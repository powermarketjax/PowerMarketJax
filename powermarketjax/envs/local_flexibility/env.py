"""Local flexibility environment layer: ``EnvState``, ``EnvParams``, ``reset``, ``step``.

Wires this package's operators -- the action map, the clearing linear program,
the nonlinear verification and the pay-as-bid settlement -- into the standard
6-tuple step interface.  Nothing here computes a market quantity itself: this
module routes what the operators return and advances the state of charge.

    reset, step, step_auto_reset, spec = make_local_flex_env(case, sens, agent_bus)
    params = make_local_flex_params(load_series, pv, energy_price, battery,
                                    cycle_cost, load_scale, learner_mask, H)
    obs, state = reset(key, params)
    obs, state, reward, costs, done, info = step(key, state, action, params)

``pv`` and ``energy_price`` are ordinary ``EnvParams`` leaves the caller
supplies; there is no default for either.  With ``energy_price`` unset the
markup has no origin, so ``price`` and ``reward`` carry no economic meaning even
though every accounting identity still holds.  Passing ``pv = 0`` is the
load-driven scenario, which the caller declares rather than falls into.

Behaviour of ``step``, in the order it applies:

* The key is split into ``transition_key`` and ``reset_key`` before anything
  uses it.  The transition consumes no randomness today -- demand, photovoltaic
  output and price are exogenous data, and the clearing breaks ties by the
  analytic centre -- so ``transition_key`` is idle.
* The action is cast to float32 on entry, so the environment fixes where the
  boundary rounds rather than the caller's dtype.
* ``learner_mask`` selects, before the action map, between the policy's action
  and ``baseline_action``: truthful price, full deliverable quantity, no planned
  charging.  Non-learners are actions, not a second code path.
* ``done`` is a time-limit truncation rather than a terminal state, so ``info``
  carries ``terminal_obs``: at ``done`` it is the observation of the true
  successor state that auto-reset overwrites, and elsewhere it equals ``obs``.
* Auto-reset merges with ``jnp.where``.  That is free here because ``reset`` is
  a gather, and ``lax.cond`` would execute both branches under ``vmap`` anyway.

Three readings this module fixes:

* The requirement in the observation is the one published for the period just
  cleared, not the current one -- agents submit before the period is realised,
  so the current requirement cannot be on the table.  At ``reset`` both
  requirement fields are zero, with every other previous-result field.
* The aggregator's own load is the load registered at its bus.  The data does
  not resolve a bus load into participant and background shares, and the physics
  depends only on the sum, so the bus load is attributed to the aggregator
  sitting there and reported as ``load_prev``.  The alternative labelling
  changes no injection, no award and no payment.
* The observation carries a charging and a discharging rating column.  They are
  equal for as long as the underlying battery bundle carries a single rating.

``jax_enable_x64`` must be on; ``make_clearing`` and ``make_requirement`` raise
when it is not, and this constructor calls both.
"""
import warnings
from typing import Callable, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from powermarketjax.resources.battery import BatteryBundle, update_soc_batch

from .action import ACTION_SATURATION, make_action_map
from .clearing import MAX_ITER, VOLL, make_clearing
from .requirement import make_requirement
from .sensitivity import VoltageSensitivity
from .settlement import make_settlement
from .verification import cleared_injection, make_verification

#: 7 static + 1 state of charge + 2 realised + 3 own previous result + 2 own
#: requirement + 4 feeder + 2 public + 2 calendar.
OBS_DIM = 23

#: The non-learner's action, reachable and exact in float32: `softplus(-128)`
#: is zero, so the price is the replacement cost; `sigmoid(+128)` is one, so
#: nothing is withheld; `sigmoid(-128)` is zero, so no charging is planned and
#: the baseline is not positioned.  Written in terms of `ACTION_SATURATION`
#: because the two are one quantity: the action box published in `spec` is the
#: frame these three saturation points sit on, so a baseline that stopped
#: agreeing with the box would be a baseline outside this market's own action
#: space.
BASELINE_ACTION = (-ACTION_SATURATION, ACTION_SATURATION, -ACTION_SATURATION)

#: Denominator floor for the average price paid.  A period in which nothing
#: clears is common, and one NaN lane pollutes a whole `vmap` batch.  An idle
#: feeder leaves interior-point dust rather than an exact zero, so the ratio
#: needs a floor even when volume is not exactly zero.
VOLUME_EPS = 1e-9


@struct.dataclass
class LocalFlexState:
    """Everything that crosses a period boundary."""
    cursor: chex.Array            # int32   ()     index into the exogenous series
    step_in_episode: chex.Array   # int32   ()     counts toward episode_len
    soc: chex.Array               # float32 (N,)   the only physical carry
    req_v_own: chex.Array         # float32 (N,)   requirement at own bus, MW
    req_th_own: chex.Array        # float32 (N,)   largest on the path, MW
    # req_v_max/req_th_max are feeder extremes, not sums: relieving one bus
    # relieves its neighbours and lines on a path are nested, so only an
    # extreme and a count are meaningful aggregates.
    req_v_max: chex.Array         # float32 ()     feeder extreme, voltage
    req_th_max: chex.Array        # float32 ()     feeder extreme, thermal
    req_v_count: chex.Array       # int32   ()     count of buses in violation
    req_th_count: chex.Array      # int32   ()     count of lines in violation
    award_prev: chex.Array        # float32 (N,)   previous cleared quantity, MW
    payment_prev: chex.Array      # float32 (N,)   previous payment, \\$
    profit_prev: chex.Array       # float32 (N,)   previous profit, \\$
    pv_prev: chex.Array           # float32 (N,)   previous realised PV, MW
    load_prev: chex.Array         # float32 (N,)   previous realised own load, MW
    volume_prev: chex.Array       # float32 ()     published volume, MW
    price_avg_prev: chex.Array    # float32 ()     published average price,
    #                                              \\$/MWh; zero when nothing cleared


@struct.dataclass
class LocalFlexParams:
    """Sweepable and data-carrying quantities; built by
    ``make_local_flex_params`` only, which owns the dtype/axis/window
    contracts."""
    load_series: chex.Array       # float32 (n_periods,)    feeder total, MW
    pv: chex.Array                # float32 (n_periods, N)  per aggregator, MW
    energy_price: chex.Array      # float32 (n_periods,)    exogenous, \\$/MWh
    battery: BatteryBundle        # parameter container only, never .step()
    cycle_cost: chex.Array        # float32 (N,)   degradation, \\$/MWh
    load_scale: chex.Array        # float32 ()     feeder demand scale factor; no default
    learner_mask: chex.Array      # bool    (N,)   True where the policy acts
    episode_len: chex.Array       # int32   ()     episode length in periods
    #: int32 (n_starts,).  The episode starts `reset` may draw, which is every
    #: legal start unless the caller restricts it.  It exists so an experiment
    #: can hold days out of training and evaluate on them: without it `reset`
    #: samples the whole series and there is no set of days a policy has not
    #: been trained on.  Carrying the pool as data rather than as a bound keeps
    #: `reset` a gather and leaves it jittable and vmappable unchanged.
    cursor_pool: chex.Array
    #: bool ().  When true, the published requirement is computed against a
    #: baseline that excludes the planned charging of each participant, so a
    #: participant raises no requirement, and earns nothing, by scheduling a
    #: charge that deepens the constraint it would then be paid to relieve.
    #: The specification keeps that positioning exposure deliberately and names
    #: a market monitor as what would address it; this flag is that monitor in
    #: its strongest form, and exists so that the two can be compared on one
    #: scenario.  The physics, the verification and the settlement all continue
    #: to use the real injection, so a participant that charges anyway still
    #: moves the feeder and is simply not paid for it.  Default false, which is
    #: the mechanism as specified.
    monitor_baseline: chex.Array


def baseline_action(n_agent: int) -> chex.Array:
    """The non-learner's submission, ``(n_agent, 3)`` float32.

    Constant because none of its three components depends on the state: the
    truthful price is a scale factor of one on the replacement cost, withholding
    nothing is a fraction of one of whatever is deliverable, and not positioning
    the baseline is zero planned charging.
    """
    return jnp.tile(jnp.asarray(BASELINE_ACTION, jnp.float32), (n_agent, 1))


def agent_bus_candidates(case, sens: VoltageSensitivity):
    """Buses eligible to host an aggregator, and the draw weight of each.

    Weighted by registered load rather than by taking the largest loads: that
    alternative is deterministic and puts every aggregator next to the
    substation, where an injection is electrically ineffective.

    Returns ``(buses, weights)`` as numpy arrays, setup time.
    """
    load = np.asarray(case.node_pd, np.float64)
    buses = np.flatnonzero(load > 0.0)
    buses = buses[buses != sens.slack]
    if buses.size == 0:
        raise ValueError(
            "no bus carries registered load, so no aggregator can be placed; "
            "this case does not support the local flexibility market")
    return buses, load[buses] / load[buses].sum()


def draw_agent_buses(case, sens: VoltageSensitivity, n_agent: int,
                     seed: int) -> np.ndarray:
    """Draw a population placement, load-weighted and without replacement.

    ``seed`` has no default: it must be reported with any result obtained
    under the draw.  Returns the bus index of each aggregator, sorted.
    """
    buses, weights = agent_bus_candidates(case, sens)
    if n_agent > len(buses):
        raise ValueError(
            f"cannot place {n_agent} aggregators without replacement on "
            f"{len(buses)} load-carrying buses")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(buses, n_agent, replace=False, p=weights))


def make_local_flex_params(
    load_series: np.ndarray,
    pv: np.ndarray,
    energy_price: np.ndarray,
    battery: BatteryBundle,
    cycle_cost: np.ndarray,
    load_scale: float,
    learner_mask: np.ndarray,
    episode_len: int,
    cursor_pool: np.ndarray = None,
    monitor_baseline: bool = False,
) -> LocalFlexParams:
    """Build and validate ``LocalFlexParams`` (setup time, numpy, may raise).

    Fixes dtypes so no caller has to pass them by luck: the series and the
    per-aggregator arrays go to float32, the mask to bool, every axis is
    checked against the battery's device axis, and the window is checked
    against the series length.  ``battery`` itself is not re-validated, since
    ``make_battery_bundle`` already rejects degenerate devices at construction.

    Args:
        load_series: ``(n_periods,)`` feeder total in MW.  The bus-level
            allocation is a property of the case and happens in ``step``.
        pv: ``(n_periods, n_agent)`` photovoltaic output in MW.  Zeros are the
            load-driven fallback, a declared scenario.
        energy_price: ``(n_periods,)`` exogenous energy price in \\$/MWh.
        battery: ``BatteryBundle``; its device axis fixes ``n_agent``.
        cycle_cost: ``(n_agent,)`` degradation cost per MWh of throughput.
        load_scale: strictly positive scale factor on the feeder demand
            series, with no default: without it the feeder never leaves its
            voltage band and the market clears nothing.
        learner_mask: ``(n_agent,)`` bool, True where the policy acts.
        episode_len: episode length in periods.
        cursor_pool: episode starts ``reset`` may draw, as indices into the
            series.  ``None``, the default, is every legal start, which is
            what ``reset`` drew before the pool existed; pass a subset to
            hold days out of training.
        monitor_baseline: when True, publish the requirement against a
            baseline with each participant's planned charging removed.
            False, the default, is the mechanism as specified.
    """
    load_series = np.asarray(load_series, np.float32)
    pv = np.asarray(pv, np.float32)
    energy_price = np.asarray(energy_price, np.float32)
    cycle_cost = np.asarray(cycle_cost, np.float32)
    learner_mask = np.asarray(learner_mask, bool)

    n = int(battery.n_devices)
    if load_series.ndim != 1:
        raise ValueError(f"load_series must be (n_periods,), got {load_series.shape}")
    n_periods = load_series.shape[0]
    if pv.shape != (n_periods, n):
        raise ValueError(f"pv must be ({n_periods}, {n}), got {pv.shape}")
    if energy_price.shape != (n_periods,):
        raise ValueError(
            f"energy_price must be ({n_periods},), got {energy_price.shape}")
    if cycle_cost.shape != (n,):
        raise ValueError(f"cycle_cost must be ({n},), got {cycle_cost.shape}")
    if learner_mask.shape != (n,):
        raise ValueError(f"learner_mask must be ({n},), got {learner_mask.shape}")

    load_scale = float(load_scale)
    if not np.isfinite(load_scale) or load_scale <= 0.0:
        raise ValueError(
            f"load_scale must be finite and positive, got {load_scale}: it "
            "multiplies the feeder demand series")

    episode_len = int(episode_len)
    if not 1 <= episode_len <= n_periods:
        raise ValueError(
            f"episode_len must lie in [1, {n_periods}], got {episode_len}: the "
            "legal starts are [0, n_periods - episode_len] and that interval "
            "must be non-empty")
    if episode_len < 2:
        # legal, but a one-step episode leaves nothing for terminal_obs to
        # bootstrap from
        warnings.warn("episode_len < 2: every step is the truncation step",
                      stacklevel=2)

    # Default: every legal start, which is what `reset` sampled before the pool
    # existed, so an omitted argument reproduces the previous behaviour exactly.
    n_starts = n_periods - episode_len + 1
    if cursor_pool is None:
        cursor_pool = np.arange(n_starts, dtype=np.int32)
    else:
        cursor_pool = np.asarray(cursor_pool, np.int64).ravel()
        if cursor_pool.size == 0:
            raise ValueError("cursor_pool is empty: reset would have no start "
                             "to draw and the restriction is almost certainly "
                             "a filter that matched nothing")
        bad = cursor_pool[(cursor_pool < 0) | (cursor_pool >= n_starts)]
        if bad.size:
            raise ValueError(
                f"cursor_pool holds {bad.size} start(s) outside "
                f"[0, {n_starts - 1}], e.g. {bad[:5].tolist()}: a start past "
                f"that runs the episode off the end of the series")
        if np.unique(cursor_pool).size != cursor_pool.size:
            raise ValueError(
                "cursor_pool holds duplicates, which silently reweights the "
                "episode distribution rather than restricting it")
        cursor_pool = cursor_pool.astype(np.int32)

    return LocalFlexParams(
        load_series=jnp.asarray(load_series), pv=jnp.asarray(pv),
        energy_price=jnp.asarray(energy_price), battery=battery,
        cycle_cost=jnp.asarray(cycle_cost),
        load_scale=jnp.asarray(load_scale, jnp.float32),
        learner_mask=jnp.asarray(learner_mask),
        episode_len=jnp.asarray(episode_len, jnp.int32),
        cursor_pool=jnp.asarray(cursor_pool, jnp.int32),
        monitor_baseline=jnp.asarray(bool(monitor_baseline)))


def make_local_flex_env(
    case,
    sens: VoltageSensitivity,
    agent_bus: np.ndarray,
    voltage_margin: float = 0.0,
    thermal_margin: float = 0.0,
    period_hours: float = 0.25,
    voll: float = VOLL,
    max_iter: int = MAX_ITER,
) -> Tuple[Callable, Callable, Callable, Dict]:
    """Build the local flexibility environment for one case and one population.

    Returns ``(reset, step, step_auto_reset, spec)``:

        reset(key, params)                          -> (obs, state)
        step(key, state, action, params)            -> 6-tuple step interface
        step_auto_reset(key, state, action, params) -> same, obs and state
                                                       behind stop_gradient

    The case, the network constants, the placement, the two safety margins and
    the period length are closed over: the placement enters the constraint
    matrix and the margins are checked at construction, so none of them can be
    swept without rebuilding.  ``load_scale`` is not among them -- it is a
    ``params`` leaf and sweeping it recompiles nothing.

    ``spec`` carries the static shapes plus ``get_obs`` and ``baseline_action``
    for tests that recompute an observation or a baseline without reaching into
    the closure, and the clearing operator's own ``spec`` under ``clearing``.
    """
    agent_bus = np.asarray(agent_bus, np.int64)
    if agent_bus.ndim != 1 or agent_bus.size == 0:
        raise ValueError(f"agent_bus must be a non-empty 1-D array of bus "
                         f"indices, got shape {agent_bus.shape}")
    n_agent = int(agent_bus.size)

    # The series supplies the feeder total per period and the registered load
    # vector supplies each bus's participation in it.  Carrying it as one
    # factor on the registered vector keeps the ratio between active and
    # reactive demand exactly as registered.  Checked before anything is
    # built, since a case that fails it fails for good.
    pd_mw = np.asarray(case.node_pd, np.float64)
    qd_mw = np.asarray(case.node_qd, np.float64)
    registered_total = float(pd_mw.sum())
    if registered_total <= 0.0:
        raise ValueError(
            f"the registered load of this case sums to {registered_total:.4f} "
            "MW, so a feeder total cannot be allocated to buses in proportion "
            "to it.  `case533mt_lo` is the net-exporting case this "
            "rejects; the primary case is `case533mt_hi`")

    clear, clearing_spec = make_clearing(
        case, sens, agent_bus, voltage_margin=voltage_margin,
        thermal_margin=thermal_margin, period_hours=period_hours, voll=voll,
        max_iter=max_iter)
    publish = make_requirement(case, sens)
    verify = make_verification(case, sens)
    settle = make_settlement(sens, period_hours)
    act_map, action_spec = make_action_map(n_agent, period_hours)

    base_mva = float(sens.base_mva)
    pd_pu = jnp.asarray(pd_mw / (base_mva * registered_total))
    qd_pu = jnp.asarray(qd_mw / (base_mva * registered_total))
    pd_share_mw = jnp.asarray(pd_mw / registered_total)
    agent_bus_j = jnp.asarray(agent_bus)
    phi_j = jnp.asarray(clearing_spec["phi"])
    # A[l, bus(i)] is one exactly on the lines between the substation and
    # bus(i) -- the lines that aggregator i's injection relieves -- so the
    # largest requirement on the path is a masked maximum.
    path_mask = jnp.asarray(sens.A[:, agent_bus])          # (n_line, n_agent)

    money_factor = period_hours * base_mva
    steps_per_day = 24.0 / period_hours
    baseline = baseline_action(n_agent)

    def _get_obs(state: LocalFlexState, params: LocalFlexParams) -> chex.Array:
        """The observation vector as a pure function of ``(state, params)``.

        Nothing is read from the exogenous series here: the realised
        photovoltaic output and own load an agent observes are the previous
        period's and live in the state, because an agent submits before the
        current period is realised.
        """
        battery = params.battery
        phase = 2.0 * jnp.pi * (state.cursor.astype(jnp.float32) / steps_per_day)
        broadcast = lambda x: jnp.full((n_agent,), x, jnp.float32)
        return jnp.stack([
            # static registered parameters.  The two rating columns are equal
            # for as long as the battery bundle carries a single rating.
            battery.capacity, battery.power_max, battery.power_max,
            battery.eta_charge, battery.eta_discharge,
            battery.soc_min, battery.soc_max,
            state.soc,                                            # own carry
            state.pv_prev, state.load_prev,                       # own realised
            state.award_prev, state.payment_prev, state.profit_prev,
            state.req_v_own, state.req_th_own,                    # own signals
            broadcast(state.req_v_max), broadcast(state.req_th_max),
            broadcast(state.req_v_count.astype(jnp.float32)),     # feeder level
            broadcast(state.req_th_count.astype(jnp.float32)),
            broadcast(state.volume_prev), broadcast(state.price_avg_prev),
            broadcast(jnp.sin(phase)), broadcast(jnp.cos(phase)),  # calendar
        ], axis=1).astype(jnp.float32)

    def reset(key: chex.PRNGKey, params: LocalFlexParams):
        """Draw a start uniformly from ``params.cursor_pool``, which defaults
        to every legal start; everything else is a gather.  Previous-result
        fields, the two requirements included, are zero by convention rather
        than by estimate."""
        # Drawn from `cursor_pool` rather than from the whole series, so that a
        # caller holding days out gets episodes only from the days it kept.  The
        # pool defaults to every legal start, so this is the previous uniform
        # draw whenever the caller does not restrict it.
        pool = params.cursor_pool
        cursor = pool[jax.random.randint(key, (), 0, pool.shape[0])]
        zeros_n = jnp.zeros((n_agent,), jnp.float32)
        state = LocalFlexState(
            cursor=jnp.asarray(cursor, jnp.int32),
            step_in_episode=jnp.asarray(0, jnp.int32),
            soc=jnp.asarray(params.battery.initial_soc, jnp.float32),
            req_v_own=zeros_n, req_th_own=zeros_n,
            req_v_max=jnp.float32(0.0), req_th_max=jnp.float32(0.0),
            req_v_count=jnp.int32(0), req_th_count=jnp.int32(0),
            award_prev=zeros_n, payment_prev=zeros_n, profit_prev=zeros_n,
            pv_prev=zeros_n, load_prev=zeros_n,
            volume_prev=jnp.float32(0.0), price_avg_prev=jnp.float32(0.0))
        return _get_obs(state, params), state

    def step(key: chex.PRNGKey, state: LocalFlexState, action: chex.Array,
             params: LocalFlexParams):
        """One `period`: submit, clear, verify, settle, advance the state of charge.

        A `step` of this market spans a single period rather than a market
        day, so one auction runs per call.  Auto-reset is embedded, so this is
        already the function `lax.scan` drives; `step_auto_reset` adds only
        `stop_gradient`.

        Args:
            key: split once at the entry into a transition key, which nothing
                consumes today, and the key `reset` draws the next start from.
            state: the carry; ``cursor`` indexes the exogenous series.
            action: ``(n_agent, 3)``, unbounded and cast to float32 here.  Rows
                where ``learner_mask`` is False are replaced by
                `baseline_action` before the action map sees them, so a
                non-learner is an action and not a second code path.
            params: the sweepable leaves; ``load_scale`` among them.

        Returns:
            The 6-tuple step interface.  ``reward`` is ``(n_agent,)`` in
            dollars and is the pay-as-bid profit computed from ``award``;
            ``costs`` is ``(n_agent, 3)`` in the order ``spec["cost_names"]``
            fixes and never reaches ``reward``; ``done`` is a truncation, so
            ``info["terminal_obs"]`` carries the observation of the true
            successor state that auto-reset overwrote.
        """
        # split before any use; the transition consumes no randomness today
        transition_key, reset_key = jax.random.split(key)
        del transition_key
        action = jnp.asarray(action, jnp.float32)
        action = jnp.where(params.learner_mask[:, None], action, baseline)

        cursor = state.cursor
        pv_t = params.pv[cursor]
        energy_price = params.energy_price[cursor]
        # load_scale scales the feeder demand series before it is allocated
        # to buses.  The registered vectors are float64 and the two scalars
        # are float32, so the product is formed against the vector first: the
        # series value is already rounded to float32, and rounding their
        # product as well would add a second rounding for nothing.
        bus_load = params.load_series[cursor] * pd_pu * params.load_scale
        bus_q = params.load_series[cursor] * qd_pu * params.load_scale

        sub = act_map(action, state.soc, energy_price, params.battery,
                      params.cycle_cost)

        # The baseline operating point: realised demand and photovoltaic
        # output and planned charging, no award and no directed curtailment.
        # It is measured here from the physical state rather than declared by
        # the participants, so there is no room to overstate it; the one
        # participant input that reaches it is the planned charging, which is
        # reported through `plan_selling` below.
        p_base = (-bus_load).at[agent_bus_j].add(
            (pv_t - sub["plan"]) / base_mva)
        q_base = -bus_q

        # The baseline the operator procures against.  It is the physical one
        # unless `monitor_baseline` is set, in which case the planned charging
        # is taken out of it, so congestion a participant created by planning to
        # charge raises no requirement and earns nothing.  `p_base` itself is
        # left alone: the swept verification, the unserved requirement and the
        # settlement all keep the real injection, so the charge still moves the
        # feeder whether or not anyone is paid for stopping it.
        p_procure = jnp.where(
            params.monitor_baseline,
            (-bus_load).at[agent_bus_j].add(pv_t / base_mva),
            p_base)

        out = clear(sub["price"], sub["qty_max"] / base_mva, p_procure, q_base,
                    bus_load)
        award, shed = out["award"], out["shed"]

        p_cleared, q_cleared = cleared_injection(
            sens, agent_bus_j, phi_j, p_base, q_base, award, shed)
        swept = verify(p_cleared, q_cleared)
        # what the clearing left unrelieved, reported per index, never summed
        unserved = publish(p_cleared, q_cleared)

        money = settle(sub["price"], award, sub["plan"] / base_mva,
                       energy_price, params.cycle_cost)

        # The powers come back per unit from the settlement, and the
        # vendored advance works in MW; `BatteryBundle.step` is bypassed
        # entirely.
        p_signed = (money["p_dis"] - money["p_ch"]) * base_mva
        soc_next = update_soc_batch(
            state.soc, p_signed, params.battery.capacity,
            params.battery.eta_charge, params.battery.eta_discharge,
            params.battery.soc_min, params.battery.soc_max, period_hours)

        volume = base_mva * jnp.sum(award)                       # MW
        payment_total = jnp.sum(money["revenue"])                # \\$
        # The published average: total payment over total energy.  This is a
        # statistic of what pay-as-bid actually paid, not a price the market
        # formed -- it enters the observation and nothing else; no
        # settlement expression reads it.
        price_avg = jnp.where(
            volume > 0.0,
            payment_total / jnp.maximum(period_hours * volume, VOLUME_EPS), 0.0)

        next_state = LocalFlexState(
            cursor=cursor + 1,
            step_in_episode=state.step_in_episode + 1,
            soc=soc_next.astype(jnp.float32),
            # `requirement` works in per unit throughout (its module doc), and
            # every other power column of this observation is in MW: `award_prev`
            # and `volume_prev` are scaled here, `pv_prev` and `load_prev` never
            # left MW.  So the four requirement fields are converted here too,
            # which is what their declared unit says and what lets an aggregator
            # read "what is needed" against "what I sold" in one row.  `info`
            # keeps the per-unit vectors, since that is the operator's quantity
            # and §16 reports it that way.
            req_v_own=(base_mva * out["req_v"][agent_bus_j]).astype(jnp.float32),
            req_th_own=(base_mva * jnp.max(path_mask * out["req_th"][:, None],
                                           axis=0)).astype(jnp.float32),
            req_v_max=(base_mva * out["req_v_max"]).astype(jnp.float32),
            req_th_max=(base_mva * out["req_th_max"]).astype(jnp.float32),
            req_v_count=out["req_v_count"].astype(jnp.int32),
            req_th_count=out["req_th_count"].astype(jnp.int32),
            award_prev=(base_mva * award).astype(jnp.float32),
            payment_prev=money["revenue"].astype(jnp.float32),
            profit_prev=money["profit"].astype(jnp.float32),
            pv_prev=pv_t,
            load_prev=(params.load_series[cursor] * pd_share_mw[agent_bus_j]
                       * params.load_scale).astype(jnp.float32),
            volume_prev=volume.astype(jnp.float32),
            price_avg_prev=price_avg.astype(jnp.float32))
        done = next_state.step_in_episode >= params.episode_len

        # evaluating reset unconditionally is free here -- it is a gather --
        # and lax.cond would run both branches under vmap anyway
        _, fresh = reset(reset_key, params)
        merged = jax.tree_util.tree_map(
            lambda nxt, new: jnp.where(done, new, nxt), next_state, fresh)

        obs = _get_obs(merged, params)
        # the true successor observation, which the merge above overwrites at
        # done; a truncated episode bootstraps from this, not from obs
        terminal_obs = jnp.where(done, _get_obs(next_state, params), obs)

        reward = money["reward"].astype(jnp.float32)
        # Three system quantities: the two violation depths surviving the
        # sweep and the energy curtailed.  All three are copied along the
        # agent axis rather than attributed, since there is no attribution
        # rule for them.
        #
        # The two sums along a bus and a line index are violation depths --
        # a measure of constraint violation that is zero exactly at a
        # feasible point, so summing them is a valid aggregation.  The
        # published requirements are a different quantity that may not be
        # summed, and they are nowhere in this vector: they travel per index
        # in `info` below.
        system_costs = jnp.stack([
            # `money_factor` is `period_hours * base_mva`, which turns a
            # per-unit power into MWh here and into dollars in the settlement
            money_factor * jnp.sum(shed),                        # MWh
            jnp.sum(swept["v_under"] + swept["v_over"]),         # per unit
            jnp.sum(swept["overload"]),                          # per unit
        ]).astype(jnp.float32)
        costs = jnp.broadcast_to(system_costs, (n_agent, 3))

        info = dict(
            # solver and sweep diagnostics; none of them is a feasibility
            # criterion, and `converged` must never be merged with the depths
            mu=out["mu"], dual_residual=out["dual_residual"], z=out["z"],
            iterations=swept["iterations"], converged=swept["converged"],
            floor_active=swept["floor_active"],
            # per-index quantities: what the sweep found, what was published
            # against the baseline, and what the clearing left
            v_under=swept["v_under"], v_over=swept["v_over"],
            overload=swept["overload"],
            req_v=out["req_v"], req_th=out["req_th"],
            req_v_unserved=unserved["req_v"], req_th_unserved=unserved["req_th"],
            req_v_max=out["req_v_max"], req_th_max=out["req_th_max"],
            req_v_count=out["req_v_count"], req_th_count=out["req_th_count"],
            traded_volume=volume, price_avg=price_avg,
            # the charging planned in total, and the part of it planned by
            # aggregators that then sold flexibility
            plan_total=jnp.sum(sub["plan"]),
            plan_selling=jnp.sum(jnp.where(award > 0.0, sub["plan"], 0.0)),
            terminal_obs=terminal_obs)
        return obs, merged, reward, costs, done, info

    def step_auto_reset(key, state, action, params):
        """`step` with ``obs`` and the new state behind `stop_gradient`.

        The reset itself is already inside `step`, so this adds nothing to the
        transition; the barrier keeps a gradient from crossing an episode
        boundary in a `lax.scan` rollout.  ``reward``, ``costs``, ``done`` and
        ``info`` are passed through untouched.
        """
        obs, new_state, reward, costs, done, info = step(key, state, action,
                                                         params)
        obs = jax.lax.stop_gradient(obs)
        new_state = jax.lax.stop_gradient(new_state)
        return obs, new_state, reward, costs, done, info

    spec = dict(n_agent=n_agent, obs_dim=OBS_DIM, action_shape=(n_agent, 3),
                # Read off the action map's own spec rather than restated.  Two
                # copies of the action space is the failure `bounds_for`
                # describes in the small: until 2026-09-10 the map's spec was
                # discarded at the call above and these two ends were written
                # again here, so the box the map declared reached nothing.
                action_low=action_spec["action_low"],
                action_high=action_spec["action_high"],
                costs_dim=3,
                cost_names=("shed_energy_mwh", "voltage_violation_sum_pu",
                            "thermal_violation_sum_pu"),
                termination="truncation", agent_bus=agent_bus,
                voltage_margin=float(voltage_margin),
                thermal_margin=float(thermal_margin),
                period_hours=float(period_hours), voll=float(voll),
                max_iter=int(max_iter), base_mva=base_mva,
                registered_total_mw=registered_total,
                dtype=jnp.float32, clearing=clearing_spec,
                get_obs=_get_obs, baseline_action=baseline)
    return reset, step, step_auto_reset, spec
