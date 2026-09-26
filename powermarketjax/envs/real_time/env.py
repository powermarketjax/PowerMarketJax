"""Real-time balancing environment: one step is one half-hour dispatch period.

    env, spec = make_env(case, load_da_position(chain="step1prime_seasons"),
                         load_gb_demand_half_hourly(), forecast, markup_max=2.0,
                         cap_scale=0.60, ramp_scale=1.00)
    params = env.make_params(episode_len=48)
    obs, state = env.reset(key, params)
    obs, state, reward, costs, done, info = env.step(key, state, action, params)

**One step, one period, one clearing.**  The clearing is the day-ahead operator
at `T = 1` (`clearing.make_rt_clearing`), so no clearing code is added; what this
market adds is the two-settlement rule and the sequential coupling.  The ramp
constraint reaches `p_prev`, the realised dispatch of the previous period, which
is the only physical carry and what makes the problem sequential.  One step is
one 30-minute dispatch period, unlike the day-ahead market's whole market day.

**The day-ahead position is exogenous input data, not the return value of a
day-ahead clearing run here.**  The commitment, the schedule, the shed and the
price all come from a frozen fixture and no action of any agent moves them.
Only the deviation from that schedule is settled at the real-time price, so
what an action can move is the deviation and its price -- which is the
decision problem this market poses.

**`episode_len` does not cross a day.**  Crossing would carry `p_prev` over a
day boundary *and* switch to the next day's fixture commitment in the same
step, and neither seam has been verified; the two would arrive together and
could not be separated if something went wrong.

**`info` carries `terminal_obs`.**  `done` here is truncation, not termination,
and `step_auto_reset` replaces the state on the final step, so the observation
returned alongside `done` belongs to the *new* episode.  A learner bootstrapping
a truncated episode needs the observation of the state the episode actually
ended in.
"""
from typing import Callable, Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from powermarketjax.envs.day_ahead.action import make_offer_map, truthful_action
from powermarketjax.envs.day_ahead.clearing import VOLL, segment_costs

from .boundary import make_boundary
from .clearing import MAX_ITER, make_rt_clearing
from .demand import T_RT
from .position import PERIODS_PER_HOUR
from .settlement import make_settlement

#: Convergence threshold on the clearing's complementarity gap.  In this market
#: `mu` is also the **infeasibility** detector: shed covers a generation
#: shortfall and never absorbs a surplus, so a net bus load falling faster than
#: the committed units can ramp down has no feasible point at all and `mu` runs
#: to 1e26x.
MU_TOL = 1e-6
#: This market's own dual-residual tolerance: above it, a period's duals cannot
#: be used as prices.  **Derived from the price consequence, not from the floor
#: of healthy periods.**  Measured on `case813nem`: `|dlmp|` (against the full
#: line set) is about 20 x `dual_residual` -- 5.0e-03 goes with 0.108 $/MWh,
#: 1e-09 with ~1e-08 -- so the "price error within 1e-03 $/MWh" tier puts the
#: tolerance at 5e-05.  Market 03 measured 0.3-3.3x on its own clearing, a
#: different point; **this ratio is measured per market, not carried over**.
#:
#: **Re-measured on 2026-09-17, this derivation has a hole and the conclusion
#: still holds -- both reasons are recorded here.**
#:
#: The hole is in the reference arm: the "20x" was measured against the
#: **full-line dense route** as the gold standard, and **the dense route itself
#: breaks on `case813nem`**.  Measured (CPU, x64, 813nem market 02 rated,
#: truthful offers, 4 days, 192 periods): in **1 of the 192 periods the
#: reference arm's own `dual_residual` is 5.0003e-03**, while the arm under
#: test has 7.47e-08 in the same period -- the 0.108 $/MWh `|dlmp|` of that
#: period is the reference arm's error.  On the tier where the reference arm is
#: 9.1e6x cleaner, **k = 21.6**, the same order as the original 20; on healthy
#: tiers the reference and tested arms are of the same order (0.9-2.7x), so the
#: k measured there is the reference arm's noise and unusable.
#:
#: **A second, independent sample** (2026-09-17, replay of day 351, CPU 8
#: cores, `make_clearing` jitted directly): the dense full-line arm at step 13
#: has `dual_residual` 5.00e-3 and `mu` 5.3e-9, while in the same period dense
#: rated / (8,523)+seq / (151,813)+arrow are all <= 1e-10; every rated arm is
#: 0.108 $/MWh `|dlmp|` away from the full-line arm, and the reference arm is
#: the one that is wrong.  The same cell reads the same on a separately
#: exported tree.
#:
#: The two samples reach the same fact from two directions -- one while using
#: the dense route as the gold standard, one while comparing routes.  **Any
#: number obtained by "comparing against the full-line dense route" must list
#: separately the periods in which the reference arm itself broke.**
#:
#: **So this number now acts as a safety net, not as a gate "just tight enough
#: to catch bad periods".**  What actually stops bad periods is the recipe
#: `reg_coef=1e-16` plus stop: on the same setup it holds `dual_residual` to at
#: most **9.6e-08** (**0/192** periods above 5e-5; with the reference arm's bad
#: periods excluded, `|dlmp|` peaks at 2.44e-05 $/MWh, a 41x margin to "within
#: 1e-3 $/MWh"); over the full window of 2 544 periods the count is also 0.
#: **With the recipe on, no period comes near 5e-05; the gap is 500x.**
#: Tightening this number (to 1e-7, say) needs the reference arm fixed first
#: -- the recipe turned on in the reference arm too, or its bad periods
#: excluded one by one -- and until the recipe is the default there is no
#: consumer for that.
#:
#: **This ratio was measured under the current `REG_COEF = 1e-14`.**  On
#: 2026-09-17 the 5.0e-03 tier was traced to the fixed point that
#: `reg = REG_COEF * mean(D)` reaches once the slack of the load-shed columns
#: hits `SLACK_FLOOR` (on those rows D ~ 1e18, `mean(D)` ~ 3.6e17, `reg` ~
#: 3.6e3, and each Newton step lets x move only `r1/reg` ~ 1.4e-6), whereas
#: `reg_coef=1e-16` pushed r1 down to 8.4e-12 on one cell.  **If `REG_COEF`
#: changes, the 20x above must be re-measured** and this tolerance re-set with
#: it.
#:
#: `MU_TOL` alone is not this criterion: on 2026-09-16 periods were measured
#: with `mu` 1.3e-11 (passing the gate cleanly) and a dual residual of
#: 8.2e-03, and `converged` looks only at `mu`.
#:
#: **This is the only definition in the repository.**  `tools/benchmark/run_rl_02.py`
#: imports it from here instead of carrying a constant of the same name: two
#: setups giving different defaults for the same quantity is worse than both
#: giving the old value.
#:
#: **What switching from 1e-6 to 5e-5 moves has been measured**: over the full
#: window of 17 520 periods of `case813nem` market 02 with truthful offers, the
#: periods with `dual_ok=False` go from 1 292 to about 1 218 (measured on the
#: same setup over a 1 488-period subset: 112 -> 106).  **The readings barely
#: move** -- the distribution is bimodal, healthy periods sit below 1e-8 and
#: bad ones pile up at 5.0e-03, only a dozen or so periods live in the six
#: decades from 1e-7 to 1e-3, and both thresholds fall in the same gap.  What
#: moves is whether the tolerance matches the "price error within 1e-3 $/MWh"
#: criterion.  **The threshold that would actually change readings is near
#: 1e-8** (moving it there shifts hundreds of periods); setting it there needs
#: the price consequence of the 1e-8 tier measured first.
DUAL_RES_TOL = 5e-5

#: The `costs` columns.  One column: this market has no minimum up/down time,
#: so the day-ahead market's second column does not apply.
COST_NAMES = ("shed_mwh",)

#: Floor below which shed energy is treated as zero, MWh.  Absolute, not
#: relative: a period that sheds nothing leaves ~1e-20 behind, so a relative
#: tolerance there would compare noise.
SHED_FLOOR = 1e-6


@struct.dataclass
class RealTimeState:
    """Three categories of carry.

    `cursor` and `step_in_episode` locate the step in the data and the episode.
    `p_prev` is the **physical** carry: it enters the ramp rows of the next
    clearing and is the only thing that makes the problem sequential.  The last
    three are previous-step results that the observation reads; they are carried
    rather than recomputed so that the observation stays a function of the state
    alone, which is what makes it survive an auto-reset.
    """
    cursor: chex.Array               # int32, index into the period series
    step_in_episode: chex.Array      # int32
    p_prev: chex.Array               # (n_units,) float32, the physical carry
    lmp_prev: chex.Array             # (n_buses,) float32
    award_prev: chex.Array           # (n_units,) float32
    profit_prev: chex.Array          # (n_agents,) float32


@struct.dataclass
class EnvParams:
    """Exogenous series and the agent population.

    Every leaf is indexed by absolute period so that a step reads one slice and
    no arithmetic on day/period appears in `step`.  The day-ahead quantities are
    already expanded onto real-time periods here rather than mapped on the fly:
    the map is a repeat, and doing it once at construction keeps the seam in one
    place, which matters because the money-balance identity cannot see an error
    in it.
    """
    demand_actual: chex.Array        # (n_periods,) float32, what clearing runs on
    demand_forecast: chex.Array      # (n_periods,) float32, shown in the observation
    q_da: chex.Array                 # (n_periods, n_units) float32
    lmp_da: chex.Array               # (n_periods, n_buses) float32
    s_da: chex.Array                 # (n_periods, n_buses) float32
    d_da: chex.Array                 # (n_periods, n_buses) float32
    u_da: chex.Array                 # (n_periods, n_units) float32
    day_of: chex.Array               # (n_periods,) int32, which fixture day
    learner_mask: chex.Array         # (n_agents,) bool
    episode_len: int = struct.field(pytree_node=False)


def _expand(x, axis=0):
    """Day-ahead hourly onto real-time periods; a repeat, never an interpolation."""
    return np.repeat(np.asarray(x), PERIODS_PER_HOUR, axis=axis)


def make_env(
    case,
    position: Dict,
    demand_actual_hh: np.ndarray,
    demand_forecast_hourly: np.ndarray,
    n_segments: int = 1,
    markup_max: Optional[float] = None,
    cap_scale: float = 1.0,
    ramp_scale: float = 1.0,
    unit_to_agent: Optional[np.ndarray] = None,
    max_iter: int = MAX_ITER,
    n_lookahead: int = 1,
    window_demand: str = "forecast",
    terminal_demand: str = "persist",
    monitored_lines=None,
    lowrank_free=None,
    lu_batching: str = "auto",
    reg_coef=None,
    stop_tol=None,
    dual_res_tol: float = DUAL_RES_TOL,
) -> Tuple[object, Dict]:
    """Build the environment for one case, position fixture and demand pair.

    Args:
        case: a `CaseData`.
        position: the dict `position.load_da_position` returns.
        demand_actual_hh: `(n_days_all, 48)` realised demand.
        demand_forecast_hourly: `(n_days_all, 24)` day-ahead forecast, shown to
            the agent and never used to clear.
        markup_max: the action ceiling.  No default: it decides how far an
            agent can move the price and therefore the whole reward scale.
        cap_scale, ramp_scale: scenario factors, declared rather than defaulted.
        n_lookahead: periods solved together at each step; only the first is
            realised.  1 reproduces the market as it ran before this parameter
            existed, bitwise.
        window_demand: what the operator is assumed to see for the periods after
            the one being cleared.  ``"forecast"`` is the day-ahead hourly
            forecast held across the half-hours it covers -- what a real
            operator has, and the default.  ``"actual"`` hands the solve the
            realised demand, which is perfect foresight: it is **an upper bound
            on what look-ahead can buy, not a policy**, and any figure taken
            from it has to be reported as such.  The period being cleared always
            uses realised demand under both.
        terminal_demand: what the window is padded with once it reaches the end
            of the day, since an episode never crosses into the next one.
            ``"persist"`` repeats the last period's realised demand;
            ``"forecast"`` repeats the day-ahead forecast for the last hour.
            The two differ by that hour's forecast error, which can reach
            several thousand MW.  `"persist"` is the default because it needs
            no forecast; `"forecast"` exists so the cost of this modelling
            choice can be measured rather than disclosed as a caveat.
        monitored_lines: which line limits both of this market's clearings
            carry.  ``None``, the default, enforces every line on the dense KKT
            route -- bit-for-bit the market as it ran before this argument
            existed.  **This market builds two operators**: the clearing each
            step calls and the boundary that produces the opening dispatch, so
            the value goes to both and the pair is checked below.  The
            effective set is published as ``spec["clearing"]`` and
            ``spec["boundary_clearing"]``; a driver stamps what it reads there
            rather than what it asked for.
    """
    if window_demand not in ("forecast", "actual"):
        raise ValueError(f"window_demand must be 'forecast' or 'actual', "
                         f"got {window_demand!r}")
    if terminal_demand not in ("persist", "forecast"):
        raise ValueError(f"terminal_demand must be 'persist' or 'forecast', "
                         f"got {terminal_demand!r}")
    if markup_max is None:
        raise ValueError("markup_max has no safe default: it "
                         "bounds how far an action can move the price")
    if not jax.config.jax_enable_x64:
        raise RuntimeError("the clearing requires float64: set "
                           "jax.config.update('jax_enable_x64', True)")

    period_hours = 1.0 / PERIODS_PER_HOUR
    clear, cspec = make_rt_clearing(case, n_segments=n_segments,
                                    cap_scale=cap_scale, ramp_scale=ramp_scale,
                                    period_hours=period_hours, max_iter=max_iter,
                                    n_lookahead=n_lookahead,
                                    monitored_lines=monitored_lines,
                                    lowrank_free=lowrank_free, lu_batching=lu_batching,
                                    reg_coef=reg_coef, stop_tol=stop_tol)
    bspec = {}
    boundary, _boundary_offer = make_boundary(case, n_segments=n_segments,
                                              cap_scale=cap_scale,
                                              period_hours=period_hours,
                                              max_iter=max_iter,
                                              monitored_lines=monitored_lines,
                                              spec_out=bspec,
                                              lowrank_free=lowrank_free, lu_batching=lu_batching,
                                              reg_coef=reg_coef, stop_tol=stop_tol)
    #: **Both operators, read back off what each returned, never off the
    #: argument.**  The failure this catches leaves no trace anywhere else: a
    #: `monitored_lines` that reached the step clearing but not the boundary
    #: opens every episode at the dense route's dispatch and then clears it on
    #: the low-rank one, and every array still comes back well-formed.
    #: Compared as lists so `None` and an index array cannot
    #: compare equal by broadcasting.
    _eff = [None if s["monitored_lines"] is None
            else [int(i) for i in np.asarray(s["monitored_lines"])]
            for s in (cspec, bspec)]
    if _eff[0] != _eff[1]:
        raise ValueError(
            f"the step clearing was built on monitored_lines={_eff[0]} but the "
            f"boundary on {_eff[1]}; the episode would open at one route's "
            f"vertex and be cleared on the other")
    if str(cspec["kkt_route"]) != str(bspec["kkt_route"]):
        raise ValueError(
            f"the step clearing took the {cspec['kkt_route']} KKT route but "
            f"the boundary took {bspec['kkt_route']}")
    #: **The IPM flags are the same shape of defect and need the same guard.**
    #: Before this, `reg_coef` and `stop_tol` reached the step clearing and not
    #: the boundary, while `ipm_reg_coef` / `ipm_stop_tol` in every product were
    #: read off the step operator alone -- so the stamp said the recipe was in
    #: effect while the opening dispatch was solved on the defaults.  Nothing
    #: else records it: the boundary's own `p_prev` comes back well-formed
    #: either way (the consequence was measured on `case813nem` rated, 13 days
    #: every 30: the default boundary's `dual_residual` stays under 3.2e-9 and
    #: `|dp_prev|` under 5.3e-8 MW, because the opening solve is a truthful
    #: zero-ramp clearing and does not reach the fixed point).  **The numbers
    #: were right and the stamp was wrong**, which is the harder half to notice.
    for _k in ("reg_coef", "stop_tol"):
        if cspec.get(_k) != bspec.get(_k):
            raise ValueError(
                f"the step clearing was built with {_k}={cspec.get(_k)} but the "
                f"boundary with {bspec.get(_k)}; the stamp would name one "
                f"operator's value while the other opened the episode")
    settle, money_balance = make_settlement(case, unit_to_agent=unit_to_agent,
                                            period_hours=period_hours,
                                            cap_scale=cap_scale)
    offer_map, aspec = make_offer_map(case, n_segments, 1, kind="markup",
                                      markup_max=markup_max)
    truthful = truthful_action(case, n_segments, 1, kind="markup")

    p_min = np.asarray(case.unit_p_min, np.float64)
    p_max = np.asarray(case.unit_p_max, np.float64)
    unit_bus = np.asarray(case.unit_node_idx, np.int64)
    n_units, n_buses = len(p_min), int(case.n_nodes)
    _width, seg_cost = segment_costs(case, n_segments)
    # registered ramp rates are a fraction of `p_max` per hour rather than MW
    # (`case_data.py`), so the products below are MW **per period**: they halve
    # with `delta` while the half-hourly demand step they have to follow does not
    ramp_up = np.asarray(case.unit_ramp_up, np.float64) * p_max * period_hours * ramp_scale
    ramp_dn = np.asarray(case.unit_ramp_down, np.float64) * p_max * period_hours * ramp_scale

    if unit_to_agent is None:
        unit_to_agent = np.arange(n_units)
    unit_to_agent = np.asarray(unit_to_agent, np.int64)
    n_agents = int(unit_to_agent.max()) + 1
    agent_of = jnp.asarray(unit_to_agent)
    seg = lambda x: jax.ops.segment_sum(x, agent_of, num_segments=n_agents)

    # the static observation block, one row per unit: two capacity limits, K
    # segment costs, two ramp limits in MW per period
    static = np.concatenate([p_min[:, None], p_max[:, None], seg_cost,
                             ramp_up[:, None], ramp_dn[:, None]], 1)
    # one row per agent, averaged over the units it owns rather than summed, so
    # the row keeps the units of the quantities it describes.  Under the
    # default of one agent per unit the sum and the division are the identity.
    static_agent = np.zeros((n_agents, static.shape[1]))
    np.add.at(static_agent, unit_to_agent, static)
    counts = np.bincount(unit_to_agent, minlength=n_agents)[:, None]
    static_agent = static_agent / np.maximum(counts, 1)

    # exogenous series, expanded onto real-time periods once
    day_index = np.asarray(position["day_index"], np.int64)
    n_days = len(day_index)
    P = n_days * T_RT
    params_arrays = dict(
        demand_actual=np.asarray(demand_actual_hh, np.float64)[day_index].reshape(P),
        demand_forecast=_expand(
            np.asarray(demand_forecast_hourly, np.float64)[day_index], axis=1).reshape(P),
        q_da=np.concatenate([_expand(position["q_da"][d].T) for d in range(n_days)]),
        lmp_da=np.concatenate([_expand(position["lmp_da"][d]) for d in range(n_days)]),
        s_da=np.concatenate([_expand(position["s_da"][d]) for d in range(n_days)]),
        d_da=np.concatenate([_expand(position["d_da"][d]) for d in range(n_days)]),
        u_da=np.concatenate([_expand(position["u"][d].T) for d in range(n_days)]),
        day_of=np.repeat(np.arange(n_days), T_RT),
    )

    unit_bus_j = jnp.asarray(unit_bus)
    static_j = jnp.asarray(static_agent, jnp.float32)

    def get_obs(state, params):
        """Exhaustive: every column is the agent's own or a public signal."""
        t = state.cursor
        own = lambda x: jnp.asarray(x, jnp.float32)
        per_unit = jnp.stack([
            state.p_prev,                                   # own dispatch, previous
            params.u_da[t], params.q_da[t],                 # own commitment, schedule
            params.lmp_da[t][unit_bus_j],                   # own bus, day-ahead price
            state.award_prev,                               # own award, previous
            state.lmp_prev[unit_bus_j],                     # own bus, previous price
        ], 1)
        # per-unit rows summed onto the agent that owns them; the identity under
        # one agent per unit
        block = jnp.zeros((n_agents, per_unit.shape[1])).at[agent_of].add(per_unit)
        # position within the market day as a sin/cos pair, so the last period
        # of the day is adjacent to the first rather than 47 apart
        angle = 2.0 * jnp.pi * (state.cursor % T_RT) / T_RT
        # The divisor here and the one on `profit_prev` below are fixed
        # constants, not a running normalisation: demand arrives in MW and
        # profit in \$, and a constant keeps the observation a function of the
        # state alone rather than of the episode's history.
        public = jnp.stack([
            params.demand_forecast[t] / 1e4,
            (params.episode_len - state.step_in_episode).astype(jnp.float32),
            jnp.sin(angle), jnp.cos(angle),
        ])
        return jnp.concatenate([
            static_j, own(block), state.profit_prev[:, None] / 1e3,
            jnp.broadcast_to(own(public)[None, :], (n_agents, 4)),
        ], 1).astype(jnp.float32)

    # `obs_dim` is **derived by evaluating `get_obs`**, never written down: a
    # hard-coded width can drift from the concatenation, and an arithmetic
    # restatement of the concatenation is the same mistake one step removed.
    def _probe_obs_dim() -> int:
        """Width of one observation row, read off an abstract evaluation.

        `jax.eval_shape` traces `get_obs` on a zero state and a dummy
        `EnvParams` and returns its shape without executing it, so the number
        published in `spec` is the concatenation's own width and not a second
        statement of it.
        """
        zero = RealTimeState(
            cursor=jnp.zeros((), jnp.int32), step_in_episode=jnp.zeros((), jnp.int32),
            p_prev=jnp.zeros((n_units,), jnp.float32),
            lmp_prev=jnp.zeros((n_buses,), jnp.float32),
            award_prev=jnp.zeros((n_units,), jnp.float32),
            profit_prev=jnp.zeros((n_agents,), jnp.float32))
        dummy = EnvParams(learner_mask=jnp.ones((n_agents,), bool), episode_len=1,
                          **{k: jnp.asarray(v, jnp.int32 if k == "day_of" else jnp.float32)
                             for k, v in params_arrays.items()})
        return int(jax.eval_shape(get_obs, zero, dummy).shape[1])

    def make_params(episode_len: int, learner_mask=None) -> EnvParams:
        """Bind the exogenous series to one episode length and population.

        The series themselves are fixed when the environment is built; what is
        chosen here is how many periods an episode runs for and which agents
        learn.  ``episode_len`` is bounded by the 48 periods of a market day
        because an episode does not cross a day boundary.

        ``learner_mask`` defaults to every agent learning; a mask with a
        single ``True`` is the single-agent case, every other agent bidding
        its true cost.  ``step`` substitutes the truthful action for a
        non-learner's row before the offer map runs, so there is one
        code path: the baseline is a point of the action space rather than an
        offer put in after the map.  The mask is per agent and the action is
        per unit, so it is gathered along ``unit_to_agent`` first -- the two
        axes coincide only when that map is the identity.
        """
        if not 1 <= episode_len <= T_RT:
            raise ValueError(f"episode_len must be in 1..{T_RT}: v1 does not cross "
                             f"a day boundary, got {episode_len}")
        mask = (jnp.ones((n_agents,), bool) if learner_mask is None
                else jnp.asarray(learner_mask, bool))
        leaves = {k: jnp.asarray(v, jnp.int32 if k == "day_of" else jnp.float32)
                  for k, v in params_arrays.items()}
        return EnvParams(learner_mask=mask, episode_len=int(episode_len), **leaves)

    def _open(cursor, params):
        """Construct `p_prev` for a first step by clearing that period alone."""
        u = jnp.asarray(params.u_da[cursor], jnp.float64)[:, None]
        d = jnp.asarray(params.demand_actual[cursor], jnp.float64)[None]
        p_prev, _ = boundary(u, d)
        return p_prev

    def reset(key, params):
        """Start at period 0 of a day drawn from the fixture window."""
        day = jax.random.randint(key, (), 0, n_days, dtype=jnp.int32)
        cursor = day * T_RT
        state = RealTimeState(
            cursor=cursor, step_in_episode=jnp.zeros((), jnp.int32),
            p_prev=_open(cursor, params).astype(jnp.float32),
            lmp_prev=jnp.zeros((n_buses,), jnp.float32),
            award_prev=jnp.zeros((n_units,), jnp.float32),
            profit_prev=jnp.zeros((n_agents,), jnp.float32))
        return get_obs(state, params), state

    def step(key, state, action, params):
        """Clear and settle one 30-minute period.

        ``key`` is accepted for signature consistency across the five markets
        and never consumed: once the period is fixed the transition is
        deterministic, since the demand, the commitment and the day-ahead
        position are all exogenous series.  This function does **not** reset --
        a caller who wants the terminal state can have it -- and
        `step_auto_reset` is the wrapper that does.

        Args:
            state: a `RealTimeState`.  Its ``p_prev`` enters the ramp rows of
                the clearing and is the only physical coupling between steps.
            action: the raw markup action, mapped to an energy offer by the
                day-ahead offer map at one period.  A non-learning agent's row
                is replaced by the truthful action before the map runs, per
                ``params.learner_mask``.

        Returns:
            ``(obs, state, reward, costs, done, info)``.  ``obs`` is the
            observation of the successor state and ``state`` is that state.
            ``reward`` is `(n_agents,)` float32, the two-settlement profit in
            \\$, computed from the realised award and never from the offer.
            ``costs`` is `(n_agents, 1)` float32, the single column `COST_NAMES`
            names: the shed energy of the period in MWh, the same value for
            every agent because unserved energy is a property of the period
            rather than of one bidder.  It is the CMDP constraint vector and
            enters no reward; the money it stands for is reported separately as
            ``info["voll_cost"]``.  ``done`` is the time-limit truncation.
            ``info`` carries the solver diagnostics ``mu``, ``dual_residual``
            and ``converged``, the shed quantities ``shed_mwh`` and
            ``voll_cost``, the money split ``revenue_da`` / ``revenue_rt`` /
            ``cost``, and ``terminal_obs``.
        """
        t = state.cursor
        # The decision window is [t, t+N-1], clamped at the end of the day: an
        # episode stays inside one day, and `jit` needs the window's shape
        # fixed, so it is padded rather than shortened.  Every padded period
        # takes `terminal_demand`; the commitment is held at its last value,
        # which is not a demand assumption and has no second candidate.
        #
        # **Only column 0 is realised.**  The padded periods therefore shape how
        # `t` is optimised and never what `t` settles -- which is what keeps the
        # invented tail out of the money.
        win = t + jnp.arange(n_lookahead)
        day_end = (t // T_RT + 1) * T_RT - 1
        idx = jnp.minimum(win, day_end)
        padded = win > day_end

        # A non-learner's row becomes the truthful
        # action and then runs through the same offer map as everyone else.
        # `learner_mask` is per agent while `action` is per unit, so it is
        # gathered onto the unit axis before being broadcast over whatever
        # else the action shape carries.  An all-True mask -- the default and
        # what every caller in this repository passes -- leaves `action`
        # untouched.
        mask = params.learner_mask[agent_of].reshape(
            (-1,) + (1,) * (jnp.ndim(action) - 1))
        action = jnp.where(mask, action, truthful)

        offer = jnp.broadcast_to(offer_map(action).astype(jnp.float64),
                                 (n_units, n_segments, n_lookahead))
        u = jnp.asarray(params.u_da[idx], jnp.float64).T

        actual = jnp.asarray(params.demand_actual[idx], jnp.float64)
        forecast = jnp.asarray(params.demand_forecast[idx], jnp.float64)
        # Three sources, in the order the window is built:
        #   offset 0        the realised demand, always -- it is the period that
        #                   actually clears and settles;
        #   offsets 1..N-1  `window_demand`.  "forecast" is what an operator can
        #                   actually see and is the default; "actual" is perfect
        #                   foresight and is an upper bound, not a policy;
        #   padded offsets  `terminal_demand` (see `make_env`).
        # At N = 1 only the first exists, which is why neither parameter can
        # disturb the bitwise reproduction of the pre-lookahead market.
        ahead = forecast if window_demand == "forecast" else actual
        terminal = forecast if terminal_demand == "forecast" else actual
        is_now = jnp.arange(n_lookahead) == 0
        demand = jnp.where(is_now, actual, jnp.where(padded, terminal, ahead))
        out = clear(offer, u, demand, jnp.asarray(state.p_prev, jnp.float64))

        # the realised period is the first column of everything the window
        # returned; the rest is look-ahead and is discarded here
        award = out["award"][:, :1]
        lmp, shed = out["lmp"][:1], out["shed"][:1]
        q_da = jnp.asarray(params.q_da[t], jnp.float64)[:, None]
        lmp_da = jnp.asarray(params.lmp_da[t], jnp.float64)[None, :]
        # `commitment_status`: the previous period's commitment, so a unit the
        # schedule starts in this period pays its start-up cost once
        prev_u = jnp.asarray(params.u_da[jnp.maximum(t - 1, 0)], jnp.float64)
        money = settle(award, lmp, u[:, :1], prev_u, q_da, lmp_da)

        # `shed` is MW at each bus of the realised period, so the factor is
        # `delta` and the channel is MWh.  The outer `maximum` clamps the
        # slightly negative sum an interior point can leave behind, and
        # `SHED_FLOOR` then drops what is left of a period that shed nothing.
        shed_mwh = jnp.maximum(jnp.sum(shed) * (1.0 / PERIODS_PER_HOUR), 0.0)
        shed_mwh = jnp.where(shed_mwh < SHED_FLOOR, 0.0, shed_mwh)
        costs = jnp.broadcast_to(shed_mwh.astype(jnp.float32),
                                 (n_agents, len(COST_NAMES)))

        #: **Acts only on the two fields that enter the observation; the
        #: market is not changed.**  `lmp_prev` and `profit_prev` are two
        #: columns of `get_obs`, and a clearing that does not converge returns
        #: unbounded duals: measured 2026-09-16 on `case813nem`, under truthful
        #: offers `lmp_prev` reached 2.2e+19, within a few steps `profit_prev`
        #: overflowed float32 to `inf`, so the mean in `observation_statistics`
        #: was non-finite, every normalised observation was NaN, the policy
        #: output NaN, and all 3072 cells of the round failed to converge.
        #: Clipping keeps that quantity out of the observation.
        #:
        #: **It does not repair the price itself.**  `reward`,
        #: `info["revenue_rt"]` and the other settlement quantities still use
        #: the unclipped `money`: reward is the market's settlement profit and
        #: must be computed from the actual awards and this market's own
        #: clearing price, not rewritten by bounds set for feeding a network.
        #: A bad price is still bad; it just no longer turns the whole policy
        #: into NaN through the observation.
        #:
        #: +/-VOLL is this market's own bound, not a new number: the penalty
        #: price of load shedding is VOLL, and a legitimate LMP lies in
        #: [-VOLL, VOLL] (`clearing.py` section 7 and `VOLL_MUST_BE_BELOW` are
        #: the same statement).  So on periods with `|lmp| <= VOLL` the clip is
        #: the identity -- two existing archives were scanned on 2026-09-16;
        #: on 29gb 0/576 cells were out of range (max|lmp| over the whole run
        #: exactly 1.0e+04 = VOLL).
        lmp_obs = jnp.clip(lmp, -VOLL, VOLL)
        #: **`clip` does not remove NaN** (measured: `clip(nan, -VOLL, VOLL)`
        #: is still `nan`), and a non-converged solution can in principle
        #: return NaN, so the observation needs both steps.  The number of
        #: replacements is published on `info`: a price that was replaced
        #: without a trace cannot be told apart from one that was fine.
        #:
        #: **The guard sits only on prices and money, not on energy, and that
        #: was measured.**  Replaying the worst period of the `case813nem` full
        #: window (2025-11-11 period 17, `dual_residual` 2.2542e+290, `mu`
        #: 9.3426e+296), the `award` it returns peaks at 511 MW, and none of
        #: its 7 248 elements is non-finite or above the float32 maximum.
        #: `award` is a box-constrained primal variable that the IPM barrier
        #: keeps inside its box; **what blows up is the dual**.  So the
        #: `p_prev` / `award_prev` observations do not need this replacement,
        #: and `lmp_prev` / `profit_prev` do.
        lmp_replaced = jnp.sum((~jnp.isfinite(lmp)) | (jnp.abs(lmp) > VOLL))
        lmp_obs = jnp.nan_to_num(lmp_obs, nan=0.0, posinf=VOLL, neginf=-VOLL)
        #: The second settlement is only for `profit_prev`: profit is linear
        #: in price, so recomputing it at the clipped price is more honest than
        #: clipping the profit itself -- the latter would stop at a number that
        #: matches no price at all.
        money_obs = settle(award, lmp_obs, u[:, :1], prev_u, q_da, lmp_da)
        nxt = RealTimeState(
            cursor=t + 1, step_in_episode=state.step_in_episode + 1,
            p_prev=award[:, 0].astype(jnp.float32),
            lmp_prev=lmp_obs[0].astype(jnp.float32),
            award_prev=award[:, 0].astype(jnp.float32),
            profit_prev=money_obs["profit"].astype(jnp.float32))
        done = nxt.step_in_episode >= params.episode_len
        #: **`usable` is not `converged`.**  `converged` looks only at `mu`,
        #: and its meaning is pinned by every product on disk.  `usable` is
        #: what a learner asks before putting a sample into a gradient: both
        #: residuals are within tolerance, and the money it produces still
        #: fits in the float32 the learner runs in.
        #:
        #: **Which of the four conjuncts actually binds over the full window
        #: has been measured.**  Over the 17 520 periods of the `case813nem`
        #: market 02 full window under truthful offers: only 1 period has
        #: `mu >= MU_TOL` (2025-11-11 period 17, `mu` 9.3426e+296), and it is
        #: also the period with `dual_residual` 2.2542e+290, i.e. the only one
        #: whose float64 profit in `settle` exceeds 3.4028e+38, becomes `inf`
        #: in float32, and turns into `NaN` when two opposite-signed infs are
        #: added.  **So the 2026-09-17 crash was not "the gate did not see it"
        #: -- `converged` saw it; nobody did anything with it.**  Another 1 292
        #: periods (7.37%) pass `mu` cleanly while `dual_residual` stays at
        #: 5.0e-03; their price gap is about 0.108 $/MWh and does not overflow:
        #: those are what the second conjunct handles, unrelated to the crash.
        #: All four stay, because "`mu` happened to blow up too" is a fact of
        #: this one window, not an identity.
        _f32_max = jnp.finfo(jnp.float32).max
        usable = ((out["mu"] < MU_TOL)
                  & (out["dual_residual"] <= dual_res_tol)
                  & jnp.all(jnp.isfinite(money["reward"]))
                  & jnp.all(jnp.abs(money["reward"]) <= _f32_max))
        info = dict(mu=out["mu"], dual_residual=out["dual_residual"],
                    converged=out["mu"] < MU_TOL,
                    dual_ok=out["dual_residual"] <= dual_res_tol,
                    usable=usable, lmp_replaced=lmp_replaced,
                    shed_mwh=shed_mwh, voll_cost=VOLL * shed_mwh,
                    revenue_da=money["revenue_da"], revenue_rt=money["revenue_rt"],
                    cost=money["cost"],
                    terminal_obs=get_obs(nxt, params))
        return (get_obs(nxt, params), nxt, money["reward"].astype(jnp.float32),
                costs, done, info)

    def step_auto_reset(key, state, action, params):
        """`step` plus a reset on `done`; the wrapper a fixed-length `lax.scan` runs.

        The same ``key`` goes to `reset` unsplit.  That is admissible only
        because `step` consumes no randomness at all, so there is nothing for
        the new episode's start day to correlate with; the split is otherwise
        required of any environment that resets inside its own step
        (`resources.env_base.Environment.step`).

        Returns `step`'s 6-tuple, with `jax.lax.stop_gradient` applied to the
        returned observation and the returned state, which is what the other
        four markets and `resources.env_base.Environment.step_auto_reset` do.
        ``info["terminal_obs"]`` is not stopped, so a learner that bootstraps
        a truncation from it still sees the gradient the barrier removes from
        ``obs``.  On the step where ``done`` fires ``obs`` becomes the
        fresh episode's first observation and ``state`` its initial state;
        ``reward``, ``costs`` and ``info`` are the truncated step's throughout,
        and ``info["terminal_obs"]`` is the observation of the state that step
        really ended in -- what a learner bootstrapping a truncation needs, and
        exactly what ``obs`` is no longer.
        """
        obs, nxt, reward, costs, done, info = step(key, state, action, params)
        r_obs, r_state = reset(key, params)
        pick = lambda a, b: jnp.where(done, a, b)
        state = jax.tree.map(pick, r_state, nxt)
        # `terminal_obs` keeps the observation of the state the episode ended in,
        # which is exactly what `obs` no longer is once `done` fires
        return (jax.lax.stop_gradient(pick(r_obs, obs)),
                jax.lax.stop_gradient(state), reward, costs, done, info)

    env = type("RealTimeEnv", (), dict(
        reset=staticmethod(reset), step=staticmethod(step),
        step_auto_reset=staticmethod(step_auto_reset),
        get_obs=staticmethod(get_obs), make_params=staticmethod(make_params),
        truthful_action=staticmethod(lambda: truthful)))()
    spec = dict(n_agents=n_agents, n_units=n_units, n_buses=n_buses,
                obs_dim=_probe_obs_dim(), action_shape=aspec["shape"],
                # the action map's own spec, carrying `low` and `high`.  The
                # day-ahead market publishes `action=aspec` the same way; the
                # other three publish `action_low`/`_high` directly, and
                # `learning.adapters.unpack_env` reads either form.
                action=aspec, n_days=n_days,
                n_periods=P, periods_per_day=T_RT, period_hours=period_hours,
                cost_names=COST_NAMES, termination="truncation",
                max_iter=max_iter, cap_scale=cap_scale, ramp_scale=ramp_scale,
                n_lookahead=n_lookahead, window_demand=window_demand,
                terminal_demand=terminal_demand,
                #: the two operators' own specs, so a driver stamps the line set
                #: and KKT route it reads off them rather than the flag it
                #: parsed.  The day-ahead market publishes `clearing=cspec` the
                #: same way; `boundary_clearing` has no counterpart there
                #: because that market has no second operator.
                dual_res_tol=float(dual_res_tol),
                clearing=cspec, boundary_clearing=bspec)
    return env, spec
