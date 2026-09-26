"""P2P environment layer: ``EnvState``, ``EnvParams``, ``reset`` and ``step``.

Wires this package's operators -- the action map, the double auction, the
settlement and the state-of-charge advance -- into the standard 6-tuple step
interface.  Nothing here computes a market quantity itself: every number in
``reward``, ``costs`` and ``info`` comes out of an operator with its own
acceptance tests, and this module only routes them.

    reset, step, step_auto_reset, spec = make_p2p_env(N, PI_EXP, PI_RET, 0.5)
    params = make_p2p_params(p_pv, load, battery, kappa, learner_mask, episode_len)
    obs, state = reset(key, params)
    obs, state, reward, costs, done, info = step(key, state, action, params)

Behaviour of ``step``, in the order it applies:

* The key splits into ``transition_key`` and ``reset_key`` before anything
  uses it.  The transition consumes no randomness today -- the exogenous
  series are data and ties break by index -- so ``transition_key`` is idle;
  the split still keeps a reset from ever sharing randomness with the step
  that triggered it.
* The action is cast to float32 on entry, so the environment fixes where the
  boundary rounds rather than the caller's dtype.
* ``learner_mask`` selects, before the action map, between the policy's
  action and ``baseline_action``: battery still, truthful price, the side
  read off the net position at a zero battery command.  Non-learners are
  actions, not a second code path.
* ``done`` is a time limit and is settled rather than truncated, so nothing
  may be bootstrapped across it: the stock left in the battery at that
  boundary is paid for by a terminal leg added to ``reward`` on the step
  where ``done`` fires, which makes the episode self-contained in value even
  though the underlying process keeps going.  ``spec["termination"] =
  "terminal"`` states that once.  ``info["terminal_obs"]`` is the observation
  of the state the episode would have entered had it continued, and
  deliberately excludes the terminal leg -- bootstrapping from it while also
  taking that payment would price the stock twice.
* Auto-reset merges with ``jnp.where``.  ``reset`` is a gather -- draw a
  start, read ``initial_soc``, zero the previous-result fields -- so
  evaluating it on every step is free, and ``lax.cond`` would run both
  branches under ``vmap`` anyway.
* No x64 guard: this market is bit-identical with ``jax_enable_x64`` on and
  off, unlike the two markets that solve linear programs, so the invariant is
  checked by the L0 dtype assertions rather than by a construction-time raise.

The previous-result fields are zero at reset by convention, not estimate: an
invented mid price would misrepresent the market, and zero sits outside the
tariff bracket, so an agent can tell the first step of an episode apart.
"""
import warnings
from typing import Callable, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from powermarketjax.resources.battery import BatteryBundle

from .action import make_action_map, make_soc_advance
from .baseline_pricing import make_baseline_pricing
from .clearing import make_clearing
from .settlement import make_settlement, make_terminal_settlement

#: Observation width: 5 static + 3 current + 3 previous own + 2 public +
#: 2 calendar.
OBS_DIM = 15


@struct.dataclass
class P2PState:
    """Everything that crosses a period boundary."""
    cursor: chex.Array            # int32   ()     index into the exogenous series
    step_in_episode: chex.Array   # int32   ()     counts toward episode_len
    soc: chex.Array               # float32 (N,)   the only physical carry
    price_prev: chex.Array        # float32 ()     previous clearing price
    volume_prev: chex.Array       # float32 ()     previous traded volume
    net_prev: chex.Array          # float32 (N,)   previous own net position
    award_prev: chex.Array        # float32 (N,)   signed: award_sell - award_buy
    profit_prev: chex.Array       # float32 (N,)   previous own profit


@struct.dataclass
class P2PParams:
    """Sweepable and data-carrying quantities; built by ``make_p2p_params``
    only, which owns the dtype/axis/window contracts."""
    p_pv: chex.Array              # float32 (n_periods, N)  exogenous PV
    load: chex.Array              # float32 (n_periods, N)  exogenous own load
    battery: BatteryBundle        # parameter container only, never .step()
    kappa: chex.Array             # float32 (N,)   degradation price, may be swept
    learner_mask: chex.Array      # bool    (N,)   selects policy vs baseline action
    episode_len: chex.Array       # int32   ()     episode length in periods


def baseline_action(p_pv_t: chex.Array, load_t: chex.Array) -> chex.Array:
    """The non-learner's action: battery still, truthful price.

    With a zero battery command the net position is exactly ``p_pv - load``,
    and its sign decides the side: a seller asks ``pi_exp``, which the action
    map's affine price map hits exactly at ``alpha_price = -1``, and a buyer
    bids ``pi_ret`` at ``+1``.  A balanced participant submits zero quantity
    either way, so the -1 it gets here is inconsequential.
    """
    side = jnp.where(p_pv_t - load_t >= 0.0, -1.0, 1.0).astype(jnp.float32)
    return jnp.stack([jnp.zeros_like(side), side], axis=1)


def make_p2p_params(
    p_pv: np.ndarray,
    load: np.ndarray,
    battery: BatteryBundle,
    kappa: np.ndarray,
    learner_mask: np.ndarray,
    episode_len: int,
) -> P2PParams:
    """Build and validate ``P2PParams`` (setup time, numpy, may raise).

    The dtype, axis and window contracts live here so that no caller has to
    pass the right dtypes by luck: series and per-agent arrays go to float32,
    the mask to bool, the window is checked against the series length, and
    every axis is checked against the battery's device axis.  ``battery``
    itself is not re-validated -- ``make_battery_bundle`` already rejected
    degenerate devices at construction, and a second copy of that check would
    be the defensive duplication this repository forbids.
    """
    p_pv = np.asarray(p_pv, np.float32)
    load = np.asarray(load, np.float32)
    kappa = np.asarray(kappa, np.float32)
    learner_mask = np.asarray(learner_mask, bool)

    n = int(battery.n_devices)
    if p_pv.ndim != 2 or p_pv.shape[1] != n:
        raise ValueError(f"p_pv must be (n_periods, {n}), got {p_pv.shape}")
    if load.shape != p_pv.shape:
        raise ValueError(f"load must match p_pv {p_pv.shape}, got {load.shape}")
    if kappa.shape != (n,):
        raise ValueError(f"kappa must be ({n},), got {kappa.shape}")
    if learner_mask.shape != (n,):
        raise ValueError(f"learner_mask must be ({n},), got {learner_mask.shape}")

    n_periods = p_pv.shape[0]
    episode_len = int(episode_len)
    if not 1 <= episode_len <= n_periods:
        raise ValueError(
            f"episode_len must lie in [1, {n_periods}], got {episode_len}: the "
            f"legal starts are [0, n_periods - episode_len] and that interval "
            "must be non-empty")
    if episode_len < 2:
        # legal, but a one-step episode leaves nothing for terminal_obs to
        # bootstrap from
        warnings.warn("episode_len < 2: every step is the truncation step",
                      stacklevel=2)

    return P2PParams(
        p_pv=jnp.asarray(p_pv), load=jnp.asarray(load), battery=battery,
        kappa=jnp.asarray(kappa), learner_mask=jnp.asarray(learner_mask),
        episode_len=jnp.asarray(episode_len, jnp.int32))


def make_p2p_env(
    n_agents: int,
    pi_exp: float,
    pi_ret: float,
    period_hours: float,
    *,
    pricing_rule: str = "award-consistent-midpoint",
) -> Tuple[Callable, Callable, Callable, Dict]:
    """Build the P2P environment for one population and tariff pair.

    Returns ``(reset, step, step_auto_reset, spec)``:

        reset(key, params)                          -> (obs, state)
        step(key, state, action, params)            -> 6-tuple
        step_auto_reset(key, state, action, params) -> same, obs and state
                                                       behind stop_gradient

    ``spec`` carries the static shapes (``obs_dim``, ``action_shape``,
    ``cost_names``, ``termination``) plus ``get_obs`` and ``baseline_action``
    for tests that need to recompute an observation or a baseline without
    reaching into the closure.  The tariff pair and the period length are
    closed over and held constant over an episode; the
    ``0 <= pi_exp < pi_ret`` check happens only here, at construction.

    ``pricing_rule`` is handed to `make_clearing` unchanged; it is keyword-only,
    its default is the market's own rule, and the two alternatives are the
    diagnostics that module's docstring describes.  What took effect is read
    back from the clearing's own ``spec`` into ``spec["pricing_rule"]`` and
    ``spec["is_market_pricing_rule"]`` rather than restated from the argument,
    so a caller can show that the operator received it and not merely that the
    driver sent it.
    """
    act_map, _ = make_action_map(n_agents, pi_exp, pi_ret, period_hours)
    clear, clear_spec = make_clearing(n_agents, pi_exp, pi_ret,
                                      pricing_rule=pricing_rule)
    settle = make_settlement(pi_exp, pi_ret)
    settle_terminal = make_terminal_settlement(pi_exp)
    advance = make_soc_advance(period_hours)
    price_baselines, _ = make_baseline_pricing(pi_exp, pi_ret)

    # periods per day for the calendar encoding; a step is one period here, so
    # env_base.time_features (which encodes time of day per step) happens to be
    # the same formula, but the phase here is fixed on `cursor` explicitly
    steps_per_day = 24.0 / period_hours

    def _get_obs(state: P2PState, params: P2PParams) -> chex.Array:
        """The observation as a pure function of ``(state, params)``.

        The index is clamped to the last row for exactly one reachable case:
        the terminal observation of an episode whose start sat on the upper
        edge of the legal window reads row ``n_periods``, one past the data,
        since every in-episode read otherwise stays within ``n_periods - 1``.
        The clamp is explicit here rather than left to jnp's own
        out-of-range indexing behaviour.
        """
        battery = params.battery
        idx = jnp.minimum(state.cursor, params.p_pv.shape[0] - 1)
        phase = 2.0 * jnp.pi * (state.cursor.astype(jnp.float32) / steps_per_day)
        broadcast = lambda x: jnp.full((n_agents,), x, jnp.float32)
        return jnp.stack([
            battery.power_max, battery.capacity,                 # static
            battery.eta_charge, battery.eta_discharge,
            params.kappa,
            state.soc, params.p_pv[idx], params.load[idx],       # current period
            state.net_prev, state.award_prev, state.profit_prev,  # own, previous
            broadcast(state.price_prev), broadcast(state.volume_prev),  # public
            broadcast(jnp.sin(phase)), broadcast(jnp.cos(phase)),  # calendar
        ], axis=1).astype(jnp.float32)

    def reset(key: chex.PRNGKey, params: P2PParams):
        """Draw a start uniformly over the legal window; everything else is a
        gather.  Previous-result fields are zero by convention (module doc)."""
        n_periods = params.p_pv.shape[0]
        cursor = jax.random.randint(key, (), 0,
                                    n_periods - params.episode_len + 1)
        zeros_n = jnp.zeros((n_agents,), jnp.float32)
        state = P2PState(
            cursor=jnp.asarray(cursor, jnp.int32),
            step_in_episode=jnp.asarray(0, jnp.int32),
            soc=jnp.asarray(params.battery.initial_soc, jnp.float32),
            price_prev=jnp.float32(0.0), volume_prev=jnp.float32(0.0),
            net_prev=zeros_n, award_prev=zeros_n, profit_prev=zeros_n)
        return _get_obs(state, params), state

    def step(key: chex.PRNGKey, state: P2PState, action: chex.Array,
             params: P2PParams):
        """Advance the market by one ``period`` and return the 6-tuple.

        One ``step`` is one ``period`` here, unlike the day-ahead market where
        a step is a market day of 24 of them.  ``action`` is ``(n_agents, 2)``
        -- battery command and price command -- and is cast to float32 at
        entry, so the environment and not the caller's dtype decides where
        the boundary rounds.

        Returns ``(obs, state, reward, costs, done, info)``:

            obs     (n_agents, 15) float32  already past the auto-reset
                                            merge, so at ``done`` it is the
                                            first observation of the next
                                            episode
            state   ``P2PState``            the next state, merged the same way
            reward  (n_agents,)   float32   profit for the period, in the
                                            currency of the tariff pair, plus
                                            the terminal leg on the step where
                                            ``done`` is set
            costs   (n_agents, 1) float32   the CMDP constraint vector, its one
                                            channel being ``clip``: how far the
                                            state of charge fell short of
                                            delivering the battery command.
                                            Nobody pays it, so it never enters
                                            ``reward``; the degradation a
                                            participant does pay is in
                                            ``cost`` inside `settlement`
            done    bool scalar             the time limit, one flag for the
                                            whole population.  It is settled
                                            rather than truncated
                                            (``spec["termination"]``), so
                                            nothing may be bootstrapped
                                            across it
            info    dict                    ``clearing_price``,
                                            ``traded_volume``, the two
                                            ``price_interval`` endpoints,
                                            ``terminal_stock_value``,
                                            ``terminal_obs``, and the six
                                            scalars of `baseline_pricing`

        ``key`` is split before anything reads it and the reset half is never
        the transition half; the transition half is idle today (module doc).
        Auto-reset is inside this function, so ``step`` alone drives a
        fixed-length ``lax.scan`` and `step_auto_reset` adds only
        ``stop_gradient``.
        """
        # split before any use; the transition consumes no randomness today
        # (module doc), so transition_key is deliberately idle
        transition_key, reset_key = jax.random.split(key)
        del transition_key
        action = jnp.asarray(action, jnp.float32)

        p_pv_t, load_t = params.p_pv[state.cursor], params.load[state.cursor]
        action = jnp.where(params.learner_mask[:, None], action,
                           baseline_action(p_pv_t, load_t))

        sub = act_map(action, state.soc, p_pv_t, load_t, params.battery)
        out = clear(sub["price"], sub["q_sell"], sub["q_buy"])
        money = settle(sub["q_sell"], sub["q_buy"], out["award_sell"],
                       out["award_buy"], out["clearing_price"], params.kappa,
                       sub["throughput"])

        next_state = P2PState(
            cursor=state.cursor + 1,
            step_in_episode=state.step_in_episode + 1,
            soc=advance(state.soc, sub["p_signed"], params.battery),
            price_prev=out["clearing_price"],
            volume_prev=out["traded_volume"],
            net_prev=sub["net_position"],
            award_prev=out["award_sell"] - out["award_buy"],
            profit_prev=money["profit"])
        done = next_state.step_in_episode >= params.episode_len

        # The terminal leg.  The episode is cut by a time limit, not by an
        # absorbing state, so the stock left in the battery is still worth
        # something; valuing it at zero would pay for ending empty.  It fires on
        # the truncated step only, and it is added to the reward rather than
        # folded into `money` because it prices a stock and not an award.
        # It is deliberately absent from `profit_prev`, and so from
        # `terminal_obs`.  That observation exists to be the state the episode
        # would have entered had it continued, and in that continuation no
        # terminal settlement happens; carrying the residual into it would let a
        # bootstrap count the stock once in the reward and again in the value.
        residual = jnp.where(done, settle_terminal(next_state.soc,
                                                   params.battery), 0.0)

        # evaluating reset unconditionally is free here -- it is a gather --
        # and lax.cond would run both branches under vmap anyway
        _, fresh = reset(reset_key, params)
        merged = jax.tree_util.tree_map(
            lambda nxt, new: jnp.where(done, new, nxt), next_state, fresh)

        obs = _get_obs(merged, params)
        # the true successor observation, which the merge above overwrites at
        # done; a truncated episode bootstraps from this, not from obs
        terminal_obs = jnp.where(done, _get_obs(next_state, params), obs)

        reward = money["reward"] + residual
        costs = sub["clip"][:, None]
        info = dict(clearing_price=out["clearing_price"],
                    traded_volume=out["traded_volume"],
                    terminal_stock_value=residual,
                    # The only real resource cost among the modelled
                    # participants: the auction leg is a transfer that nets to
                    # zero across them.  Whether that makes it the whole of this
                    # market's system cost is a judgement about the boundary,
                    # argued at `make_settlement`.  Both are published, not one,
                    # because `kappa` is per agent and may be swept, so a
                    # throughput alone cannot be priced by a reader and a price
                    # alone cannot say how much was cycled.
                    # `degradation_cost` comes back from `settle`, which already
                    # formed the product for `cost`; recomputing it here would
                    # be a second implementation of that product, and the two
                    # could then disagree about which `kappa` was in force.
                    throughput=sub["throughput"],
                    degradation_cost=money["degradation_cost"],
                    price_interval_lo=out["price_interval_lo"],
                    price_interval_hi=out["price_interval_hi"],
                    terminal_obs=terminal_obs,
                    **price_baselines(sub["q_sell"], sub["q_buy"]))
        return obs, merged, reward, costs, done, info

    def step_auto_reset(key, state, action, params):
        """`step` with the gradient stopped at the episode boundary.

        The reset itself is already inside ``step`` here, so this wrapper adds
        only ``stop_gradient`` on ``obs`` and ``new_state``, which is what keeps
        a gradient from flowing across a boundary in a ``lax.scan`` rollout.
        ``reward``, ``costs``, ``done`` and ``info`` pass through untouched, so
        ``info["terminal_obs"]`` is not stopped either.  Same signature and same
        6-tuple as ``step``; a caller that wants the terminal state rather than
        the fresh one reads it from that ``info`` entry, since neither function
        returns it in ``state``.
        """
        obs, new_state, reward, costs, done, info = step(key, state, action,
                                                         params)
        obs = jax.lax.stop_gradient(obs)
        new_state = jax.lax.stop_gradient(new_state)
        return obs, new_state, reward, costs, done, info

    spec = dict(n_agents=n_agents, obs_dim=OBS_DIM,
                action_shape=(n_agents, 2), action_low=-1.0, action_high=1.0,
                costs_dim=1, cost_names=("clip",), termination="terminal",
                pi_exp=float(pi_exp), pi_ret=float(pi_ret),
                period_hours=float(period_hours), dtype=jnp.float32,
                pricing_rule=clear_spec["pricing_rule"],
                is_market_pricing_rule=clear_spec["is_market_pricing_rule"],
                get_obs=_get_obs, baseline_action=baseline_action)
    return reset, step, step_auto_reset, spec
