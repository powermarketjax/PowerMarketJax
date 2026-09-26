"""The day-ahead market as a Markov game: one step is one market day.

    env, spec = make_env(case, fixture, demand, n_segments=1, kind="markup",
                         markup_max=..., cap_scale=0.60, ramp_scale=1.00)
    params = env.make_params(episode_len=7)
    obs, state = env.reset(key, params)
    obs, state, reward, costs, done, info = env.step(key, state, action, params)

Agents are an **array axis**, not dictionary keys: ``reward`` is ``(N,)``,
``action`` is ``(N, ...)``, ``obs`` is ``(N, obs_dim)`` and ``N`` is fixed at
build time.  One generating company per unit, so ``N`` is the unit count and the
action axis of `action` is also the unit axis of `clearing`.  This module does
**not** inherit `resources/env_base.py`, which is single-agent; it keeps that
module's two conventions -- pure functions with the state in a pytree, and
`step_auto_reset` -- and nothing else.

**The commitment is solved from the offers.**  ``step`` runs all three clearing
stages: the relaxed unit commitment (`relax.py`), the rounding of the relaxed
commitment to integers, and the fixed-commitment dispatch (`clearing.py`) whose
duals are the prices.  So who runs is a function of what the agents bid: a unit
can avoid a loss-making day by bidding differently, and start-up cost recovery
can drive a commitment strategy.  The fixture supplies the day window and the
initial boundary only.

One consequence decides the observation.  The commitment of the day being
cleared **does not exist at the moment the offers are made**, so it cannot appear
in the observation.  What an agent has instead is what predicts it: its own
costs, the demand forecast, and its run lengths at the day boundary.

**`step` does not consume its key.**  The transition is deterministic: demand is
a realised data series and both solves are deterministic given the offers, so
nothing in a market day is sampled.  `reset` is where the key is used, to pick
the start day of the episode, and `step_auto_reset` passes it there.

`step` returns the six-tuple with `costs` separate from `reward`.  `costs`
carries two feasibility quantities -- shed energy and the minimum up-/downtime
violation count -- and nothing else.  The remaining per-day diagnostics (the VOLL
cost, the congested line-periods, the solver's `mu` and dual residual, and the
relaxation's `integrality_gap`) are observations rather than constraints and go
to `info`.

Two of those `info` entries are not decoration.  Both operators return
**garbage rather than an error** on an infeasible input, with `mu` rising many
orders of magnitude while the prices stay finite, so the two `mu` values and the
`converged` flag derived from them are the only signal a caller has that the
reward is meaningful.  The `integrality_gap` is computed from the solve: how far
the relaxed commitment sat from {0, 1}, which is exactly how far the rounding
step had to move it.

Precision matters: the clearing runs in float64 and every `DayAheadState` array
is explicitly float32, so each constructor below writes its dtype -- under
`jax_enable_x64` an omitted dtype silently becomes float64 and the state's
pytree structure then depends on a global flag.

Observations are in **physical units**, unnormalised.  A scale that maps profit
in dollars and power in megawatts onto a common range is a property of the
learning setup rather than of the market, so normalisation belongs in
`wrappers`.
"""
from typing import Callable, Dict, NamedTuple, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import struct

from .action import make_offer_map, truthful_action
from .clearing import VOLL, make_clearing, segment_costs
from .relax import make_relax, round_commitment
from .settlement import make_settlement

#: Convergence threshold for **step 3**, the dispatch and the prices.  A converged
#: solve leaves `mu` near 1e-11; an infeasible one leaves it near 1e265 with
#: prices to match, so the threshold is not delicate.  What it guards is **dual
#: accuracy**, because the LMP this market settles on is a dual.
MU_TOL = 1e-8

#: `dual_residual` from both solves reaches `info` and is deliberately not part
#: of `converged`.  The rounding step between the relaxed commitment and the
#: reward is a wide absorber of that residual -- it only consumes `sign(u)` --
#: so `mu` alone is the meaningful gate here; an operator with no such absorber
#: between its solve and its reward would need the residual in its gate too.
#:
#: Convergence threshold for **step 1'**, the relaxed commitment.  Numerically the
#: same value as `MU_TOL` but a separate constant, because it guards a different
#: property: `relax.MAX_ITER` is calibrated against reproducing the discrete
#: commitment cell for cell, not against dual accuracy.  One shared constant
#: would let a recalibration of either criterion move the other silently.
RELAX_MU_TOL = 1e-8

#: A line counts as congested when its flow is within this many megawatts of the
#: rating.  Flows on a binding line sit at the rating to 1e-6 MW.
CONGESTION_TOL = 1e-3

#: `costs` in this order.  The first is a system quantity and is reported
#: identically to every agent: shed load is a property of the day, not of one
#: participant, and there is no defensible way to attribute megawatt-hours of
#: unserved energy to a bidder.  The second is per-agent, over its own units.
COST_NAMES = ("shed_energy_mwh", "min_up_down_violation")


@struct.dataclass
class DayAheadState:
    """The state after a market day.

    The first six fields are the quantities that constrain the next clearing,
    and each is used by it -- `p_final` is the `p_init` of the next day's ramp
    rows, `commitment_status` decides whether period 1 pays to start, and the two
    run lengths are what the minimum up-/downtime violation count is measured
    against.

    The last three are not physical carry.  An agent has to observe its cleared
    schedule and profit from the previous day, and the observation has to be a
    function of the state alone or it changes meaning across an auto-reset, so the
    previous day's settlement is carried here as well.
    """
    cursor: chex.Array               # int32, position in the day series
    step_in_episode: chex.Array      # int32, steps taken in this episode
    commitment_status: chex.Array     # (n_units,) float32, end-of-day commitment
    p_final: chex.Array              # (n_units,) float32, final-period output
    up_time: chex.Array              # (n_units,) int32, periods on at the boundary
    down_time: chex.Array            # (n_units,) int32, periods off at the boundary
    lmp_prev: chex.Array             # (T, n_buses) float32
    award_prev: chex.Array           # (n_units, T) float32
    profit_prev: chex.Array          # (n_agents,) float32


@struct.dataclass
class EnvParams:
    """Per-day data and the agent population.

    The arrays are indexed by `cursor`, which runs over the days the
    commitment fixture covers rather than over the whole demand series, so the
    two cannot fall out of step.

    The fixture supplies the day window and the **initial** boundary only.  It
    does not supply the commitment itself: that is solved inside `step` from the
    offers, so the four `boundary_*` arrays are what this environment keeps of it.

    `learner_mask` fixes part of the population to truthful bidding: it is a
    mask, not a second interface, and the action of a non-learning agent runs
    through the same offer map as everyone else.

    `episode_len` is static -- it changes the pytree structure, not a leaf -- and
    has no default, because a silent default would pick an episode length for
    the caller rather than leave that choice explicit.
    """
    forecast: chex.Array             # (n_days, T) float32, what the observation shows
    actual: chex.Array               # (n_days, T) float32, what clearing runs on
    boundary_p_init: chex.Array      # (n_days, n_units) float32
    boundary_commitment: chex.Array   # (n_days, n_units) float32
    boundary_up_time: chex.Array     # (n_days, n_units) int32
    boundary_down_time: chex.Array   # (n_days, n_units) int32
    calendar: chex.Array             # (n_days, 4) float32
    learner_mask: chex.Array         # (n_agents,) bool
    episode_len: int = struct.field(pytree_node=False)


class DayAheadEnv(NamedTuple):
    """The pure functions of one built environment."""
    reset: Callable
    step: Callable
    step_auto_reset: Callable
    get_obs: Callable
    make_params: Callable


def calendar_encoding(days) -> np.ndarray:
    """Calendar encoding for the observation: day of year and day of week, each
    as sin/cos.

    Two cycles rather than an index, so that 31 December is adjacent to 1 January
    and Sunday to Monday.
    """
    idx = pd.DatetimeIndex(days)
    doy = idx.dayofyear.to_numpy(np.float64) / 366.0
    dow = idx.dayofweek.to_numpy(np.float64) / 7.0
    return np.stack([np.sin(2 * np.pi * doy), np.cos(2 * np.pi * doy),
                     np.sin(2 * np.pi * dow), np.cos(2 * np.pi * dow)], 1)


def make_env(
    case,
    fixture: Dict,
    demand: Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex],
    n_segments: int = 1,
    kind: str = "markup",
    markup_max: Optional[float] = None,
    cap_scale: Optional[float] = None,
    ramp_scale: Optional[float] = None,
    period_hours: float = 1.0,
    monitored_lines=None,
) -> Tuple[DayAheadEnv, Dict]:
    """Build the environment for one case, commitment fixture and demand series.

    ``monitored_lines`` is handed to both clearing operators unchanged: ``None``
    enforces every line limit, an index array enforces those lines and solves
    the two Newton systems by the low-rank route (`clearing.make_clearing`).
    The congestion count in ``info`` is taken over every line either way.

    ``fixture`` comes from `commitment.load_commitment` and ``demand`` from
    `demand.load_gb_demand`; the fixture's ``day_index`` selects the days of the
    demand series that the environment runs over, so the episode window is
    whatever was pre-committed offline and nothing else.

    ``cap_scale`` and ``ramp_scale`` are required.  At the registered values
    `case29gb` neither congests nor couples across periods, so an environment
    built without them studies a different market silently.

    Returns ``(env, spec)``.  ``spec`` carries the static shapes -- the
    observation dimension, the action shape and bounds, the `costs` names and the
    agent count -- plus the clearing operator's own spec.
    """
    if cap_scale is None or ramp_scale is None:
        raise ValueError("cap_scale and ramp_scale are scenario parameters with "
                         "no defensible default; pass both explicitly")
    meta = fixture["meta"]
    if (meta["cap_scale"], meta["ramp_scale"], meta["n_segments"]) != \
            (cap_scale, ramp_scale, n_segments):
        # a schedule committed under a different network or offer resolution
        # describes a different market, and nothing downstream would notice
        raise ValueError(
            f"fixture was built at cap_scale={meta['cap_scale']}, "
            f"ramp_scale={meta['ramp_scale']}, K={meta['n_segments']}, but this "
            f"environment asks for {cap_scale}, {ramp_scale}, {n_segments}")

    forecast, actual, days = demand
    day_index = np.asarray(fixture["day_index"], np.int64)
    # the window is the third thing a fixture can disagree about, and the check
    # above is blind to it.  Comparing lengths would not help, since two windows
    # of equal length can still cover different days.  The dates the fixture
    # recorded are compared against the dates its `day_index` selects out of the
    # demand series actually passed in, which is what makes a fixture carrying
    # one window and a series carrying another an error rather than a silent
    # substitution of the scenario.
    fixture_dates = [str(d) for d in meta["dates"]]
    series_dates = [d.strftime("%Y-%m-%d") for d in days[day_index]]
    if fixture_dates != series_dates:
        first = next((i for i, (a_, b_) in enumerate(zip(fixture_dates, series_dates))
                      if a_ != b_), 0)
        raise ValueError(
            f"fixture covers {len(fixture_dates)} days beginning "
            f"{fixture_dates[0]} but its day_index selects "
            f"{len(series_dates)} beginning {series_dates[0]} out of the demand "
            f"series passed in; first disagreement at position {first} "
            f"({fixture_dates[first]} against {series_dates[first]})")
    T = int(meta["n_periods"])
    K = n_segments
    n_units = len(np.asarray(case.unit_p_min))
    n_agents = n_units                       # one generating company per unit
    n_buses = int(case.n_nodes)
    n_days = len(day_index)

    clear, cspec = make_clearing(case, T, n_segments=K, cap_scale=cap_scale,
                                 ramp_scale=ramp_scale, period_hours=period_hours,
                                 monitored_lines=monitored_lines)
    # the relaxed commitment and its rounding, which together make the commitment
    # a function of the offers.  Both operators are built here rather than
    # shared, because their Newton budgets are calibrated against different
    # criteria (`relax.MAX_ITER` against reproducing the commitment,
    # `clearing.MAX_ITER` against dual accuracy) and one budget for both would
    # silently adopt the looser one.
    relax, rspec = make_relax(case, T, n_segments=K, cap_scale=cap_scale,
                              ramp_scale=ramp_scale, period_hours=period_hours,
                              monitored_lines=monitored_lines)
    settle = make_settlement(case, period_hours=period_hours)
    offer_map, aspec = make_offer_map(case, K, T, kind=kind, markup_max=markup_max)
    truthful = truthful_action(case, K, T, kind=kind)

    p_min = np.asarray(case.unit_p_min, np.float64)
    p_max = np.asarray(case.unit_p_max, np.float64)
    unit_bus = np.asarray(case.unit_node_idx, np.int64)
    _, seg_cost = segment_costs(case, K)                     # the offer envelope
    ramp_up_mw = np.asarray(case.unit_ramp_up, np.float64) * p_max * period_hours * ramp_scale
    ramp_dn_mw = np.asarray(case.unit_ramp_down, np.float64) * p_max * period_hours * ramp_scale
    # minimum up/down time are registered in hours; one period is `period_hours`
    # of them, and a window that does not divide evenly is rounded up rather than
    # truncated, since truncating would weaken the constraint it counts violations of
    ut = np.ceil(np.maximum(np.asarray(case.unit_min_up_time, np.float64), 1.0)
                 / period_hours).astype(np.int32)
    dt = np.ceil(np.maximum(np.asarray(case.unit_min_down_time, np.float64), 1.0)
                 / period_hours).astype(np.int32)

    # the observation's static block, one row per agent: capacity, segment costs,
    # ramp in the megawatts per period that actually bind, and the two windows
    static_obs = jnp.asarray(np.concatenate(
        [p_min[:, None], p_max[:, None], seg_cost, ramp_up_mw[:, None],
         ramp_dn_mw[:, None], ut[:, None], dt[:, None]], 1), jnp.float32)
    # four per-period blocks: today's commitment is not among them, since a
    # market that solves the commitment from the offers cannot publish it before
    # bidding.  That is the removal of a quantity that does not exist at decision
    # time, not a reduction in what the agent is told.
    obs_dim = static_obs.shape[1] + 4 * T + 9

    UTj, DTj = jnp.asarray(ut), jnp.asarray(dt)
    unit_busj = jnp.asarray(unit_bus)
    PTDFj = jnp.asarray(np.asarray(case.PTDF, np.float64))
    Fj = jnp.asarray(np.asarray(case.line_cap, np.float64) * cap_scale)
    sharej = jnp.asarray(cspec["demand_share"])

    P = dict(
        forecast=jnp.asarray(forecast[day_index, :T], jnp.float32),
        actual=jnp.asarray(actual[day_index, :T], jnp.float32),
        boundary_p_init=jnp.asarray(fixture["p_init"], jnp.float32),
        boundary_commitment=jnp.asarray(fixture["commitment_prev"], jnp.float32),
        boundary_up_time=jnp.asarray(fixture["up_time"], jnp.int32),
        boundary_down_time=jnp.asarray(fixture["down_time"], jnp.int32),
        calendar=jnp.asarray(calendar_encoding(days[day_index]), jnp.float32),
    )

    def make_params(episode_len: int, learner_mask=None) -> EnvParams:
        """Fill in the population and the episode length.

        ``learner_mask`` defaults to every agent learning, the full Markov game;
        a mask with a single `True` is the single-agent case, with every other
        agent bidding its true cost.
        """
        if not 1 <= episode_len <= n_days:
            raise ValueError(f"episode_len must be in 1..{n_days}, the days the "
                             f"commitment fixture covers, got {episode_len}")
        mask = jnp.ones((n_agents,), bool) if learner_mask is None else \
            jnp.asarray(learner_mask, bool)
        return EnvParams(learner_mask=mask, episode_len=int(episode_len), **P)

    def _obs(state, params):
        """The observation, ``(n_agents, obs_dim)``, in physical units.

        One row per agent, concatenated in this order: the static block built
        above, then the previous day's award over the T periods, its profit, and
        the four boundary quantities of `DayAheadState`, then the previous day's
        LMP at that agent's own bus over the T periods.  The last three blocks are
        system-wide and identical in every row -- the previous day's LMP averaged
        over buses, period by period, then the demand forecast for the day about
        to be cleared and that day's calendar encoding.

        Everything about the *previous* day comes from the state rather than being
        recomputed, which is what keeps the observation a function of the state
        alone and therefore unchanged in meaning across an auto-reset.
        """
        own_price = state.lmp_prev[:, unit_busj].T                  # (n_units, T)
        wide = lambda v: jnp.broadcast_to(v, (n_agents,) + v.shape[-1:])
        return jnp.concatenate([
            static_obs,
            state.award_prev,
            state.profit_prev[:, None],
            state.commitment_status[:, None],
            state.p_final[:, None],
            state.up_time[:, None].astype(jnp.float32),
            state.down_time[:, None].astype(jnp.float32),
            own_price,
            wide(state.lmp_prev.mean(1)),                # the system average
            wide(params.forecast[state.cursor]),
            wide(params.calendar[state.cursor]),
        ], 1)

    def _run_lengths(u, u_prev, up0, down0):
        """Run lengths at the day boundary, and the minimum up-/downtime violations
        inside it.

        A `lax.scan` over the periods, not a Python loop: the count depends on the
        order of the switches, and this is the only place where `up_time` and
        `down_time` from the previous day do any work.  A unit that shuts down
        having been on for fewer than UT periods violates the minimum uptime; the
        mirror image violates the minimum downtime.  With the commitment solved
        from the offers these counts are a consequence of the action rather than a
        property of the day: the relaxed commitment drops the minimum up-/downtime
        rows, so which pattern the rounding produces, and therefore how many
        windows it breaks, moves with the bids.  That is why the second column of
        `costs` is not constant in the action.

        `u` and `u_prev` must arrive in the same dtype.  The scan carries `u_prev`
        forward and replaces it with a row of `u`, so a float32 boundary and a
        float64 commitment would change the carry dtype on the first period.
        """
        def body(carry, u_t):
            """One period.  Carry is ``(previous commitment, up run, down run)``.

            The scan runs over ``u.T``, so ``u_t`` is one period's commitment
            across all units and every quantity here is ``(n_units,)``.  Each
            counter counts the periods the unit has spent in its own state,
            this one included, and is held at zero while it is in the other, so
            the two are never both positive.  The emitted flag is the violation
            that this period's switch, if any, causes.
            """
            prev, up, down = carry
            started = (prev < 0.5) & (u_t > 0.5)
            stopped = (prev > 0.5) & (u_t < 0.5)
            viol = (stopped & (up < UTj)) | (started & (down < DTj))
            return (u_t, jnp.where(u_t > 0.5, up + 1, 0),
                    jnp.where(u_t < 0.5, down + 1, 0)), viol
        (_, up, down), viol = jax.lax.scan(body, (u_prev, up0, down0), u.T)
        return up, down, viol.sum(0)

    def reset(key, params):
        """Start the episode at a day drawn from the pre-committed window.

        The previous day's prices, schedule and profit are zero because there is
        no previous day.

        The physical boundary is **the one the fixture's own sweep used for that
        day**, not one reconstructed here.  Reconstructing it does not work: a
        commitment met by a boundary it was not computed against sheds and prices
        at VOLL from the second day onwards, so the boundary and the schedule
        have to come from the same place.
        """
        j = jax.random.randint(key, (), 0, n_days - params.episode_len + 1,
                               dtype=jnp.int32)
        state = DayAheadState(
            step_in_episode=jnp.zeros((), jnp.int32),
            cursor=j,
            commitment_status=params.boundary_commitment[j],
            p_final=params.boundary_p_init[j],
            up_time=params.boundary_up_time[j],
            down_time=params.boundary_down_time[j],
            lmp_prev=jnp.zeros((T, n_buses), jnp.float32),
            award_prev=jnp.zeros((n_units, T), jnp.float32),
            profit_prev=jnp.zeros((n_agents,), jnp.float32))
        return _obs(state, params), state

    def step(key, state, action, params):
        """Clear, settle and advance by one market day.

        Runs all three clearing stages on the offers the actions map to -- the
        relaxed commitment, its rounding to integers, and the fixed-commitment
        dispatch whose duals are the prices -- then settles the awards against
        those prices.  It does **not** reset on `done`; `step_auto_reset` is the
        wrapper that does.

        Args:
            key: unused, and deleted immediately.  The market day is deterministic;
                see the module docstring.
            state: the `DayAheadState` the previous day ended at.  Its physical
                carry is the boundary both solves are run against, and its
                observation carry is what `_obs` reports as "yesterday".
            action: ``(n_agents, ...)`` in the shape ``spec["action"]`` declares.
                A non-learning agent's row is replaced by the truthful action
                before the map, so there is one code path.
            params: the per-day arrays and the agent population.

        Returns:
            ``(obs, state, reward, costs, done, info)``.  ``reward`` is
            ``(n_agents,)`` profit in \\$ per market day; ``costs`` is
            ``(n_agents, 2)`` in the order of `COST_NAMES` and never enters
            ``reward``; ``done`` is a time-limit truncation rather than a terminal
            state; ``info`` carries the per-day diagnostics, among them the two
            ``mu`` values and the ``converged`` flag that say whether the reward
            means anything at all.
        """
        del key                      # the market day is deterministic; see module
        pos = state.cursor
        # one code path: a non-learning agent's action is replaced by the action
        # whose offer is its true cost, and the same map runs on both.
        mask = params.learner_mask.reshape((-1,) + (1,) * (jnp.ndim(action) - 1))
        offer = offer_map(jnp.where(mask, action, truthful))

        demand = params.actual[pos]
        # float64 for the solver, float32 for the state.  The cast is written
        # out because the operator's own constants are float64 and a silent
        # promotion is what makes precision depend on a global flag.
        offer64 = offer.astype(jnp.float64)
        demand64 = demand.astype(jnp.float64)
        p_init64 = state.p_final.astype(jnp.float64)

        # the commitment is solved from the offers, so it is a function of the
        # action rather than a per-day constant.  The relaxation reads the day
        # boundary out of the state, which is what couples the days.
        rlx = relax(offer64, demand64, p_init64,
                    state.commitment_status.astype(jnp.float64),
                    state.up_time, state.down_time)
        u = round_commitment(rlx["u"])
        # how far the relaxed commitment sat from {0, 1}, which is exactly how
        # far the rounding step had to move it
        integrality_gap = jnp.max(jnp.abs(rlx["u"] - u))

        out = clear(offer64, u, demand64, p_init64)
        award, lmp, shed = out["award"], out["lmp"], out["shed"]
        money = settle(award, lmp, u.astype(award.dtype), state.commitment_status)

        # both in float32: the run lengths compare against 0.5 and nothing else,
        # and the scan carry has to keep one dtype (see `_run_lengths`)
        up, down, viol = _run_lengths(u.astype(jnp.float32), state.commitment_status,
                                      state.up_time, state.down_time)
        shed_mwh = period_hours * shed.sum()
        # `costs` carries these two and nothing else.  The VOLL cost below is
        # money and belongs to `info`; putting it here would make a price a
        # feasibility constraint.
        costs = jnp.stack([jnp.full((n_agents,), shed_mwh, jnp.float32),
                           viol.astype(jnp.float32)], 1)

        # net nodal injection through the PTDF, re-formed here only to count the
        # congested line-periods below.  `.add` rather than `.set`, since several
        # units share a bus; `shed` counts as injection, because a shed megawatt
        # is load the network does not have to carry.
        gen = jnp.zeros((T, n_buses), award.dtype).at[:, unit_busj].add(award.T)
        d = sharej[None, :] * demand.astype(award.dtype)[:, None]
        flow = (gen + shed - d) @ PTDFj.T
        state = DayAheadState(
            step_in_episode=state.step_in_episode + 1,
            # the last day of an episode has no successor, and an out-of-range
            # index clamps to the final element without a word from JAX.  The
            # clamp is written here so that the terminal observation is a stated
            # choice rather than a silent one; `step_auto_reset` replaces that
            # observation anyway.
            cursor=jnp.minimum(pos + 1, n_days - 1),
            commitment_status=u[:, -1].astype(jnp.float32),
            p_final=award[:, -1].astype(jnp.float32),
            up_time=up.astype(jnp.int32), down_time=down.astype(jnp.int32),
            lmp_prev=lmp.astype(jnp.float32),
            award_prev=award.astype(jnp.float32),
            profit_prev=money["profit"].astype(jnp.float32))
        info = dict(
            # two solves, so two complementarity gaps.  `mu` keeps its meaning --
            # the dispatch solve's, the one the price comes out of -- and the
            # relaxed commitment's gets a new name, so that `converged` being
            # false can be traced to a step; a single conjunction alone would
            # hide which one failed.
            mu=out["mu"], mu_relax=rlx["mu"],
            dual_residual=out["dual_residual"],
            relax_dual_residual=rlx["dual_residual"],
            converged=(out["mu"] < MU_TOL) & (rlx["mu"] < RELAX_MU_TOL),
            voll_cost=VOLL * shed_mwh,
            shed_mwh=shed_mwh,
            congested_line_periods=(jnp.abs(flow) >= Fj[None, :] - CONGESTION_TOL
                                    ).sum().astype(jnp.int32),
            integrality_gap=integrality_gap,
            # the commitment is now an output of the market rather than an input
            # to it, so it has to be reported: without it nothing downstream can
            # say which schedule the offers bought, and every check of capacity,
            # ramp or start-up cost needs it
            commitment=u.astype(jnp.float32),
            revenue=money["revenue"].astype(jnp.float32),
            cost=money["cost"].astype(jnp.float32),
            # the split of `cost`, for the same reason `commitment` is reported:
            # a share cannot be recovered from the total afterwards
            energy_cost=money["energy_cost"].astype(jnp.float32),
            no_load_cost=money["no_load_cost"].astype(jnp.float32),
            startup_cost=money["startup_cost"].astype(jnp.float32))
        done = state.step_in_episode >= params.episode_len
        obs = _obs(state, params)
        # `done` is a time-limit truncation and not a terminal state -- the
        # demand series, the units and the run lengths all continue past it -- so
        # a learner has to bootstrap from the observation that follows the
        # truncated step, and `step_auto_reset` overwrites the returned one.  Here
        # the two coincide, since this function does not reset.
        info["terminal_obs"] = obs
        return (obs, state, money["reward"].astype(jnp.float32),
                costs, done, info)

    def step_auto_reset(key, state, action, params):
        """Reset on `done` and stop the gradient at the boundary.

        `step` itself does not reset, so a caller who wants the terminal state can
        have it; this is the wrapper a fixed-length `lax.scan` runs.  The key goes
        to `reset` unsplit because `step` does not consume it, so there is no
        randomness for the new episode to correlate with.

        `info["terminal_obs"]` survives the reset and is what a learner bootstraps
        from, since `done` is a truncation.  One caveat belongs to the data rather
        than to this function: an episode that starts at the last legal day has no
        successor day, so the forecast, commitment and calendar columns of that
        terminal observation repeat the final day of the fixture.
        """
        obs, nxt, reward, costs, done, info = step(key, state, action, params)
        obs_re, state_re = reset(key, params)
        # `info["terminal_obs"]` keeps what `step` produced, which is the
        # successor of the truncated step; only the returned observation is
        # replaced
        obs = jnp.where(done, obs_re, obs)
        state = jax.tree.map(lambda a, b: jnp.where(done, a, b), state_re, nxt)
        return (jax.lax.stop_gradient(obs), jax.lax.stop_gradient(state),
                reward, costs, done, info)

    spec = dict(n_agents=n_agents, n_units=n_units, n_buses=n_buses, T=T, K=K,
                n_days=n_days, obs_dim=int(obs_dim), action=aspec,
                cost_names=COST_NAMES,
                # `done` is a time-limit truncation and no market here has a
                # natural terminal state, so a `truncated` field would be
                # identically `done`.  It is declared once, and a wrapper that needs
                # the gym pair builds it from this.
                termination="truncation",
                kind=kind, markup_max=markup_max,
                cap_scale=cap_scale, ramp_scale=ramp_scale,
                period_hours=period_hours, unit_bus=unit_bus,
                dates=meta["dates"], commitment_mode=meta["mode"],
                clearing=cspec)
    return DayAheadEnv(reset=reset, step=step, step_auto_reset=step_auto_reset,
                       get_obs=_obs, make_params=make_params), spec
