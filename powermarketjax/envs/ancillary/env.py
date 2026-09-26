"""Ancillary services environment: the Markov game of energy plus reserve.

    reset, step, step_auto_reset, spec = make_ancillary_env(case, theta=(1/6, 0.5),
        volr=..., beta=..., pi_scale=...)
    obs, state, reward, costs, done, info = step(key, state, action, params)

One step is one period of `period_hours`.  An episode is a fixed number of
consecutive periods ending in a time limit rather than an absorbing state, so
the final step has a genuine successor and `info` carries it.

**Reserve adds no physical carry.**  A capacity payment settles inside the
period and holding reserve moves no energy, so the state is the state of the
real-time market: where the exogenous series stands, the realised dispatch
that (RMP) reads, and the previous results the observation shows.  The
requirement is a function of the demand forecast, which the exogenous series
already supplies.

**`converged` reads `mu` and `dual_residual`**, each at a tolerance calibrated
for this market rather than inherited from the day-ahead market's (`MU_TOL`,
`DUAL_RES_TOL`).  Because the `mu` tolerance sits below what the tied region
produces, `converged` is also a partial detector of that region; the dual
half catches a solve whose prices drifted while `mu` stayed on the floor
(`case813nem`).

**A period that does not converge still pays.**  Reward is computed the same
way whether or not `converged` is true, and nothing here branches on it:
dropping unconverged periods would put an invisible selection bias into the
learning curve, and making reward depend on a solver diagnostic would break
the Markov property.  `converged` is reported so an experiment can count them.

**The learner may exclude such a sample from its loss, and that does not
weaken either half of the paragraph above.**  `info["usable"]` carries the same
boolean under the name `make_ippo(valid_key=...)` reads.  Under the driver's
`--mask-unusable` the sample's **reward keeps the value this market produced**
-- nothing here branches, the Markov chain still advances on the dispatch that
actually cleared -- and the excluded count is published as `masked_samples`, so
the selection is on the record rather than invisible.  What the paragraph above
rules out is dropping the *period*; what this allows is dropping the *gradient
contribution* of a sample whose price this market itself says is not usable.
Default off, so a run that passes nothing is the run it was before.

**Offer separation is computed here**, since it is a property of the offers
the action produced and the clearing receives them already built.  It goes
into `info` three times: over the whole profile, over the learners alone, and
over the units committed in that period.  The whole profile says whether the
program contains a degenerate block; under a learner mask that number is zero
by construction, because the non-learners are exactly tied at the baseline;
and the committed set is the population that actually trades.

`costs` carries physical quantities only: the shed energy, reported
identically to every agent because unserved energy is a property of the period
rather than of one bidder, and the unmet requirement of each product.  The
money those two penalties represent is reported in `info` as a diagnostic and
enters no reward.
"""
from typing import Callable, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from ..day_ahead.action import make_offer_map, truthful_action
from ..day_ahead.clearing import segment_costs
from .action import (make_reserve_offer_map, offer_separation,
                     truthful_reserve_action)
from .clearing import MAX_ITER, VOLL, make_clearing
from .requirement import make_requirement
from .settlement import make_settlement

#: `mu` above this is not a converged solve.  Measured on this market rather
#: than inherited: over six operating points outside the tied region and two
#: processes, converged solves leave `mu` between 4.79e-12 and 6.07e-12, a
#: spread of 1.27 and bit-identical across processes, while the lightest known
#: failure leaves 1.6e-01.  1e-09 sits 165x above the measured maximum and seven
#: orders below that failure.  Shrinking `ipm.SLACK_FLOOR` by ten orders leaves
#: those figures unchanged, so this floor is a convergence limit of the problem
#: and not an artefact of that constant.
MU_TOL = 1e-9

#: The second half of `converged`: the stationarity residual of the solve,
#: `dual_residual`, must be below this as well as `mu` below `MU_TOL`.  `mu`
#: alone cannot see a solve whose duals -- the prices -- are wrong: measured
#: 2026-09-16 on `case813nem`, the fixed 60-step loop leaves `mu` at
#: 1.4e-11 on 97.6% of 1 728 truthful periods while the dual residual is above
#: 1e-8 on 80.6% of them and two same-algorithm solvers disagree on the LMP
#: by up to 5.9 \$/MWh.  The value is derived, not chosen: it is the
#: dual-residual tier at which the price error against a reference running
#: the same solve is <= 1e-3 \$/MWh: over fifteen (dual residual, price
#: difference) pairs spanning eleven orders of magnitude on `case813nem` the
#: price difference is 0.19 to 3.30 times the residual, the residual being the
#: larger of the two solvers' (on this operator's alone the two unfrozen pairs
#: whose reference side was the bad one read 135 and 765; recomputed
#: 2026-09-17), so the 1e-3 \$/MWh tier is a residual of 3e-4, rounded down
#: one decade.  It sits three decades
#: above the unrestricted, converged solves of `case29gb` / `case73rts`
#: (truthful replay of the twelve evaluation days each, 2026-09-17:
#: dual residual p99 3.0e-8 /
#: 1.1e-12, max 6.0e-8 / 6.9e-8, no period above 1e-4 on either).  It does
#: move the label on TRAINED policies' periods: replaying one archive of each
#: case under this gate, `case29gb` 16 of 576 periods flip to unconverged
#: (residuals up to 1.6e-2) and `case73rts` 110 of 1 728 (residuals up to
#: 1.5, 30 of 36 days touched) -- those are solves whose
#: duals were unusable all along, which the mu half could not see.  Same name and
#: meaning as `tools/benchmark/run_rl_02.DUAL_RES_TOL`; that market derived
#: its own value.
#:
#: What a failed dual check does NOT say: how wrong the price is.  Measured on
#: market 02 (2026-09-17): at the production trip count two kinds of period read the same
#: residual, one that a longer budget would drive to 1e-9 and one that is a
#: fixed point of the restricted LP under this solver; the products cannot
#: tell them apart, so ``converged=False`` on this half means "the duals are
#: not usable at this budget", never a price error of a known size.
DUAL_RES_TOL = 1e-4

#: `costs` in the order the shed energy first, then one column per product
#: for the unmet requirement.
SHED_COST_NAME = "shed_energy_mwh"


@struct.dataclass
class AncillaryState:
    """Everything that crosses a period boundary."""
    cursor: chex.Array              # int32   ()          index into the series
    step_in_episode: chex.Array     # int32   ()          episode counter
    p_prev: chex.Array              # float64 (n_units,)  (RMP) reads this
    award_prev: chex.Array          # float64 (n_units,)  own previous
    profit_prev: chex.Array         # float64 (n_units,)  own previous
    lmp_prev: chex.Array            # float64 (n_units,)  own bus price
    reserve_price_prev: chex.Array  # float64 (n_prod,)   public


@struct.dataclass
class AncillaryParams:
    """Sweepable and data-carrying quantities."""
    demand: chex.Array          # float64 (n_periods,)            realised
    forecast: chex.Array        # float64 (n_periods,)            published
    commitment: chex.Array      # float64 (n_periods, n_units)    exogenous u
    q_da: chex.Array            # float64 (n_periods, n_units)    day-ahead schedule
    lmp_da: chex.Array          # float64 (n_periods, n_buses)    day-ahead price
    learner_mask: chex.Array    # bool    (n_units,)              which agents learn
    episode_len: int = struct.field(pytree_node=False)


def make_ancillary_env(
    case,
    theta,
    volr: float,
    beta,
    pi_scale: float,
    n_segments: int = 1,
    cap_scale: float = 1.0,
    ramp_scale: float = 1.0,
    period_hours: float = 0.5,
    kind: str = "markup",
    markup_max: Optional[float] = None,
    max_iter: int = MAX_ITER,
    #: Exposed so that two regularisations can be compared inside one process.
    #: Without it the only way to ask "is this quantity an artefact of the
    #: solver" is to edit the module constant in a copy of the repository and
    #: run twice, which compares two processes as well as two regularisations
    #: and cannot hold anything else fixed.  Defaults to the market's adopted
    #: value, so every existing caller is unchanged.
    reg_coef: float = None,
    mu_tol: float = MU_TOL,
    monitored_lines=None,
    freeze_mu: Optional[float] = None,
    dual_res_tol: float = DUAL_RES_TOL,
    stop_tol: Optional[Tuple[float, float]] = None,
    kkt: str = "auto",
    lowrank_free: Optional[Tuple[int, int]] = None,
    lu_batching: str = "auto",
) -> Tuple[Callable, Callable, Callable, dict]:
    """Build the environment for one case, product ladder and scenario.

    `theta`, `volr`, `beta` and `pi_scale` have no defensible defaults and are
    required arguments.  `cap_scale` and `ramp_scale` describe a different
    market at a different value, so they are declared here too rather than
    defaulted.

    `monitored_lines` goes straight to `make_clearing` (``None`` = every line,
    the default and what every archive was produced on); the set and route the
    operator was actually built with are on ``spec["clearing_spec"]`` as
    ``monitored_lines`` / ``kkt_route``, which is where a driver stamps its
    products from.  `freeze_mu` likewise (``None`` = off, the default;
    stamped as ``clearing_spec["freeze_mu"]``), and `stop_tol` (``None`` =
    the fixed trip count; stamped as ``clearing_spec["stop_tol"]``).  `kkt`,
    `lowrank_free` and `lu_batching` likewise: the Newton system's
    route and the low-rank route's block sizing and solver, ``"auto"`` /
    ``None`` / ``"auto"`` being the operator's own choice (dense on every
    line, low-rank with every column free on a proper subset), stamped as
    ``clearing_spec["kkt_route"]`` / ``["lowrank_free"]`` / ``["lu_batching"]``.
    """
    clear, cspec = make_clearing(case, theta, volr, n_segments=n_segments,
                                 cap_scale=cap_scale, ramp_scale=ramp_scale,
                                 period_hours=period_hours, max_iter=max_iter,
                                 monitored_lines=monitored_lines,
                                 freeze_mu=freeze_mu, stop_tol=stop_tol,
                                 kkt=kkt, lowrank_free=lowrank_free,
                                 lu_batching=lu_batching,
                                 **({} if reg_coef is None
                                    else {"reg_coef": float(reg_coef)}))
    settle = make_settlement(case, period_hours=period_hours)
    requirement = make_requirement(beta)
    n_units, n_prod = cspec["n_units"], cspec["n_prod"]
    n_buses = cspec["n_buses"]

    energy_map, espec = make_offer_map(case, n_segments, 1, kind=kind,
                                       markup_max=markup_max)
    reserve_map, rspec = make_reserve_offer_map(n_units, n_prod, pi_scale,
                                                volr)
    _, seg_cost = segment_costs(case, n_segments)
    unit_bus = jnp.asarray(np.asarray(case.unit_node_idx, np.int64))

    n_energy = int(np.prod(espec["shape"])) // n_units
    action_shape = (n_units, n_energy + n_prod)
    truth_energy = truthful_action(case, n_segments, 1, kind=kind)
    truth_reserve = truthful_reserve_action(n_units, n_prod, jnp.float64)
    baseline = jnp.concatenate(
        [jnp.reshape(truth_energy, (n_units, n_energy)), truth_reserve], axis=1)

    #: static per agent, so the observation carries them once rather than the
    #: agent inferring them
    p_max = jnp.asarray(np.asarray(case.unit_p_max, np.float64))
    p_min = jnp.asarray(np.asarray(case.unit_p_min, np.float64))
    res_cap = jnp.asarray(np.asarray(cspec["res_cap"], np.float64))
    #: Periods in a calendar day at this market's period length.  ONE number
    #: with two consumers: the observation's calendar phase below, and
    #: `reset_on_day`'s "first period of day `d`".  They were two derivations of
    #: `24 / period_hours` until they were merged -- and two derivations of
    #: a day length are exactly the divergence this market was just hurt by: an
    #: episode filed under day `d` by one of them and opened elsewhere by the
    #: other is what made the arms score on different periods.  Derived from
    #: `period_hours` rather than passed in, so it cannot disagree with the
    #: period the clearing and the settlement were built for.
    periods_per_day = int(round(24.0 / float(period_hours)))
    if abs(periods_per_day * float(period_hours) - 24.0) > 1e-9:
        raise ValueError(
            f"period_hours={period_hours} does not divide a 24 h day, so "
            f"'the first period of day d' has no meaning and `reset_on_day` "
            f"would silently open somewhere inside a neighbouring day")
    steps_per_day = periods_per_day

    def _split(action):
        """Split one action into its energy and reserve halves.

        The first ``n_energy`` columns are the day-ahead markup action, reshaped
        to that map's own shape; the remaining ``n_prod`` are the raw reserve
        actions.  Returns the two **offers**, not the two actions.
        """
        energy = jnp.reshape(action[:, :n_energy], espec["shape"])
        return energy_map(energy), reserve_map(action[:, n_energy:])

    def _get_obs(state: AncillaryState, params: AncillaryParams) -> chex.Array:
        """The observation, `(n_units, obs_dim)`, one row per agent.

        Static description of the unit first, then the commitment it is about to
        be cleared under and its own previous results, then its day-ahead
        schedule, then a public block: the published forecast, the calendar
        pair, the requirement of each product and each product's previous
        price.  Every column is either the agent's own or public; no column is
        another agent's.

        ``cursor`` is clamped to the last period of the series, so the
        observation of a state one past the end is still defined -- which is
        what the terminal observation of the last episode needs.
        """
        idx = jnp.minimum(state.cursor, params.demand.shape[0] - 1)
        d_res = requirement(params.forecast[idx])
        # sin/cos of the absolute cursor: periodic in the day without a modulus,
        # and adjacent across a day boundary rather than a full turn apart
        phase = 2.0 * jnp.pi * (state.cursor.astype(jnp.float64) / steps_per_day)
        broadcast = lambda x: jnp.full((n_units,), x, jnp.float64)
        # `res_cap` is `(n_units, n_prod)`, so the static block carries one
        # column per product and is built from `n_prod` like the two blocks
        # appended below it.  At the two-product ladder this is column for
        # column what naming the two indices gave.
        columns = [p_min, p_max]                                   # static
        columns += [res_cap[:, j] for j in range(n_prod)]
        columns += [
            params.commitment[idx],                                # own, coming
            state.p_prev, state.award_prev, state.profit_prev,     # own, previous
            state.lmp_prev,
            params.q_da[idx],                                      # own position
            broadcast(params.forecast[idx]),                       # public
            broadcast(jnp.sin(phase)), broadcast(jnp.cos(phase)),  # calendar
        ]
        columns += [broadcast(d_res[j]) for j in range(n_prod)]
        columns += [broadcast(state.reserve_price_prev[j]) for j in range(n_prod)]
        return jnp.stack(columns, axis=1)

    def _probe_obs_dim() -> int:
        """Width of one observation row, read off an abstract evaluation.

        `jax.eval_shape` traces `_get_obs` on a zero state and a dummy
        `AncillaryParams` and returns the shape without executing it, so the
        number published in `spec` is the concatenation's own width and not a
        second statement of it.  The same probe is what `real_time.env` uses;
        a hand-written width here agreed with the columns only at ``n_prod``
        equal to two.
        """
        zero_u = jnp.zeros((n_units,), jnp.float64)
        state = AncillaryState(
            cursor=jnp.zeros((), jnp.int32),
            step_in_episode=jnp.zeros((), jnp.int32),
            p_prev=zero_u, award_prev=zero_u, profit_prev=zero_u,
            lmp_prev=zero_u,
            reserve_price_prev=jnp.zeros((n_prod,), jnp.float64))
        dummy = AncillaryParams(
            demand=jnp.zeros((1,), jnp.float64),
            forecast=jnp.zeros((1,), jnp.float64),
            commitment=jnp.zeros((1, n_units), jnp.float64),
            q_da=jnp.zeros((1, n_units), jnp.float64),
            lmp_da=jnp.zeros((1, n_buses), jnp.float64),
            learner_mask=jnp.ones((n_units,), bool), episode_len=1)
        return int(jax.eval_shape(_get_obs, state, dummy).shape[1])

    obs_dim = _probe_obs_dim()

    def _open_at(cursor, params):
        """The opening state at an explicit start period, and its observation.

        Factored out of `reset` so that `reset_on_day` cannot build a carry that
        differs from the one `reset` builds: the two entry points choose the
        start period differently and construct the state through this one body.
        A second copy of these seven fields is the failure this avoids -- it
        would look correct and open a state belonging to no episode.
        """
        cursor = jnp.asarray(cursor, jnp.int32)
        zeros_u = jnp.zeros((n_units,), jnp.float64)
        state = AncillaryState(
            cursor=cursor, step_in_episode=jnp.asarray(0, jnp.int32),
            # the first period starts from the day-ahead schedule of that
            # period, which is the schedule the system is running when the
            # episode opens -- not the registered initial output or the
            # committed minimum
            p_prev=params.q_da[cursor],
            award_prev=zeros_u, profit_prev=zeros_u, lmp_prev=zeros_u,
            reserve_price_prev=jnp.zeros((n_prod,), jnp.float64))
        return _get_obs(state, params), state

    def reset(key: chex.PRNGKey, params: AncillaryParams):
        """Draw a start period and open the episode there.

        The draw is uniform over the starts that leave ``episode_len`` whole
        periods in the series, so an episode never runs off the end of the
        exogenous data.  Returns ``(obs, state)``; every previous-step field of
        the state is zero because there is no previous step, and the opening
        `p_prev` is the one quantity that has to be chosen rather than zeroed.

        **The draw is over periods, not over days.**  A start uniform on periods
        lands inside a day 47 times out of 48, so an ``episode_len`` of 48 runs
        across the day boundary; evaluation that wants one named day must use
        `reset_on_day` instead.  This behaviour is deliberately left
        as it is -- training draws should not be locked to day boundaries.
        """
        n_periods = params.demand.shape[0]
        cursor = jax.random.randint(key, (), 0,
                                    n_periods - params.episode_len + 1)
        return _open_at(cursor, params)

    def reset_on_day(key: chex.PRNGKey, params: AncillaryParams, day):
        """Open the episode at the **first** period of ``day``.

        Purely additive: `reset` is untouched and nothing in the environment
        calls this.  It exists because evaluation has to score every arm on one
        named batch of periods, and `reset` cannot be asked for a day -- the
        stopgap it replaces searched for a key whose draw happened to land in
        day ``day``, which fixes ``cursor // periods_per_day`` and leaves the
        offset inside the day free.  Market 03 measured the consequence: over
        twelve evaluation days the open-loop arms spent 265 of 576 half hours
        outside the day they were filed under, all of them on training days,
        while the learning arm spent 353 outside and none on a training day --
        so the two arms were not scored on the same periods at all.

        ``key`` is accepted and ignored, so this has `reset`'s signature plus
        the day and can be substituted for it wherever a driver holds one.
        ``day`` may be traced; no bound check is done here because a Python
        check on a tracer is not one, and the caller reads the day back out of
        the returned state (`evaluation.open_day_start`).
        """
        del key
        return _open_at(jnp.asarray(day, jnp.int32) * periods_per_day,
                        params)

    def step(key: chex.PRNGKey, state: AncillaryState, action: chex.Array,
             params: AncillaryParams):
        """Clear and settle one period of ``period_hours``.

        Auto-reset is embedded here rather than in the wrapper, so on the step
        where ``done`` fires the returned state already belongs to a fresh
        episode.  ``key`` is therefore split, as
        `resources.env_base.Environment.step` requires of any environment that
        resets inside its own step: the reset draws its start period from a key
        the transition never touches.  The transition itself draws nothing.

        Args:
            action: `(n_units, n_energy + n_prod)`.  A non-learner's row is
                replaced by the declared baseline before the split, so the whole
                population goes through one code path.

        Returns:
            ``(obs, state, reward, costs, done, info)``.  ``reward`` is
            `(n_units,)` in \\$ for this period: the energy leg of the
            two-settlement rule plus the capacity payment, computed from
            cleared quantities and never from an offer.  ``costs`` is
            `(n_units, 1 + n_prod)` in the order `spec["cost_names"]` gives --
            the shed energy in MWh, identical for every agent because unserved
            energy is a property of the period rather than of one bidder, then
            the unmet requirement of each product in MW.  It is the CMDP
            constraint vector and enters no reward; the money those two
            shortfalls stand for is reported separately as ``info["voll_cost"]``
            and ``info["volr_cost"]``.  ``done`` is a time limit rather than an
            absorbing state, so ``info["terminal_obs"]`` carries the observation
            of the successor the truncated episode really had.  ``info`` also
            carries ``mu``, ``dual_residual`` and ``converged``, the two offer
            separations, the cleared ``award`` and ``reserve``, the prices
            ``lmp``, ``reserve_price`` and ``capacity_dual``, the
            ``requirement`` and the period ``cost``.

            A period that does not converge is settled and paid like any other;
            nothing here branches on ``converged`` (see the module docstring).
        """
        transition_key, reset_key = jax.random.split(key)
        del transition_key                     # the transition draws nothing
        action = jnp.asarray(action, jnp.float64)
        # one code path.  A non-learner's row becomes the declared baseline
        # action, and the same two maps run on learners and baseline alike
        # rather than the offers being substituted after the map.
        action = jnp.where(params.learner_mask[:, None], action, baseline)

        offer, offer_res = _split(action)
        idx = state.cursor
        d_res = requirement(params.forecast[idx])
        u = params.commitment[idx]
        # the energy map is built at one period, so its period axis is dropped
        # here rather than the clearing being given a horizon it does not have
        out = clear(offer[:, :, 0], offer_res, u, params.demand[idx], d_res,
                    state.p_prev)
        # who pays to start is decided by the previous period's commitment, and
        # that is read off the previous output exactly as (RMP) reads it inside
        # the clearing -- one convention, so the two cannot disagree
        money = settle(out["award"], out["reserve"], out["lmp"],
                       out["reserve_price"], params.q_da[idx],
                       params.lmp_da[idx], u,
                       (state.p_prev > 0.0).astype(jnp.float64))

        next_state = AncillaryState(
            cursor=state.cursor + 1,
            step_in_episode=state.step_in_episode + 1,
            p_prev=out["award"], award_prev=out["award"],
            profit_prev=money["profit"], lmp_prev=out["lmp"][unit_bus],
            reserve_price_prev=out["reserve_price"])
        done = next_state.step_in_episode >= params.episode_len

        _, fresh = reset(reset_key, params)
        merged = jax.tree_util.tree_map(
            lambda nxt, new: jnp.where(done, new, nxt), next_state, fresh)
        obs = _get_obs(merged, params)
        terminal_obs = jnp.where(done, _get_obs(next_state, params), obs)

        shed_mwh = period_hours * jnp.sum(out["shed"])   # MW over buses -> MWh
        costs = jnp.concatenate(
            [jnp.full((n_units, 1), shed_mwh),
             jnp.broadcast_to(out["reserve_shortfall"][None, :],
                              (n_units, n_prod))], axis=1)
        info = dict(
            mu=out["mu"], dual_residual=out["dual_residual"],
            # both halves: the complementarity gap AND the stationarity
            # residual; a period whose prices are wrong with `mu` on the
            # floor is not a converged clearing (see `DUAL_RES_TOL`)
            converged=(out["mu"] < mu_tol) & (out["dual_residual"] < dual_res_tol),
            # `usable` is the same boolean under the name the learner reads
            # (`make_ippo(valid_key=...)`, added for market 02 in `606d6ea`).
            # It is a second name and not a second quantity on purpose: 02 had
            # to build one because its `converged` looks at `mu` alone, while
            # this market's already reads both halves.  Renaming rather than
            # reusing keeps `converged`'s meaning pinned to every product on
            # disk and leaves the learner's key free to mean what the learner
            # needs.
            usable=(out["mu"] < mu_tol) & (out["dual_residual"] < dual_res_tol),
            # two populations, two questions (see `action.offer_separation`):
            # the whole profile says whether the program has a degenerate
            # block, the learners alone say whether the learning signal is
            # polluted
            offer_separation_all=offer_separation(offer_res),
            offer_separation_learners=offer_separation(
                offer_res, params.learner_mask),
            # The separation that bears on a price is the one among the
            # providers of the period, and only committed units provide.  The
            # two series above
            # answer different questions and neither answers this one: taking
            # the gap over all 66 units includes the de-committed ones, which
            # all sit at the same offer, so the reported separation is
            # systematically too small and belongs to no population that trades.
            # That denominator error has been retracted once in this market
            # already, which is why this is a third series rather than a
            # redefinition of either existing one.
            offer_separation_committed=offer_separation(offer_res, u > 0.0),
            # The offers themselves, because every question about whether a
            # given provider *set* the price needs the offer beside the price.
            # `offer_separation_*` collapses the profile to one number per
            # product, and a collapsed statistic cannot say which unit is at the
            # margin.  The distinction is load-bearing: separation lives on the
            # population of all provider pairs, while the price lives on the
            # marginal provider, and a zero separation between two units that
            # never price says nothing about any price.
            # both offer blocks, because reconstructing the LP that produced a
            # period (to solve it with an independent solver) needs the offers
            # exactly as the operator saw them; deriving them again from the
            # action would be a second implementation of the offer map
            offer=offer, offer_res=offer_res,
            reserve_price=out["reserve_price"], lmp=out["lmp"],
            # the cleared reserve itself, so that a comparison against a
            # reference can test the quantity rather than only the price it is
            # multiplied by; an experiment also needs it to report who held
            # capacity
            reserve=out["reserve"], award=out["award"],
            requirement=d_res, capacity_dual=out["capacity_dual"],
            # `cost` and `shed_mwh` are what `tools/benchmark/evaluation.py`
            # pairs into the system-cost figure every market's baselines
            # report.  Reported here rather than recomputed by the caller: the
            # cost of a period is a settlement quantity, and a driver that
            # rebuilt it from `award` would be a second implementation of the
            # cost envelope.
            cost=money["cost"], shed_mwh=shed_mwh,
            # The shed PER BUS, beside the scalar in MWh that `evaluation.py`
            # consumes.  Both, not one: the scalar is what the system-cost
            # figure needs and the vector is what an independent solver needs,
            # and neither recovers the other -- summing the vector loses the
            # period-hours factor, and the scalar loses which bus.
            #
            # Added 2026-08-29 for a comparison that could not be finished
            # without it.  Re-solving a period's LP with HiGHS and comparing
            # reserve prices reports a DISAGREEMENT; turning that into an ERROR
            # needs each side's objective at its own point, and assembling the
            # LP's `x` needs this vector.  Without it the note had to stop at
            # "the two differ", which is a weaker statement than the measurement
            # could support.
            shed=out["shed"],
            cost_energy=money["cost_energy"], cost_noload=money["cost_noload"],
            cost_startup=money["cost_startup"],
            reserve_shortfall=out["reserve_shortfall"],
            voll_cost=period_hours * VOLL * jnp.sum(out["shed"]),
            volr_cost=period_hours * volr * jnp.sum(out["reserve_shortfall"]),
            terminal_obs=terminal_obs)
        return obs, merged, money["reward"], costs, done, info

    def step_auto_reset(key, state, action, params):
        """`step` with the gradient stopped at the episode boundary.

        `step` already resets on ``done``, so this wrapper adds only
        ``stop_gradient`` on the observation and the state, which is what keeps
        a `lax.scan` rollout from carrying a gradient across a seam.  The
        6-tuple and every element of it are `step`'s.
        """
        obs, new_state, reward, costs, done, info = step(key, state, action,
                                                         params)
        return (jax.lax.stop_gradient(obs), jax.lax.stop_gradient(new_state),
                reward, costs, done, info)

    spec = dict(n_agents=n_units, obs_dim=obs_dim, action_shape=action_shape,
                action_low=espec["low"], action_high=espec["high"],
                costs_dim=1 + n_prod,
                cost_names=(SHED_COST_NAME,)
                + tuple(f"unmet_requirement_mw_{j}" for j in range(n_prod)),
                termination="truncation", n_prod=n_prod, n_buses=n_buses,
                period_hours=float(period_hours), mu_tol=float(mu_tol),
                dual_res_tol=float(dual_res_tol),
                pi_scale=float(pi_scale), volr=float(volr),
                #: The reserve half of the action box, in pre-softplus units.
                #: `action_low` / `action_high` above are the energy map's and
                #: cover the leading columns only, which is the gap
                #: `learning/policy.py` describes; these two close it for the
                #: trailing `n_prod` columns.  They are the offer map's own
                #: numbers rather than restated here, so the market and the
                #: learner cannot come to disagree about the action space.
                reserve_low=float(rspec["low"]),
                reserve_high=float(rspec["high"]),
                dtype=jnp.float64, get_obs=_get_obs, baseline_action=baseline,
                periods_per_day=periods_per_day, reset_on_day=reset_on_day,
                clearing_spec=cspec)
    return reset, step, step_auto_reset, spec
