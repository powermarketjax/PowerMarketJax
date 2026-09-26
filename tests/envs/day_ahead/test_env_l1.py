"""L1 domain correctness for the day-ahead environment (§16).

What this module checks is what only the *environment* can get wrong.  The
clearing operator's own invariants are verified in `test_clearing_l1.py` against
the operator directly, in the form that needs its duals: the pricing formula
against a finite difference, the money-balance identity of §8 term by term, and
per-period power balance.  Those are not repeated here.  What is checked below in
a weaker form is the part of them the environment can break by wiring the
operator wrongly, namely balance summed over the day against the demand of that
day, flows against the rating alongside the congestion count the environment
itself computes, and payments against charges per agent.  Beyond that the
environment adds four things and each has a test of its own:

* it **chains days**, so the ramp rows of period 0 act on the previous day's final
  output.  Nothing in the clearing tests crosses a day boundary;
* it **wires the settlement**, so the reward has to be the profit of the day
  computed from the realised award and the true cost, with the start-up charge
  reading the commitment carried across the boundary rather than assuming it;
* it **counts (MU)/(MD) violations** of an exogenous commitment, which is §6.4's
  second option -- leave the pattern and report the violation rate;
* it **substitutes truthful actions** for non-learning agents under
  `learner_mask`, and that substitution has to reach exactly those agents.

**This file is the primal side of the L1 split**, plus
the settlement.  The price side is in `test_clearing_l1.py`: everything that reads
a dual lives there, and a new assertion belongs on whichever side its quantity
comes from rather than in whichever file is open.  The split exists because this
side is blind to a price error, since power balance, the capacity bounds and the
ramp limits all hold whatever the duals say.

The discriminating injection on this side is the shut-down allowance of §6.3:
reverting it to $\\underline{p}_i$ fails the chained feasibility test on the first
day with `mu` at 3.93e+265.  That test also records why the injection has to draw
markups per agent rather than set a common ceiling, which is the shape that made
an earlier version of it vacuous.

Tolerances are float32-bound and measured, not assumed.  `DayAheadState` is float32 by
§15 while the clearing runs in float64, so a quantity recomputed from the state
differs from the one the environment computed internally by the rounding of the
state itself.  Measured over one episode at a markup of 1.3: the reward
recomputation below is out by at most 3.5 cents, 1.2e-6 relative against profits
of order 7e5 \\$, so 1e-4 carries two orders of margin without hiding a wiring
error, which would be relative-order-one.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.day_ahead import (VOLL, load_commitment, load_gb_demand,
                                           make_env, truthful_action)
from powermarketjax.envs.day_ahead.commitment import FIXTURE_DIR
from powermarketjax.envs.day_ahead.env import MU_TOL, RELAX_MU_TOL

T = 4
K = 1
CAP_SCALE = 0.6        # adopted scenario (2026-08-17)
RAMP_SCALE = 1.0       # adopted scenario (2026-08-17); registered rates undiscounted
MARKUP_MAX = 2.0
EPISODE_LEN = 3
#: See the module docstring: float32 state rounding, measured at 1.2e-6 relative
#: (3.5 cents against profits of order 7e5 \$).
MONEY_RTOL = 1e-4


@pytest.fixture(scope="module", autouse=True)
def x64():
    prev, prev_mm = jax.config.jax_enable_x64, jax.config.jax_default_matmul_precision
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    yield
    jax.config.update("jax_enable_x64", prev)
    jax.config.update("jax_default_matmul_precision", prev_mm)


@pytest.fixture(scope="module")
def case(x64):
    return load_case("29gb")


@pytest.fixture(scope="module")
def built(case):
    env, spec = make_env(case, load_commitment(n_periods=T), load_gb_demand(),
                         n_segments=K, kind="markup", markup_max=MARKUP_MAX,
                         cap_scale=CAP_SCALE, ramp_scale=RAMP_SCALE)
    return env, spec, env.make_params(episode_len=EPISODE_LEN)


@pytest.fixture(scope="module")
def rollout(built):
    """One episode at a markup of 1.3, kept as (state_before, outputs) pairs."""
    env, spec, params = built
    key = jax.random.PRNGKey(0)
    action = jnp.full((spec["n_agents"],), 1.3)
    step = jax.jit(env.step)
    _, state = env.reset(key, params)
    out = []
    for _ in range(EPISODE_LEN):
        before = state
        obs, state, reward, costs, done, info = step(key, state, action, params)
        out.append((before, state, reward, costs, info))
    return out, action


def _true_cost(case, award, u, u_boundary, period_hours=1.0):
    """Cost of §4 in numpy: the integral of the marginal-cost curve, no-load, start-up.

    Written out rather than taken from `physics.power_flow.compute_generation_cost`,
    which is what the settlement itself calls, so that the two are independent
    paths.  The integral is the one `test_settlement_l1.py` pinned against
    `scipy.integrate.quad`; writing `a p^2 + b p + c` instead -- the MATPOWER
    total-cost reading that `case_data.py` documented until 2026-08-05 -- gives a
    plausible number that is wrong by orders of magnitude (§12).
    """
    a = np.asarray(case.unit_cost_a, np.float64)[:, None]
    b = np.asarray(case.unit_cost_b, np.float64)[:, None]
    c = np.asarray(case.unit_cost_c, np.float64)[:, None]
    energy = (a / 3 * award ** 3 + b / 2 * award ** 2 + c * award).sum(1)
    idle = np.asarray(case.unit_no_load_cost, np.float64) * u.sum(1)
    prev = np.hstack([np.asarray(u_boundary)[:, None], u[:, :-1]])
    start = np.asarray(case.unit_startup_cost, np.float64) * \
        np.maximum(u - prev, 0.0).sum(1)
    return period_hours * (energy + idle) + start


def test_reward_is_the_profit_of_the_realised_award(case, built, rollout):
    """§8: reward from the award, not from the offer.

    Recomputed here from the state's own record of the day -- the award, the price
    at each unit's bus, and the commitment carried across the boundary -- with the
    true cost written independently.  A feasibility penalty leaking into the reward
    would show as an order-one discrepancy, since the only penalty available is
    VOLL times shed at 10 000 \\$/MWh.
    """
    env, spec, params = built
    steps, _ = rollout
    unit_bus = spec["unit_bus"]
    worst = 0.0
    for before, after, reward, _, info in steps:
        # the commitment is solved from the offers, so it comes back in `info`
        u = np.asarray(info["commitment"], np.float64)
        award = np.asarray(after.award_prev, np.float64)
        lmp = np.asarray(after.lmp_prev, np.float64)
        revenue = (lmp.T[unit_bus] * award).sum(1)
        got = revenue - _true_cost(case, award, u, np.asarray(before.commitment_status))
        np.testing.assert_allclose(got, np.asarray(reward, np.float64),
                                   rtol=MONEY_RTOL, atol=1.0)
        worst = max(worst, float(np.abs(got - np.asarray(reward, np.float64)).max()
                                 / max(np.abs(got).max(), 1.0)))
    assert worst < MONEY_RTOL, f"worst relative money error {worst:.2e}"


def test_ramp_is_respected_across_the_day_boundary(case, built, rollout):
    """The one physical constraint that only a chained rollout can violate.

    Period 0 of every day ramps from the previous day's final output.  At the
    retired `ramp_scale = 0.25` that limit was 0.175 of capacity per period against
    a committed minimum of about 0.200, so the two allowances of §6.3 were what
    made a switch possible at all, on 65 of 66 units.  At the adopted
    `ramp_scale = 1.0` the limit is 0.700 of capacity and **no unit needs the
    allowance** (0 of 66, measured 2026-08-17), so this test no longer exercises
    that necessity -- it checks that the constraint including the allowances is
    respected, which is a different and still-worthwhile statement.  The test that
    exercises the necessity constructs its own run point:
    `test_clearing_l1.py::test_commitment_may_switch_under_binding_ramp` at
    RAMP_BINDING.  They are asymmetric, `p_min` upwards and `p_max`
    downwards, because starting only has to reach the committed minimum while
    stopping has to leave whatever output the previous period chose; `p_min` on the
    shut-down side leaves a high-output unit unable to stop, which is infeasibility
    rather than expense.  A boundary carried wrongly -- the registered
    `unit_init_power`, or the committed minimum -- does not show up as an error but
    as a market that sheds and prices at VOLL (`commitment.py`).
    """
    env, spec, params = built
    steps, _ = rollout
    p_min, p_max = spec["clearing"]["p_min"], spec["clearing"]["p_max"]
    ramp_up = np.asarray(case.unit_ramp_up, np.float64) * p_max * RAMP_SCALE
    ramp_dn = np.asarray(case.unit_ramp_down, np.float64) * p_max * RAMP_SCALE
    for before, after, _, _, info in steps:
        u = np.asarray(info["commitment"], np.float64)
        award = np.asarray(after.award_prev, np.float64)
        p_init = np.asarray(before.p_final, np.float64)
        u_prev = np.hstack([(p_init > 0).astype(np.float64)[:, None], u[:, :-1]])
        start = np.maximum(u - u_prev, 0.0) * p_min[:, None]
        stop = np.maximum(u_prev - u, 0.0) * p_max[:, None]
        delta = np.diff(np.hstack([p_init[:, None], award]), axis=1)
        # OFF_EPS-scale tolerance, as in the clearing L1: the ramp rows act on the
        # LP variables, which keep a phantom of up to 1e-3 MW on a de-committed
        # unit, while `award` has that phantom multiplied away
        tol = 1e-3 + 1e-5
        assert (delta - (ramp_up[:, None] + start)).max() < tol
        assert (-delta - (ramp_dn[:, None] + stop)).max() < tol
        # and the boundary carried forward is the final period's award, which is
        # what the next day's period 0 will ramp from
        np.testing.assert_allclose(np.asarray(after.p_final), award[:, -1],
                                   rtol=1e-6)


def test_power_balance_over_the_day(built, rollout):
    """Generation plus shed equals the demand of **that** day.

    Summed over the day rather than per period, which the clearing L1 already
    covers per period.  What this adds is the day index: an environment reading
    `actual[day + 1]` or a transposed commitment would balance against the wrong
    day, and every other check here would still pass.
    """
    env, spec, params = built
    steps, _ = rollout
    for before, after, _, costs, info in steps:
        demand = float(np.asarray(params.actual[before.cursor]).sum())
        served = float(np.asarray(after.award_prev).sum()) + float(costs[0, 0])
        np.testing.assert_allclose(served, demand, rtol=1e-5)
        assert float(info["shed_mwh"]) == pytest.approx(float(costs[0, 0]))


def test_award_within_committed_capacity(built, rollout):
    env, spec, params = built
    steps, _ = rollout
    p_min, p_max = spec["clearing"]["p_min"], spec["clearing"]["p_max"]
    for before, after, _, _, info in steps:
        u = np.asarray(info["commitment"], np.float64)
        award = np.asarray(after.award_prev, np.float64)
        assert (award >= p_min[:, None] * u - 1e-3).all()
        assert (award <= p_max[:, None] * u + 1e-3).all()
        assert np.abs(award[u == 0]).max() < 1e-6, "a de-committed unit generated"


def test_costs_channel_carries_energy_and_counts_only(built, rollout):
    """§16: `costs` holds the two feasibility quantities.

    The VOLL cost is money and stays in `info`; putting it in `costs` would make a
    price into a constraint.  The check is dimensional: column 0 is megawatt-hours
    and the VOLL cost is 10 000 times it, so the two cannot be confused silently.
    """
    env, spec, params = built
    steps, _ = rollout
    assert spec["cost_names"] == ("shed_energy_mwh", "min_up_down_violation")
    for _, _, reward, costs, info in steps:
        assert costs.shape == (spec["n_agents"], 2)
        assert float(info["voll_cost"]) == pytest.approx(VOLL * float(costs[0, 0]))
        # the shed column is a system quantity, reported identically to everyone
        assert float(np.ptp(np.asarray(costs[:, 0]))) == 0.0
        assert (np.asarray(costs) >= 0).all()


def test_the_solved_commitment_sheds_nothing_and_breaks_windows(built, rollout):
    """What the commitment this market solves for actually does.

    Two behaviours, recorded rather than merely asserted, and the second is the
    one that changed when the commitment stopped coming from the offline sweep.

    Shed stays zero on every day of **this** window, which is three days at
    `T = 4` and `cap_scale = 0.6`.  That is a property of the window and not of the
    route, and the distinction matters because the two differ: measured
    independently over sixty days at `T = 24` and a markup of one, the chain this
    environment now runs sheds on four of them, 377.1 MWh in total, while the
    offline fixture chain that carried (MU)/(MD) shed on none.  So the assertion
    below is a fixture-window regression guard; a wider window is expected to shed
    and must not be made to pass by loosening it.

    The (MU)/(MD) violation count does **not** stay zero, and that is by
    construction rather than a defect: step 1' drops (MU)/(MD) from the relaxation,
    so nothing stops step 2 from rounding to a
    pattern that switches a unit faster than its window allows.  Measured over
    this episode at a markup of 1.3: one violation on three of the five days and
    none on the other two, against one to three in-day switches per day.  Both
    solves converge throughout, with the relaxation near 1.96e-11 and the dispatch
    near 1.06e-11.  A day with zero switches would have nothing to violate, so the
    assertion below is that the count is not identically zero across the episode,
    which is what distinguishes a live counter from a dead one.
    """
    steps, _ = rollout
    total = 0.0
    for _, _, _, costs, info in steps:
        assert float(costs[0, 0]) == pytest.approx(0.0, abs=1e-6)
        assert bool(info["converged"])
        total += float(costs[:, 1].sum())
    assert total > 0.0, ("no window was broken anywhere in the episode; either the "
                         "counter is dead or step 1' has started enforcing (MU)/(MD)")


def test_violation_count_matches_an_independent_recount(case, built, rollout):
    """The (MU)/(MD) counter, checked against §6.3 recomputed from the outputs.

    The commitment can no longer be injected: it is solved from the offers, so a
    hand-made infeasible pattern cannot be fed to `step` any more.  Nor can the
    boundary be used to manufacture one, because the relaxation writes the forced
    states in -- a unit that has not served its minimum uptime is held on, which is
    the opposite of a violation.  Violations therefore arise only from switching
    **inside** the day, which step 1' permits because it carries no (MU)/(MD) rows.

    So the check becomes a recount rather than an injection.  The loop below is
    written from §6.3's definition (a unit that stops having run fewer than UT
    periods violates the uptime window; a unit that starts having been off fewer
    than DT violates the downtime window) as a plain Python loop over periods,
    which is a different shape from the `lax.scan` in the environment, so it is a
    second implementation rather than a transcription of the first.

    Measured over this episode: three of the five days carry one violation and two
    carry none, so the comparison is not vacuous; the assertion on the total is
    what guarantees that.
    """
    steps, _ = rollout
    UT = np.maximum(np.asarray(case.unit_min_up_time, np.int32), 1)
    DT = np.maximum(np.asarray(case.unit_min_down_time, np.int32), 1)
    total = 0.0
    for before, _, _, costs, info in steps:
        u = np.asarray(info["commitment"], np.float64) > 0.5
        prev = np.asarray(before.commitment_status, np.float64) > 0.5
        up = np.asarray(before.up_time, np.int64).copy()
        down = np.asarray(before.down_time, np.int64).copy()
        expected = np.zeros(u.shape[0])
        for t in range(u.shape[1]):
            now = u[:, t]
            started, stopped = (~prev) & now, prev & (~now)
            expected += (stopped & (up < UT)) | (started & (down < DT))
            up = np.where(now, up + 1, 0)
            down = np.where(now, 0, down + 1)
            prev = now
        got = np.asarray(costs[:, 1], np.float64)
        np.testing.assert_array_equal(got, expected)
        total += expected.sum()
    assert total > 0.0, "the recount found nothing to compare; see the test above"


def test_start_up_is_not_recharged_across_the_day_boundary(case, built, rollout):
    """§4's `commitment_status`: a unit already running does not pay to start again.

    **One** solve, settled against two candidate cost models: the boundary the
    state carries, and a boundary forced to zero.  The reward must match the first
    and must differ from the second by exactly the start-up cost of every unit that
    was already running.  Without the carry the market recharges a start every day,
    which is a five-figure error per day that no other check reacts to.

    Re-running `step` with the boundary zeroed is what this test used to do, and it
    stopped being available when the commitment became a function of the offers:
    the boundary enters the relaxation, so perturbing it changes which units are
    committed and the two runs are then no longer the same day.  Holding one solve
    and varying only the cost model isolates the same property without that
    confound, and it keeps the discriminating power in the same place -- a
    settlement that ignored `commitment_status` would match the zero boundary.
    """
    env, spec, params = built
    steps, action = rollout
    before, after, reward, _, info = steps[1]
    u = np.asarray(info["commitment"], np.float64)
    award = np.asarray(after.award_prev, np.float64)
    lmp = np.asarray(after.lmp_prev, np.float64)
    revenue = (lmp.T[spec["unit_bus"]] * award).sum(1)
    carried = np.asarray(before.commitment_status, np.float64)

    with_carry = revenue - _true_cost(case, award, u, carried)
    without = revenue - _true_cost(case, award, u, np.zeros_like(carried))
    np.testing.assert_allclose(np.asarray(reward, np.float64), with_carry,
                               rtol=MONEY_RTOL, atol=1.0)

    startup = np.asarray(case.unit_startup_cost, np.float64)
    already_on = startup * np.minimum(carried, u[:, 0])
    np.testing.assert_allclose(with_carry - without, already_on,
                               rtol=MONEY_RTOL, atol=1.0)
    assert already_on.sum() > 0, "no unit ran across this boundary; nothing is proved"


# --------------------------------------------------------------------------
# learner_mask: one code path, not a second interface
# --------------------------------------------------------------------------

def test_learner_mask_makes_non_learners_bid_their_cost(built):
    """A fully masked-off population must ignore the action it is handed.

    Two very different actions, bit-identical rewards, and both equal to the
    reward of the all-learning population bidding the truthful action.  That last
    equality is what makes this a check of §9.3 rather than of the mask: the
    truthful action of the markup map is its own lower bound, 1.
    """
    env, spec, params = built
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key, params)
    step = jax.jit(env.step)
    off = params.replace(learner_mask=jnp.zeros((spec["n_agents"],), bool))

    a1 = jnp.full((spec["n_agents"],), 1.0)
    a2 = jnp.full((spec["n_agents"],), 1.9)
    r1 = step(key, state, a1, off)[2]
    r2 = step(key, state, a2, off)[2]
    np.testing.assert_array_equal(np.asarray(r1), np.asarray(r2))
    r_true = step(key, state, truthful_action(load_case("29gb"), K, T, "markup"),
                  params)[2]
    np.testing.assert_array_equal(np.asarray(r1), np.asarray(r_true))
    # and the mask must actually be doing something: with everyone learning, the
    # marked-up action has to move the reward
    r_learn = step(key, state, a2, params)[2]
    assert not np.allclose(np.asarray(r_learn), np.asarray(r2))


def test_learner_mask_reaches_exactly_the_masked_agents(case, built):
    """A partly-learning population, on the action space where the mask can slip.

    `kind="full"` has action shape `(N, K, T)`, so the mask has to broadcast along
    two axes it does not own.  Changing the action of a masked-off agent must leave
    the day bit-identical; changing the action of a learning agent must not.

    Both halves are aimed at units whose offer can matter, which is not every
    unit: §19 leaves the must-run block `p_min * u` without an offer price of its
    own, so a committed unit with no accepted segment is dispatched and paid the
    same whatever it bids.  Picking a unit by index instead of by accepted
    quantity gives a test that passes for the wrong reason -- unit 0 of this
    scenario is exactly such a unit.
    """
    env, spec = make_env(load_case("29gb"), load_commitment(n_periods=T),
                         load_gb_demand(), n_segments=K, kind="full",
                         cap_scale=CAP_SCALE, ramp_scale=RAMP_SCALE)
    N = spec["n_agents"]
    all_learning = env.make_params(episode_len=EPISODE_LEN)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key, all_learning)
    step = jax.jit(env.step)

    base = jnp.zeros((N, K, T))
    _, after, _, _, _, info = step(key, state, base, all_learning)
    u = np.asarray(info["commitment"], np.float64)
    accepted = (np.asarray(after.award_prev, np.float64)
                - spec["clearing"]["p_min"][:, None] * u).sum(1)
    learner, other = (int(i) for i in np.argsort(accepted)[::-1][:2])
    assert accepted[learner] > 1.0 and accepted[other] > 1.0

    mask = np.zeros(N, bool)
    mask[learner] = True
    params = all_learning.replace(learner_mask=jnp.asarray(mask))
    r0 = step(key, state, base, params)[2]
    silent = step(key, state, base.at[other].set(4.0), params)[2]   # masked off
    heard = step(key, state, base.at[learner].set(4.0), params)[2]  # learning
    np.testing.assert_array_equal(np.asarray(r0), np.asarray(silent))
    assert not np.allclose(np.asarray(r0), np.asarray(heard))


# --------------------------------------------------------------------------
# The price criterion, on an uncongested scenario (§16)
# --------------------------------------------------------------------------

def test_price_is_a_marked_up_unit_offer_when_nothing_binds(case, x64):
    """§16's price criterion, at the level of the environment rather than the LP.

    With no line binding and no bus shedding the congestion and shed-bound terms of
    §7 vanish, so the price is the offer of the marginal unit -- and in this
    environment that offer is the *agent's*, so the price has to be the markup
    times some unit's envelope cost.  `test_clearing_l1.py` pins the same identity
    with offers handed to the operator directly; what this adds is the action map
    and the mask sitting in front of it, which is the path an agent's decision
    actually travels.

    It needs its own fixture, committed at `cap_scale = 1.0` with ramp effectively
    off, because at the cap_scale the rest of this module uses, line-periods bind on
    every day: `info["congested_line_periods"]` over this module's own three-day
    episode is 7, 7, 8 at the adopted cap 0.6 / ramp 1.0, and was 16, 16, 16 at the
    retired cap 0.4 / ramp 0.25 (measured 2026-08-17).  Fewer, but never zero, so
    the separate uncongested fixture is still required.
    """
    fixture = load_commitment(
        n_periods=T,
        path=FIXTURE_DIR / "day_ahead_commitment_29gb_T4_relax_uncongested.npz")
    env, spec = make_env(case, fixture, load_gb_demand(), n_segments=K,
                         kind="markup", markup_max=3.0, cap_scale=1.0,
                         ramp_scale=2.0)
    params = env.make_params(episode_len=1)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key, params)
    from powermarketjax.envs.day_ahead import segment_costs
    _, envelope = segment_costs(case, K)

    for markup in (1.0, 1.4):
        _, after, _, costs, _, info = jax.jit(env.step)(
            key, state, jnp.full((spec["n_agents"],), markup), params)
        assert int(info["congested_line_periods"]) == 0, "scenario congested"
        assert float(costs[0, 0]) < 1e-6, "scenario shed; §7's rho term is live"
        lmp = np.asarray(after.lmp_prev, np.float64)
        assert np.ptp(lmp, axis=1).max() < 1e-3, "prices are not uniform"
        offers = markup * np.asarray(envelope)[:, 0]
        gap = np.abs(lmp[:, 0][:, None] - offers[None, :]).min(1)
        assert gap.max() < 1e-2, f"no unit offers at the clearing price: {gap.max()}"


# --------------------------------------------------------------------------
# Line flows and money balance, recomputed from the state (§16).  Both need the
# nodal shed to be zero, which every day of the chained fixture is, so each test
# asserts that first rather than assuming it.
# --------------------------------------------------------------------------

def test_line_flows_within_rating_and_the_congestion_count_matches(built, rollout):
    """§16: flows respect the scaled rating unless shed is active.

    The count of congested line-periods that `step` reports is new arithmetic
    rather than something the clearing operator returns, so it is recounted here
    from the flows.  A count derived from the wrong rating, or from an untransposed
    PTDF, would be a plausible number that nothing else reacts to.
    """
    env, spec, params = built
    steps, _ = rollout
    PTDF = spec["clearing"]["PTDF"]
    F = np.asarray(load_case("29gb").line_cap, np.float64) * CAP_SCALE
    share = spec["clearing"]["demand_share"]
    for before, after, _, costs, info in steps:
        assert float(costs[0, 0]) < 1e-6, "shed is active; flows may exceed the rating"
        award = np.asarray(after.award_prev, np.float64)
        gen = np.zeros((T, spec["n_buses"]))
        np.add.at(gen, (slice(None), spec["unit_bus"]), award.T)
        demand = share[None, :] * np.asarray(params.actual[before.cursor], np.float64)[:, None]
        flow = (gen - demand) @ PTDF.T
        assert (np.abs(flow) - F[None, :]).max() < 1e-2
        recount = int((np.abs(flow) >= F[None, :] - 1e-3).sum())
        assert abs(recount - int(info["congested_line_periods"])) <= 1, \
            f"reported {int(info['congested_line_periods'])} congested cells, recount {recount}"
        assert recount > 0, "scenario did not congest; the count proves nothing"


def test_money_balance_leaves_a_non_negative_congestion_rent(built, rollout):
    """§8, at the level of the environment: load pays at least what generation earns.

    The identity of §8 is verified against the duals in `test_clearing_l1.py`.  What
    is checked here is the settlement wiring: the revenue the environment reports
    per agent has to be the nodal payment recomputed from the price at each unit
    bus, and the difference between charges and payments is the congestion rent,
    which is non-negative because the line duals and the ratings are.  A unit
    mapped to the wrong bus passes every other check in this module.
    """
    env, spec, params = built
    steps, _ = rollout
    share = spec["clearing"]["demand_share"]
    for before, after, _, costs, info in steps:
        assert float(costs[0, 0]) < 1e-6, "shed is active; the rho rent term is live"
        lmp = np.asarray(after.lmp_prev, np.float64)
        award = np.asarray(after.award_prev, np.float64)
        payments = float((lmp.T[spec["unit_bus"]] * award).sum())
        demand = share[None, :] * np.asarray(params.actual[before.cursor], np.float64)[:, None]
        charges = float((lmp * demand).sum())
        np.testing.assert_allclose(payments, float(np.asarray(info["revenue"]).sum()),
                                   rtol=MONEY_RTOL)
        rent = charges - payments
        assert rent > -1e-5 * abs(charges), f"payments exceed charges by {-rent:.3e}"
        assert rent > 0.0, "scenario has no congestion rent; the check is vacuous"


def test_commitment_status_agrees_with_the_run_lengths(built, rollout):
    """`commitment_status > 0` iff `up_time > 0`, at every boundary.

    The two are redundant by construction, and the redundancy is kept so
    that this check exists.  It is the one that catches a run-length recursion
    drifting out of step with the commitment carry: the recursion runs over the
    periods in a `lax.scan` while `commitment_status` is read straight off the last
    period, so an off-by-one in either would leave a unit recorded as running with a
    zero run length, and nothing else in this module reacts to that.  Exactly one of
    the two counters is non-zero for a unit, which is what the offline sweep's
    boundary convention also assumes.
    """
    env, spec, params = built
    steps, _ = rollout
    _, state0 = env.reset(jax.random.PRNGKey(0), params)
    for state in [state0] + [after for _, after, _, _, _ in steps]:
        on = np.asarray(state.commitment_status) > 0.5
        up = np.asarray(state.up_time) > 0
        down = np.asarray(state.down_time) > 0
        np.testing.assert_array_equal(on, up)
        np.testing.assert_array_equal(~on, down)
        assert not (up & down).any(), "a unit is both running and stopped"


def test_the_markup_ceiling_stays_feasible_over_a_chained_episode(case, built):
    """The failure class that made the commitment endogenous dangerous at all.

    With the commitment fixed before bidding, offering away from true cost
    redistributed the dispatch, left a unit at a day-end output the next day could
    not ramp down from, and made step 3 **infeasible** rather than merely
    expensive: 11 of 59 day transitions at a markup ceiling of 2, with one of them
    putting a NaN into the state that every later day inherited.  Solving the commitment from the offers
    does not by itself remove that class, because the boundary still carries
    forward; it only removes the mismatch between a schedule and the offers that
    were not used to build it.  The shut-down allowance of §6.3 being `p_max * w`
    is what makes the transition reachable, and this is the assertion that says so
    at the one action where the effect is largest.

    The check reads step 3's `mu`, not the finiteness of the award.  That is the
    lesson of the original failure: an infeasible input returns prices that look
    perfectly ordinary while `mu` sits above 1e260, so finiteness is not a gate and
    a test that asserted it would have passed throughout.

    **The markups are drawn per agent, not set to a common ceiling.**  A common
    ceiling multiplies every offer by the same factor, which leaves the merit order
    unchanged and therefore barely moves the dispatch, so it does not exercise this
    at all: with the allowance reverted to `p_min` a uniform ceiling of 2 still
    converges on every day.  The original measurement drew `U[1, alpha_bar]` per
    agent per day for the same reason, and heterogeneous markups are what reorder
    the merit list and leave units at outputs the next day has to descend from.

    Teeth, measured: with the shut-down allowance of §6.3 reverted to `p_min` this
    test fails; under `p_max` it passes.  If it ever stops failing under the
    reverted allowance, the scenario has stopped reaching the boundary case and the
    guard is no longer guarding anything.

    This exercises the `mu` half of `converged`, and there is no residual half to
    exercise, which is a result rather than a gap: the shape that a residual gate
    would catch **does occur** on these operators, in two of the eighteen scenarios
    the solver was calibrated on, and on those two the rounded commitment matches
    the reference cell for cell, so the residual is large where nothing is wrong
    (measured by a separate precision study).  That is why `converged` carries
    no residual term, and why one should not be added without first repeating that
    measurement.
    """
    env, spec, params = built
    key = jax.random.PRNGKey(0)
    step = jax.jit(env.step)
    _, state = env.reset(key, params)
    reachable = 0
    for day in range(EPISODE_LEN):
        markup = jax.random.uniform(jax.random.fold_in(key, day),
                                    (spec["n_agents"],), minval=1.0,
                                    maxval=MARKUP_MAX)
        before = state
        _, state, reward, _, _, info = step(key, state, markup, params)
        # the guard on the injection: the failure it reproduces needs a unit
        # entering the day above p_min + R_dn, since that is the one a p_min
        # allowance cannot bring to zero.  Nothing in this file mentions
        # `cap_scale`, yet a looser network changes the dispatch and can empty
        # the scenario, so count the configuration rather than assume it
        p_prev = np.asarray(before.p_final, np.float64)
        floor = spec["clearing"]["p_min"] + np.asarray(
            case.unit_ramp_down, np.float64) * spec["clearing"]["p_max"] * RAMP_SCALE
        reachable += int((p_prev > floor + 1e-6).sum())
        assert float(info["mu"]) < MU_TOL, \
            f"day {day}: step 3 did not converge, mu={float(info['mu']):.2e}"
        assert float(info["mu_relax"]) < RELAX_MU_TOL, \
            f"day {day}: step 1' did not converge, mu_relax={float(info['mu_relax']):.2e}"
        assert bool(info["converged"])
        assert np.isfinite(np.asarray(reward)).all(), f"day {day}: reward is not finite"
        assert np.isfinite(np.asarray(state.p_final)).all(), \
            f"day {day}: the boundary carried a non-finite output forward"
    assert reachable > 0, (
        "no unit entered any day above p_min + R_dn, so the shut-down allowance "
        "is never exercised and reverting it would not fail: the scenario has "
        "stopped reproducing the failure this test guards")


def test_the_commitment_is_solved_and_not_read_from_the_fixture(built, rollout):
    """The environment must not be quietly running the fixture's schedule.

    The fixture still carries a `commitment` array and still has to: the real-time
    market builds its day-ahead position on top of it.  So the presence of that key
    cannot be an error, and a guard that rejected it would break a fixture the rest
    of the repository depends on.  What can be checked instead is the positive
    statement -- that what `step` clears is not what the fixture holds for the same
    day.

    The two are close but never equal, which is the reason this needs an assertion
    rather than an eyeball: the fixture is relaxed-and-rounded from true costs with
    (MU)/(MD) enforced, step 1' drops those rows and runs on the submitted offers,
    and at a markup of 1.3 the two differ by 10, 4 and 3 unit-periods out of 264
    over this episode.  A wiring error that read the fixture would make the
    difference exactly zero on every day, which no other test in this module would
    notice, since every one of them takes the commitment from `info`.
    """
    env, spec, params = built
    steps, _ = rollout
    fixture_u = np.asarray(load_commitment(n_periods=T)["commitment"]) > 0.5
    differences = []
    for before, _, _, _, info in steps:
        solved = np.asarray(info["commitment"]) > 0.5
        differences.append(int((solved != fixture_u[int(before.cursor)]).sum()))
    assert all(d > 0 for d in differences), (
        f"the cleared commitment equals the fixture's on some day ({differences}); "
        "the environment is reading the schedule rather than solving for it")
