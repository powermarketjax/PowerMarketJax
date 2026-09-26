"""L0 and L1 for the ancillary environment (§12).

Two assertions here exist because a rule that is only written down is not
enforced, and both were required by review rather than invented:

* `offer_separation` must be computed on the **mapped offers**, not on the raw
  actions.  Two agents whose raw actions differ but whose offers coincide are
  tied, and that is the commonest case in early training, since softplus
  underflows to exactly zero.  A metric computed on actions would call that
  point well separated, which is exactly backwards.
* the reward must not depend on `converged`.  Asserted by rebuilding the
  environment with a tolerance that makes the same solve converge, and requiring
  the reward to be bit-identical.  Without this, a later "discount the reward
  when the solve is poor" branch would turn the same action in the same state
  into two different payoffs, which breaks the Markov property rather than
  merely adding noise.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from powermarketjax.case import load_case
from powermarketjax.envs.ancillary.action import (TRUTHFUL_ACTION,
                                                  offer_separation)
from powermarketjax.envs.ancillary.env import (AncillaryParams,
                                               make_ancillary_env)
from powermarketjax.envs.day_ahead.demand import load_gb_demand
from tests.envs.ancillary.test_clearing_l0 import FIXTURE

CASE = "29gb"
THETA, VOLR, BETA = (1.0 / 6.0, 0.5), 250.0, (0.020, 0.050)
PI_SCALE, DELTA, DAYS, EPISODE = 50.0, 0.5, 4, 4


@pytest.fixture
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _build(x64_unused=None, mu_tol=None, learner_mask=None):
    case = load_case(CASE)
    fx = np.load(FIXTURE, allow_pickle=True)
    idx = fx["day_index"]
    _, actual, _ = load_gb_demand()
    u = np.repeat(fx["commitment"][:DAYS], 2, axis=2).transpose(0, 2, 1).reshape(-1, 66)
    demand = np.repeat(actual[idx[:DAYS]], 2, axis=1).reshape(-1)
    pmin = np.asarray(case.unit_p_min, np.float64)
    pmax = np.asarray(case.unit_p_max, np.float64)
    # The day-ahead schedule doubles as the previous dispatch that (RMP) reads
    # at reset, so it has to be a schedule that actually serves the demand.  A
    # schedule at 1.5 times the committed minimum leaves the fleet unable to
    # reach demand within one half-hour of ramp: measured, that sheds 7 333 MW,
    # pins every price at VOLL, and makes the market indifferent to every
    # offer, which is the trivialised operating point §20 forbids comparing at.
    room = np.maximum(((pmax - pmin) * u).sum(1, keepdims=True), 1.0)
    frac = np.clip((demand[:, None] - (pmin * u).sum(1, keepdims=True)) / room,
                   0.0, 1.0)
    q_da = (pmin + frac * (pmax - pmin)) * u
    kwargs = dict(n_segments=1, cap_scale=0.6, ramp_scale=1.0,
                  period_hours=DELTA, kind="markup", markup_max=2.0)
    if mu_tol is not None:
        kwargs["mu_tol"] = mu_tol
    env = make_ancillary_env(case, THETA, VOLR, BETA, PI_SCALE, **kwargs)
    mask = jnp.ones(66, bool) if learner_mask is None else learner_mask
    params = AncillaryParams(
        demand=jnp.asarray(demand), forecast=jnp.asarray(demand * 1.02),
        commitment=jnp.asarray(u), q_da=jnp.asarray(q_da),
        lmp_da=jnp.full((len(demand), int(case.n_nodes)), 40.0),
        learner_mask=mask, episode_len=EPISODE)
    return env, params


def _action(energy=1.2, reserve=0.5, n=66, n_prod=2, spread=0.0):
    res = reserve + spread * np.linspace(-1.0, 1.0, n)[:, None]
    return jnp.concatenate([jnp.full((n, 1), energy),
                            jnp.asarray(np.repeat(res, n_prod, axis=1))], axis=1)


def _assert_operating_point_reacts(reset, step, params):
    """A fixture precondition, not a property under test.

    An operating point where the market cannot respond to any offer passes
    every comparison while testing nothing, and this repository has met that in
    several forms: the ramp allowance pinning output, the capacity never being
    scarce, the whole period shedding at VOLL.  Rather than enumerate the
    mechanisms, perturb one learner's offer and require the result to move.
    Failing here means the fixture is wrong, which is why it is checked once at
    construction instead of being discovered through a puzzling assertion.

    **This does not cover an infeasible fixture, and that was measured rather
    than assumed.**  A commitment that the network cannot serve produces garbage
    rather than a solution, and garbage still moves when an offer moves, so this
    assertion passes on it: measured on a masked eight-unit commitment and on an
    all-units commitment, both of which both solvers refuse, the perturbation
    changed the output every time while `mu` sat at 1e+09 and 1e+15.
    Infeasibility therefore needs the convergence gate, which is asserted
    separately below, and the two together are what make the fixture sound.
    """
    _, state = jax.jit(reset)(jax.random.PRNGKey(0), params)
    quiet = np.asarray(_action(energy=1.1, reserve=0.2, spread=0.2))
    loud = quiet.copy()
    loud[0, :] = np.array([2.0, 6.0, 6.0])
    a = jax.jit(step)(jax.random.PRNGKey(1), state, jnp.asarray(quiet), params)
    b = jax.jit(step)(jax.random.PRNGKey(1), state, jnp.asarray(loud), params)
    moved = (not np.array_equal(np.asarray(a[2]), np.asarray(b[2]))
             or not np.array_equal(np.asarray(a[5]["lmp"]),
                                   np.asarray(b[5]["lmp"]))
             or not np.array_equal(np.asarray(a[5]["reserve_price"]),
                                   np.asarray(b[5]["reserve_price"])))
    assert moved, (
        "this operating point does not react to an offer, so every comparison "
        "on it would pass without testing anything; the fixture must change")
    assert bool(a[5]["converged"]), (
        f"this operating point does not solve (mu {float(a[5]['mu']):.1e}); a "
        "commitment the network cannot serve returns garbage that still reacts "
        "to an offer, so the reaction check above cannot see it")


def test_the_fixture_operating_point_is_not_trivial(x64):
    (reset, step, _, _), params = _build()
    _assert_operating_point_reacts(reset, step, params)


def test_obs_dim_matches_the_columns_on_a_ladder_that_is_not_two_products(x64):
    """`spec["obs_dim"]` is the width of the observation, at any ``n_prod``.

    The two-product ladder every other test runs on cannot see this: the
    published width and the columns agree there by coincidence, so the shape
    assertion in `test_reset_and_step_shapes` is already on the passing side of
    the question before anything is changed.  A three-product ladder separates
    them -- measured, a hand-written width published 20 for 19 columns -- so
    the ladder is what this test exists to vary.  Nothing else about the run
    matters, so the exogenous series here are the smallest ones that index.
    """
    case = load_case(CASE)
    n_units, n_buses = 66, int(case.n_nodes)
    env = make_ancillary_env(case, (1.0 / 6.0, 0.5, 1.0), VOLR,
                             (0.020, 0.050, 0.010), PI_SCALE,
                             n_segments=1, cap_scale=0.6, ramp_scale=1.0,
                             period_hours=DELTA, kind="markup", markup_max=2.0)
    (reset, _step, _sar, spec) = env
    assert spec["n_prod"] == 3
    n_periods = 2
    demand = jnp.full((n_periods,), 20_000.0)
    u = jnp.ones((n_periods, n_units))
    pmin = jnp.asarray(np.asarray(case.unit_p_min, np.float64))
    pmax = jnp.asarray(np.asarray(case.unit_p_max, np.float64))
    params = AncillaryParams(
        demand=demand, forecast=demand,
        commitment=u, q_da=(pmin + 0.5 * (pmax - pmin)) * u,
        lmp_da=jnp.full((n_periods, n_buses), 40.0),
        learner_mask=jnp.ones((n_units,), bool), episode_len=1)

    obs, _state = jax.jit(reset)(jax.random.PRNGKey(0), params)
    assert obs.shape == (spec["n_agents"], spec["obs_dim"])
    # and the width does move with the ladder, or the assertion above would
    # hold for a width that ignored `n_prod` altogether
    (_r2, _s2, _a2, spec2) = make_ancillary_env(
        case, THETA, VOLR, BETA, PI_SCALE, n_segments=1, cap_scale=0.6,
        ramp_scale=1.0, period_hours=DELTA, kind="markup", markup_max=2.0)
    assert spec["obs_dim"] - spec2["obs_dim"] == 3


def test_reset_and_step_shapes(x64):
    (reset, step, _, spec), params = _build()
    obs, state = jax.jit(reset)(jax.random.PRNGKey(0), params)
    assert obs.shape == (spec["n_agents"], spec["obs_dim"])
    obs2, _, reward, costs, done, info = jax.jit(step)(
        jax.random.PRNGKey(1), state, _action(), params)
    assert obs2.shape == obs.shape
    assert reward.shape == (spec["n_agents"],)
    assert costs.shape == (spec["n_agents"], spec["costs_dim"])
    assert done.shape == ()
    assert info["terminal_obs"].shape == obs.shape
    for key in ("mu", "dual_residual", "converged",
                "offer_separation_all", "offer_separation_learners"):
        assert key in info, key


def test_obs_is_a_pure_function_of_state_and_params(x64):
    (reset, step, _, spec), params = _build()
    _, state = jax.jit(reset)(jax.random.PRNGKey(0), params)
    obs, next_state, *_ = jax.jit(step)(jax.random.PRNGKey(1), state,
                                        _action(), params)
    recomputed = jax.jit(spec["get_obs"])(next_state, params)
    np.testing.assert_array_equal(np.asarray(obs), np.asarray(recomputed))


def test_episode_ends_on_the_time_limit_and_carries_its_successor(x64):
    (reset, step, _, spec), params = _build()
    assert spec["termination"] == "truncation"
    key = jax.random.PRNGKey(0)
    _, state = jax.jit(reset)(key, params)
    for i in range(EPISODE):
        obs, state, _, _, done, info = jax.jit(step)(
            jax.random.PRNGKey(i + 1), state, _action(), params)
    assert bool(done)
    # after the final step the observation is the reset one, while the info
    # carries the observation the episode would have continued into
    assert not np.array_equal(np.asarray(obs),
                              np.asarray(info["terminal_obs"]))


def test_vmap_over_environments(x64):
    (reset, step, _, spec), params = _build()
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    obs, state = jax.jit(jax.vmap(reset, in_axes=(0, None)))(keys, params)
    assert obs.shape == (4, spec["n_agents"], spec["obs_dim"])
    out = jax.jit(jax.vmap(step, in_axes=(0, 0, None, None)))(
        keys, state, _action(), params)
    assert out[2].shape == (4, spec["n_agents"])


def test_scan_over_an_episode(x64):
    (reset, step, step_auto, _), params = _build()
    _, state = jax.jit(reset)(jax.random.PRNGKey(0), params)

    def body(carry, key):
        st = carry
        _, st, reward, _, done, _ = step_auto(key, st, _action(), params)
        return st, (reward, done)

    _, (rewards, dones) = lax.scan(body, state,
                                   jax.random.split(jax.random.PRNGKey(2), 6))
    assert rewards.shape == (6, 66)
    assert bool(dones[EPISODE - 1])


def test_learner_mask_replaces_the_action_of_a_non_learner(x64):
    """A masked agent bids the truthful baseline whatever it is handed."""
    mask = jnp.asarray(np.arange(66) < 33)
    (reset, step, _, _), params = _build(learner_mask=mask)
    _, state = jax.jit(reset)(jax.random.PRNGKey(0), params)
    base = np.asarray(_action(energy=1.5, reserve=0.5, spread=0.2))
    # vary only the half that is masked: the market must not move at all, since
    # those actions are replaced before they reach the offer map.  Comparing
    # rewards of the masked agents under two *learner* actions would not test
    # this, because their rewards move with the prices the learners set.
    other = base.copy()
    other[33:, :] = np.array([2.0, 9.0, 9.0])
    a, b = jnp.asarray(base), jnp.asarray(other)
    out_a = jax.jit(step)(jax.random.PRNGKey(1), state, a, params)
    out_b = jax.jit(step)(jax.random.PRNGKey(1), state, b, params)
    np.testing.assert_array_equal(np.asarray(out_a[2]), np.asarray(out_b[2]))
    np.testing.assert_array_equal(np.asarray(out_a[5]["reserve_price"]),
                                  np.asarray(out_b[5]["reserve_price"]))
    np.testing.assert_array_equal(np.asarray(out_a[5]["lmp"]),
                                  np.asarray(out_b[5]["lmp"]))
    # and the same variation on the learning half does move it
    other2 = base.copy()
    other2[:33, :] = np.array([2.0, 9.0, 9.0])
    out_c = jax.jit(step)(jax.random.PRNGKey(1), state, jnp.asarray(other2),
                          params)
    assert not np.array_equal(np.asarray(out_a[2]), np.asarray(out_c[2]))


# --------------------------------------------------------------------------
# the two assertions review required


def test_offer_separation_reads_the_mapped_offers_not_the_actions(x64):
    """Raw actions far apart in the saturated region map to the same offer."""
    (reset, step, _, _), params = _build()
    _, state = jax.jit(reset)(jax.random.PRNGKey(0), params)

    def sep(action):
        return np.asarray(jax.jit(step)(jax.random.PRNGKey(1), state, action,
                                        params)[5]["offer_separation_all"])

    spread_out = sep(_action(reserve=0.5, spread=0.4))
    tied = sep(_action(reserve=0.5, spread=0.0))
    # actions that differ by tens but whose offers are all exactly zero
    saturated = jnp.concatenate(
        [jnp.full((66, 1), 1.2),
         jnp.asarray(np.repeat((TRUTHFUL_ACTION
                                + np.linspace(-40.0, 40.0, 66))[:, None], 2, 1))],
        axis=1)
    assert spread_out.min() > 1e-3, spread_out
    np.testing.assert_array_equal(tied, np.zeros_like(tied))
    np.testing.assert_array_equal(sep(saturated), np.zeros_like(tied))


def test_separation_over_a_subset_measures_that_subset():
    """The subset path had a defect the whole-population path could not show:
    the excluded providers sort to the end as infinities, and the difference
    between two of them is not a number, so the minimum came back as nan and
    every subset read as completely tied.  The case that catches it is a
    population where the excluded part is tied and the subset is not, which is
    exactly the shape a learner mask produces."""
    offers = jnp.asarray(np.concatenate([
        np.zeros((6, 2)),                                    # the tied block
        np.repeat(np.array([100.0, 101.0, 102.0])[:, None], 2, axis=1),
    ]))
    mask = jnp.asarray(np.array([False] * 6 + [True] * 3))
    whole = np.asarray(offer_separation(offers))
    subset = np.asarray(offer_separation(offers, mask))
    np.testing.assert_array_equal(whole, np.zeros(2))        # tied block wins
    assert subset.min() > 1e-3, subset                       # the learners are not
    # and a subset that is itself tied still reads as tied
    tied_subset = jnp.asarray(np.array([True] * 6 + [False] * 3))
    np.testing.assert_array_equal(np.asarray(offer_separation(offers,
                                                              tied_subset)),
                                  np.zeros(2))


def test_separation_of_a_tied_profile_is_zero_by_construction():
    """The metric itself, away from the environment."""
    tied = jnp.full((10, 2), 3.0)
    np.testing.assert_array_equal(np.asarray(offer_separation(tied)),
                                  np.zeros(2))
    spread = jnp.asarray(np.repeat(np.linspace(1.0, 2.0, 10)[:, None], 2, 1))
    assert float(offer_separation(spread).min()) > 0.0


def test_the_reward_does_not_depend_on_convergence(x64):
    """Same solve, two tolerances, one converged flag flipped, reward unmoved."""
    (reset, step, _, _), params = _build(mu_tol=1e-30)      # nothing converges
    (_, step_loose, _, _), _ = _build(mu_tol=1e30)          # everything does
    _, state = jax.jit(reset)(jax.random.PRNGKey(0), params)
    action = _action(spread=0.2)
    strict = jax.jit(step)(jax.random.PRNGKey(1), state, action, params)
    loose = jax.jit(step_loose)(jax.random.PRNGKey(1), state, action, params)
    assert not bool(strict[5]["converged"])
    assert bool(loose[5]["converged"])
    np.testing.assert_array_equal(np.asarray(strict[2]), np.asarray(loose[2]))
    np.testing.assert_array_equal(np.asarray(strict[3]), np.asarray(loose[3]))


# --------------------------------------------------------------------------
# `reset_on_day`.  Purely additive; `reset` is unchanged and nothing
# in the environment calls the new entry point.  These four are what make the
# evaluation claim ("every arm scored on the same batch of periods") checkable.
# --------------------------------------------------------------------------

PERIODS_PER_DAY = int(round(24.0 / DELTA))


def test_reset_on_day_opens_the_first_period_of_that_day(x64):
    (reset, _step, _sp, spec), params = _build()
    reset_on_day = spec["reset_on_day"]
    assert int(spec["periods_per_day"]) == PERIODS_PER_DAY
    for day in range(DAYS):
        _obs, state = reset_on_day(jax.random.PRNGKey(7), params, day)
        assert int(state.cursor) == day * PERIODS_PER_DAY, (
            f"day {day} opened at period {int(state.cursor)} rather than "
            f"{day * PERIODS_PER_DAY}")
        assert int(state.step_in_episode) == 0


def test_reset_on_day_ignores_its_key(x64):
    """It takes a key so it can be substituted for `reset`, and uses none.

    If a future change makes the opening state depend on the key, the day would
    stop being the only thing the caller controls, and an evaluation would
    silently vary with a key nobody chose deliberately.
    """
    (_reset, _step, _sp, spec), params = _build()
    a = spec["reset_on_day"](jax.random.PRNGKey(0), params, 2)
    b = spec["reset_on_day"](jax.random.PRNGKey(12345), params, 2)
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def test_reset_on_day_agrees_bitwise_with_reset_on_a_boundary_draw(x64):
    """The two entry points build one carry, not two that look alike.

    Checked against `reset` itself rather than against a hand-built state: the
    opening carry (`p_prev` from the day-ahead schedule) is constructed inside
    the environment, and a hand-built comparison would only restate this test's
    own arithmetic.  Keys are enumerated until one draws a start that already
    sits on a day boundary, which is the only draw where the two must agree.
    """
    (reset, _step, _sp, spec), params = _build()
    wanted, found = 2 * PERIODS_PER_DAY, None
    for i in range(20_000):
        _o, st = reset(jax.random.PRNGKey(i), params)
        if int(st.cursor) == wanted:
            found = st
            break
    assert found is not None, (
        f"no key in 20 000 draws opened period {wanted}, so this test cannot "
        f"compare the two entry points on the draw where they must agree")
    _o2, on_day = spec["reset_on_day"](jax.random.PRNGKey(0), params, 2)
    for a, b in zip(jax.tree_util.tree_leaves(found),
                    jax.tree_util.tree_leaves(on_day)):
        a, b = np.asarray(a), np.asarray(b)
        assert a.dtype == b.dtype and np.array_equal(a, b), (a, b)


def test_reset_draws_inside_days_which_is_what_reset_on_day_fixes(x64):
    """The measured 'it bites' number for the check above.

    `reset` draws uniformly over start *periods*, so the day a draw is filed
    under (`cursor // periods_per_day`) leaves the offset inside that day free.
    Asserting the minority here is what keeps the two entry points from being
    trivially interchangeable: if a later change makes `reset` draw on day
    boundaries, this fails and `reset_on_day` stops carrying content.
    """
    (reset, _step, _sp, spec), params = _build()
    offsets = [int(reset(jax.random.PRNGKey(i), params)[1].cursor)
               % PERIODS_PER_DAY for i in range(200)]
    on_boundary = sum(o == 0 for o in offsets)
    assert on_boundary < len(offsets) // 4, (
        f"{on_boundary}/{len(offsets)} draws already start a day, so `reset` no "
        f"longer draws inside days and the defect `reset_on_day` was added for "
        f"is gone; re-derive the evaluation claim rather than deleting this")
    assert max(offsets) > 0


def test_reset_on_day_jits_and_vmaps(x64):
    (_reset, _step, _sp, spec), params = _build()
    on_day = spec["reset_on_day"]
    jitted = jax.jit(lambda d: on_day(jax.random.PRNGKey(0), params, d))
    assert int(jitted(1)[1].cursor) == PERIODS_PER_DAY
    keys = jax.random.split(jax.random.PRNGKey(0), DAYS)
    days = jnp.arange(DAYS)
    _obs, batch = jax.vmap(lambda k, d: on_day(k, params, d))(keys, days)
    assert np.array_equal(np.asarray(batch.cursor),
                          np.arange(DAYS) * PERIODS_PER_DAY)
