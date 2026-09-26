"""L1 for `tools/benchmark/arms.py`, which had no test of any kind.

Two of the four baseline arms -- truthful offers and the constant instrument --
run through `rollout_action`, and every full-suite run so far has been *blind* to
that file rather than passing it.  What this module pins is the one property the
rest of the benchmark rests on: **the number the arm reports is the number the
environment computed**, not a second quantity that merely looks like it.

The reference is built by a different route from the thing under test.
`rollout_action` reaches its answer through `vmap` over the environment axis and
`lax.scan` over the horizon; the reference below steps a single unbatched state
once, with no `vmap` and no `scan`, and the two are compared exactly.  A test
that re-implemented the batched path would agree with it for reasons that have
nothing to do with correctness.

The key derivation is the one place the reference must copy rather than
re-derive: the day an episode lands on comes out of the key, so a reference that
split keys differently would disagree for a legitimate reason and the test would
be measuring key handling instead of reward wiring.  The derivation is therefore
taken verbatim from `rollout_action` and is the contract this test also pins --
if it changes there and not here, this fails, which is the intended behaviour.
"""
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# `powermarketjax.learning` imports the IPPO module, which needs `optax` from the
# `rl` extra while CI installs `dev`, so a bare import would turn an absent
# optional dependency into a failure.
pytest.importorskip("optax", reason="powermarketjax.learning needs the rl extra")

from powermarketjax.case import load_case
from powermarketjax.envs.day_ahead import load_commitment, load_gb_demand, make_env
from powermarketjax.learning.adapters import unpack_env

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
from benchmark import arms                                       # noqa: E402

T = 4
K = 1
CAP_SCALE = 0.6        # adopted scenario (2026-08-17)
RAMP_SCALE = 1.0       # adopted scenario (2026-08-17); registered rates undiscounted
MARKUP_MAX = 2.0
EPISODE_LEN = 2
ACTION = 1.3           # a fixed interior markup; nothing here depends on the value


@pytest.fixture(scope="module", autouse=True)
def x64():
    """float64 for this module; the clearing operator refuses to build without it."""
    prev, prev_mm = jax.config.jax_enable_x64, jax.config.jax_default_matmul_precision
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    yield
    jax.config.update("jax_enable_x64", prev)
    jax.config.update("jax_default_matmul_precision", prev_mm)


@pytest.fixture(scope="module")
def built(x64):
    case = load_case("29gb")
    fixture = load_commitment(n_periods=T)
    env, spec = make_env(case, fixture, load_gb_demand(), n_segments=K,
                         kind="markup", markup_max=MARKUP_MAX,
                         cap_scale=CAP_SCALE, ramp_scale=RAMP_SCALE)
    four = unpack_env((env, spec))
    params = env.make_params(episode_len=EPISODE_LEN)
    # the *normalised* spec, not the market's own: `baseline_action` is supplied
    # by the adapter and is absent from what `make_env` returns, which is the
    # whole reason `unpack_env` exists
    return four, params, four[3]


def _reference(four, params, action, key, n_envs):
    """One step of the environment reached without `vmap` and without `scan`.

    The key derivation is `rollout_action`'s, for the reason in the module
    docstring; everything after it deliberately is not.
    """
    reset, _step, step_auto_reset, _spec = four
    keys = jax.random.split(key, n_envs + 1)
    _obs, state = jax.vmap(reset, in_axes=(0, None))(keys[:n_envs], params)
    _k, k_env = jax.random.split(keys[n_envs])
    env_keys = jax.random.split(k_env, n_envs)
    # unbatched: pull environment 0 out of the reset state and step it alone
    state0 = jax.tree_util.tree_map(lambda x: x[0], state)
    _o, _nxt, reward, costs, _done, info = step_auto_reset(
        env_keys[0], state0, action, params)
    return reward, costs, info


def test_rollout_action_reports_the_environments_own_reward(built):
    """The arm's reward is the environment's reward, exactly.

    Horizon and environment count are both one, so `rollout_action`'s means are
    over a single element and no reassociation is available to either side; the
    comparison is therefore exact rather than to a tolerance.
    """
    four, params, spec = built
    key = jax.random.PRNGKey(0)
    action = jnp.full((spec["n_agents"],), ACTION)

    got = arms.rollout_action(four, params, action, n_envs=1, horizon=1, key=key)
    reward, costs, info = _reference(four, params, action, key, n_envs=1)

    # non-triviality: an all-zero reward would make every assertion below hold
    # for a rollout that did nothing at all
    assert float(jnp.abs(reward).max()) > 1.0, (
        f"reward is ~zero (max |reward| = {float(jnp.abs(reward).max()):.3e}); "
        "this scenario cannot tell a correct arm from an idle one")
    assert bool(info["converged"]), "reference solve did not converge"

    assert float(got["reward_mean"]) == float(jnp.mean(reward))
    np.testing.assert_array_equal(np.asarray(got["reward_per_agent"]),
                                  np.asarray(reward))
    assert float(got["costs_mean"]) == float(jnp.mean(costs))
    assert float(got["costs_max"]) == float(jnp.max(costs))
    assert float(got["unconverged_frac"]) == 0.0


def test_the_means_are_taken_over_both_axes(built):
    """The reductions over horizon and over environments, which the test above
    cannot see.

    At one environment and one step, "mean over everything" and "take the first"
    are the same operation, so a reduction that silently dropped either axis
    would pass.  Measured: with `reward_mean` changed to `traj["reward"][0]`
    (first horizon step only) or to `traj["reward"][:, 0]` (first environment
    only), the single-step test above still reports 2 passed.  This one runs two
    environments for two steps and compares against an explicit Python loop that
    steps each environment on its own -- no `vmap`, no `scan` -- so both axes
    carry information and either drop is visible.
    """
    four, params, spec = built
    reset, _step, step_auto_reset, _spec = four
    # The seed is chosen, not arbitrary: `reset` draws the start day from the
    # key, and on a six-day fixture two environments land on the same day often
    # enough to matter -- seed 2 gives cursors (2, 2), which makes the vmap axis
    # carry no information and the assertions below vacuous.  Seed 4 gives
    # (0, 3).  The pre-assertions further down are what make this a stated
    # requirement rather than a lucky constant, and they fire if it ever lapses.
    key = jax.random.PRNGKey(4)
    n_envs, horizon = 2, 2
    action = jnp.full((spec["n_agents"],), ACTION)

    got = arms.rollout_action(four, params, action, n_envs=n_envs,
                              horizon=horizon, key=key)

    keys = jax.random.split(key, n_envs + 1)
    states = [reset(keys[i], params)[1] for i in range(n_envs)]
    k = keys[n_envs]
    rewards = []
    for _ in range(horizon):
        k, k_env = jax.random.split(k)
        env_keys = jax.random.split(k_env, n_envs)
        nxt = []
        for i in range(n_envs):
            _o, s, r, _c, _d, _inf = step_auto_reset(env_keys[i], states[i],
                                                     action, params)
            nxt.append(s)
            rewards.append(np.asarray(r))
        states = nxt
    stacked = np.stack(rewards)                       # (horizon * n_envs, agents)

    # the two axes must actually differ, or dropping one would be invisible here
    # for the same reason it is invisible above
    per_step = stacked.reshape(horizon, n_envs, -1)
    assert float(np.abs(per_step[0] - per_step[1]).max()) > 1e-6, \
        "the two horizon steps carry the same reward; this test cannot see the scan axis"
    assert float(np.abs(per_step[:, 0] - per_step[:, 1]).max()) > 1e-6, \
        "the two environments carry the same reward; this test cannot see the vmap axis"

    # Not exact here, and the reason is the reduction rather than the wiring.
    # `reward` is float32 (§15 keeps the state in float32 while the clearing runs
    # in float64), and this mean is over 264 elements, so `jnp.mean` inside the
    # scan and numpy's mean outside it accumulate in different orders.  Measured
    # 2026-08-17: the two differ by 8.98e-08 relative against a float64
    # accumulation of the same rewards, and by 1.73e-07 against a float32 one.
    # One float32 ULP is 1.19e-07 relative, so the disagreement is under a single
    # ULP and `rtol=1e-6` leaves about a factor of eleven.  The single-step test
    # above stays exact precisely because it reduces one element and no
    # reassociation is available to either side: **the two tolerances differ
    # because the element counts differ, not because the two tests disagree
    # about what correctness is.**
    np.testing.assert_allclose(float(got["reward_mean"]),
                               float(stacked.astype(np.float64).mean()),
                               rtol=1e-6, atol=0.0)


def test_rollout_action_system_cost_is_additive_and_opt_in(built):
    """`voll`/`other_shortfall_cost` change nothing by default and, given both,
    report the environment's own `info["cost"]` / `info["shed_mwh"]` correctly.

    Two things this test exists to catch, matched to the two ways the additive
    design could fail silently: (1) passing the new arguments changes a value
    an existing caller already relies on (checked by comparing every
    pre-existing key between the default call and the opt-in call, not just
    eyeballing that they look similar); (2) the new fields are wired to the
    wrong environment output or the wrong arithmetic (checked against the same
    unbatched, unvmapped, unscanned reference `_reference` builds, exactly as
    the reward test above does for `reward`/`costs`).
    """
    four, params, spec = built
    key = jax.random.PRNGKey(0)
    action = jnp.full((spec["n_agents"],), ACTION)
    voll, other_shortfall_cost = 10_000.0, 0.0

    default = arms.rollout_action(four, params, action, n_envs=1, horizon=1,
                                  key=key)
    assert "system_cost_sum_per_env" not in default, (
        "system_cost_sum_per_env present without voll/other_shortfall_cost "
        "being passed -- the opt-in default has changed")

    got = arms.rollout_action(four, params, action, n_envs=1, horizon=1,
                              key=key, voll=voll,
                              other_shortfall_cost=other_shortfall_cost)
    # every pre-existing key must be untouched by opting in to the new ones
    for k in default:
        np.testing.assert_array_equal(
            np.asarray(got[k]), np.asarray(default[k]),
            err_msg=f"opting in to system_cost changed pre-existing key {k!r}")

    _reward, _costs, info = _reference(four, params, action, key, n_envs=1)
    ref_production_cost = float(jnp.sum(info["cost"]))
    ref_shed_mwh = float(info["shed_mwh"])
    # non-triviality: a scenario with zero production cost would make the
    # production-cost assertion below hold for a wire that reads nothing
    assert ref_production_cost > 1.0, (
        f"reference production cost is ~zero ({ref_production_cost:.3e}); "
        "this scenario cannot tell a correct wire from an idle one")

    assert float(got["production_cost_sum_per_env"][0]) == ref_production_cost
    # `shed_mwh` itself is ~2e-20 here (below `SHED_FLOOR`, i.e. numerically
    # zero) rather than exactly reproduced: the vmap+scan reduction and the
    # reference's direct read accumulate in different orders, the same
    # single-ULP-class disagreement `test_the_means_are_taken_over_both_axes`
    # documents for `reward_mean` above, just visible here because the
    # quantity itself is tiny rather than because the wire is wrong
    np.testing.assert_allclose(float(got["shed_mwh_sum_per_env"][0]),
                               ref_shed_mwh, rtol=0, atol=1e-15)
    expected_system_cost = (ref_production_cost + voll * ref_shed_mwh
                            + other_shortfall_cost)
    np.testing.assert_allclose(float(got["system_cost_sum_per_env"][0]),
                               expected_system_cost, rtol=1e-12, atol=0)
    np.testing.assert_allclose(float(got["system_cost_mean"]),
                               expected_system_cost, rtol=1e-12, atol=0)


def test_rollout_action_raises_without_other_shortfall_cost(built):
    """`voll` alone must raise, not silently assume `other_shortfall_cost=0.0`.

    `evaluation.system_cost` takes no default for this argument on purpose
    (no market's convention ships to another's as a default), and
    this wire is required to inherit that rule rather than quietly relaxing it.
    """
    four, params, spec = built
    key = jax.random.PRNGKey(0)
    action = jnp.full((spec["n_agents"],), ACTION)
    with pytest.raises(ValueError, match="other_shortfall_cost"):
        arms.rollout_action(four, params, action, n_envs=1, horizon=1,
                            key=key, voll=10_000.0)


def test_other_shortfall_cost_key_sums_the_named_info_field(built):
    """`other_shortfall_cost_key` locates, sums over the horizon, and adds in
    an `info` field named at call time -- market 03's `info["volr_cost"]`
    wiring, exercised here against day-ahead's fixture rather than a new
    ancillary one.

    This does not claim `info["congested_line_periods"]` is a real third cost
    term for day-ahead -- it is deliberately repurposed as a stand-in so the
    test can pin the *mechanism* (find this key, sum it, add it on top of
    `other_shortfall_cost`) independently of any one market's physics, the
    same way the reward tests above use an unbatched reference rather than a
    second real market to check `vmap`/`scan` wiring. Chosen over
    `info["shed_mwh"]` specifically because that one is supposed to be ~0 on
    this fixture (the market operating correctly, not a defect) and would
    make the non-triviality check below skip on every run; every period on
    this scenario is congested (the scenario-confirmation figures elsewhere
    in this report series measure 8760/8760), so this field is reliably
    nonzero instead of reliably zero.
    """
    four, params, spec = built
    key = jax.random.PRNGKey(0)
    action = jnp.full((spec["n_agents"],), ACTION)
    voll, base = 10_000.0, 5.0

    plain = arms.rollout_action(four, params, action, n_envs=1, horizon=1,
                                key=key, voll=voll, other_shortfall_cost=base)
    keyed = arms.rollout_action(four, params, action, n_envs=1, horizon=1,
                                key=key, voll=voll, other_shortfall_cost=base,
                                other_shortfall_cost_key="congested_line_periods")

    _reward, _costs, info = _reference(four, params, action, key, n_envs=1)
    ref_congested = float(info["congested_line_periods"])
    # non-triviality: a stand-in that happened to be zero would make the
    # assertions below pass whether or not the key was ever read
    assert ref_congested > 0, (
        f"reference congested_line_periods is {ref_congested}; this test "
        "needs a nonzero stand-in value to tell 'read' from 'ignored'")

    assert "other_shortfall_cost_sum_per_env" not in plain, (
        "other_shortfall_cost_sum_per_env present without "
        "other_shortfall_cost_key being passed")
    expected = base + ref_congested
    np.testing.assert_allclose(
        float(keyed["other_shortfall_cost_sum_per_env"][0]), expected,
        rtol=0, atol=1e-9)
    # and it must actually have landed in system_cost, not just in its own
    # key: `keyed` adds one more `ref_congested` on top of `plain` (the
    # stand-in key's own contribution), nothing else changes between the calls
    np.testing.assert_allclose(
        float(keyed["system_cost_sum_per_env"][0]),
        float(plain["system_cost_sum_per_env"][0]) + ref_congested,
        rtol=1e-12, atol=0)


def test_constant_arm_defaults_to_the_baseline_the_adapter_supplies(built):
    """`constant_arm` with no action must use `spec["baseline_action"]`.

    The baseline design separates the truthful point from the best constant action and
    forbids reporting the first as if it were the second, so which action the
    default reaches for is a property worth pinning rather than assuming.  The
    check is that the default agrees with passing that same action explicitly,
    and disagrees with passing a different one -- the second half is what stops
    this from passing for an implementation that ignored `action` entirely.
    """
    four, params, spec = built
    key = jax.random.PRNGKey(1)
    baseline = spec["baseline_action"]

    default = arms.constant_arm(four, params, n_envs=1, horizon=1, key=key)
    explicit = arms.constant_arm(four, params, n_envs=1, horizon=1, key=key,
                                 action=baseline)
    assert float(default["reward_mean"]) == float(explicit["reward_mean"])

    other = arms.constant_arm(four, params, n_envs=1, horizon=1, key=key,
                              action=jnp.full((spec["n_agents"],), 1.9))
    assert float(default["reward_mean"]) != float(other["reward_mean"]), (
        "a different constant action produced the same reward, so this test "
        "cannot tell whether `action` is read at all")
