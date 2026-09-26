"""L0 JAX contract for the real-time environment (§13).

`jit`; `vmap` over parallel environments; a `RealTimeState` pytree whose structure
and dtypes are identical across `jit` and across `done`; an episode under
`lax.scan`; and exactly one clearing per step.

**The dtype half of the pytree check is the load-bearing half**, for the reason
the day-ahead L0 records: `jax_enable_x64` is on for the solver while every state
array is built at float32, so an omitted dtype produces a float64 leaf that
leaves the treedef and the shapes identical.  Only comparing dtypes catches it.
"""
import jax
import jax.extend
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.day_ahead import load_gb_demand
from powermarketjax.envs.real_time import load_da_position
from powermarketjax.envs.real_time.clearing import MAX_ITER
from powermarketjax.envs.real_time.demand import load_gb_demand_half_hourly
from powermarketjax.envs.real_time.env import COST_NAMES, RealTimeState, make_env

EPISODE_LEN = 4
MARKUP_MAX = 2.0

#: The adopted scenario.  Stated here rather than read back from the
#: position's `meta`, so that the fixture and this file remain two independent
#: statements that can be compared -- reading the scales out of the fixture makes
#: any comparison pass by construction.
#:
#: **`real_time.make_env` does not check them**: the refusal that compares a
#: fixture's `meta` against the requested scales lives in `day_ahead.make_env`,
#: and this market's own constructor takes `cap_scale` with a default of 1.0 and
#: validates nothing.  Restoring this constant to 0.4 against the seasonal
#: fixture was measured to break no test at all, so the assertion in the fixture
#: below is the whole guard, not a second opinion.
CAP_SCALE = 0.60
RAMP_SCALE = 1.00
#: `_seasons` is the expand-phase name; batch 3 renames it back to the canonical
#: one, and `grep -rn "_seasons"` is that batch's acceptance criterion.
CHAIN = "step1prime_seasons"


@pytest.fixture(scope="module", autouse=True)
def x64():
    prev = jax.config.jax_enable_x64, jax.config.jax_default_matmul_precision
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    yield
    jax.config.update("jax_enable_x64", prev[0])
    jax.config.update("jax_default_matmul_precision", prev[1])


@pytest.fixture(scope="module")
def built(x64):
    pos = load_da_position(chain=CHAIN)
    assert (pos["meta"]["cap_scale"], pos["meta"]["ramp_scale"]) == \
        (CAP_SCALE, RAMP_SCALE), (
            f"position built at cap {pos['meta']['cap_scale']} / ramp "
            f"{pos['meta']['ramp_scale']}, but this module builds the "
            f"environment at {CAP_SCALE} / {RAMP_SCALE}")
    case = load_case(pos["meta"]["case"])
    hh, _d = load_gb_demand_half_hourly()
    forecast, _a, _days = load_gb_demand()
    env, spec = make_env(case, pos, hh, forecast, n_segments=1,
                         markup_max=MARKUP_MAX, cap_scale=CAP_SCALE,
                         ramp_scale=RAMP_SCALE)
    return env, spec, env.make_params(episode_len=EPISODE_LEN)


def _leaves(state):
    return {n: (getattr(state, n).shape, getattr(state, n).dtype)
            for n in state.__dataclass_fields__}


def test_step_under_jit(built):
    env, spec, params = built
    N = spec["n_agents"]
    obs, state = env.reset(jax.random.PRNGKey(0), params)
    action = env.truthful_action()
    obs2, state2, reward, costs, done, info = jax.jit(env.step)(
        jax.random.PRNGKey(1), state, action, params)
    assert obs2.shape == (N, spec["obs_dim"]) and obs2.dtype == jnp.float32
    assert reward.shape == (N,) and reward.dtype == jnp.float32
    assert costs.shape == (N, len(COST_NAMES)) and costs.dtype == jnp.float32
    assert done.shape == () and done.dtype == jnp.bool_
    assert bool(info["converged"]), f"clearing did not converge: {info['mu']:.2e}"


def test_obs_dim_is_derived_and_not_declared(built):
    """`obs_dim` must be what `get_obs` returns, not a restatement.

    This one is not hypothetical: the first version of this environment computed
    `obs_dim` arithmetically from the concatenation and got 14 against an actual
    16.  The spec value is now produced by evaluating the function, so the two
    cannot disagree; this asserts that they do not.
    """
    env, spec, params = built
    _obs, state = env.reset(jax.random.PRNGKey(0), params)
    assert env.get_obs(state, params).shape[1] == spec["obs_dim"]


def test_state_pytree_identical_across_jit_and_done(built):
    env, spec, params = built
    action = env.truthful_action()
    key = jax.random.PRNGKey(0)
    _obs, state = env.reset(key, params)
    ref = _leaves(state)

    step, auto = jax.jit(env.step), jax.jit(env.step_auto_reset)
    for i in range(EPISODE_LEN):
        _, stepped, _, _, done, _ = step(key, state, action, params)
        assert _leaves(stepped) == ref, f"step changed the layout at period {i}"
        _, state, _, _, done_a, _ = auto(key, state, action, params)
        assert _leaves(state) == ref, f"auto-reset changed the layout at {i}"
        assert bool(done) == bool(done_a) == (i == EPISODE_LEN - 1)


def test_vmap_over_parallel_environments(built):
    """Agents are an array axis, so environments vmap on top of them."""
    env, spec, params = built
    n_env = 4
    keys = jax.random.split(jax.random.PRNGKey(0), n_env)
    obs, states = jax.vmap(env.reset, in_axes=(0, None))(keys, params)
    assert obs.shape == (n_env, spec["n_agents"], spec["obs_dim"])
    action = jnp.broadcast_to(env.truthful_action(),
                              (n_env,) + env.truthful_action().shape)
    obs2, states2, reward, costs, done, info = jax.vmap(
        env.step, in_axes=(0, 0, 0, None))(keys, states, action, params)
    assert reward.shape == (n_env, spec["n_agents"])
    assert costs.shape == (n_env, spec["n_agents"], len(COST_NAMES))
    assert bool(jnp.all(info["converged"]))


def test_full_episode_under_scan(built):
    """A whole episode with no Python loop, and no NaN in the state afterwards."""
    env, spec, params = built
    action = env.truthful_action()
    _obs, state = env.reset(jax.random.PRNGKey(0), params)

    def body(carry, _):
        obs, st, r, c, d, _info = env.step_auto_reset(
            jax.random.PRNGKey(0), carry, action, params)
        return st, (r, c, d)

    final, (reward, costs, done) = jax.lax.scan(
        body, state, None, length=EPISODE_LEN)
    assert reward.shape == (EPISODE_LEN, spec["n_agents"])
    assert bool(done[-1]) and not bool(jnp.any(done[:-1]))
    for name in final.__dataclass_fields__:
        assert not bool(jnp.any(jnp.isnan(getattr(final, name)))), name


def _scans_with_trip_count(jaxpr, length):
    """Count `scan` equations whose trip count is exactly `length`.

    Counting `scan` equations outright reports the wrong number, and here there
    are **two** interior-point loops in reach: this market's clearing and the
    ramp-free boundary solve `reset` uses.  Both are built from the same operator
    and would have the same trip count if the boundary did not take its own, so
    the count is qualified by the constant rather than by the shape of the node.
    """
    n = 0
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "scan" and eqn.params.get("length") == length:
            n += 1
        for sub in jax.extend.core.jaxprs_in_params(eqn.params):
            n += _scans_with_trip_count(sub, length)
    return n


def test_step_solves_exactly_one_clearing(built):
    """The environment layer's one-clearing-per-step clause, counted by the
    solver's trip count.

    The trip count is `real_time.clearing.MAX_ITER`, **not** the day-ahead
    market's: the two markets calibrate separately and are 40 against 60, so a
    test that read the day-ahead constant would count zero here and pass for the
    wrong reason.
    """
    env, spec, params = built
    action = env.truthful_action()
    _obs, state = env.reset(jax.random.PRNGKey(0), params)
    closed = jax.make_jaxpr(env.step)(jax.random.PRNGKey(0), state, action, params)
    assert _scans_with_trip_count(closed.jaxpr, MAX_ITER) == 1
    assert spec["max_iter"] == MAX_ITER


def test_reset_constructs_the_carry_and_does_not_clear_inside_step(built):
    """`reset` runs the boundary solve; `step` must not run a second one."""
    env, spec, params = built
    closed = jax.make_jaxpr(env.reset)(jax.random.PRNGKey(0), params)
    assert _scans_with_trip_count(closed.jaxpr, MAX_ITER) == 1


def test_an_all_false_mask_bids_the_truthful_action(built):
    """The baseline is a mask, not a second interface.

    `learner_mask` was an `EnvParams` leaf that `step` never read, so the POMDP
    degeneration §9.1 of the market description names -- every agent but one
    fixed to truthful bidding -- was not expressible in this market, while the
    other four all implemented it.  Nothing raised, because the mask reaches
    `make_params` and is validated there.

    The second half is what gives the first content: without it the assertion
    would pass on a step that ignored its action outright.
    """
    env, spec, params = built
    key = jax.random.PRNGKey(3)
    _obs, state = env.reset(key, params)
    truthful = env.truthful_action()
    arbitrary = jnp.full_like(truthful, MARKUP_MAX)

    silenced = env.make_params(episode_len=EPISODE_LEN,
                               learner_mask=np.zeros(spec["n_agents"], bool))
    got = jax.jit(env.step)(key, state, arbitrary, silenced)
    want = jax.jit(env.step)(key, state, truthful, params)
    for j, name in ((0, "obs"), (2, "reward"), (3, "costs")):
        np.testing.assert_array_equal(np.asarray(got[j]), np.asarray(want[j]),
                                      err_msg=name)

    # the arbitrary action is not already the truthful one
    loud = jax.jit(env.step)(key, state, arbitrary, params)
    assert not np.array_equal(np.asarray(loud[2]), np.asarray(want[2]))


def test_step_auto_reset_stops_the_gradient_on_the_observation(built):
    """The barrier covers ``obs``, not only the state.

    `resources.env_base.Environment.step_auto_reset` stops both and the other
    four markets stop both; this one stopped the state alone, so a `lax.scan`
    rollout could carry a gradient across an episode seam through the
    observation handed to the next step.

    Forward mode rather than reverse: the clearing is a `lax.while_loop`, which
    has no transpose rule, and `jax.grad` through `step` therefore raises
    instead of measuring anything.  The same tangent through `step` is what
    says the zero below is the barrier and not an environment that happens to
    be flat in the action.
    """
    env, _spec, params = built
    key = jax.random.PRNGKey(5)
    _obs, state = env.reset(key, params)
    action = env.truthful_action()
    tangent = jnp.ones_like(action)

    seen = {}
    for name, fn in (("step", env.step), ("auto", env.step_auto_reset)):
        _, d_obs = jax.jvp(lambda a: fn(key, state, a, params)[0],
                           (action,), (tangent,))
        seen[name] = float(jnp.max(jnp.abs(d_obs)))
    assert seen["step"] > 0.0, (
        "the observation does not move with the action at this operating "
        f"point ({seen['step']}), so the barrier below cannot be seen")
    assert seen["auto"] == 0.0, seen
