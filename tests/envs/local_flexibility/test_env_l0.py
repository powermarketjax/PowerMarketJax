"""L0 JAX contract for the local flexibility environment layer.

Beyond the standard jit / vmap / `lax.scan` / static-pytree checks, four
assertions come straight from the environment layer's acceptance criteria:

* **dtype leaf by leaf**, params included.  With `jax_enable_x64` on -- which
  this market requires, since its clearing raises without it -- any constructor
  that forgets its dtype silently becomes float64, and the suite has to fail on
  the leaf rather than on a downstream number that still looks right.
* **exactly one clearing inside `step`**.  The clearing's fingerprint here is
  its interior point iteration, a `scan` of length `MAX_ITER`; the verification
  sweep is a `while_loop` and is counted separately, so the two cannot mask
  each other.  A first design of the layer would have put a second solve
  into every step.
* **`obs` is a pure function of `(state, params)`**, bitwise.  This observation
  contains no reduction -- it is a stack of state fields and two broadcast
  scalars -- so unlike the day-ahead one it admits no fusion tolerance.
* **the window's three off-by-ones**: a start on the upper edge runs to the
  end, `episode_len == n_periods` leaves exactly one legal start, one more
  raises at setup.

The jit/eager comparison keeps a tolerance on the money-like leaves and on the
state fields derived from them.  Two effects reach them: the FMA fusion of the
action map's price, measured at 6.8e-8 relative in `test_action_l0.py`, and the
interior point solve, which starts from an input that fusion has moved by that
much.  The development case is `case33bw` throughout, whose line ratings are
the 1e6 sentinel: nothing here is meant to exercise (LIM), which
`test_env_l1.py` runs on the primary case for exactly that reason.
"""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from powermarketjax.case import load_case
from powermarketjax.envs.local_flexibility import (build_voltage_sensitivity,
                                                   make_action_map,
                                                   draw_agent_buses,
                                                   make_local_flex_env,
                                                   make_local_flex_params)
from powermarketjax.envs.local_flexibility.clearing import MAX_ITER
from powermarketjax.envs.local_flexibility.env import baseline_action
from powermarketjax.resources.battery import make_battery_bundle

CASE = "33bw"
N = 6
N_PERIODS = 24
EPISODE_LEN = 5
KAPPA = 2.0
DELTA = 0.25
BATCH = 4


@pytest.fixture(scope="module", autouse=True)
def x64():
    """Module scope, and that is not cosmetic: `built` is a module-scoped
    fixture and pytest sets higher-scoped fixtures up first, so a
    function-scoped switch would still leave the environment being constructed
    with x64 off -- which is exactly the guard `make_clearing` raises on."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def build(episode_len=EPISODE_LEN, n_periods=N_PERIODS, **extra):
    case = load_case(CASE)
    sens = build_voltage_sensitivity(case)
    agent_bus = draw_agent_buses(case, sens, N, seed=0)
    env = make_local_flex_env(case, sens, agent_bus, period_hours=DELTA)

    rng = np.random.default_rng(0)
    battery = make_battery_bundle(
        n_devices=N, dt_hours=DELTA,
        capacity_mwh=rng.uniform(0.2, 1.0, N).tolist(),
        power_mw=rng.uniform(0.05, 0.3, N).tolist())
    total = float(np.asarray(case.node_pd, np.float64).sum())
    params = make_local_flex_params(
        load_series=total * rng.uniform(0.8, 1.2, n_periods),
        pv=rng.uniform(0.0, 0.05, (n_periods, N)),
        energy_price=rng.uniform(30.0, 90.0, n_periods),
        battery=battery, cycle_cost=rng.uniform(5.0, 20.0, N),
        load_scale=KAPPA, learner_mask=np.ones(N, bool),
        episode_len=episode_len, **extra)
    return env, params, case, sens, agent_bus


@pytest.fixture(scope="module")
def built():
    return build()[:2]


def some_action(key):
    return jax.random.normal(key, (N, 3), jnp.float32)


def test_reset_and_step_jit(built):
    (reset, step, _, spec), params = built
    key = jax.random.PRNGKey(0)
    obs, state = jax.jit(reset)(key, params)
    obs2, _, reward, costs, done, info = jax.jit(step)(
        key, state, some_action(key), params)
    assert obs.shape == (N, 23) and obs2.shape == (N, 23)
    assert spec["obs_dim"] == 23 and spec["costs_dim"] == 3
    assert reward.shape == (N,) and costs.shape == (N, 3) and done.shape == ()
    assert info["terminal_obs"].shape == (N, 23)
    assert float(info["mu"]) < 1e-6, "the solve inside step did not converge"


def test_spec_baseline_action_is_the_array_not_the_factory(built):
    """`spec["baseline_action"]` must be the array, not the function that builds it.

    `learning/adapters.py` documents two shapes for this key and only two: an
    array (ancillary) or a function of the **state** (P2P, whose truthful price
    depends on the current net position).  This market's baseline is neither --
    `baseline_action`'s own docstring says none of its three components depends
    on the state -- so it is the array, and a function of `n_agent` is a third
    shape no caller reads.  It matters because `adapters.py` passes whatever it
    finds through unchanged and `learning/ippo.py` feeds this key straight into
    `vmap(step_auto_reset)` as an action, so a callable here fails far from the
    cause rather than at the point that made it wrong.
    """
    (_, _, _, spec), _ = built
    got = spec["baseline_action"]
    assert not callable(got), (
        "spec['baseline_action'] is a callable; adapters.py does not normalise "
        "this key, so the caller receives a function where an action array is "
        "expected")
    np.testing.assert_array_equal(np.asarray(got), np.asarray(baseline_action(N)))


def test_jit_matches_eager_within_fusion(built):
    (reset, step, _, _), params = built
    key = jax.random.PRNGKey(1)
    _, state = reset(key, params)
    action = some_action(key)
    eager = step(key, state, action, params)
    jitted = jax.jit(step)(key, state, action, params)
    for name, a, b in (("obs", eager[0], jitted[0]),
                       ("reward", eager[2], jitted[2]),
                       ("costs", eager[3], jitted[3])):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b),
                                   rtol=1e-5, atol=1e-6, err_msg=name)
    assert bool(eager[4]) == bool(jitted[4])


def test_dtypes_leaf_by_leaf(built):
    (reset, step, _, _), params = built
    key = jax.random.PRNGKey(2)
    obs, state = reset(key, params)
    obs2, state2, reward, costs, done, info = jax.jit(step)(
        key, state, some_action(key), params)

    for leaf in (params.load_series, params.pv, params.energy_price,
                 params.cycle_cost, params.load_scale):
        assert leaf.dtype == jnp.float32
    assert params.learner_mask.dtype == jnp.bool_
    assert params.episode_len.dtype == jnp.int32
    assert params.cursor_pool.dtype == jnp.int32
    assert params.monitor_baseline.dtype == jnp.bool_

    for st in (state, state2):
        for name in ("cursor", "step_in_episode", "req_v_count", "req_th_count"):
            assert getattr(st, name).dtype == jnp.int32, name
        for name in ("soc", "req_v_own", "req_th_own", "req_v_max",
                     "req_th_max", "award_prev", "payment_prev", "profit_prev",
                     "pv_prev", "load_prev", "volume_prev", "price_avg_prev"):
            assert getattr(st, name).dtype == jnp.float32, name

    assert obs.dtype == jnp.float32 and obs2.dtype == jnp.float32
    assert reward.dtype == jnp.float32 and costs.dtype == jnp.float32
    assert done.dtype == jnp.bool_
    assert info["terminal_obs"].dtype == jnp.float32


def test_step_contains_exactly_one_clearing_and_one_sweep(built):
    """Regression test for a second solve inside `step`, on this market's two
    loops."""
    from powermarketjax.envs.local_flexibility.clearing import make_clearing

    (reset, step, _, spec), params = built
    key = jax.random.PRNGKey(3)
    _, state = reset(key, params)
    case = load_case(CASE)
    sens = build_voltage_sensitivity(case)

    clear, _ = make_clearing(case, sens, spec["agent_bus"])
    n_bus = int(case.n_nodes)
    args = (jnp.zeros(N), jnp.zeros(N), jnp.zeros(n_bus), jnp.zeros(n_bus),
            jnp.zeros(n_bus))
    in_clear = str(jax.make_jaxpr(clear)(*args))
    in_step = str(jax.make_jaxpr(step)(key, state, some_action(key), params))

    fingerprint = f"length={MAX_ITER}"
    assert in_clear.count(fingerprint) == 1
    assert in_step.count(fingerprint) == 1
    # and the nonlinear sweep is the one while_loop, not a second solve
    assert in_step.count("while") == 1


def test_obs_is_pure_function_of_state(built):
    (reset, step, _, spec), params = built
    key = jax.random.PRNGKey(4)
    _, state = reset(key, params)
    obs, state2, *_ = jax.jit(step)(key, state, some_action(key), params)
    recomputed = spec["get_obs"](state2, params)
    np.testing.assert_array_equal(np.asarray(obs), np.asarray(recomputed))


def test_vmap_over_envs(built):
    (reset, step, _, _), params = built
    keys = jax.random.split(jax.random.PRNGKey(5), BATCH)
    _, states = jax.vmap(reset, in_axes=(0, None))(keys, params)
    actions = jax.vmap(some_action)(keys)
    obs, states2, reward, costs, done, info = jax.jit(
        jax.vmap(step, in_axes=(0, 0, 0, None)))(keys, states, actions, params)
    assert obs.shape == (BATCH, N, 23)
    assert reward.shape == (BATCH, N) and costs.shape == (BATCH, N, 3)
    assert done.shape == (BATCH,)
    assert np.isfinite(np.asarray(reward)).all()
    assert (np.asarray(info["mu"]) < 1e-6).all()


def test_scan_full_episode_no_nan_static_pytree(built):
    """Two episodes back to back under `lax.scan`, through the auto-reset."""
    (reset, _, step_auto_reset, _), params = built
    key = jax.random.PRNGKey(6)
    _, state0 = reset(key, params)
    structure0 = jax.tree_util.tree_structure(state0)

    def body(carry, key):
        obs, carry, reward, costs, done, _ = step_auto_reset(
            key, carry, some_action(key), params)
        return carry, (obs, reward, costs, done)

    n_steps = 2 * EPISODE_LEN + 2
    keys = jax.random.split(jax.random.PRNGKey(7), n_steps)
    final, (obs, reward, costs, done) = jax.jit(
        lambda s, k: lax.scan(body, s, k))(state0, keys)

    assert jax.tree_util.tree_structure(final) == structure0
    assert int(done.sum()) == 2
    for leaf in jax.tree_util.tree_leaves((final, obs, reward, costs)):
        assert np.isfinite(np.asarray(leaf, np.float64)).all()
    assert int(final.step_in_episode) == n_steps % EPISODE_LEN


def test_vmap_of_scan_full_rollout(built):
    """The rollout shape M1 asks for: batched environments, each scanned
    through an episode boundary, one compiled program."""
    (reset, _, step_auto_reset, _), params = built
    keys = jax.random.split(jax.random.PRNGKey(8), BATCH)
    _, states = jax.vmap(reset, in_axes=(0, None))(keys, params)

    def rollout(state, key):
        def body(carry, k):
            _, carry, reward, _, done, _ = step_auto_reset(
                k, carry, some_action(k), params)
            return carry, (reward, done)
        return lax.scan(body, state, jax.random.split(key, EPISODE_LEN + 1))

    final, (reward, done) = jax.jit(jax.vmap(rollout))(states, keys)
    assert reward.shape == (BATCH, EPISODE_LEN + 1, N)
    assert int(done.sum()) == BATCH
    for leaf in jax.tree_util.tree_leaves(final):
        assert np.isfinite(np.asarray(leaf, np.float64)).all()


def test_window_start_on_upper_edge_runs_to_the_end():
    """Off-by-one 1: a start at n_periods - episode_len reads its last row at
    n_periods - 1, and its terminal observation needs no row at all."""
    (reset, step, _, _), params, *_ = build(episode_len=4, n_periods=10)
    state = reset(jax.random.PRNGKey(0), params)[1]
    state = state.replace(cursor=jnp.int32(10 - 4))
    key = jax.random.PRNGKey(1)
    for _ in range(4):
        _, state, _, _, done, _ = jax.jit(step)(key, state, some_action(key),
                                                params)
    assert bool(done)


def test_window_episode_equal_to_series_has_one_start():
    """Off-by-one 2: exactly one legal start, and it is zero."""
    (reset, _, _, _), params, *_ = build(episode_len=10, n_periods=10)
    starts = {int(reset(jax.random.PRNGKey(seed), params)[1].cursor)
              for seed in range(20)}
    assert starts == {0}


def test_window_episode_longer_than_series_raises():
    """Off-by-one 3: setup time, not run time."""
    with pytest.raises(ValueError, match="episode_len"):
        build(episode_len=11, n_periods=10)


def test_cursor_pool_restricts_the_starts_and_the_default_does_not():
    """The pool is the set `reset` draws from; an omitted pool is every legal
    start.

    Both halves are asserted here because **either alone is satisfied by a
    `reset` broken in the other direction**: one that ignored the pool
    entirely would still pass the default half, and one that returned a
    constant would still pass the restricted half.  The default half is also
    the compatibility statement -- the pool was added under a default that has
    to reproduce the uniform draw that preceded it, and this is where that is
    measured rather than asserted in a comment.
    """
    n_periods, episode_len = 20, 4
    n_starts = n_periods - episode_len + 1
    keys = [jax.random.PRNGKey(seed) for seed in range(400)]

    (reset, *_), default, *_ = build(episode_len=episode_len,
                                     n_periods=n_periods)
    assert {int(reset(k, default)[1].cursor) for k in keys} == set(
        range(n_starts))

    (reset, *_), pooled, *_ = build(episode_len=episode_len,
                                    n_periods=n_periods,
                                    cursor_pool=np.array([3, 7]))
    assert {int(reset(k, pooled)[1].cursor) for k in keys} == {3, 7}


def test_cursor_pool_is_drawn_uniformly_over_the_pool_and_not_over_its_range():
    """A pool of two starts, drawn 400 times: neither member dominates.

    The set assertion above is satisfied by 399 draws of one member and one of
    the other, so it says the pool is *reachable* and not that it is drawn
    from.  What that would hide is a draw taken over something other than the
    pool's own index -- the pool's values, or its first element with the rest
    reached only by an edge case -- which restricts the episodes correctly and
    reweights them silently.  The interval is wide because this is a check on
    the estimator, not on the seed: at 400 draws the standard error of the
    share is 0.025, so 0.35 to 0.65 is six standard errors from either edge.
    """
    (reset, *_), pooled, *_ = build(episode_len=4, n_periods=20,
                                    cursor_pool=np.array([3, 7]))
    drawn = [int(reset(jax.random.PRNGKey(seed), pooled)[1].cursor)
             for seed in range(400)]
    share = drawn.count(3) / len(drawn)
    assert 0.35 < share < 0.65, share


def test_cursor_pool_is_rejected_when_it_cannot_be_honoured():
    """Setup time, like the window's third off-by-one.

    Each of the three is silent at run time and wrong in a different way: an
    empty pool is a filter that matched nothing, a start past the last legal
    one runs the episode off the end of the series, and a duplicate reweights
    the episode distribution instead of restricting it.
    """
    for pool, match in ((np.array([], np.int32), "empty"),
                        (np.array([17]), "outside"),
                        (np.array([3, 3, 7]), "duplicates")):
        with pytest.raises(ValueError, match=match):
            build(episode_len=4, n_periods=20, cursor_pool=pool)


def test_cursor_pool_survives_jit_and_vmap_as_a_params_leaf():
    """It is data, not a bound, and this is what that buys: `reset` stays a
    gather, so it compiles and maps like any other leaf and two pools can be
    mapped over in one call."""
    (reset, *_), pooled, *_ = build(episode_len=4, n_periods=20,
                                    cursor_pool=np.array([3, 7]))
    keys = jax.random.split(jax.random.PRNGKey(0), 64)
    cursors = jax.jit(jax.vmap(reset, in_axes=(0, None)))(keys, pooled)[1].cursor
    assert set(np.asarray(cursors).tolist()) == {3, 7}


def test_one_step_episode_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build(episode_len=1, n_periods=10)
    assert any("episode_len < 2" in str(w.message) for w in caught)


def test_params_axis_and_scenario_validation():
    """Every axis is checked against the battery's device axis, and
    `load_scale` has no default and no non-positive value."""
    battery = make_battery_bundle(n_devices=N, dt_hours=DELTA,
                                  capacity_mwh=1.0, power_mw=0.2)
    good = dict(load_series=np.ones(8), pv=np.zeros((8, N)),
                energy_price=np.full(8, 50.0), battery=battery,
                cycle_cost=np.full(N, 10.0), load_scale=1.5,
                learner_mask=np.ones(N, bool), episode_len=4)
    make_local_flex_params(**good)

    for field, value, match in (
            ("pv", np.zeros((8, N + 1)), "pv must be"),
            ("energy_price", np.full(7, 50.0), "energy_price must be"),
            ("cycle_cost", np.full(N + 1, 10.0), "cycle_cost must be"),
            ("learner_mask", np.ones(N + 1, bool), "learner_mask must be"),
            ("load_scale", 0.0, "load_scale"),
            ("load_scale", np.nan, "load_scale")):
        with pytest.raises(ValueError, match=match):
            make_local_flex_params(**{**good, field: value})


def test_net_exporting_case_is_rejected_at_construction():
    """`case533mt_lo` registers negative net load in aggregate, so §14's
    allocation has no meaning on it."""
    case = load_case("533mt_lo")
    sens = build_voltage_sensitivity(case)
    with pytest.raises(ValueError, match="registered load"):
        make_local_flex_env(case, sens, np.array([5, 9]))


def test_the_env_spec_republishes_the_action_maps_box_and_not_a_second_one(built):
    """One action space, declared once, in the place that owns it.

    Until 2026-09-10 `make_local_flex_env` discarded `make_action_map`'s `spec`
    at the call that builds the map and wrote `action_low` / `action_high`
    again itself, so the map's declaration reached nothing and the two could
    have parted company without any test noticing -- the same dead-spec defect
    the ancillary market found on its reserve map.  This asserts the env
    republishes the map's two ends rather than computing its own, which is what
    makes `envs/local_flexibility/action.py` the single place the box is
    chosen.
    """
    (_, _, _, spec), _ = built
    aspec = make_action_map(N, DELTA)[1]
    assert (spec["action_low"], spec["action_high"]) \
        == (aspec["action_low"], aspec["action_high"])
    assert np.isfinite(spec["action_low"]) and np.isfinite(spec["action_high"])
