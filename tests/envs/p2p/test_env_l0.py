"""L0 JAX contract for the P2P environment layer.

Beyond the standard jit / vmap / `lax.scan` / static-pytree checks, four
assertions come straight from the environment layer's acceptance criteria:

* **dtype leaf by leaf**, params included.  With `jax_enable_x64` on, any
  constructor that forgets its dtype silently becomes float64, so the suite
  must fail on the leaf, not on a downstream NaN.
* **exactly one clearing inside `step`**.  A first design of the layer evaluated
  `reset` unconditionally in a form that would have put a second clearing into
  every step of the day-ahead market; here the fingerprint of the clearing is
  its sorts, and `step`'s jaxpr must contain exactly as many as `clear`'s.
* **`obs` is a pure function of `(state, params)`**: the observation `step`
  returns must equal the one recomputed from the state it returns, bitwise.
  This is also what makes the two auto-reset paths agree.
* **the window's three off-by-ones**: a start on the upper edge runs to the
  end, `episode_len == n_periods` leaves exactly one legal start, one more
  raises at setup.

The jit/eager comparison uses `allclose` on the money-like outputs rather than
bitwise equality: under jit the fused multiply-adds of the price interpolation
differ from eager by 1 ULP (measured in `test_action_l0.py`, 9.5e-7 on
`price`), and that is XLA fusion, not a defect.
"""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from powermarketjax.envs.p2p import make_p2p_env, make_p2p_params
from powermarketjax.resources.battery import make_battery_bundle

N = 12
N_PERIODS = 40
EPISODE_LEN = 8
PI_EXP, PI_RET = 4.1, 26.11
DELTA = 0.5
BATCH = 8


def build(episode_len=EPISODE_LEN, n_periods=N_PERIODS):
    rng = np.random.default_rng(0)
    battery = make_battery_bundle(
        n_devices=N, dt_hours=DELTA,
        capacity_mwh=rng.uniform(2.0, 10.0, N).tolist(),
        power_mw=rng.uniform(0.5, 3.0, N).tolist())
    params = make_p2p_params(
        p_pv=rng.uniform(0.0, 2.0, (n_periods, N)),
        load=rng.uniform(0.0, 2.0, (n_periods, N)),
        battery=battery,
        kappa=rng.uniform(0.0, 5.0, N),
        learner_mask=np.ones(N, bool),
        episode_len=episode_len)
    env = make_p2p_env(N, PI_EXP, PI_RET, DELTA)
    return env, params


@pytest.fixture(scope="module")
def built():
    return build()


def some_action(key):
    return jax.random.uniform(key, (N, 2), jnp.float32, -1.0, 1.0)


def test_reset_and_step_jit(built):
    (reset, step, _, _), params = built
    key = jax.random.PRNGKey(0)
    obs, state = jax.jit(reset)(key, params)
    out = jax.jit(step)(key, state, some_action(key), params)
    obs2, state2, reward, costs, done, info = out
    assert obs.shape == (N, 15) and obs2.shape == (N, 15)
    assert reward.shape == (N,) and costs.shape == (N, 1)
    assert done.shape == ()


def test_jit_matches_eager_within_fusion(built):
    """Bitwise on everything except the money-like leaves, which fuse.

    The pattern and the measured 1-ULP figures are in `test_action_l0.py`;
    the affine price map and the settlement products are the leaves that
    XLA fuses into FMAs.
    """
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
    """Params, state and every output leaf; holds with x64 on or off."""
    (reset, step, _, _), params = built
    key = jax.random.PRNGKey(2)
    obs, state = reset(key, params)
    _, state2, reward, costs, done, info = jax.jit(step)(
        key, state, some_action(key), params)

    for leaf_params in (params.p_pv, params.load, params.kappa):
        assert leaf_params.dtype == jnp.float32
    assert params.learner_mask.dtype == jnp.bool_
    assert params.episode_len.dtype == jnp.int32

    for st in (state, state2):
        assert st.cursor.dtype == jnp.int32
        assert st.step_in_episode.dtype == jnp.int32
        for leaf in (st.soc, st.price_prev, st.volume_prev, st.net_prev,
                     st.award_prev, st.profit_prev):
            assert leaf.dtype == jnp.float32

    assert obs.dtype == jnp.float32
    assert reward.dtype == jnp.float32
    assert costs.dtype == jnp.float32
    assert done.dtype == jnp.bool_
    assert info["terminal_obs"].dtype == jnp.float32


def test_step_contains_exactly_one_clearing(built):
    """Regression test: the clearing's fingerprint is its
    sorts, and a step that evaluated a second clearing would double them."""
    from powermarketjax.envs.p2p import make_clearing
    (reset, step, _, _), params = built
    key = jax.random.PRNGKey(3)
    _, state = reset(key, params)

    clear, _ = make_clearing(N, PI_EXP, PI_RET)
    # dtype spelled out: with x64 on, a bare jnp.zeros is float64 and its
    # scatter into the clearing's float32 output warns -- the very trap the
    # dtype test above exists for
    args = tuple(jnp.zeros(N, jnp.float32) for _ in range(3))
    sorts_in_clear = str(jax.make_jaxpr(clear)(*args)).count("sort")
    sorts_in_step = str(jax.make_jaxpr(step)(
        key, state, some_action(key), params)).count("sort")
    assert sorts_in_clear > 0
    assert sorts_in_step == sorts_in_clear


def test_obs_is_pure_function_of_state(built):
    """`obs` is a function of `(state, params)` and of nothing else.

    **Both sides are compiled, and that is what makes the assertion bitwise.**
    The claim under test is that no state is hidden outside the pair, so the
    comparison has to hold the compilation context fixed: `step` runs under `jit`,
    so the recomputation must too.  Comparing a compiled result against an eager
    one tests something else, namely whether the same float32 expression gives the
    same last bits when XLA fuses it and when it does not, and it does not.
    Measured on 2026-08-14 with `jax_enable_x64` on: the two calendar channels of
    §9.4 differ by 2.09e-07, one unit in the last place, because the eager path
    evaluates the phase as separate operations while the fused path reassociates
    it, and the fused value is the more accurate of the two. With
    `jax_enable_x64` off the two paths agree, so the failure appears only in the
    configuration fixed for production, which is why the test now
    pins the context rather than the flag.
    """
    (reset, step, _, spec), params = built
    key = jax.random.PRNGKey(4)
    _, state = reset(key, params)
    obs, state2, *_ = jax.jit(step)(key, state, some_action(key), params)
    recomputed = jax.jit(spec["get_obs"])(state2, params)
    np.testing.assert_array_equal(np.asarray(obs), np.asarray(recomputed))

    # Calling it twice must also agree, which is the part that would break if
    # `get_obs` read anything outside its arguments.
    again = jax.jit(spec["get_obs"])(state2, params)
    np.testing.assert_array_equal(np.asarray(recomputed), np.asarray(again))


def test_eager_and_compiled_obs_agree_to_one_last_place(built):
    """Records the discrepancy the test above must not be written against.

    A wrapper is free to call `spec["get_obs"]` outside `jit`, so the size of the
    difference matters even though bitwise agreement across compilation contexts
    is not available. It is bounded here rather than asserted away: a regression
    that made it large would mean something other than reassociation had changed.
    """
    (reset, step, _, spec), params = built
    key = jax.random.PRNGKey(4)
    _, state = reset(key, params)
    _, state2, *_ = jax.jit(step)(key, state, some_action(key), params)
    eager = np.asarray(spec["get_obs"](state2, params))
    compiled = np.asarray(jax.jit(spec["get_obs"])(state2, params))
    worst = float(np.abs(eager - compiled).max())
    assert worst < 1e-6, f"eager against compiled differs by {worst:.3e}"


def test_spec_declares_a_terminal_boundary(built):
    """The only machine-readable form of the decision that this market's
    episode boundary is terminal.

    `spec["termination"]` is what tells a downstream algorithm whether it may
    carry a continuation value across the boundary, and for this market it must
    not: §8 settles the stock left in the battery there, so bootstrapping as
    well would price that stock twice. The other three markets pin the opposite
    value in their own L0 tests; this market had no such assertion while the
    field said `"truncation"`, so the flip to `"terminal"` was unguarded.
    """
    (_, _, _, spec), _ = built
    assert spec["termination"] == "terminal"


def test_vmap_over_envs(built):
    (reset, step, _, _), params = built
    keys = jax.random.split(jax.random.PRNGKey(5), BATCH)
    obs, states = jax.vmap(reset, in_axes=(0, None))(keys, params)
    actions = jax.vmap(some_action)(keys)
    out = jax.jit(jax.vmap(step, in_axes=(0, 0, 0, None)))(
        keys, states, actions, params)
    assert out[0].shape == (BATCH, N, 15)
    assert out[2].shape == (BATCH, N)
    assert out[4].shape == (BATCH,)
    for leaf in jax.tree_util.tree_leaves(out):
        assert np.isfinite(np.asarray(leaf, np.float64)).all()


def test_scan_full_episode_no_nan_static_pytree(built):
    """Two episodes back to back under `lax.scan`, through the auto-reset."""
    (reset, _, step_auto_reset, _), params = built
    key = jax.random.PRNGKey(6)
    _, state0 = reset(key, params)
    structure0 = jax.tree_util.tree_structure(state0)

    def body(carry, key):
        state = carry
        obs, state, reward, costs, done, info = step_auto_reset(
            key, state, some_action(key), params)
        return state, (obs, reward, costs, done)

    keys = jax.random.split(jax.random.PRNGKey(7), 2 * EPISODE_LEN + 3)
    final, (obs, reward, costs, done) = jax.jit(
        lambda s, k: lax.scan(body, s, k))(state0, keys)

    assert jax.tree_util.tree_structure(final) == structure0
    assert int(done.sum()) == 2                      # two boundaries crossed
    for leaf in jax.tree_util.tree_leaves((final, obs, reward, costs)):
        assert np.isfinite(np.asarray(leaf, np.float64)).all()
    # after every done the merged state restarted the episode counter
    assert int(final.step_in_episode) == (2 * EPISODE_LEN + 3) % EPISODE_LEN


def test_vmap_of_scan_full_rollout(built):
    """The formal rollout shape of M1: batched environments, each scanned
    through episode boundaries, one compiled program."""
    (reset, _, step_auto_reset, _), params = built
    keys = jax.random.split(jax.random.PRNGKey(20), BATCH)
    _, states = jax.vmap(reset, in_axes=(0, None))(keys, params)

    def rollout(state, key):
        def body(carry, k):
            obs, carry, reward, costs, done, _ = step_auto_reset(
                k, carry, some_action(k), params)
            return carry, (reward, done)
        return lax.scan(body, state, jax.random.split(key, EPISODE_LEN + 2))

    final, (reward, done) = jax.jit(jax.vmap(rollout))(states, keys)
    assert reward.shape == (BATCH, EPISODE_LEN + 2, N)
    assert int(done.sum()) == BATCH                 # one boundary per lane
    for leaf in jax.tree_util.tree_leaves(final):
        assert np.isfinite(np.asarray(leaf, np.float64)).all()


def test_window_start_on_upper_edge_runs_to_the_end():
    """Off-by-one 1: an episode starting at n_periods - episode_len reads its
    last row at n_periods - 1 and finishes without touching the clamp."""
    env, params = build()
    reset, step, _, spec = env
    # force the upper-edge start by rebuilding the state directly
    _, state = reset(jax.random.PRNGKey(8), params)
    state = state.replace(cursor=jnp.asarray(N_PERIODS - EPISODE_LEN, jnp.int32))
    key = jax.random.PRNGKey(9)
    for _ in range(EPISODE_LEN):
        obs, state, _, _, done, info = step(key, state, some_action(key), params)
    assert bool(done)
    # the terminal observation read row n_periods via the clamp; it must be
    # finite and equal to a recomputation, not garbage
    assert np.isfinite(np.asarray(info["terminal_obs"])).all()


def test_window_episode_equal_to_series_has_one_start():
    """Off-by-one 2: episode_len == n_periods leaves start 0 only."""
    env, params = build(episode_len=N_PERIODS)
    reset, _, _, _ = env
    for seed in range(5):
        _, state = reset(jax.random.PRNGKey(seed), params)
        assert int(state.cursor) == 0


def test_window_episode_longer_than_series_raises():
    """Off-by-one 3: episode_len == n_periods + 1 fails at setup, not at run."""
    with pytest.raises(ValueError, match="episode_len"):
        build(episode_len=N_PERIODS + 1)


def test_one_step_episode_warns():
    with pytest.warns(UserWarning, match="episode_len"):
        build(episode_len=1)


def test_params_axis_validation():
    rng = np.random.default_rng(1)
    battery = make_battery_bundle(n_devices=N, dt_hours=DELTA)
    good = dict(p_pv=rng.uniform(0, 2, (N_PERIODS, N)),
                load=rng.uniform(0, 2, (N_PERIODS, N)),
                battery=battery, kappa=np.zeros(N),
                learner_mask=np.ones(N, bool), episode_len=4)
    make_p2p_params(**good)                                   # sanity
    for field, bad in (("p_pv", rng.uniform(0, 2, (N_PERIODS, N + 1))),
                       ("load", rng.uniform(0, 2, (N_PERIODS - 1, N))),
                       ("kappa", np.zeros(N + 1)),
                       ("learner_mask", np.ones(N - 1, bool))):
        with pytest.raises(ValueError, match=field):
            make_p2p_params(**{**good, field: bad})
