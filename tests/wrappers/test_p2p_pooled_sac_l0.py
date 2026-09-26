"""L0: `make_pooled_sac` is the same pool discipline as `make_pooled_ippo`, on SAC.

Added 2026-09-20.  Market 04's SAC columns had only ever run on
`tools/p2p_experiment/constrained_baseline.py`, which stores no policy, so their
learned prices could not be read back; `powermarketjax/wrappers/p2p.py` gained
`make_pooled_sac` and `tools/benchmark/run_rl_04.py` gained `--algo sac`.  These
tests fix the wrapper in place with the assertions the IPPO wrapper already
carries (`test_p2p_adapter_l1.py`): the learner jits, its parameter pytree does
not change structure across iterations, no emitted cell leaves the pool, the
carry it is handed is replaced, a horizon that is not one episode is refused at
construction, and no convergence flag is reported.  The per-agent layout is
checked by the leading axis of the actor's leaves, read off the tree `init`
returned rather than off the flag.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.learning.policy import bounds_for
from powermarketjax.learning.sac import SACConfig
from powermarketjax.wrappers.p2p import make_pooled_sac

from tests.wrappers.test_p2p_adapter_l1 import (EPISODE_LEN, N, N_ENVS, POOL,
                                                _market, _stats)


def _cfg(horizon, n_envs=N_ENVS):
    return SACConfig(n_envs=n_envs, horizon=horizon, buffer_size=4096,
                     batch_size=32, utd_ratio=0.01, gamma=0.99, tau=0.005,
                     policy_lr=3e-4, q_lr=1e-3, alpha_lr=1e-3, init_alpha=0.2,
                     hidden=(16,), log_std_min=-5.0, log_std_max=2.0,
                     reward_scale=1.0)


def _build(per_agent=False):
    env, params = _market()
    spec = env[3]
    obs_mean, obs_std = _stats(spec)
    init, iterate = make_pooled_sac(env, POOL, bounds_for(spec), _cfg(EPISODE_LEN),
                                    obs_mean, obs_std, episode_len=EPISODE_LEN,
                                    per_agent_params=per_agent)
    return env, params, init, iterate


def test_pooled_sac_jits_and_keeps_its_pytree_structure():
    """L0: one jitted iteration runs, and a second returns the same treedef."""
    _env, params, init, iterate = _build()
    p, tx, learner, env_state, env_obs = init(jax.random.PRNGKey(0), params)
    step = jax.jit(iterate, static_argnums=(1,))
    p1, l1, s1, o1, k1, m1 = step(p, tx, learner, env_state, env_obs,
                                  jax.random.PRNGKey(1), params)
    p2, l2, _s2, _o2, _k2, m2 = step(p1, tx, l1, s1, o1, k1, params)
    assert jax.tree_util.tree_structure(p) == jax.tree_util.tree_structure(p1) \
        == jax.tree_util.tree_structure(p2)
    assert jax.tree_util.tree_structure(learner) == jax.tree_util.tree_structure(l1) \
        == jax.tree_util.tree_structure(l2)
    for a, b in zip(jax.tree_util.tree_leaves(p), jax.tree_util.tree_leaves(p2)):
        assert a.shape == b.shape and a.dtype == b.dtype
    assert sorted(m1) == sorted(m2)
    assert bool(jnp.isfinite(m2["reward_mean"]))


def test_pooled_sac_never_leaves_the_pool():
    """Every start is in the pool and every emitted cursor continues its own episode."""
    _env, params, init, iterate = _build()
    p, tx, learner, env_state, env_obs = init(jax.random.PRNGKey(0), params)
    metrics = iterate(p, tx, learner, env_state, env_obs, jax.random.PRNGKey(1),
                      params)[-1]
    cursor = np.asarray(metrics["step_cursor"])
    assert cursor.shape == (EPISODE_LEN, N_ENVS)
    assert set(cursor[0].tolist()) <= set(POOL.tolist()), (
        f"starts outside the pool: {sorted(set(cursor[0].tolist()))}")
    expected = cursor[0][None, :] + np.arange(EPISODE_LEN)[:, None]
    assert int(np.count_nonzero(cursor != expected)) == 0


def test_pooled_sac_replaces_the_carry_it_is_handed():
    """A carry drawn over the whole panel never reaches the rollout: same metrics bit for bit."""
    env, params, init, iterate = _build()
    p, tx, learner, env_state, env_obs = init(jax.random.PRNGKey(0), params)
    keys = jax.random.split(jax.random.PRNGKey(99), N_ENVS)
    other_obs, other_state = jax.vmap(env[0], in_axes=(0, None))(keys, params)
    assert not set(np.asarray(other_state.cursor).tolist()) <= set(POOL.tolist())
    good = iterate(p, tx, learner, env_state, env_obs, jax.random.PRNGKey(1), params)[-1]
    bad = iterate(p, tx, learner, other_state, other_obs, jax.random.PRNGKey(1),
                  params)[-1]
    for name in sorted(good):
        assert np.array_equal(np.asarray(good[name]), np.asarray(bad[name])), (
            f"{name} depended on the carry")


def test_pooled_sac_refuses_a_horizon_that_is_not_one_episode():
    env, _params = _market()
    spec = env[3]
    obs_mean, obs_std = _stats(spec)
    for horizon in (EPISODE_LEN + 1, 2 * EPISODE_LEN, EPISODE_LEN - 1):
        with pytest.raises(ValueError, match="must equal episode_len"):
            make_pooled_sac(env, POOL, bounds_for(spec), _cfg(horizon),
                            obs_mean, obs_std, episode_len=EPISODE_LEN)


def test_pooled_sac_reports_no_convergence_flag():
    _env, params, init, iterate = _build()
    p, tx, learner, env_state, env_obs = init(jax.random.PRNGKey(0), params)
    metrics = jax.jit(iterate, static_argnums=(1,))(
        p, tx, learner, env_state, env_obs, jax.random.PRNGKey(1), params)[-1]
    assert "unconverged_frac" not in metrics and "step_converged" not in metrics
    assert "reward_mean" in metrics and "step_cursor" in metrics


def test_pooled_sac_per_agent_layout_is_read_off_the_returned_tree():
    """Under `per_agent_params` every actor leaf leads with `n_agents`; shared leads with none."""
    _env, params, init, _iterate = _build(per_agent=True)
    p, *_ = init(jax.random.PRNGKey(0), params)
    leading = {int(jnp.shape(leaf)[0]) for leaf in jax.tree_util.tree_leaves(p["actor"])
               if jnp.ndim(leaf) > 0}
    assert leading == {N}, leading
    _env, params, init_shared, _it = _build(per_agent=False)
    q, *_ = init_shared(jax.random.PRNGKey(0), params)
    leading_shared = {int(jnp.shape(leaf)[0])
                      for leaf in jax.tree_util.tree_leaves(q["actor"]) if jnp.ndim(leaf) > 0}
    assert N not in leading_shared, leading_shared
