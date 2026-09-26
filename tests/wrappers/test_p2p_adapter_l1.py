"""L1: market 04 reaches the shared learner without either side being bent.

`powermarketjax/wrappers/p2p.py` carries two of the three
divergences between this market and the learner the other four are driven by;
the third was a false conflict and was repaired in `ippo.py` itself
(`tests/learning/test_ippo_convergence_key_l0.py`).

**The measurement this file fixes in place.**  The out-of-package experiment
restricts which episode starts a run may draw, and its own `restrict_starts`
wraps ``reset`` alone, stating that a rollout longer than one episode would need
that revisited.  The shared learner is exactly such a rollout: `_rollout` scans
`horizon` steps and the auto-reset inside `envs/p2p/env.py`'s ``step`` calls the
``reset`` it closed over, not the wrapped one.  Measured 2026-08-27 on the real
Fluvius panel at ``episode_len = 96``, ``horizon = 192``, ``n_envs = 64``: half
the emitted cells belonged to a second episode, 25.0% of the boundary draws
(16 of 64) fell outside the training pool and 14.1% (9 of 64) squarely inside a
held-out block, against 26.38% and 13.53% for a uniform draw over the panel.
That is training on held-out days, and the test named
`test_a_rollout_that_crosses_a_boundary_leaves_the_pool` is the same
phenomenon on a fixture small enough to run here.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.envs.p2p import make_p2p_env, make_p2p_params
from powermarketjax.learning.ippo import IPPOConfig, make_ippo
from powermarketjax.learning.policy import bounds_for
from powermarketjax.resources.battery import make_battery_bundle
from powermarketjax.wrappers.p2p import (baseline_observation_statistics,
                                         make_pooled_ippo, restrict_starts)

N = 6
N_PERIODS = 64
EPISODE_LEN = 8
N_ENVS = 16
PI_EXP, PI_RET = 4.1, 26.11
DELTA = 0.5
#: Three starts out of the 57 the panel admits.  Small on purpose: a draw that
#: ignores the pool lands outside it with probability 54/57, so the check below
#: does not depend on being lucky.
POOL = np.array([0, 8, 16], np.int32)


def _market(learner_mask=None):
    rng = np.random.default_rng(0)
    battery = make_battery_bundle(
        n_devices=N, dt_hours=DELTA,
        capacity_mwh=rng.uniform(2.0, 10.0, N).tolist(),
        power_mw=rng.uniform(0.5, 3.0, N).tolist())
    mask = np.ones(N, bool) if learner_mask is None else learner_mask
    params = make_p2p_params(p_pv=rng.uniform(0.0, 2.0, (N_PERIODS, N)),
                             load=rng.uniform(0.0, 2.0, (N_PERIODS, N)),
                             battery=battery, kappa=rng.uniform(0.0, 5.0, N),
                             learner_mask=mask, episode_len=EPISODE_LEN)
    return make_p2p_env(N, PI_EXP, PI_RET, DELTA), params


def _cfg(horizon):
    return IPPOConfig(n_envs=N_ENVS, horizon=horizon, epochs=1, minibatches=1,
                      lr=0.0, clip_eps=0.2, gamma=0.99, gae_lambda=0.95,
                      vf_coef=0.5, ent_coef=0.0, max_grad_norm=0.5,
                      hidden=(16,), init_scale=1.0, weight_decay=0.0)


def _stats(spec):
    return (jnp.zeros((spec["obs_dim"],), jnp.float32),
            jnp.ones((spec["obs_dim"],), jnp.float32))


def test_restricted_reset_draws_from_the_pool_and_only_from_it():
    """Every start is in the pool, and more than one of them is used.

    The second half is what keeps this from passing on a degenerate wrapper: a
    device that always returned the same start would satisfy the first.
    """
    env, params = _market()
    reset = restrict_starts(env, POOL)[0]
    keys = jax.random.split(jax.random.PRNGKey(0), 128)
    _, state = jax.vmap(reset, in_axes=(0, None))(keys, params)
    drawn = np.asarray(state.cursor)
    assert set(drawn.tolist()) <= set(POOL.tolist()), (
        f"starts outside the pool: "
        f"{sorted(set(drawn.tolist()) - set(POOL.tolist()))}")
    assert len(set(drawn.tolist())) == len(POOL), (
        f"only {sorted(set(drawn.tolist()))} of {POOL.tolist()} ever drawn")


def test_an_empty_pool_is_refused():
    """An empty pool draws without complaining, so it has to be refused here.

    A pool of one is deliberately still allowed: scoring on one fixed day is a
    thing a caller may want, and refusing it would be this module deciding an
    experiment's business.
    """
    env, _params = _market()
    with pytest.raises(ValueError, match="non-empty"):
        restrict_starts(env, np.zeros((0,), np.int32))
    restrict_starts(env, np.array([8], np.int32))


def test_a_rollout_that_crosses_a_boundary_leaves_the_pool():
    """The defect, on this fixture: wrapping `reset` alone is not enough.

    `restrict_starts` holds the first episode inside the pool and the auto-reset
    that starts the second one does not see it, because it is the market's own
    closed-over reset.  This is the check that bites: it fails if the wrapper
    ever grows to cover that path, and then this test is what has to be
    re-pointed rather than deleted.
    """
    env, params = _market()
    spec = env[3]
    obs_mean, obs_std = _stats(spec)
    cfg = _cfg(2 * EPISODE_LEN)
    init, iterate = make_ippo(restrict_starts(env, POOL), bounds_for(spec), cfg,
                              obs_mean, obs_std, convergence_key=None)
    p, tx, opt, env_state, env_obs = init(jax.random.PRNGKey(0), params)
    metrics = iterate(p, tx, opt, env_state, env_obs, jax.random.PRNGKey(1),
                      params)[-1]
    cursor = np.asarray(metrics["step_cursor"])
    assert cursor.shape == (2 * EPISODE_LEN, N_ENVS)
    assert set(cursor[0].tolist()) <= set(POOL.tolist()), (
        "the first episode already left the pool, so this test is not measuring "
        "what it claims to")
    fresh = cursor[EPISODE_LEN]
    outside = [int(c) for c in fresh if int(c) not in set(POOL.tolist())]
    # measured on this fixture: 16 of 16
    assert outside, (
        "every boundary draw landed in the pool by chance; with 3 starts out of "
        "57 that has probability 1e-21, so the likelier cause is that the "
        "fixture no longer crosses a boundary")


def test_the_pooled_learner_never_leaves_the_pool():
    """Under `make_pooled_ippo` no emitted cell belongs to another episode.

    Two assertions, because either alone would pass on a device that broke the
    other: every start is in the pool, and every emitted cursor is its own
    episode's start plus the step index.
    """
    env, params = _market()
    spec = env[3]
    obs_mean, obs_std = _stats(spec)
    init, iterate = make_pooled_ippo(env, POOL, bounds_for(spec),
                                     _cfg(EPISODE_LEN), obs_mean, obs_std,
                                     episode_len=EPISODE_LEN)
    p, tx, opt, env_state, env_obs = init(jax.random.PRNGKey(0), params)
    metrics = iterate(p, tx, opt, env_state, env_obs, jax.random.PRNGKey(1),
                      params)[-1]
    cursor = np.asarray(metrics["step_cursor"])
    assert cursor.shape == (EPISODE_LEN, N_ENVS)
    assert set(cursor[0].tolist()) <= set(POOL.tolist()), (
        f"starts outside the pool: {sorted(set(cursor[0].tolist()))}")
    expected = cursor[0][None, :] + np.arange(EPISODE_LEN)[:, None]
    off = int(np.count_nonzero(cursor != expected))
    assert off == 0, (
        f"{off} of {cursor.size} emitted cells are not the continuation of "
        f"their own episode")


def test_a_horizon_that_is_not_one_episode_is_refused_at_construction():
    """Both directions, and at wiring time rather than on the first iteration."""
    env, params = _market()
    spec = env[3]
    obs_mean, obs_std = _stats(spec)
    for horizon in (EPISODE_LEN + 1, 2 * EPISODE_LEN, EPISODE_LEN - 1):
        with pytest.raises(ValueError, match="must equal episode_len"):
            make_pooled_ippo(env, POOL, bounds_for(spec), _cfg(horizon),
                             obs_mean, obs_std, episode_len=EPISODE_LEN)


def test_the_pooled_iterate_replaces_the_carry_it_is_handed():
    """A carry from outside the pool does not reach the rollout.

    The driver cannot forget the fresh reset, because there is nothing for it to
    forget: the same iteration run from a carry drawn over the whole panel and
    from the pooled one returns the same metrics, bit for bit.
    """
    env, params = _market()
    spec = env[3]
    obs_mean, obs_std = _stats(spec)
    init, iterate = make_pooled_ippo(env, POOL, bounds_for(spec),
                                     _cfg(EPISODE_LEN), obs_mean, obs_std,
                                     episode_len=EPISODE_LEN)
    p, tx, opt, env_state, env_obs = init(jax.random.PRNGKey(0), params)
    keys = jax.random.split(jax.random.PRNGKey(99), N_ENVS)
    other_obs, other_state = jax.vmap(env[0], in_axes=(0, None))(keys, params)
    assert not set(np.asarray(other_state.cursor).tolist()) <= set(POOL.tolist()), (
        "the deliberately wrong carry happens to be inside the pool, so this "
        "test would pass without the replacement it is checking")
    good = iterate(p, tx, opt, env_state, env_obs, jax.random.PRNGKey(1), params)[-1]
    bad = iterate(p, tx, opt, other_state, other_obs, jax.random.PRNGKey(1),
                  params)[-1]
    for name in sorted(good):
        a, b = np.asarray(good[name]), np.asarray(bad[name])
        assert np.array_equal(a, b), f"{name} depended on the carry"


def test_the_pooled_learner_reports_no_convergence_flag():
    """This market publishes none, so the metrics carry none."""
    env, params = _market()
    spec = env[3]
    obs_mean, obs_std = _stats(spec)
    init, iterate = make_pooled_ippo(env, POOL, bounds_for(spec),
                                     _cfg(EPISODE_LEN), obs_mean, obs_std,
                                     episode_len=EPISODE_LEN)
    p, tx, opt, env_state, env_obs = init(jax.random.PRNGKey(0), params)
    # `tx` is the optimiser itself, static as in every driver on disk
    step = jax.jit(iterate, static_argnums=(1,))
    metrics = step(p, tx, opt, env_state, env_obs, jax.random.PRNGKey(1),
                   params)[-1]
    assert "unconverged_frac" not in metrics and "step_converged" not in metrics
    assert "reward_mean" in metrics and "step_cursor" in metrics


def test_baseline_statistics_match_the_action_supplied_by_hand():
    """One quantity, two paths: the mask substitutes what the caller can spell out.

    The device under test lets the environment insert its own truthful action
    through ``learner_mask``; the control computes ``baseline_action`` from the
    same state and submits it with the mask on.  Both paths take their
    observations from the environment -- neither recomputes one -- so agreement
    says the substitution is the truthful action and not merely something
    reproducible.
    """
    env, params = _market()
    reset, _step, step_auto_reset, spec = env
    key, n_envs, horizon = jax.random.PRNGKey(7), 8, 3 * EPISODE_LEN
    mean, std = baseline_observation_statistics(env, params, key, n_envs, horizon)

    baseline = spec["baseline_action"]
    keys = jax.random.split(key, n_envs)
    obs, state = jax.vmap(reset, in_axes=(0, None))(keys, params)

    def one(carry, _):
        state, obs, k = carry
        k, k_env = jax.random.split(k)
        act = jax.vmap(lambda s: baseline(params.p_pv[s.cursor],
                                          params.load[s.cursor]))(state)
        nxt_obs, nxt_state, *_ = jax.vmap(
            step_auto_reset, in_axes=(0, 0, 0, None))(
                jax.random.split(k_env, n_envs), state, act, params)
        return (nxt_state, nxt_obs, k), obs

    _, seen = jax.lax.scan(one, (state, obs, keys[0]), None, length=horizon)
    ctl_mean = np.asarray(jnp.mean(seen, axis=(0, 1, 2)))
    ctl_std = np.asarray(jnp.std(seen, axis=(0, 1, 2)))
    ctl_std = np.where(ctl_std > 1e-8, ctl_std, 1.0)

    assert np.array_equal(np.asarray(mean), ctl_mean), (
        f"max|diff| {np.max(np.abs(np.asarray(mean) - ctl_mean))}")
    assert np.array_equal(np.asarray(std), ctl_std)
    # non-trivial: the statistics have to be of something that moved
    assert float(np.max(ctl_std)) > 1e-6, (
        "every observation coordinate is constant on this fixture, so the "
        "comparison above would hold for any action at all")


def test_baseline_statistics_refuse_parameters_already_masked_off():
    """Handed an all-False mask, the device cannot tell its own doing apart."""
    env, params = _market(learner_mask=np.zeros(N, bool))
    with pytest.raises(ValueError, match="already all False"):
        baseline_observation_statistics(env, params, jax.random.PRNGKey(0), 4, 4)
