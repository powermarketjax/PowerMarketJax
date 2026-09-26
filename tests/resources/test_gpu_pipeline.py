# Vendored test from PowerZooJax.
# Source        : tests/resource/test_gpu_pipeline.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
"""GPU Pipeline tests — auto_reset, scan rollout, vmap + scan combo.

Tests the three missing capabilities for full GPU training loops:
1. step_auto_reset: done → auto reset for lax.scan compatibility
2. scan_rollout: lax.scan over step_auto_reset, no Python loop
3. vmap + scan combo: batch of parallel envs each doing a scan rollout

These tests apply to ALL four resource envs: Battery, Renewable, Vehicle, FlexLoad.
"""

import pytest
import jax
import jax.numpy as jnp

from powermarketjax.resources.battery import BatteryEnv, make_battery_params
from powermarketjax.resources.renewable import RenewableEnv, RenewableParams
from powermarketjax.resources.vehicle import VehicleEnv, make_vehicle_params
from powermarketjax.resources.flexload import FlexLoadEnv, FlexLoadParams
from powermarketjax.utils.jax_utils import (
    split_key_for_envs, batch_reset, batch_step, scan_rollout,
)


# ========================== Fixtures ==========================

@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


def _short_battery():
    env = BatteryEnv()
    params = make_battery_params(max_steps=8, delta_t_hours=0.5, steps_per_day=48)
    return env, params


def _short_renewable():
    env = RenewableEnv()
    profiles = jnp.full((48,), 0.5, dtype=jnp.float32)
    params = RenewableParams(capacity_mw=10.0, profiles=profiles, max_steps=8)
    return env, params


def _short_vehicle():
    env = VehicleEnv()
    params = make_vehicle_params(max_steps=8, delta_t_minutes=30.0)
    return env, params


def _short_flexload():
    env = FlexLoadEnv()
    params = FlexLoadParams(max_steps=8)
    return env, params


ALL_ENVS = [
    pytest.param(_short_battery, id="battery"),
    pytest.param(_short_renewable, id="renewable"),
    pytest.param(_short_vehicle, id="vehicle"),
    pytest.param(_short_flexload, id="flexload"),
]


def _zero_action(env, params):
    """Return a single zero-action with the correct shape for this env."""
    space = env.action_space(params)
    return jnp.zeros(space.shape, dtype=jnp.float32)


def _zero_actions(env, params, T):
    """Return T zero-actions with shape (T, *action_shape)."""
    space = env.action_space(params)
    return jnp.zeros((T, *space.shape), dtype=jnp.float32)


def _batch_zero_actions(env, params, n_envs, T):
    """Return batched zero-actions with shape (n_envs, T, *action_shape)."""
    space = env.action_space(params)
    return jnp.zeros((n_envs, T, *space.shape), dtype=jnp.float32)


# ========================== step_auto_reset ==========================

class TestStepAutoReset:
    """step_auto_reset returns reset state/obs when done=True."""

    @pytest.mark.parametrize("make_env", ALL_ENVS)
    def test_auto_reset_fires_on_done(self, make_env, key):
        """After max_steps, state should be auto-reset (time_step=0, done=False)."""
        env, params = make_env()
        obs, state = env.reset(key, params)

        # Step until done
        for i in range(params.max_steps):
            k = jax.random.fold_in(key, i)
            action = _zero_action(env, params)
            obs, state, reward, costs, done, info = env.step_auto_reset(
                k, state, action, params
            )

        # After max_steps, done should be True but state should be auto-reset
        assert bool(done), "Expected done=True at max_steps"
        assert int(state.time_step) == 0, "Auto-reset should set time_step=0"
        assert not bool(state.done), "Auto-reset should set done=False"

    @pytest.mark.parametrize("make_env", ALL_ENVS)
    def test_auto_reset_preserves_reward(self, make_env, key):
        """Reward from the terminal step should be preserved (not overwritten by reset)."""
        env, params = make_env()
        obs, state = env.reset(key, params)

        # Step to just before done
        for i in range(params.max_steps - 1):
            k = jax.random.fold_in(key, i)
            obs, state, _, _, _, _ = env.step_auto_reset(
                k, state, _zero_action(env, params), params
            )

        # Terminal step
        k_final = jax.random.fold_in(key, params.max_steps - 1)
        obs, state, reward, costs, done, info = env.step_auto_reset(
            k_final, state, _zero_action(env, params), params
        )
        # reward should be a valid scalar
        assert reward.shape == ()
        assert costs.shape == (len(env.constraint_names(params)),)

    @pytest.mark.parametrize("make_env", ALL_ENVS)
    def test_auto_reset_jit(self, make_env, key):
        """step_auto_reset compiles under jax.jit."""
        env, params = make_env()
        obs, state = env.reset(key, params)

        step_fn = jax.jit(lambda k, s, a: env.step_auto_reset(k, s, a, params))
        obs2, state2, reward, costs, done, info = step_fn(
            key, state, _zero_action(env, params)
        )
        assert obs2.shape == obs.shape

    @pytest.mark.parametrize("make_env", ALL_ENVS)
    def test_auto_reset_vmap(self, make_env, key):
        """step_auto_reset works under vmap."""
        env, params = make_env()
        n = 4
        keys = jax.random.split(key, n)
        obs_b, states_b = jax.vmap(env.reset, in_axes=(0, None))(keys, params)

        actions = jax.vmap(lambda _: _zero_action(env, params))(jnp.arange(n))
        step_keys = jax.random.split(jax.random.fold_in(key, 99), n)

        vmapped = jax.vmap(env.step_auto_reset, in_axes=(0, 0, 0, None))
        obs2, states2, rewards, costs, dones, infos = vmapped(
            step_keys, states_b, actions, params
        )
        assert obs2.shape[0] == n
        assert rewards.shape == (n,)
        assert costs.shape == (n, len(env.constraint_names(params)))

    @pytest.mark.parametrize("make_env", ALL_ENVS)
    def test_auto_reset_pytree_structure_stable(self, make_env, key):
        """Pytree structure is the same before/after auto-reset fires."""
        env, params = make_env()
        _, state0 = env.reset(key, params)
        tree_init = jax.tree_util.tree_structure(state0)

        # Run past done
        state = state0
        for i in range(params.max_steps + 2):
            k = jax.random.fold_in(key, i)
            _, state, _, _, _, _ = env.step_auto_reset(
                k, state, _zero_action(env, params), params
            )

        tree_after = jax.tree_util.tree_structure(state)
        assert tree_init == tree_after


# ========================== scan_rollout ==========================

class TestScanRollout:
    """lax.scan-based trajectory collection."""

    @pytest.mark.parametrize("make_env", ALL_ENVS)
    def test_scan_rollout_compiles(self, make_env, key):
        """scan_rollout should JIT-compile and run without error."""
        env, params = make_env()
        k1, k2 = jax.random.split(key)
        _, init_state = env.reset(k1, params)

        T = params.max_steps
        actions = _zero_actions(env, params, T)

        rollout_fn = jax.jit(lambda k, s: scan_rollout(env, k, s, params, actions))
        final_state, obs_traj, reward_traj, cost_traj, done_traj, info_traj = rollout_fn(
            k2, init_state
        )

        assert obs_traj.shape[0] == T
        assert reward_traj.shape == (T,)
        assert cost_traj.shape == (T, len(env.constraint_names(params)))
        assert done_traj.shape == (T,)

    @pytest.mark.parametrize("make_env", ALL_ENVS)
    def test_scan_rollout_done_fires(self, make_env, key):
        """At least one done=True should appear in a rollout of length max_steps."""
        env, params = make_env()
        k1, k2 = jax.random.split(key)
        _, init_state = env.reset(k1, params)

        T = params.max_steps
        actions = _zero_actions(env, params, T)
        _, _, _, cost_traj, done_traj, _ = scan_rollout(env, k2, init_state, params, actions)

        assert bool(jnp.any(done_traj)), "Expected at least one done in rollout"

    @pytest.mark.parametrize("make_env", ALL_ENVS)
    def test_scan_rollout_multi_episode(self, make_env, key):
        """Rollout longer than one episode should auto-reset and continue."""
        env, params = make_env()
        k1, k2 = jax.random.split(key)
        _, init_state = env.reset(k1, params)

        T = params.max_steps * 3  # 3 episodes worth
        actions = _zero_actions(env, params, T)
        final_state, obs_traj, reward_traj, cost_traj, done_traj, _ = scan_rollout(
            env, k2, init_state, params, actions
        )

        # Should see multiple done=True
        n_done = int(jnp.sum(done_traj))
        assert n_done >= 2, f"Expected >=2 dones in 3×max_steps rollout, got {n_done}"


# ========================== vmap + scan combo ==========================

class TestVmapScanCombo:
    """Batch of parallel envs each doing a scan rollout — full GPU pipeline."""

    @pytest.mark.parametrize("make_env", ALL_ENVS)
    def test_vmap_scan_compiles(self, make_env, key):
        """vmap over scan_rollout should JIT-compile."""
        env, params = make_env()
        n_envs = 4
        T = params.max_steps

        # Batch reset
        reset_keys = jax.random.split(key, n_envs)
        obs_b, states_b = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, params)

        actions = _batch_zero_actions(env, params, n_envs, T)
        rollout_keys = jax.random.split(jax.random.fold_in(key, 1), n_envs)

        def single_rollout(k, state, acts):
            return scan_rollout(env, k, state, params, acts)

        vmapped_rollout = jax.jit(jax.vmap(single_rollout))
        final_states, obs_traj, reward_traj, cost_traj, done_traj, info_traj = vmapped_rollout(
            rollout_keys, states_b, actions
        )

        assert obs_traj.shape[0] == n_envs
        assert obs_traj.shape[1] == T
        assert reward_traj.shape == (n_envs, T)
        assert cost_traj.shape == (n_envs, T, len(env.constraint_names(params)))
        assert done_traj.shape == (n_envs, T)

    @pytest.mark.parametrize("make_env", ALL_ENVS)
    def test_vmap_scan_multi_episode(self, make_env, key):
        """Batch of envs running 3-episode rollouts."""
        env, params = make_env()
        n_envs = 4
        T = params.max_steps * 3

        reset_keys = jax.random.split(key, n_envs)
        _, states_b = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, params)

        actions = _batch_zero_actions(env, params, n_envs, T)
        rollout_keys = jax.random.split(jax.random.fold_in(key, 2), n_envs)

        def single_rollout(k, state, acts):
            return scan_rollout(env, k, state, params, acts)

        _, _, _, cost_traj, done_traj, _ = jax.vmap(single_rollout)(
            rollout_keys, states_b, actions
        )

        # Each env should have ≥2 done=True
        dones_per_env = jnp.sum(done_traj, axis=1)
        assert bool(jnp.all(dones_per_env >= 2)), (
            f"Expected >=2 dones per env, got {dones_per_env}"
        )


# ========================== batch_reset / batch_step from jax_utils ==========================

class TestJaxUtilsFunctions:
    """Test the jax_utils.py public API: batch_reset, batch_step, scan_rollout."""

    @pytest.mark.parametrize("make_env", ALL_ENVS)
    def test_batch_reset(self, make_env, key):
        env, params = make_env()
        n = 8
        keys = split_key_for_envs(key, n)
        obs, states = batch_reset(env, keys, params)
        assert obs.shape[0] == n

    @pytest.mark.parametrize("make_env", ALL_ENVS)
    def test_batch_step(self, make_env, key):
        env, params = make_env()
        n = 8
        keys = split_key_for_envs(key, n)
        obs, states = batch_reset(env, keys, params)

        step_keys = jax.random.split(jax.random.fold_in(key, 1), n)
        actions = jax.vmap(lambda _: _zero_action(env, params))(jnp.arange(n))
        obs2, states2, rewards, costs, dones, infos = batch_step(
            env, step_keys, states, actions, params
        )
        assert obs2.shape[0] == n
        assert rewards.shape == (n,)
        assert costs.shape == (n, len(env.constraint_names(params)))
        assert dones.shape == (n,)
