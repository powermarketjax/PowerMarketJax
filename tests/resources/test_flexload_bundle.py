# Vendored test from PowerZooJax.
# Source        : tests/resource/test_flexload_bundle.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
"""L0 + L1 tests for FlexLoadBundle — SoA bundle for grid env resource attachment.

L0: JAX contracts — JIT, vmap, pytree structure stability
L1: Physics correctness — ring buffer dynamics, action scaling, obs layout
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.tree_util as tu
import numpy as np
import pytest

from powermarketjax.case import create_case33bw
from powermarketjax.resources.flexload import (
    DEFAULT_MAX_BUFFER,
    FlexLoadBundle,
    FlexLoadBundleState,
    make_flexload_bundle,
)


# ========================== Fixtures ==========================

@pytest.fixture(scope="module")
def case33():
    return create_case33bw()


@pytest.fixture(scope="module")
def bundle(case33):
    """6-device DSO main config bundle."""
    return make_flexload_bundle(
        case33,
        bus_ids=[6, 14, 18, 22, 28, 33],
        curtail_cap_mw=[0.15, 0.10, 0.10, 0.08, 0.12, 0.10],
        shift_cap_mw=[0.15, 0.10, 0.10, 0.08, 0.12, 0.10],
        shift_horizon=4,
        dt_hours=0.5,
    )


@pytest.fixture(scope="module")
def small_bundle(case33):
    """Single-device bundle for scalar checks."""
    return make_flexload_bundle(
        case33,
        bus_ids=[18],
        curtail_cap_mw=0.10,
        shift_cap_mw=0.10,
        shift_horizon=4,
        dt_hours=0.5,
    )


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


# ========================== L0: JAX Contracts ==========================

class TestFlexLoadBundleJaxContracts:

    def test_reset_shapes(self, bundle, key):
        state = bundle.reset(key)
        n = bundle.n_devices
        assert state.curtailed_mw.shape == (n,)
        assert state.shift_out_mw.shape == (n,)
        assert state.shift_in_mw.shape == (n,)
        assert state.deferred_buffer.shape == (n, DEFAULT_MAX_BUFFER)
        assert state.buffer_head.shape == (n,)
        assert state.buffer_size.shape == (n,)

    def test_reset_zeros(self, bundle, key):
        state = bundle.reset(key)
        assert jnp.all(state.curtailed_mw == 0.0)
        assert jnp.all(state.shift_out_mw == 0.0)
        assert jnp.all(state.buffer_size == 0)

    def test_step_output_shapes(self, bundle, key):
        state = bundle.reset(key)
        action = jnp.zeros(bundle.action_dim)
        new_state, p_inj, q_inj, obs, cost_info = bundle.step(state, action, {})
        n = bundle.n_devices
        assert p_inj.shape == (n,)
        assert q_inj.shape == (n,)
        assert obs.shape == (bundle.obs_dim,)
        assert obs.shape == (n * bundle.per_device_obs_dim,)

    # ---- JIT ----

    def test_step_jit_compiles(self, bundle, key):
        state = bundle.reset(key)
        action = jnp.zeros(bundle.action_dim)

        @jax.jit
        def jit_step(s, a):
            return bundle.step(s, a, {})

        new_state, p, q, obs, info = jit_step(state, action)
        assert obs.shape == (bundle.obs_dim,)

    def test_observe_jit_compiles(self, bundle, key):
        state = bundle.reset(key)

        @jax.jit
        def jit_observe(s):
            return bundle.observe(s, {})

        obs = jit_observe(state)
        assert obs.shape == (bundle.obs_dim,)

    def test_reset_jit_compiles(self, bundle, key):
        @jax.jit
        def jit_reset(k):
            return bundle.reset(k)

        state = jit_reset(key)
        assert state.curtailed_mw.shape == (bundle.n_devices,)

    # ---- vmap ----

    def test_step_vmap_over_batch(self, bundle, key):
        """vmap over a batch of environments (parallel envs pattern)."""
        batch = 4
        keys = jax.random.split(key, batch)
        states = jax.vmap(bundle.reset)(keys)

        actions = jnp.zeros((batch, bundle.action_dim))

        def step_fn(s, a):
            return bundle.step(s, a, {})

        batch_step = jax.vmap(step_fn)
        new_states, p, q, obs, info = batch_step(states, actions)

        assert p.shape == (batch, bundle.n_devices)
        assert obs.shape == (batch, bundle.obs_dim)

    def test_vmap_reset(self, bundle, key):
        keys = jax.random.split(key, 8)
        states = jax.vmap(bundle.reset)(keys)
        assert states.curtailed_mw.shape == (8, bundle.n_devices)

    # ---- pytree structure stability ----

    def test_pytree_structure_stable_after_step(self, bundle, key):
        state = bundle.reset(key)
        action = jnp.zeros(bundle.action_dim)
        new_state, *_ = bundle.step(state, action, {})

        leaves_before = tu.tree_leaves(state)
        leaves_after = tu.tree_leaves(new_state)
        assert len(leaves_before) == len(leaves_after)

        for lb, la in zip(leaves_before, leaves_after):
            assert lb.shape == la.shape, (
                f"Shape mismatch: before {lb.shape}, after {la.shape}"
            )

    def test_pytree_structure_stable_across_multiple_steps(self, bundle, key):
        state = bundle.reset(key)
        action = jnp.ones(bundle.action_dim) * 0.3
        for _ in range(5):
            state, *_ = bundle.step(state, action, {})

        leaves = tu.tree_leaves(state)
        n = bundle.n_devices
        expected_shapes = [
            (n,),   # curtailed_mw
            (n,),   # shift_out_mw
            (n,),   # shift_in_mw
            (n, DEFAULT_MAX_BUFFER),  # deferred_buffer
            (n,),   # buffer_head
            (n,),   # buffer_size
        ]
        actual_shapes = [l.shape for l in leaves]
        assert actual_shapes == expected_shapes


# ========================== L1: Physics Correctness ==========================

class TestFlexLoadBundlePhysics:

    def test_zero_action_zero_curtailment(self, bundle, key):
        state = bundle.reset(key)
        action = jnp.zeros(bundle.action_dim)
        new_state, p_inj, q_inj, obs, info = bundle.step(state, action, {})
        assert jnp.all(new_state.curtailed_mw == 0.0)
        assert jnp.all(new_state.shift_out_mw == 0.0)
        # p_inject should be ≤ 0 (shift_in from buffer is 0 at first step)
        assert jnp.all(p_inj == 0.0)

    def test_curtailment_scales_with_cap(self, case33, key):
        bundle = make_flexload_bundle(
            case33, bus_ids=[6, 14], curtail_cap_mw=[1.0, 2.0],
            shift_cap_mw=0.0, shift_horizon=1,
        )
        state = bundle.reset(key)
        action = jnp.ones(bundle.action_dim)  # full curtail, no shift
        new_state, p_inj, _, _, _ = bundle.step(state, action, {})
        np.testing.assert_allclose(
            np.array(new_state.curtailed_mw), [1.0, 2.0], atol=1e-5
        )
        np.testing.assert_allclose(
            np.array(p_inj), [1.0, 2.0], atol=1e-5
        )

    def test_q_inject_always_zero(self, bundle, key):
        state = bundle.reset(key)
        action = jnp.ones(bundle.action_dim) * 0.5
        _, _, q_inj, _, _ = bundle.step(state, action, {})
        assert jnp.all(q_inj == 0.0)

    def test_obs_values_in_range(self, bundle, key):
        state = bundle.reset(key)
        action = jnp.ones(bundle.action_dim) * 0.5
        _, _, _, obs, _ = bundle.step(state, action, {})
        assert jnp.all(obs >= 0.0), f"obs has negative entries: {obs}"
        assert jnp.all(obs <= 1.0), f"obs exceeds 1: {obs}"

    def test_shift_out_fills_buffer(self, bundle, key):
        state = bundle.reset(key)
        # Only shift (action[1] = 1.0), no curtailment
        action = jnp.zeros(bundle.action_dim).at[1::2].set(1.0)
        new_state, _, _, _, _ = bundle.step(state, action, {})
        # Buffer size should be > 0 for all devices
        assert jnp.all(new_state.buffer_size > 0)

    def test_shift_in_released_after_horizon(self, case33, key):
        """With shift_horizon=1, deferred demand should be released next step."""
        bundle = make_flexload_bundle(
            case33, bus_ids=[6], curtail_cap_mw=0.0,
            shift_cap_mw=1.0, shift_horizon=1,
        )
        state = bundle.reset(key)
        # Step 1: shift 1 MW
        action_shift = jnp.array([0.0, 1.0])
        state, _, _, _, _ = bundle.step(state, action_shift, {})
        assert jnp.all(state.buffer_size > 0)

        # Step 2: no action — deferred should be released
        action_zero = jnp.zeros(bundle.action_dim)
        state, p_inj, _, _, _ = bundle.step(state, action_zero, {})
        # p_inject should be negative (load increase from release)
        # p = curtail + shift_out - shift_in = 0 + 0 - released
        assert float(p_inj[0]) < 0.0

    def test_action_clamped_to_unit_interval(self, bundle, key):
        state = bundle.reset(key)
        action = jnp.ones(bundle.action_dim) * 2.0  # out of [0, 1]
        new_state, p_inj, _, obs, _ = bundle.step(state, action, {})
        # Should behave identically to action=1.0
        state2 = bundle.reset(key)
        action2 = jnp.ones(bundle.action_dim)
        new_state2, p_inj2, _, obs2, _ = bundle.step(state2, action2, {})
        np.testing.assert_allclose(
            np.array(p_inj), np.array(p_inj2), atol=1e-6
        )

    def test_cost_info_keys_present(self, bundle, key):
        state = bundle.reset(key)
        action = jnp.ones(bundle.action_dim) * 0.3
        _, _, _, _, cost_info = bundle.step(state, action, {})
        assert "cost_sum" in cost_info
        assert "cost_curtailment" in cost_info
        assert "cost_shift_discomfort" in cost_info

    def test_cost_non_negative(self, bundle, key):
        state = bundle.reset(key)
        action = jnp.ones(bundle.action_dim) * 0.5
        _, _, _, _, cost_info = bundle.step(state, action, {})
        assert float(cost_info["cost_sum"]) >= 0.0

    def test_action_dim_matches_n_devices(self, bundle):
        assert bundle.action_dim == bundle.n_devices * 2

    def test_obs_dim_matches_n_devices(self, bundle):
        assert bundle.obs_dim == bundle.n_devices * 5


# ========================== make_flexload_bundle ==========================

class TestMakeFlexLoadBundle:

    def test_bus_id_not_found_raises(self, case33):
        with pytest.raises(ValueError, match="not found in case.node_ids"):
            make_flexload_bundle(case33, bus_ids=[9999])

    def test_empty_bus_ids_raises(self, case33):
        with pytest.raises(ValueError, match="non-empty"):
            make_flexload_bundle(case33, bus_ids=[])

    def test_per_device_scalar_broadcast(self, case33):
        b = make_flexload_bundle(
            case33, bus_ids=[6, 14, 18],
            curtail_cap_mw=0.1, shift_cap_mw=0.2,
        )
        assert b.n_devices == 3
        np.testing.assert_allclose(np.array(b.curtail_cap_mw), [0.1, 0.1, 0.1])
        np.testing.assert_allclose(np.array(b.shift_cap_mw), [0.2, 0.2, 0.2])

    def test_per_device_list_spec(self, case33):
        b = make_flexload_bundle(
            case33, bus_ids=[6, 14, 18],
            curtail_cap_mw=[0.1, 0.2, 0.3],
            shift_cap_mw=[0.3, 0.2, 0.1],
        )
        np.testing.assert_allclose(
            np.array(b.curtail_cap_mw), [0.1, 0.2, 0.3], atol=1e-6
        )

    def test_bus_idx_resolved_correctly(self, case33, key):
        b = make_flexload_bundle(case33, bus_ids=[1], curtail_cap_mw=0.1, shift_cap_mw=0.1)
        assert int(b.bus_idx[0]) == 0  # bus_id=1 → internal index 0

    def test_single_device_bundle_works(self, case33, key):
        b = make_flexload_bundle(
            case33, bus_ids=[18], curtail_cap_mw=0.1, shift_cap_mw=0.1,
        )
        state = b.reset(key)
        action = jnp.array([0.5, 0.5])
        new_state, p, q, obs, info = b.step(state, action, {})
        assert obs.shape == (5,)
        assert p.shape == (1,)
