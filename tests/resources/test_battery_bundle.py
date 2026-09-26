# Vendored test from PowerZooJax.
# Source        : tests/resource/test_battery_bundle.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
"""L0 + L1 tests for BatteryBundle — SoA bundle for grid env resource attachment.

L0: JAX contracts — JIT, vmap, pytree structure stability
L1: Physics correctness — SOC dynamics, feasibility clipping, obs layout
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from powermarketjax.case import create_case5
from powermarketjax.resources.battery import (
    BatteryBundle,
    BatteryBundleState,
    make_battery_bundle,
    compute_feasible_power_batch,
)


# ========================== Fixtures ==========================

@pytest.fixture(scope="module")
def case5():
    return create_case5()


@pytest.fixture(scope="module")
def bundle(case5):
    return make_battery_bundle(
        case5,
        bus_ids=[2, 4],
        power_mw=20.0,
        capacity_mwh=50.0,
        soc_min=0.1,
        soc_max=0.9,
        eta_charge=0.95,
        eta_discharge=0.95,
        initial_soc=0.5,
        dt_hours=0.5,
    )


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


# ========================== L0: JAX Contracts ==========================

class TestBundleJaxContracts:

    def test_bundle_default_state_shapes(self, bundle, key):
        state = bundle.reset(key)
        assert state.soc.shape == (2,)
        assert jnp.allclose(state.soc, jnp.array([0.5, 0.5]))

    def test_bundle_step_jit(self, bundle, key):
        state = bundle.reset(key)
        action = jnp.zeros(bundle.action_dim)
        step_fn = jax.jit(bundle.step)
        new_state, p_inj, q_inj, obs, cost = step_fn(state, action, {})
        assert new_state.soc.shape == (2,)
        assert p_inj.shape == (2,)
        assert q_inj.shape == (2,)
        assert obs.shape == (bundle.obs_dim,)

    def test_bundle_step_vmap(self, bundle, key):
        state = bundle.reset(key)
        batch = 8
        states = jax.tree.map(lambda x: jnp.tile(x[None], (batch,) + (1,) * x.ndim), state)
        actions = jnp.zeros((batch, bundle.action_dim))

        def step_single(s, a):
            return bundle.step(s, a, {})

        new_states, p_inj, q_inj, obs, cost = jax.vmap(step_single)(states, actions)
        assert new_states.soc.shape == (batch, 2)
        assert p_inj.shape == (batch, 2)
        assert obs.shape == (batch, bundle.obs_dim)

    def test_bundle_action_dim_obs_dim_derivation(self, bundle):
        assert bundle.action_dim == bundle.n_devices * bundle.per_device_action_dim
        assert bundle.obs_dim == bundle.n_devices * bundle.per_device_obs_dim
        assert bundle.action_dim == 2
        assert bundle.obs_dim == 4


# ========================== L1: Cycle throughput cost ==========================

class TestBundleCycleCost:

    def test_cost_info_includes_cost_cycle(self, bundle, key):
        state = bundle.reset(key)
        _, _, _, _, cost_info = bundle.step(state, jnp.zeros(bundle.action_dim), {})
        assert "cost_cycle" in cost_info
        assert "cost_soc_clip" in cost_info
        assert "cost_sum" in cost_info

    def test_cost_cycle_zero_by_default(self, case5):
        b = make_battery_bundle(case5, bus_ids=[2, 4], cycle_cost_per_mwh=0.0)
        state = b.reset(jax.random.PRNGKey(0))
        _, _, _, _, cost_info = b.step(state, jnp.array([1.0, -1.0]), {})
        assert float(cost_info["cost_cycle"]) == pytest.approx(0.0)

    def test_cost_cycle_formula_two_devices(self, case5):
        rate = 10.0
        dt = 0.5
        b = make_battery_bundle(
            case5,
            bus_ids=[2, 4],
            power_mw=20.0,
            capacity_mwh=50.0,
            initial_soc=0.5,
            dt_hours=dt,
            cycle_cost_per_mwh=rate,
        )
        state = b.reset(jax.random.PRNGKey(0))
        p_des = jnp.array([1.0, -1.0])  # ±20 MW
        _, p_inj, _, _, cost_info = b.step(state, p_des, {})
        expected = float(jnp.sum(jnp.abs(p_inj)) * dt * rate)
        assert float(cost_info["cost_cycle"]) == pytest.approx(expected, rel=1e-5)
        assert float(cost_info["cost_sum"]) == pytest.approx(expected, rel=1e-5)

    def test_make_battery_bundle_passes_cycle_cost(self, case5):
        b = make_battery_bundle(case5, bus_ids=[2], cycle_cost_per_mwh=7.5)
        assert b.cycle_cost_per_mwh == pytest.approx(7.5)


# ========================== L1: Physics Correctness ==========================

class TestBundlePhysics:

    def test_bundle_step_soc_update_discharge(self, bundle, key):
        state = bundle.reset(key)
        action = jnp.array([1.0, 1.0])  # max discharge
        new_state, p_inj, _, _, _ = bundle.step(state, action, {})
        assert jnp.all(p_inj > 0), "Discharge should produce positive injection"
        assert jnp.all(new_state.soc < state.soc), "SOC should decrease on discharge"

    def test_bundle_step_soc_update_charge(self, bundle, key):
        state = bundle.reset(key)
        action = jnp.array([-1.0, -1.0])  # max charge
        new_state, p_inj, _, _, _ = bundle.step(state, action, {})
        assert jnp.all(p_inj < 0), "Charge should produce negative injection"
        assert jnp.all(new_state.soc > state.soc), "SOC should increase on charge"

    def test_bundle_step_soc_clipping(self, bundle, key):
        """Action exceeding SOC range should be clipped with cost > 0."""
        low_soc = BatteryBundleState(soc=jnp.array([0.11, 0.11]))
        action = jnp.array([1.0, 1.0])  # try max discharge near soc_min
        new_state, p_inj, _, _, cost_info = bundle.step(low_soc, action, {})
        assert new_state.soc.shape == (2,)
        # With very limited SOC headroom, feasible power < desired
        desired = 1.0 * bundle.power_max
        feasible = compute_feasible_power_batch(
            low_soc.soc, desired, bundle.power_max, bundle.capacity,
            bundle.soc_min, bundle.soc_max, bundle.eta_charge, bundle.eta_discharge,
            bundle.dt_hours)
        assert jnp.all(feasible < desired), "Feasible should be less than desired near soc_min"

    def test_bundle_obs_device_major_layout(self, bundle, key):
        """obs_slice should be device-major: [soc_0, p_0, soc_1, p_1]."""
        state = bundle.reset(key)
        action = jnp.array([0.5, -0.3])
        new_state, p_inj, _, obs_slice, _ = bundle.step(state, action, {})
        reshaped = obs_slice.reshape(bundle.n_devices, bundle.per_device_obs_dim)
        safe_p = jnp.maximum(bundle.power_max, 1e-6)
        for i in range(bundle.n_devices):
            assert jnp.allclose(reshaped[i, 0], new_state.soc[i], atol=1e-6), \
                f"Device {i}: obs soc should match state soc"
            assert jnp.allclose(reshaped[i, 1], p_inj[i] / safe_p[i], atol=1e-6), \
                f"Device {i}: obs p_norm should match feasible/power_max"

    def test_bundle_observe_without_stepping(self, bundle, key):
        state = bundle.reset(key)
        obs = bundle.observe(state, {})
        assert obs.shape == (bundle.obs_dim,)
        reshaped = obs.reshape(bundle.n_devices, bundle.per_device_obs_dim)
        for i in range(bundle.n_devices):
            assert jnp.allclose(reshaped[i, 0], state.soc[i], atol=1e-6)
            assert jnp.allclose(reshaped[i, 1], 0.0, atol=1e-6)


# ========================== make_battery_bundle ==========================

class TestMakeBatteryBundle:

    def test_bus_resolution(self, case5):
        bundle = make_battery_bundle(case5, bus_ids=[1, 3, 5])
        node_ids = np.asarray(case5.node_ids)
        for i, bid in enumerate([1, 3, 5]):
            expected_idx = int(np.where(node_ids == bid)[0][0])
            assert int(bundle.bus_idx[i]) == expected_idx

    def test_scalar_broadcast(self, case5):
        bundle = make_battery_bundle(case5, bus_ids=[2, 4], power_mw=15.0)
        assert bundle.power_max.shape == (2,)
        assert jnp.allclose(bundle.power_max, jnp.array([15.0, 15.0]))

    def test_sequence_kwargs(self, case5):
        bundle = make_battery_bundle(
            case5, bus_ids=[2, 4],
            power_mw=[20.0, 15.0],
            capacity_mwh=[50.0, 30.0],
        )
        assert jnp.allclose(bundle.power_max, jnp.array([20.0, 15.0]))
        assert jnp.allclose(bundle.capacity, jnp.array([50.0, 30.0]))

    def test_invalid_bus_raises(self, case5):
        with pytest.raises(ValueError, match="not found"):
            make_battery_bundle(case5, bus_ids=[999])

    def test_empty_bus_ids_raises(self, case5):
        with pytest.raises(ValueError, match="non-empty"):
            make_battery_bundle(case5, bus_ids=[])

    def test_enable_q_control_dims(self, case5):
        bundle = make_battery_bundle(
            case5, bus_ids=[2, 4], enable_q_control=True)
        assert bundle.per_device_action_dim == 2
        assert bundle.per_device_obs_dim == 3
        assert bundle.action_dim == 4   # 2 devices × 2
        assert bundle.obs_dim == 6      # 2 devices × 3
        assert bundle.enable_q_control is True

    def test_s_rated_defaults_to_power_max(self, case5):
        bundle = make_battery_bundle(
            case5, bus_ids=[2, 4], power_mw=15.0, enable_q_control=True)
        assert jnp.allclose(bundle.s_rated, bundle.power_max)

    def test_s_rated_custom(self, case5):
        bundle = make_battery_bundle(
            case5, bus_ids=[2, 4], power_mw=20.0,
            s_rated_mva=24.0, enable_q_control=True)
        assert jnp.allclose(bundle.s_rated, jnp.array([24.0, 24.0]))


# ========================== P+Q Control Mode ==========================

@pytest.fixture(scope="module")
def bundle_pq(case5):
    return make_battery_bundle(
        case5,
        bus_ids=[2, 4],
        power_mw=20.0,
        capacity_mwh=50.0,
        soc_min=0.1,
        soc_max=0.9,
        eta_charge=0.95,
        eta_discharge=0.95,
        initial_soc=0.5,
        dt_hours=0.5,
        enable_q_control=True,
    )


class TestBundlePQJaxContracts:

    def test_pq_step_jit(self, bundle_pq, key):
        state = bundle_pq.reset(key)
        action = jnp.zeros(bundle_pq.action_dim)
        new_state, p_inj, q_inj, obs, cost = jax.jit(bundle_pq.step)(state, action, {})
        assert p_inj.shape == (2,)
        assert q_inj.shape == (2,)
        assert obs.shape == (bundle_pq.obs_dim,)  # 6

    def test_pq_step_vmap(self, bundle_pq, key):
        state = bundle_pq.reset(key)
        batch = 8
        states = jax.tree.map(
            lambda x: jnp.tile(x[None], (batch,) + (1,) * x.ndim), state)
        actions = jnp.zeros((batch, bundle_pq.action_dim))
        new_states, p_inj, q_inj, obs, cost = jax.vmap(
            lambda s, a: bundle_pq.step(s, a, {}))(states, actions)
        assert q_inj.shape == (batch, 2)
        assert obs.shape == (batch, bundle_pq.obs_dim)


class TestBundlePQPhysics:

    def test_pure_q_injection(self, bundle_pq, key):
        """P=0, Q=1 → SOC unchanged, Q = S_rated."""
        state = bundle_pq.reset(key)
        # [P_0, Q_0, P_1, Q_1] — P=0, Q=max
        action = jnp.array([0.0, 1.0, 0.0, 1.0])
        new_state, p_inj, q_inj, _, _ = bundle_pq.step(state, action, {})
        assert jnp.allclose(new_state.soc, state.soc, atol=1e-6), \
            "SOC should be unchanged when P=0"
        assert jnp.allclose(q_inj, bundle_pq.s_rated, atol=1e-4), \
            "Q should equal S_rated when P=0 and Q_norm=1"
        assert jnp.allclose(p_inj, 0.0, atol=1e-6)

    def test_pure_q_absorption(self, bundle_pq, key):
        """P=0, Q=-1 → negative Q injection (absorb reactive)."""
        state = bundle_pq.reset(key)
        action = jnp.array([0.0, -1.0, 0.0, -1.0])
        _, _, q_inj, _, _ = bundle_pq.step(state, action, {})
        assert jnp.all(q_inj < 0)

    def test_pq_circle_constraint(self, bundle_pq, key):
        """Full P discharge + full Q → S should not exceed S_rated."""
        state = bundle_pq.reset(key)
        # Both P and Q at max → should be projected by PQ circle
        action = jnp.array([1.0, 1.0, 1.0, 1.0])
        _, p_inj, q_inj, _, _ = bundle_pq.step(state, action, {})
        s = jnp.sqrt(p_inj ** 2 + q_inj ** 2)
        assert jnp.all(s <= bundle_pq.s_rated + 1e-5), \
            "Apparent power must not exceed S_rated"

    def test_p_priority_soc_clip_frees_q_headroom(self, bundle_pq, key):
        """When SOC clips P down, Q gets more headroom from PQ circle."""
        from powermarketjax.resources.battery import BatteryBundleState
        # Near soc_min → P discharge heavily clipped → Q has almost full S_rated
        low_soc = BatteryBundleState(soc=jnp.array([0.11, 0.11]))
        action = jnp.array([1.0, 1.0, 1.0, 1.0])
        _, p_inj, q_inj, _, _ = bundle_pq.step(low_soc, action, {})
        # P should be small (SOC-limited)
        assert jnp.all(p_inj < bundle_pq.power_max * 0.5)
        # Q should be close to S_rated since P is small
        q_max_expected = jnp.sqrt(bundle_pq.s_rated ** 2 - p_inj ** 2)
        assert jnp.allclose(q_inj, q_max_expected, atol=1e-4)

    def test_pq_obs_device_major_layout(self, bundle_pq, key):
        """obs_slice should be [soc_0, p_0, q_0, soc_1, p_1, q_1]."""
        state = bundle_pq.reset(key)
        action = jnp.array([0.5, 0.3, -0.2, -0.4])
        new_state, p_inj, q_inj, obs_slice, _ = bundle_pq.step(state, action, {})
        reshaped = obs_slice.reshape(bundle_pq.n_devices, bundle_pq.per_device_obs_dim)
        safe_p = jnp.maximum(bundle_pq.power_max, 1e-6)
        safe_s = jnp.maximum(bundle_pq.s_rated, 1e-6)
        for i in range(bundle_pq.n_devices):
            assert jnp.allclose(reshaped[i, 0], new_state.soc[i], atol=1e-6)
            assert jnp.allclose(reshaped[i, 1], p_inj[i] / safe_p[i], atol=1e-6)
            assert jnp.allclose(reshaped[i, 2], q_inj[i] / safe_s[i], atol=1e-6)

    def test_pq_observe_without_stepping(self, bundle_pq, key):
        state = bundle_pq.reset(key)
        obs = bundle_pq.observe(state, {})
        assert obs.shape == (bundle_pq.obs_dim,)
        reshaped = obs.reshape(bundle_pq.n_devices, bundle_pq.per_device_obs_dim)
        for i in range(bundle_pq.n_devices):
            assert jnp.allclose(reshaped[i, 0], state.soc[i], atol=1e-6)
            assert jnp.allclose(reshaped[i, 1], 0.0, atol=1e-6)  # p=0
            assert jnp.allclose(reshaped[i, 2], 0.0, atol=1e-6)  # q=0
