# Vendored test from PowerZooJax.
# Source        : tests/resource/test_diesel.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
"""Tests for DieselBundle (and pure helpers)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.tree_util as tu
import pytest

from powermarketjax.resources.diesel import (
    DieselParams,
    DieselBundle,
    DieselBundleState,
    compute_dg_power,
    compute_dg_fuel_cost,
    compute_dg_emissions,
    make_diesel_bundle,
)


# ---------------------------------------------------------------------------
# Pure helper formulas
# ---------------------------------------------------------------------------

class TestPureHelpers:

    def test_compute_dg_power_clamp(self):
        assert float(compute_dg_power(jnp.float32(-0.5), 0.6)) == pytest.approx(0.0)
        assert float(compute_dg_power(jnp.float32(1.5), 0.6)) == pytest.approx(0.6)
        assert float(compute_dg_power(jnp.float32(0.5), 0.6)) == pytest.approx(0.3)

    def test_compute_dg_fuel_cost_formula(self):
        cost = compute_dg_fuel_cost(jnp.float32(0.3), dt_h=5 / 60, fuel_cost_per_mwh=300.0)
        assert float(cost) == pytest.approx(0.3 * 5 / 60 * 300.0, rel=1e-5)

    def test_compute_dg_emissions_formula(self):
        em = compute_dg_emissions(jnp.float32(0.6), dt_h=5 / 60, emission_factor=0.80)
        assert float(em) == pytest.approx(0.6 * 5 / 60 * 1e3 * 0.80, rel=1e-5)

    def test_diesel_params_defaults(self):
        dp = DieselParams()
        assert dp.p_dg_max_mw == pytest.approx(0.6)
        assert dp.fuel_cost_per_mwh == pytest.approx(300.0)
        assert dp.emission_factor == pytest.approx(0.80)


# ---------------------------------------------------------------------------
# Bundle protocol & JAX contract
# ---------------------------------------------------------------------------

class TestDieselBundle:

    def _make(self, n=2, p_max=0.6, fuel=300.0, ef=0.80):
        return make_diesel_bundle(
            n_devices=n, p_max_mw=p_max, fuel_cost_per_mwh=fuel,
            emission_factor=ef, dt_hours=5 / 60,
        )

    def test_busless_factory_shape(self):
        b = self._make(n=3)
        assert b.n_devices == 3
        assert b.bus_idx.shape == (3,)
        assert int(b.bus_idx[0]) == 0 and int(b.bus_idx[2]) == 2
        assert b.action_dim == 3 and b.obs_dim == 6

    def test_reset_state(self):
        b = self._make(n=2)
        st = b.reset(jax.random.PRNGKey(0))
        assert isinstance(st, DieselBundleState)
        assert st.p_dg_mw.shape == (2,)
        assert float(jnp.sum(st.p_dg_mw)) == pytest.approx(0.0)

    def test_step_protocol_returns_5_tuple(self):
        b = self._make(n=2)
        st = b.reset(jax.random.PRNGKey(0))
        a = jnp.array([0.5, 0.8], dtype=jnp.float32)
        new_st, p_inj, q_inj, obs, info = b.step(st, a, ctx={})
        assert isinstance(new_st, DieselBundleState)
        assert p_inj.shape == (2,) and float(p_inj[0]) == pytest.approx(0.3, rel=1e-5)
        assert float(p_inj[1]) == pytest.approx(0.48, rel=1e-5)
        assert q_inj.shape == (2,) and float(jnp.sum(q_inj)) == 0.0
        assert obs.shape == (4,)  # n_devices * per_device_obs_dim = 2*2
        for key in ("cost", "fuel_cost", "carbon_kg"):
            assert key in info

    def test_action_clipping(self):
        b = self._make(n=2)
        st = b.reset(jax.random.PRNGKey(0))
        a = jnp.array([-1.0, 5.0], dtype=jnp.float32)
        _, p_inj, _, _, _ = b.step(st, a, ctx={})
        assert float(p_inj[0]) == pytest.approx(0.0)       # clamped to 0
        assert float(p_inj[1]) == pytest.approx(0.6, rel=1e-5)  # saturated

    def test_economics_consistency_with_helpers(self):
        b = self._make(n=1, p_max=0.6, fuel=300.0, ef=0.80)
        st = b.reset(jax.random.PRNGKey(0))
        a = jnp.array([0.5], dtype=jnp.float32)
        _, _, _, _, info = b.step(st, a, ctx={})
        expected_p = 0.5 * 0.6
        expected_fuel = expected_p * (5 / 60) * 300.0
        expected_carbon = expected_p * (5 / 60) * 1e3 * 0.80
        assert float(info["fuel_cost"]) == pytest.approx(expected_fuel, rel=1e-5)
        assert float(info["carbon_kg"]) == pytest.approx(expected_carbon, rel=1e-5)

    def test_jit_step(self):
        b = self._make(n=2)
        st = b.reset(jax.random.PRNGKey(0))
        a = jnp.array([0.3, 0.6], dtype=jnp.float32)
        jstep = jax.jit(b.step)
        new_st, p, q, o, info = jstep(st, a, {})
        assert p.shape == (2,)

    def test_vmap_over_envs(self):
        b = self._make(n=2)
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        states = jax.vmap(b.reset)(keys)
        actions = jnp.full((4, 2), 0.4, dtype=jnp.float32)
        new_st, p, q, o, info = jax.vmap(lambda s, a: b.step(s, a, {}))(states, actions)
        assert p.shape == (4, 2)
        # Identical inputs → identical outputs across the batch
        for i in range(1, 4):
            assert jnp.allclose(p[0], p[i], atol=0.0)

    def test_pytree_structure_stable(self):
        b = self._make(n=2)
        st0 = b.reset(jax.random.PRNGKey(0))
        a = jnp.zeros(2, dtype=jnp.float32)
        st1, *_ = b.step(st0, a, {})
        l0 = tu.tree_leaves(st0)
        l1 = tu.tree_leaves(st1)
        assert len(l0) == len(l1)
        for x, y in zip(l0, l1):
            assert x.shape == y.shape and x.dtype == y.dtype

    def test_observe_matches_step_obs(self):
        b = self._make(n=2)
        st = b.reset(jax.random.PRNGKey(0))
        a = jnp.array([0.4, 0.7], dtype=jnp.float32)
        new_st, _, _, obs_step, _ = b.step(st, a, {})
        obs_observe = b.observe(new_st, {})
        assert jnp.allclose(obs_step, obs_observe, atol=1e-6)


# ---------------------------------------------------------------------------
# Minimum-loading soft constraint (opt-in, default no-op)
# ---------------------------------------------------------------------------

class TestMinLoading:
    """``p_min_norm`` enables a deadband + clamp, default 0 keeps old behaviour."""

    def _make(self, *, p_min_norm=0.0, n=1, p_max=1.0):
        return make_diesel_bundle(
            n_devices=n, p_max_mw=p_max, fuel_cost_per_mwh=300.0,
            emission_factor=0.80, p_min_norm=p_min_norm, dt_hours=5 / 60,
        )

    def test_default_zero_is_noop(self):
        """With p_min_norm=0 (default), p_dg = norm * p_max for any norm."""
        b = self._make(p_min_norm=0.0, n=1, p_max=1.0)
        st = b.reset(jax.random.PRNGKey(0))
        for a_val in [0.0, 0.05, 0.1, 0.5, 1.0]:
            _, p_inj, _, _, _ = b.step(st, jnp.array([a_val], dtype=jnp.float32), {})
            assert float(p_inj[0]) == pytest.approx(a_val, abs=1e-6)

    def test_setpoint_below_deadband_shuts_off(self):
        """Requested below p_min_norm/2 → DG OFF (p = 0)."""
        b = self._make(p_min_norm=0.3, n=1, p_max=1.0)
        st = b.reset(jax.random.PRNGKey(0))
        # Deadband = 0.15; below it → off
        for a_val in [0.0, 0.05, 0.1, 0.149]:
            _, p_inj, _, _, _ = b.step(st, jnp.array([a_val], dtype=jnp.float32), {})
            assert float(p_inj[0]) == pytest.approx(0.0, abs=1e-6), (
                f"a={a_val} should be OFF, got p={float(p_inj[0])}"
            )

    def test_setpoint_above_deadband_clamped_to_min(self):
        """Requested above p_min_norm/2 but below p_min_norm → clamped UP to p_min_norm."""
        b = self._make(p_min_norm=0.3, n=1, p_max=1.0)
        st = b.reset(jax.random.PRNGKey(0))
        # Deadband = 0.15, p_min_norm = 0.3
        # Requests in (0.15, 0.3] → clamped to 0.3
        for a_val in [0.16, 0.20, 0.25, 0.30]:
            _, p_inj, _, _, _ = b.step(st, jnp.array([a_val], dtype=jnp.float32), {})
            assert float(p_inj[0]) == pytest.approx(0.30, abs=1e-6), (
                f"a={a_val} should clamp to p_min=0.30, got p={float(p_inj[0])}"
            )

    def test_setpoint_above_min_passes_through(self):
        """Requested above p_min_norm → linear (no clamp)."""
        b = self._make(p_min_norm=0.3, n=1, p_max=1.0)
        st = b.reset(jax.random.PRNGKey(0))
        for a_val in [0.4, 0.5, 0.7, 1.0]:
            _, p_inj, _, _, _ = b.step(st, jnp.array([a_val], dtype=jnp.float32), {})
            assert float(p_inj[0]) == pytest.approx(a_val, abs=1e-6)

    def test_per_device_min_loading(self):
        """p_min_norm broadcasts per-device when given a sequence."""
        b = make_diesel_bundle(
            n_devices=2, p_max_mw=1.0, p_min_norm=[0.0, 0.5], dt_hours=5 / 60,
        )
        st = b.reset(jax.random.PRNGKey(0))
        a = jnp.array([0.1, 0.1], dtype=jnp.float32)
        _, p_inj, _, _, _ = b.step(st, a, {})
        # Device 0: no constraint → p = 0.1
        # Device 1: deadband=0.25, request 0.1 < deadband → OFF
        assert float(p_inj[0]) == pytest.approx(0.1, abs=1e-6)
        assert float(p_inj[1]) == pytest.approx(0.0, abs=1e-6)

    def test_jit_compatible(self):
        b = self._make(p_min_norm=0.3, n=2, p_max=0.6)
        st = b.reset(jax.random.PRNGKey(0))
        jstep = jax.jit(b.step)
        _, p_inj, _, _, _ = jstep(st, jnp.array([0.1, 0.5], dtype=jnp.float32), {})
        assert float(p_inj[0]) == pytest.approx(0.0)
        assert float(p_inj[1]) == pytest.approx(0.3, rel=1e-5)
