# Vendored test from PowerZooJax.
# Source        : tests/grid/test_power_flow.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
"""Tests for powermarketjax/physics/power_flow.py — L0 JAX + L1 physics."""

import jax
import jax.numpy as jnp
import pytest

from powermarketjax.case import create_case5
from powermarketjax.physics.power_flow import (
    ac_thermal_check,
    dc_power_flow,
    dc_power_flow_with_check,
    safety_check,
    compute_generation_cost,
    proportional_dispatch,
)


@pytest.fixture(scope="module")
def case5():
    return create_case5()


# ==================== dc_power_flow ====================

class TestDCPowerFlow:
    """L1: DCPF physics correctness + L0 JIT compatibility."""

    def test_output_shapes(self, case5):
        p = jnp.array([40.0, 170.0, 520.0, 200.0, 600.0])
        node_load = case5.nodes_loads_map @ jnp.array([0.0, 500.0, 600.0, 400.0, 0.0])
        flow, inj, actual_up = dc_power_flow(case5, p, node_load)
        assert flow.shape == (case5.n_lines,)
        assert inj.shape == (case5.n_nodes,)
        assert actual_up.shape == (case5.n_units,)

    def test_power_balance(self, case5):
        """Net injection at slack absorbs imbalance → total inj = 0."""
        p = jnp.array([10.0, 50.0, 100.0, 50.0, 100.0])  # total = 310
        load_demand = jnp.array([0.0, 100.0, 100.0, 100.0, 0.0])  # total = 300
        node_load = case5.nodes_loads_map @ load_demand
        flow, inj, actual_up = dc_power_flow(case5, p, node_load)
        # After slack absorption, total inj ~ 0
        assert jnp.abs(jnp.sum(inj)) < 1e-3
        # Actual unit power must sum to total load (slack picks up deficit)
        assert jnp.abs(jnp.sum(actual_up) - jnp.sum(node_load)) < 1e-3

    def test_zero_flow_at_uniform_load(self, case5):
        """If gen exactly meets load everywhere, flows should be near zero."""
        # One generator per bus, load matching gen exactly
        p = jnp.array([0.0, 0.0, 100.0, 0.0, 0.0])
        load_demand = jnp.array([0.0, 0.0, 100.0, 0.0, 0.0])
        node_load = case5.nodes_loads_map @ load_demand
        flow, inj, actual_up = dc_power_flow(case5, p, node_load)
        # All injection zero → flows zero
        assert jnp.allclose(flow, 0.0, atol=1e-3)

    def test_jit_compile(self, case5):
        """JIT compilation must succeed."""
        p = jnp.ones(case5.n_units) * 100.0
        load = jnp.ones(case5.n_nodes) * 80.0
        jit_fn = jax.jit(dc_power_flow)
        flow, inj, actual_up = jit_fn(case5, p, load)
        assert flow.shape == (case5.n_lines,)

    def test_vmap(self, case5):
        """vmap over batch of dispatches."""
        batch = 4
        ps = jnp.tile(jnp.ones(case5.n_units) * 100.0, (batch, 1))
        loads = jnp.tile(jnp.ones(case5.n_nodes) * 80.0, (batch, 1))
        vfn = jax.vmap(lambda p, l: dc_power_flow(case5, p, l))
        flows, injs, actual_ups = vfn(ps, loads)
        assert flows.shape == (batch, case5.n_lines)
        assert injs.shape == (batch, case5.n_nodes)
        assert actual_ups.shape == (batch, case5.n_units)

    def test_slack_cost_accountability(self, case5):
        """Slack-adjusted dispatch must cost more than naive dispatch when under-generated.

        If cost used raw ``unit_power_mw`` instead of ``actual_unit_power_mw``, an RL
        agent could under-dispatch and avoid paying for slack make-up energy.
        """
        p_low = jnp.array([5.0, 10.0, 20.0, 10.0, 20.0])
        load_demand = jnp.array([0.0, 300.0, 300.0, 400.0, 0.0])
        node_load = case5.nodes_loads_map @ load_demand

        _, _, actual_up = dc_power_flow(case5, p_low, node_load)

        assert float(jnp.sum(actual_up)) > float(jnp.sum(p_low))

        c = case5.unit_cost_c
        zeros = jnp.zeros(case5.n_units)
        cost_actual = compute_generation_cost(actual_up, zeros, zeros, c)
        cost_naive = compute_generation_cost(p_low, zeros, zeros, c)
        assert float(cost_actual) > float(cost_naive)

    def test_slack_unit_no_clamp(self, case5):
        """Slack-bus units may exceed p_max when balancing a huge load (no clamp in DCPF).

        If this fails, someone added clamping in ``dc_power_flow``; re-check power balance.
        """
        p = case5.unit_p_min
        total_load = float(jnp.sum(case5.unit_p_max))
        node_load = jnp.ones(case5.n_nodes, dtype=jnp.float32) * (
            total_load / float(case5.n_nodes)
        )

        _, _, actual_up = dc_power_flow(case5, p, node_load)

        slack_idx = int(case5.slack_bus_idx)
        slack_unit_mask = case5.nodes_units_map[slack_idx] > 0
        assert bool(jnp.any(slack_unit_mask))
        slack_actual = actual_up[slack_unit_mask]
        slack_pmax = case5.unit_p_max[slack_unit_mask]
        assert jnp.any(slack_actual > slack_pmax)


# ==================== safety_check ====================

class TestSafetyCheck:
    def test_safe(self):
        flow = jnp.array([100.0, -50.0, 200.0])
        cap = jnp.array([400.0, 400.0, 400.0])
        floor = jnp.array([-400.0, -400.0, -400.0])
        is_safe, n_viol, cost = safety_check(flow, cap, floor)
        assert bool(is_safe)
        assert int(n_viol) == 0
        assert float(cost) == 0.0

    def test_violation(self):
        flow = jnp.array([500.0, -50.0])
        cap = jnp.array([400.0, 400.0])
        floor = jnp.array([-400.0, -400.0])
        is_safe, n_viol, cost = safety_check(flow, cap, floor)
        assert not bool(is_safe)
        assert int(n_viol) == 1
        assert float(cost) == pytest.approx(100.0)

    def test_floor_violation(self):
        flow = jnp.array([-500.0, 0.0])
        cap = jnp.array([400.0, 400.0])
        floor = jnp.array([-400.0, -400.0])
        is_safe, n_viol, cost = safety_check(flow, cap, floor)
        assert not bool(is_safe)
        assert int(n_viol) == 1
        assert float(cost) == pytest.approx(100.0)

    def test_jit(self):
        jit_fn = jax.jit(safety_check)
        flow = jnp.array([100.0])
        cap = jnp.array([200.0])
        floor = jnp.array([-200.0])
        is_safe, n_viol, cost = jit_fn(flow, cap, floor)
        assert bool(is_safe)


# ==================== ac_thermal_check ====================

class TestAcThermalCheck:
    """AC apparent-power (MVA) thermal check vs DC ``safety_check``."""

    def test_jit(self):
        z = jnp.zeros(3)
        is_safe, n_viol, cost = jax.jit(ac_thermal_check)(
            jnp.array([100.0, 200.0, 50.0]),
            jnp.array([10.0, 20.0, 5.0]),
            jnp.array([400.0, 400.0, 400.0]),
            z, z, False)
        assert is_safe.dtype == jnp.bool_

    def test_mva_dominates_active_component(self):
        """|S| = sqrt(P^2+Q^2) >= |P| for each line."""
        pf = jnp.array([30.0, -40.0, 100.0])
        qf = jnp.array([40.0, 30.0, 5.0])
        sf = jnp.sqrt(pf ** 2 + qf ** 2)
        assert jnp.all(sf + 1e-6 >= jnp.abs(pf))

    def test_both_ends_uses_max_apparent(self):
        """With two ends, thermal flow is max(|Sf|, |St|); overload cost uses that max."""
        pf = jnp.array([100.0])
        qf = jnp.array([0.0])
        pt = jnp.array([80.0])
        qt = jnp.array([60.0])
        cap = jnp.array([90.0])
        z = jnp.zeros(1)
        sf = 100.0
        st = float(jnp.sqrt(80.0 ** 2 + 60.0 ** 2))
        assert st == pytest.approx(100.0)
        _, _, c_one = ac_thermal_check(pf, qf, cap, z, z, False)
        _, _, c_two = ac_thermal_check(pf, qf, cap, pt, qt, True)
        assert float(c_one) == pytest.approx(sf - 90.0, rel=1e-5)
        assert float(c_two) == pytest.approx(max(sf, st) - 90.0, rel=1e-5)


# ==================== compute_generation_cost ====================

class TestGenerationCost:
    def test_constant_marginal_cost(self):
        # MC(p) = c (constant) → TC = c * p
        p = jnp.array([100.0, 200.0])
        a = jnp.zeros(2)
        b = jnp.zeros(2)
        c = jnp.array([10.0, 20.0])
        cost = compute_generation_cost(p, a, b, c)
        expected = 10.0 * 100.0 + 20.0 * 200.0  # 5000
        assert float(cost) == pytest.approx(expected)

    def test_linear_cost(self):
        # MC(p) = 0*p^2 + 2*p + 5 → TC = 0 + p^2 + 5*p
        p = jnp.array([100.0])
        a = jnp.zeros(1)
        b = jnp.array([2.0])
        c = jnp.array([5.0])
        cost = compute_generation_cost(p, a, b, c)
        expected = (2.0 / 2) * 100 ** 2 + 5.0 * 100  # 10000 + 500
        assert float(cost) == pytest.approx(expected)

    def test_quadratic_cost(self):
        # MC(p) = 0.5*p^2 + 2*p + 1 → TC = (0.5/3)*p^3 + (2/2)*p^2 + 1*p
        p = jnp.array([10.0])
        a = jnp.array([0.5])
        b = jnp.array([2.0])
        c = jnp.array([1.0])
        cost = compute_generation_cost(p, a, b, c)
        expected = (0.5 / 3) * 1000 + 1.0 * 100 + 1.0 * 10  # ~276.67
        assert float(cost) == pytest.approx(expected, rel=1e-4)

    def test_cost_convention_is_mc_not_tc(self):
        """cost_a is an MC coefficient; integrated TC differs from MATPOWER-style quadratic TC."""
        p = jnp.array([6.0])
        a = jnp.array([1.0])
        b = jnp.zeros(1)
        c = jnp.zeros(1)

        cost = compute_generation_cost(p, a, b, c)

        # This codebase: MC = p^2 → TC = (1/3) p^3 = 72 at p=6
        assert float(cost) == pytest.approx(72.0, rel=1e-5)
        # MATPOWER-style TC = a·p^2 would give 36 at p=6, a=1 — not equal
        assert abs(float(cost) - 36.0) > 1.0


# ==================== proportional_dispatch ====================

class TestProportionalDispatch:
    def test_meets_load(self, case5):
        total_load = 800.0
        p = proportional_dispatch(jnp.float32(total_load), case5.unit_p_min, case5.unit_p_max)
        assert jnp.abs(jnp.sum(p) - total_load) < 1.0

    def test_within_limits(self, case5):
        total_load = 600.0
        p = proportional_dispatch(jnp.float32(total_load), case5.unit_p_min, case5.unit_p_max)
        assert jnp.all(p >= case5.unit_p_min - 1e-5)
        assert jnp.all(p <= case5.unit_p_max + 1e-5)

    def test_min_load(self, case5):
        """If load equals sum(p_min), dispatch should be p_min."""
        total_load = float(jnp.sum(case5.unit_p_min))
        p = proportional_dispatch(jnp.float32(total_load), case5.unit_p_min, case5.unit_p_max)
        assert jnp.allclose(p, case5.unit_p_min, atol=1e-3)

    def test_max_load(self, case5):
        """If load equals sum(p_max), dispatch should be p_max."""
        total_load = float(jnp.sum(case5.unit_p_max))
        p = proportional_dispatch(jnp.float32(total_load), case5.unit_p_min, case5.unit_p_max)
        assert jnp.allclose(p, case5.unit_p_max, atol=1e-3)

    def test_jit(self, case5):
        jit_fn = jax.jit(proportional_dispatch)
        p = jit_fn(jnp.float32(500.0), case5.unit_p_min, case5.unit_p_max)
        assert p.shape == (case5.n_units,)


# ==================== dc_power_flow_with_check ====================

class TestDCPFWithCheck:
    def test_safe_dispatch(self, case5):
        """A balanced dispatch within limits should be safe."""
        total_load = 800.0
        p = proportional_dispatch(jnp.float32(total_load), case5.unit_p_min, case5.unit_p_max)
        load_demand = jnp.array([0.0, 300.0, 300.0, 200.0, 0.0])
        node_load = case5.nodes_loads_map @ load_demand
        flow, inj, actual_up, is_safe, n_viol, cost = dc_power_flow_with_check(case5, p, node_load)
        assert flow.shape == (case5.n_lines,)
        assert actual_up.shape == (case5.n_units,)
        # These checks depend on the specific limits, but no assertion on safety value
        # since it depends on PTDF and limits. Just check types.
        assert is_safe.dtype == jnp.bool_
        assert n_viol.dtype == jnp.int32
