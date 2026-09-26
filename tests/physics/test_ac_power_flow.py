# Vendored test from PowerZooJax.
# Source        : tests/grid/test_ac_power_flow.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
"""Tests for powermarketjax/physics/ac_power_flow.py — L0 JAX + L1 physics.

L0: JIT compilation, vmap batching, pytree structure stability.
L1: Physical correctness (convergence, power balance, branch flows, voltage).
"""

import jax
import jax.numpy as jnp
import jax.tree_util as tu
import pytest

from powermarketjax.case import create_case5
from powermarketjax.physics.ac_power_flow import (
    ACPFSetup,
    ACPFResult,
    prepare_acpf,
    ac_power_flow,
    calc_branch_flows,
    ac_power_flow_with_check,
    _build_jacobian,
)
from powermarketjax.physics.power_flow import ac_thermal_check


@pytest.fixture(scope="module")
def case5():
    return create_case5()


@pytest.fixture(scope="module")
def setup5(case5):
    return prepare_acpf(case5)


@pytest.fixture(scope="module")
def result5(setup5):
    return ac_power_flow(setup5)


# ==================== L0: JAX Contract Tests ====================

class TestACPF_L0_JAX:
    """L0: JIT, vmap, pytree stability."""

    def test_acpf_jit_compile(self, setup5):
        jit_fn = jax.jit(ac_power_flow, static_argnums=(1, 2))
        result = jit_fn(setup5, 30, 1e-5)
        assert result.vm.shape == (setup5.n_bus,)
        assert result.converged

    def test_acpf_vmap_batch(self, setup5):
        batch_setup = tu.tree_map(lambda x: jnp.stack([x, x, x]), setup5)
        batch_fn = jax.vmap(lambda s: ac_power_flow(s, 30, 1e-5))
        results = batch_fn(batch_setup)
        assert results.vm.shape == (3, setup5.n_bus,)
        assert jnp.all(results.converged)

    def test_acpf_result_pytree(self, result5):
        leaves = tu.tree_leaves(result5)
        assert len(leaves) == 10  # vm, va, p_calc, q_calc, pf_from, qf_from, pf_to, qf_to, converged, iterations

    def test_acpf_setup_pytree(self, setup5):
        leaves = tu.tree_leaves(setup5)
        assert len(leaves) > 0
        # Reconstruct from flatten/unflatten
        flat, treedef = tu.tree_flatten(setup5)
        setup_rebuilt = tu.tree_unflatten(treedef, flat)
        assert setup_rebuilt.n_bus == setup5.n_bus

    def test_build_jacobian_jit(self, setup5):
        n_bus = setup5.n_bus
        Vm = jnp.ones(n_bus)
        Va = jnp.zeros(n_bus)
        G = setup5.Y_bus_real
        B = setup5.Y_bus_imag
        P = jnp.zeros(n_bus)
        Q = jnp.zeros(n_bus)
        n_pvpq = setup5.pv_pq_idx.shape[0]

        # bus_is_pq: PV buses (first n_pv) = False, PQ buses = True
        bus_is_pq = jnp.concatenate([
            jnp.zeros(setup5.n_pv, dtype=jnp.bool_),
            jnp.ones(setup5.n_pq, dtype=jnp.bool_),
        ])

        jit_jac = jax.jit(_build_jacobian)
        J = jit_jac(Vm, Va, G, B, P, Q, setup5.pv_pq_idx, bus_is_pq)
        # New fixed shape: (2*n_pvpq, 2*n_pvpq)
        assert J.shape == (2 * n_pvpq, 2 * n_pvpq)

    def test_calc_branch_flows_jit(self, setup5, result5):
        jit_fn = jax.jit(calc_branch_flows)
        pf, qf, pt, qt = jit_fn(result5.vm, result5.va, setup5)
        assert pf.shape == (setup5.br_from_idx.shape[0],)

    def test_calc_branch_flows_vmap(self, setup5, result5):
        batch_vm = jnp.stack([result5.vm, result5.vm])
        batch_va = jnp.stack([result5.va, result5.va])
        batch_setup = tu.tree_map(lambda x: jnp.stack([x, x]), setup5)
        vfn = jax.vmap(calc_branch_flows)
        pf, qf, pt, qt = vfn(batch_vm, batch_va, batch_setup)
        assert pf.shape == (2, setup5.br_from_idx.shape[0])

    def test_make_jaxpr(self, setup5):
        """jax.make_jaxpr must succeed (no Python side effects)."""
        jaxpr = jax.make_jaxpr(lambda s: ac_power_flow(s, 30, 1e-5))(setup5)
        assert jaxpr is not None


# ==================== L1: Physical Correctness Tests ====================

class TestACPF_L1_Physics:
    """L1: Physical correctness of the NR solver."""

    def test_nr_convergence_case5(self, result5):
        assert bool(result5.converged)
        assert int(result5.iterations) <= 10

    def test_voltage_at_slack(self, result5, setup5):
        """Slack bus (bus 0) maintains scheduled Vm and Va=0."""
        slack_idx = 0
        assert jnp.abs(result5.vm[slack_idx] - setup5.Vm_init[slack_idx]) < 1e-4
        assert jnp.abs(result5.va[slack_idx]) < 1e-4

    def test_pv_bus_voltage(self, result5, setup5):
        """PV buses not hitting Q-limits maintain their scheduled voltage magnitude."""
        pv_idx = setup5.pv_idx
        Qg_max = setup5.Qg_max_bus[pv_idx]
        Qg_min = setup5.Qg_min_bus[pv_idx]
        Q_calc_pv = result5.q_calc[pv_idx]
        q_tol = 1e-2  # tolerance for "near limit" detection (p.u.)
        for i_local, i_bus in enumerate(pv_idx):
            q = float(Q_calc_pv[i_local])
            qmax = float(Qg_max[i_local])
            qmin = float(Qg_min[i_local])
            # If Q is well within limits, the bus is not Q-switched → vm held at setpoint
            if qmin + q_tol < q < qmax - q_tol:
                assert jnp.abs(result5.vm[i_bus] - setup5.Vm_init[i_bus]) < 1e-3, \
                    f"PV bus {i_bus}: vm={result5.vm[i_bus]:.4f}, expected≈{setup5.Vm_init[i_bus]:.4f}"

    def test_power_balance(self, result5, setup5):
        """Sum of bus injections = sum of branch losses."""
        base_mva = setup5.base_mva
        net_inj_sum = float(jnp.sum(result5.p_calc) * base_mva)
        branch_loss_sum = float(jnp.sum(result5.pf_from + result5.pf_to))
        # Both should agree: net injection = total losses
        assert abs(net_inj_sum - branch_loss_sum) < 1.0, \
            f"Injection sum {net_inj_sum:.2f} != branch loss sum {branch_loss_sum:.2f}"
        # Losses should be positive
        assert branch_loss_sum >= -0.1

    def test_branch_flow_conservation(self, result5):
        """Branch losses (pf_from + pf_to) should be non-negative for each branch."""
        p_loss = result5.pf_from + result5.pf_to
        assert jnp.all(p_loss >= -1e-2), f"Negative loss: {p_loss}"

    def test_zero_load_flat_voltage(self, case5):
        """With zero loads, zero gen, and all Vg=1.0, voltage profile is flat."""
        case_flat = case5.replace(
            node_pd=jnp.zeros(5),
            node_qd=jnp.zeros(5),
            unit_pg=jnp.zeros(5),
            unit_qg=jnp.zeros(5),
            unit_vg=jnp.ones(5),
        )
        setup_flat = prepare_acpf(case_flat)
        result = ac_power_flow(setup_flat)
        assert bool(result.converged)
        assert jnp.allclose(result.vm, 1.0, atol=1e-3)
        assert jnp.allclose(result.va, 0.0, atol=1e-3)

    def test_voltage_magnitudes_reasonable(self, result5):
        """All voltages should be in [0.8, 1.2] p.u. for a normal system."""
        assert jnp.all(result5.vm > 0.8)
        assert jnp.all(result5.vm < 1.2)

    def test_voltage_angles_small(self, result5):
        """Voltage angles should be small (< 30 degrees) for a normal system."""
        assert jnp.all(jnp.abs(jnp.degrees(result5.va)) < 30.0)

    def test_branch_flows_shapes(self, result5, setup5):
        n_branch = setup5.br_from_idx.shape[0]
        assert result5.pf_from.shape == (n_branch,)
        assert result5.qf_from.shape == (n_branch,)
        assert result5.pf_to.shape == (n_branch,)
        assert result5.qf_to.shape == (n_branch,)

    def test_with_check(self, setup5, case5):
        (flow, inj, vm, va, safe, n_viol, cost_t, cost_v, converged) = \
            ac_power_flow_with_check(
                setup5, case5.line_cap, case5.line_floor,
                case5.node_v_min, case5.node_v_max)
        assert flow.shape == (case5.n_lines,)
        assert vm.shape == (case5.n_nodes,)
        assert safe.dtype == jnp.bool_
        assert converged.dtype == jnp.bool_

    def test_apparent_mva_ge_active_mw(self, result5):
        """|S| >= |P| at from end (reactive flow makes apparent > active)."""
        s_from = jnp.sqrt(result5.pf_from ** 2 + result5.qf_from ** 2)
        assert jnp.all(s_from + 1e-4 >= jnp.abs(result5.pf_from))

    def test_two_end_thermal_max_differs_from_from_only_when_asymmetric(self, result5, case5):
        """max(|Sf|,|St|) can differ from |Sf| when line losses split P across ends."""
        sf = jnp.sqrt(result5.pf_from ** 2 + result5.qf_from ** 2)
        st = jnp.sqrt(result5.pf_to ** 2 + result5.qf_to ** 2)
        thermal_max = jnp.maximum(sf, st)
        z = jnp.zeros_like(result5.pf_from)
        _, _, c_from = ac_thermal_check(
            result5.pf_from, result5.qf_from, case5.line_cap, z, z, False)
        _, _, c_max = ac_thermal_check(
            result5.pf_from, result5.qf_from, case5.line_cap,
            result5.pf_to, result5.qf_to, True)
        assert float(c_max) + 1e-4 >= float(c_from)
        assert jnp.all(thermal_max + 1e-5 >= jnp.abs(result5.pf_from))


# ==================== L1.5: Q-limit Switching Tests ====================

class TestACPF_Qlim:
    """L1.5: PV→PQ Q-limit switching behavior."""

    def test_qlim_no_switch_when_limits_generous(self, case5):
        """With ±inf Q limits (no Q data), all PV buses should hold their setpoints."""
        case_unlimited = case5.replace(unit_q_min=None, unit_q_max=None)
        setup_unlimited = prepare_acpf(case_unlimited)
        result_unlimited = ac_power_flow(setup_unlimited)
        assert bool(result_unlimited.converged)
        pv_idx = setup_unlimited.pv_idx
        for i in pv_idx:
            assert jnp.abs(result_unlimited.vm[i] - setup_unlimited.Vm_init[i]) < 1e-3, \
                f"PV bus {i}: vm={result_unlimited.vm[i]:.4f}, expected≈{setup_unlimited.Vm_init[i]:.4f}"

    def test_qlim_fields_in_setup(self, setup5):
        """ACPFSetup must contain Vm_setpoint, Qg_min_bus, Qg_max_bus."""
        assert setup5.Vm_setpoint is not None
        assert setup5.Qg_min_bus is not None
        assert setup5.Qg_max_bus is not None
        assert setup5.Vm_setpoint.shape == (setup5.n_bus,)
        assert setup5.Qg_min_bus.shape == (setup5.n_bus,)
        assert setup5.Qg_max_bus.shape == (setup5.n_bus,)

    def test_qlim_pq_buses_unlimited(self, setup5):
        """PQ buses must have Qg_min=-inf / Qg_max=+inf (no limit enforced)."""
        pq_idx = setup5.pq_idx
        assert jnp.all(jnp.isinf(setup5.Qg_min_bus[pq_idx]))
        assert jnp.all(jnp.isinf(setup5.Qg_max_bus[pq_idx]))

    def test_qlim_tight_limits_still_converges(self, case5):
        """PV buses with very tight Q limits switch to PQ but solver still converges."""
        case_tight = case5.replace(
            unit_q_min=jnp.full(5, -1.0),   # ±1 MVAr in p.u. = ±0.01
            unit_q_max=jnp.full(5,  1.0),
        )
        setup_tight = prepare_acpf(case_tight)
        result_tight = ac_power_flow(setup_tight)
        assert bool(result_tight.converged)
        assert result_tight.vm.shape == (5,)
        assert result_tight.va.shape == (5,)

    def test_qlim_switched_pv_voltage_drifts(self, case5):
        """PV buses that hit Q-limits switch to PQ and their voltages drift from setpoints."""
        # Case5 bus 5 (index 4) genuinely saturates at Qmin=-450 MVAr (no load, large reactive demand).
        # With Q-limits enabled, its voltage should differ from the setpoint.
        setup5 = prepare_acpf(case5)
        result5 = ac_power_flow(setup5)
        assert bool(result5.converged)

        # Identify which PV buses are Q-saturated (with tolerance for float32).
        # After switching, the NR drives Q_calc → Q_sched_eff ≈ limit, so
        # the absolute check is unreliable. Instead, detect switching by
        # whether vm deviates significantly from the setpoint.
        pv_idx = setup5.pv_idx
        vm_deviation = jnp.abs(result5.vm[pv_idx] - setup5.Vm_init[pv_idx])
        saturated = vm_deviation > 1e-3

        # At least one bus should be saturated (bus 5 in case5)
        assert jnp.any(saturated), "Expected at least one PV bus to be Q-saturated in case5"

        # Saturated bus must deviate from setpoint (lost voltage control)
        for i_local, i_bus in enumerate(pv_idx):
            if bool(saturated[i_local]):
                deviation = float(jnp.abs(result5.vm[i_bus] - setup5.Vm_init[i_bus]))
                assert deviation > 1e-3, \
                    f"Q-saturated bus {i_bus} should deviate from setpoint, got {deviation:.5f}"

    def test_qlim_jit_vmap_stable(self, case5):
        """Q-limit logic must survive JIT + vmap without shape errors."""
        case_tight = case5.replace(
            unit_q_min=jnp.full(5, -1.0),
            unit_q_max=jnp.full(5,  1.0),
        )
        setup_tight = prepare_acpf(case_tight)
        import jax.tree_util as tu
        batch_setup = tu.tree_map(lambda x: jnp.stack([x, x]), setup_tight)
        batch_fn = jax.vmap(lambda s: ac_power_flow(s, 30, 1e-5))
        results = batch_fn(batch_setup)
        assert results.vm.shape == (2, 5)
        assert jnp.all(results.converged)
