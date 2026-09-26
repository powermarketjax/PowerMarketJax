# Vendored test from PowerZooJax.
# Source        : tests/grid/test_bfs_3phase_power_flow.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
"""Tests for bfs_3phase_power_flow.py — L0 JAX + L1 physics.

L0: JIT, vmap, pytree stability for 3-phase BFS.
L1: Convergence, voltage profile, balanced load symmetry.
"""

import jax
import jax.numpy as jnp
import jax.tree_util as tu
import numpy as np
import pytest

from powermarketjax.physics.bfs_3phase_power_flow import (
    ThreePhaseTopoData,
    BFS3PhResult,
    build_3phase_topology,
    bfs_3phase_power_flow,
)


@pytest.fixture(autouse=True, scope="module")
def _force_float32_mode():
    """Keep 3-phase BFS tests isolated from ACOPF's global x64 side effects."""
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", False)


@pytest.fixture(scope="module")
def small_3ph_topo():
    """4-bus radial network with balanced 3-phase impedances."""
    n_nodes = 4
    from_nodes = np.array([0, 1, 2])
    to_nodes = np.array([1, 2, 3])
    Z = np.zeros((3, 3, 3), dtype=complex)
    for i in range(3):
        Z[i] = (0.1 + 0.2j) * np.eye(3)
    return build_3phase_topology(n_nodes, from_nodes, to_nodes, Z, ref_bus=0)


@pytest.fixture(scope="module")
def balanced_load():
    """Balanced 3-phase load: 0.01 p.u. per bus-phase."""
    return jnp.ones(9) * 0.01, jnp.ones(9) * 0.005


@pytest.fixture(scope="module")
def result_3ph(small_3ph_topo, balanced_load):
    P, Q = balanced_load
    return bfs_3phase_power_flow(small_3ph_topo, P, Q)


# ==================== L0: JAX Contract ====================

class TestBFS3Ph_L0_JAX:

    def test_jit_compile(self, small_3ph_topo, balanced_load):
        P, Q = balanced_load
        jit_fn = jax.jit(lambda t, p, q: bfs_3phase_power_flow(t, p, q, 100, 1e-6))
        r = jit_fn(small_3ph_topo, P, Q)
        assert r.v_mag.shape == (9,)
        assert bool(r.converged)

    def test_vmap_batch(self, small_3ph_topo, balanced_load):
        P, Q = balanced_load
        batch_topo = tu.tree_map(lambda x: jnp.stack([x, x, x]), small_3ph_topo)
        batch_P = jnp.stack([P, P * 1.5, P * 0.5])
        batch_Q = jnp.stack([Q, Q * 1.5, Q * 0.5])
        vfn = jax.vmap(lambda t, p, q: bfs_3phase_power_flow(t, p, q, 100, 1e-6))
        results = vfn(batch_topo, batch_P, batch_Q)
        assert results.v_mag.shape == (3, 9)
        assert jnp.all(results.converged)

    def test_result_pytree(self, result_3ph):
        leaves = tu.tree_leaves(result_3ph)
        assert len(leaves) == 9  # BFS3PhResult has 9 fields

    def test_topo_pytree(self, small_3ph_topo):
        flat, td = tu.tree_flatten(small_3ph_topo)
        rebuilt = tu.tree_unflatten(td, flat)
        assert rebuilt.n_nodes == small_3ph_topo.n_nodes

    def test_make_jaxpr(self, small_3ph_topo, balanced_load):
        P, Q = balanced_load
        jaxpr = jax.make_jaxpr(
            lambda t, p, q: bfs_3phase_power_flow(t, p, q, 100, 1e-6)
        )(small_3ph_topo, P, Q)
        assert jaxpr is not None


# ==================== L1: Physical Correctness ====================

class TestBFS3Ph_L1_Physics:

    def test_convergence(self, result_3ph):
        assert bool(result_3ph.converged)
        assert int(result_3ph.iterations) <= 20

    def test_voltage_reasonable(self, result_3ph):
        """All voltages in [0.8, 1.1]."""
        assert jnp.all(result_3ph.v_mag > 0.8)
        assert jnp.all(result_3ph.v_mag < 1.1)

    def test_balanced_load_balanced_voltage(self, result_3ph):
        """With balanced load, per-phase voltages at each bus should be similar."""
        v_mat = result_3ph.v_mag.reshape(-1, 3)  # (n_lines, 3)
        for i in range(v_mat.shape[0]):
            phase_range = float(v_mat[i].max() - v_mat[i].min())
            assert phase_range < 0.01, f"Bus {i} phase imbalance = {phase_range}"

    def test_zero_load_flat(self, small_3ph_topo):
        """Zero load → flat voltage profile."""
        P_zero = jnp.zeros(9)
        Q_zero = jnp.zeros(9)
        r = bfs_3phase_power_flow(small_3ph_topo, P_zero, Q_zero)
        assert bool(r.converged)
        assert jnp.allclose(r.v_mag, 1.0, atol=1e-4)

    def test_heavy_load_lower_voltage(self, small_3ph_topo, balanced_load):
        """Heavier load → lower voltages."""
        P1, Q1 = balanced_load
        r1 = bfs_3phase_power_flow(small_3ph_topo, P1, Q1)
        r2 = bfs_3phase_power_flow(small_3ph_topo, P1 * 3.0, Q1 * 3.0)
        assert float(r2.v_mag.min()) < float(r1.v_mag.min())

    def test_voltage_drop_from_root(self, result_3ph):
        """Voltage should decrease from reference to far end."""
        v_mat = result_3ph.v_mag.reshape(-1, 3)
        # Average per bus
        v_avg = v_mat.mean(axis=1)
        assert float(v_avg[-1]) < float(v_avg[0])


# ==================== L2: Robustness ====================

class TestBFS3Ph_L2_Robustness:

    def test_branch_current_magnitude(self, small_3ph_topo, balanced_load):
        """I_branch: valid shape, no NaNs, nonzero under load."""
        P, Q = balanced_load
        result = bfs_3phase_power_flow(small_3ph_topo, P, Q)
        I_mag = jnp.sqrt(result.I_branch_real ** 2 + result.I_branch_imag ** 2)
        assert I_mag.shape == (9,)
        assert jnp.all(I_mag >= 0.0)
        assert float(jnp.max(I_mag)) > 1e-6

    def test_branch_current_vs_P_branch_approximation(self, small_3ph_topo):
        """Compare |I_branch| vs P_branch-based |S|; thermal limits should use |I_branch|.

        S_from_I = |I_branch| * |V| — branch current × receiving-end |V| (lower bound on |S|).
        S_from_P = sqrt(P_branch^2 + Q_branch^2) — |S| from P,Q at receiving bus; misses I²R.

        The gap is I²R slack; ampacity checks should use I_mag * line_rating, not S_from_P.
        This test only checks same order of magnitude (max ratio in [0.5, 2.0]).
        """
        P = jnp.ones(9) * 0.05
        Q = jnp.ones(9) * 0.02
        result = bfs_3phase_power_flow(small_3ph_topo, P, Q)

        I_mag = jnp.sqrt(result.I_branch_real ** 2 + result.I_branch_imag ** 2)
        S_from_I = I_mag * result.v_mag
        S_from_P = jnp.sqrt(result.P_branch ** 2 + result.Q_branch ** 2)

        max_I = float(jnp.max(S_from_I))
        max_P = float(jnp.max(S_from_P))
        assert max_P > 1e-9, "S_from_P should not be all zero"
        ratio = max_I / max_P
        assert 0.5 <= ratio <= 2.0, (
            f"max(S_from_I)/max(S_from_P) = {ratio:.3f}, expected in [0.5, 2.0]"
        )

    def test_non_zero_ref_bus(self):
        """Regression: ref_bus != 0 — far-end vs root slack both converge; profiles differ."""
        n_nodes = 5
        from_nodes = np.array([0, 1, 2, 3])
        to_nodes = np.array([1, 2, 3, 4])
        Z = np.zeros((4, 3, 3), dtype=complex)
        for i in range(4):
            Z[i] = (0.05 + 0.1j) * np.eye(3)

        topo_ref0 = build_3phase_topology(n_nodes, from_nodes, to_nodes, Z, ref_bus=0)
        topo_ref4 = build_3phase_topology(n_nodes, from_nodes, to_nodes, Z, ref_bus=4)

        P = jnp.ones(12) * 0.02
        Q = jnp.ones(12) * 0.01

        r0 = bfs_3phase_power_flow(topo_ref0, P, Q)
        r4 = bfs_3phase_power_flow(topo_ref4, P, Q)

        assert bool(r0.converged), "expected convergence with ref_bus=0"
        assert bool(r4.converged), "expected convergence with ref_bus=4"

        assert jnp.all(r0.v_mag > 0.7) and jnp.all(r0.v_mag < 1.3)
        assert jnp.all(r4.v_mag > 0.7) and jnp.all(r4.v_mag < 1.3)

        # Different slack ends → different voltage profiles (opposing flow sense).
        assert not jnp.allclose(r0.v_mag, r4.v_mag, atol=1e-3), (
            "v_mag should differ between ref_bus=0 and ref_bus=4"
        )

    def test_strongly_unbalanced_load(self, small_3ph_topo):
        """Strongly unbalanced load: solver runs; phase a voltage below b/c.

        Load: phase a = 0.10 p.u., b = c = 0.01 p.u. (10:1).
        Expect converged=True and per-line max−min phase voltage > 0.01.
        """
        # P shape (9,) = (3 lines × 3 phases); per line [a=0.10, b=0.01, c=0.01]
        P = jnp.array([0.10, 0.01, 0.01] * 3)
        Q = P * 0.5
        result = bfs_3phase_power_flow(small_3ph_topo, P, Q)

        assert bool(result.converged), "expected convergence under strong imbalance"

        v_mat = result.v_mag.reshape(-1, 3)  # (n_lines, 3_phases)
        for line_idx in range(v_mat.shape[0]):
            phase_range = float(v_mat[line_idx].max() - v_mat[line_idx].min())
            assert phase_range > 0.01, (
                f"line {line_idx}: phase voltage spread = {phase_range:.4f}, "
                f"expected > 0.01 under strong imbalance"
            )
