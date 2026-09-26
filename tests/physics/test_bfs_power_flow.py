# Vendored test from PowerZooJax.
# Source        : tests/grid/test_bfs_power_flow.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# NOT verbatim. Beyond the package rename (powerzoojax -> powermarketjax, in
# both import statements and path strings): on 2026-08-05 the expected
# BFSResult leaf count in test_result_pytree went 8 -> 9, because BFSResult
# gained ``floor_active``.
# No other assertion changed; every ``converged`` assertion in this file is at
# normal loading and still holds under the stricter flag.
"""Tests for powermarketjax/physics/bfs_power_flow.py — L0 JAX + L1 physics.

L0: JIT, vmap, pytree stability for BFS solver.
L1: Convergence, voltage profile, losses, zero-load trivial case.
"""

import jax
import jax.numpy as jnp
import jax.tree_util as tu
import pytest

from powermarketjax.case import create_case33bw
from powermarketjax.physics.bfs_power_flow import (
    BFSTopoData,
    BFSResult,
    prepare_bfs,
    bfs_power_flow,
    backward_sweep,
    forward_sweep,
)


@pytest.fixture(scope="module")
def case33():
    return create_case33bw()


@pytest.fixture(scope="module")
def topo33(case33):
    return prepare_bfs(case33)


@pytest.fixture(scope="module")
def result33(topo33, case33):
    p_load = case33.node_pd / case33.base_mva
    q_load = case33.node_qd / case33.base_mva
    return bfs_power_flow(topo33, p_load, q_load)


# ==================== L0: JAX Contract ====================

class TestBFS_L0_JAX:

    def test_jit_compile(self, topo33, case33):
        p = case33.node_pd / case33.base_mva
        q = case33.node_qd / case33.base_mva
        jit_fn = jax.jit(bfs_power_flow, static_argnums=(3, 4, 5))
        r = jit_fn(topo33, p, q, 1.0, 100, 1e-6)
        assert r.v_mag.shape == (case33.n_nodes,)
        assert bool(r.converged)

    def test_vmap_batch(self, topo33, case33):
        p = case33.node_pd / case33.base_mva
        q = case33.node_qd / case33.base_mva
        batch_topo = tu.tree_map(lambda x: jnp.stack([x, x, x]), topo33)
        batch_p = jnp.stack([p, p * 1.2, p * 0.8])
        batch_q = jnp.stack([q, q * 1.2, q * 0.8])
        vfn = jax.vmap(lambda t, pl, ql: bfs_power_flow(t, pl, ql, 1.0, 100, 1e-6))
        results = vfn(batch_topo, batch_p, batch_q)
        assert results.v_mag.shape == (3, case33.n_nodes)
        assert jnp.all(results.converged)

    def test_result_pytree(self, result33):
        # Was 8 upstream; 9 since BFSResult gained ``floor_active`` on
        # 2026-08-05 (see the note at the top of this file).
        leaves = tu.tree_leaves(result33)
        assert len(leaves) == 9

    def test_topo_pytree(self, topo33):
        flat, td = tu.tree_flatten(topo33)
        rebuilt = tu.tree_unflatten(td, flat)
        assert rebuilt.n_nodes == topo33.n_nodes

    def test_backward_sweep_jit(self, topo33, case33):
        p = case33.node_pd / case33.base_mva
        q = case33.node_qd / case33.base_mva
        v_sq = jnp.ones(case33.n_nodes)
        jit_bs = jax.jit(backward_sweep)
        pb, qb = jit_bs(topo33, p, q, v_sq)
        assert pb.shape == (topo33.n_lines,)

    def test_forward_sweep_jit(self, topo33, case33):
        p_br = jnp.zeros(topo33.n_lines)
        q_br = jnp.zeros(topo33.n_lines)
        v_sq = jnp.ones(case33.n_nodes)
        jit_fs = jax.jit(forward_sweep, static_argnums=(4,))
        v_new = jit_fs(topo33, p_br, q_br, v_sq, 1.0)
        assert v_new.shape == (case33.n_nodes,)

    def test_make_jaxpr(self, topo33, case33):
        p = case33.node_pd / case33.base_mva
        q = case33.node_qd / case33.base_mva
        jaxpr = jax.make_jaxpr(
            lambda t, pl, ql: bfs_power_flow(t, pl, ql, 1.0, 100, 1e-6)
        )(topo33, p, q)
        assert jaxpr is not None


# ==================== L1: Physical Correctness ====================

class TestBFS_L1_Physics:

    def test_convergence(self, result33):
        assert bool(result33.converged)
        assert int(result33.iterations) <= 20

    def test_slack_bus_voltage(self, result33):
        """Slack bus (bus 0) voltage should be 1.0."""
        assert abs(float(result33.v_mag[0]) - 1.0) < 1e-4

    def test_voltage_drop_monotonic(self, result33):
        """Voltage should generally decrease from slack to far ends."""
        assert float(result33.v_mag.min()) < float(result33.v_mag[0])

    def test_voltage_reasonable(self, result33):
        """All voltages in [0.8, 1.1] for normal loading."""
        assert jnp.all(result33.v_mag > 0.8)
        assert jnp.all(result33.v_mag < 1.1)

    def test_total_loss_positive(self, result33):
        assert float(result33.p_loss.sum()) > 0.0
        assert float(result33.q_loss.sum()) > 0.0

    def test_zero_load_flat(self, topo33, case33):
        """Zero load → flat voltage profile."""
        p_zero = jnp.zeros(case33.n_nodes)
        q_zero = jnp.zeros(case33.n_nodes)
        r = bfs_power_flow(topo33, p_zero, q_zero, 1.0, 100, 1e-6)
        assert bool(r.converged)
        assert jnp.allclose(r.v_mag, 1.0, atol=1e-4)
        assert jnp.allclose(r.p_branch, 0.0, atol=1e-6)

    def test_heavy_load_lower_voltage(self, topo33, case33):
        """Heavier load → lower voltages at far end."""
        p1 = case33.node_pd / case33.base_mva
        q1 = case33.node_qd / case33.base_mva
        r1 = bfs_power_flow(topo33, p1, q1, 1.0, 100, 1e-6)

        p2 = p1 * 2.0
        q2 = q1 * 2.0
        r2 = bfs_power_flow(topo33, p2, q2, 1.0, 100, 1e-6)

        assert float(r2.v_mag.min()) < float(r1.v_mag.min())

    def test_branch_flows_shape(self, result33, topo33):
        assert result33.p_branch.shape == (topo33.n_lines,)
        assert result33.q_branch.shape == (topo33.n_lines,)
        assert result33.p_loss.shape == (topo33.n_lines,)
