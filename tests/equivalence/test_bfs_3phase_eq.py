# Vendored test from PowerZooJax.
# Source        : tests/equivalence/test_bfs_3phase_eq.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# NOT verbatim -- restructured; the assertions themselves are unchanged.
#   Upstream imported the sibling PowerZoo repo at module scope and computed
#   the reference live, skipping the whole file when PowerZoo was absent.
#   This repository takes no runtime dependency on a sibling repo, and here that
#   skip would have been permanent: three tests that could never run.
#
#   The reference is therefore FROZEN below -- the same shape as the other two
#   files in this directory (test_bfs_eq.py, test_acpf_eq.py), and the rule
#   for offline references.
#
#   Generated 2026-08-05 by running PowerZoo's
#   powerzoo.envs.grid.cal_pf_dist_3phase.run_3phase_bfs_power_flow on the
#   4-bus case defined below, from PowerZooJax/PowerZoo (conda env
#   `market_env`, numpy 1.26.4, float64). Regenerating is an explicit manual
#   step -- see _REFERENCE_PROVENANCE.
# Upstream licence: MIT, Copyright (c) 2026 PowerZooJax Contributors.
"""L2 Equivalence: 3-phase BFS (JAX, float32) vs a frozen PowerZoo reference.

Tolerance: atol=1e-2 (float32 + different DLF construction), unchanged from
upstream.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from powermarketjax.physics.bfs_3phase_power_flow import (
    build_3phase_topology,
    bfs_3phase_power_flow,
)

_REFERENCE_PROVENANCE = """\
Reference producer -- run manually to regenerate; never part of CI:

    import sys, numpy as np
    sys.path.insert(0, "<path to>/PowerZooJax/PowerZoo")
    from powerzoo.envs.grid.cal_pf_dist_3phase import (
        build_3phase_topology as pz_build, run_3phase_bfs_power_flow as pz_run,
    )
    n, fn, tn = 4, np.array([0, 1, 2]), np.array([1, 2, 3])
    Z = np.zeros((3, 3, 3), dtype=complex)
    for i in range(3):
        Z[i] = (0.1 + 0.2j) * np.eye(3)
    P, Q = np.ones((3, 3)) * 0.01, np.ones((3, 3)) * 0.005
    res = pz_run(pz_build(n, fn, tn, Z, ref_bus=0), P.flatten(), Q.flatten())
    print(res["converged"], res["V_mag"].tolist(), res["P_branch"].tolist())
"""

# --- Frozen PowerZoo reference (float64) ---------------------------------
PZ_CONVERGED = True
PZ_V_MAG = np.array([
    0.9939208860719604, 0.9939208860719602, 0.9939208860719604,
    0.9898688558387959, 0.9898688558387960, 0.9898688558387960,
    0.9878430781831344, 0.9878430781831344, 0.9878430781831344,
])
PZ_P_BRANCH = np.array([
    -0.030178602116017007, -0.030178602116017007, -0.030178602116017007,
    -0.020063942944107002, -0.020063942944106995, -0.020063942944107002,
    -0.010012809555993056, -0.010012809555993055, -0.010012809555993056,
])


@pytest.fixture(autouse=True, scope="module")
def _force_float32_mode():
    """Keep this float32 equivalence test isolated from ACOPF's global x64 side effects."""
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", False)


@pytest.fixture(scope="module")
def small_case():
    """4-bus test case with balanced impedances."""
    n_nodes = 4
    from_nodes = np.array([0, 1, 2])
    to_nodes = np.array([1, 2, 3])
    Z = np.zeros((3, 3, 3), dtype=complex)
    for i in range(3):
        Z[i] = (0.1 + 0.2j) * np.eye(3)
    P = np.ones((3, 3)) * 0.01  # 3 buses * 3 phases
    Q = np.ones((3, 3)) * 0.005
    return n_nodes, from_nodes, to_nodes, Z, P, Q


@pytest.fixture(scope="module")
def jax_result(small_case):
    n_nodes, from_nodes, to_nodes, Z, P, Q = small_case
    topo = build_3phase_topology(n_nodes, from_nodes, to_nodes, Z, ref_bus=0)
    P_flat = jnp.array(P.flatten())
    Q_flat = jnp.array(Q.flatten())
    return bfs_3phase_power_flow(topo, P_flat, Q_flat)


class TestBFS3PhEquivalence:
    ATOL = 1e-2

    def test_both_converge(self, jax_result):
        assert PZ_CONVERGED
        assert bool(jax_result.converged)

    def test_v_mag_eq(self, jax_result):
        v_jax = np.asarray(jax_result.v_mag)
        np.testing.assert_allclose(v_jax, PZ_V_MAG, atol=self.ATOL,
                                   err_msg="3-phase voltage magnitudes differ")

    def test_p_branch_eq(self, jax_result):
        p_jax = np.asarray(jax_result.P_branch)
        np.testing.assert_allclose(p_jax, PZ_P_BRANCH, atol=self.ATOL,
                                   err_msg="3-phase branch P flows differ")
