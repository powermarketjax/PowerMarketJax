# Vendored test from PowerZooJax.
# Source        : tests/equivalence/test_bfs_eq.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
"""L2 Equivalence: BFS JAX vs PowerZoo reference on case33bw.

Reference values obtained from PowerZoo run_bfs_power_flow() using the
IEEE 33-bus standard impedance data (32 active branches, no tie switches).

Tolerance: atol=1e-3 (JAX float32 vs PowerZoo float64 reference).
"""

import numpy as np
import jax.numpy as jnp
import pytest

from powermarketjax.case import create_case33bw
from powermarketjax.physics.bfs_power_flow import prepare_bfs, bfs_power_flow

# PowerZoo reference (float64, 33-bus, 32 active branches)
_REF_CONVERGED = True

_REF_V_MAG = np.array([
    1.0,        0.99704452, 0.98301121, 0.97555899, 0.96818628,
    0.94983502, 0.94635351, 0.94151371, 0.93525007, 0.92943867,
    0.92857902, 0.92708005, 0.92096949, 0.91870336, 0.9172914,
    0.9159238,  0.91389707, 0.91329013, 0.99651618, 0.99293875,
    0.99223426, 0.99159685, 0.97942648, 0.97275673, 0.96943208,
    0.94790834, 0.94534787, 0.93392127, 0.92570762, 0.92215128,
    0.91799125, 0.91707603, 0.91679245,
])

_REF_P_BRANCH = np.array([
    0.03901482, 0.03429371, 0.02353757, 0.02215978, 0.02139106,
    0.01094568, 0.00892735, 0.00688108, 0.00624108, 0.00560665,
    0.00515124, 0.00454264, 0.00391658, 0.00270937, 0.00210583,
    0.00150303, 0.00090053, 0.00361129, 0.00270969, 0.00180144,
    0.00090044, 0.00939444, 0.00846334, 0.00421278, 0.00949576,
    0.00887121, 0.00823973, 0.00753252, 0.00625647, 0.00421802,
    0.00270226, 0.00060013,
])

_REF_Q_BRANCH = np.array([
    0.02425106, 0.02198437, 0.01677803, 0.01588749, 0.01550155,
    0.00527259, 0.00421201, 0.00319672, 0.00296798, 0.00274358,
    0.00244179, 0.00208894, 0.00171844, 0.00090895, 0.0008058,
    0.00060376, 0.00040042, 0.00161071, 0.00120919, 0.00080175,
    0.00040058, 0.00457117, 0.00404993, 0.00201,    0.00972716,
    0.00946465, 0.00919862, 0.0089041,  0.00813785, 0.00211826,
    0.00140268, 0.0004002,
])


@pytest.fixture(scope="module")
def jax_result():
    case = create_case33bw()
    topo = prepare_bfs(case)
    p_load = case.node_pd / case.base_mva
    q_load = case.node_qd / case.base_mva
    return bfs_power_flow(topo, p_load, q_load)


class TestBFSEquivalence:
    ATOL = 1e-3

    def test_converge(self, jax_result):
        assert _REF_CONVERGED
        assert bool(jax_result.converged)

    def test_v_mag_eq(self, jax_result):
        v_jax = np.asarray(jax_result.v_mag)
        np.testing.assert_allclose(v_jax, _REF_V_MAG, atol=self.ATOL,
                                   err_msg="Voltage magnitudes differ")

    def test_p_branch_eq(self, jax_result):
        p_jax = np.asarray(jax_result.p_branch)
        np.testing.assert_allclose(p_jax, _REF_P_BRANCH, atol=self.ATOL,
                                   err_msg="Branch P flows differ")

    def test_q_branch_eq(self, jax_result):
        q_jax = np.asarray(jax_result.q_branch)
        np.testing.assert_allclose(q_jax, _REF_Q_BRANCH, atol=self.ATOL,
                                   err_msg="Branch Q flows differ")

    def test_total_loss_eq(self, jax_result):
        p_jax_sum = float(jnp.sum(jax_result.p_branch))
        p_ref_sum = float(np.sum(_REF_P_BRANCH))
        np.testing.assert_allclose(p_jax_sum, p_ref_sum, atol=self.ATOL,
                                   err_msg="Total branch P sum differs")
