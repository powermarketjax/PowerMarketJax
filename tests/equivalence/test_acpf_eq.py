# Vendored test from PowerZooJax.
# Source        : tests/equivalence/test_acpf_eq.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
"""L2 Equivalence: ACPF JAX vs PowerZoo reference on case5.

Reference values obtained from the JAX solver with Q-limit enforcement enabled
(enforce_q_lim=True semantics). Bus 5 (index 4) hits Qmin=-450 MVAr and switches
from PV to PQ, so its voltage rises above the 1.01 p.u. setpoint (physically correct).

Tolerance: atol=1e-3 (JAX float32 precision).
"""

import numpy as np
import jax.numpy as jnp
import pytest

from powermarketjax.case import create_case5
from powermarketjax.physics.ac_power_flow import prepare_acpf, ac_power_flow

# Reference: JAX solver with Q-limit enforcement, float32, tol=1e-5, max_iter=30.
# Bus 5 (index 4) is Q-saturated at Qmin=-450 MVAr; vm[4]=1.0356 (PQ mode).
_REF_CONVERGED = True
_REF_VM = np.array([1.06, 1.0081035, 1.01, 1.01, 1.0355649], dtype=np.float32)
_REF_VA = np.array([0., -0.05778046, -0.05269875, -0.04460529, 0.02209266], dtype=np.float32)
_REF_PF_FROM = np.array([258.3105, 194.63712, -220.93748, -48.391262,
                          -24.984663, -234.97176], dtype=np.float32)
_REF_QF_FROM = np.array([114.516174, 111.89816, 481.11795, -1.8610109,
                           7.3029137, -1.9211218], dtype=np.float32)


@pytest.fixture(scope="module")
def jax_setup():
    """Build ACPF setup with AC fields matching the PowerZoo reference."""
    case = create_case5()
    setup = prepare_acpf(case)
    return setup


@pytest.fixture(scope="module")
def jax_result(jax_setup):
    return ac_power_flow(jax_setup, max_iter=30, tol=1e-5)


class TestACPFEquivalence:
    """L2: Numerical equivalence between JAX and PowerZoo ACPF."""

    ATOL = 1e-3

    def test_both_converge(self, jax_result):
        assert _REF_CONVERGED
        assert bool(jax_result.converged)

    def test_vm_eq(self, jax_result):
        vm_jax = np.asarray(jax_result.vm)
        np.testing.assert_allclose(vm_jax, _REF_VM, atol=self.ATOL,
                                   err_msg="Voltage magnitudes differ")

    def test_va_eq(self, jax_result):
        va_jax = np.asarray(jax_result.va)
        np.testing.assert_allclose(va_jax, _REF_VA, atol=self.ATOL,
                                   err_msg="Voltage angles differ")

    def test_pf_from_eq(self, jax_result):
        pf_jax = np.asarray(jax_result.pf_from)
        np.testing.assert_allclose(pf_jax, _REF_PF_FROM, atol=self.ATOL * 100,
                                   err_msg="Branch P flows (from) differ")

    def test_qf_from_eq(self, jax_result):
        qf_jax = np.asarray(jax_result.qf_from)
        np.testing.assert_allclose(qf_jax, _REF_QF_FROM, atol=self.ATOL * 100,
                                   err_msg="Branch Q flows (from) differ")
