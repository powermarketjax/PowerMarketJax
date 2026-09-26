"""L2 review of the low-rank ``kkt`` route (2026-09-16): what
`test_lowrank_l2.py` leaves untested.

1. **K > 1 end to end.**  `test_lowrank_l2.py` runs every case at K = 1, where
   two layout mistakes are invisible: the capacity weights of the relaxation
   are sliced as ``(T, n_units, K)`` and a unit's ``K`` segment columns are
   gathered as one block -- both collapse to the K = 1 case whatever order the
   rows are actually in.  The same same-LP comparison (dense vs low-rank on
   the monitored-lines LP) is run here at K = 2 for the three cases, for both
   operators, to the tolerances `test_clearing_l2.py` derived.
   Measured 2026-09-16 (CPU, K = 2, T = 2): see the docstring of each test.

2. **The refinement is load-bearing.**  `kkt_lowrank.N_REFINE` is what turns
   the not-backward-stable Schur solve into one that matches the dense route;
   a review that only checks the shipped constant cannot tell whether the L2
   comparison would notice its loss.  Building the operators with
   ``N_REFINE = 0`` (module attribute, read at trace time) must move the
   813nem result past the L2 tolerance: measured 2026-09-16 on the 813nem
   T = 2 K = 1 LP, plain Schur solve vs dense: see `test_refinement_is_load_bearing`.
   If this test ever passes with the injection, the same-LP tolerance has
   stopped being a check on the solver.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.day_ahead import make_clearing, segment_costs
from powermarketjax.envs.day_ahead import kkt_lowrank
from powermarketjax.envs.day_ahead.relax import make_relax, round_commitment

from . import reference
from .test_clearing_l2 import ATOL_LMP, ATOL_MW, RTOL_LMP
from .test_lowrank_l2 import (CASE_NAMES, CASE_SCALE, _demand_for, _feasible_u_ones,
                              _monitored_lines, _reference_clear, x64)  # noqa: F401
from .test_lowrank_l2 import PC

T = 2
K2 = 2


def _clear_inputs(case_name, K):
    case = load_case(case_name)
    cap_scale, ramp_scale = CASE_SCALE[case_name]
    demand = _demand_for(case_name, case)
    monitored = _monitored_lines(case_name, np.asarray(case.line_cap).shape[0])
    _, cost = segment_costs(case, K)
    # a K-segment offer, non-decreasing in the segment index as `segment_costs`
    # returns it; constant over the periods
    offer = np.repeat(cost[:, :, None], T, axis=2)
    p_min = np.asarray(case.unit_p_min, np.float64)
    u = np.broadcast_to(_feasible_u_ones(case, demand)[:, None], (len(p_min), T)).copy()
    p_init = p_min * u[:, 0]
    dem = np.full(T, demand)
    return case, cap_scale, ramp_scale, monitored, offer, u, p_init, dem


def _assert_close(new, other, label):
    np.testing.assert_allclose(np.asarray(new["lmp"]), np.asarray(other["lmp"]),
                               rtol=RTOL_LMP, atol=ATOL_LMP, err_msg=f"lmp ({label})")
    np.testing.assert_allclose(np.asarray(new["award"]), np.asarray(other["award"]),
                               rtol=1e-6, atol=ATOL_MW, err_msg=f"award ({label})")
    np.testing.assert_allclose(np.asarray(new["shed"]).sum(1), np.asarray(other["shed"]).sum(1),
                               rtol=1e-6, atol=1e-4, err_msg=f"shed total ({label})")


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_clear_k2_lowrank_matches_dense_and_reference(case_name):
    """Same monitored-lines LP at K = 2, low-rank vs dense vs the numpy
    reference.  A segment-order or capacity-reshape mistake shows up here and
    not at K = 1."""
    case, cap_scale, ramp_scale, monitored, offer, u, p_init, dem = _clear_inputs(case_name, K2)
    n_lines_all = np.asarray(case.line_cap).shape[0]
    dense_clear, _ = make_clearing(case, T, n_segments=K2, cap_scale=cap_scale,
                                   ramp_scale=ramp_scale, monitored_lines=monitored, kkt="dense")
    new_clear, spec = make_clearing(case, T, n_segments=K2, cap_scale=cap_scale,
                                    ramp_scale=ramp_scale, monitored_lines=monitored)
    assert spec["K"] == K2
    args = (jnp.asarray(offer), jnp.asarray(u), jnp.asarray(dem), jnp.asarray(p_init))
    dense = jax.jit(dense_clear)(*args)
    new = jax.jit(new_clear)(*args)
    assert float(dense["mu"]) < 1e-8 and float(new["mu"]) < 1e-8
    _assert_close(new, dense, f"{case_name} K=2 new vs dense")
    want = _reference_clear(case, monitored, n_lines_all, offer, u, dem, p_init, cap_scale, ramp_scale)
    assert want["mu"] < 1e-8
    _assert_close(new, want, f"{case_name} K=2 new vs numpy reference")


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_relax_k2_lowrank_matches_dense(case_name):
    """Same monitored-lines relaxation LP at K = 2, low-rank vs dense."""
    case = load_case(case_name)
    cap_scale, ramp_scale = CASE_SCALE[case_name]
    demand = _demand_for(case_name, case)
    monitored = _monitored_lines(case_name, np.asarray(case.line_cap).shape[0])
    boundary = PC.first_boundary(case, demand, cap_scale=cap_scale, ramp_scale=ramp_scale,
                                 K=K2, delta_h=1.0)
    _, cost = segment_costs(case, K2)
    offer = np.repeat(cost[:, :, None], T, axis=2)
    dense_relax, _ = make_relax(case, T, n_segments=K2, cap_scale=cap_scale, ramp_scale=ramp_scale,
                                monitored_lines=monitored, kkt="dense")
    new_relax, spec = make_relax(case, T, n_segments=K2, cap_scale=cap_scale, ramp_scale=ramp_scale,
                                 monitored_lines=monitored)
    assert spec["K"] == K2
    args = (jnp.asarray(offer), jnp.asarray(np.full(T, demand)), jnp.asarray(boundary["p_init"]),
            jnp.asarray(boundary["u_prev"]), jnp.asarray(boundary["up_time"].astype(float)),
            jnp.asarray(boundary["down_time"].astype(float)))
    dense = jax.jit(dense_relax)(*args)
    new = jax.jit(new_relax)(*args)
    assert float(dense["mu"]) < 1e-6 and float(new["mu"]) < 1e-6
    np.testing.assert_allclose(float(new["obj"]), float(dense["obj"]), rtol=1e-9)
    np.testing.assert_allclose(np.asarray(new["u"]), np.asarray(dense["u"]), atol=1e-7)
    np.testing.assert_allclose(np.asarray(new["p"]), np.asarray(dense["p"]), atol=1e-6)
    assert np.array_equal(np.asarray(round_commitment(new["u"])),
                          np.asarray(round_commitment(dense["u"])))


def test_refinement_is_load_bearing(monkeypatch):
    """Injection: with ``N_REFINE = 0`` the plain Schur solve on the 813nem
    T = 2 K = 1 LP must land outside the same-LP tolerance against the dense
    route on at least one of lmp / award, otherwise `test_lowrank_l2.py`'s
    tolerance is not a check on the solver's numerics."""
    case, cap_scale, ramp_scale, monitored, offer, u, p_init, dem = _clear_inputs("813nem", 1)
    dense_clear, _ = make_clearing(case, T, n_segments=1, cap_scale=cap_scale,
                                   ramp_scale=ramp_scale, monitored_lines=monitored, kkt="dense")
    monkeypatch.setattr(kkt_lowrank, "N_REFINE", 0)
    plain_clear, _ = make_clearing(case, T, n_segments=1, cap_scale=cap_scale,
                                   ramp_scale=ramp_scale, monitored_lines=monitored)
    args = (jnp.asarray(offer), jnp.asarray(u), jnp.asarray(dem), jnp.asarray(p_init))
    dense = jax.jit(dense_clear)(*args)
    plain = jax.jit(plain_clear)(*args)
    d_lmp = float(np.abs(np.asarray(plain["lmp"]) - np.asarray(dense["lmp"])).max())
    d_award = float(np.abs(np.asarray(plain["award"]) - np.asarray(dense["award"])).max())
    print(f"\nN_REFINE=0 vs dense, 813nem T=2 K=1: max|dlmp| {d_lmp:.3e} $/MWh, "
          f"max|daward| {d_award:.3e} MW, mu plain {float(plain['mu']):.2e} dense {float(dense['mu']):.2e}")
    assert d_lmp > ATOL_LMP or d_award > 1e-3 or not np.isfinite(float(plain["mu"])), (
        "the unrefined Schur solve is within the L2 tolerance of the dense route; "
        "the same-LP comparison cannot see the refinement any more")


def test_default_path_bit_identical_at_same_thread_count_813nem():
    """Positive control on the case the keyword exists for: on 813nem the
    default path (no keyword) and ``monitored_lines=None`` must be
    bit-identical **within one process**, i.e. at one thread count.  Measured
    2026-09-16: across processes the
    813nem T=2 outputs are bit-identical at 4 cores and differ at 16 against
    the same pre-change baseline, so the byte-identity of the default path is
    a same-thread-count property on this case; 29gb and 73rts agree at both.
    """
    case, cap_scale, ramp_scale, _, offer, u, p_init, dem = _clear_inputs("813nem", 1)
    no_kw, _ = make_clearing(case, T, n_segments=1, cap_scale=cap_scale, ramp_scale=ramp_scale)
    explicit, _ = make_clearing(case, T, n_segments=1, cap_scale=cap_scale, ramp_scale=ramp_scale,
                                monitored_lines=None)
    args = (jnp.asarray(offer), jnp.asarray(u), jnp.asarray(dem), jnp.asarray(p_init))
    a, b = jax.jit(no_kw)(*args), jax.jit(explicit)(*args)
    for key in a:
        assert np.array_equal(np.asarray(a[key]), np.asarray(b[key])), key
