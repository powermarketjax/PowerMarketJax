"""`monitored_lines` on the joint clearing: the `case813nem` defect, the fix, and
the reference on the same row set (2026-09-16).

**The defect.**  On `case813nem` every truthful period of this market read
``converged = False``.  Not a wrong solve: `ipm.make_solver` floors every
multiplier at `ipm.SLACK_FLOOR` (1e-14), so a row that never binds still
contributes ``s * 1e-14`` to ``mu = sum(s * lam) / m``, and the 1 271 lines the
case registers at 1e6 MW leave ``s`` at 1e6 on 2 542 rows: a floor of
2 542 * 1e6 * 1e-14 / 5 547 = 4.6e-9 under ``mu``, above `env.MU_TOL` (1e-9),
whatever the dispatch does.  Measured on five truthful periods: ``mu`` 4.59e-9 on every one,
99.8% of it from those rows.

**The fix** is the day-ahead operator's row-set switch
on this operator: ``monitored_lines=rated_lines(case)`` carries the 7 rated
lines and drops the 1 271 placeholders.  The Newton system's order does not
change -- it is a row-count change, not a route change.

**What is asserted, and on which tree it fails.**  On the tree before the
switch existed, `_build` falls back to the no-argument constructor, so every
test below runs on the default rows and the convergence assertions fail
*because of the defect* (``mu`` 4.6e-9 against 1e-9), not because of a missing
keyword.  The characterisation test passes on both trees: it states the defect
and is the negative control for the fix (the placeholder rows put back = the
failure back).

**The frozen point depends on the thread count.**  The merit gate is
discontinuous: a change in floating-point rounding (XLA and BLAS thread
count) moves the step at which ``mu`` first crosses ``freeze_mu`` and the
step at which the first refusal happens, so this operator and the numpy
reference can freeze at different points on the same cell.  Measured
2026-09-17 (cells 0_0 / 0_24 / 0_36 at ``FREEZE_MU``): at 4 cores (where this
file was written) the two
solvers agree to 4e-9 $/MWh on every cell; at 8 cores cell 0_0 freezes on
this operator at ``mu`` 2.6e-7 / dual residual 9.3e-4 against the
reference's 6.5e-9 / 1.4e-8, |dlmp| 1.7e-3 $/MWh, and cell 0_24 reads
|dlmp| 1.4e-3; at 1 core the worst cell is 0_24 at 3.0e-5.  The tolerances
below are therefore derived over the three core counts, and the plain-loop
price test on cell 0_36 -- red at 4 cores, green at 8 -- is a non-strict
xfail.  A run of this file at one core count says nothing about another.
The gate only exposes this: the same cells measured under a
data-dependent stop (`ipm.make_solver(stop_tol=...)`, no merit gate) and the
Newton path already differs by core count before any gate is crossed (cell
0_0 passes the gate at step 35 on 1 core and step 60 on 4), so the sensitivity is this LP's, not the gate's.

The fixture holds the raw inputs of `clear` for four periods of the truthful
813nem replay (evaluation days of the rated-lines day-ahead position); nothing
derived is stored.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.ancillary.clearing import (MAX_ITER, REG_COEF,
                                                    make_clearing)
from powermarketjax.envs.ancillary.env import MU_TOL, make_ancillary_env
from powermarketjax.envs.day_ahead.clearing import rated_lines
from powermarketjax.solvers import ipm
from tests.envs.ancillary.reference import clear_reference
from tests.envs.ancillary.test_clearing_l2 import (ATOL, RESERVE_MW_ATOL,
                                                   Z_RTOL, _clean)

CASE = "813nem"
THETA = (1.0 / 6.0, 0.5)
#: the adopted pair for this case (`run_eval_03.VOLR_BY_CASE`)
VOLR, PI_SCALE = 147.0, 29.4
BETA = (0.05, 0.05)
DELTA = 0.5
CAP_SCALE, RAMP_SCALE = 1.0, 1.0
DUAL_START = "cost_norm"
FIXTURE = (Path(__file__).resolve().parents[2] / "fixtures"
           / "ge05_03_cells_813nem.npz")
#: The cells the L2 runs.  Three of the four the fixture holds (0_12 is the
#: tied-units cell, kept for the freeze table; 0_36 has a binding reserve price): a reference solve on this case is
#: a 1 269-order dense factorisation per Newton step, tens of seconds per cell.
L2_CELLS = ("0_0", "0_24", "0_36")


@pytest.fixture(scope="module")
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _cells():
    fx = np.load(FIXTURE, allow_pickle=True)
    names = [str(s) for s in fx["cells"]]
    out = {}
    for c in names:
        out[c] = tuple(np.asarray(fx[f"{c}_{k}"], np.float64)
                       for k in ("offer", "offer_res", "u", "demand", "d_res", "p_prev"))
    return out


def _build(case, monitored):
    """`make_clearing` with the row set, or without it on a tree that has no
    such keyword -- so that on that tree the tests below fail on the defect.

    Pinned to the dense route (``kkt="dense"``) since 2026-09-17:
    every number in this file -- the frozen points, the tolerances derived
    over 1 / 4 / 8 cores, the numpy reference -- was taken on the dense
    factorisation, and this file is about the row set and the merit gate,
    not about the route.  With the keyword absent a proper subset now takes
    the low-rank route by default; that route against the dense one is
    `test_lowrank_cells_l2.py`."""
    kw = dict(n_segments=1, cap_scale=CAP_SCALE, ramp_scale=RAMP_SCALE,
              period_hours=DELTA)
    try:
        return make_clearing(case, THETA, VOLR, monitored_lines=monitored, kkt="dense", **kw)
    except TypeError:
        try:
            return make_clearing(case, THETA, VOLR, monitored_lines=monitored, **kw)
        except TypeError:
            return make_clearing(case, THETA, VOLR, **kw)


def _run(clear, args):
    got = jax.jit(clear)(*(jnp.asarray(a) for a in args))
    return {k: np.asarray(v) for k, v in got.items()}


@pytest.fixture(scope="module")
def case(x64):
    return load_case(CASE)


@pytest.fixture(scope="module")
def rated(case):
    clear, spec = _build(case, rated_lines(case))
    return clear, spec


def test_spec_stamps_the_row_set_and_the_route(case):
    """L0: what the LP carries and how it is solved are on the spec, read by the
    drivers (`evaluation.effective_monitored_stamp`), never off a flag."""
    _, dflt = _build(case, None)
    assert dflt["monitored_lines"] is None
    assert dflt["n_lines"] == 1278 and dflt["n_l"] == 1278
    assert dflt["kkt_route"] == "dense"
    assert dflt["m"] == 2 * 1278 + 3 * 151 + 2 + 2 * dflt["n"] == 5547

    _, r = _build(case, rated_lines(case))
    assert list(np.asarray(r["monitored_lines"])) == [10, 134, 188, 361, 363, 738, 1040]
    assert r["n_lines"] == 1278 and r["n_l"] == 7
    assert r["kkt_route"] == "dense"
    assert r["m"] == 2 * 7 + 3 * 151 + 2 + 2 * r["n"] == 3005
    assert r["n"] == dflt["n"] == 1268, "a row-set change must not touch the columns"

    with pytest.raises(ValueError):
        make_clearing(case, THETA, VOLR, monitored_lines=np.array([1278]))
    with pytest.raises(ValueError):
        make_clearing(case, THETA, VOLR, monitored_lines=np.array([], np.int64))


def test_environment_passes_the_row_set_to_its_clearing(case):
    """The env's `clearing_spec` is where a driver reads the stamp from."""
    *_, spec = make_ancillary_env(case, THETA, VOLR, BETA, PI_SCALE,
                                  cap_scale=CAP_SCALE, ramp_scale=RAMP_SCALE,
                                  period_hours=DELTA, kind="markup", markup_max=2.0,
                                  monitored_lines=rated_lines(case))
    assert spec["clearing_spec"]["m"] == 3005
    # a proper subset takes the low-rank route by the operator's own rule
    # since 2026-09-17; the route is a separate switch from the row set
    assert spec["clearing_spec"]["kkt_route"] == "lowrank+free(151,815)+lu:arrow"
    *_, spec_dense = make_ancillary_env(case, THETA, VOLR, BETA, PI_SCALE,
                                        cap_scale=CAP_SCALE, ramp_scale=RAMP_SCALE,
                                        period_hours=DELTA, kind="markup", markup_max=2.0,
                                        monitored_lines=rated_lines(case), kkt="dense")
    assert spec_dense["clearing_spec"]["m"] == 3005
    assert spec_dense["clearing_spec"]["kkt_route"] == "dense"
    *_, spec = make_ancillary_env(case, THETA, VOLR, BETA, PI_SCALE,
                                  cap_scale=CAP_SCALE, ramp_scale=RAMP_SCALE,
                                  period_hours=DELTA, kind="markup", markup_max=2.0)
    assert spec["clearing_spec"]["monitored_lines"] is None
    assert spec["clearing_spec"]["m"] == 5547


def test_default_rows_read_unconverged_from_the_placeholder_rows(case):
    """The defect, stated: on the default rows ``mu`` sits on the floor the
    2 542 placeholder rows make, above `MU_TOL`, and those rows carry it.
    Passes on both trees; it is the negative control for the fix (put the
    rows back and the failure is back).  Two-sided: the floor is bounded
    from both ends, so a ``mu`` that were large for another reason would
    fail here too."""
    clear, spec = _build(case, None)
    args = _cells()["0_0"]
    got = _run(clear, args)
    n_unrated = int((np.asarray(case.line_cap) >= 1e5).sum())
    floor = 2 * n_unrated * 1e6 * ipm.SLACK_FLOOR / spec["m"]
    assert n_unrated == 1271 and abs(floor - 4.583e-9) < 1e-11
    assert got["mu"] > MU_TOL
    assert 0.95 * floor < got["mu"] < 1.1 * floor, (got["mu"], floor)
    # the split from the returned duals: lam = dual * period_hours, s = F -/+ flow
    PTDF = np.asarray(case.PTDF); F = np.asarray(case.line_cap) * CAP_SCALE
    UB = np.zeros((int(case.n_nodes), spec["n_units"]))
    UB[np.asarray(case.unit_node_idx), np.arange(spec["n_units"])] = 1.0
    d = np.asarray(spec["demand_share"]) * float(args[3])
    flow = PTDF @ (UB @ got["award"] - d + got["shed"])
    unrated = F >= 1e5
    share = ((F - flow) * got["line_dual_up"] * DELTA
             + (F + flow) * got["line_dual_dn"] * DELTA)[unrated].sum() / spec["m"] / got["mu"]
    assert share > 0.9, share
    # and the dispatch is a solution: primal feasibility to the floor
    assert got["primal_residual"] < 1e-6


def _reference(case, spec, args):
    return clear_reference(case, *args, theta=THETA, volr=VOLR, cap_scale=CAP_SCALE,
                           ramp_scale=RAMP_SCALE, period_hours=DELTA, max_iter=MAX_ITER,
                           dual_start=DUAL_START, reg_coef=REG_COEF,
                           monitored_lines=spec["monitored_lines"])


@pytest.mark.parametrize("cell", L2_CELLS)
def test_rated_rows_converge_and_match_the_reference_on_the_same_rows(case, rated, cell):
    """L2 on the fixed operator, for the quantities the market's calibration
    does determine on this case: ``mu`` below `MU_TOL`, the objective, shed,
    reserve shortfall, reserve price and reserve award agree with the numpy
    reference built on the same seven rows (a reference on all rows would
    compare two LPs), to the tolerances of
    `test_clearing_l2`.  The energy price and the per-unit energy award are
    NOT asserted here -- see the next test.  On the tree before the fix this
    runs on the default rows and fails on ``mu``."""
    clear, spec = rated
    args = _cells()[cell]
    got = _run(clear, args)
    assert got["mu"] < MU_TOL, (cell, got["mu"])
    assert got["primal_residual"] < 1e-6
    want = _reference(case, spec, args)
    assert want["mu"] < MU_TOL
    assert abs(got["z"] - want["z"]) <= Z_RTOL * abs(want["z"])
    for key in ("shed", "reserve_shortfall", "reserve_price"):
        np.testing.assert_allclose(got[key], want[key], rtol=0, atol=ATOL[key], err_msg=key)
    np.testing.assert_allclose(got["reserve"], want["reserve"], rtol=0,
                               atol=RESERVE_MW_ATOL, err_msg="reserve")


_PRICES_XFAIL = dict(reason=(
    "813nem 03: after mu converges (step ~25) the fixed 60-step loop keeps "
    "stepping on a near-singular Newton system and the stationarity residual "
    "drifts to 1e-2..1e1 on both this operator and the numpy reference, so "
    "the two land on different vertices and duals of the same optimal face: "
    "measured 2026-09-16 |dlmp| 1.8..5.9 $/MWh, |daward| 75..226 MW, objective "
    "equal to 1e-13. A merit-gated freeze was measured to bring both to "
    "<= 3e-4 $/MWh; adopting it, or a mu tolerance for this case, is an "
    "open decision. strict: this test turns red the day the prices agree, "
    "which is the signal to retire the marker."))
#: 0_0 and 0_24 stay strict: the two solvers disagree by 0.7..5.9 $/MWh at
#: every core count measured (1 / 4 / 8).  0_36 is not strict: at
#: 4 cores the prices differ, at 8 cores (2026-09-17) they
#: agree within `ATOL` -- the plain loop's end point moves with the thread
#: count too (module docstring), so a strict marker here would read a core
#: count as a code change.
@pytest.mark.parametrize("cell", [
    pytest.param("0_0", marks=pytest.mark.xfail(strict=True, **_PRICES_XFAIL)),
    pytest.param("0_24", marks=pytest.mark.xfail(strict=True, **_PRICES_XFAIL)),
    pytest.param("0_36", marks=pytest.mark.xfail(strict=False, **_PRICES_XFAIL)),
])
def test_rated_rows_prices_match_the_reference(case, rated, cell):
    """The part of the L2 the market's calibration does NOT determine on this
    case: the energy price and, through it, the per-unit energy revenue."""
    clear, spec = rated
    args = _cells()[cell]
    got = _run(clear, args)
    want = _reference(case, spec, args)
    _clean(got, want)
    np.testing.assert_allclose(got["lmp"], want["lmp"], rtol=0, atol=ATOL["lmp"], err_msg="lmp")
    bus = np.asarray(case.unit_node_idx)
    np.testing.assert_allclose(got["award"] @ got["lmp"][bus], want["award"] @ want["lmp"][bus],
                               rtol=1e-6, atol=0, err_msg="energy revenue")


#: The gate's arming value the controls were measured at.  What the
#: adopted value should be (this, or the market's `MU_TOL`) is left open;
#: the tests below assert the mechanism at the measured value.
FREEZE_MU = 1e-6
#: Measured on `case813nem` rated rows, cells 0_0 / 0_24 / 0_36, this operator
#: against the numpy reference under the same gate, CPU, 2026-09-17,
#: at 1 / 4 / 8 cores -- the frozen
#: point moves with the thread count (module docstring).  Worst of the three
#: core counts, then a 2-3x margin:
#:   dual residual, either side   1c 9.0e-6   4c 8.7e-7   8c 9.3e-4   -> 3e-3
#:   |dlmp| $/MWh                 1c 3.0e-5   4c 4.4e-9   8c 1.7e-3   -> 5e-3
#:   |dz| / |z|                   1c 1.0e-13  4c 2.2e-16  8c 1.8e-9   -> 5e-9
#:   mu, either side              1c 2.8e-9   4c 1.6e-7   8c 2.6e-7   -> FREEZE_MU holds
#: The first reading (2026-09-16, five cells: |dlmp| 4e-9..2.9e-4, dual residual
#: 5e-11..8.8e-5, bounds 1e-3 / 1e-3) was a 4-core reading only.
FREEZE_LMP_ATOL = 5e-3
FREEZE_DUAL_RES = 3e-3
FREEZE_Z_RTOL = 5e-9


@pytest.fixture(scope="module")
def rated_frozen(case):
    return make_clearing(case, THETA, VOLR, n_segments=1, cap_scale=CAP_SCALE,
                         ramp_scale=RAMP_SCALE, period_hours=DELTA,
                         monitored_lines=rated_lines(case), freeze_mu=FREEZE_MU,
                         kkt="dense")


@pytest.mark.parametrize("cell", L2_CELLS)
def test_frozen_solver_holds_the_converged_iterate_and_prices_agree(case, rated_frozen, cell):
    """`ipm.make_solver(freeze_mu=...)`: once ``mu`` is below the arming
    value, a step that would raise the merit is refused, so the loop holds the
    point it converged to instead of drifting off it.  Asserted two-sided on
    the same cells the unfrozen test above is red on: the dual residual is
    small AND this operator agrees with the numpy reference running the same
    gate on the price -- under the plain loop they disagree by 1.8..5.9 $/MWh
    (the strict xfail above), so neither bound is vacuous.  The spec stamps
    the value in force.  Pinned to the dense route like `_build`
    (2026-09-17): the fixture above had no ``kkt`` and so ran the low-rank
    route once that became the default -- a leak injected into that route's requirement rows
    turned this test red -- while the gate values here were taken on the
    dense factorisation."""
    clear, spec = rated_frozen
    assert spec["freeze_mu"] == FREEZE_MU
    args = _cells()[cell]
    got = _run(clear, args)
    want = clear_reference(case, *args, theta=THETA, volr=VOLR, cap_scale=CAP_SCALE,
                           ramp_scale=RAMP_SCALE, period_hours=DELTA, max_iter=MAX_ITER,
                           dual_start=DUAL_START, reg_coef=REG_COEF,
                           monitored_lines=spec["monitored_lines"], freeze_mu=FREEZE_MU)
    assert got["dual_residual"] < FREEZE_DUAL_RES, (cell, got["dual_residual"])
    assert want["dual_residual"] < FREEZE_DUAL_RES, (cell, want["dual_residual"])
    assert got["mu"] < FREEZE_MU and want["mu"] < FREEZE_MU
    assert abs(got["z"] - want["z"]) <= FREEZE_Z_RTOL * abs(want["z"])
    np.testing.assert_allclose(got["lmp"], want["lmp"], rtol=0, atol=FREEZE_LMP_ATOL, err_msg="lmp")
    np.testing.assert_allclose(got["shed"], want["shed"], rtol=0, atol=ATOL["shed"])
    np.testing.assert_allclose(got["reserve_price"], want["reserve_price"], rtol=0,
                               atol=ATOL["reserve_price"])


def test_frozen_solver_changes_nothing_where_the_plain_loop_is_clean(x64):
    """Negative control on ``case29gb``: with the gate armed the result is the
    plain loop's to L2 tolerance (measured 2026-09-16: mu 6.3e-12 either way,
    |dlmp| 4.6e-13), because a clean plateau never raises the merit."""
    from tests.envs.ancillary.test_clearing_l0 import FIXTURE as FX29, HOUR
    from powermarketjax.envs.day_ahead.clearing import segment_costs
    c29 = load_case("29gb")
    _, cost = segment_costs(c29, 1)
    fx = np.load(FX29, allow_pickle=True)
    u = fx["commitment"][0, :, HOUR].astype(np.float64)
    pmin = np.asarray(c29.unit_p_min); pmax = np.asarray(c29.unit_p_max)
    demand = float((pmin * u).sum() + 0.85 * ((pmax - pmin) * u).sum())
    p_prev = (pmin + 0.85 * (pmax - pmin)) * u
    kw = dict(n_segments=1, cap_scale=0.6, ramp_scale=1.0, period_hours=DELTA)
    plain, sp = make_clearing(c29, THETA, 250.0, **kw)
    frozen, sf = make_clearing(c29, THETA, 250.0, freeze_mu=FREEZE_MU, **kw)
    assert sp["freeze_mu"] is None and sf["freeze_mu"] == FREEZE_MU
    args = (cost, np.zeros((len(u), 2)), u, demand, np.array([0.05, 0.05]) * demand, p_prev)
    a, b = _run(plain, args), _run(frozen, args)
    assert a["mu"] < MU_TOL and b["mu"] < MU_TOL
    np.testing.assert_allclose(a["lmp"], b["lmp"], rtol=0, atol=ATOL["lmp"])
    np.testing.assert_allclose(a["award"], b["award"], rtol=0, atol=ATOL["award"])
    assert abs(a["z"] - b["z"]) <= Z_RTOL * abs(b["z"])


def test_line_duals_are_reported_over_every_line(case, rated):
    """`line_dual_up` / `line_dual_dn` keep the all-lines shape and are zero
    off the monitored set, so a consumer can mix the two row sets."""
    clear, spec = rated
    got = _run(clear, _cells()["0_0"])
    mon = np.asarray(spec["monitored_lines"])
    assert got["line_dual_up"].shape == got["line_dual_dn"].shape == (1278,)
    off = np.ones(1278, bool); off[mon] = False
    assert (got["line_dual_up"][off] == 0.0).all() and (got["line_dual_dn"][off] == 0.0).all()
