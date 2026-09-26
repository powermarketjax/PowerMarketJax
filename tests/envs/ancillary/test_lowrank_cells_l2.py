"""L2: the low-rank KKT route of the joint energy-reserve clearing against
the dense route on the same rows -- `case813nem`, the rated lines, the four
truthful periods of `tests/fixtures/ge05_03_cells_813nem.npz`
(2026-09-17).

Both arms run this case's adopted recipe, ``stop_tol=(1e-9, 1e-4)``,
``max_iter=200``: the data-dependent stop is what the market runs,
and on this LP the fixed 60 steps walk the dense route off its plateau,
which would compare a converged point with a drifted one.  Under
the recipe each arm stops at its own first iterate under the gate, so the
two are compared as two stops of one interior-point path on one LP:

1. Both converged: the low-rank route's ``mu`` under `MU_TOL` (measured
   4.3e-10..6.4e-10 at 1, 4 and 8 cores) and its dual residual at or under
   1e-8 (measured 4.2e-11..3.8e-9); the dense route's ``mu`` under
   `MU_DENSE_CELLS` -- on cell 0_24 it reaches the cap and hands back its
   best iterate, whose ``mu`` is 1.4e-11 at 1 and 4 cores and 2.84e-9 at 8
   cores on two core segments alike (measured on cores 24-31 and 56-63; the
   gate at `MU_TOL` went red there, 2026-09-17) -- and its dual residual
   within 2e-4 (it stops at its 1e-4 gate or hands back its best iterate at
   the cap, 8.5e-5..1.1e-4 over the core counts:
   its Newton solves are the less accurate ones on this LP).
2. Prices: reserve prices to 1e-6 \\$/MWh (measured 5.6e-11), LMP to
   `ATOL_LMP_CELLS` (measured 2.7e-5..2.2e-4 \\$/MWh over 1 / 4 / 8 cores;
   the spread is the dense
   arm's stop point at a residual of 1e-5, not the route -- on the cell
   whose dense stop is at 1e-10, 0_36, the two agree to 9e-11).
3. Quantities: total award, shed and reserve shortfall to 1e-6 MW (measured
   6.4e-9); per-unit award may differ on tied units (measured up to 139 MW
   moved between units of identical offer, a degenerate face of
   the LP), so it is not compared per unit, and the reserve
   cleared above the requirement is free at a zero reserve offer (measured
   3.2e-3 MW apart in the total), so the requirement met is what is checked.
4. Objective to 1e-9 relative (measured 4.8e-12).
5. The negative controls: the plain Schur solve
   (``lowrank_free=(0, 0)``) on the fixed 60 steps leaves a dual residual
   above 1 (measured 4.5e13, cell 0_0); both refinements switched off
   (`ARROW_REFINE` and `N_REFINE` at 0) leave it above 1e-3 under the
   recipe (measured 8.2e-2; either one alone still reaches 1e-6..1e-9, so
   the control switches both).  If either control ever passes, the
   comparison above has stopped being a check.
6. The route stamp names the sizing and the solver.

The thread count moves both arms' stop points (measured on the dense route);
every tolerance here is the widest of the three core counts, and a run at
another count says nothing about these.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.ancillary.clearing import make_clearing
from powermarketjax.envs.ancillary.env import MU_TOL
from powermarketjax.envs.day_ahead import kkt_lowrank
from powermarketjax.envs.day_ahead.clearing import rated_lines

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "ge05_03_cells_813nem.npz"
THETA = (1.0 / 6.0, 0.5)
VOLR = 147.0
DELTA = 0.5
#: this case's adopted recipe (`run_rl_03.py --stop-tol 1e-9,1e-4 --max-iter 200`)
STOP_TOL, MAX_ITER = (1e-9, 1e-4), 200
#: measured 2026-09-17, CPU, 1 / 4 / 8 cores:
#: 1.1e-4 / 1.45e-4 / 1.64e-4 worst over the four cells; 2.2e-4 at one core
ATOL_LMP_CELLS = 1e-3
ATOL_RESERVE_PRICE = 1e-6
ATOL_MW = 1e-6
RTOL_Z = 1e-9
DUAL_RES_LOWRANK = 1e-8
DUAL_RES_DENSE = 2e-4
#: the dense arm's ``mu`` at its returned iterate: measured 2026-09-17, CPU,
#: cell 0_24 at the cap, 1.4e-11 (1 / 4 cores) and 2.84e-9 (8 cores, on cores
#: 16-23, 24-31 and 56-63 alike); the low-rank arm stays under `MU_TOL`
MU_DENSE_CELLS = 1e-8


@pytest.fixture(scope="module", autouse=True)
def x64():
    prev = jax.config.jax_enable_x64, jax.config.jax_default_matmul_precision
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    yield
    jax.config.update("jax_enable_x64", prev[0])
    jax.config.update("jax_default_matmul_precision", prev[1])


@pytest.fixture(scope="module")
def rig():
    fx = np.load(FIXTURE, allow_pickle=True)
    meta = json.loads(str(fx["meta"]))
    cells = {c: tuple(jnp.asarray(np.asarray(fx[f"{c}_{k}"], np.float64))
                      for k in ("offer", "offer_res", "u", "demand", "d_res", "p_prev"))
             for c in (str(s) for s in fx["cells"])}
    case = load_case("813nem")
    kw = dict(n_segments=1, cap_scale=meta["cap_scale"], ramp_scale=meta["ramp_scale"],
              period_hours=DELTA, monitored_lines=rated_lines(case))
    lr, lr_spec = make_clearing(case, THETA, VOLR, stop_tol=STOP_TOL, max_iter=MAX_ITER, **kw)
    dn, dn_spec = make_clearing(case, THETA, VOLR, stop_tol=STOP_TOL, max_iter=MAX_ITER, kkt="dense", **kw)
    return dict(case=case, kw=kw, cells=cells, lr=jax.jit(lr), dn=jax.jit(dn),
                lr_spec=lr_spec, dn_spec=dn_spec)


def _np(out):
    return {k: np.asarray(v, np.float64) for k, v in out.items()}


def test_route_stamps(rig):
    s = rig["lr_spec"]
    assert s["kkt_route"] == f"lowrank+free({s['n_units']},{s['n_buses'] + s['n_prod']})+lu:arrow"
    assert s["lowrank_free"] == (151, 815) and s["lu_batching"] == "arrow"
    assert rig["dn_spec"]["kkt_route"] == "dense" and rig["dn_spec"]["lowrank_free"] == (0, 0)
    assert s["m"] == rig["dn_spec"]["m"] == 3005 and s["n"] == 1268


@pytest.mark.parametrize("cell", ["0_0", "0_12", "0_24", "0_36"])
def test_lowrank_matches_dense_on_the_recipe(rig, cell):
    lr, dn = _np(rig["lr"](*rig["cells"][cell])), _np(rig["dn"](*rig["cells"][cell]))
    assert lr["mu"] < MU_TOL and dn["mu"] < MU_DENSE_CELLS, (lr["mu"], dn["mu"])
    assert lr["dual_residual"] <= DUAL_RES_LOWRANK, lr["dual_residual"]
    # the dense arm stops at its gate, or hands back its best iterate at the
    # cap: measured 8.5e-5 at 4 and 8 cores, 1.1e-4 at one core (cell 0_24)
    assert dn["dual_residual"] <= DUAL_RES_DENSE, dn["dual_residual"]
    assert lr["primal_residual"] < 1e-8 and dn["primal_residual"] < 1e-8
    np.testing.assert_allclose(lr["reserve_price"], dn["reserve_price"], rtol=0,
                               atol=ATOL_RESERVE_PRICE, err_msg="reserve price")
    np.testing.assert_allclose(lr["lmp"], dn["lmp"], rtol=0, atol=ATOL_LMP_CELLS, err_msg="lmp")
    np.testing.assert_allclose(lr["award"].sum(), dn["award"].sum(), rtol=0, atol=ATOL_MW,
                               err_msg="total award")
    # the truthful reserve offer is zero on every unit, so the reserve cleared
    # above the requirement is not determined by the LP (measured: the two
    # arms differ by up to 3.2e-3 MW in the total); what is consumed is the
    # requirement met, i.e. total reserve + shortfall against the requirement
    d_res = np.asarray(rig["cells"][cell][4])
    for o, name in ((lr, "lowrank"), (dn, "dense")):
        assert (o["reserve"].sum(0) + o["reserve_shortfall"] >= d_res - ATOL_MW).all(), name
    np.testing.assert_allclose(lr["shed"].sum(), dn["shed"].sum(), rtol=0, atol=ATOL_MW,
                               err_msg="total shed")
    np.testing.assert_allclose(lr["reserve_shortfall"], dn["reserve_shortfall"], rtol=0,
                               atol=ATOL_MW, err_msg="reserve shortfall")
    np.testing.assert_allclose(lr["z"], dn["z"], rtol=RTOL_Z, atol=0, err_msg="objective")
    # per-unit award moves only between units of one offer price
    moved = np.abs(lr["award"] - dn["award"]) > 1e-3
    if moved.any():
        offer = np.asarray(rig["cells"][cell][0])[:, 0]
        prices = np.unique(np.round(offer[moved], 6))
        for p in prices:
            group = np.abs(offer - p) < 1e-6
            assert group.sum() >= 2, ("award moved on a unit with no tied partner", p)
            np.testing.assert_allclose(lr["award"][group].sum(), dn["award"][group].sum(),
                                       rtol=0, atol=1e-3, err_msg=f"award within the tie at {p}")


def test_plain_schur_route_is_the_negative_control(rig):
    plain, spec = make_clearing(rig["case"], THETA, VOLR, kkt="lowrank", lowrank_free=(0, 0), **rig["kw"])
    assert spec["kkt_route"] == "lowrank"
    out = _np(jax.jit(plain)(*rig["cells"]["0_0"]))
    assert out["dual_residual"] > 1.0, out["dual_residual"]


def test_both_refinements_off_is_the_negative_control(rig, monkeypatch):
    monkeypatch.setattr(kkt_lowrank, "ARROW_REFINE", 0)
    monkeypatch.setattr(kkt_lowrank, "N_REFINE", 0)
    off, spec = make_clearing(rig["case"], THETA, VOLR, stop_tol=STOP_TOL, max_iter=MAX_ITER, **rig["kw"])
    assert spec["kkt_route"].endswith("+lu:arrow")
    out = _np(jax.jit(off)(*rig["cells"]["0_0"]))     # traced now, under the patched constants
    assert out["dual_residual"] > 1e-3, out["dual_residual"]
