"""L2: the arrowhead solve of the low-rank route's pivoted block
(`kkt_lowrank`, "Arrowhead solve"; ``lu_batching="arrow"``) on this market's
shape, against the dense route and against the pivoted-LU solve of the same
block.

1. The five recorded shed periods of `case813nem`
   (`tests/fixtures/ge04_t1_shed_cells_813nem.npz`), through
   `make_rt_clearing` with ``lu_batching="arrow"``: the dense route's result
   to the arrowhead tolerances (both converged, dual residual at the dense level,
   ``|dlmp| <= 1e-6`` \$/MWh), at the case's sizing (32, 523 on this case
   as it stands), at the earlier (8, 523), and with every
   column free ``(n_units, n_buses)`` -- the sizing the arrowhead makes
   affordable, where no region can be left out.
2. Arrow against the pivoted LU (``sequential``) on the same block: the same
   prices to 1e-6, so the two solvers of one block are interchangeable.
3. The negative control: with `ARROW_REFINE` set to 0 the border
   rows are left at 6e-4 and the outer refinement diverges; the five periods
   must then fail the way the plain route does (thread-count-robust criterion
   of `test_lowrank_t1_l2`).  If this ever passes, the inner pass has stopped
   mattering and the comparison above has stopped being a check of it.
4. The route is refused on any shape but ``q * T = 1`` and stamped
   ``+lu:arrow``; `householder_qr` is orthonormal on columns spanning
   1e-7..1e7 and finite on a zero column.
"""
import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.case import load_case, scale_min_output
from powermarketjax.envs.day_ahead import kkt_lowrank
from powermarketjax.envs.day_ahead.clearing import make_clearing, rated_lines
from powermarketjax.envs.real_time.clearing import (MAX_ITER, default_lowrank_free,
                                                    lowrank_free_for, make_rt_clearing)
from powermarketjax.envs.real_time.env import MU_TOL
from tests.envs.day_ahead.test_clearing_l2 import ATOL_LMP
from tests.envs.real_time.test_lowrank_t1_l2 import _as_np, _assert_same_clearing, _cell

FIXTURES = pathlib.Path(__file__).resolve().parents[2] / "fixtures"
T1_CELLS = FIXTURES / "ge04_t1_shed_cells_813nem.npz"

#: The price tolerance between the arrowhead solve and the dense route,
#: and between the two solvers of the same block; measured 2026-09-17 (CPU).
ATOL_LMP_ARROW = 1e-6


@pytest.fixture(scope="module", autouse=True)
def x64():
    prev = jax.config.jax_enable_x64, jax.config.jax_default_matmul_precision
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    yield
    jax.config.update("jax_enable_x64", prev[0])
    jax.config.update("jax_default_matmul_precision", prev[1])


@pytest.fixture(scope="module")
def nem():
    z = np.load(T1_CELLS)
    meta = json.loads(str(z["meta"]))
    case = scale_min_output(load_case(meta["case"]), float(meta["p_min_scale"]))
    mon = rated_lines(case)
    kw = dict(n_segments=meta["n_segments"], cap_scale=meta["cap_scale"],
              ramp_scale=meta["ramp_scale"], period_hours=meta["period_hours"],
              max_iter=MAX_ITER, monitored_lines=mon)
    n_units, n_buses = len(np.asarray(case.unit_p_min)), np.asarray(case.PTDF).shape[1]
    sizings = {"all": default_lowrank_free(case, mon), "ge04": lowrank_free_for(case, mon), "small": (8, 523)}
    assert sizings["all"] == (n_units, n_buses) and sizings["ge04"] == (32, 523)
    arrow = {k: make_rt_clearing(case, lowrank_free=v, lu_batching="arrow", **kw)
             for k, v in sizings.items()}
    lu_clear, lu_spec = make_rt_clearing(case, lowrank_free=sizings["ge04"],
                                         lu_batching="sequential", **kw)
    dn_clear, dn_spec = make_clearing(case, 1, kkt="dense", **kw)
    return dict(z=z, case=case, mon=mon, kw=kw, sizings=sizings,
                arrow={k: (jax.jit(c), s) for k, (c, s) in arrow.items()},
                lu=jax.jit(lu_clear), lu_spec=lu_spec, dense=jax.jit(dn_clear), dense_spec=dn_spec)


@pytest.mark.parametrize("sizing", ["all", "ge04", "small"])
@pytest.mark.parametrize("i", range(5))
def test_arrow_matches_dense_on_recorded_periods(nem, sizing, i):
    clear, spec = nem["arrow"][sizing]
    assert spec["lu_batching"] == "arrow"
    args = _cell(nem["z"], i)
    ar, dn = _as_np(clear(*args)), _as_np(nem["dense"](*args))
    label = f"813nem cell {i} arrow {nem['sizings'][sizing]}"
    _assert_same_clearing(ar, dn, label)
    assert np.abs(ar["lmp"] - dn["lmp"]).max() <= ATOL_LMP_ARROW, label


@pytest.mark.parametrize("i", range(5))
def test_arrow_matches_pivoted_lu(nem, i):
    args = _cell(nem["z"], i)
    ar, lu = _as_np(nem["arrow"]["ge04"][0](*args)), _as_np(nem["lu"](*args))
    assert ar["mu"] < MU_TOL and lu["mu"] < MU_TOL
    assert np.abs(ar["lmp"] - lu["lmp"]).max() <= ATOL_LMP_ARROW, i
    np.testing.assert_allclose(ar["shed"].sum(), lu["shed"].sum(), rtol=1e-6, atol=1e-3)


def test_inner_refinement_is_what_makes_it_work(nem, monkeypatch):
    """Negative control: `ARROW_REFINE = 0` must leave the five periods wrong
    the way the plain route is wrong.  The criterion is the one
    `test_lowrank_t1_l2` uses for the plain route, robust to the thread
    count: every period violates at least one of the three checks, and at
    least one period's price is off by more than 1 \$/MWh."""
    monkeypatch.setattr(kkt_lowrank, "ARROW_REFINE", 0)
    clear, spec = make_rt_clearing(nem["case"], **nem["kw"])   # the default: all free, arrow
    assert spec["lu_batching"] == "arrow" and spec["lowrank_free"] == nem["sizings"]["all"]
    clear = jax.jit(clear)
    worst_dlmp = 0.0
    for i in range(5):
        args = _cell(nem["z"], i)
        bad, dn = _as_np(clear(*args)), _as_np(nem["dense"](*args))
        dlmp = float(np.nan_to_num(np.abs(bad["lmp"] - dn["lmp"]), nan=np.inf).max())
        worst_dlmp = max(worst_dlmp, dlmp)
        assert not (bad["mu"] < MU_TOL and bad["dual_residual"] < 1e-6 and dlmp < ATOL_LMP), (
            f"cell {i}: without the inner pass the arrow route passes all three checks "
            f"(mu {bad['mu']:.3e}, dual {bad['dual_residual']:.3e}, |dlmp| {dlmp:.3e}); "
            "the negative control has lost its teeth")
    assert worst_dlmp > 1.0, worst_dlmp


def test_arrow_is_refused_off_the_single_period_shape(nem):
    with pytest.raises(ValueError, match="arrow"):
        make_clearing(nem["case"], 2, kkt="lowrank", lowrank_free=(8, 523), lu_batching="arrow",
                      n_segments=1, cap_scale=1.0, ramp_scale=1.0, period_hours=0.5,
                      max_iter=MAX_ITER, monitored_lines=nem["mon"])


def test_route_stamp_names_the_solver(nem):
    s = nem["sizings"]
    assert nem["arrow"]["all"][1]["kkt_route"] == f"lowrank+free({s['all'][0]},{s['all'][1]})+lu:arrow"
    assert nem["arrow"]["ge04"][1]["kkt_route"] == f"lowrank+free({s['ge04'][0]},{s['ge04'][1]})+lu:arrow"
    assert nem["lu_spec"]["kkt_route"] == f"lowrank+free({s['ge04'][0]},{s['ge04'][1]})+lu:sequential"
    assert nem["dense_spec"]["kkt_route"] == "dense"
    # `auto` resolves to arrow on the arrowhead shape only
    assert kkt_lowrank.resolve_lu_batching("arrow") == "arrow"
    assert kkt_lowrank.resolve_lu_batching("auto", arrowhead=True) == "arrow"
    assert kkt_lowrank.resolve_lu_batching("auto", arrowhead=False) in ("sequential", "batched")
    default_clear, default_spec = make_rt_clearing(nem["case"], **nem["kw"])
    assert default_spec["kkt_route"] == f"lowrank+free({s['all'][0]},{s['all'][1]})+lu:arrow"


def test_householder_qr_on_spanning_columns():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((531, 8)) * (10.0 ** np.linspace(-7, 7, 8))[None, :]
    A[:, 3] = 0.0                                   # a line with nothing behind it
    Q, T = (np.asarray(t) for t in jax.jit(kkt_lowrank.householder_qr)(jnp.asarray(A)))
    assert np.isfinite(Q).all() and np.isfinite(T).all()
    assert np.abs(Q.T @ Q - np.eye(8)).max() < 1e-13
    assert np.abs(Q @ T - A).max() <= 1e-13 * np.abs(A).max()
    assert np.abs(np.tril(T, -1)).max() == 0.0
    assert np.abs(T[:, 3]).max() == 0.0
