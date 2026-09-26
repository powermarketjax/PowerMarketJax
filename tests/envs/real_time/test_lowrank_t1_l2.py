"""L2: the low-rank ``kkt`` route on this market's shape (T = 1) against the
dense route, on the periods that shed load.

`tests/envs/day_ahead/test_lowrank_l2.py` compares the two routes at T = 2
and T = 24 on days that shed nothing.  On this market the failing shape is a
period that sheds load behind a binding line: every shed column of the
import-constrained region is then dual-degenerate (its price is VOLL, so its
only curvature is the solver's ``reg``) while the binding line's barrier
weight is 1e14..1e18, and the plain Schur route of `kkt_lowrank` computes
those columns as ``(r_f - U_f w) / reg``.  Measured 2026-09-16 on five such
periods of `case813nem`: Newton residual
1e1..4e1 against the dense route's 1e-12, refinement diverging by ~20x per
step, ``mu`` 6e-8..1e3 on the clearing.  The fix keeps the highest-ranked
free columns inside a pivoted block (`kkt_lowrank`, "Free columns");
`clearing.make_rt_clearing` sizes that block from the case
(`lowrank_free_for`).

What is checked, each against the dense route on the same monitored-lines
LP:

1. T = 1 on the three cases with the sizing `make_rt_clearing` applies.
2. The five recorded shed periods of `case813nem`
   (`tests/fixtures/ge04_t1_shed_cells_813nem.npz`; inputs only, nothing
   derived), through `make_rt_clearing` itself so the wiring is what the
   environment runs.  Tolerances are the day-ahead L2 ones for ``lmp``; the
   per-unit ``award`` is compared to 1e-2 MW because the two routes may land
   on different vertices of a degenerate optimal face (measured 4.6e-4 MW on
   one of the five, 1e-12..1e-10 on the others), and the period total to
   1e-4 MW.
3. A day-ahead day that sheds (T = 24, 230 MW,
   `tests/fixtures/ge04_t24_shed_days_813nem.npz`), with the same sizing
   passed explicitly: the defect is a property of shed periods, not of T = 1.
4. The negative control: the plain route, ``lowrank_free=(0, 0)``,
   on the first recorded period must still leave a dual residual above 1.
   If this ever passes, the comparison above has stopped being a check.
5. The route stamp names the sizing, so a product made on the pivoted-block
   variant can be told from one made on the plain route.
"""
import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.case import load_case, scale_min_output
from powermarketjax.envs.day_ahead.action import make_offer_map
from powermarketjax.envs.day_ahead.clearing import make_clearing, rated_lines, segment_costs
from powermarketjax.envs.day_ahead.demand import load_nem_demand
from powermarketjax.envs.real_time.clearing import (LOWRANK_FREE_UNITS, MAX_ITER,
                                                    default_lowrank_free, lowrank_free_for,
                                                    make_rt_clearing)
from powermarketjax.envs.real_time.env import MU_TOL
from tests.envs.day_ahead.test_clearing_l2 import ATOL_LMP, RTOL_LMP
from tests.envs.day_ahead.test_lowrank_l2 import (CASE_NAMES, CASE_SCALE, _demand_for,
                                                  _feasible_u_ones, _monitored_lines)

FIXTURES = pathlib.Path(__file__).resolve().parents[2] / "fixtures"
T1_CELLS = FIXTURES / "ge04_t1_shed_cells_813nem.npz"
T24_DAYS = FIXTURES / "ge04_t24_shed_days_813nem.npz"

#: Half-hour periods, this market's `period_hours`.
DELTA = 0.5
#: Per-unit award between the two routes on a degenerate face; measured
#: 4.6e-4 MW on one of the five recorded periods (2026-09-16, CPU).
ATOL_AWARD_UNIT = 1e-2
ATOL_AWARD_TOTAL = 1e-4
ATOL_SHED_TOTAL = 1e-3


@pytest.fixture(scope="module", autouse=True)
def x64():
    prev = jax.config.jax_enable_x64, jax.config.jax_default_matmul_precision
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    yield
    jax.config.update("jax_enable_x64", prev[0])
    jax.config.update("jax_default_matmul_precision", prev[1])


def _as_np(out):
    return {k: np.asarray(v, np.float64) for k, v in out.items()}


def _assert_same_clearing(lr, dn, label):
    """Both converged, dual residual at the dense route's level, same prices,
    same dispatch up to the degenerate face, same shed total."""
    assert lr["mu"] < MU_TOL and dn["mu"] < MU_TOL, (label, lr["mu"], dn["mu"])
    assert lr["dual_residual"] <= 10.0 * dn["dual_residual"] + 1e-8, (
        label, lr["dual_residual"], dn["dual_residual"])
    np.testing.assert_allclose(lr["lmp"], dn["lmp"], rtol=RTOL_LMP, atol=ATOL_LMP,
                               err_msg=f"lmp ({label})")
    np.testing.assert_allclose(lr["award"], dn["award"], rtol=1e-6, atol=ATOL_AWARD_UNIT,
                               err_msg=f"award per unit ({label})")
    np.testing.assert_allclose(lr["award"].sum(0), dn["award"].sum(0), rtol=1e-8,
                               atol=ATOL_AWARD_TOTAL, err_msg=f"award total ({label})")
    np.testing.assert_allclose(lr["shed"].sum(1), dn["shed"].sum(1), rtol=1e-6,
                               atol=ATOL_SHED_TOTAL, err_msg=f"shed total ({label})")


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_t1_lowrank_matches_dense(case_name):
    """T = 1, half-hour ramps, the sizing `make_rt_clearing` would apply."""
    case = load_case(case_name)
    cap_scale, ramp_scale = CASE_SCALE[case_name]
    demand = _demand_for(case_name, case)
    monitored = _monitored_lines(case_name, np.asarray(case.line_cap).shape[0])
    _, cost = segment_costs(case, 1)
    offer = cost[:, :, None]
    p_min = np.asarray(case.unit_p_min, np.float64)
    u = _feasible_u_ones(case, demand)[:, None]
    p_init = p_min * u[:, 0]
    dem = np.full(1, demand)
    kw = dict(n_segments=1, cap_scale=cap_scale, ramp_scale=ramp_scale, period_hours=DELTA,
              max_iter=MAX_ITER, monitored_lines=monitored)
    sizing = lowrank_free_for(case, monitored)
    lr_clear, lr_spec = make_clearing(case, 1, kkt="lowrank", lowrank_free=sizing, **kw)
    dn_clear, dn_spec = make_clearing(case, 1, kkt="dense", **kw)
    assert lr_spec["kkt_route"].startswith("lowrank") and dn_spec["kkt_route"] == "dense"
    args = (jnp.asarray(offer), jnp.asarray(u), jnp.asarray(dem), jnp.asarray(p_init))
    _assert_same_clearing(_as_np(jax.jit(lr_clear)(*args)), _as_np(jax.jit(dn_clear)(*args)),
                          f"{case_name} T=1 lowrank_free={sizing}")


@pytest.fixture(scope="module")
def nem():
    z = np.load(T1_CELLS)
    meta = json.loads(str(z["meta"]))
    case = scale_min_output(load_case(meta["case"]), float(meta["p_min_scale"]))
    mon = rated_lines(case)
    kw = dict(n_segments=meta["n_segments"], cap_scale=meta["cap_scale"],
              ramp_scale=meta["ramp_scale"], period_hours=meta["period_hours"])
    rt_clear, rt_spec = make_rt_clearing(case, max_iter=MAX_ITER, monitored_lines=mon, **kw)
    dn_clear, dn_spec = make_clearing(case, 1, kkt="dense", max_iter=MAX_ITER,
                                      monitored_lines=mon, **kw)
    plain_clear, plain_spec = make_clearing(case, 1, kkt="lowrank", lowrank_free=(0, 0),
                                            max_iter=MAX_ITER, monitored_lines=mon, **kw)
    return dict(z=z, case=case, mon=mon, rt=jax.jit(rt_clear), rt_spec=rt_spec,
                dense=jax.jit(dn_clear), dense_spec=dn_spec, plain=jax.jit(plain_clear),
                plain_spec=plain_spec)


def _cell(z, i):
    return (jnp.asarray(z["offer"][i]), jnp.asarray(z["u"][i]),
            jnp.asarray(z["demand"][i]), jnp.asarray(z["p_init"][i]))


@pytest.mark.parametrize("i", range(5))
def test_recorded_shed_periods_813nem(nem, i):
    """The five periods the real-time environment did not converge on, through
    `make_rt_clearing` as the environment builds it."""
    z = nem["z"]
    lr = _as_np(nem["rt"](*_cell(z, i)))
    dn = _as_np(nem["dense"](*_cell(z, i)))
    shed_dense = json.loads(str(z["meta"]))["shed_mw_dense"][i]
    assert abs(dn["shed"].sum() - shed_dense) < 1e-3, "the recorded period no longer sheds"
    _assert_same_clearing(lr, dn, f"813nem T=1 cell {i} (step {int(z['step'][i])}, env {int(z['env'][i])})")


def test_plain_route_still_fails_on_a_shed_period(nem):
    """Negative control: without the pivoted block the same periods are solved
    wrong.  **Which quantity carries the failure depends on the thread count**:
    the Newton solve is wrong at the linear-algebra level on every platform
    (residual 1e1..4e1 against the dense route's 1e-12), but the iterate the
    fixed-trip IPM lands on afterwards is round-off chaos, so the final
    ``mu`` / ``dual_residual`` are not reproducible across thread counts
    (measured 2026-09-17, CPU): cell 0's
    dual residual is 6.9e-1 at 1 core, 5.6e-4 at 4, 9.9e3 at 8 -- the 4-core
    value is *below* the dense route's own 5.0e-3 there, which is why an
    earlier version of this test asserting ``> 1.0`` went red on 4 cores
    (2026-09-17).  What holds at 1, 4 and 8 cores on all five cells
    is that the plain route fails at least one of the three consumed checks
    the fixed route passes -- ``mu`` below `MU_TOL`, dual residual at the
    healthy level, prices within `ATOL_LMP` of the dense route -- and that on
    at least one cell the price is off by more than 1 $/MWh (2.3e3 at 1 core,
    1.3e16 at 4, 1.4e5 at 8).  So the assertion is that disjunction, per
    cell, not a magnitude.  If this ever passes with the plain route, the
    comparison above has stopped being a check."""
    z = nem["z"]
    assert nem["plain_spec"]["kkt_route"] == "lowrank"
    worst_dlmp = 0.0
    for i in range(5):
        plain = _as_np(nem["plain"](*_cell(z, i)))
        dn = _as_np(nem["dense"](*_cell(z, i)))
        dlmp = float(np.abs(plain["lmp"] - dn["lmp"]).max())
        worst_dlmp = max(worst_dlmp, dlmp)
        looks_solved = (plain["mu"] < MU_TOL and plain["dual_residual"] < 1e-6
                        and dlmp < ATOL_LMP)
        assert not looks_solved, (
            f"cell {i}: the plain route now passes all three checks "
            f"(mu {plain['mu']:.3e}, dual {plain['dual_residual']:.3e}, |dlmp| {dlmp:.3e}); "
            "the negative control has lost its teeth")
    assert worst_dlmp > 1.0, worst_dlmp


def test_route_stamp_names_the_sizing(nem):
    """`spec["kkt_route"]` tells the two low-rank variants apart.  The
    real-time wrapper's default is every column free under the arrowhead
    solve; `lowrank_free_for` -- the sizing the LU tiers and the
    lookahead shape use -- is still the case's largest behind-a-line region
    (523 buses on `case813nem`, line 10) plus `LOWRANK_FREE_UNITS`."""
    sizing = lowrank_free_for(nem["case"], nem["mon"])
    assert sizing == (LOWRANK_FREE_UNITS, 523)
    default = default_lowrank_free(nem["case"], nem["mon"])
    assert default == (151, 813)
    assert nem["rt_spec"]["lu_batching"] == "arrow"
    assert nem["rt_spec"]["kkt_route"] == f"lowrank+free({default[0]},{default[1]})+lu:arrow"
    assert nem["rt_spec"]["lowrank_free"] == default
    assert default_lowrank_free(nem["case"], nem["mon"], n_lookahead=2) == sizing
    assert nem["plain_spec"]["kkt_route"] == "lowrank"
    assert nem["dense_spec"]["kkt_route"] == "dense"
    assert lowrank_free_for(nem["case"], None) == (0, 0)
    assert default_lowrank_free(nem["case"], None) == (0, 0)


def test_day_ahead_shed_day_813nem():
    """T = 24, the day that sheds 230 MW: the same sizing brings the low-rank
    route to the dense route's result.  The day-ahead default stays ``(0, 0)``
    (bit-identity of every day-ahead product), so the sizing is passed here."""
    z = np.load(T24_DAYS)
    meta = json.loads(str(z["meta"]))
    case = scale_min_output(load_case(meta["case"]), float(meta["p_min_scale"]))
    mon = rated_lines(case)
    T, K = int(z["u"].shape[2]), meta["n_segments"]
    day = int(z["day"][0])
    demand_all, _a, _d = load_nem_demand(floor_mw=11500.0)
    offer_map, _asp = make_offer_map(case, K, T, kind="markup", markup_max=meta["markup_max"])
    n_units = len(np.asarray(case.unit_p_min))
    args = (jnp.broadcast_to(offer_map(jnp.asarray(z["action"][0])).astype(jnp.float64), (n_units, K, T)),
            jnp.asarray(z["u"][0]), jnp.asarray(demand_all[day], jnp.float64), jnp.asarray(z["p_init"][0]))
    kw = dict(n_segments=K, cap_scale=meta["cap_scale"], ramp_scale=meta["ramp_scale"],
              period_hours=meta["period_hours"], monitored_lines=mon)
    lr_clear, lr_spec = make_clearing(case, T, kkt="lowrank", lowrank_free=lowrank_free_for(case, mon), **kw)
    dn_clear, _s = make_clearing(case, T, kkt="dense", **kw)
    lr, dn = _as_np(jax.jit(lr_clear)(*args)), _as_np(jax.jit(dn_clear)(*args))
    assert abs(dn["shed"].sum() - meta["shed_mw_dense"][0]) < 1e-2, "the recorded day no longer sheds"
    assert lr_spec["kkt_route"].startswith(f"lowrank+free({LOWRANK_FREE_UNITS},523)+lu:")
    _assert_same_clearing(lr, dn, f"813nem T=24 day {day}")
