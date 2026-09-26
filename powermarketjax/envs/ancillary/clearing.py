"""Joint energy and reserve clearing for one period.

Energy and reserve are co-optimised in a single LP.  This operator is the
real-time clearing with reserve added, and the real-time clearing is the
day-ahead one at `n_periods=1`, so the layout follows `envs.day_ahead.clearing`:
the same variable ordering within the period block, row ordering, dual sign
convention, and phantom-quantity treatment of de-committed units.  Three kinds
of row are new, all of them inside the period:

    (CS)  sum_k g[i,k] + sum_j r[i,j] <= (pmax[i] - pmin[i]) * u[i]
    (RD)  sum_i r[i,j] + s_res[j] >= d_res[j]
    (RA)  0 <= r[i,j] <= R_up[i] * theta[j]

(RA) folds into the box and adds no row.  (CS) is the only constraint where
energy and reserve variables meet, so it is the single channel coupling the two
commodities, and its dual is the scarcity value of one megawatt of that unit's
capacity.  (RD) carries the reserve price: the requirement `d_res` is its
right-hand side, and the dual of that constraint is the clearing price of the
product.

The dense Newton path is used rather than the block-tridiagonal backend of
`envs.day_ahead.kkt`, which stays available for a multi-period look-ahead but
is not exercised here.

``monitored_lines`` is the day-ahead operator's modelling switch with the same
meaning: ``None`` carries every line's two rows, an index
array carries those lines only.  It exists here for a reason the day-ahead
market does not have.  `ipm.make_solver` floors every multiplier at
`ipm.SLACK_FLOOR`, so a row that can never bind still contributes
``s * SLACK_FLOOR`` to ``mu``; on ``case813nem`` the 1 271 lines registered at
1e6 MW leave ``s`` at 1e6 on 2 542 rows, and ``mu = sum(s * lam) / m`` cannot
fall below 2 542 * 1e6 * 1e-14 / 5 547 = 4.6e-9 whatever the dispatch does.
Measured 2026-09-16: those rows
carry 99.8% of the final ``mu`` and every truthful period reads above
`env.MU_TOL` (1e-9).  Dropping the rows that cannot bind is the fix; the
Newton system's order ``n + 1`` does not change, so this is a row-count change
and not a change of linear-algebra route -- `spec["kkt_route"]` reports
``"dense"`` either way.

Every price is a dual of this solve and nothing else.  The energy price is the
day-ahead market's three-term LMP formula, the reserve price of product j is
the dual of its (RD) row, and the capacity scarcity value is the dual of the
(CS) row.  No price is reconstructed after the fact.
"""
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np

from powermarketjax.solvers import ipm

from ..day_ahead import kkt_lowrank
from ..day_ahead.clearing import KKT_ROUTES, segment_costs

#: Value of lost load, \$/MWh.  Inherited from the day-ahead scenario, since
#: this market clears the same case and the same demand series.
VOLL = 10_000.0

#: Value of lost reserve must stay strictly below `VOLL`, or the clearing would
#: shed firm load to hold a reserve margin -- sacrificing a customer to guard
#: against an event that has not happened.  The constructor enforces this and
#: refuses a value at or above `VOLL`.
VOLR_MUST_BE_BELOW = VOLL

#: Newton steps for this market's joint solve.  Fewer risks a `mu` that has not
#: converged and an objective off by tens of percent.  A converging `mu`
#: beside a stalled `dual_residual` is the symptom of a regularisation-scale
#: problem, not of too few steps -- a caller should watch both.
MAX_ITER = 60

#: Primal-dual regularisation for this market's solve, passed at the call site
#: so it does not move the other markets' default.  Too large biases the
#: reserve and energy prices; too small erodes conditioning without silencing
#: a genuine infeasibility, which still drives `mu` far above the floor.
REG_COEF = 1e-20

#: Slack kept on any box whose two bounds would otherwise coincide -- e.g. a
#: de-committed unit's zero segment width, or a bus with no load's zero
#: sheddable quantity -- since `0 <= x <= 0` destroys the strict interior the
#: method needs.  Phantom quantities are at most `OFF_EPS` and are zeroed out.
OFF_EPS = 1e-9

#: The five places `OFF_EPS` is spent.  Four are upper bounds on a variable's
#: box and fail the same way if narrowed too far, by destroying the strict
#: interior.  The fifth, ``cs``, is the right-hand side of an inequality row
#: rather than a box, and its slack is held separately at `ipm.SLACK_FLOOR`, so
#: it goes interior-infeasible once `OFF_EPS` approaches that floor -- two
#: mechanisms, which is why one value for all five would hide whichever fails
#: first.
OFF_EPS_SITES = ("g", "r", "s", "sr", "cs")


def default_lowrank_free(case, monitored_lines, n_prod: int) -> Tuple[int, int]:
    """What `make_clearing` sizes the low-rank route's pivoted block to when
    not told: every column, ``(n_units, n_buses + n_prod)`` -- the units, the
    shed columns and the ``n_prod`` reserve-shortfall columns -- on a proper
    subset of the lines, ``(0, 0)`` when every line is carried (``None`` or
    an index array naming every line), which takes the dense route where the
    sizing is unused.  `tools/benchmark/run_rl_03.py` reads the same
    function, so the value it expects is the value in effect.  Every column
    free is the real-time market's default too: under the arrowhead
    solve the block's width is a linear cost and no region can be left out.
    """
    n_lines_all = int(np.asarray(case.PTDF).shape[0])
    if monitored_lines is None or np.unique(np.asarray(monitored_lines, np.int64)).size == n_lines_all:
        return (0, 0)
    return (len(np.asarray(case.unit_p_min)), int(case.n_nodes) + int(n_prod))


def _resolve_off_eps(off_eps):
    """One value per site, from a scalar or a per-site mapping."""
    if isinstance(off_eps, Mapping):
        missing = set(OFF_EPS_SITES) - set(off_eps)
        if missing:
            raise ValueError(
                f"off_eps mapping is missing {sorted(missing)}; it must give a "
                f"value for every one of {OFF_EPS_SITES}")
        return {k: float(off_eps[k]) for k in OFF_EPS_SITES}
    return {k: float(off_eps) for k in OFF_EPS_SITES}


def make_clearing(
    case,
    theta: Sequence[float],
    volr: float,
    n_segments: int = 1,
    cap_scale: float = 1.0,
    ramp_scale: float = 1.0,
    period_hours: float = 0.5,
    voll: float = VOLL,
    max_iter: int = MAX_ITER,
    dual_start: str = "cost_norm",
    reg_coef: float = REG_COEF,
    slack_floor: float = ipm.SLACK_FLOOR,
    off_eps=OFF_EPS,
    margin: float = 0.05,
    monitored_lines=None,
    freeze_mu: Optional[float] = None,
    stop_tol: Optional[Tuple[float, float]] = None,
    kkt: str = "auto",
    lowrank_free: Optional[Tuple[int, int]] = None,
    lu_batching: str = "auto",
) -> Tuple[Callable, Dict]:
    """Build the joint clearing operator for one case and one product ladder.

    Args:
        case: a `CaseData`; read for the unit registry, the network and the
            demand split, exactly as the day-ahead operator reads it.
        theta: response time of each product in hours, in product order;
            `(1/6, 1/2)` is the adopted pair.
        volr: penalty price on unmet requirement, \\$/MWh, which must be below
            `voll` or the clearing would shed load to hold reserve.  No default.
        n_segments: K, the number of offer segments per unit.
        cap_scale: line ratings are scaled by this.
        ramp_scale: registered ramp rates are scaled by this.  It enters (RA)
            as well as (RMP), so it also sets what the fleet can sell as
            reserve.
        period_hours: the period length, half an hour for this market.
        voll: penalty price on load shed, \\$/MWh.
        max_iter: Newton steps; see `MAX_ITER`.
        margin: fraction of each box kept clear at the starting point.
        monitored_lines: which line limits the LP carries.  ``None`` (the
            default) enforces every line, which is what every archive was
            produced on; an index array enforces those lines only and drops
            the other rows (see the module docstring for why that matters
            here).  `envs.day_ahead.clearing.rated_lines(case)` is the set
            whose limit the case publishes.  ``line_dual_up`` / ``line_dual_dn`` are reported over
            **all** lines either way, zero on a line not monitored.  With the
            default every operation below is the one it was before the
            argument existed.
        freeze_mu: `ipm.make_solver`'s merit gate, armed once ``mu`` is below
            this value; ``None`` (the default) is the solver as it always
            was.  On ``case813nem`` the fixed 60 steps drift the duals after
            convergence (module docstring of `solvers.ipm`); the value in
            force is stamped as ``spec["freeze_mu"]``.
        stop_tol: `ipm.make_solver`'s data-dependent trip count, ``(mu_tol,
            dual_tol)``: the loop stops at the first iterate under both and
            ``max_iter`` is the cap; ``None`` (the default) is the fixed
            count, to the bit.  On ``case813nem`` (rated rows) the market's
            own gate ``(1e-9, 1e-4)`` stops in 26-66 steps where the fixed 60
            walk off the plateau (2026-09-17); stamped as
            ``spec["stop_tol"]``.  ``freeze_mu`` and ``stop_tol`` may both
            be given; the market uses one.
        kkt: the linear-algebra route of the Newton system, one of
            `envs.day_ahead.clearing.KKT_ROUTES` (2026-09-17).
            ``"auto"``, the default, is the dense factorisation of `ipm`
            when every line is carried (``monitored_lines=None`` or an index
            array naming every line) and the low-rank route of
            `kkt_lowrank.make_kkt_ancillary` on a proper subset -- the same
            rule as the other two markets' operators.  So a caller passing a
            proper subset gets the low-rank route unless it says
            ``kkt="dense"``; the dense route on the same rows is the ruler
            the low-rank one is measured against
            (`tests/envs/ancillary/test_lowrank_cells_l2.py`).  On the
            default rows every operation is the one it was before the
            argument existed.
        lowrank_free: the low-rank route's ``(units, diagonal columns)`` kept
            in its pivoted block (`kkt_lowrank`, "Free columns"); ``None``
            takes every column on the low-rank route
            (``(n_units, n_buses + n_prod)``) and ``(0, 0)`` on the dense
            one.  ``(0, 0)`` on the low-rank route is the plain Schur
            solve, which on a period that sheds behind a binding line is
            wrong; it is kept as the negative control.
        lu_batching: how that block is solved under `vmap`
            (`kkt_lowrank.LU_BATCHINGS`); ``"auto"`` is the local-border
            arrowhead on this operator's single-period shape.  Both are
            stamped as ``spec["lowrank_free"]`` / ``spec["lu_batching"]``
            and, when in force, into ``spec["kkt_route"]`` as
            ``lowrank+free(u,c)+lu:arrow``, the form the real-time market
            stamps.

    Returns:
        ``(clear, spec)``.  ``clear(offer, offer_res, u, demand, d_res, p_prev)``
        is pure and jittable with

            offer      (n_units, n_segments)  as-bid energy price, non-decreasing
            offer_res  (n_units, n_prod)      as-bid reserve price, nonnegative
            u          (n_units,)             fixed commitment of this period
            demand     ()                     realised total system demand, MW
            d_res      (n_prod,)              published requirement, MW
            p_prev     (n_units,)             previous period output, MW

        and returns a dict carrying ``award``, ``reserve`` and ``shed``, the
        prices ``lmp``, ``reserve_price`` and ``capacity_dual``, the unmet
        requirement ``reserve_shortfall``, and the solver diagnostics ``mu`` and
        ``dual_residual``.
    """
    eps = _resolve_off_eps(off_eps)

    if not jax.config.jax_enable_x64:
        # Same guard as the other two clearing operators: under float32 this
        # method returns a finite, plausible-looking solution whose duals are
        # meaningless, and the duals are the prices here.
        raise RuntimeError(
            "the ancillary clearing requires float64: set "
            "jax.config.update('jax_enable_x64', True) before building it")
    theta = np.asarray(theta, np.float64)
    if theta.ndim != 1 or len(theta) == 0:
        raise ValueError(f"theta must be a non-empty 1-D sequence, got {theta!r}")
    if (theta <= 0.0).any():
        raise ValueError(f"a response time must be positive: {theta!r}")
    if not 0.0 < volr < voll:
        # Above VOLL the program would curtail firm load to hold a reserve
        # margin, which inverts the order this market applies.
        raise ValueError(
            f"volr={volr} must satisfy 0 < volr < voll={voll}; above voll "
            "the clearing sheds load in order to hold reserve")
    if kkt not in KKT_ROUTES:
        raise ValueError(f"kkt must be one of {KKT_ROUTES}, got {kkt!r}")

    K, P = n_segments, len(theta)
    pmin = np.asarray(case.unit_p_min, np.float64)
    pmax = np.asarray(case.unit_p_max, np.float64)
    unit_bus = np.asarray(case.unit_node_idx, np.int64)
    width, _ = segment_costs(case, K)
    n_units, n_buses = len(pmin), int(case.n_nodes)
    PTDF_all = np.asarray(case.PTDF, np.float64)
    n_lines_all = PTDF_all.shape[0]
    F_all = np.asarray(case.line_cap, np.float64) * cap_scale
    if monitored_lines is None:
        # the same three objects the operator always built from: no indexing,
        # so the default path is the old path to the bit
        mon, PTDF, F = None, PTDF_all, F_all
    else:
        mon = np.unique(np.asarray(monitored_lines, np.int64))
        if mon.size == 0 or mon.min() < 0 or mon.max() >= n_lines_all:
            raise ValueError(
                f"monitored_lines must be a non-empty subset of 0..{n_lines_all - 1}")
        PTDF, F = PTDF_all[mon], F_all[mon]
    n_lines = PTDF.shape[0]
    route = kkt if kkt != "auto" else (
        # the same rule as the day-ahead operator: the low-rank route pays
        # off only on a proper subset of the lines
        "dense" if mon is None or mon.size == n_lines_all else "lowrank")

    n_g, n_r = n_units * K, n_units * P
    n = n_g + n_r + n_buses + P
    g0, r0, s0, sr0 = 0, n_g, n_g + n_r, n_g + n_r + n_buses

    # nodal demand split, read exactly as the day-ahead operator reads it:
    # `case29gb` registers participation factors rather than MW
    node_pd = np.asarray(case.node_pd, np.float64)
    if node_pd.sum() > 1e-6:
        share = node_pd / node_pd.sum()
    else:
        load_bus = np.asarray(case.load_node_idx, np.int64)
        w = np.asarray(case.load_d_max, np.float64)
        share = np.zeros(n_buses)
        np.add.at(share, load_bus, w)
        share = share / share.sum()

    # ramp limits are a fraction of p_max per hour, not MW
    rate = np.asarray(case.unit_ramp_up, np.float64) * pmax * ramp_scale   # MW/h
    ramp_up = rate * period_hours
    ramp_dn = np.asarray(case.unit_ramp_down, np.float64) * pmax * period_hours * ramp_scale
    # (RA): what one unit may sell of product j within its response time.  The
    # products are otherwise interchangeable inside this operator -- they share
    # the (CS) row, the box shape, the shortfall price `volr` and the
    # settlement -- so `theta` is the only property of a product built in here,
    # and everything else that separates them arrives as data in `d_res`.
    res_cap = rate[:, None] * theta[None, :]                     # (n_units, P)

    # bus injection from the segment variables, and segment -> unit sums
    seg_bus = np.zeros((n_buses, n_g))
    seg_bus[np.repeat(unit_bus, K), np.arange(n_g)] = 1.0
    unit_bus_mat = np.zeros((n_buses, n_units))
    unit_bus_mat[unit_bus, np.arange(n_units)] = 1.0
    seg_of_unit = np.zeros((n_units, n_g))
    seg_of_unit[np.repeat(np.arange(n_units), K), np.arange(n_g)] = 1.0
    res_of_unit = np.zeros((n_units, n_r))
    res_of_unit[np.repeat(np.arange(n_units), P), np.arange(n_r)] = 1.0
    prod_of_res = np.zeros((P, n_r))
    prod_of_res[np.tile(np.arange(P), n_units), np.arange(n_r)] = 1.0

    zeros = lambda rows, cols: np.zeros((rows, cols))
    # line flow reads the segment and shed variables only
    M = np.hstack([PTDF @ seg_bus, zeros(n_lines, n_r), PTDF, zeros(n_lines, P)])
    # (RMP) reads the segment variables only, since p = pmin*u + sum_k g
    S = np.hstack([seg_of_unit, zeros(n_units, n_r + n_buses + P)])
    # (CS) reads a unit's segments and all of its reserve
    CS = np.hstack([seg_of_unit, res_of_unit, zeros(n_units, n_buses + P)])
    # (RD) is written as a <= row: -sum_i r[i,j] - s_res[j] <= -d_res[j]
    RD = np.hstack([zeros(P, n_g), -prod_of_res, zeros(P, n_buses), -np.eye(P)])
    identity = np.eye(n)
    G = np.vstack([M, -M, S, -S, CS, RD, identity, -identity])

    row0 = dict(line_up=0, line_dn=n_lines, ramp_up=2 * n_lines,
                ramp_dn=2 * n_lines + n_units, cs=2 * n_lines + 2 * n_units,
                rd=2 * n_lines + 3 * n_units,
                box_hi=2 * n_lines + 3 * n_units + P,
                box_lo=2 * n_lines + 3 * n_units + P + n)
    m = G.shape[0]
    assert m == 2 * n_lines + 3 * n_units + P + 2 * n, (m, row0)

    spec = dict(K=K, n_prod=P, n_units=n_units, n_buses=n_buses, n_l=n_lines,
                n=n, m=m, n_eq=1, col0=dict(g=g0, r=r0, s=s0, s_res=sr0),
                row0=row0, PTDF=PTDF, unit_bus=unit_bus, p_min=pmin, p_max=pmax,
                theta=theta, res_cap=res_cap, demand_share=share,
                cap_scale=cap_scale, ramp_scale=ramp_scale,
                period_hours=period_hours, voll=voll, volr=volr,
                max_iter=max_iter, dual_start=dual_start, reg_coef=reg_coef,
                slack_floor=slack_floor)
    # What the LP carries and how its Newton system is solved, stamped whatever
    # the caller passed, so a driver reads its products' stamp off the operator
    # (`evaluation.effective_monitored_stamp`) and not off its own command line.
    # Until 2026-09-17 this operator had one route and the stamp was the constant
    # "dense"; the default rows still read exactly that.  The stamp takes the
    # real-time market's form, "lowrank+free(u,c)+lu:arrow", on the low-rank
    # route with free columns, and "lowrank" for the plain Schur solve.
    spec["monitored_lines"], spec["n_lines"] = mon, n_lines_all
    if lowrank_free is None:
        lowrank_free = default_lowrank_free(case, mon, P) if route == "lowrank" else (0, 0)
    spec["lowrank_free"] = (int(lowrank_free[0]), int(lowrank_free[1]))
    spec["lu_batching"] = kkt_lowrank.resolve_lu_batching(lu_batching, arrowhead=True)
    spec["kkt_route"] = (route if route == "dense" or spec["lowrank_free"] == (0, 0)
                         else f"lowrank+free({spec['lowrank_free'][0]},{spec['lowrank_free'][1]})"
                              f"+lu:{spec['lu_batching']}")
    spec["freeze_mu"] = None if freeze_mu is None else float(freeze_mu)
    spec["stop_tol"] = None if stop_tol is None else (float(stop_tol[0]), float(stop_tol[1]))

    A_eq = np.zeros((1, n))
    A_eq[0, g0: g0 + n_g] = 1.0                    # generation above must-run
    A_eq[0, s0: s0 + n_buses] = 1.0                # plus what is shed
    # `reg_coef`, `max_iter` and `dual_start` are one calibration, not three
    # independent knobs, and the regularisation scale is not scale free.
    # Passing it here keeps this market's value local: the default of
    # `make_solver` is what the other markets calibrated for themselves and
    # must not move on their behalf.
    if route == "dense":
        solver = ipm.make_solver(n, m, max_iter, dual_start=dual_start, n_eq=1,
                                 reg_coef=reg_coef, slack_floor=slack_floor,
                                 freeze_mu=freeze_mu, stop_tol=spec["stop_tol"])
    else:
        # the same Newton system through `kkt_lowrank` and the row products
        # without the matrix; `G` below is then a placeholder to `solve`
        solver = ipm.make_solver(n, m, max_iter, dual_start=dual_start, n_eq=1,
                                 reg_coef=reg_coef, slack_floor=slack_floor,
                                 freeze_mu=freeze_mu, stop_tol=spec["stop_tol"],
                                 ops=kkt_lowrank.make_ops_ancillary(spec, M),
                                 kkt=kkt_lowrank.make_kkt_ancillary(
                                     spec, M, S, CS, RD, A_eq[0], n_free=spec["lowrank_free"],
                                     lu_batching=spec["lu_batching"]))

    G_j, A_j = jnp.asarray(G), jnp.asarray(A_eq)
    PTDF_j = jnp.asarray(PTDF)
    F_j = jnp.asarray(F)
    mon_j = None if mon is None else jnp.asarray(mon)
    pmin_j, pmax_j = jnp.asarray(pmin), jnp.asarray(pmax)
    width_j = jnp.asarray(width)
    res_cap_j = jnp.asarray(res_cap)
    share_j = jnp.asarray(share)
    ramp_up_j, ramp_dn_j = jnp.asarray(ramp_up), jnp.asarray(ramp_dn)
    UB_j = jnp.asarray(unit_bus_mat)

    def clear(offer, offer_res, u, demand, d_res, p_prev):
        """Clear energy and reserve jointly for one period.

        The six arguments, their shapes and their units are listed in
        `make_clearing`'s ``Returns``.

        Returns:
            A dict holding the primal quantities ``award``, ``reserve``,
            ``shed`` and ``reserve_shortfall`` in MW, the objective value
            ``z``, the prices ``lmp``, ``reserve_price`` and ``capacity_dual``,
            the three duals ``line_dual_up``, ``line_dual_dn`` and ``shed_dual``
            the money-balance identity needs from this same solve, and the
            diagnostics ``mu``, ``dual_residual``, ``primal_residual``,
            ``eq_residual`` and ``ineq_residual``.

            Every price is in \\$/MWh, already divided by ``period_hours``
            below, so a caller settling with ``Delta * price * quantity`` must
            not divide again.  The prices are the duals of this solve and of
            nothing else.
        """
        d = share_j * demand                                     # (n_buses,)
        must_run = pmin_j * u                                    # (n_units,)
        on = u > 0.0

        # box upper bounds.  Everything a de-committed unit could sell is zero
        # and every bound below is widened to OFF_EPS for the strict interior.
        # every segment of a unit has the same width, so the per-unit width is
        # repeated across the segment axis rather than indexed by it
        hi_g = jnp.repeat(jnp.where(on, width_j, eps["g"])[:, None], K,
                          axis=1).reshape(n_g)
        hi_r = jnp.where(on[:, None], res_cap_j, eps["r"]).reshape(n_r)
        hi_s = jnp.maximum(d, eps["s"])
        # the slack needs a finite box, and taking d_res for it would make that
        # bound active exactly when no reserve clears, which puts a bound dual
        # into the reserve price the way the shed bound puts rho into the
        # energy price.  Doubling keeps it slack at the solution instead.
        hi_sr = 2.0 * jnp.maximum(d_res, eps["sr"])
        hi = jnp.concatenate([hi_g, hi_r, hi_s, hi_sr])

        # (E): the segments and the shed carry demand net of must-run output
        b_eq = jnp.atleast_1d(jnp.sum(d) - jnp.sum(must_run))

        line_rhs = PTDF_j @ (d - UB_j @ must_run)                # (n_lines,)
        # (RMP) with the start-up and shut-down allowances of the day-ahead
        # market: a unit switching on needs to reach pmin and
        # a unit switching off needs to leave whatever the previous period
        # chose, so the two allowances are pmin and pmax and are not symmetric.
        # The previous commitment is read off the previous output rather than
        # carried in the state: a unit that produced was on, and a unit at
        # exactly zero output reads as off.
        u_prev = (p_prev > 0.0).astype(u.dtype)
        start = jnp.maximum(u - u_prev, 0.0) * pmin_j
        stop = jnp.maximum(u_prev - u, 0.0) * pmax_j
        ramp = jnp.concatenate([ramp_up_j + start - must_run + p_prev,
                                ramp_dn_j + stop + must_run - p_prev])
        # (CS): capacity above must-run, which is what the segments and the
        # reserve share.  A de-committed unit gets OFF_EPS for the interior.
        cs = jnp.maximum((pmax_j - pmin_j) * u, eps["cs"])

        h = jnp.concatenate([F_j + line_rhs, F_j - line_rhs, ramp, cs,
                             -d_res, hi, jnp.zeros(n)])
        c = period_hours * jnp.concatenate([
            offer.reshape(n_g), offer_res.reshape(n_r),
            jnp.full((n_buses,), voll), jnp.full((P,), volr)])

        # strictly interior, and exact on the balance row
        lo_x, up_x = margin * hi, (1.0 - margin) * hi
        span = jnp.maximum(up_x - lo_x, 1e-9)
        x0 = lo_x + 0.5 * span
        # move only the two blocks the balance row reads
        reads = jnp.concatenate([jnp.ones(n_g), jnp.zeros(n_r),
                                 jnp.ones(n_buses), jnp.zeros(P)])
        gap = b_eq[0] - jnp.sum(x0 * reads)
        x0 = x0 + gap * reads / jnp.sum(reads)
        x0 = jnp.clip(x0, lo_x, up_x)

        x, lam, nu, mu, r1, r3 = solver(c, A_j, b_eq, G_j, h, x0)

        # duals to prices.  `lam >= 0` for `G x <= h`; the negation below is the
        # sign convention the day-ahead operator fixed, and a missing negation
        # stays invisible until a line binds.
        # Every multiplier below is a derivative of the objective, and the
        # objective carries the period length, so a raw multiplier is dollars
        # per megawatt held for one period.  Dividing by `period_hours` puts
        # every price in \$/MWh, which is what the settlement writes as
        # `Delta * lambda * quantity`.  The day-ahead operator omits this step
        # because its period is one hour, so the factor is invisible there and
        # would silently halve every price of this market.
        per_mwh = 1.0 / period_hours
        line_up = -lam[row0["line_up"]: row0["line_up"] + n_lines] * per_mwh
        line_dn = -lam[row0["line_dn"]: row0["line_dn"] + n_lines] * per_mwh
        rho = -lam[row0["box_hi"] + s0: row0["box_hi"] + s0 + n_buses] * per_mwh
        lmp = -nu[0] * per_mwh + (line_up - line_dn) @ PTDF_j + rho
        # (RD) was written with negated coefficients, so its multiplier is
        # already the derivative of the objective with respect to the
        # requirement, which is the price of the product.
        reserve_price = lam[row0["rd"]: row0["rd"] + P] * per_mwh
        # (CS) has no interior for a de-committed unit, whose right-hand side is
        # the `OFF_EPS` allowance rather than a real capacity, so its multiplier
        # there is an artefact of that allowance and carries no economic
        # content.  It is zeroed for the same reason the phantom quantities are.
        capacity_dual = lam[row0["cs"]: row0["cs"] + n_units] * per_mwh * u

        if mon_j is None:
            line_up_all, line_dn_all = line_up, line_dn
        else:
            # the duals are reported over every line; an unmonitored line has none
            line_up_all = jnp.zeros((n_lines_all,)).at[mon_j].set(line_up)
            line_dn_all = jnp.zeros((n_lines_all,)).at[mon_j].set(line_dn)

        g = x[g0: g0 + n_g].reshape(n_units, K)
        r = x[r0: r0 + n_r].reshape(n_units, P)
        # multiplying by u drops the OFF_EPS phantom quantities of a
        # de-committed unit, in both commodities
        award = (must_run + jnp.sum(g, axis=1)) * u
        reserve = r * u[:, None]
        shed = jnp.where(d > 0.0, x[s0: s0 + n_buses], 0.0)
        shortfall = x[sr0: sr0 + P]
        # Primal feasibility of the returned point.  `mu` and `dual_residual`
        # between them do not cover this: the complementarity gap of the whole
        # problem is `mu * m`, far below the objective error a violated
        # balance row can produce, since that error is the violation times a
        # price and this market prices scarcity at VOLL.  A caller that reads
        # only `mu` cannot see it.
        eq_residual = jnp.max(jnp.abs(A_j @ x - b_eq))
        ineq_residual = jnp.max(jnp.maximum(G_j @ x - h, 0.0))
        return dict(award=award, reserve=reserve, shed=shed,
                    reserve_shortfall=shortfall, z=jnp.dot(c, x), lmp=lmp,
                    eq_residual=eq_residual, ineq_residual=ineq_residual,
                    reserve_price=reserve_price, capacity_dual=capacity_dual,
                    line_dual_up=-line_up_all, line_dual_dn=-line_dn_all,
                    shed_dual=-rho, mu=mu, dual_residual=r1,
                    primal_residual=r3)

    return clear, spec
