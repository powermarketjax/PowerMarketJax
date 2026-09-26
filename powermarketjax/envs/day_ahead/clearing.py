"""Day-ahead clearing operator: the fixed-commitment dispatch, over T coupled
periods.

Array in, array out, pure.  Given offers, a fixed integer commitment, realised
demand and the previous day's final output, it returns the `award` per unit and
period and the `lmp` per bus and period.  This is the fixed-commitment
security-constrained economic dispatch whose duals are the prices, so it is also
the real-time market's clearing engine.

The relaxed unit commitment and its rounding to integers are **not implemented**
here.  `u` is a parameter, supplied by the caller.

Output is the committed minimum plus the accepted segments,

    p[i,t] = p_min[i] * u[i,t] + sum_k g[i,k,t]

which makes the lower bound of the capacity constraint hold by construction, so
it never appears as a row.  Variables are ordered **block per period**, which is
what makes the Newton system block-tridiagonal (see `kkt`).

Prices carry three terms, not two:

    lmp[n,t] = lambda[t] - sum_l (mu+ - mu-)[l,t] PTDF[l,n] - rho[n,t]

`rho` is the dual of the shed bound `s <= d`.  Demand enters the objective in
three places -- the balance rhs, the line rhs through PTDF@d, and that bound --
and dropping the third overstates the price at any bus whose load is fully shed.
It is exactly zero when nothing sheds to its cap.  The money-balance identity
needs the matching `rho . P` rent term; applying one correction without the
other turns the strongest available check into a false alarm.

Scenario parameters `cap_scale` and `ramp_scale` are **not defaults**: at the
registered values `case29gb` neither congests nor binds on ramp, so both must be
declared explicitly.
"""
from typing import Callable, Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np

from powermarketjax.solvers import ipm

from . import kkt as kkt_mod
from . import kkt_lowrank

#: Routes for the Newton system.  ``"auto"`` is the dense block-tridiagonal
#: sweep of `kkt` when every line is monitored -- ``monitored_lines=None`` or an
#: index array naming every line -- and the low-rank Schur route of
#: `kkt_lowrank` when `make_clearing` is given a proper subset; the other two
#: force one route, which is how the two are compared on one LP.  The route
#: built is reported as ``spec["kkt_route"]``.
KKT_ROUTES = ("auto", "dense", "lowrank")

#: A line rated at or above this many MW carries no published limit:
#: `case813nem` registers 1e6 MW on the 1 271 lines the source gives no rating
#: for, against 39 GW installed, so
#: no dispatch can reach it.  `rated_lines` is the monitored set that keeps
#: every constraint that can bind.
UNRATED_MW = 1e5


def rated_lines(case, unrated_mw: float = UNRATED_MW) -> np.ndarray:
    """Indices of the lines whose rating is below ``unrated_mw``, i.e. the
    lines whose limit the case actually publishes.  On ``case29gb`` and
    ``case73rts`` that is every line; on ``case813nem`` it is 7 of 1 278."""
    return np.where(np.asarray(case.line_cap, np.float64) < unrated_mw)[0]

#: Value of lost load, the penalty price on shed load.  A cost parameter -- it
#: must never appear in a settlement expression.
VOLL = 10_000.0

#: Newton steps.  Chosen for **dual** accuracy, because the LMP this market
#: settles on is a dual; other markets that consult no dual calibrate their own
#: value.  A converged solve leaves `mu` near 1e-11; short of this the LMP is
#: not accurate as a dual.
MAX_ITER = 60

#: Slack kept on any variable whose bounds would otherwise coincide.  Exactly
#: zero destroys the interior point method's strict interior and the dual of
#: that bound blows up.  Two places need it: a de-committed unit's segments,
#: whose upper and lower bound both sit at zero; and the shed variable of a
#: zero-demand bus, where ``s <= d`` becomes ``s <= 0`` and its dual enters the
#: price directly as rho.  The resulting phantom quantities are <= OFF_EPS per
#: variable and are zeroed out of `award` and `shed`.
OFF_EPS = 1e-3


def segment_costs(case, n_segments: int, monotone: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Segment widths and `segment_cost`: mean true marginal cost per segment.

    `unit_cost_a/b/c` are **marginal**-cost coefficients, not MATPOWER total-cost
    ones; the integral below is the correct total cost.

    `monotone` takes the running maximum, which offers must satisfy and which the
    action map supplies by construction.  It is not cosmetic: units with
    non-monotone true marginal cost get an envelope that differs from that raw
    cost, so a larger `n_segments` is not automatically more faithful.
    """
    pmin = np.asarray(case.unit_p_min, np.float64)
    pmax = np.asarray(case.unit_p_max, np.float64)
    a = np.asarray(case.unit_cost_a, np.float64)
    b = np.asarray(case.unit_cost_b, np.float64)
    c = np.asarray(case.unit_cost_c, np.float64)
    # coefficients keep a unit axis so they broadcast per unit, not per segment
    total_cost = lambda p: a[:, None] / 3 * p ** 3 + b[:, None] / 2 * p ** 2 + c[:, None] * p

    width = np.maximum(pmax - pmin, 1e-9) / n_segments
    edge = pmin[:, None] + np.arange(n_segments + 1)[None, :] * width[:, None]
    cost = (total_cost(edge[:, 1:]) - total_cost(edge[:, :-1])) / width[:, None]
    if monotone:
        cost = np.maximum.accumulate(cost, axis=1)
    return width, cost


def make_clearing(
    case,
    n_periods: int,
    n_segments: int = 1,
    cap_scale: float = 1.0,
    ramp_scale: float = 1.0,
    period_hours: float = 1.0,
    max_iter: int = MAX_ITER,
    dual_start: str = "cost_norm",
    margin: float = 0.05,
    monitored_lines=None,
    kkt: str = "auto",
    lowrank_free: Tuple[int, int] = (0, 0),
    lu_batching: str = "auto",
    reg_coef: Optional[float] = None,
    stop_tol: Optional[Tuple[float, float]] = None,
) -> Tuple[Callable, Dict]:
    """Build the clearing operator for one fixed case and horizon.

    Returns ``(clear, spec)`` where ``clear(offer, u, demand, p_init)`` is pure
    and jittable, and ``spec`` carries the static shapes plus the arrays the
    caller needs to interpret the result.

    ``monitored_lines`` is the modelling choice of which line limits the LP
    carries: ``None`` (the default) enforces every line, an index array
    enforces those lines only and drops the other rows from the LP.  Dropping
    a line that can bind changes the problem; dropping one that cannot -- on
    ``case813nem`` the 1 271 lines rated 1e6 MW against 39 GW of capacity --
    leaves the optimum untouched.  ``kkt`` picks the linear-algebra route,
    `KKT_ROUTES`; the low-rank route solves the same Newton system to the
    dense route's accuracy (`kkt_lowrank`), so ``kkt`` never changes the LP.
    With both at their defaults every operation below is the one it was
    before the two arguments existed.  ``lowrank_free`` is the low-rank
    route's ``(units, shed columns)`` kept in its pivoted block
    (`kkt_lowrank`, "Free columns"); ``(0, 0)`` is the plain Schur route the
    day-ahead products were made on, and the real-time wrapper passes its
    own sizing.  Ignored on the dense route; reported as
    ``spec["lowrank_free"]`` on both.  ``lu_batching`` is how that block is
    solved under `vmap` (`kkt_lowrank.LU_BATCHINGS`; ``"auto"`` is the
    arrowhead solve at ``n_periods * n_segments == 1``, else a sequential LU
    on CPU and a batched one on GPU), resolved at build time into
    ``spec["lu_batching"]`` and into the stamp.  ``line_dual_up`` / ``line_dual_dn`` are
    always ``(n_periods, n_lines)`` over **all** lines, zero on the ones not
    monitored, so the two paths can be mixed by a caller.

    ``reg_coef`` and ``stop_tol`` go straight to `ipm.make_solver` and are stamped
    as ``spec["reg_coef"]`` (the value in force, `ipm.REG_COEF` when ``None``) and
    ``spec["stop_tol"]`` (``None`` = the fixed trip count).  They are separate keys,
    not part of ``kkt_route``: the route names how the Newton system is solved,
    these name how the loop is regularised and when it stops.  ``None`` for both
    is the operator as it always was, to the bit; a case that needs another
    regulariser passes it here (2026-09-17: ``case813nem`` at T=1 on the
    rated rows stalls at a stationarity residual of 5e-3 under 1e-14 and reaches
    1e-11 under 1e-16 -- `solvers.ipm`'s module docstring has the mechanism).

        offer   (n_units, n_segments, n_periods)  as-bid price, non-decreasing
                                       in segment.  Offers are per delivery
                                       period, so a unit may bid the peak
                                       differently from the trough; a day-flat
                                       offer is that array with its last axis
                                       broadcast.
        u       (n_units, n_periods)   fixed integer commitment
        demand  (n_periods,)           realised total system demand, MW
        p_init  (n_units,)             previous period's output, the day carry

    ``clear`` returns a dict with ``award`` (n_units, n_periods), ``lmp``
    (n_periods, n_buses), ``shed`` (n_periods, n_buses), plus ``mu`` and
    ``dual_residual`` as the per-day solver diagnostics.  ``mu`` is the
    convergence check: a converged solve leaves it near 1e-11.
    """
    if not jax.config.jax_enable_x64:
        # float32 does not merely lose accuracy here, it loses the duals: the
        # KKT condition number reaches extreme values and the errors are
        # chaotic.  The failure is not loud -- mu stays small and the LMPs stay
        # finite and plausible-looking -- so a caller who does not check mu gets
        # a wrong reward with no warning.
        raise RuntimeError(
            "day-ahead clearing requires float64: set "
            "jax.config.update('jax_enable_x64', True) before building it")

    if kkt not in KKT_ROUTES:
        raise ValueError(f"kkt must be one of {KKT_ROUTES}, got {kkt!r}")
    T, K = n_periods, n_segments
    pmin = np.asarray(case.unit_p_min, np.float64)
    pmax = np.asarray(case.unit_p_max, np.float64)
    unit_bus = np.asarray(case.unit_node_idx, np.int64)
    width, _ = segment_costs(case, K)
    n_units, n_buses = len(pmin), int(case.n_nodes)
    PTDF_all = np.asarray(case.PTDF, np.float64)
    n_lines_all = PTDF_all.shape[0]
    F_all = np.asarray(case.line_cap, np.float64) * cap_scale
    if monitored_lines is None:
        mon, PTDF, F = None, PTDF_all, F_all
    else:
        mon = np.unique(np.asarray(monitored_lines, np.int64))
        if mon.size == 0 or mon.min() < 0 or mon.max() >= n_lines_all:
            raise ValueError(
                f"monitored_lines must be a non-empty subset of 0..{n_lines_all - 1}")
        PTDF, F = PTDF_all[mon], F_all[mon]
    route = kkt if kkt != "auto" else (
        # the low-rank route pays off only when the monitored set is a proper
        # subset: on `case29gb` / `case73rts` `rated_lines` is every line and the
        # capacitance matrix would be T * (n_lines + 1) wide (measured 13-22x
        # slower than the sweep on CPU)
        "dense" if mon is None or mon.size == n_lines_all else "lowrank")
    n_lines = PTDF.shape[0]
    nb = n_units * K + n_buses

    # nodal demand split: `case29gb`'s load_d_max sums to 1.0, i.e. it is
    # participation factors rather than MW
    node_pd = np.asarray(case.node_pd, np.float64)
    if node_pd.sum() > 1e-6:
        share = node_pd / node_pd.sum()
    else:
        load_bus = np.asarray(case.load_node_idx, np.int64)
        w = np.asarray(case.load_d_max, np.float64)
        share = np.zeros(n_buses); np.add.at(share, load_bus, w)
        share = share / share.sum()

    # per-period blocks: bus injection from segments, and segment->unit sum
    seg_bus = np.zeros((n_buses, n_units * K))
    seg_bus[np.repeat(unit_bus, K), np.arange(n_units * K)] = 1.0
    Mblk = np.hstack([PTDF @ seg_bus, PTDF])                 # (n_lines, nb)
    Sblk = np.zeros((n_units, nb))
    Sblk[np.repeat(np.arange(n_units), K), np.arange(n_units * K)] = 1.0
    unit_bus_mat = np.zeros((n_buses, n_units))
    unit_bus_mat[unit_bus, np.arange(n_units)] = 1.0

    # ramp limits are a FRACTION OF p_max PER HOUR, not MW
    ramp_up = np.asarray(case.unit_ramp_up, np.float64) * pmax * period_hours * ramp_scale
    ramp_dn = np.asarray(case.unit_ramp_down, np.float64) * pmax * period_hours * ramp_scale

    spec = dict(T=T, K=K, n_units=n_units, n_buses=n_buses, n_l=n_lines, nb=nb,
                n=nb * T, ramp_row0=2 * n_lines * T, n_ramp_rows=2 * n_units * T,
                PTDF=PTDF, unit_bus=unit_bus, p_min=pmin, p_max=pmax,
                demand_share=share, cap_scale=cap_scale, ramp_scale=ramp_scale)
    spec["m"] = spec["ramp_row0"] + spec["n_ramp_rows"] + 2 * spec["n"]
    # What the LP carries and how its Newton system is solved, written whatever
    # the flags were: a driver stamps its products off these, not off its own
    # command line.  `monitored_lines`
    # is None when every line's row is carried, as `make_clearing` was given.
    spec["monitored_lines"], spec["n_lines"] = mon, n_lines_all
    spec["lowrank_free"] = (int(lowrank_free[0]), int(lowrank_free[1]))
    # the stamp names the free-column sizing when it is on, so a product made
    # on the pivoted-block variant is distinguishable from one made on the
    # plain Schur route: "lowrank+free(8,523)" against "lowrank".  The sizing
    # is per period: a day that sheds in several periods needs roughly the
    # region's bus count times the number of shedding periods (813nem, T=24,
    # 4 047 MW shed: (8,523) fails, (8,1046) matches the dense route), and a shortfall shows only in
    # `dual_residual` / `mu`, so a day-ahead run that starts shedding is the
    # signal to size and switch this on -- it stays (0, 0) there until then.
    # ``auto`` is the arrowhead solve on the single-period, single-segment
    # shape and a platform choice of LU batching elsewhere; unused,
    # and therefore stamped nowhere, when the sizing is (0, 0)
    spec["lu_batching"] = kkt_lowrank.resolve_lu_batching(lu_batching, arrowhead=(T * K == 1))
    spec["kkt_route"] = (route if route == "dense" or spec["lowrank_free"] == (0, 0)
                         else f"lowrank+free({spec['lowrank_free'][0]},{spec['lowrank_free'][1]})"
                              f"+lu:{spec['lu_batching']}")
    spec["reg_coef"] = float(ipm.REG_COEF if reg_coef is None else reg_coef)
    spec["stop_tol"] = None if stop_tol is None else (float(stop_tol[0]), float(stop_tol[1]))

    A_eq = np.zeros((T, spec["n"]))
    for t in range(T):
        A_eq[t, t * nb: (t + 1) * nb] = 1.0

    Fj, Aj = jnp.asarray(F), jnp.asarray(A_eq)
    Mj, Sj, PTDFj = jnp.asarray(Mblk), jnp.asarray(Sblk), jnp.asarray(PTDF)
    UBj, pminj, widthj = jnp.asarray(unit_bus_mat), jnp.asarray(pmin), jnp.asarray(width)
    pmaxj = jnp.asarray(pmax)
    sharej, rupj, rdnj = jnp.asarray(share), jnp.asarray(ramp_up), jnp.asarray(ramp_dn)
    if route == "dense":
        kkt_pair = kkt_mod.make_kkt(spec, Mj, Sj)
    else:
        kkt_pair = kkt_lowrank.make_kkt_sced(spec, Mblk, Sblk, n_free=spec["lowrank_free"],
                                             lu_batching=spec["lu_batching"])
    solver = ipm.make_solver(spec["n"], spec["m"], n_eq=T, max_iter=max_iter,
                             ops=kkt_mod.make_ops(spec, Mj, Sj),
                             kkt=kkt_pair,
                             dual_start=dual_start,
                             reg_coef=spec["reg_coef"], stop_tol=spec["stop_tol"])
    monj = None if mon is None else jnp.asarray(mon)
    box0 = spec["ramp_row0"] + spec["n_ramp_rows"]
    _unused_G = jnp.zeros((1, 1))

    def clear(offer, u, demand, p_init):
        """Clear one market day: offers and a commitment in, awards and prices out.

        Args:
            offer: ``(n_units, n_segments, n_periods)`` as-bid price in \\$/MWh,
                non-decreasing in the segment index.
            u: ``(n_units, n_periods)`` fixed integer commitment, step 3's parameter.
            demand: ``(n_periods,)`` realised total system demand in MW.
            p_init: ``(n_units,)`` output in the period before the first, i.e. the
                day boundary the ramp rows of period 0 are measured against.

        Returns:
            A dict of ``award`` ``(n_units, n_periods)`` in MW, ``lmp`` and ``shed``
            ``(n_periods, n_buses)``, the solver diagnostics ``mu``,
            ``dual_residual`` and ``primal_residual``, and three duals --
            ``line_dual_up`` and ``line_dual_dn`` ``(n_periods, n_lines)`` and
            ``shed_dual`` ``(n_periods, n_buses)``.  ``mu`` is the convergence
            check; the duals are the solve's own, never reconstructed afterwards.
        """
        # nodal demand, and the part of output the commitment already fixes.  The
        # segment variables carry only what sits above must-run, so `must_run`
        # leaves the variables entirely and appears on the right-hand sides below.
        d = sharej[None, :] * demand[:, None]                  # (T, n_buses)
        must_run = pminj[:, None] * u                          # (n_units, T)
        # a de-committed unit's segments would have upper bound zero, which
        # coincides with the lower bound and destroys the interior; see `OFF_EPS`
        width_box = jnp.where(u > 0, widthj[:, None], OFF_EPS)

        # per period: the n_units*K segment prices of that period, then the
        # shed penalty at every bus.  Segments are unit-major within a period,
        # matching Sblk, so the period axis moves to the front and the rest
        # ravels in place.
        c = jnp.concatenate(
            [jnp.moveaxis(offer, 2, 0).reshape(T, n_units * K),
             jnp.full((T, n_buses), VOLL)], 1).ravel()
        shed_cap = jnp.maximum(d, OFF_EPS)     # zero-demand buses need slack too
        # box upper bounds, one period per row and in the same unit-major segment
        # order as `c`: segment widths first, then the shed bound `s <= d`
        hi = jnp.concatenate([jnp.repeat(width_box.T, K, axis=1), shed_cap], 1)
        # (E) per period, net of must-run: the variables only have to cover what
        # the committed minima do not
        b_eq = d.sum(1) - must_run.sum(0)

        # (L) per period, likewise net: must-run injection and demand are both
        # constants here, so their flow contribution moves to the rhs and the two
        # directions become `F +/- PTDF (d - must_run)`
        line_rhs = (d - (UBj @ must_run).T) @ PTDFj.T           # (T, n_lines)
        line = jnp.concatenate([Fj[None, :] + line_rhs, Fj[None, :] - line_rhs], 1)

        # the segment variables carry only the part above must-run, so the
        # previous period's must-run moves to the right-hand side
        prev = jnp.concatenate([p_init[:, None], must_run[:, :-1]], 1)
        # the ramp rows carry a start-up allowance of p_min and a shut-down
        # allowance of p_max.  Without it a unit switching on would jump from 0
        # to p_min in one period, which the ordinary ramp limit forbids whenever
        # p_min exceeds the per-period ramp rate.  `u` is fixed here, so the
        # indicators are constants and this only moves the right-hand side.
        u_prev = jnp.concatenate([(p_init > 0.0).astype(u.dtype)[:, None],
                                  u[:, :-1]], 1)
        # start-up needs p_min, since a unit only has to reach its minimum;
        # shut-down needs p_max, because a unit already running above
        # p_min + R_dn * period otherwise cannot reach zero at all.  With only
        # the tighter p_min allowance, a commitment that de-commits such a unit
        # is infeasible rather than expensive.
        start = jnp.maximum(u - u_prev, 0.0) * pminj[:, None]
        stop = jnp.maximum(u_prev - u, 0.0) * pmaxj[:, None]
        ramp = jnp.concatenate([(rupj[:, None] + start - must_run + prev).T,
                                (rdnj[:, None] + stop + must_run - prev).T], 1)

        # row order: lines, ramp, `+I` then `-I` for the box.  `kkt` slices the
        # barrier weights on exactly this layout, so the two have to be changed
        # together; `spec["ramp_row0"]` and `spec["n_ramp_rows"]` are where the two
        # agree on where each group starts.
        h = jnp.concatenate([line.ravel(), ramp.ravel(), hi.ravel(),
                             jnp.zeros(spec["n"])])             # lo == 0

        # strictly interior and exact on every period's balance
        li, ui = margin * hi, (1.0 - margin) * hi
        frac = (b_eq - li.sum(1)) / jnp.maximum(ui.sum(1) - li.sum(1), 1e-9)
        xt = li + jnp.clip(frac, 0.02, 0.98)[:, None] * (ui - li)
        # every column of a period block has coefficient one in (E), so spreading
        # the residual evenly over all `nb` of them restores the balance exactly
        # while moving each variable by as little as possible off the interior
        # point above.  `relax` cannot do this -- its `u` columns carry `p_min` --
        # and corrects on the coefficient-one columns only.
        x0 = (xt + ((b_eq - xt.sum(1)) / nb)[:, None]).ravel()

        x, lam, nu, mu, r1, r3 = solver(c, Aj, b_eq, _unused_G, h, x0)

        # duals -> prices.  lam >= 0 for G x <= h; the sign convention here is
        # the one a missing negation silently breaks, and it is invisible until a
        # line binds.
        line_dual = -lam[: 2 * n_lines * T].reshape(T, 2, n_lines)
        # `rho` is the dual of the shed bound `s <= d` of the module docstring: the
        # `+I` box rows begin at `box0`, and within a period block the shed columns
        # are the tail, after the `n_units * K` segment columns.
        rho = -lam[box0: box0 + spec["n"]].reshape(T, nb)[:, n_units * K:]
        lmp = -nu[:, None] + (line_dual[:, 0] - line_dual[:, 1]) @ PTDFj + rho
        if monj is not None:
            # the duals are reported over every line; an unmonitored line has none
            line_dual = jnp.zeros((T, 2, n_lines_all)).at[:, :, monj].set(line_dual)

        xb = x.reshape(T, nb)
        # multiplying by u drops the OFF_EPS phantom output of de-committed units
        award = (pminj[None, :] * u.T + xb[:, : n_units * K].reshape(T, n_units, K).sum(2)) * u.T
        # and the same at a zero-demand bus, whose shed bound is OFF_EPS not zero
        shed = jnp.where(d > 0.0, xb[:, n_units * K:], 0.0)
        # The three duals, all nonnegative, so that the money-balance identity
        # can be assembled from the same solve that produced the price.  `mu`
        # above is the interior point complementarity gap and is a different
        # quantity; these carry the role in the name rather than the symbol for
        # that reason.
        return dict(award=award.T, lmp=lmp, shed=shed, mu=mu, dual_residual=r1,
                    primal_residual=r3,
                    line_dual_up=-line_dual[:, 0], line_dual_dn=-line_dual[:, 1],
                    shed_dual=-rho)

    return clear, spec
