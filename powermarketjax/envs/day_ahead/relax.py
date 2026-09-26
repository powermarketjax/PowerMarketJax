"""The relaxed commitment LP: minimum up-/downtime dropped, solved by the same
interior point method as the fixed-commitment dispatch.

Array in, array out, pure.  Given offers, realised demand and the day boundary it
returns the relaxed commitment ``u`` in [0, 1], which `round_commitment` rounds.
Prices are **not** taken from here: they come from the fixed-commitment dispatch
in `clearing.py`, and nothing in this module produces a settlement quantity.

What is dropped and why is in `relax_kkt`.  What is kept:

* the two ramp allowances, asymmetric, ``p_min`` on a start and ``p_max`` on a
  stop, the second of which is what keeps a commitment that de-commits a
  high-output unit feasible rather than merely expensive;
* the minimum up-/downtime **initial conditions**, which are variable bounds
  rather than rows -- a unit whose run length at the boundary is shorter than
  its window has the remainder of that window forced.  These are the only route
  by which yesterday reaches today, they cost nothing structurally, and
  dropping them would silently decouple the market days.

Each clause of that forcing applies only in the state it is about: a unit that is
off carries ``up_time = 0``, and reading a remaining uptime off that would force
it *on* for its whole minimum uptime while the mirror error forces a running unit
*off*.  Both at once make the first day infeasible.
"""
from typing import Callable, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np

from powermarketjax.solvers import ipm
from . import kkt_lowrank, relax_kkt
from .clearing import KKT_ROUTES, VOLL, OFF_EPS, segment_costs

#: Newton steps.  Calibrated separately from `clearing.MAX_ITER`, against the
#: rounded commitment rather than against dual accuracy: what `round_commitment`
#: consumes is ``sign(u)``, so the criterion is that it reproduces the correct
#: commitment cell for cell.  A converged solve lands near ``mu`` 1e-11; short
#: of convergence a caller gets a wrong commitment with no warning, since the
#: transition between the two regimes is abrupt rather than gradual.
MAX_ITER = 60

#: Rounding tolerance for `round_commitment`.  The rounding rule is ``u > 0``,
#: and that form is not implementable directly: an interior point method
#: converges to a strictly interior point and never returns an exact zero, so
#: ``u > 0`` would commit every unit-period and the fixed-commitment dispatch
#: would come back infeasible.  This tolerance stands in for the exact zero a
#: simplex method would have returned.
ROUND_EPS = 1e-9


def round_commitment(u, eps: float = ROUND_EPS):
    """Round the relaxed commitment to integers: commit every unit the
    relaxation touches at all.

    The threshold leans towards committing because the two errors are orders of
    magnitude apart -- a spare committed unit costs its no-load rate in \\$/h,
    while a unit left off costs load shed at VOLL in \\$/MWh -- and `eps` is the
    solver tolerance of `ROUND_EPS`, not a second economic threshold.
    """
    return (u > eps).astype(u.dtype)


def make_relax(
    case,
    n_periods: int,
    n_segments: int = 1,
    cap_scale: float = 1.0,
    ramp_scale: float = 1.0,
    period_hours: float = 1.0,
    max_iter: int = MAX_ITER,
    dual_start: str = "cost_norm",
    monitored_lines=None,
    kkt: str = "auto",
) -> Tuple[Callable, Dict]:
    """Build the relaxed-commitment operator for one fixed case and horizon.

    Returns ``(relax, spec)`` where ``relax(offer, demand, p_init, u_prev,
    up_time, down_time)`` is pure and jittable.

    ``monitored_lines`` and ``kkt`` mean what they mean in
    `clearing.make_clearing`: which line limits the LP carries, and which
    linear-algebra route solves its Newton system.  Both default to today's
    operator, operation for operation.

        offer      (n_units, n_segments, n_periods)  as-bid price
        demand     (n_periods,)                      total system demand, MW
        p_init     (n_units,)                        previous day's final output
        u_prev     (n_units,)                        previous day's final commitment
        up_time    (n_units,)                        periods on at the boundary
        down_time  (n_units,)                        periods off at the boundary

    ``relax`` returns ``u`` (n_units, n_periods) in [0, 1] plus ``mu`` and
    ``dual_residual``.  ``mu`` is the convergence check, exactly as in the
    fixed-commitment dispatch.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "relaxed commitment requires float64: set "
            "jax.config.update('jax_enable_x64', True)")

    if kkt not in KKT_ROUTES:
        raise ValueError(f"kkt must be one of {KKT_ROUTES}, got {kkt!r}")
    T, K = n_periods, n_segments
    p_min = np.asarray(case.unit_p_min, np.float64)
    p_max = np.asarray(case.unit_p_max, np.float64)
    unit_bus = np.asarray(case.unit_node_idx, np.int64)
    NL = np.asarray(case.unit_no_load_cost, np.float64)
    SU = np.asarray(case.unit_startup_cost, np.float64)
    UT = np.maximum(np.asarray(case.unit_min_up_time, np.int64), 1)
    DT = np.maximum(np.asarray(case.unit_min_down_time, np.int64), 1)
    width, _ = segment_costs(case, K)
    n_u, n_b = len(p_min), int(case.n_nodes)
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
    n_l = PTDF.shape[0]
    nb = n_u * K + n_b + 2 * n_u
    n = nb * T

    node_pd = np.asarray(case.node_pd, np.float64)
    if node_pd.sum() > 1e-6:
        share = node_pd / node_pd.sum()
    else:
        share = np.zeros(n_b)
        np.add.at(share, np.asarray(case.load_node_idx, np.int64),
                  np.asarray(case.load_d_max, np.float64))
        share = share / share.sum()

    ramp_up = np.asarray(case.unit_ramp_up, np.float64) * p_max * period_hours * ramp_scale
    ramp_dn = np.asarray(case.unit_ramp_down, np.float64) * p_max * period_hours * ramp_scale

    # column slices of one period block
    o_g, o_s, o_u, o_v = 0, n_u * K, n_u * K + n_b, n_u * K + n_b + n_u

    seg_bus = np.zeros((n_b, n_u * K))
    seg_bus[np.repeat(unit_bus, K), np.arange(n_u * K)] = 1.0
    unit_bus_mat = np.zeros((n_b, n_u))
    unit_bus_mat[unit_bus, np.arange(n_u)] = 1.0
    sum_k = np.zeros((n_u, n_u * K))
    sum_k[np.repeat(np.arange(n_u), K), np.arange(n_u * K)] = 1.0

    # (L): flow of the block's own injections; the demand part is a constant rhs
    M = np.zeros((n_l, nb))
    M[:, o_g:o_s] = PTDF @ seg_bus
    M[:, o_s:o_u] = PTDF
    M[:, o_u:o_v] = (PTDF @ unit_bus_mat) * p_min

    # (CAP): g - width * u <= 0
    C = np.zeros((n_u * K, nb))
    C[np.arange(n_u * K), np.arange(n_u * K)] = 1.0
    C[np.arange(n_u * K), o_u + np.repeat(np.arange(n_u), K)] = -np.repeat(width, K)

    # (RMP) after eliminating w = v - (u_t - u_{t-1}).  Up row uses Sa on t and
    # Sb on t-1; down row uses -Sc on t and Sd on t-1.
    def _S(coef_u, coef_v):
        """One per-unit (RMP) row group: ``(n_units, nb)``, one row per unit.

        Every row sums that unit's own ``g`` segments and adds a diagonal weight
        on its ``u`` column and on its ``v`` column, which is enough to express
        both ramp directions once ``w`` has been eliminated.  Four operators are
        needed rather than one because the start-up allowance is ``p_min`` and the
        shut-down allowance ``p_max``, so the two directions differ in the
        ``v`` weight *and*, after the elimination, in the ``u`` weight as well;
        `relax_kkt` records what that costs the block structure.
        """
        S = np.zeros((n_u, nb))
        S[:, o_g:o_s] = sum_k
        S[:, o_u:o_v] = np.diag(coef_u)
        S[:, o_v:] = np.diag(coef_v)
        return S

    Sa = _S(p_min, -p_min)
    Sb = _S(p_min, np.zeros(n_u))
    Sc = _S(p_min - p_max, p_max)
    Sd = _S(p_min - p_max, np.zeros(n_u))

    # bounds on the eliminated w: 0 <= v - (u_t - u_{t-1}) <= 1
    Se = np.zeros((n_u, nb)); Se[:, o_u:o_v] = np.eye(n_u); Se[:, o_v:] = -np.eye(n_u)
    Sf = np.zeros((n_u, nb)); Sf[:, o_u:o_v] = np.eye(n_u)

    # (E): sum_k g + sum_n s + sum_i p_min u = demand
    a_row = np.zeros(nb)
    a_row[o_g:o_s] = 1.0
    a_row[o_s:o_u] = 1.0
    a_row[o_u:o_v] = p_min

    spec = dict(T=T, K=K, n_units=n_u, n_buses=n_b, n_l=n_l, nb=nb, n=n,
                o_g=o_g, o_s=o_s, o_u=o_u, o_v=o_v,
                p_min=p_min, p_max=p_max, width=width, demand_share=share,
                unit_bus=unit_bus, cap_scale=cap_scale, ramp_scale=ramp_scale)
    spec["m"] = (2 * n_l * T) + (n_u * K * T) + (2 * n_u * T) + (2 * n_u * T) + 2 * n
    # What the LP carries and how its Newton system is solved, written whatever
    # the flags were: a driver stamps its products off these, not off its own
    # command line.  `monitored_lines`
    # is None when every line's row is carried, as `make_relax` was given.
    spec["monitored_lines"], spec["n_lines"], spec["kkt_route"] = mon, n_lines_all, route
    n_row_before_box = spec["m"] - 2 * n

    ops_np = dict(M=M, C=C, Sa=Sa, Sb=Sb, Sc=Sc, Sd=Sd, Se=Se, Sf=Sf, a=a_row)
    ops_j = {k: jnp.asarray(v) for k, v in ops_np.items()}

    A_eq = np.zeros((T, n))
    for t in range(T):
        A_eq[t, t * nb:(t + 1) * nb] = a_row
    Aj = jnp.asarray(A_eq)

    Fj = jnp.asarray(F)
    PTDFj = jnp.asarray(PTDF)
    pminj, pmaxj, widthj = jnp.asarray(p_min), jnp.asarray(p_max), jnp.asarray(width)
    NLj, SUj = jnp.asarray(NL), jnp.asarray(SU)
    rupj, rdnj = jnp.asarray(ramp_up), jnp.asarray(ramp_dn)
    sharej = jnp.asarray(share)
    UTj, DTj = jnp.asarray(UT), jnp.asarray(DT)
    UBj = jnp.asarray(unit_bus_mat)
    ar = jnp.arange(T)

    if route == "dense":
        kkt_pair = relax_kkt.make_kkt(spec, ops_j)
    else:
        kkt_pair = kkt_lowrank.make_kkt_relax(spec, ops_np)
    solver = ipm.make_solver(
        n, spec["m"], max_iter=max_iter, n_eq=T,
        ops=relax_kkt.make_ops(spec, ops_j),
        kkt=kkt_pair,
        dual_start=dual_start)
    _unused_G = jnp.zeros((1, 1))

    def relax(offer, demand, p_init, u_prev, up_time, down_time):
        """Solve step 1' for one market day: offers in, a relaxed commitment out.

        Args:
            offer: ``(n_units, n_segments, n_periods)`` as-bid price in \\$/MWh.
            demand: ``(n_periods,)`` realised total system demand in MW.
            p_init: ``(n_units,)`` output at the day boundary.
            u_prev: ``(n_units,)`` commitment in the previous day's last period;
                read as a threshold at 0.5, so a float32 carry is fine.
            up_time: ``(n_units,)`` periods on at the boundary.
            down_time: ``(n_units,)`` periods off at the boundary.

        Returns:
            A dict of ``u`` ``(n_units, n_periods)`` in [0, 1], the matching output
            ``p`` ``(n_units, n_periods)`` and ``shed`` ``(n_periods, n_buses)``,
            the objective value ``obj``, and the solver diagnostics ``mu``,
            ``dual_residual`` and ``primal_residual``.  Only ``u`` is consumed
            downstream: `round_commitment` rounds it and the fixed-commitment
            dispatch recomputes the dispatch from scratch.  **No dual of this
            solve is a price** -- prices come from the fixed-commitment dispatch
            alone -- and ``obj`` is a bound on the commitment problem, not a
            settlement quantity.
        """
        d = sharej[None, :] * demand[:, None]                     # (T, n_b)
        uprev = (u_prev > 0.5).astype(jnp.float64)

        # (MU)/(MD) initial conditions as bounds.  Each clause reads only the
        # counter that belongs to its own state, which is what keeps a stopped
        # unit from being forced on and vice versa.
        force_on = (uprev[:, None] > 0.5) & (ar[None, :] < (UTj - up_time)[:, None])
        force_off = (uprev[:, None] < 0.5) & (ar[None, :] < (DTj - down_time)[:, None])
        u_lo = jnp.where(force_on, 1.0 - OFF_EPS, 0.0)            # (n_u, T)
        u_hi = jnp.where(force_off, OFF_EPS, 1.0)

        # boxes, per period block
        g_hi = jnp.broadcast_to(jnp.repeat(widthj, K)[None, :], (T, n_u * K))
        s_hi = jnp.maximum(d, OFF_EPS)
        v_hi = jnp.ones((T, n_u))
        hi = jnp.concatenate([g_hi, s_hi, u_hi.T, v_hi], 1)       # (T, nb)
        lo = jnp.concatenate([jnp.zeros((T, n_u * K + n_b)), u_lo.T,
                              jnp.zeros((T, n_u))], 1)

        # objective coefficients, one period per row and in the block's own column
        # order.  Energy and shed are $/MWh on a megawatt variable and no-load is
        # $/h on an indicator, so all three carry `period_hours`; start-up is $ per
        # start on the start-up indicator and must not.
        c = jnp.concatenate([
            jnp.moveaxis(offer, 2, 0).reshape(T, n_u * K) * period_hours,
            jnp.full((T, n_b), VOLL * period_hours),
            jnp.broadcast_to(NLj[None, :] * period_hours, (T, n_u)),
            jnp.broadcast_to(SUj[None, :], (T, n_u))], 1).ravel()

        b_eq = d.sum(1)

        # (L): the demand at each bus is a constant and moves to the rhs
        line_rhs = d @ PTDFj.T                                    # (T, n_l)
        line = jnp.concatenate([Fj[None, :] + line_rhs, Fj[None, :] - line_rhs], 1)

        cap_h = jnp.zeros((T, n_u * K))          # (CAP) is g - width * u <= 0

        # (RMP).  Period 0 carries the boundary output and commitment on the rhs;
        # later periods have both sides among the variables.
        zeros_u = jnp.zeros((T - 1, n_u))
        up_h = jnp.concatenate([(rupj + p_init)[None, :],
                                jnp.broadcast_to(rupj[None, :], (T - 1, n_u))], 0)
        dn_h = jnp.concatenate([(rdnj - p_init + pmaxj * uprev)[None, :],
                                jnp.broadcast_to(rdnj[None, :], (T - 1, n_u))], 0)
        ramp = jnp.concatenate([up_h, dn_h], 1)

        # the two bounds on the eliminated ``w = v - (u_t - u_{t-1})``, one row for
        # ``w >= 0`` and one for ``w <= 1``.  Period 0's ``u_{t-1}`` is the boundary
        # commitment and therefore a constant, so it moves to the right-hand side;
        # later periods have both ends among the variables and a constant bound.
        w0_h = jnp.concatenate([uprev[None, :], zeros_u], 0)
        w1_h = jnp.concatenate([(1.0 - uprev)[None, :],
                                jnp.ones((T - 1, n_u))], 0)
        wbnd = jnp.concatenate([w0_h, w1_h], 1)

        # row order: lines, capacity, ramp, the two w bounds, `+I` then `-I` for
        # the box.  `relax_kkt._split_barrier` slices the barrier weights on
        # exactly this layout and the module docstring there tabulates it, so the
        # two have to be changed together.  The lower box is `-x <= -lo`, hence the
        # negation rather than a second `lo`.
        h = jnp.concatenate([line.ravel(), cap_h.ravel(), ramp.ravel(),
                             wbnd.ravel(), hi.ravel(), -lo.ravel()])

        # strictly interior, and exact on every period's balance.  Only the
        # coefficient-one columns absorb the correction, so the balance stays
        # exact whatever p_min does.
        mid = lo + 0.5 * (hi - lo)
        n_free = n_u * K + n_b
        resid = b_eq - mid @ ops_j["a"]
        adj = jnp.zeros((T, nb)).at[:, :n_free].add((resid / n_free)[:, None])
        x0 = jnp.clip(mid + adj, lo + 1e-6 * (hi - lo), hi - 1e-6 * (hi - lo))
        # one more exact correction after the clip, spread over the same columns
        resid = b_eq - x0 @ ops_j["a"]
        x0 = x0.at[:, :n_free].add((resid / n_free)[:, None]).ravel()

        x, lam, nu, mu, r1, r3 = solver(c, Aj, b_eq, _unused_G, h, x0)

        xb = x.reshape(T, nb)
        # The two forced states are known exactly, so they are written in rather
        # than read back out of the solver.  Their bounds carry OFF_EPS to keep
        # the interior non-empty, and a forced-off unit therefore leaves the
        # solver at OFF_EPS rather than at zero -- which is above any rounding
        # tolerance small enough to be safe elsewhere, so reading it back would
        # commit exactly the units the day boundary forbids.  Same discipline as
        # `clearing.award`, which multiplies out its own OFF_EPS phantom.
        u = jnp.clip(xb[:, o_u:o_v], 0.0, 1.0).T                  # (n_u, T)
        u = jnp.where(force_off, 0.0, jnp.where(force_on, 1.0, u))
        g = xb[:, o_g:o_s].reshape(T, n_u, K).sum(2).T
        shed = jnp.where(d > 0.0, xb[:, o_s:o_u], 0.0)
        obj = jnp.dot(c, x)
        return dict(u=u, p=pminj[:, None] * u + g, shed=shed, obj=obj,
                    mu=mu, dual_residual=r1, primal_residual=r3)

    return relax, spec
