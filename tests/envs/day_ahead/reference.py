"""numpy reference for the day-ahead clearing operator.

This answers "is the JAX implementation written correctly", which is a different
question from "is the mechanism right".  The latter is answered in
`tools/lp_bench` by comparing against HiGHS (a *different* algorithm) and against
a finite-difference derivative of the objective.  This file runs the **same**
algorithm, so the two can be compared elementwise.

To be worth anything it must not be a transcription of the implementation with
`jnp` swapped for `np`.  It therefore takes a deliberately different route to the
same mathematics:

* constraints are assembled as an **explicit dense** ``A_ub`` matrix, row by row
  in a Python loop over periods, rather than the implementation's vectorised
  block expressions;
* bounds are folded into ``G`` explicitly, so ``G`` exists here as a real matrix
  where the implementation never forms it;
* the Newton system is solved by a plain dense factorisation, not the block
  tridiagonal sweep.

What is shared is the algorithm itself, and only that: Mehrotra predictor-
corrector, the same regularisation rule, the same fraction-to-boundary rule, the
same fixed trip count.  An elementwise comparison requires that; a reference on a different
algorithm could not be compared elementwise.  That Newton loop is model-agnostic
and lives in `tests/solvers/reference_ipm.py`; what stays here is the day-ahead
LP and the day-ahead starting point.

Kept in `tests/` because it exists only to be compared against.
"""
import numpy as np

from powermarketjax.envs.day_ahead import OFF_EPS, VOLL, segment_costs
from powermarketjax.envs.day_ahead.clearing import MAX_ITER
from tests.solvers.reference_ipm import solve_ipm


def build_lp(case, offer, u, demand, p_init, cap_scale, ramp_scale, period_hours=1.0):
    """Assemble the LP explicitly: c, A_eq, b_eq, A_ub, b_ub, lo, hi.

    Variables are ordered block per period, matching the implementation, because
    the comparison is elementwise on `x`.  Everything else is built the long way.
    """
    offer = np.asarray(offer, np.float64)
    u = np.asarray(u, np.float64)
    demand = np.asarray(demand, np.float64)
    p_init = np.asarray(p_init, np.float64)

    p_min = np.asarray(case.unit_p_min, np.float64)
    p_max = np.asarray(case.unit_p_max, np.float64)
    unit_bus = np.asarray(case.unit_node_idx, np.int64)
    n_units, K, _ = offer.shape
    T = len(demand)
    n_buses = int(case.n_nodes)
    PTDF = np.asarray(case.PTDF, np.float64)
    n_lines = PTDF.shape[0]
    nb = n_units * K + n_buses
    n = nb * T

    width, _ = segment_costs(case, K)
    F = np.asarray(case.line_cap, np.float64) * cap_scale
    ramp_up = np.asarray(case.unit_ramp_up, np.float64) * p_max * period_hours * ramp_scale
    ramp_dn = np.asarray(case.unit_ramp_down, np.float64) * p_max * period_hours * ramp_scale

    node_pd = np.asarray(case.node_pd, np.float64)
    if node_pd.sum() > 1e-6:
        share = node_pd / node_pd.sum()
    else:
        load_bus = np.asarray(case.load_node_idx, np.int64)
        w = np.asarray(case.load_d_max, np.float64)
        share = np.zeros(n_buses); np.add.at(share, load_bus, w)
        share = share / share.sum()
    d = share[None, :] * demand[:, None]                       # (T, n_buses)

    must_run = p_min[:, None] * u                              # (n_units, T)
    width_box = np.where(u > 0, width[:, None], OFF_EPS)

    seg_bus = np.zeros((n_buses, n_units * K))
    seg_bus[np.repeat(unit_bus, K), np.arange(n_units * K)] = 1.0
    Mblk = np.hstack([PTDF @ seg_bus, PTDF])                   # (n_lines, nb)
    Sblk = np.zeros((n_units, nb))
    Sblk[np.repeat(np.arange(n_units), K), np.arange(n_units * K)] = 1.0

    c = np.zeros(n); lo = np.zeros(n); hi = np.zeros(n)
    A_eq = np.zeros((T, n)); b_eq = np.zeros(T)
    n_line_rows, n_ramp_rows = 2 * n_lines * T, 2 * n_units * T
    A_ub = np.zeros((n_line_rows + n_ramp_rows, n)); b_ub = np.zeros(A_ub.shape[0])

    for t in range(T):
        sl = slice(t * nb, (t + 1) * nb)
        c[sl] = np.concatenate([offer[:, :, t].ravel(), np.full(n_buses, VOLL)])
        hi[sl] = np.concatenate([np.repeat(width_box[:, t], K),
                                 np.maximum(d[t], OFF_EPS)])
        A_eq[t, sl] = 1.0
        b_eq[t] = d[t].sum() - must_run[:, t].sum()

        inj = np.zeros(n_buses); np.add.at(inj, unit_bus, must_run[:, t])
        rhs = PTDF @ (d[t] - inj)
        A_ub[2 * n_lines * t: 2 * n_lines * t + n_lines, sl] = Mblk
        A_ub[2 * n_lines * t + n_lines: 2 * n_lines * (t + 1), sl] = -Mblk
        b_ub[2 * n_lines * t: 2 * n_lines * t + n_lines] = F + rhs
        b_ub[2 * n_lines * t + n_lines: 2 * n_lines * (t + 1)] = F - rhs

        # ramp couples t to t-1: p[t] - p[t-1] <= R_up and p[t-1] - p[t] <= R_dn.
        # The segment variables carry only the part above must-run, so the
        # must-run difference moves to the right-hand side.  For t = 0 the
        # previous output is p_init and there is no earlier block to reference.
        r0 = n_line_rows + 2 * n_units * t
        A_ub[r0: r0 + n_units, sl] = Sblk
        A_ub[r0 + n_units: r0 + 2 * n_units, sl] = -Sblk
        if t > 0:
            pv = slice((t - 1) * nb, t * nb)
            A_ub[r0: r0 + n_units, pv] = -Sblk
            A_ub[r0 + n_units: r0 + 2 * n_units, pv] = Sblk
        prev = p_init if t == 0 else must_run[:, t - 1]
        # start-up allowance of p_min, shut-down allowance of p_max (§6.3);
        # u is fixed, so the
        # indicators are constants of the right-hand side
        u_prev = (p_init > 0.0).astype(np.float64) if t == 0 else u[:, t - 1]
        start = np.maximum(u[:, t] - u_prev, 0.0) * p_min
        stop = np.maximum(u_prev - u[:, t], 0.0) * p_max
        b_ub[r0: r0 + n_units] = ramp_up + start - must_run[:, t] + prev
        b_ub[r0 + n_units: r0 + 2 * n_units] = ramp_dn + stop + must_run[:, t] - prev

    return dict(c=c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, lo=lo, hi=hi,
                n=n, nb=nb, T=T, K=K, n_units=n_units, n_buses=n_buses,
                n_lines=n_lines, PTDF=PTDF, p_min=p_min, u=u, d=d,
                n_line_rows=n_line_rows, n_ramp_rows=n_ramp_rows)


def start_point(lp, margin=0.05):
    """Strictly interior, exact on every period's balance."""
    nb, T = lp["nb"], lp["T"]
    x = np.zeros(lp["n"])
    for t in range(T):
        sl = slice(t * nb, (t + 1) * nb)
        h = lp["hi"][sl]
        li, ui = margin * h, (1.0 - margin) * h
        D = lp["b_eq"][t]
        f = (D - li.sum()) / max(ui.sum() - li.sum(), 1e-9)
        xt = li + min(max(f, 0.02), 0.98) * (ui - li)
        x[sl] = xt + (D - xt.sum()) / nb
    return x


def clear(case, offer, u, demand, p_init, cap_scale, ramp_scale, period_hours=1.0,
          max_iter=MAX_ITER):
    """Reference clearing: same signature and outputs as the implementation."""
    lp = build_lp(case, offer, u, demand, p_init, cap_scale, ramp_scale, period_hours)
    # same dual start as the implementation, for the reason in `solve_ipm`'s docstring
    x, lam, nu, mu, r1, r3 = solve_ipm(lp, start_point(lp), max_iter,
                                       dual_start="cost_norm")

    T, nb, n_units, K = lp["T"], lp["nb"], lp["n_units"], lp["K"]
    n_lines, n_buses, PTDF = lp["n_lines"], lp["n_buses"], lp["PTDF"]
    box0 = lp["n_line_rows"] + lp["n_ramp_rows"]

    lmp = np.zeros((T, n_buses)); award = np.zeros((n_units, T))
    shed = np.zeros((T, n_buses))
    for t in range(T):
        mu_ub = -lam[2 * n_lines * t: 2 * n_lines * (t + 1)]     # scipy sign
        rho = -lam[box0 + t * nb + n_units * K: box0 + (t + 1) * nb]
        lmp[t] = -nu[t] + (mu_ub[:n_lines] - mu_ub[n_lines:]) @ PTDF + rho
        xt = x[t * nb: (t + 1) * nb]
        award[:, t] = (lp["p_min"] * lp["u"][:, t]
                       + xt[:n_units * K].reshape(n_units, K).sum(1)) * lp["u"][:, t]
        shed[t] = np.where(lp["d"][t] > 0.0, xt[n_units * K:], 0.0)
    return dict(award=award, lmp=lmp, shed=shed, mu=mu, dual_residual=r1,
                primal_residual=r3)
