"""numpy reference for the joint clearing operator.

This answers "is the JAX implementation written correctly", not "is the
mechanism right".  The latter is answered against HiGHS, a *different*
algorithm, in `tools/ancillary`; this file runs the **same** algorithm so that
the two can be compared elementwise.

To be worth anything it must not be a transcription with `jnp` swapped for `np`,
so it takes a different route to the same mathematics:

* every constraint row is written into a dense ``A_ub`` **by a Python loop over
  lines, units and products**, with an explicit row-index map, where the
  implementation builds the same rows by stacking vectorised blocks;
* the bounds are folded into ``G`` as real rows by the shared Newton loop, while
  the implementation keeps them as a box and never materialises ``G``;
* the reserve columns are addressed per (unit, product) pair rather than through
  the implementation's `res_of_unit` / `prod_of_res` incidence matrices.

**What is shared, and why each is right to share.** The `CaseData` and
`segment_costs` are shared: recomputing them here would test the data loader
rather than the clearing.  The (RA) allowance is recomputed from the registered
ramp rate here rather than read from `spec["res_cap"]`, because it is part of
what the clearing must get right.  The Newton loop itself is shared
(`tests/solvers/reference_ipm.py`), as an elementwise L2 requires: a reference on a
different algorithm could not be compared elementwise.

**The three calibration parameters must match the implementation and are passed
explicitly**: `max_iter`, `dual_start` and `reg_coef`.  They are one calibration,
not three knobs, and a reference running a different one compares two
calibrations rather than two implementations.  This is not hypothetical: a
reference running 120 steps against an implementation running 80 once left every
L2 assertion passing at 4.3e-18 while testing nothing.  This market runs
`reg_coef = 1e-18`, which is **not** the solver default, so a reference left on
the default would disagree even above the separation threshold and the
disagreement would look like an implementation error.
"""
import numpy as np

from powermarketjax.envs.ancillary.clearing import (OFF_EPS, VOLL,
                                                    _resolve_off_eps)
from powermarketjax.envs.day_ahead.clearing import segment_costs
from tests.solvers.reference_ipm import solve_ipm


def build_lp(case, offer, offer_res, u, demand, d_res, p_prev, theta, volr,
             cap_scale, ramp_scale, period_hours, n_segments=1, voll=VOLL,
             off_eps=OFF_EPS, monitored_lines=None):
    """Assemble the LP explicitly: c, A_eq, b_eq, A_ub, b_ub, lo, hi.

    Column order matches the implementation, since the comparison is
    elementwise: ``[g (unit-major, K per unit) ; r (unit-major, P per unit) ;
    s (per bus) ; s_res (per product)]``.  Row order is this file's own, and the
    map is returned so the duals can be read back.

    ``monitored_lines`` has the implementation's meaning: ``None`` writes both
    rows of every line, an index array writes those lines only.  An L2 on a
    restricted row set must build the reference on the same set, or it
    compares two LPs (a row-set difference) rather than
    two implementations of one.
    """
    eps = _resolve_off_eps(off_eps)
    pmin = np.asarray(case.unit_p_min, np.float64)
    pmax = np.asarray(case.unit_p_max, np.float64)
    unit_bus = np.asarray(case.unit_node_idx, np.int64)
    PTDF = np.asarray(case.PTDF, np.float64)
    line_cap = np.asarray(case.line_cap, np.float64)
    if monitored_lines is not None:
        mon = np.unique(np.asarray(monitored_lines, np.int64))
        PTDF, line_cap = PTDF[mon], line_cap[mon]
    theta = np.asarray(theta, np.float64)
    K, P = n_segments, len(theta)
    n_u, n_b, n_l = len(pmin), int(case.n_nodes), PTDF.shape[0]
    width, _ = segment_costs(case, K)
    rate = np.asarray(case.unit_ramp_up, np.float64) * pmax * ramp_scale
    ramp_up = rate * period_hours
    ramp_dn = (np.asarray(case.unit_ramp_down, np.float64) * pmax
               * period_hours * ramp_scale)

    node_pd = np.asarray(case.node_pd, np.float64)
    if node_pd.sum() > 1e-6:
        share = node_pd / node_pd.sum()
    else:
        share = np.zeros(n_b)
        np.add.at(share, np.asarray(case.load_node_idx, np.int64),
                  np.asarray(case.load_d_max, np.float64))
        share = share / share.sum()

    g0, r0, s0, sr0 = 0, n_u * K, n_u * (K + P), n_u * (K + P) + n_b
    n = n_u * (K + P) + n_b + P
    gcol = lambda i, k: g0 + i * K + k
    rcol = lambda i, j: r0 + i * P + j

    d = share * demand
    must = pmin * u
    on = u > 0.0

    rows, rhs, row0 = [], [], {}

    row0["line_up"] = 0
    for sign in (+1.0, -1.0):                       # +M x <= F + rhs, -M x <= F - rhs
        if sign < 0:
            row0["line_dn"] = len(rows)
        for l in range(n_l):
            row = np.zeros(n)
            for i in range(n_u):
                for k in range(K):
                    row[gcol(i, k)] = sign * PTDF[l, unit_bus[i]]
            for b in range(n_b):
                row[s0 + b] = sign * PTDF[l, b]
            rows.append(row)
            base = float(PTDF[l] @ (d - np.bincount(unit_bus, must, n_b)))
            rhs.append(float(line_cap[l]) * cap_scale + sign * base)

    u_prev = (p_prev > 0.0).astype(np.float64)
    start = np.maximum(u - u_prev, 0.0) * pmin
    stop = np.maximum(u_prev - u, 0.0) * pmax
    row0["ramp_up"] = len(rows)
    for i in range(n_u):                            # p_i - p_prev_i <= R_up + start
        row = np.zeros(n)
        for k in range(K):
            row[gcol(i, k)] = 1.0
        rows.append(row)
        rhs.append(ramp_up[i] + start[i] - must[i] + p_prev[i])
    row0["ramp_dn"] = len(rows)
    for i in range(n_u):                            # p_prev_i - p_i <= R_dn + stop
        row = np.zeros(n)
        for k in range(K):
            row[gcol(i, k)] = -1.0
        rows.append(row)
        rhs.append(ramp_dn[i] + stop[i] + must[i] - p_prev[i])

    row0["cs"] = len(rows)
    for i in range(n_u):                            # (CS)
        row = np.zeros(n)
        for k in range(K):
            row[gcol(i, k)] = 1.0
        for j in range(P):
            row[rcol(i, j)] = 1.0
        rows.append(row)
        rhs.append(max((pmax[i] - pmin[i]) * u[i], eps["cs"]))

    row0["rd"] = len(rows)
    for j in range(P):                              # (RD), written as a <= row
        row = np.zeros(n)
        for i in range(n_u):
            row[rcol(i, j)] = -1.0
        row[sr0 + j] = -1.0
        rows.append(row)
        rhs.append(-float(d_res[j]))

    A_ub = np.array(rows)
    b_ub = np.array(rhs)

    lo = np.zeros(n)
    hi = np.empty(n)
    for i in range(n_u):
        for k in range(K):
            hi[gcol(i, k)] = width[i] if on[i] else eps["g"]
        for j in range(P):                          # (RA), folded into the box
            hi[rcol(i, j)] = rate[i] * theta[j] if on[i] else eps["r"]
    for b in range(n_b):
        hi[s0 + b] = max(d[b], eps["s"])
    for j in range(P):
        hi[sr0 + j] = 2.0 * max(float(d_res[j]), eps["sr"])

    A_eq = np.zeros((1, n))
    for i in range(n_u):
        for k in range(K):
            A_eq[0, gcol(i, k)] = 1.0
    for b in range(n_b):
        A_eq[0, s0 + b] = 1.0
    b_eq = np.array([d.sum() - must.sum()])

    c = np.empty(n)
    for i in range(n_u):
        for k in range(K):
            c[gcol(i, k)] = period_hours * offer[i, k]
        for j in range(P):
            c[rcol(i, j)] = period_hours * offer_res[i, j]
    c[s0: s0 + n_b] = period_hours * voll
    c[sr0: sr0 + P] = period_hours * volr

    return dict(c=c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, lo=lo, hi=hi,
                n=n, row0=row0, n_u=n_u, n_b=n_b, n_l=n_l, K=K, P=P,
                g0=g0, r0=r0, s0=s0, sr0=sr0, d=d, must=must, share=share,
                PTDF=PTDF, unit_bus=unit_bus, period_hours=period_hours)


def start_point(lp, margin=0.05):
    """Strictly interior, and exact on the balance row where it can be."""
    lo_x, up_x = margin * lp["hi"], (1.0 - margin) * lp["hi"]
    x0 = lo_x + 0.5 * (up_x - lo_x)
    reads = np.zeros(lp["n"])
    reads[lp["g0"]: lp["r0"]] = 1.0
    reads[lp["s0"]: lp["sr0"]] = 1.0
    gap = lp["b_eq"][0] - float(x0 @ reads)
    x0 = x0 + gap * reads / reads.sum()
    return np.clip(x0, lo_x, up_x)


def clear_reference(case, offer, offer_res, u, demand, d_res, p_prev, *, theta,
                    volr, cap_scale, ramp_scale, period_hours, max_iter,
                    dual_start, reg_coef, n_segments=1, monitored_lines=None,
                    freeze_mu=None):
    """Run the reference clearing and return the same quantities as `clear`."""
    lp = build_lp(case, np.asarray(offer), np.asarray(offer_res), np.asarray(u),
                  float(demand), np.asarray(d_res), np.asarray(p_prev), theta,
                  volr, cap_scale, ramp_scale, period_hours, n_segments,
                  monitored_lines=monitored_lines)
    x, lam, nu, mu, r1, r3 = solve_ipm(lp, start_point(lp), max_iter,
                                       dual_start=dual_start, reg_coef=reg_coef,
                                       freeze_mu=freeze_mu)
    n_u, n_b, n_l, K, P = lp["n_u"], lp["n_b"], lp["n_l"], lp["K"], lp["P"]
    row0, per = lp["row0"], 1.0 / lp["period_hours"]

    line_up = -lam[row0["line_up"]: row0["line_up"] + n_l] * per
    line_dn = -lam[row0["line_dn"]: row0["line_dn"] + n_l] * per
    # the bound duals sit after the coupling rows, in the order the shared
    # Newton loop folds them in: first the upper bounds, then the lower
    n_cpl = lp["A_ub"].shape[0]
    rho = -lam[n_cpl + lp["s0"]: n_cpl + lp["s0"] + n_b] * per
    lmp = -nu[0] * per + (line_up - line_dn) @ lp["PTDF"] + rho

    g = x[lp["g0"]: lp["r0"]].reshape(n_u, K)
    r = x[lp["r0"]: lp["s0"]].reshape(n_u, P)
    award = (lp["must"] + g.sum(1)) * u
    return dict(award=award, reserve=r * np.asarray(u)[:, None],
                shed=np.where(lp["d"] > 0.0, x[lp["s0"]: lp["sr0"]], 0.0),
                reserve_shortfall=x[lp["sr0"]:], z=float(lp["c"] @ x), lmp=lmp,
                reserve_price=lam[row0["rd"]: row0["rd"] + P] * per,
                capacity_dual=lam[row0["cs"]: row0["cs"] + n_u] * per * np.asarray(u),
                mu=mu, dual_residual=r1, primal_residual=r3)
