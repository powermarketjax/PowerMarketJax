"""Offline pre-commitment for the day-ahead market: steps 1 and 2 of the three-step clearing.

This is not part of the package and CI does not run it.  It exists because §19
left "what produces the commitment" open after step 1 was measured at 16.1 s per
clearing on the accelerator, and one of the three answers there is *taking the
commitment as given from outside*.  This script produces that outside: one
integer commitment per market day, frozen into a fixture that
`envs/day_ahead/env.py` reads.  Being offline, it may use a CPU solver
-- which is what lets it write §6.3 out as it is written and hand
it to HiGHS rather than approximate it.

Two modes, both of them the same problem definition:

    relax   step 1 as §6.3 writes it -- u, v, w relaxed to [0,1] -- then step 2's
            rounding `u_int = u > 0` (revised 2026-08-10)
    milp    the exact problem of §6.1, u binary, via `scipy.optimize.milp`

`relax` is what the fixture is built from; `milp` is the reference the gap is
measured against, and at T=24 it costs 8 minutes a day against 0.3 s, so it runs
over a handful of days rather than all of them.  Which mode produced a fixture
and which days it covers are written into the fixture's metadata, because those
two facts are the difference between a measured optimality gap and an assumed
one.

**The offer basis is true cost.**  Step 1 is solved against the monotone
envelope of the segment costs at markup 1, so the schedule is what an operator
computes from true costs before anyone bids.  Nothing here reads an agent action,
and that is exactly why this mode narrows what the environment can claim; §9.2
of the specification records the consequence.

Two constructions are load bearing and both have been measured to fail the
obvious way.

**p_init is built, never assumed** (§15).  Each day's boundary comes from
running the *same three steps* on a one-period horizon, which is §15's "clearing
the first period on its own".  Taking the registered `unit_init_power` instead
puts first-period generation above demand and the LP is simply infeasible;
taking the committed minimum sheds 2 572 MWh, all of it in period 0, and reports
a maximum price of 10 000 \\$/MWh against 205.9.  Both failures look like a
scarce market rather than a mis-specified initial condition.

**(RMP) carries the start-up and shut-down allowances** of §6.3, and here -- unlike
in step 3, where u is fixed and both are constants on the right-hand side -- v and
w are variables, so they sit on the left.  Without them no commitment that
switches any unit admits a feasible dispatch at `ramp_scale = 0.25`, which is
every commitment step 2 produces.  The two are asymmetric, `p_min * v` upwards
against `p_max * w` downwards, because starting only has to reach the committed
minimum while stopping has to leave whatever output the previous period chose.

**The sweep is chained, and that is not a refinement.**  Day d is solved against
the boundary day d-1 ended at: its final output, its final commitment, and the
run lengths that shorten the (MU) and (MD) windows reaching back across the
boundary (§6.3).  Solving each day independently instead was measured and it does
not work.  The environment carries the previous day's final output forward, as
§9.2 requires, so an independently-solved commitment meets a boundary it was not
computed for.  Measured over the same 60 days at truthful offers, chained against
`--no-chain`: nothing shed on any day against 9 373 MWh shed over 16 of the 60, no
day priced at VOLL against 16 of them, and a total profit of -6.72e8 against
+2.08e8, which is a change of sign.  The reward becomes an artefact of the fixture
rather than a property of the market.

The symptom is milder than it was before the shut-down allowance of §6.3 was
widened: an unchained boundary used to leave the linear program infeasible, with
`mu` above 1e260, and now both sweeps converge at 1e-11 and the cost appears as
shed energy instead.  A converged solve is therefore not evidence that the
boundary is right, and one day is not enough to tell the two apart -- at T = 4 they
agree, both shedding nothing, while one unit differs by 1 532 MW at the boundary.
`--no-chain` exists so that this comparison can be re-run.

Chaining removes that at the truthful baseline exactly, because the environment
reproduces this script's boundary: HiGHS and the accelerator's own clearing
operator agree on end-of-day output to 6.3e-3 MW, which is the `OFF_EPS` phantom
scale rather than a disagreement.  What remains is the honest part of the same
effect -- an agent bidding away from cost redistributes the dispatch under a
commitment that was fixed before it bid, so the boundary drifts by however much
strategic bidding moves the final period.

Usage.  The `relax` line read `--mode relax --days 60` alone until 2026-08-24,
and the derived output path carries mode, case and period count but no scenario,
so at the CLI defaults it landed on a committed `cap 0.6 / ramp 1.0` fixture and
wrote `0.4 / 0.25` over it.  `refuse_if_scenario_moved` now stops that; the flags
below are the adopted scenario.

    python tools/commitment/precommit.py --mode relax --days 60 \
        --cap-scale 0.6 --ramp-scale 1.0 --out relax_c0.6_r1.00.npz
    python tools/commitment/precommit.py --mode milp --days 3

Fixtures land in `tests/fixtures/`, which `.gitignore` re-includes for exactly
this purpose.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from powermarketjax.case import load_case, scale_min_output
from powermarketjax.envs.day_ahead.clearing import VOLL, segment_costs
from powermarketjax.envs.day_ahead.demand import (CASE_DEMAND, RTS_NETTED,
                                                  demand_from_meta, demand_meta,
                                                  demand_pairing)

#: The scenario scales.  Neither has a safe default: at the registered values `case29gb`
#: neither congests nor binds on ramp.
CAP_SCALE = 0.4
RAMP_SCALE = 0.25
#: `mip_rel_gap` for the exact mode.  1e-4 is the gap the timings in the module
#: docstring were measured at; loosening it is the first thing to try if they grow.
MIP_REL_GAP = 1e-4

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def build(case, demand_t, cap_scale=CAP_SCALE, ramp_scale=RAMP_SCALE, K=1,
          delta_h=1.0, markup=1.0, p_init=None, u_prev=None,
          up_time=None, down_time=None, u_fixed=None):
    """The linear program of §6.3, sparse, in the canonical form

        min c'x  s.t.  A_eq x = b_eq ,  A_ub x <= b_ub ,  lo <= x <= hi

    with ``x = [g (n_units*K*T) ; s (T*n_buses) ; u (n_units*T) ; v ; w]``.

    Ordering is by variable type, not block per period: the accelerator's block
    tridiagonal structure (§15) buys nothing under a sparse simplex, and this
    ordering makes the (MU)/(MD) windows, which reach back up to 96 periods,
    plain slices.

    ``u_fixed`` selects step 3 of §6.5: u, v and w are pinned by their bounds and
    (MU), (MD) drop out, which is what §6.5 means by dropping them -- a rounded
    pattern is allowed to violate them, and pinning u while keeping the rows
    would report that as infeasibility instead.

    ``p_init=None`` drops the period-0 ramp rows, which is how §15's "clear the
    first period on its own" is stated as a problem rather than as a fix.
    """
    demand_t = np.asarray(demand_t, np.float64)
    T = len(demand_t)
    p_min = np.asarray(case.unit_p_min, np.float64)
    p_max = np.asarray(case.unit_p_max, np.float64)
    unit_bus = np.asarray(case.unit_node_idx, np.int64)
    no_load = np.asarray(case.unit_no_load_cost, np.float64)
    startup = np.asarray(case.unit_startup_cost, np.float64)
    UT = np.maximum(np.asarray(case.unit_min_up_time, np.int64), 1)
    DT = np.maximum(np.asarray(case.unit_min_down_time, np.int64), 1)
    PTDF = np.asarray(case.PTDF, np.float64)
    F = np.asarray(case.line_cap, np.float64) * cap_scale
    n_u, n_n, n_l = len(p_min), int(case.n_nodes), PTDF.shape[0]

    width, cost = segment_costs(case, K)          # the envelope of §9.3
    offer = markup * cost                         # (n_u, K)
    ramp_up = np.asarray(case.unit_ramp_up, np.float64) * p_max * delta_h * ramp_scale
    ramp_dn = np.asarray(case.unit_ramp_down, np.float64) * p_max * delta_h * ramp_scale

    # nodal demand split: `case29gb`'s load_d_max sums to 1.0, so it is bus
    # participation factors rather than MW (note lmp-dual-and-congestion §2)
    node_pd = np.asarray(case.node_pd, np.float64)
    if node_pd.sum() > 1e-6:
        share = node_pd / node_pd.sum()
    else:
        share = np.zeros(n_n)
        np.add.at(share, np.asarray(case.load_node_idx, np.int64),
                  np.asarray(case.load_d_max, np.float64))
        share = share / share.sum()
    d = share[None, :] * demand_t[:, None]                       # (T, n_n)

    n_g, n_s, n_ut = n_u * K * T, T * n_n, n_u * T
    off_g, off_s, off_u, off_v, off_w = 0, n_g, n_g + n_s, n_g + n_s + n_ut, \
        n_g + n_s + 2 * n_ut
    n = n_g + n_s + 3 * n_ut
    gi = lambda i, k, t: off_g + (i * K + k) * T + t
    si = lambda t, m: off_s + t * n_n + m
    ui = lambda i, t: off_u + i * T + t
    vi = lambda i, t: off_v + i * T + t
    wi = lambda i, t: off_w + i * T + t

    c = np.zeros(n)
    c[off_g: off_g + n_g] = delta_h * np.repeat(offer.ravel(), T)
    c[off_s: off_s + n_s] = delta_h * VOLL
    c[off_u: off_u + n_ut] = delta_h * np.repeat(no_load, T)
    c[off_v: off_v + n_ut] = np.repeat(startup, T)   # §6.3: no delta on start-up

    lo, hi = np.zeros(n), np.ones(n)
    hi[off_g: off_g + n_g] = np.repeat(np.broadcast_to(width[:, None], (n_u, K)).ravel(), T)
    hi[off_s: off_s + n_s] = d.ravel()               # (SHED), as a bound

    if u_fixed is None:
        # (MU)/(MD) initial conditions (§6.3): a unit whose run length at the
        # boundary is shorter than its window has the remainder of that window
        # forced, which is the only route by which yesterday reaches today.
        #
        # Each clause applies only in the state it is about.  A unit that is off
        # carries `up_time = 0`, and reading the remaining uptime off that would
        # force it *on* for its whole minimum uptime -- the mirror error forces a
        # running unit off.  Measured: both at once make the day-0 problem
        # infeasible, since the forced-on minima then exceed demand while the
        # capacity that could serve it is forced off.
        up_time = UT.copy() if up_time is None else np.asarray(up_time, np.int64)
        down_time = DT.copy() if down_time is None else np.asarray(down_time, np.int64)
        for i in range(n_u):
            if up_time[i] > 0:                     # on at the boundary
                for t in range(min(T, max(0, int(UT[i] - up_time[i])))):
                    lo[ui(i, t)] = 1.0
            if down_time[i] > 0:                   # off at the boundary
                for t in range(min(T, max(0, int(DT[i] - down_time[i])))):
                    hi[ui(i, t)] = 0.0
    else:
        u_fx = np.asarray(u_fixed, np.float64)
        u_fx = np.repeat(u_fx[:, None], T, axis=1) if u_fx.ndim == 1 else u_fx
        prev = np.zeros(n_u) if u_prev is None else np.asarray(u_prev, np.float64)
        v_fx = np.maximum(u_fx - np.hstack([prev[:, None], u_fx[:, :-1]]), 0.0)
        w_fx = np.maximum(np.hstack([prev[:, None], u_fx[:, :-1]]) - u_fx, 0.0)
        for arr, off in ((u_fx, off_u), (v_fx, off_v), (w_fx, off_w)):
            lo[off: off + n_ut] = hi[off: off + n_ut] = arr.ravel()
        # with u fixed the segment-width row is replaced by the box, which is
        # what §6.5 means by (CAP) holding by construction: a de-committed unit
        # gets zero width and its output collapses to p_min * 0 = 0
        hi[off_g: off_g + n_g] = (width[:, None, None] * u_fx[:, None, :]).repeat(K, 1).ravel()

    rows, cols, vals, b_eq = [], [], [], []
    rows_u, cols_u, vals_u, b_ub = [], [], [], []
    # one label per inequality row, so a caller holding `ineqlin.marginals` can
    # say which line and which period a shadow price belongs to.  Only the (L)
    # rows are labelled; everything else stays None.  Recording it here rather
    # than reconstructing the row order outside is the point: a reconstruction
    # drifts silently when a row is added, and nothing would report the drift.
    tags_u, tags_eq = [], []

    def eq(entries, rhs, tag=None):
        r = len(b_eq)
        for j, a in entries:
            rows.append(r); cols.append(j); vals.append(a)
        b_eq.append(rhs)
        tags_eq.append(tag)

    def ub(entries, rhs, tag=None):
        r = len(b_ub)
        for j, a in entries:
            rows_u.append(r); cols_u.append(j); vals_u.append(a)
        b_ub.append(rhs)
        tags_u.append(tag)

    for t in range(T):
        # (E) generation + shed = demand, with the must-run block carrying u
        e = [(gi(i, k, t), 1.0) for i in range(n_u) for k in range(K)]
        e += [(si(t, m), 1.0) for m in range(n_n)]
        e += [(ui(i, t), p_min[i]) for i in range(n_u)]
        eq(e, float(d[t].sum()), ("E", t))

        # (L) both directions on every line.  Injection at bus m is the units
        # there plus the shed there minus the demand there; the demand part is a
        # constant and moves to the right-hand side.
        for l in range(n_l):
            e = [(gi(i, k, t), PTDF[l, unit_bus[i]]) for i in range(n_u) for k in range(K)]
            e += [(ui(i, t), PTDF[l, unit_bus[i]] * p_min[i]) for i in range(n_u)]
            e += [(si(t, m), PTDF[l, m]) for m in range(n_n)]
            rhs = float(PTDF[l] @ d[t])
            ub(e, F[l] + rhs, ("L", t, l, "up"))
            ub([(j, -a) for j, a in e], F[l] - rhs, ("L", t, l, "dn"))

        # segment width is available only when the unit is committed.  With u a
        # variable this is a row; step 3 gets it for free from the box.
        if u_fixed is None:
            for i in range(n_u):
                for k in range(K):
                    ub([(gi(i, k, t), 1.0), (ui(i, t), -width[i])], 0.0)

        for i in range(n_u):
            # (RMP).  p[t] = p_min u[t] + sum_k g[t], and the two allowances sit
            # on the left because v and w are variables here rather than the
            # constants they are in step 3.  Dropping them makes every switching
            # commitment infeasible.
            #
            # The allowances are asymmetric and §6.3 says why: starting only has
            # to reach the committed minimum, so p_min*v suffices, while stopping
            # has to leave whatever output the previous period put the unit at,
            # which can be as high as p_max.  An allowance of p_min*w there leaves
            # a high-output unit unable to shut down at all, and the consequence
            # is infeasibility rather than expense.  This step and step 3 must
            # share one formulation (§6.5), which is what rules out the tighter
            # p[i,t-1]*w: that product is bilinear here, where both factors are
            # variables.
            up = [(gi(i, k, t), 1.0) for k in range(K)] + [(ui(i, t), p_min[i]),
                                                           (vi(i, t), -p_min[i])]
            dn = [(gi(i, k, t), -1.0) for k in range(K)] + [(ui(i, t), -p_min[i]),
                                                            (wi(i, t), -p_max[i])]
            # tagged for the same reason the (L) rows are: a caller holding
            # `ineqlin.marginals` cannot otherwise say which rows are ramp rows,
            # and "how many ramp rows carry a non-zero dual" is the measurement
            # that separates "ramp does not bind here" from "ramp binds but does
            # not change the commitment".  Tags are recorded, never read by the
            # solve, so this changes no number.
            if t == 0:
                if p_init is not None:
                    ub(up, float(ramp_up[i] + p_init[i]), ("RMP", t, i, "up"))
                    ub(dn, float(ramp_dn[i] - p_init[i]), ("RMP", t, i, "dn"))
            else:
                back = [(gi(i, k, t - 1), -1.0) for k in range(K)] + [(ui(i, t - 1), -p_min[i])]
                ub(up + back, float(ramp_up[i]), ("RMP", t, i, "up"))
                ub(dn + [(j, -a) for j, a in back], float(ramp_dn[i]), ("RMP", t, i, "dn"))

            # (LOG) u[t] - u[t-1] = v[t] - w[t]
            e = [(ui(i, t), 1.0), (vi(i, t), -1.0), (wi(i, t), 1.0)]
            if t == 0:
                eq(e, float(0.0 if u_prev is None else np.asarray(u_prev).ravel()[i]))
            else:
                eq(e + [(ui(i, t - 1), -1.0)], 0.0)

            if u_fixed is None:
                # (MU) a unit started within the last UT periods is still on;
                # (MD) the same on shut-downs.  The window is truncated at the
                # day start, which is what `up_time`/`down_time` above stand in
                # for.
                lo_t = max(0, t - int(UT[i]) + 1)
                ub([(vi(i, tau), 1.0) for tau in range(lo_t, t + 1)]
                   + [(ui(i, t), -1.0)], 0.0)
                lo_t = max(0, t - int(DT[i]) + 1)
                ub([(wi(i, tau), 1.0) for tau in range(lo_t, t + 1)]
                   + [(ui(i, t), 1.0)], 1.0)

    A_eq = sp.coo_matrix((vals, (rows, cols)), shape=(len(b_eq), n)).tocsr()
    A_ub = sp.coo_matrix((vals_u, (rows_u, cols_u)), shape=(len(b_ub), n)).tocsr()
    return dict(c=c, A_eq=A_eq, b_eq=np.asarray(b_eq), A_ub=A_ub,
                b_ub=np.asarray(b_ub), lo=lo, hi=hi, T=T, K=K, n_u=n_u, n_n=n_n,
                n_l=n_l, off_g=off_g, off_s=off_s, off_u=off_u, off_v=off_v,
                off_w=off_w, p_min=p_min, offer=offer, d=d, delta_h=delta_h,
                no_load=no_load, startup=startup, integral_slice=slice(off_u, off_u + n_ut),
                # echoed so a caller can print the scales the LP was actually
                # built with rather than the ones it believes it passed: the two
                # scripts in this directory carry different defaults for them
                cap_scale=cap_scale, ramp_scale=ramp_scale, ub_tags=tags_u, eq_tags=tags_eq,
                line_cap=F, PTDF=PTDF, unit_bus=unit_bus, demand_nodal=d)


def unpack(lp, x):
    """Dispatch, shed, commitment and the objective split by term."""
    T, K, n_u, n_n = lp["T"], lp["K"], lp["n_u"], lp["n_n"]
    g = x[lp["off_g"]: lp["off_s"]].reshape(n_u, K, T)
    s = x[lp["off_s"]: lp["off_u"]].reshape(T, n_n)
    u = x[lp["off_u"]: lp["off_v"]].reshape(n_u, T)
    v = x[lp["off_v"]: lp["off_w"]].reshape(n_u, T)
    p = lp["p_min"][:, None] * u + g.sum(1)
    dh = lp["delta_h"]
    return dict(
        p=p, shed=s, u=u, v=v,
        energy=dh * float((lp["offer"][:, :, None] * g).sum()),
        no_load=dh * float(lp["no_load"] @ u.sum(1)),
        start=float(lp["startup"] @ v.sum(1)),
        voll=dh * VOLL * float(s.sum()),
        shed_mwh=dh * float(s.sum()))


def solve(lp, integral=False, mip_rel_gap=MIP_REL_GAP):
    """HiGHS through scipy: `linprog` for the relaxation, `milp` when u is binary."""
    t0 = time.perf_counter()
    if integral:
        integrality = np.zeros(len(lp["c"]))
        integrality[lp["integral_slice"]] = 1        # only u; (LOG) pins v and w
        cons = [LinearConstraint(lp["A_eq"], lp["b_eq"], lp["b_eq"]),
                LinearConstraint(lp["A_ub"], -np.inf, lp["b_ub"])]
        r = milp(lp["c"], constraints=cons, integrality=integrality,
                 bounds=Bounds(lp["lo"], lp["hi"]),
                 options=dict(mip_rel_gap=mip_rel_gap))
    else:
        r = linprog(lp["c"], A_ub=lp["A_ub"], b_ub=lp["b_ub"], A_eq=lp["A_eq"],
                    b_eq=lp["b_eq"], bounds=np.stack([lp["lo"], lp["hi"]], 1),
                    method="highs")
    assert r.status == 0, f"solver status {r.status}: {r.message}"
    out = unpack(lp, np.asarray(r.x))
    out.update(obj=float(r.fun), seconds=time.perf_counter() - t0, status=int(r.status))
    return out


def run_lengths(u, up_time, down_time):
    """Advance the consecutive-on and consecutive-off counts across a day.

    The same recursion the environment runs, so the boundary this script hands to
    day d+1 is the boundary the environment carries there.  Only one of the two
    is non-zero for a given unit at a given period.
    """
    up, down = np.asarray(up_time).copy(), np.asarray(down_time).copy()
    for t in range(u.shape[1]):
        on = u[:, t] > 0.5
        up = np.where(on, up + 1, 0)
        down = np.where(on, 0, down + 1)
    return up.astype(np.int64), down.astype(np.int64)


def first_boundary(case, demand_0, **kw):
    """§15: p_init and the commitment behind it, from clearing period 0 alone.

    The same three steps on a one-period horizon, so the boundary is produced by
    the market's own mechanism rather than by a rule invented for it.  There is
    no previous period, so the period-0 ramp rows are absent (`p_init=None`).

    `u_prev` is all-on for this solve alone.  There is no earlier day to have
    started from, and the alternative -- all-off -- charges every committed unit
    a start-up cost for a start that no day made, which biases the boundary
    towards committing fewer units than the period can use.

    Both modes take their boundary from here, relaxed, so that the two fixtures
    describe the same first day and their objectives subtract.  A boundary that
    differed between the modes would fold into the measured gap.

    The run lengths of this one boundary are unconstrained -- `up_time = UT` for a
    unit on, `down_time = DT` for a unit off -- because there is no earlier day to
    read them from.  Every later day of the sweep has real ones.
    """
    n_u = len(np.asarray(case.unit_p_min))
    prev = np.ones(n_u)
    relaxed = solve(build(case, [demand_0], p_init=None, u_prev=prev, **kw))
    u_int = (relaxed["u"] > 0.0).astype(np.float64)
    fixed = solve(build(case, [demand_0], p_init=None, u_prev=prev,
                        u_fixed=u_int, **kw))
    on = u_int[:, 0] > 0.5
    UT = np.maximum(np.asarray(case.unit_min_up_time, np.int64), 1)
    DT = np.maximum(np.asarray(case.unit_min_down_time, np.int64), 1)
    return dict(p_init=fixed["p"][:, 0], u_prev=u_int[:, 0],
                up_time=np.where(on, UT, 0), down_time=np.where(on, 0, DT))


def precommit_day(case, demand_t, mode, boundary, **kw):
    """One market day from a given boundary: step 1 (or §6.1), step 2, step 3."""
    lp = build(case, demand_t, **boundary, **kw)
    first = solve(lp, integral=(mode == "milp"))
    # step 2, as revised 2026-08-10: commit every unit the relaxation
    # touches at all, because committing a spare unit costs a no-load rate in
    # $/h while leaving one off costs shed at VOLL in $/MWh, and the two are
    # orders of magnitude apart.  §17 carries the gap against the
    # exact optimum; no number is quoted here, since the anchor scenario is
    # its own and this script measures its own (see the fixture metadata).
    # Under `milp` u is already integer and this is a no-op up to rounding.
    u_int = (first["u"] > 0.0).astype(np.float64)
    third = solve(build(case, demand_t, u_fixed=u_int, **boundary, **kw))
    up, down = run_lengths(u_int, boundary["up_time"], boundary["down_time"])
    return dict(u=u_int, boundary=boundary, first=first, third=third,
                next_boundary=dict(p_init=third["p"][:, -1], u_prev=u_int[:, -1],
                                   up_time=up, down_time=down),
                integrality_gap=float(np.abs(first["u"] - u_int).max()),
                n_committed=int(u_int.sum()))


#: The fields that make two fixtures at one path different markets rather than two
#: runs of one.  `make_env` compares the first three of the scales and refuses a
#: mismatch (`envs/day_ahead/env.py`), but it never sees the day window, so an
#: overwrite that keeps the scales and moves the 60 days is silent all the way
#: through.  Compared here because the derived path names none of them.
#:
#: **The demand pair joined on 2026-09-05** and it belongs for the same reason
#: the scales do: `floor_mw` is which market, not a tuning knob.  At 2 500 MW the
#: RTS floor pins forecast equal to realisation on 4 499 of 8 784 hours (51.2%),
#: at 2 800 on 5 348 (60.9%) (`load_rts_demand`).  **Those two are properties of
#: the demand series alone and stand.**  The shed figures measured beside them
#: are deliberately not quoted here: a year-long RTS driver passed `down_time = 24` every
#: day and never advanced the counter, so the three RTS units with
#: `min_down_time = 48` were never committed in any of the 366 days, and the
#: clearing-side readings of the 2026-09-04 adoption are pending a re-read
#: (the correction withdrew the "two paths differ by 18%" finding).  Two RTS runs that disagree
#: on the floor landed on one derived path and the second overwrote the first in
#: silence.
#:
#: **Naming the demand in the path instead was the other candidate and does not
#: work here**: seven places in this tree spell that filename out, and five of
#: them are readers (`envs/day_ahead/commitment.load_commitment` and four
#: `tools/callback_arch/` drivers) that cannot know a floor before they open the
#: file -- the floor is what the file records.  A path they cannot derive is a
#: naming rule kept in sync by hand in seven copies, which is the failure
#: `demand_from_meta` was built to end rather than one to add to.
#:
#: `milp_reference.py` did put its scenario in the name (2026-08, its derived path
#: carries both scales, the window and `--exact-mumd`) and that is not the same
#: situation: **one** place derives that name, the writer, and every consumer
#: spells the file out as a literal or a glob.  Nothing reads it by rebuilding
#: the name from parameters, and the parameters in it are ones a consumer holds
#: anyway -- they are the run point being compared at.  The demand pairing is the
#: one scenario class a consumer is *supposed* to learn from the product rather
#: than restate, which is what `demand_from_meta` is for; putting it in the name
#: would ask every reader to know it in advance.
SCENARIO_KEYS = ("mode", "case", "n_periods", "n_segments", "cap_scale",
                 "ramp_scale", "p_min_scale", "demand_source", "demand_kwargs")

#: Fields whose *absence* from a prior product has a known meaning, mapped to the
#: value it must have had.  `refuse_if_scenario_moved` fills these in before
#: comparing, so an old product is compared rather than skipped.
#:
#: The distinction this table draws, and why it is not a patch on the ABSENT
#: branch: `p_min_scale` has an implied prior value because before the field
#: existed nothing was scaled, so it was 1.0, and reading it that way is exact in
#: both directions.  The demand pairing has none -- a product that does not
#: record it does not tell you which series it was built on -- so it stays
#: uncompared and rests on the external fact `refuse_if_scenario_moved`
#: documents.  Membership here is therefore a claim about the field, not a
#: convenience: put a field in only if its pre-field value is knowable.
#:
#: The failure this closes (named by `powermarketjax-ba` when the field was
#: proposed, and held by
#: `tests/envs/day_ahead/test_p_min_scale_l0.py::test_a_prior_without_the_scale_is_read_as_unscaled_not_as_uncomparable`):
#: an existing `29gb` product built before the field, overwritten by a `29gb` run
#: at `p_min_scale = 0.80`.  Same case, every other scenario field equal, ABSENT
#: on this one, and a genuinely different market -- silently overwritten.
IMPLIED_PRIOR = {"p_min_scale": 1.0}


def demand_meta_for(case, floor_mw=None, netted=None):
    """This script's demand flags, as the record `demand_meta` validates.

    A flag left off is a pairing argument not supplied, so `demand_meta` refuses
    the case that needs one rather than this function inventing a value: that is
    why `load_rts_demand` and `load_nem_demand` give ``floor_mw`` no default.
    ``netted`` reads as an exception and is not one -- `load_rts_demand` does
    have a default for it, and this mirrors that default rather than adding one,
    so what ran is still what the record says ran.
    """
    return demand_meta(case, **demand_pairing(case, floor_mw=floor_mw,
                                              netted=netted))


def scenario_at(path):
    """The scenario a fixture already at `path` describes, or None if unreadable.

    **Only the fields it actually records.**  A key the fixture does not carry is
    left out rather than mapped to `None`, so that "this product predates the
    field" stays distinguishable from "this product was built at another value";
    `meta.get` folds the two together and the caller then cannot tell them apart.
    The mirror-image failure is an identity key over possibly-missing fields
    reading "neither side recorded it" as "the two agree"; this is the same rule
    from the other side, and the
    failure direction it avoids here is a refusal for a move never measured.

    `meta` is stored as a JSON string, not a dict, so `str()` then `json.loads`
    is the only route in: `.item()` returns the text and iterating that yields
    characters, which makes every key lookup miss and return `None` silently.
    """
    try:
        z = np.load(path, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        out = {k: meta[k] for k in SCENARIO_KEYS if k in meta}
        out["day_index"] = np.asarray(z["day_index"]).tolist()
        return out
    except Exception:
        return None


def refuse_if_scenario_moved(path, now, force):
    """Print the field-by-field comparison, and stop unless nothing moved.

    Shaped after `da_position.py`, which refuses when a regeneration moves the
    position it is about to overwrite.  The case this exists for: the derived
    path carries mode, case and period count and no scenario, so a run at the
    CLI defaults lands on a committed fixture built at other scales and the write
    itself says nothing.

    **A field the prior does not record is printed as ABSENT and is not a move.**
    It cannot be one: nothing was compared.  Refusing on it would refuse every
    product built before the field existed, which for the demand pair is every
    commitment product in `tests/fixtures/` -- five of them, 16 to 22 meta keys
    each and not one about demand.  This fails open on such a field, and for the
    demand pair the compensating fact is checkable rather than hoped for: a
    non-GB product carrying no demand stamp cannot be read at all
    (`demand_from_meta` raises, `LEGACY_DEMAND_CASE` scopes the fallback to
    `29gb`, and a test holds that), so a prior that is absent here *and* agrees on `case` can only
    be a `29gb` one, whose only pairing is `load_gb_demand()` with no arguments
    -- which is what the run comparing against it must also have used.
    """
    prior = scenario_at(path)
    if prior is None:
        print(f"  NOTE {path.name} exists, and its scenario could not be read; "
              f"overwriting with no comparison")
        return
    prior = dict(prior)
    implied = [k for k in now if k not in prior and k in IMPLIED_PRIOR]
    for k in implied:
        prior[k] = IMPLIED_PRIOR[k]
    absent = [k for k in now if k not in prior]
    moved = [k for k in now if k in prior and prior[k] != now[k]]
    for k in now:
        if k in absent:
            print(f"  ABSENT {k}: not recorded by {path.name}, not compared")
            continue
        if k in implied:
            print(f"  IMPLIED {k}: not recorded by {path.name}, read as "
                  f"{IMPLIED_PRIOR[k]} (its value before the field existed)")
        a, b = prior[k], now[k]
        if k == "day_index":
            a, b = _days_summary(a), _days_summary(b)
        print(f"  {'MOVED' if k in moved else 'same '} {k}: {a}"
              + (f" -> {b}" if k in moved else ""))
    if moved and not force:
        print(f"  {path} already holds a different scenario, moved in "
              f"{len(moved)} of {len(now) - len(absent)} compared fields; "
              f"nothing written.  Pass --force to overwrite it, or --out to "
              f"write elsewhere")
        raise SystemExit(2)


def _days_summary(idx):
    """`[5, 6, ..., 299]` as `60 days, 5..299, runs=4` -- the gaps are the point."""
    if not idx:
        return "none"
    runs = 1 + sum(1 for a, b in zip(idx, idx[1:]) if b != a + 1)
    return f"{len(idx)} days, {idx[0]}..{idx[-1]}, runs={runs}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("relax", "milp"), default="relax")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--start-day", type=int, default=0)
    ap.add_argument("--case", default="29gb")
    ap.add_argument("--demand-floor-mw", type=float, default=None,
                    help="the floor the case's demand series is clipped at, in "
                         "MW.  Has no default because `load_rts_demand` and "
                         "`load_nem_demand` have none: it is part of the run "
                         "point, and 29gb takes no floor at all")
    ap.add_argument("--demand-netted", nargs="+", default=None,
                    choices=list(RTS_NETTED),
                    help="73rts only: which renewable classes to net off "
                         "demand.  Defaults to all four, as `load_rts_demand` "
                         "does; whichever ran is written into the fixture")
    ap.add_argument("--periods", type=int, default=24)
    ap.add_argument("--segments", type=int, default=1)
    #: **The default lives in the code rather than in `default=`, so that
    #: "passed or not" can be told apart.**  The value is unchanged (not given
    #: is still `CAP_SCALE` / `RAMP_SCALE`), whereas `default=CAP_SCALE` would
    #: make `args.cap_scale is None` always false -- the "flag or default"
    #: criterion would then only ever show one direction, and what it prints
    #: would look perfectly normal.
    ap.add_argument("--cap-scale", type=float, default=None,
                    help=f"if not given, this file's {CAP_SCALE}; `da_position` "
                         f"defaults to 0.60 and `milp_reference` falls back to "
                         f"the 29gb fixture meta's 0.6 -- the three tools each "
                         f"have their own default for this quantity, so give it "
                         f"explicitly when comparing across tools.")
    ap.add_argument("--ramp-scale", type=float, default=None,
                    help=f"if not given, this file's {RAMP_SCALE} (`da_position`: 1.00).")
    ap.add_argument("--p-min-scale", type=float, default=1.0,
                    help="scale on every unit's minimum output.  1.0 is the "
                         "registered case and the default, so a run that does "
                         "not ask for it gets RTS-96 as registered.  73rts is "
                         "the case this exists for: its aggregate p_min/p_max "
                         "is 0.464 against 0.274 (813nem) and 0.200 (29gb), "
                         "and a commitment sized for the peak cannot shut down "
                         "to the trough")
    ap.add_argument("--no-chain", action="store_true",
                    help="solve every day from a boundary reconstructed for it "
                         "rather than from the one the previous day ended at.  "
                         "Kept so that the comparison in the module docstring can "
                         "be re-run rather than taken on trust")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing fixture whose scenario differs "
                         "from this run's.  Without it such a write is refused, "
                         "because the derived path carries no scenario and the "
                         "two products are indistinguishable by name")
    args = ap.parse_args()

    # the scenario scale is applied once, to the case, so that every operator
    # reading `unit_p_min` -- `segment_costs` inside `build`, and the accelerator
    # if this fixture is later compared against it -- sees the same one
    case = scale_min_output(load_case(args.case), args.p_min_scale)
    # the demand comes from the same case the network does, and the record of
    # which series that was is built first so that what runs below is literally
    # what the fixture will carry
    demand_record = demand_meta_for(args.case, floor_mw=args.demand_floor_mw,
                                    netted=args.demand_netted)
    _, actual, days = demand_from_meta(dict(demand_record, case=args.case))
    idx = np.arange(args.start_day, args.start_day + args.days)
    assert idx[-1] < len(days), f"only {len(days)} days available"
    cap = CAP_SCALE if args.cap_scale is None else args.cap_scale
    ramp = RAMP_SCALE if args.ramp_scale is None else args.ramp_scale
    kw = dict(K=args.segments, cap_scale=cap, ramp_scale=ramp)

    #: **Print the resolved scenario and its source, as `da_position` and
    #: `milp_reference` do.**  Those two print this line at start-up and this
    #: file previously did not -- while the three tools each have their own
    #: default for the same named quantity (this file 0.4 / 0.25, `da_position`
    #: 0.60 / 1.00, `milp_reference` no module constant, falling back to the
    #: 29gb fixture meta's 0.6 / 1.0).  **Running the three tools without flags
    #: gets three scenarios, and the three products each record theirs
    #: faithfully, with nothing turning red.**
    #:
    #: Why print rather than only document it in `--help`: the launch log
    #: records the full command, **so the flags that were passed are in it and
    #: the ones that were not are invisible**, and this line supplies exactly
    #: the missing half; `--help` is seen only by whoever asks.
    src = " ".join(
        f"{k}={v}<-{'flag' if a is not None else 'default(' + __file__.split('/')[-1] + ')'}"
        for k, v, a in (("cap_scale", cap, args.cap_scale),
                        ("ramp_scale", ramp, args.ramp_scale)))
    print(f"scenario: {src} p_min_scale={args.p_min_scale} K={kw['K']} "
          f"periods={args.periods} days={args.days} start_day={args.start_day}",
          flush=True)

    out, wall = [], time.perf_counter()
    boundary = first_boundary(case, float(actual[idx[0], 0]), **kw)
    for j, day in enumerate(idx):
        demand_t = actual[day, : args.periods].astype(np.float64)
        if args.no_chain and j > 0:
            boundary = first_boundary(case, float(demand_t[0]), **kw)
        r = precommit_day(case, demand_t, args.mode, boundary, **kw)
        boundary = r["next_boundary"]
        out.append(r)
        print(f"[{j + 1}/{len(idx)}] {days[day].date()} demand "
              f"{demand_t.sum():.0f} MWh  committed {r['n_committed']}/"
              f"{r['u'].size}  obj {r['third']['obj']:.6e}  shed "
              f"{r['third']['shed_mwh']:.1f} MWh  "
              f"{r['first']['seconds']:.2f}+{r['third']['seconds']:.2f} s",
              flush=True)

    meta = dict(
        mode=args.mode, case=args.case, n_periods=args.periods,
        n_segments=args.segments, cap_scale=cap,
        ramp_scale=ramp, p_min_scale=args.p_min_scale,
        markup=1.0, voll=VOLL,
        rounding="u > 0 (ADR-0003, revised 2026-08-10)",
        mip_rel_gap=(MIP_REL_GAP if args.mode == "milp" else None),
        offer_basis="monotone envelope of the true segment costs, markup 1",
        chained=not args.no_chain,
        boundary="chained: day d is solved against the boundary day d-1 ended at "
                 "(final output, final commitment, and the run lengths that "
                 "shorten the (MU)/(MD) windows across it).  The first day of the "
                 "sweep comes from the same three steps on a one-period horizon "
                 "(§15) with unconstrained run lengths",
        dates=[str(days[d].date()) for d in idx],
        day_index=idx.tolist(), scipy=scipy.__version__,
        #: **`seconds_total` is the timed region, `seconds` below is per day --
        #: and `seconds_total / n_days` is the per-day cost, while the `seconds`
        #: array is NOT.**  This corrects what stood here earlier the same day.
        #:
        #: The earlier reading was `case813nem --days 1`: **61.24 s** in the timed
        #: region against **24.19 s** in that day's own two solves, and the 37 s
        #: gap was written up as a one-off §15 cold start, making the `seconds`
        #: array the slope.  **A second point refutes that.**  The 365-day sweep
        #: of the same case took **23 172.8 s** with its `seconds` array summing
        #: to **9 207.0 s**.  Two points, 1 and 365 days, give a slope of
        #: **63.49 s/day and an intercept of -2.25 s**, i.e. no intercept worth
        #: naming; the 37 s is **per-day work outside the two solve calls** (the
        #: LP assembly, the rounding, the boundary bookkeeping), which recurs
        #: every day rather than once.  Had the earlier model held, 365 days
        #: would have cost 37 + 365 x 25 = 9 162 s -- out by 14 000 s.
        #:
        #: `case73rts` behaves the same way: **707.7 s / 366 = 1.93 s/day**, two
        #: point slope also **1.93**, intercept **-0.14 s**, and its `seconds`
        #: array median (0.90 s) is 47% of that.  **So the per-day array is 39%
        #: (813nem) to 47% (73rts) of the real per-day cost** -- it is the solver
        #: time, useful as a lower bound and for comparing solve difficulty, and
        #: wrong by more than a factor of two if quoted as throughput.
        seconds_total=time.perf_counter() - wall,
        **demand_record)
    path = Path(args.out) if args.out else FIXTURE_DIR / (
        f"day_ahead_commitment_{args.case}_T{args.periods}_{args.mode}.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        refuse_if_scenario_moved(
            path, dict({k: meta[k] for k in SCENARIO_KEYS},
                       day_index=idx.tolist()), args.force)
    np.savez_compressed(
        path,
        commitment=np.stack([r["u"] for r in out]).astype(np.int8),
        # the boundary each day was solved against, which is what `reset` needs:
        # an episode may start at any day of the fixture, and the day it starts
        # at has no predecessor inside the episode
        p_init=np.stack([r["boundary"]["p_init"] for r in out]).astype(np.float64),
        commitment_prev=np.stack([r["boundary"]["u_prev"] for r in out]).astype(np.int8),
        up_time=np.stack([r["boundary"]["up_time"] for r in out]).astype(np.int32),
        down_time=np.stack([r["boundary"]["down_time"] for r in out]).astype(np.int32),
        day_index=idx.astype(np.int32),
        objective=np.array([r["third"]["obj"] for r in out]),
        objective_relaxed=np.array([r["first"]["obj"] for r in out]),
        shed_mwh=np.array([r["third"]["shed_mwh"] for r in out]),
        integrality_gap=np.array([r["integrality_gap"] for r in out]),
        seconds=np.array([r["first"]["seconds"] + r["third"]["seconds"] for r in out]),
        meta=np.array(json.dumps(meta, indent=1)))
    print(f"\nwrote {path}  ({path.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
