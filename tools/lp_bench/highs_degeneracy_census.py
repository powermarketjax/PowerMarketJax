r"""Is the day-ahead operator's disagreement with HiGHS confined to degenerate points?

The ancillary-services market (03) answered the same question with
`tools/ancillary/highs_arbitration.py`; this is that question asked of 01, and
it is asked with 01's own numbers, because the boundary 03 measured (relative
spacing 1e-5 clean, 1e-7 not) is a property of 03's solver and does not
transfer.

**Why the question is not "are they equal".**  The interior-point method reports
the analytic centre of the dual optimal face; a simplex method reports a vertex
of it.  Where that face is
a single point the two must agree.  Where it is not -- where demand falls on a
capacity breakpoint -- they must be expected to differ *without either being
wrong*, and a comparison that does not separate the two cases reports an
implementation error that is not there.  So the census records, for every
period, both the disagreement and an independent verdict on whether that
period is degenerate.

**The degeneracy verdict is computed from the HiGHS solution alone.**  Not from
the operator under test, which would make the argument circular: the thing being
asked about would be supplying the excuse.  Two indicators, one per form of
degeneracy §7 names:

* `n_frac`, the number of strictly interior segments.  When it is zero, demand
  sits exactly on a breakpoint and the balance dual is an *interval*; `dual_gap`
  is that interval's width, in $/MWh, and it is the second form -- the one that
  makes the price non-unique.
* `n_tie_at_lambda`, how many committed units offer at the balance price.  That
  is the first form, which leaves the price single-valued and the award
  undetermined.  `case29gb` at K=1 has 65 distinct offers among 66 units, so
  this form is nearly absent here; it is measured rather than assumed.

**The judgement is on revenue, not on megawatts and not on the objective.**  A
megawatt difference between two optima of the same face costs nobody anything;
the repository has recorded four times that a quantity difference need not be a
money difference, and once that the objective is blind to a 159 600 $ revenue
error.  So the gate is per-unit energy revenue against HiGHS, and `mu`, `d_lmp`,
`d_award` and the objective are recorded beside it and never used to decide
whether a period disagrees.

**Reading HiGHS's duals is itself a derivation that can be wrong**, and a sign
slip produces a perfectly ordinary-looking nodal price.  Two checks stop the run
rather than report: the stationarity identity has to close, which pins the
convention of all four multiplier families at once, and the nodal price is
computed twice by routes that are equal only if that identity holds.

    JAX_PLATFORMS=cpu python tools/lp_bench/highs_degeneracy_census.py \
        --days 60 --periods 24 --out runs/highs_census/census.npz
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# The census compares against the numpy reference in `tests/envs/day_ahead/`, so
# the repository root has to be importable when the script is run as documented.
import sys                                                        # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax                                                        # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                           # noqa: E402
from scipy.optimize import linprog                                # noqa: E402

from powermarketjax.case import (load_case,       # noqa: E402
                                 scale_min_output)
from powermarketjax.envs.day_ahead import make_clearing, segment_costs  # noqa: E402
from powermarketjax.envs.day_ahead.clearing import (MAX_ITER, OFF_EPS,  # noqa: E402
                                                    VOLL)
from powermarketjax.envs.day_ahead.commitment import load_commitment  # noqa: E402
from powermarketjax.envs.day_ahead.demand import demand_from_meta  # noqa: E402
from powermarketjax.envs.day_ahead.env import MU_TOL              # noqa: E402
from tests.envs.day_ahead import reference                        # noqa: E402

#: Judgement threshold, per (unit, period) energy revenue in dollars, against
#: HiGHS.  The 03 arbitration used this same figure for the same reason: it is
#: the repository's L2 money budget, so a disagreement below it is one this
#: project has already declared it does not distinguish.  Fixed before the
#: first solve.
REVENUE_TOL = 1e-4

#: Largest tolerated relative stationarity residual when reading HiGHS duals.
#: This is a convention check, not a precision one: a wrong sign convention
#: misses by order 1e-2, so anything near machine precision passes and anything
#: structurally wrong misses by nine orders.
DUAL_RESIDUAL_TOL = 1e-9

#: A segment counts as strictly interior between these fractions of its width.
#: Relative, so it means the same thing for a 20 MW segment and a 2000 MW one.
FRAC_EPS = 1e-9

#: "Offering at the balance price", relative.  Looser than FRAC_EPS on purpose:
#: `case29gb`'s smallest positive gap between two distinct offers is 8.9e-3
#: $/MWh, so anything below that separates every genuinely distinct pair.
TIE_REL = 1e-6

#: Binding, for counting active rows out of HiGHS's inequality marginals.
BINDING_TOL = 1e-9


def _blob(path):
    """Content identity of the apparatus that actually ran; see the twin note in
    `renumber_determinism.py`.  `git hash-object` on the working tree, because a
    HEAD blob names the file as last committed and a commit hash dies to rebase."""
    try:
        return subprocess.run(["git", "hash-object", path], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def highs(lp):
    """Solve the reference LP with HiGHS and recover what the market consumes.

    Returns None if HiGHS fails.  The award and shed come straight off `x`; the
    nodal price does not exist in the solver's output at all and is built from
    the duals, so it carries its own two checks back to the caller.
    """
    r = linprog(lp["c"], A_ub=lp["A_ub"], b_ub=lp["b_ub"], A_eq=lp["A_eq"],
                b_eq=lp["b_eq"], bounds=np.stack([lp["lo"], lp["hi"]], 1),
                method="highs")
    if r.status != 0:
        return None
    x = np.asarray(r.x)
    mu_i = np.asarray(r.ineqlin.marginals)
    lam_e = np.asarray(r.eqlin.marginals)
    z_lo = np.asarray(r.lower.marginals)
    z_up = np.asarray(r.upper.marginals)

    # pins the sign convention of all four multiplier families at once
    aty = lp["A_ub"].T @ mu_i + lp["A_eq"].T @ lam_e
    c_inf = max(float(np.abs(lp["c"]).max()), 1.0)
    stat = float(np.abs(lp["c"] - (aty + z_lo + z_up)).max()) / c_inf

    T, nb, n_u, K = lp["T"], lp["nb"], lp["n_units"], lp["K"]
    n_lines, n_buses = lp["n_lines"], lp["n_buses"]
    lmp_a = np.zeros((T, n_buses)); lmp_b = np.zeros((T, n_buses))
    award = np.zeros((n_u, T)); shed = np.zeros((T, n_buses)); g = np.zeros((n_u, T))
    for t in range(T):
        sl = slice(t * nb + n_u * K, (t + 1) * nb)              # the shed columns
        # two constructions of the same number: through A^T y, and through the
        # shed column's own reduced cost.  Equal iff the identity above holds.
        lmp_a[t] = aty[sl] + z_up[sl]
        lmp_b[t] = lp["c"][sl] - z_lo[sl]
        xt = x[t * nb: (t + 1) * nb]
        g[:, t] = xt[:n_u * K].reshape(n_u, K).sum(1)
        award[:, t] = (lp["p_min"] * lp["u"][:, t] + g[:, t]) * lp["u"][:, t]
        shed[t] = np.where(lp["d"][t] > 0.0, xt[n_u * K:], 0.0)
    return dict(award=award, lmp=lmp_a, shed=shed, g=g, x=x, lam_e=lam_e,
                mu_i=mu_i, fun=float(r.fun), stationarity_rel=stat,
                d_lmp_two_ways=float(np.abs(lmp_a - lmp_b).max()))


def degeneracy(lp, ref, offer, u, width, t):
    """Per-period degeneracy indicators, from the HiGHS vertex only.

    `n_frac == 0` is §7's second form (demand on a capacity breakpoint, price an
    interval of width `dual_gap`); `n_tie_at_lambda` is §7's first form (award
    undetermined, price single-valued).
    """
    n_lines, n_u = lp["n_lines"], lp["n_units"]
    on = u[:, t] > 0.0
    box = np.where(on, width, OFF_EPS)
    frac = np.where(box > 0, ref["g"][:, t] / np.maximum(box, 1e-30), 0.0)
    interior = on & (frac > FRAC_EPS) & (frac < 1.0 - FRAC_EPS)
    full = on & (frac >= 1.0 - FRAC_EPS)
    empty = on & (frac <= FRAC_EPS)
    px = offer[:, 0, t]
    lam_t = float(ref["lam_e"][t])
    gap = np.nan
    if not interior.any() and empty.any() and full.any():
        gap = float(px[empty].min() - px[full].max())
    mt = ref["mu_i"][2 * n_lines * t: 2 * n_lines * (t + 1)]
    r0 = lp["n_line_rows"] + 2 * n_u * t
    mr = ref["mu_i"][r0: r0 + 2 * n_u]
    return dict(
        n_frac=int(interior.sum()), n_on=int(on.sum()),
        dual_gap=gap,
        n_tie_at_lambda=int((on & (np.abs(px - lam_t)
                                   <= TIE_REL * max(abs(lam_t), 1.0))).sum()),
        lambda_highs=lam_t,
        n_line_binding=int((np.abs(mt) > BINDING_TOL).sum()),
        n_ramp_binding=int((np.abs(mr) > BINDING_TOL).sum()),
        shed_mw=float(ref["shed"][t].sum()))


def run_constructed(case, cost, width, args):
    """Operating points built to *be* degenerate, because the real ones are not.

    A census over the days the market actually runs answers "do they disagree".
    It cannot answer "is the disagreement confined to degenerate points" unless
    it contains some, and if it contains none then the second question is empty
    on that population -- `vacuous`, not `pass`.  So the run point is
    constructed here rather than the gate being widened.

    The construction is exact and needs no search.  Order the committed segments
    by offer price and set demand to must-run plus the first `k` segment widths.
    Then the first `k` segments are exactly full, the rest exactly empty, no
    segment is interior, and the balance dual is the interval between the `k`-th
    and `(k+1)`-th offer price -- §7's second form, the one that makes the price
    non-unique.  `p_init` is set to that same merit dispatch so the ramp rows are
    slack and the breakpoint is not perturbed by them.

    Whether the construction took is asserted, not assumed: congestion can force
    an out-of-merit dispatch, and then the point is simply not degenerate.  Those
    rows are kept and marked `construction_took=0` rather than dropped, because
    a silently shrunk sample reads exactly like a full one.
    """
    T = 1
    n_u = int(case.n_units)
    pmin = np.asarray(case.unit_p_min, np.float64)
    offer = np.broadcast_to(cost[:, :, None] * args.markup, (n_u, 1, T)).copy()
    u = np.ones((n_u, T))
    px = offer[:, 0, 0]
    order = np.argsort(px, kind="stable")
    cum = np.cumsum(width[order])
    must = float(pmin.sum())
    ks = np.unique(np.linspace(3, n_u - 3, args.constructed_points).astype(int))
    print(f"constructed: {len(ks)} breakpoints k={list(ks)} over the merit "
          f"order of {n_u} committed segments, must-run {must:.1f} MW, "
          f"caps {args.constructed_caps}")

    rows = []
    for cap in args.constructed_caps:
        clear, spec = make_clearing(case, T, n_segments=1, cap_scale=cap,
                                    ramp_scale=args.ramp_scale,
                                    max_iter=args.max_iter)
        clear = jax.jit(clear)
        for k in ks:
            D = must + float(cum[k - 1])
            p_init = pmin.copy()
            p_init[order[:k]] += width[order[:k]]
            dem = np.array([D])
            lp = reference.build_lp(case, offer, u, dem, p_init, cap,
                                    args.ramp_scale)
            ref = highs(lp)
            if ref is None:
                print(f"  cap {cap} k {k}: HiGHS failed, skipped")
                del lp
                continue
            if ref["stationarity_rel"] > DUAL_RESIDUAL_TOL:
                raise RuntimeError(f"HiGHS duals did not close at cap {cap} k "
                                   f"{k}: {ref['stationarity_rel']:.3e}")
            out = {k2: np.asarray(v) for k2, v in
                   clear(jnp.asarray(offer), jnp.asarray(u), jnp.asarray(dem),
                         jnp.asarray(p_init)).items()}
            deg = degeneracy(lp, ref, offer, u, width, 0)
            unit_bus = np.asarray(case.unit_node_idx, np.int64)
            rev_i = out["award"][:, 0] * out["lmp"][0, unit_bus]
            rev_h = ref["award"][:, 0] * ref["lmp"][0, unit_bus]
            Z_i = float((offer[:, 0, :] * out["award"]).sum()
                        + VOLL * out["shed"].sum())
            Z_h = float((offer[:, 0, :] * ref["award"]).sum()
                        + VOLL * ref["shed"].sum())
            took = float(deg["n_frac"] == 0 and np.isfinite(deg["dual_gap"])
                         and deg["dual_gap"] > 0.0)
            rows.append(dict(
                day=-1, period=int(k), cap_scale=float(cap), demand=D,
                construction_took=took,
                dual_gap_theory=float(px[order[k]] - px[order[k - 1]]),
                d_energy_revenue=float(np.abs(rev_i - rev_h).max()),
                d_lmp=float(np.abs(out["lmp"][0] - ref["lmp"][0]).max()),
                d_award=float(np.abs(out["award"][:, 0] - ref["award"][:, 0]).max()),
                revenue_scale=float(np.abs(rev_h).max()),
                d_objective_rel=abs(Z_i - Z_h) / max(abs(Z_h), 1e-30),
                mu=float(out["mu"]), dual_residual=float(out["dual_residual"]),
                stationarity_rel=ref["stationarity_rel"], **deg))
            print(f"  cap {cap} k {k:>3d}: took {int(took)} n_frac "
                  f"{deg['n_frac']:>2d} dual_gap {deg['dual_gap']:>10.4f} "
                  f"(theory {rows[-1]['dual_gap_theory']:.4f}) shed "
                  f"{deg['shed_mw']:>9.2f} MW | d_rev "
                  f"{rows[-1]['d_energy_revenue']:.3e} $ d_lmp "
                  f"{rows[-1]['d_lmp']:.3e} d_award {rows[-1]['d_award']:.3e} "
                  f"relZ {rows[-1]['d_objective_rel']:.3e}", flush=True)
            del lp
    if not rows:
        raise RuntimeError("no constructed row produced")
    took = np.array([r["construction_took"] for r in rows])
    print(f"\nconstruction took at {int(took.sum())}/{len(rows)} points "
          f"(front-asserted, not assumed)")
    if took.sum() == 0:
        raise RuntimeError("the construction took nowhere: every point still "
                           "has an interior segment, so this run contains no "
                           "degenerate point and answers nothing")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("census", "constructed"),
                    default="census")
    ap.add_argument("--constructed-points", type=int, default=12)
    ap.add_argument("--constructed-caps", type=float, nargs="+",
                    default=[0.6, 1.0])
    ap.add_argument("--fixture", default="tests/fixtures/"
                    "day_ahead_commitment_29gb_T24_relax_seasons.npz")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--periods", type=int, default=24)
    ap.add_argument("--cap-scale", type=float, default=0.6)
    ap.add_argument("--ramp-scale", type=float, default=1.0)
    ap.add_argument("--max-iter", type=int, default=MAX_ITER)
    ap.add_argument("--markup", type=float, default=1.0)
    #: The third scenario scale.  `make_clearing` is blind to it (it is applied
    #: to the case, not passed to the operators), so a `case73rts` fixture built
    #: at 0.80 would be cleared here at the registered 1.00 -- the same silent
    #: mismatch measured on `run_eval_01.py` 2026-09-13.  Defaults to the
    #: fixture's own value so a command line naming nothing runs the fixture.
    ap.add_argument("--p-min-scale", type=float, default=None)
    #: Which days, by index into the fixture window.  `--days N` takes the
    #: leading N, which cannot reach day 183 without solving 184 of them.
    ap.add_argument("--only-days", type=int, nargs="+", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    K, T = 1, args.periods

    fx = load_commitment(path=Path(args.fixture), n_periods=24,
                         p_min_scale=args.p_min_scale)
    meta = fx["meta"]
    got = (meta["cap_scale"], meta["ramp_scale"], meta["n_segments"])
    if got != (args.cap_scale, args.ramp_scale, K):
        raise ValueError(f"fixture is cap/ramp/K {got}, this run asks for "
                         f"{(args.cap_scale, args.ramp_scale, K)}")
    pms = (float(meta.get("p_min_scale", 1.0)) if args.p_min_scale is None
           else float(args.p_min_scale))
    print(f"p_min_scale in effect: {pms}", flush=True)
    case = scale_min_output(load_case(meta["case"]), pms)
    width, cost = segment_costs(case, K)
    forecast, actual, days = demand_from_meta(meta)
    day_index = np.asarray(fx["day_index"], np.int64)
    n_u, n_b = int(case.n_units), int(case.n_nodes)
    unit_bus = np.asarray(case.unit_node_idx, np.int64)

    import powermarketjax
    print(f"powermarketjax {powermarketjax.__file__}")
    if args.only_days is not None:
        sel = np.asarray(sorted(set(args.only_days)), np.int64)
        bad = [int(d) for d in sel if d >= len(day_index)]
        if bad:
            raise SystemExit(f"--only-days {bad} past the window's "
                             f"{len(day_index)} days")
    else:
        sel = np.arange(min(args.days, len(day_index)))
    drawn = len(sel) == len(day_index)
    print(f"case {meta['case']}  n_units {n_u}  n_buses {n_b}  T {T}  K {K}")
    if drawn:
        print(f"days: all {len(sel)} in the fixture -- a census.  No draw is "
              f"made, so no selection can follow a result")
    elif args.only_days is not None:
        # A new way of choosing days has to show in the output which days it took:
        # the old wording "the first N days" is wrong for a named set, and being
        # wrong would make nothing fail.
        print(f"days: the {len(sel)} named by --only-days, of "
              f"{len(day_index)}: {[int(d) for d in sel]}")
    else:
        print(f"days: the {len(sel)} leading days of {len(day_index)}, taken in "
              f"the fixture's own order")

    if args.mode == "constructed":
        rows = run_constructed(case, cost, width, args)
        _write(rows, args, meta, dict(cap_scale=None, ramp_scale=None,
                                      voll_module=float(VOLL),
                                      off_eps_module=float(OFF_EPS),
                                      max_iter_sent=int(args.max_iter),
                                      dual_start_sent="cost_norm (default)"),
               sel=[], drawn=False, n_failed=0, secs_h=None, secs_i=None)
        return

    clear, spec = make_clearing(case, T, n_segments=K, cap_scale=args.cap_scale,
                                ramp_scale=args.ramp_scale,
                                max_iter=args.max_iter)
    in_force = dict(cap_scale=float(spec["cap_scale"]),
                    ramp_scale=float(spec["ramp_scale"]),
                    voll_module=float(VOLL), off_eps_module=float(OFF_EPS),
                    max_iter_sent=int(args.max_iter),
                    dual_start_sent="cost_norm (make_clearing default)")
    print("in force:", in_force)
    if (in_force["cap_scale"], in_force["ramp_scale"]) != (args.cap_scale,
                                                           args.ramp_scale):
        raise RuntimeError(f"operator reports {in_force}")
    clear = jax.jit(clear)
    offer = np.broadcast_to(cost[:, :, None] * args.markup, (n_u, K, T)).copy()

    rows, n_failed, secs_h, secs_i = [], 0, [], []
    for d in sel:
        u = np.asarray(fx["commitment"][d], np.float64)[:, :T]
        p_init = np.asarray(fx["p_init"][d], np.float64)
        dem = np.asarray(actual[day_index[d]], np.float64)[:T]
        lp = reference.build_lp(case, offer, u, dem, p_init, args.cap_scale,
                                args.ramp_scale)
        t0 = time.time(); ref = highs(lp); secs_h.append(time.time() - t0)
        if ref is None:
            n_failed += 1
            del lp
            continue
        if ref["stationarity_rel"] > DUAL_RESIDUAL_TOL:
            raise RuntimeError(
                f"HiGHS duals did not close on day {d}: relative stationarity "
                f"residual {ref['stationarity_rel']:.3e} > {DUAL_RESIDUAL_TOL:g}; "
                f"the nodal price below would be reconstructed from multipliers "
                f"whose convention is not established, so this stops")
        if ref["d_lmp_two_ways"] > 1e-6:
            raise RuntimeError(f"the two nodal-price constructions disagree by "
                               f"{ref['d_lmp_two_ways']:.3e} $/MWh on day {d}")
        t0 = time.time()
        out = {k: np.asarray(v) for k, v in
               clear(jnp.asarray(offer), jnp.asarray(u), jnp.asarray(dem),
                     jnp.asarray(p_init)).items()}
        secs_i.append(time.time() - t0)

        rev_i = out["award"] * out["lmp"][:, unit_bus].T
        rev_h = ref["award"] * ref["lmp"][:, unit_bus].T
        # objective by one formula on both sides, from the consumed quantities
        Z_i = float((offer[:, 0, :] * out["award"]).sum()
                    + VOLL * out["shed"].sum())
        Z_h = float((offer[:, 0, :] * ref["award"]).sum()
                    + VOLL * ref["shed"].sum())
        for t in range(T):
            deg = degeneracy(lp, ref, offer, u, width, t)
            rows.append(dict(
                day=int(d), period=int(t),
                d_energy_revenue=float(np.abs(rev_i[:, t] - rev_h[:, t]).max()),
                d_lmp=float(np.abs(out["lmp"][t] - ref["lmp"][t]).max()),
                d_award=float(np.abs(out["award"][:, t] - ref["award"][:, t]).max()),
                revenue_scale=float(np.abs(rev_h[:, t]).max()),
                d_objective_rel=abs(Z_i - Z_h) / max(abs(Z_h), 1e-30),
                mu=float(out["mu"]), dual_residual=float(out["dual_residual"]),
                stationarity_rel=ref["stationarity_rel"], **deg))
        del lp
        if (int(d) + 1) % 10 == 0:
            print(f"  day {int(d) + 1}/{len(sel)} done ({len(rows)} rows)",
                  flush=True)

    _write(rows, args, meta, in_force, sel, drawn, n_failed,
           secs_h, secs_i)


def _write(rows, args, meta, in_force, sel, drawn, n_failed, secs_h, secs_i):
    """Save the product and print the verdict.  Both modes go through here,
    so the columns, the meta keys and the gate are the same object in both."""
    T, K = args.periods, 1
    if not rows:
        raise RuntimeError("no row produced; the sample or the gate is wrong")
    keys = list(rows[0])
    a = {k: np.array([r[k] for r in rows], dtype=float) for k in keys}
    summary = dict(
        case=meta["case"], T=T, K=K, days=[int(d) for d in sel],
        census=bool(drawn), fixture=args.fixture, markup_in_force=float(args.markup),
        **{k + "_in_force": v for k, v in in_force.items()},
        revenue_tol=REVENUE_TOL, dual_residual_tol=DUAL_RESIDUAL_TOL,
        frac_eps=FRAC_EPS, tie_rel=TIE_REL, binding_tol=BINDING_TOL,
        mu_tol=MU_TOL, n_highs_failed=n_failed,
        judged_on="per-(unit, period) energy revenue award*lmp[bus] in dollars "
                  "against HiGHS, a different algorithm.  mu, d_lmp, d_award "
                  "and the objective are recorded beside it and are not used to "
                  "decide whether a period disagrees",
        degeneracy_from="the HiGHS vertex solution only, never from the "
                        "operator under test",
        platform=str(jax.devices()),
        head=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip(),
        tree_dirty_tracked=len([l for l in subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=no"],
            capture_output=True, text=True).stdout.split("\n") if l]),
        apparatus_blob={p: _blob(p) for p in (
            "tools/lp_bench/highs_degeneracy_census.py",
            "powermarketjax/envs/day_ahead/clearing.py",
            "tests/envs/day_ahead/reference.py")},
        # None in constructed mode: that run is not a timing measurement and a
        # placeholder 0.0 would read as one
        seconds_highs_per_day=None if secs_h is None else float(np.mean(secs_h)),
        seconds_impl_per_day=(
            None if secs_i is None else
            float(np.mean(secs_i[1:]) if len(secs_i) > 1 else secs_i[0])),
    )
    out_arr = dict(a); out_arr["meta"] = np.array(json.dumps(summary, indent=1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out_arr)
    back = np.load(args.out, allow_pickle=True)
    for k in ("d_energy_revenue", "n_frac", "dual_gap", "n_tie_at_lambda"):
        if k not in back.files:
            raise RuntimeError(f"column {k} asked for but not in the product")
    print(f"\nwrote {args.out}, {len(rows)} rows, HiGHS failures {n_failed}")

    bad = a["d_energy_revenue"] > REVENUE_TOL
    breakpoint_ = a["n_frac"] == 0
    tied = a["n_tie_at_lambda"] > 1
    print(f"\n=== does 01 disagree with HiGHS only at degenerate points? ===")
    print(f"periods {len(rows)};  disagreeing (>{REVENUE_TOL:g} $) "
          f"{int(bad.sum())};  n_frac==0 (price an interval) "
          f"{int(breakpoint_.sum())};  tied at lambda {int(tied.sum())}")
    print(f"{'':>26s} {'n_frac==0':>10s} {'n_frac>0':>10s}")
    for lab, m in (("disagree", bad), ("agree", ~bad)):
        print(f"{lab:>26s} {int((m & breakpoint_).sum()):>10d} "
              f"{int((m & ~breakpoint_).sum()):>10d}")
    unexplained = bad & ~breakpoint_ & ~tied
    print(f"\ndisagreeing, not on a breakpoint, no tie at lambda: "
          f"{int(unexplained.sum())}  <- these are the ones degeneracy cannot "
          f"explain")
    if unexplained.any():
        i = int(np.argmax(np.where(unexplained, a["d_energy_revenue"], -1)))
        print(f"  worst such: day {int(a['day'][i])} period {int(a['period'][i])} "
              f"revenue {a['d_energy_revenue'][i]:.3e} $ n_frac "
              f"{int(a['n_frac'][i])} d_lmp {a['d_lmp'][i]:.3e} "
              f"rel dZ {a['d_objective_rel'][i]:.3e}")
    for lab, m in (("n_frac==0", breakpoint_), ("n_frac>0", ~breakpoint_)):
        if m.any():
            print(f"worst revenue disagreement on {lab:>9s}: "
                  f"{a['d_energy_revenue'][m].max():.3e} $  "
                  f"(worst d_lmp {a['d_lmp'][m].max():.3e} $/MWh, worst d_award "
                  f"{a['d_award'][m].max():.3e} MW)")
    fin = np.isfinite(a["dual_gap"])
    if fin.any():
        q = np.percentile(a["dual_gap"][fin], [0, 25, 50, 75, 100])
        print(f"dual_gap where defined ({int(fin.sum())} periods) $/MWh: "
              f"min {q[0]:.3e} q25 {q[1]:.3e} med {q[2]:.3e} q75 {q[3]:.3e} "
              f"max {q[4]:.3e}")
    print(f"worst mu {a['mu'].max():.3e} (MU_TOL {MU_TOL:g}), unconverged "
          f"{int((a['mu'] > MU_TOL).sum())}/{len(rows)}; worst stationarity "
          f"{a['stationarity_rel'].max():.3e}")
    if summary["seconds_highs_per_day"] is None:
        print("seconds/day: not measured in this mode")
    else:
        print(f"seconds/day: HiGHS {summary['seconds_highs_per_day']:.2f}, "
              f"implementation {summary['seconds_impl_per_day']:.2f}")


if __name__ == "__main__":
    main()
