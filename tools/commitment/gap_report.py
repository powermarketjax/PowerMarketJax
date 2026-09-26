"""The optimality gap of the three-step clearing, read off a `milp_reference` product.

The product stores raw materials -- per-day objectives, dual bounds, statuses,
commitments -- and no percentages, so every figure quoted in a note is derived
here and can be re-derived by running this.  That is the point: the note is a
transcription of this output, not a second source for it.

Three things this refuses to average over.

**Days that stopped at the time limit are not points, they are intervals.**  A day
whose branch-and-bound ran out returns an incumbent (an upper bound on the true
optimum) and HiGHS's dual bound (a lower bound).  The gap computed against the
incumbent therefore *understates* the true gap, and the gap computed against the
dual bound overstates it.  Such a day is reported as the pair, never as the first
number alone, and it is listed by name rather than folded into a quantile.

**The twelve evaluation days are listed separately** (`tools/benchmark/evaluation
.split_days`, the rule of report §2.2), because every learning result in this
market is measured on those twelve and on no others.  A distribution over sixty
days answers a different question from the one a reader of those results is
asking.

**Quantiles come before `min`/`max`**, and the extremes are asked whether they are
scattered or concentrated.  A single day at the top of this distribution has
repeatedly been the whole of a "maximum" in this repository.

Usage:

    python tools/commitment/gap_report.py \\
        --product tests/fixtures/day_ahead_milp_reference_29gb_T24_c0.6_r1.00_\\
d5-103-187-285x15_mumd-drop.npz \\
        --against milp_reference_29gb_T24_c0.6_r1.00_d5.npz ...

`--against` is optional and takes any number of products covering the same days
at the same scenario: for each it checks that the step 1' chain came out bitwise
identical (which is what says the two runs share a boundary and differ only in the
program the exact side solved) and that the exact objective moved in the only
direction dropping rows can move it.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "benchmark"))

#: The chain columns: bitwise-identical across two runs iff they share a chain.
CHAIN_KEYS = ("obj_lp", "obj_three", "shed_three", "n_on_three", "phys_three",
              "obj_lp_fixture_chain", "obj_three_fixture_chain")


def meta(z):
    return json.loads(str(z["meta"]))


def quantiles(v):
    """`(q10, q25, median, q75, q90)` in percent, quantiles before extremes."""
    return np.percentile(100 * v, [10, 25, 50, 75, 90])


def describe(name, v, dates):
    q10, q1, med, q3, q90 = quantiles(v)
    lo, hi = 100 * v.min(), 100 * v.max()
    i_lo, i_hi = int(np.argmin(v)), int(np.argmax(v))
    print(f"  {name:<10} q10 {q10:+.2f}  q25 {q1:+.2f}  median {med:+.2f}  "
          f"q75 {q3:+.2f}  q90 {q90:+.2f}   then min {lo:+.2f} ({dates[i_lo]})  "
          f"max {hi:+.2f} ({dates[i_hi]})")
    # scattered or concentrated: is the worst day's excess over the runner-up
    # bigger than the whole spread of the rest of the top six
    top = np.sort(100 * v)[-6:]
    shape = "concentrated" if top[-1] - top[-2] > top[-2] - top[0] else "scattered"
    listed = np.array2string(top, precision=2, floatmode="fixed")
    print(f"  {'':<10} top 6 days {listed}, {shape}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--product", required=True)
    ap.add_argument("--against", nargs="*", default=[])
    ap.add_argument("--table", action="store_true",
                    help="print the full per-day markdown table as well")
    args = ap.parse_args()

    z = np.load(args.product, allow_pickle=True)
    m = meta(z)
    day = np.asarray(z["day"])
    dates = list(m["dates"])
    n = len(day)

    print(f"# {Path(args.product).name}\n")
    print(f"case {m['case']}  T={m['n_periods']}  K={m['n_segments']}  "
          f"cap_scale {m['cap_scale']}  ramp_scale {m['ramp_scale']}  "
          f"markup {m['markup']}  VOLL {m['voll']}")
    print(f"window: {n} days, segment starts {m.get('segment_starts')}, "
          f"{m.get('days_per_segment')} per segment, chained "
          f"{m.get('chained')!r}; {dates[0]} .. {dates[-1]}")
    print(f"exact side: exact_mumd={m.get('exact_mumd')!r} -- {m['exact']}")
    print(f"solver: mip_rel_gap {m['mip_rel_gap']}, time_limit {m['time_limit']} s, "
          f"{m['workers']} workers, scipy {m['scipy']}")
    print(f"run: commit {m.get('commit', 'n/a')} dirty {m.get('commit_dirty')} "
          f"date {m.get('date')}; wall {m['seconds_wall'] / 60:.1f} min, "
          f"cpu {m['seconds_cpu'] / 3600:.2f} h")

    # ---- cross-checks against another product over the same days -------------
    for other in args.against:
        zo = np.load(other, allow_pickle=True)
        mo, dayo = meta(zo), np.asarray(zo["day"])
        sel = np.isin(day, dayo)
        selo = np.isin(dayo, day)
        assert sel.sum() == selo.sum() and sel.sum() > 0, "no days in common"
        print(f"\n## against {Path(other).name} "
              f"(exact_mumd={mo.get('exact_mumd', 'keep (pre-flag)')!r}, "
              f"{int(sel.sum())} days in common)")
        same = all(np.array_equal(np.asarray(z[k])[sel], np.asarray(zo[k])[selo])
                   for k in CHAIN_KEYS if k in z and k in zo)
        worst = max(float(np.max(np.abs(np.asarray(z[k])[sel]
                                        - np.asarray(zo[k])[selo])))
                    for k in CHAIN_KEYS if k in z and k in zo)
        print(f"  step 1' chain bitwise identical on {CHAIN_KEYS}: {same} "
              f"(max abs difference {worst:.3e})")
        a, b = np.asarray(z["obj_milp"])[sel], np.asarray(zo["obj_milp"])[selo]
        rel = a / b - 1
        tol = 2 * float(m["mip_rel_gap"])
        n_bad = int((rel > tol).sum())
        print(f"  exact objective, this product against that one: "
              f"median {100 * np.median(rel):+.3f}%  min {100 * rel.min():+.3f}%  "
              f"max {100 * rel.max():+.3f}%; strictly cheaper on "
              f"{int((rel < -tol).sum())}/{len(rel)} days, above tolerance on "
              f"{n_bad} (must be 0 when this product dropped rows the other kept)")

    # ---- the gaps -------------------------------------------------------------
    obj3, objm = np.asarray(z["obj_three"]), np.asarray(z["obj_milp"])
    dual = np.asarray(z["milp_dual_bound"])
    status = np.asarray(z["milp_status"])
    mgap = np.asarray(z["milp_gap"])
    total = obj3 / objm - 1
    total_ub = obj3 / dual - 1                 # against the dual bound
    phys = np.asarray(z["phys_three"]) / np.asarray(z["phys_milp"]) - 1
    voll_pp = (np.asarray(z["voll_three"]) - np.asarray(z["voll_milp"])) / objm

    print(f"\n## the gap over all {n} days (quantiles first, then extremes)")
    describe("total", total, dates)
    describe("physical", phys, dates)
    describe("voll pp", voll_pp, dates)

    proved = status == 0
    print(f"\n## proved optimal: {int(proved.sum())}/{n}; "
          f"stopped at the {m['time_limit']:.0f} s limit: {int((~proved).sum())}")
    if (~proved).any():
        print("  On these days the exact side is an incumbent, so the gap below is "
              "a LOWER bound on the true gap (the incumbent is above the optimum, "
              "which makes the ratio too small); the bracket's other end uses "
              "HiGHS's dual bound.")
        print(f"  | day | date | milp_gap | gap vs incumbent | gap vs dual bound |")
        print(f"  |---|---|---|---|---|")
        for i in np.flatnonzero(~proved):
            print(f"  | {day[i]} | {dates[i]} | {mgap[i]:.2e} | "
                  f"{100 * total[i]:+.2f}% | {100 * total_ub[i]:+.2f}% |")
    print(f"  the same bracket over all {n} days: median "
          f"{100 * np.median(total):+.2f}% to {100 * np.median(total_ub):+.2f}%")

    # ---- the twelve evaluation days ------------------------------------------
    from evaluation import REPORT_EVAL_DATES, split_days                # noqa: E402
    try:
        ev, _tr = split_days(n, m["case"])
    except ValueError as exc:
        # `split_days` refuses a window its rule was not written for, which is the
        # right refusal: selecting days by a rule for another window picks the
        # wrong ones silently.  A partial product is still worth the rest of this
        # report, so say why the section is missing rather than dying here.
        print(f"\n## the evaluation days: not selectable on this product\n  {exc}")
        return
    ev_dates = [dates[i] for i in ev]
    print(f"\n## the {len(ev)} evaluation days (report §2.2)")
    print(f"  dates reproduce REPORT_EVAL_DATES: "
          f"{tuple(ev_dates) == tuple(REPORT_EVAL_DATES)}")
    e = np.array(ev)
    describe("total", total[e], ev_dates)
    describe("physical", phys[e], ev_dates)
    print(f"  proved optimal {int(proved[e].sum())}/{len(e)}; at the limit: "
          f"{[dates[i] for i in e[~proved[e]]]}")
    print(f"  | day | date | three-step | exact | gap | milp_gap | status |")
    print(f"  |---|---|---|---|---|---|---|")
    for i in e:
        print(f"  | {day[i]} | {dates[i]} | {obj3[i]:.6e} | {objm[i]:.6e} | "
              f"{100 * total[i]:+.2f}% | {mgap[i]:.2e} | "
              f"{'optimal' if proved[i] else 'TIME LIMIT'} |")

    # ---- commitments ---------------------------------------------------------
    n3, nm = np.asarray(z["n_on_three"]), np.asarray(z["n_on_milp"])
    ham = np.asarray(z["hamming"])
    more = int((n3 > nm).sum())
    print(f"\n## commitment")
    print(f"  three-step commits more unit-periods than exact on {more}/{n} days; "
          f"difference median {np.median(n3 - nm):.0f}, "
          f"range {int((n3 - nm).min())} to {int((n3 - nm).max())}")
    print(f"  hamming q25 {np.percentile(ham, 25):.0f} median {np.median(ham):.0f} "
          f"q75 {np.percentile(ham, 75):.0f}, then min {ham.min():.0f} "
          f"max {ham.max():.0f}, out of {z['commitment_milp'][0].size} cells")
    print(f"  days shedding: three-step "
          f"{int((np.asarray(z['shed_three']) > 1e-6).sum())}, exact "
          f"{int((np.asarray(z['shed_milp']) > 1e-6).sum())}")

    if args.table:
        print(f"\n## per-day\n")
        print("| day | date | three-step | exact | gap | physical diff | `milp_gap` | status | "
              "committed three-step/exact | Hamming |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for i in range(n):
            print(f"| {day[i]} | {dates[i]} | {obj3[i]:.6e} | {objm[i]:.6e} | "
                  f"{100 * total[i]:+.2f}% | {100 * phys[i]:+.2f}% | "
                  f"{mgap[i]:.2e} | {'optimal' if proved[i] else '**time limit**'} | "
                  f"{int(n3[i])}/{int(nm[i])} | {int(ham[i])} |")


if __name__ == "__main__":
    main()
