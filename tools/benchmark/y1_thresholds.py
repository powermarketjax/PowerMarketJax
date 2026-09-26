"""Per-market thresholds from the seed spread, and each arm's displacement against them.

The rule is the 60-day report's section 5, unchanged: a threshold is the RANGE
across the seeds of one market divided by their MEAN, on the evaluation days,
computed separately for system cost and for per-agent profit and never combined
into one score.  Displacement is measured against the untrained control of the
same window, also unchanged.

**Only the numbers move between windows, not the method.**  This script exists so
that the year window's thresholds are derived by the same arithmetic rather than
by a second implementation of it, and so that the derivation can be re-run.

    python tools/benchmark/y1_thresholds.py <untrained-dir> <seed-dir> <seed-dir> ...

Each directory must hold `*_day*.npz` files, written by `evaluation.write_day`.
The drivers that call it: `tools/benchmark/run_eval_03.py`, `run_rl_01.py`,
`run_rl_02.py` and `run_rl_03.py`.

**Two additions, 2026-09-11, for the 60-day SAC cells.**  Both are extra output.
Neither changes the arithmetic above, and neither changes what the invocation
spelled above prints -- one baseline and no `--per-day` gives the same bytes it
gave before.

`--baseline DIR`, repeatable, names FURTHER untrained directories.  Those cells
have three untrained seeds apiece rather than one, and which of them is the
displacement denominator is a published convention, and changing it would move
published magnitudes.  Up to three denominators are printed: (a) the positional
baseline alone, (b) the mean over all the baselines, and (c) the paired one,
below.  Each quantity then ends with a line saying whether they agree on every
arm's pass/fail.  The baselines' own cross-seed spread is printed as well, by
the same section-5 arithmetic: the threshold is sized on the TRAINED side and
says nothing about how far the denominator itself moves when its seed changes.

**(c), the paired denominator, adopted 2026-09-12.**  Arm `sN` is
divided by the untrained run of the SAME seed, `sN`, rather than by one seed's
run or by the mean.  The sentence the displacement is asked to support is what
TRAINING bought, and only the paired form is true of it arm by arm; (a) is not a
single convention at all but a mixture -- seed 0 paired, the other seeds not --
which is what having only one untrained run produced.  **This is not a second
implementation of anything**: (a), (b) and (c) differ only in which number each
arm is divided by, and all three go through the one loop below.

(c) is printed only when there are as many baselines as arms, and it pairs them
BY POSITION, so the k-th `--baseline` must be the k-th arm's own seed.  A wrong
order is silent -- every percentage would print normally -- so when every
directory carries a `curve.jsonl` the seeds are read and a mismatch is refused;
when any is missing, the run says so and says the pairing went unchecked, rather
than reading two absent fields as agreement.

`--per-day` prints the per-item form of the same two quantities, because a total
can hide and even reverse what the days say.  No verdict is derived from
the per-day numbers; they are reported beside the totals, never instead of them.

    python tools/benchmark/y1_thresholds.py --per-day \
        x_untrained_s0 x_s0 x_s1 x_s2 \
        --baseline x_untrained_s1 --baseline x_untrained_s2

`--baseline` and `--per-day` go after the directories, not between them: the
positional list is greedy.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

# Index into a row of `day_rows`, which is `(date, system_cost, agent_profit_sum)`.
QTY = (("system cost", 1), ("per-agent profit", 2))


def day_rows(d):
    """`[(date, system_cost, agent_profit_sum)]` for one directory, in file order.

    File order is `sorted(glob(...))`, which is the order the year-window figure
    script and the per-day totals exporter accumulate in, so the sums below
    are bit-identical to theirs rather than merely close to them.
    """
    fs = sorted(glob.glob(str(Path(d) / "*_day*.npz")))
    if not fs:
        raise SystemExit(
            f"{d}: no day products; *_day*.npz is written by "
            "evaluation.write_day, called from tools/benchmark/run_eval_03.py, "
            "run_rl_01.py, run_rl_02.py and run_rl_03.py")
    #: One arm per directory.  The evaluation drivers write every arm they run
    #: into the same --out-dir (`honest_day*.npz` beside `constant_day*.npz`),
    #: and the glob above would then add the arms' days together; when every
    #: directory is mixed the same way the day-count and date checks below all
    #: pass and the totals come out silently wrong.  A single-prefix directory
    #: takes exactly the path it took before.
    prefixes = sorted({Path(f).name.rsplit("_day", 1)[0] for f in fs})
    if len(prefixes) > 1:
        raise SystemExit(
            f"{d}: day products of {len(prefixes)} arms ({', '.join(prefixes)}) "
            "in one directory; summing them would add the arms together. Give "
            "a directory that holds one arm's *_day*.npz only")
    rows = []
    for f in fs:
        b = np.load(f, allow_pickle=True)
        m = json.loads(str(b["meta"]))
        rows.append((m["date"], float(m["system_cost"]),
                     float(np.sum(np.asarray(b["agent_profit"], np.float64)))))
    return rows


def sums(rows):
    """`(days, sum system_cost, sum agent_profit)` over one directory's rows."""
    sc = pf = 0.0
    for _date, a, b in rows:
        sc += a
        pf += b
    return len(rows), sc, pf


def totals(d):
    """One directory's `(days, system cost, profit)`; the whole basis of the report."""
    return sums(day_rows(d))


def refuse_unless_aligned(pairs):
    """Refuse a partial or a differently-dated run rather than average over it.

    `pairs` is `[(dir, rows)]`, baselines first.

    **Day counts.**  A driver writes its day products one at a time, so a
    directory read while the evaluation is still going yields a short, silently
    wrong total -- and the percentages below would look entirely normal.
    Measured 2026-08-30: read at the wrong moment, three arms held 30 / 26 / 20
    of 36 days and produced displacements of -15.8% / -25.9% / -44.8%, none of
    which meant anything.  The day count is printed above and that is what caught
    it; this makes the catch automatic rather than dependent on the reader.

    **Dates.**  The same count can still be a different set of days: a 12-day
    60-day-window arm and a 12-day arm from another window's eval split both
    print `12 days`, and a displacement between them is a difference of weather
    and load, not of policy.  `meta['date']` is written by `evaluation.write_day`
    on every product this repository has (checked 2026-09-11 over the 18 SAC
    60-day directories and 5 further 01/02/03 directories, year window and 60-day
    both), so this is a real comparison and not a skipped one.
    """
    counts = {len(r) for _d, r in pairs}
    if len(counts) > 1:
        n0 = len(pairs[0][1])
        raise SystemExit(
            f"day counts disagree: baseline {n0}, arms "
            + ", ".join(f"{d}={len(r)}" for d, r in pairs[1:])
            + ". A short directory is an evaluation still being written, and a "
              "total over a subset of days is not comparable with one over all "
              "of them. Re-run when every arm has finished.")
    dates = {d: [r[0] for r in rows] for d, rows in pairs}
    ref_dir, ref = pairs[0][0], dates[pairs[0][0]]
    bad = {d: v for d, v in dates.items() if v != ref}
    if bad:
        raise SystemExit(
            f"evaluation dates disagree with {ref_dir} ({ref[0]}..{ref[-1]}): "
            + "; ".join(f"{d} {v[0]}..{v[-1]}" for d, v in bad.items())
            + ". The same day count over a different set of days is not the same "
              "comparison, and the percentages would look entirely normal.")


def seed_of(d):
    """The run's seed from `curve.jsonl`'s first line, or `None` if there is none.

    Written by the same drivers that write the day products, so this is the one
    field on disk that says which seed a directory IS rather than which seed its
    name claims.
    """
    p = Path(d) / "curve.jsonl"
    if not p.exists():
        return None
    with open(p) as f:
        return json.loads(f.readline()).get("seed")


def refuse_unless_seeds_pair(base_dirs, seed_dirs):
    """Refuse a (c) whose k-th baseline is not the k-th arm's own seed.

    Pairing is positional and a wrong order is silent: the displacements would
    print normally and be wrong by the difference between two initialisations.
    Measured on the L0 fixture: exchanging two of three baselines leaves (a) and
    (b) bit-identical -- (a) reads only the first, (b)'s mean is permutation
    invariant -- while one arm's cost displacement moves 1.9973 percentage
    points and its verdict flips.  So (c) is the only one of the three that can
    be wrong this way, and it is the adopted one.

    A missing `curve.jsonl` downgrades to a printed caveat and never to a silent
    pass: two absent fields must not read as two equal ones.
    """
    n = len(base_dirs)
    seeds = [seed_of(d) for d in base_dirs + seed_dirs]
    missing = [d for d, s in zip(base_dirs + seed_dirs, seeds) if s is None]
    if missing:
        print(f"(c) paired by position: seeds unchecked, {len(missing)} directories have no "
              f"curve.jsonl ({', '.join(Path(d).name for d in missing)})")
        return
    bad = [(base_dirs[k], seeds[k], seed_dirs[k], seeds[n + k])
           for k in range(n) if seeds[k] != seeds[n + k]]
    if bad:
        raise SystemExit(
            "paired denominator: the k-th baseline is not the k-th arm's seed: "
            + "; ".join(f"{b} seed {sb} against {t} seed {st}"
                        for b, sb, t, st in bad)
            + ". Pairing is positional, and a wrong order is silent: every "
              "percentage would print normally. Re-order --baseline so that it "
              "runs seed by seed with the arms.")
    print("(c) pairing checked: untrained seeds "
          + "/".join(str(s) for s in seeds[:n])
          + " match trained seeds " + "/".join(str(s) for s in seeds[n:])
          + " one to one (read from the first line of each curve.jsonl)")


def spread(values):
    """`(range, mean, 100*range/|mean|)` -- section 5's arithmetic, one quantity."""
    v = np.asarray(values, np.float64)
    rng, mean = v.max() - v.min(), v.mean()
    return rng, mean, 100.0 * rng / abs(mean)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("base_dir", help="the untrained control; with --baseline, the "
                                    "one the published denominator is taken from")
    p.add_argument("seed_dirs", nargs="+", help="one directory per trained seed")
    p.add_argument("--baseline", action="append", default=[], metavar="DIR",
                   help="a further untrained seed of the same cell (repeatable); "
                        "give as many as there are arms, seed by seed, and the "
                        "paired denominator (c) is printed too")
    p.add_argument("--per-day", action="store_true",
                   help="also print the per-item form of the threshold and the "
                        "displacements")
    a = p.parse_args()

    base_dirs = [a.base_dir] + a.baseline
    rows = {d: day_rows(d) for d in base_dirs + a.seed_dirs}
    tot = {d: sums(r) for d, r in rows.items()}

    for d in base_dirs:
        n, sc, pf = tot[d]
        print(f"untrained {d}  {n} days")
        print(f"  system_cost {sc:.10e}   profit {pf:.10e}")
    for d in a.seed_dirs:
        n, sc, pf = tot[d]
        print(f"{d}  {n} days  system_cost {sc:.10e}  profit {pf:.10e}")

    refuse_unless_aligned([(d, rows[d]) for d in base_dirs + a.seed_dirs])
    n_days = len(rows[base_dirs[0]])

    if len(base_dirs) > 1:
        print(f"\nuntrained spread across {len(base_dirs)} seeds, same days, "
              f"section 5's arithmetic applied to the denominator itself:")
        for name, i in QTY:
            rng, mean, s = spread([tot[d][i] for d in base_dirs])
            print(f"  {name}: range {rng:.6e} / |mean| {abs(mean):.6e} -> {s:.6f}%")

    if len(base_dirs) > 1 and len(base_dirs) == len(a.seed_dirs):
        refuse_unless_seeds_pair(base_dirs, a.seed_dirs)

    for name, i in QTY:
        rng, mean, thr = spread([tot[d][i] for d in a.seed_dirs])
        print(f"\n{name}: range {rng:.6e} / |mean| {abs(mean):.6e} "
              f"-> threshold {thr:.6f}%")

        day_of = {d: np.array([r[i] for r in rows[d]], np.float64) for d in rows}
        n_arms = len(a.seed_dirs)
        # One entry per denominator: (short name, label, one reference total per
        # arm, one reference per-day vector per arm).  (a) and (b) give every arm
        # the same reference, (c) gives each arm its own; that list is the ONLY
        # thing the three differ in, and the one loop below consumes all three.
        denoms = [("single", f"(a) one untrained seed: {base_dirs[0]}",
                   [tot[base_dirs[0]][i]] * n_arms,
                   [day_of[base_dirs[0]]] * n_arms)]
        if len(base_dirs) > 1:
            per_day_mean = np.mean([day_of[d] for d in base_dirs], axis=0)
            denoms.append(("mean", f"(b) mean of {len(base_dirs)} untrained seeds",
                           [float(np.mean([tot[d][i] for d in base_dirs]))] * n_arms,
                           [per_day_mean] * n_arms))
            if len(base_dirs) == n_arms:
                denoms.append(("paired",
                               f"(c) paired, arm sN over untrained sN "
                               f"({n_arms} seeds) -- adopted 2026-09-12",
                               [tot[d][i] for d in base_dirs],
                               [day_of[d] for d in base_dirs]))
        pad = "  " if len(denoms) == 1 else "    "

        sides = []
        for _short, label, refs, _vecs in denoms:
            one = len(set(refs)) == 1
            if len(denoms) > 1:
                print(f"  denominator {label}" + (f"   {refs[0]:.10e}" if one else ""))
            row = []
            for k, d in enumerate(a.seed_dirs):
                ref = refs[k]
                disp = 100.0 * (tot[d][i] - ref) / abs(ref)
                side = "pass" if abs(disp) > thr else "fail"
                row.append((disp, side))
                print(f"{pad}{d:28s} displacement {disp:+9.4f}%   {side}"
                      f"   (|disp|/thr = {abs(disp) / thr:.2f}x)"
                      + ("" if one else
                         f"   ÷ {Path(base_dirs[k]).name} {ref:.10e}"))
            sides.append(row)
        if len(sides) > 1:
            shorts = [x[0] for x in denoms]
            split = []
            for k, d in enumerate(a.seed_dirs):
                v = [row[k][1] for row in sides]
                if len(set(v)) > 1:
                    split.append(f"{d} (" + ", ".join(
                        f"{t} {x}" for t, x in zip(shorts, v)) + ")")
            print(f"  verdict across {len(denoms)} denominators: "
                  + ("all agree" if not split else "**flipped** " + "; ".join(split)))

        if a.per_day:
            arms = np.array([day_of[d] for d in a.seed_dirs], np.float64)
            pdt = 100.0 * (arms.max(0) - arms.min(0)) / np.abs(arms.mean(0))
            print(f"  per day ({n_days} days), cross-seed range/|mean|: "
                  f"median {np.median(pdt):.6f}%  min {pdt.min():.6f}%  "
                  f"max {pdt.max():.6f}%")
            for (_short, label, _refs, vecs), row in zip(denoms, sides):
                print(f"  per day, denominator {label}")
                for k, d in enumerate(a.seed_dirs):
                    dd = 100.0 * (arms[k] - vecs[k]) / np.abs(vecs[k])
                    same = int(np.sum(np.sign(dd) == np.sign(row[k][0])))
                    print(f"    {d:28s} median {np.median(dd):+9.4f}%  "
                          f"[{dd.min():+9.4f}%, {dd.max():+9.4f}%]  "
                          f"same sign as the total on {same}/{n_days} days")


if __name__ == "__main__":
    main()
