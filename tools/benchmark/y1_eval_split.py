"""The evaluation-day split of the one-year window (Y1), derived from the calendar alone.

The 60-day window's rule fits in three lines (`evaluation.split_days`: segment
`s` contributes its `s`-th, `s+5`-th and `s+10`-th day) because four equal
15-day segments make every stratum the same size.  A 365-day window does not
have that property, so the rule here is a stratified systematic sample and its
output is frozen as `evaluation.YEAR_EVAL_OFFSETS` rather than recomputed at
import time.  This script is what derives that constant, and `--verify`
recomputes it and compares, so the constant can be audited rather than trusted.

**The sampling criterion is exogenous to everything reported.**  It reads the
calendar and nothing else -- not demand, not price, not any run's output.  That
is the whole point: a sampling criterion must be exogenous to the quantity
reported.

Four strata dimensions, eight strata:

    season      meteorological, on the window's own dates.  Summer wraps the
                two ends of the window (2023-07-10..08-31 and 2024-06-01..07-08)
                because they are the same season of the annual cycle; the two
                pieces are 53 + 38 = 91 days, and autumn / winter / spring are
                91 / 91 / 92, so the four seasons partition the 365 days almost
                equally.  Evaluation days need no contiguity, so a wrapped
                season costs nothing.
    day type    weekday (Mon-Fri) against weekend (Sat-Sun)

Allocation: 36 days split across the eight strata in proportion to stratum size
by largest remainder, ties broken by season order (summer, autumn, winter,
spring) and then weekday before weekend.  Selection inside a stratum of L days
taking n: the ordered positions ``floor((i + 0.5) * L / n)``, i.e. systematic
sampling at mid-interval offsets, which spreads the picks over the season
instead of clustering them at either end.

    python tools/benchmark/y1_eval_split.py            # print the split
    python tools/benchmark/y1_eval_split.py --verify   # compare to the constant
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: The window: day indices into `load_gb_demand()`'s 648-day inventory.
#: 5 is 2023-07-10 (Mon), the first day of the 60-day window's first segment,
#: so the year window is a strict superset of the 60-day one.
Y1_START_DAY = 5
Y1_N_DAYS = 365
#: Days taken for evaluation, held out of training.
Y1_N_EVAL = 36

SEASONS = ("summer", "autumn", "winter", "spring")
#: Meteorological seasons by (month, day) ranges, summer last so it collects
#: both ends of the window.
_SEASON_OF_MONTH = {12: "winter", 1: "winter", 2: "winter",
                    3: "spring", 4: "spring", 5: "spring",
                    6: "summer", 7: "summer", 8: "summer",
                    9: "autumn", 10: "autumn", 11: "autumn"}


def window_dates():
    """The 365 `datetime64[D]`-like dates of the window, in order."""
    from powermarketjax.envs.day_ahead.demand import load_gb_demand
    _f, _a, days = load_gb_demand()
    idx = np.arange(Y1_START_DAY, Y1_START_DAY + Y1_N_DAYS)
    assert idx[-1] < len(days), f"only {len(days)} days available"
    return days[idx]


def strata(dates):
    """`{(season, daytype): [offsets]}` -- window offsets, ordered, per stratum."""
    out = {(s, t): [] for s in SEASONS for t in ("weekday", "weekend")}
    for i, d in enumerate(dates):
        season = _SEASON_OF_MONTH[d.month]
        daytype = "weekend" if d.weekday() >= 5 else "weekday"
        out[(season, daytype)].append(i)
    return out


def allocate(sizes, total):
    """Largest-remainder allocation of `total` over `sizes`, in key order.

    `sizes` is an ordered mapping; ties in the remainder go to the earlier key,
    which is why the caller's key order is part of the rule rather than an
    implementation detail.
    """
    keys = list(sizes)
    n = sum(sizes.values())
    exact = {k: total * sizes[k] / n for k in keys}
    base = {k: int(np.floor(exact[k])) for k in keys}
    left = total - sum(base.values())
    order = sorted(keys, key=lambda k: (-(exact[k] - base[k]), keys.index(k)))
    for k in order[:left]:
        base[k] += 1
    return base


def systematic(pool, n):
    """`n` entries of the ordered list `pool`, at mid-interval offsets."""
    L = len(pool)
    return [pool[int(np.floor((i + 0.5) * L / n))] for i in range(n)]


def build():
    """`(eval_offsets, table)` -- the split and the per-stratum bookkeeping."""
    dates = window_dates()
    st = strata(dates)
    keys = [(s, t) for s in SEASONS for t in ("weekday", "weekend")]
    sizes = {k: len(st[k]) for k in keys}
    quota = allocate(sizes, Y1_N_EVAL)
    picks, table = [], []
    for k in keys:
        chosen = systematic(st[k], quota[k]) if quota[k] else []
        picks += chosen
        table.append((k[0], k[1], sizes[k], quota[k],
                      [str(dates[i].date()) for i in chosen]))
    return sorted(picks), table


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verify", action="store_true",
                    help="recompute and compare against evaluation.YEAR_EVAL_OFFSETS")
    args = ap.parse_args()

    ev, table = build()
    dates = window_dates()
    print(f"window: day {Y1_START_DAY}..{Y1_START_DAY + Y1_N_DAYS - 1}  "
          f"{dates[0].date()} ({dates[0].strftime('%a')}) .. "
          f"{dates[-1].date()} ({dates[-1].strftime('%a')})  {Y1_N_DAYS} days")
    print(f"{'season':8} {'type':8} {'pool':>5} {'take':>5}  dates")
    for season, daytype, size, take, ds in table:
        print(f"{season:8} {daytype:8} {size:5d} {take:5d}  {' '.join(ds)}")
    n_we = sum(t[3] for t in table if t[1] == "weekend")
    pool_we = sum(t[2] for t in table if t[1] == "weekend")
    print(f"\neval {len(ev)} days; weekend {n_we}/{len(ev)} = "
          f"{100 * n_we / len(ev):.2f}%, window {pool_we}/{Y1_N_DAYS} = "
          f"{100 * pool_we / Y1_N_DAYS:.2f}%")
    print("weekday counts in the eval set: " + " ".join(
        f"{d}={sum(1 for i in ev if dates[i].strftime('%a') == d)}"
        for d in ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")))
    print("\noffsets = " + repr(tuple(ev)))
    print("dates   = " + repr(tuple(str(dates[i].date()) for i in ev)))

    if args.verify:
        from evaluation import YEAR_EVAL_DATES, YEAR_EVAL_OFFSETS
        ok_o = tuple(ev) == tuple(YEAR_EVAL_OFFSETS)
        ok_d = tuple(str(dates[i].date()) for i in ev) == tuple(YEAR_EVAL_DATES)
        print(f"\nverify offsets {'PASS' if ok_o else 'FAIL'}   "
              f"dates {'PASS' if ok_d else 'FAIL'}")
        raise SystemExit(0 if (ok_o and ok_d) else 1)


if __name__ == "__main__":
    main()
