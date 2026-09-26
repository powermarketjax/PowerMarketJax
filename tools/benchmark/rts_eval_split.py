"""The evaluation-day split of `case73rts`'s window: calendar 2020, all 366 days.

**The rule is not restated here, it is imported.**  `y1_eval_split` derives GB's
split -- stratify by meteorological season and weekday/weekend, allocate the
evaluation quota over the eight strata by largest remainder, then take
systematic mid-interval positions inside each stratum -- and this script calls
`strata`, `allocate` and `systematic` from it rather than copying them.  Two
cases sampled by two transcriptions of one rule are two rules the moment either
is edited; importing makes "the same criterion was used on both cases" a fact
about the code instead of a claim in a note.

**The sampling criterion is exogenous to everything reported.**  It reads the
calendar and nothing else -- not demand, not price, not the curtailment floor,
not any run's output (a sampling criterion must be exogenous to the quantity
reported).  In
particular it does not read `floor_mw`, so the split is unchanged by the run
point the scenario adopts and does not have to be re-derived if the floor moves.

Two differences from the GB window, both forced by the calendar:

    366 days   2020 is a leap year and the window is the whole of it, so there
               is no start offset: window offset i is day i of the year.  GB's
               window starts at day 5 of a 648-day inventory.
    seasons    RTS-GMLC places its buses in Arizona, southern California and
               Nevada, so the northern-hemisphere season map `y1_eval_split`
               already uses applies unchanged.  Winter wraps the two ends of the
               window (January-February and December) exactly as GB's summer
               wraps its own; evaluation days need no contiguity.

The quota is 36 days, the same count GB holds out, which on 366 days is 9.84%
against GB's 9.86%.

    python tools/benchmark/rts_eval_split.py            # print the split
    python tools/benchmark/rts_eval_split.py --verify   # compare to the constant below
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from y1_eval_split import SEASONS, allocate, strata, systematic  # noqa: E402

#: Days held out for evaluation, as offsets into `load_rts_demand`'s 366-day
#: output.  Frozen here rather than recomputed at import time, for the reason
#: `y1_eval_split` freezes GB's: a split that moves when the rule is edited is
#: not a held-out set.  `--verify` recomputes and compares.
RTS_EVAL_OFFSETS = (
    7, 22, 24, 37, 54, 68, 74, 83, 98, 108, 112, 127,
    137, 142, 156, 170, 171, 183, 197, 200, 210, 223, 228, 237,
    251, 262, 266, 281, 291, 295, 310, 319, 327, 340, 343, 358,
)

#: Number of days held out, matching the GB window's count.
RTS_N_EVAL = 36


def window_dates():
    """The 366 dates of calendar 2020, in order, as `load_rts_demand` returns them."""
    from powermarketjax.envs.day_ahead.demand import load_rts_demand
    # floor_mw is deliberately not passed: the split reads the calendar only,
    # and load_rts_demand returns the same `days` index whatever the floor is.
    _f, _a, days = load_rts_demand()
    assert len(days) == 366, f"expected the whole of 2020, got {len(days)} days"
    return days


def build():
    """`(eval_offsets, table)` -- the split and the per-stratum bookkeeping."""
    dates = window_dates()
    st = strata(dates)
    keys = [(s, t) for s in SEASONS for t in ("weekday", "weekend")]
    sizes = {k: len(st[k]) for k in keys}
    quota = allocate(sizes, RTS_N_EVAL)
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
                    help="recompute and compare against RTS_EVAL_OFFSETS")
    args = ap.parse_args()

    ev, table = build()
    dates = window_dates()
    print(f"window: {dates[0].date()} ({dates[0].strftime('%a')}) .. "
          f"{dates[-1].date()} ({dates[-1].strftime('%a')})  {len(dates)} days")
    print(f"{'season':8} {'type':8} {'pool':>5} {'take':>5}  dates")
    for season, daytype, size, take, ds in table:
        print(f"{season:8} {daytype:8} {size:5d} {take:5d}  {' '.join(ds)}")
    n_we = sum(t[3] for t in table if t[1] == "weekend")
    pool_we = sum(t[2] for t in table if t[1] == "weekend")
    print(f"\neval {len(ev)} days; weekend {n_we}/{len(ev)} = "
          f"{100 * n_we / len(ev):.2f}%, window {pool_we}/{len(dates)} = "
          f"{100 * pool_we / len(dates):.2f}%")
    print("weekday counts in the eval set: " + " ".join(
        f"{d}={sum(1 for i in ev if dates[i].strftime('%a') == d)}"
        for d in ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")))
    print("month counts in the eval set:   " + " ".join(
        f"{m}={sum(1 for i in ev if dates[i].month == m)}" for m in range(1, 13)))
    print("\nRTS_EVAL_OFFSETS = " + repr(tuple(ev)))
    print("RTS_EVAL_DATES   = " + repr(tuple(str(dates[i].date()) for i in ev)))

    if args.verify:
        # offsets against this module's constant, and **dates against
        # `evaluation`'s**: that is the copy the drivers consume, and a copy
        # nothing compares is a copy that drifts.  Same shape as
        # `y1_eval_split.py --verify`, which checks GB's two the same way.
        from evaluation import RTS_EVAL_DATES, RTS_EVAL_OFFSETS as EV_OFFSETS
        got_dates = tuple(str(dates[i].date()) for i in ev)
        ok_o = tuple(ev) == tuple(RTS_EVAL_OFFSETS)
        ok_e = tuple(RTS_EVAL_OFFSETS) == tuple(EV_OFFSETS)
        ok_d = got_dates == tuple(RTS_EVAL_DATES)
        print(f"\nverify offsets {'PASS' if ok_o else 'FAIL'}   "
              f"evaluation offsets {'PASS' if ok_e else 'FAIL'}   "
              f"evaluation dates {'PASS' if ok_d else 'FAIL'}")
        raise SystemExit(0 if (ok_o and ok_e and ok_d) else 1)


if __name__ == "__main__":
    main()
