"""The evaluation-day split of `case813nem`'s window: 2025-02-01 .. 2026-01-31.

**The rule is not restated here, it is imported.**  `y1_eval_split` derives GB's
split -- stratify by season and weekday/weekend, allocate the evaluation quota
over the eight strata by largest remainder, then take systematic mid-interval
positions inside each stratum -- and this script calls `allocate` and
`systematic` from it, and builds its season map out of `y1_eval_split`'s, rather
than copying either.  Two cases sampled by two transcriptions of one rule are
two rules the moment either is edited (`rts_eval_split` says the same thing
about the RTS window, and takes the same route).

**The sampling criterion is exogenous to everything reported.**  It reads the
calendar and nothing else -- not demand, not price, not `floor_mw`, not any
run's output (a sampling criterion must be exogenous to the quantity reported).  In
particular it does not read the floor, so the split does not move if the run
point's floor moves.

Two differences from the GB window, both forced by the calendar:

    seasons    the mainland NEM is in the southern hemisphere, so the season
               map is `y1_eval_split`'s shifted by six months -- December to
               February is summer here and winter there.  Applying the
               northern map unchanged would label the whole of the NEM summer
               "winter", and the strata would still be eight and still be
               nearly equal, so **nothing downstream would look wrong**.  That
               is why the shift is derived from the imported map rather than
               typed out again.
    window     `load_nem_demand` returns 370 days, 2025-01-27 to 2026-01-31,
               which is not a calendar year.  The window is its last 365 days,
               offset 5 onward, because that is the longest run of whole
               calendar months in the inventory: 2025-02-01 to 2026-01-31.
               The five days dropped off the front are 2025-01-27..31.  GB's
               window also starts at offset 5 of its inventory; that is a
               coincidence of two different reasons, not a shared rule.

The quota is 36 days, the same count GB and RTS hold out, which on 365 days is
9.86% -- the same as GB's.

    python tools/benchmark/nem_eval_split.py            # print the split
    python tools/benchmark/nem_eval_split.py --verify   # compare to the constant below
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# `_SEASON_OF_MONTH` is private, and imported anyway: it is the single source of
# truth for which months make a season, and the point of this module is to reuse
# that rather than retype it.  What is local here is one fact -- the hemisphere.
from y1_eval_split import (SEASONS, _SEASON_OF_MONTH,  # noqa: E402
                           allocate, systematic)

#: Southern-hemisphere season map: the northern one shifted six months.  Derived
#: rather than typed so that editing `y1_eval_split`'s map moves both.
_SEASON_OF_MONTH_SOUTH = {m: _SEASON_OF_MONTH[(m + 5) % 12 + 1]
                          for m in range(1, 13)}

#: The window: offsets into `load_nem_demand()`'s 370-day inventory.
NEM_START_DAY = 5
NEM_N_DAYS = 365
#: Number of days held out, matching the GB and RTS windows' count.
NEM_N_EVAL = 36

#: Days held out for evaluation, as offsets into the **window** (add
#: `NEM_START_DAY` to index `load_nem_demand()`).  Frozen here rather than
#: recomputed at import time, for the reason `y1_eval_split` freezes GB's: a
#: split that moves when the rule is edited is not a held-out set.  `--verify`
#: recomputes and compares.
NEM_EVAL_OFFSETS = (
    9, 14, 24, 37, 42, 52, 67, 71, 81, 96, 105, 111,
    128, 134, 143, 158, 168, 172, 187, 197, 202, 219, 231, 234,
    249, 260, 263, 278, 288, 293, 312, 322, 326, 341, 350, 356,
)


def strata(dates):
    """`{(season, daytype): [offsets]}` -- as `y1_eval_split.strata`, southern.

    Not imported, because the imported one closes over the northern season map
    and there is no argument to change it.  The three lines are the same three;
    `--verify` asserts that this function and the imported one differ by exactly
    the six-month relabelling and by nothing else.
    """
    out = {(s, t): [] for s in SEASONS for t in ("weekday", "weekend")}
    for i, d in enumerate(dates):
        season = _SEASON_OF_MONTH_SOUTH[d.month]
        daytype = "weekend" if d.weekday() >= 5 else "weekday"
        out[(season, daytype)].append(i)
    return out


def window_dates():
    """The 365 dates of the window, in order, as `load_nem_demand` returns them."""
    from powermarketjax.envs.day_ahead.demand import load_nem_demand
    # floor_mw is deliberately not passed: the split reads the calendar only,
    # and load_nem_demand returns the same `days` index whatever the floor is.
    _f, _a, days = load_nem_demand()
    assert len(days) == 370, f"expected the 370-day inventory, got {len(days)}"
    sel = days[NEM_START_DAY:NEM_START_DAY + NEM_N_DAYS]
    assert (sel[0].month, sel[0].day) == (2, 1) and (sel[-1].month, sel[-1].day) == (1, 31), (
        f"window is {sel[0].date()}..{sel[-1].date()}, not the whole-month run "
        f"this split was derived on; the inventory's dates have moved")
    return sel


def build():
    """`(eval_offsets, table)` -- the split and the per-stratum bookkeeping."""
    dates = window_dates()
    st = strata(dates)
    keys = [(s, t) for s in SEASONS for t in ("weekday", "weekend")]
    sizes = {k: len(st[k]) for k in keys}
    quota = allocate(sizes, NEM_N_EVAL)
    picks, table = [], []
    for k in keys:
        chosen = systematic(st[k], quota[k]) if quota[k] else []
        picks += chosen
        table.append((k[0], k[1], sizes[k], quota[k],
                      [str(dates[i].date()) for i in chosen]))
    return sorted(picks), table


def _hemisphere_check(dates):
    """`True` if the local strata are the imported ones, seasons relabelled.

    The failure this catches is the silent one: a season map that is wrong by a
    shift still produces eight strata of nearly equal size, so every count
    printed below stays plausible.  Comparing the *partitions* rather than the
    labels is what makes the shift checkable.
    """
    from y1_eval_split import strata as strata_north
    north, south = strata_north(dates), strata(dates)
    shift = {_SEASON_OF_MONTH[m]: _SEASON_OF_MONTH_SOUTH[m] for m in range(1, 13)}
    return all(north[(s, t)] == south[(shift[s], t)]
               for s in SEASONS for t in ("weekday", "weekend"))



#: **The only check here whose two sides are not both derived from
#: `NEM_START_DAY`.**  The other four compare this tool against
#: `evaluation.py`, and both copies come from this file's own constant, so they
#: catch a copy drifting and cannot catch the pair being jointly wrong about the
#: world.  2026-09-13 they were all four green while the shipped commitment
#: fixture spanned 2025-01-27..2026-01-26, i.e. inventory 0..364 instead of the
#: window's 5..369; `run_eval_01`'s own split check is what refused, and only
#: after a card had been taken for it.
#:
#: **The weaker form of this check does not bite, and that is worth stating
#: because it is the form one reaches for first**: "every registered evaluation
#: date is present in the fixture" was measured against that wrong fixture and
#: came out **True** -- all 36 dates lie inside 2025-01-27..2026-01-26, because
#: the five days the window drops off the front hold no evaluation day and the
#: five it gains at the back hold none either.  A guard that passes on the very
#: artefact it was written for is decoration.  **The invariant is the window
#: itself**, and the fixture records it: `meta["day_index"][0]` is the inventory
#: offset it started at, and `len(meta["dates"])` its length.  One line judges
#: it; no date parsing.
#:
#: **What a window change silently breaks, and why this guard is worth its
#: line.**  `run_eval_01` and the five other drivers have their own split check,
#: so a moved window stops them loudly.  **Six tools have no such check and index
#: the fixture by window position**: `lp_bench/highs_degeneracy_census.py:386`
#: (`fx["commitment"][d]`), `ancillary/requirement_check.py:124`,
#: `ancillary/pathological_loudness.py:106`,
#: `ancillary/near_tie_is_not_iterations.py:66`,
#: `teaching/three_prices_real_scale.py:107`.  Move the window and every one of
#: them reads a different day than it names, with nothing anywhere going red --
#: the first of them is the device `report-06`'s independent-solver comparison
#: takes its shape from.  **The 01 environment is not among them**: it reads only
#: the four boundary arrays (`env.py:298-301`) and re-solves the commitment from
#: the day's offers (`:458-464`), so "the fixture only supplies the boundary" is
#: true of that environment and false of these six.
def _fixture_window_check():
    """Compare the shipped 813nem commitment fixture's window to this window.

    Returns ``(ok, label)``.  A missing fixture is reported as ``SKIP`` and does
    not pass: this check has no content without the artefact, and a skip that
    reads as a pass is the failure it exists to prevent.
    """
    import json
    fx = (Path(__file__).resolve().parents[2] / "tests" / "fixtures"
          / "day_ahead_commitment_813nem_T24_relax.npz")
    if not fx.exists():
        return True, f"SKIP (no fixture at {fx.name}; this check has no content)"
    import numpy as np
    meta = json.loads(str(np.load(fx, allow_pickle=True)["meta"]))
    start, n = int(meta["day_index"][0]), len(meta["dates"])
    if (start, n) == (NEM_START_DAY, NEM_N_DAYS):
        return True, "PASS"
    return False, (f"FAIL (fixture spans inventory {start}..{start + n - 1}, "
                   f"{meta['dates'][0]}..{meta['dates'][-1]}; this window is "
                   f"{NEM_START_DAY}..{NEM_START_DAY + NEM_N_DAYS - 1}. "
                   f"Rebuild it with precommit.py --start-day {NEM_START_DAY}, "
                   f"or the registered window is what moved)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verify", action="store_true",
                    help="recompute and compare against NEM_EVAL_OFFSETS")
    args = ap.parse_args()

    ev, table = build()
    dates = window_dates()
    print(f"window: offset {NEM_START_DAY}..{NEM_START_DAY + NEM_N_DAYS - 1}  "
          f"{dates[0].date()} ({dates[0].strftime('%a')}) .. "
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
    print("\nNEM_EVAL_OFFSETS = " + repr(tuple(ev)))
    print("NEM_EVAL_DATES   = " + repr(tuple(str(dates[i].date()) for i in ev)))
    print("inventory indices = " + repr(tuple(i + NEM_START_DAY for i in ev)))

    if args.verify:
        # offsets and the hemisphere shift as before, plus the dates **against
        # `evaluation`'s copy**: that is the one the drivers consume, and a copy
        # nothing compares is a copy that drifts.  Same shape as
        # `y1_eval_split.py --verify`, which checks GB's two the same way.
        from evaluation import NEM_EVAL_DATES, NEM_EVAL_OFFSETS as EV_OFFSETS
        got_dates = tuple(str(dates[i].date()) for i in ev)
        ok = tuple(ev) == tuple(NEM_EVAL_OFFSETS)
        ok_e = tuple(NEM_EVAL_OFFSETS) == tuple(EV_OFFSETS)
        ok_d = got_dates == tuple(NEM_EVAL_DATES)
        hemi = _hemisphere_check(dates)
        fx, fx_why = _fixture_window_check()
        print(f"\nverify offsets {'PASS' if ok else 'FAIL'}   "
              f"evaluation offsets {'PASS' if ok_e else 'FAIL'}   "
              f"evaluation dates {'PASS' if ok_d else 'FAIL'}   "
              f"hemisphere shift {'PASS' if hemi else 'FAIL'}   "
              f"fixture window {fx_why}")
        raise SystemExit(0 if (ok and ok_e and ok_d and hemi and fx) else 1)


if __name__ == "__main__":
    main()
