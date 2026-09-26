"""L0: `tools/benchmark/y1_thresholds.py` -- section 5's arithmetic, its two
refusals, and the per-item form the totals cannot witness.

The device is the one place the threshold rule of the five-market baseline
matrix is implemented, and five documents cite it as the definition rather than restating it, so what is gated here is its
OUTPUT: every assertion below reads the text `main()` prints, not a recomputation
of the same formula beside it.  A test that recomputed the arithmetic would pass
against a device that printed it into the wrong column.

**Nothing here is a pinned measurement.**  Every fixture value is a small
integer, so it is exact in float64 and every sum, range and mean in this file is
exact; the constants transcribed into the assertions are ratios of those
integers worked out by hand (a tolerance or a constant needs a derivation,
not a previous run of the thing under test).

**The bite.**  `test_a_total_preserving_reattribution...` swaps
two days of one arm's `system_cost`.  The multiset of the directory's values is
unchanged and each is exactly representable, so the directory total is
bit-identical -- an attribution error with no magnitude in it.  Measured
2026-09-11 on Linux-6.17.0-35-generic x86_64 (python 3.11.15, numpy 2.4.6): the
total form moves **0 percentage points** (all four thresholds and all twelve
displacements print the same characters, and `totals()` returns the same bits),
while in the per-day form that arm's largest single-day cost displacement moves
from `+0.0000%` to `+95.0000%` -- **95 percentage points** -- and the per-day
cross-seed range's median moves from `5.128205%` to `76.595745%`.  That is the
whole reason `--per-day` exists; the totals are blind to it by construction.

**The second bite, for (c)**, added 2026-09-12 with the paired
denominator.  `test_a_permutation_of_the_untrained_seeds...` exchanges two of the
three baselines.  The multiset of denominators is unchanged, so (a) -- which
reads only the first -- and (b) -- whose mean is permutation invariant -- are
blind BY CONSTRUCTION and not by luck, and every fixture value is a small integer
so "blind" means bit-identical.  (c) is not: `arm_s1`'s cost displacement moves
from `-4.1958%` to `-6.1644%` (**1.9686 percentage points**) and `arm_s2`'s from
`-4.7945%` to `-2.7972%` (**1.9973 percentage points**), the latter flipping a
verdict -- 1.64x the threshold becomes 0.96x, so a cell reported 3/3 would be
reported 2/3.  Measured 2026-09-12 on the same platform as above.  Five mutants
of the device were run against this file and none survived: (c) falling back to
the single seed and (c) pairing in reverse each fail 6 tests, (c) paired in the
totals but not per day fails 2, and the two pairing gates fail 1 each.

**What this does not catch**, so a green result is not read as more: whether the
`.npz` products on disk are right, and whether the adopted denominator is the
right one -- that second one is a choice of definition and the device settles
nothing, it prints all three and names any arm where they disagree.  Changing
the definition would move magnitudes already published; (c) was adopted as the
denominator on 2026-09-12.
"""
import io
import pathlib
import re
import sys
from contextlib import redirect_stdout

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "benchmark"))

import evaluation                                                 # noqa: E402
import y1_thresholds                                              # noqa: E402

DATES = ("2024-01-01", "2024-01-02", "2024-01-03")
RUN_POINT = dict(cap_scale=0.6, ramp_scale=1.0, window="l0 fixture, 3 days",
                 voll=10000.0)

# Per-day system cost.  Baseline totals 7000 / 7150 / 7300, mean 7150; arm
# totals 6750 / 6850 / 6950, mean 6850, range 200.
COST = {"base_s0": (1000.0, 2000.0, 4000.0),
        "base_s1": (1150.0, 2000.0, 4000.0),
        "base_s2": (1300.0, 2000.0, 4000.0),
        "arm_s0": (800.0, 1950.0, 4000.0),
        "arm_s1": (900.0, 1900.0, 4050.0),
        "arm_s2": (1000.0, 2000.0, 3950.0)}
# Per-day profit, summed over the two agents.  Baseline totals 70 / 71 / 72,
# mean 71; arm totals 100 / 110 / 120, mean 110, range 20.
PROFIT = {"base_s0": (10.0, 20.0, 40.0),
          "base_s1": (11.0, 20.0, 40.0),
          "base_s2": (12.0, 20.0, 40.0),
          "arm_s0": (40.0, 20.0, 40.0),
          "arm_s1": (45.0, 25.0, 40.0),
          "arm_s2": (50.0, 30.0, 40.0)}

# The four thresholds, as range/|mean| of the integers above:
# cost 200/6850, profit 20/110, untrained cost 300/7150, untrained profit 2/71.
THRESHOLD = {"system cost": 2.919708, "per-agent profit": 18.181818}
UNTRAINED_SPREAD = {"system cost": 4.195804, "per-agent profit": 2.816901}

# (quantity, denominator, arm) -> (displacement %, verdict).  Denominator (a) is
# base_s0 alone (7000 cost, 70 profit), (b) the mean of the three (7150, 71).
TOTAL_FORM = {
    ("system cost", "a", "arm_s0"): (-3.5714, "pass"),
    ("system cost", "a", "arm_s1"): (-2.1429, "fail"),
    ("system cost", "a", "arm_s2"): (-0.7143, "fail"),
    ("system cost", "b", "arm_s0"): (-5.5944, "pass"),
    ("system cost", "b", "arm_s1"): (-4.1958, "pass"),
    ("system cost", "b", "arm_s2"): (-2.7972, "fail"),
    ("per-agent profit", "a", "arm_s0"): (+42.8571, "pass"),
    ("per-agent profit", "a", "arm_s1"): (+57.1429, "pass"),
    ("per-agent profit", "a", "arm_s2"): (+71.4286, "pass"),
    ("per-agent profit", "b", "arm_s0"): (+40.8451, "pass"),
    ("per-agent profit", "b", "arm_s1"): (+54.9296, "pass"),
    ("per-agent profit", "b", "arm_s2"): (+69.0141, "pass"),
    # (c) pairs arm sN with base_sN: 6750/7000, 6850/7150, 6950/7300 for cost
    # and 100/70, 110/71, 120/72 for profit.
    ("system cost", "c", "arm_s0"): (-3.5714, "pass"),
    ("system cost", "c", "arm_s1"): (-4.1958, "pass"),
    ("system cost", "c", "arm_s2"): (-4.7945, "pass"),
    ("per-agent profit", "c", "arm_s0"): (+42.8571, "pass"),
    ("per-agent profit", "c", "arm_s1"): (+54.9296, "pass"),
    ("per-agent profit", "c", "arm_s2"): (+66.6667, "pass"),
}

# (quantity, denominator, arm) -> (median, min, max, same-sign days) over 3 days.
PER_ITEM_FORM = {
    ("system cost", "a", "arm_s0"): (-2.5000, -20.0000, +0.0000, 2),
    ("system cost", "a", "arm_s1"): (-5.0000, -10.0000, +1.2500, 2),
    ("system cost", "a", "arm_s2"): (+0.0000, -1.2500, +0.0000, 1),
    ("system cost", "b", "arm_s0"): (-2.5000, -30.4348, +0.0000, 2),
    ("system cost", "b", "arm_s1"): (-5.0000, -21.7391, +1.2500, 2),
    ("system cost", "b", "arm_s2"): (-1.2500, -13.0435, +0.0000, 2),
    ("per-agent profit", "a", "arm_s0"): (+0.0000, +0.0000, +300.0000, 1),
    ("per-agent profit", "a", "arm_s1"): (+25.0000, +0.0000, +350.0000, 2),
    ("per-agent profit", "a", "arm_s2"): (+50.0000, +0.0000, +400.0000, 2),
    ("per-agent profit", "b", "arm_s0"): (+0.0000, +0.0000, +263.6364, 1),
    ("per-agent profit", "b", "arm_s1"): (+25.0000, +0.0000, +309.0909, 2),
    ("per-agent profit", "b", "arm_s2"): (+50.0000, +0.0000, +354.5455, 2),
    ("system cost", "c", "arm_s0"): (-2.5000, -20.0000, +0.0000, 2),
    ("system cost", "c", "arm_s1"): (-5.0000, -21.7391, +1.2500, 2),
    ("system cost", "c", "arm_s2"): (-1.2500, -23.0769, +0.0000, 2),
    ("per-agent profit", "c", "arm_s0"): (+0.0000, +0.0000, +300.0000, 1),
    ("per-agent profit", "c", "arm_s1"): (+25.0000, +0.0000, +309.0909, 2),
    ("per-agent profit", "c", "arm_s2"): (+50.0000, +0.0000, +316.6667, 2),
}
# The per-day cross-seed range, quantity -> (median, min, max).  Cost days are
# 200/900, 100/1950, 100/4000; profit days are 10/45, 10/25, 0/40.
PER_DAY_RANGE = {"system cost": (5.128205, 2.500000, 22.222222),
                 "per-agent profit": (22.222222, 0.000000, 40.000000)}


def _write(cell, name, *, dates=DATES, cost=None, profit=None):
    """One run directory, written by the writer the real products come from.

    `evaluation.write_day` rather than a hand-rolled `np.savez`, so that a change
    to the product format fails here instead of leaving this file testing a
    format nothing writes any more.
    """
    out = cell / name
    for k, (date, sc, pf) in enumerate(zip(dates, cost or COST[name],
                                           profit or PROFIT[name])):
        evaluation.write_day(out, "sac", k, date, system_cost_value=sc,
                             agent_profit=np.array([pf - 1.0, 1.0], np.float64),
                             shed_mwh=np.zeros(1), production_cost=np.zeros(1),
                             run_point=RUN_POINT)
    return out


def _cell(tmp_path, **kw):
    for name in COST:
        _write(tmp_path, name, **kw.pop(name, {}))
    assert not kw, kw
    return tmp_path


def _argv(cell, *, per_day=True, extra_baselines=(1, 2)):
    out = [cell / "base_s0"] + [cell / f"arm_s{s}" for s in (0, 1, 2)]
    for s in extra_baselines:
        out += ["--baseline", cell / f"base_s{s}"]
    return [str(x) for x in out] + (["--per-day"] if per_day else [])


def _curves(cell, seeds=(0, 1, 2)):
    """Give every directory the `curve.jsonl` the real drivers write.

    Only its first line is read, and only its `seed`.  Without these files the
    device cannot check the pairing and says so; with them a wrong `--baseline`
    order is refused.
    """
    import json
    for prefix in ("base_s", "arm_s"):
        for k, seed in enumerate(seeds if prefix == "base_s" else (0, 1, 2)):
            with open(cell / f"{prefix}{k}" / "curve.jsonl", "w") as f:
                f.write(json.dumps({"kind": "meta", "algo": "sac",
                                    "seed": seed}) + "\n")
    return cell


def _run(argv):
    buf, saved = io.StringIO(), sys.argv
    sys.argv = ["y1_thresholds.py"] + list(argv)
    try:
        with redirect_stdout(buf):
            y1_thresholds.main()
    finally:
        sys.argv = saved
    return buf.getvalue()


_THR = re.compile(r"^(system cost|per-agent profit): range \S+ / \|mean\| \S+ "
                  r"-> threshold ([\d.]+)%$")
_SPREAD = re.compile(r"^  (system cost|per-agent profit): range \S+ / \|mean\| "
                     r"\S+ -> ([\d.]+)%$")
_DENOM = re.compile(r"^  denominator \((a|b|c)\) ")
_FLIP = re.compile(r"^  verdict across \d+ denominators: (.*)$")
_DISP = re.compile(r"^ +(\S+) +displacement +([-+][\d.]+)% +(pass|fail) +"
                   r"\(\|disp\|/thr = ([\d.]+)x\)(   ÷ \S+ \S+)?$")
_PDRANGE = re.compile(r"^  per day \((\d+) days\), cross-seed range/\|mean\|: "
                      r"median ([\d.]+)% +min ([\d.]+)% +max ([\d.]+)%$")
_PDDENOM = re.compile(r"^  per day, denominator \((a|b|c)\) ")
_PAIR = re.compile(r"^\(c\) (pairing checked|paired by position): (.*)$")
_PDARM = re.compile(r"^ {4}(\S+) +median +([-+][\d.]+)% +"
                    r"\[ *([-+][\d.]+)%, *([-+][\d.]+)%\] +"
                    r"same sign as the total on (\d+)/(\d+) days$")


def _parse(text):
    """The device's printed report, as a dict keyed the way it is read."""
    out = {"threshold": {}, "untrained_spread": {}, "flip": {}, "total": {},
           "per_day_range": {}, "per_day": {}, "pairing": None}
    qty = denom = None
    for line in text.splitlines():
        if (m := _PAIR.match(line)):
            out["pairing"] = (m.group(1), m.group(2))
        elif (m := _THR.match(line)):
            qty, denom = m.group(1), "a"
            out["threshold"][qty] = float(m.group(2))
        elif qty is None and (m := _SPREAD.match(line)):
            out["untrained_spread"][m.group(1)] = float(m.group(2))
        elif (m := _DENOM.match(line)) or (m := _PDDENOM.match(line)):
            denom = m.group(1)
        elif (m := _FLIP.match(line)):
            out["flip"][qty] = m.group(1)
        elif (m := _DISP.match(line)):
            out["total"][(qty, denom, pathlib.Path(m.group(1)).name)] = (
                float(m.group(2)), m.group(3))
        elif (m := _PDRANGE.match(line)):
            out["per_day_range"][qty] = (int(m.group(1)), float(m.group(2)),
                                         float(m.group(3)), float(m.group(4)))
        elif (m := _PDARM.match(line)):
            out["per_day"][(qty, denom, pathlib.Path(m.group(1)).name)] = (
                float(m.group(2)), float(m.group(3)), float(m.group(4)),
                int(m.group(5)), int(m.group(6)))
    return out


def _assert_total_form(p):
    """The four numbers section 5 defines, and the twelve verdicts under them."""
    assert p["threshold"] == THRESHOLD
    assert p["total"] == TOTAL_FORM


def _assert_per_item_form(p):
    """The same two quantities read day by day."""
    assert p["per_day_range"] == {q: (3, *v) for q, v in PER_DAY_RANGE.items()}
    assert p["per_day"] == {k: (*v[:3], v[3], 3) for k, v in PER_ITEM_FORM.items()}


def test_thresholds_and_both_denominators_are_section_five_arithmetic(tmp_path):
    _assert_total_form(_parse(_run(_argv(_cell(tmp_path)))))


def test_the_per_item_form_is_printed_and_is_the_same_two_quantities(tmp_path):
    _assert_per_item_form(_parse(_run(_argv(_cell(tmp_path)))))


def test_the_untrained_side_gets_its_own_spread(tmp_path):
    """The threshold is sized on the trained side, so it cannot say how far the
    denominator moves when ITS seed changes -- 4.195804% and 2.816901% here."""
    assert _parse(_run(_argv(_cell(tmp_path))))["untrained_spread"] == UNTRAINED_SPREAD


def test_a_denominator_flip_is_named_arm_by_arm(tmp_path):
    """Two of the three arms get a different verdict from a different denominator.

    `arm_s1`'s cost is 0.73x the threshold under (a), 1.44x under (b) and 1.44x
    under (c); `arm_s2`'s is 0.24x, 0.96x and 1.64x.  So the cost channel splits
    on both of them and the profit channel on none, and the device prints every
    denominator's verdict for each split arm rather than reducing them to one.
    """
    flip = _parse(_run(_argv(_cell(tmp_path))))["flip"]
    assert flip["per-agent profit"] == "all agree"
    assert flip["system cost"].startswith("**flipped** ")
    assert "arm_s1 (single fail, mean pass, paired pass)" in flip["system cost"]
    assert "arm_s2 (single fail, mean fail, paired pass)" in flip["system cost"]
    assert "arm_s0 (" not in flip["system cost"]


def test_one_baseline_and_no_per_day_prints_what_it_printed_before(tmp_path):
    """The documented invocation is cited as the definition in five documents,
    so the addition is additive: no denominator header, no per-day block, and
    the arm lines still start at two spaces rather than four."""
    text = _run(_argv(_cell(tmp_path), per_day=False, extra_baselines=()))
    assert "denominator" not in text
    assert "per day" not in text
    assert "verdict across" not in text
    assert "untrained spread" not in text
    assert "pair" not in text            # (c) needs one baseline per arm
    arms = [l for l in text.splitlines() if "displacement" in l]
    assert len(arms) == 6
    assert all(l.startswith("  /") and not l.startswith("   ") for l in arms)
    p = _parse(text)
    assert p["threshold"] == THRESHOLD
    assert p["total"] == {k: v for k, v in TOTAL_FORM.items() if k[1] == "a"}


def test_a_short_directory_is_still_refused(tmp_path):
    """The 2026-08-30 gate, unweakened: three arms read mid-write held 30 / 26 /
    20 of 36 days and produced displacements that meant nothing."""
    cell = _cell(tmp_path, arm_s1=dict(dates=DATES[:2], cost=COST["arm_s1"][:2],
                                       profit=PROFIT["arm_s1"][:2]))
    with pytest.raises(SystemExit, match="day counts disagree"):
        _run(_argv(cell))


def test_the_same_day_count_over_different_days_is_refused(tmp_path):
    """Three days against three other days passes the count gate and is still
    not a comparison: the difference would be weather and load, not policy."""
    cell = _cell(tmp_path, arm_s2=dict(dates=("2024-01-01", "2024-01-02",
                                              "2024-01-04")))
    with pytest.raises(SystemExit, match="evaluation dates disagree"):
        _run(_argv(cell))


def test_an_empty_directory_is_refused_rather_than_read_as_complete(tmp_path):
    cell = _cell(tmp_path)
    (cell / "empty").mkdir()
    with pytest.raises(SystemExit, match="no day products"):
        _run([str(cell / "base_s0"), str(cell / "empty")])


def _swap_days(run_dir, i, j):
    """Move value between two days of one directory, preserving its total.

    Rewrites both products through `np.savez` with the two `system_cost` fields
    exchanged.  Both are integers, so the directory's sum is bit-identical: this
    is an attribution error with no magnitude in it, which is the only kind that
    a total-only check provably cannot see.
    """
    import json
    paths = sorted(run_dir.glob("*_day*.npz"))
    blobs = [dict(np.load(p, allow_pickle=True)) for p in paths]
    metas = [json.loads(str(b["meta"])) for b in blobs]
    metas[i]["system_cost"], metas[j]["system_cost"] = (metas[j]["system_cost"],
                                                        metas[i]["system_cost"])
    for p, b, m in zip(paths, blobs, metas):
        b["meta"] = np.asarray(json.dumps(m))
        np.savez(p, **b)


def test_the_pairing_is_unchecked_and_says_so_when_there_is_no_curve_jsonl(tmp_path):
    """A missing `seed` must not read as an agreeing one.

    The fixture writes day products only, so the device cannot know which seed
    any directory is; it pairs by position and prints that it did not check,
    naming how many directories it could not read.
    """
    p = _parse(_run(_argv(_cell(tmp_path))))
    assert p["pairing"][0] == "paired by position"
    assert p["pairing"][1].startswith("seeds unchecked, 6 directories have no curve.jsonl (")
    assert p["total"] == TOTAL_FORM          # and (c) is still computed


def test_the_pairing_is_checked_seed_by_seed_when_curve_jsonl_is_there(tmp_path):
    assert _parse(_run(_argv(_curves(_cell(tmp_path)))))["pairing"] == (
        "pairing checked", "untrained seeds 0/1/2 match trained seeds 0/1/2 one to one"
        " (read from the first line of each curve.jsonl)")


def test_a_baseline_order_that_does_not_match_the_arms_is_refused(tmp_path):
    """The seeds are on disk, so the mis-pairing below is caught rather than run."""
    cell = _curves(_cell(tmp_path))
    with pytest.raises(SystemExit, match="the k-th baseline is not the k-th arm"):
        _run(_argv(cell, extra_baselines=(2, 1)))


def test_a_permutation_of_the_untrained_seeds_is_invisible_to_a_and_b_and_moves_c(tmp_path):
    """Exchange two of the three baselines: only (c) can see it, and it does.

    This is the (c) counterpart of the day swap below, and the same shape of
    injection: the multiset of denominators is unchanged, so (a) --
    which reads only the first -- and (b) -- whose mean is permutation invariant
    -- are blind by construction rather than by luck, and every value here is a
    small integer so "blind" means bit-identical and not merely close.  Without
    `curve.jsonl` the device cannot refuse the order, which is exactly the
    situation this measures.
    """
    cell = _cell(tmp_path)
    before = _parse(_run(_argv(cell)))
    after = _parse(_run(_argv(cell, extra_baselines=(2, 1))))

    # (a) and (b) print the same characters, thresholds and spread included.
    assert after["threshold"] == before["threshold"] == THRESHOLD
    assert after["untrained_spread"] == before["untrained_spread"] == UNTRAINED_SPREAD
    for k in ("total", "per_day"):
        assert {j: v for j, v in after[k].items() if j[1] in "ab"} == \
               {j: v for j, v in before[k].items() if j[1] in "ab"}
    assert after["per_day_range"] == before["per_day_range"]

    # (c) is not blind.  arm_s1 is now divided by base_s2 (6850/7300) and arm_s2
    # by base_s1 (6950/7150), and the measured bite, in percentage points of
    # displacement:
    q = "system cost"
    assert before["total"][(q, "c", "arm_s1")] == (-4.1958, "pass")
    assert after["total"][(q, "c", "arm_s1")] == (-6.1644, "pass")      # 1.9686 pp
    assert before["total"][(q, "c", "arm_s2")] == (-4.7945, "pass")
    assert after["total"][(q, "c", "arm_s2")] == (-2.7972, "fail")    # 1.9973 pp

    # That one moves a verdict, not only a number: 1.64x the threshold becomes
    # 0.96x, so the cell would be reported 2/3 where it is 3/3.  The comparison
    # line shows the same thing from the other side -- before the swap (c)
    # disagreed with (a) and (b) about arm_s2 and it was named; after, all three
    # say "fail" and there is nothing left to name.
    assert "arm_s2 (single fail, mean fail, paired pass)" in before["flip"][q]
    assert "arm_s2 (" not in after["flip"][q]

    # The arm that was paired correctly all along did not move.
    assert after["total"][(q, "c", "arm_s0")] == before["total"][(q, "c", "arm_s0")]


def test_a_total_preserving_reattribution_passes_the_total_form_and_fails_the_per_item_form(tmp_path):
    cell = _cell(tmp_path)
    before = _parse(_run(_argv(cell)))
    _assert_total_form(before)
    _assert_per_item_form(before)
    totals_before = y1_thresholds.totals(cell / "arm_s0")

    _swap_days(cell / "arm_s0", 0, 1)          # 800 <-> 1950, sum 6750 either way
    after = _parse(_run(_argv(cell)))

    # The total form is blind to it, and provably so rather than approximately:
    # the same bits out of `totals`, and the same characters out of the report.
    assert y1_thresholds.totals(cell / "arm_s0") == totals_before
    _assert_total_form(after)

    # The per-item form is not.
    with pytest.raises(AssertionError):
        _assert_per_item_form(after)

    # The measured bite, in percentage points of displacement.
    q, arm = "system cost", "arm_s0"
    assert before["per_day"][(q, "a", arm)][:3] == (-2.5000, -20.0000, +0.0000)
    assert after["per_day"][(q, "a", arm)][:3] == (+0.0000, -60.0000, +95.0000)
    assert after["per_day"][(q, "a", arm)][3] == 1        # was 2 of 3 days
    assert before["per_day_range"][q] == (3, 5.128205, 2.500000, 22.222222)
    assert after["per_day_range"][q] == (3, 76.595745, 2.500000, 81.818182)
    # ... and nothing in the profit channel moved, so the failure is attributable.
    assert {k: v for k, v in after["per_day"].items() if k[0] != q} == \
           {k: v for k, v in before["per_day"].items() if k[0] != q}


def test_a_directory_holding_two_arms_is_refused(tmp_path):
    """The evaluation drivers write every arm into one --out-dir, so a directory
    can hold `honest_day*.npz` beside `sac_day*.npz`.  When every directory is
    mixed the same way, the day-count and date gates all pass and the glob
    sums the two arms into one total -- silently.  The device must refuse and
    name the prefixes instead."""
    cell = _cell(tmp_path)
    for name in COST:                       # the same second arm in every directory
        for k, (date, sc, pf) in enumerate(zip(DATES, COST[name], PROFIT[name])):
            evaluation.write_day(cell / name, "honest", k, date,
                                 system_cost_value=sc,
                                 agent_profit=np.array([pf - 1.0, 1.0], np.float64),
                                 shed_mwh=np.zeros(1), production_cost=np.zeros(1),
                                 run_point=RUN_POINT)
    with pytest.raises(SystemExit, match=r"2 arms \(honest, sac\)"):
        _run(_argv(cell))
