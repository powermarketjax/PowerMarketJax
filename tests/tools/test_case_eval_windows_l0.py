"""L0: each case's evaluation days come from that case's own frozen split.

`evaluation.split_days` keyed its answer on the window's **length** until
2026-09-12, and `case813nem`'s window is 365 days long -- exactly `case29gb`'s Y1
window.  So `split_days(len(dates))` on a NEM position returned GB's thirty-six
offsets and raised nothing: the driver would have trained on days it then
evaluated on, and every product would have looked well-formed.  `case73rts`'s
366-day window was loud by accident rather than by design (366 matched no
branch, so it raised).

**The measured "it bites" datum**, as required before a check counts: on a
365-day window, GB's frozen offsets and NEM's differ on **66 of the 72**
(36 + 36) entries -- they share only six days by coincidence of the arithmetic.
So the defect this file rules out is not a near miss; it is a different
evaluation set.

**What is checked here, and why each half is needed.**

    the table     one entry per case, offsets and dates the same length, inside
                  the window, sorted and distinct
    the copies    `evaluation`'s offsets equal the deriving devices' own
                  constants.  The duplication is deliberate -- `evaluation` is
                  imported both bare and as `benchmark.evaluation`, and under the
                  second the devices are not importable by bare name -- so this
                  test is what makes the copy unable to drift
    the calendar  the frozen dates are what those offsets actually select out of
                  each case's calendar.  Without this the table could be
                  internally consistent and still describe days that are not
                  there
    the dispatch  passing the case changes the answer, and not passing it on a
                  case that needs it fails rather than silently substituting
    the gate      `case` has no default, so a call that does not name it cannot
                  run at all; and no call in `tools/` or `tests/` is bare, checked
                  on the syntax tree rather than by grep

**The second hardening (2026-09-12, later the same day).**  The dispatch above
was added with `case` defaulting to `29gb`, which protected exactly the callers
that named their case -- the four 02/03 drivers -- and left thirty-two others in
`tools/` taking GB's days on any 365-day window without a word.  Measured before
the default was removed: 46 bare calls under `tools/` and `tests/` (32 to
`split_days` and 9 to `check_split_against_report` in `tools/`, 5 in this file);
after: 0.  The default is gone rather than guarded, because a length check is
known not to distinguish NEM from GB and a "default only when declared" flag is
a default under another name.

The splits themselves are not re-derived here and must not be: they are frozen
run points (RTS 2026-09-04, NEM 2026-09-06), and the rule
that produced them reads the calendar and nothing else -- the sampling
criterion is exogenous to the quantity being reported.  `rts_eval_split.py --verify`
and `nem_eval_split.py --verify` recompute them from the calendar on demand.
"""
import ast
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "benchmark"))

import evaluation as E                                            # noqa: E402
import nem_eval_split as NEM                                      # noqa: E402
import rts_eval_split as RTS                                      # noqa: E402
import y1_eval_split as Y1                                        # noqa: E402

#: `case -> (the module that derives it, its own frozen offsets)`.
DEVICES = {
    "29gb": (Y1, E.YEAR_EVAL_OFFSETS),
    "73rts": (RTS, RTS.RTS_EVAL_OFFSETS),
    "813nem": (NEM, NEM.NEM_EVAL_OFFSETS),
}


@pytest.mark.parametrize("case", sorted(E.CASE_EVAL_WINDOWS))
def test_the_table_entry_is_well_formed(case):
    n_days, offsets, dates = E.CASE_EVAL_WINDOWS[case]
    assert len(offsets) == len(dates) == 36, (
        f"{case} holds out {len(offsets)} days and lists {len(dates)} dates")
    assert list(offsets) == sorted(set(offsets)), (
        f"{case}'s offsets are not sorted and distinct")
    assert 0 <= min(offsets) and max(offsets) < n_days, (
        f"{case}'s offsets run outside its own {n_days}-day window")


@pytest.mark.parametrize("case", sorted(DEVICES))
def test_the_table_matches_the_device_that_derives_it(case):
    """The one gate that stops the deliberate copy from drifting."""
    _dev, own = DEVICES[case]
    assert tuple(E.CASE_EVAL_WINDOWS[case][1]) == tuple(own), (
        f"{case}'s offsets in evaluation.py and in its deriving device have "
        f"come apart; the drivers read the first and the notes cite the second")


@pytest.mark.parametrize("case", sorted(DEVICES))
def test_the_frozen_dates_are_what_the_calendar_gives_at_those_offsets(case):
    """The half that cannot be got right by editing one tuple consistently."""
    dev, _own = DEVICES[case]
    dates = dev.window_dates()
    n_days, offsets, want = E.CASE_EVAL_WINDOWS[case]
    assert len(dates) == n_days, (
        f"{case}'s window is {len(dates)} days on the calendar and {n_days} in "
        f"the table")
    got = tuple(str(dates[i].date()) for i in offsets)
    assert got == tuple(want)


def test_the_nem_and_gb_windows_are_the_same_length_and_different_days():
    """The collision this table exists for, with the number that makes it bite."""
    assert E.NEM_LEN == E.YEAR_LEN == 365
    gb, nem = set(E.YEAR_EVAL_OFFSETS), set(E.NEM_EVAL_OFFSETS)
    assert len(gb ^ nem) == 66, (
        f"GB's and NEM's 365-day splits differ on {len(gb ^ nem)} of 72 offsets; "
        f"if this ever reached zero the dispatch would have nothing to protect")


def test_the_case_selects_the_answer():
    assert E.split_days(365, "813nem")[0] == sorted(E.NEM_EVAL_OFFSETS)
    assert E.split_days(365, "29gb")[0] == sorted(E.YEAR_EVAL_OFFSETS)
    assert E.split_days(366, "73rts")[0] == sorted(E.RTS_EVAL_OFFSETS)
    # the 60-day rule is still the 60-day rule, and still only 29gb's
    assert E.split_days(60, "29gb")[0] == [0, 5, 10, 16, 21, 26, 32, 37, 42, 48, 53, 58]


def test_a_window_a_case_does_not_have_is_refused():
    with pytest.raises(ValueError, match="366"):
        E.split_days(366, "29gb")               # RTS's length, GB's table
    with pytest.raises(ValueError, match="60"):
        E.split_days(60, "73rts")               # no 60-day window was adopted
    with pytest.raises(ValueError, match="118ieee"):
        E.split_days(365, "118ieee")            # no split at all


@pytest.mark.parametrize("case", sorted(DEVICES))
def test_check_split_passes_on_the_case_it_is_told(case):
    dev, _own = DEVICES[case]
    dates = [str(d.date()) for d in dev.window_dates()]
    ok, _got = E.check_split_against_report(dates, case)
    assert ok, f"{case}'s own window no longer reproduces its own frozen dates"


def test_check_split_refuses_a_nem_window_handed_over_as_gb():
    """The loud half of the collision: same length, so the dates must catch it."""
    dates = [str(d.date()) for d in NEM.window_dates()]
    ok, got = E.check_split_against_report(dates, "29gb")  # a NEM window, called GB
    assert not ok, (
        "a NEM window compared against GB's evaluation dates came back as a "
        "match, so the caller's SystemExit would never fire")
    assert got != tuple(E.NEM_EVAL_DATES)


@pytest.mark.parametrize("case", sorted(DEVICES))
def test_the_check_has_discriminating_power(case, monkeypatch):
    """Move one day; the check must go red.  An injection, not an inspection."""
    dev, _own = DEVICES[case]
    dates = [str(d.date()) for d in dev.window_dates()]
    n_days, offsets, want = E.CASE_EVAL_WINDOWS[case]
    # shift the last held-out day by one, which is the smallest possible change
    # to the split and the one a transcription error would make
    moved = tuple(offsets[:-1]) + (offsets[-1] - 1,)
    assert moved[-1] not in offsets[:-1]
    monkeypatch.setitem(E.CASE_EVAL_WINDOWS, case, (n_days, moved, want))
    ok, _got = E.check_split_against_report(dates, case)
    assert not ok, (
        f"moving {case}'s last evaluation day by one left the check green, so "
        f"it is not comparing the days it claims to")


def test_a_call_that_does_not_name_its_case_cannot_run():
    """The gate itself: no default, and `None` is not a name.

    `None` matters because the natural way to read a fixture written before the
    case was stamped is `meta.get("case")`, which hands `None` on; that must
    fail here, not select GB's days.
    """
    dates = [str(d.date()) for d in NEM.window_dates()]
    with pytest.raises(TypeError):
        E.split_days(365)
    with pytest.raises(TypeError):
        E.split_days()
    with pytest.raises(TypeError):
        E.check_split_against_report(dates)
    with pytest.raises(ValueError, match="None"):
        E.split_days(365, None)
    with pytest.raises(ValueError, match="None"):
        E.check_split_against_report(dates, None)


#: Both selectors, by name; a call to either with fewer than two positional
#: arguments and no `case=` keyword is bare.
_SELECTORS = ("split_days", "check_split_against_report")


def bare_selector_calls(tree):
    """`[(name, lineno)]` of bare calls in one syntax tree.

    On the tree rather than on text, because the text has a decoy: the docstring
    of `check_split_against_report` spells out `split_days(365)` as the thing it
    is not, and a line-count of "split_days( without case" counted it as a call
    (three lines were miscounted that way on 2026-09-12 before the numbers
    agreed).  Attribute calls (`E.split_days(...)`) are looked at too, since
    that is how this file calls it.
    """
    # the one place a bare call is allowed is where its failure is what is
    # being asserted: the body of `with pytest.raises(TypeError)`
    asserted_to_fail = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            ctx = item.context_expr
            if (isinstance(ctx, ast.Call) and isinstance(ctx.func, ast.Attribute)
                    and ctx.func.attr == "raises" and ctx.args
                    and isinstance(ctx.args[0], ast.Name)
                    and ctx.args[0].id == "TypeError"):
                asserted_to_fail.append((node.lineno, node.end_lineno))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = (fn.id if isinstance(fn, ast.Name)
                else fn.attr if isinstance(fn, ast.Attribute) else None)
        if name not in _SELECTORS:
            continue
        if any(lo <= node.lineno <= hi for lo, hi in asserted_to_fail):
            continue
        if len(node.args) < 2 and not any(k.arg == "case" for k in node.keywords):
            out.append((name, node.lineno))
    return out


def test_the_scanner_bites():
    """An injection: the scanner must flag bare calls of both shapes and pass
    named ones of both shapes, or the tree-wide test below is vacuous."""
    bare = ast.parse(
        "ev, tr = split_days(len(dates))\n"
        "ok, got = E.check_split_against_report(dates)\n"
        "E.split_days(365)\n")
    assert sorted(bare_selector_calls(bare)) == [
        ("check_split_against_report", 2), ("split_days", 1), ("split_days", 3)]
    named = ast.parse(
        'ev, tr = split_days(len(dates), meta["case"])\n'
        'ok, got = E.check_split_against_report(dates, case=args.case)\n'
        'E.split_days(365, "813nem")\n'
        '"""a docstring that says split_days(365) is not a call"""\n'
        'with pytest.raises(TypeError):\n'
        '    E.split_days(365)\n')
    assert bare_selector_calls(named) == []
    # and the exemption is only for TypeError: a bare call whose ValueError is
    # asserted would be one that ran, which the gate says cannot happen
    other = ast.parse(
        'with pytest.raises(ValueError):\n'
        '    E.split_days(365)\n')
    assert bare_selector_calls(other) == [("split_days", 2)]


def test_no_call_in_the_tree_leaves_the_case_to_a_default():
    """The wiring, checked on every file because that is what regressed.

    A driver that reads `meta["case"]` for its network and then calls
    `split_days(len(dates))` bare was the whole defect, one level along: the
    network is Australian and the held-out days are British.  Thirty-two callers
    had that shape when the default was removed; the signature now refuses them
    at run time, and this test refuses them at test time, which is earlier than
    most of those tools ever run.
    """
    offenders = {}
    for sub in ("tools", "tests"):
        for path in sorted((REPO / sub).rglob("*.py")):
            tree = ast.parse(path.read_text())
            hits = bare_selector_calls(tree)
            if hits:
                rel = str(path.relative_to(REPO))
                offenders[rel] = [f"{name} at line {ln}" for name, ln in hits]
    assert not offenders, (
        f"{offenders} select held-out days without naming the case; on a "
        f"365-day non-GB window that is silently the wrong thirty-six days")
