"""L0: `run_rl_02.py` can evaluate on a day set other than the frozen held-out
one, and says in the product which days those were.

**What this is for** (2026-09-19): 813nem's training
reward rises on all three seeds while the 36-day held-out profit falls on two,
and the two numbers are read on disjoint day sets.  Measuring that gap needs one
thing the driver did not have -- the same archive, the same device, evaluated on
*training* days.  `split_days` freezes the evaluation days per case and no flag
could change them (`--eval-only` picks which archive, not which days).

**The vocabulary is copied, not invented**: `--days {eval,train,all}` is
`run_eval_02.py`'s flag, with its name, choices and meaning.  Two drivers giving
the same quantity two different names is worse than both keeping the old one.  The one thing that could not be copied is that
driver's `--day-from/--day-to`, which slices a *contiguous* range: the held-out
set is scattered across the year (adjacent gaps median 10 days), so a contiguous
run of training days would put seasonality back into the comparison.  Hence the
equally spaced comb, `--day-take` with `--day-phase`.

**The regression test that matters most is the whole-tooth phase.**  Measured
before the flag was wired up: pool 48 taken 12 (spacing 4) with `phase=4` gives
index-for-index what `phase=0` gives, because the rotation lands on the next
tooth.  That would have made the gap measurement's negative control -- "another equally spaced
set at a different phase must agree in sign and magnitude" -- *vacuous*: the two
sets would be the same days, so of course they agree.  A green criterion that
verifies nothing is worse than a red one.

`equally_spaced` is exercised by compiling the driver's own function out of its
source rather than by importing the driver (which would pull in jax and a GPU
context for an arithmetic check).  `test_the_function_under_test_is_the_driver_s`
is what keeps that honest: it fails if the function ever starts depending on a
module-level name, which is the only way the compiled copy could diverge from
the one that runs.
"""
import ast
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
BENCH = REPO / "tools" / "benchmark"
SRC = (BENCH / "run_rl_02.py").read_text()
SRC_EVAL_02 = (BENCH / "run_eval_02.py").read_text()

_TREE = ast.parse(SRC)
_FN = next((n for n in _TREE.body
            if isinstance(n, ast.FunctionDef) and n.name == "equally_spaced"),
           None)


def _compiled():
    assert _FN is not None, "run_rl_02.py has no module-level `equally_spaced`"
    ns = {}
    mod = ast.Module(body=[_FN], type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), "<run_rl_02>", "exec"), ns)
    return ns["equally_spaced"]


def _flag_call(src, flag):
    """The `ap.add_argument("<flag>", ...)` call node, or None."""
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call) and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == flag):
            return node
    return None


def _kw(call, name):
    for k in call.keywords:
        if k.arg == name:
            return ast.literal_eval(k.value)
    return None


def test_the_function_under_test_is_the_driver_s():
    """The compiled copy can only diverge from the running one by picking up a
    module-level name, so assert it has none beyond builtins."""
    assigned = set()
    for node in ast.walk(_FN):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            assigned.add(node.id)
    for a in _FN.args.args:
        assigned.add(a.arg)
    for node in ast.walk(_FN):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            assert node.id in assigned or node.id in dir(__builtins__) or \
                node.id in ("len", "sorted", "set", "int", "range", "ValueError",
                            "AssertionError"), \
                (f"equally_spaced reads the module-level name {node.id!r}; the "
                 f"exec-compiled copy in this test would not see it")


def test_days_flag_copies_run_eval_02_vocabulary():
    here, there = _flag_call(SRC, "--days"), _flag_call(SRC_EVAL_02, "--days")
    assert here is not None, "run_rl_02.py declares no --days"
    assert there is not None, "run_eval_02.py declares no --days (did it move?)"
    assert tuple(_kw(here, "choices")) == tuple(_kw(there, "choices")), \
        "the two drivers must offer the same day pools under the same names"
    assert _kw(here, "default") == "eval", \
        "the default must be the frozen held-out set, so a command line without " \
        "the flag is what this driver always did"


def test_default_pool_is_the_frozen_split_and_train_is_its_complement():
    """`--days train` must read `tr_days` -- the complement of the evaluation
    set, not a subset of it.  This is the property that made option A the only
    path: `eval_policy_cpu.py --days` filters *within* the held-out set, which
    reads like the same feature and cannot answer that question."""
    src = ast.get_source_segment(SRC, _pool_assignment()) or ""
    assert '"train": tr_days' in src.replace("'", '"'), \
        f"the train pool must be tr_days; the assignment reads: {src}"
    assert '"eval": ev_days' in src.replace("'", '"')


def _pool_assignment():
    for node in ast.walk(_TREE):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "_pool"
                and isinstance(node.value, ast.Subscript)):
            return node
    raise AssertionError("no `_pool = {...}[args.days]` assignment found")


def test_training_days_are_never_narrowed():
    """The flag changes what is evaluated, not what is trained on.  If `tr_days`
    were reassigned after the split, `--days train` would also shrink the
    training environment and the run would no longer be the same device."""
    main = next(n for n in _TREE.body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    split_line = next(n.lineno for n in ast.walk(main)
                      if isinstance(n, ast.Assign)
                      and isinstance(n.value, ast.Call)
                      and getattr(n.value.func, "id", "") == "split_days")
    for node in ast.walk(main):
        if (isinstance(node, ast.Assign) and node.lineno > split_line):
            for t in node.targets:
                assert getattr(t, "id", None) != "tr_days", \
                    f"tr_days is reassigned at line {node.lineno}; the training " \
                    f"environment must keep the whole training window"


def test_both_stamps_are_written_unconditionally():
    """A product without these keys cannot be told apart from one written before
    they existed, which is the same false pass as a missing `commit`.
    So they are stamped on every path, and `eval_day_set` carries
    the day numbers rather than a label naming how they were chosen -- a label
    does not let a consumer answer "which days was this".
    """
    assert SRC.count("eval_day_set=[int(d) for d in ev_days]") == 2, \
        "both stamp sites (curve meta and run point) must carry eval_day_set"
    assert SRC.count("day_select=day_select") + SRC.count("day_select=day_select,") >= 1
    assert 'day_select = "split_days" if args.days == "eval" else args.days' in SRC, \
        "the default path must stamp the literal `split_days`, which is what " \
        "the bit-for-bit gate asserts against"
    for node in ast.walk(_TREE):
        if (isinstance(node, ast.keyword) and node.arg == "eval_day_set"):
            assert isinstance(node.value, ast.ListComp), \
                "stamp the day numbers themselves, not a count or a label"


@pytest.mark.parametrize("n,k", [(48, 12), (329, 36), (60, 60), (365, 36), (7, 1)])
def test_comb_takes_k_distinct_and_spreads_them(n, k):
    equally_spaced = _compiled()
    pool = list(range(100, 100 + n))
    got, idx = equally_spaced(pool, k)
    assert len(got) == k == len(set(got))
    assert got == sorted(got)
    assert set(got) <= set(pool)
    if k > 1:
        gaps = [b - a for a, b in zip(idx, idx[1:])]
        assert max(gaps) - min(gaps) <= 1, \
            f"spacing must be even to within one day; gaps {sorted(set(gaps))}"


def test_take_all_is_the_identity():
    equally_spaced = _compiled()
    pool = list(range(48))
    got, _ = equally_spaced(pool, len(pool))
    assert got == pool


def test_whole_tooth_phase_is_refused():
    """The measured regression: pool 48 taken 12 has spacing 4, and phase 4 is a
    whole tooth, so it returns phase 0's days.  Refused rather than normalised,
    because the caller asked for a different set."""
    equally_spaced = _compiled()
    pool = list(range(48))
    base, _ = equally_spaced(pool, 12)
    with pytest.raises(ValueError, match="SAME days as phase 0"):
        equally_spaced(pool, 12, 4)
    with pytest.raises(ValueError, match="SAME days as phase 0"):
        equally_spaced(pool, 12, 8)
    # and the negative control of the guard itself: it must not fire on phases
    # that do change the set, or the flag would be unusable
    for phase in (1, 2, 3):
        got, _ = equally_spaced(pool, 12, phase)
        assert got != base and len(got) == 12


def test_take_larger_than_the_pool_is_refused():
    equally_spaced = _compiled()
    with pytest.raises(ValueError, match="cannot take more teeth"):
        _compiled()(list(range(10)), 11)
    with pytest.raises(ValueError, match="must be >= 1"):
        equally_spaced(list(range(10)), 0)


def test_day_phase_alone_is_refused_before_any_work():
    """`--day-phase` without `--day-take` rotates nothing.  The guard sits in
    argument validation so it bites in seconds rather than after the fixture and
    the environments are built ("this parameter has no
    effect" is first of all "I never passed it in")."""
    out = subprocess.run(
        [sys.executable, str(BENCH / "run_rl_02.py"),
         "--position", "/nonexistent.npz", "--out-dir", "/tmp/unused",
         "--day-phase", "1"],
        capture_output=True, text=True, timeout=300,
        env={**__import__("os").environ, "JAX_PLATFORMS": "cpu"})
    assert out.returncode != 0
    assert "without --day-take" in out.stdout + out.stderr, \
        (out.stdout + out.stderr)[-800:]
