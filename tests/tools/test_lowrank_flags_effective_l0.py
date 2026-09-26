"""L0: `run_rl_02.py` refuses a run whose operators were not built on the
low-rank sizing and block solver its command line asked for.

A review on 2026-09-17 injected "``--lowrank-free-shed 128`` on
the command line, ``lowrank_free=`` dropped from the `make_env` call": the
operators were built at (8, 523), stamped 523, and nothing refused -- 128 was
recorded nowhere.  `refuse_ineffective_lowrank_flags` is the guard, the same
shape as `effective_monitored_stamp`: it reads the value in effect off each
clearing's ``spec`` and exits on a mismatch.  Checked here on fake specs
(matching passes, each kind of mismatch exits, ``auto`` is not compared),
and the call site is anchored by source count so a driver that drops the
call goes red rather than silently matching nothing.
"""
import ast
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
BENCH = REPO / "tools" / "benchmark"
sys.path.insert(0, str(BENCH))
from run_rl_02 import refuse_ineffective_lowrank_flags  # noqa: E402

SRC = (BENCH / "run_rl_02.py").read_text()


def _specs(free=(8, 523), lu="sequential", **override):
    base = {"lowrank_free": free, "lu_batching": lu}
    out = []
    for name in ("step", "boundary", "eval"):
        c = dict(base)
        c.update(override.get(name, {}))
        out.append((name, c))
    return tuple(out)


def test_matching_flags_pass():
    refuse_ineffective_lowrank_flags(_specs(), (8, 523), "sequential")
    refuse_ineffective_lowrank_flags(_specs(), (8, 523), "auto")        # auto is not compared
    refuse_ineffective_lowrank_flags(_specs(lu="arrow"), (8, 523), "arrow")


@pytest.mark.parametrize("which", ["step", "boundary", "eval"])
def test_sizing_mismatch_exits_on_any_operator(which):
    """The injection: 128 asked for, the operator built at 523."""
    specs = _specs(**{which: {"lowrank_free": (8, 523)}})
    # the other two carry the requested value, so only `which` is wrong
    specs = tuple((n, dict(c, lowrank_free=(8, 128))) if n != which else (n, c) for n, c in specs)
    with pytest.raises(SystemExit, match=rf"{which} clearing was built on \(8, 523\)"):
        refuse_ineffective_lowrank_flags(specs, (8, 128), "auto")


def test_batching_mismatch_exits():
    with pytest.raises(SystemExit, match="--lu-batching asked for arrow"):
        refuse_ineffective_lowrank_flags(_specs(lu="sequential"), (8, 523), "arrow")


def test_auto_batching_is_not_compared():
    refuse_ineffective_lowrank_flags(_specs(lu="batched"), (8, 523), "auto")


def test_call_site_is_anchored():
    """The guard is called once, on the three clearings, right after the
    monitored-lines stamp; a copy with the call removed still parses and
    would refuse nothing, so the count is what this test rests on."""
    anchor = "    refuse_ineffective_lowrank_flags(\n"
    assert SRC.count(anchor) == 1
    tree = ast.parse(SRC)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call)
             and isinstance(c.func, ast.Name) and c.func.id == "refuse_ineffective_lowrank_flags"]
    assert len(calls) == 1
    names = [e.elts[0].value for e in calls[0].args[0].elts]
    assert names == ["step", "boundary", "eval"]
    # the mutation: without the call the parse still succeeds and the guard is gone
    broken = SRC.replace(anchor, "    _no_guard = (\n")
    fn_b = next(n for n in ast.walk(ast.parse(broken))
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    assert not [c for c in ast.walk(fn_b) if isinstance(c, ast.Call)
                and isinstance(c.func, ast.Name) and c.func.id == "refuse_ineffective_lowrank_flags"]
