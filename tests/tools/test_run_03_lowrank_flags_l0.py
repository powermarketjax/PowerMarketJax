"""L0: the two ancillary drivers carry the low-rank route's flags to the
operator and refuse a run whose operator disagrees (2026-09-17).

Same shape as `test_lowrank_flags_effective_l0.py` for market 02: the guard
`evaluation.refuse_ineffective_lowrank_flags` is the one function (moved there
from `run_rl_02.py` unchanged; `run_rl_02` still exposes it under the same
name), and each driver's call site is anchored by source count so a driver
that drops the call or the keyword goes red rather than silently building the
operator on its default (the injection found on market 02).
"""
import ast
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
BENCH = REPO / "tools" / "benchmark"
sys.path.insert(0, str(BENCH))
from evaluation import refuse_ineffective_lowrank_flags  # noqa: E402
import run_rl_02  # noqa: E402

SRC = {name: (BENCH / f"{name}.py").read_text() for name in ("run_rl_03", "run_eval_03")}


def test_market_02_still_exposes_the_guard():
    assert run_rl_02.refuse_ineffective_lowrank_flags is refuse_ineffective_lowrank_flags


def test_one_clearing_is_enough_for_the_guard():
    spec = {"lowrank_free": (151, 815), "lu_batching": "arrow"}
    refuse_ineffective_lowrank_flags((("clearing", spec),), (151, 815), "auto")
    refuse_ineffective_lowrank_flags((("clearing", spec),), (151, 815), "arrow")
    with pytest.raises(SystemExit, match=r"clearing was built on \(151, 815\)"):
        refuse_ineffective_lowrank_flags((("clearing", spec),), (32, 815), "auto")
    with pytest.raises(SystemExit, match="--lu-batching asked for sequential"):
        refuse_ineffective_lowrank_flags((("clearing", spec),), (151, 815), "sequential")


def _main(src):
    return next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "main")


@pytest.mark.parametrize("name", sorted(SRC))
def test_flags_reach_the_operator_and_the_guard_is_called(name):
    src = SRC[name]
    for flag in ("--kkt", "--lowrank-free-units", "--lowrank-free-shed", "--lu-batching"):
        assert src.count(f'ap.add_argument("{flag}"') == 1, flag
    fn = _main(src)
    env_calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call)
                 and isinstance(c.func, ast.Name) and c.func.id == "make_ancillary_env"]
    assert len(env_calls) == 1
    kws = {k.arg: k.value for k in env_calls[0].keywords}
    assert isinstance(kws["kkt"], ast.Attribute) and kws["kkt"].attr == "kkt"
    assert isinstance(kws["lowrank_free"], ast.Name) and kws["lowrank_free"].id == "lowrank_free"
    assert isinstance(kws["lu_batching"], ast.Attribute) and kws["lu_batching"].attr == "lu_batching"
    guard = [c for c in ast.walk(fn) if isinstance(c, ast.Call)
             and isinstance(c.func, ast.Name) and c.func.id == "refuse_ineffective_lowrank_flags"]
    assert len(guard) == 1
    assert [e.elts[0].value for e in guard[0].args[0].elts] == ["clearing"]
    # the stamps travel with the route stamp into every product meta
    n_route = src.count("kkt_route=kkt_route,")
    assert n_route >= 1
    assert src.count("lowrank_free=lowrank_free_stamp, lu_batching=lu_batching_stamp,") == n_route
    # the mutation: without the call the parse still succeeds and the guard is gone
    anchor = "    refuse_ineffective_lowrank_flags(\n"
    assert src.count(anchor) == 1
    fn_b = _main(src.replace(anchor, "    _no_guard = (\n"))
    assert not [c for c in ast.walk(fn_b) if isinstance(c, ast.Call)
                and isinstance(c.func, ast.Name) and c.func.id == "refuse_ineffective_lowrank_flags"]


@pytest.mark.parametrize("name", sorted(SRC))
def test_requested_route_is_checked_against_the_stamp(name):
    src = SRC[name]
    assert src.count('if (args.kkt == "dense" and kkt_route != "dense") or (args.kkt == "lowrank" and not kkt_route.startswith("lowrank")):') == 1
