"""L0: market 03's two drivers name their case on the command line, and one case.

`run_eval_03.py` and `run_rl_03.py` declared `CASE = "29gb"` and called
`load_gb_demand_half_hourly()` unconditionally until 2026-09-12.  They are the
two drivers deliberately left out when nine others were moved onto
`half_hourly_from_meta` -- the reason recorded then: "`build_03` hard-codes
`run_eval_03.CASE`", i.e. they declared their case rather than reading it, so the
pairing rule had nothing to attach to.  With markets 02 and 03 wanted on
`case73rts` and `case813nem`, declaring it is no longer an option.

**Why a flag rather than moving `CASE`.**  Two reasons, and either alone decides
it.  The constant has produced results that are in effect and cited, so changing
its value is "change the value *and* re-run what it produced"; and
`run_rl_03.py` takes it through `from run_eval_03 import (..., CASE, ...)`, which
binds the value at import time, so rebinding it in the process would change
nothing.

**The measured "it bites" datum.**  The three realised half-hourly series are not
near each other, measured 2026-09-12 at the adopted run points:

    29gb     (648, 48)   16 612 .. 47 517 MW
    73rts    (366, 48)    2 500 ..  5 979 MW   (floor 2 500, all four netted)
    813nem   (370, 48)   11 500 .. 32 475 MW   (floor 11 500)

So a `73rts` network served British half-hours fails on shape before it fails on
values, and would still be wrong by a factor of seven if the day counts happened
to agree.  That is the failure this file's second half rules out.

**And the default is unchanged, verified by effect**: `run_eval_03.py`
at its default, run from `HEAD` and from the working tree over the same twelve
evaluation days, wrote 24 products whose every array is equal element for
element, with `meta` gaining exactly one key, `case: "29gb"`.
"""
import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

#: The two drivers of market 03.  Both now select the case at runtime.
DRIVERS = ["tools/benchmark/run_eval_03.py", "tools/benchmark/run_rl_03.py"]

#: The realised loaders that answer for one case only.  Same set as
#: `test_demand_case_pairing_l0.SINGLE_CASE_REALISED`, asserted below to still be
#: that set rather than a copy that can drift.
SINGLE_CASE_REALISED = {"load_gb_demand_half_hourly",
                        "load_rts_demand_half_hourly",
                        "load_nem_demand_half_hourly"}


def _tree(rel):
    return ast.parse((REPO / rel).read_text())


def _add_argument_calls(tree):
    out = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant)):
            out[node.args[0].value] = node
    return out


def _called_names(tree):
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            out.setdefault(node.func.id, []).append(node.lineno)
    return out


def test_the_single_case_loader_set_is_the_shared_one():
    """One answer to "which loaders answer for one case", not two."""
    import sys
    sys.path.insert(0, str(REPO / "tests" / "tools"))
    # the other list lives in a test that is not in every checkout
    shared = pytest.importorskip("test_demand_case_pairing_l0").SINGLE_CASE_REALISED
    assert SINGLE_CASE_REALISED == shared


@pytest.mark.parametrize("rel", DRIVERS)
def test_the_driver_takes_a_case_flag(rel):
    assert "--case" in _add_argument_calls(_tree(rel)), (
        f"{rel} cannot be pointed at a case, so markets 02 and 03 cannot be run "
        f"on 73rts or 813nem at all")


@pytest.mark.parametrize("rel", DRIVERS)
def test_the_case_flag_defaults_to_the_constant(rel):
    """The default is `CASE`, so every command line written before this runs on."""
    default = [kw.value for kw in _add_argument_calls(_tree(rel))["--case"].keywords
               if kw.arg == "default"]
    assert default and isinstance(default[0], ast.Name) and default[0].id == "CASE", (
        f"{rel}'s --case does not default to CASE; a second place would then "
        f"state which case is the default and the two can disagree")


@pytest.mark.parametrize("rel", DRIVERS)
def test_the_driver_names_no_single_case_realised_loader(rel):
    """The half `af5952e` could not reach while these two declared their case."""
    called = _called_names(_tree(rel))
    named = {n: sorted(called[n]) for n in SINGLE_CASE_REALISED & set(called)}
    assert not named, (
        f"{rel} selects its case at runtime but calls {named} (name: lines); the "
        f"realised series has to come from the fixture's own record, via "
        f"`half_hourly_from_meta`")


@pytest.mark.parametrize("rel", DRIVERS)
def test_the_driver_refuses_a_case_that_the_fixture_disagrees_with(rel):
    """`--case` is an assertion about the fixture, not an override of it.

    `half_hourly_from_meta` takes the realised series from `meta["case"]` while
    `load_case(args.case)` takes the network from the flag.  Letting the two
    differ is the original defect one level along: the run would be well-formed,
    would raise nothing, and would serve one country's demand to another
    country's network.
    """
    src = (REPO / rel).read_text()
    assert 'meta.get("case")) != args.case' in src, (
        f"{rel} does not check its --case against the fixture's own case")
    assert "two different cases" in src


@pytest.mark.parametrize("rel", DRIVERS)
def test_the_network_comes_from_the_flag(rel):
    """`load_case(args.case)`, not `load_case(CASE)`: the flag must reach it."""
    tree = _tree(rel)
    calls = [c for c in ast.walk(tree) if isinstance(c, ast.Call)
             and isinstance(c.func, ast.Name) and c.func.id == "load_case"]
    assert calls, f"{rel} does not call load_case"
    for c in calls:
        assert (c.args and isinstance(c.args[0], ast.Attribute)
                and c.args[0].attr == "case"
                and isinstance(c.args[0].value, ast.Name)
                and c.args[0].value.id == "args"), (
            f"{rel} builds its network from something other than args.case at "
            f"line {c.lineno}, so --case would be recorded and not applied")


@pytest.mark.parametrize("rel", DRIVERS)
def test_the_product_records_the_case_that_ran(rel):
    """A product that does not say which case it is cannot be filtered later."""
    src = (REPO / rel).read_text()
    assert "case=args.case" in src, (
        f"{rel} does not stamp `case` from the flag into its products")


def test_the_realised_series_are_far_apart():
    """The datum the checks above are worth having for, recomputed not quoted."""
    import numpy as np
    from powermarketjax.envs.day_ahead.demand import RTS_NETTED
    from powermarketjax.envs.real_time.demand import (
        load_gb_demand_half_hourly, load_nem_demand_half_hourly,
        load_rts_demand_half_hourly)
    gb, _ = load_gb_demand_half_hourly()
    rts, _ = load_rts_demand_half_hourly(netted=RTS_NETTED, floor_mw=2500.0)
    nem, _ = load_nem_demand_half_hourly(floor_mw=11500.0)
    # shape first: a mismatch cannot even be compared element for element
    assert gb.shape[0] != rts.shape[0] != nem.shape[0] != gb.shape[0]
    # and the levels, so the check would still bite if the day counts agreed
    assert float(rts.max()) < float(gb.min()) / 2.0, (
        "RTS net load should sit far below GB demand at these run points")
    assert float(nem.max()) < float(gb.max())
    assert float(gb.min()) / float(rts.max()) > 2.0
