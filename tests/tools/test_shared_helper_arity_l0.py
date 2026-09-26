"""Every call to a shared benchmark helper must match that helper's signature.

Written for one failure that happened on 2026-08-19 and was invisible to the
full suite.  `evaluation.system_cost` gained a fourth **required** parameter
(`other_shortfall_cost`) precisely so that adding it would be an act somebody
had to perform per market rather than a default that silently priced the
ancillary market's reserve shortfall at zero.  That design worked: it did force
the change.  What it did not do is force it *everywhere at once*, and the two
drivers that were missed -- `run_rl_01.py` and `run_eval_03.py` -- were left in
a state where they raise `TypeError` on the line that writes their first
product, after the training or the sweep has already run.

**The full suite was green through all of it.**  1703 tests passed on `main`
with both drivers broken, because nothing under `tests/` imports anything under
`tools/`, and the breakage is at call time rather than import time, so even an
import-smoke test would have missed it.  Market 01 found it by running a driver
with `--iterations 0` and watching it die before writing anything.

So this checks the one thing that is both cheap and exactly the failure mode:
for a small registry of shared helpers, every call site in `tools/` is bound
against the helper's live signature.  It is static -- no driver is imported and
no `main()` runs -- so it costs milliseconds and cannot be defeated by a driver
that is expensive to start.

**What it does not catch**, stated so a clean result is not read as more than it
is.  Argument *order* among same-typed positionals is invisible here; so is a
wrong value, a wrong unit, and any helper not in the registry.  A call that
splats (`f(*args)`) is skipped and reported, because binding it statically is
not possible -- if that ever becomes common the registry approach needs
rethinking rather than a looser check.
"""
import ast
import inspect
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from benchmark import evaluation                                   # noqa: E402
from powermarketjax.learning.ippo import IPPOConfig                # noqa: E402
from powermarketjax.learning.policy import SharedActorCritic       # noqa: E402

#: The helpers whose signatures other lines depend on.  Kept small on purpose:
#: the value of this check is that a name in here is one whose arity nobody may
#: get wrong, not that every function is covered.
REGISTRY = {
    "system_cost": evaluation.system_cost,
    "write_day": evaluation.write_day,
    "count_cells": evaluation.count_cells,
    "runtime_stamp": evaluation.runtime_stamp,
    # A dataclass, not a function, and that is the point: `IPPOConfig` declares
    # every field required with no default, so adding one is meant to be an act
    # somebody performs per call site rather than a value that appears silently.
    # On 2026-08-19 `weight_decay` was added that way and six construction sites
    # had to follow; three of them are under `tests/`, which is why the search
    # roots below are not only `tools/`.  The first version of this file scanned
    # `tools/` alone because the bug that prompted it lived there -- the same
    # scan-by-directory mistake that missed three `platform="cpu"` literals in
    # `tools/rt_scenario/` earlier the same day.
    "IPPOConfig": IPPOConfig,
    # Added 2026-08-19 after the registry above failed to catch a real break the
    # same hour: the commit that made `weight_decay` a required `IPPOConfig`
    # field also passed `weight_decay=0.0` to `SharedActorCritic`, which has no
    # such field and raises `TypeError`.  `tests/learning` and `tests/tools` were
    # both green, because nothing under `tests/` imports that driver -- the same
    # coverage hole that let two drivers sit broken on `main` earlier that day.
    # A constructor whose fields other lines pass by keyword belongs here for
    # exactly the reason `IPPOConfig` does.
    "SharedActorCritic": SharedActorCritic,
    # Added because each has already caused one incident, not for completeness.
    # `open_day` was called with the wrong argument on 2026-08-19 (a day index
    # where a restricted-fixture cursor was meant), which sent a whole 66-unit
    # sweep to the wrong days; `subset_position` is the other of the two shared
    # helpers every market's driver goes through.
    "open_day": evaluation.open_day,
    "subset_position": evaluation.subset_position,
}

#: Measured 2026-08-19 on `main` at `6a77b19`: **47** call sites across 220 files
#: under the roots below (32 before `open_day` and `subset_position` joined the
#: registry, which brought 15 more with them).  The floor is here because
#: `2 passed` looks the same whether the scan bound 47 sites or zero -- if the
#: drivers move, or an import fails and the walk finds nothing, the count
#: collapses and this fires rather than the suite going green on an empty check.
#: Raise it when the real count grows; do not lower it without saying which
#: sites went away.
MIN_CALL_SITES = 40

#: `tools/` holds the drivers; `tests/` holds construction sites for the shared
#: config that no driver touches.  Both are scanned because the failure mode is
#: "a required parameter grew and a call site did not follow", and that call site
#: can be anywhere.
SEARCH_ROOTS = (REPO / "tools", REPO / "tests")


def _driver_files():
    out = []
    for root in SEARCH_ROOTS:
        if root.is_dir():
            out.extend(sorted(f for f in root.rglob("*.py")
                              # this file constructs nothing; scanning it would
                              # only find the registry literal above
                              if f != pathlib.Path(__file__).resolve()))
    return out


def _called_name(node: ast.Call):
    """`f(...)` and `mod.f(...)` both count; anything else is not a registry call."""
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def test_registry_is_not_empty_and_names_resolve():
    """Guard the guard: an empty or stale registry would make this vacuous.

    A registry key that no longer names anything in `evaluation` would be
    silently dropped by the loop below, and the test would pass while checking
    nothing -- the failure mode this whole file exists to catch.
    """
    assert REGISTRY, "registry is empty; this test would check nothing"
    for name, fn in REGISTRY.items():
        assert callable(fn), f"{name!r} in the registry is not callable"
        assert inspect.signature(fn), f"{name!r} has no inspectable signature"


def test_every_call_site_binds_against_the_live_signature():
    files = _driver_files()
    # non-triviality: if the search directories move, this test would pass by
    # finding nothing to check
    assert len(files) >= 5, (
        f"only {len(files)} files found under {SEARCH_ROOTS}; the search "
        f"roots have probably moved and this test is checking almost nothing")

    checked, skipped, failures = 0, [], []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _called_name(node)
            if name not in REGISTRY:
                continue
            if any(isinstance(a, ast.Starred) for a in node.args) or \
               any(k.arg is None for k in node.keywords):
                skipped.append(f"{path.relative_to(REPO)}:{node.lineno} {name}(*splat)")
                continue
            sig = inspect.signature(REGISTRY[name])
            args = [inspect.Parameter.empty] * len(node.args)
            kwargs = {k.arg: inspect.Parameter.empty for k in node.keywords}
            try:
                sig.bind(*args, **kwargs)
                checked += 1
            except TypeError as exc:
                failures.append(
                    f"{path.relative_to(REPO)}:{node.lineno}  {name}(...)  {exc}")

    print(f"\nbound {checked} registry call sites across {len(files)} files")
    assert checked >= MIN_CALL_SITES, (
        f"only {checked} registry call sites were bound, floor is "
        f"{MIN_CALL_SITES} (measured 47 on 2026-08-19). Either the registry "
        f"names nothing the drivers call, the drivers moved, or the walk found "
        f"nothing -- and a green result on an empty scan is the failure this "
        f"floor exists to catch. Skipped: {skipped}")
    assert not failures, (
        "these call sites do not match the helper's current signature, and each "
        "one raises `TypeError` at run time on the line that writes a product, "
        "after the expensive part has already run:\n  " + "\n  ".join(failures)
        + (f"\n(skipped, not statically bindable: {skipped})" if skipped else ""))
