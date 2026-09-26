"""L0: market 03's two drivers land the per-cell prices, not only their maxima.

The coverage matrix's "price level and distribution of market 03" row read
**cannot be given** for as long as neither
`run_rl_03.py` nor `run_eval_03.py` passed `arrays=` to `write_day`: the clearing
publishes `info["lmp"]` and `info["reserve_price"]` every period, and both
drivers reduced them to `reserve_price_max` and `reserve_price_at_cap_periods`
inside the rollout and dropped the rest.  Quantiles over bus-periods cannot be
recovered from two scalars, so the row could only be filled by a re-run of the
training that produced the products.

Two halves, because either alone would pass while the row stays unanswerable.

*Static*, over the call sites: the fix is one keyword at two places, and a
keyword is exactly the kind of edit a later refactor drops without any test going
red -- nothing under `tests/` runs these drivers (the coverage hole
`test_shared_helper_arity_l0.py` was written for).  So the two calls are parsed
and their `arrays=` keys read.

*Behavioural*, over `write_day` itself: that the container round-trips both
arrays under their own names, and that a name collision with a standard array
raises rather than overwriting.  Without this half the static check would still
pass if `arrays=` silently discarded what it was handed.

**What this does not catch**, so a green result is not read as more than it is:
that the arrays hold the right *values*, or that they are stacked in period
order.  Both are properties of the rollout, and this file runs no rollout.  The
first real product written after this landed was checked by hand instead --
an untrained market-03 day product, `lmp` (48, 29) and `reserve_price`
(48, 2), whose own maximum 250.000000 matches that product's
`reserve_price_max` 250.00000000006867 -- a cross-check the meta scalar
provides and the arrays alone would not.
"""
import ast
import json
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from benchmark import evaluation                                   # noqa: E402

#: The two drivers of market 03 and the price arrays each must land.  Named
#: rather than discovered: a driver that stopped calling `write_day` at all
#: would make a discovery-based check vacuous, and vacuous is the failure mode
#: this file exists to prevent.
DRIVERS = ("benchmark/run_rl_03.py", "benchmark/run_eval_03.py")
REQUIRED_KEYS = {"lmp", "reserve_price"}


def _write_day_calls(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        if name == "write_day":
            yield node


@pytest.mark.parametrize("driver", DRIVERS)
def test_both_03_drivers_land_both_price_arrays(driver):
    path = REPO / "tools" / driver
    calls = list(_write_day_calls(path))
    assert calls, f"{driver} has no write_day call; this check went vacuous"
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert "arrays" in kw, (
            f"{driver}:{call.lineno} calls write_day without arrays=; the "
            f"per-cell prices are dropped and the matrix's price row goes back "
            f"to 'cannot be given'")
        arrays = kw["arrays"]
        assert isinstance(arrays, ast.Call), (
            f"{driver}:{call.lineno} passes arrays= as {type(arrays).__name__}; "
            f"this check can only read a literal dict(...) call")
        got = {k.arg for k in arrays.keywords}
        assert got == REQUIRED_KEYS, (
            f"{driver}:{call.lineno} lands {sorted(got)}, wanted "
            f"{sorted(REQUIRED_KEYS)}")


def _run_point():
    return dict(cap_scale=0.6, ramp_scale=1.0, window="four-season", voll=1e4)


def test_write_day_round_trips_both_price_arrays(tmp_path):
    lmp = np.arange(48 * 29, dtype=np.float64).reshape(48, 29)
    res = np.linspace(0.0, 5.0, 48).reshape(48, 1)
    path = evaluation.write_day(
        tmp_path, "ippo", 4, "2024-01-05", system_cost_value=1.0,
        agent_profit=np.zeros(66), shed_mwh=np.zeros(48), production_cost=2.0,
        run_point=_run_point(), arrays=dict(lmp=lmp, reserve_price=res))
    blob = np.load(path, allow_pickle=True)
    assert set(blob.files) >= {"lmp", "reserve_price"}
    np.testing.assert_array_equal(blob["lmp"], lmp)
    np.testing.assert_array_equal(blob["reserve_price"], res)
    # the standard three survive beside them
    assert blob["production_cost"] == 2.0
    assert json.loads(str(blob["meta"]))["arm"] == "ippo"


def test_a_price_array_may_not_shadow_a_standard_one(tmp_path):
    with pytest.raises(ValueError, match="overwrite a standard array"):
        evaluation.write_day(
            tmp_path, "ippo", 4, "2024-01-05", system_cost_value=1.0,
            agent_profit=np.zeros(66), shed_mwh=np.zeros(48),
            production_cost=2.0, run_point=_run_point(),
            arrays=dict(shed_mwh=np.ones(48)))
