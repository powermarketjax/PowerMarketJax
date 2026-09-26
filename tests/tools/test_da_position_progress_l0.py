"""L0: `--progress-every` adds progress lines and changes nothing when off.

**What this exists for.**  `harvest`'s per-day print fires only on the first
three days, days that shed, and the last one.  On a window where nothing sheds
that is silence from day 3 to the end -- and `da_position.py` writes its `.npz`
only at the end, so the out-dir is silent too.  On 2026-09-17 two runs of it sat
at 67 h and 27 h with no way to say which day they were on: the log said
nothing, the out-dir age said STALE (correctly, and uselessly), and RSS was
measured non-monotone.  One of them was nearly killed as dead, and "day 2/365"
in the other's log was read by a second line as progress when it only means
"past day 2".

So the load-bearing assertion here is the **default**: with `progress_every=0`
the printed set is exactly what it was before this keyword existed.  The second
assertion is that the keyword does something, without which a no-op
implementation would pass the first.
"""
import re
import sys
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "commitment"))
sys.path.insert(0, str(REPO / "tools" / "benchmark"))

N_DAYS = 6
#: 6 days on 29gb measured 19.4 s (2026-09-17, CPU, x64, 4 cores).  Two calls
#: take about 40 s, so this file does not suit the "run on every change" tier,
#: but it lives in tests/tools/.


@pytest.fixture(scope="module")
def inputs():
    """**x64 is turned on here, and must be turned back off here.**

    `jax.config` is process-global, and pytest runs a whole directory in one
    process in alphabetical order.  The first version of this fixture turned x64
    on and left it, so the tests that sort after it in the same directory and
    assert that x64 is **off** all went red (the 2026-09-17 comparison: the same
    subset in the same order on 4 cores gave 9 failed, four of them from this).
    The failure shows up **on someone else's test**: this file run alone is
    always green, and only a directory-order run sees it.  The x64 fixtures of
    `tests/envs/ancillary/test_clearing_l0.py` and
    `tests/envs/real_time/test_monitored_lines_l0.py` both save the old value and
    restore it; this one copies that shape.
    """
    import jax
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    from powermarketjax.case import load_case, scale_min_output
    from powermarketjax.envs.day_ahead import (demand_for_case, demand_pairing,
                                               load_commitment)
    from powermarketjax.envs.day_ahead.demand import T
    case = scale_min_output(load_case("29gb"), 1.0)
    fixture = load_commitment(mode="relax", case="29gb", n_periods=T, path=None,
                              p_min_scale=1.0)
    demand = demand_for_case("29gb", **demand_pairing("29gb"))
    yield case, fixture, demand
    jax.config.update("jax_enable_x64", previous)


def _days_printed(capsys, **kw):
    from da_position import harvest
    case, fixture, demand = kw.pop("inputs")
    harvest(case, fixture, demand, N_DAYS, n_segments=1, verbose=True, **kw)
    out = capsys.readouterr().out
    return sorted(int(m) for m in re.findall(r"^  day\s+(\d+)\s", out, re.M))


def test_default_prints_exactly_what_it_did_before(inputs, capsys):
    """`progress_every=0` must leave the printed set untouched.

    Days 0/1/2 by `d < 3`, day 5 by `d == n_days - 1`; no day sheds on this
    window, so the shed branch contributes nothing.  A fourth branch that fired
    at the default would change every existing log.
    """
    got = _days_printed(capsys, inputs=inputs)
    assert got == [0, 1, 2, 5], (
        f"default printed {got}, not the pre-existing set [0, 1, 2, 5]; the new "
        f"branch is firing when it is off")


def test_progress_every_actually_adds_a_line(inputs, capsys):
    """And the keyword has to do something, or the test above proves nothing.

    `progress_every=2` adds the days where ``d % 2 == 0``: 0, 2, 4.  Only 4 is
    new -- 0 and 2 were already in by `d < 3` -- which is why this asserts on
    the whole set rather than on a count.
    """
    got = _days_printed(capsys, inputs=inputs, progress_every=2)
    assert got == [0, 1, 2, 4, 5], (
        f"progress_every=2 printed {got}, expected [0, 1, 2, 4, 5] (day 4 is the "
        f"one the keyword adds; 0 and 2 were already printed by `d < 3`)")
