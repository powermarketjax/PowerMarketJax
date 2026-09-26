"""L0: `evaluation.open_day_start` reads the day back and the offset back.

The function it replaces (`open_day`) searched for a key whose draw
landed *somewhere inside* the wanted day and read the day back out of the state,
which is a real check that misses the thing market 03 was hurt by: the offset
inside the day.  Over twelve evaluation days, 265 of 576 half hours belonged to
the following day for the open-loop arms and 353 for the learning arm, so the
two were never scored on the same periods.

Both read-backs are exercised against a stub environment rather than against a
market, because what is under test is the checking, not any market's arithmetic:
a stub is the only way to produce the wrong state deliberately.  The markets'
own `reset_on_day` is tested in `tests/envs/ancillary/test_env_l0.py`.
"""
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "benchmark"))

import evaluation                                                 # noqa: E402

PERIODS_PER_DAY = 48


class _State:
    def __init__(self, cursor):
        self.cursor = cursor


def _stub(offset):
    """A `reset_on_day` that opens `offset` periods into the day it was asked for."""
    def reset_on_day(key, params, day):
        return None, _State(day * PERIODS_PER_DAY + offset)
    return reset_on_day


DAY_OF = lambda st: st.cursor // PERIODS_PER_DAY
PERIOD_OF = lambda st: st.cursor


def test_a_correct_open_returns_the_state_and_the_key():
    key, state = evaluation.open_day_start(
        _stub(0), None, 5, DAY_OF, PERIOD_OF, PERIODS_PER_DAY)
    assert PERIOD_OF(state) == 5 * PERIODS_PER_DAY
    assert key is not None


@pytest.mark.parametrize("offset", [1, 9, 47])
def test_an_offset_inside_the_right_day_raises(offset):
    """The case `open_day`'s key search accepted.

    `day_of_state` agrees -- the state really is filed under day 5 -- and the
    episode still runs `offset` periods past the day's end.
    """
    stub = _stub(offset)
    assert DAY_OF(stub(None, None, 5)[1]) == 5, (
        "this stub no longer reproduces the accepted-by-open_day case, so the "
        "test below would be catching a different mistake")
    with pytest.raises(AssertionError, match="periods into itself"):
        evaluation.open_day_start(stub, None, 5, DAY_OF, PERIOD_OF,
                                  PERIODS_PER_DAY)


def test_opening_the_wrong_day_raises():
    wrong = lambda key, params, day: (None, _State((day + 1) * PERIODS_PER_DAY))
    with pytest.raises(AssertionError, match="opened day"):
        evaluation.open_day_start(wrong, None, 5, DAY_OF, PERIOD_OF,
                                  PERIODS_PER_DAY)


def test_a_wrong_periods_per_day_is_not_silently_absorbed():
    """`periods_per_day` comes from the environment's `spec`, not from here.

    Passing the wrong one has to fail rather than shift the accepted offset,
    which is why the offset is checked against `day * periods_per_day` and not
    against `cursor % periods_per_day`: the second would pass for any day at
    any period length that divides the cursor.
    """
    with pytest.raises(AssertionError, match="periods into itself"):
        evaluation.open_day_start(_stub(0), None, 5, DAY_OF, PERIOD_OF, 24)
