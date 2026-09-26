"""Market 02's economic-withholding arm: the two decisions that are not market 01's.

Market 01's driver could read a per-unit total as `reward_per_agent * n_days`
because its horizon is one step.  Market 02 runs 48 half-hours per day, so the
same expression is off by a factor of 48 -- and off *silently*, because the sign
and the ranking of the levels are unchanged and only the magnitude moves.  The
measured pair is the bite: on the 36 fitting days of the 365-day window the
honest arm's per-unit totals sum to -4.1348721749e+08 dollars, and the market-01
expression gives -8.6143170311e+06 (2026-09-03).

The tie criterion is checked on this market's own curves rather than on market
01's, for a reason particular to 02: its reward is float32
(`envs/real_time/env.py`'s state carries `profit_prev` as float32), so the
per-unit totals are quantised, and the tolerance has to sit above that quantum
and below the smallest real level-to-level gap.  All three were measured on the
2026-09-03 train36 sweep and they separate cleanly:

    float32 quantum   7.0e-8 relative (median unit), 1.18e-7 (worst unit)
    TIE_REL           1e-6            -- 8.5x the worst quantum
    smallest real gap 5.21e-6         -- 5.2x TIE_REL (unit 0, 281 dollars)

No unit in this market has a non-zero level-to-level gap below `TIE_REL`, so the
tolerance arbitrates nothing here; sweeping it over seven orders of magnitude
moves no unit's best level until 1e-3, where two move.  Market 03 is the
opposite case and its own test says so.
"""
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "benchmark"))
from run_withholding import (EXIT_REL, TIE_REL, _totals, alpha_layout,  # noqa: E402
                             best_index, classify)

#: unit 17 on the 2026-09-03 train36 sweep: twenty levels at exactly the same
#: float32 profit, the twenty-first (alpha = 2.0) below them
U17_PLATEAU = [20697280.3125] * 20 + [20600000.0]
#: unit 31, same sweep: consecutive levels 1.27e-4 apart in relative terms,
#: which is a hundred times the tie tolerance and must stay distinct
U31_HEAD = [1534955.0625, 1535149.75781, 1535344.98047]
#: unit 37, same sweep: the unit that comes nearest the `exit` band in market 02
U37 = dict(honest=-85098134.25, best=-77298448.5)


def test_totals_counts_the_horizon_not_only_the_days():
    row = {"reward_per_agent": [1.0, -2.0]}
    assert _totals(row, 36, 48).tolist() == [1728.0, -3456.0]
    # market 01's horizon, where the two conventions coincide
    assert _totals(row, 36, 1).tolist() == [36.0, -72.0]


def test_totals_bite_is_a_factor_of_the_horizon():
    # the measured pair from the module docstring, reproduced from one number
    market_01_expression = -8.6143170311e06
    assert np.isclose(market_01_expression * 48, -4.1348721749e08, rtol=1e-9)


def test_alpha_layout_of_a_markup_vector_is_the_vector_itself():
    base = np.array([1.0, 1.0, 1.0])
    cfg = {"four": (None, None, None, {"baseline_action": base})}
    n, truthful, action_of = alpha_layout(cfg)
    assert n == 3 and truthful.tolist() == [1.0, 1.0, 1.0]
    assert action_of([1.0, 1.7, 1.0]).tolist() == [1.0, 1.7, 1.0]
    # the truthful vector is a copy: deviating one unit must not edit the profile
    truthful[1] = 2.0
    assert base.tolist() == [1.0, 1.0, 1.0]


def test_tie_criterion_takes_the_lowest_level_of_the_measured_plateau():
    best, tied = best_index(U17_PLATEAU)
    assert best == 0 and len(tied) == 20


def test_tie_criterion_bites_on_both_sides_of_the_measured_plateau():
    top = U17_PLATEAU[0]
    # a dent 5e-7 relative below the plateau is still tied, so level 0 wins
    dented = [top * (1 - 5e-7)] + U17_PLATEAU[1:]
    assert best_index(dented)[0] == 0
    # a dent 2e-6 relative below it is a distinct, worse level: level 1 wins
    dented = [top * (1 - 2e-6)] + U17_PLATEAU[1:]
    assert best_index(dented)[0] == 1
    assert TIE_REL == 1e-6


def test_measured_consecutive_levels_stay_distinct():
    best, tied = best_index(U31_HEAD)
    assert best == 2 and tied == [2]
    gap = (U31_HEAD[1] - U31_HEAD[0]) / U31_HEAD[0]
    assert gap > 100 * TIE_REL          # 1.27e-4 against a 1e-6 tolerance


def test_tie_tolerance_sits_between_the_quantum_and_the_smallest_real_gap():
    """The three measured numbers of the module docstring, as one ordering.

    A tolerance below the float32 quantum would call two representations of the
    same number distinct levels; one above the smallest real gap would merge two
    levels that differ.  Both failures are silent -- they move a unit's best
    level and nothing else changes -- so the ordering is pinned here rather than
    left to the reader of the docstring.
    """
    float32_quantum_worst = 1.18e-7
    smallest_real_gap = 5.208e-6
    assert float32_quantum_worst < TIE_REL < smallest_real_gap
    assert TIE_REL / float32_quantum_worst > 8.0
    assert smallest_real_gap / TIE_REL > 5.0


def test_market_02_has_no_exit_and_the_margin_is_wide():
    own = np.array([U37["honest"], U37["best"]])
    assert classify(own, 1, U37["honest"]) == "partial"
    # how far it is from being called `exit`: 0.908 against a band of 0.02
    ratio = abs(U37["best"]) / abs(U37["honest"])
    assert ratio > 40 * EXIT_REL
    # and the band still bites: the same curve with the loss actually erased
    erased = np.array([U37["honest"], 0.01 * U37["honest"]])
    assert classify(erased, 1, U37["honest"]) == "exit"
