"""The two pure decisions of the economic-withholding arm: which level is the best
response, and what class a unit's curve falls in.

Both are checked with a case that must bite.  `best_index` has to return the
*lowest* level of a plateau (a unit priced out is indifferent along the whole
plateau, and the arm keeps the one nearest the market), and its tie tolerance has
to separate two levels 2e-6 apart while merging two 5e-7 apart -- the measured
cross-process GPU scatter is 1.4e-7 and the smallest real level-to-level gap was
1e-3 (`run_withholding_01.TIE_REL`).  `classify` has to put the 2026-09-02 unit 28
shape -- a few hundred dollars of profit at the best level against a 3.4e7 loss
when honest -- in `exit`, not `raise`, which is the misclassification the band
was introduced for.
"""
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "benchmark"))
from run_withholding_01 import EXIT_REL, TIE_REL, best_index, classify  # noqa: E402


def test_best_index_takes_lowest_level_of_the_plateau():
    own = [-1.0e5, -1.2e5, -0.4e5, 0.0, 0.0, 0.0]
    best, tied = best_index(own)
    assert best == 3 and tied == [3, 4, 5]


def test_best_index_tie_tolerance_bites_on_both_sides():
    top = 5.0e7
    # 5e-7 relative below the top: tied, so the lower level wins
    best, tied = best_index([0.0, top * (1 - 5e-7), top])
    assert best == 1 and tied == [1, 2]
    # 2e-6 relative below the top: distinct, the top wins
    best, tied = best_index([0.0, top * (1 - 2e-6), top])
    assert best == 2 and tied == [2]
    assert TIE_REL == 1e-6


def test_best_index_tolerance_floor_for_small_profits():
    # |top| < 1: the tolerance is absolute 1e-6, so 0.0 and -5e-7 tie
    best, tied = best_index([-5e-7, 0.0])
    assert best == 0 and tied == [0, 1]


def test_classify_four_classes():
    honest = -3.4e7
    assert classify(np.array([honest, -1e7, 0.0]), 0, honest) == "honest"
    assert classify(np.array([honest, -1e7, 0.0]), 2, honest) == "exit"
    assert classify(np.array([honest, -1e7, 0.0]), 1, honest) == "partial"
    assert classify(np.array([honest, 2e6, 0.0]), 1, honest) == "raise"


def test_classify_unit_28_shape_is_exit_not_raise():
    honest = -3.4181e7
    own = np.array([honest, -2.0e7, 760.0, 0.0])
    best, _ = best_index(own)
    assert best == 2                      # 760 beats 0, so it is the best level...
    assert classify(own, best, honest) == "exit"   # ...but earns nothing worth the name
    # and the band bites: 3% of the honest loss at the best level is `raise`
    own_r = np.array([honest, -2.0e7, 0.03 * abs(honest), 0.0])
    assert classify(own_r, 2, honest) == "raise"
    assert EXIT_REL == 0.02
