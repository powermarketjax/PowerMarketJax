"""Market 03's economic-withholding arm: the action layout and the tie criterion.

**The action layout is the one that fails silently.**  This market's action is
`(n_units, 3)`; only column 0 is a markup, and the two reserve columns are raw
pre-softplus values the baseline holds at -800, which softplus sends to exactly
zero.  A sweep that filled every column with the level would move those two by
801 units and report the result under the markup's name -- it would sweep a
different axis and nothing downstream would look wrong.  The arm therefore
carries the profile as an alpha *vector* and hands it to a market-specific
mapping at the last moment; these tests pin that mapping.

The tie criterion is checked on this market's own measured curve.  Market 03
makes it bite harder than market 01 did: unit 0's best level beats honest by 125
dollars out of 1.979633e+06, which is 6.3e-5 relative -- sixty times `TIE_REL`,
so genuinely distinct, but small enough that a tolerance chosen an order of
magnitude looser would erase it (2026-09-03).

`classify` is checked on unit 58, the unit that comes nearest the `exit` band in
this market: it avoids 93.2% of its honest loss and still lands at 6.80% of it,
which is 3.4 times the band.  That margin is the reason market 03 reports no
`exit` unit at all, and it is a measurement rather than a choice of band.
"""
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "benchmark"))
from run_withholding import (EXIT_REL, TIE_REL, _totals, alpha_layout,  # noqa: E402
                             best_index, classify)

#: the pre-softplus reserve sentinel `run_eval_03.py` and `build_03` both use
SENTINEL = -800.0
#: unit 0's first four levels (alpha 1.00 to 1.15) on the 2026-09-03 train36
#: sweep.  Level 3 sits above level 0 too, so a test that wants the plateau to
#: swallow the peak has to take the head of the curve and say so.
U0 = [1979633.40073, 1979633.40073, 1979758.43235, 1979723.09864]
#: unit 58, same sweep: nearest this market has to an `exit`
U58 = dict(honest=-5594839.96503, best=-380346.878054)


def _cfg(n=4):
    base = np.full((n, 3), SENTINEL)
    base[:, 0] = 1.0
    return {"four": (None, None, None, {"baseline_action": base})}, base


def test_alpha_layout_reads_the_markup_out_of_column_zero():
    cfg, _ = _cfg()
    n, truthful, _ = alpha_layout(cfg)
    assert n == 4 and truthful.tolist() == [1.0, 1.0, 1.0, 1.0]


def test_alpha_layout_writes_only_the_markup_column():
    cfg, base = _cfg()
    _n, _t, action_of = alpha_layout(cfg)
    a = action_of([1.0, 1.85, 1.0, 1.0])
    assert a.shape == (4, 3)
    assert a[:, 0].tolist() == [1.0, 1.85, 1.0, 1.0]
    # the bite: the reserve columns are untouched.  Filling them with the level
    # instead would move each by 801.85, and this is what says it did not happen
    assert np.all(a[:, 1:] == SENTINEL)
    assert abs(1.85 - SENTINEL) > 800.0
    # and the baseline itself is not edited, so the next level starts from it
    assert np.all(base[:, 1:] == SENTINEL) and base[:, 0].tolist() == [1.0] * 4


def test_alpha_layout_returns_an_independent_array_each_call():
    cfg, _ = _cfg()
    _n, _t, action_of = alpha_layout(cfg)
    first = action_of([1.0, 1.5, 1.0, 1.0])
    first[0, 1] = 0.0
    second = action_of([1.0, 1.5, 1.0, 1.0])
    assert second[0, 1] == SENTINEL


def test_tie_criterion_keeps_the_measured_125_dollar_gain():
    best, tied = best_index(U0)
    assert best == 2 and tied == [2]
    gain = (U0[2] - U0[0]) / U0[0]
    assert gain > 60 * TIE_REL           # 6.3e-5 against a 1e-6 tolerance


def test_tie_criterion_bites_the_other_way_on_the_same_curve():
    # the head of the same curve, alpha 1.00 to 1.10, with that gain shrunk to
    # 5e-7 relative: now the three levels are one plateau and the *lowest* wins,
    # which here is the honest level
    top = U0[0]
    shrunk = [top, top, top * (1 + 5e-7)]
    best, tied = best_index(shrunk)
    assert best == 0 and tied == [0, 1, 2]
    # unshrunk, the same three levels do not tie -- so the merge above is the
    # tolerance acting and not the shape of the head
    assert best_index(U0[:3]) == (2, [2])


def test_measured_nearest_unit_to_the_exit_band_is_partial():
    own = np.array([U58["honest"], U58["best"]])
    assert classify(own, 1, U58["honest"]) == "partial"
    ratio = abs(U58["best"]) / abs(U58["honest"])
    assert 3.0 * EXIT_REL < ratio < 4.0 * EXIT_REL       # 6.80% against 2%
    # the band would have to be widened past that ratio to call it `exit`
    assert classify(own, 1, U58["honest"], exit_rel=0.07) == "exit"


def test_totals_counts_the_forty_eight_periods():
    row = {"reward_per_agent": [-1.0, 3.0]}
    assert _totals(row, 36, 48).tolist() == [-1728.0, 5184.0]
