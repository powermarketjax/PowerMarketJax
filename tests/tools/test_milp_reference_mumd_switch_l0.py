"""L0/L1: `--exact-mumd` selects which program `milp_reference.py` calls exact.

The reference this script produces is the denominator of every "the three-step
clearing is X% from the optimum" statement in market 01.  Until 2026-08-29 the
exact side always carried the (MU)/(MD) rows while step 1' -- the relaxation the
environment actually solves -- always dropped them, so the difference priced two
things at once and only one of them was rounding.  The model this environment describes has no
(MU)/(MD) at all; on that model the reference has to drop the rows too, and this
file is what says the switch that does it is wired to the solve rather than only
to the metadata.

Three properties, and the third is the one with a number behind it:

    structure    `drop_mumd` removes exactly the rows that touch no dispatch and
                 no shed column, which at T periods and n units is 2*n*T of them
    wiring       `one_day` hands the solver the smaller program under `drop` and
                 the larger one under `keep` -- checked by capture, not by
                 solving, so it costs a build
    discrimination  dropping them is strictly cheaper on a day where it bites

**The measured "it bites" datum, as required before a check counts.**  On
`case29gb`, T=4, `cap 0.6 / ramp 1.00`, markup 1, day 12 solved from the boundary
the step 1' chain reaches after days 5..11: `keep` gives 3.809768625e6 and `drop`
3.797927629e6, i.e. -3.108e-3 relative.  The same day solved from a *cold start*
gives -5.8e-10, which is zero at this tolerance -- so the discrimination lives in
the chained history, not in the day, and a version of this test written against
`first_boundary` would pass against a broken switch.  That is why the fixture
below pays for the chain.

Kept at T=4 because the same pair at T=24 is two solves of several minutes each;
the commitment-fixture rebuild test draws the same line for the
same reason.  The adopted run point is T=24 and is measured offline, not here.
"""
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "commitment"))

import milp_reference as mr                                       # noqa: E402
import precommit                                                  # noqa: E402

PERIODS = 4
KW = dict(K=1, cap_scale=0.6, ramp_scale=1.0)
#: The day the discrimination lives on, and the days of chain it needs.
DAY = 12
CHAIN_FROM = 5


@pytest.fixture(scope="module")
def case():
    from powermarketjax.case import load_case
    return load_case("29gb")


@pytest.fixture(scope="module")
def actual():
    from powermarketjax.envs.day_ahead.demand import load_gb_demand
    return load_gb_demand()[1][:, :PERIODS].astype(np.float64)


@pytest.fixture(scope="module")
def chained_boundary(case, actual):
    """The boundary the step 1' chain reaches at the end of day `DAY - 1`."""
    rec = mr.chain(case, actual, np.arange(CHAIN_FROM, DAY + 1), KW, drop=True)
    assert rec[-1]["day"] == DAY
    return rec[-1]["boundary"]


@pytest.fixture(scope="module")
def program(case, actual, chained_boundary):
    return precommit.build(case, actual[DAY], **chained_boundary, **KW)


def test_drop_mumd_removes_exactly_the_min_up_down_rows(case, program):
    """Two rows per unit-period, and every survivor touches a `g` or an `s`."""
    dropped = mr.drop_mumd(program)
    n_rows = program["A_ub"].shape[0] - dropped["A_ub"].shape[0]
    n_units = len(case.unit_p_min)
    assert n_rows == 2 * n_units * PERIODS, (
        f"expected (MU) and (MD) for each of {n_units} units in each of "
        f"{PERIODS} periods; {n_rows} rows went")

    A = program["A_ub"].tocsr()
    n_gs = program["off_u"]
    touches = np.diff(A[:, :n_gs].tocsr().indptr) > 0
    assert touches.sum() == dropped["A_ub"].shape[0]
    # and the selection is not vacuous in either direction
    assert touches.any() and not touches.all()
    # the right-hand side is dropped with its row, not left behind
    assert len(dropped["b_ub"]) == dropped["A_ub"].shape[0]
    # nothing else moves: the equalities, the bounds and the objective are the
    # same object, so the cross-day history in `lo`/`hi` survives the drop
    for k in ("A_eq", "b_eq", "lo", "hi", "c", "integral_slice", "off_u"):
        assert dropped[k] is program[k], k


def test_the_switch_reaches_the_solver(case, actual, chained_boundary,
                                       program, monkeypatch):
    """`one_day` solves the smaller program under `drop`.  By capture, not by
    solving: what is under test is the wiring, and a solve would cost minutes at
    the horizon anyone cares about."""
    seen = {}

    class Stop(Exception):
        pass

    def spy(lp, *a, **kw):
        seen["rows"] = lp["A_ub"].shape[0]
        raise Stop

    monkeypatch.setattr(mr, "solve_milp", spy)
    monkeypatch.setattr(mr, "_CASE", case)
    monkeypatch.setattr(mr, "_ACTUAL", actual)

    rows = {}
    for arm in ("keep", "drop"):
        with pytest.raises(Stop):
            mr.one_day((DAY, chained_boundary, KW, mr.MIP_REL_GAP, 60.0, arm))
        rows[arm] = seen["rows"]

    assert rows["keep"] == program["A_ub"].shape[0]
    assert rows["drop"] == mr.drop_mumd(program)["A_ub"].shape[0]
    assert rows["keep"] - rows["drop"] == 2 * len(case.unit_p_min) * PERIODS


def test_dropping_the_rows_is_strictly_cheaper_where_it_bites(program):
    """The direction is forced and the size is measured.

    Forced: dropping rows enlarges the feasible set, so the exact optimum can only
    fall.  A `drop` objective above `keep` beyond the MIP tolerance is a bug in the
    selection, not a finding.  Measured: -3.108e-3 relative on this instance, which
    is what makes this an assertion rather than a tautology -- the same comparison
    at a cold-start boundary comes out at -5.8e-10 and would pass against a switch
    wired to nothing.
    """
    keep = mr.solve_milp(program, mr.MIP_REL_GAP, 300.0)
    drop = mr.solve_milp(mr.drop_mumd(program), mr.MIP_REL_GAP, 300.0)
    assert keep["status"] == 0 and drop["status"] == 0

    rel = drop["obj"] / keep["obj"] - 1
    assert rel <= 2 * mr.MIP_REL_GAP, f"drop above keep by {rel:+.3e}"
    assert rel < -1e-3, (
        f"the instance stopped discriminating: {rel:+.3e}, measured -3.108e-3 on "
        f"2026-08-29.  A switch wired to nothing also gives ~0 here")


def test_the_derived_name_cannot_land_on_a_committed_product():
    """Every input that changes the meaning changes the name, and none of them
    reaches the committed file's name."""
    base = dict(case="29gb", periods=24, cap_scale=0.6, ramp_scale=1.0,
                starts=(5, 103, 187, 285), days=15, exact_mumd="drop")
    moved = [dict(base, **d) for d in
             (dict(cap_scale=0.4), dict(ramp_scale=0.25), dict(days=14),
              dict(starts=(0,)), dict(exact_mumd="keep"), dict(periods=4),
              dict(case="29gb_x"), dict(p_min_scale=0.8))]
    names = [mr.derived_name(**base)] + [mr.derived_name(**m) for m in moved]
    assert len(set(names)) == len(names), "two run points share a path"
    #: The default has to leave the name alone, or every product written before
    #: the scale existed would be looked for under a name it was never given.
    assert mr.derived_name(**base, p_min_scale=mr.P_MIN_SCALE) == names[0]
    for n in names:
        assert n != mr.COMMITTED
    assert (mr.FIXTURE_DIR / mr.COMMITTED).exists(), (
        "the name this check is written against is gone; if the committed "
        "product moved, this file has to be told where")
