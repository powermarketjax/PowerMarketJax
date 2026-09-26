"""`load_commitment` refuses a fixture whose `p_min_scale` the caller did not name.

That factor is applied to the *case*, not passed to the operators, so `make_env`
never sees it and the drivers that build their case from ``load_case(meta["case"])``
drop it without any error.  The failure is a wrong market rather than an exception:
measured 2026-09-13 on `case73rts`, a commitment solved at 0.80x the minimum output
and cleared against the registered case gave 34 of 36 days not converging, negative
shedding (over-generation) and ``profit -inf``.

Both directions are held here, because a guard that only refuses is indistinguishable
from a guard wired to refuse everything: the pre-2026-09-05 fixtures record no field
and must keep loading exactly as before.
"""
import pytest

from powermarketjax.envs.day_ahead.commitment import (
    P_MIN_SCALE_IMPLIED_PRIOR, load_commitment)

#: The two cases are not interchangeable here: `29gb` records no `p_min_scale` at
#: all (its fixtures predate the field) and `73rts` declares 0.80, so together they
#: cover both the implied-prior branch and the declared branch.
IMPLIED = "29gb"
DECLARED, DECLARED_VALUE = "73rts", 0.80


def test_a_fixture_without_the_field_loads_as_it_did_before():
    """The absent field reads as 1.0, so every fixture written before it existed
    keeps loading with no argument -- this is the half that proves the guard is not
    simply refusing everything."""
    fx = load_commitment(case=IMPLIED, n_periods=24)
    assert "meta" in fx
    assert fx["meta"].get("p_min_scale", P_MIN_SCALE_IMPLIED_PRIOR) == P_MIN_SCALE_IMPLIED_PRIOR
    # and naming the prior explicitly is the same call
    assert load_commitment(case=IMPLIED, n_periods=24,
                           p_min_scale=P_MIN_SCALE_IMPLIED_PRIOR)["meta"] == fx["meta"]


def test_naming_a_value_a_fixture_does_not_declare_is_refused():
    with pytest.raises(ValueError, match="p_min_scale"):
        load_commitment(case=IMPLIED, n_periods=24, p_min_scale=DECLARED_VALUE)


def test_a_declared_fixture_is_refused_when_the_caller_names_nothing():
    """This is the silent path the guard exists for: before it, this call returned a
    commitment built at 0.80 and the caller went on to build the registered case."""
    with pytest.raises(ValueError, match="did not name one"):
        load_commitment(case=DECLARED, n_periods=24)


def test_a_declared_fixture_loads_when_the_caller_names_the_same_value():
    fx = load_commitment(case=DECLARED, n_periods=24, p_min_scale=DECLARED_VALUE)
    assert fx["meta"]["p_min_scale"] == DECLARED_VALUE


def test_a_declared_fixture_is_refused_when_the_caller_names_a_different_value():
    with pytest.raises(ValueError, match="different case than it was solved for"):
        load_commitment(case=DECLARED, n_periods=24, p_min_scale=1.0)
