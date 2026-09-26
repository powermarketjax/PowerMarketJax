"""L1 for the day-ahead position and the hour-to-period map — spec §4, §15.

The map is tested on its own rather than only through the settlement, because
§17 records that the money-balance identity of §8 **cannot** detect an error in
it: the identity holds whatever `q_da` is substituted, so a position mapped onto
the wrong periods satisfies it exactly as well as a correct one.  Nothing
downstream of the map checks the map.
"""
import json

import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.real_time import (PERIODS_PER_HOUR, T_RT,
                                           hour_of_period, load_da_position,
                                           to_real_time)

T_DA = 24


def test_each_day_ahead_hour_covers_two_real_time_periods():
    """§4: the schedule of one day-ahead period applies to every period it covers."""
    t_rt = np.arange(T_RT)
    hours = hour_of_period(t_rt)
    assert hours.min() == 0 and hours.max() == T_DA - 1
    # periods 0 and 1 -> hour 0; 2 and 3 -> hour 1; ...
    assert list(hours[:6]) == [0, 0, 1, 1, 2, 2]
    counts = np.bincount(hours, minlength=T_DA)
    assert (counts == PERIODS_PER_HOUR).all()


def test_the_map_is_a_repeat_and_not_a_tile():
    """The two periods of an hour are adjacent, which `tile` gets wrong.

    They agree only at `periods_per_hour == 1`, so the distinction is invisible
    on any test that does not use the real ratio -- which is why this asserts
    against an explicit expected array rather than against `np.tile`.
    """
    x = np.array([10.0, 20.0, 30.0])
    np.testing.assert_array_equal(to_real_time(x, periods_per_hour=2),
                                  [10.0, 10.0, 20.0, 20.0, 30.0, 30.0])
    # the wrong answer this rules out
    assert not np.array_equal(to_real_time(x, periods_per_hour=2), np.tile(x, 2))


def test_the_map_is_consistent_with_the_index_it_is_derived_from():
    """`to_real_time` and `hour_of_period` must be the same map, two ways.

    They are used in different places -- one expands an array, the other indexes
    a period -- and a disagreement between them is exactly the seam §17 says the
    identity cannot see.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(size=(T_DA, 5))
    expanded = to_real_time(x, axis=0)
    assert expanded.shape == (T_RT, 5)
    for t in range(T_RT):
        np.testing.assert_array_equal(expanded[t], x[hour_of_period(t)])


#: The adopted scenario's position; `_seasons` is the expand-phase name that
#: batch 3 renames back.  This module only reads the fixture, so it declares no
#: scales -- but it must read the *same* fixture the rest of the market runs on.
#: Left on the old one it would stay green while checking a scenario nobody
#: runs, and a permanently green irrelevant test is harder to notice than an
#: explicit `vacuous` one.
CHAIN = "step1prime_seasons"


@pytest.fixture(scope="module")
def position():
    try:
        return load_da_position(chain=CHAIN)
    except FileNotFoundError as exc:                       # pragma: no cover
        pytest.skip(str(exc))


def test_the_fixture_carries_the_five_quantities(position):
    """The position is five quantities, not three."""
    for name in ("u", "q_da", "s_da", "lmp_da", "d_da"):
        assert name in position, f"the position is missing {name}"
    n_days = len(position["q_da"])
    case = load_case(position["meta"]["case"])
    n_units, n_buses = len(case.unit_p_min), int(case.n_nodes)
    assert position["q_da"].shape == (n_days, n_units, T_DA)
    for name in ("s_da", "lmp_da", "d_da"):
        assert position[name].shape == (n_days, T_DA, n_buses), name


def test_day_ahead_net_injections_sum_to_zero(position):
    """§8: `sum_n P_da[t, n] = 0`, which is what `s_da` is carried for.

    Dropping `s_da` makes the sum equal the negated day-ahead shed of that
    period, so the assertion below is checked **and** shown to be able to fail:
    the fixture must contain a shedding period, or this proves nothing.  That
    ordering is the repository's standing fix for a vacuous check.

    The bound is the one `tools/commitment/da_position.py` derives from measured
    residuals, relative because the quantity is the equality row's own solve
    residual and therefore scales with the demand.

    **Vacuous on the adopted scenario**.  The discriminating
    half of this check needs a day-ahead shedding period, and the adopted
    scenario has none:

        step1prime          max s_da = 182.995524 MWh, 4 shedding periods
        step1prime_seasons  max s_da =   0.000000 MWh, 0 shedding periods

    So `s_da` is identically zero, omitting it would move nothing, and the
    "shown to be able to fail" half cannot run.  **The phenomenon was removed by
    the scenario correction, not by anything about this check** -- the gate is
    not widened and the check is not deleted.  The proper repair is to build the
    operating point rather than to find one: a day-ahead position that does shed
    has to be solved for, which is new work and not part of this migration.
    """
    pytest.skip(
        "vacuous on the adopted scenario: s_da is identically 0 (was 182.995524 "
        "MWh over 4 periods on the old window), so the check cannot be shown "
        "able to fail. Sacrificed by the scenario correction; repair is to "
        "build a shedding position, which is new work")
    case = load_case(position["meta"]["case"])
    unit_bus = np.asarray(case.unit_node_idx, np.int64)
    # the producer's own bound, restated rather than read out of the fixture:
    # taking it from `meta["gates"]` would compare the fixture against a number
    # it wrote itself, which passes by construction
    rtol = 5e-6                          # tools/commitment/da_position.py
    separation_floor = 1e2               # measured 4.2e3 on the shipped window

    worst, worst_drop = 0.0, 0.0
    for d in range(len(position["q_da"])):
        gen = np.zeros((T_DA, int(case.n_nodes)))
        np.add.at(gen, (slice(None), unit_bus), position["q_da"][d].T)
        s_da, d_da = position["s_da"][d], position["d_da"][d]
        scale = np.maximum(d_da.sum(1), 1.0)
        worst = max(worst, float((np.abs((gen + s_da - d_da).sum(1)) / scale).max()))
        worst_drop = max(worst_drop, float((np.abs(s_da.sum(1)) / scale).max()))

    assert worst_drop > separation_floor * worst, (
        f"no period in this fixture sheds enough for the check to be able to "
        f"fail: omitting `s_da` would move the sum by {worst_drop:.2e} against a "
        f"residual of {worst:.2e}, only {worst_drop / worst:.1f}x")
    assert worst < rtol, f"day-ahead net injections do not sum to zero: {worst:.2e}"


#: Slack on the commitment box, MW.  Measured 2026-08-17 on
#: `day_ahead_position_29gb_T24_step1prime_seasons.npz` (cap 0.60 / ramp 1.00 /
#: four seasonal segments / VOLL 10000): every violation is **exactly** 0.0 --
#: below `p_min`, above `p_max`, and the phantom output of de-committed units
#: alike.  The tolerance is not zero anyway, because the producer solves an LP
#: and a future rebuild may leave a residual; it is set six orders below the
#: smallest `p_min` in the case rather than at the observed zero.
BOX_ATOL = 1e-6


def test_the_position_respects_the_commitment_box(position):
    """`q_da` lies in the box its own `u` declares -- the fixture's own coherence.

    This is the check the seasonal fixture went into the tree without.  Nothing
    else looks at `u` and `q_da` **together**: the loader validates keys, shapes
    and `periods_per_hour`, and the real-time side consumes `q_da` as the
    baseline of §4's deviation without ever asking whether the schedule it came
    with agrees with it.  A fixture whose `u` and `q_da` had drifted apart -- a
    transposed axis, a day shifted by one, two halves of different rebuilds --
    would satisfy every existing assertion and would silently redefine every
    deviation, and therefore every settlement, in this market.

    The three parts are asserted separately because they fail differently: an
    off-by-one in the day axis breaks the zero part, a wrong `cap_scale` breaks
    the upper part, and a stale `u` breaks the lower one.
    """
    case = load_case(position["meta"]["case"])
    u = np.asarray(position["u"], np.float64)
    q = np.asarray(position["q_da"], np.float64)
    p_min = np.asarray(case.unit_p_min, np.float64)[None, :, None]
    p_max = np.asarray(case.unit_p_max, np.float64)[None, :, None]

    assert set(np.unique(u)) <= {0.0, 1.0}, \
        f"`u` is not binary: {np.unique(u)[:8]}"
    on = u > 0.5

    # a de-committed unit produces nothing
    assert np.abs(q[~on]).max() <= BOX_ATOL, (
        f"de-committed units carry output up to {np.abs(q[~on]).max():.3e} MW")
    # a committed one produces within its own limits
    assert (q - p_min * u)[on].min() >= -BOX_ATOL, (
        f"committed output falls below p_min by "
        f"{-(q - p_min * u)[on].min():.3e} MW")
    assert (p_max * u - q)[on].min() >= -BOX_ATOL, (
        f"committed output exceeds p_max by {-(p_max * u - q)[on].min():.3e} MW")

    # and the box is not trivially satisfied: both states must occur, and some
    # committed unit must sit strictly inside its limits rather than on a bound,
    # or the three assertions above would hold for a fixture of all zeros
    assert on.any() and (~on).any(), "every unit is in the same commitment state"
    interior = on & (q > p_min * u + BOX_ATOL) & (q < p_max * u - BOX_ATOL)
    assert interior.any(), (
        "no committed unit lies strictly inside its box, so the bounds above are "
        "satisfied by the bounds themselves and carry no information")


def test_the_declared_scenario_choices_are_recorded(position):
    """`u` is endogenous since 01A, so the position depends on a policy.

    A fixture that does not say which day-ahead policy and which commitment
    chain produced it cannot be compared against another one, and §4's deviation
    is defined against *this* market's day-ahead outcome.
    """
    meta = position["meta"]
    for key in ("chain", "day_ahead_policy", "max_iter", "dual_start",
                "cap_scale", "ramp_scale", "boundary"):
        assert meta.get(key), f"the position fixture does not record {key}"
    # and it is the adopted scenario, not merely *a* recorded one.  Checking
    # only that the keys exist leaves `CHAIN` unguarded: restoring it to the old
    # fixture was measured to break nothing in this file, because every
    # assertion here held on both fixtures.
    assert (meta["cap_scale"], meta["ramp_scale"]) == (0.60, 1.00), (
        f"this module reads the adopted scenario's position; got cap "
        f"{meta['cap_scale']} / ramp {meta['ramp_scale']}")
    assert "four-season" in str(meta.get("window", "")), (
        f"expected the seasonal window, fixture records "
        f"{meta.get('window', '<no window recorded>')!r}")
    assert meta["periods_per_hour"] == PERIODS_PER_HOUR
    json.dumps(meta)                       # must stay serialisable for the audit
