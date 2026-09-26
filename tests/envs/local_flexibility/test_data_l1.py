"""L1 for the three exogenous series of the local flexibility market (item 39).

All three have a dataset since 2026-08-15, so the second of the two jobs below
has changed shape; the first is unaltered.

**What is present has to be usable.**  The zone-substation demand series is
loaded on a regular 15-minute grid that starts at a local midnight, since the
environment counts periods by row and its calendar encoding would otherwise not
mean time of day.  The dataset carries both daylight-saving transitions
unresolved -- one timestamp bearing five rows in October, on 174 of the 175
substations, and a 75-minute hole in April, on all 175 -- so the loader has to
reject a window spanning one instead of silently dropping rows, and that
rejection is tested rather than assumed.

**What was absent had to fail loudly, and this file was the tripwire.**  The
photovoltaic and price loaders were required to raise `MissingSeries`, so that
the day the datasets landed those two tests would fail and that failure would
be the signal to finish item 39 rather than a regression to paper over.  They
fired on 2026-08-15 and are discharged: what stood there now asserts the
coverage, the price against its cap and its floor (which are not symmetric:
one is reached bitwise, the other only approached) and the night-time zero of
the photovoltaic output, each against the measured number rather than against
"not empty".  One thing had to be added rather than converted -- the
`MissingSeries` branch itself lost its only exercise when every call on the
real registry began to succeed, so a test now reaches it through an empty
registry.  Turning a completed item into a fresh hole is the failure mode that
kept it.

**The path between the two is tested on a fixture dataset.**  A parquet and a
manifest written into a temporary directory exercise the resolution, the
region filter, the resampling and the coverage check, which is what says the
loaders will work when real datasets arrive rather than only that they raise.
That fixture is a test of the loader's mechanics and **not** a stand-in for the
missing series: no market result may be produced on it, since synthetic data must not stand
in for real data.
"""
import json

import numpy as np
import pandas as pd
import pytest

from powermarketjax.envs.local_flexibility import (MissingSeries,
                                                   load_energy_price,
                                                   load_pv_series,
                                                   load_substation_demand)
from powermarketjax.envs.local_flexibility.data import (AEDT_WINDOW,
                                                        AEST_WINDOW,
                                                        DEMAND_TIMEZONE,
                                                        PERIOD_HOURS,
                                                        load_elcom_tariff,
                                                        load_swiss_flex_series)

#: One of the 173 substations complete over the summer window, with a mean load
#: of 11.9 MW -- the same order as the 14.9 MW registered on `case533mt_hi`,
#: which is what keeps the adopted kappa meaningful (§14).
REGION = "Mona Vale 33_11kV"
SUMMER_PERIODS = 17_376


@pytest.fixture(scope="module")
def summer():
    return load_substation_demand(REGION, start=AEDT_WINDOW[0],
                                  end=AEDT_WINDOW[1])


def test_summer_window_is_a_regular_grid_from_local_midnight(summer):
    load_mw, index = summer
    assert load_mw.shape == (SUMMER_PERIODS,) and len(index) == SUMMER_PERIODS
    assert load_mw.dtype == np.float32
    assert np.isfinite(load_mw).all()

    steps = np.unique(np.diff(index.asi8))
    assert steps.size == 1
    assert pd.Timedelta(int(steps[0])) == pd.Timedelta(hours=PERIOD_HOURS)

    local = index.tz_convert(DEMAND_TIMEZONE)
    assert (local[0].hour, local[0].minute) == (0, 0)
    # one offset throughout, which is what makes the row index a time of day
    assert len(set(local.strftime("%z"))) == 1


def test_winter_window_is_the_other_clean_one():
    load_mw, index = load_substation_demand(REGION, start=AEST_WINDOW[0],
                                            end=AEST_WINDOW[1])
    assert len(index) == 15_176 and load_mw.shape == index.shape
    local = index.tz_convert(DEMAND_TIMEZONE)
    assert (local[0].hour, local[0].minute) == (0, 0)


def test_window_spanning_a_transition_is_rejected():
    """The October transition left four rows on one timestamp on every
    substation, so no regular grid exists across it (module docstring)."""
    with pytest.raises(ValueError, match="repeated timestamps"):
        load_substation_demand(REGION)


def test_unknown_region_lists_what_the_dataset_carries():
    with pytest.raises(ValueError, match="is not in dataset"):
        load_substation_demand("Nowhere 11kV")


def _window_grid(window):
    return pd.date_range(*[pd.Timestamp(t, tz="UTC") for t in window],
                         freq="15min")


def test_both_series_cover_both_windows_with_no_hole_after_resampling():
    """The tripwires of the module docstring, discharged 2026-08-15.

    They asserted `MissingSeries` while §14's gap was open; the datasets landed
    and the assertion is now the coverage itself, stated as the measured
    numbers rather than as "not empty".  Measured on the shipped parquets
    (`AEMO_NSW1_Rooftop_PV_30min`, `AEMO_NSW1_Dispatch_Price_5min`), CPU,
    2026-08-15: 17 376 periods on the summer window and 15 176 on the winter
    one, no NaN in either series on either.
    """
    for window, n in ((AEDT_WINDOW, 17376), (AEST_WINDOW, 15176)):
        grid = _window_grid(window)
        assert len(grid) == n
        for load in (load_pv_series, load_energy_price):
            series = np.asarray(load(grid))
            assert series.shape == (n,)
            assert not np.isnan(series).any()


def test_the_two_missing_winter_intervals_are_bridged_by_the_resampling():
    """The gap the source has is not the gap the market sees, and that is the
    caveat this asserts rather than leaves in a sidecar.

    `ROOFTOP_PV_ACTUAL` is missing 2024-09-05 02:30 and 03:00 UTC -- local
    12:30 and 13:00, solar noon -- and the sidecar records them, unfilled, as
    the repository's discipline requires.  Resampling the 30-minute series onto
    the 15-minute grid interpolates in time, so by the time a rollout on the
    winter window reads those periods they carry interpolated values and no
    `NaN` marks them.  Interpolating a power series is the right choice (§14,
    and the sidecar's `resampling_declaration`), so this is not a defect; but
    "the source has no hole here" and "the market sees no hole here" are then
    two different statements, and only the second one is observable downstream.
    The summer window has no such interval on either series.
    """
    gap = pd.Timestamp("2024-09-05 02:30", tz="UTC")
    assert pd.Timestamp(AEST_WINDOW[0], tz="UTC") < gap
    assert gap < pd.Timestamp(AEST_WINDOW[1], tz="UTC")

    grid = _window_grid(AEST_WINDOW)
    pv = np.asarray(load_pv_series(grid))
    bridged = pv[(grid >= gap) & (grid < gap + pd.Timedelta("1h"))]
    assert bridged.shape == (4,)
    assert not np.isnan(bridged).any()
    assert (bridged > 0.0).all()            # local noon: not a night period


def test_the_price_reaches_its_cap_bitwise_and_stays_inside_its_floor():
    """The cap is a quoted constant; the floor is a bound the data approaches.

    The two ends are **not** symmetric and were reported as if they were on
    2026-08-15, from a two-decimal print: the maximum is exactly the 2024-25 NEM
    market price cap, but the minimum of the five-minute source is -999.99797,
    which never reaches the -1000 market floor.  Asserting equality on the cap is
    admissible because it is a quoted administrative constant the source
    publishes at that value, so a resampling or unit error moves it off 17500
    entirely rather than by an ulp; asserting equality on the floor would be
    asserting a coincidence, so what is asserted there is the invariant that
    actually holds, that the series never crosses it.

    The negative side matters on its own: §9.3's markup map inverts once the cost
    basis goes negative, and the exposure this series carries is
    11 307 of 113 760 five-minute intervals, 9.94 per cent.
    """
    price = np.asarray(load_energy_price(_window_grid(AEDT_WINDOW)))
    assert price.max() == 17500.0
    assert price.min() < 0.0
    assert price.min() > -1000.0


def test_the_photovoltaic_series_is_zero_at_night_on_the_local_clock():
    """The check that a ten-hour offset error would fail and a range check
    would not.

    The series is regional rooftop output, so it is exactly zero between local
    22:00 and 04:00 and peaks near local noon.  Reading the market time of the
    archive as UTC shifts the profile by ten or eleven hours and moves the daily
    peak into the local night, which leaves the range, the mean and the count
    untouched -- so those would all still pass.  Measured 2026-08-15: the
    night-time maximum is exactly 0.0.
    """
    grid = _window_grid(AEDT_WINDOW)
    pv = np.asarray(load_pv_series(grid))
    local = grid.tz_convert(DEMAND_TIMEZONE)
    night = (local.hour >= 22) | (local.hour < 4)
    assert night.sum() > 0
    assert pv[night].max() == 0.0
    assert pv[~night].max() > 1000.0


def test_an_unregistered_signal_still_says_what_a_manifest_would_have_to_give(
        tmp_path):
    """The `MissingSeries` branch the two tripwires used to cover.

    When the datasets landed, every loader call on the real registry began to
    succeed, and this branch -- "no registered dataset provides this signal" --
    lost its only exercise.  Flipping the tripwires to positive assertions
    without this one would have turned a completed item into a fresh hole, so
    the branch is kept, pointed at an empty registry instead of at an empty
    repository.
    """
    grid = pd.date_range("2024-10-07", periods=4, freq="15min", tz="UTC")
    with pytest.raises(MissingSeries, match="no registered dataset provides"):
        load_pv_series(grid, manifest_dir=tmp_path)
    with pytest.raises(MissingSeries, match="no registered dataset provides"):
        load_energy_price(grid, manifest_dir=tmp_path)


# ------------------------------------------------------------------ fixture
# A parquet and a manifest in a temporary directory: the mechanics of the two
# loaders that have nothing to load yet.  Not data for any market result.

def write_dataset(tmp_path, name, signal, resolution, index, values,
                  region=None):
    frame = pd.DataFrame({"ts": index, "value": values})
    column_map = {"value": signal}
    index_map = {"ts": "datetime"}
    if region is not None:
        frame["zone"] = region
        index_map["zone"] = "region"
    frame.to_parquet(tmp_path / f"{name}.parquet")
    manifest = dict(name=name, source="fixture", data_type="actual_series",
                    time_mode="calendar", resolution=resolution,
                    parquet_file=f"{name}.parquet", column_map=column_map,
                    index_map=index_map, derived={}, normalize={},
                    region_values=[region] if region else [])
    (tmp_path / f"{name}.json").write_text(json.dumps(manifest))
    return name


def test_coarser_series_is_held_and_finer_series_is_averaged(tmp_path):
    """A 30-minute price is held across its own interval and a 5-minute
    photovoltaic series is averaged into the period, which is what those two
    quantities mean rather than a choice of interpolation."""
    grid = pd.date_range("2024-10-07", periods=8, freq="15min", tz="UTC")

    price_name = write_dataset(
        tmp_path, "fixture_price", "market.spot_price_aud_mwh", "30min",
        pd.date_range("2024-10-07", periods=4, freq="30min", tz="UTC"),
        [10.0, 20.0, 30.0, 40.0])
    price = load_energy_price(grid, dataset=price_name, data_dir=tmp_path,
                              manifest_dir=tmp_path)
    np.testing.assert_allclose(price, [10, 10, 20, 20, 30, 30, 40, 40])

    pv_name = write_dataset(
        tmp_path, "fixture_pv", "solar.substation_mw", "5min",
        pd.date_range("2024-10-07", periods=24, freq="5min", tz="UTC"),
        np.arange(24, dtype=float), region="NSW1")
    pv = load_pv_series(grid, dataset=pv_name, region="NSW1",
                        data_dir=tmp_path, manifest_dir=tmp_path)
    # three 5-minute values average into each 15-minute period
    np.testing.assert_allclose(pv[:8], np.arange(24).reshape(8, 3).mean(axis=1))
    assert pv.dtype == np.float32


def test_a_series_that_stops_inside_the_window_raises(tmp_path):
    grid = pd.date_range("2024-10-07", periods=8, freq="15min", tz="UTC")
    name = write_dataset(
        tmp_path, "fixture_short", "market.spot_price_aud_mwh", "15min",
        pd.date_range("2024-10-07", periods=3, freq="15min", tz="UTC"),
        [10.0, 11.0, 12.0])
    with pytest.raises(ValueError, match="series that stops"):
        load_energy_price(grid, dataset=name, data_dir=tmp_path,
                          manifest_dir=tmp_path)


def test_naming_a_dataset_that_does_not_provide_the_signal(tmp_path):
    name = write_dataset(
        tmp_path, "fixture_other", "load.actual_mw", "15min",
        pd.date_range("2024-10-07", periods=8, freq="15min", tz="UTC"),
        np.ones(8))
    grid = pd.date_range("2024-10-07", periods=8, freq="15min", tz="UTC")
    with pytest.raises(MissingSeries, match="does not provide"):
        load_pv_series(grid, dataset=name, data_dir=tmp_path,
                       manifest_dir=tmp_path)


# --------------------------------------------------------------------------
# The Swiss configuration of §14.  It is a second configuration and not a
# replacement, so every test below has an Australian counterpart above that
# must keep passing unchanged; the regression that pins that is the last one.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def swiss():
    return load_swiss_flex_series(projection_year=2030, tariff_category="C2",
                                  tariff_period=2026)


def test_the_swiss_grid_is_hourly_and_the_step_is_read_not_assumed(swiss):
    """§14 puts this configuration on $\\Delta$ = 1 h against 0.25 h above.

    The module constant was the Australian period, and a constant that does not
    match the series does not fail loudly: it turns every hour into three absent
    quarter-hours, which reads as a defect in the data rather than as a
    mismatched period length.  Before the step was inferred, asking for this
    series reported 26 277 of 35 037 periods missing.
    """
    assert len(swiss.index) == 8760
    step = swiss.index[1] - swiss.index[0]
    assert step == pd.Timedelta(hours=1)
    assert (swiss.index.to_series().diff().dropna() == step).all()


def test_the_swiss_labels_are_read_as_local_clock_time(swiss):
    """The source carries no timezone, so UTC is a carrier and not a conversion.

    Its rows are labelled `MM-DD HH:MM:SS` with no year, under a stated
    convention of 365 days beginning on a Monday; the ingest stamped them onto
    2029 UTC because that year satisfies the convention and lies outside the
    span of `ch_dayahead_price`.  Converting such a label to Europe/Zurich would
    shift §9.4's row index by an hour and put noon two rows off solar noon.
    """
    assert str(swiss.index[0]) == "2029-01-01 00:00:00+00:00"
    assert swiss.index.tz is not None


def test_the_feeder_total_carries_the_profile_maximum_of_the_source(swiss):
    """0.852801 rather than unity, which §14 records and which is load bearing.

    The source describes its profile as max-normalised and its methods divide by
    the maximum, so this would be 1.0; it ships at 0.852801.  A scaling factor
    calibrated on this series therefore carries a factor of 1.1726 that belongs
    to the data, which is why §14 forbids quoting the Australian $\\kappa$ here.
    """
    registered = 13.719738
    assert float(swiss.load_mw.max()) == pytest.approx(
        registered * 0.852801, rel=1e-5)
    assert float(swiss.load_mw.min()) > 0.0


def test_the_photovoltaic_matrix_is_per_bus_and_not_a_regional_series(swiss):
    """The Swiss table publishes output per equipped bus, so no outer product.

    `make_local_flex_params` takes `(n_periods, n_agent)` either way, so the
    difference between the two configurations does not reach the environment.
    """
    assert swiss.pv.shape == (8760, 15)
    assert swiss.pv.dtype == np.float32
    assert (swiss.pv >= 0.0).all()


def test_the_representative_days_are_carried_across_their_month_not_tiled():
    """288 hourly values become 8760 by month, which is the whole point.

    `TimeAligner.align_profile` tiles a repeatable profile end to end, which
    would turn twelve monthly days into a twelve-day cycle and remove the
    seasonal structure §14 wants photovoltaic output for.  So this table is
    deliberately unreachable through the registry, and the check that the
    expansion is seasonal rather than cyclic is that every day inside one month
    is identical under `repeat` while months differ from one another.
    """
    s = load_swiss_flex_series(projection_year=2050, tariff_category="C2",
                               tariff_period=2026, pv_rule="repeat")
    total = s.pv.sum(axis=1)
    by_day = total.reshape(365, 24)
    jan = by_day[:31]
    assert np.allclose(jan, jan[0]), "days inside January must share one shape"
    jun = by_day[151]
    assert jun.max() > jan[0].max(), "June must out-produce January"


def test_sampling_needs_a_seed_and_repeats_under_one():
    """The second rule of §18 draws a variate, so it is a declared parameter."""
    with pytest.raises(ValueError, match="seed"):
        load_swiss_flex_series(projection_year=2030, tariff_category="C2",
                               tariff_period=2026, pv_rule="sample")
    a = load_swiss_flex_series(projection_year=2030, tariff_category="C2",
                               tariff_period=2026, pv_rule="sample", pv_seed=7)
    b = load_swiss_flex_series(projection_year=2030, tariff_category="C2",
                               tariff_period=2026, pv_rule="sample", pv_seed=7)
    c = load_swiss_flex_series(projection_year=2030, tariff_category="C2",
                               tariff_period=2026, pv_rule="sample", pv_seed=8)
    assert np.array_equal(a.pv, b.pv)
    assert not np.array_equal(a.pv, c.pv)


def test_the_tariff_is_annual_and_therefore_constant_in_an_episode(swiss):
    """§14 records the consequence: the offer prices location and nothing else.

    Taking the energy component and not the total is the other half: §9.5 prices
    the energy an aggregator stores, not the network service delivering it, and
    the two differ by roughly a factor of two here.
    """
    assert len(np.unique(swiss.energy_price)) == 1
    assert float(swiss.energy_price[0]) == pytest.approx(99.65, abs=5e-3)


def test_a_category_this_operator_does_not_offer_raises(swiss):
    """C5 ships as a row of zeros rather than as an absent row.

    Selecting it would give a replacement cost of zero and offers priced at
    zero, with every reported quantity finite and nothing raising.
    """
    with pytest.raises(ValueError, match="does not offer"):
        load_elcom_tariff(swiss.index, category="C5", period=2020)


def test_the_projection_year_fixes_the_population_it_does_not_choose_it():
    """15 / 24 / 34 buses carry a battery, which is where $N^{agent}$ comes from."""
    sizes = {y: load_swiss_flex_series(projection_year=y, tariff_category="C2",
                                       tariff_period=2026).pv.shape[1]
             for y in (2030, 2040, 2050)}
    assert sizes == {2030: 15, 2040: 24, 2050: 34}


def test_the_australian_path_is_unchanged_by_the_second_configuration(summer):
    """The regression that makes this an addition rather than a replacement.

    The step is now read from the series instead of taken from the module
    constant, and the timezone is a parameter instead of a constant, so the
    Australian window has to come back on the same grid it did before.
    """
    load_mw, index = summer
    assert index[1] - index[0] == pd.Timedelta(hours=PERIOD_HOURS)
    assert len(index) == 17376
    assert load_mw.dtype == np.float32
