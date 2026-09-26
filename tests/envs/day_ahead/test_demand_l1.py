"""L1 correctness for the GB demand pair (§14).

No JAX here: the pair is assembled once at construction, so there is no jit or
vmap contract to check.  What there is to check is that the two series are the
*right* two, because the failure this module exists to prevent is silent.  The
`Actual` column that ships beside the forecast is a gross demand measure, and
pairing the forecast with it produces an 11% "forecast error" that is really a
definitional offset; §14 says building an uncertainty model on that would make
the task about learning the offset rather than about forecast uncertainty.

The discriminating property is bias.  A like-for-like forecast/realisation pair
is unbiased, so the share of periods where the realisation exceeds the forecast
sits near one half.  Measured 2026-08-09 over the overlap: 42.3% for the pair
this module returns, against 82.2% for the shipped `Actual` and 7.6% for `ND`.
That one number separates all three candidates.
"""
import numpy as np
import pandas as pd
import pytest

from powermarketjax.envs.day_ahead.demand import T, load_gb_demand

#: Measured 2026-08-09; the overlap runs 2023-07-05 to 2025-04-12 after the
#: truncated final day is dropped.
N_DAYS = 648


@pytest.fixture(scope="module")
def pair():
    return load_gb_demand()


def test_shape_dtype_and_no_gaps(pair):
    forecast, actual, days = pair
    assert forecast.shape == actual.shape == (N_DAYS, T)
    # float32 because the EnvState arrays are (§15)
    assert forecast.dtype == np.float32 and actual.dtype == np.float32
    assert not np.isnan(forecast).any() and not np.isnan(actual).any()
    assert len(days) == N_DAYS
    assert days.is_monotonic_increasing and days.is_unique


def test_series_is_gb_transmission_demand_not_another_country(pair):
    """`load.actual_mw` is mapped by three datasets and the registry resolves it
    to AEMO, so a wrong signal name returns 5-minute Australian data without an
    error.  GB transmission demand runs 16 to 48 GW; NSW runs under 15 GW."""
    _, actual, _ = pair
    assert 16_000 < actual.min() < 20_000
    assert 45_000 < actual.max() < 50_000
    assert 25_000 < np.median(actual) < 32_000


def test_the_pair_is_unbiased(pair):
    """The check that separates this pair from the two wrong ones.

    Shipped `Actual`: 82.2% one-sided.  NESO `ND`: 7.6%.  This pair: 42.3%.
    """
    forecast, actual, _ = pair
    residual = actual - forecast
    share_positive = float((residual > 0).mean())
    assert 0.35 < share_positive < 0.65, f"one-sided residual: {share_positive:.3f}"
    assert abs(float(np.median(residual))) < 1_000.0


def test_forecast_error_is_the_measured_magnitude(pair):
    """6.05% relative, measured.  Higher than the 1-3% a national day-ahead
    forecast usually achieves, which is why §14 records it rather than assuming
    it; the band here is wide enough to survive a data refresh and narrow enough
    to catch a re-pairing."""
    forecast, actual, _ = pair
    relative = np.median(np.abs(actual - forecast) / actual)
    assert 0.04 < relative < 0.08, f"relative error {relative:.4f}"


def test_hours_are_in_order_within_the_day(pair):
    """A reshape that scrambled the hour axis would flatten the daily profile.

    GB transmission demand peaks in the evening; averaged over the whole window
    the highest hour is 16:00 to 19:00 UTC and the lowest is before dawn.
    """
    _, actual, _ = pair
    profile = actual.mean(axis=0)
    assert 16 <= int(profile.argmax()) <= 19
    assert 1 <= int(profile.argmin()) <= 5
    assert float(np.median(actual.max(1) / actual.min(1))) > 1.3


def test_days_are_consecutive_apart_from_the_dropped_one(pair):
    """Episodes are consecutive days (§9.1), and the physical state crosses the
    day boundary, so a hole in the calendar would join two days that are not
    adjacent."""
    _, _, days = pair
    gaps = pd.Series(days).diff().dropna()
    assert (gaps == pd.Timedelta("1D")).all(), f"non-consecutive days: {gaps.value_counts()}"
