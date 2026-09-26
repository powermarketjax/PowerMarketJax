# Written for this repository on 2026-08-15 -- no upstream counterpart.
"""Tests for the NSW1 spot price and rooftop photovoltaic series.

These defend the data layer: what is in the parquet and what the sidecar claims
about it.  The loader path is defended separately in
``tests/envs/local_flexibility/test_data_l1.py``; the price bounds and the night
zeros appear in both on purpose, because a fault in the stored bytes and a fault
in the alignment produce the same symptom and only two layers separate them.

Three properties of this data would survive a wrong value while still looking
reasonable, which is why each gets an assertion rather than a note.

**The two extremes are administrative constants, not measurements.**  The price
floor of -1000 and the cap of 17500 AUD/MWh are set by the rules, so they are the
one place where a bit-exact assertion is legitimate: a unit slip or a botched
resample moves them wholesale rather than by an ulp.  It is also why they must
never be clipped as outliers -- they mark the scarcity and negative-price
intervals this market exists to price.

**The archive contradicted itself twice** and both traps are silent.  Its file
naming changed between 2024-07 and 2024-08 into two disjoint forms, so fetching
with one pattern truncates the window exactly where the winter demand window
sits; and the older files each carry one extra day, so consecutive months overlap
by 48 half hours.  The overlap was verified to be identical republication rather
than revision before one copy was dropped, and the sidecar has to keep saying so.

**Averaging and interpolation are indistinguishable on flat data.**  A 15-minute
mean equals the endpoints of a flat five minutes, and a linear interpolation
equals a forward fill across a flat half hour.  So the resampling tests below
pick their sample points at the sharpest jump and the steepest ramp in the whole
series -- anywhere else they would pass under the wrong rule.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from powermarketjax.envs.local_flexibility.data import (load_energy_price,
                                                        load_pv_series)

DATA = Path(__file__).resolve().parents[2] / "powermarketjax/data/parquet"
MANIFESTS = Path(__file__).resolve().parents[2] / "powermarketjax/data/manifests"
PRICE = DATA / "AEMO_NSW1_Dispatch_Price_5min.parquet"
PV = DATA / "AEMO_NSW1_Rooftop_PV_30min.parquet"

#: NEM market floor and the 2024-25 market price cap, in AUD/MWh.
FLOOR, CAP = -1000.0, 17500.0
#: 395 days from 2024-04-01 to 2025-04-30 inclusive, in market time.
DAYS = 395
#: The rooftop series is short exactly the two half hours the source omits.
PV_GAPS = ["2024-09-05 02:30:00", "2024-09-05 03:00:00"]

# The sharpest 15-minute disagreement in the price series: an interval whose
# three 5-minute prices span nearly the whole regulated range.  Written out so
# the expected mean is arithmetic anyone can check, not a number this test
# recomputed from the same series it is testing.
JUMP_SLOT = "2024-11-11 08:00:00"
JUMP_FIVES = (300.8, 17499.96, 48.41487)
JUMP_MEAN = sum(JUMP_FIVES) / 3.0                      # 5949.724956666...

# The steepest half-hourly ramp in the photovoltaic series, a morning sunrise.
RAMP_FROM, RAMP_TO = "2024-12-26 22:00:00", "2024-12-26 22:30:00"
RAMP_VALUES = (1812.0240, 3094.0710)
RAMP_MID = "2024-12-26 22:15:00"
RAMP_MID_INTERP = sum(RAMP_VALUES) / 2.0               # 2453.0475


@pytest.fixture(scope="module")
def price():
    return pd.read_parquet(PRICE).set_index("datetime")["spot_price_aud_mwh"]


@pytest.fixture(scope="module")
def pv():
    return pd.read_parquet(PV).set_index("datetime")["rooftop_pv_mw"]


@pytest.fixture(scope="module")
def price_meta():
    return json.loads(PRICE.with_suffix(".json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pv_meta():
    return json.loads(PV.with_suffix(".json").read_text(encoding="utf-8"))


# --------------------------------------------------------------- the parquet

def test_the_price_series_is_a_complete_five_minute_grid(price):
    assert len(price) == DAYS * 288
    assert not price.index.has_duplicates
    assert (price.index.to_series().diff().dropna()
            == pd.Timedelta(minutes=5)).all()


def test_the_photovoltaic_series_is_half_hourly_and_short_exactly_two(pv):
    assert len(pv) == DAYS * 48 - len(PV_GAPS)
    assert not pv.index.has_duplicates
    full = pd.date_range(pv.index.min(), pv.index.max(), freq="30min")
    assert [str(g) for g in full.difference(pv.index)] == PV_GAPS


def test_the_cap_is_attained_exactly_and_the_floor_is_never_breached(price):
    """The two ends are administrative constants, but only one of them is *hit*.

    The cap is attained bit-exactly, 21 times, so equality is legitimate there:
    it is a quoted constant, and a unit slip or a mis-scaled resample would move
    it wholesale rather than by an ulp.  **The floor is not attained** -- the
    minimum is -999.99797, within 0.003 of it and never below.  So the floor gets
    the invariant it actually has (dispatch prices cannot go below it) plus a
    bound saying the series does exercise the bottom of the range; asserting
    equality there would be asserting a coincidence.

    Clipping either end would delete the scarcity and negative-price intervals
    this market exists to price.
    """
    assert price.max() == CAP
    assert (price == CAP).sum() == 21
    assert price.min() >= FLOOR
    assert price.min() < FLOOR + 1.0
    assert (price < 0).sum() > 0


def test_rooftop_output_is_exactly_zero_at_night(pv):
    """Zero at night is a real value, not a gap, so nothing should fill it."""
    local = pv.index.tz_localize("UTC").tz_convert("Australia/Sydney")
    night = pv[(local.hour >= 22) | (local.hour <= 4)]
    assert len(night) > 5000
    assert night.max() == 0.0
    assert pv.max() > 1000.0            # and the daytime signal is really there


# --------------------------------------------------------------- the sidecar

@pytest.mark.parametrize("which", ["price", "pv"])
def test_every_month_records_a_distinct_archive_and_a_usable_url(which, request):
    """13 months, each with its own checksum.

    Distinctness is the point: a copy-paste in the fetch loop would leave two
    months claiming the same bytes, and every other field would still look right.
    """
    meta = request.getfixturevalue(f"{which}_meta")
    archives = meta["archives"]
    assert len(archives) == 13
    digests = [a["sha256"] for a in archives.values()]
    assert len(set(digests)) == 13
    assert all(len(d) == 64 and set(d) <= set("0123456789abcdef") for d in digests)
    for key, a in archives.items():
        month = key.rsplit("_", 1)[1]                  # e.g. 202411
        assert a["url"].startswith("https://nemweb.com.au/")
        assert month in a["url"] and a["bytes"] > 0
        assert a["naming"] in ("ARCHIVE", "DVD")


def test_the_archive_renaming_is_recorded_and_both_forms_were_used(price_meta):
    """Both naming schemes appear, which is the evidence the changeover was
    crossed rather than silently truncated at it."""
    used = {a["naming"] for a in price_meta["archives"].values()}
    assert used == {"DVD", "ARCHIVE"}
    dvd = {k.rsplit("_", 1)[1] for k, a in price_meta["archives"].items()
           if a["naming"] == "DVD"}
    assert dvd == {"202404", "202405", "202406", "202407"}


def test_the_dropped_duplicates_are_recorded_as_republication(pv_meta, price_meta):
    """4 changeovers x 48 half hours; and none at all on the price series."""
    assert pv_meta["duplicate_rows_dropped"] == 192
    assert price_meta["duplicate_rows_dropped"] == 0
    assert "republication" in pv_meta["duplicate_note"]


@pytest.mark.parametrize("which,column", [("price", "spot_price_aud_mwh"),
                                          ("pv", "rooftop_pv_mw")])
def test_the_window_coverage_claim_recomputes(which, column, request):
    """The sidecar's coverage is the fact that picks the window, so recompute it.

    A stale claim here is the failure that matters: it would send the next reader
    to a window with a hole in it.
    """
    from powermarketjax.envs.local_flexibility.data import (AEDT_WINDOW,
                                                            AEST_WINDOW)
    meta = request.getfixturevalue(f"{which}_meta")
    series = request.getfixturevalue(which)
    step = f"{meta['resolution_minutes']}min"
    for window, (lo, hi) in (("AEDT_summer", AEDT_WINDOW),
                             ("AEST_winter", AEST_WINDOW)):
        want = pd.date_range(lo, hi, freq=step)
        missing = [str(m) for m in want.difference(series.index)]
        claim = meta["window_coverage"][window]
        assert claim["intervals_expected"] == len(want)
        assert claim["intervals_present"] == len(want) - len(missing)
        assert claim["missing"] == missing

    summer = meta["window_coverage"]["AEDT_summer"]
    assert summer["missing"] == []
    assert summer["intervals_present"] == summer["intervals_expected"]
    winter = meta["window_coverage"]["AEST_winter"]
    assert winter["missing"] == (PV_GAPS if which == "pv" else [])


def test_the_licence_record_separates_permission_from_evidence(price_meta):
    """Clearance changed whether the data may be used, not what tier the
    evidence is, and the sidecar has to keep those apart."""
    assert price_meta["licence_permits_redistribution"] is True
    assert price_meta["licence_in_retrieval_response"] is False
    assert "Internet Archive" in price_meta["licence_text_read_from"]
    assert "CC BY" in price_meta["licence_note"]        # recorded as excluded
    assert "够，放行并结案那三个" in price_meta["licence_ruling"]


@pytest.mark.parametrize("name,signal", [
    ("aemo_nsw1_dispatch_price", "market.spot_price_aud_mwh"),
    ("aemo_nsw1_rooftop_pv", "solar.substation_mw")])
def test_the_manifest_registers_the_signal_the_market_asks_for(name, signal):
    manifest = json.loads((MANIFESTS / f"{name}.json").read_text(encoding="utf-8"))
    assert list(manifest["column_map"].values()) == [signal]
    assert manifest["source_organization"].startswith("Australian Energy Market")


# ------------------------------------------------------ the resampling itself

def test_the_price_is_averaged_three_to_one_and_not_sampled(price):
    """Picked at the sharpest jump in the series, because nowhere else can fail.

    Across a flat five minutes the mean, the first value and the last value
    agree, so a test sited there passes whether the loader averages or samples.
    Here the three 5-minute prices are 300.80, 17499.96 and 48.41 -- one of them
    near the market cap -- so the three rules give three different answers and
    only the mean is right for a 15-minute period.
    """
    slot = pd.Timestamp(JUMP_SLOT)
    fives = price.loc[slot:slot + pd.Timedelta(minutes=10)]
    assert [float(v) for v in fives.values] == list(JUMP_FIVES)

    index = pd.date_range(slot, periods=4, freq="15min", tz="UTC")
    got = load_energy_price(index)
    assert np.isclose(got[0], JUMP_MEAN, rtol=1e-6)
    # and the two rules this one has to beat are genuinely elsewhere
    assert not np.isclose(got[0], JUMP_FIVES[0], rtol=1e-3)
    assert not np.isclose(got[0], JUMP_FIVES[-1], rtol=1e-3)


def test_the_photovoltaic_series_is_interpolated_and_not_held(pv):
    """Picked at the steepest half-hourly ramp, for the same reason.

    Power is a rate: holding a half-hourly value across both quarter hours says
    output was flat and then stepped, interpolating says it ramped, and for a
    solar profile the ramp is the physical claim.  The two rules differ by
    641 MW here and by nothing at all across a flat half hour.
    """
    lo, hi = pd.Timestamp(RAMP_FROM), pd.Timestamp(RAMP_TO)
    assert (float(pv.loc[lo]), float(pv.loc[hi])) == RAMP_VALUES

    index = pd.date_range(lo, hi, freq="15min", tz="UTC")
    interpolated = load_pv_series(index)                       # the default
    held = load_pv_series(index, upsample="ffill")
    assert np.isclose(interpolated[1], RAMP_MID_INTERP, rtol=1e-6)
    assert np.isclose(held[1], RAMP_VALUES[0], rtol=1e-6)
    assert abs(interpolated[1] - held[1]) > 600.0
    assert interpolated[0] == pytest.approx(RAMP_VALUES[0], rel=1e-6)
    assert interpolated[-1] == pytest.approx(RAMP_VALUES[1], rel=1e-6)


def test_interpolation_invents_no_output_at_night(pv):
    """Interpolating between two zeros must stay zero, or the market would see
    generation the sun did not make."""
    night = pd.date_range("2024-12-26 12:00", periods=16, freq="15min", tz="UTC")
    got = load_pv_series(night)
    local = night.tz_convert("Australia/Sydney")
    assert set(local.hour) <= {22, 23, 0, 1, 2, 3, 4}      # the window wraps midnight
    assert got.max() == 0.0
