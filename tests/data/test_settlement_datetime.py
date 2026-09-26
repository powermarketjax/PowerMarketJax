# Written for this repository on 2026-08-06 -- no upstream counterpart.
"""gb_neso_demand stores settlement date + period, not a timestamp.

Before the loader rebuilt the timestamp, this dataset came back with no time
axis at all (a single value column on a RangeIndex), and because
``align_calendar`` returns unchanged when there is no time column, asking for
a 7-day window handed back all 285454 rows spanning 2009-2025 -- silently.

The rebuild adds the period offset **in UTC to the instant of local
midnight**. Adding it to local wall-clock time instead would put period 50 at
``24:30`` of the settlement date, past midnight, which no later
``tz_localize`` can undo. The DST tests below are what distinguish the two.
"""

from pathlib import Path

import dataclasses
import pandas as pd
import pytest

from powermarketjax.data.data_loader import DataLoader

DATA_DIR = Path(__file__).resolve().parents[2] / "powermarketjax/data/parquet"
MANIFEST_DIR = Path(__file__).resolve().parents[2] / "powermarketjax/data/manifests"

NESO_SIGNAL = "load.england_wales_mw"   # NESO-only; load.actual_mw resolves to AEMO


@pytest.fixture(scope="module")
def loader():
    return DataLoader(data_dir=DATA_DIR, manifest_dir=MANIFEST_DIR)


@pytest.fixture(scope="module")
def neso(loader):
    return loader.load_signals([NESO_SIGNAL])


def test_timestamps_are_utc_strictly_increasing_and_uniform(neso):
    t = pd.to_datetime(neso["datetime"])
    assert str(t.dt.tz) == "UTC"
    steps = t.diff().dropna().unique()
    assert list(steps) == [pd.Timedelta("30min")], (
        f"expected a single 30-minute step over the whole series, got {steps}")


@pytest.mark.parametrize("day,n_periods", [
    ("2024-03-31", 46),   # BST starts: 23-hour local day
    ("2024-10-27", 50),   # GMT returns: 25-hour local day
    ("2024-06-15", 48),   # ordinary day
    ("2016-02-29", 48),   # leap day
])
def test_local_day_length_follows_dst(neso, day, n_periods):
    local = pd.to_datetime(neso["datetime"]).dt.tz_convert("Europe/London")
    assert int((local.dt.date == pd.Timestamp(day).date()).sum()) == n_periods


def test_fall_back_hour_appears_twice_with_distinct_offsets(neso):
    """01:00 and 01:30 occur once in BST and once in GMT on the fall-back day."""
    local = pd.to_datetime(neso["datetime"]).dt.tz_convert("Europe/London")
    day = local[local.dt.date == pd.Timestamp("2024-10-27").date()].sort_values()
    repeated = [x for x in day if x.strftime("%H:%M") in ("01:00", "01:30")]
    assert len(repeated) == 4
    assert sum(1 for x in repeated if x.dst() != pd.Timedelta(0)) == 2


def test_alignment_window_now_filters(loader, neso):
    """The original silent failure: a 7-day window returned all 16 years."""
    from powermarketjax.data.alignment import TimeAligner

    out = TimeAligner.align_calendar(
        neso,
        sim_start=pd.Timestamp("2025-01-01", tz="UTC"),
        sim_end=pd.Timestamp("2025-01-07", tz="UTC"),
    )
    assert len(out) == 7 * 48
    assert out["datetime"].min() == pd.Timestamp("2025-01-01 00:00", tz="UTC")
    assert out["datetime"].max() == pd.Timestamp("2025-01-07 23:30", tz="UTC")


def test_settlement_columns_without_timezone_raise(loader):
    """The manifest must say which settlement calendar the periods belong to."""
    manifest = dataclasses.replace(
        loader.registry.get_manifest("gb_neso_demand"), timezone=None)
    raw = pd.DataFrame({
        "settlement_date": pd.to_datetime(["2024-01-01"] * 2),
        "settlement_period": [1, 2],
    })
    with pytest.raises(ValueError, match="timezone"):
        DataLoader._datetime_from_settlement(raw, manifest)
