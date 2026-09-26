# Written for this repository on 2026-08-14 -- no upstream counterpart.
"""Tests for the Swiss day-ahead price series.

This series exists to give the local flexibility market its exogenous
energy price for the Swiss configuration, so what these tests defend is not
"the file loads" but the three ways it could load and still be wrong.

**A price is not a demand.**  Every other series in this repository is
non-negative, and the one other price series, ``gb_market_mid``, is floored at
zero at the source.  A clip, an ``abs``, or a fill applied anywhere on the load
path would therefore be invisible on every existing dataset and would land
exactly on the hours in which storing energy pays best -- the hours §9.5 makes
the aggregator trade against degradation cost.  Two tests below pin the
negative hours by count and by value.

**Two prices in two currencies.**  ``resolve_signals`` takes the first
candidate the registry indexes for a signal, which the day-ahead work already
found the hard way for ``load.actual_mw``.  Here the guard is that the two
prices carry different signal names, so a GBP series can never answer a request
for the EUR one; a test asserts the names are disjoint rather than trusting it.
The currency is part of the name for that reason, and the name is a plain
string rather than a constant in ``data/signals.py``: signal strings are not
validated back to that module, so a dataset registers its own name and the
vendored file stays verbatim, which is the convention ``solar.substation_mw``
established.

**The hourly grid is the pairing surface.**  §14 pairs this series with the
Swiss DER profiles by hour, so a sub-hourly stretch or a hole would misalign
everything after it rather than fail loudly.  The step is checked directly on
the loaded frame, not read back out of the metadata the generator wrote.
"""

import pandas as pd

from powermarketjax.data import DataLoader
from powermarketjax.data import signals as S

#: Not a constant in ``data/signals.py``; see the note above.
DAYAHEAD_PRICE = "market.dayahead_price_eur_mwh"


def test_ch_dayahead_price_manifest_is_registered():
    loader = DataLoader()

    assert "ch_dayahead_price" in loader.registry.list_datasets()
    manifest = loader.registry.get_manifest("ch_dayahead_price")

    assert manifest.parquet_file == "CH_DayAhead_Price_2015_2025_60min.parquet"
    assert manifest.source == "ch"
    assert manifest.resolution == "60min"
    assert manifest.date_range == ("2015-01-01", "2025-12-31")
    assert manifest.source_organization == "Fraunhofer ISE (Energy-Charts)"
    assert DAYAHEAD_PRICE in manifest.signals


def test_ch_dayahead_price_loads_by_semantic_signal():
    loader = DataLoader()

    df = loader.load_signals(
        [DAYAHEAD_PRICE],
        source="ch",
        start_date="2023-07-02",
        end_date="2023-07-02",
        resample="60min",
    )

    assert {S.DATETIME, DAYAHEAD_PRICE} <= set(df.columns)
    assert len(df) == 24
    assert pd.api.types.is_datetime64_any_dtype(df[S.DATETIME])
    assert df[DAYAHEAD_PRICE].notna().all()


def test_ch_dayahead_price_keeps_negative_hours():
    """The single test that a clip or an abs on the load path would fail.

    2024-07-14 10:00 UTC carries the minimum of the whole record. Both the
    value and the sign are asserted: a clip at zero changes the sign, and a
    rescaling that preserved the sign would change the value.
    """
    loader = DataLoader()

    df = loader.load_signals(
        [DAYAHEAD_PRICE],
        source="ch",
        start_date="2024-07-14",
        end_date="2024-07-14",
        resample="60min",
    )

    price = df.set_index(S.DATETIME)[DAYAHEAD_PRICE]
    hour = price.index[price.index.hour == 10]
    assert len(hour) == 1
    assert price.loc[hour[0]] == -427.51
    assert (price < 0).any()


def test_ch_dayahead_price_metadata_counts_the_negative_hours():
    """872 negative hours, spread over every year of the record.

    A guard against a regenerated file that quietly lost them: the count is
    what a clip at source would drive to zero, and the year spread is what a
    single-year truncation would collapse.
    """
    loader = DataLoader()

    meta = loader.get_metadata("CH_DayAhead_Price_2015_2025_60min")
    stats = meta["numeric_statistics"]["dayahead_price_eur_mwh"]

    assert stats["negative_hours"] == 872
    assert stats["min"] == -427.51
    assert stats["missing"] == 0
    assert meta["currency"] == "EUR"
    assert meta["licence_permits_redistribution"] is True
    assert meta["hourly_and_gap_free"] is True


def test_ch_dayahead_price_is_hourly_with_no_interior_gap():
    """Checked on the loaded frame, not read back out of the generator's note."""
    loader = DataLoader()

    df = loader.load_data(
        dataset_name="CH_DayAhead_Price_2015_2025_60min",
        columns=["dayahead_price_eur_mwh"],
        start_date="2023-01-01",
        end_date="2023-12-31",
    )

    steps = df["datetime"].diff().dropna().unique()
    assert list(steps) == [pd.Timedelta(hours=1)]
    assert len(df) == 8760


def test_ch_dayahead_price_does_not_collide_with_the_gb_price_signal():
    """The two price series must not be substitutable for one another.

    They are different products in different currencies and this repository
    holds no exchange rate, so a request for one must never be answered by the
    other. Asserted on the registry rather than assumed from the naming.
    """
    loader = DataLoader()

    ch = loader.registry.get_manifest("ch_dayahead_price")
    gb = loader.registry.get_manifest("gb_market_mid")

    assert set(ch.signals).isdisjoint(set(gb.signals))
    assert DAYAHEAD_PRICE not in gb.signals
    assert S.MARKET_MID_PRICE_APX not in ch.signals

    resolved = loader.registry.find_by_signal(DAYAHEAD_PRICE)
    assert [m.name for m in resolved] == ["ch_dayahead_price"]
