# Vendored test from PowerZooJax.
# Source        : tests/data/test_gb_market_mid_data.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
"""Tests for the GB MID market index dataset."""

import pandas as pd

from powermarketjax.data import DataLoader
from powermarketjax.data import signals as S


def test_gb_market_mid_manifest_is_registered():
    loader = DataLoader()

    assert "gb_market_mid" in loader.registry.list_datasets()
    manifest = loader.registry.get_manifest("gb_market_mid")

    assert manifest.parquet_file == "MID_GB_30min_aligned_to_gen.parquet"
    assert manifest.date_range == ("2023-07-05", "2026-04-05")
    assert manifest.source_url == "https://data.elexon.co.uk/bmrs/api/v1/datasets/MID"
    assert manifest.source_organization == "Elexon Limited"
    assert S.MARKET_MID_PRICE_APX in manifest.signals
    assert S.MARKET_MID_PRICE_N2EX in manifest.signals
    assert S.MARKET_MID_VOLUME_APX in manifest.signals
    assert S.MARKET_MID_VOLUME_N2EX in manifest.signals


def test_gb_market_mid_loads_by_semantic_signal():
    loader = DataLoader()

    df = loader.load_signals(
        [S.MARKET_MID_PRICE_APX, S.MARKET_MID_VOLUME_APX],
        source="gb",
        start_date="2025-12-14",
        end_date="2025-12-14",
        resample="30min",
    )

    assert {
        S.DATETIME,
        S.MARKET_MID_PRICE_APX,
        S.MARKET_MID_VOLUME_APX,
    } <= set(df.columns)
    assert len(df) == 48
    assert pd.api.types.is_datetime64_any_dtype(df[S.DATETIME])
    assert df[S.MARKET_MID_PRICE_APX].notna().any()
    assert df[S.MARKET_MID_VOLUME_APX].notna().any()


def test_gb_market_mid_metadata_matches_gb_demand_timeline():
    loader = DataLoader()

    market_meta = loader.get_metadata("MID_GB_30min_aligned_to_gen")
    demand_meta = loader.get_metadata(
        "GB_Forecast_Actual_Demand_2023_2025_30min"
    )

    assert market_meta["shape"]["rows"] == demand_meta["shape"]["rows"]
    assert (
        market_meta["date_ranges"]["startTime"]
        == demand_meta["date_ranges"]["startTime"]
    )


def test_gb_market_mid_aligns_with_gb_load_series():
    loader = DataLoader()

    df = loader.load_signals(
        [S.LOAD_ACTUAL_MW, S.MARKET_MID_PRICE_APX],
        source="gb",
        start_date="2025-12-14",
        end_date="2025-12-14",
        resample="30min",
    )

    assert {
        S.DATETIME,
        S.LOAD_ACTUAL_MW,
        S.MARKET_MID_PRICE_APX,
    } <= set(df.columns)
    assert len(df) == 48
    assert df[S.LOAD_ACTUAL_MW].notna().all()
    assert df[S.MARKET_MID_PRICE_APX].notna().any()


def test_gb_market_mid_legacy_loader_accepts_naive_date_filters():
    loader = DataLoader()

    df = loader.load_data(
        dataset_name="MID_GB_30min_aligned_to_gen",
        columns=["mid_price_APXMIDP"],
        start_date="2025-12-14",
        end_date="2025-12-14",
    )

    assert list(df.columns) == ["datetime", "mid_price_APXMIDP"]
    assert len(df) == 48
    assert df["mid_price_APXMIDP"].notna().any()
