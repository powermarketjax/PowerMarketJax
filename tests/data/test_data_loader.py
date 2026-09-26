# Vendored test from PowerZooJax.
# Source        : tests/data/test_data_loader.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# NOT verbatim. Beyond the package rename (powerzoojax -> powermarketjax, in
# both import statements and path strings): three datacenter-only tests were
# removed, one was retargeted to GB signals, and one dataset-count threshold
# was lowered from 9 to 7. Each site carries an inline note. The four
# datacenter datasets are out of scope here.
#
# KNOWN COVERAGE GAP:
#   All four dropped datasets were the only manifests with
#   time_mode="profile" / cyclical=True. Every vendored dataset is
#   time_mode="calendar". The profile-mode and cyclical-tiling branches of
#   data_loader.py are therefore present but UNTESTED in this repository.
#   Adding alibaba_gpu_2020 back (4.6 KB) would restore that coverage.
"""Tests for powermarketjax.data.data_loader — DataLoader."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from powermarketjax.data import signals as S
from powermarketjax.data.data_loader import DataLoader

DATA_DIR = Path(__file__).resolve().parents[2] / "powermarketjax/data/parquet"
MANIFEST_DIR = Path(__file__).resolve().parents[2] / "powermarketjax/data/manifests"


@pytest.fixture(scope="module")
def loader():
    return DataLoader(data_dir=DATA_DIR, manifest_dir=MANIFEST_DIR)


# ======================================================================
# Semantic API
# ======================================================================


class TestLoadSignals:

    def test_load_single_signal(self, loader):
        df = loader.load_signals([S.LOAD_ACTUAL_MW], source="gb")
        assert S.LOAD_ACTUAL_MW in df.columns
        assert S.DATETIME in df.columns
        assert len(df) > 0

    def test_load_multiple_signals_same_manifest(self, loader):
        df = loader.load_signals(
            [S.LOAD_ACTUAL_MW, S.LOAD_FORECAST_DA_MW], source="gb",
        )
        assert S.LOAD_ACTUAL_MW in df.columns
        assert S.LOAD_FORECAST_DA_MW in df.columns

    def test_load_derived_wind(self, loader):
        df = loader.load_signals([S.WIND_AVAILABLE_MW], source="gb")
        assert S.WIND_AVAILABLE_MW in df.columns
        assert len(df) > 0
        assert df[S.WIND_AVAILABLE_MW].mean() > 0

    def test_load_solar(self, loader):
        df = loader.load_signals([S.SOLAR_AVAILABLE_MW], source="gb")
        assert S.SOLAR_AVAILABLE_MW in df.columns

    # REMOVED when vendoring (2026-08-05): test_load_datacenter_signal.
    # It loaded S.DC_CPU_UTIL from google_dc_2019, which is not vendored.

    def test_load_with_date_range(self, loader):
        df = loader.load_signals(
            [S.LOAD_ACTUAL_MW],
            source="gb",
            start_date="2024-01-01",
            end_date="2024-01-07",
        )
        assert len(df) > 0
        assert len(df) < 10000

    def test_load_empty_signals_raises(self, loader):
        with pytest.raises(ValueError, match="At least one signal"):
            loader.load_signals([])

    def test_load_nonexistent_signal_raises(self, loader):
        with pytest.raises(ValueError, match="Cannot resolve"):
            loader.load_signals(["nonexistent.signal"])

    def test_load_with_resample(self, loader):
        df = loader.load_signals(
            [S.LOAD_ACTUAL_MW],
            source="gb",
            start_date="2024-01-01",
            end_date="2024-01-02",
            resample="60min",
        )
        assert len(df) > 0

    # REMOVED when vendoring (2026-08-05): test_load_profile_mode.
    # It exercised manifests with time_mode="profile" / cyclical=True. All four
    # such datasets are the datacenter traces, which are not vendored; every
    # vendored dataset is time_mode="calendar". See the KNOWN COVERAGE GAP note
    # at the top of this file.

    def test_load_aemo_with_region(self, loader):
        df = loader.load_signals(
            [S.LOAD_ACTUAL_MW],
            source="aemo",
            region="NSW1",
            start_date="2025-01-01",
            end_date="2025-01-07",
        )
        assert len(df) > 0
        if S.REGION in df.columns:
            assert (df[S.REGION] == "NSW1").all()


# ======================================================================
# JAX API
# ======================================================================


class TestLoadJaxProfiles:

    def test_returns_jnp_array(self, loader):
        arr = loader.load_jax_profiles(
            [S.LOAD_ACTUAL_MW],
            source="gb",
            start_date="2024-01-01",
            end_date="2024-01-07",
        )
        assert isinstance(arr, jnp.ndarray)
        assert arr.ndim == 2
        assert arr.shape[1] == 1
        assert arr.dtype == jnp.float32

    def test_multiple_signals_column_order(self, loader):
        # Retargeted when vendoring (2026-08-05): upstream used the two
        # datacenter signals from google_dc_2019. Two GB generation signals
        # from the same manifest exercise the same multi-column path.
        signals = [S.WIND_AVAILABLE_MW, S.SOLAR_AVAILABLE_MW]
        arr = loader.load_jax_profiles(
            signals,
            source="gb",
            start_date="2024-01-01",
            end_date="2024-01-02",
        )
        assert arr.shape[1] == 2
        assert arr.dtype == jnp.float32

    def test_no_nan_in_output(self, loader):
        arr = loader.load_jax_profiles(
            [S.LOAD_ACTUAL_MW],
            source="gb",
            start_date="2024-06-01",
            end_date="2024-06-07",
        )
        assert not jnp.any(jnp.isnan(arr))

    # REMOVED when vendoring (2026-08-05): test_profile_tiling_produces_data.
    # Cyclical tiling only applies to cyclical=True manifests; all four are
    # datacenter traces and are not vendored. See the KNOWN COVERAGE GAP note
    # at the top of this file.

    def test_default_dtype_is_float32(self, loader):
        arr = loader.load_jax_profiles(
            [S.LOAD_ACTUAL_MW],
            source="gb",
            start_date="2024-01-01",
            end_date="2024-01-03",
        )
        assert arr.dtype == jnp.float32


class TestLoadJaxArray:

    def test_from_dataframe(self, loader):
        df = loader.load_signals(
            [S.LOAD_ACTUAL_MW, S.LOAD_FORECAST_DA_MW],
            source="gb",
            start_date="2024-01-01",
            end_date="2024-01-03",
        )
        arr = loader.load_jax_array(df, [S.LOAD_ACTUAL_MW])
        assert isinstance(arr, jnp.ndarray)
        assert arr.shape[1] == 1

    def test_all_numeric_columns(self, loader):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        arr = loader.load_jax_array(df)
        assert arr.shape == (2, 2)

    def test_missing_column_raises(self, loader):
        df = pd.DataFrame({"a": [1.0]})
        with pytest.raises(KeyError, match="not found"):
            loader.load_jax_array(df, ["nonexistent"])


# ======================================================================
# Forecast panel
# ======================================================================


class TestForecastPanel:

    def test_load_forecast_panel(self, loader):
        df = loader.load_forecast_panel(
            [S.LOAD_ACTUAL_MW], source="aemo",
        )
        assert len(df) > 0
        assert S.LOAD_ACTUAL_MW in df.columns

    def test_forecast_panel_with_region(self, loader):
        df = loader.load_forecast_panel(
            [S.LOAD_ACTUAL_MW], source="aemo", region="NSW1",
        )
        if S.REGION in df.columns:
            assert (df[S.REGION] == "NSW1").all()

    def test_forecast_panel_empty_raises(self, loader):
        with pytest.raises(ValueError, match="At least one signal"):
            loader.load_forecast_panel([])


# ======================================================================
# Legacy API
# ======================================================================


class TestLegacyAPI:

    def test_list_datasets(self, loader):
        # Threshold lowered from 9 to 7 when vendoring (2026-08-05): this
        # repository ships 7 datasets, not upstream's 11.
        datasets = loader.list_available_datasets()
        assert len(datasets) >= 7

    def test_get_metadata(self, loader):
        datasets = loader.list_available_datasets()
        json_datasets = [
            d for d in datasets
            if (DATA_DIR / f"{d}.json").exists()
        ]
        if json_datasets:
            meta = loader.get_metadata(json_datasets[0])
            assert "columns" in meta
            assert "shape" in meta

    def test_missing_parquet_raises(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.get_metadata("totally_nonexistent_dataset")


# ======================================================================
# Constructor
# ======================================================================


class TestDataLoaderInit:

    def test_default_paths(self):
        loader = DataLoader()
        assert loader.data_dir.exists()
        assert loader.registry is not None

    def test_custom_paths(self, tmp_path):
        pq_dir = tmp_path / "parquet"
        pq_dir.mkdir()
        manifest_dir = tmp_path / "manifests"
        manifest_dir.mkdir()
        loader = DataLoader(data_dir=pq_dir, manifest_dir=manifest_dir)
        assert loader.data_dir == pq_dir

    def test_missing_data_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DataLoader(data_dir=tmp_path / "nonexistent")
