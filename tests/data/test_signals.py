# Vendored test from PowerZooJax.
# Source        : tests/data/test_signals.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
"""Tests for powermarketjax.data.signals — semantic signal constants."""

import pytest

from powermarketjax.data import signals as S


class TestSignalConstants:
    """All signal constants exist and follow naming convention."""

    EXPECTED_LOAD = [
        S.LOAD_ACTUAL_MW,
        S.LOAD_FORECAST_DA_MW,
        S.LOAD_FORECAST_P10_MW,
        S.LOAD_FORECAST_P50_MW,
        S.LOAD_FORECAST_P90_MW,
    ]

    EXPECTED_RENEWABLE = [
        S.SOLAR_AVAILABLE_MW,
        S.WIND_AVAILABLE_MW,
    ]

    EXPECTED_MARKET = [
        S.MARKET_MID_PRICE_APX,
        S.MARKET_MID_PRICE_N2EX,
        S.MARKET_MID_VOLUME_APX,
        S.MARKET_MID_VOLUME_N2EX,
    ]

    EXPECTED_DC = [
        S.DC_CPU_UTIL,
        S.DC_MEM_UTIL,
        S.DC_NET_IN,
        S.DC_NET_OUT,
        S.DC_DISK_IO,
        S.DC_POWER_MW,
    ]

    EXPECTED_GPU = [
        S.DC_GPU_UTIL,
        S.DC_GPU_MEM_UTIL,
        S.DC_CYCLES_PER_INST,
        S.DC_ASSIGNED_MEM,
    ]

    @pytest.mark.parametrize("sig", EXPECTED_LOAD + EXPECTED_RENEWABLE)
    def test_power_system_signals_naming(self, sig):
        """Power system signals follow domain.metric pattern."""
        assert "." in sig
        domain, _ = sig.split(".", 1)
        assert domain in {"load", "solar", "wind"}

    @pytest.mark.parametrize("sig", EXPECTED_MARKET)
    def test_market_signals_naming(self, sig):
        assert sig.startswith("market.")

    @pytest.mark.parametrize("sig", EXPECTED_DC + EXPECTED_GPU)
    def test_datacenter_signals_naming(self, sig):
        assert sig.startswith("datacenter.")

    def test_weather_signal(self):
        assert S.TEMPERATURE_OUTDOOR_C == "weather.temperature_c"

    def test_index_columns(self):
        assert S.REGION == "region"
        assert S.DATETIME == "datetime"
        assert S.ISSUE_TIME == "issue_time"
        assert S.TARGET_TIME == "target_time"

    def test_data_shape_types(self):
        assert S.ACTUAL_SERIES == "actual_series"
        assert S.FORECAST_PANEL == "forecast_panel"

    def test_time_modes(self):
        assert S.TIME_MODE_CALENDAR == "calendar"
        assert S.TIME_MODE_PROFILE == "profile"

    def test_legacy_column_map_exists(self):
        assert isinstance(S._LEGACY_COLUMN_MAP, dict)
        assert len(S._LEGACY_COLUMN_MAP) > 0
        assert S._LEGACY_COLUMN_MAP["Actual"] == S.LOAD_ACTUAL_MW
        assert S._LEGACY_COLUMN_MAP["DAForecast"] == S.LOAD_FORECAST_DA_MW
        assert S._LEGACY_COLUMN_MAP["ActualDemand"] == S.LOAD_ACTUAL_MW
        assert S._LEGACY_COLUMN_MAP["Solar"] == S.SOLAR_AVAILABLE_MW
