# Vendored from PowerZooJax. Do not edit to track upstream;
# this file is now this repository's code.
# Source        : powerzoojax/data/signals.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
# Upstream licence: MIT, Copyright (c) 2026 PowerZooJax Contributors.
"""Semantic signal definitions for the PowerZooJax data layer.

Signals are the stable public contract between the data layer and
env/task/resource code.  External code should request data by signal
name, never by raw source column name.

Naming convention: ``domain.metric[_unit]``
"""

from __future__ import annotations

# Power system — load
LOAD_ACTUAL_MW = "load.actual_mw"
LOAD_FORECAST_DA_MW = "load.forecast_da_mw"
LOAD_FORECAST_P10_MW = "load.forecast_p10_mw"
LOAD_FORECAST_P50_MW = "load.forecast_p50_mw"
LOAD_FORECAST_P90_MW = "load.forecast_p90_mw"

# Power system — renewable generation
SOLAR_AVAILABLE_MW = "solar.available_mw"
WIND_AVAILABLE_MW = "wind.available_mw"

# Power system — market index data
MARKET_MID_PRICE_APX = "market.mid_price_apx"
MARKET_MID_PRICE_N2EX = "market.mid_price_n2ex"
MARKET_MID_VOLUME_APX = "market.mid_volume_apx"
MARKET_MID_VOLUME_N2EX = "market.mid_volume_n2ex"

# Data center — general
DC_CPU_UTIL = "datacenter.cpu_util"
DC_MEM_UTIL = "datacenter.mem_util"
DC_NET_IN = "datacenter.net_in"
DC_NET_OUT = "datacenter.net_out"
DC_DISK_IO = "datacenter.disk_io"
DC_POWER_MW = "datacenter.power_mw"

# Data center — GPU (Alibaba PAI GPU traces, etc.)
DC_GPU_UTIL = "datacenter.gpu_util"
DC_GPU_MEM_UTIL = "datacenter.gpu_mem_util"
DC_CYCLES_PER_INST = "datacenter.cycles_per_instruction"
DC_ASSIGNED_MEM = "datacenter.assigned_mem"

# Weather / environmental
TEMPERATURE_OUTDOOR_C = "weather.temperature_c"

# Canonical index columns
REGION = "region"
DATETIME = "datetime"
ISSUE_TIME = "issue_time"
TARGET_TIME = "target_time"

# Data shape types
ACTUAL_SERIES = "actual_series"
FORECAST_PANEL = "forecast_panel"

# Time modes (used in manifests)
TIME_MODE_CALENDAR = "calendar"
TIME_MODE_PROFILE = "profile"

# Legacy column name -> signal mapping (for backward compatibility)
_LEGACY_COLUMN_MAP: dict[str, str] = {
    # GB forecast/actual parquet (current raw names)
    "Actual": LOAD_ACTUAL_MW,
    "DAForecast": LOAD_FORECAST_DA_MW,
    # Older GB CSV-style names (same semantics)
    "ActualDemand": LOAD_ACTUAL_MW,
    "DAForecastDemand": LOAD_FORECAST_DA_MW,
    "Solar": SOLAR_AVAILABLE_MW,
    "Wind": WIND_AVAILABLE_MW,
    "Wind Offshore": WIND_AVAILABLE_MW,
    "Wind Onshore": WIND_AVAILABLE_MW,
}
