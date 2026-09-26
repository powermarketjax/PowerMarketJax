# Vendored from PowerZooJax. Do not edit to track upstream;
# this file is now this repository's code.
# Source        : powerzoojax/data/__init__.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Trimmed to this repository's scope -- NOT verbatim.
#   Removed: dc_microgrid_profiles (DC-microgrid / datacenter workload
#   profiles). Its four backing datasets (alibaba, google, azure traces) are
#   not vendored either -- they belong to the datacenter line, not to any of
#   this repository's five markets.
# Upstream licence: MIT, Copyright (c) 2026 PowerZooJax Contributors.
"""Data layer for benchmark profiles and split logic.

This package handles setup-time data work only:
- manifest-driven parquet loading
- time alignment and windowing
- real-data split definitions for GB / Ausgrid-style experiments
- controlled nonstationarity / OOD transforms

Outputs are regular arrays ready to be packed into `EnvParams`; nothing in
this package is meant to run inside jitted training inner loops.
"""

from .data_loader import DataLoader
from . import signals
from .manifest import DatasetManifest
from .registry import DatasetRegistry
from .alignment import TimeAligner
from . import splits
from .splits import (
    GB_TRAIN_START,
    GB_TRAIN_END,
    GB_IID_START,
    GB_IID_END,
    AUSGRID_TRAIN_START,
    AUSGRID_TRAIN_END,
    AUSGRID_IID_START,
    AUSGRID_IID_END,
    AUSGRID_SUMMER_START,
    AUSGRID_SUMMER_END,
    gb_windows,
    ausgrid_windows,
)
from . import ausgrid_utils
from .ausgrid_utils import (
    AUSGRID_FEEDER_POOLS,
    AUSGRID_TRAIN_POOL,
    AUSGRID_ZONE_HOLDOUT_POOL,
    get_ausgrid_split,
    get_feeder_substations,
    select_full_coverage_substations,
)
from .nonstationary import (
    EpisodeConfig,
    NonstationarySampler,
    apply_drift,
)

__all__ = [
    "DataLoader",
    "signals",
    "DatasetManifest",
    "DatasetRegistry",
    "TimeAligner",
    "splits",
    "GB_TRAIN_START",
    "GB_TRAIN_END",
    "GB_IID_START",
    "GB_IID_END",
    "AUSGRID_TRAIN_START",
    "AUSGRID_TRAIN_END",
    "AUSGRID_IID_START",
    "AUSGRID_IID_END",
    "AUSGRID_SUMMER_START",
    "AUSGRID_SUMMER_END",
    "gb_windows",
    "ausgrid_windows",
    "ausgrid_utils",
    "AUSGRID_FEEDER_POOLS",
    "AUSGRID_TRAIN_POOL",
    "AUSGRID_ZONE_HOLDOUT_POOL",
    "get_ausgrid_split",
    "get_feeder_substations",
    "select_full_coverage_substations",
    "EpisodeConfig",
    "NonstationarySampler",
    "apply_drift",
]
