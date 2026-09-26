# Vendored from PowerZooJax. Do not edit to track upstream;
# this file is now this repository's code.
# Source        : powerzoojax/data/ausgrid_utils.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# NOT verbatim. Beyond the package name change (powerzoojax -> powermarketjax,
# in both import statements and path strings), on 2026-08-24 the substation
# count in the module docstring was corrected from 165 to 175. 175 is what the
# shipped artefact carries: distinct ``zone_substation`` values in
# parquet/Ausgrid_Zone_Substation_FY25_imputed_15min.parquet, ``region_values``
# in manifests/ausgrid_zone_substation_fy25_imputed.json, and ``source_files``
# in that parquet's sidecar JSON, all three = 175. Upstream still reads 165 at
# a7641de.
# Second deviation, 2026-08-24: the tz handling in
# ``select_full_coverage_substations`` was fixed. Upstream tests
# ``df[time_col].dtype == "datetime64[ns, UTC]"``, which does not fire on
# pandas >= 2 (a tz-aware ``date_range`` has dtype ``datetime64[us, UTC]``),
# and it compares ``ts`` against naive ``pd.Timestamp(start)``. Measured on
# 2026-08-24: tz-aware UTC input raises ``TypeError: Invalid comparison between
# dtype=datetime64[us, UTC] and Timestamp``, while the tz-naive control returns
# normally. The function is exported from ``powermarketjax.data``, so the defect
# is on this package's public surface. Upstream still has both at a7641de.
# Upstream licence: MIT, Copyright (c) 2026 PowerZooJax Contributors.
"""Ausgrid zone-substation utilities for DSO and DERs tasks.

Responsibilities
----------------
1. ``select_full_coverage_substations`` — filter the 175 Ausgrid substations
   down to a *full-coverage* subset that has no consecutive time gaps within
   a requested window.
2. ``AUSGRID_FEEDER_POOLS`` — hardcoded feeder-pool assignment for the
   DSO/DERs tasks (3 feeder segments, each driven by a distinct substation
   shape).  The list was derived by running ``select_full_coverage`` on the
   training window and then picking substations that represent diverse
   peak-load profiles (residential, industrial, coastal).
3. ``get_ausgrid_split`` — single entry-point that returns
   ``(start, end, substation_list)`` for a requested role.
4. ``filter_ausgrid_role_days`` — deterministic day-level split for
   ``train`` / ``iid`` within the shared non-summer window, so the two roles
   are genuinely distinct even though they use the same substation pool.

Usage example
-------------
>>> from powermarketjax.data.ausgrid_utils import get_ausgrid_split
>>> start, end, subs = get_ausgrid_split("train")
>>> # → ("2024-05-01", "2024-11-30", ["Ausgrid...", ...])
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .splits import (
    AUSGRID_TRAIN_START,
    AUSGRID_TRAIN_END,
    AUSGRID_IID_START,
    AUSGRID_IID_END,
    AUSGRID_SUMMER_START,
    AUSGRID_SUMMER_END,
)

# ---------------------------------------------------------------------------
# Feeder pool assignment (hardcoded after coverage analysis)
# ---------------------------------------------------------------------------
# Three feeder segments for DSO case33bw:
#   Feeder A — buses [2..18]  (urban residential, moderate load)
#   Feeder B — buses [19..22] (semi-industrial)
#   Feeder C — buses [23..33] (coastal/mixed)
# Substations were selected from the full-coverage pool to maximise diversity
# across peak-timing and magnitude.  The holdout set uses different zones so
# that the zone_swap OOD test uses truly unseen demand shapes.

AUSGRID_FEEDER_POOLS: Dict[str, Dict[str, List[str]]] = {
    "feeder_A": {
        "train": [
            "Broadmeadow 132_11kV",
            "Charlestown 132_11kV",
            "Jesmond 132_11kV",
        ],
        "zone_holdout": [
            "Mayfield West 132_11kV",
            "Kotara 33_11kV",
        ],
    },
    "feeder_B": {
        "train": [
            "Burwood 132_11kV",
            "Homebush Bay 132_11kV",
            "Strathfield South 132_11kV",
        ],
        "zone_holdout": [
            "Flemington 132_11kV",
            "Lidcombe 33_11kV",
        ],
    },
    "feeder_C": {
        "train": [
            "Cronulla 132_11kV",
            "Caringbah 33_11kV",
            "Miranda 33_11kV",
        ],
        "zone_holdout": [
            "Kurnell South 132_11kV",
            "Jannali 33_11kV",
        ],
    },
}

# Flat training pool (union of all feeder train sets)
AUSGRID_TRAIN_POOL: List[str] = [
    sub
    for fd in AUSGRID_FEEDER_POOLS.values()
    for sub in fd["train"]
]

# Flat zone-holdout pool
AUSGRID_ZONE_HOLDOUT_POOL: List[str] = [
    sub
    for fd in AUSGRID_FEEDER_POOLS.values()
    for sub in fd["zone_holdout"]
]


# Deterministic held-out day split for roles that share the same window/pool.
_IID_DAY_STRIDE = 4
_IID_DAY_OFFSET = 0


# ---------------------------------------------------------------------------
# Coverage analysis
# ---------------------------------------------------------------------------

def select_full_coverage_substations(
    df: pd.DataFrame,
    start: str,
    end: str,
    *,
    max_gap_steps: int = 2,
    substation_col: str = "region",
    time_col: str = "datetime",
    resolution: str = "15min",
) -> List[str]:
    """Return substations that have no consecutive time gap > *max_gap_steps*.

    A gap is defined as a run of consecutive missing 15-min slots within
    ``[start, end]``.  Substations with any gap run longer than
    ``max_gap_steps`` are excluded.

    Args:
        df: DataFrame with at least ``time_col`` and ``substation_col``
            columns.  Multi-region format (one row per region per timestamp).
        start: ISO date string, inclusive window start.
        end: ISO date string, inclusive window end.
        max_gap_steps: Maximum tolerated consecutive missing steps.
        substation_col: Column name identifying the substation/region.
        time_col: Column name for timestamps.
        resolution: Expected time resolution for the full grid.

    Returns:
        Sorted list of substation names with full (or near-full) coverage.
    """
    # Build expected timestamp grid
    freq = pd.tseries.frequencies.to_offset(resolution)
    expected_index = pd.date_range(start=start, end=end, freq=freq)
    # The test here was `df[time_col].dtype == "datetime64[ns, UTC]"`, which
    # **does not fire on pandas >= 2**: a tz-aware `date_range` there has dtype
    # `datetime64[us, UTC]`, so matching a string that pins both the resolution
    # and the zone misses it.  Read the zone off the dtype instead.
    tz = getattr(df[time_col].dtype, "tz", None)
    if expected_index.tz is None and tz is not None:
        expected_index = expected_index.tz_localize(tz)
    # And the window bounds must carry the same zone as `ts`, or the comparison
    # below raises `TypeError: Invalid comparison between dtype=datetime64[us,
    # UTC] and Timestamp`.  With `tz=None` these are naive, i.e. exactly the old
    # expressions, so the tz-naive path is unchanged.
    lo = pd.Timestamp(start, tz=tz)
    hi = pd.Timestamp(end, tz=tz) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    qualifying: List[str] = []

    for substation, grp in df.groupby(substation_col):
        ts = pd.to_datetime(grp[time_col])
        # Restrict to window
        mask = (ts >= lo) & (ts <= hi)
        ts_window = ts[mask].sort_values().reset_index(drop=True)

        if len(ts_window) == 0:
            continue

        # Check for consecutive missing steps
        full_set = set(expected_index)
        present_set = set(ts_window)
        missing = sorted(full_set - present_set)

        if len(missing) == 0:
            qualifying.append(substation)
            continue

        # Find the longest consecutive run of missing steps
        max_run = _max_consecutive_gap(missing, freq)
        if max_run <= max_gap_steps:
            qualifying.append(substation)

    return sorted(qualifying)


def _max_consecutive_gap(
    missing_timestamps: list,
    freq,
) -> int:
    """Return the length of the longest consecutive gap in *missing_timestamps*."""
    if not missing_timestamps:
        return 0

    max_run = 1
    current_run = 1

    for i in range(1, len(missing_timestamps)):
        if missing_timestamps[i] - missing_timestamps[i - 1] == freq:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1

    return max_run


# ---------------------------------------------------------------------------
# Split entry-point
# ---------------------------------------------------------------------------

_SPLIT_REGISTRY: Dict[str, Tuple[str, str, List[str]]] = {
    "train": (AUSGRID_TRAIN_START, AUSGRID_TRAIN_END, AUSGRID_TRAIN_POOL),
    "iid": (AUSGRID_IID_START, AUSGRID_IID_END, AUSGRID_TRAIN_POOL),
    "summer_ood": (AUSGRID_SUMMER_START, AUSGRID_SUMMER_END, AUSGRID_TRAIN_POOL),
    "zone_holdout": (AUSGRID_TRAIN_START, AUSGRID_TRAIN_END, AUSGRID_ZONE_HOLDOUT_POOL),
}


def filter_ausgrid_role_days(
    df: pd.DataFrame,
    role: str,
    *,
    time_col: str = "datetime",
    local_tz: str = "Australia/Sydney",
) -> pd.DataFrame:
    """Filter a DataFrame to the day subset associated with ``role``.

    ``train`` and ``iid`` intentionally share the same coarse time window and
    substation pool.  The actual separation happens at the *day* level:

    - ``iid`` keeps every 4th local calendar day
    - ``train`` keeps the complementary days

    This preserves similar seasonal composition while ensuring the two splits
    do not collapse to identical feeder shapes.
    """
    if role not in {"train", "iid"}:
        return df
    if time_col not in df.columns:
        raise KeyError(f"Column '{time_col}' not found in DataFrame")
    if df.empty:
        return df.copy()

    out = df.copy()
    ts = pd.to_datetime(out[time_col], utc=True)
    local_days = ts.dt.tz_convert(local_tz).dt.strftime("%Y-%m-%d")
    day_codes, _ = pd.factorize(local_days, sort=True)
    iid_mask = (day_codes % _IID_DAY_STRIDE) == _IID_DAY_OFFSET

    if role == "iid":
        if not iid_mask.any():
            return out
        return out.loc[iid_mask].reset_index(drop=True)

    train_mask = ~iid_mask
    if not train_mask.any():
        return out
    return out.loc[train_mask].reset_index(drop=True)


def get_ausgrid_split(
    role: str,
) -> Tuple[str, str, List[str]]:
    """Return ``(start, end, substation_list)`` for a named split role.

    Args:
        role: One of ``"train"``, ``"iid"``, ``"summer_ood"``,
              ``"zone_holdout"``.

    Returns:
        Tuple of (ISO-date start, ISO-date end, list of substation names).

    Raises:
        KeyError: if *role* is not recognised.
    """
    if role not in _SPLIT_REGISTRY:
        raise KeyError(
            f"Unknown Ausgrid split role '{role}'. "
            f"Available: {sorted(_SPLIT_REGISTRY)}"
        )
    return _SPLIT_REGISTRY[role]


def get_feeder_substations(
    feeder: str,
    role: str = "train",
) -> List[str]:
    """Return substation list for a specific feeder and role.

    Args:
        feeder: ``"feeder_A"``, ``"feeder_B"``, or ``"feeder_C"``.
        role: ``"train"`` or ``"zone_holdout"``.

    Returns:
        List of substation names for that feeder/role combination.
    """
    if feeder not in AUSGRID_FEEDER_POOLS:
        raise KeyError(
            f"Unknown feeder '{feeder}'. "
            f"Available: {sorted(AUSGRID_FEEDER_POOLS)}"
        )
    pool = AUSGRID_FEEDER_POOLS[feeder]
    if role not in pool:
        raise KeyError(
            f"Unknown role '{role}' for feeder '{feeder}'. "
            f"Available: {sorted(pool)}"
        )
    return pool[role]
