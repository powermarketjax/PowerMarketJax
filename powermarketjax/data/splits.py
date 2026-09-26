# Vendored from PowerZooJax. Do not edit to track upstream;
# this file is now this repository's code.
# Source        : powerzoojax/data/splits.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
# Upstream licence: MIT, Copyright (c) 2026 PowerZooJax Contributors.
"""Frozen train/iid/ood split definitions for all benchmark tasks.

These constants are the single source of truth for all time-window and
substation-pool definitions.  No task should hard-code dates or substation
names elsewhere.

GB window (GenCos, TSO):
    Train  : 2025-04-01 → 2025-12-31  (spring/summer/autumn/winter)
    IID    : 2026-01-01 → 2026-03-31  (held-out quarter, same distribution)
    OOD    : parametric perturbation axes (not a time range):
               demand_shift    load × 1.10
               renewable_shock wind × 0.30
               opponent_shift  different seed pool (GenCos only)
               load_stress     load × 1.15 (TSO only)
               renewable_surge wind × 1.50 (TSO only)

Ausgrid window (DSO, DERs):
    Train       : 2024-05-01 → 2024-11-30  (7 months, seasonal drift included)
    IID         : held-out days within same window (deterministic day-level split)
    Summer OOD  : 2024-12-01 → 2025-02-28  (high-temp season)
    Zone holdout: different substation pool (see ausgrid_utils.AUSGRID_FEEDER_POOLS)
"""

from __future__ import annotations

import datetime
from typing import Tuple

# ---------------------------------------------------------------------------
# GB main window
# ---------------------------------------------------------------------------

GB_TRAIN_START: str = "2025-04-01"
GB_TRAIN_END: str = "2025-12-31"

GB_IID_START: str = "2026-01-01"
GB_IID_END: str = "2026-03-31"

# ---------------------------------------------------------------------------
# Ausgrid window
# ---------------------------------------------------------------------------

AUSGRID_TRAIN_START: str = "2024-05-01"
AUSGRID_TRAIN_END: str = "2024-11-30"

# IID: same substation pool, held-out days (sampled by ausgrid_utils)
AUSGRID_IID_START: str = "2024-05-01"
AUSGRID_IID_END: str = "2024-11-30"

# Summer OOD: high-temperature season outside training window
AUSGRID_SUMMER_START: str = "2024-12-01"
AUSGRID_SUMMER_END: str = "2025-02-28"


# ---------------------------------------------------------------------------
# Validation helpers (used by tests and ausgrid_utils)
# ---------------------------------------------------------------------------

def _parse(date_str: str) -> datetime.date:
    return datetime.date.fromisoformat(date_str)


def gb_windows() -> Tuple[Tuple[str, str], Tuple[str, str]]:
    """Return ((train_start, train_end), (iid_start, iid_end))."""
    return (GB_TRAIN_START, GB_TRAIN_END), (GB_IID_START, GB_IID_END)


def ausgrid_windows() -> Tuple[Tuple[str, str], Tuple[str, str], Tuple[str, str]]:
    """Return (train_window, iid_window, summer_ood_window)."""
    return (
        (AUSGRID_TRAIN_START, AUSGRID_TRAIN_END),
        (AUSGRID_IID_START, AUSGRID_IID_END),
        (AUSGRID_SUMMER_START, AUSGRID_SUMMER_END),
    )


def assert_no_overlap(start_a: str, end_a: str, start_b: str, end_b: str) -> None:
    """Raise AssertionError if [start_a, end_a] and [start_b, end_b] overlap."""
    a_end = _parse(end_a)
    b_start = _parse(start_b)
    assert a_end < b_start, (
        f"Windows overlap: [{start_a}, {end_a}] overlaps [{start_b}, {end_b}]"
    )
