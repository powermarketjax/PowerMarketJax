# Vendored test from PowerZooJax.
# Source        : tests/data/test_ausgrid_utils.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
"""Tests for powermarketjax.data.ausgrid_utils."""

from __future__ import annotations

from typing import List

import pandas as pd
import pytest

from powermarketjax.data.ausgrid_utils import (
    AUSGRID_FEEDER_POOLS,
    AUSGRID_TRAIN_POOL,
    AUSGRID_ZONE_HOLDOUT_POOL,
    _max_consecutive_gap,
    filter_ausgrid_role_days,
    get_ausgrid_split,
    get_feeder_substations,
    select_full_coverage_substations,
)
from powermarketjax.data.splits import (
    AUSGRID_TRAIN_START,
    AUSGRID_TRAIN_END,
    AUSGRID_IID_START,
    AUSGRID_IID_END,
    AUSGRID_SUMMER_START,
    AUSGRID_SUMMER_END,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_full_df(
    substations: List[str],
    start: str,
    end: str,
    resolution: str = "15min",
) -> pd.DataFrame:
    """Build a gapless multi-substation DataFrame."""
    idx = pd.date_range(start=start, end=end, freq=resolution)
    rows = []
    for sub in substations:
        for ts in idx:
            rows.append({"datetime": ts, "region": sub, "load_mw": 1.0})
    return pd.DataFrame(rows)


def _make_gapped_df(
    substation: str,
    start: str,
    end: str,
    gap_positions: List[int],
    resolution: str = "15min",
) -> pd.DataFrame:
    """Build a single-substation DataFrame with gaps at specific positions."""
    idx = list(pd.date_range(start=start, end=end, freq=resolution))
    gap_set = set(gap_positions)
    rows = [
        {"datetime": ts, "region": substation, "load_mw": 1.0}
        for i, ts in enumerate(idx)
        if i not in gap_set
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# select_full_coverage_substations
# ---------------------------------------------------------------------------

class TestSelectFullCoverage:

    def test_gapless_passes(self):
        df = _make_full_df(["Sub_A", "Sub_B"], "2024-05-01", "2024-05-03")
        result = select_full_coverage_substations(df, "2024-05-01", "2024-05-03")
        assert "Sub_A" in result
        assert "Sub_B" in result

    def test_single_gap_within_tolerance_passes(self):
        df_full = _make_full_df(["Sub_A"], "2024-05-01", "2024-05-03")
        # Remove one row → 1-step gap, max_gap_steps=2 → should pass
        df_gapped = df_full[df_full.index != 5].reset_index(drop=True)
        result = select_full_coverage_substations(
            df_gapped, "2024-05-01", "2024-05-03", max_gap_steps=2
        )
        assert "Sub_A" in result

    def test_large_gap_excluded(self):
        # Build Sub_A with a 10-step gap starting at position 10
        df_gapped = _make_gapped_df(
            "Sub_X", "2024-05-01", "2024-05-03",
            gap_positions=list(range(10, 20)),
        )
        result = select_full_coverage_substations(
            df_gapped, "2024-05-01", "2024-05-03", max_gap_steps=2
        )
        assert "Sub_X" not in result

    def test_returns_sorted_list(self):
        df = _make_full_df(["Zeta", "Alpha", "Beta"], "2024-05-01", "2024-05-02")
        result = select_full_coverage_substations(df, "2024-05-01", "2024-05-02")
        assert result == sorted(result)

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame(columns=["datetime", "region", "load_mw"])
        result = select_full_coverage_substations(df, "2024-05-01", "2024-05-02")
        assert result == []

    def test_exactly_at_tolerance_boundary_passes(self):
        # A gap of exactly max_gap_steps should still pass (<=)
        df = _make_gapped_df(
            "Sub_B", "2024-05-01", "2024-05-03",
            gap_positions=[5, 6],  # 2-step gap
        )
        result = select_full_coverage_substations(
            df, "2024-05-01", "2024-05-03", max_gap_steps=2
        )
        assert "Sub_B" in result

    def test_one_over_tolerance_excluded(self):
        df = _make_gapped_df(
            "Sub_C", "2024-05-01", "2024-05-03",
            gap_positions=[5, 6, 7],  # 3-step gap
        )
        result = select_full_coverage_substations(
            df, "2024-05-01", "2024-05-03", max_gap_steps=2
        )
        assert "Sub_C" not in result


# ---------------------------------------------------------------------------
# _max_consecutive_gap (internal helper)
# ---------------------------------------------------------------------------

class TestMaxConsecutiveGap:

    def test_empty(self):
        assert _max_consecutive_gap([], pd.tseries.frequencies.to_offset("15min")) == 0

    def test_single(self):
        ts = [pd.Timestamp("2024-05-01 00:00")]
        assert _max_consecutive_gap(ts, pd.tseries.frequencies.to_offset("15min")) == 1

    def test_consecutive_run_of_three(self):
        base = pd.Timestamp("2024-05-01")
        freq = pd.tseries.frequencies.to_offset("15min")
        ts = [base + freq * i for i in range(3)]
        assert _max_consecutive_gap(ts, freq) == 3

    def test_two_separate_runs(self):
        base = pd.Timestamp("2024-05-01")
        freq = pd.tseries.frequencies.to_offset("15min")
        # run of 2, gap, run of 4
        ts = (
            [base + freq * i for i in range(2)]
            + [base + freq * (i + 10) for i in range(4)]
        )
        assert _max_consecutive_gap(ts, freq) == 4


# ---------------------------------------------------------------------------
# get_ausgrid_split
# ---------------------------------------------------------------------------

class TestGetAusgridSplit:

    @pytest.mark.parametrize("role", ["train", "iid", "summer_ood", "zone_holdout"])
    def test_valid_roles_return_tuple(self, role):
        result = get_ausgrid_split(role)
        assert isinstance(result, tuple)
        assert len(result) == 3
        start, end, subs = result
        assert isinstance(start, str)
        assert isinstance(end, str)
        assert isinstance(subs, list)
        assert len(subs) > 0

    def test_train_window_matches_constants(self):
        start, end, _ = get_ausgrid_split("train")
        assert start == AUSGRID_TRAIN_START
        assert end == AUSGRID_TRAIN_END

    def test_iid_window_matches_constants(self):
        start, end, _ = get_ausgrid_split("iid")
        assert start == AUSGRID_IID_START
        assert end == AUSGRID_IID_END

    def test_summer_ood_window_matches_constants(self):
        start, end, _ = get_ausgrid_split("summer_ood")
        assert start == AUSGRID_SUMMER_START
        assert end == AUSGRID_SUMMER_END

    def test_unknown_role_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown Ausgrid split role"):
            get_ausgrid_split("invalid_role")

    def test_train_uses_train_pool(self):
        _, _, subs = get_ausgrid_split("train")
        assert set(subs) == set(AUSGRID_TRAIN_POOL)

    def test_zone_holdout_uses_holdout_pool(self):
        _, _, subs = get_ausgrid_split("zone_holdout")
        assert set(subs) == set(AUSGRID_ZONE_HOLDOUT_POOL)

    def test_train_and_zone_holdout_pools_are_disjoint(self):
        _, _, train_subs = get_ausgrid_split("train")
        _, _, holdout_subs = get_ausgrid_split("zone_holdout")
        assert set(train_subs).isdisjoint(set(holdout_subs))


class TestFilterAusgridRoleDays:

    def test_train_and_iid_partition_days(self):
        idx = pd.date_range("2024-05-01", periods=8 * 48, freq="30min", tz="UTC")
        df = pd.DataFrame(
            {"datetime": idx, "region": "Sub_A", "load.actual_mw": 1.0}
        )
        train_df = filter_ausgrid_role_days(df, "train")
        iid_df = filter_ausgrid_role_days(df, "iid")
        assert len(train_df) > 0
        assert len(iid_df) > 0
        train_days = set(
            pd.to_datetime(train_df["datetime"], utc=True)
            .dt.tz_convert("Australia/Sydney")
            .dt.strftime("%Y-%m-%d")
        )
        iid_days = set(
            pd.to_datetime(iid_df["datetime"], utc=True)
            .dt.tz_convert("Australia/Sydney")
            .dt.strftime("%Y-%m-%d")
        )
        assert train_days.isdisjoint(iid_days)

    def test_non_train_iid_roles_passthrough(self):
        idx = pd.date_range("2024-05-01", periods=16, freq="30min", tz="UTC")
        df = pd.DataFrame(
            {"datetime": idx, "region": "Sub_A", "load.actual_mw": 1.0}
        )
        out = filter_ausgrid_role_days(df, "summer_ood")
        pd.testing.assert_frame_equal(out.reset_index(drop=True), df.reset_index(drop=True))


# ---------------------------------------------------------------------------
# AUSGRID_FEEDER_POOLS structure
# ---------------------------------------------------------------------------

class TestFeederPools:

    def test_three_feeders_defined(self):
        assert set(AUSGRID_FEEDER_POOLS) == {"feeder_A", "feeder_B", "feeder_C"}

    def test_each_feeder_has_train_and_holdout(self):
        for feeder, pool in AUSGRID_FEEDER_POOLS.items():
            assert "train" in pool, f"{feeder} missing 'train'"
            assert "zone_holdout" in pool, f"{feeder} missing 'zone_holdout'"

    def test_no_overlap_within_feeder(self):
        for feeder, pool in AUSGRID_FEEDER_POOLS.items():
            train_set = set(pool["train"])
            holdout_set = set(pool["zone_holdout"])
            assert train_set.isdisjoint(holdout_set), (
                f"{feeder}: train and zone_holdout share substations"
            )

    def test_get_feeder_substations(self):
        subs = get_feeder_substations("feeder_A", "train")
        assert subs == AUSGRID_FEEDER_POOLS["feeder_A"]["train"]

    def test_get_feeder_substations_invalid_feeder_raises(self):
        with pytest.raises(KeyError):
            get_feeder_substations("feeder_Z", "train")

    def test_get_feeder_substations_invalid_role_raises(self):
        with pytest.raises(KeyError):
            get_feeder_substations("feeder_A", "test_ood")
