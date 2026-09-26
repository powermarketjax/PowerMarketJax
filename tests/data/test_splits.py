# Vendored test from PowerZooJax.
# Source        : tests/data/test_splits.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
"""Tests for powermarketjax.data.splits — frozen train/iid/ood window constants."""

import datetime

import pytest

from powermarketjax.data.splits import (
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
    assert_no_overlap,
    _parse,
)
from powermarketjax.data import (
    GB_TRAIN_START as PKG_GB_TRAIN_START,
    GB_IID_START as PKG_GB_IID_START,
    AUSGRID_TRAIN_START as PKG_AUSGRID_TRAIN_START,
    AUSGRID_SUMMER_START as PKG_AUSGRID_SUMMER_START,
    gb_windows as pkg_gb_windows,
    ausgrid_windows as pkg_ausgrid_windows,
)


class TestGBWindow:

    def test_constants_are_strings(self):
        for val in (GB_TRAIN_START, GB_TRAIN_END, GB_IID_START, GB_IID_END):
            assert isinstance(val, str)

    def test_constants_are_valid_iso_dates(self):
        for val in (GB_TRAIN_START, GB_TRAIN_END, GB_IID_START, GB_IID_END):
            d = datetime.date.fromisoformat(val)
            assert d.year >= 2020

    def test_train_before_iid(self):
        assert _parse(GB_TRAIN_END) < _parse(GB_IID_START)

    def test_no_overlap(self):
        assert_no_overlap(GB_TRAIN_START, GB_TRAIN_END, GB_IID_START, GB_IID_END)

    def test_train_window_positive_duration(self):
        assert _parse(GB_TRAIN_START) < _parse(GB_TRAIN_END)

    def test_iid_window_positive_duration(self):
        assert _parse(GB_IID_START) < _parse(GB_IID_END)

    def test_gb_windows_returns_tuple(self):
        train, iid = gb_windows()
        assert train == (GB_TRAIN_START, GB_TRAIN_END)
        assert iid == (GB_IID_START, GB_IID_END)


class TestAusgridWindow:

    def test_constants_are_strings(self):
        for val in (
            AUSGRID_TRAIN_START, AUSGRID_TRAIN_END,
            AUSGRID_IID_START, AUSGRID_IID_END,
            AUSGRID_SUMMER_START, AUSGRID_SUMMER_END,
        ):
            assert isinstance(val, str)

    def test_constants_are_valid_iso_dates(self):
        for val in (
            AUSGRID_TRAIN_START, AUSGRID_TRAIN_END,
            AUSGRID_IID_START, AUSGRID_IID_END,
            AUSGRID_SUMMER_START, AUSGRID_SUMMER_END,
        ):
            d = datetime.date.fromisoformat(val)
            assert d.year >= 2024

    def test_train_window_positive_duration(self):
        assert _parse(AUSGRID_TRAIN_START) < _parse(AUSGRID_TRAIN_END)

    def test_summer_after_train(self):
        assert _parse(AUSGRID_TRAIN_END) < _parse(AUSGRID_SUMMER_START)

    def test_summer_window_positive_duration(self):
        assert _parse(AUSGRID_SUMMER_START) < _parse(AUSGRID_SUMMER_END)

    def test_ausgrid_windows_returns_tuple(self):
        train, iid, summer = ausgrid_windows()
        assert train == (AUSGRID_TRAIN_START, AUSGRID_TRAIN_END)
        assert iid == (AUSGRID_IID_START, AUSGRID_IID_END)
        assert summer == (AUSGRID_SUMMER_START, AUSGRID_SUMMER_END)


class TestPackageExports:
    """Ensure data.__init__ re-exports splits correctly."""

    def test_gb_train_start_exported(self):
        assert PKG_GB_TRAIN_START == GB_TRAIN_START

    def test_gb_iid_start_exported(self):
        assert PKG_GB_IID_START == GB_IID_START

    def test_ausgrid_train_start_exported(self):
        assert PKG_AUSGRID_TRAIN_START == AUSGRID_TRAIN_START

    def test_ausgrid_summer_start_exported(self):
        assert PKG_AUSGRID_SUMMER_START == AUSGRID_SUMMER_START

    def test_gb_windows_function_exported(self):
        assert pkg_gb_windows() == gb_windows()

    def test_ausgrid_windows_function_exported(self):
        assert pkg_ausgrid_windows() == ausgrid_windows()


class TestAssertNoOverlap:

    def test_non_overlapping_passes(self):
        assert_no_overlap("2025-01-01", "2025-06-30", "2025-07-01", "2025-12-31")

    def test_overlapping_raises(self):
        with pytest.raises(AssertionError):
            assert_no_overlap("2025-01-01", "2025-07-01", "2025-06-30", "2025-12-31")

    def test_adjacent_passes(self):
        # end of A strictly before start of B
        assert_no_overlap("2025-01-01", "2025-06-30", "2025-07-01", "2025-12-31")
