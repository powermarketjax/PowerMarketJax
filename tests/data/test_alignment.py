# Vendored test from PowerZooJax.
# Source        : tests/data/test_alignment.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
"""Tests for powermarketjax.data.alignment — TimeAligner."""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from powermarketjax.data.alignment import TimeAligner


class TestAlignCalendar:

    def test_basic_filter(self):
        dates = pd.date_range("2024-01-01", periods=100, freq="30min", tz="UTC")
        df = pd.DataFrame({
            "datetime": dates,
            "value": np.arange(100, dtype=np.float64),
        })
        result = TimeAligner.align_calendar(
            df,
            sim_start=pd.Timestamp("2024-01-01 02:00", tz="UTC"),
            sim_end=pd.Timestamp("2024-01-01 05:00", tz="UTC"),
        )
        assert len(result) > 0
        assert result["datetime"].min() >= pd.Timestamp("2024-01-01 02:00", tz="UTC")

    def test_with_offset(self):
        dates = pd.date_range("2020-06-01", periods=48, freq="30min", tz="UTC")
        df = pd.DataFrame({
            "datetime": dates,
            "load": np.ones(48),
        })
        result = TimeAligner.align_calendar(
            df,
            sim_start=pd.Timestamp("2024-01-01", tz="UTC"),
            sim_end=pd.Timestamp("2024-01-01 23:30", tz="UTC"),
            align_from=pd.Timestamp("2020-06-01", tz="UTC"),
        )
        assert len(result) == 48
        assert result["datetime"].min() == pd.Timestamp("2024-01-01", tz="UTC")

    def test_naive_timestamps(self):
        """Naive (no tz) timestamps should work."""
        dates = pd.date_range("2024-01-01", periods=10, freq="30min")
        df = pd.DataFrame({"datetime": dates, "v": range(10)})
        result = TimeAligner.align_calendar(
            df,
            sim_start=pd.Timestamp("2024-01-01"),
            sim_end=pd.Timestamp("2024-01-01"),
        )
        assert len(result) > 0

    def test_empty_result(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="30min", tz="UTC")
        df = pd.DataFrame({"datetime": dates, "v": range(10)})
        result = TimeAligner.align_calendar(
            df,
            sim_start=pd.Timestamp("2025-01-01", tz="UTC"),
            sim_end=pd.Timestamp("2025-01-02", tz="UTC"),
        )
        assert len(result) == 0

    def test_no_datetime_column(self):
        df = pd.DataFrame({"value": [1, 2, 3]})
        result = TimeAligner.align_calendar(
            df,
            sim_start=pd.Timestamp("2024-01-01"),
            sim_end=pd.Timestamp("2024-01-02"),
        )
        assert len(result) == 3


class TestAlignProfile:

    def test_tile_shorter_profile(self):
        """Profile shorter than sim window should be tiled."""
        df = pd.DataFrame({
            "datetime": pd.date_range("2020-01-01", periods=3, freq="300s"),
            "cpu": [0.1, 0.5, 0.9],
        })
        result = TimeAligner.align_profile(
            df,
            sim_start=pd.Timestamp("2024-01-01"),
            sim_end=pd.Timestamp("2024-01-01 00:30:00"),
            resolution="300s",
        )
        assert len(result) > 3
        assert "cpu" in result.columns
        assert "datetime" in result.columns
        values = result["cpu"].values
        assert np.isclose(values[0], 0.1)
        assert np.isclose(values[1], 0.5)
        assert np.isclose(values[2], 0.9)
        assert np.isclose(values[3], 0.1)

    def test_empty_profile(self):
        df = pd.DataFrame({"datetime": pd.Series(dtype="datetime64[ns]"), "v": pd.Series(dtype=float)})
        result = TimeAligner.align_profile(
            df,
            sim_start=pd.Timestamp("2024-01-01"),
            sim_end=pd.Timestamp("2024-01-01 01:00"),
            resolution="300s",
        )
        assert "datetime" in result.columns

    def test_profile_exact_multiple(self):
        """Profile length is exact multiple of needed length."""
        df = pd.DataFrame({
            "datetime": pd.date_range("2020-01-01", periods=6, freq="300s"),
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        result = TimeAligner.align_profile(
            df,
            sim_start=pd.Timestamp("2024-01-01"),
            sim_end=pd.Timestamp("2024-01-01 00:29:59"),
            resolution="300s",
        )
        n = len(result)
        assert n >= 6
        for i in range(min(6, n)):
            assert np.isclose(result["v"].iloc[i], float(i % 6 + 1), atol=1e-5)


class TestTileProfileJax:

    def test_basic_tile(self):
        values = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = TimeAligner.tile_profile_jax(values, n_needed=7)
        assert result.shape == (7, 2)
        assert result.dtype == jnp.float32
        np.testing.assert_allclose(result[0], [1.0, 2.0])
        np.testing.assert_allclose(result[2], [1.0, 2.0])
        np.testing.assert_allclose(result[6], [1.0, 2.0])

    def test_exact_multiple(self):
        values = jnp.array([[1.0], [2.0], [3.0]])
        result = TimeAligner.tile_profile_jax(values, n_needed=6)
        assert result.shape == (6, 1)
        expected = jnp.array([[1.0], [2.0], [3.0], [1.0], [2.0], [3.0]])
        np.testing.assert_allclose(result, expected)

    def test_empty_profile(self):
        values = jnp.zeros((0, 3), dtype=jnp.float32)
        result = TimeAligner.tile_profile_jax(values, n_needed=10)
        assert result.shape == (10, 3)
        assert jnp.all(result == 0.0)

    def test_shorter_than_profile(self):
        values = jnp.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        result = TimeAligner.tile_profile_jax(values, n_needed=2)
        assert result.shape == (2, 1)
        np.testing.assert_allclose(result, jnp.array([[1.0], [2.0]]))


class TestAlignDispatcher:

    def test_dispatch_calendar(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="30min", tz="UTC")
        df = pd.DataFrame({"datetime": dates, "v": range(10)})
        result = TimeAligner.align(
            df,
            time_mode="calendar",
            sim_start=pd.Timestamp("2024-01-01", tz="UTC"),
            sim_end=pd.Timestamp("2024-01-01", tz="UTC"),
        )
        assert len(result) > 0

    def test_dispatch_profile(self):
        df = pd.DataFrame({
            "datetime": pd.date_range("2020-01-01", periods=3, freq="300s"),
            "v": [1.0, 2.0, 3.0],
        })
        result = TimeAligner.align(
            df,
            time_mode="profile",
            sim_start=pd.Timestamp("2024-01-01"),
            sim_end=pd.Timestamp("2024-01-01 00:30:00"),
            resolution="300s",
        )
        assert len(result) > 3
