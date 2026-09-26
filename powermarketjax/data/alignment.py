# Vendored from PowerZooJax. Do not edit to track upstream;
# this file is now this repository's code.
# Source        : powerzoojax/data/alignment.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
# Upstream licence: MIT, Copyright (c) 2026 PowerZooJax Contributors.
"""Time alignment engine for heterogeneous data sources.

Two modes:
* **calendar** -- data has real-world timestamps.  An optional *offset*
  shifts the data timeline onto the simulation timeline.
* **profile** -- data is a short, repeatable pattern (e.g. 8-day DC
  trace).  It is tiled cyclically to cover the simulation window.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
import numpy as np
import pandas as pd


class TimeAligner:
    """Map raw-data timestamps onto a unified simulation timeline."""

    # ------------------------------------------------------------------
    # Calendar mode (pandas — involves datetime comparison/filtering)
    # ------------------------------------------------------------------

    @staticmethod
    def align_calendar(
        df: pd.DataFrame,
        sim_start: pd.Timestamp,
        sim_end: pd.Timestamp,
        align_from: Optional[pd.Timestamp] = None,
        time_col: str = "datetime",
    ) -> pd.DataFrame:
        """Shift a calendar-mode dataset onto ``[sim_start, sim_end]``."""
        out = df.copy()

        has_col = time_col in out.columns
        if not has_col and isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index()
            if time_col not in out.columns and "index" in out.columns:
                out = out.rename(columns={"index": time_col})

        if time_col not in out.columns:
            return out

        out[time_col] = pd.to_datetime(out[time_col], utc=True)

        if align_from is not None:
            align_from = pd.Timestamp(align_from)
            if align_from.tzinfo is None:
                align_from = align_from.tz_localize("UTC")
            _sim_start = pd.Timestamp(sim_start)
            if _sim_start.tzinfo is None:
                _sim_start = _sim_start.tz_localize("UTC")
            offset = _sim_start - align_from
            out[time_col] = out[time_col] + offset

        _sim_start_utc = (
            pd.Timestamp(sim_start, tz="UTC")
            if pd.Timestamp(sim_start).tzinfo is None
            else pd.Timestamp(sim_start)
        )
        _sim_end_utc = (
            pd.Timestamp(sim_end, tz="UTC")
            if pd.Timestamp(sim_end).tzinfo is None
            else pd.Timestamp(sim_end)
        )
        _sim_end_inclusive = (
            _sim_end_utc.normalize()
            + pd.Timedelta(days=1)
            - pd.Timedelta(seconds=1)
        )

        mask = (out[time_col] >= _sim_start_utc) & (
            out[time_col] <= _sim_end_inclusive
        )
        out = out.loc[mask].reset_index(drop=True)
        return out

    # ------------------------------------------------------------------
    # Profile mode — uses jnp.tile for JAX-native tiling
    # ------------------------------------------------------------------

    @staticmethod
    def align_profile(
        df: pd.DataFrame,
        sim_start: pd.Timestamp,
        sim_end: pd.Timestamp,
        resolution: str,
        time_col: str = "datetime",
    ) -> pd.DataFrame:
        """Tile a profile cyclically to cover ``[sim_start, sim_end]``.

        Uses ``jnp.tile`` so the tiled numeric data stays on the JAX
        default device (GPU when available).
        """
        sim_start = pd.Timestamp(sim_start)
        sim_end = pd.Timestamp(sim_end)
        if sim_start.tzinfo is None:
            sim_start = sim_start.tz_localize("UTC")
        if sim_end.tzinfo is None:
            sim_end = sim_end.tz_localize("UTC")

        sim_end_inclusive = (
            sim_end.normalize()
            + pd.Timedelta(days=1)
            - pd.Timedelta(seconds=1)
        )
        sim_index = pd.date_range(
            start=sim_start, end=sim_end_inclusive, freq=resolution
        )
        n_needed = len(sim_index)

        value_cols = [c for c in df.columns if c != time_col]
        values = np.asarray(df[value_cols].values, dtype=np.float32)
        n_profile = len(values)

        if n_profile == 0:
            return pd.DataFrame({time_col: sim_index})

        n_full_tiles = n_needed // n_profile
        remainder = n_needed % n_profile

        jax_values = jnp.asarray(values)
        parts: list[jnp.ndarray] = []
        if n_full_tiles > 0:
            parts.append(jnp.tile(jax_values, (n_full_tiles, 1)))
        if remainder > 0:
            parts.append(jax_values[:remainder])

        tiled = (
            jnp.concatenate(parts, axis=0)
            if parts
            else jax_values[:0]
        )

        tiled_np = np.asarray(tiled)
        result = pd.DataFrame(tiled_np, columns=value_cols)
        result[time_col] = sim_index[: len(result)]
        return result

    # ------------------------------------------------------------------
    # JAX-native profile tiling (returns jnp.ndarray directly)
    # ------------------------------------------------------------------

    @staticmethod
    def tile_profile_jax(
        values: jnp.ndarray,
        n_needed: int,
    ) -> jnp.ndarray:
        """Tile a profile array to length *n_needed* using ``jnp.tile``.

        Parameters
        ----------
        values : (n_profile, n_signals) JAX array.
        n_needed : target number of time steps.

        Returns
        -------
        (n_needed, n_signals) JAX array, dtype float32.
        """
        n_profile = values.shape[0]
        if n_profile == 0:
            return jnp.zeros((n_needed, values.shape[1]), dtype=jnp.float32)

        n_full_tiles = n_needed // n_profile
        remainder = n_needed % n_profile
        parts: list[jnp.ndarray] = []
        if n_full_tiles > 0:
            parts.append(jnp.tile(values, (n_full_tiles, 1)))
        if remainder > 0:
            parts.append(values[:remainder])

        if not parts:
            return jnp.zeros((n_needed, values.shape[1]), dtype=jnp.float32)
        return jnp.concatenate(parts, axis=0).astype(jnp.float32)

    # ------------------------------------------------------------------
    # Convenience dispatcher
    # ------------------------------------------------------------------

    @classmethod
    def align(
        cls,
        df: pd.DataFrame,
        *,
        time_mode: str,
        sim_start: pd.Timestamp,
        sim_end: pd.Timestamp,
        align_from: Optional[pd.Timestamp] = None,
        resolution: str = "30min",
        time_col: str = "datetime",
    ) -> pd.DataFrame:
        """Dispatch to the correct alignment strategy."""
        if time_mode == "profile":
            return cls.align_profile(
                df, sim_start, sim_end, resolution, time_col
            )
        return cls.align_calendar(
            df, sim_start, sim_end, align_from, time_col
        )
