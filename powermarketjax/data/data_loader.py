# Vendored from PowerZooJax. Do not edit to track upstream;
# this file is now this repository's code.
# Source        : powerzoojax/data/data_loader.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# NOT verbatim. Beyond the package rename (powerzoojax -> powermarketjax, in
# both import statements and path strings), on 2026-08-06
# ``_datetime_from_settlement`` was added and called from
# ``_apply_index_map``: gb_neso_demand stores settlement date + period rather
# than a timestamp, so it used to load with no time axis at all and any
# alignment window silently returned all 16 years.
# On 2026-08-24, four docstring/comment claims that measurement contradicted:
# ``load_signals`` says what it actually returns (a RangeIndex frame with a
# ``datetime`` column, which is why envs/day_ahead/demand.py:176 calls
# ``.set_index("datetime")`` itself); "alibaba" dropped from the source
# examples (not one of the seven sources the 14 manifests declare);
# ``make_trans_params`` is upstream-only and is no longer named as the
# destination; and the derived-expression comment now says that subtraction
# does work.
# Upstream licence: MIT, Copyright (c) 2026 PowerZooJax Contributors.
"""Data Loader for PowerZooJax benchmark data.

This module provides :class:`DataLoader`, the **single public data facade**
for PowerZooJax.  External code (envs, tasks, resources) should load data
exclusively through this class, using *semantic signal names* rather than
raw source column names.

Three loading APIs are available:

* **JAX API** (preferred for env construction): :meth:`load_jax_profiles`,
  :meth:`load_jax_array`.  Returns ``jnp.ndarray`` (float32) ready for
  ``EnvParams``.
* **Semantic API**: :meth:`load_signals`, :meth:`load_actual_series`,
  :meth:`load_forecast_panel`.  Returns ``pd.DataFrame``.
* **Legacy API** (backward compatibility): :meth:`load_data` with raw
  column names.

All loading happens at **setup-time** (CPU, pandas).  The JAX API converts
the final result to ``jnp.float32`` so it can reside on GPU for training.
"""

from __future__ import annotations

import json
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd

from . import signals as S
from .alignment import TimeAligner
from .manifest import DatasetManifest
from .registry import DatasetRegistry


_SETTLEMENT_COLS = {"settlement_date", "settlement_period"}


class DataLoader:
    """Benchmark data facade backed by parquet files and manifest metadata.

    Features:
    - JAX array output via :meth:`load_jax_profiles` (float32, GPU-ready)
    - Semantic signal loading via :meth:`load_signals`
    - Calendar / profile time-alignment across heterogeneous sources
    - Forecast-panel loading with ``issue_time`` / ``target_time``
    - Legacy column-based loading via :meth:`load_data`
    - Resampling, date filtering, multi-dataset merge
    """

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        manifest_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize DataLoader.

        Args:
            data_dir: Directory containing parquet files.
                If *None*, defaults to ``powermarketjax/data/parquet``.
            manifest_dir: Directory containing manifest JSON files.
                If *None*, defaults to ``powermarketjax/data/manifests``.
        """
        script_dir = Path(__file__).resolve().parent
        if data_dir is None:
            self.data_dir = script_dir / "parquet"
        else:
            self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}"
            )

        if manifest_dir is None:
            manifest_dir = script_dir / "manifests"
        self._registry = DatasetRegistry(Path(manifest_dir))

        self._metadata_cache: Dict[str, Dict] = {}
        self._column_index: Optional[Dict[str, List[str]]] = None

        self.data: Optional[pd.DataFrame] = None
        self.current_source: Optional[str] = None

    # ==================================================================
    # JAX API (preferred for env construction)
    # ==================================================================

    def load_jax_profiles(
        self,
        signals: List[str],
        *,
        source: Optional[str] = None,
        region: Optional[str] = None,
        start_date: Optional[Union[str, datetime, date]] = None,
        end_date: Optional[Union[str, datetime, date]] = None,
        resample: Optional[str] = None,
        time_alignment: Optional[Dict[str, str]] = None,
        interpolation: str = "linear",
        dtype: jnp.dtype = jnp.float32,
    ) -> jnp.ndarray:
        """Load semantic signals and return as a JAX array.

        This is the **intended entry-point for env construction**.  It calls
        :meth:`load_signals` internally, then converts numeric columns to
        a ``jnp.ndarray`` of shape ``(T, n_signals)``.  Intended, not yet in
        use: this repository has no call site for it outside its own tests.
        Env-side loading goes through :meth:`load_signals` instead -- three
        call sites, in ``envs/day_ahead/demand.py`` and
        ``envs/real_time/demand.py`` -- and converts to JAX itself.

        The result is a JAX array on the default device (GPU when available),
        ready to be packed into an ``EnvParams``.  Upstream named
        ``make_trans_params(case, load_profiles=...)`` here; that function is
        not vendored into this repository, so no such call exists.

        Args:
            signals: Semantic signal names (e.g.
                ``["load.actual_mw", "solar.available_mw"]``).
            source: Restrict lookup to a specific data source.
            region: Filter by region (e.g. ``"NSW1"``).
            start_date: Simulation time window start (inclusive).
            end_date: Simulation time window end (inclusive).
            resample: Target frequency (``"5min"``, ``"30min"``, ...).
            time_alignment: Per-signal calendar-shift overrides.
            interpolation: Interpolation method when resampling.
            dtype: JAX dtype for the output array.  Default ``jnp.float32``.

        Returns:
            ``jnp.ndarray`` of shape ``(T, n_signals)`` with columns
            ordered as *signals*.
        """
        df = self.load_signals(
            signals,
            source=source,
            region=region,
            start_date=start_date,
            end_date=end_date,
            resample=resample,
            time_alignment=time_alignment,
            interpolation=interpolation,
        )
        return self._df_to_jax(df, signals, dtype=dtype)

    def load_jax_array(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        dtype: jnp.dtype = jnp.float32,
    ) -> jnp.ndarray:
        """Convert a DataFrame (e.g. from :meth:`load_signals`) to JAX.

        Args:
            df: Source DataFrame.
            columns: Columns to extract (order preserved).  If *None*,
                all numeric columns are used.
            dtype: Target JAX dtype.

        Returns:
            ``jnp.ndarray`` of shape ``(n_rows, n_columns)``.
        """
        return self._df_to_jax(df, columns, dtype=dtype)

    @staticmethod
    def _df_to_jax(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        dtype: jnp.dtype = jnp.float32,
    ) -> jnp.ndarray:
        """Extract numeric columns from *df* and return as JAX array."""
        if columns is None:
            numeric = df.select_dtypes(include=[np.number])
            columns = list(numeric.columns)
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise KeyError(f"Columns not found in DataFrame: {missing}")
        values = df[columns].values.astype(np.float32)
        return jnp.asarray(values, dtype=dtype)

    # ==================================================================
    # Semantic API
    # ==================================================================

    @property
    def registry(self) -> DatasetRegistry:
        """Expose the dataset registry for advanced querying."""
        return self._registry

    def load_signals(
        self,
        signals: List[str],
        *,
        source: Optional[str] = None,
        region: Optional[str] = None,
        start_date: Optional[Union[str, datetime, date]] = None,
        end_date: Optional[Union[str, datetime, date]] = None,
        resample: Optional[str] = None,
        time_alignment: Optional[Dict[str, str]] = None,
        interpolation: str = "linear",
    ) -> pd.DataFrame:
        """Load one or more semantic signals into a single DataFrame.

        Args:
            signals: Semantic signal names (e.g.
                ``["load.actual_mw", "solar.available_mw"]``).
            source: Restrict lookup to a specific data source
                (``"gb"``, ``"aemo"``, ``"swissdn"``, ...).
            region: Filter by region (e.g. ``"NSW1"``).
            start_date: Simulation time window start (inclusive).
            end_date: Simulation time window end (inclusive).
            resample: Target frequency (``"5min"``, ``"30min"``, ...).
            time_alignment: Per-signal calendar-shift overrides.
            interpolation: Interpolation method when resampling.

        Returns:
            DataFrame carrying a ``datetime`` column on the simulation
            timeline, under a ``RangeIndex`` -- ``datetime`` is a column, not
            the index.
        """
        if not signals:
            raise ValueError("At least one signal must be requested")

        if time_alignment is None:
            time_alignment = {}

        sig_to_manifest = self._registry.resolve_signals(
            signals, source=source
        )

        manifest_groups: Dict[str, list[str]] = {}
        for sig, m in sig_to_manifest.items():
            manifest_groups.setdefault(m.name, []).append(sig)

        frames: list[pd.DataFrame] = []
        sim_start = (
            pd.Timestamp(start_date) if start_date is not None else None
        )
        sim_end = pd.Timestamp(end_date) if end_date is not None else None

        for mname, sigs in manifest_groups.items():
            manifest = self._registry.get_manifest(mname)
            df = self._load_manifest_signals(
                manifest,
                sigs,
                sim_start=sim_start,
                sim_end=sim_end,
                region=region,
                time_alignment=time_alignment,
            )
            frames.append(df)

        if len(frames) == 1:
            result = frames[0]
        else:
            result = frames[0]
            for df_part in frames[1:]:
                df_part = self._normalize_datetime_for_merge(df_part)
                result = self._normalize_datetime_for_merge(result)
                result = result.merge(df_part, on=S.DATETIME, how="inner")

        if S.DATETIME in result.columns:
            result = result.sort_values(S.DATETIME).reset_index(drop=True)

        if resample is not None:
            result = self._resample_data(result, resample, interpolation)

        self.data = result
        self.current_source = ", ".join(manifest_groups.keys())
        return result

    def load_actual_series(
        self,
        signals: List[str],
        **kwargs,
    ) -> pd.DataFrame:
        """Convenience wrapper: load only ``actual_series`` data."""
        kwargs.setdefault("source", None)
        return self.load_signals(signals, **kwargs)

    def load_forecast_panel(
        self,
        signals: List[str],
        *,
        source: Optional[str] = None,
        region: Optional[str] = None,
        start_date: Optional[Union[str, datetime, date]] = None,
        end_date: Optional[Union[str, datetime, date]] = None,
    ) -> pd.DataFrame:
        """Load forecast-panel data with ``issue_time`` + ``target_time``."""
        if not signals:
            raise ValueError("At least one signal must be requested")

        result_map: Dict[str, DatasetManifest] = {}
        missing: list[str] = []
        for sig in signals:
            candidates = self._registry.find_by_signal(
                sig, source=source, data_type=S.FORECAST_PANEL,
            )
            if not candidates:
                missing.append(sig)
            else:
                result_map[sig] = candidates[0]
        if missing:
            raise ValueError(
                f"Cannot resolve forecast-panel signals: {missing}. "
                f"Make sure there is a manifest with data_type='forecast_panel' "
                f"providing these signals."
            )

        manifest_groups: Dict[str, list[str]] = {}
        for sig, m in result_map.items():
            manifest_groups.setdefault(m.name, []).append(sig)

        frames: list[pd.DataFrame] = []
        for mname, sigs in manifest_groups.items():
            manifest = self._registry.get_manifest(mname)
            df = self._load_raw_parquet(manifest)
            df = self._apply_index_map(df, manifest)
            df = self._apply_column_map(df, manifest, sigs)
            if region is not None and S.REGION in df.columns:
                df = df[df[S.REGION] == region].reset_index(drop=True)
            frames.append(df)

        if len(frames) == 1:
            return frames[0]
        result = frames[0]
        merge_keys = [
            c
            for c in [S.REGION, S.ISSUE_TIME, S.TARGET_TIME]
            if c in result.columns
        ]
        for df_part in frames[1:]:
            result = result.merge(df_part, on=merge_keys, how="inner")
        return result

    # ------------------------------------------------------------------
    # Internal helpers for the semantic API
    # ------------------------------------------------------------------

    def _load_manifest_signals(
        self,
        manifest: DatasetManifest,
        sigs: List[str],
        *,
        sim_start: Optional[pd.Timestamp],
        sim_end: Optional[pd.Timestamp],
        region: Optional[str],
        time_alignment: Dict[str, str],
    ) -> pd.DataFrame:
        """Load requested signals from one manifest, with alignment."""
        df = self._load_raw_parquet(manifest)
        df = self._apply_index_map(df, manifest)
        df = self._apply_derived(df, manifest, sigs)
        df = self._apply_column_map(df, manifest, sigs)
        df = self._apply_normalization(df, manifest, sigs)

        if region is not None and S.REGION in df.columns:
            df = df[df[S.REGION] == region].reset_index(drop=True)

        if sim_start is not None and sim_end is not None:
            align_from = None
            for sig in sigs:
                if sig in time_alignment:
                    align_from = pd.Timestamp(time_alignment[sig])
                    break
            df = TimeAligner.align(
                df,
                time_mode=manifest.time_mode,
                sim_start=sim_start,
                sim_end=sim_end,
                align_from=align_from,
                resolution=manifest.resolution,
                time_col=S.DATETIME,
            )
        return df

    def _load_raw_parquet(self, manifest: DatasetManifest) -> pd.DataFrame:
        """Read a parquet file referenced by *manifest*."""
        path = self.data_dir / manifest.parquet_file
        if not path.exists():
            raise FileNotFoundError(
                f"Parquet file not found: {path} "
                f"(referenced by manifest '{manifest.name}')"
            )
        cols_needed = manifest.raw_columns_needed
        try:
            df = pd.read_parquet(path, columns=cols_needed)
        except Exception:
            df = pd.read_parquet(path)
        return df

    def _apply_index_map(
        self, df: pd.DataFrame, manifest: DatasetManifest
    ) -> pd.DataFrame:
        """Rename raw index columns to canonical names."""
        rename = {}
        for raw_col, canon in manifest.index_map.items():
            if raw_col in df.columns and canon not in df.columns:
                rename[raw_col] = canon
        if rename:
            df = df.rename(columns=rename)
        if S.DATETIME not in df.columns and _SETTLEMENT_COLS <= set(df.columns):
            df = self._datetime_from_settlement(df, manifest)
        if S.DATETIME in df.columns:
            df[S.DATETIME] = pd.to_datetime(df[S.DATETIME], utc=True)
        if S.ISSUE_TIME in df.columns:
            df[S.ISSUE_TIME] = pd.to_datetime(df[S.ISSUE_TIME], utc=True)
        if S.TARGET_TIME in df.columns:
            df[S.TARGET_TIME] = pd.to_datetime(df[S.TARGET_TIME], utc=True)
        return df

    @staticmethod
    def _datetime_from_settlement(
        df: pd.DataFrame, manifest: DatasetManifest
    ) -> pd.DataFrame:
        """Build ``datetime`` from settlement date + period (GB/Elexon format).

        NESO publishes a settlement date plus a period index instead of a
        timestamp, and the period index is what absorbs the DST clock change:
        a spring-forward day has 46 periods and a fall-back day 50.

        The offset is therefore added in UTC to the instant of local midnight,
        **not** to local wall-clock time. Adding in local time would put
        period 50 at ``24:30`` of the settlement date, past midnight, and no
        subsequent ``tz_localize`` can recover from that.
        """
        if manifest.timezone is None:
            raise ValueError(
                f"manifest '{manifest.name}' carries settlement_date and "
                f"settlement_period but no 'timezone'; the settlement calendar "
                f"is needed to place them on a UTC timeline."
            )
        day_start_utc = (
            pd.to_datetime(df["settlement_date"])
            .dt.tz_localize(manifest.timezone)
            .dt.tz_convert("UTC")
        )
        step = pd.Timedelta(manifest.resolution)
        period = df["settlement_period"].astype("int64")
        df = df.copy()
        df[S.DATETIME] = day_start_utc + (period - 1) * step
        return df

    def _apply_derived(
        self,
        df: pd.DataFrame,
        manifest: DatasetManifest,
        requested_sigs: List[str],
    ) -> pd.DataFrame:
        """Compute derived signals (e.g. ``wind = offshore + onshore``)."""
        for sig, expr in manifest.derived.items():
            if sig not in requested_sigs:
                continue
            # Parse additive expressions: "A + B", "A + B + C", "A -B".
            # "*" and "/" are not supported.  Subtraction is, but only spelled
            # with no space after the minus: "A -B" and "A-B" both give A - B,
            # while "A - B" tokenises to ["A", "- B"] and warns that column
            # " B" is missing.  No shipped manifest uses a minus.
            tokens = [t.strip() for t in expr.replace("-", "+-").split("+")]
            tokens = [t for t in tokens if t]  # drop empty from leading -
            missing = [t.lstrip("-") for t in tokens if t.lstrip("-") not in df.columns]
            if missing:
                warnings.warn(
                    f"Cannot derive '{sig}': missing columns {missing}",
                    stacklevel=3,
                )
                continue
            result = df[tokens[0].lstrip("-")] * (-1 if tokens[0].startswith("-") else 1)
            for t in tokens[1:]:
                col = t.lstrip("-")
                if t.startswith("-"):
                    result = result - df[col]
                else:
                    result = result + df[col]
            df[sig] = result
        return df

    def _apply_column_map(
        self,
        df: pd.DataFrame,
        manifest: DatasetManifest,
        requested_sigs: List[str],
    ) -> pd.DataFrame:
        """Rename raw data columns to signal names and drop unrequested."""
        rename = {}
        for raw_col, sig in manifest.column_map.items():
            if sig in requested_sigs and raw_col in df.columns:
                rename[raw_col] = sig
        if rename:
            df = df.rename(columns=rename)

        keep = set(requested_sigs) | {
            S.DATETIME, S.REGION, S.ISSUE_TIME, S.TARGET_TIME,
        }
        drop = [c for c in df.columns if c not in keep]
        if drop:
            df = df.drop(columns=drop)
        return df

    def _apply_normalization(
        self,
        df: pd.DataFrame,
        manifest: DatasetManifest,
        sigs: List[str],
    ) -> pd.DataFrame:
        """Divide by the normalizing factor declared in the manifest."""
        for sig in sigs:
            factor = manifest.normalize.get(sig)
            if factor and factor != 0 and sig in df.columns:
                df[sig] = df[sig] / factor
        return df

    # Dataset metadata helpers (used internally by load_signals and tests)

    def list_available_datasets(self) -> List[str]:
        """List all available datasets (parquet files)."""
        parquet_files = list(self.data_dir.glob("*.parquet"))
        return [f.stem for f in parquet_files]

    def get_metadata(self, dataset_name: str) -> Dict:
        """Get metadata for a dataset."""
        if dataset_name in self._metadata_cache:
            return self._metadata_cache[dataset_name]

        json_file = self.data_dir / f"{dataset_name}.json"
        if not json_file.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {json_file}"
            )

        with open(json_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self._metadata_cache[dataset_name] = metadata
        return metadata

    def get_available_columns(self, dataset_name: str) -> List[str]:
        """Get available columns for a dataset."""
        metadata = self.get_metadata(dataset_name)
        return metadata.get("columns", [])

    def get_date_range(
        self, dataset_name: str
    ) -> Optional[Tuple[datetime, datetime]]:
        """Get date range for a dataset."""
        metadata = self.get_metadata(dataset_name)
        date_ranges = metadata.get("date_ranges")

        if not date_ranges:
            return None

        range_info = next(iter(date_ranges.values()))
        return (pd.to_datetime(range_info["min"]), pd.to_datetime(range_info["max"]))

    def _build_column_index(self) -> Dict[str, List[str]]:
        """Build reverse index: column name -> list of datasets."""
        if self._column_index is not None:
            return self._column_index

        index: Dict[str, List[str]] = {}
        for dataset in self.list_available_datasets():
            try:
                cols = self.get_available_columns(dataset)
            except FileNotFoundError:
                continue
            for col in cols:
                col_lower = col.lower()
                if col_lower in {
                    "datetime", "settlement_datetime",
                    "interval_datetime", "starttime", "start_time",
                    "settlement_date", "date",
                    "settlementperiod", "settlement_period",
                }:
                    continue
                index.setdefault(col, []).append(dataset)

        self._column_index = index
        return index

    def find_datasets_for_columns(
        self, columns: List[str]
    ) -> Dict[str, List[str]]:
        """Find which datasets contain the requested columns."""
        col_index = self._build_column_index()
        dataset_to_cols: Dict[str, List[str]] = {}
        missing_cols = []

        for col in columns:
            datasets = col_index.get(col)
            if not datasets:
                missing_cols.append(col)
            else:
                ds = datasets[0]
                dataset_to_cols.setdefault(ds, []).append(col)

        if missing_cols:
            available = sorted(col_index.keys())[:20]
            raise ValueError(
                f"Columns not found: {missing_cols}. "
                f"Available columns (first 20): {available}"
            )

        return dataset_to_cols

    def _detect_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """Detect date column in DataFrame."""
        date_column_names = [
            "SETTLEMENT_DATETIME", "INTERVAL_DATETIME",
            "startTime", "START_TIME", "STARTTIME",
            "SETTLEMENT_DATE", "datetime", "date", "time",
        ]

        for col_name in date_column_names:
            if col_name in df.columns:
                return col_name

        datetime_cols = df.select_dtypes(
            include=["datetime64"]
        ).columns.tolist()
        if datetime_cols:
            return datetime_cols[0]

        return None

    @staticmethod
    def _normalize_datetime_for_merge(df: pd.DataFrame) -> pd.DataFrame:
        """UTC-normalize ``datetime`` and drop duplicate labels."""
        if df is None or "datetime" not in df.columns:
            return df
        out = df.copy()
        out["datetime"] = pd.to_datetime(out["datetime"], utc=True)
        if out["datetime"].duplicated().any():
            out = out.drop_duplicates(subset=["datetime"], keep="first")
        return out

    def load_data(
        self,
        dataset_name: Optional[str] = None,
        columns: Optional[List[str]] = None,
        start_date: Optional[Union[str, datetime, date]] = None,
        end_date: Optional[Union[str, datetime, date]] = None,
        validate_dates: bool = True,
        merge_how: str = "inner",
        resample: Optional[str] = None,
        interpolation: str = "linear",
    ) -> pd.DataFrame:
        """Load data from parquet file(s) with optional resampling.

        If *dataset_name* is not provided, automatically finds datasets
        based on *columns*.
        """
        if dataset_name is None:
            if not columns:
                raise ValueError(
                    "Either dataset_name or columns must be provided"
                )

            dataset_to_cols = self.find_datasets_for_columns(columns)

            if len(dataset_to_cols) == 1:
                dataset_name = list(dataset_to_cols.keys())[0]
                df = self._load_single_dataset(
                    dataset_name, columns, start_date, end_date,
                    validate_dates,
                )
            else:
                df = self._load_and_merge_datasets(
                    dataset_to_cols, start_date, end_date,
                    validate_dates, merge_how,
                )
        else:
            df = self._load_single_dataset(
                dataset_name, columns, start_date, end_date, validate_dates,
            )

        if resample is not None:
            df = self._resample_data(df, resample, interpolation)

        return df

    def _load_single_dataset(
        self,
        dataset_name: str,
        columns: Optional[List[str]],
        start_date: Optional[Union[str, datetime, date]],
        end_date: Optional[Union[str, datetime, date]],
        validate_dates: bool,
    ) -> pd.DataFrame:
        """Load data from a single parquet file."""
        metadata = self.get_metadata(dataset_name)

        parquet_file = self.data_dir / f"{dataset_name}.parquet"
        if not parquet_file.exists():
            raise FileNotFoundError(
                f"Parquet file not found: {parquet_file}"
            )

        available_columns = metadata.get("columns", [])
        if columns:
            invalid_columns = [
                col for col in columns if col not in available_columns
            ]
            if invalid_columns:
                raise ValueError(
                    f"Invalid columns: {invalid_columns}. "
                    f"Available columns: {available_columns}"
                )

        if start_date is not None:
            start_date = pd.to_datetime(start_date)
        if end_date is not None:
            end_date = pd.to_datetime(end_date)

        if validate_dates and (start_date or end_date):
            dataset_date_range = self.get_date_range(dataset_name)
            if dataset_date_range is not None:
                min_date, max_date = dataset_date_range
                if start_date:
                    start_cmp = (
                        start_date.tz_localize(min_date.tzinfo)
                        if start_date.tzinfo is None and min_date.tzinfo
                        else start_date
                    )
                    min_cmp = (
                        min_date.tz_localize(start_date.tzinfo)
                        if min_date.tzinfo is None and start_date.tzinfo
                        else min_date
                    )
                    if start_cmp < min_cmp:
                        raise ValueError(
                            f"Start date {start_date.date()} is before "
                            f"dataset start date {min_date.date()}"
                        )
                if end_date:
                    end_cmp = (
                        end_date.tz_localize(max_date.tzinfo)
                        if end_date.tzinfo is None and max_date.tzinfo
                        else end_date
                    )
                    max_cmp = (
                        max_date.tz_localize(end_date.tzinfo)
                        if max_date.tzinfo is None and end_date.tzinfo
                        else max_date
                    )
                    if end_cmp > max_cmp:
                        raise ValueError(
                            f"End date {end_date.date()} is after "
                            f"dataset end date {max_date.date()}"
                        )

        date_col_in_metadata = None
        date_ranges = metadata.get("date_ranges", {})
        if date_ranges:
            date_col_in_metadata = list(date_ranges.keys())[0]

        load_columns = None
        if columns:
            load_columns = columns.copy()
            if (
                date_col_in_metadata
                and date_col_in_metadata not in load_columns
            ):
                load_columns.insert(0, date_col_in_metadata)

        try:
            import pyarrow.parquet as pq
            import pyarrow.compute as pc

            table = pq.read_table(parquet_file, columns=load_columns)

            if date_col_in_metadata and (
                start_date is not None or end_date is not None
            ):
                date_col_data = table.column(date_col_in_metadata)

                mask = None
                if start_date is not None:
                    start_ts = pd.Timestamp(start_date)
                    if (
                        date_col_data.type.tz is not None
                        and start_ts.tzinfo is None
                    ):
                        start_ts = start_ts.tz_localize(
                            str(date_col_data.type.tz)
                        )
                    start_mask = pc.greater_equal(date_col_data, start_ts)
                    mask = (
                        start_mask
                        if mask is None
                        else pc.and_(mask, start_mask)
                    )

                if end_date is not None:
                    end_ts = (
                        pd.Timestamp(end_date).normalize()
                        + pd.Timedelta(days=1)
                        - pd.Timedelta(seconds=1)
                    )
                    if (
                        date_col_data.type.tz is not None
                        and end_ts.tzinfo is None
                    ):
                        end_ts = end_ts.tz_localize(
                            str(date_col_data.type.tz)
                        )
                    end_mask = pc.less_equal(date_col_data, end_ts)
                    mask = (
                        end_mask
                        if mask is None
                        else pc.and_(mask, end_mask)
                    )

                if mask is not None:
                    table = table.filter(mask)

            df = table.to_pandas()

        except ImportError:
            if columns:
                try:
                    df = pd.read_parquet(
                        parquet_file, columns=load_columns
                    )
                except Exception:
                    df = pd.read_parquet(parquet_file)
                    if load_columns:
                        df = df[load_columns]
            else:
                df = pd.read_parquet(parquet_file)

        except Exception:
            df = pd.read_parquet(parquet_file, columns=load_columns)

        date_col = self._detect_date_column(df)
        if date_col:
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                df[date_col] = pd.to_datetime(df[date_col])
            if date_col != "datetime":
                df = df.rename(columns={date_col: "datetime"})

        if "datetime" in df.columns and (
            start_date is not None or end_date is not None
        ):
            if start_date is not None:
                min_dt = df["datetime"].min()
                start_cmp = start_date
                min_cmp = min_dt
                if min_dt.tzinfo is not None and start_date.tzinfo is None:
                    start_cmp = start_date.tz_localize(min_dt.tzinfo)
                elif (
                    min_dt.tzinfo is None and start_date.tzinfo is not None
                ):
                    min_cmp = min_dt.tz_localize(start_date.tzinfo)

                if min_cmp < start_cmp:
                    if (
                        df["datetime"].dt.tz is not None
                        and start_date.tzinfo is None
                    ):
                        start_date = start_date.tz_localize(
                            df["datetime"].dt.tz
                        )
                    elif (
                        df["datetime"].dt.tz is None
                        and start_date.tzinfo is not None
                    ):
                        start_date = start_date.tz_localize(None)
                    df = df[df["datetime"] >= start_date]

            if end_date is not None:
                end_datetime = (
                    pd.Timestamp(end_date).normalize()
                    + pd.Timedelta(days=1)
                    - pd.Timedelta(seconds=1)
                )
                max_dt = df["datetime"].max()
                end_cmp = end_datetime
                max_cmp = max_dt
                if (
                    max_dt.tzinfo is not None
                    and end_datetime.tzinfo is None
                ):
                    end_cmp = end_datetime.tz_localize(max_dt.tzinfo)
                elif (
                    max_dt.tzinfo is None
                    and end_datetime.tzinfo is not None
                ):
                    max_cmp = max_dt.tz_localize(end_datetime.tzinfo)

                if max_cmp > end_cmp:
                    if (
                        df["datetime"].dt.tz is not None
                        and end_datetime.tzinfo is None
                    ):
                        end_datetime = end_datetime.tz_localize(
                            df["datetime"].dt.tz
                        )
                    elif (
                        df["datetime"].dt.tz is None
                        and end_datetime.tzinfo is not None
                    ):
                        end_datetime = end_datetime.tz_localize(None)
                    df = df[df["datetime"] <= end_datetime]

        df = df.reset_index(drop=True)

        self.data = df
        self.current_source = dataset_name

        return df

    def _resample_data(
        self,
        df: pd.DataFrame,
        resample_freq: str,
        interpolation: str,
    ) -> pd.DataFrame:
        """Resample time series data to a different frequency."""
        if "datetime" not in df.columns:
            raise ValueError(
                "DataFrame must have 'datetime' column for resampling"
            )

        valid_freqs = ["5min", "15min", "30min", "60min"]
        if resample_freq not in valid_freqs:
            raise ValueError(
                f"resample must be one of {valid_freqs}, "
                f"got: {resample_freq}"
            )

        df_resampled = df.set_index("datetime")

        freq_minutes = {"5min": 5, "15min": 15, "30min": 30, "60min": 60}
        target_minutes = freq_minutes[resample_freq]

        if len(df_resampled) > 1:
            time_diff = df_resampled.index[1] - df_resampled.index[0]
            original_minutes = time_diff.total_seconds() / 60
        else:
            original_minutes = 30

        if target_minutes < original_minutes:
            df_resampled = df_resampled.resample(resample_freq).asfreq()
            numeric_cols = df_resampled.select_dtypes(
                include=[np.number]
            ).columns

            if interpolation in [
                "linear", "nearest", "zero", "slinear",
                "quadratic", "cubic",
            ]:
                for col in numeric_cols:
                    df_resampled[col] = df_resampled[col].interpolate(
                        method=interpolation, limit_direction="both",
                    )
            elif interpolation in ["spline", "pchip", "akima"]:
                from scipy import interpolate as scipy_interp

                for col in numeric_cols:
                    col_mask = df_resampled[col].notna()
                    if col_mask.sum() < 2:
                        continue
                    x = np.arange(len(df_resampled))[col_mask]
                    y = df_resampled[col][col_mask].values
                    if interpolation == "spline":
                        f = scipy_interp.UnivariateSpline(x, y, s=0, k=3)
                    elif interpolation == "pchip":
                        f = scipy_interp.PchipInterpolator(x, y)
                    else:
                        f = scipy_interp.Akima1DInterpolator(x, y)
                    x_new = np.arange(len(df_resampled))
                    df_resampled[col] = f(x_new)
            else:
                raise ValueError(
                    f"Unknown interpolation method: {interpolation}"
                )

        elif target_minutes > original_minutes:
            df_resampled = df_resampled.resample(resample_freq).mean(numeric_only=True)

        df_resampled = df_resampled.reset_index()
        return df_resampled

    def _load_and_merge_datasets(
        self,
        dataset_to_cols: Dict[str, List[str]],
        start_date: Optional[Union[str, datetime, date]],
        end_date: Optional[Union[str, datetime, date]],
        validate_dates: bool,
        merge_how: str,
    ) -> pd.DataFrame:
        """Load data from multiple datasets and merge on datetime."""
        if merge_how not in ("inner", "outer"):
            raise ValueError("merge_how must be 'inner' or 'outer'")

        loaded_frames = []
        for dataset_name, cols in dataset_to_cols.items():
            df_part = self._load_single_dataset(
                dataset_name, cols, start_date, end_date, validate_dates,
            )
            df_part = self._normalize_datetime_for_merge(df_part)
            loaded_frames.append(df_part)

        result = loaded_frames[0]
        for df_part in loaded_frames[1:]:
            result = result.merge(df_part, on="datetime", how=merge_how)

        if "datetime" in result.columns:
            result = result.sort_values("datetime").reset_index(drop=True)

        self.data = result
        self.current_source = ", ".join(dataset_to_cols.keys())
        return result

    def print_dataset_info(self, dataset_name: str):
        """Print information about a dataset."""
        metadata = self.get_metadata(dataset_name)

        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 60}")
        print(f"\nSource file: {metadata.get('source_file')}")
        print(
            f"Shape: {metadata['shape']['rows']} rows "
            f"x {metadata['shape']['columns']} columns"
        )
        print("\nColumns:")
        for col in metadata["columns"]:
            dtype = metadata["dtypes"].get(col, "unknown")
            print(f"  - {col} ({dtype})")

        date_ranges = metadata.get("date_ranges")
        if date_ranges:
            print("\nDate Range:")
            for col, range_info in date_ranges.items():
                print(f"  {col}:")
                print(f"    Min: {range_info['min']}")
                print(f"    Max: {range_info['max']}")
                print(f"    Count: {range_info['count']:,}")
                if range_info.get("missing", 0) > 0:
                    print(f"    Missing: {range_info['missing']}")

        print(f"\n{'=' * 60}")

    def get_summary(self) -> Optional[pd.DataFrame]:
        """Get summary of currently loaded data."""
        if self.data is None:
            print("No data loaded. Use load_data() first.")
            return None

        print(f"\nLoaded data from: {self.current_source}")
        print(f"Shape: {self.data.shape}")
        print(f"\nColumns: {list(self.data.columns)}")

        if "datetime" in self.data.columns:
            print(f"\nDate range:")
            print(f"  Min: {self.data['datetime'].min()}")
            print(f"  Max: {self.data['datetime'].max()}")

        return self.data.describe()
