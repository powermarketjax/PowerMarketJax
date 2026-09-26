"""Shared plumbing for the download scripts in this directory.

Only what every download script needs identically: the ``--out-dir`` guard,
one retried HTTP GET, and the parquet + sidecar writer.  Nothing here knows
about any particular dataset.  No JAX import, so these scripts run on any
machine with pandas and pyarrow.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
#: The vendored copies.  A download script never writes here: a re-download
#: that differs from the vendored file (upstream revisions, a shorter source
#: window) would otherwise silently replace the numbers the paper was run on.
VENDORED_PARQUET_DIR = REPO / "powermarketjax/data/parquet"

USER_AGENT = "powermarketjax-data-prep/1.0 (+research; python-urllib)"


def parser(description: str) -> argparse.ArgumentParser:
    """An argument parser with the mandatory ``--out-dir``."""
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--out-dir", type=Path, required=True,
        help="directory to write the parquet and its sidecar json into; "
             "required, and refused if it is the directory of the vendored copy")
    return p


def out_dir(args: argparse.Namespace, vendored: Path = VENDORED_PARQUET_DIR) -> Path:
    """Resolve ``--out-dir``, refusing the directory holding the vendored copy."""
    d = args.out_dir.expanduser().resolve()
    if d == vendored.resolve():
        raise SystemExit(
            f"refusing to write into {vendored}: pass a scratch "
            f"directory and compare against the vendored file instead")
    d.mkdir(parents=True, exist_ok=True)
    return d


def get(url: str, *, retries: int = 6, pause: float = 5.0,
        timeout: float = 300.0) -> bytes:
    """One GET, retried with doubling backoff on 429, 5xx and transport errors."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    delay = pause
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as fh:
                return fh.read()
        except urllib.error.HTTPError as exc:
            if (exc.code != 429 and exc.code < 500) or attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries - 1:
                raise
        time.sleep(delay)
        delay *= 2
    raise AssertionError("unreachable")


def get_json(url: str, **kw) -> object:
    return json.loads(get(url, **kw))


def to_ns(s: pd.Series) -> pd.Series:
    """Cast a datetime column to nanosecond resolution.

    The vendored files store ``timestamp[ns]``.  pandas 3 parses strings to
    microsecond resolution, which compares equal value-for-value but is a
    different dtype, so the cast is explicit.
    """
    tz = getattr(s.dt, "tz", None)
    return s.astype(f"datetime64[ns, {tz}]" if tz is not None else "datetime64[ns]")


def _plain(v: object) -> object:
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, (datetime, pd.Timestamp)):
        return v.isoformat()
    return v


def describe(df: pd.DataFrame) -> dict:
    """Shape, dtypes, first five rows, datetime ranges and numeric summaries.

    The same fields, in the same layout, as the sidecars of the vendored GB
    files, so a downloaded sidecar can be diffed against the vendored one key
    by key.  Numeric summaries cover the first ten numeric columns only, as
    the vendored sidecars do.
    """
    meta: dict = {
        "shape": {"rows": int(len(df)), "columns": int(len(df.columns))},
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "sample_data": {
            f"row_{i}": {c: _plain(row[c]) for c in df.columns}
            for i, (_, row) in enumerate(df.head(5).iterrows())},
    }
    ranges = {}
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            v = df[c].dropna()
            ranges[c] = {"min": v.min().isoformat(), "max": v.max().isoformat(),
                         "count": int(len(v)), "missing": int(len(df) - len(v))}
    meta["date_ranges"] = ranges or None
    stats = {}
    for c in df.select_dtypes(include=[np.number]).columns[:10]:
        d = df[c].describe()
        stats[c] = {k: _plain(d.get(k)) for k in ("count", "mean", "std", "min", "max")}
        stats[c]["count"] = int(d["count"])
        stats[c]["missing"] = int(df[c].isna().sum())
    meta["numeric_statistics"] = stats
    return meta


def write(df: pd.DataFrame, directory: Path, stem: str, sidecar: dict) -> Path:
    """Write ``<stem>.parquet`` and ``<stem>.json`` side by side."""
    pq = directory / f"{stem}.parquet"
    # pandas 3 string columns are written as arrow ``large_string``; the
    # vendored files, written by pandas 2, hold ``string``.
    df = df.astype({c: object for c in df.columns if isinstance(df[c].dtype, pd.StringDtype)})
    df.to_parquet(pq, index=False, compression="snappy")
    meta = {"parquet_file": pq.name,
            "generated_at": datetime.now().isoformat(), **sidecar}
    with open(directory / f"{stem}.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False, default=str)
        fh.write("\n")
    print(f"wrote {pq} ({len(df):,} rows) and {pq.with_suffix('.json').name}")
    return pq
