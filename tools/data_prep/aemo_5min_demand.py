"""Download AEMO 5-minute operational demand into the form of ``aemo_5min_demand``.

Source: AEMO NEMWEB, report ``ACTUAL_OPERATIONAL_DEMAND_5MIN``, directory
https://nemweb.com.au/Reports/Current/Operational_Demand/ACTUAL_5MIN/.  No
registration and no key.

Licence: not yet reviewed for redistribution; see the Data section of the
README.  This script only downloads; the vendored copy is not redistributed.

Output: ``AEMO_5min_Demand_2025_2026.parquet`` and its sidecar json, columns
``REGIONID``, ``INTERVAL_DATETIME``, ``OPERATIONAL_DEMAND``,
``OPERATIONAL_DEMAND_ADJUSTMENT``, ``WDR_ESTIMATE`` (MW), one row per region
and 5-minute interval, sorted by region then time.

**Only part of the vendored window can still be downloaded.**  The report is a
single file holding a rolling twelve months, and NEMWEB keeps no archive of it
(``Reports/Archive/Operational_Demand/`` has no ``ACTUAL_5MIN``; the MMSDM
monthly archive has only the 30-minute ``DEMANDOPERATIONALACTUAL``).  On
2026-09-22 the file began at 2025-09-22 00:05, so the vendored rows before
that date (2024-09-18 onwards) are no longer published; the rows that remain
agree exactly with the vendored ones.  The window is clipped to ``--start`` ..
``--end`` (default the vendored window) and the coverage actually obtained is
recorded in the sidecar.

Processing decisions:

* **``INTERVAL_DATETIME`` is AEMO market time (UTC+10, interval-ending) and is
  stored with a UTC label unchanged**, which is what the vendored file does.
  The label is wrong by ten hours; this script reproduces it so the output is
  comparable with the vendored copy, and does not correct it.
* ``LASTCHANGED`` is dropped, as in the vendored file.

    python tools/data_prep/aemo_5min_demand.py --out-dir /tmp/pmj-data
"""
from __future__ import annotations

import io
import re
import zipfile

import pandas as pd

import _fetch

NAME = "aemo_5min_demand"
STEM = "AEMO_5min_Demand_2025_2026"
DIR = "https://nemweb.com.au/Reports/Current/Operational_Demand/ACTUAL_5MIN/"
START, END = "2024-09-18 00:05", "2026-02-12 02:00"
COLUMNS = ["REGIONID", "INTERVAL_DATETIME", "OPERATIONAL_DEMAND",
           "OPERATIONAL_DEMAND_ADJUSTMENT", "WDR_ESTIMATE"]


def mms_csv(blob: bytes) -> pd.DataFrame:
    """Parse an MMS flat file: 'C' comment rows, one 'I' header row, 'D' data rows.

    The first four fields of every row are the record type, report, table and
    version; they are dropped so that a column named like the report (here
    ``OPERATIONAL_DEMAND``) does not collide with it.
    """
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0]).decode("utf-8", "replace")
    header, rows = None, []
    for line in raw.splitlines():
        if line.startswith("I,"):
            if header is not None:
                raise RuntimeError("more than one table in the file")
            header = [c.strip('"') for c in line.split(",")][4:]
        elif line.startswith("D,"):
            rows.append([c.strip('"') for c in line.split(",")][4:])
    if header is None:
        raise RuntimeError("no 'I' header row found; the MMS layout changed")
    return pd.DataFrame(rows, columns=header)


def latest_file() -> str:
    listing = _fetch.get(DIR).decode("utf-8", "replace")
    names = sorted(set(re.findall(r"PUBLIC_ACTUAL_OPERATIONAL_DEMAND_5MIN_\d+_\d+\.zip", listing)))
    if not names:
        raise RuntimeError(f"no report file listed at {DIR}")
    return DIR + names[-1]


def build(start: str, end: str) -> tuple[pd.DataFrame, dict]:
    url = latest_file()
    blob = _fetch.get(url)
    df = mms_csv(blob)
    df["INTERVAL_DATETIME"] = pd.to_datetime(df["INTERVAL_DATETIME"], format="%Y/%m/%d %H:%M:%S", utc=True)
    published = (df["INTERVAL_DATETIME"].min(), df["INTERVAL_DATETIME"].max())
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    df = df[(df["INTERVAL_DATETIME"] >= lo) & (df["INTERVAL_DATETIME"] <= hi)]
    for c in COLUMNS[2:]:
        df[c] = pd.to_numeric(df[c], errors="raise").astype("int64")
    if df.duplicated(["REGIONID", "INTERVAL_DATETIME"]).any():
        raise RuntimeError("a (region, interval) pair appears twice")
    df = df[COLUMNS].sort_values(["REGIONID", "INTERVAL_DATETIME"], kind="mergesort").reset_index(drop=True)
    df["INTERVAL_DATETIME"] = _fetch.to_ns(df["INTERVAL_DATETIME"])
    got = (df["INTERVAL_DATETIME"].min(), df["INTERVAL_DATETIME"].max())
    return df, {"source_file": url.rsplit("/", 1)[-1], "source_file_url": url, "bytes": len(blob),
                "published_range": [str(t) for t in published],
                "requested_range": [str(lo), str(hi)],
                "obtained_range": [str(t) for t in got],
                "regions": sorted(df["REGIONID"].unique().tolist())}


def main() -> None:
    p = _fetch.parser(__doc__)
    p.add_argument("--start", default=START, help=f"first interval (default {START})")
    p.add_argument("--end", default=END, help=f"last interval (default {END})")
    args = p.parse_args()
    out = _fetch.out_dir(args)
    df, info = build(args.start, args.end)
    _fetch.write(df, out, STEM, {
        "source_url": DIR,
        "source_organization": "Australian Energy Market Operator (AEMO)",
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        **info})


if __name__ == "__main__":
    main()
