"""Download AEMO day-ahead operational demand forecasts into the form of ``aemo_forecast``.

Sources, AEMO NEMWEB, no registration and no key:

* forecasts: report ``FORECAST_OPERATIONAL_DEMAND_HH``, one file per half-hour
  issue, weekly bundles in
  https://nemweb.com.au/Reports/Archive/Operational_Demand/FORECAST_HH/;
* outturn: MMSDM monthly archive table ``DEMANDOPERATIONALACTUAL``,
  https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/.

Licence: not yet reviewed for redistribution; see the Data section of the
README.  This script only downloads; the vendored copy is not redistributed.

Output: ``AEMO_Forecast_vs_Actual_2025.parquet`` and its sidecar json: per
region and target half-hour, the forecast issued exactly 24 hours earlier
(``FORECAST_DATETIME``, ``INTERVAL_DATETIME``, ``LOAD_DATE``,
``OPERATIONAL_DEMAND_POE10/50/90``) and the outturn for the same half-hour
(``OPERATIONAL_DEMAND``, ``OPERATIONAL_DEMAND_ADJUSTMENT``, ``WDR_ESTIMATE``).

**Only part of the vendored window can still be downloaded.**  NEMWEB keeps
about a year of weekly forecast bundles; on 2026-09-22 the oldest was
``..._20250831.zip``, so vendored targets before about 2025-09-01 (the window
starts 2025-01-27) are no longer published.  The MMSDM archive keeps only the
last forecast per interval, not the day-ahead one, so it cannot stand in.  The
coverage obtained is recorded in the sidecar.  Outturn is complete: the MMSDM
archive covers the whole window.

Processing decisions:

* **``FORECAST_DATETIME`` is the issue time in the report's file name** (the
  first stamp of ``..._HH_<issue>_<created>.zip``), and a row is kept when its
  ``INTERVAL_DATETIME`` is exactly 24 hours after it.  That gives one forecast
  per region and target, as in the vendored file.  ``LOAD_DATE`` is the run
  time inside the report and is kept as published.
* **All timestamps are AEMO market time (UTC+10) stored with a UTC label**,
  which is what the vendored file does; the label is wrong by ten hours and is
  reproduced, not corrected.
* **The outturn is joined onto the forecasts**, so a target whose outturn is
  not yet in the archive is missing there.  The vendored file lacks outturn
  for its last day (2026-02-01); the MMSDM archive has it now, so those 240
  values are filled here.
* A report issued twice for the same issue time gives the same rows twice;
  identical copies are dropped, and differing copies keep the later-created
  report and are counted in the sidecar.

    python tools/data_prep/aemo_forecast.py --out-dir /tmp/pmj-data
"""
from __future__ import annotations

import io
import re
import zipfile

import pandas as pd

import _fetch
from aemo_5min_demand import mms_csv

NAME = "aemo_forecast"
STEM = "AEMO_Forecast_vs_Actual_2025"
ARCHIVE = "https://nemweb.com.au/Reports/Archive/Operational_Demand/FORECAST_HH/"
MMSDM = ("https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/{y}/MMSDM_{y}_{m:02d}/"
         "MMSDM_Historical_Data_SQLLoader/DATA/PUBLIC_ARCHIVE%23DEMANDOPERATIONALACTUAL%23FILE01%23{y}{m:02d}010000.zip")
START, END = "2025-01-27 00:00", "2026-02-01 23:30"      # target (INTERVAL_DATETIME) window
LEAD = pd.Timedelta(hours=24)
COLUMNS = ["REGIONID", "FORECAST_DATETIME", "INTERVAL_DATETIME", "LOAD_DATE",
           "OPERATIONAL_DEMAND_POE10", "OPERATIONAL_DEMAND_POE50", "OPERATIONAL_DEMAND_POE90",
           "OPERATIONAL_DEMAND", "OPERATIONAL_DEMAND_ADJUSTMENT", "WDR_ESTIMATE"]
STAMP = "%Y/%m/%d %H:%M:%S"
INNER = re.compile(r"PUBLIC_FORECAST_OPERATIONAL_DEMAND_HH_(\d{12})_(\d{14})\.zip")


def forecasts(lo: pd.Timestamp, hi: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    listing = _fetch.get(ARCHIVE).decode("utf-8", "replace")
    bundles = sorted(set(re.findall(r"PUBLIC_FORECAST_OPERATIONAL_DEMAND_HH_(\d{8})\.zip", listing)))
    issue_lo, issue_hi = lo - LEAD, hi - LEAD
    frames, used, reissued = [], [], 0
    for day in bundles:
        first = pd.Timestamp(day, tz="UTC")
        if first > issue_hi or first + pd.Timedelta(days=7) <= issue_lo:
            continue
        blob = _fetch.get(f"{ARCHIVE}PUBLIC_FORECAST_OPERATIONAL_DEMAND_HH_{day}.zip")
        used.append(day)
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for name in z.namelist():
                m = INNER.fullmatch(name)
                if not m:
                    continue
                issue = pd.Timestamp(m.group(1), tz="UTC")
                if not issue_lo <= issue <= issue_hi:
                    continue
                df = mms_csv(z.read(name))
                df["INTERVAL_DATETIME"] = pd.to_datetime(df["INTERVAL_DATETIME"], format=STAMP, utc=True)
                df = df[df["INTERVAL_DATETIME"] == issue + LEAD].copy()
                df["FORECAST_DATETIME"] = issue
                df["_created"] = m.group(2)
                frames.append(df)
        print(f"  bundle {day}: {sum(len(f) for f in frames):,} rows so far")
    df = pd.concat(frames, ignore_index=True).drop(columns="LASTCHANGED")
    df = df.drop_duplicates([c for c in df.columns if c != "_created"])
    key = ["REGIONID", "FORECAST_DATETIME"]
    reissued = int(df.duplicated(key).sum())
    df = df.sort_values("_created").drop_duplicates(key, keep="last").drop(columns="_created")
    df["LOAD_DATE"] = pd.to_datetime(df["LOAD_DATE"], format=STAMP, utc=True)
    for c in ("OPERATIONAL_DEMAND_POE10", "OPERATIONAL_DEMAND_POE50", "OPERATIONAL_DEMAND_POE90"):
        df[c] = pd.to_numeric(df[c], errors="raise").astype("int64")
    return df, {"bundles": used, "reissued_reports_later_kept": reissued,
                "oldest_bundle_listed": bundles[0] if bundles else None}


def outturn(lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    frames = []
    # MMSDM files are by calendar month of the interval-ending market time; the
    # month before ``lo`` holds the interval ending at its first midnight.
    for p in pd.period_range(lo - pd.Timedelta(days=1), hi, freq="M"):
        df = mms_csv(_fetch.get(MMSDM.format(y=p.year, m=p.month)))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["INTERVAL_DATETIME"] = pd.to_datetime(df["INTERVAL_DATETIME"], format=STAMP, utc=True)
    df = df[(df["INTERVAL_DATETIME"] >= lo) & (df["INTERVAL_DATETIME"] <= hi)]
    for c in ("OPERATIONAL_DEMAND", "OPERATIONAL_DEMAND_ADJUSTMENT", "WDR_ESTIMATE"):
        df[c] = pd.to_numeric(df[c], errors="raise").astype("float64")
    if df.duplicated(["REGIONID", "INTERVAL_DATETIME"]).any():
        raise RuntimeError("outturn has a (region, interval) pair twice")
    return df[["REGIONID", "INTERVAL_DATETIME", "OPERATIONAL_DEMAND",
               "OPERATIONAL_DEMAND_ADJUSTMENT", "WDR_ESTIMATE"]]


def build(start: str, end: str) -> tuple[pd.DataFrame, dict]:
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    fc, info = forecasts(lo, hi)
    df = fc.merge(outturn(lo, hi), on=["REGIONID", "INTERVAL_DATETIME"], how="left")
    df = df[COLUMNS].sort_values(["FORECAST_DATETIME", "REGIONID"], kind="mergesort").reset_index(drop=True)
    for c in ("FORECAST_DATETIME", "INTERVAL_DATETIME", "LOAD_DATE"):
        df[c] = _fetch.to_ns(df[c])
    info.update(requested_target_range=[str(lo), str(hi)],
                obtained_target_range=[str(df["INTERVAL_DATETIME"].min()), str(df["INTERVAL_DATETIME"].max())],
                targets_without_outturn=int(df["OPERATIONAL_DEMAND"].isna().sum()),
                regions=sorted(df["REGIONID"].unique().tolist()))
    return df, info


def main() -> None:
    p = _fetch.parser(__doc__)
    p.add_argument("--start", default=START, help=f"first target half-hour (default {START})")
    p.add_argument("--end", default=END, help=f"last target half-hour (default {END})")
    args = p.parse_args()
    out = _fetch.out_dir(args)
    df, info = build(args.start, args.end)
    _fetch.write(df, out, STEM, {
        "source_urls": [ARCHIVE, MMSDM.split("{y}")[0]],
        "source_organization": "Australian Energy Market Operator (AEMO)",
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        **info})


if __name__ == "__main__":
    main()
