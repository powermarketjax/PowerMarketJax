"""Download NESO historic demand data into the form of ``gb_neso_demand``.

Source: National Energy System Operator (NESO) data portal, dataset
"Historic Demand Data", https://www.neso.energy/data-portal/historic-demand-data.
One CSV per calendar year.  The resource list is read from the portal's CKAN
API (``package_show?id=historic-demand-data``) rather than hard-coded, because
NESO re-issues a year as a new resource when it revises it.  No registration
and no key.

Licence: not yet reviewed for redistribution; see the Data section of the
README.  This script only downloads; the vendored copy is not redistributed.

Output: ``GB_NESO_Demand_2009_2025_30min.parquet`` and its sidecar json, one
row per settlement period, the CSV columns unchanged.

Processing decisions:

* **The date column comes in more than one format across years** (for
  example ``01-JAN-2009`` in 2009, ``01-Jan-23`` in 2023 and ``2025-01-01``
  in 2025).  Each year is parsed with the formats tried in turn, and a year
  that no format parses completely raises, so a new format cannot silently
  turn into missing dates.
* **Columns that some years lack are kept and left missing there**, never
  filled.  Interconnectors commissioned later (NSL, ElecLink, Viking,
  Greenlink) and ``SCOTTISH_TRANSFER`` are absent from the early files; those
  columns are therefore float with missing values, every other column is
  integer.  Column order is fixed to that of the vendored file.
* **The window is cut to ``--start`` .. ``--end``** (default: the
  ``date_range`` of the vendored manifest).  The vendored copy ends on
  2025-04-13 because that is where the 2025 file ended when it was taken;
  NESO has since extended that file, so a longer window is available but
  would not be the vendored series.
* **NESO revised the 2025 file after the vendored copy was taken** (the
  portal lists it modified 2026-06-09; the vendored copy was generated
  2026-04-09).  A download therefore differs from the vendored copy in
  ``EMBEDDED_SOLAR_GENERATION`` and ``EMBEDDED_SOLAR_CAPACITY`` between
  2025-01-01 and 2025-04-13, and nowhere else (measured 2026-09-22).
  ``TSD``, the column ``case29gb`` uses, is identical.
* ``SETTLEMENT_PERIOD`` runs 1..46 on the spring clock-change day and 1..50 on
  the autumn one.  It is stored as published; conversion to UTC is the
  loader's job (see the manifest's ``datetime_recipe``).

    python tools/data_prep/gb_neso_demand.py --out-dir /tmp/pmj-data
"""
from __future__ import annotations

import io
import re

import pandas as pd

import _fetch

NAME = "gb_neso_demand"
STEM = "GB_NESO_Demand_2009_2025_30min"
PORTAL = "https://www.neso.energy/data-portal/historic-demand-data"
PACKAGE = "https://api.neso.energy/api/3/action/package_show?id=historic-demand-data"
START, END = "2009-01-01", "2025-04-13"

COLUMNS = [
    "SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND", "TSD", "ENGLAND_WALES_DEMAND",
    "EMBEDDED_WIND_GENERATION", "EMBEDDED_WIND_CAPACITY",
    "EMBEDDED_SOLAR_GENERATION", "EMBEDDED_SOLAR_CAPACITY", "NON_BM_STOR",
    "PUMP_STORAGE_PUMPING", "IFA_FLOW", "IFA2_FLOW", "BRITNED_FLOW",
    "MOYLE_FLOW", "EAST_WEST_FLOW", "NEMO_FLOW", "NSL_FLOW", "ELECLINK_FLOW",
    "VIKING_FLOW", "GREENLINK_FLOW", "SCOTTISH_TRANSFER",
]
DATE_FORMATS = ("%d-%b-%Y", "%d-%b-%y", "%Y-%m-%d", "%d/%m/%Y")


def _resources() -> dict[int, str]:
    """Map year -> CSV URL from the CKAN package listing."""
    pkg = _fetch.get_json(PACKAGE)["result"]
    out = {}
    for r in pkg["resources"]:
        m = re.fullmatch(r"Historic Demand Data (\d{4})", r["name"].strip())
        if m and (r.get("format") or "").upper() == "CSV":
            out[int(m.group(1))] = r["url"]
    return out


def _parse_dates(raw: pd.Series, year: int) -> pd.Series:
    for fmt in DATE_FORMATS:
        d = pd.to_datetime(raw, format=fmt, errors="coerce")
        if d.notna().all():
            return d
    raise RuntimeError(f"{year}: SETTLEMENT_DATE matches none of {DATE_FORMATS}; "
                       f"first value {raw.iloc[0]!r}")


def build(start: str, end: str) -> tuple[pd.DataFrame, dict]:
    urls = _resources()
    years = [y for y in sorted(urls) if int(start[:4]) <= y <= int(end[:4])]
    frames, files = [], {}
    for y in years:
        blob = _fetch.get(urls[y])
        df = pd.read_csv(io.BytesIO(blob))
        df["SETTLEMENT_DATE"] = _parse_dates(df["SETTLEMENT_DATE"].astype(str), y)
        unknown = set(df.columns) - set(COLUMNS)
        if unknown:
            raise RuntimeError(f"{y}: columns not in the vendored schema: {sorted(unknown)}")
        frames.append(df)
        files[str(y)] = {"url": urls[y], "bytes": len(blob), "rows": len(df)}
        print(f"  {y}: {len(df):,} rows")
    df = pd.concat(frames, ignore_index=True).reindex(columns=COLUMNS)
    df = df[(df["SETTLEMENT_DATE"] >= start) & (df["SETTLEMENT_DATE"] <= end)]
    df = df.reset_index(drop=True)
    for c in COLUMNS[1:]:
        if df[c].notna().all():
            df[c] = df[c].astype("int64")
        else:
            df[c] = df[c].astype("float64")
    df["SETTLEMENT_DATE"] = _fetch.to_ns(df["SETTLEMENT_DATE"])
    return df, files


def main() -> None:
    p = _fetch.parser(__doc__)
    p.add_argument("--start", default=START, help=f"first settlement date (default {START})")
    p.add_argument("--end", default=END, help=f"last settlement date (default {END})")
    args = p.parse_args()
    out = _fetch.out_dir(args)
    df, files = build(args.start, args.end)
    _fetch.write(df, out, STEM, {
        "source_url": PORTAL,
        "source_organization": "National Energy System Operator (NESO)",
        "source_files": files,
        "window": [args.start, args.end],
        **_fetch.describe(df)})


if __name__ == "__main__":
    main()
