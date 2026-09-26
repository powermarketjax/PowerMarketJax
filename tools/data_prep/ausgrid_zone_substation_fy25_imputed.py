"""Download Ausgrid zone substation load as Ausgrid publishes it, gaps included.

**This is the raw publication, not the gap-filled copy this repository runs
on** (``ausgrid_zone_substation_fy25_imputed``).  How that copy's gaps were
filled is not recorded and cannot be reproduced by this script.  The data are
Ausgrid's copyright and this repository does not redistribute them.

Source: Ausgrid, "Distribution zone substation data", file "Distribution
zone substation data 2025" (a zip of one CSV per zone substation), listed at
https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Distribution-zone-substation-data
(which now redirects to
https://www.ausgrid.com.au/about-us/about-ausgrid/research-data-sets/distribution-zone-substation-data).
No registration and no key.  The file's download URL is read from that page,
because the page links it through a content store whose URL carries a
version parameter.

Licence: Ausgrid's; not redistributed, see the Data section of the README.
This script only downloads.

Output: ``Ausgrid_Zone_Substation_FY25_15min.parquet`` and its sidecar
json, long form: ``interval_start`` (UTC), ``zone_substation``, ``load_mw``,
``fiscal_year``, sorted by substation then time.

Processing decisions:

* **Ausgrid's financial-year file is taken as published; nothing is imputed
  here, so the output has gaps the vendored file does not.**  The vendored
  sidecar names its source folder ``... FY25_imputed`` and its ``load_mw``
  has no missing value; the zip Ausgrid serves today (published 2025-12-10)
  names the folder ``... FY25`` and leaves 153 167 of 6 095 040 values empty
  (measured 2026-09-22).  Where both have a value they agree to 1.5e-14 MW.
  How the vendored values were filled is not recorded and is neither linear
  interpolation nor the same quarter hour a week earlier.  Each CSV row is
  one local calendar day, 2024-05-01 to 2025-04-29, with 96 quarter-hour
  columns.
* **A column label ``HH:MM`` is the end of its quarter hour in local civil
  time** (``00:15`` is the first, ``24:00`` the last), so the start is the
  label minus 15 minutes.  The labels are not in time order in the CSV; they
  are sorted.
* **Local time is Australia/Sydney and is converted to UTC** with
  ``tz_localize(ambiguous=False, nonexistent="shift_forward")``, as in the
  vendored file.  Every CSV day has 96 columns, including the two
  clock-change days, so the conversion is not one-to-one there: on
  2024-10-06 the four labels that fall in the skipped hour are shifted onto
  03:00, so 174 of the 175 substations carry five rows with one
  ``interval_start`` (870 rows in all, the same in the vendored file), and a
  substation's series also has one 75-minute step (measured on Mona Vale).
  This is reproduced, not corrected.

    python tools/data_prep/ausgrid_zone_substation_fy25_imputed.py --out-dir /tmp/pmj-data
"""
from __future__ import annotations

import io
import json
import re
import zipfile

import pandas as pd

import _fetch

NAME = "ausgrid_zone_substation_fy25_imputed"
STEM = "Ausgrid_Zone_Substation_FY25_15min"
PAGE = "https://www.ausgrid.com.au/about-us/about-ausgrid/research-data-sets/distribution-zone-substation-data"
FILE_NAME = "Distribution zone substation data 2025"
TZ = "Australia/Sydney"
LABEL = re.compile(r"^\d{2}:\d{2}$")


def download_url() -> str:
    page = _fetch.get(PAGE).decode("utf-8", "replace")
    m = re.search(r'"name":"' + re.escape(FILE_NAME) + r'".{0,400}?"href":"([^"]+)"', page)
    if not m:
        raise RuntimeError(f"{FILE_NAME!r} is no longer linked from {PAGE}")
    return json.loads(f'"{m.group(1)}"')


def start_minutes(label: str) -> int:
    h, m = map(int, label.split(":"))
    return h * 60 + m - 15


def one_csv(blob: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(blob), low_memory=False)
    labels = sorted((c for c in df.columns if LABEL.match(str(c).strip())), key=start_minutes)
    if len(labels) != 96:
        raise RuntimeError(f"{len(labels)} quarter-hour columns, expected 96")
    long = df.melt(id_vars=["year", "Zone Substation", "Date"], value_vars=labels,
                   var_name="label", value_name="load_mw")
    local = (pd.to_datetime(long["Date"], format="%Y-%m-%d")
             + pd.to_timedelta(long["label"].map(start_minutes), unit="m"))
    return pd.DataFrame({
        "interval_start": local.dt.tz_localize(TZ, ambiguous=False, nonexistent="shift_forward")
                               .dt.tz_convert("UTC"),
        "zone_substation": long["Zone Substation"].astype(object),
        "load_mw": pd.to_numeric(long["load_mw"], errors="coerce").astype("float64"),
        "fiscal_year": pd.to_numeric(long["year"], errors="coerce").astype("Int64"),
    })


def build() -> tuple[pd.DataFrame, dict]:
    url = download_url()
    blob = _fetch.get(url)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith(".csv"))
        folder = names[0].split("/")[0] if "/" in names[0] else ""
        df = pd.concat([one_csv(z.read(n)) for n in names], ignore_index=True)
    df = df.sort_values(["zone_substation", "interval_start"], kind="mergesort").reset_index(drop=True)
    df["interval_start"] = _fetch.to_ns(df["interval_start"])
    stations = sorted(df["zone_substation"].unique().tolist())
    return df, {"source_dir": folder, "source_files": len(names), "source_file_url": url,
                "bytes": len(blob), "zone_substation_values": stations,
                "missing_load_mw": int(df["load_mw"].isna().sum()),
                "date_range": [df["interval_start"].min().date().isoformat(),
                               df["interval_start"].max().date().isoformat()]}


def main() -> None:
    args = _fetch.parser(__doc__).parse_args()
    out = _fetch.out_dir(args)
    df, info = build()
    _fetch.write(df, out, STEM, {
        "source_url": PAGE,
        "source_organization": "Ausgrid",
        "timezone_local": TZ,
        "timezone_stored": "UTC",
        "interval_convention": "interval_start is UTC start of 15-minute block; CSV column "
                               "HH:MM is interval end in local civil time. Local->UTC uses pandas "
                               "tz_localize (ambiguous=False, nonexistent=shift_forward).",
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        **info})


if __name__ == "__main__":
    main()
