"""Fetch the Swiss day-ahead wholesale price into this repository's parquet form.

Offline, run once, not imported by anything.  It exists to supply the price
series of the Swiss configuration of the local flexibility
market: it needs an exogenous
energy price :math:`\\pi^{\\mathrm{e}}_t`, and without it the replacement cost
has no value, so the markup map has no origin.

**Why a national series is the local price here.**  Switzerland is a single
bidding zone.  There is no Swiss zonal or nodal day-ahead price to be more
local than this one, so the CH day-ahead price *is* the wholesale price facing
every Swiss node, and pairing it with a Swiss distribution grid is same-market
pairing rather than the cross-market mixing §14 forbids.  That is the whole
argument, and it does not carry over to a country with several bidding zones.

Source and provenance:

* Energy-Charts, operated by Fraunhofer ISE, ``GET /price?bzn=CH``.  No
  registration and no token; the endpoint returns ``unix_seconds`` and
  ``price`` arrays plus ``unit`` and ``license_info``.
* Licence CC BY 4.0.  The value recorded below is the ``license_info`` string
  the API itself returned on the day of the fetch, not a claim read off a web
  page -- ``"CC BY 4.0 (creativecommons.org/licenses/by/4.0) from
  Bundesnetzagentur | SMARD.de"`` -- because a licence read from the response
  is evidence about the bytes actually retrieved.  Attribution therefore names
  both the distributor and the upstream publisher it points at.

Three properties of the retrieved series drive what this script does.

**The currency is EUR and it is not converted.**  ``unit`` is ``EUR / MWh``.
The market specifications write prices in \\$/MWh as a nominal unit, and the
only other price series in this repository, ``gb_market_mid``, is in £/MWh.
Converting would require an exchange-rate series, which is one more data
dependency with its own provenance and its own alignment problem, so nothing
is converted and the currency is carried in the column name
(``dayahead_price_eur_mwh``), in the signal name, and in the metadata.  A
number crossing between the two price series without an explicit conversion is
then a visible error rather than a silent one.

**Gaps are counted, never filled.**  A complete hourly year is 8760 rows in UTC
terms -- the two daylight-saving transitions cancel, since the spring day loses
an hour and the autumn day gains one.  Any year short of that is missing data
at the source.  The counts go into the metadata and a test asserts them,
because an interpolated hour is indistinguishable from a measured one once it
is in the file.

**Negative prices are real and are kept.**  The measured minimum over
2015-2025 is far below zero, which is a genuine feature of a coupled European
day-ahead market and not a sentinel.  Clipping it would remove exactly the
periods in which storing energy is most attractive, which is the behaviour
§9.5 makes the aggregator trade off against degradation.

    python tools/data_prep/ch_dayahead_price.py [first_year] [last_year]
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT_PARQUET = REPO / "powermarketjax/data/parquet/CH_DayAhead_Price_2015_2025_60min.parquet"
OUT_META = OUT_PARQUET.with_suffix(".json")

API = "https://api.energy-charts.info/price"
BIDDING_ZONE = "CH"
FIRST_YEAR, LAST_YEAR = 2015, 2025

LICENCE = "Creative Commons Attribution 4.0 International"
LICENCE_URL = "https://creativecommons.org/licenses/by/4.0/"
ATTRIBUTION = (
    "Day-ahead spot price, bidding zone CH, retrieved from the Energy-Charts "
    "API operated by Fraunhofer ISE; the API reports the data as CC BY 4.0 "
    "from Bundesnetzagentur | SMARD.de. Changes: per-year responses "
    "concatenated, unix seconds converted to UTC timestamps, no unit "
    "conversion and no gap filling."
)


#: The endpoint rate-limits an unpaced loop over the years with HTTP 429, which
#: is what a first run actually hit after four requests.  Pacing plus backoff
#: rather than a shorter series, because dropping years to stay under the limit
#: would make the available window an artefact of the fetch loop.
PAUSE_SECONDS = 5.0
RETRIES = 6


def _get_json(url: str) -> dict:
    """One GET, retried on 429 and on transient transport errors."""
    delay = PAUSE_SECONDS
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=180) as fh:
                return json.load(fh)
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == RETRIES - 1:
                raise
        except urllib.error.URLError:
            if attempt == RETRIES - 1:
                raise
        time.sleep(delay)
        delay *= 2
    raise SystemExit(f"exhausted {RETRIES} attempts on {url}")


def _fetch_year(year: int) -> tuple[pd.DataFrame, str, str]:
    """One calendar year of hourly prices, still exactly as the API gave them."""
    url = f"{API}?bzn={BIDDING_ZONE}&start={year}-01-01&end={year}-12-31"
    payload = _get_json(url)

    seconds = payload["unix_seconds"]
    prices = payload["price"]
    if len(seconds) != len(prices):
        raise SystemExit(f"{year}: {len(seconds)} timestamps against {len(prices)} prices")

    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(seconds, unit="s", utc=True),
            "dayahead_price_eur_mwh": pd.Series(prices, dtype="float64"),
        }
    )
    return frame, payload.get("unit", ""), payload.get("license_info", "")


def main(first_year: int = FIRST_YEAR, last_year: int = LAST_YEAR) -> None:
    frames, per_year, units, licences = [], {}, set(), set()
    for n, year in enumerate(range(first_year, last_year + 1)):
        if n:
            time.sleep(PAUSE_SECONDS)
        frame, unit, licence = _fetch_year(year)
        units.add(unit)
        licences.add(licence)

        # A complete hourly year in UTC is 24 x the number of calendar days:
        # the two daylight-saving transitions cancel out, so 8760 except in a
        # leap year.  Anything short of it is a hole at the source.
        expected = 24 * (366 if pd.Timestamp(year=year, month=12, day=31).dayofyear == 366 else 365)
        per_year[str(year)] = {
            "rows": int(len(frame)),
            "expected_rows": expected,
            "missing_rows": int(expected - len(frame)),
            "null_prices": int(frame["dayahead_price_eur_mwh"].isna().sum()),
        }
        frames.append(frame)
        print(f"  {year}: {len(frame):5d} rows (expected {expected}), "
              f"{per_year[str(year)]['null_prices']} null")

    if len(units) != 1:
        raise SystemExit(f"the API changed units mid-series: {units}")
    if len(licences) != 1:
        raise SystemExit(f"the API changed its licence statement mid-series: {licences}")

    df = pd.concat(frames, ignore_index=True).sort_values("datetime", kind="stable")

    duplicated = int(df.duplicated("datetime").sum())
    if duplicated:
        raise SystemExit(f"unexpected duplicate timestamps: {duplicated}")

    # The step between consecutive rows says whether the series is hourly
    # throughout.  It is checked rather than assumed because EPEX SPOT has been
    # adding sub-hourly day-ahead products, and a series that turns 15-minute
    # part way through would silently quadruple the weight of its later years.
    steps = df["datetime"].diff().dropna().value_counts()
    step_summary = {str(k): int(v) for k, v in steps.items()}
    # A single step value is a stronger statement than the per-year row counts:
    # it says no hour is missing anywhere *inside* the record.  A year short of
    # its expected count while every step is one hour therefore locates the
    # shortfall at a year boundary, which for the first year is the start of
    # the available history rather than a hole.
    contiguous = list(step_summary) == [str(pd.Timedelta(hours=1))]

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, compression="zstd", index=False)

    unit = units.pop()
    meta = {
        "parquet_file": OUT_PARQUET.name,
        "source_organization": "Fraunhofer ISE (Energy-Charts)",
        "source_url": f"{API}?bzn={BIDDING_ZONE}",
        "upstream_publisher_reported_by_api": licences.copy().pop(),
        "bidding_zone": BIDDING_ZONE,
        "bidding_zone_note": (
            "Switzerland is a single bidding zone, so this national series is "
            "the day-ahead wholesale price at every Swiss node; there is no "
            "more local Swiss day-ahead price to prefer over it."),
        "licence": LICENCE,
        "licence_url": LICENCE_URL,
        "licence_permits_redistribution": True,
        "licence_evidence": (
            "the licence recorded here is the license_info string returned by "
            "the API in the same responses these rows came from"),
        "attribution": ATTRIBUTION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "currency": "EUR",
        "currency_note": (
            "not converted. gb_market_mid is GBP and the market specifications "
            "write $/MWh as a nominal unit; an exchange-rate series would be a "
            "further data dependency, so the currency is carried in the column "
            "name, the signal name and here instead."),
        "units": {"dayahead_price_eur_mwh": f"{unit}, day-ahead auction price"},
        "resolution": "60min",
        "timestamp_convention": (
            "datetime is the UTC start of the delivery hour, converted from the "
            "unix_seconds the API returns; no local-time column is stored, so "
            "the daylight-saving transitions need no convention"),
        "interval_steps_observed": step_summary,
        "hourly_and_gap_free": contiguous,
        "hourly_and_gap_free_note": (
            "true means every consecutive pair of rows is exactly one hour "
            "apart, so the series carries no interior hole and no sub-hourly "
            "stretch. Read it together with per_year: a year whose rows fall "
            "short while this stays true is short at its boundary, and for the "
            "first year of the record that is where the history begins."),
        "per_year": per_year,
        "gap_policy": "gaps are counted in per_year.missing_rows and never filled",
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "date_range": [str(df["datetime"].min()), str(df["datetime"].max())],
        # ``date_ranges`` keyed by column name is not a duplicate of the list
        # above: ``DataLoader._load_single_dataset`` takes the first key of
        # this dict as the name of the time column, so a sidecar without it
        # makes the legacy ``load_data`` path fail with a bare KeyError on
        # 'datetime' -- measured, not anticipated. The list form is the one
        # the manifest and the human reader use.
        "date_ranges": {
            "datetime": {
                "min": str(df["datetime"].min()),
                "max": str(df["datetime"].max()),
                "count": int(len(df)),
                "missing": int(df["datetime"].isna().sum()),
            }
        },
        "numeric_statistics": {
            "dayahead_price_eur_mwh": {
                "count": int(df["dayahead_price_eur_mwh"].count()),
                "mean": float(df["dayahead_price_eur_mwh"].mean()),
                "std": float(df["dayahead_price_eur_mwh"].std()),
                "min": float(df["dayahead_price_eur_mwh"].min()),
                "max": float(df["dayahead_price_eur_mwh"].max()),
                "negative_hours": int((df["dayahead_price_eur_mwh"] < 0).sum()),
                "missing": int(df["dayahead_price_eur_mwh"].isna().sum()),
            }
        },
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{OUT_PARQUET.name}: {OUT_PARQUET.stat().st_size / 1e3:.1f} kB, "
          f"{len(df):,} rows, {df['datetime'].min()} .. {df['datetime'].max()}, "
          f"{meta['numeric_statistics']['dayahead_price_eur_mwh']['negative_hours']} negative hours")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(int(args[0]) if args else FIRST_YEAR,
         int(args[1]) if len(args) > 1 else LAST_YEAR)
