"""Download Elexon outturn and day-ahead demand into the form of ``gb_forecast_actual_demand``.

Sources: Elexon Insights Solution API, no registration and no key:

* ``GET https://data.elexon.co.uk/bmrs/api/v1/demand/actual/total``
  (dataset ATL, actual total load, B0610) -> ``Actual``;
* ``GET https://data.elexon.co.uk/bmrs/api/v1/forecast/demand/total/day-ahead``
  (dataset DATL, day-ahead total load forecast, B0620) -> ``DAForecast``.

Licence: not yet reviewed for redistribution; see the Data section of the
README.  This script only downloads; the vendored copy is not redistributed.

Output: ``GB_Forecast_Actual_Demand_2023_2025_30min.parquet`` and its sidecar
json, columns ``startTime`` (UTC period start), ``DAForecast``, ``Actual``, MW.

What ``load_gb_demand`` (case29gb) reads from this file is ``DAForecast``
only, over its overlap with NESO transmission demand (2023-07-05 to
2025-04-13); ``Actual`` is carried but not paired with it.

Processing decisions, each one the rule that reproduces the vendored file:

* **The time axis is the union of the periods ATL and DATL return**, not a
  full grid.  Five periods in the default window are in neither and are
  absent from the output rather than filled.
* **A period ATL published more than once keeps its first publication**
  (earliest ``publishTime``).  DATL had no republished period in the default
  window as measured 2026-09-22.
* **Gaps on that axis are filled by linear interpolation over positions**
  (``Series.interpolate(method="linear")``, which ignores the spacing of the
  timestamps), **rounded to 0.1 MW**.  Both are what the vendored values show:
  the time-weighted variant and the unrounded one each leave differences.
  Without the fill, ``load_gb_demand`` would drop every day DATL misses (23
  whole days before 2025-04-13) instead of seeing a complete window.
* **``DAForecast`` gaps are filled only up to ``--fill-until``** (default
  2025-12-31).  The vendored file leaves DATL's 2026 gaps missing; any date
  from 2025-04-05 to 2026-01-21 reproduces that, since no gap starts in
  between.  ``Actual`` has no such cut: the vendored column has no missing
  value.
* **Outturn is not otherwise cleaned.**  ATL has a few implausibly low
  readings (minimum 750 MW); they are kept.
* **Requests are cut into 7-day windows**, the API's limit.  ``from`` and
  ``to`` are inclusive, so boundary periods arrive twice; identical copies are
  dropped.

    python tools/data_prep/gb_forecast_actual_demand.py --out-dir /tmp/pmj-data
"""
from __future__ import annotations

import pandas as pd

import _fetch

NAME = "gb_forecast_actual_demand"
STEM = "GB_Forecast_Actual_Demand_2023_2025_30min"
BASE = "https://data.elexon.co.uk/bmrs/api/v1"
ACTUAL = f"{BASE}/demand/actual/total"
DAYAHEAD = f"{BASE}/forecast/demand/total/day-ahead"
START, END = "2023-07-05", "2026-04-05"
FILL_UNTIL = "2025-12-31"
WINDOW = pd.Timedelta(days=7)


def series(url: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Series, int]:
    """One Elexon quantity series indexed by UTC period start, first publication
    of each period, and the number of periods published more than once."""
    rows = []
    t = start
    while t < end:
        u = min(t + WINDOW, end)
        rows += _fetch.get_json(f"{url}?from={t:%Y-%m-%dT%H:%MZ}&to={u:%Y-%m-%dT%H:%MZ}&format=json")["data"]
        t = u
    df = pd.DataFrame(rows)[["startTime", "publishTime", "quantity"]].drop_duplicates()
    df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
    df = df[(df["startTime"] >= start) & (df["startTime"] <= end)]
    revised = int(df["startTime"].duplicated().sum())
    df = df.sort_values(["startTime", "publishTime"]).drop_duplicates("startTime", keep="first")
    print(f"  {url.rsplit('/', 2)[-2:]}: {len(df):,} periods, {revised} revised")
    return df.set_index("startTime")["quantity"].astype("float64"), revised


def fill(s: pd.Series) -> pd.Series:
    """Interior gaps filled linearly over positions, filled points rounded to 0.1."""
    filled = s.interpolate(method="linear", limit_area="inside").round(1)
    return s.where(s.notna(), filled)


def build(start: str, end: str, fill_until: str) -> tuple[pd.DataFrame, dict]:
    t0 = pd.Timestamp(start, tz="UTC")
    t1 = pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23, minutes=30)
    actual, rev_a = series(ACTUAL, t0, t1)
    dayahead, rev_d = series(DAYAHEAD, t0, t1)
    axis = actual.index.union(dayahead.index)
    actual, dayahead = actual.reindex(axis), dayahead.reindex(axis)
    cut = axis > pd.Timestamp(fill_until, tz="UTC") + pd.Timedelta(hours=23, minutes=30)
    filled = {"Actual": int(actual.isna().sum()),
              "DAForecast": int((dayahead.isna() & ~cut).sum())}
    df = pd.DataFrame({"DAForecast": fill(dayahead).where(~cut, dayahead),
                       "Actual": fill(actual)}, index=axis)
    df = df.rename_axis("startTime").reset_index()
    df["startTime"] = _fetch.to_ns(df["startTime"])
    grid = pd.date_range(t0, t1, freq="30min")
    return df, {"republished_periods_first_kept": {"Actual": rev_a, "DAForecast": rev_d},
                "filled_points": filled,
                "dayahead_fill_until": fill_until,
                "periods_in_neither_series": [str(t) for t in grid.difference(axis)]}


def main() -> None:
    p = _fetch.parser(__doc__)
    p.add_argument("--start", default=START, help=f"first UTC day (default {START})")
    p.add_argument("--end", default=END, help=f"last UTC day, inclusive (default {END})")
    p.add_argument("--fill-until", default=FILL_UNTIL,
                   help=f"last UTC day on which DAForecast gaps are filled (default {FILL_UNTIL})")
    args = p.parse_args()
    out = _fetch.out_dir(args)
    df, info = build(args.start, args.end, args.fill_until)
    _fetch.write(df, out, STEM, {
        "source_urls": [ACTUAL, DAYAHEAD],
        "source_organization": "Elexon Limited",
        "window": [args.start, args.end],
        **info,
        **_fetch.describe(df)})


if __name__ == "__main__":
    main()
