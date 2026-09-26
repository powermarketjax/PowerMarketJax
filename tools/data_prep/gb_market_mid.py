"""Download Elexon market index data (MID) into the form of ``gb_market_mid``.

Source: Elexon Insights Solution API,
``GET https://data.elexon.co.uk/bmrs/api/v1/datasets/MID``, the market index
price and volume reported by the two market index data providers, APX
(``APXMIDP``) and N2EX (``N2EXMIDP``).  No registration and no key.

Licence: not yet reviewed for redistribution; see the Data section of the
README.  This script only downloads; the vendored copy is not redistributed.

Output: ``MID_GB_30min_aligned_to_gen.parquet`` and its sidecar json:
``startTime`` (UTC period start), ``settlementPeriod``, and
``mid_{price,volume}_{APXMIDP,N2EXMIDP}`` (GBP/MWh and MWh).

Processing decisions, each one the rule that reproduces the vendored file:

* **The time axis is that of ``gb_forecast_actual_demand``** (the union of
  the periods Elexon ATL and DATL return), so the two GB files align row for
  row; it is fetched here with the same code, not read from a file.
* **Negative prices are removed and filled by default.**  The vendored file
  has no negative price, while MID has 1 326 negative APX half-hours in the
  default window (measured 2026-09-22), and every one of them is replaced in
  the vendored file by a value interpolated from its neighbours.  This script
  does the same so that it reproduces the vendored series; ``--keep-negative``
  keeps MID's published prices instead.  The removal is a property of the
  vendored file, not of the market: negative GB day-ahead prices are real.
* **Gaps are filled by linear interpolation over positions**, and a gap at
  either end of the series by 0, which is how the vendored file carries
  periods MID does not report.

    python tools/data_prep/gb_market_mid.py --out-dir /tmp/pmj-data
"""
from __future__ import annotations

import pandas as pd

import _fetch
import gb_forecast_actual_demand as demand
from gb_gen_by_type import settlement_period

NAME = "gb_market_mid"
STEM = "MID_GB_30min_aligned_to_gen"
API = "https://data.elexon.co.uk/bmrs/api/v1/datasets/MID"
START, END = demand.START, demand.END
PROVIDERS = ("APXMIDP", "N2EXMIDP")
COLUMNS = ["startTime", "settlementPeriod",
           *(f"mid_price_{p}" for p in PROVIDERS), *(f"mid_volume_{p}" for p in PROVIDERS)]


def fetch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = []
    t = start
    while t < end:
        u = min(t + demand.WINDOW, end)
        rows += _fetch.get_json(f"{API}?from={t:%Y-%m-%dT%H:%MZ}&to={u:%Y-%m-%dT%H:%MZ}&format=json")["data"]
        t = u
    df = pd.DataFrame(rows).drop_duplicates()
    df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
    if df.duplicated(["startTime", "dataProvider"]).any():
        raise RuntimeError("a (period, provider) pair carries two different records")
    return df[(df["startTime"] >= start) & (df["startTime"] <= end)]


def build(start: str, end: str, keep_negative: bool) -> tuple[pd.DataFrame, dict]:
    t0 = pd.Timestamp(start, tz="UTC")
    t1 = pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23, minutes=30)
    actual, _ = demand.series(demand.ACTUAL, t0, t1)
    dayahead, _ = demand.series(demand.DAYAHEAD, t0, t1)
    axis = actual.index.union(dayahead.index)

    raw = fetch(t0, t1)
    wide = raw.pivot(index="startTime", columns="dataProvider", values=["price", "volume"])
    wide.columns = [f"mid_{q}_{p}" for q, p in wide.columns]
    wide = wide.reindex(axis)
    info = {"negative_prices_removed": {}, "filled_points": {}}
    for c in COLUMNS[2:]:
        s = wide[c].astype("float64")
        if c.startswith("mid_price_") and not keep_negative:
            info["negative_prices_removed"][c] = int((s < 0).sum())
            s = s.mask(s < 0)
        info["filled_points"][c] = int(s.isna().sum())
        wide[c] = s.interpolate(method="linear", limit_area="inside").fillna(0.0)

    sp = raw.drop_duplicates("startTime").set_index("startTime")["settlementPeriod"].reindex(axis)
    sp = sp.fillna(settlement_period(axis.to_series()))
    wide["settlementPeriod"] = sp.astype("int64")
    df = wide.rename_axis("startTime").reset_index()[COLUMNS]
    df["startTime"] = _fetch.to_ns(df["startTime"])
    return df, info


def main() -> None:
    p = _fetch.parser(__doc__)
    p.add_argument("--start", default=START, help=f"first UTC day (default {START})")
    p.add_argument("--end", default=END, help=f"last UTC day, inclusive (default {END})")
    p.add_argument("--keep-negative", action="store_true",
                   help="keep negative prices as published instead of reproducing the vendored fill")
    args = p.parse_args()
    out = _fetch.out_dir(args)
    df, info = build(args.start, args.end, args.keep_negative)
    _fetch.write(df, out, STEM, {
        "source_url": API,
        "source_organization": "Elexon Limited",
        "window": [args.start, args.end],
        "keep_negative": args.keep_negative,
        **info,
        **_fetch.describe(df)})


if __name__ == "__main__":
    main()
