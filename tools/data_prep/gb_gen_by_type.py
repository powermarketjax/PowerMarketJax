"""Download Elexon actual generation by fuel type into the form of ``gb_gen_by_type``.

Source: Elexon Insights Solution API,
``GET https://data.elexon.co.uk/bmrs/api/v1/generation/actual/per-type``
(dataset AGPT, the B1620 "actual aggregated generation per type" report).
No registration and no key.

Licence: not yet reviewed for redistribution; see the Data section of the
README.  This script only downloads; the vendored copy is not redistributed.

Output: ``GB_Gen_by_Type_2016_2025_30min.parquet`` and its sidecar json: one
row per settlement period in UTC (``startTime``), ``settlementPeriod``, and
one MW column per fuel type (``psrType``) in the order of the vendored file.

Processing decisions:

* **Requests are cut into 7-day windows and every window's length is
  checked.**  The endpoint does not reject a longer range, it truncates it
  (a one-month request measured 2026-09-22 returned 732 of 1 488 periods).
  ``from`` and ``to`` are both inclusive, so the shared boundary period of two
  windows is fetched twice; the copies must agree or the script raises.
* **The time grid is the full 30-minute UTC grid from the first to the last
  period.**  Periods the API does not return get their ``settlementPeriod``
  from the Europe/London calendar, the same rule the API uses (asserted on
  every period it does return).
* **Missing values inside the series are linearly interpolated in time**,
  and counted per column in the sidecar (``interpolated_points``).
* **This does not reproduce the vendored file.**  Compared on 2026-09-22 the
  two agree on the time axis and ``settlementPeriod`` but differ in every
  fuel column, from 9 rows (``Fossil Oil``) to 32 274 (``Hydro Pumped
  Storage``) of 180 048.  In 97-99% of the differing runs the vendored values
  are a straight line between two points that both files share, across
  points the API publishes today; which points the vendored file replaced,
  and why, is not recoverable from it.  Nothing in the environments reads
  this dataset.
* **A fuel type not yet reported is zero, not interpolated.**  ``Biomass`` is
  not reported before late 2017; its leading missing values are set to 0, as
  in the vendored file.

    python tools/data_prep/gb_gen_by_type.py --out-dir /tmp/pmj-data
"""
from __future__ import annotations

import pandas as pd

import _fetch

NAME = "gb_gen_by_type"
STEM = "GB_Gen_by_Type_2016_2025_30min"
API = "https://data.elexon.co.uk/bmrs/api/v1/generation/actual/per-type"
START, END = "2016-01-01", "2026-04-08"
FUELS = [
    "Fossil Gas", "Fossil Hard coal", "Fossil Oil", "Hydro Pumped Storage",
    "Hydro Run-of-river and poundage", "Nuclear", "Other", "Solar",
    "Wind Offshore", "Wind Onshore", "Biomass",
]
WINDOW = pd.Timedelta(days=7)


def settlement_period(t: pd.Series) -> pd.Series:
    """GB settlement period of a UTC period start: half hours since London midnight, plus one."""
    local = t.dt.tz_convert("Europe/London")
    midnight = local.dt.normalize().dt.tz_convert("UTC")
    return ((t - midnight) // pd.Timedelta(minutes=30) + 1).astype("int64")


def fetch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = []
    t = start
    while t < end:
        u = min(t + WINDOW, end)
        got = _fetch.get_json(f"{API}?from={t:%Y-%m-%dT%H:%MZ}&to={u:%Y-%m-%dT%H:%MZ}&format=json")["data"]
        want = int((u - t) / pd.Timedelta(minutes=30)) + 1
        if len(got) > want:
            raise RuntimeError(f"{t}..{u}: {len(got)} periods returned, at most {want} possible")
        for r in got:
            rec = {"startTime": r["startTime"], "settlementPeriod": r["settlementPeriod"]}
            for d in r["data"]:
                if d["psrType"] in rec:
                    raise RuntimeError(f"{r['startTime']}: {d['psrType']} reported twice")
                rec[d["psrType"]] = d["quantity"]
            rows.append(rec)
        print(f"  {t:%Y-%m-%d}: {len(got)} of {want} periods")
        t = u
    df = pd.DataFrame(rows)
    df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
    return df


def build(start: str, end: str) -> tuple[pd.DataFrame, dict]:
    t0 = pd.Timestamp(start, tz="UTC")
    t1 = pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23, minutes=30)
    raw = fetch(t0, t1)
    unknown = set(raw.columns) - {"startTime", "settlementPeriod", *FUELS}
    if unknown:
        raise RuntimeError(f"fuel types not in the vendored schema: {sorted(unknown)}")
    raw = raw.reindex(columns=["startTime", "settlementPeriod", *FUELS])

    dup = raw.duplicated("startTime", keep=False)
    if dup.any():
        spread = raw[dup].groupby("startTime")[FUELS].nunique(dropna=False)
        if (spread > 1).any().any():
            raise RuntimeError("a period fetched twice has two different values")
        raw = raw[~raw.duplicated("startTime")]
    raw = raw.set_index("startTime").sort_index()

    grid = pd.date_range(t0, t1, freq="30min", name="startTime")
    df = raw.reindex(grid)
    sp = settlement_period(grid.to_series())
    have = df["settlementPeriod"].notna()
    if not (df.loc[have, "settlementPeriod"].astype("int64") == sp[have]).all():
        raise RuntimeError("API settlementPeriod disagrees with the Europe/London calendar")
    df["settlementPeriod"] = sp

    first = df["Biomass"].first_valid_index()
    df.loc[df.index < first, "Biomass"] = 0.0
    missing = {c: int(df[c].isna().sum()) for c in FUELS}
    df[FUELS] = df[FUELS].astype("float64").interpolate(method="time", limit_area="inside")
    if df[FUELS].isna().any().any():
        raise RuntimeError("missing values at the series edges; widen or narrow the window")
    df = df.reset_index()
    df["startTime"] = _fetch.to_ns(df["startTime"])
    return df, {"missing_before_fill": missing,
                "biomass_zero_before": str(first),
                "api_periods_returned": int(have.sum())}


def main() -> None:
    p = _fetch.parser(__doc__)
    p.add_argument("--start", default=START, help=f"first UTC day (default {START})")
    p.add_argument("--end", default=END, help=f"last UTC day, inclusive (default {END})")
    args = p.parse_args()
    out = _fetch.out_dir(args)
    df, info = build(args.start, args.end)
    _fetch.write(df, out, STEM, {
        "source_url": API,
        "source_organization": "Elexon Limited",
        "window": [args.start, args.end],
        "interpolated_points": info["missing_before_fill"],
        **{k: v for k, v in info.items() if k != "missing_before_fill"},
        **_fetch.describe(df)})


if __name__ == "__main__":
    main()
