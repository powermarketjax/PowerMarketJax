"""Fetch the RTS-GMLC load and renewable time series into this repository's parquet form.

Offline, run once, not imported by anything.  It gives `case73rts` the demand
series it does not carry itself::

    python tools/data_prep/rts_gmlc_timeseries.py

**What is stored is the ingredients, not the net load.**  Eleven columns: the
system load and the four renewable classes RTS-GMLC gives as time series
(hydro, utility PV, rooftop PV, wind), each in both its day-ahead and its
real-time vintage, hourly.  Net load is a subtraction the consumer performs, so
a run that nets hydro but not rooftop PV, or nets nothing, re-derives its own
series from this file rather than needing a second fetch.  The 5-minute
real-time series are aggregated 12:1 to hourly here.

**`--resolution 30min` writes a second file and it is not the same shape.**
Market 02 runs 48 periods a day, so it needs the real-time series at 30 minutes;
that pass aggregates the same 5-minute sources 6:1 instead and writes the five
real-time columns only::

    python tools/data_prep/rts_gmlc_timeseries.py --resolution 30min

The day-ahead columns are absent from it deliberately: they are hourly at the
source, so putting them in a 30-minute frame would mean upsampling a forecast
this repository did not make.  A consumer that wants both reads the hourly file
for the day-ahead leg, which is what `load_rts_demand` already does.  The
5-minute resolution itself is still left at the source: storing it would make
the file six times larger again for a period length nothing here runs at.

**The day-ahead load is not a forecast of the real-time load.**  Measured over
the 8 784 hours of 2020: within each of the three regions the ratio of hourly
real-time load to day-ahead load is **constant to within the rounding of the
source files** -- means 0.9198, 0.9981 and 0.9943 with standard deviations of
2.2e-5, 3.0e-5 and 2.6e-5.  The two series are one signal times a constant to
five significant figures, so the load carries no forecast error at all, and
the day-ahead market's like-for-like forecast test fails as loudly as it
can: the realisation exceeds the forecast in 0.0% of hours where an unbiased
pair sits near half.  Rescaling the day-ahead load by the mean hourly ratio
(0.97165, the sidecar's `system_ratio_mean`) restores 50.6% and leaves a
median relative gap of 0.16%, i.e. nothing.  The factor has to be named: the
same sidecar also carries `rescaling_factor` = 0.97147, the sum ratio used for
the net-load residual below, and rescaling by that one gives 53.6% instead.

The renewables are different, and that is what makes a real forecast pair
available here at all.  Their two vintages are genuinely different series --
the wind residual has a standard deviation of 462 MW against a 814 MW mean, the
utility PV residual 104 MW -- so **net** load does carry a real forecast error
even though load alone does not.  Measured on the sum of the four classes, with
the day-ahead load first rescaled by its constant: median bias +43.5 MW, median
relative gap 11.61%, realisation above forecast in 57.0% of hours.  A consumer
that nets renewables gets a forecast/realisation pair whose error is real data
and lives entirely in the renewable term; a consumer that does not net them
gets no error at all.  Both facts have to travel with the file, which is why
neither the netting nor the rescaling is applied here.

**Timestamps are naive.**  The source gives Year/Month/Day/Period with no zone
and no zone is invented: `datetime` is a naive hourly timestamp over calendar
2020, Period 1 mapping to hour 0.  RTS-GMLC places its buses in Arizona,
southern California and Nevada, but nothing in the files says which clock the
periods are on, and a guessed zone would silently shift the solar profile.

Source: RTS-GMLC, <https://github.com/GridMod/RTS-GMLC>,
``RTS_Data/timeseries_data_files/``.  The data use notice in
``powermarketjax/case/raw_cases/rts_gmlc/LICENCE.md`` grants use, copying and
distribution provided the notice travels with the data, and requires that any
publication using it credit DOE/NREL/ALLIANCE.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT_PARQUET = REPO / "powermarketjax/data/parquet/RTS_GMLC_Load_and_Renewables_2020_60min.parquet"
OUT_META = OUT_PARQUET.with_suffix(".json")
OUT_MANIFEST = REPO / "powermarketjax/data/manifests/rts_gmlc_timeseries.json"

OUT_PARQUET_30 = REPO / ("powermarketjax/data/parquet/"
                         "RTS_GMLC_RealTime_Load_and_Renewables_2020_30min.parquet")
OUT_META_30 = OUT_PARQUET_30.with_suffix(".json")
OUT_MANIFEST_30 = REPO / "powermarketjax/data/manifests/rts_gmlc_timeseries_30min.json"

BASE = ("https://raw.githubusercontent.com/GridMod/RTS-GMLC/master/"
        "RTS_Data/timeseries_data_files")

#: ``(column stem, directory, day-ahead file, real-time file)``.
SERIES = [
    ("load", "Load", "DAY_AHEAD_regional_Load.csv", "REAL_TIME_regional_Load.csv"),
    ("hydro", "Hydro", "DAY_AHEAD_hydro.csv", "REAL_TIME_hydro.csv"),
    ("pv", "PV", "DAY_AHEAD_pv.csv", "REAL_TIME_pv.csv"),
    ("rtpv", "RTPV", "DAY_AHEAD_rtpv.csv", "REAL_TIME_rtpv.csv"),
    ("wind", "WIND", "DAY_AHEAD_wind.csv", "REAL_TIME_wind.csv"),
]

LICENCE = "DOE/NREL/ALLIANCE data use disclaimer agreement"
ATTRIBUTION = (
    "RTS-GMLC time series, National Renewable Energy Laboratory (NREL) operated "
    "by Alliance for Sustainable Energy LLC for the U.S. Department of Energy. "
    "Changes: per-generator columns summed to a system total, real-time "
    "5-minute series aggregated to period means (12:1 for the hourly file, 6:1 "
    "for the 30-minute one), Year/Month/Day/Period replaced by a naive "
    "timestamp. No netting and no rescaling.")


def _fetch(url: str) -> pd.DataFrame:
    with urllib.request.urlopen(url, timeout=300) as fh:
        return pd.read_csv(io.BytesIO(fh.read()))


def _system_total(frame: pd.DataFrame, periods: int, factor: int) -> np.ndarray:
    """Sum the per-generator (or per-region) columns and average ``factor``:1.

    The first four columns are Year/Month/Day/Period in every one of these
    files.  ``factor`` is stated by the caller rather than inferred from the row
    count, so a file whose resolution moved upstream fails here instead of being
    silently averaged at whatever ratio happens to divide: a day-ahead file that
    became 5-minute would previously have been taken for the hourly one.
    """
    total = frame.iloc[:, 4:].sum(axis=1).to_numpy(np.float64)
    if len(total) != periods * factor:
        raise SystemExit(f"unexpected row count {len(total)} against "
                         f"{periods} periods x {factor}")
    if factor == 1:
        return total
    return total.reshape(periods, factor).mean(axis=1)


def main() -> None:
    index = pd.date_range("2020-01-01", "2020-12-31 23:00", freq="1h")
    hours = len(index)                       # 8784, 2020 being a leap year
    out = {"datetime": index}
    regional_ratio_std = {}
    for stem, folder, da_file, rt_file in SERIES:
        frames = {}
        for vintage, name in (("da", da_file), ("rt", rt_file)):
            frames[vintage] = _fetch(f"{BASE}/{folder}/{name}")
            out[f"{stem}_{vintage}_mw"] = _system_total(
                frames[vintage], hours, 1 if vintage == "da" else 12)
            print(f"  {stem}_{vintage}: {len(frames[vintage]):,} source rows", flush=True)
        if stem == "load":
            # the claim in the docstring is about each region separately, so it
            # is measured on each region separately before the columns are summed
            for region in frames["da"].columns[4:]:
                da = frames["da"][region].to_numpy(np.float64)
                rt = frames["rt"][region].to_numpy(np.float64).reshape(hours, 12).mean(axis=1)
                ratio_r = rt / da
                regional_ratio_std[str(region)] = {
                    "mean": float(ratio_r.mean()), "std": float(ratio_r.std()),
                    "share_rt_above_da": float((ratio_r > 1).mean())}
    df = pd.DataFrame(out)
    df.to_parquet(OUT_PARQUET, compression="zstd", index=False)

    ratio = df.load_rt_mw / df.load_da_mw
    renew_da = df[[f"{s}_da_mw" for s, *_ in SERIES if s != "load"]].sum(axis=1)
    renew_rt = df[[f"{s}_rt_mw" for s, *_ in SERIES if s != "load"]].sum(axis=1)
    scale = df.load_rt_mw.sum() / df.load_da_mw.sum()
    net_da = df.load_da_mw * scale - renew_da
    net_rt = df.load_rt_mw - renew_rt
    residual = net_rt - net_da

    meta = {
        "parquet_file": OUT_PARQUET.name,
        "source_organization": "National Renewable Energy Laboratory (NREL) / GridMod",
        "source_url": f"{BASE}/",
        "licence": LICENCE,
        "licence_url": "https://github.com/GridMod/RTS-GMLC#data-use-disclaimer-agreement",
        "licence_permits_redistribution": True,
        "licence_evidence": (
            "the notice in the source repository's README grants use, copying "
            "and distribution for any purpose provided the entire notice "
            "appears in all copies; it is reproduced verbatim in "
            "powermarketjax/case/raw_cases/rts_gmlc/LICENCE.md"),
        "attribution": ATTRIBUTION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "resolution": "60min",
        "timestamp_convention": (
            "naive hourly timestamps over calendar 2020, Period 1 -> hour 0. "
            "The source gives Year/Month/Day/Period with no zone and none is "
            "invented here; a guessed zone would shift the solar profile."),
        "day_ahead_load_is_not_a_forecast": {
            "note": (
                "within each region the hourly real-time/day-ahead load ratio is "
                "constant to the source's rounding (std ~3e-5 against means of "
                "0.92-1.00), so the two load series are one signal times "
                "a constant and the load carries no forecast error. The "
                "renewable pairs do differ, so a consumer that nets renewables "
                "gets a real forecast error and one that does not gets none."),
            "regional_ratio": regional_ratio_std,
            "system_ratio_mean": float(ratio.mean()),
            "system_ratio_std": float(ratio.std()),
            "net_load_residual_after_rescaling": {
                "median_bias_mw": float(residual.median()),
                "median_relative": float((residual.abs() / net_rt.abs().clip(lower=1)).median()),
                "share_positive": float((residual > 0).mean()),
                "rescaling_factor": float(scale),
            },
        },
        "net_load_if_all_four_classes_netted": {
            "day_ahead_min_mw": float(net_da.min()),
            "day_ahead_max_mw": float(net_da.max()),
            "real_time_min_mw": float(net_rt.min()),
            "real_time_max_mw": float(net_rt.max()),
            "hours_below_zero_real_time": int((net_rt < 0).sum()),
            "note": (
                "negative hours are real: the four classes together exceed "
                "demand in them. A thermal-only clearing cannot serve a "
                "negative net load and has no over-generation slack, so the "
                "consumer decides what to do about them and says so."),
        },
        "gap_policy": "no gaps observed and none would be filled",
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "date_range": [str(df["datetime"].min()), str(df["datetime"].max())],
        "date_ranges": {
            "datetime": {
                "min": str(df["datetime"].min()), "max": str(df["datetime"].max()),
                "count": int(len(df)), "missing": int(df["datetime"].isna().sum()),
            }
        },
        "numeric_statistics": {
            c: {"count": int(df[c].count()), "mean": float(df[c].mean()),
                "min": float(df[c].min()), "max": float(df[c].max()),
                "missing": int(df[c].isna().sum())}
            for c in df.columns if c != "datetime"
        },
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "name": "rts_gmlc_timeseries",
        "source": "rts_gmlc",
        "data_type": "actual_series",
        "time_mode": "calendar",
        "resolution": "60min",
        "parquet_file": OUT_PARQUET.name,
        "metadata_json": OUT_META.name,
        "column_map": {
            "load_da_mw": "load.rts_da_mw",
            "load_rt_mw": "load.rts_rt_mw",
            "hydro_da_mw": "hydro.rts_da_mw",
            "hydro_rt_mw": "hydro.rts_rt_mw",
            "pv_da_mw": "solar.rts_pv_da_mw",
            "pv_rt_mw": "solar.rts_pv_rt_mw",
            "rtpv_da_mw": "solar.rts_rtpv_da_mw",
            "rtpv_rt_mw": "solar.rts_rtpv_rt_mw",
            "wind_da_mw": "wind.rts_da_mw",
            "wind_rt_mw": "wind.rts_rt_mw",
        },
        "index_map": {"datetime": "datetime"},
        "derived": {},
        "normalize": {},
        "data_epoch": None,
        "cyclical": False,
        "region_values": [],
        "date_range": [str(df["datetime"].min().date()), str(df["datetime"].max().date())],
        "source_url": f"{BASE}/",
        "source_organization": "National Renewable Energy Laboratory (NREL) / GridMod",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{OUT_PARQUET.name}: {OUT_PARQUET.stat().st_size / 1e3:.1f} kB, {len(df):,} rows")
    print(f"  load ratio rt/da: mean {ratio.mean():.4f}, std {ratio.std():.6f}")
    print(f"  net load (all four netted): da {net_da.min():.0f}..{net_da.max():.0f}, "
          f"rt {net_rt.min():.0f}..{net_rt.max():.0f}, "
          f"{int((net_rt < 0).sum())} negative hours")
    print(f"  net-load residual after rescaling: median |rel| "
          f"{100 * (residual.abs() / net_rt.abs().clip(lower=1)).median():.2f}%, "
          f"share positive {100 * (residual > 0).mean():.1f}%")


def main_30min() -> None:
    """The 30-minute real-time sibling: five columns, 6:1 from the same sources.

    It is checked against the hourly file rather than only against itself: a
    2:1 aggregation of what this writes has to reproduce the ``*_rt_mw`` columns
    already in the tree, because both are means of the same 5-minute samples.
    That number goes into the sidecar, so a consumer can recompute it without
    re-fetching (a product needs a content criterion of its own, and a commit
    stamp is not one).
    """
    index = pd.date_range("2020-01-01", "2020-12-31 23:30", freq="30min")
    periods = len(index)                     # 17568, 2020 being a leap year
    out = {"datetime": index}
    for stem, folder, _da_file, rt_file in SERIES:
        frame = _fetch(f"{BASE}/{folder}/{rt_file}")
        out[f"{stem}_rt_mw"] = _system_total(frame, periods, 6)
        print(f"  {stem}_rt: {len(frame):,} source rows", flush=True)
    df = pd.DataFrame(out)
    df.to_parquet(OUT_PARQUET_30, compression="zstd", index=False)

    hourly = pd.read_parquet(OUT_PARQUET)
    hours = len(hourly)
    worst = {}
    for stem, *_ in SERIES:
        col = f"{stem}_rt_mw"
        two_to_one = df[col].to_numpy(np.float64).reshape(hours, 2).mean(axis=1)
        worst[col] = float(np.abs(two_to_one - hourly[col].to_numpy(np.float64)).max())

    renew_rt = df[[f"{s}_rt_mw" for s, *_ in SERIES if s != "load"]].sum(axis=1)
    net_rt = df.load_rt_mw - renew_rt

    meta = {
        "parquet_file": OUT_PARQUET_30.name,
        "source_organization": "National Renewable Energy Laboratory (NREL) / GridMod",
        "source_url": f"{BASE}/",
        "licence": LICENCE,
        "licence_url": "https://github.com/GridMod/RTS-GMLC#data-use-disclaimer-agreement",
        "licence_permits_redistribution": True,
        "licence_evidence": (
            "the notice in the source repository's README grants use, copying "
            "and distribution for any purpose provided the entire notice "
            "appears in all copies; it is reproduced verbatim in "
            "powermarketjax/case/raw_cases/rts_gmlc/LICENCE.md"),
        "attribution": ATTRIBUTION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "resolution": "30min",
        "real_time_vintage_only": (
            "the five real-time series and nothing else. The day-ahead columns "
            "are hourly at the source and are not upsampled here; a consumer "
            "that needs the day-ahead leg reads the 60-minute file, which is "
            "what `load_rts_demand` does."),
        "timestamp_convention": (
            "naive 30-minute timestamps over calendar 2020, Period 1 -> 00:00. "
            "The source gives Year/Month/Day/Period with no zone and none is "
            "invented here; a guessed zone would shift the solar profile."),
        "aggregation": (
            "each column is the mean of six consecutive 5-minute samples of the "
            "per-generator sum, the same samples the 60-minute file averages "
            "twelve at a time."),
        "consistency_with_hourly_file": {
            "note": (
                "this file aggregated 2:1 against the `*_rt_mw` columns of "
                f"{OUT_PARQUET.name}; both are means of the same 5-minute "
                "samples, so the difference is float64 rounding only. It is a "
                "content criterion for this artefact: recomputable from the two "
                "files alone, without the source or this script."),
            "hourly_file": OUT_PARQUET.name,
            "max_abs_difference_mw": worst,
        },
        "net_load_if_all_four_classes_netted": {
            "real_time_min_mw": float(net_rt.min()),
            "real_time_max_mw": float(net_rt.max()),
            "periods_below_zero_real_time": int((net_rt < 0).sum()),
            "note": (
                "negative periods are real and nothing is removed: the four "
                "classes together exceed demand in them. What to do about them "
                "is the consumer's choice and `load_rts_demand_half_hourly` "
                "takes it as `floor_mw`."),
        },
        "gap_policy": "no gaps observed and none would be filled",
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "date_range": [str(df["datetime"].min()), str(df["datetime"].max())],
        "date_ranges": {
            "datetime": {
                "min": str(df["datetime"].min()), "max": str(df["datetime"].max()),
                "count": int(len(df)), "missing": int(df["datetime"].isna().sum()),
            }
        },
        "numeric_statistics": {
            c: {"count": int(df[c].count()), "mean": float(df[c].mean()),
                "min": float(df[c].min()), "max": float(df[c].max()),
                "missing": int(df[c].isna().sum())}
            for c in df.columns if c != "datetime"
        },
    }
    OUT_META_30.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                           encoding="utf-8")

    manifest = {
        "name": "rts_gmlc_timeseries_30min",
        "source": "rts_gmlc",
        "data_type": "actual_series",
        "time_mode": "calendar",
        "resolution": "30min",
        "parquet_file": OUT_PARQUET_30.name,
        "metadata_json": OUT_META_30.name,
        # the signal names carry `_30min` because `DatasetRegistry.resolve_signals`
        # takes the first manifest that claims a name and says nothing when two
        # do; reusing `load.rts_rt_mw` here would make which file a caller gets
        # depend on the glob order of the manifest directory
        "column_map": {
            "load_rt_mw": "load.rts_rt_30min_mw",
            "hydro_rt_mw": "hydro.rts_rt_30min_mw",
            "pv_rt_mw": "solar.rts_pv_rt_30min_mw",
            "rtpv_rt_mw": "solar.rts_rtpv_rt_30min_mw",
            "wind_rt_mw": "wind.rts_rt_30min_mw",
        },
        "index_map": {"datetime": "datetime"},
        "derived": {},
        "normalize": {},
        "data_epoch": None,
        "cyclical": False,
        "region_values": [],
        "date_range": [str(df["datetime"].min().date()), str(df["datetime"].max().date())],
        "source_url": f"{BASE}/",
        "source_organization": "National Renewable Energy Laboratory (NREL) / GridMod",
    }
    OUT_MANIFEST_30.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                               encoding="utf-8")

    print(f"{OUT_PARQUET_30.name}: {OUT_PARQUET_30.stat().st_size / 1e3:.1f} kB, "
          f"{len(df):,} rows")
    print(f"  2:1 back to {OUT_PARQUET.name}: worst column "
          f"{max(worst.values()):.3e} MW")
    print(f"  net load (all four netted): {net_rt.min():.0f}..{net_rt.max():.0f} MW, "
          f"{int((net_rt < 0).sum())} negative half-hours")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--resolution", choices=("60min", "30min"), default="60min",
                    help="60min writes the eleven-column day-ahead + real-time "
                         "file; 30min writes the five real-time columns only")
    args = ap.parse_args()
    sys.exit(main() if args.resolution == "60min" else main_30min())
