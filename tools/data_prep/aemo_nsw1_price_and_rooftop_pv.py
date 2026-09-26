"""Fetch the NSW1 spot price and rooftop photovoltaic actuals into parquet form.

Offline, run once, not imported by anything.  It supplies the two series still
missing from the **Australian** configuration of the local flexibility
market: it needs an exogenous energy
price, and without it the replacement cost has no value, so the markup map
has no origin and the reward has no economic meaning.

**This script will not fetch anything until the redistribution question is
settled, and it enforces that itself** (see ``--licence-cleared``).  The state of
that question as of 2026-08-15:

* AEMO's own distribution, NEMWeb, **does not report a licence in the response**.
  There is no ``Link: rel="license"`` header, no licence field and no licence file
  in the archive directories.  This is the opposite of the two Swiss sources,
  whose licences were read out of the retrieval response itself.
* ``aemo.com.au`` returns **403 to every automated client**, so the copyright
  notice cannot be re-verified programmatically.  Its text was read from an
  Internet Archive capture dated 2026-07-29.
* That text grants "general permission for anyone to use AEMO Material for any
  purpose, but only with accurate and appropriate attribution".  It says **use**.
  It does not contain words granting reproduction, redistribution or
  republication, which is what vendoring into a public repository is.
* "AEMO Material" is material "created by or on behalf of AEMO".  Dispatch prices
  and rooftop photovoltaic estimates are computed by AEMO's own systems and sit on
  the favourable side of that line; participant-submitted data would not.
* Two search summaries claimed the notice is, and is not, compatible with CC BY
  4.0.  **The text never mentions CC BY.**  That claim is excluded, not merely
  unverified, and must not re-enter from anywhere.

Routing around this through an aggregator does not solve it.  A second-hand
distributor that self-reports CC BY satisfies the *form* of "licence read from the
response" while sourcing the same AEMO bytes; if AEMO's terms do not permit
redistribution, a downstream CC BY claim launders the licence rather than
resolving it.  Such a second-hand source is, independently, only second-tier.

The same unresolved notice already covers ``aemo_5min_demand``, ``aemo_forecast``
and ``ausgrid_zone_substation_fy25_imputed``, which are in the tree today, so one
ruling settles four datasets rather than one.

Two series, and why each:

``DISPATCHPRICE``   the 5-minute regional reference price, **not** the 30-minute
                    ``TRADINGPRICE``.  The demand series is 15-minute, so 5-minute
                    aggregates 3:1 onto its grid exactly, while 30 minutes would
                    have to be interpolated *down*.  Take ``INTERVENTION = 0``:
                    intervention runs are re-runs of the same interval and taking
                    both duplicates timestamps, which the loader rejects.
``ROOFTOP_PV_ACTUAL`` AEMO's 30-minute regional estimate of distributed rooftop
                    output.  It is the right pairing for a zone-substation demand
                    series because both sit at distribution level; the utility
                    solar in ``DISPATCH_UNIT_SCADA`` does not.  Take
                    ``TYPE = MEASUREMENT``.

**Two conventions in this data are traps and both are recorded in the sidecar
rather than left in a function signature.**

*NEM timestamps are interval-ENDING and are in fixed UTC+10 all year* -- "market
time" never observes daylight saving, unlike the Ausgrid demand series it pairs
with, which is local civil time and does.  A naive read puts every price 5 minutes
late and 10 hours west, and neither error is visible in a plot.

*The photovoltaic series is upsampled by time interpolation, not forward fill.*
It is 30-minute and the demand grid is 15-minute.  Power is a rate, so a
half-hourly value forward-filled onto both quarter hours states that output was
flat across the half hour and then stepped; interpolating states it ramped.  For a
solar profile the ramp is the physical claim.  ``data.py`` leaves this choice to
the caller, which is right for a library and wrong for a stored artefact, so the
choice this file made is written into the sidecar as a declaration.

    python tools/data_prep/aemo_nsw1_price_and_rooftop_pv.py --dry-run
    python tools/data_prep/aemo_nsw1_price_and_rooftop_pv.py --licence-cleared "<ruling>"
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Imported rather than restated, so the coverage recorded in each sidecar cannot
# drift from the windows the loader actually offers.
from powermarketjax.envs.local_flexibility.data import AEDT_WINDOW, AEST_WINDOW

REPO = Path(__file__).resolve().parents[2]
PARQUET_DIR = REPO / "powermarketjax/data/parquet"
MANIFEST_DIR = REPO / "powermarketjax/data/manifests"

_DIR = ("https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/"
        "{year}/MMSDM_{year}_{month:02d}/MMSDM_Historical_Data_SQLLoader/DATA/")
#: **The archive renamed its files between 2024-07 and 2024-08** and the two forms
#: are disjoint -- each month answers 404 for the other one.  Measured 2026-08-15:
#: `PUBLIC_DVD_...` serves 2024-04 through 2024-07 and 404s from 2024-08;
#: `PUBLIC_ARCHIVE#...#FILE01#...` is the mirror image.  Trying one form only
#: silently truncates the window at the changeover, which lands in the middle of
#: the winter demand window this pairs with, so both are tried and the one that
#: answered is recorded per month.
ARCHIVE = _DIR + "PUBLIC_ARCHIVE%23{table}%23FILE01%23{year}{month:02d}010000.zip"
LEGACY = _DIR + "PUBLIC_DVD_{table}_{year}{month:02d}010000.zip"

REGION = "NSW1"
#: Market time: fixed UTC+10 the whole year, no daylight saving.  Not
#: Australia/Sydney, which is what the demand series is published in.
MARKET_UTC_OFFSET_HOURS = 10

SERIES = {
    "price": dict(
        table="DISPATCHPRICE", stamp="SETTLEMENTDATE", value="RRP",
        minutes=5, filters={"INTERVENTION": "0"},
        column="spot_price_aud_mwh", signal="market.spot_price_aud_mwh",
        parquet="AEMO_NSW1_Dispatch_Price_5min.parquet",
        manifest="aemo_nsw1_dispatch_price"),
    "pv": dict(
        table="ROOFTOP_PV_ACTUAL", stamp="INTERVAL_DATETIME", value="POWER",
        minutes=30, filters={"TYPE": "MEASUREMENT"},
        column="rooftop_pv_mw", signal="solar.substation_mw",
        parquet="AEMO_NSW1_Rooftop_PV_30min.parquet",
        manifest="aemo_nsw1_rooftop_pv"),
}

LICENCE = dict(
    licence="AEMO Copyright Permissions Notice",
    licence_url="https://www.aemo.com.au/privacy-and-legal-notices/copyright-permissions",
    licence_permits_redistribution=None,       # overwritten by --licence-cleared
    licence_text_read_from="Internet Archive capture 20260729145136 of the AEMO "
                           "page; aemo.com.au returns 403 to every automated "
                           "client, so the notice cannot be re-verified "
                           "programmatically",
    licence_in_retrieval_response=False,
    licence_note="The notice grants 'general permission for anyone to use AEMO "
                 "Material for any purpose, but only with accurate and appropriate "
                 "attribution of the relevant AEMO Material and AEMO as its "
                 "author'.  It grants *use*; it contains no words granting "
                 "reproduction, redistribution or republication.  'AEMO Material' "
                 "is material 'created by or on behalf of AEMO', which covers "
                 "dispatch prices and rooftop photovoltaic estimates (AEMO's own "
                 "systems compute both) but would not cover participant-submitted "
                 "data.  The claim that the notice is compatible with CC BY 4.0 "
                 "appears in search summaries and NOT in the notice, which never "
                 "mentions CC BY; that claim is excluded.",
    attribution="Australian Energy Market Operator (AEMO), NEMWeb MMSDM archive, "
                "tables DISPATCHPRICE and ROOFTOP_PV_ACTUAL, region NSW1.  Used "
                "with attribution to AEMO as author under the AEMO Copyright "
                "Permissions Notice.  Changes: non-intervention dispatch runs and "
                "MEASUREMENT-type rooftop estimates selected, region filtered to "
                "NSW1, interval-ending market time (UTC+10, no daylight saving) "
                "converted to interval-start UTC.")

INTERVAL_CONVENTION = (
    "Stored `datetime` is the UTC **start** of each interval.  The source stamps "
    "are interval-ENDING and are expressed in NEM market time, a fixed UTC+10 "
    "offset that never observes daylight saving -- unlike the "
    "`ausgrid_zone_substation_fy25_imputed` demand series this pairs with, which "
    "is Australia/Sydney local civil time and does.  Conversion is therefore "
    "`utc_start = stamp - 10h - interval_length`, applied uniformly with no "
    "transition handling, because market time has no transitions.  Skipping "
    "either term is invisible in a plot: the offset alone shifts the daily solar "
    "peak by ten hours, and the interval-ending term alone shifts every value by "
    "one period.")

RESAMPLING_DECLARATION = (
    "The rooftop photovoltaic series is 30-minute and the demand grid is "
    "15-minute.  **This artefact declares time interpolation, not forward fill.** "
    "Power is a rate: forward-filling a half-hourly value onto both quarter hours "
    "asserts that output was constant across the half hour and then stepped, "
    "while interpolating asserts it ramped, and for a solar profile the ramp is "
    "the physical claim.  `envs/local_flexibility/data.py` leaves the choice to "
    "the caller, which is correct for a library and insufficient for a stored "
    "artefact, so the choice is recorded here.  The 5-minute price needs no such "
    "declaration: it aggregates 3:1 onto the 15-minute grid by mean, exactly.")


def _fetch(url: str) -> tuple[bytes, str]:
    with urllib.request.urlopen(url, timeout=300) as r:
        blob = r.read()
    return blob, hashlib.sha256(blob).hexdigest()


def _fetch_month(table: str, year: int, month: int) -> tuple[bytes, str, str]:
    """Try both archive naming schemes; return the one that answered."""
    tried = []
    for pattern in (ARCHIVE, LEGACY):
        url = pattern.format(year=year, month=month, table=table)
        try:
            blob, digest = _fetch(url)
            return blob, digest, url
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            tried.append(url)
    raise RuntimeError(f"{table} {year}-{month:02d}: both naming schemes 404:\n  "
                       + "\n  ".join(tried))


def verify_conventions(frame: pd.DataFrame, kind: str, column: str) -> dict:
    """Falsifiable checks on the timestamp convention, not a restatement of it.

    A declaration in a docstring cannot fail.  These can, and each is aimed at one
    of the two terms in `utc_start = stamp - 10h - interval`:

    *the ten hours* -- rooftop photovoltaic output is zero at local night and
    peaks near local noon.  Get the offset wrong and the daily peak lands at local
    midnight, which this rejects.  The price series has no such shape, so the
    check runs on the photovoltaic series and the offset is shared code.

    *the interval* -- the first interval of an archive month must start exactly at
    the month boundary in market time.  Forget the interval-ending term and the
    earliest start is one period late, which this rejects.
    """
    out = {}
    local = (frame["datetime"].dt.tz_localize("UTC")
             .dt.tz_convert("Australia/Sydney"))
    market = frame["datetime"] + pd.Timedelta(hours=MARKET_UTC_OFFSET_HOURS)
    first = market.min()
    assert first == first.normalize().replace(day=1), (
        f"first interval starts at {first} in market time, not at a month "
        f"boundary: the interval-ending term is wrong")
    out["first_interval_start_market_time"] = str(first)

    if kind == "pv":
        night = frame[column][(local.dt.hour >= 22) | (local.dt.hour <= 4)]
        peak_hour = (frame.assign(h=local.dt.hour, d=local.dt.date)
                     .loc[frame.groupby(local.dt.date)[column].idxmax()]["h"])
        out["night_max_mw"] = float(night.max())
        out["median_local_peak_hour"] = float(peak_hour.median())
        assert out["night_max_mw"] < 1.0, (
            f"rooftop photovoltaic output reaches {out['night_max_mw']:.1f} MW "
            f"between 22:00 and 04:00 local: the UTC offset is wrong")
        assert 10 <= out["median_local_peak_hour"] <= 14, (
            f"median daily peak at local hour {out['median_local_peak_hour']}, "
            f"not near solar noon: the UTC offset is wrong")
    return out


def _mms_csv(blob: bytes) -> pd.DataFrame:
    """Parse one MMS flat file: 'C' comment rows, an 'I' header row, 'D' data rows.

    The header is not on line 1 and the file may carry several tables, so the
    columns are taken from the 'I' row rather than assumed.
    """
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0]).decode("utf-8", "replace")
    header, rows = None, []
    for line in raw.splitlines():
        if line.startswith("I,"):
            header = line.split(",")
        elif line.startswith("D,") and header is not None:
            cells = line.split(",")
            if len(cells) == len(header):
                rows.append(cells)
    if header is None:
        raise RuntimeError("no 'I' header row found; the MMS layout changed")
    return pd.DataFrame(rows, columns=[c.strip('"') for c in header])


def build_series(kind: str, months: list[tuple[int, int]], cleared: str) -> dict:
    spec = SERIES[kind]
    frames, archives = [], {}
    for year, month in months:
        blob, digest, url = _fetch_month(spec["table"], year, month)
        archives[f"{spec['table']}_{year}{month:02d}"] = dict(
            url=url, sha256=digest, bytes=len(blob),
            naming="ARCHIVE" if "PUBLIC_ARCHIVE" in url else "DVD")
        df = _mms_csv(blob)
        df = df[df["REGIONID"].str.strip('"') == REGION]
        for col, want in spec["filters"].items():
            df = df[df[col].str.strip('"') == want]
        out = pd.DataFrame({
            "stamp": pd.to_datetime(df[spec["stamp"]].str.strip('"')),
            spec["column"]: pd.to_numeric(df[spec["value"]], errors="raise")})
        frames.append(out)

    s = pd.concat(frames).sort_values("stamp")
    # interval-ending market time (UTC+10, no DST) -> interval-start UTC
    s["datetime"] = (s["stamp"]
                     - pd.Timedelta(hours=MARKET_UTC_OFFSET_HOURS)
                     - pd.Timedelta(minutes=spec["minutes"]))
    s = s.drop(columns="stamp")[["datetime", spec["column"]]]

    # The DVD-era files carry one extra day: `ROOFTOP_PV_ACTUAL` for 2024-04 runs
    # to 2024-05-02, so each DVD month overlaps the next by 48 half hours (4
    # changeovers x 48 = the 192 duplicates measured 2026-08-15).  The overlapping
    # rows are byte-identical republications, not revisions -- POWER agreed to
    # 0.000000 MW on all 48 and LASTCHANGED matched -- so dropping one copy is
    # lossless.  That is asserted rather than assumed: if AEMO ever republishes a
    # *revised* value, this raises instead of silently keeping an arbitrary copy.
    dup_mask = s["datetime"].duplicated(keep=False)
    n_dropped = 0
    if dup_mask.any():
        spread = (s[dup_mask].groupby("datetime")[spec["column"]]
                  .agg(lambda v: v.max() - v.min()))
        if float(spread.max()) != 0.0:
            worst = spread.idxmax()
            raise RuntimeError(
                f"duplicate timestamps disagree by up to {spread.max()} at "
                f"{worst}: these are revisions, not republications, and which "
                f"copy to keep is a decision this script must not make silently")
        n_dropped = int(s["datetime"].duplicated().sum())
        s = s[~s["datetime"].duplicated()].reset_index(drop=True)
    gaps = pd.date_range(s["datetime"].min(), s["datetime"].max(),
                         freq=f"{spec['minutes']}min").difference(s["datetime"])
    checks = verify_conventions(s, kind, spec["column"])

    # Coverage of the two windows the demand series actually offers, computed
    # rather than asserted, because which window to run is a scenario parameter
    # and this is the fact that decides it.
    idx = s.set_index("datetime").index
    coverage = {}
    for wname, (lo, hi) in (("AEDT_summer", AEDT_WINDOW), ("AEST_winter", AEST_WINDOW)):
        want = pd.date_range(lo, hi, freq=f"{spec['minutes']}min")
        have = idx.intersection(want)
        coverage[wname] = dict(intervals_expected=len(want), intervals_present=len(have),
                               missing=[str(g) for g in want.difference(idx)])

    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    s.to_parquet(PARQUET_DIR / spec["parquet"], index=False)

    meta = dict(parquet_file=spec["parquet"],
                source_organization="Australian Energy Market Operator (AEMO)",
                source_url="https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/",
                **LICENCE,
                interval_convention=INTERVAL_CONVENTION,
                resampling_declaration=RESAMPLING_DECLARATION,
                region=REGION, table=spec["table"],
                filters_applied=spec["filters"],
                resolution_minutes=spec["minutes"],
                value_bounds_note=(
                    "The NEM market floor is -1000 and the 2024-25 market price "
                    "cap is 17500 AUD/MWh.  These are REGULATED VALUES, not "
                    "outliers, and clipping them would remove exactly the "
                    "scarcity and negative-price intervals the market is about.  "
                    "Only the cap is actually attained: max == 17500.0 exactly, "
                    "on 21 intervals.  The floor is approached but NOT reached -- "
                    "the minimum is -999.99797, within 0.003 of it and never "
                    "below -- so an equality assertion is legitimate at the cap "
                    "and would be asserting a coincidence at the floor."
                    if kind == "price" else
                    "zero at night is real, not missing: rooftop output is "
                    "genuinely 0 MW between dusk and dawn, which is what the "
                    "night-window check asserts."),
                rows=int(len(s)), missing_intervals=int(len(gaps)),
                missing_interval_starts_utc=[str(g) for g in gaps],
                duplicate_rows_dropped=n_dropped,
                duplicate_note="identical republications, not revisions: the "
                               "DVD-era archive files carry one extra day, so "
                               "each overlaps the next month.  The overlapping "
                               "values were verified identical before one copy "
                               "was dropped; a disagreement raises instead.",
                convention_checks=checks,
                window_coverage=coverage,
                window_note="The demand series this pairs with offers two usable "
                            "windows.  **Both series are complete over the summer "
                            "(AEDT) window, so choosing it costs no gap at all.** "
                            "The winter (AEST) window is complete for price and "
                            "misses two rooftop photovoltaic half hours at local "
                            "midday on 2024-09-05.  Which window to run is a "
                            "scenario parameter; this is the fact that decides it.",
                source_defects=[
                    "the MMSDM archive renamed its files between 2024-07 and "
                    "2024-08: PUBLIC_DVD_<TABLE>_<YYYYMM>010000.zip serves months "
                    "up to 2024-07 and 404s from 2024-08, while "
                    "PUBLIC_ARCHIVE#<TABLE>#FILE01#<YYYYMM>010000.zip is the "
                    "mirror image.  The two forms are disjoint, so fetching with "
                    "one pattern truncates the window silently at the changeover "
                    "-- which falls inside the winter demand window.  Measured "
                    "2026-08-15; the form each month answered on is in `archives`."],
                first_interval_start_utc=str(s["datetime"].min()),
                last_interval_start_utc=str(s["datetime"].max()),
                timezone_source=f"NEM market time, fixed UTC+{MARKET_UTC_OFFSET_HOURS}, no DST",
                timezone_stored="UTC",
                archives=archives,
                generated_at=datetime.now(timezone.utc).isoformat())
    meta["licence_permits_redistribution"] = True
    meta["licence_ruling"] = cleared
    (PARQUET_DIR / spec["parquet"]).with_suffix(".json").write_text(
        json.dumps(meta, indent=2))

    manifest = dict(
        name=spec["manifest"], source="aemo", data_type="actual_series",
        time_mode="calendar", resolution=f"{spec['minutes']}min",
        parquet_file=spec["parquet"],
        metadata_json=str(Path(spec["parquet"]).with_suffix(".json")),
        column_map={spec["column"]: spec["signal"]},
        index_map={"datetime": "datetime"}, derived={}, normalize={},
        data_epoch=None, cyclical=False, region_values=[],
        date_range=[str(s["datetime"].min().date()), str(s["datetime"].max().date())],
        source_url=ARCHIVE.split("{")[0],
        source_organization="Australian Energy Market Operator (AEMO)")
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFEST_DIR / f"{spec['manifest']}.json").write_text(json.dumps(manifest, indent=2))
    return dict(rows=len(s), missing=len(gaps), checks=checks, dropped=n_dropped,
                gaps=[str(g) for g in gaps[:6]],
                span=(str(s["datetime"].min()), str(s["datetime"].max())),
                naming=sorted({a["naming"] for a in archives.values()}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--first", default="2024-04", help="first month, YYYY-MM")
    ap.add_argument("--last", default="2025-04", help="last month, YYYY-MM")
    ap.add_argument("--licence-cleared", default=None, metavar="RULING",
                    help="the author's ruling that the AEMO Copyright Permissions "
                         "Notice supports redistributing AEMO data with this "
                         "repository.  Recorded verbatim in each sidecar.  Without "
                         "it this script fetches nothing.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the months and URLs that would be fetched")
    args = ap.parse_args()

    months = [(d.year, d.month) for d in
              pd.date_range(f"{args.first}-01", f"{args.last}-01", freq="MS")]

    if args.dry_run:
        for kind, spec in SERIES.items():
            print(f"{kind}: {spec['table']} -> {spec['parquet']}  "
                  f"({len(months)} months, {spec['minutes']}min, "
                  f"filters {spec['filters']})")
            shown = months if len(months) <= 3 else months[:2] + months[-1:]
            for j, (y, mth) in enumerate(shown):
                if len(months) > 3 and j == 2:
                    print("     ...")
                print("    ", ARCHIVE.format(year=y, month=mth, table=spec["table"]))
        return

    if not args.licence_cleared:
        print(__doc__.split("Two series, and why each:")[0].strip())
        print("\nREFUSING TO FETCH.  The redistribution question is open and this "
              "script will not resolve it by fetching first.  Re-run with "
              "--licence-cleared '<the ruling>' once it is settled.")
        sys.exit(2)

    for kind in ("price", "pv"):          # price first: it blocks the reward
        r = build_series(kind, months, args.licence_cleared)
        print(f"{kind}: {r['rows']} rows, {r['missing']} missing intervals, "
              f"{r['span'][0]} .. {r['span'][1]}, naming {r['naming']}, "
              f"{r['dropped']} duplicate rows dropped")
        if r["gaps"]:
            print(f"    missing interval starts (UTC): {r['gaps']}")
        for k, v in r["checks"].items():
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
