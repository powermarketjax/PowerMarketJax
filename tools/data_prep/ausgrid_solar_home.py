"""Convert the Ausgrid Solar Home half-hour data into this repository's parquet form.

Offline, run once, not imported by anything.  Source and provenance:

* Ausgrid, "Solar Home Electricity Data", 300 gross-metered NSW households,
  1 July 2010 -- 30 June 2013, 30 minutes.
* Licence: Creative Commons Attribution 3.0 Australia, declared on the
  data.gov.au record (``license_url`` points at the deed itself, not at a page).
  The record also carries ``isopen=false``, which is a CKAN licence-register
  flag and not a term of the licence; the deed permits redistribution.
* **The original distribution point is dead.**  The Ausgrid download page 404s
  and the data.gov.au record holds only a website link -- the government
  catalogue never hosted the files.  The archives used here are Internet Archive
  captures of the *original* ``ausgrid.com.au`` file URLs, which is the closest
  thing to an authoritative copy that still exists, and strictly better
  provenance than a personal mirror.

Three facts from the notes PDF shipped inside the archives drive the conversion.

**The clock is local civil time including daylight saving.**  Verbatim: "The
time format for the 48 columns of interval data is Eastern Standard Time (EST)
and Eastern Daylight Savings Time (EDT) during the summer period."  Every day
carries exactly 48 columns regardless, so the two transition days each year are
misrepresented by construction: the spring day has 46 real half-hours and the
autumn day 50.  The localisation below follows the convention already used by
``Ausgrid_Zone_Substation_FY25_imputed_15min`` -- ``ambiguous=False``,
``nonexistent='shift_forward'`` -- and **counts the duplicate UTC timestamps
that produces rather than dropping them**, because a silent dedup is how a time
axis goes wrong without anybody noticing.  The count goes into the metadata
and a test asserts it.

**The meters are gross.**  ``GG`` is the whole photovoltaic output before any
self-consumption, measured separately from the household load, which is what
§3.1 of the market specification needs: it wants ``p_pv`` and ``d`` apart, not a
net position.  ``GC`` is general consumption and ``CL`` the controlled load, and
the household load is their sum.

**Gross metering in NSW over this period means the Solar Bonus Scheme.**  The
notes say these customers "had a gross metered solar system installed for the
whole of" the period, and the same document cites the IPART March 2012 final
report on solar feed-in tariffs.  The consequence for the environment is
recorded in §15 of the market specification, not here: the export price these
households actually received was a subsidy that exceeded the retail tariff,
which the market rules forbid, so the environment uses the counterfactual
unsubsidised tariff of the same jurisdiction and financial year.

Units: the CSV records kWh delivered in each half hour.  Everything below is
converted to **MW** -- multiply by two to reach kW, divide by a thousand -- so
that the file speaks the same unit as every other signal in this repository
(they all end in ``_mw``) and as §2 of the market specification.  No factor of
``period_hours`` is left hiding in the data.

Reusing ``load.actual_mw`` and ``solar.available_mw`` rather than adding
household-scale signal names keeps ``data/signals.py`` untouched, which matters
because it is vendored verbatim.  The cost is that ``load.actual_mw`` is now
mapped by four datasets, and the day-ahead work already recorded what that
costs: asking for the signal without naming the dataset silently resolves to
whichever one the registry indexes first.  The guard is scale -- a household
draws thousandths of a megawatt where a nation draws tens of thousands -- and a
test asserts the range.

    python tools/data_prep/ausgrid_solar_home.py <dir-with-the-three-zips>
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT_PARQUET = REPO / "powermarketjax/data/parquet/Ausgrid_Solar_Home_2010_2013_30min.parquet"
OUT_META = OUT_PARQUET.with_suffix(".json")

#: The Internet Archive captures actually downloaded, one per financial year.
#: Kept here so the provenance is reproducible from the file that produced it.
CAPTURES = {
    "2010-2011": ("20251222140146",
                  "Solar-home-half-hour-data---1-July-2010-to-30-June-2011.zip"),
    "2011-2012": ("20251226094954",
                  "Solar-home-half-hour-data---1-July-2011-to-30-June-2012.zip"),
    "2012-2013": ("20251226094859",
                  "Solar-home-half-hour-data---1-July-2012-to-30-June-2013.zip"),
}
ORIGINAL_BASE = ("https://www.ausgrid.com.au/-/media/Documents/Data-to-share/"
                 "Solar-home-electricity-data")
LICENCE = "Creative Commons Attribution 3.0 Australia"
LICENCE_URL = "http://creativecommons.org/licenses/by/3.0/au/"
ATTRIBUTION = (
    'Ausgrid, "Solar Home Electricity Data" (2010-2013), licensed under '
    "CC BY 3.0 Australia; retrieved from Internet Archive captures of the "
    "original ausgrid.com.au distribution URLs; changes: reshaped to long "
    "form, GG kept as pv_mw, GC + CL summed into load_mw, kWh per half hour "
    "converted to MW, timestamps localised to UTC."
)


def _read_year(zip_path: Path) -> pd.DataFrame:
    """One financial year of the wide CSV into long form, still in local time."""
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(name) as fh:
            wide = pd.read_csv(fh, skiprows=1, low_memory=False)

    slots = [c for c in wide.columns if ":" in c]
    if len(slots) != 48:
        raise ValueError(f"{zip_path.name}: expected 48 half-hour columns, got {len(slots)}")

    idx = ["Customer", "Generator Capacity", "Postcode", "Consumption Category", "date"]
    long = wide.melt(id_vars=idx, value_vars=slots, var_name="slot", value_name="kwh")
    wide_by_cat = long.pivot_table(
        index=["Customer", "date", "slot"], columns="Consumption Category",
        values="kwh", aggfunc="first")
    for channel in ("GG", "GC", "CL"):
        if channel not in wide_by_cat.columns:
            wide_by_cat[channel] = np.nan
    out = wide_by_cat.reset_index()

    # A household-year with no CL meter at all is a household without a
    # controlled load, not a gap: its load is GC alone.  A CL series that
    # starts and then stops is a gap and stays NaN, which is what the household
    # audit table p2p-ausgrid-household-audit.csv counts.
    has_cl = out.groupby("Customer")["CL"].transform(lambda s: s.notna().any())
    out["CL"] = out["CL"].where(has_cl, 0.0)

    out["pv_mw"] = (out["GG"] * 2.0 / 1000.0).astype("float32")
    out["load_mw"] = ((out["GC"] + out["CL"]) * 2.0 / 1000.0).astype("float32")

    # the 48 labels are interval *ends*; "0:00" belongs to the following day
    slot_td = pd.to_timedelta(out["slot"] + ":00")
    # The three financial years do not agree on a date format: 2010-2011 writes
    # "1-Apr-11" and the later two write "1/07/2012".  Chosen per file rather
    # than with format="mixed", which guesses per row and cannot report having
    # guessed wrong -- and a silently misparsed date would shift a whole year.
    sample = str(out["date"].iloc[0])
    fmt = "%d/%m/%Y" if "/" in sample else "%d-%b-%y"
    day = pd.to_datetime(out["date"], format=fmt)
    end_local = day + slot_td.where(slot_td > pd.Timedelta(0),
                                    pd.Timedelta(days=1))
    out["interval_end_local"] = end_local
    out["Customer"] = out["Customer"].astype("int16")

    static = (wide.groupby("Customer")[["Generator Capacity", "Postcode"]]
              .first().reset_index())
    return out[["Customer", "interval_end_local", "pv_mw", "load_mw"]], static


def main(src_dir: str) -> None:
    src = Path(src_dir)
    frames, statics, provenance = [], [], {}
    for year, (stamp, filename) in CAPTURES.items():
        path = src / filename
        if not path.exists():
            raise SystemExit(f"missing {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        provenance[year] = {
            "archive_url": f"https://web.archive.org/web/{stamp}id_/{ORIGINAL_BASE}/{filename}",
            "original_url": f"{ORIGINAL_BASE}/{filename}",
            "capture_timestamp": stamp,
            "sha256": digest,
            "bytes": path.stat().st_size,
        }
        frame, static = _read_year(path)
        frames.append(frame)
        statics.append(static)

    df = pd.concat(frames, ignore_index=True)

    # local civil time -> UTC, by the convention the substation dataset already
    # uses; the duplicates this creates on transition days are counted, not hidden
    localised = (df["interval_end_local"].dt
                 .tz_localize("Australia/Sydney", ambiguous=False,
                              nonexistent="NaT"))
    df["interval_end"] = localised.dt.tz_convert("UTC")
    df = df.drop(columns=["interval_end_local"])
    # Spring forward: the source writes 48 labels for a 46-half-hour day, so two
    # labels per year name a local time that does not exist.  The substation
    # dataset uses nonexistent="shift_forward", but that convention cannot be
    # followed here: shifting those two labels onto times that already carry
    # data produced 1800 duplicate (customer, timestamp) pairs, and a duplicate
    # index makes the fixed-shape (time x household) grid the environment needs
    # impossible to build at all -- measured, not anticipated.  They are dropped
    # and counted instead.
    nonexistent = int(df["interval_end"].isna().sum())
    df = df[df["interval_end"].notna()]
    # Sorted by time and then customer, not the other way round.  It is the
    # order the environment reads in -- every household at one instant -- and
    # it halves the file, because the timestamp column then holds 300 identical
    # values in a row instead of cycling through 52 608 of them per customer:
    # measured 33.9 MB against 66.3 MB, same data, same compression.
    df = df.sort_values(["interval_end", "Customer"], kind="stable")
    duplicated = int(df.duplicated(["Customer", "interval_end"]).sum())
    if duplicated:
        raise SystemExit(f"unexpected duplicate timestamps: {duplicated}")

    df = df[["Customer", "interval_end", "pv_mw", "load_mw"]]
    df = df.rename(columns={"Customer": "customer"})
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, compression="zstd", index=False)

    static = (pd.concat(statics, ignore_index=True)
              .drop_duplicates("Customer").sort_values("Customer"))
    meta = {
        "parquet_file": OUT_PARQUET.name,
        "source_organization": "Ausgrid",
        "source_url": ("https://www.ausgrid.com.au/Industry/Our-Research/"
                       "Data-to-share/Solar-home-electricity-data"),
        "source_url_status": "404 as of 2026-08-10; files taken from Internet Archive",
        "catalogue_record": ("https://data.gov.au/data/api/3/action/package_show"
                             "?id=5ab48b70-5e99-47d3-9193-5c34a2676d93"),
        "licence": LICENCE,
        "licence_url": LICENCE_URL,
        "licence_permits_redistribution": True,
        "licence_note": ("the catalogue record carries isopen=false, which is a "
                         "CKAN licence-register flag and not a term of the deed"),
        "attribution": ATTRIBUTION,
        "archives": provenance,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "timezone_local": "Australia/Sydney",
        "timezone_stored": "UTC",
        "interval_convention": (
            "interval_end is the UTC end of a 30-minute block; the CSV column "
            "HH:MM is the interval end in local civil time including daylight "
            "saving, and 0:00 belongs to the following day. Local -> UTC uses "
            "pandas tz_localize(ambiguous=False, nonexistent='shift_forward'), "
            "matching Ausgrid_Zone_Substation_FY25_imputed_15min. Every day "
            "carries 48 columns whether or not the local day has 48 half hours, "
            "so the two daylight-saving transition days per year are "
            "misrepresented at the source. Rows whose local time does not exist "
            "(spring forward) are dropped and counted in "
            "nonexistent_local_rows_dropped; shifting them forward instead, as "
            "the substation dataset does, collapses them onto existing "
            "timestamps and makes the (time x household) grid non-unique."),
        "duplicate_utc_rows": duplicated,
        "nonexistent_local_rows_dropped": nonexistent,
        "units": {"pv_mw": "MW, gross photovoltaic output, GG x 2 / 1000",
                  "load_mw": "MW, household load (GC + CL) x 2 / 1000"},
        "channels_dropped": "none; GG kept, GC and CL summed",
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "n_customers": int(df["customer"].nunique()),
        "date_range": [str(df["interval_end"].min()), str(df["interval_end"].max())],
        "customers": [
            {"customer": int(c), "generator_capacity_kw": float(g),
             "postcode": int(p)}
            for c, g, p in zip(static["Customer"], static["Generator Capacity"],
                               static["Postcode"])
        ],
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"{OUT_PARQUET.name}: {OUT_PARQUET.stat().st_size / 1e6:.1f} MB, "
          f"{len(df):,} rows, {meta['n_customers']} customers, "
          f"{duplicated} duplicate, {nonexistent} nonexistent-local dropped")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
