"""Convert the Fluvius digital-meter quarter-hour data into this repository's parquet form.

Offline, run once, not imported by anything.  Source and provenance:

* Fluvius, "Verbruiksprofielen digitale elektriciteitsmeters: kwartierwaarden
  voor een volledig jaar", 2 400 anonymised residential digital electricity
  meters in Flanders, calendar year 2024, 15 minutes.  The publisher splits the
  sample into eight groups of 300 by three attributes -- photovoltaic array,
  heat pump as main heating, home-charging electric vehicle -- and ships one CSV
  per group inside two ZIP attachments.
* Licence: the Fluvius open data licence, which grants the licensee a
  non-exclusive, free right of reuse "wereldwijd en voor onbeperkte duur"
  covering, verbatim, "de informatie reproduceren, kopieren, publiceren en
  doorgeven", "de informatie **verspreiden en herverdelen**", adaptation and
  extraction, and commercial exploitation.  **The attribution obligation is
  wider than CC BY's**: it requires the name of Fluvius *and the date of the
  last update* on every reuse, so that date is carried in the metadata below
  and belongs in any published attribution string.  Belgian law governs and the
  licence terminates automatically on breach.

**Only the four groups that have photovoltaic arrays are taken**, which is
1 200 households.  A household without an array is in deficit in every period
and can only ever be a buyer; it is a legitimate participant and the publisher
ships 1 200 of them too, but it is not needed to make a market and it would
double the parquet.  Whether the demand side is thin without them is an
empirical question about the surplus/deficit mix, which the audit answers; if it
is, `geen_ZP` is the group to add and nothing else here changes.

**Offtake and injection are stored, not photovoltaic output and load, and that
is a gain rather than a compromise.**  §3.1 of the market specification needs
the net position of a premises, and at a digital meter the net position *is* the
measured difference of the two registers.  The Ausgrid household set had to
reconstruct it from gross channels, which §15 records as a declared
recombination; here there is nothing to reconstruct.  The price is that gross
photovoltaic output is not recoverable, so the array cannot be rescaled and
behind-the-meter self-consumption cannot be separated from the load.

**Storage systems are excluded by the publisher, with a residual risk the
publisher states.**  Verbatim from the legend: "EANs waarvoor we weten dat er
opslagsystemen aanwezig zijn, werden gedescoped. Het is dus mogelijk dat er EANs
in deze dataset zitten waartoe opslagsystemen behoren waarvan we geen weet
hebben."  This matters because the environment adds a battery model on top of
the metered net position, and a meter that already contains battery action would
be counted twice.

Three defects in the source were measured on 2026-08-14 and are handled here
rather than passed on.

**The clock is CET/CEST local civil time and the `Z` suffix on it is wrong.**
Every `Datum_Startuur` is written `2024-01-01T00:00:00.000Z`, which claims UTC,
while the legend says CET/CEST.  The legend is right, and the two European
transition days prove it at the row level: 2024-03-31 carries 92 quarters and
the labels jump from 01:45 to 03:00, and 2024-10-27 carries 100, with each of
02:00, 02:15, 02:30 and 02:45 written twice with different values.  **The two
copies are written pairwise per label rather than as two consecutive blocks**,
so the file is not in time order inside that hour and `ambiguous="infer"` fails
outright -- measured, "There are 4 dst switches when there should only be 1".
The disambiguation is therefore explicit, and it is the only assumption in this
conversion; see the comment at the localisation itself.  No row is dropped and
no timestamp collides, which is strictly better than the Ausgrid conversion:
that source fabricated 48 labels for every day including the 46-half-hour one
and 1 800 rows had to go.

**The row index means time of day only inside one UTC offset.**  The
observation's calendar encoding is computed from the row cursor rather than from the clock,
so a window spanning a transition rotates local time of day by an hour against
the phase for the rest of it, and no market quantity shows it.  The grid stored
here is regular, unique and gap-free across the whole year, which is what makes
that a window question for the loader instead of a data defect; the two
transition days are the only local days that are not 96 periods long.

**`EAN_ID` is the publisher's identifier and it is kept as `household`
unchanged.**  It is globally unique over the eight groups exactly as the legend
says, running 1 to 2 400 in blocks of 300 per group -- measured on the four files
taken: 1 to 300, 601 to 900, 1 201 to 1 500 and 1 801 to 2 100, with the gaps
being the four groups without an array.  An earlier version of this script
renumbered the households on the belief that the identifier restarted per file,
which came from reading one file's range and generalising it; the assertion
below is what a claim of that kind should have been checked against in the first
place.  Keeping the publisher's numbering means a household in this parquet can
be matched against the published CSV without a translation table.

**The legend's description of `Warmtepomp_Indicator` is a copy of the electric
vehicle one** -- it reads "Laadpaal gemeld door de klant of EV gedetecteerd door
het algoritme", which is about charge points.  The indicator itself is usable
and is checked against the group each file claims to be; its documentation is
not quotable.  Separately the CSV header writes `Volume_Afname_KWh` with a
capital K where the legend writes `kWh`.

Units: the CSV records kWh delivered in each quarter hour.  Everything below is
converted to **MW** -- multiply by four to reach kW, divide by a thousand -- so
the file speaks the same unit as every other signal in this repository and as §2
of the market specification.  No factor of `period_hours` is left hiding in the
data.

Signals: `load.offtake_mw` and `load.injection_mw` are new names, which costs
nothing because nothing validates signal strings against `data/signals.py` --
`DatasetManifest.signals` is just `column_map.values()` -- so the vendored file
stays untouched *and* this dataset stays out of the six-way collision on
`load.actual_mw` that the day-ahead and local-flexibility loaders both had to
work around.  Reading by dataset name is still the house rule.

    python tools/data_prep/fluvius_dm_quarterly.py <dir-with-the-two-zips>
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
OUT_PARQUET = REPO / "powermarketjax/data/parquet/Fluvius_DM_Residential_2024_15min.parquet"
OUT_META = OUT_PARQUET.with_suffix(".json")

DATASET_SLUG = "1_50-verbruiksprofielen-dm-elek-kwartierwaarden-voor-een-volledig-jaar"
PORTAL = f"https://opendata.fluvius.be/explore/dataset/{DATASET_SLUG}/"
ATTACHMENT_BASE = (f"https://opendata.fluvius.be/api/explore/v2.1/catalog/datasets/"
                   f"{DATASET_SLUG}/attachments")
LICENCE = "Open data licentie Fluvius"
LICENCE_URL = "https://opendata.fluvius.be/p/licentieopendatafluvius/"
#: The licence obliges the licensee to state the date of the last update, so it
#: is part of the provenance and not a nicety.  Value is the dataset's
#: ``modified`` field as served by the portal's catalogue API on 2026-08-14.
LAST_UPDATED = "2025-09-29"
ATTRIBUTION = (
    'Fluvius, "Verbruiksprofielen digitale elektriciteitsmeters: kwartierwaarden '
    f'voor een volledig jaar" (2024), last updated {LAST_UPDATED}, reused under '
    "the Fluvius open data licence; changes: the four groups with a photovoltaic "
    "array retained, kWh per quarter hour "
    "converted to MW, timestamps localised from CET/CEST to UTC."
)

#: The two ZIP attachments, by the archive name the portal serves them under.
ARCHIVES = {
    "Deel_1.zip": "p6269_1_50_dmk_sample_elek_2024_deel_1_zip",
    "Deel_2.zip": "p6269_1_50_dmk_sample_elek_2024_deel_2_zip",
}

#: The four groups taken, in the order the ``household`` identifier runs, each
#: with the archive holding it and the indicator triple it must carry.  The
#: triple is asserted rather than trusted: it is the only check that the file
#: named after a group actually contains that group.
GROUPS = (
    ("enkel_ZP", "Deel_1.zip", "P6269_Open_Data_enkel_ZP.csv", (0, 0, 1)),
    ("WP_met_ZP", "Deel_1.zip", "P6269_Open_Data_WP_met_ZP.csv", (1, 0, 1)),
    ("EV_met_ZP", "Deel_2.zip", "P6269_Open_Data_EV_met_ZP.csv", (0, 1, 1)),
    ("WP_EV_met_ZP", "Deel_2.zip", "P6269_Open_Data_WP_EV_met_ZP.csv", (1, 1, 1)),
)
#: Groups the publisher ships that are deliberately not taken; recorded so the
#: omission is a decision on the record rather than an oversight.
GROUPS_OMITTED = ("geen_ZP", "WP_geen_ZP", "EV_geen_ZP", "WP_EV_geen_ZP")

INDICATORS = ("Warmtepomp_Indicator", "Elektrisch_Voertuig_Indicator",
              "PV_Installatie_Indicator")
USECOLS = ("EAN_ID", "Datum_Startuur", "Volume_Afname_KWh", "Volume_Injectie_KWh",
           *INDICATORS, "Contract_Categorie")
LOCAL_TZ = "Europe/Brussels"
PERIODS_PER_DAY = 96


def _read_group(zip_path: Path, member: str, expected: tuple[int, int, int]
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One group's CSV into (long frame in UTC, per-household static frame)."""
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as fh:
            raw = pd.read_csv(
                fh, usecols=list(USECOLS),
                dtype={"EAN_ID": "int16", "Contract_Categorie": "category",
                       **{c: "int8" for c in INDICATORS}},
            )

    # The publisher's own scope statement, asserted rather than assumed.
    bad = set(raw["Contract_Categorie"].unique()) - {"Residentieel"}
    if bad:
        raise ValueError(f"{member}: non-residential contract categories {bad}")
    per_household = raw.groupby("EAN_ID", observed=True)[list(INDICATORS)].nunique()
    if (per_household != 1).any().any():
        raise ValueError(f"{member}: an indicator changes within a household")
    got = raw.groupby("EAN_ID", observed=True)[list(INDICATORS)].first()
    if not (got.to_numpy() == np.asarray(expected, dtype=np.int8)).all():
        raise ValueError(f"{member}: indicators do not match the group triple {expected}")

    # The Z suffix is a lie: parse the label as a naive local wall clock, then
    # localise.  Explicit format, not inference: a guessed format cannot report
    # having guessed wrong, and a misparsed stamp would shift a whole household.
    naive = pd.to_datetime(raw["Datum_Startuur"].str.slice(0, 19),
                           format="%Y-%m-%dT%H:%M:%S")
    # ``ambiguous="infer"`` cannot be used, and the reason is a property of the
    # source rather than of pandas: **the repeated autumn hour is written
    # pairwise per label**, 02:00, 02:00, 02:15, 02:15, ... rather than as two
    # consecutive blocks, so the file is not in time order inside that hour and
    # the inference sees four transitions where it demands one (measured:
    # "There are 4 dst switches when there should only be 1").  The rule applied
    # instead is that the first of each pair is the earlier instant, hence
    # summer time.  It is an assumption, it is the only ordering that makes the
    # file sorted by instant within a label, and **it is immaterial downstream**:
    # it decides which of two adjacent quarters eight rows per household land
    # in, on a local day that no single-offset window contains.
    dup_rank = naive.groupby(
        [raw["EAN_ID"], naive], observed=True).cumcount().to_numpy()
    frames = []
    for ean, block in naive.groupby(raw["EAN_ID"], observed=True):
        if not block.is_monotonic_increasing:
            raise ValueError(f"{member}: household {ean} is not in time order")
        utc = (pd.DatetimeIndex(block)
               .tz_localize(LOCAL_TZ,
                            ambiguous=(dup_rank[block.index.to_numpy()] == 0),
                            nonexistent="raise")
               .tz_convert("UTC"))
        frames.append(pd.Series(utc, index=block.index))
    interval_start = pd.concat(frames).sort_index()

    out = pd.DataFrame({
        "ean_id": raw["EAN_ID"].to_numpy(),
        "interval_start": interval_start.to_numpy(),
        "offtake_mw": (raw["Volume_Afname_KWh"].to_numpy(np.float64)
                       * 4.0 / 1000.0).astype("float32"),
        "injection_mw": (raw["Volume_Injectie_KWh"].to_numpy(np.float64)
                         * 4.0 / 1000.0).astype("float32"),
    })
    return out, got.reset_index()


def _local_day_lengths(utc: pd.Series) -> pd.Series:
    """Periods per *local* day, for one household, to locate the transitions."""
    local = pd.DatetimeIndex(utc).tz_convert(LOCAL_TZ)
    return pd.Series(1, index=local.date).groupby(level=0).sum()


def main(src_dir: str) -> None:
    src = Path(src_dir)
    provenance = {}
    for name, asset in ARCHIVES.items():
        path = src / name
        if not path.exists():
            raise SystemExit(f"missing {path}")
        provenance[name] = {
            "attachment_url": f"{ATTACHMENT_BASE}/{asset}",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }

    frames, statics, seen = [], [], set()
    for group, archive, member, expected in GROUPS:
        frame, static = _read_group(src / archive, member, expected)
        ids = set(static["EAN_ID"].astype(int))
        # The publisher's identifier is kept, so it has to be globally unique;
        # the claim that it is not was what the earlier renumbering rested on.
        clash = sorted(ids & seen)
        if clash:
            raise SystemExit(
                f"{member}: EAN_ID is not unique across groups, {len(clash)} "
                f"collide, for example {clash[:5]}")
        seen |= ids
        frame = frame.rename(columns={"ean_id": "household"})
        frame["household"] = frame["household"].astype("int16")
        statics.append(static.assign(group=group))
        frames.append(frame)
        print(f"  {group}: {len(frame):,} rows, {len(ids)} households, "
              f"EAN_ID {min(ids)}..{max(ids)}")

    df = pd.concat(frames, ignore_index=True)
    del frames

    # Sorted by time and then household, not the other way round: it is the
    # order the environment reads in -- every household at one instant -- and
    # the Ausgrid conversion measured the file-size consequence of the choice
    # (33.9 MB against 66.3 MB, same data, same compression).
    df = df.sort_values(["interval_start", "household"], kind="stable")
    duplicated = int(df.duplicated(["household", "interval_start"]).sum())
    if duplicated:
        raise SystemExit(f"duplicate (household, timestamp) pairs: {duplicated}")

    # The grid must be regular, unique and gap-free, or the row-cursor calendar
    # encoding of the observation silently rotates.  Checked here rather than left
    # to the loader, because a defect in the file cannot be fixed downstream.
    stamps = pd.DatetimeIndex(df["interval_start"].unique()).sort_values()
    gaps = stamps.to_series().diff().dropna().unique()
    if len(gaps) != 1 or gaps[0] != pd.Timedelta(minutes=15):
        raise SystemExit(f"irregular UTC grid, step values seen: {gaps}")
    counts = df.groupby("interval_start", observed=True).size()
    incomplete = int((counts != len(seen)).sum())

    df = df[["household", "interval_start", "offtake_mw", "injection_mw"]]
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, compression="zstd", index=False)

    static = (pd.concat(statics, ignore_index=True)
              .sort_values("EAN_ID"))
    day_lengths = _local_day_lengths(
        df.loc[df["household"] == 1, "interval_start"])
    short = day_lengths[day_lengths != PERIODS_PER_DAY]

    meta = {
        "parquet_file": OUT_PARQUET.name,
        "source_organization": "Fluvius",
        "source_url": PORTAL,
        "licence": LICENCE,
        "licence_url": LICENCE_URL,
        "licence_permits_redistribution": True,
        "licence_note": (
            "the grant lists 'de informatie verspreiden en herverdelen' and "
            "commercial exploitation explicitly, worldwide and for unlimited "
            "duration; the attribution obligation is wider than CC BY's in that "
            "it requires the date of the last update as well as the name"),
        "last_updated": LAST_UPDATED,
        "attribution": ATTRIBUTION,
        "archives": provenance,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "timezone_local": LOCAL_TZ,
        "timezone_stored": "UTC",
        "groups_taken": [g for g, _, _, _ in GROUPS],
        "groups_omitted": list(GROUPS_OMITTED),
        "interval_convention": (
            "interval_start is the UTC start of a 15-minute block. The source "
            "column Datum_Startuur is the block start in CET/CEST local civil "
            "time despite carrying a Z suffix, which was established from the "
            "two transition days: 2024-03-31 holds 92 quarters with the labels "
            "jumping 01:45 -> 03:00, and 2024-10-27 holds 100 with 02:00 to "
            "02:45 each written twice with different values. Local -> UTC uses "
            "tz_localize(ambiguous='infer', nonexistent='raise'), which is "
            "lossless here: no row is dropped and no timestamp collides. The "
            "stored UTC grid is regular at 15 minutes with no gaps and no "
            "duplicates, so a window spanning a transition is well defined in "
            "UTC but rotates local time of day by an hour against the "
            "row-cursor calendar encoding of ADR-0010 (7); confining a window "
            "to one UTC offset is the loader's job."),
        "duplicate_rows": duplicated,
        "timestamps_with_missing_households": incomplete,
        "local_days_not_96_periods": {str(d): int(n) for d, n in short.items()},
        "source_defects": [
            "Datum_Startuur carries a Z suffix but is CET/CEST local civil time",
            "the legend's Warmtepomp_Indicator description is a copy of the "
            "Elektrisch_Voertuig_Indicator one and describes charge points",
            "the CSV header writes Volume_Afname_KWh where the legend writes kWh",
        ],
        "publisher_storage_note": (
            "EANs known to hold storage systems were de-scoped by the publisher, "
            "which also states that EANs with storage unknown to it may remain"),
        "units": {
            "offtake_mw": "MW, metered offtake, Volume_Afname_KWh x 4 / 1000",
            "injection_mw": "MW, metered injection, Volume_Injectie_KWh x 4 / 1000",
        },
        "net_position_note": (
            "the net position of specification 3.1 is injection_mw - offtake_mw, "
            "a measured difference rather than a recombination; gross "
            "photovoltaic output is not recoverable from these two registers"),
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "n_households": int(df["household"].nunique()),
        "n_periods": int(len(stamps)),
        "date_range": [str(stamps.min()), str(stamps.max())],
        "households": [
            {"household": int(r.EAN_ID), "group": r.group,
             "heat_pump": int(r.Warmtepomp_Indicator),
             "electric_vehicle": int(r.Elektrisch_Voertuig_Indicator),
             "pv": int(r.PV_Installatie_Indicator)}
            for r in static.itertuples()
        ],
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"{OUT_PARQUET.name}: {OUT_PARQUET.stat().st_size / 1e6:.1f} MB, "
          f"{len(df):,} rows, {meta['n_households']} households, "
          f"{meta['n_periods']:,} periods, {duplicated} duplicate, "
          f"{incomplete} timestamps short of a full panel")
    print(f"  local days not {PERIODS_PER_DAY} periods: "
          f"{meta['local_days_not_96_periods']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
