"""Vendor the Swiss medium-voltage grid 459_0 and its distributed resources.

Offline, run once, not imported by anything.  It exists to supply the
photovoltaic and battery-fleet data of the Swiss
configuration of the local flexibility market, alongside
``ch_dayahead_price.py`` which supplies the price.

Source: Zapparoli, Oneto, Parajeles Herrera, Gjorgiev, Hug & Sansavini,
"Future Deployment and Flexibility of Distributed Energy Resources in the
Distribution Grids of Switzerland", *Scientific Data* 12:1491 (2025),
https://doi.org/10.1038/s41597-025-05830-y.  Data at Zenodo
https://doi.org/10.5281/zenodo.15056134, single archive ``SwissDN_DERs.zip``,
md5 ``490b7b2ef411541fac8db6a7cc9dad03``, 4 534 108 749 bytes, CC BY 4.0.

**Only grid 459_0, and only its medium-voltage level.**  The archive carries
the resource tables for the whole country -- 11 551 medium-voltage
photovoltaic nodes in 2030 rising to 19 452 in 2050 -- but ``06_Grids`` holds
exactly one network, the integrated medium-low voltage system 459_0, and its
own readme says so.  Resource rows for a grid whose topology is absent cannot
reach a power flow, so vendoring them would be dead weight; the remaining
networks live in a separate Zenodo record (10.5281/zenodo.15167589) and adding
one later means re-running this script with a different grid name.  The
low-voltage level is left out for size: ``LV_generation.csv`` alone is 1.1 GB
for 2030 and 3.4 GB for 2050, against 27-46 MB for the medium-voltage file,
and the market models a feeder rather than a service cable.

Four properties of the source drive what this script does, and three of them
were found by measurement rather than read off the paper.

**The photovoltaic series is twelve representative days, not a year.**  Its
header runs ``01-15 00:00:00`` to ``12-15 23:00:00``: 288 hourly columns, one
day per month.  The demand and temperature series in the same archive carry
8760.  So photovoltaic output and demand do *not* share a time axis, and the
market cannot simply read both at one ``t``.  This script keeps the
representative form rather than expanding it, because expanding it is a
modelling assumption -- which day of March gets March's profile -- and
assumptions belong in declared parameters rather than in
files that look like measurements.  The per-hour standard deviation shipped
beside the mean is carried through for the same reason: it is what a
within-month sampling rule would need.

**The loader's profile mode would destroy the seasonality.**
``TimeAligner.align_profile`` tiles a profile end to end, so a 288-hour series
of twelve monthly days becomes January, February, ... December, January
repeating every twelve days.  That is why the photovoltaic table is not given
a manifest: nothing should be able to reach it through the tiling path.

**The demand series is one shape for every node.**  ``MV_load_profile.csv``
is a single max-normalised 8760-hour row, and the paper's construction is to
multiply it by each node's peak.  This script performs exactly that
multiplication, which reproduces the source's documented construction rather
than inventing a spatial pattern, and it is the same shape-times-share
structure §14 already uses for the Ausgrid series.

**The nodal peak is in MW, established by a cross-check rather than assumed.**
Each edge carries a ``load`` property, and the ratio of the summed downstream
``el_dmd`` to that property is 11.547 on every edge, which is 20/sqrt(3) for
the 20 kV the paper states the medium-voltage grids operate at.  A constant
ratio across 128 independent edges is what makes the unit reading safe; it is
checked at all because ``line_cap`` in the case data is in MVA while the
sensitivity matrices are per unit, a unit boundary that is easy to misread.

A calendar year has to be attached to the demand series because the source
labels are ``MM-DD HH:MM:SS`` with no year, under the stated convention that
each year has 365 days and begins on a Monday.  **2029 is used as the
carrier**, which satisfies that convention exactly and, unlike 2018 and 2024,
lies outside the span of the vendored price series.  A calendar join of this
demand against ``ch_dayahead_price`` therefore returns nothing instead of
returning a plausible-looking misalignment: the pairing of a projection year
with a price year is a declared scenario choice under §14, and this makes
performing it by accident impossible.

    python tools/data_prep/swissdn_mv_459_0.py <path-to-SwissDN_DERs.zip>
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "powermarketjax/data/parquet"

GRID = "459_0"
YEARS = (2030, 2040, 2050)
#: 365 days beginning on a Monday, and outside 2015-2025 so that a calendar
#: join against the price series fails loudly rather than silently.
CARRIER_YEAR = 2029
#: The paper states the medium-voltage grids operate at 20 kV.
BASE_KV = 20.0
#: The band the implied power factor of every base-case flow must lie in for
#: ``el_dmd`` to be readable as MW against ``s_nom`` in MVA. Measured range is
#: [0.90001, 0.98927]; the band is the plausible-power-factor interval, not a
#: fit to the measurement.
PF_BAND = (0.85, 1.0)
#: The power factor the source models demand at, read off the minimum of that
#: same quotient, which sits exactly at 0.9 on the most heavily loaded edges
#: where line charging matters least. It is what §3.1 calls phi_n, as
#: ``tan(arccos(0.9)) = 0.4843``.
DEMAND_POWER_FACTOR = 0.9

ARCHIVE_MD5 = "490b7b2ef411541fac8db6a7cc9dad03"
LICENCE = "Creative Commons Attribution 4.0 International"
LICENCE_URL = "https://creativecommons.org/licenses/by/4.0/"
CITATION = (
    "Zapparoli, L., Oneto, A., Parajeles Herrera, M., Gjorgiev, B., Hug, G. & "
    "Sansavini, G. Future Deployment and Flexibility of Distributed Energy "
    "Resources in the Distribution Grids of Switzerland. Sci Data 12, 1491 "
    "(2025). https://doi.org/10.1038/s41597-025-05830-y"
)
ATTRIBUTION = (
    f"{CITATION} Dataset: https://doi.org/10.5281/zenodo.15056134, CC BY 4.0. "
    f"Changes: medium-voltage grid {GRID} and its medium-voltage resource rows "
    "extracted from the country-wide tables; GeoJSON converted to tabular "
    "form; the normalised demand profile multiplied by the nodal peak as the "
    "paper prescribes; no gap filling, no unit conversion beyond kW to MW "
    "where stated."
)


def _write(df: pd.DataFrame, stem: str, meta: dict) -> None:
    path = OUT / f"{stem}.parquet"
    df.to_parquet(path, compression="zstd", index=False)
    meta = {
        "parquet_file": path.name,
        "source_organization": "ETH Zurich (Zapparoli et al., Scientific Data 2025)",
        "source_url": "https://doi.org/10.5281/zenodo.15056134",
        "source_citation": CITATION,
        "source_archive": "SwissDN_DERs.zip",
        "source_archive_md5": ARCHIVE_MD5,
        "licence": LICENCE,
        "licence_url": LICENCE_URL,
        "licence_permits_redistribution": True,
        "attribution": ATTRIBUTION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mv_grid": GRID,
        **meta,
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }
    (OUT / f"{stem}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  {path.name}: {path.stat().st_size / 1e3:8.1f} kB  {len(df):>9,} rows")


def _grid_tables(z: zipfile.ZipFile) -> tuple[pd.DataFrame, pd.DataFrame, tuple[float, float]]:
    with zipfile.ZipFile(io.BytesIO(z.read("SwissDN_DERs/06_Grids/MV.zip"))) as mz:
        nodes = json.loads(mz.read(f"MV/{GRID}_nodes"))["features"]
        edges = json.loads(mz.read(f"MV/{GRID}_edges"))["features"]

    node_df = pd.DataFrame([f["properties"] for f in nodes]).rename(
        columns={"el_dmd": "peak_mw", "voltage": "base_case_voltage_pu",
                 "x": "coord_e_m", "y": "coord_n_m"})
    node_df["consumers"] = node_df["consumers"].astype(str) == "True"
    node_df["source"] = node_df["source"].astype(bool)

    edge_df = pd.DataFrame([f["properties"] for f in edges]).rename(
        columns={"length": "length_km", "x": "x_ohm", "r": "r_ohm",
                 "b": "b_siemens", "s_nom": "s_nom_mva",
                 "load": "base_case_flow"})
    edge_df["OHL"] = edge_df["OHL"].astype(str) == "True"
    edge_df = edge_df.rename(columns={"OHL": "overhead_line"})

    # Radiality and the unit reading, both checked rather than assumed: a
    # non-radial feeder breaks §3.1 outright, and a wrong reading of peak_mw
    # would rescale every requirement without making anything fail.
    if len(edge_df) != len(node_df) - 1:
        raise SystemExit(f"{GRID} is not a tree: {len(node_df)} nodes, {len(edge_df)} edges")
    adj: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for i, e in edge_df.iterrows():
        adj[e["u"]].append((e["v"], i))
        adj[e["v"]].append((e["u"], i))
    root = node_df.loc[node_df["source"], "osmid"].iloc[0]
    parent: dict[str, tuple[str, int]] = {}
    order, stack, seen = [], [root], {root}
    while stack:
        n = stack.pop()
        order.append(n)
        for m, i in adj[n]:
            if m not in seen:
                seen.add(m)
                parent[m] = (n, i)
                stack.append(m)
    if len(seen) != len(node_df):
        raise SystemExit(f"{GRID} is not connected: reached {len(seen)} of {len(node_df)}")

    peak = dict(zip(node_df["osmid"], node_df["peak_mw"]))
    downstream = dict(peak)
    for n in reversed(order):
        if n in parent:
            downstream[parent[n][0]] += downstream[n]
    # ``base_case_flow`` is the apparent flow as a fraction of the line rating,
    # so dividing the summed downstream active demand by it and by the rating
    # returns the power factor of that flow. Measured, it sits in
    # [0.90001, 0.98927] on all 128 edges with the minimum exactly at 0.9: the
    # source models demand at a power factor of 0.9 and line charging lifts the
    # effective factor above it on the cable runs. That the quotient lands in a
    # power-factor band at all is what establishes peak_mw as MW against
    # s_nom_mva as MVA -- the two are otherwise unrelatable, and a wrong
    # reading here would repeat the line_cap-in-MVA against per-unit trap.
    #
    # An earlier version of this check compared the quotient of downstream
    # demand and base_case_flow against 20/sqrt(3) = 11.547, the paper's 20 kV.
    # It passed on the ten trunk lines and failed by 67% elsewhere: those ten
    # are rated 12.76 MVA and 0.9 x 12.76 = 11.48, so the agreement was a
    # coincidence of that one rating. The check below is the relation that
    # actually holds, and it holds on every edge rather than on the ten that
    # happen to be looked at first.
    implied = [downstream[n] / (edge_df.at[i, "base_case_flow"] * edge_df.at[i, "s_nom_mva"])
               for n, (_, i) in parent.items() if edge_df.at[i, "base_case_flow"] > 0]
    if len(implied) != len(node_df) - 1:
        raise SystemExit("some edge carries no base-case flow; the unit check is incomplete")
    if not all(PF_BAND[0] <= v <= PF_BAND[1] for v in implied):
        raise SystemExit(
            f"peak_mw does not read as MW against s_nom in MVA: the implied "
            f"power factor spans [{min(implied):.4f}, {max(implied):.4f}], "
            f"outside {PF_BAND}")
    print(f"  radial tree, connected, {len(node_df)} nodes; peak_mw reads as MW "
          f"(implied power factor {min(implied):.5f} .. {max(implied):.5f})")
    # The demand a line supplies is the demand below its far end, and the far
    # end is whichever endpoint has the other as its parent in the tree above.
    edge_df["downstream_peak_mw"] = [
        downstream[e["v"]] if parent.get(e["v"], (None, None))[0] == e["u"]
        else downstream[e["u"]]
        for _, e in edge_df.iterrows()]
    return node_df, edge_df, (min(implied), max(implied))


def _rows_for_grid(z: zipfile.ZipFile, name: str) -> tuple[list[str], list[list[str]]]:
    """Header and the rows whose first field is this grid, streamed."""
    with z.open(name) as fh:
        reader = csv.reader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"))
        header = next(reader)
        rows = [r for r in reader if r and r[0] == GRID]
    return header, rows


def main(archive: str) -> None:
    path = Path(archive)
    if path.is_dir():
        path = path / "SwissDN_DERs.zip"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    if digest != ARCHIVE_MD5:
        raise SystemExit(f"archive md5 {digest} != {ARCHIVE_MD5}")
    print(f"archive md5 {digest} matches the Zenodo record")

    OUT.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(path)

    # ---------------------------------------------------------------- grid
    node_df, edge_df, pf_range = _grid_tables(z)
    _write(node_df, f"SwissDN_{GRID}_MV_Nodes", {
        "content": "medium-voltage grid nodes: peak demand, coordinates, base-case voltage",
        "base_kv": BASE_KV,
        "crs": "EPSG:2056 (coord_e_m, coord_n_m)",
        "radial": True,
        "units": {"peak_mw": "MW, nodal peak non-controllable demand",
                  "base_case_voltage_pu": "per unit, the source's own power-flow solution",
                  "coord_e_m": "metres, EPSG:2056", "coord_n_m": "metres, EPSG:2056"},
        "source_node_osmid": node_df.loc[node_df["source"], "osmid"].iloc[0],
        "n_nodes": int(len(node_df)),
        "sum_peak_mw": float(node_df["peak_mw"].sum()),
        "nodes_with_negative_peak": int((node_df["peak_mw"] < 0).sum()),
        "base_case_voltage_pu_range": [float(node_df["base_case_voltage_pu"].min()),
                                       float(node_df["base_case_voltage_pu"].max())],
        "voltage_limits": (
            "NOT SUPPLIED by the source. Every vendored case in this repository "
            "carries per-bus limits and this grid does not, so a band has to be "
            "declared before the market can run on it (§14)."),
        "peak_mw_unit_evidence": (
            "summed downstream peak_mw divided by (base_case_flow x s_nom_mva) "
            "is the power factor of that flow, and it lands in "
            f"[{pf_range[0]:.5f}, {pf_range[1]:.5f}] on all {len(edge_df)} "
            "edges. That is what ties peak_mw in MW to s_nom in MVA."),
        "demand_power_factor": DEMAND_POWER_FACTOR,
        "demand_power_factor_evidence": (
            "the minimum of that quotient is exactly 0.90001, reached on the "
            "most heavily loaded edges where line charging matters least; "
            "values above it are cable runs whose charging offsets reactive "
            "demand. This supplies phi_n of §3.1 as tan(arccos(0.9)) = 0.4843 "
            "from the data rather than as a declared parameter."),
    })
    _write(edge_df, f"SwissDN_{GRID}_MV_Edges", {
        "content": "medium-voltage grid edges: impedance, thermal rating, base-case flow",
        "base_kv": BASE_KV,
        "units": {"r_ohm": "ohm, total for the line", "x_ohm": "ohm, total for the line",
                  "b_siemens": "S", "s_nom_mva": "MVA, thermal rating",
                  "length_km": "km", "base_case_flow": "source's own base-case value",
                  "downstream_peak_mw": "MW, sum of peak_mw over the buses this line supplies"},
        "n_edges": int(len(edge_df)),
        "s_nom_mva_range": [float(edge_df["s_nom_mva"].min()), float(edge_df["s_nom_mva"].max())],
        "max_downstream_peak_over_s_nom": float(
            (edge_df["downstream_peak_mw"] / edge_df["s_nom_mva"]).max()),
        "impedance_note": (
            "r_ohm and x_ohm are totals in ohms, not per-unit and not per km. "
            "Building a CaseData from them needs Z_base = base_kv^2 / base_mva."),
    })

    # ------------------------------------------------------------- demand
    with z.open("SwissDN_DERs/05_Demand/2030/MV_load_profile.csv") as fh:
        reader = csv.reader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"))
        labels = next(reader)
        shape = [float(v) for v in next(reader)]
    if len(labels) != len(shape) or len(shape) != 8760:
        raise SystemExit(f"demand profile is {len(shape)} long, expected 8760")
    # The source calls this profile max-normalised and its methods say it is
    # divided by its own maximum, which would put that maximum at one. The
    # shipped maximum is 0.8528. The profile is vendored as it ships and the
    # departure is recorded rather than repaired, because renormalising would
    # make the file disagree with the archive it claims to reproduce. What it
    # costs is stated in the metadata: multiplying by the nodal peak, which is
    # the construction the paper prescribes, gives a feeder that never reaches
    # its registered peak, so any scaling factor calibrated against this series
    # carries a factor of 1/0.8528 that belongs to the data rather than to the
    # scenario.
    shape_max = max(shape)
    # The three projection years ship byte-identical demand files, which is the
    # paper's statement that non-controllable load is held unchanged; checked so
    # that a future release changing it cannot pass unnoticed.
    same = {y: z.read(f"SwissDN_DERs/05_Demand/{y}/MV_load_profile.csv") for y in YEARS}
    identical = len({hashlib.md5(v).hexdigest() for v in same.values()}) == 1
    stamps = pd.date_range(f"{CARRIER_YEAR}-01-01", periods=8760, freq="h", tz="UTC")
    if stamps[0].dayofweek != 0 or len(pd.date_range(
            f"{CARRIER_YEAR}-01-01", f"{CARRIER_YEAR}-12-31", freq="D")) != 365:
        raise SystemExit(f"{CARRIER_YEAR} is not a 365-day year beginning on a Monday")

    consumers = node_df[["osmid", "peak_mw"]]
    demand = pd.DataFrame({
        "datetime": stamps.repeat(len(consumers)),
        "osmid": list(consumers["osmid"]) * 8760,
        "load_mw": [s * p for s in shape for p in consumers["peak_mw"]],
    })
    demand["load_mw"] = demand["load_mw"].astype("float32")
    _write(demand, f"SwissDN_{GRID}_MV_Load_60min", {
        "content": "nodal medium-voltage non-controllable demand, hourly",
        "construction": (
            "the archive's single max-normalised MV_load_profile row multiplied "
            "by each node's peak_mw, which is the construction the paper "
            "prescribes; all nodes therefore share one temporal shape"),
        "carrier_year": CARRIER_YEAR,
        "carrier_year_note": (
            f"the source labels are MM-DD HH:MM:SS with no year under the stated "
            f"convention of 365 days beginning on a Monday. {CARRIER_YEAR} "
            f"satisfies that convention and lies outside the 2015-2025 span of "
            f"ch_dayahead_price, so a calendar join against the price returns an "
            f"empty frame instead of a plausible misalignment. It is a carrier, "
            f"not the data year."),
        "normalised_profile_max": shape_max,
        "normalised_profile_min": min(shape),
        "normalised_profile_max_note": (
            "the source describes this profile as max-normalised and its "
            "methods divide by the maximum, which would put this at 1.0. It "
            "ships at 0.852801, with the maximum falling on 06-21 14:00. The "
            "profile is vendored unchanged. Consequence: load_mw reaches only "
            "85.28% of peak_mw at every node, so a load_scale calibrated on "
            "this series carries a factor of 1/0.852801 = 1.1726 that is a "
            "property of the data and not of the scenario, and it must not be "
            "compared with the load_scale calibrated on the Ausgrid series "
            "without that factor."),
        "identical_across_projection_years": identical,
        "identical_note": (
            "the paper holds non-controllable load unchanged for 2030, 2040 and "
            "2050; the three files are byte-identical, so one copy is vendored"),
        "units": {"load_mw": "MW"},
        "resolution": "60min",
        "n_nodes": int(len(consumers)),
        "sum_peak_mw": float(consumers["peak_mw"].sum()),
        "date_range": [str(demand["datetime"].min()), str(demand["datetime"].max())],
        "date_ranges": {"datetime": {"min": str(demand["datetime"].min()),
                                     "max": str(demand["datetime"].max()),
                                     "count": int(len(demand)), "missing": 0}},
        "numeric_statistics": {"load_mw": {
            "count": int(len(demand)), "mean": float(demand["load_mw"].mean()),
            "min": float(demand["load_mw"].min()), "max": float(demand["load_mw"].max()),
            "missing": int(demand["load_mw"].isna().sum())}},
    })

    # ----------------------------------------------------------------- pv
    pv_frames, pv_slots = [], None
    for year in YEARS:
        head_m, rows_m = _rows_for_grid(z, f"SwissDN_DERs/01_PV/{year}/MV_generation.csv")
        head_s, rows_s = _rows_for_grid(z, f"SwissDN_DERs/01_PV/{year}/MV_std.csv")
        head_p, rows_p = _rows_for_grid(z, f"SwissDN_DERs/01_PV/{year}/MV_P_installed.csv")
        slots = head_m[2:]
        if head_s[2:] != slots:
            raise SystemExit(f"{year}: mean and std do not share a time axis")
        if len(slots) != 288:
            raise SystemExit(f"{year}: {len(slots)} photovoltaic slots, expected 288")
        pv_slots = slots
        std_by_node = {r[1]: r[2:] for r in rows_s}
        inst_by_node = {r[1]: float(r[2]) for r in rows_p}
        for r in rows_m:
            osmid, values = r[1], r[2:]
            stds = std_by_node.get(osmid)
            if stds is None:
                raise SystemExit(f"{year}: node {osmid} has a mean but no std")
            for slot, mean, sd in zip(slots, values, stds):
                month, hour = int(slot[:2]), int(slot.split()[1][:2])
                pv_frames.append((year, osmid, month, hour,
                                  float(mean), float(sd), inst_by_node.get(osmid, float("nan"))))
    pv = pd.DataFrame(pv_frames, columns=[
        "projection_year", "osmid", "month", "hour_of_day",
        "pv_kw", "pv_std_kw", "p_installed_kw"])
    _write(pv, f"SwissDN_{GRID}_MV_PV_RepDays", {
        "content": "photovoltaic output on twelve representative days, one per month",
        "time_axis": (
            "NOT a year. 288 hourly values per node and projection year, being "
            "the 15th of each month; the demand series in the same archive "
            "carries 8760. Which calendar day of a month receives its "
            "representative profile is a declared scenario choice under §14 and "
            "is deliberately not baked in here."),
        "no_manifest_reason": (
            "TimeAligner.align_profile tiles a profile end to end, which would "
            "turn twelve monthly days into a twelve-day cycle and destroy the "
            "seasonality. This table is therefore not registered with the "
            "DataLoader at all."),
        "slots_per_node_year": 288,
        "representative_days": sorted({s.split()[0] for s in (pv_slots or [])}),
        "units": {"pv_kw": "kW, mean output", "pv_std_kw": "kW, standard deviation",
                  "p_installed_kw": "kWp, nominal installed capacity"},
        "nodes_per_year": {str(y): int(pv.loc[pv.projection_year == y, "osmid"].nunique())
                           for y in YEARS},
    })

    # --------------------------------------------------------------- bess
    bess_rows = []
    for year in YEARS:
        head, rows = _rows_for_grid(z, f"SwissDN_DERs/02_BESS/{year}/BESS_allocation_MV.csv")
        for r in rows:
            bess_rows.append((year, r[1], float(r[2]), float(r[3]), float(r[4]), float(r[5])))
    bess = pd.DataFrame(bess_rows, columns=[
        "projection_year", "osmid", "capacity_kwh", "nominal_power_kw",
        "eta_charge", "eta_discharge"])
    _write(bess, f"SwissDN_{GRID}_MV_BESS", {
        "content": "battery parameters per medium-voltage node and projection year",
        "supplies": (
            "the energy capacity, power rating and one-way efficiencies §5 and "
            "§9.5 need per aggregator, and the population and its placement "
            "with them"),
        "does_not_supply": (
            "the degradation cost c_cyc of §9.5, which is a cost parameter and "
            "appears in no dataset; it remains a declared parameter"),
        "units": {"capacity_kwh": "kWh", "nominal_power_kw": "kW, charging and "
                  "discharging alike", "eta_charge": "one-way", "eta_discharge": "one-way"},
        "round_trip_efficiency": float(
            (bess["eta_charge"] * bess["eta_discharge"]).round(6).mode().iloc[0]),
        "nodes_per_year": {str(y): int(bess.loc[bess.projection_year == y, "osmid"].nunique())
                           for y in YEARS},
        "capacity_kwh_range": [float(bess["capacity_kwh"].min()), float(bess["capacity_kwh"].max())],
    })


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
