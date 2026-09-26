"""Fetch the retail tariff of the municipalities that grid 459_0 actually lies in.

Offline, run once, not imported by anything.  It supplies the exogenous energy
price of the local flexibility market for the Swiss configuration, as the
price of the place rather than the price of the country.

**Why an address price is available at all.**  Every low-voltage grid code in
the SwissDN dataset begins with the number of the municipality it sits in, and
the nodes of the medium-voltage grid 459_0 carry those codes.  Reading them
gives nine municipalities, all in canton Geneva on the eastern shore of the
lake: Anières, Choulex, Collonge-Bellerive, Cologny, Corsier, Gy, Hermance,
Jussy and Meinier.  ElCom publishes tariffs per municipality, so the feeder can
be priced where it stands.

**The nine agree, which is what makes the address price well defined.**  All
nine are served by one distribution utility, and every published figure is
identical across them in every year and every category.  The script asserts
that rather than assuming it: if a future boundary change split the feeder
between two utilities, an address price would stop being a single number and
the assertion is what would say so.

Source and provenance:

* Federal Electricity Commission ElCom, cube
  ``https://energy.ld.admin.ch/elcom/electricityprice``, queried over the
  LINDAS SPARQL endpoint ``https://lindas.admin.ch/query``.  No registration
  and no token.
* ElCom is the publisher and the creator of that cube, so this is the
  regulator's own interface rather than a secondary aggregation, which is what
  a first-tier source requires.
* Terms of use ``https://ld.admin.ch/vocabulary/TermsOfUse/Open-Use``, read
  from the cube itself in the same session as the observations, on the same
  reasoning applied to the day-ahead series: a licence read off the object
  queried is evidence about the values retrieved.

Three properties of the result decide how it may be used.

**The energy component is published apart from the network charge.**  §9.5
defines the exogenous price as what the aggregator pays for the energy it
stores, which is the energy component; charging a participant the network fee
and the public levies as well would price a different transaction.  Both are
carried here so that the choice is visible rather than silent.

**It is an annual figure, not a series.**  One value per year and consumption
category, so the price is constant over an episode of any length below a year.
That removes the intraday arbitrage motive from the market and leaves the
locational structure of §3.2 as the only thing an offer prices, which is a
modelling consequence to declare rather than a defect.  It also sidesteps
rather than settles the sign inversion of §9.3 recorded in §18: a retail
tariff is positive in every year published here, so the inversion cannot
occur under this price and returns the moment a wholesale series is used.

**The unit is centimes per kilowatt hour.**  Both the published figure and the
same number in CHF per megawatt hour are stored, the second being the first
times ten.  The market specifications work in per megawatt hour, and an
undeclared factor of ten in a cost basis is the kind of error that leaves
every reported quantity finite.

    python tools/data_prep/elcom_swissdn_tariff.py
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT_PARQUET = REPO / "powermarketjax/data/parquet/ElCom_SwissDN_459_0_Tariff_Annual.parquet"
OUT_META = OUT_PARQUET.with_suffix(".json")
#: The same figures before the nine municipalities are collapsed into one.
#: They are the raw material behind the agreement this script asserts, and the
#: collapsed table cannot reproduce it, and the material that recomputes a
#: conclusion is stored rather than the conclusion alone.
OUT_BY_MUNICIPALITY = OUT_PARQUET.with_name(
    "ElCom_SwissDN_459_0_Tariff_Annual_ByMunicipality.parquet")

ENDPOINT = "https://lindas.admin.ch/query"
CUBE = "https://energy.ld.admin.ch/elcom/electricityprice"

#: BFS numbers read off the low-voltage grid codes carried by the nodes of
#: medium-voltage grid 459_0, with the names from the municipality boundaries
#: shipped in the same archive.
MUNICIPALITIES = {
    "6602": "Anières", "6614": "Choulex", "6616": "Collonge-Bellerive",
    "6617": "Cologny", "6619": "Corsier (GE)", "6624": "Gy",
    "6625": "Hermance", "6626": "Jussy", "6629": "Meinier",
}
CANTON = "Geneva (GE)"
#: Centimes per kWh to CHF per MWh.
RP_KWH_TO_CHF_MWH = 10.0

LICENCE = "opendata.swiss Open use"
LICENCE_URL = "https://ld.admin.ch/vocabulary/TermsOfUse/Open-Use"
ATTRIBUTION = (
    "Federal Electricity Commission ElCom, electricity tariff per provider and "
    "municipality, retrieved from the LINDAS SPARQL endpoint. Terms of use: "
    "Open use. Changes: restricted to the nine municipalities of medium-voltage "
    "grid 459_0 and to the standard product, agreement across the nine "
    "verified, centimes per kWh carried through unchanged alongside the same "
    "figure in CHF per MWh."
)

QUERY = """
PREFIX e: <https://energy.ld.admin.ch/elcom/electricityprice/dimension/>
SELECT ?bfs ?period ?category ?energy ?gridusage ?charge ?aidfee ?metering ?total ?operator WHERE {
  VALUES ?m { %s }
  ?o e:municipality ?m ; e:category ?c ; e:period ?period ; e:operator ?opu ;
     e:product <https://energy.ld.admin.ch/elcom/electricityprice/product/standard> ;
     e:energy ?energy ; e:gridusage ?gridusage ; e:charge ?charge ;
     e:aidfee ?aidfee ; e:total ?total .
  OPTIONAL { ?o e:meteringrate ?mr }
  BIND(IF(BOUND(?mr) && STR(?mr) != "", ?mr, "0") AS ?metering)
  BIND(REPLACE(STR(?m), "^.*/", "") AS ?bfs)
  BIND(REPLACE(STR(?c), "^.*/", "") AS ?category)
  BIND(REPLACE(STR(?opu), "^.*/", "") AS ?operator)
}
"""


def _sparql(query: str) -> list[dict]:
    data = urllib.parse.urlencode({"query": query}).encode()
    req = urllib.request.Request(
        ENDPOINT, data=data,
        headers={"Accept": "application/sparql-results+json"})
    with urllib.request.urlopen(req, timeout=300) as fh:
        payload = json.load(fh)
    return [{k: v["value"] for k, v in row.items()}
            for row in payload["results"]["bindings"]]


def _licence_from_cube() -> dict[str, str]:
    rows = _sparql(f"""
    PREFIX schema: <http://schema.org/>
    PREFIX dcterms: <http://purl.org/dc/terms/>
    SELECT ?p ?o WHERE {{
      <{CUBE}> ?p ?o .
      FILTER(?p IN (dcterms:license, schema:publisher, schema:dateModified))
    }}""")
    return {r["p"].rsplit("/", 1)[-1].rsplit("#", 1)[-1]: r["o"] for r in rows}


def main() -> None:
    values = " ".join(f"<https://ld.admin.ch/municipality/{b}>" for b in MUNICIPALITIES)
    rows = _sparql(QUERY % values)
    if not rows:
        raise SystemExit("the endpoint returned nothing; the cube or the dimensions moved")
    raw = pd.DataFrame(rows)
    for col in ("energy", "gridusage", "charge", "aidfee", "metering", "total"):
        raw[col] = raw[col].astype("float64")
    # `int` is the platform integer, which is int64 here and was int32 in the
    # file this script first produced, so re-running it did not reproduce its
    # own output. Pinned so that it does.
    raw["period"] = raw["period"].astype("int32")

    operators = sorted(raw["operator"].unique())
    if len(operators) != 1:
        raise SystemExit(
            f"the nine municipalities are served by {len(operators)} operators "
            f"({operators}); an address price is no longer a single number")
    if sorted(raw["bfs"].unique()) != sorted(MUNICIPALITIES):
        raise SystemExit("the endpoint did not return every municipality asked for")

    # The agreement across the nine is the property that makes one tariff the
    # tariff of the feeder. Checked on every published figure, not just the one
    # that is going to be used.
    key = ["period", "category"]
    spread = raw.groupby(key)[
        ["energy", "gridusage", "charge", "aidfee", "metering", "total"]].agg(
        lambda s: s.max() - s.min())
    worst = float(spread.to_numpy().max())
    if worst > 0.0:
        raise SystemExit(
            f"the nine municipalities do not agree: largest spread within one "
            f"period and category is {worst} Rp./kWh")
    counts = raw.groupby(key).size()
    if set(counts) != {len(MUNICIPALITIES)}:
        raise SystemExit(f"uneven coverage across municipalities: {sorted(set(counts))}")

    df = (raw.drop(columns=["bfs"]).drop_duplicates(key)
             .sort_values(key).reset_index(drop=True))
    df = df.rename(columns={
        "energy": "energy_rp_kwh", "gridusage": "gridusage_rp_kwh",
        "charge": "charge_rp_kwh", "aidfee": "aidfee_rp_kwh",
        "metering": "metering_rp_kwh", "total": "total_rp_kwh"})
    df["energy_chf_mwh"] = df["energy_rp_kwh"] * RP_KWH_TO_CHF_MWH
    df["total_chf_mwh"] = df["total_rp_kwh"] * RP_KWH_TO_CHF_MWH

    # The five components published per kWh must reconstruct the total. The
    # metering rate is the one that is easy to miss: it is empty before 2026,
    # and it enters the total as an annual charge divided by the reference
    # consumption of the category, so it is largest exactly where consumption
    # is smallest. Omitting it left the smallest household category short by
    # 5.03 Rp./kWh, which is a fifth of that tariff and still a finite number.
    parts = (df["energy_rp_kwh"] + df["gridusage_rp_kwh"] + df["charge_rp_kwh"]
             + df["aidfee_rp_kwh"] + df["metering_rp_kwh"])
    residual = float((parts - df["total_rp_kwh"]).abs().max())
    if residual > 5e-3:
        raise SystemExit(
            f"the published components do not reconstruct the total; largest "
            f"residual {residual:.4f} Rp./kWh, so a component is missing here")

    # A category the operator does not offer is published as a row of zeros
    # rather than omitted, so a caller that selects it gets a cost basis of
    # zero and offers priced at zero, with nothing raising.
    empty = sorted(
        {str(c) for c in df.loc[df["energy_rp_kwh"] == 0.0, "category"].unique()})

    cube_meta = _licence_from_cube()
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, compression="zstd", index=False)

    # The per-municipality rows go in beside the collapsed table. Without them
    # the agreement checked above survives only as a boolean written by this
    # script, which a test can assert and never fail on; with them the spread
    # is recomputable from what is in the repository.
    by_muni = raw.rename(columns={
        "energy": "energy_rp_kwh", "gridusage": "gridusage_rp_kwh",
        "charge": "charge_rp_kwh", "aidfee": "aidfee_rp_kwh",
        "metering": "metering_rp_kwh", "total": "total_rp_kwh"})
    by_muni = by_muni.sort_values(["period", "category", "bfs"]).reset_index(drop=True)
    by_muni.to_parquet(OUT_BY_MUNICIPALITY, compression="zstd", index=False)

    meta = {
        "parquet_file": OUT_PARQUET.name,
        "source_organization": "Federal Electricity Commission ElCom",
        "source_url": ENDPOINT,
        "source_cube": CUBE,
        "source_tier": "first",
        "source_tier_evidence": (
            "ElCom is the publisher and creator of the cube queried, so this is "
            "the regulator's own interface rather than a secondary aggregation "
            "(docs/rtm.md §5.4)"),
        "cube_publisher": cube_meta.get("publisher"),
        "cube_date_modified": cube_meta.get("dateModified"),
        "licence": LICENCE,
        "licence_url": cube_meta.get("license", LICENCE_URL),
        "licence_permits_redistribution": True,
        "licence_evidence": (
            "read from the cube in the same session as the observations"),
        "attribution": ATTRIBUTION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mv_grid": "459_0",
        "municipalities": MUNICIPALITIES,
        "canton": CANTON,
        "municipality_source": (
            "the BFS number is the prefix of each low-voltage grid code carried "
            "by the nodes of 459_0; names from the municipality boundaries "
            "shipped in the same archive"),
        "operator_id": operators[0],
        "by_municipality_file": OUT_BY_MUNICIPALITY.name,
        "largest_spread_rp_kwh": worst,
        "agreement_note": (
            "every published figure is identical across the nine in every "
            f"period and category, largest spread {worst} Rp./kWh, so one "
            "tariff is the tariff of the feeder. Asserted rather than assumed: "
            "a boundary change splitting the feeder between two utilities "
            "would make an address price stop being a single number. The "
            "figure above is the measured spread rather than a constant, and "
            f"the rows it was measured on are in {OUT_BY_MUNICIPALITY.name}, "
            "so a reader can recompute it instead of trusting this line."),
        "product": "standard",
        "temporal_resolution": "annual",
        "temporal_note": (
            "one value per period and category, so the price is constant over "
            "any episode shorter than a year. This removes the intraday "
            "arbitrage motive and leaves the locational structure of §3.2 as "
            "the only thing an offer prices. It also sidesteps rather than "
            "settles the sign inversion of §9.3: a retail tariff is positive "
            "in every year published here, so the inversion cannot occur under "
            "this price and returns the moment a wholesale series is used."),
        "units": {
            "energy_rp_kwh": "Rp./kWh, energy component as published",
            "gridusage_rp_kwh": "Rp./kWh, network charge as published",
            "charge_rp_kwh": "Rp./kWh, community charges as published",
            "aidfee_rp_kwh": "Rp./kWh, federal levy as published",
            "metering_rp_kwh": ("Rp./kWh, metering charge as published; empty "
                                "before 2026 and stored as zero there"),
            "total_rp_kwh": "Rp./kWh, all components as published",
            "energy_chf_mwh": f"CHF/MWh, energy_rp_kwh x {RP_KWH_TO_CHF_MWH}",
            "total_chf_mwh": f"CHF/MWh, total_rp_kwh x {RP_KWH_TO_CHF_MWH}",
        },
        "which_component_is_the_exogenous_price": (
            "energy_chf_mwh. §9.5 prices the energy the aggregator stores, not "
            "the network service; total_chf_mwh is carried so that the choice "
            "is visible rather than silent."),
        "no_manifest_reason": (
            "an annual scalar keyed by period and category is neither a "
            "calendar series nor a repeatable profile, so it is not registered "
            "with the DataLoader; the market layer looks it up."),
        "component_identity_residual_rp_kwh": residual,
        "component_identity_note": (
            "energy + gridusage + charge + aidfee + metering reconstructs total "
            "to within the residual above. The metering rate is empty before "
            "2026 and enters the total as an annual charge divided by the "
            "reference consumption of the category, so it is largest where "
            "consumption is smallest; omitting it understated the smallest "
            "household category by 5.03 Rp./kWh."),
        "categories_without_a_tariff": empty,
        "categories_without_a_tariff_note": (
            "the operator publishes these as rows of zeros rather than omitting "
            "them, so selecting one yields a cost basis of zero and offers "
            "priced at zero with nothing raising. C5 is never a real tariff "
            "here: it is zero-filled in twelve years and absent in the rest."),
        "periods": [int(df["period"].min()), int(df["period"].max())],
        "categories": sorted(df["category"].unique()),
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{OUT_PARQUET.name}: {OUT_PARQUET.stat().st_size / 1e3:.1f} kB, "
          f"{len(df):,} rows, periods {meta['periods'][0]}-{meta['periods'][1]}, "
          f"{len(meta['categories'])} categories, operator {operators[0]}, "
          f"all nine municipalities agree")


if __name__ == "__main__":
    main()
