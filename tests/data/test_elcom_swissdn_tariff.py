# Written for this repository on 2026-08-14 -- no upstream counterpart.
"""Tests for the retail tariff of the municipalities grid 459_0 lies in.

What these defend is the chain that makes this an address price rather than a
national one, because every link in it is a place where a wrong value would
still look reasonable.

**The nine municipalities agree.**  That is what lets one number stand for the
feeder. If a boundary change ever split it between two utilities the agreement
would break, and the number would silently become the tariff of whichever row
happened to be read first.

**The energy component is not the total.**  §9.5 prices the energy an
aggregator stores, not the network service delivering it. The two differ by
roughly a factor of two here, so taking the wrong column doubles the cost basis
and every offer built on it, while leaving all reported quantities finite.

**The factor of ten is real.**  The source publishes centimes per kilowatt
hour; the specifications work in per megawatt hour. An undeclared factor of ten
in a cost basis is the same class of error as the MVA-versus-per-unit trap.
"""

from pathlib import Path

import pandas as pd
import pytest

DATA = Path(__file__).resolve().parents[2] / "powermarketjax/data/parquet"
TARIFF = DATA / "ElCom_SwissDN_459_0_Tariff_Annual.parquet"
META = TARIFF.with_suffix(".json")
BY_MUNICIPALITY = DATA / "ElCom_SwissDN_459_0_Tariff_Annual_ByMunicipality.parquet"

BFS_OF_459_0 = {
    "6602", "6614", "6616", "6617", "6619", "6624", "6625", "6626", "6629",
}


@pytest.fixture(scope="module")
def tariff():
    return pd.read_parquet(TARIFF)


@pytest.fixture(scope="module")
def priced(tariff):
    """Every row that carries an actual tariff; see the C5 test for the rest."""
    return tariff[tariff["energy_rp_kwh"] > 0]


@pytest.fixture(scope="module")
def meta():
    import json
    return json.loads(META.read_text(encoding="utf-8"))


def test_the_tariff_covers_the_municipalities_of_grid_459_0(meta):
    """The address, established from the grid rather than assumed."""
    assert set(meta["municipalities"]) == BFS_OF_459_0
    assert meta["municipalities"]["6616"] == "Collonge-Bellerive"
    assert meta["canton"] == "Geneva (GE)"
    assert meta["mv_grid"] == "459_0"


@pytest.fixture(scope="module")
def by_municipality():
    return pd.read_parquet(BY_MUNICIPALITY)


def test_one_operator_serves_all_nine(meta, by_municipality):
    """The property that makes an address price a single number.

    Computed from the per-municipality rows rather than read back out of the
    sidecar. An earlier version asserted a boolean the prep script had written
    there, which made the test incapable of failing on the data: the only
    check on the agreement lived in the script, and the script runs against
    the live endpoint rather than in CI.
    """
    assert meta["operator_id"] == "692"
    assert set(by_municipality["bfs"]) == BFS_OF_459_0
    assert set(by_municipality["operator"].astype(str)) == {"692"}

    components = ["energy_rp_kwh", "gridusage_rp_kwh", "charge_rp_kwh",
                  "aidfee_rp_kwh", "metering_rp_kwh", "total_rp_kwh"]
    spread = (by_municipality.groupby(["period", "category"])[components]
              .agg(lambda s: s.max() - s.min()))
    assert float(spread.to_numpy().max()) == 0.0
    # Every (period, category) is covered by all nine, so the zero above is
    # agreement rather than a group that happens to hold one municipality.
    assert set(by_municipality.groupby(["period", "category"]).size()) == {9}
    assert float(meta["largest_spread_rp_kwh"]) == float(spread.to_numpy().max())


def test_the_collapsed_table_is_the_agreed_row(tariff, by_municipality):
    """The published table must be the nine agreeing, not one of them picked."""
    components = ["energy_rp_kwh", "gridusage_rp_kwh", "charge_rp_kwh",
                  "aidfee_rp_kwh", "metering_rp_kwh", "total_rp_kwh"]
    key = ["period", "category"]
    collapsed = tariff.set_index(key)[components].sort_index()
    from_muni = (by_municipality.groupby(key)[components].first().sort_index())
    assert len(collapsed) * 9 == len(by_municipality)
    pd.testing.assert_frame_equal(collapsed, from_muni)


def test_the_energy_component_is_distinct_from_the_total(priced):
    """Taking total where §9.5 wants energy would roughly double the basis."""
    assert (priced["energy_chf_mwh"] < priced["total_chf_mwh"]).all()
    ratio = priced["total_chf_mwh"] / priced["energy_chf_mwh"]
    # Both bounds are measured, and the operating point of each is recorded
    # with it: the minimum falls on C7 in 2011, the largest commercial category,
    # where the network charge is proportionally smallest, and the maximum on H1
    # in 2026, the smallest household one, where the annual metering charge is
    # spread over the least consumption. Two earlier drafts carried numbers that
    # were plausible rather than measured, 1.5 and 3.286, and both failed
    # against the file.
    assert ratio.min() == pytest.approx(1.376, abs=5e-3)
    assert ratio.max() == pytest.approx(2.894, abs=5e-3)
    assert priced.loc[ratio.idxmin(), ["category", "period"]].tolist() == ["C7", 2011]
    assert priced.loc[ratio.idxmax(), ["category", "period"]].tolist() == ["H1", 2026]


def test_the_five_published_components_reconstruct_the_total(tariff):
    """The metering rate is the one that is easy to miss.

    It is empty before 2026 and enters the total as an annual charge divided by
    the reference consumption of the category, so it is largest exactly where
    consumption is smallest. Summing only the four obvious components left the
    smallest household category short by 5.03 Rp./kWh, a fifth of that tariff,
    and every quantity stayed finite.
    """
    parts = (tariff["energy_rp_kwh"] + tariff["gridusage_rp_kwh"]
             + tariff["charge_rp_kwh"] + tariff["aidfee_rp_kwh"]
             + tariff["metering_rp_kwh"])
    assert (parts - tariff["total_rp_kwh"]).abs().max() < 5e-3
    # Present but zero before 2026, non-zero after: the shape that made it easy
    # to omit without any year looking wrong on its own.
    assert (tariff.loc[tariff["period"] < 2026, "metering_rp_kwh"] == 0).all()
    assert (tariff.loc[tariff["period"] == 2026, "metering_rp_kwh"] > 0).all()


def test_the_unit_conversion_is_exactly_ten(priced):
    assert (priced["energy_chf_mwh"] / priced["energy_rp_kwh"]).round(9).eq(10.0).all()
    assert (priced["total_chf_mwh"] / priced["total_rp_kwh"]).round(9).eq(10.0).all()


def test_category_c5_is_a_row_of_zeros_rather_than_an_absent_row(tariff, meta):
    """The trap this file exists to pin.

    The operator does not offer C5, and ElCom publishes that as zeros rather
    than by omitting the row. A caller selecting C5 therefore gets a
    replacement cost of zero and offers priced at zero, and nothing raises.
    """
    c5 = tariff[tariff["category"] == "C5"]
    assert len(c5) == 12
    assert (c5["energy_rp_kwh"] == 0.0).all()
    assert (c5["gridusage_rp_kwh"] == 0.0).all()
    assert meta["categories_without_a_tariff"] == ["C5"]
    # Every other category carries a real energy component in every year.
    assert (tariff.loc[tariff["category"] != "C5", "energy_rp_kwh"] > 0).all()


def test_coverage_and_the_gaps_that_are_real(tariff):
    """232 rows rather than 240: the three largest commercial categories are
    not published for this operator in recent years. Pinned so that a silently
    truncated refetch is distinguishable from the source's own gaps."""
    assert tariff["period"].min() == 2011
    assert tariff["period"].max() == 2026
    assert len(tariff) == 232
    assert sorted(tariff["category"].unique()) == [
        "C1", "C2", "C3", "C4", "C5", "C6", "C7",
        "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"]

    present = set(zip(tariff["period"], tariff["category"]))
    missing = {(y, c) for y in range(2011, 2027)
               for c in tariff["category"].unique()} - present
    assert missing == {
        (2022, "C5"), (2023, "C5"), (2025, "C5"), (2025, "C6"),
        (2025, "C7"), (2026, "C5"), (2026, "C6"), (2026, "C7")}
    assert tariff.notna().all().all()


def test_the_price_is_positive_in_every_year(priced):
    """A retail tariff cannot go negative, which is why adopting it sidesteps
    the sign inversion of §9.3 rather than settling it. The test states the
    property the specification then relies on."""
    assert (priced["energy_chf_mwh"] > 0).all()
    assert priced["energy_chf_mwh"].min() > 50.0
    assert priced["energy_chf_mwh"].max() < 200.0


def test_the_source_is_recorded_as_first_tier(meta):
    """Sources are split into tiers and the second must
    be flagged. This one is the regulator's own interface, and the licence was
    read from the queried cube rather than off a web page."""
    assert meta["source_tier"] == "first"
    assert meta["source_organization"] == "Federal Electricity Commission ElCom"
    assert meta["licence_permits_redistribution"] is True
    assert "Open-Use" in meta["licence_url"]
    assert meta["temporal_resolution"] == "annual"
