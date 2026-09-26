# Written for this repository on 2026-08-14 -- no upstream counterpart.
"""Tests for the Swiss medium-voltage grid 459_0 and its distributed resources.

Four of these tests exist because the vendoring script found the corresponding
fact by measurement and would have shipped a wrong claim without it.

**The unit reading.**  ``peak_mw`` and ``s_nom_mva`` are only relatable through
the base-case flow, and an earlier version of the vendoring script tied them
together with 20/sqrt(3) instead -- which agreed on the ten trunk lines,
because 0.9 x 12.76 = 11.48, and was 67% wrong elsewhere. The relation
recomputed below is the one that holds on all 128 edges. A units error here
rescales every requirement of §4 without making anything fail.

**The profile does not reach one.**  The source calls its demand profile
max-normalised and ships it with a maximum of 0.852801. The nodal series
therefore tops out at 85.28% of the registered peak, and a ``load_scale``
calibrated against it carries that factor. The test pins the number so that a
future release which fixes the normalisation cannot slip through as though
nothing had changed.

**The photovoltaic table is not a year.**  288 hourly values on twelve monthly
days, against 8760 for demand. The assertion is written as an inequality
against 8760 as well as an equality against 288, because the failure being
guarded is somebody expanding the table and leaving the tests passing.

**The carrier year is a firebreak.**  The demand series is stamped on 2029,
outside the span of ``ch_dayahead_price``, so joining the two on the calendar
returns nothing. That is deliberate: pairing a projection with a price year is
a declared choice under §14, and this makes doing it by accident impossible.
"""

from collections import defaultdict
from pathlib import Path

import pandas as pd
import pytest

from powermarketjax.data import DataLoader
from powermarketjax.data import signals as S

DATA = Path(__file__).resolve().parents[2] / "powermarketjax/data/parquet"
NODES = DATA / "SwissDN_459_0_MV_Nodes.parquet"
EDGES = DATA / "SwissDN_459_0_MV_Edges.parquet"
LOAD = DATA / "SwissDN_459_0_MV_Load_60min.parquet"
PV = DATA / "SwissDN_459_0_MV_PV_RepDays.parquet"
BESS = DATA / "SwissDN_459_0_MV_BESS.parquet"

PROFILE_MAX = 0.852801


@pytest.fixture(scope="module")
def nodes():
    return pd.read_parquet(NODES)


@pytest.fixture(scope="module")
def edges():
    return pd.read_parquet(EDGES)


def test_grid_is_a_connected_radial_tree(nodes, edges):
    """§3.1 admits a radial feeder only, and the sensitivity matrices of §3.2
    are defined by paths that exist only on a tree."""
    assert len(nodes) == 129
    assert len(edges) == len(nodes) - 1

    adj = defaultdict(set)
    for u, v in zip(edges["u"], edges["v"]):
        adj[u].add(v)
        adj[v].add(u)

    root = nodes.loc[nodes["source"], "osmid"]
    assert len(root) == 1, "a radial feeder has exactly one substation"

    seen, stack = {root.iloc[0]}, [root.iloc[0]]
    while stack:
        for m in adj[stack.pop()]:
            if m not in seen:
                seen.add(m)
                stack.append(m)
    assert len(seen) == len(nodes)


def test_the_feeder_is_load_dominated(nodes):
    """The feasibility argument of §6 needs every baseline injection to be a
    withdrawal; `case533mt_lo` was demoted for failing exactly this."""
    assert (nodes["peak_mw"] >= 0).all()
    assert nodes["peak_mw"].sum() == pytest.approx(13.7197, abs=1e-3)


def test_peak_mw_reads_as_mw_against_s_nom_in_mva(nodes, edges):
    """Recomputed here, not read out of the generator's metadata."""
    adj = defaultdict(list)
    for i, (u, v) in enumerate(zip(edges["u"], edges["v"])):
        adj[u].append((v, i))
        adj[v].append((u, i))

    root = nodes.loc[nodes["source"], "osmid"].iloc[0]
    parent, order, stack, seen = {}, [], [root], {root}
    while stack:
        n = stack.pop()
        order.append(n)
        for m, i in adj[n]:
            if m not in seen:
                seen.add(m)
                parent[m] = (n, i)
                stack.append(m)

    downstream = dict(zip(nodes["osmid"], nodes["peak_mw"]))
    for n in reversed(order):
        if n in parent:
            downstream[parent[n][0]] += downstream[n]

    implied = [
        downstream[n] / (edges.at[i, "base_case_flow"] * edges.at[i, "s_nom_mva"])
        for n, (_, i) in parent.items()
    ]
    assert len(implied) == len(edges)
    # A power factor, on every edge. Nothing else makes MW and MVA comparable.
    assert min(implied) == pytest.approx(0.90001, abs=1e-4)
    assert max(implied) <= 1.0


def test_line_ratings_are_real_and_leave_headroom_at_the_registered_peak(edges):
    """Only two of the seven vendored cases carry real ratings, and (LIM) is
    invisible without them. Headroom at the registered peak is what makes
    `load_scale` the thing that creates a thermal requirement."""
    assert (edges["s_nom_mva"] > 0).all()
    assert edges["s_nom_mva"].min() == pytest.approx(4.2, abs=1e-6)
    assert edges["s_nom_mva"].max() == pytest.approx(12.9, abs=1e-6)

    # The tightest line is not the one carrying the most demand: edge 33 is
    # rated 7.24 MVA against 5.88 MW downstream, where the trunk lines carry
    # more power into a 12.76 MVA rating. Reading the maximum off a list
    # sorted by demand gives 0.746 and is wrong.
    utilisation = edges["downstream_peak_mw"] / edges["s_nom_mva"]
    assert utilisation.max() < 1.0
    assert utilisation.max() == pytest.approx(0.8125, abs=1e-3)
    # Hence the scaling factor at which a thermal requirement first appears:
    # 1.231 against the registered peak, 1.443 once the profile's own maximum
    # of 0.8528 is accounted for. §14 records 1.2 on the Australian
    # configuration, so the two windows sit in the same place.
    assert 1.0 / utilisation.max() == pytest.approx(1.231, abs=5e-3)
    assert 1.0 / (utilisation.max() * PROFILE_MAX) == pytest.approx(1.443, abs=5e-3)


def test_nodal_demand_is_the_shipped_profile_times_the_peak():
    """Pins the 0.8528 the source ships in place of a maximum of one."""
    nodes_df = pd.read_parquet(NODES).set_index("osmid")["peak_mw"]
    load = pd.read_parquet(LOAD)

    assert len(load) == 8760 * 129
    assert load["osmid"].nunique() == 129

    reached = load.groupby("osmid")["load_mw"].max() / nodes_df.reindex(
        load.groupby("osmid")["load_mw"].max().index)
    assert reached.min() == pytest.approx(PROFILE_MAX, rel=1e-4)
    assert reached.max() == pytest.approx(PROFILE_MAX, rel=1e-4)


def test_the_carrier_year_cannot_be_joined_to_the_price_series():
    """The firebreak. If this ever overlaps, an accidental calendar join stops
    failing and starts returning a plausible misalignment instead."""
    load = pd.read_parquet(LOAD)
    assert sorted(load["datetime"].dt.year.unique()) == [2029]
    assert load["datetime"].min() == pd.Timestamp("2029-01-01 00:00", tz="UTC")

    price = pd.read_parquet(DATA / "CH_DayAhead_Price_2015_2025_60min.parquet")
    assert 2029 not in set(price["datetime"].dt.year.unique())
    assert load.merge(price, on="datetime", how="inner").empty


def test_photovoltaic_output_is_twelve_representative_days_not_a_year():
    pv = pd.read_parquet(PV)

    per_node_year = pv.groupby(["projection_year", "osmid"]).size()
    assert set(per_node_year) == {288}
    assert 8760 not in set(per_node_year)
    assert sorted(pv["month"].unique()) == list(range(1, 13))
    assert sorted(pv["hour_of_day"].unique()) == list(range(24))
    assert pv.groupby("projection_year")["osmid"].nunique().to_dict() == {
        2030: 29, 2040: 37, 2050: 47}
    assert (pv["pv_kw"] >= 0).all()
    assert (pv["pv_std_kw"] >= 0).all()


def test_battery_fleet_is_heterogeneous_and_sits_on_grid_nodes(nodes):
    """§5.4 records that a homogeneous portfolio degenerates the competition,
    so the spread is the property worth asserting, not the mere presence."""
    bess = pd.read_parquet(BESS)

    assert bess.groupby("projection_year")["osmid"].nunique().to_dict() == {
        2030: 15, 2040: 24, 2050: 34}
    assert bess["osmid"].isin(nodes["osmid"]).all()
    assert pd.read_parquet(PV)["osmid"].isin(nodes["osmid"]).all()

    assert bess["capacity_kwh"].max() / bess["capacity_kwh"].min() > 20
    assert bess["nominal_power_kw"].max() / bess["nominal_power_kw"].min() > 20
    # 85% round trip, as the paper states; sqrt of it one way.
    assert (bess["eta_charge"] * bess["eta_discharge"]).round(4).eq(0.85).all()
    assert (bess["eta_charge"] == bess["eta_discharge"]).all()


def test_swissdn_load_manifest_is_registered_and_resolves_by_source():
    loader = DataLoader()

    assert "swissdn_459_0_mv_load" in loader.registry.list_datasets()
    manifest = loader.registry.get_manifest("swissdn_459_0_mv_load")

    assert manifest.source == "swissdn"
    assert manifest.resolution == "60min"
    assert manifest.time_mode == "calendar"
    assert S.LOAD_ACTUAL_MW in manifest.signals
    assert len(manifest.region_values) == 129

    # load.actual_mw is mapped by several datasets, so the source filter is
    # what keeps them apart; asserted rather than assumed.
    by_source = loader.registry.find_by_signal(S.LOAD_ACTUAL_MW, source="swissdn")
    assert [m.name for m in by_source] == ["swissdn_459_0_mv_load"]


def test_the_photovoltaic_table_is_deliberately_not_registered():
    """`TimeAligner.align_profile` tiles end to end, which would turn twelve
    monthly days into a twelve-day cycle. Nothing may reach this table through
    the loader."""
    loader = DataLoader()

    names = loader.registry.list_datasets()
    assert not any("pv" in n.lower() and "swissdn" in n.lower() for n in names)
    assert PV.exists(), "the table must exist even though it carries no manifest"
