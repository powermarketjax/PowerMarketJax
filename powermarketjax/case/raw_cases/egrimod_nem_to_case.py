"""Convert the egrimod-nem dataset into a ``case29gb``-style case file.

Offline, run once, not imported by anything.  It reads the CSVs vendored in
``raw_cases/egrimod_nem/`` and writes ``cases/transmission/case813nem.py``::

    python -m powermarketjax.case.raw_cases.egrimod_nem_to_case

**Read ``raw_cases/egrimod_nem/LICENCE.md`` first.**  The network half of this
dataset is Geoscience Australia data under CC BY 4.0 and redistributing it is
what that licence is for.  The generator half is compiled from AEMO's MMSDM and
NTNDP databases and redistributed under a downstream CC BY claim, which is the
construction `tools/data_prep/aemo_nsw1_price_and_rooftop_pv.py` already
declines to treat as settling AEMO redistribution.  The unit table below is a
fifth dataset waiting on that one ruling.

**Mainland only, and one pocket beyond that.**  A PTDF is built from line
reactances, and a DC link has none -- its flow is set by a controller rather than
by Kirchhoff's laws -- so a DC power-flow model cannot carry a controllable DC
tie.  Anything whose every path to the rest of the network is DC therefore has to
leave, because the susceptance matrix of a disconnected graph is singular and
there is no PTDF to compute at all.

That rule removes two things, not one.

*Tasmania*, 97 nodes, 24 scheduled units, 2 542 MW: it reaches the mainland only
through Basslink.  The 912-node AC graph splits into 815 + 97 without it.

*The Terranora pocket*, buses 298 (Bungalora) and 604 (Terranora), two 132 kV
buses at the far north-eastern tip of New South Wales near Tweed Heads.  Their
paths out are Directlink (HVDC, 180 MW, 605-298), which the rule above removes,
and the 110 kV Terranora-Mudgeeraba line into Queensland, which the dataset files
under AC interconnectors as ``N-Q-MNSP1``.  **MNSP is the NEM's term for a market
network service provider, and every NEM MNSP is a DC link** (Basslink, Murraylink
and Directlink/Terranora all are), so both of this pocket's ties are DC in
reality and it is the Tasmanian case one size down.  That last point is an
inference from the identifier and from NEM practice, not something this dataset
states; what the dataset does state is that Directlink sits in the HVDC file.

Keeping the pocket while dropping Directlink leaves it hanging off Queensland by
a single ±107 MW line while it consumes 129.1 MW (0.4303% of system demand at a
30 GW system) with **zero scheduled generation of its own** -- 22 MW short, every
hour of every day.  Measured before it was removed, that shortfall was the entire
source of the load shed in this case: on 359 clean days of a 370-day run the shed
landed on buses 298 and 604 and on no other bus ever, up to 152 MWh in a day, and
it pinned those days' price ceiling at ``VOLL``.  Removing the pocket costs 0.43%
of demand, no generation at all, and the smallest of the four interconnectors;
leaving it in costs every price statistic in the case.

The two remaining HVDC links, Murraylink (Victoria-South Australia, 220 MW) and
the Directlink corridor itself, are dropped by the same rule but cost transfer
capability rather than connectivity, because each parallels an AC path.

**Reactances are typological, not measured.**  ``X_PU / LENGTH_KM`` is the same
constant within each voltage class over all 1 406 edges -- 2.826e-3 at 110 kV,
2.011e-3 at 132 kV, 7.736e-4 at 220 kV, 5.109e-4 at 275 kV, 3.640e-4 at 330 kV,
1.682e-4 at 500 kV -- so the dataset derives reactance from length and voltage
rather than from per-line data.  The same measurement settles a question the
column definitions leave open: the ratio does not change with ``NUM_LINES``, so
``X_PU`` is the reactance of **one** circuit and a group of ``N`` parallel
circuits has ``X_PU / N``.  Using ``X_PU`` directly would overstate the impedance
of the 227 multi-circuit edges by a factor of 2 to 4 and move the flows.  Only
relative reactances enter a PTDF, so the per-unit base is irrelevant here.

**Line ratings: only the interconnectors carry one.**  The dataset has no thermal
rating column at all; the only real limits it publishes are the four AC
interconnector flow limits, and **three of the four reach the case** -- seven
edges in all.  The fourth is N-Q-MNSP1, whose edge 604-599 is the Terranora
pocket's tie: bus 604 left with the pocket, so that limit has nothing to sit on.
Every other line is left unlimited rather than given a rating constructed from
its voltage class, which is a deliberate choice: an intra-regional rating would
be a number this project invented, and the project does not invent data.
The price consequence is narrower than it looks.  With only inter-regional
corridors constrained the case cannot produce intra-regional congestion, but
what that removes is intra-regional congestion *rent*, not intra-regional price
differences: when a rated line binds, its dual enters every bus's price through
that line's whole PTDF row, and that row differs bus by bus.  Over the 813 buses
the seven rated rows take 211 distinct columns at 1e-6, and still 47 at 1e-2; the
rounding is quoted because the claim this replaces -- "the LMP takes at most four
distinct values in any period" -- came from picking one and not saying so.
A later measurement retracted that claim and found a
median of 187 distinct prices per period, on the 815-bus variant that still
carried the Terranora pocket.

**The published limits are directional and the case carries the smaller of each
pair.**  The day-ahead clearing enforces ``|flow| <= line_cap * cap_scale`` and
never reads ``line_floor``, so a case that wrote the two directions into the two
fields would have the reverse limit quietly replaced by the forward one. What
the symmetric minimum costs is one number per corridor: VIC1-NSW1 1 600 / 1 350
becomes 1 350, NSW1-QLD1 600 / 1 078 becomes 600, V-SA 600 / 500 becomes 500,
N-Q-MNSP1 107 / 210 becomes 107 -- that last one is the ±107 MW the pocket
paragraph weighs, and it does not reach the case.  Every one is inside both
published limits, so the case is conservative in the loose direction rather than
wrong in the tight one.

Two of the three that reach the case are single edges and take that limit
directly.  VIC1-NSW1 is five edges and its 1 600 / 1 350 MW limit is on their
**sum**, which a per-line cap cannot express.  It is split in proportion to how a
1 MW Victoria-to-New-South-Wales transfer between the two regional reference
nodes distributes over the five, measured on this network: 0.3862, 0.2700,
0.2427, 0.0617, 0.0394, summing to 1.  Under that transfer pattern the split
reproduces the aggregate limit exactly; under any other pattern it does not, so
the constraint is a stand-in for the aggregate one and not equal to it.

**Demand.**  ``PROP_REG_D`` gives each node's share of its **region's** demand and
sums to 1 within each region, so turning it into a share of *system* demand needs
a regional weighting.  The weights are the mean shares of mainland operational
demand over the AEMO forecast/actual panel already in this tree: NSW1 0.3751,
QLD1 0.3072, SA1 0.0664, VIC1 0.2513.  The dataset's own June-2017 regional
signals give 0.4010 / 0.2776 / 0.0703 / 0.2511 instead, so the choice moves a
node's share by up to three percentage points of system demand; the AEMO panel is
used because it is the series that will drive the case.

**What is paired with what.**  The fleet is the 2016-17 registered fleet and the
demand series is 2025.  That is an era mismatch and it is not a small one: the
mainland NEM has added grid-scale wind and solar and retired coal since, so this
case asks a 2016-17 scheduled fleet to serve 2025 operational demand with no
grid-scale renewables in front of it.  Measured on the AEMO panel, mainland
operational demand peaks at 32 475 MW against 39 164 MW of scheduled capacity, a
ratio of 0.83 where `case29gb` sits at 0.58, so the case is tighter than the
system it is named after.  The dataset ships one month of contemporaneous
regional demand (June 2017) which would remove the mismatch and the forecast
pair with it.

Source: <https://github.com/akxen/egrimod-nem-dataset>, retrieved 2026-08-21,
CC BY 4.0; see Xenophon & Hill, *Sci. Data* **5**, 180203 (2018).
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).parent / "egrimod_nem"
OUT_PATH = (Path(__file__).parents[1] / "cases" / "transmission" / "case813nem.py")

#: Buses excluded on top of the connectivity rule, because every path from them
#: to the rest of the NEM is a DC link.  See the module docstring: this is the
#: Tasmanian rule applied to a two-bus pocket rather than to an island.
DC_POCKET = (298, 604)

#: Mean share of mainland operational demand per NEM region, from the AEMO
#: forecast/actual panel in ``powermarketjax/data/parquet``; see the docstring.
REGION_SHARE = {"NSW1": 0.3751, "QLD1": 0.3072, "SA1": 0.0664, "VIC1": 0.2513}

#: ``FUEL_TYPE`` → the ``type`` string ``case_builder.FUEL_MAP`` understands.
FUEL_TO_TYPE = {
    "Black coal": "coal", "Brown coal": "coal",
    "Natural Gas (Pipeline)": "gas", "Coal seam methane": "gas",
    "Diesel oil": "oil", "Kerosene - non aviation": "oil",
    "Hydro": "hydro",
}


def mainland_nodes(nodes: pd.DataFrame, edges: pd.DataFrame) -> List[int]:
    """Node IDs of the largest connected component of the AC graph."""
    adj: Dict[int, List[int]] = {int(i): [] for i in nodes.NODE_ID}
    for a, b in zip(edges.FROM_NODE, edges.TO_NODE):
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    best: List[int] = []
    seen: set = set()
    for start in adj:
        if start in seen:
            continue
        queue, comp = deque([start]), []
        seen.add(start)
        while queue:
            u = queue.popleft()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    queue.append(v)
        if len(comp) > len(best):
            best = comp
    return sorted(best)


def ptdf(edges: pd.DataFrame, node_index: Dict[int, int], slack: int) -> np.ndarray:
    """DC PTDF of the mainland network, float64, reactance ``X_PU / NUM_LINES``."""
    n_lines, n_nodes = len(edges), len(node_index)
    x = (edges.X_PU / edges.NUM_LINES).to_numpy(np.float64)
    incidence = np.zeros((n_lines, n_nodes))
    incidence[np.arange(n_lines), [node_index[int(b)] for b in edges.FROM_NODE]] = 1.0
    incidence[np.arange(n_lines), [node_index[int(b)] for b in edges.TO_NODE]] = -1.0
    b_diag = np.diag(1.0 / x)
    b_bus = incidence.T @ b_diag @ incidence
    keep = [i for i in range(n_nodes) if i != slack]
    b_inv = np.zeros((n_nodes, n_nodes))
    b_inv[np.ix_(keep, keep)] = np.linalg.inv(b_bus[np.ix_(keep, keep)])
    return b_diag @ incidence @ b_inv


def interconnector_caps(
    edges: pd.DataFrame, nodes: pd.DataFrame, links: pd.DataFrame,
    limits: pd.DataFrame, node_index: Dict[int, int],
) -> Dict[int, Tuple[float, float]]:
    """Map edge row index → ``(floor, cap)``, symmetric, in MW.

    A single-edge interconnector takes its published limit directly.  A
    multi-edge one is split by DC transfer share between the two regional
    reference nodes; see the module docstring for why that is a stand-in for the
    aggregate limit rather than equal to it.

    **The limit is symmetric and equals ``min(forward, reverse)``**, because the
    day-ahead clearing enforces ``|flow| <= line_cap * cap_scale`` and never
    reads ``line_floor``. An asymmetric pair written into the two fields would
    have had its reverse limit silently replaced by the forward one, which on
    three of the four corridors is the wrong number in the tighter direction.
    """
    rrn = nodes[nodes.RRN == 1].set_index("NEM_REGION").NODE_ID.to_dict()
    slack = node_index[int(rrn["NSW1"])]
    shift = ptdf(edges, node_index, slack)
    limit = limits.set_index("INTERCONNECTOR_ID")
    out: Dict[int, Tuple[float, float]] = {}
    for ic_id, group in links.groupby("INTERCONNECTOR_ID"):
        if ic_id not in limit.index:
            continue
        forward = float(limit.loc[ic_id, "FORWARD_LIMIT_MW"])
        reverse = float(limit.loc[ic_id, "REVERSE_LIMIT_MW"])
        symmetric = min(forward, reverse)
        from_region = str(limit.loc[ic_id, "FROM_REGION"])
        to_region = str(limit.loc[ic_id, "TO_REGION"])
        injection = np.zeros(len(node_index))
        injection[node_index[int(rrn[from_region])]] = 1.0
        injection[node_index[int(rrn[to_region])]] = -1.0
        flow = shift @ injection
        rows, signs = [], []
        for _, link in group.iterrows():
            match = (((edges.FROM_NODE == link.FROM_NODE) & (edges.TO_NODE == link.TO_NODE))
                     | ((edges.FROM_NODE == link.TO_NODE) & (edges.TO_NODE == link.FROM_NODE)))
            for j in np.where(match.to_numpy())[0]:
                rows.append(int(j))
                signs.append(1.0 if edges.FROM_NODE.iloc[j] == link.FROM_NODE else -1.0)
        if len(rows) == 1:
            shares = [1.0]
        else:
            raw = np.array([s * flow[j] for j, s in zip(rows, signs)])
            shares = [float(v) for v in raw / raw.sum()]
        for j, _sign, share in zip(rows, signs, shares):
            cap = share * symmetric
            out[j] = (-cap, cap)
    return out


def build_tables(raw_dir: Path) -> Tuple[List, List, List, List, dict]:
    """Read the CSVs and return the four ``build_case_from_tables`` tables."""
    nodes_all = pd.read_csv(raw_dir / "network_nodes.csv")
    edges_all = pd.read_csv(raw_dir / "network_edges.csv")
    gens_all = pd.read_csv(raw_dir / "generators.csv")
    links = pd.read_csv(raw_dir / "network_ac_interconnector_links.csv")
    limits = pd.read_csv(raw_dir / "network_ac_interconnector_flow_limits.csv")

    keep = set(mainland_nodes(nodes_all, edges_all)) - set(DC_POCKET)
    # re-run the component search on the survivors: removing a pocket can strand
    # whatever sat behind it, and a stranded node is a singular matrix later
    surviving = edges_all[edges_all.FROM_NODE.isin(keep) & edges_all.TO_NODE.isin(keep)]
    keep = set(mainland_nodes(nodes_all[nodes_all.NODE_ID.isin(keep)], surviving))
    nodes_df = nodes_all[nodes_all.NODE_ID.isin(keep)].reset_index(drop=True)
    edges_df = edges_all[edges_all.FROM_NODE.isin(keep)
                         & edges_all.TO_NODE.isin(keep)].reset_index(drop=True)
    node_index = {int(b): i for i, b in enumerate(nodes_df.NODE_ID)}

    nodes = [["id", "x", "y"]]
    for _, node in nodes_df.iterrows():
        nodes.append([int(node.NODE_ID), round(float(node.LONGITUDE), 6),
                      round(float(node.LATITUDE), 6)])

    caps = interconnector_caps(edges_df, nodes_df, links, limits, node_index)
    lines = [["id", "from", "to", "x", "floor", "cap"]]
    for j, edge in edges_df.iterrows():
        floor, cap = caps.get(int(j), (0.0, 0.0))     # 0 → the builder's 1e6
        lines.append([j + 1, int(edge.FROM_NODE), int(edge.TO_NODE),
                      round(float(edge.X_PU) / int(edge.NUM_LINES), 9),
                      round(float(floor), 2), round(float(cap), 2)])

    gens = gens_all[(gens_all.SCHEDULE_TYPE == "SCHEDULED")
                    & gens_all.NODE.isin(keep)].reset_index(drop=True)
    units = [["id", "bus_id", "type", "mc_a", "mc_b", "mc_c", "p_max", "p_min",
              "ramp_up", "ramp_down", "init_start_up_cost", "keep_time",
              "init_power", "init_state", "min_up_time", "min_down_time",
              "init_no_load_cost"]]
    for i, gen in gens.iterrows():
        p_max = float(gen.REG_CAP)
        min_up, min_down = float(gen.MIN_ON_TIME), float(gen.MIN_OFF_TIME)
        # NL_FUEL_CONS is no-load fuel as a proportion of full-load consumption
        no_load = float(gen.NL_FUEL_CONS) * float(gen.HEAT_RATE) * p_max * float(gen["FC_2016-17"])
        units.append([
            i + 1, int(gen.NODE), FUEL_TO_TYPE[gen.FUEL_TYPE],
            0, 0, round(float(gen["SRMC_2016-17"]), 5),
            p_max, float(gen.MIN_GEN),
            round(float(gen.RR_UP) / p_max, 5), round(float(gen.RR_DOWN) / p_max, 5),
            float(gen.SU_COST_HOT), max(min_up, min_down), 0.0, 0,
            min_up, min_down, round(no_load, 2),
        ])

    # renormalise: PROP_REG_D sums to 1 within each *whole* region, so dropping
    # any bus leaves the participation factors summing to less than 1 and the
    # case would quietly serve less than the demand series asks for
    raw = np.array([float(node.PROP_REG_D) * REGION_SHARE[node.NEM_REGION]
                    for _, node in nodes_df.iterrows()])
    raw = raw / raw.sum()
    loads = [["id", "bus_id", "mc_a", "mc_b", "mc_c", "d_max", "d_min"]]
    for i, node in nodes_df.iterrows():
        share = round(float(raw[i]), 9)
        loads.append([i + 1, int(node.NODE_ID), 0, 0, 0, share, share])
    total = sum(row[5] for row in loads[1:])

    stats = dict(n_nodes=len(nodes_df), n_units=len(gens), n_lines=len(edges_df),
                 dropped_pocket=", ".join(str(b) for b in DC_POCKET),
                 n_loads=len(nodes_df), unit_mw=float(gens.REG_CAP.sum()),
                 n_capped=len(caps), share_total=total,
                 dropped_nodes=len(nodes_all) - len(nodes_df),
                 dropped_units=int(((gens_all.SCHEDULE_TYPE == "SCHEDULED")
                                    & ~gens_all.NODE.isin(keep)).sum()))
    return nodes, units, lines, loads, stats


def _fmt(table: List) -> str:
    return "\n".join("        %r," % (row,) for row in table)


HEADER = '''"""Case813NEM: mainland Australian National Electricity Market, egrimod-nem.

Grid type: transmission
{n_nodes} buses, {n_units} scheduled units, {n_lines} branches, {n_loads} loads
Base MVA: 100.0

Generated by ``powermarketjax.case.raw_cases.egrimod_nem_to_case`` from the CSVs
vendored in ``raw_cases/egrimod_nem/``; edit that converter, not this file.  Read
its docstring before using the case: it records why Tasmania is absent, why the
reactance of a multi-circuit edge is ``X_PU / NUM_LINES``, how the VIC1-NSW1
aggregate limit becomes five per-line caps, and what it costs to pair a 2016-17
fleet with a 2025 demand series.

Two buses beyond Tasmania are absent, {dropped_pocket}: the Terranora pocket,
whose every tie to the rest of the NEM is a DC link.  The converter docstring
records why, and what leaving it in cost.

**Only the interconnectors carry a line rating** ({n_capped} of the {n_lines}
edges).  The dataset publishes no thermal ratings, and rather than construct one
per voltage class this case leaves intra-regional lines unlimited, so it can
produce inter-regional congestion and cannot produce intra-regional congestion.

Demand is carried as bus participation factors summing to {share_total:.6f}, the
product of each node's share of its region's demand and that region's mean share
of mainland operational demand.

Source: egrimod-nem, <https://github.com/akxen/egrimod-nem-dataset>, CC BY 4.0,
retrieved 2026-08-21.  Xenophon, A. K. & Hill, D. J., "Open grid model of
Australia's National Electricity Market allowing backtesting against historic
data", *Scientific Data* **5**, 180203 (2018).  The unit table derives from AEMO
MMSDM and NTNDP data; see ``raw_cases/egrimod_nem/LICENCE.md`` for the
redistribution question that licence does not settle.
"""

from powermarketjax.case.case_builder import build_case_from_tables
from powermarketjax.case.case_data import CaseData


def create_case813nem() -> CaseData:
    """Build Case813NEM: {n_nodes}-bus mainland NEM transmission case."""

'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    nodes, units, lines, loads, stats = build_tables(args.raw_dir)
    body = HEADER.format(**stats)
    for name, table in (("_nodes", nodes), ("_units", units),
                        ("_lines", lines), ("_loads", loads)):
        body += "    %s = [\n%s\n    ]\n\n" % (name, _fmt(table))
    body += (
        "    return build_case_from_tables(\n"
        "        nodes_cols=_nodes[0], nodes_data=_nodes[1:],\n"
        "        units_cols=_units[0], units_data=_units[1:],\n"
        "        lines_cols=_lines[0], lines_data=_lines[1:],\n"
        "        loads_cols=_loads[0], loads_data=_loads[1:],\n"
        "        base_mva=100.0,\n"
        "    )\n"
    )
    args.out.write_text(body)
    print("wrote %s" % args.out)
    print("  %(n_nodes)d buses, %(n_units)d scheduled units (%(unit_mw).0f MW), "
          "%(n_lines)d branches, %(n_loads)d loads" % stats)
    print("  dropped, Tasmania plus the Terranora pocket: %(dropped_nodes)d nodes, "
          "%(dropped_units)d scheduled units" % stats)
    print("  edges carrying a rating: %(n_capped)d ; participation factors sum to "
          "%(share_total).6f" % stats)


if __name__ == "__main__":
    main()
