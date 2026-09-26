"""Convert the RTS-GMLC source CSVs into a ``case29gb``-style case file.

Offline, run once, not imported by anything.  It reads the three vendored CSVs in
``raw_cases/rts_gmlc/`` and writes ``cases/transmission/case73rts.py``::

    python -m powermarketjax.case.raw_cases.rts_gmlc_to_case

**What this case is.**  RTS-GMLC is the 2019 update of the IEEE Reliability Test
System, placed on real geography in the US southwest (every bus carries a
latitude and longitude in Arizona, southern California and Nevada) and given
generator, branch and time-series data built for production-cost modelling.  It
is not a copy of an operating grid; it is a test system whose *parameters* are
real in the sense that they were assembled from utility practice rather than
invented per-run.  What makes it usable here, and what `case14` / `case118` /
`case300` are not, is that all three ingredients the day-ahead market
needs are present at once: real branch ratings, unit-commitment parameters,
and a demand interface.

**Only thermal units become units.**  Of the 158 generators, the 73 with fuel
Oil / Coal / NG / Nuclear (8 076 MW) are dispatchable and carry commitment data;
the other 85 do not belong in a thermal unit-commitment market:

* the 20 hydro units are 50 MW each with ``PMin = 0``, zero fuel price and zero
  VOM, so as units they would be 1 000 MW of free unconstrained capacity and
  would set the price to zero whenever they are marginal.  RTS-GMLC does not
  intend them that way — their output is given by a time series, which is what
  makes them energy-limited.
* wind, PV, rooftop PV, CSP and the one storage unit are likewise time series,
  and the three synchronous condensers produce no active power at all.

All of them are netted off demand in the data layer instead, where the
time-varying data lives.  Measured on the 2020 day-ahead series: gross demand
peaks at 8 192 MW against 8 076 MW of thermal capacity, a ratio of 1.01 that no
thermal-only clearing can serve, while net demand peaks at 6 228 MW, a ratio of
0.77 against the same capacity, next to 0.58 for `case29gb`.  Netting is
therefore not a convenience, it is what makes the case solvable at all.

**Cost curves.**  ``gen.csv`` gives an average heat rate at the first output
point and incremental heat rates for the segments above it, in BTU/kWh, plus a
fuel price in \\$/MMBTU and a VOM in \\$/MWh.  Those become total-cost points

    C(p_0) = HR_avg_0 · p_0 · fuel / 1000 + VOM · p_0
    C(p_k) = C(p_{k-1}) + HR_incr_k · (p_k - p_{k-1}) · fuel / 1000 + VOM · (p_k - p_{k-1})

which are fitted to ``TC(p) = NL + (a/3)p³ + (b/2)p² + c·p`` by **non-negative**
least squares.  The non-negativity is not cosmetic: an unconstrained fit of the
same design (``numpy.linalg.lstsq``) gives 10 of the 73 units a negative no-load
cost, and 5 of them an ``MC(p)`` that goes negative somewhere in ``[0, PMax]``,
because the RTS points start at ``PMin`` and the cubic is free to extrapolate
below it.  Inside ``[PMin, PMax]`` nothing goes properly negative: the only unit
that dips is ``121_NUCLEAR_1`` at −7e-13, four identical cost points over a 4 MW
span, which is the fit's conditioning and not a datum.  With all four
coefficients constrained non-negative, ``NL ≥ 0`` and
``MC(p) = a p² + b p + c`` is non-negative and non-decreasing by construction,
which is what §5 of the market document requires of an offer curve before the
action map ever sees it.

**What RTS-GMLC does not supply, and what is put in its place.**  ``gen.csv``
carries no initial on/off history, so ``keep_time`` is set to
``max(min_up_time, min_down_time)`` per unit, which is the value that makes the
initial minimum-up and minimum-down windows non-binding on the first period.
That is a declared starting condition, not a datum; a run that wants a binding
initial window has to state its own.  ``init_power`` and ``init_state`` come
from the base-case injection ``MW Inj``, clipped into ``[PMin, PMax]``.

The single DC branch (113–316, 100 MW) is absent, and nothing in this converter
drops it: ``branch.csv`` has no 113–316 row, so the drop happened where the three
CSVs were chosen.  The AC graph is connected without it — one component over all
73 buses — so it costs a 100 MW controllable tie inside an already-connected
network rather than the connectivity of the case.

Source: <https://github.com/GridMod/RTS-GMLC>, ``RTS_Data/SourceData/``,
retrieved 2026-08-21.  The data use notice is in ``raw_cases/rts_gmlc/LICENCE.md``
and requires that credit be given to DOE/NREL/ALLIANCE in any publication.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import nnls

#: Fuels whose generators become dispatchable units; see the module docstring.
THERMAL_FUELS = ("Oil", "Coal", "NG", "Nuclear")

#: ``gen.csv`` fuel → the ``type`` string ``case_builder.FUEL_MAP`` understands.
FUEL_TO_TYPE = {"Nuclear": "nuclear", "Coal": "coal", "NG": "gas", "Oil": "oil"}

RAW_DIR = Path(__file__).parent / "rts_gmlc"
OUT_PATH = (Path(__file__).parents[1] / "cases" / "transmission" / "case73rts.py")


def fit_cost(mw: np.ndarray, cost: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Fit ``TC(p) = NL + (a/3)p³ + (b/2)p² + c·p`` with all four terms non-negative.

    Returns ``(no_load, a, b, c, max_abs_residual)``.  Non-negative least squares
    rather than an unconstrained solve, because the points start at ``PMin`` and
    an unconstrained cubic extrapolates to a negative no-load cost on 10 of the
    73 units; see the module docstring.
    """
    design = np.column_stack([np.ones_like(mw), mw ** 3 / 3.0, mw ** 2 / 2.0, mw])
    coef, _ = nnls(design, cost)
    residual = float(np.max(np.abs(design @ coef - cost)))
    return float(coef[0]), float(coef[1]), float(coef[2]), float(coef[3]), residual


def cost_points(row: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Heat-rate table of one generator → (MW, \\$/h) total-cost points."""
    p_max, fuel, vom = row["PMax MW"], row["Fuel Price $/MMBTU"], row["VOM"]
    mw = [row["Output_pct_0"] * p_max]
    cost = [row["HR_avg_0"] * mw[0] * fuel / 1000.0 + vom * mw[0]]
    for k in range(1, 5):
        pct, hr_incr = row[f"Output_pct_{k}"], row[f"HR_incr_{k}"]
        if pd.isna(pct) or pd.isna(hr_incr):
            break
        p_k = pct * p_max
        cost.append(cost[-1] + (hr_incr * fuel / 1000.0 + vom) * (p_k - mw[-1]))
        mw.append(p_k)
    return np.asarray(mw, dtype=np.float64), np.asarray(cost, dtype=np.float64)


def build_tables(raw_dir: Path) -> Tuple[List, List, List, List, dict]:
    """Read the three CSVs and return the four ``build_case_from_tables`` tables."""
    gen = pd.read_csv(raw_dir / "gen.csv")
    branch = pd.read_csv(raw_dir / "branch.csv")
    bus = pd.read_csv(raw_dir / "bus.csv")

    nodes = [["id", "x", "y"]]
    for _, b in bus.iterrows():
        nodes.append([int(b["Bus ID"]), round(float(b["lng"]), 6), round(float(b["lat"]), 6)])

    thermal = gen[gen["Fuel"].isin(THERMAL_FUELS)].reset_index(drop=True)
    units = [["id", "bus_id", "type", "mc_a", "mc_b", "mc_c", "p_max", "p_min",
              "ramp_up", "ramp_down", "init_start_up_cost", "keep_time",
              "init_power", "init_state", "min_up_time", "min_down_time",
              "init_no_load_cost"]]
    worst_fit, worst_uid = 0.0, ""
    for i, g in thermal.iterrows():
        p_max, p_min = float(g["PMax MW"]), float(g["PMin MW"])
        no_load, a, b, c, residual = fit_cost(*cost_points(g))
        scale = max(abs(cost_points(g)[1]).max(), 1e-9)
        if residual / scale > worst_fit:
            worst_fit, worst_uid = residual / scale, g["GEN UID"]
        ramp = float(g["Ramp Rate MW/Min"]) * 60.0 / p_max      # fraction of p_max per hour
        min_up, min_down = float(g["Min Up Time Hr"]), float(g["Min Down Time Hr"])
        start_up = (float(g["Start Heat Hot MBTU"]) * float(g["Fuel Price $/MMBTU"])
                    + float(g["Non Fuel Start Cost $"]))
        init_power = float(np.clip(g["MW Inj"], p_min, p_max))
        units.append([
            i + 1, int(g["Bus ID"]), FUEL_TO_TYPE[g["Fuel"]],
            round(a, 9), round(b, 7), round(c, 5),
            p_max, p_min, round(ramp, 5), round(ramp, 5), round(start_up, 2),
            max(min_up, min_down), init_power, 1 if g["MW Inj"] > 0 else 0,
            min_up, min_down, round(no_load, 3),
        ])

    lines = [["id", "from", "to", "x", "floor", "cap"]]
    for i, (_, ln) in enumerate(branch.iterrows()):
        cap = float(ln["Cont Rating"])
        lines.append([i + 1, int(ln["From Bus"]), int(ln["To Bus"]),
                      float(ln["X"]), -cap, cap])

    load_bus = bus[bus["MW Load"] > 0].reset_index(drop=True)
    total = float(load_bus["MW Load"].sum())
    loads = [["id", "bus_id", "mc_a", "mc_b", "mc_c", "d_max", "d_min"]]
    for i, (_, b) in enumerate(load_bus.iterrows()):
        share = round(float(b["MW Load"]) / total, 6)
        loads.append([i + 1, int(b["Bus ID"]), 0, 0, 0, share, share])

    stats = dict(n_nodes=len(bus), n_units=len(thermal), n_lines=len(branch),
                 n_loads=len(load_bus), worst_fit=worst_fit, worst_uid=worst_uid,
                 thermal_mw=float(thermal["PMax MW"].sum()),
                 load_mw=total)
    return nodes, units, lines, loads, stats


def _fmt(table: List) -> str:
    """Render a header+rows table as the literal Python the case files carry."""
    out = ["        %r," % (table[0],)]
    for row in table[1:]:
        out.append("        %r," % (row,))
    return "\n".join(out)


HEADER = '''"""Case73RTS: RTS-GMLC 73-bus transmission, thermal units only.

Grid type: transmission
{n_nodes} buses, {n_units} thermal units, {n_lines} branches, {n_loads} loads
Base MVA: 100.0

Generated by ``powermarketjax.case.raw_cases.rts_gmlc_to_case`` from the source
CSVs vendored in ``raw_cases/rts_gmlc/``; edit that converter, not this file.
Read its docstring before using the case -- it records which generators were
dropped and why, how the heat-rate table becomes a marginal-cost polynomial, and
which two fields RTS-GMLC does not supply.

Demand is carried as bus participation factors summing to 1, as in `case29gb`,
because the megawatts come from a time series rather than from the case.  The
factors are the base-case bus loads of ``bus.csv``, which total {load_mw:.0f} MW
across {n_loads} of the {n_nodes} buses.  **Net demand, not gross**: the hydro,
wind, PV, rooftop-PV and CSP output that RTS-GMLC gives as time series is netted
off in the data layer, without which the {thermal_mw:.0f} MW of thermal capacity
here cannot serve the 8 192 MW gross peak.

Source: RTS-GMLC, <https://github.com/GridMod/RTS-GMLC>, retrieved 2026-08-21.
The data use notice in ``raw_cases/rts_gmlc/LICENCE.md`` requires that any
publication using this case credit DOE/NREL/ALLIANCE.
"""

from powermarketjax.case.case_builder import build_case_from_tables
from powermarketjax.case.case_data import CaseData


def create_case73rts() -> CaseData:
    """Build Case73RTS: {n_nodes}-bus RTS-GMLC transmission case, thermal units only."""

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
    print("  %(n_nodes)d buses, %(n_units)d thermal units (%(thermal_mw).0f MW), "
          "%(n_lines)d branches, %(n_loads)d loads" % stats)
    print("  worst cost-curve fit residual: %.4f%% of total cost (unit %s)"
          % (100 * stats["worst_fit"], stats["worst_uid"]))


if __name__ == "__main__":
    main()
