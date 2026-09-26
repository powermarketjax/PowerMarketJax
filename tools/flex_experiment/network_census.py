"""Full-year census of the two thermal thresholds this market carries.

    JAX_PLATFORMS=cpu PYTHONPATH=.:tools/flex_experiment python -m network_census

`requirement.py` publishes ``req_th = max(flow - p_max, 0)`` against the
**registered** rating, while `clearing.py` imposes (LIM) at ``p_max * (1 -
thermal_margin)``.  The two are different quantities and the report's Section
1.2 used the first while describing what the second procures against, so both
are counted here over all 8760 hours with no battery acting.

The margin exists because §6's linearisation drops the loss terms, so a line
cleared onto its registered rating is over that rating in the sweep of §7; it
is a safety margin on the constraint, not a second requirement.  The
consequence is that the clearing buys in hours the published signal reports
zero in, and the count of those hours is what a claim about "the need this
feeder has" has to be written against.

Nothing here depends on a policy, a seed or a device: with the photovoltaic
arrays zeroed (§1.1) and no award, the baseline injection is the scaled
registered load alone, so the census is a property of the feeder and the
scaling.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from powermarketjax.case.cases.distribution.case459_0 import (  # noqa: E402
    EN50160_MV_V_MAX, EN50160_MV_V_MIN, create_case459_0)
from powermarketjax.envs.local_flexibility.data import (  # noqa: E402
    load_swiss_flex_series)
from powermarketjax.envs.local_flexibility.data import \
    _default_data_dir  # noqa: E402
from powermarketjax.envs.local_flexibility.sensitivity import \
    build_voltage_sensitivity  # noqa: E402

from concentration_baseline import (EPISODE_LEN, KAPPA,  # noqa: E402
                                    SWISS_BESS_FILE, TARIFF_CATEGORY,
                                    TARIFF_PERIOD, THERMAL_MARGIN, day_split,
                                    series_day_of_month)

SCALINGS = (1.50, 1.55, 1.65, 1.75)


def placement_duty(kappa: float) -> list:
    """What the two published photovoltaic arrays do to the requirement.

    §1.1 zeroes them, and the reason is that they are not a nuisance parameter:
    §3.1 defines the output per aggregator, so the 34-bus placement arrives
    with more modelled generation and hence a different net demand.  The share
    of hours carrying a requirement under each published array is the quantity
    that says how different, and it is computed here rather than quoted from a
    console so that it can be recomputed.
    """
    case = create_case459_0(node_v_min=EN50160_MV_V_MIN,
                            node_v_max=EN50160_MV_V_MAX)
    sens = build_voltage_sensitivity(case)
    nodes = pd.read_parquet(_default_data_dir() / "SwissDN_459_0_MV_Nodes.parquet")
    bus_of = {str(o): i for i, o in enumerate(nodes["osmid"])}
    table = pd.read_parquet(_default_data_dir() / SWISS_BESS_FILE)
    pd_mw = np.asarray(case.node_pd, np.float64)
    total, base = float(pd_mw.sum()), float(sens.base_mva)
    pd_pu = pd_mw / (base * total)
    A = np.asarray(sens.A, np.float64)
    p_max = np.asarray(sens.p_max, np.float64) * (1.0 - THERMAL_MARGIN)
    rows = []
    for year in (2040, 2050):
        series = load_swiss_flex_series(projection_year=year,
                                        tariff_category=TARIFF_CATEGORY,
                                        tariff_period=TARIFF_PERIOD)
        fleet = table.query("projection_year == @year").sort_values("osmid")
        ab = np.array([bus_of[str(o)] for o in fleet["osmid"]], np.int64)
        load = np.asarray(series.load_mw, np.float64) * kappa
        pv = np.asarray(series.pv, np.float64)                # (8760, n_agent) MW
        for label, arr in (("published", pv), ("zeroed", np.zeros_like(pv))):
            p_inj = -load[:, None] * pd_pu[None, :]
            p_inj = p_inj.copy()
            np.add.at(p_inj, (slice(None), ab), arr / base)
            req = np.maximum(-(p_inj @ A.T) - p_max[None, :], 0.0).max(1)
            rows.append(dict(kappa=kappa, projection_year=year, pv=label,
                             n_agent=int(ab.size),
                             pv_capacity_mw=float(pv.max(0).sum()),
                             duty=float((req > 0).mean()),
                             hours=int((req > 0).sum()),
                             peak_mw=float(req.max() * base)))
    return rows


def census(kappa: float) -> dict:
    case = create_case459_0(node_v_min=EN50160_MV_V_MIN,
                            node_v_max=EN50160_MV_V_MAX)
    sens = build_voltage_sensitivity(case)
    series = load_swiss_flex_series(projection_year=2040,
                                    tariff_category=TARIFF_CATEGORY,
                                    tariff_period=TARIFF_PERIOD)
    pd_mw = np.asarray(case.node_pd, np.float64)
    qd_mw = np.asarray(case.node_qd, np.float64)
    total = float(pd_mw.sum())
    base = float(sens.base_mva)
    pd_pu, qd_pu = pd_mw / (base * total), qd_mw / (base * total)

    # (8760, n_bus) baseline injections: load only, negative into the feeder
    load = np.asarray(series.load_mw, np.float64) * kappa            # (8760,)
    p_inj = -load[:, None] * pd_pu[None, :]
    q_inj = -load[:, None] * qd_pu[None, :]

    flow = -(p_inj @ np.asarray(sens.A, np.float64).T)               # (8760, n_line) pu
    v_sq = 1.0 + 2.0 * (p_inj @ np.asarray(sens.R, np.float64).T
                        + q_inj @ np.asarray(sens.X, np.float64).T)

    p_max = np.asarray(sens.p_max, np.float64)                       # pu
    out = {"kappa": kappa, "thermal_margin": THERMAL_MARGIN,
           "min_voltage_pu": float(np.sqrt(v_sq.min())),
           "v_lo_pu": float(np.min(case.node_v_min))}
    _, dom = series_day_of_month()
    # The runs score 36 held-out days, so the requirement over exactly those
    # hours is the only denominator a procured volume from a run may be
    # divided by; the year figure answers a different question and the two
    # were mixed in the second version of the report.
    _, _, test = day_split(dom, EPISODE_LEN)
    test_hours = (np.asarray(test, np.int64)[:, None]
                  + np.arange(EPISODE_LEN)[None, :]).reshape(-1)
    out["test_hours"] = int(test_hours.size)
    for label, thr in (("registered", p_max),
                       ("procured", p_max * (1.0 - THERMAL_MARGIN))):
        req = np.maximum(flow - thr[None, :], 0.0)
        hour_has = req.max(1) > 0.0
        lines = np.flatnonzero(req.max(0) > 0.0)
        hours = int(hour_has.sum())
        days = int(np.unique(np.arange(len(load))[hour_has] // 24).size)
        # never summed across lines (`requirement.py`): nested lines on one
        # path would count the same injection more than once
        per_hour = req.max(1)
        out[label] = dict(
            hours=hours, days=days,
            hours_per_day=round(hours / days, 3) if days else None,
            lines=int(lines.size), line_ids=lines.tolist(),
            peak_mw=float(per_hour.max() * base),
            mean_given_positive_mw=float(per_hour[hour_has].mean() * base)
            if hours else 0.0,
            energy_mwh=float(per_hour.sum() * base),
            test_hours_with_requirement=int(hour_has[test_hours].sum()),
            test_peak_mw=float(per_hour[test_hours].max() * base),
            test_mean_all_periods_mw=float(per_hour[test_hours].mean() * base),
            test_energy_mwh=float(per_hour[test_hours].sum() * base))
    r, p = out["registered"], out["procured"]
    out["ratio_hours"] = round(p["hours"] / r["hours"], 3) if r["hours"] else None
    out["ratio_peak"] = round(p["peak_mw"] / r["peak_mw"], 3) if r["peak_mw"] else None
    return out


def main() -> int:
    #: No options: the census is fixed by the module constants.  The parser
    #: exists so that `--help` prints this and exits instead of running the
    #: whole census and overwriting the JSON below; run without arguments it
    #: does exactly what it did before.
    argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Takes no options.  Writes network_census.json into the "
               "flex-concentration figures directory, relative to the "
               "working directory.").parse_args()
    rows = [census(k) for k in SCALINGS]
    print("%-7s %-11s %7s %6s %8s %7s %10s %9s"
          % ("kappa", "threshold", "hours", "days", "h/day", "lines",
             "peak MW", "MWh/yr"))
    for row in rows:
        for label in ("registered", "procured"):
            d = row[label]
            print("%-7.2f %-11s %7d %6d %8.2f %7d %10.4f %9.1f"
                  % (row["kappa"], label, d["hours"], d["days"],
                     d["hours_per_day"] or 0.0, d["lines"], d["peak_mw"],
                     d["energy_mwh"]))
    duty = placement_duty(KAPPA)
    print("\nwhat the published photovoltaic arrays do, at the procured "
          "threshold and scaling %.2f:" % KAPPA)
    print("%-6s %-11s %7s %10s %8s %10s"
          % ("year", "pv", "agents", "pv peak MW", "duty", "hours"))
    for r in duty:
        print("%-6d %-11s %7d %10.3f %8.4f %10d"
              % (r["projection_year"], r["pv"], r["n_agent"],
                 r["pv_capacity_mw"], r["duty"], r["hours"]))
    out = Path("docs/figures/flex-concentration/network_census.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"scalings": rows, "placement_duty": duty},
                              indent=1))
    print(f"\nwrote {out}")
    a = next(r for r in rows if r["kappa"] == KAPPA)
    print("at the adopted scaling %.2f: the published signal reports %d hours, "
          "the clearing procures against %d (x%.2f); peak %.4f against %.4f MW"
          % (KAPPA, a["registered"]["hours"], a["procured"]["hours"],
             a["ratio_hours"], a["registered"]["peak_mw"],
             a["procured"]["peak_mw"]))
    print("minimum voltage %.4f pu against a %.3f bound"
          % (a["min_voltage_pu"], a["v_lo_pu"]))
    t = a["procured"]
    print("over the %d held-out hours: %d carry a requirement, peak %.4f MW, "
          "mean over all of them %.4f MW, total %.2f MWh"
          % (a["test_hours"], t["test_hours_with_requirement"],
             t["test_peak_mw"], t["test_mean_all_periods_mw"],
             t["test_energy_mwh"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
