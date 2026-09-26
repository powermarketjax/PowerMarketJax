"""PyPSA `Network.optimize()` as an out-of-the-box market-02 clearing.

Every PyPSA / linopy / HiGHS setting is the package default: `n.optimize()` is
called with no arguments.  What this file decides is only how 02's single-period
SCED is written in PyPSA's own vocabulary:

  unit i with u=1   Generator, p_nom=p_max, p_min_pu=p_min/p_max (must-run),
                    marginal_cost=offer, p_init=previous output,
                    ramp_limit_down=R_dn/p_max,
                    ramp_limit_up=(R_up + p_min*[starting])/p_max -- the start-up
                    allowance of `envs/day_ahead/clearing.py`, starting = p_init==0
  unit i with u=0   active=False
  shed at bus b     Generator, p_nom=max(d_b, 1e-3), marginal_cost=VOLL
  load at bus b     Load, p_set=share_b * demand
  line l            Line, x=line_x, s_nom=F_l

L2 (`python pypsa_arm.py l2 --rec rec02.npz --out l2.json`): over every recorded
step of the recording, in ONE
execution, the correct arm and an injected arm (demand x 1.01) are both judged
by the same predicate; the first must pass and the second must fail.
"""
import argparse, json, sys, time, logging
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

HERE = Path(__file__).resolve().parent
R = None


def load_rec(path):
    """Load the recording written by `record_02.py` into the module globals."""
    global R, NU, NB, NL, VOLL, EPS, G, S, D
    R = np.load(path)
    NU, NB, NL = len(R["p_min"]), len(R["share"]), len(R["line_x"])
    VOLL, EPS = float(R["voll"]), float(R["off_eps"])
    G = [f"g{i}" for i in range(NU)]
    S = [f"s{i}" for i in range(NB)]
    D = [f"d{i}" for i in range(NB)]


def build():
    n = pypsa.Network()
    n.set_snapshots([0])
    buses = [f"b{i}" for i in range(NB)]
    n.add("Bus", buses)
    n.add("Line", [f"l{i}" for i in range(NL)], bus0=[buses[i] for i in R["line_from"]],
          bus1=[buses[i] for i in R["line_to"]], x=R["line_x"], s_nom=R["F"])
    n.add("Load", [f"d{i}" for i in range(NB)], bus=buses, p_set=0.0)
    n.add("Generator", [f"g{i}" for i in range(NU)], bus=[buses[i] for i in R["unit_bus"]],
          p_nom=R["p_max"], marginal_cost=0.0)
    n.add("Generator", [f"s{i}" for i in range(NB)], bus=buses, p_nom=1.0,
          marginal_cost=VOLL)
    return n


def set_step(n, offer, u, demand, p_init):
    on = u > 0.5
    start = on & (p_init <= 0.0)
    gen = n.generators
    gen.loc[G, "active"] = on
    gen.loc[G, "marginal_cost"] = offer
    gen.loc[G, "p_min_pu"] = np.where(on, R["p_min"] / R["p_max"], 0.0)
    gen.loc[G, "p_init"] = p_init
    gen.loc[G, "ramp_limit_up"] = (R["ramp_up"] + R["p_min"] * start) / R["p_max"]
    gen.loc[G, "ramp_limit_down"] = R["ramp_dn"] / R["p_max"]
    d = R["share"] * demand
    gen.loc[S, "p_nom"] = np.maximum(d, EPS)
    n.loads.loc[D, "p_set"] = d
    return d


def read(n, d, u):
    p = n.generators_t.p.iloc[0]
    award = p.reindex(G).fillna(0.0).to_numpy() * (u > 0.5)
    shed = np.where(d > 0, p.reindex(S).to_numpy(), 0.0)
    lmp = n.buses_t.marginal_price.iloc[0].reindex([f"b{i}" for i in range(NB)]).to_numpy()
    return award, shed, lmp


def objective(offer, u, award, shed):
    """02's own objective: offers on output above must-run, VOLL on shed."""
    return float(offer @ (award - R["p_min"] * u) + VOLL * shed.sum())


def feasible(award, shed, t, demand):
    """Largest violation of 02's LP rows by (award, shed) at step t, MW."""
    u, p0 = R["u"][t], R["p_init"][t]
    d = R["share"] * demand
    inj = np.zeros(NB); np.add.at(inj, R["unit_bus"], award)
    flow = R["PTDF"] @ (inj + shed - d)
    start = (u > 0.5) & (p0 <= 0)
    viol = [abs(award.sum() + shed.sum() - demand),
            (np.abs(flow) - R["F"]).max(),
            (award - p0 - R["ramp_up"] - R["p_min"] * start)[u > 0.5].max(),
            (p0 - award - R["ramp_dn"])[u > 0.5].max(),
            (R["p_min"] * u - award).max(), (award - R["p_max"] * u).max(),
            (-shed).max(), (shed - d).max()]
    return float(max(viol))


def l2(scale=1.0):
    logging.disable(logging.WARNING)
    n = build()
    rows = []
    for t in range(len(R["demand"])):
        offer, u, p0 = R["offer"][t], R["u"][t], R["p_init"][t]
        dem = float(R["demand"][t]) * scale
        d = set_step(n, offer, u, dem, p0)
        status, cond = n.optimize()
        award, shed, lmp = read(n, d, u)
        ours = objective(offer, u, R["award"][t], R["shed"][t])
        theirs = objective(offer, u, award, shed)
        # phantom mass of 02's own LP (`OFF_EPS` of envs/day_ahead/clearing.py):
        # every de-committed unit's segment and every zero-demand bus's shed
        # keeps a [0, OFF_EPS] box, which carries balance and is zeroed out of
        # `award` / `shed`; PyPSA has no such columns
        phi = EPS * (int((u < 0.5).sum()) + int((R["share"] * float(R["demand"][t]) <= 0).sum()))
        price = VOLL if R["shed"][t].sum() > 0 else float(offer[u > 0.5].max())
        rows.append(dict(t=t, status=status, cond=cond, phi_mw=phi,
                         dobj_abs=abs(theirs - ours), obj_bound=phi * price,
                         dobj_rel=abs(theirs - ours) / max(abs(ours), 1.0),
                         dlmp=float(np.abs(lmp - R["lmp"][t]).max()),
                         dq=float(np.abs(award - R["award"][t]).max()),
                         n_dq_gt_1e3=int((np.abs(award - R["award"][t]) > 1e-3).sum()),
                         viol_theirs=feasible(award, shed, t, float(R["demand"][t])),
                         viol_ours=feasible(R["award"][t], R["shed"][t], t, float(R["demand"][t]))))
    return rows


# Tolerances.  Objective: the phantom bound `phi * price`, i.e.
# the most the OFF_EPS columns can move 02's objective, derived from the
# mechanism rather than read off the data (a first pass at 1e-6 relative,
# chosen before any run, failed 14/192 steps, all within this bound).  Price:
# 1e-3 $/MWh.  Feasibility of PyPSA's point in 02's rows: 1e-4 MW.
TOL_LMP, TOL_VIOL = 1e-3, 1e-4


def classify(r):
    """Per-unit dispatch difference: 'none' (<=1e-6 MW), 'phantom' (<= phi,
    the OFF_EPS displacement), 'face' (> phi but both points feasible and the
    objectives within bound: another vertex of the same optimal face),
    'not_equivalent' otherwise."""
    if r["dq"] <= 1e-6:
        return "none"
    if r["dq"] <= r["phi_mw"]:
        return "phantom"
    ok = r["dobj_abs"] <= r["obj_bound"] and r["viol_theirs"] <= TOL_VIOL
    return "face" if ok else "not_equivalent"


def judge(rows):
    for r in rows:
        r["dq_class"] = classify(r)
    bad = [r for r in rows if not (r["status"] == "ok" and r["dobj_abs"] <= r["obj_bound"]
                                   and r["dlmp"] <= TOL_LMP and r["viol_theirs"] <= TOL_VIOL
                                   and r["dq_class"] != "not_equivalent")]
    return len(rows), len(bad), bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="L2 of the PyPSA arm against a record_02.py recording.")
    ap.add_argument("mode", choices=["l2"])
    ap.add_argument("--rec", type=Path, required=True, help="the .npz written by record_02.py")
    ap.add_argument("--out", type=Path, required=True, help="the L2 report (.json) to write")
    args = ap.parse_args()
    load_rec(args.rec)
    out = {}
    for name, sc in (("correct", 1.0), ("inject_demand_x1.01", 1.01)):
        rows = l2(sc)
        n_all, n_bad, bad = judge(rows)
        cls = {k: sum(r["dq_class"] == k for r in rows) for k in ("none", "phantom", "face", "not_equivalent")}
        out[name] = dict(n_steps=n_all, n_fail=n_bad, dq_class=cls, rows=rows)
        print(name, "dq_class", cls)
        print(name, "steps", n_all, "fail", n_bad,
              "max dobj_rel %.3g max dlmp %.3g max dq %.3g max viol_theirs %.3g max viol_ours %.3g" % tuple(
                  max(r[k] for r in rows) for k in ("dobj_rel", "dlmp", "dq", "viol_theirs", "viol_ours")))
    out["tol"] = dict(obj="phi*price per step", lmp=TOL_LMP, viol_mw=TOL_VIOL)
    out["versions"] = dict(pypsa=pypsa.__version__, pandas=pd.__version__)
    out["rec_commit"] = str(R["commit"])
    json.dump(out, open(args.out, "w"), indent=1)
