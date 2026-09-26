"""Time PyPSA `n.optimize()` (all defaults) as market 02's environment step.

Single environment, Python loop, one `n.optimize()` per step, plus a policy
forward -- the rollout measure of `tools/env_bench/rollout_bench.py` (env +
policy forward, no gradient).  Offers are the recorded ones of rec02.npz, so
every timed solve is an instance that passed L2 (`pypsa_arm.py l2`); the policy
output is computed every step and fed nothing but the timing.

The breakdown wraps methods from the outside for timing only; the call is still
`n.optimize()` with no arguments:
  highs_run     highspy.Highs.run            -- the solve itself
  create_model  OptimizationAccessor.create_model  (linopy model assembly)
  lp_solve      linopy.Model.solve minus highs_run (linopy -> HiGHS transfer, readback)
  assign        assign_solution + assign_duals + post_processing
  consistency   Network.consistency_check (called by n.optimize)
  set_read      writing the step's attributes into the Network, reading p / prices
  policy        torch forward
Criterion: the parts sum to the step time within 10%.

usage: python pypsa_time.py <repeats> <out.json> --rec <rec02.npz> [--inject]
  --inject: the control of the breakdown check: 0.1 s of untimed sleep per
  step (~20% of it), so the named parts fall short of the total by more than
  10% and the check must go red.  No part is defined as a residual, so the
  check is not closed by construction.
"""
import argparse, json, os, sys, time, logging, platform, socket, subprocess
from pathlib import Path

import numpy as np
import torch
import highspy
import linopy
import pypsa
from pypsa.optimization.optimize import OptimizationAccessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pypsa_arm as A

logging.disable(logging.WARNING)
T = {k: 0.0 for k in ("highs_run", "create_model", "lp_solve", "assign", "consistency", "optimize", "set_read", "policy")}


def timed(owner, name, key, sub=None):
    f = getattr(owner, name)

    def w(*a, **k):
        t0 = time.perf_counter()
        h0 = T["highs_run"]
        try:
            return f(*a, **k)
        finally:
            dt = time.perf_counter() - t0
            T[key] += dt - ((T["highs_run"] - h0) if sub else 0.0)
    setattr(owner, name, w)


timed(highspy.Highs, "run", "highs_run")
timed(OptimizationAccessor, "create_model", "create_model")
timed(linopy.Model, "solve", "lp_solve", sub=True)
for nm in ("assign_solution", "assign_duals", "post_processing"):
    timed(OptimizationAccessor, nm, "assign")
timed(pypsa.Network, "consistency_check", "consistency")


class Policy(torch.nn.Module):
    """rollout_bench's SharedActorCritic shape: obs 16 -> 64 -> 64 -> (mean, value)."""
    def __init__(self, obs_dim=16, act_dim=1):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(obs_dim, 64), torch.nn.Tanh(),
                                        torch.nn.Linear(64, 64), torch.nn.Tanh())
        self.mu, self.v = torch.nn.Linear(64, act_dim), torch.nn.Linear(64, 1)

    def forward(self, x):
        h = self.body(x)
        return self.mu(h), self.v(h)


def main():
    ap = argparse.ArgumentParser(description="Time PyPSA n.optimize() as market 02's environment step.")
    ap.add_argument("repeats", type=int)
    ap.add_argument("out", help="the timing record (.json) to write")
    ap.add_argument("--rec", type=Path, required=True, help="the .npz written by record_02.py")
    ap.add_argument("--inject", action="store_true")
    args = ap.parse_args()
    repeats, out, inject = args.repeats, args.out, args.inject
    A.load_rec(args.rec)
    torch.manual_seed(0)
    pol = Policy().double()
    R = A.R
    nsteps = len(R["demand"])
    n = A.build()
    wall, per_rep = [], []
    cpu0, w0 = os.times(), time.perf_counter()
    obs = np.zeros((A.NU, 16))
    for rep in range(repeats):
        for k in T:
            T[k] = 0.0
        t_rep = time.perf_counter()
        for t in range(nsteps):
            tp = time.perf_counter()
            with torch.no_grad():
                act, _v = pol(torch.from_numpy(obs))
            T["policy"] += time.perf_counter() - tp
            if inject:
                time.sleep(0.1)
            ts = time.perf_counter()
            offer, u, p0 = R["offer"][t], R["u"][t], R["p_init"][t]
            d = A.set_step(n, offer, u, float(R["demand"][t]), p0)
            T["set_read"] += time.perf_counter() - ts
            to = time.perf_counter()
            status, cond = n.optimize()
            T["optimize"] += time.perf_counter() - to
            ts = time.perf_counter()
            award, shed, lmp = A.read(n, d, u)
            # the observation a market-02 agent would see next: own award, the
            # price at its bus, demand, and the previous action; 16 columns
            obs = np.zeros((A.NU, 16))
            obs[:, 0], obs[:, 1] = award / R["p_max"], lmp[R["unit_bus"]] / A.VOLL
            obs[:, 2], obs[:, 3] = float(R["demand"][t]) / 4e4, act.numpy()[:, 0]
            T["set_read"] += time.perf_counter() - ts
            assert status == "ok", (t, status, cond)
        dt = time.perf_counter() - t_rep
        wall.append(dt)
        per_rep.append(dict(T))
    cpu1, w1 = os.times(), time.perf_counter()
    cores_used = ((cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)) / (w1 - w0)
    med = int(np.argsort(wall)[len(wall) // 2])
    Tm, tot = per_rep[med], wall[med]
    parts = dict(highs_run=Tm["highs_run"], create_model=Tm["create_model"], lp_solve=Tm["lp_solve"],
                 assign=Tm["assign"], consistency=Tm["consistency"], set_read=Tm["set_read"],
                 policy=Tm["policy"])
    # not a part: n.optimize's own wall time, kept to show what of it is unnamed
    optimize_unnamed = Tm["optimize"] - sum(parts[k] for k in ("highs_run", "create_model",
                                                            "lp_solve", "assign", "consistency"))
    # the closure check: named parts vs the step total (loop overhead is what is left)
    closure = abs(tot - sum(parts.values())) / tot
    res = dict(
        candidate="PyPSA Network.optimize() out-of-the-box (linopy + HiGHS, all defaults)",
        market="02", running_point=str(R["running_point"]), rec_commit=str(R["commit"]),
        n_envs=1, horizon=nsteps, steps_per_repeat=nsteps, repeats=repeats, inject=inject,
        wall_s_all=wall, ms_per_env_step_median=1e3 * tot / nsteps,
        ms_per_env_step_all=[1e3 * w / nsteps for w in wall],
        parts_ms_per_step={k: 1e3 * v / nsteps for k, v in parts.items()},
        parts_pct={k: 100 * v / tot for k, v in parts.items()},
        closure_rel=closure, closure_pass=closure < 0.10,
        optimize_ms_per_step=1e3 * Tm["optimize"] / nsteps,
        optimize_unnamed_ms_per_step=1e3 * optimize_unnamed / nsteps,
        cores_used=cores_used, affinity=sorted(os.sched_getaffinity(0)),
        torch_threads=torch.get_num_threads(), loadavg=os.getloadavg(),
        versions=dict(pypsa=pypsa.__version__, linopy=linopy.__version__,
                      highspy=getattr(highspy, "__version__", "1.15.1"), torch=torch.__version__,
                      numpy=np.__version__, python=platform.python_version()),
        host=socket.gethostname(), stamped_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        hn04_head=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("ms_per_env_step_median", "ms_per_env_step_all", "parts_pct",
                                          "closure_rel", "closure_pass", "cores_used")}, indent=1))


if __name__ == "__main__":
    main()
