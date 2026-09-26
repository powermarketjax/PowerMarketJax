"""Record market 02's per-step clearing inputs and outputs, for the L2 check of
external implementations (`pypsa_arm.py`).

Runs our own 02 environment exactly as `tools/env_bench/rollout_bench.build_02`
builds it, with the rollout_bench policy (random init, greedy), one env, a few
episodes.  `make_rt_clearing` is wrapped from the outside (nothing in
`powermarketjax/` is edited): the wrapped `clear` hands its inputs and outputs
to a `jax.debug.callback`, so the step still runs under `jit`.

Writes the .npz named by `--out`: the case arrays the LP is built from, and per
step `offer (66,)`, `u (66,)`, `demand ()`, `p_init (66,)`, `award (66,)`,
`lmp (29,)`, `shed (29,)`, `mu ()`.
"""
import argparse, os, subprocess, sys
from pathlib import Path

ap = argparse.ArgumentParser(description="Record market 02's per-step clearing inputs and outputs.")
ap.add_argument("--episodes", type=int, default=4)
ap.add_argument("--out", type=Path, required=True, help="the .npz to write (e.g. rec02.npz)")
args = ap.parse_args()
n_ep = args.episodes

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "env_bench"))
import rollout_bench as RB
import powermarketjax.envs.real_time.env as RTE

REC = []
CAPT = {}
_orig = RTE.make_rt_clearing


def _wrapped(*a, **k):
    clear, cspec = _orig(*a, **k)
    CAPT["spec"], CAPT["kw"] = cspec, dict(k)

    def rec(offer, u, demand, p_init, award, lmp, shed, mu):
        REC.append(tuple(np.asarray(v, np.float64) for v in
                         (offer, u, demand, p_init, award, lmp, shed, mu)))

    def clear_rec(offer, u, demand, p_init):
        out = clear(offer, u, demand, p_init)
        jax.debug.callback(rec, offer, u, demand, p_init, out["award"],
                           out["lmp"], out["shed"], out["mu"], ordered=True)
        return out
    return clear_rec, cspec


RTE.make_rt_clearing = _wrapped

reset, step_ar, params, spec, horizon, point = RB.build_02()
key = jax.random.PRNGKey(0)
obs0, st0 = reset(key, params)
apply, pol, _ = RB.make_policy(spec, int(obs0.shape[-1]), 0, 0)
roll = jax.jit(RB.make_rollout(step_ar, apply, params, horizon, False))
for e in range(n_ep):
    k = jax.random.PRNGKey(100 + e)
    ob, stt = reset(k, params)
    out = roll(k, stt, ob, pol)
    jax.block_until_ready(out)
    jax.effects_barrier()

from powermarketjax.case import load_case, scale_min_output
from powermarketjax.envs.real_time import load_da_position
meta = load_da_position(chain="step1prime_seasons")["meta"]
case = scale_min_output(load_case(meta["case"]), float(meta.get("p_min_scale", 1.0)))
cs, kw = CAPT["spec"], CAPT["kw"]
ph = kw["period_hours"]
arr = [np.stack([r[i] for r in REC]) for i in range(8)]
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                        text=True, cwd=ROOT).stdout.strip()
np.savez(args.out,
         offer=arr[0].reshape(len(REC), -1), u=arr[1].reshape(len(REC), -1),
         demand=arr[2].reshape(len(REC)), p_init=arr[3],
         award=arr[4].reshape(len(REC), -1), lmp=arr[5].reshape(len(REC), -1),
         shed=arr[6].reshape(len(REC), -1), mu=arr[7].reshape(len(REC)),
         p_min=cs["p_min"], p_max=cs["p_max"], unit_bus=cs["unit_bus"],
         share=cs["demand_share"], PTDF=cs["PTDF"],
         F=np.asarray(case.line_cap, np.float64) * cs["cap_scale"],
         line_from=np.asarray(case.line_from_idx), line_to=np.asarray(case.line_to_idx),
         line_x=np.asarray(case.line_x, np.float64), slack=int(case.slack_bus_idx),
         ramp_up=np.asarray(case.unit_ramp_up, np.float64) * cs["p_max"] * ph * cs["ramp_scale"],
         ramp_dn=np.asarray(case.unit_ramp_down, np.float64) * cs["p_max"] * ph * cs["ramp_scale"],
         voll=10_000.0, off_eps=1e-3, running_point=point, commit=commit,
         kkt_route=str(cs["kkt_route"]), max_iter=int(kw["max_iter"]))
print("steps", len(REC), "point", point, "commit", commit[:10],
      "mu max", float(arr[7].max()), "route", cs["kkt_route"])
