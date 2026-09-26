"""External-solver arm for market 02: HiGHS clearing, policy on the card.

Same shape as `rollout_bench.run_cell_comm` (Python loop over the 48 steps,
policy forward vmapped on the card, one device_put each way per step, same key
schedule), with one change: both of the market's clearing operators -- the step
clearing and the boundary that opens an episode -- are HiGHS via
`ext_clear.ExtClear`, reached from the env's jitted step by `jax.pure_callback`
(the whole batch in one callback).  Settlement, state and observation stay the
market's own JAX code on the host CPU.

Reset: `step_auto_reset` evaluates `reset` (and so a boundary clearing) on every
step and selects it on `done`; a Python loop knows `done` and resets only then,
which is what a hand-written loop does.  The selection and the key are the
auto-reset wrapper's (same `k_step`, unsplit), so the fingerprint is comparable.

Nothing in tools/env_bench or powermarketjax is modified: the two module
attributes `make_rt_clearing` (env, boundary) and `adapters.unpack_env` are
rebound in this process only, and the rebinding is verified by counting solves.
"""
from __future__ import annotations

import argparse, json, os, platform, subprocess, sys, time
from pathlib import Path

import jax, jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rollout_bench as RB
import powermarketjax.envs.real_time.env as E
import powermarketjax.envs.real_time.boundary as Bd
import powermarketjax.learning.adapters as AD
from powermarketjax.envs.real_time.clearing import make_rt_clearing as _orig
from highs_clear import ExtClear

EXT = {}
_STEP = {}


def install(n_lanes, threads, warm, solver="highs", procs=0):
    def ext_factory(case, **kw):
        assert kw.get("n_lookahead", 1) == 1 and kw.get("monitored_lines") is None, kw
        _clear, spec = _orig(case, **kw)          # spec only; its IPM is never called
        tag = "boundary" if kw.get("ramp_scale", 1.0) > 100 else "step"
        if solver == "cvxpy":
            from cvxpy_clear import CvxClear
            X = CvxClear(case, kw.get("n_segments", 1), kw["cap_scale"], kw["ramp_scale"],
                         kw["period_hours"], n_lanes=1, threads=1)
        elif procs and tag == "step":
            from highs_clear import ExtClearProc
            X = ExtClearProc(case, kw.get("n_segments", 1), kw["cap_scale"], kw["ramp_scale"],
                             kw["period_hours"], n_lanes=n_lanes, procs=procs)
        elif procs:
            #: the boundary solves once per episode; in-process, serial
            X = ExtClear(case, kw.get("n_segments", 1), kw["cap_scale"], kw["ramp_scale"],
                         kw["period_hours"], n_lanes=n_lanes, threads=1, warm=warm)
        else:
            X = ExtClear(case, kw.get("n_segments", 1), kw["cap_scale"], kw["ramp_scale"],
                         kw["period_hours"], n_lanes=n_lanes, threads=threads, warm=warm)
        EXT[tag] = X
        return X.jax_clear(), spec
    E.make_rt_clearing = ext_factory
    Bd.make_rt_clearing = ext_factory
    _unpack = AD.unpack_env

    def unpack_keep_step(built):
        out = _unpack(built)
        _STEP["step"] = out[1]
        return out
    AD.unpack_env = unpack_keep_step


def timed(X):
    """Wrap solve_batch to accumulate the wall time spent inside HiGHS calls."""
    X.solve_s = 0.0
    inner = X.solve_batch

    def sb(*a):
        t = time.perf_counter(); r = inner(*a); X.solve_s += time.perf_counter() - t
        return r
    X.solve_batch = sb


def run(n_envs, repeats, threads, warm, inject, seed=0, solver="highs", procs=0,
        perturb_ulp=0, perturb_rel=0.0, policy_on_cpu=False):
    install(n_envs, threads, warm, solver, procs)
    reset, step_ar, env_params, spec, horizon, point = RB.BUILDS["02"]()
    for X in EXT.values():
        X.perturb_ulp, X.perturb_rel = perturb_ulp, perturb_rel
    step = _STEP["step"]
    for X in EXT.values():
        timed(X)
    cpus = jax.devices("cpu")
    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    if not gpus and not policy_on_cpu:
        raise SystemExit("needs a card for the policy (or --policy-on-cpu for a smoke run)")
    cpu = cpus[0]
    gpu = cpu if policy_on_cpu else gpus[0]
    keys0 = jax.device_put(jax.random.split(jax.random.PRNGKey(seed), n_envs), cpu)
    reset_j = jax.jit(jax.vmap(reset, in_axes=(0, None)))
    obs0, state0 = reset_j(keys0, env_params)
    apply, pol_params, act_dim = RB.make_policy(spec, int(obs0.shape[-1]), 0, seed)
    obs0 = jax.device_put(obs0, cpu); state0 = jax.device_put(state0, cpu)
    env_params = jax.device_put(env_params, cpu)
    pol = jax.device_put(pol_params, gpu)
    step_j = jax.jit(jax.vmap(step, in_axes=(0, 0, 0, None)))
    pol_j = jax.jit(jax.vmap(lambda p, o: apply(p, o, inject), in_axes=(None, 0)))
    split_j = jax.jit(jax.vmap(jax.random.split))

    def pick_j(done, a, b):
        return jax.tree.map(lambda x, y: jnp.where(done.reshape((-1,) + (1,) * (x.ndim - 1)), x, y), a, b)
    pick_j = jax.jit(pick_j)

    def rollout():
        keys, state, obs, rews = keys0, state0, obs0, []
        for _ in range(horizon):
            act = jax.device_put(pol_j(pol, jax.device_put(obs, gpu)), cpu)
            two = split_j(keys); keys, k_step = two[:, 0], two[:, 1]
            obs, state, rew, _c, done, _i = step_j(k_step, state, act, env_params)
            if bool(jnp.any(done)):
                r_obs, r_state = reset_j(k_step, env_params)
                obs, state = pick_j(done, (r_obs, r_state), (obs, state))
            rews.append(rew)
        r = jnp.stack(rews)
        return obs, jnp.moveaxis(r, 1, 0)

    t0 = time.perf_counter()
    ob, rr = rollout(); jax.block_until_ready((ob, rr))
    compile_s = time.perf_counter() - t0
    for X in EXT.values():
        X.solve_s = 0.0; X.n_solves = 0; X.simplex_iters = 0
        if hasattr(X, "log"):
            X.log.clear()

    def worker_ticks():
        t = 0
        for X in EXT.values():
            for pr in getattr(X, "procs", []) or []:
                f = open(f"/proc/{pr.pid}/stat").read().rsplit(")", 1)[1].split()
                t += int(f[11]) + int(f[12])
        return t
    load0, nb0 = os.getloadavg(), RB._neighbours()
    wk0 = worker_ticks()
    times, c0, w0 = [], RB._cpu_ticks(), time.perf_counter()
    for _ in range(repeats):
        t = time.perf_counter()
        ob, rr = rollout(); jax.block_until_ready((ob, rr))
        times.append(time.perf_counter() - t)
    cpu_s = (RB._cpu_ticks() - c0) / RB._TICK
    wall_s = time.perf_counter() - w0
    worker_cpu_s = (worker_ticks() - wk0) / RB._TICK
    fp = float(RB.fingerprint(ob[0], rr[0]))
    env_steps = n_envs * horizon
    med, mn = float(np.median(times)), float(np.min(times))
    st, bd = EXT["step"], EXT["boundary"]
    extra = {}
    if solver == "cvxpy":
        import cvxpy
        lg = st.log + bd.log
        wall = sum(r[0] for r in lg); slv = sum(r[1] for r in lg)
        comp = sum(r[2] for r in lg if r[2] is not None)
        tot = sum(times)
        extra = {
            "cvxpy_version": cvxpy.__version__,
            "cvxpy_solver_names": sorted({r[3] for r in lg}),
            "cvxpy_calls": len(lg),
            "cvxpy_solve_wall_s": wall, "cvxpy_solver_solve_time_s": slv,
            "cvxpy_compilation_time_s": comp,
            "cvxpy_canon_s": wall - slv,
            "pct_step_canon": 100 * (wall - slv) / tot,
            "pct_step_solver": 100 * slv / tot,
            "pct_step_outside_solve_call": 100 * (tot - wall) / tot,
            "ms_per_call_canon": 1e3 * (wall - slv) / len(lg),
            "ms_per_call_solver": 1e3 * slv / len(lg),
            "ms_per_call_compilation_time": 1e3 * comp / len(lg),
            "defaults": "Problem.solve() with no arguments; fresh Variable/Problem each call; no Parameter/DPP; no warm_start",
        }
    return {**extra,
        "market": "02", "running_point": point, "n_envs": n_envs, "horizon": horizon,
        "env_steps_per_rollout": env_steps, "obs_dim": int(obs0.shape[-1]), "act_dim": act_dim,
        "n_agents": int(spec.get("n_agents", -1)), "inject": bool(inject), "repeats": repeats,
        "compile_s": compile_s,
        "rollout_s_median": med, "rollout_s_min": mn, "rollout_s_all": times,
        "spread_max_over_min": max(times) / mn,
        "ms_per_env_step_median": 1e3 * med / env_steps,
        "ms_per_env_step_min": 1e3 * mn / env_steps,
        "env_steps_per_s_median": env_steps / med,
        "fingerprint_lane0": repr(fp),
        "cores_used": cpu_s / wall_s,
        #: `cores_used` reads this process only; the worker processes are separate
        "cores_used_workers": worker_cpu_s / wall_s,
        "cores_used_incl_workers": (cpu_s + worker_cpu_s) / wall_s, "cores_available": RB._affinity(),
        "platform": f"{cpu.platform}+{gpu.platform}",
        "device_kind": f"{cpu.device_kind}|{gpu.device_kind}",
        "x64": bool(jax.config.jax_enable_x64),
        "ext_solver": "cvxpy" if solver == "cvxpy" else "highspy",
        "highspy_version": __import__("highspy").Highs().version(),
        "perturb_ulp": perturb_ulp, "perturb_rel": perturb_rel,
        "ext_threads": threads, "ext_procs": procs,
        "ext_parallelism": (f"{procs} spawned processes, one core each, lanes split evenly"
                            if procs else f"thread pool of {threads}"), "ext_warm": bool(warm), "ext_presolve": "off",
        "ext_highs_threads_per_instance": 1,
        "ext_step_solves": st.n_solves, "ext_boundary_solves": bd.n_solves,
        "ext_step_simplex_iters": st.simplex_iters,
        "ext_simplex_iters_per_step_solve": st.simplex_iters / max(st.n_solves, 1),
        "ext_solve_s_total": st.solve_s + bd.solve_s,
        "ext_solve_share_of_rollout": (st.solve_s + bd.solve_s) / sum(times),
        "threads_env": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "XLA_FLAGS", "JAX_PLATFORMS", "XLA_PYTHON_CLIENT_PREALLOCATE",
                         "XLA_PYTHON_CLIENT_MEM_FRACTION", "CUDA_VISIBLE_DEVICES")},
        "threads_in_process": len(os.listdir("/proc/self/task")),
        "loadavg_before": load0, "loadavg_after": os.getloadavg(),
        "my_cores": sorted(os.sched_getaffinity(0)),
        "neighbours_before": nb0, "neighbours_after": RB._neighbours(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=None, help="default: len(affinity)")
    ap.add_argument("--cold", action="store_true", help="clearSolver before each solve")
    ap.add_argument("--inject", action="store_true")
    ap.add_argument("--procs", type=int, default=0,
                    help="HiGHS lanes over this many worker processes instead of threads")
    ap.add_argument("--perturb-ulp", type=int, default=0,
                    help="move every award and price the callback returns by this many ULP")
    ap.add_argument("--perturb-rel", type=float, default=0.0,
                    help="move every award and price the callback returns by this relative amount")
    ap.add_argument("--solver", choices=("highs", "cvxpy"), default="highs")
    ap.add_argument("--policy-on-cpu", action="store_true",
                    help="smoke run on a host without a card: the policy runs on the CPU "
                         "too, which is not the measured configuration")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not jax.config.jax_enable_x64:
        raise SystemExit("set JAX_ENABLE_X64=1")
    others = RB._neighbours()
    blocking = [o for o in others if not o.startswith("unpinned:")]
    if blocking:
        raise SystemExit(f"refusing to time a cell beside {len(blocking)} avoidable "
                         f"tenant(s) on its cores: {sorted(set(blocking))}")
    threads = args.threads or RB._affinity()
    args.out.mkdir(parents=True, exist_ok=True)
    tag = (f"02_ext-{args.solver}_n{args.n_envs}_c{RB._affinity()}_t{threads}"
           f"{'_p%d' % args.procs if args.procs else ''}{'_cold' if args.cold else ''}{'_ulp%d' % args.perturb_ulp if args.perturb_ulp else ''}{'_rel%g' % args.perturb_rel if args.perturb_rel else ''}{'_inject' if args.inject else ''}{'_polcpu' if args.policy_on_cpu else ''}")
    dest = args.out / f"{tag}.json"
    if dest.exists():
        print(f"skip {tag} (exists)"); return
    rec = run(args.n_envs, args.repeats, threads, not args.cold, args.inject, solver=args.solver,
              procs=args.procs, perturb_ulp=args.perturb_ulp, perturb_rel=args.perturb_rel,
              policy_on_cpu=args.policy_on_cpu)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    rec.update(arm=f"ext-{args.solver}", commit=head, host=platform.node(),
               stamped_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    dest.write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(f"{tag}  {rec['ms_per_env_step_median']:.4f} ms/env-step  spread={rec['spread_max_over_min']:.3f} "
          f"cores={rec['cores_used']:.1f}/{rec['cores_available']}  solve_share={rec['ext_solve_share_of_rollout']:.2f} "
          f"step_solves={rec['ext_step_solves']} bnd_solves={rec['ext_boundary_solves']} fp={rec['fingerprint_lane0']}",
          flush=True)


if __name__ == "__main__":
    main()
