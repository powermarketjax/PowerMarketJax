"""One full IPPO training iteration -- rollout, GAE and the gradient update --
timed on two arms.  The gradient-including counterpart of `rollout_bench.py`.

`rollout_bench.py` times a rollout and excludes the update.  Published
throughput comparisons of RL environments time full training, so a
rollout-only ratio set beside it is a different quantity.  This device times
the iteration a training run actually repeats:

    gpu    jit(iterate)                      `make_ippo`'s own iterate, one jit:
                                             scan rollout + GAE + 320 Adam steps
    comm   env on CPU, policy on GPU         Python loop over the horizon,
                                             device_put each step; the buffer
                                             goes to the card once and GAE +
                                             update run there under one jit

**Both arms run the same update function.**  `make_ippo` exposes only
`(init, iterate)`; its `_act`, `_gae` and `_update` are closures of `iterate`.
They are read out of the closure cells rather than restated here, so the comm
arm's gradient step is the gpu arm's gradient step by identity, not by a copy
that could drift: a restated update goes stale silently when the learner
changes.  The comm arm's key schedule is `iterate`'s and
`_rollout`'s step for step, so with the same inputs both arms produce the same
trajectory up to rounding, and the same parameter update.

The comm arm's policy forward is `vmap` over environments, one call per step
for every participant at once -- the SB3/SBX shape with batched inference, not
a Python loop over agents, which would make the conventional arm a naive
baseline rather than a like-for-like one.

Every cell writes, besides the timing json, an `.npz` with the parameter
update `params_after - params_before` and lane 0's reward series, so the
cross-arm gates (`train_gates.py`) read both arms off disk.

    JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=0 taskset -c 16-47 \
      python tools/env_bench/train_bench.py --market 04 --arm gpu --out DIR
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rollout_bench as RB                                     # noqa: E402


def _closure(f):
    """`f`'s free variables by name.  Refuses a name that is not there."""
    return dict(zip(f.__code__.co_freevars,
                    (c.cell_contents for c in f.__closure__)))


def _pick(f, *names):
    cells = _closure(f)
    missing = [n for n in names if n not in cells]
    if missing:
        raise SystemExit(f"{f.__name__} no longer closes over {missing}; it "
                         f"closes over {sorted(cells)}. The comm arm reads the "
                         f"learner's own functions out of these cells.")
    return [cells[n] for n in names]


def learner(market, n_envs, inject, seed=0, epochs=None):
    """The market's env, `make_ippo` on it, and the shared starting carry."""
    from powermarketjax.learning.ippo import action_layout, make_ippo
    from powermarketjax.learning.policy import bounds_for
    RB._driver_path()
    import hyperparams as HP                                   # tools/benchmark
    reset, step_ar, env_params, spec, horizon, point = RB.BUILDS[market]()
    bounds = bounds_for(spec)
    if inject:
        #: the control: every action the environment sees is the box's lower
        #: corner, whatever the policy says.  The learner still runs in full --
        #: sampling, log-probs, the update -- so the timed region is the same
        #: work; only the action's route into the environment is cut.
        _, _, act_shape, low, _ = action_layout(spec, bounds)
        const = jnp.asarray(low).reshape(act_shape)
        _real = step_ar
        step_ar = lambda k, s, a, p: _real(k, s, jnp.broadcast_to(const, a.shape), p)
    #: `SHARED` (CleanRL's PPO: 10 epochs x 32 minibatches, adam 3e-4, clip
    #: 0.2, grad-norm 0.5) with this market's horizon.  Observations are not standardised
    #: (`obs_mean=0`, `obs_std=1`), for the reason `rollout_bench.make_policy`
    #: gives: a throughput reading does not depend on the affine map.
    cfg = dataclasses.replace(HP.SHARED, n_envs=n_envs, horizon=int(horizon))
    if epochs is not None:
        #: a parity cell, not a timing claim: after 320 Adam steps a 1e-7
        #: relative change in the batch moves the update by 2.7e-2 inside ONE
        #: arm (`train_amplify.py`, 04, 2026-09-22), so the cross-arm update
        #: gate only has resolution at one epoch (floor 3e-7 there).
        cfg = dataclasses.replace(cfg, epochs=int(epochs))
    obs0, _ = reset(jax.random.PRNGKey(0), env_params)
    obs_mean = jnp.zeros(obs0.shape[-1])
    obs_std = jnp.ones(obs0.shape[-1])
    #: 04 has no solver and so no convergence flag (`make_pooled_ippo` passes
    #: None on its behalf); the other markets carry `converged`.
    conv = None if market == "04" else "converged"
    init, iterate = make_ippo((reset, None, step_ar, spec), bounds, cfg,
                              obs_mean, obs_std, convergence_key=conv)
    key = jax.random.PRNGKey(seed)
    k_init, k_iter = jax.random.split(key)
    params, tx, opt_state, env_state, env_obs = init(k_init, env_params)
    return dict(iterate=iterate, tx=tx, cfg=cfg, params=params,
                opt_state=opt_state, env_state=env_state, env_obs=env_obs,
                key=k_iter, env_params=env_params, spec=spec, point=point,
                horizon=int(horizon), step_ar=step_ar)


def _fp(env_obs_lane0, reward_lane0):
    return float(RB.fingerprint(env_obs_lane0, reward_lane0))


def _flat(tree):
    return np.concatenate([np.asarray(x, np.float64).ravel()
                           for x in jax.tree_util.tree_leaves(tree)])


def run_gpu(L, repeats):
    iterate, tx = L["iterate"], L["tx"]
    fn = jax.jit(lambda p, o, s, ob, k, ep: iterate(p, tx, o, s, ob, k, ep))
    args = (L["params"], L["opt_state"], L["env_state"], L["env_obs"], L["key"],
            L["env_params"])
    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0
    times = []
    load0, nb0, c0, w0 = os.getloadavg(), RB._neighbours(), RB._cpu_ticks(), time.perf_counter()
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t)
    cpu_s = (RB._cpu_ticks() - c0) / RB._TICK
    wall_s = time.perf_counter() - w0
    params1, _o, _s, env_obs1, _k, metrics = out
    rew = metrics["step_reward"]                               # (H, n_envs, ...)
    return dict(compile_s=compile_s, times=times, cpu_s=cpu_s, wall_s=wall_s,
                load0=load0, nb0=nb0, params1=params1,
                obs_lane0=np.asarray(env_obs1[0]),
                rew_lane0=np.asarray(rew[:, 0]),
                pg_loss=float(metrics["pg_loss"]),
                platform=jax.devices()[0].platform,
                device_kind=jax.devices()[0].device_kind)


def run_chain(L, n_iter):
    """`n_iter` iterations back to back, each starting where the last ended.

    `run_gpu` repeats ONE iteration from the same carry, which answers "how
    long is an iteration"; a training run's wall clock is a different number
    (the first call compiles, and the carry moves).  This is that number, for
    set-ups that compare against a library timed on `learn(total_timesteps)`.
    The env-step count is read off the returned reward array, not computed
    from the configuration, so the cell says how many steps it actually ran.
    """
    iterate, tx = L["iterate"], L["tx"]
    fn = jax.jit(lambda p, o, s, ob, k, ep: iterate(p, tx, o, s, ob, k, ep))
    p, o, s, ob, k = (L["params"], L["opt_state"], L["env_state"], L["env_obs"],
                      L["key"])
    per, steps = [], 0
    load0, nb0, c0, w0 = os.getloadavg(), RB._neighbours(), RB._cpu_ticks(), time.perf_counter()
    for _ in range(n_iter):
        t = time.perf_counter()
        p, o, s, ob, k, metrics = fn(p, o, s, ob, k, L["env_params"])
        jax.block_until_ready(p)
        per.append(time.perf_counter() - t)
        steps += int(np.prod(metrics["step_reward"].shape[:2]))   # (H, n_envs)
    wall_s = time.perf_counter() - w0
    cpu_s = (RB._cpu_ticks() - c0) / RB._TICK
    return dict(per_iter_s=per, total_s=wall_s, env_steps_counted=steps,
                cpu_s=cpu_s, wall_s=wall_s, load0=load0, nb0=nb0, params1=p,
                pg_loss=float(metrics["pg_loss"]),
                platform=jax.devices()[0].platform,
                device_kind=jax.devices()[0].device_kind)


def main_chain(a):
    """`--chain N`: one training run of N iterations, end to end."""
    import powermarketjax
    t0 = time.perf_counter()
    L = learner(a.market, a.n_envs, a.inject)
    setup_s = time.perf_counter() - t0
    r = run_chain(L, a.chain)
    d_params = _flat(r["params1"]) - _flat(L["params"])
    if not np.all(np.isfinite(d_params)):
        raise SystemExit("non-finite parameter update: not a cell")
    rec = {
        "market": a.market, "arm": "gpu-chain", "running_point": L["point"],
        "quantity": f"{a.chain} chained IPPO iterations, wall clock, compile included",
        "n_envs": a.n_envs, "horizon": L["horizon"], "n_iter": a.chain,
        "env_steps_counted": r["env_steps_counted"],
        "env_steps_configured": a.chain * a.n_envs * L["horizon"],
        "setup_s": setup_s, "train_s": r["total_s"],
        "first_iter_s": r["per_iter_s"][0],
        "steady_s": float(sum(r["per_iter_s"][1:])),
        "per_iter_s": r["per_iter_s"],
        "update_norm": float(np.linalg.norm(d_params)), "pg_loss": r["pg_loss"],
        "cores_used": r["cpu_s"] / r["wall_s"], "cores_available": RB._affinity(),
        "my_cores": sorted(os.sched_getaffinity(0)),
        "platform": r["platform"], "device_kind": r["device_kind"],
        "powermarketjax_file": powermarketjax.__file__,
        "x64": bool(jax.config.jax_enable_x64),
        "threads_env": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "XLA_FLAGS", "JAX_PLATFORMS",
                         "CUDA_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_MEM_FRACTION")},
        "loadavg_before": r["load0"], "loadavg_after": os.getloadavg(),
        "neighbours_before": r["nb0"], "neighbours_after": RB._neighbours(),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip(),
        "host": platform.node(),
        "stamped_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return rec


def run_comm(L, repeats):
    """Env on the host, learner on the card; the loop is the thing measured."""
    cpus = jax.devices("cpu")
    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    if not gpus:
        raise SystemExit("arm=comm needs a card; do not set JAX_PLATFORMS=cpu")
    cpu, gpu = cpus[0], gpus[0]
    iterate, tx, cfg = L["iterate"], L["tx"], L["cfg"]
    _rollout, _gae, _update = _pick(iterate, "_rollout", "_gae", "_update")
    _act, _step_envs, _apply, _norm = _pick(_rollout, "_act", "_step_envs",
                                            "_apply", "_norm")
    n_envs, H = cfg.n_envs, cfg.horizon

    act_j = jax.jit(jax.vmap(_act, in_axes=(None, 0, 0)))
    step_j = jax.jit(_step_envs)
    last_j = jax.jit(lambda p, o: _apply(p, _norm(o))[2])

    @jax.jit
    def keys_j(k):                          # `_rollout.one`'s split, one dispatch
        k, k_act, k_env = jax.random.split(k, 3)
        return k, jax.random.split(k_act, n_envs), jax.random.split(k_env, n_envs)

    @jax.jit
    def learn_j(params, opt_state, traj, last_value, k_upd):
        adv, ret = _gae(traj, last_value)
        params, opt_state, aux = _update(params, tx, opt_state, traj, adv, ret,
                                         k_upd)
        return params, opt_state, aux

    params0 = jax.device_put(L["params"], gpu)
    opt0 = jax.device_put(L["opt_state"], gpu)
    state0 = jax.device_put(L["env_state"], cpu)
    obs0 = jax.device_put(L["env_obs"], cpu)
    env_params = jax.device_put(L["env_params"], cpu)
    key0 = jax.device_put(L["key"], cpu)

    def iteration():
        key, k_roll, k_upd = jax.random.split(key0, 3)          # `iterate`'s split
        k, state, obs, buf = k_roll, state0, obs0, []
        t_r = time.perf_counter()
        for _ in range(H):
            k, act_keys, env_keys = keys_j(k)
            action, pre, logp, value = act_j(
                params0, jax.device_put(obs, gpu), jax.device_put(act_keys, gpu))
            action, pre, logp, value = jax.device_put((action, pre, logp, value), cpu)
            nobs, state, reward, _c, done, _i = step_j(env_keys, state, action,
                                                        env_params)
            buf.append(dict(obs=obs, pre=pre, logp=logp, value=value,
                            reward=reward, done=done))
            obs = nobs
        traj = jax.tree.map(lambda *x: jnp.stack(x), *buf)     # (H, n_envs, ...)
        jax.block_until_ready(traj)
        t_r = time.perf_counter() - t_r
        t_u = time.perf_counter()
        traj_g = jax.device_put(traj, gpu)                       # buffer -> card
        lv = last_j(params0, jax.device_put(obs, gpu))
        params1, _o, aux = learn_j(params0, opt0, traj_g, lv,
                                   jax.device_put(k_upd, gpu))
        jax.block_until_ready(params1)
        t_u = time.perf_counter() - t_u
        return params1, obs, traj["reward"], aux, t_r, t_u

    t0 = time.perf_counter()
    out = iteration()
    compile_s = time.perf_counter() - t0
    times, split = [], []
    load0, nb0, c0, w0 = os.getloadavg(), RB._neighbours(), RB._cpu_ticks(), time.perf_counter()
    for _ in range(repeats):
        t = time.perf_counter()
        out = iteration()
        times.append(time.perf_counter() - t)
        split.append([out[4], out[5]])
    cpu_s = (RB._cpu_ticks() - c0) / RB._TICK
    wall_s = time.perf_counter() - w0
    params1, obs1, rew, aux, _tr, _tu = out
    return dict(compile_s=compile_s, times=times, cpu_s=cpu_s, wall_s=wall_s,
                load0=load0, nb0=nb0, params1=params1,
                obs_lane0=np.asarray(obs1[0]), rew_lane0=np.asarray(rew[:, 0]),
                pg_loss=float(aux["pg_loss"]),
                rollout_update_s=split,
                platform=f"{cpu.platform}+{gpu.platform}",
                device_kind=f"{cpu.device_kind}|{gpu.device_kind}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True, choices=sorted(RB.BUILDS),
                    help="any market `rollout_bench.BUILDS` builds; the device is "
                         "market-agnostic (it calls `RB.BUILDS[market]()`).")
    ap.add_argument("--arm", required=True, choices=("gpu", "comm"))
    ap.add_argument("--n-envs", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--inject", action="store_true")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override SHARED.epochs; parity cells only")
    ap.add_argument("--chain", type=int, default=None,
                    help="gpu arm only: one run of N chained iterations, end to end")
    ap.add_argument("--tag", default="", help="suffix for a re-measurement")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if not jax.config.jax_enable_x64:
        raise SystemExit("set JAX_ENABLE_X64=1: the clearing operators need float64")
    others = RB._neighbours()
    blocking = [o for o in others if not o.startswith("unpinned:")]
    if blocking:
        raise SystemExit(f"refusing: avoidable tenants on my cores: {blocking}")
    if a.arm == "gpu" and jax.devices()[0].platform != "gpu":
        raise SystemExit("--arm gpu but the default device is not a card")
    a.out.mkdir(parents=True, exist_ok=True)
    if a.chain is not None:
        if a.arm != "gpu" or a.epochs is not None:
            raise SystemExit("--chain times the gpu arm at SHARED's epochs only")
        tag = (f"{a.market}_gpu_chain{a.chain}_n{a.n_envs}_c{RB._affinity()}"
               f"{'_inject' if a.inject else ''}{'_' + a.tag if a.tag else ''}")
        dest = a.out / f"{tag}.json"
        if dest.exists():
            print(f"skip {tag} (exists)", flush=True)
            return
        rec = main_chain(a)
        dest.write_text(json.dumps(rec, indent=1, sort_keys=True))
        print(f"{tag}  train={rec['train_s']:.1f} s  first={rec['first_iter_s']:.1f} s  "
              f"env_steps={rec['env_steps_counted']}  "
              f"cores={rec['cores_used']:.1f}/{rec['cores_available']}", flush=True)
        return
    tag = (f"{a.market}_{a.arm}_train_n{a.n_envs}_c{RB._affinity()}"
           f"{'_inject' if a.inject else ''}"
           f"{'_ep%d' % a.epochs if a.epochs is not None else ''}"
           f"{'_' + a.tag if a.tag else ''}")
    dest = a.out / f"{tag}.json"
    if dest.exists():
        print(f"skip {tag} (exists)", flush=True)
        return
    L = learner(a.market, a.n_envs, a.inject, epochs=a.epochs)
    r = (run_gpu if a.arm == "gpu" else run_comm)(L, a.repeats)
    d_params = _flat(r["params1"]) - _flat(L["params"])
    if not np.all(np.isfinite(d_params)):
        raise SystemExit("non-finite parameter update: not a cell")
    steps = a.n_envs * L["horizon"]
    med = float(np.median(r["times"]))
    spread = (max(r["times"]) - min(r["times"])) / min(r["times"])
    rec = {
        "market": a.market, "arm": a.arm, "running_point": L["point"],
        "quantity": "one IPPO iteration: rollout + GAE + update",
        "n_envs": a.n_envs, "horizon": L["horizon"],
        "env_steps_per_iteration": steps,
        "epochs": L["cfg"].epochs, "minibatches": L["cfg"].minibatches,
        "optimiser_steps_per_iteration": L["cfg"].epochs * L["cfg"].minibatches,
        "n_agents": int(L["spec"].get("n_agents", -1)),
        "n_params": int(_flat(L["params"]).size),
        "inject": a.inject, "repeats": a.repeats,
        "compile_s": r["compile_s"], "iter_s_all": r["times"],
        "iter_s_median": med, "iter_s_min": float(min(r["times"])),
        "spread_max_over_min": spread,
        "ms_per_env_step_median": 1e3 * med / steps,
        "rollout_update_s": r.get("rollout_update_s"),
        "fingerprint_lane0": repr(_fp(r["obs_lane0"], r["rew_lane0"])),
        "update_norm": float(np.linalg.norm(d_params)),
        "pg_loss": r["pg_loss"],
        "cores_used": r["cpu_s"] / r["wall_s"], "cores_available": RB._affinity(),
        "my_cores": sorted(os.sched_getaffinity(0)),
        "platform": r["platform"], "device_kind": r["device_kind"],
        "x64": bool(jax.config.jax_enable_x64),
        "threads_env": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "XLA_FLAGS", "JAX_PLATFORMS",
                         "CUDA_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_MEM_FRACTION")},
        "loadavg_before": r["load0"], "loadavg_after": os.getloadavg(),
        "neighbours_before": r["nb0"], "neighbours_after": RB._neighbours(),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip(),
        "host": platform.node(),
        "stamped_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    np.savez(a.out / f"{tag}.npz", d_params=d_params, rew_lane0=r["rew_lane0"],
             obs_lane0=r["obs_lane0"])
    dest.write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(f"{tag}  {1e3 * med:.1f} ms/iter  spread={100 * spread:.2f}%  "
          f"cores={rec['cores_used']:.1f}/{rec['cores_available']}  "
          f"fp={rec['fingerprint_lane0']}  |dp|={rec['update_norm']:.6g}",
          flush=True)


if __name__ == "__main__":
    main()
