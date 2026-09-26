"""Rollout throughput across host sizes, for the five markets and three arms.

The question this answers is **not** "is it fast".  It is "does the throughput
depend on how big the host is".  An earlier measurement on market 01 found
the all-GPU arm flat to 7e-5 across
4 / 8 / 128 logical CPUs while the host-solver arm moved 6.6x; this device
carries the same measurement to the other four markets and to two more arms.

Three arms, one timed quantity each -- a rollout of `horizon` steps over
`n_envs` environments, policy forward included, **gradient excluded**:

    gpu    jit(vmap(scan(step)))            everything on the accelerator
    cpu    the same, JAX_PLATFORMS=cpu      everything on the host
    comm   env on CPU, policy on GPU        Python loop, device_put each step

`comm` is the architecture an off-device RL loop has (SB3/SBX shape).  It runs
the *same* env code as the other two, so `gpu / comm` is a pure architecture
ratio and does not mix in "somebody else's env is slow".

Compile time is excluded from the steady reading and reported separately.
Every cell also reports a **fingerprint** -- a float64 scalar that depends on
the whole chain -- because XLA folds away computation whose result nobody
consumes, and a benchmark that folded half its work away reports a clean,
reproducible, wrong number with nothing to raise a flag.  Lane 0's fingerprint must agree across `n_envs` **to a relative tolerance**,
not bit for bit: `lu_batching` can take a different route at batch 1 and batch
64, and bitwise agreement across a change of program shape is not something
to assume.  Measured 2026-09-21 on market 02, n=1 against
n=64: relative difference 4.6e-12.  `check_cells.py` holds the threshold.

The `--inject` control is read off the **fingerprint**, not off the clock.
Replacing the policy with a constant moved market 02's fingerprint by 6.4% but
its time by only 1.1%, because the policy forward is about 1% of a step -- which
an independent measurement on the same market also found.  A timing gate on a 1% component has no resolution; the
fingerprint moving proves the action reached the environment and propagated.

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu taskset -c 0-31 \
      python tools/env_bench/rollout_bench.py --market 02 --arm cpu \
        --n-envs 64 --out cells

`JAX_ENABLE_X64=1` is not optional: the clearing operators raise without it.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------- host metering

_TICK = os.sysconf("SC_CLK_TCK")


def _cpu_ticks():
    """utime + stime of this process **and its children**, in clock ticks.

    Children matter: nothing here forks today, but a future arm that farms out
    to workers would otherwise read as using no CPU at all.
    """
    with open("/proc/self/stat", "rb") as fh:
        f = fh.read().rsplit(b")", 1)[1].split()
    # after the comm field: state is f[0]; utime/stime/cutime/cstime are 11..14
    return sum(int(f[i]) for i in (11, 12, 13, 14))


def _affinity():
    n = len(os.sched_getaffinity(0))
    return n


def _neighbours():
    """Which other cells of this device were running beside this one.

    Cells run in parallel lanes from 2026-09-22, so "how fast was this cell"
    is no longer answerable from the cell alone.  Every cell therefore records
    its own core range and every sibling's, by reading /proc rather than by
    being told: a lane script that forgot to pass its id would otherwise
    produce a cell that looks like it ran alone.
    """
    me, mypg, out = os.getpid(), os.getpgid(0), []
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        #: this call's own `timeout` and `conda run` wrappers carry
        #: `rollout_bench.py` on their command line too, and counting them
        #: would make the neighbour gate refuse every solo run.  They share
        #: this process group; another lane's cell does not, because each lane
        #: is started under `setsid`.
        try:
            if os.getpgid(int(d)) == mypg:
                continue
        except (OSError, ProcessLookupError):
            continue
        try:
            with open(f"/proc/{d}/cmdline", "rb") as fh:
                argv = fh.read().split(b"\0")
            bits = [b.decode("utf8", "replace") for b in argv if b]
            #: exactly `python .../rollout_bench.py ...`, which the wrappers are
            #: not: `timeout 7200 conda run ... python ...` and
            #: `/opt/conda/bin/python /opt/conda/condabin/conda run ...` both
            #: carry the script name further along their argv, and matching the
            #: substring counted this call's own two wrappers as neighbours.
            mine = set(os.sched_getaffinity(0))
            own = (len(bits) >= 2 and Path(bits[0]).name.startswith("python")
                   and bits[1].endswith("rollout_bench.py"))
            if not own:
                #: **Anything** busy on this line's cores contends, not only
                #: this device's own cells.  On 2026-09-22 another line pinned
                #: two `run_eval_02.py` jobs to cores 0-3 and 4-7 ten minutes
                #: before a calibration was started on 0-3; the calibration's
                #: "solo" phase therefore shared four cores with a job drawing
                #: 192% CPU, read 8x slow in both compile and rollout, and was
                #: within one step of deciding that parallel running was safe.
                #: A gate that only sees its own kind is a gate against
                #: carelessness, not against the machine.
                try:
                    theirs = set(os.sched_getaffinity(int(d)))
                    if not (theirs & mine):
                        continue
                    #: **Pinned** neighbours can be avoided by choosing another
                    #: core range, so they are a red.  **Unpinned** ones cannot
                    #: -- on this shared machine another user's `pytest` and
                    #: three of this project's own GPU jobs run with the full
                    #: mask -- so refusing on them would block every cell for
                    #: hours.  Those are recorded as a condition instead, which
                    #: is the honest split: the gate stops what the operator
                    #: controls and the record carries what they do not.
                    unpinned = len(theirs) >= os.cpu_count()
                    with open(f"/proc/{d}/stat", "rb") as fh:
                        st = fh.read().rsplit(b")", 1)[1].split()
                    busy = sum(int(st[i]) for i in (11, 12)) / _TICK
                    up = float(open("/proc/uptime").read().split()[0])
                    start = int(st[19]) / _TICK
                    if up - start < 1 or busy / (up - start) < 0.2:
                        continue        # idle or barely running: not a tenant
                except (OSError, ValueError, IndexError, ZeroDivisionError):
                    continue
                name = (f"{Path(bits[1]).name}" if len(bits) > 1
                        else Path(bits[0]).name)
                out.append(("unpinned:" if unpinned else "pinned:") + name)
                continue
            mk = bits[bits.index("--market") + 1] if "--market" in bits else "?"
            n = bits[bits.index("--n-envs") + 1] if "--n-envs" in bits else "?"
            cores = len(os.sched_getaffinity(int(d)))
            out.append(f"m{mk}/n{n}/c{cores}")
        except (OSError, ValueError, IndexError):
            continue
    return sorted(out)


# ---------------------------------------------------------------- market builds
# Each build returns (reset, step_auto_reset, env_params, spec, horizon, label).
# The running point of each is the one its headline result was produced at; the
# constants are read off that market's own driver, not invented here.

def build_01():
    from powermarketjax.case import load_case
    from powermarketjax.envs.day_ahead import (load_commitment, load_gb_demand,
                                               make_env)
    from powermarketjax.learning.adapters import unpack_env
    _driver_path()
    import run_rl_01 as D
    case = load_case("29gb")
    #: `run_rl_01.CAP_SCALE` / `RAMP_SCALE`, not the 0.4 / 0.25 that
    #: `day_ahead_step.py` still carries: that pair is the 2026-08-11 running
    #: point and today's commitment fixture refuses it outright.  **Readings
    #: taken at that older pair are therefore at a different running point
    #: and do not belong beside these cells.**
    built = make_env(case, load_commitment(n_periods=24), load_gb_demand(),
                     n_segments=1, kind="markup", markup_max=2.0,
                     cap_scale=D.CAP_SCALE, ramp_scale=D.RAMP_SCALE)
    reset, _step, step_ar, spec = unpack_env(built)
    params = built[0].make_params(episode_len=1)     # one market day per episode
    return (reset, step_ar, params, spec, 4,
            f"case29gb T24 K1 cap{D.CAP_SCALE} ramp{D.RAMP_SCALE} ep_len1")


def build_02():
    from powermarketjax.case import load_case, scale_min_output
    from powermarketjax.envs.day_ahead import demand_from_meta
    from powermarketjax.envs.real_time import load_da_position
    from powermarketjax.envs.real_time.demand import T_RT, half_hourly_from_meta
    from powermarketjax.envs.real_time.env import make_env
    from powermarketjax.learning.adapters import unpack_env
    pos = load_da_position(chain="step1prime_seasons")
    meta = pos["meta"]
    case = scale_min_output(load_case(meta["case"]),
                            float(meta.get("p_min_scale", 1.0)))
    hh, _ = half_hourly_from_meta(meta)
    fc, _a, _d = demand_from_meta(meta)
    built = make_env(case, pos, hh, fc, n_segments=1, markup_max=2.0,
                     cap_scale=0.60, ramp_scale=1.00)
    reset, _step, step_ar, spec = unpack_env(built)
    params = built[0].make_params(episode_len=T_RT)
    return (reset, step_ar, params, spec, T_RT,
            f"case29gb seasons60 cap0.6 ramp1.0 T_RT{T_RT}")


def _driver_path():
    """`tools/benchmark`, `tools/p2p_experiment`, `tools/flex_experiment` on the path.

    The three later markets build their environment inside their own driver,
    and their scenario constants live there too.  Those are imported rather
    than copied: a benchmark that restates `THETA`, `BETA` or `KAPPA` becomes a
    second declaration of the running point, and the copy goes stale silently.
    """
    import sys
    root = Path(__file__).resolve().parents[2]
    for sub in ("benchmark", "p2p_experiment", "flex_experiment"):
        d = str(root / "tools" / sub)
        if d not in sys.path:
            sys.path.insert(0, d)
    return root


def build_03():
    _driver_path()
    import run_eval_03 as E
    import run_rl_03 as D
    from powermarketjax.case import load_case, scale_min_output
    from powermarketjax.envs.ancillary.env import make_ancillary_env
    from powermarketjax.envs.real_time import load_da_position
    from powermarketjax.learning.adapters import unpack_env
    pos = load_da_position(chain="step1prime_seasons")
    meta = pos["meta"]
    case = scale_min_output(load_case(E.CASE), float(meta.get("p_min_scale", 1.0)))
    volr, pi_scale = E.volr_pi_scale(E.CASE)
    built = make_ancillary_env(case, E.THETA, volr, E.BETA, pi_scale,
                               n_segments=1, cap_scale=E.CAP_SCALE,
                               ramp_scale=E.RAMP_SCALE, period_hours=E.DELTA,
                               kind="markup", markup_max=E.MARKUP_MAX)
    reset, _step, step_ar, spec = unpack_env(built)
    params = D.build_params(pos, case, jnp)
    return (reset, step_ar, params, spec, E.T_DAY,
            f"case{E.CASE} seasons60 cap{E.CAP_SCALE} volr{volr:g} pi{pi_scale:g}")


def build_04():
    _driver_path()
    import run_rl_04 as D
    from powermarketjax.learning.adapters import unpack_env
    import preliminary_reference as R
    run = D.build_run(1200, 0.5)                 # the driver's own defaults
    reset, _step, step_ar, spec = unpack_env(run["env"])
    #: `run["n_periods"]` is the whole readable series (20 064 quarter-hours),
    #: not an episode.  The horizon is the market's episode length, which is
    #: where every other market's horizon comes from too.
    return (reset, step_ar, run["params"], spec, int(R.EPISODE_LEN),
            f"{D.CASE} agents1200 soc0.5 ep_len{R.EPISODE_LEN}")


def build_05():
    _driver_path()
    import concentration_baseline as CB
    from powermarketjax.learning.adapters import unpack_env
    env, params, _n = CB.scenario(2040, 2040, split="train")
    reset, _step, step_ar, spec = unpack_env(env)
    #: this market declares `n_agent`, the other four `n_agents`, and
    #: `adapters.unpack_env` normalises the action keys but not this one.
    #: Filled on the local copy: editing the shared module to take a
    #: measurement would change it for every other caller.
    spec = dict(spec)
    spec.setdefault("n_agents", int(spec["n_agent"]))
    return (reset, step_ar, params, spec, CB.EPISODE_LEN,
            "SwissDN 459_0 place2040 cap2040 train")


BUILDS = {"01": build_01, "02": build_02, "03": build_03,
          "04": build_04, "05": build_05}


# ---------------------------------------------------------------- the rollout

def make_policy(spec, obs_dim, reserve_columns=0, seed=0, per_agent=False):
    """The market's real IPPO shared actor, greedy (pre-squash mean -> box).

    The layout comes from `ippo.action_layout` and the box from
    `policy.bounds_for`, not from anything worked out here: this device has to
    put the *market's* action through the *market's* policy, and a shape rule
    reinvented in a benchmark is a second declaration of the action space.

    Observations are **not** standardised (`obs_mean=0`, `obs_std=1`).  A
    throughput reading does not depend on the affine map, and the frozen
    statistics a real run uses belong to that run's training window; carrying
    one in would make this cell quote a window it never measured.
    """
    from powermarketjax.learning.ippo import action_layout
    from powermarketjax.learning.policy import (SharedActorCritic, bounds_for,
                                                to_action)
    bounds = bounds_for(spec, reserve_columns=reserve_columns)
    n_agents, act_dim, act_shape, low, high = action_layout(spec, bounds)
    net = SharedActorCritic(act_dim=act_dim, hidden=(64, 64), init_scale=1.0)
    const = jnp.asarray(low).reshape(act_shape)

    if per_agent:
        #: one network PER PARTICIPANT, not one shared across them.  This is the
        #: market's own `--per-agent-params` layout (`run_rl_01.py`,
        #: `run_rl_04.py`, `concentration_baseline.py` all carry the flag), and
        #: `ippo._apply_per_agent` maps the parameter axis against the agent
        #: axis with a single `vmap`, so 1 200 networks are one call rather than
        #: 1 200.  Market 02's shared layout is 5 379 scalars and its per-unit
        #: layout 355 014 -- the two are different amounts of work, so cells
        #: carry the layout in their filename and are never compared across it.
        from powermarketjax.learning.ippo import _apply_per_agent
        keys = jax.random.split(jax.random.PRNGKey(seed), n_agents)
        pol = jax.vmap(net.init, in_axes=(0, None))(keys, jnp.zeros((1, obs_dim)))
        _ap = partial(_apply_per_agent, net)
    else:
        pol = net.init(jax.random.PRNGKey(seed), jnp.zeros((1, obs_dim)))
        _ap = net.apply

    def apply(pol_params, obs, inject=False):
        if inject:            # the control: a constant action, the net unused
            return const
        mean, _log_std, _v = _ap(pol_params, obs)
        return to_action(mean, low, high).reshape(act_shape)

    return apply, pol, act_dim


def make_rollout(step_ar, apply, env_params, horizon, inject):
    """One `lax.scan` of `horizon` steps; returns the fingerprint inputs."""
    def rollout(key, state, obs, pol_params):
        def body(carry, _):
            k, st, ob = carry
            k, k_step = jax.random.split(k)
            act = apply(pol_params, ob, inject)
            nob, nst, rew, _costs, _done, _info = step_ar(k_step, st, act,
                                                          env_params)
            return (k, nst, nob), rew
        (k, st, ob), rews = jax.lax.scan(body, (key, state, obs), None,
                                         length=horizon)
        return ob, rews
    return rollout


def fingerprint(obs, rews):
    """One float64 that every step of the chain feeds into.

    Sum alone would cancel sign errors, so each term is weighted by its index:
    a permutation of the same numbers lands somewhere else.
    """
    o = jnp.asarray(obs, jnp.float64).ravel()
    r = jnp.asarray(rews, jnp.float64).ravel()
    wo = jnp.arange(o.size, dtype=jnp.float64) + 1.0
    wr = jnp.arange(r.size, dtype=jnp.float64) + 1.0
    return jnp.sum(o * wo) + jnp.sum(r * wr)


# ---------------------------------------------------------------- one cell

def run_cell(market, n_envs, repeats, inject, seed=0, perturb_ulp=0, per_agent=False):
    reset, step_ar, env_params, spec, horizon, point = BUILDS[market]()
    keys = jax.random.split(jax.random.PRNGKey(seed), n_envs)
    obs0, state0 = jax.vmap(reset, in_axes=(0, None))(keys, env_params)
    obs_dim = int(obs0.shape[-1])
    reserve = int(spec.get("n_prod", 0)) if "reserve_low" in spec else 0
    apply, pol_params, act_dim = make_policy(spec, obs_dim, reserve, seed,
                                             per_agent=per_agent)
    if perturb_ulp:
        #: How much does THIS market amplify a last-bit change in its input?
        #: The gate on cross-`n_envs` fingerprint agreement needs a threshold,
        #: and a threshold picked after seeing the data is not a gate.  This
        #: measures one: nudge one policy weight by `perturb_ulp` ULP and read
        #: how far the fingerprint moves.  Market 03 clears by a **fixed** 60
        #: Newton trips (`ancillary/clearing.py MAX_ITER`), so it is expected
        #: to amplify far more than market 02 -- expected, and now measured
        #: rather than assumed.
        leaves, tree = jax.tree_util.tree_flatten(pol_params)
        big = max(range(len(leaves)), key=lambda i: leaves[i].size)
        flat = leaves[big].ravel()
        step = jnp.abs(flat[0]) * jnp.finfo(flat.dtype).eps * perturb_ulp
        flat = flat.at[0].add(jnp.where(step == 0, jnp.finfo(flat.dtype).eps, step))
        leaves[big] = flat.reshape(leaves[big].shape)
        pol_params = jax.tree_util.tree_unflatten(tree, leaves)
    roll = make_rollout(step_ar, apply, env_params, horizon, inject)

    def batched(keys, state, obs, pol):
        ob, rews = jax.vmap(roll, in_axes=(0, 0, 0, None))(keys, state, obs, pol)
        return ob, rews, fingerprint(ob[0], rews[0])       # lane 0 only

    fn = jax.jit(batched)

    t0 = time.perf_counter()
    out = fn(keys, state0, obs0, pol_params)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0

    load0, nb0 = os.getloadavg(), _neighbours()
    times, c0, w0 = [], _cpu_ticks(), time.perf_counter()
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn(keys, state0, obs0, pol_params)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t)
    cpu_s = (_cpu_ticks() - c0) / _TICK
    wall_s = time.perf_counter() - w0

    env_steps = n_envs * horizon
    med, mn = float(np.median(times)), float(np.min(times))
    return {
        "market": market, "running_point": point, "n_envs": n_envs,
        "horizon": horizon, "env_steps_per_rollout": env_steps,
        "obs_dim": obs_dim, "act_dim": act_dim,
        "n_agents": int(spec.get("n_agents", -1)),
        "inject": bool(inject), "repeats": repeats,
        "perturb_ulp": int(perturb_ulp), "per_agent_params": bool(per_agent),
        "compile_s": compile_s,
        "rollout_s_median": med, "rollout_s_min": mn,
        "rollout_s_all": times,
        "ms_per_env_step_median": 1e3 * med / env_steps,
        "ms_per_env_step_min": 1e3 * mn / env_steps,
        "env_steps_per_s_median": env_steps / med,
        "fingerprint_lane0": repr(float(out[2])),
        "cores_used": cpu_s / wall_s,
        "cores_available": _affinity(),
        "platform": jax.devices()[0].platform,
        "device_kind": jax.devices()[0].device_kind,
        "x64": bool(jax.config.jax_enable_x64),
        #: at 128 logical CPUs OpenBLAS prints "precompiled NUM_THREADS
        #: exceeded" and takes an auxiliary-array path, so that level is not a
        #: clean "more cores" point.  Recorded rather than pinned: setting
        #: OPENBLAS_NUM_THREADS would change what the level means.
        "threads_env": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "MKL_NUM_THREADS", "XLA_FLAGS", "JAX_PLATFORMS")},
        "threads_in_process": len(os.listdir("/proc/self/task")),
        #: Load average is reported beside every timing, because the same code on this machine has
        #: differed 1.7x between a quiet box and a busy one.  Read before and
        #: after the timed region: a cell that started quiet and ended loud is
        #: not the same measurement as one that was quiet throughout.
        "loadavg_before": load0,
        "loadavg_after": os.getloadavg(),
        "my_cores": sorted(os.sched_getaffinity(0)),
        "neighbours_before": nb0,
        "neighbours_after": _neighbours(),
    }


def run_cell_comm(market, n_envs, repeats, inject, seed=0, per_agent=False):
    """Arm C: the environment on the host, the policy on the card.

    This is the architecture an off-device RL loop has -- SB3/SBX shape -- and
    it runs the **same** environment code as the other two arms, so `gpu / comm`
    is an architecture ratio with nothing else moving.  There is no `lax.scan`
    here by construction: the loop is the thing being measured.

    The key schedule is `make_rollout`'s, step for step (`k, k_step =
    split(k)`), so lane 0's fingerprint is comparable with the scanned arms.
    Placement is by **committed** arrays rather than a per-call device
    argument, which recent JAX no longer accepts: the environment's state and
    parameters are committed to the host, the policy's parameters to the card,
    and each hop is an explicit `device_put` that the clock therefore sees.
    """
    cpus = jax.devices("cpu")
    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    if not gpus:
        raise SystemExit("arm=comm needs a card; jax.devices() shows none. Do "
                         "not set JAX_PLATFORMS=cpu for this arm.")
    cpu, gpu = cpus[0], gpus[0]

    reset, step_ar, env_params, spec, horizon, point = BUILDS[market]()
    keys0 = jax.device_put(jax.random.split(jax.random.PRNGKey(seed), n_envs), cpu)
    obs0, state0 = jax.vmap(reset, in_axes=(0, None))(keys0, env_params)
    obs_dim = int(obs0.shape[-1])
    reserve = int(spec.get("n_prod", 0)) if "reserve_low" in spec else 0
    apply, pol_params, act_dim = make_policy(spec, obs_dim, reserve, seed,
                                             per_agent=per_agent)

    obs0 = jax.device_put(obs0, cpu)
    state0 = jax.device_put(state0, cpu)
    env_params = jax.device_put(env_params, cpu)
    pol = jax.device_put(pol_params, gpu)

    step_j = jax.jit(jax.vmap(step_ar, in_axes=(0, 0, 0, None)))
    pol_j = jax.jit(jax.vmap(lambda p, o: apply(p, o, inject), in_axes=(None, 0)))
    split_j = jax.jit(jax.vmap(jax.random.split))

    def rollout():
        keys, state, obs, rews = keys0, state0, obs0, []
        for _ in range(horizon):
            obs_g = jax.device_put(obs, gpu)          # host -> card
            act_g = pol_j(pol, obs_g)
            act = jax.device_put(act_g, cpu)          # card -> host
            two = split_j(keys)
            keys, k_step = two[:, 0], two[:, 1]
            obs, state, rew, _c, _d, _i = step_j(k_step, state, act, env_params)
            rews.append(rew)
        r = jnp.stack(rews)                           # (horizon, n_envs, ...)
        return obs, jnp.moveaxis(r, 1, 0)             # -> (n_envs, horizon, ...)

    t0 = time.perf_counter()
    ob, rr = rollout()
    jax.block_until_ready((ob, rr))
    compile_s = time.perf_counter() - t0

    load0, nb0 = os.getloadavg(), _neighbours()
    times, c0, w0 = [], _cpu_ticks(), time.perf_counter()
    for _ in range(repeats):
        t = time.perf_counter()
        ob, rr = rollout()
        jax.block_until_ready((ob, rr))
        times.append(time.perf_counter() - t)
    cpu_s = (_cpu_ticks() - c0) / _TICK
    wall_s = time.perf_counter() - w0

    fp = float(fingerprint(ob[0], rr[0]))
    env_steps = n_envs * horizon
    med, mn = float(np.median(times)), float(np.min(times))
    return {
        "market": market, "running_point": point, "n_envs": n_envs,
        "horizon": horizon, "env_steps_per_rollout": env_steps,
        "obs_dim": obs_dim, "act_dim": act_dim,
        "n_agents": int(spec.get("n_agents", -1)),
        "inject": bool(inject), "repeats": repeats, "compile_s": compile_s,
        "rollout_s_median": med, "rollout_s_min": mn, "rollout_s_all": times,
        "ms_per_env_step_median": 1e3 * med / env_steps,
        "ms_per_env_step_min": 1e3 * mn / env_steps,
        "env_steps_per_s_median": env_steps / med,
        "fingerprint_lane0": repr(fp),
        "cores_used": cpu_s / wall_s, "cores_available": _affinity(),
        "platform": f"{cpu.platform}+{gpu.platform}",
        "device_kind": f"{cpu.device_kind}|{gpu.device_kind}",
        "x64": bool(jax.config.jax_enable_x64),
        "round_trips_per_rollout": 2 * horizon,
        "threads_env": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "MKL_NUM_THREADS", "XLA_FLAGS", "JAX_PLATFORMS")},
        "threads_in_process": len(os.listdir("/proc/self/task")),
        "loadavg_before": load0, "loadavg_after": os.getloadavg(),
        "my_cores": sorted(os.sched_getaffinity(0)),
        "neighbours_before": nb0,
        "neighbours_after": _neighbours(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True, choices=sorted(BUILDS))
    ap.add_argument("--arm", default="cpu", choices=("gpu", "cpu", "comm"))
    ap.add_argument("--n-envs", type=int, nargs="+", default=[64])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--inject", action="store_true",
                    help="the control: replace the policy with a constant. The "
                         "fingerprint must change and the cell must get faster; "
                         "if neither moves, the timed region never covered it.")
    ap.add_argument("--perturb-ulp", type=int, default=0,
                    help="nudge one policy weight by this many ULP and read how "
                         "far the fingerprint moves. Calibrates the cross-n_envs "
                         "gate threshold per market instead of inventing one.")
    ap.add_argument("--per-agent-params", action="store_true",
                    help="one network per participant instead of one shared. "
                         "The market's own layout; 04 has 1 200 participants.")
    ap.add_argument("--allow-neighbours", action="store_true",
                    help="time this cell even though other cells are running. "
                         "Off by default: see the gate below for the 1.71x.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if not jax.config.jax_enable_x64:
        raise SystemExit("set JAX_ENABLE_X64=1: the clearing operators need float64")
    #: A cell measured beside other cells is not comparable with one measured
    #: alone, and on 2026-09-22 that was not a worry but a measurement: market
    #: 02 at 8 cores read 30.695 ms/env-step alone and 52.400 beside four
    #: others -- 1.71x -- while the quantity the device exists to measure, the
    #: spread across core counts, is 1.13x to 1.54x.  **The contamination was
    #: larger than the signal.**  A pairing penalty of +1.7% had been measured
    #: beforehand and did not protect against it, because it was measured on
    #: market 05, the lightest of the five.  So this is a gate rather than a
    #: note: the run refuses, and `--allow-neighbours` has to be typed.
    others = _neighbours()
    blocking = [o for o in others if not o.startswith("unpinned:")]
    if blocking and not args.allow_neighbours:
        raise SystemExit(
            f"refusing to time a cell beside {len(blocking)} avoidable "
            f"tenant(s) on its cores: {sorted(set(blocking))}. "
            f"(unavoidable, recorded not refused: {sorted(set(others) - set(blocking))}) Timings taken beside neighbours are not "
            f"comparable with timings taken alone (measured 1.71x on market 02 "
            f"at 8 cores). Wait, or pass --allow-neighbours and accept that the "
            f"cell records `neighbours_before` and must not be compared across "
            f"different neighbour counts.")
    args.out.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    for n in args.n_envs:
        #: the thread count is part of the cell's identity, not a footnote:
        #: market 02 at n=64 on 32 cores reads 85.150 ms/env-step untuned and
        #: 17.629 with OMP_NUM_THREADS=8, so a tuned run landing on an untuned
        #: run's filename would overwrite a 4.83x different number in place.
        omp = os.environ.get("OMP_NUM_THREADS")
        tag = (f"{args.market}_{args.arm}_n{n}_c{_affinity()}"
               f"{'_omp' + omp if omp else ''}"
               f"{'_inject' if args.inject else ''}"
               f"{'_ulp%d' % args.perturb_ulp if args.perturb_ulp else ''}"
               f"{'_pa' if args.per_agent_params else ''}")
        dest = args.out / f"{tag}.json"
        if dest.exists():                       # resume: one cell, one file
            print(f"skip {tag} (exists)", flush=True)
            continue
        rec = (run_cell_comm(args.market, n, args.repeats, args.inject,
                             per_agent=args.per_agent_params)
               if args.arm == "comm" else
               run_cell(args.market, n, args.repeats, args.inject,
                        perturb_ulp=args.perturb_ulp,
                        per_agent=args.per_agent_params))
        #: the arm is a label until something checks it.  A `--arm gpu` cell
        #: that silently ran on the host would be a clean, wrong number with
        #: nothing to raise a flag, and JAX_PLATFORMS is one env var away.
        want = {"gpu": "gpu", "cpu": "cpu"}.get(args.arm)
        if want and rec["platform"] != want:
            raise SystemExit(f"--arm {args.arm} but jax.devices()[0].platform "
                             f"is {rec['platform']!r}: refusing to stamp this "
                             f"cell with an arm it did not run on")
        rec.update(arm=args.arm, commit=head, host=platform.node(),
                   stamped_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        dest.write_text(json.dumps(rec, indent=1, sort_keys=True))
        print(f"{tag}  {rec['ms_per_env_step_median']:.3f} ms/env-step  "
              f"cores={rec['cores_used']:.1f}/{rec['cores_available']}  "
              f"fp={rec['fingerprint_lane0']}", flush=True)


if __name__ == "__main__":
    main()
