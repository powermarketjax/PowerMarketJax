"""Market 02 trained by stable-baselines3's PPO at its out-of-the-box defaults,
timed end to end on a fixed env-step budget.  The counterpart of
`train_bench.py --chain`.

The question is the reviewer's: "why not just use stable-baselines3?".  The arm
is SB3 **as a user gets it**: `PPO("MlpPolicy", env)` and `learn(budget)` --
no `n_steps`, no `SubprocVecEnv`, no `device`.  A tuned SB3 is a different arm
and is not measured here; nothing this file prints is a claim about it.

The environment is this repository's own 02 (`rollout_bench.build_02`, the
same running point the gpu arm trains on), wrapped as a single-agent
`gymnasium.Env`:

    observation   (66, 16) -> (1056,)    all participants' observations, flat
    action        (66,)                  one markup per participant, in [1, 2]
    reward        sum over participants  one scalar, since SB3 is single-agent

The flattening changes the policy: IPPO shares one small network across 66
agents, each seeing its own 16 numbers; SB3 learns one network from 1056 inputs
to 66 outputs on the summed reward.  What they learn is not comparable and is
not compared; only the wall clock is.

The env step is the market's jitted `step_auto_reset` on the host CPU with one
environment -- the per-step work is the gpu arm's per-step work, without the
batch.  `truncated` is raised at `done`, because 02's `termination` is
"truncation" (`ippo._BOOTSTRAP_AT_DONE`).

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=2 taskset -c 32-39 \\
      python tools/env_bench/sb3_bench.py --budget 61440 --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rollout_bench as RB                                     # noqa: E402


class RealTime02(gym.Env):
    """02 as one gymnasium env: 66 participants flattened into one agent."""

    def __init__(self, seed=0):
        reset, step_ar, params, spec, horizon, point = RB.build_02()
        self.point, self.horizon = point, int(horizon)
        obs0, _ = reset(jax.random.PRNGKey(0), params)
        self._obs_shape = tuple(obs0.shape)
        act_shape = tuple(int(d) for d in spec["action_shape"])
        low = np.broadcast_to(np.asarray(spec["action_low"], np.float32), act_shape)
        high = np.broadcast_to(np.asarray(spec["action_high"], np.float32), act_shape)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (int(np.prod(obs0.shape)),), np.float32)
        self.action_space = gym.spaces.Box(low.ravel(), high.ravel(), dtype=np.float32)
        self._act_shape = act_shape

        @jax.jit
        def _reset(k):
            k, sub = jax.random.split(k)
            obs, state = reset(sub, params)
            return k, obs, state

        @jax.jit
        def _step(k, state, action):
            k, sub = jax.random.split(k)
            obs, state, reward, _c, done, _i = step_ar(sub, state, action, params)
            return k, obs, state, jnp.sum(reward), done

        self._reset_j, self._step_j = _reset, _step
        self._key = jax.random.PRNGKey(seed)
        self._state = None
        self.env_steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._key = jax.random.PRNGKey(seed)
        self._key, obs, self._state = self._reset_j(self._key)
        return np.asarray(obs, np.float32).ravel(), {}

    def step(self, action):
        a = jnp.asarray(np.asarray(action, np.float32).reshape(self._act_shape))
        self._key, obs, self._state, r, done = self._step_j(self._key, self._state, a)
        self.env_steps += 1
        return (np.asarray(obs, np.float32).ravel(), float(r), False, bool(done), {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, required=True, help="env-steps")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if not jax.config.jax_enable_x64:
        raise SystemExit("set JAX_ENABLE_X64=1: the clearing operators need float64")
    if jax.devices()[0].platform != "cpu":
        raise SystemExit("the env runs on the host: set JAX_PLATFORMS=cpu")
    blocking = [o for o in RB._neighbours() if not o.startswith("unpinned:")]
    if blocking:
        raise SystemExit(f"refusing: avoidable tenants on my cores: {blocking}")
    a.out.mkdir(parents=True, exist_ok=True)
    tag = f"02_sb3default_b{a.budget}_c{RB._affinity()}{'_' + a.tag if a.tag else ''}"
    dest = a.out / f"{tag}.json"
    if dest.exists():
        print(f"skip {tag} (exists)", flush=True)
        return

    import powermarketjax
    import stable_baselines3
    import torch
    from stable_baselines3 import PPO

    t0 = time.perf_counter()
    env = RealTime02()
    build_s = time.perf_counter() - t0

    load0, nb0, c0 = os.getloadavg(), RB._neighbours(), RB._cpu_ticks()
    t1 = time.perf_counter()
    model = PPO("MlpPolicy", env)                    # the out-of-the-box defaults
    t2 = time.perf_counter()
    model.learn(total_timesteps=a.budget)
    t3 = time.perf_counter()
    cpu_s = (RB._cpu_ticks() - c0) / RB._TICK

    if env.env_steps != model.num_timesteps:
        raise SystemExit(f"env counted {env.env_steps} steps, SB3 {model.num_timesteps}")
    rec = {
        "market": "02", "arm": "sb3-default", "running_point": env.point,
        "quantity": "PPO('MlpPolicy', env) + learn(budget), wall clock",
        "budget": a.budget, "env_steps_counted": env.env_steps,
        "sb3_num_timesteps": int(model.num_timesteps),
        "build_s": build_s, "setup_s": t2 - t1, "train_s": t3 - t2,
        "setup_plus_train_s": t3 - t1,
        "sb3_config": {"n_steps": model.n_steps, "batch_size": model.batch_size,
                       "n_epochs": model.n_epochs, "n_envs": model.n_envs,
                       "learning_rate": float(model.learning_rate),
                       "device": str(model.device),
                       "vec_env": type(model.env).__name__,
                       "net_arch": str(model.policy.net_arch),
                       "activation": model.policy.activation_fn.__name__,
                       "n_params": int(sum(p.numel() for p in model.policy.parameters()))},
        "obs_dim": int(env.observation_space.shape[0]),
        "act_dim": int(env.action_space.shape[0]),
        "versions": {"stable_baselines3": stable_baselines3.__version__,
                     "torch": torch.__version__, "gymnasium": gym.__version__,
                     "jax": jax.__version__},
        "torch_threads": torch.get_num_threads(),
        "powermarketjax_file": powermarketjax.__file__,
        "cores_used": cpu_s / (t3 - t1), "cores_available": RB._affinity(),
        "my_cores": sorted(os.sched_getaffinity(0)),
        "threads_env": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "XLA_FLAGS", "JAX_PLATFORMS",
                         "CUDA_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_MEM_FRACTION")},
        "loadavg_before": load0, "loadavg_after": os.getloadavg(),
        "neighbours_before": nb0, "neighbours_after": RB._neighbours(),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip(),
        "host": platform.node(),
        "stamped_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    dest.write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(f"{tag}  train={rec['train_s']:.1f} s  setup={rec['setup_s']:.1f} s  "
          f"env_steps={env.env_steps}  device={rec['sb3_config']['device']}  "
          f"cores={rec['cores_used']:.1f}/{rec['cores_available']}", flush=True)


if __name__ == "__main__":
    main()
