"""Market 04 P2P: the in-package learner, driven the way the out-of-package one is.

Market 04's publishable scores have to come from `powermarketjax/wrappers/`;
this is the driver that gets them there.  Its completion criterion is an
item-by-item comparison
against `tools/p2p_experiment/`, so every choice below that could move a number
is either taken from that experiment or recorded as a difference.

**The scenario is not restated here.**  `constrained_baseline.build` is called
for its `params`, which carries the panel, the battery bundle, the degradation
coefficients, the learner mask and the episode length; the tariff pair and the
period length come from `preliminary_reference`.  Nothing in this file spells a
scenario constant, so the two paths cannot drift apart by one being edited.  The
two environments `build` also returns are NOT used: they carry that experiment's
own observation scaling, which here is applied inside the learner instead (see
below), and applying it twice is exactly the failure that would look ordinary in
every printed number.

**Three divergences between this market and the shared learner are handled, and
the meta line records which disposition each one got** -- in the product, not in
a comment, because a reader holding only the `.jsonl` has no comment:

    convergence_key       null.  04 publishes no convergence flag: no solver, no
                          tolerance, no iteration bound.  `metrics` therefore
                          carries neither `step_converged` nor
                          `unconverged_frac`, and this driver ASSERTS their
                          absence out of the returned metrics rather than
                          trusting the flag it passed.
    observation scaling   `constrained_baseline.observation_scale` divisors, fed
                          as `obs_std` with `obs_mean = 0`, which makes the
                          learner's `_norm` the same division the out-of-package
                          runs used.  Not `wrappers.p2p.baseline_observation_
                          statistics`: under the truthful action the battery
                          never moves, so `soc` has zero variance there and gets
                          floored to a divisor of 1.0 -- the coordinate a
                          battery policy is most about, calibrated on a run
                          where it was constant.
    episode pool          `horizon == episode_len`, enforced by
                          `make_pooled_ippo` at construction.  The guard is
                          EXERCISED at start-up here, not merely relied on: the
                          driver builds a second learner one step longer and
                          requires it to raise, and records that it did.

**One difference from the out-of-package pipeline is deliberate and is not a
free choice.**  There, `restrict_starts` is applied after `scale_observations`
and recomputes the reset observation from the raw `spec["get_obs"]`, so the
first observation of every episode reaches the network unscaled while the other
95 are scaled (measured 2026-08-27: the reset observation is bit-identical to
the raw one, and the per-channel ratio against the next step reproduces the
divisor vector).  Here the standardisation happens inside `_norm`, which every
observation the network sees goes through, so there is no such step.  It is a
difference between the two learning arms and belongs on that list; it is not
reproduced here, because reproducing a defect to match a number is how the
number stops meaning anything.

**x64 stays off.**  `envs/p2p/clearing.py` writes float32 at every entry point
and the out-of-package numbers were produced with x64 off; turning it on would
change the market rather than record it.  `runtime_stamp(requires_x64=False)`
carries that assertion, and the meta line repeats it as `x64`.

    JAX_PLATFORMS=cpu python -m run_rl_04 --agents 16 --iterations 3 \\
        --n-envs 4 --eval-episodes 64 --out-dir r1_04_smoke \\
        --curve-out r1_04_smoke/curve.npz
"""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
#: the scenario's one definition lives there, and this driver consumes it rather
#: than restating it
sys.path.insert(0, str(REPO / "tools" / "p2p_experiment"))

import jax                                                        # noqa: E402
import jax.numpy as jnp                                           # noqa: E402

from curve_jsonl import CurveLog, repo_relative                   # noqa: E402
from evaluation import _START_DIRTY, commit_hash, runtime_stamp   # noqa: E402
from hyperparams import PROVENANCE, SAC_SHARED, SHARED            # noqa: E402
from run_rl_01 import write_params                                # noqa: E402

import constrained_baseline as CB                                 # noqa: E402
import preliminary_reference as R                                 # noqa: E402

from powermarketjax.envs.p2p import make_p2p_env                  # noqa: E402
from powermarketjax.learning.ippo import make_greedy_action       # noqa: E402
from powermarketjax.learning.sac import make_sac_greedy_action   # noqa: E402
from powermarketjax.learning.policy import bounds_for             # noqa: E402
import optax                                                      # noqa: E402

from powermarketjax.wrappers.p2p import (make_pooled_ippo,        # noqa: E402
                                         make_pooled_sac,
                                         restrict_starts,
                                         unscaled_first_obs)

MARKET = "04 p2p local energy"
CASE = "fluvius-2024-quarter-hourly"


def build_run(agents, initial_soc):
    """Scenario, environment and the two start pools, all from the experiment.

    `build` is called for `params` alone.  Its two environments carry that
    experiment's observation scaling, which this driver applies inside the
    learner instead; taking them as well would scale twice.
    """
    params, _train_env, _eval_env = CB.build(agents, initial_soc)
    env = make_p2p_env(agents, R.PI_EXP, R.PI_RET, R.DELTA)
    n_periods = int(params.p_pv.shape[0])
    train_starts, eval_starts = CB.make_start_pools(n_periods, R.EPISODE_LEN)
    scale = CB.observation_scale(params, agents)
    return dict(params=params, env=env, n_periods=n_periods,
                train_starts=train_starts, eval_starts=eval_starts, scale=scale)


def episode_returns(env, env_params, act, key, episode_len):
    """One episode's per-period reward, constraint channel and degradation cost.

    Three arrays, each `(T, n_agents)`.

    The key schedule is `preliminary_reference.make_rollout`'s, step for step --
    one split before the reset and a three-way split per step, with the policy
    half drawn even by arms that ignore it.  That is not decoration: the two
    paths must score the SAME episodes for the paired comparison this run exists
    for, and the episodes are whatever those keys select.
    """
    reset, _step, step_auto_reset, _spec = env
    key, sub = jax.random.split(key)
    obs, state = reset(sub, env_params)

    def body(carry, _):
        obs, state, key = carry
        key, a_key, s_key = jax.random.split(key, 3)
        action = act(obs, a_key)
        nxt, new_state, reward, costs, _done, info = step_auto_reset(
            s_key, state, action, env_params)
        # Battery degradation is this market's whole system cost: there is no
        # generation and no shedding here, and the other two terms of `cost` are
        # transfers.  It is carried out of the scan rather than reduced inside
        # it for the reason the market's own note gives -- the matrix asks for a
        # displacement in system cost, and a scalar mean cannot be re-read per
        # household or per period afterwards.
        return (nxt, new_state, key), (reward, costs[:, 0],
                                       info["degradation_cost"])

    (_o, _s, _k), (reward, cost, deg) = jax.lax.scan(
        body, (obs, state, key), None, length=episode_len)
    return reward, cost, deg


def evaluate(env, env_params, act, key, episodes, chunk, episode_len):
    """Per-episode, per-household return and constraint total.

    Chunked exactly as `preliminary_reference.evaluate` chunks, including the
    `fold_in(key, start)` that derives each chunk's keys: it was measured that
    reading 64 episodes as four chunks of 16 selects a DIFFERENT 64 episodes, so
    the chunk size is part of the episode set and not a memory knob.
    """
    one = jax.jit(jax.vmap(
        lambda k: episode_returns(env, env_params, act, k, episode_len)))
    rewards, costs, degs = [], [], []
    for start in range(0, episodes, chunk):
        take = min(chunk, episodes - start)
        keys = jax.random.split(jax.random.fold_in(key, start), take)
        reward, cost, deg = one(keys)
        # sum over the period axis, leaving (episodes, n_agents)
        rewards.append(np.asarray(reward.sum(1), np.float64))
        costs.append(np.asarray(cost.sum(1), np.float64))
        degs.append(np.asarray(deg.sum(1), np.float64))
    return (np.concatenate(rewards), np.concatenate(costs),
            np.concatenate(degs))


def device_memory():
    """Peak and pool bytes as the device reports them, or `(None, None)`.

    Read off the device and not off `nvidia-smi`, for the reason `run_rl_01`
    records at its own call site: under the default preallocation `nvidia-smi`
    shows the POOL, which is the fraction this process was told to reserve, so
    the number it prints is the one that was set rather than the one the run
    needed.  Both are returned because the decision they feed -- how many seeds
    fit on one card -- needs the second to read the first.

    **Which axis the peak scales with is not established for this market.**
    Market 01 measured its own memory to scale with `n_envs` alone, because its
    dominant buffers are the per-period Newton systems; 04 has no solver and its
    trajectory is `(horizon, n_envs, n_agents, obs_dim)`, which scales with the
    product.  So the per-environment figure below is printed beside the batch
    shape that produced it and is not offered as a rate to extrapolate from.
    """
    try:
        st = jax.local_devices()[0].memory_stats() or {}
    except Exception:
        return None, None
    return st.get("peak_bytes_in_use"), st.get("bytes_limit")


def report_memory(where, cfg):
    """Print the two figures and return them, or say plainly that there are none."""
    peak, pool = device_memory()
    if peak is None:
        print(f"{where}: the device reports no memory statistics on this "
              f"backend, so peak memory is unmeasured here rather than zero",
              flush=True)
        return None, None
    pool_txt = ("unreported" if pool is None
                else f"{pool / 2**30:.2f} GiB")
    print(f"{where}: peak device memory {peak / 2**30:.2f} GiB "
          f"({peak / 2**20 / max(cfg.n_envs, 1):.1f} MiB per environment at "
          f"n_envs={cfg.n_envs} x horizon={cfg.horizon}; the axis it scales "
          f"with is not established for this market), pool {pool_txt}",
          flush=True)
    return peak, pool


#: The out-of-package output-head scale, read off `preliminary_reference`'s
#: `init_policy` rather than recalled: it multiplies the Glorot draw of both the
#: mean head (`wm`) and the value head (`wv`) there, while `SharedActorCritic`
#: scales only the mean head and leaves the value head at 1.0.  So aligning it
#: takes a config field for one head and a rescale of the initialised kernel for
#: the other.  What this does NOT align is the initialiser family; that is named
#: in the flag's help and in the product rather than left for a reader to find.
EXTERNAL_OUTPUT_SCALE = 0.01

#: The out-of-package settings the calibration **adopted**, which are NOT the
#: constants sitting in `constrained_baseline.py`.  That file's own docstring
#: says so in as many words: "Every constant below is therefore a placeholder
#: the calibration overrides, and a run that leaves them at these values is not
#: a run any reported number came from", and it names the cost of the one that
#: matters -- the shared 1e-4 step size "is fifty times the optimum of the
#: unconstrained arm ... the two arms lose 0.46 and 0.18 of return at it".  The
#: adopted values: the 0.5-dock **unconstrained** arm, which is the one the -0.0260
#: paired difference comes from, was calibrated at a step size of 2e-6 (grid
#: interior, three cells), and the exploration width was held at -5.0 for the
#: whole of every calibrated run rather than annealed to anything.
#:
#: **An earlier run got this wrong once.**  Its first six alignment arms read
#: `CB.LEARNING_RATE` and `CB.LOG_STD_START/END` straight out of the source
#: file, i.e. the placeholders, so those arms aligned to 1e-4 annealed and to a
#: -0.5 -> -3.0 schedule and not to anything the out-of-package results were
#: produced at.  The two overrides below exist so that run stays reproducible;
#: the defaults are the adopted values.
EXTERNAL_ADOPTED_LR = 2e-6
EXTERNAL_ADOPTED_LOG_STD = -5.0


def scale_value_head(policy, factor):
    """Multiply the critic output kernel of `SharedActorCritic` by `factor`.

    The head is found by shape and not by name: it is the only `Dense` kernel
    whose trailing axis is 1.  A rename inside the module would make this raise
    instead of silently scaling the wrong layer, which is the failure this
    market has already paid for once elsewhere -- an arm that ran, produced a
    plausible number and had not applied the treatment it was named for.
    """
    tree = dict(policy)
    inner = dict(tree["params"])
    hits = [k for k, v in inner.items()
            if isinstance(v, dict) and "kernel" in v
            and jnp.shape(v["kernel"])[-1] == 1]
    if len(hits) != 1:
        raise SystemExit(
            f"expected exactly one Dense kernel with a trailing axis of 1 (the "
            f"value head) and found {hits}; --align-external-init would "
            f"otherwise scale the wrong layer or none")
    layer = dict(inner[hits[0]])
    layer["kernel"] = layer["kernel"] * factor
    inner[hits[0]] = layer
    tree["params"] = inner
    return tree, hits[0]


def set_log_std(policy, value):
    """Overwrite the module's `log_std` parameter with a constant.

    `constrained_baseline.run` does exactly this to its own policy dict before
    every update, which is what makes the exploration width a schedule there
    rather than a learned quantity.  The parameter is at the top of `params`
    because `SharedActorCritic` declares it with `self.param` outside any
    submodule; a missing key raises rather than being created, so a rename
    cannot turn this into a silent no-op.
    """
    tree = dict(policy)
    inner = dict(tree["params"])
    if "log_std" not in inner:
        raise SystemExit(
            f"policy params carry no 'log_std' leaf; the keys are "
            f"{sorted(inner)} and --align-external-log-std would have no effect")
    inner["log_std"] = jnp.full_like(inner["log_std"], value)
    tree["params"] = inner
    return tree

def redraw_init_family(policy, key, hidden, act_dim, output_scale):
    """Redraw every Dense kernel from the out-of-package initialiser family.

    Out of package every weight matrix is `N(0, 1) * sqrt(2 / fan_in)`, with a
    further factor of 0.01 on the two output heads
    (`preliminary_reference.init_policy`).  In package they come from
    `nn.initializers.orthogonal`: gain `sqrt(2)` on the hidden layers,
    `init_scale` on the mean head, 1.0 on the value head.
    `--align-external-init` matches the head SCALE and says in as many words
    that it does not match the family; this matches the family, on every layer.

    Layers are found by shape and never by name, for the reason
    `scale_value_head` records: a rename must raise rather than quietly treat a
    hidden layer as a head.  Biases are zero on both sides, and that is
    asserted here rather than assumed -- aligning half of an initialisation and
    calling it aligned is the failure this market has already paid for once.
    """
    hidden = tuple(int(h) for h in hidden)
    if act_dim == 1 or act_dim in hidden:
        raise SystemExit(
            f"the heads cannot be told from the hidden layers by shape: "
            f"act_dim={act_dim}, hidden={hidden}; --align-external-init-family "
            f"would have to guess which kernel is which")
    tree = dict(policy)
    inner = dict(tree["params"])
    dense = {k: v for k, v in inner.items()
             if isinstance(v, dict) and "kernel" in v}
    if len(dense) != len(hidden) + 2:
        raise SystemExit(
            f"expected {len(hidden) + 2} Dense kernels (hidden {hidden} plus a "
            f"mean head and a value head) and found {sorted(dense)}")
    value_head = [k for k, v in dense.items()
                  if jnp.shape(v["kernel"])[-1] == 1]
    mean_head = [k for k, v in dense.items()
                 if jnp.shape(v["kernel"])[-1] == act_dim]
    if len(value_head) != 1 or len(mean_head) != 1:
        raise SystemExit(
            f"expected exactly one kernel with a trailing axis of 1 and one "
            f"with a trailing axis of {act_dim}; found {value_head} and "
            f"{mean_head}, so this flag would redraw the wrong layer or none")
    heads = {value_head[0], mean_head[0]}
    subkeys = dict(zip(sorted(dense), jax.random.split(key, len(dense))))
    detail = {}
    for name in sorted(dense):
        layer = dict(inner[name])
        kernel = layer["kernel"]
        shape = tuple(jnp.shape(kernel))
        if "bias" in layer and not bool(jnp.all(layer["bias"] == 0)):
            raise SystemExit(
                f"{name} carries a non-zero bias at initialisation while out "
                f"of package every bias starts at zero; this flag would be "
                f"aligning only half of the initialisation")
        scale = float(np.sqrt(2.0 / shape[0]))
        if name in heads:
            scale *= float(output_scale)
        redrawn = jax.random.normal(subkeys[name], shape, jnp.float32) * scale
        detail[name] = {"shape": list(shape), "is_head": name in heads,
                        "scale": scale,
                        "kernel_sd_before": float(jnp.std(kernel)),
                        "kernel_sd_after": float(jnp.std(redrawn))}
        layer["kernel"] = redrawn
        inner[name] = layer
    tree["params"] = inner
    return tree, detail


def horizon_guard_bites(env, allowed, bounds, cfg, obs_mean, obs_std,
                        episode_len):
    """Build a learner one step too long and report whether it was refused.

    Recorded in the product as a fact about THIS process.  "The code contains a
    check" is not the same claim as "the check ran and refused", and only the
    second one is worth anything to a reader holding the curve.
    """
    try:
        make_pooled_ippo(env, allowed, bounds,
                         dataclasses.replace(cfg, horizon=episode_len + 1),
                         obs_mean, obs_std, episode_len=episode_len)
    except ValueError:
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--agents", type=int, default=1200)
    ap.add_argument("--initial-soc", type=float, default=0.5)
    ap.add_argument("--iterations", type=int, default=400,
                    help="0 is the untrained control: no update runs and the "
                         "initial policy is scored, which is the same arm "
                         "`constrained_baseline --iterations 0` reports")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=SHARED.n_envs,
                    help="episodes per update; off SHARED is recorded in the "
                         "meta line rather than left for a reader to notice")
    #: One network per household instead of one shared by all of them.  Same
    #: flag name and same meaning as `run_rl_01.py --per-agent-params` and as
    #: `constrained_baseline.py --per-agent-params`.
    #:
    #: **This path is the CROSS-CHECK and not the path the published 04
    #: learning numbers came from.**  Those come from the out-of-package driver
    #: so the layout ablation is run there and repeated here; a contrast whose
    #: two arms are two different programs is a contrast in the program as well
    #: as in the layout.
    ap.add_argument("--algo", choices=("ippo", "sac"), default="ippo",
                    help="learner: ippo (default, the path every 04 archive was "
                         "produced on) or sac (off-policy; wired 2026-09-20 so the "
                         "SAC columns of this market store a policy). Stamped as "
                         "`algo` in every product.")
    ap.add_argument("--sac-buffer-size", type=int, default=None,
                    help="SACConfig.buffer_size override (default SAC_SHARED's "
                         "32768; the 04 SAC columns of 2026-09-09 ran "
                         "constrained_baseline.py at 8192). Recorded in "
                         "off_shared. Refused with --algo ippo")
    ap.add_argument("--sac-batch-size", type=int, default=None,
                    help="SACConfig.batch_size override; see --sac-buffer-size")
    ap.add_argument("--sac-utd-ratio", type=float, default=None,
                    help="SACConfig.utd_ratio override; see --sac-buffer-size")
    ap.add_argument("--sac-hidden", type=int, nargs="+", default=None,
                    help="SACConfig.hidden override, e.g. 64 64 (default "
                         "SAC_SHARED's 256 256). Added 2026-09-20 for the "
                         "per-agent SAC column at width 64 (the 256 column OOMs "
                         "at a 0.45 share); recorded in off_shared, and a "
                         "run under it is not at the width the other SAC "
                         "readings use. Refused with --algo ippo")
    ap.add_argument("--per-agent-params", action="store_true",
                    help="give every household its own copy of the policy and "
                         "value network. Archives written under this flag are "
                         "NOT interchangeable with shared ones: same leaf "
                         "count, differing only in a leading axis. This is the "
                         "cross-check path; the published numbers come from "
                         "`constrained_baseline --per-agent-params`")
    #: Same flag as `run_rl_01.py --env-chunks`, accepted here so the four
    #: drivers stamp the same key, and **refused at any value but 1**: this
    #: driver builds its learner through `wrappers.p2p.make_pooled_ippo`, which
    #: does not take the keyword, so a value here would be accepted and dropped.
    #: Market 04 has no `case813nem` and no clearing to chunk; the
    #: pass-through is a one-keyword addition to `wrappers/p2p.py` if a run ever
    #: needs it, and it is not made here because that file is outside this
    #: flag's ticket.
    ap.add_argument("--env-chunks", type=int, default=1,
                    help="step the n_envs environments in this many sequential "
                         "pieces (lax.map over vmap) to cap peak device memory; "
                         "1 = the single vmap every archive was produced on. "
                         "Must divide n_envs. Not a hyperparameter: the batch "
                         "is unchanged, only how it sits on the device")
    ap.add_argument("--curve-out", default="")
    ap.add_argument("--checkpoint-every", type=int, default=0)
    ap.add_argument("--eval-episodes", type=int, default=1024)
    ap.add_argument("--eval-chunk", type=int, default=64,
                    help="part of the episode set, not a memory knob; see "
                         "`evaluate`")
    ap.add_argument("--replicate-external-first-obs-defect", action="store_true",
                    help="reintroduce, on purpose, the defect the out-of-package "
                         "pipeline has: the first observation of every episode "
                         "reaches the network unscaled while the other 95 are "
                         "divided. Exists to answer whether the sign flip "
                         "between the two arms' paired differences (+0.030499 "
                         "here, -0.0260 out of package) is an implementation "
                         "difference or that defect -- the evaluation-side "
                         "injection could only price the defect for a policy "
                         "NOT trained under it, and the out-of-package policies "
                         "were. A run under this flag is NOT a run of this "
                         "market and its products say so")
    ap.add_argument("--align-external-log-std", action="store_true",
                    help="hold the exploration width where the out-of-package "
                         "calibration held it instead of learning it: "
                         "`log_std` is overwritten before every update with "
                         "--external-log-std, which defaults to the adopted "
                         "-5.0 (held for the whole run, not annealed). Pass "
                         "--external-log-std-end to anneal instead, which is "
                         "what `constrained_baseline.py`'s placeholder "
                         "schedule does and what an earlier alignment run used")
    ap.add_argument("--external-log-std", type=float,
                    default=EXTERNAL_ADOPTED_LOG_STD,
                    help="the width --align-external-log-std holds; default is "
                         "the adopted -5.0")
    ap.add_argument("--external-log-std-end", type=float, default=None,
                    help="if given, --align-external-log-std anneals linearly "
                         "from --external-log-std to this instead of holding. "
                         "An earlier alignment run used -0.5 -> -3.0, the placeholder "
                         "schedule")
    ap.add_argument("--align-external-lr", action="store_true",
                    help="adopt the out-of-package step size: --external-lr "
                         "instead of SHARED.lr, linearly annealed to zero over "
                         "the run instead of held constant. The default is the "
                         "ADOPTED 2e-6, not `constrained_baseline.LEARNING_RATE`; "
                         "that constant is a placeholder its own file says the "
                         "calibration overrides")
    ap.add_argument("--external-lr", type=float, default=EXTERNAL_ADOPTED_LR,
                    help="the level --align-external-lr starts from; default is "
                         "the adopted 2e-6. An earlier alignment run used 1e-4, the "
                         "placeholder")
    ap.add_argument("--align-external-init", action="store_true",
                    help="scale both output heads down to the out-of-package "
                         "0.01: the actor head through `init_scale` and the "
                         "critic head by rescaling the initialised kernel. "
                         "The initialiser family (orthogonal here, Glorot "
                         "there) is NOT aligned by this flag")
    ap.add_argument("--align-external-init-family", action="store_true",
                    help="redraw every weight matrix from the out-of-package "
                         "family: N(0,1) * sqrt(2/fan_in), with the 0.01 on "
                         "both heads. --align-external-init matches the head "
                         "SCALE only and records that it does not match the "
                         "family; this is the family, and on the mean head it "
                         "overrides whatever init_scale produced")
    ap.add_argument("--align-external-grad-steps", action="store_true",
                    help="take the out-of-package number of optimiser steps per "
                         "update: `epochs = preliminary_reference.EPOCHS` over "
                         "one full-batch minibatch, i.e. 4 steps instead of "
                         "epochs x minibatches = 320. NOT on the note's list of "
                         "eight differences, and it is the largest of them")
    ap.add_argument("--align-external-vf", action="store_true",
                    help="match the out-of-package weight on the value loss. "
                         "`ippo._loss` computes 0.5 (v - r)^2 and then "
                         "multiplies by vf_coef, so SHARED's 0.5 gives 0.25 "
                         "(v - r)^2 while `constrained_baseline` uses "
                         "VALUE_COEF (v - r)^2 = 0.5 (v - r)^2. Also not on "
                         "the note's list")
    ap.add_argument("--arm", choices=("learned", "truthful"), default="learned",
                    help="`truthful` masks every household off the learner, so "
                         "the environment substitutes its own truthful action "
                         "and no network is consulted; it is the arm both paths "
                         "can produce without a policy and is therefore where "
                         "the item-by-item tolerance is measured")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    run = build_run(args.agents, args.initial_soc)
    params, env = run["params"], run["env"]
    spec = env[3]
    episode_len = int(np.asarray(params.episode_len))
    if episode_len != R.EPISODE_LEN:
        raise SystemExit(f"params carry episode_len={episode_len} while the "
                         f"experiment declares {R.EPISODE_LEN}")

    align = dict(log_std=bool(args.align_external_log_std),
                 lr=bool(args.align_external_lr),
                 init=bool(args.align_external_init),
                 init_family=bool(args.align_external_init_family),
                 grad_steps=bool(args.align_external_grad_steps),
                 vf=bool(args.align_external_vf))
    if args.algo == "sac" and any(align.values()):
        raise SystemExit("the --align-external-* treatments are IPPO-path "
                         "contrasts against constrained_baseline.py; refused "
                         "under --algo sac rather than accepted and dropped")
    base = SHARED if args.algo == "ippo" else SAC_SHARED
    cfg = dataclasses.replace(base, n_envs=args.n_envs, horizon=episode_len)
    _sac_over = {"buffer_size": args.sac_buffer_size, "batch_size": args.sac_batch_size,
                 "utd_ratio": args.sac_utd_ratio,
                 "hidden": (tuple(args.sac_hidden) if args.sac_hidden is not None
                            else None)}
    if any(v is not None for v in _sac_over.values()) and args.algo != "sac":
        raise SystemExit("--sac-buffer-size / --sac-batch-size / --sac-utd-ratio / "
                         "--sac-hidden belong to SACConfig; this run is --algo ippo")
    for _k, _v in _sac_over.items():
        if _v is not None and _v != getattr(base, _k):
            cfg = dataclasses.replace(cfg, **{_k: _v})
    if align["init"]:
        cfg = dataclasses.replace(cfg, init_scale=EXTERNAL_OUTPUT_SCALE)
    if align["grad_steps"]:
        # `_update` scans `epochs` permutations of `minibatches` gradient steps,
        # so one full-batch minibatch per epoch is the out-of-package shape.
        cfg = dataclasses.replace(cfg, epochs=R.EPOCHS, minibatches=1)
    if align["vf"]:
        cfg = dataclasses.replace(cfg, vf_coef=2.0 * R.VALUE_COEF)
    #: What this run does NOT share with `hyperparams.SHARED`, named rather than
    #: left as a difference a reader has to find.  `horizon` is not a tuning
    #: choice here: it is the pool constraint (module docstring).
    off_shared = {
        "horizon": {"shared": base.horizon, "here": cfg.horizon,
                    "why": "one episode of this market; enforced by "
                           "make_pooled_ippo so the rollout cannot cross a "
                           "boundary and draw a start outside the pool"},
    }
    for _k, _v in _sac_over.items():
        if _v is not None and _v != getattr(base, _k):
            off_shared[_k] = {"shared": list(getattr(base, _k)) if _k == "hidden"
                              else getattr(base, _k),
                              "here": list(_v) if _k == "hidden" else _v,
                              "why": (f"--sac-hidden: SAC width override (added "
                                      f"2026-09-20 because the per-agent 256-wide "
                                      f"column OOMs at a 0.45 share, C-119; used by "
                                      f"both layouts, see per_agent_params_effective); "
                                      f"this width is NOT the one the other SAC "
                                      f"readings use"
                                      if _k == "hidden" else
                                      f"--sac-{_k.replace('_', '-')}: the 04 SAC "
                                      f"columns of 2026-09-09 ran at this value "
                                      f"(constrained_baseline.py); device memory")}
    if args.n_envs != base.n_envs:
        off_shared["n_envs"] = {"shared": base.n_envs, "here": args.n_envs,
                                "why": "batch sized for the device this run "
                                       "was given"}
    if align["init"]:
        off_shared["init_scale"] = {
            "shared": SHARED.init_scale, "here": cfg.init_scale,
            "why": "--align-external-init: the out-of-package output-head "
                   "scale, so this run is a controlled contrast against that "
                   "arm and not a run of this market"}
    if align["lr"]:
        off_shared["lr"] = {
            "shared": SHARED.lr, "here": float(args.external_lr),
            "adopted": EXTERNAL_ADOPTED_LR,
            "is_adopted": bool(args.external_lr == EXTERNAL_ADOPTED_LR),
            "why": "--align-external-lr: annealed linearly to zero over the "
                   "run; the level is --external-lr, whose default is the "
                   "value the out-of-package calibration adopted"}
    if align["grad_steps"]:
        off_shared["epochs"] = {"shared": SHARED.epochs, "here": cfg.epochs,
                                "why": "--align-external-grad-steps"}
        off_shared["minibatches"] = {"shared": SHARED.minibatches,
                                     "here": cfg.minibatches,
                                     "why": "--align-external-grad-steps: one "
                                            "full-batch step per epoch, as "
                                            "`constrained_baseline.make_update` "
                                            "does"}
    if align["vf"]:
        off_shared["vf_coef"] = {
            "shared": SHARED.vf_coef, "here": cfg.vf_coef,
            "why": "--align-external-vf: `ippo._loss` already carries the 0.5 "
                   "of the squared error, so twice VALUE_COEF here equals "
                   "VALUE_COEF there"}

    bounds = bounds_for(spec)
    #: `obs_mean = 0` with `obs_std = scale` makes `_norm` the division the
    #: out-of-package runs used.  Read back below out of what was passed, so the
    #: product records the transform in force rather than the one intended.
    obs_std = jnp.asarray(run["scale"], jnp.float32)
    obs_mean = jnp.zeros_like(obs_std)

    guard_bit = horizon_guard_bites(env, run["train_starts"], bounds, cfg,
                                    obs_mean, obs_std, episode_len)
    if not guard_bit:
        raise SystemExit(
            "make_pooled_ippo accepted horizon = episode_len + 1; the guard "
            "this run depends on is not live, so the pool constraint is not "
            "being enforced and no number from this run is what it claims")

    if args.per_agent_params and align["init_family"]:
        # `redraw_init_family` takes fan_in from `shape[0]`, which under this
        # layout is `n_agents` rather than the input width, so the redraw would
        # silently use the wrong scale on every layer.  Refused rather than
        # fixed here: the combination has no run behind it.
        raise SystemExit(
            "--align-external-init-family and --per-agent-params cannot be "
            "combined: the redraw reads fan_in off the leading axis, which is "
            "n_agents under the per-agent layout, so it would rescale every "
            "layer by sqrt(2/n_agents) instead of sqrt(2/fan_in)")
    if args.per_agent_params:
        print(f"PER-AGENT PARAMETERS: {args.agents} independent copies of the "
              f"network. Products of this run are stamped "
              f"`per_agent_params: true` and must not be compared leaf-for-leaf "
              f"against a shared archive.", flush=True)

    truthful = args.arm == "truthful"
    if truthful:
        params = params.replace(
            learner_mask=jnp.zeros_like(params.learner_mask))

    if args.env_chunks != 1:
        raise SystemExit(
            f"--env-chunks={args.env_chunks}: this driver's learner is built by "
            f"make_pooled_ippo, which does not take the keyword; refused rather "
            f"than accepted and dropped (the flag's comment says what would "
            f"wire it)")
    key = jax.random.PRNGKey(args.seed)
    key, key_init, key_train = jax.random.split(key, 3)
    if args.algo == "ippo":
        init, iterate = make_pooled_ippo(
            env, run["train_starts"], bounds, cfg, obs_mean, obs_std,
            episode_len=episode_len,
            per_agent_params=args.per_agent_params,
            replicate_external_first_obs_defect=(
                args.replicate_external_first_obs_defect))
    else:
        if args.replicate_external_first_obs_defect:
            raise SystemExit("--replicate-external-first-obs-defect is an IPPO-"
                             "path experiment; refused under --algo sac")
        # the critic's reward scale, fitted once from the truthful rollout on the
        # training pool and frozen like `obs_mean` / `obs_std` (as run_rl_02 does)
        # `sac.reward_statistics` reads `spec["baseline_action"]` as a constant
        # array; in this market that entry is a FUNCTION of the period (the
        # truthful price depends on each household's net position), and the
        # truthful arm is produced by the environment itself under a zero
        # `learner_mask`.  Same sampling structure as `reward_statistics`, with
        # the mask doing the substitution and a zero action handed in.
        key, k_stat = jax.random.split(key)
        _train_env = restrict_starts(env, run["train_starts"])
        _truthful = params.replace(learner_mask=jnp.zeros_like(params.learner_mask))
        _zero_act = jnp.zeros(tuple(int(d) for d in spec["action_shape"]), jnp.float32)
        _reset, _s, _step_auto, _sp = _train_env
        _keys = jax.random.split(k_stat, cfg.n_envs)
        _obs, _state = jax.vmap(_reset, in_axes=(0, None))(_keys, _truthful)

        def _one(carry, _):
            st, k = carry
            k, k_env = jax.random.split(k)
            _o, nxt, reward, *_ = jax.vmap(_step_auto, in_axes=(0, 0, None, None))(
                jax.random.split(k_env, cfg.n_envs), st, _zero_act, _truthful)
            return (nxt, k), reward

        _, _seen = jax.lax.scan(_one, (_state, _keys[0]), None, length=cfg.horizon)
        _std = float(jnp.std(_seen))
        scale = _std if _std > 1e-8 else 1.0
        cfg = dataclasses.replace(cfg, reward_scale=scale)
        print(f"SAC reward_scale = {scale:.6e} (pooled std of the per-agent "
              f"reward under the truthful action over a {cfg.n_envs} x "
              f"{cfg.horizon} sample)", flush=True)
        init, iterate = make_pooled_sac(
            env, run["train_starts"], bounds, cfg, obs_mean, obs_std,
            episode_len=episode_len, per_agent_params=args.per_agent_params)
    policy, tx, opt_state, env_state, env_obs = init(key_init, params)
    #: The layout that is actually in force, read off the tree `init` RETURNED.
    #: `per_agent_params` does not enter `vars(cfg)` (it is a keyword of
    #: `make_ippo`), so without this the product records only what was asked
    #: for, and a flag that is accepted and dropped is invisible.
    _leading = sorted({int(jnp.shape(leaf)[0])
                       for leaf in jax.tree_util.tree_leaves(
                           policy if args.algo == "ippo" else policy["actor"])
                       if jnp.ndim(leaf) > 0})
    per_agent_effective = bool(_leading == [int(args.agents)])
    if per_agent_effective != bool(args.per_agent_params):
        raise SystemExit(
            f"--per-agent-params={bool(args.per_agent_params)} was requested "
            f"and the tree `init` returned has leading axes {_leading} against "
            f"agents={args.agents}; the flag did not reach the learner")

    #: The three alignment treatments, all applied here and all recorded.  Each
    #: is a driver-level change: nothing under `powermarketjax/` is touched, so
    #: no other market's unflagged device moves.
    align_effective = {}
    if align["init"]:
        policy, head = scale_value_head(policy, EXTERNAL_OUTPUT_SCALE)
        align_effective["init"] = {
            "actor_head_scale": float(cfg.init_scale),
            "value_head_layer": head,
            "value_head_rescaled_by": EXTERNAL_OUTPUT_SCALE,
            "not_aligned": "initialiser family (orthogonal here, Glorot in "
                           "`preliminary_reference.init_policy`)"}
    if align["init_family"]:
        # After `init`, so that on the mean head the redraw is what stands.
        key, key_family = jax.random.split(key)
        policy, family = redraw_init_family(
            policy, key_family, cfg.hidden,
            int(jnp.shape(bounds[0])[-1]), EXTERNAL_OUTPUT_SCALE)
        align_effective["init_family"] = {
            "family_here": "orthogonal: sqrt(2) hidden, init_scale mean head, "
                           "1.0 value head",
            "family_now": "N(0,1) * sqrt(2/fan_in), both heads * "
                          f"{EXTERNAL_OUTPUT_SCALE}",
            "source": "preliminary_reference.init_policy",
            "overrides_init_scale": bool(align["init"]),
            "layers": family}
    if align["lr"]:
        # One optimiser step per minibatch per epoch per iteration; the
        # out-of-package loop takes one per epoch because it does not
        # minibatch, so the schedule is written over the steps this run will
        # actually take rather than over its iteration count.
        n_opt_steps = int(args.iterations) * cfg.epochs * cfg.minibatches
        schedule = optax.linear_schedule(float(args.external_lr), 0.0,
                                         n_opt_steps)
        tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm),
                         optax.adam(schedule))
        opt_state = tx.init(policy)
        align_effective["lr"] = {
            "start": float(args.external_lr), "end": 0.0,
            "adopted_start": EXTERNAL_ADOPTED_LR,
            "is_adopted": bool(args.external_lr == EXTERNAL_ADOPTED_LR),
            "placeholder_in_source": float(CB.LEARNING_RATE),
            "steps": n_opt_steps,
            "steps_are": "iterations x epochs x minibatches",
            "lr_at_step_0": float(schedule(0)),
            "lr_at_last_step": float(schedule(max(n_opt_steps - 1, 0)))}
    if align["grad_steps"]:
        align_effective["grad_steps"] = {
            "epochs": cfg.epochs, "minibatches": cfg.minibatches,
            "optimiser_steps_per_update": cfg.epochs * cfg.minibatches,
            "shared_steps_per_update": SHARED.epochs * SHARED.minibatches,
            "source": "preliminary_reference.EPOCHS with one full batch"}
    if align["vf"]:
        align_effective["vf"] = {
            "vf_coef": cfg.vf_coef,
            "effective_weight_on_squared_error": 0.5 * cfg.vf_coef,
            "out_of_package": float(R.VALUE_COEF)}
    if align["log_std"]:
        align_effective["log_std"] = {
            "start": float(args.external_log_std),
            "end": (float(args.external_log_std_end)
                    if args.external_log_std_end is not None
                    else float(args.external_log_std)),
            "held_constant": args.external_log_std_end is None,
            "adopted": EXTERNAL_ADOPTED_LOG_STD,
            "is_adopted": bool(args.external_log_std == EXTERNAL_ADOPTED_LOG_STD
                               and args.external_log_std_end is None),
            "placeholder_in_source": [float(CB.LOG_STD_START),
                                      float(CB.LOG_STD_END)],
            "applied": "before every update, overwriting the learned leaf"}
    if align_effective:
        print(f"alignment in force: {json.dumps(align_effective, default=float)}",
              flush=True)

    all_starts = run["n_periods"] - episode_len + 1
    pool = dict(all=int(all_starts), train=int(run["train_starts"].size),
                held_out=int(run["eval_starts"].size),
                buffer=int(all_starts - run["train_starts"].size
                           - run["eval_starts"].size),
                block_days=CB.EVAL_BLOCK_DAYS, every_days=CB.EVAL_EVERY_DAYS,
                source="constrained_baseline.make_start_pools")

    battery = params.battery
    #: The scenario stamp, assembled once and carried by three products: the
    #: curve's `meta` line below and both `write_params` call sites.  Until
    #: 2026-08-28 only the curve had it, so a `.params.npz` from this driver
    #: could not name its case, its window, or which of its null scenario
    #: factors were null *by construction*.
    #
    # 04 has no unit commitment and no network, so four of the fields every
    # wholesale product carries have no referent here.  They are present and
    # null because the checker requires the keys, and `not_applicable` says
    # which of the nulls are "this market has no such quantity" rather than
    # "nobody recorded it" -- two statements that a bare null would merge.
    scenario_meta = dict(
        case=CASE,
        cap_scale=None, ramp_scale=None, voll=None, markup_max=None,
        not_applicable=["cap_scale", "ramp_scale", "voll", "markup_max"],
        not_applicable_why=("04 commits no units and models no network; its "
                            "only feasibility quantity is the clipped battery "
                            "command, reported on the cost channel"),
        window={"periods": run["n_periods"],
                "days": run["n_periods"] / (24.0 / R.DELTA),
                "derived_from": "the panel length, not read off the index",
                "source": "load_fluvius_households default window"})
    curve_meta = dict(
        market=MARKET, arm=args.arm, algo=args.algo, seed=args.seed,
        commit=commit_hash(), commit_dirty=_START_DIRTY,
        # The defect replicator belongs in the CURVE's metadata and not only in
        # `run.json`: the curve file is what travels to a figure, and a training
        # curve that does not say it was produced under a deliberately broken
        # observation pipeline is a curve nobody can place.  It was missing for
        # the first batch run under the flag (five seeds), which carry it in
        # `run.json` and in their launch logs instead.
        replicate_external_first_obs_defect=bool(
            args.replicate_external_first_obs_defect) or None,
        align_external=align_effective or None,
        #: Requested and effective, both, and they are two claims: the second
        #: is read off the parameter tree and is what a reader should filter on.
        per_agent_params=bool(args.per_agent_params),
        env_chunks=int(args.env_chunks),
        per_agent_params_effective=per_agent_effective,
        param_leaf_leading_axes=_leading,
        per_agent_params_is_cross_check=(
            "the published 04 learning numbers come from "
            "`constrained_baseline --per-agent-params`; this path repeats the "
            "ablation in package"),
        **scenario_meta,
        episode_len=episode_len,
        hyperparams={k: (list(v) if isinstance(v, tuple) else v)
                     for k, v in vars(cfg).items()},
        hyperparams_provenance=PROVENANCE,
        n_envs=cfg.n_envs, horizon=cfg.horizon, off_shared=off_shared,
        # the three dispositions of the module docstring, as facts about this run
        convergence_key=None,
        convergence_key_why=("04 publishes no convergence flag: no solver, no "
                             "tolerance, no iteration bound"),
        observation_standardisation={
            "obs_mean": "zeros",
            "obs_std": "constrained_baseline.observation_scale(params, agents)",
            "not": "wrappers.p2p.baseline_observation_statistics",
            "why": ("under the truthful action the battery never moves, so soc "
                    "has zero variance there and the 1e-8 floor turns its "
                    "divisor into 1.0"),
            "obs_std_checksum": float(np.sum(np.asarray(obs_std, np.float64))),
            "obs_mean_all_zero": bool(np.all(np.asarray(obs_mean) == 0.0))},
        horizon_equals_episode_len=bool(cfg.horizon == episode_len),
        horizon_guard_exercised=bool(guard_bit),
        start_pool=pool,
        initial_soc=float(np.asarray(battery.initial_soc).reshape(-1)[0]),
        battery={"capacity_mwh": float(np.max(np.asarray(battery.capacity))),
                 "power_mw": float(np.max(np.asarray(battery.power_max))),
                 "eta_charge": float(np.max(np.asarray(battery.eta_charge))),
                 "eta_discharge": float(np.max(np.asarray(battery.eta_discharge))),
                 "read_back_from": "params.battery, not from a literal here"},
        pi_exp=R.PI_EXP, pi_ret=R.PI_RET, kappa=R.KAPPA,
        period_hours=R.DELTA, agents=args.agents,
        learner_households=int(np.sum(np.asarray(params.learner_mask))),
        iterations_requested=args.iterations,
        eval_episodes=args.eval_episodes, eval_chunk=args.eval_chunk,
        x64=bool(jax.config.jax_enable_x64),
        x64_why=("float32 by design in this market and in the out-of-package "
                 "results these numbers are compared against"),
        **runtime_stamp(requires_x64=False))

    jsonl = CurveLog(args.curve_out, curve_meta)
    print(f"[{time.strftime('%H:%M:%S')}] {MARKET} arm={args.arm} "
          f"agents={args.agents} soc0={args.initial_soc} seed={args.seed} "
          f"n_envs={cfg.n_envs} horizon={cfg.horizon} "
          f"pool train={pool['train']}/{pool['all']} "
          f"held-out={pool['held_out']} buffer={pool['buffer']}", flush=True)
    if jsonl.path is not None:
        print(f"per-iteration JSONL -> {repo_relative(jsonl.path)}", flush=True)

    step_iter = jax.jit(iterate, static_argnums=(1,))
    ckpt_dir = out_dir / "checkpoints"
    peak_first = pool_first = None
    curve, t0 = [], time.time()
    log_std_trace = []
    for it in range(args.iterations):
        t_it = time.time()
        if align["log_std"]:
            frac = it / max(args.iterations - 1, 1)
            end = (args.external_log_std if args.external_log_std_end is None
                   else args.external_log_std_end)
            width = args.external_log_std + frac * (end - args.external_log_std)
            policy = set_log_std(policy, width)
            log_std_trace.append(float(width))
        policy, opt_state, env_state, env_obs, key_train, m = step_iter(
            policy, tx, opt_state, env_state, env_obs, key_train, params)
        #: JAX dispatches asynchronously, so `step_iter` returns before the
        #: iteration has run and `time.time()` here would measure the dispatch
        #: alone.  Measured 2026-08-27 on card 1 at N=1200: without this line
        #: the column read 0.0s for every iteration after the first while the
        #: printed timestamps were 2.5s apart -- and iteration 0 looked right
        #: (7.4s) only because compilation is synchronous, which is the worst
        #: shape for a defect to have.  The other drivers do not need the line
        #: because they reduce a metric into their curve row before taking the
        #: time, which forces materialisation as a side effect; this one does
        #: not, and relying on that would make the timing depend on which
        #: fields the row happens to carry.
        policy, opt_state, m = jax.block_until_ready((policy, opt_state, m))
        if it == 0:
            # read back out of the metrics, not out of the flag that was passed:
            # this is the only form that says the market's own declaration
            # reached the learner
            for absent in ("unconverged_frac", "step_converged"):
                if absent in m:
                    raise SystemExit(
                        f"metrics carry {absent!r}, so convergence_key did not "
                        f"take effect and this market is being credited with a "
                        f"convergence flag it does not publish")
            print(f"convergence flag absent from metrics as declared; "
                  f"metrics are {sorted(m)}", flush=True)
            #: read after the first iteration, which is the first moment the
            #: whole training step -- rollout, GAE and every update epoch --
            #: has been compiled and executed at least once.  Reading it
            #: before that reports the cost of the reset alone.
            peak_first, pool_first = report_memory("after iteration 0", cfg)
        # per-learner diagnostics, as run_rl_02 keeps them: PPO's surrogate
        # entropy and SAC's sampled entropy are not the same quantity (`sac.py`)
        if args.algo == "ippo":
            diag = dict(pg_loss=float(m["pg_loss"]), vf_loss=float(m["vf_loss"]),
                        entropy=float(m["entropy"]),
                        approx_kl=float(m["approx_kl"]),
                        clip_frac=float(m["clip_frac"]))
        else:
            diag = dict(q_loss=float(m["q_loss"]), q_mean=float(m["q_mean"]),
                        target_mean=float(m["target_mean"]),
                        actor_loss=float(m["actor_loss"]),
                        alpha_loss=float(m["alpha_loss"]),
                        alpha=float(m["alpha"]), entropy=float(m["entropy"]),
                        buffer_filled=int(m["buffer_filled"]))
        row = dict(seconds=time.time() - t_it, iteration=it,
                   reward_mean=float(m["reward_mean"]),
                   costs_mean=float(m["costs_mean"]),
                   reward_per_agent=np.asarray(m["reward_per_agent"],
                                               np.float64), **diag)
        curve.append(row)
        jsonl.iteration(row)
        print(f"  [{time.strftime('%H:%M:%S')}] it {it:>4}  "
              f"reward_mean {row['reward_mean']:+.6e}  "
              f"costs_mean {row['costs_mean']:.6e}  "
              f"entropy {row['entropy']:+.4f}  {row['seconds']:.1f}s",
              flush=True)
        if args.checkpoint_every > 0 and (it + 1) % args.checkpoint_every == 0:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            write_params(ckpt_dir / f"iter{it + 1:04d}.params.npz", policy,
                         obs_mean, obs_std, cfg, seed=args.seed, iteration=it + 1,
                         per_agent_params=per_agent_effective,
                         extra_meta=dict(scenario_meta, market=MARKET,
                                         arm=args.arm,
                                         initial_soc=args.initial_soc,
                                         episode_len=episode_len,
                                         start_pool=pool))

    seconds_training = time.time() - t0

    # ---- evaluation on the held-out pool, at the mean of the action distribution
    eval_env = restrict_starts(env, run["eval_starts"])
    if args.replicate_external_first_obs_defect:
        # The evaluation carries the defect too, because the arm being
        # reproduced both trains and evaluates under it.  Same one-line
        # construction as the training side (`make_pooled_ippo`): with
        # `obs_mean` at zero, handing the reset observation over
        # pre-multiplied by `obs_std` makes the learner's `_norm` return
        # the raw observation for that one step.  The reference this run
        # is paired against is a constant battery command, which reads no
        # observation, so the paired difference stays well defined.
        eval_env = unscaled_first_obs(eval_env, obs_std)

    if truthful:
        # no network is consulted; the mask makes the environment substitute
        act = lambda obs, k: jnp.zeros(tuple(int(d) for d in spec["action_shape"]),
                                       jnp.float32)
    else:
        _mk = make_greedy_action if args.algo == "ippo" else make_sac_greedy_action
        greedy = _mk(spec, bounds, cfg, obs_mean, obs_std,
                     per_agent_params=args.per_agent_params)
        act = lambda obs, k: greedy(policy, obs)
    #: the out-of-package evaluation key, so both paths score the same episodes
    eval_key = jax.random.PRNGKey(20_000 + args.agents)
    ret, cost, deg = evaluate(eval_env, params, act, eval_key,
                              args.eval_episodes, args.eval_chunk, episode_len)
    seconds_eval = time.time() - t0 - seconds_training
    if ret.shape != (args.eval_episodes, args.agents):
        raise SystemExit(
            f"the evaluation came back {ret.shape}, not "
            f"({args.eval_episodes}, {args.agents}); the per-household vector "
            f"is what the paired comparison is taken on, so a reduction that "
            f"happened one axis early would be invisible in the mean")

    peak_final, pool_final = report_memory("end of run", cfg)
    result = dict(
        curve_meta, eval_return=float(ret.mean()), eval_cost=float(cost.mean()),
        # This market's system cost, in the same reduction as `eval_return`:
        # mean over episodes and households of the per-episode sum.  It is a
        # cost, so it is reported positive and a larger value is worse.
        eval_degradation=float(deg.mean()),
        eval_degradation_total=float(deg.sum(1).mean()),
        peak_device_bytes_after_first_iteration=peak_first,
        peak_device_bytes_final=peak_final,
        device_pool_bytes=pool_final if pool_final is not None else pool_first,
        batch_env_steps=cfg.n_envs * cfg.horizon,
        seconds_per_iteration_median=(
            float(np.median([c["seconds"] for c in curve[1:]]))
            if len(curve) > 1 else None),
        seconds_first_iteration=(curve[0]["seconds"] if curve else None),
        eval_shape=list(ret.shape),
        eval_key="PRNGKey(20000 + agents)",
        eval_definition=("mean over episodes and households of the per-episode "
                         "sum, which is `constrained_baseline`'s eval_return; "
                         "the constraint total is the same reduction of the "
                         "clipped-command channel, full scale 96 per household "
                         "per episode, and it never enters the reward"),
        replicate_external_first_obs_defect=bool(
            args.replicate_external_first_obs_defect) or None,
        align_external=align_effective or None,
        #: Read back off the trained parameters, not off the flag: this is the
        #: only form that says the treatment reached the learner rather than
        #: that the driver meant to apply it.  Under the log-std treatment the
        #: last written width is `LOG_STD_END`; without it the leaf is whatever
        #: the update learned.
        #: Nested one level under `--per-agent-params`, where the leaf is
        #: `(n_agents, act_dim)` rather than `(act_dim,)`; `tolist` keeps the
        #: shape rather than flattening it, because a flat list of 2 n_agents
        #: numbers reads exactly like a shared run at a larger action dimension.
        log_std_final=(np.asarray(policy["params"]["log_std"], np.float64).tolist()
                       if args.algo == "ippo" else None),
        algo=args.algo,
        log_std_schedule_first_last=(
            [log_std_trace[0], log_std_trace[-1]] if log_std_trace else None),
        seconds_training=seconds_training, seconds_eval=seconds_eval,
        seconds_total=time.time() - t_start,
        iterations_run=len(curve))
    (out_dir / "run.json").write_text(json.dumps(result, indent=1,
                                                 default=str))
    np.savez(out_dir / "eval_per_household.npz",
             ret=ret, cost=cost, degradation=deg,
             note=json.dumps({"axes": "(episode, household)",
                              "units": "EUR per household per episode; the "
                                       "cost channel is dimensionless, 96 full "
                                       "scale; `degradation` is EUR per "
                                       "household per episode and is this "
                                       "market's only real resource cost, "
                                       "already a term of each household's "
                                       "own cost"}))
    write_params(out_dir / "final.params.npz", policy, obs_mean, obs_std, cfg,
                 seed=args.seed, per_agent_params=per_agent_effective,
                 extra_meta=dict(scenario_meta, market=MARKET, arm=args.arm,
                                 algo=args.algo,
                                 initial_soc=args.initial_soc,
                                 episode_len=episode_len, start_pool=pool))
    jsonl.close()
    print(f"[{time.strftime('%H:%M:%S')}] eval on {args.eval_episodes} held-out "
          f"episodes: return {result['eval_return']:+.6f}  "
          f"constraint total {result['eval_cost']:.6f}  "
          f"degradation {result['eval_degradation']:.6f}  "
          f"(training {seconds_training:.1f}s, eval {seconds_eval:.1f}s)",
          flush=True)
    print(f"products -> {repo_relative(out_dir)}", flush=True)


if __name__ == "__main__":
    main()
