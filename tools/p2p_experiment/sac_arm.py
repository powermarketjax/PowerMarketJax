"""SAC for the OUT-OF-PACKAGE market 04 driver, at the in-package calibre.

`constrained_baseline.py` is the program the published 04 learning numbers came
from, and until now it had one learner.  The benchmark makes SAC shared / SAC
per-agent two of the eight columns every market owes, accepted on the condition
"test it thoroughly first, wire it in only if it works".  This file is the second
learner, and it is added **on that driver** rather than only in the package:
two arms that differ in the algorithm AND in which program produced them are not
a contrast in the algorithm.

**Everything that could be imported from `powermarketjax.learning.sac` is
imported and not transcribed.**  The networks (`SACActor`, `SoftQ`), the two
per-agent reductions (`_actor_per_agent`, `_q_per_agent`), what the critic sees
of an action (`_q_action`), the action layout (`_nets`), the squash and its
log-density (`policy.to_action`, `policy.log_prob`) and the episode-boundary
table (`ippo._BOOTSTRAP_AT_DONE`) are the package's own objects, private names
included.  A transcribed `_q_action` would be a second declaration of what the
critic reads, and two declarations of one quantity is the failure `bounds_for`
records for the ancillary box.  What could not be imported is the body of
`make_sac`, whose parts are closures over its own rollout; those are transcribed
below with the same names and the same expressions, and every departure is
listed in the next paragraph rather than left to be discovered.

**The conventions this file copies, in `sac.py`'s own words** (its module
docstring):

* *"Independent learners, no centralised critic."*  Every agent's Q function
  reads its own observation and its own action; the agent axis is a batch axis
  on the shared path and a `vmap` axis on the per-agent path.
* *"The automatic temperature is one scalar on the shared path and one per agent
  on the per-agent path, because each independent learner has its own entropy
  target to meet."*  The target is CleanRL's, *"minus the action dimension, per
  agent"*, so `-2.0` here.
* *"Twin-Q regression on the soft Bellman target"*, the target networks moved by
  Polyak with `tau`, the actor stepped *"against the freshly updated critics"*,
  and the temperature stepped *"on the log-probabilities the actor step
  sampled"* -- CleanRL recomputes them under the updated actor, which
  `SAC_PROVENANCE["not_carried_over"]` records as *"a half-step lag, not an
  accumulating one"*.
* *"The episode boundary branches on `spec['termination']`"* through
  `ippo._BOOTSTRAP_AT_DONE`.  Market 04 is `terminal`, so the bootstrap is
  masked there, and `ippo.py` gives the reason -- the stock left in
  the battery is paid for by a terminal leg inside `reward`, so a continuation
  value on top of it is a second payment for the end of the episode.
* *"Rewards enter the critic divided by a frozen scale ... the pooled standard
  deviation of the per-agent reward under the truthful action, fitted once and
  frozen, like the observation statistics -- a quantity derived by the
  apparatus, not a knob."*
* *"The whole iteration runs inside one `jit`"*: the replay buffer is a
  fixed-size pytree carried between iterations, FIFO cursor, uniform sampling
  over the filled prefix, and the gradient steps are a `lax.scan`.  A host-side
  buffer would put a Python loop on the rollout path, which the all-in-`jit`
  execution rule forbids.
* *"The entropy terms of PPO and SAC are not the same quantity."*  So the two
  columns of market 04 are compared on market outcomes and never on training
  diagnostics, exactly as on markets 01 to 03.

**Seven departures, each forced by this driver and none of them a preference.**

1. *The rollout is one episode per scan, `vmap`ped over the batch* -- this
   driver's shape (`preliminary_reference.make_rollout`), not `sac._rollout`'s
   `scan(horizon)` over a carried environment state.  It is what makes the two
   arms differ in the algorithm alone: both collect `batch` freshly drawn
   episodes per iteration from the same restricted start pool.  It also keeps
   `restrict_starts` effective -- a rollout that ran past a boundary would
   continue from the auto-reset inside `step`, which draws over the whole panel
   including the held-out days.
2. *The successor stored is `info["terminal_obs"]`, not the returned `obs`.*
   `sac._rollout` stores the returned one and lets `_cont` mask it.  On this
   market the two are bit-identical at every step where `done` is false, and on
   the step where it is true `_cont` multiplies `q_next` by exactly zero, so the
   Q target is the same number under either choice.  `terminal_obs` is stored
   because it keeps the auto-reset draw -- which `restrict_starts` does not
   restrict, as that function's docstring records -- out of the buffer.
   **It does not make the buffer free of held-out observations, and the
   measurement is here rather than the claim.**  `make_start_pools` admits a
   start when periods `[s, s+95]` miss every held-out period, which says
   nothing about period `s+96`, and that period is exactly what `terminal_obs`
   reports on the step where `done` fires.  Measured 2026-09-09 on the 209-day
   panel: **13 of the 14 702 admissible training starts have their terminal
   successor inside a held-out block** (0.088 %), and one more runs off the end
   of the panel.  Those 13 observations sit in the buffer at rows where `done`
   is true, where `_cont` is exactly zero, so no gradient ever reads them -- for
   those rows the property is the arithmetic's, and for the other 14 689 it is
   the buffer's.  Both statements are needed; neither alone is true of the
   whole buffer.
3. *No second observation standardisation.*  `make_sac` divides by
   `obs_mean`/`obs_std` from `ippo.observation_statistics`; this driver divides
   by `constrained_baseline.observation_scale`, a fixed function of the scenario
   applied inside the environment tuple, before anything the learner sees.
   Writing `(obs - 0) / 1` here would put an op in the graph standing for a step
   that is taken somewhere else.
4. *The buffer carries `cost`, and the critic regresses on
   `(reward - lam * cost) / reward_scale`.*  Markets 01 to 03 have no
   constrained arm on their driver; this one has two arms and the reported
   reward must stay unshaped, which it does -- `lam` never
   touches what is measured, exactly as in `constrained_baseline.make_update`.
   The multiplier's ascent rule, its clip and its decaying step are that file's
   and are unchanged.  **Replayed transitions are shaped by the multiplier in
   force now, not by the one in force when they were collected**: storing a
   shaped reward would freeze a stale multiplier into the target, and the critic
   would be regressing on an objective that moved without it being told.
5. *The reward scale is fitted by rolling out the truthful ENVIRONMENT.*
   `sac.reward_statistics` submits `spec["baseline_action"]` as an array; on
   market 04 that key is a *function* of the period's injection and offtake, and
   the truthful action is delivered by the environment itself under
   `learner_mask = False` (`envs/p2p/env.py`).  Same quantity, different
   delivery, same floor (`std > 1e-8` or 1.0).
6. *No exploration-width schedule.*  `constrained_baseline` overwrites the PPO
   policy's `log_std` on a linear schedule each iteration because there it is a
   free parameter that does not move on its own.  SAC's `log_std` is a head of
   the network squashed into `[log_std_min, log_std_max]`, and the exploration
   scale is what the automatic temperature governs; there is nothing to
   overwrite and overwriting it would fight the temperature.  This is a
   difference between the 04 SAC column and the 04 PPO column, and it is the
   same difference markets 01 to 03 carry.
7. *No learning-rate schedule.*  That driver anneals PPO's step size to zero;
   `SACConfig` has no such field and CleanRL's SAC does not anneal, so
   `policy_lr` / `q_lr` / `alpha_lr` are constants, as in `SAC_SHARED`.

**One thing this file deliberately does NOT fix.**  `constrained_baseline.build`
applies `scale_observations` and then `restrict_starts`, and `restrict_starts`
re-derives the first observation of every episode from `spec["get_obs"]`, which
is the unscaled one.  Measured 2026-09-09 on CPU at 4 participants: the reset
observation has channel 4 at 13.88 where the scaled stream carries 0.0416, and
channel 0 at 5.2e-03 where the scaled stream carries 1.0 -- one observation in
every 96 is off by two to three orders of magnitude, in both directions.  It is
a defect of the driver and not of this file, it is identical for both arms, and
repairing it would move every published 04 learning number, so it is reported
and left alone.
"""
import numpy as np

import jax
import jax.numpy as jnp
import optax

import preliminary_reference as R
from powermarketjax.learning.ippo import _BOOTSTRAP_AT_DONE
from powermarketjax.learning.policy import bounds_for, log_prob, to_action
from powermarketjax.learning.sac import (SACConfig, _actor_per_agent, _nets,
                                         _q_action, _q_per_agent)

#: `tools/benchmark/hyperparams.py::SAC_SHARED`, field for field.  The values
#: are repeated here rather than imported because `hyperparams.py` lives under
#: `tools/benchmark`, is not on this driver's path, and carries markets 01 to
#: 03's `n_envs` and `horizon` -- two numbers that mean something else here.
#: `constrained_baseline` already keeps its own PPO constants for the same
#: reason.  The provenance of every one of them is CleanRL
#: `sac_continuous_action.py`, read from GitHub master on 2026-09-03; the four
#: that are not CleanRL's are marked.
BUFFER_SIZE = 32_768          # ours (device memory); CleanRL 1e6 transitions
BATCH_SIZE = 256              # CleanRL batch_size, in env-steps (see below)
UTD_RATIO = 1.0               # CleanRL: one update per env-step
GAMMA = 0.99                  # CleanRL gamma
TAU = 0.005                   # CleanRL tau
POLICY_LR = 3e-4              # CleanRL policy_lr
Q_LR = 1e-3                   # CleanRL q_lr
ALPHA_LR = 1e-3               # CleanRL: the temperature uses q_lr
INIT_ALPHA = 0.2              # CleanRL alpha, then autotuned
HIDDEN = (256, 256)           # CleanRL Actor / SoftQNetwork
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0      # CleanRL LOG_STD_MIN / LOG_STD_MAX

#: What one drawn sample is, stated because the number 256 is CleanRL's and the
#: unit is not: CleanRL draws 256 transitions, this draws 256 **env-steps** and
#: every participant's transition rides inside each one.  At 16 participants a
#: gradient step therefore sees 4 096 transitions, at 1 200 it sees 307 200.
#: `SAC_PROVENANCE["from_source_units"]` says the same of markets 01 to 03.
BATCH_SIZE_UNIT = ("env-steps, each carrying every participant's transition; "
                   "CleanRL draws 256 transitions")

#: One env-step of the replay buffer holds, per participant, `obs` (15) +
#: `next_obs` (15) + `pre` (2) + `reward` (1) + `cost` (1) = 34 float32, i.e.
#: 136 bytes.  `BUFFER_SIZE` env-steps is therefore 4.46 MB per participant:
#: 71 MB at 16 and 5.35 GB at 1 200.  `--buffer-size` exists for that second
#: number.
BYTES_PER_ENV_STEP_PER_AGENT = 34 * 4


def config_for(batch, reward_scale, buffer_size=BUFFER_SIZE,
               utd_ratio=UTD_RATIO, episode_len=None):
    """`SACConfig` for this driver, from the constants above and the run's batch.

    The dataclass is the package's, so a market 04 product's `hyperparams`
    block carries the same field names as markets 01 to 03's and can be diffed
    against them.  `n_envs` is the number of episodes collected in parallel and
    `horizon` is the episode length, which is what those two fields mean on the
    other markets too -- `n_envs * horizon` is the env-steps one iteration
    collects, and it is the number `utd_ratio` multiplies.
    """
    return SACConfig(
        n_envs=int(batch),
        horizon=int(R.EPISODE_LEN if episode_len is None else episode_len),
        buffer_size=int(buffer_size), batch_size=BATCH_SIZE,
        utd_ratio=float(utd_ratio), gamma=GAMMA, tau=TAU,
        policy_lr=POLICY_LR, q_lr=Q_LR, alpha_lr=ALPHA_LR,
        init_alpha=INIT_ALPHA, hidden=HIDDEN, log_std_min=LOG_STD_MIN,
        log_std_max=LOG_STD_MAX, reward_scale=float(reward_scale))


PROVENANCE = {
    "source": ("CleanRL sac_continuous_action.py, Args dataclass, Actor and "
               "SoftQNetwork, fetched 2026-09-03; the same eleven values "
               "tools/benchmark/hyperparams.py::SAC_SHARED carries for markets "
               "01 to 03"),
    "url": ("https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/"
            "cleanrl/sac_continuous_action.py"),
    "from_source": ["batch_size", "utd_ratio", "gamma", "tau", "policy_lr",
                    "q_lr", "alpha_lr", "init_alpha", "hidden", "log_std_min",
                    "log_std_max"],
    "ours": {"n_envs": "the driver's --batch, i.e. episodes in parallel",
             "horizon": ("preliminary_reference.EPISODE_LEN = 96 quarter "
                         "hours, one market day; a market definition, not "
                         "tuning"),
             "buffer_size": "env-steps, device memory; CleanRL 1e6",
             "reward_scale": ("fitted per run from the truthful rollout and "
                              "stamped into the product; not a constant here")},
    "from_source_units": {"batch_size": BATCH_SIZE_UNIT},
    "not_carried_over": ("learning_starts 5e3 random steps (none here); "
                         "policy_frequency 2 (actor and critic at the same "
                         "frequency); PyTorch default initialiser (flax's is "
                         "used); the temperature's log-probability is the one "
                         "the actor step sampled, before that step was "
                         "applied, a half-step lag"),
    "departures_from_the_04_ppo_arm": (
        "no exploration-width schedule (SAC's log_std is a network head "
        "squashed into [-5, 2] and the temperature governs the scale); no "
        "learning-rate annealing (SACConfig has no such field); the critic "
        "regresses on (reward - lam * cost) / reward_scale, and the replayed "
        "transitions carry the multiplier in force now rather than the one in "
        "force when they were collected"),
    "tuned_per_market": False,
}


def env_layout(env, cfg):
    """`(n_agents, act_dim, act_shape, low, high, actor, qnet)` from the market.

    `sac._nets` itself, so the two networks and the action layout this file
    builds are the objects `make_sac` builds; `bounds_for` reads the box off the
    market's own spec, which for market 04 is `[-1, 1]` on both coordinates.
    """
    spec = env[3]
    return _nets(spec, bounds_for(spec), cfg)


# --------------------------------------------------------------- the parameters

def init_params(key, env, cfg, per_agent=False):
    """The learner tree: `{actor, q1, q2, q1_target, q2_target, log_alpha}`.

    Transcribed from `sac.make_sac`'s `_init_nets` and the line after it,
    ``params = dict(nets, q1_target=nets["q1"], q2_target=nets["q2"])`` -- the
    targets start equal to the critics rather than at their own draw.

    `z` is one un-batched observation, ``(n_agents, obs_dim)``, which is what
    shapes each agent on its own row on the per-agent path.  On the shared path
    the same array is a batch of `n_agents` rows through one network, which is
    the whole difference between the two layouts.
    """
    n_agents, act_dim, _shape, _low, _high, actor, qnet = env_layout(env, cfg)
    obs_dim = int(env[3]["obs_dim"])
    z = jnp.zeros((n_agents, obs_dim), jnp.float32)
    ka, k1, k2 = jax.random.split(key, 3)
    if per_agent:
        a0 = jnp.zeros((n_agents, act_dim), z.dtype)
        nets = dict(
            actor=jax.vmap(actor.init)(jax.random.split(ka, n_agents), z),
            q1=jax.vmap(qnet.init)(jax.random.split(k1, n_agents), z, a0),
            q2=jax.vmap(qnet.init)(jax.random.split(k2, n_agents), z, a0),
            log_alpha=jnp.full((n_agents,), jnp.log(cfg.init_alpha),
                               jnp.float32))
    else:
        a0 = jnp.zeros(z.shape[:-1] + (act_dim,), z.dtype)
        nets = dict(actor=actor.init(ka, z), q1=qnet.init(k1, z, a0),
                    q2=qnet.init(k2, z, a0),
                    log_alpha=jnp.asarray(jnp.log(cfg.init_alpha), jnp.float32))
    return dict(nets, q1_target=nets["q1"], q2_target=nets["q2"])


#: The six top-level entries of the learner tree, in sorted order.  Used to tell
#: a SAC tree from a PPO one when a product is read back, and asserted rather
#: than described in `tests/tools/test_p2p_external_sac_l0.py`.
TOP_LEVEL = ("actor", "log_alpha", "q1", "q1_target", "q2", "q2_target")


def greedy_forward(env, cfg, per_agent=False):
    """`(policy, obs) -> (mean, log_std, value)`, shaped like `R.forward`.

    Returned in that shape so `preliminary_reference.make_rollout` can drive
    this learner's evaluation unchanged: its `learned_mean` arm computes
    ``tanh(mean)``, and `sac.make_sac_greedy_action` computes
    ``to_action(mean, -1, 1)``, which is `-1 + 0.5 * 2 * (tanh(mean) + 1)`.

    **Those two are equal in exact arithmetic and NOT equal in float32.**  The
    round trip through `+1` and `-1` loses the low bits of a small `tanh`.
    Measured 2026-09-09 on CPU over 32 768 draws of `mean ~ 2 * N(0, 1)`: the
    largest disagreement is **5.960e-08**, half an ulp at magnitude one, and
    **64.75 %** of the draws are bit-identical, falling to **0 %** among the 119
    draws with `|mean| < 0.01`.  So the two expressions agree to half an ulp
    and not to the bit, and `test_p2p_external_sac_l0.py` asserts the bound
    rather than the equality.

    `tanh` is what runs, deliberately: it is the expression the PPO arm's
    evaluation already used, so the two 04 arms are scored by the same line of
    code and differ in the learner alone.  Writing `to_action` here
    instead would move the PPO arm's evaluation or leave the two arms scored by
    two expressions, and the second is the failure this file exists to avoid.

    `value` comes back as zeros.  SAC has no value head; the third slot exists
    because `R.make_rollout` unpacks three, and the `learned_mean` arm discards
    it.  A number that is never read is safer as a constant than as an
    invented estimate.
    """
    _n, _d, _shape, _low, _high, actor, _q = env_layout(env, cfg)
    _actor = ((lambda p, z: _actor_per_agent(actor, p, z)) if per_agent
              else actor.apply)

    def forward(policy, obs):
        mean, log_std = _actor(policy["actor"], obs)
        return mean, log_std, jnp.zeros(obs.shape[:-1], obs.dtype)

    return forward


def reference_env(n_agents):
    """The market's own `(reset, step, step_auto_reset, spec)` at `n_agents`.

    Built from `make_p2p_env` alone, which needs no series and no parameters, so
    a shape question can be answered without loading the panel.  The point of
    going through it rather than writing a spec dict by hand is that the action
    box then comes from the market (`action_low`/`action_high` on its own spec)
    and not from a second declaration here.
    """
    return R.make_p2p_env(n_agents, R.PI_EXP, R.PI_RET, R.DELTA)


def leaf_shapes(tree):
    """Every leaf's path and shape, as a flat dict of strings to tuples."""
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {jax.tree_util.keystr(path): tuple(int(d) for d in jnp.shape(leaf))
            for path, leaf in flat}


def parameter_layout(policy, n_agents):
    """Which layout the SAC tree that came BACK is in, and how big it is.

    `constrained_baseline.parameter_layout`'s contract and its reason:
    read off the returned tree and not off the flag that was passed in.  Both
    references are built with `jax.eval_shape`, so neither costs a random draw
    or any device work, and matching neither is a hard error rather than a third
    value.

    The discriminator that no reshaping can fake is `log_alpha`: one scalar on
    the shared path, one per participant on the per-agent path.  `sac.py` gives
    the reason -- *"each independent learner has its own entropy target to
    meet"* -- so a tree whose actor is per-agent and whose temperature is not is
    in neither layout and is refused here rather than filed with one of them.
    """
    env = reference_env(n_agents)
    cfg = config_for(1, reward_scale=1.0)
    ref = lambda pa: leaf_shapes(jax.eval_shape(
        lambda k: init_params(k, env, cfg, pa), jax.random.PRNGKey(0)))
    shared, per_agent = ref(False), ref(True)
    got = leaf_shapes(policy)
    if sorted(got) != sorted(shared):
        raise SystemExit(
            f"the SAC policy carries leaves {sorted(got)} while `init_params` "
            f"produces {sorted(shared)}; the layout cannot be read off a tree "
            f"whose leaves are not the ones this file knows")
    is_shared, is_per_agent = got == shared, got == per_agent
    if is_shared == is_per_agent:
        raise SystemExit(
            f"the SAC parameter tree is in neither layout at "
            f"n_agents={n_agents}: shapes {got}, shared reference {shared}, "
            f"per-agent reference {per_agent}")
    count = lambda d: int(sum(int(np.prod(v)) if v else 1 for v in d.values()))
    return dict(
        algo="sac",
        per_agent_params=bool(is_per_agent),
        top_level=sorted(policy),
        leaves=len(got),
        scalars=count(got),
        scalars_shared_layout=count(shared),
        #: the temperature alone, so a reader can check the one leaf that says
        #: whether each participant got its own entropy target
        log_alpha_shape=list(got["['log_alpha']"]),
        leaf_shapes={k: list(v) for k, v in sorted(got.items())},
        read_from="the policy `run` returned, not the flag passed in")


# ------------------------------------------------------------ the reward scale

def reward_scale_for(truthful_env, truthful_params, key, batch):
    """Pooled standard deviation of the per-agent reward under the truthful action.

    `sac.reward_statistics`'s quantity and its floor, fitted through this
    driver's own truthful arm: on market 04 the truthful action is delivered by
    the environment under `learner_mask = False`, so the rollout submits zeros
    and the environment replaces them with `baseline_action` of the period.  The
    same rollout structure as the learner's is used, so the two samples are the
    same kind of sample.

    Returns a Python float, which is what `SACConfig` is built from: the scale
    is frozen at construction and cannot then move with the parameters.
    """
    rollout = jax.jit(jax.vmap(
        R.make_rollout(truthful_env, truthful_params, "truthful", 0, False),
        in_axes=(None, 0)))
    keys = jax.random.split(key, batch)
    _obs, _raw, _logp, _value, reward, _cost, _term = rollout(
        R.init_policy(jax.random.PRNGKey(0)), keys)
    std = jnp.std(reward)
    return float(jnp.where(std > 1e-8, std, 1.0))


# ---------------------------------------------------------------- the learner

def make_learner(env, env_params, cfg, constrained, cost_limit, lambda_max,
                 per_agent=False):
    """`(init, iterate)` for one market and one parameter layout.

    `init(key)` returns ``(policy, learner)``; `iterate(policy, learner, lam,
    lam_step, key)` returns ``(policy, learner, new_lam, aux)`` and is jitted
    whole.  `constrained` is a Python bool closed over, so the unconstrained arm
    compiles without the multiplier's ascent at all, which is the arrangement
    `constrained_baseline.make_update` uses.

    `cost_limit` and `lambda_max` are passed in rather than imported: they are
    `constrained_baseline`'s constants, that file is the one that scans them per
    dock, and this module must not import the module that imports it.  A second
    copy of them here would be a second declaration of the constraint.
    """
    reset, _step, step_auto, spec = env
    if spec["termination"] not in _BOOTSTRAP_AT_DONE:
        raise ValueError(
            f"spec['termination']={spec['termination']!r} is not one of "
            f"{sorted(_BOOTSTRAP_AT_DONE)}")
    bootstrap_at_done = _BOOTSTRAP_AT_DONE[spec["termination"]]
    if not (cfg.reward_scale > 0.0) or not np.isfinite(cfg.reward_scale):
        raise ValueError(f"reward_scale={cfg.reward_scale!r} must be a finite "
                         f"positive number; fit it with `reward_scale_for`")
    n_agents, act_dim, act_shape, low, high, actor, qnet = env_layout(env, cfg)
    obs_dim = int(spec["obs_dim"])
    per_iter = cfg.n_envs * cfg.horizon
    if per_iter > cfg.buffer_size:
        raise ValueError(f"one iteration collects {per_iter} env-steps but the "
                         f"buffer holds {cfg.buffer_size}; the FIFO write would "
                         f"overwrite this iteration's own transitions")
    n_updates = int(round(cfg.utd_ratio * per_iter))
    if n_updates < 1:
        raise ValueError(f"utd_ratio {cfg.utd_ratio} gives {n_updates} updates "
                         f"per iteration")
    #: CleanRL: minus the action dimension, per agent
    target_entropy = -float(act_dim)
    tx_actor = optax.adam(cfg.policy_lr)
    tx_q = optax.adam(cfg.q_lr)
    tx_alpha = optax.adam(cfg.alpha_lr)

    if per_agent:
        _actor = lambda p, z: _actor_per_agent(actor, p, z)
        _q = lambda p, z, a: _q_per_agent(qnet, p, z, a)

        def _alpha_loss(log_alpha, logp):
            """Each agent's temperature answers to its own entropy: the batch
            mean is taken per agent and the agents are summed, so the gradient
            on `log_alpha[j]` sees agent `j`'s samples only."""
            per_agent_mean = jnp.mean(logp + target_entropy, axis=0)
            return jnp.sum(-jnp.exp(log_alpha) * per_agent_mean)
    else:
        _actor = actor.apply
        _q = qnet.apply

        def _alpha_loss(log_alpha, logp):
            return -jnp.exp(log_alpha) * jnp.mean(logp + target_entropy)

    def _alpha_of(policy):
        """`alpha` broadcast against a ``(..., n_agents)`` `logp`."""
        return jnp.exp(policy["log_alpha"])

    def _act(policy, obs, key):
        """One episode's sampled action at one period: ``(action, pre)``."""
        mean, log_std = _actor(policy["actor"], obs)
        pre = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape,
                                                          mean.dtype)
        return to_action(pre, low, high).reshape(act_shape), pre

    def rollout(policy, key):
        """One episode of `cfg.horizon` periods, emitting one transition each.

        Unlike `preliminary_reference.make_rollout`, the successor observation
        is emitted alongside the observation the action was chosen from, because
        the critic's target needs it and a replay sample has no neighbour in
        time to read it from -- `sac._rollout`'s reason, in its words.
        """
        key, sub = jax.random.split(key)
        obs, state = reset(sub, env_params)

        def body(carry, _):
            obs, state, key = carry
            key, a_key, s_key = jax.random.split(key, 3)
            action, pre = _act(policy, obs, a_key)
            nxt, new_state, reward, costs, done, info = step_auto(
                s_key, state, action, env_params)
            out = dict(obs=obs, pre=pre, reward=reward, cost=costs[:, 0],
                       next_obs=info["terminal_obs"], done=done)
            return (nxt, new_state, key), out

        _carry, traj = jax.lax.scan(body, (obs, state, key), None,
                                    length=cfg.horizon)
        return traj

    def _push(buffer, traj):
        """FIFO write of one iteration's ``n_envs * horizon`` env-steps."""
        n = per_iter
        idx = (buffer["cursor"] + jnp.arange(n)) % cfg.buffer_size
        flat = lambda x: x.reshape((n,) + x.shape[2:])
        out = dict(buffer)
        for k in ("obs", "pre", "reward", "cost", "next_obs", "done"):
            out[k] = buffer[k].at[idx].set(flat(traj[k]))
        out["cursor"] = (buffer["cursor"] + n) % cfg.buffer_size
        out["filled"] = jnp.minimum(buffer["filled"] + n, cfg.buffer_size)
        return out

    def _sample(buffer, key):
        idx = jax.random.randint(key, (cfg.batch_size,), 0, buffer["filled"])
        return {k: buffer[k][idx] for k in ("obs", "pre", "reward", "cost",
                                            "next_obs", "done")}

    def _cont(done, like):
        """The continuation mask, branched in Python on the boundary rule."""
        if bootstrap_at_done:
            return jnp.ones_like(like)
        if done.ndim != like.ndim - 1:
            raise ValueError(f"`done` has shape {done.shape} against a "
                             f"per-agent value of shape {like.shape}")
        return 1.0 - done[..., None].astype(like.dtype)

    def _shaped(batch, lam):
        """The reward the critic regresses on: the surrogate, not the report.

        `constrained_baseline`'s sentence holds here unchanged -- the reported
        reward stays `reward` and `lam` never touches what is measured.  What
        differs from the PPO arm is only where the subtraction happens: there it
        is applied to a fresh rollout, here to a replay sample, so the
        multiplier that shapes a transition is the one in force at the update
        and not the one in force at the collection.
        """
        return batch["reward"] - lam * batch["cost"]

    def _q_loss(q_params, policy, batch, lam, key):
        """Twin-Q regression on the soft Bellman target."""
        q1, q2 = q_params
        z, z_next = batch["obs"], batch["next_obs"]
        mean, log_std = _actor(policy["actor"], z_next)
        pre_next = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape,
                                                               mean.dtype)
        logp_next = log_prob(pre_next, mean, log_std, low, high)
        a_next = _q_action(pre_next, low, high)
        q_next = jnp.minimum(_q(policy["q1_target"], z_next, a_next),
                             _q(policy["q2_target"], z_next, a_next))
        alpha = _alpha_of(policy)
        target = (_shaped(batch, lam) / cfg.reward_scale
                  + cfg.gamma * _cont(batch["done"], q_next)
                  * (q_next - alpha * logp_next))
        target = jax.lax.stop_gradient(target)
        a = _q_action(batch["pre"], low, high)
        q1_pred, q2_pred = _q(q1, z, a), _q(q2, z, a)
        loss = jnp.mean((q1_pred - target) ** 2) + jnp.mean((q2_pred - target) ** 2)
        return loss, dict(q_loss=loss, q_mean=jnp.mean(q1_pred),
                          target_mean=jnp.mean(target))

    def _actor_loss(actor_params, policy, batch, key):
        z = batch["obs"]
        mean, log_std = _actor(actor_params, z)
        pre = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape,
                                                          mean.dtype)
        logp = log_prob(pre, mean, log_std, low, high)
        a = _q_action(pre, low, high)
        q = jnp.minimum(_q(policy["q1"], z, a), _q(policy["q2"], z, a))
        alpha = jax.lax.stop_gradient(_alpha_of(policy))
        loss = jnp.mean(alpha * logp - q)
        return loss, dict(actor_loss=loss, logp=jax.lax.stop_gradient(logp))

    def _update(policy, learner, lam, key):
        def one(carry, k):
            policy, opt_actor, opt_q, opt_alpha = carry
            k_batch, k_q, k_pi = jax.random.split(k, 3)
            batch = _sample(learner["buffer"], k_batch)
            # critic
            (_, q_aux), g_q = jax.value_and_grad(_q_loss, has_aux=True)(
                (policy["q1"], policy["q2"]), policy, batch, lam, k_q)
            upd, opt_q = tx_q.update(g_q, opt_q, (policy["q1"], policy["q2"]))
            q1, q2 = optax.apply_updates((policy["q1"], policy["q2"]), upd)
            policy = dict(policy, q1=q1, q2=q2)
            # actor, against the freshly updated critics
            (_, pi_aux), g_pi = jax.value_and_grad(_actor_loss, has_aux=True)(
                policy["actor"], policy, batch, k_pi)
            upd, opt_actor = tx_actor.update(g_pi, opt_actor, policy["actor"])
            policy = dict(policy, actor=optax.apply_updates(policy["actor"], upd))
            # temperature, on the log-probabilities the actor step sampled
            a_loss, g_a = jax.value_and_grad(_alpha_loss)(policy["log_alpha"],
                                                          pi_aux["logp"])
            upd, opt_alpha = tx_alpha.update(g_a, opt_alpha, policy["log_alpha"])
            policy = dict(policy, log_alpha=optax.apply_updates(
                policy["log_alpha"], upd))
            # targets
            polyak = lambda t, s: (1.0 - cfg.tau) * t + cfg.tau * s
            policy = dict(policy,
                          q1_target=jax.tree.map(polyak, policy["q1_target"],
                                                 policy["q1"]),
                          q2_target=jax.tree.map(polyak, policy["q2_target"],
                                                 policy["q2"]))
            aux = dict(q_loss=q_aux["q_loss"], q_mean=q_aux["q_mean"],
                       target_mean=q_aux["target_mean"],
                       actor_loss=pi_aux["actor_loss"], alpha_loss=a_loss,
                       # `-logp` is the sampled entropy, per agent
                       entropy=-jnp.mean(pi_aux["logp"]),
                       alpha=jnp.mean(jnp.exp(policy["log_alpha"])))
            return (policy, opt_actor, opt_q, opt_alpha), aux

        (policy, opt_actor, opt_q, opt_alpha), aux = jax.lax.scan(
            one, (policy, learner["opt_actor"], learner["opt_q"],
                  learner["opt_alpha"]), jax.random.split(key, n_updates))
        learner = dict(learner, opt_actor=opt_actor, opt_q=opt_q,
                       opt_alpha=opt_alpha)
        return policy, learner, jax.tree.map(jnp.mean, aux)

    def init(key):
        """The starting carry: ``(policy, learner)``."""
        policy = init_params(key, env, cfg, per_agent)
        # the buffer's leaves are shaped on ONE env-step -- one period for the
        # whole population -- and the leading axis is the capacity
        B = cfg.buffer_size
        f32 = jnp.float32
        buffer = dict(
            obs=jnp.zeros((B, n_agents, obs_dim), f32),
            pre=jnp.zeros((B, n_agents, act_dim), f32),
            reward=jnp.zeros((B, n_agents), f32),
            cost=jnp.zeros((B, n_agents), f32),
            next_obs=jnp.zeros((B, n_agents, obs_dim), f32),
            done=jnp.zeros((B,), jnp.bool_),
            cursor=jnp.asarray(0, jnp.int32),
            filled=jnp.asarray(0, jnp.int32))
        learner = dict(opt_actor=tx_actor.init(policy["actor"]),
                       opt_q=tx_q.init((policy["q1"], policy["q2"])),
                       opt_alpha=tx_alpha.init(policy["log_alpha"]),
                       buffer=buffer)
        return policy, learner

    @jax.jit
    def iterate(policy, learner, lam, lam_step, key):
        """One iteration: `n_envs` episodes, the FIFO write, `n_updates` steps.

        The multiplier's ascent is `constrained_baseline.make_update`'s, on the
        episode cost of THIS iteration's rollout rather than of the replay
        sample: it is the constraint of the policy now in force that the
        multiplier is chasing, and the buffer holds the policies of every
        earlier iteration too.
        """
        key, k_roll, k_upd = jax.random.split(key, 3)
        traj = jax.vmap(rollout, in_axes=(None, 0))(
            policy, jax.random.split(k_roll, cfg.n_envs))
        learner = dict(learner, buffer=_push(learner["buffer"], traj))
        policy, learner, aux = _update(policy, learner, lam, k_upd)

        # `constrained_baseline.update`'s three lines, unchanged: ascent on the
        # violation expressed as a multiple of the limit, clipped, and held at
        # zero on the unconstrained arm
        episode_cost = traj["cost"].sum(1).mean()
        violation = episode_cost / cost_limit - 1.0
        new_lam = jnp.where(
            constrained,
            jnp.clip(lam + lam_step * violation, 0.0, lambda_max),
            0.0)
        metrics = dict(aux)
        metrics.update(ret=traj["reward"].sum(1).mean(), cost=episode_cost,
                       lam=lam, buffer_filled=learner["buffer"]["filled"])
        return policy, learner, new_lam, metrics

    return init, iterate


# ------------------------------------------------------------------ the arm

def run(n_agents, initial_soc, constrained, seed, iterations, batch, every,
        per_agent, buffer_size, utd_ratio, build, cost_limit, lambda_step,
        lambda_max):
    """Train on the training starts, score the curve on the held-out ones.

    The curve is evaluated every `every` iterations and at the last one, so the
    evaluated points -- and the `final_eval_return` the driver averages over the
    last three of them -- depend on `every` (`--eval-every`), as on the PPO arm.

    The same contract as `constrained_baseline.run`, and the same sentence
    holds: `per_agent` is the only thing that separates the two parameter
    layouts, and the policy that comes back is what says which layout actually
    ran.  `build` is passed in rather than imported so that this module does not
    import the driver that imports it.

    The curve rows carry SAC's own diagnostics beside the two fields the PPO
    arm writes.  They are NOT the PPO arm's diagnostics under other names:
    `entropy` here is `-log pi` of the sampled action and moves with the
    automatic temperature, where the PPO arm's is a closed-form surrogate on a
    scheduled width, and `sac.py` records that aligning one coefficient does not
    align the two learners.
    """
    params, env, env_eval = build(n_agents, initial_soc)
    truth_params, truth_env, _truth_eval = build(
        n_agents, initial_soc, np.zeros(n_agents, bool))

    key = jax.random.PRNGKey(seed)
    key, scale_key, init_key = jax.random.split(key, 3)
    scale = reward_scale_for(truth_env, truth_params, scale_key, batch)
    cfg = config_for(batch, reward_scale=scale, buffer_size=buffer_size,
                     utd_ratio=utd_ratio)
    init, iterate = make_learner(env, params, cfg, constrained, cost_limit,
                                 lambda_max, per_agent)
    policy, learner = init(init_key)
    lam = jnp.float32(0.0)
    greedy = greedy_forward(env, cfg, per_agent)

    eval_key = jax.random.PRNGKey(20_000 + n_agents)
    evaluate = lambda p: R.evaluate(env_eval, params, "learned_mean", p,
                                    eval_key, 64, per_agent=per_agent,
                                    forward_fn=greedy)
    curve = []
    if iterations == 0:
        # the untrained control, the same branch and the same reason as the PPO
        # arm's: on the iteration count, not on `curve` being empty
        ret, cost = evaluate(policy)
        curve.append(dict(iteration=-1, train_return=None, train_cost=None,
                          lam=float(lam), eval_return=float(ret.mean()),
                          eval_cost=float(cost.mean())))
    for iteration in range(iterations):
        key, sub = jax.random.split(key)
        frac = iteration / max(iterations - 1, 1)
        policy, learner, lam, aux = iterate(
            policy, learner, lam,
            jnp.float32(lambda_step * (1.0 - frac)), sub)
        row = dict(iteration=iteration, train_return=float(aux["ret"]),
                   train_cost=float(aux["cost"]), lam=float(aux["lam"]),
                   q_loss=float(aux["q_loss"]), q_mean=float(aux["q_mean"]),
                   target_mean=float(aux["target_mean"]),
                   actor_loss=float(aux["actor_loss"]),
                   alpha_loss=float(aux["alpha_loss"]),
                   alpha=float(aux["alpha"]), entropy=float(aux["entropy"]),
                   buffer_filled=int(aux["buffer_filled"]))
        if iteration % every == 0 or iteration == iterations - 1:
            ret, cost = evaluate(policy)
            row.update(eval_return=float(ret.mean()), eval_cost=float(cost.mean()))
        curve.append(row)
        if not np.isfinite(row["train_return"]):
            raise SystemExit(f"non-finite at iteration {iteration}")
    return curve, policy, env_eval, params, cfg
