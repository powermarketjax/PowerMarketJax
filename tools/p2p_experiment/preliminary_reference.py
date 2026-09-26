"""Preliminary reference scores for the P2P market: three arms, several seeds.

Not part of the package, not run by CI, and **not the publishable reference
score**.  This project does not build a training framework:
adapters to JaxMARL and PureJaxRL belong in `powermarketjax/wrappers/`, and that
directory is empty.  The learner below is written here so that one question can
be answered before that work is done, namely whether a learned policy beats
truthful bidding at all.  If it does not, the claim that this environment is a
non-trivial learning problem fails, and the wrapper would have been written for
nothing.  A published score has to come from the framework, not from this file.

Three arms are compared, and the middle one is the load-bearing comparison:

* random, actions drawn uniformly over the action space, which bounds from below;
* truthful, every participant bidding its opportunity cost, which §4 derives as
  the export price for a seller and the retail tariff for a buyer.  This is the
  game-theoretic reference point rather than a naive arm, and a naive arm is
  not to be treated as strength-equivalent;
* self-consumption, the truthful price of the arm above together with a battery
  run to absorb the premises' own surplus and cover its own deficit, which is
  what a household controller does in the absence of a market;
* learned, proximal policy optimisation with parameters shared across
  participants.

The self-consumption arm exists because the truthful arm holds the battery still.
`baseline_action` submits the truthful price and a zero battery command, so
comparing a learned policy against it would credit the learner for merely
switching the battery on, and a naive arm is not to be treated as
strength-equivalent.  The heuristic reads only the observation, so it is a policy
and not an oracle: the rated power, the state of charge, the injection and the
offtake are channels 0, 5, 6 and 7 of §9.4.

Parameters are shared rather than independent.  Sharing is what independent
proximal policy optimisation does for symmetric agents, and at 1 200
participants an independent-parameter learner gives each agent one part in 1 200
of the data, so the two settings are not comparable at equal sample budget.
A separate trial script carries the independent-parameter variant; sharing is
what a reference score should report.

**The independent layout is now also available here, and it is off by default.**
`init_policy_per_agent` and `forward_per_agent` are the two functions that
differ, and every caller binds one or the other AT CONSTRUCTION -- `make_rollout`
and `evaluate` take `per_agent` and hand the chosen function down, so the shared
arm remains the expression it always was rather than that expression with an
axis moved by zero.  That is `powermarketjax.learning.ippo`'s arrangement and it
is copied deliberately: the two out-of-package arms have to differ in the
parameter layout and in nothing else, or the contrast is not a contrast.
`main` below is untouched and still reports the shared layout;
`constrained_baseline --per-agent-params` is what runs the other one.

**Truncation is handled rather than ignored.**  `done` is a time limit and not a
terminal state, so the value target on the last step of an episode bootstraps
from the true successor observation, which the environment supplies in
`info["terminal_obs"]` because auto-reset overwrites the returned one.  Handing
this to the algorithm was an untested assumption before a
learner existed; the bootstrap below is where it stops being untested.

Statistics follow the project's reporting convention: the interquartile
mean over seeds with a bootstrap confidence interval, rather than a mean with a
standard deviation.

    PYTHONPATH=tools/p2p_experiment python -m preliminary_reference \\
        --agents 16 100 1200 --seeds 5
"""
import argparse
import json
import math
import time
from pathlib import Path

import distrax
import jax
import jax.numpy as jnp
import rlax
import numpy as np

from powermarketjax.envs.p2p import (load_fluvius_households, make_p2p_env,
                                     make_p2p_params)
from powermarketjax.resources.battery import make_battery_bundle

#: The scenario as fixed in §15 of the specification.
PI_EXP, PI_RET = 73.0, 333.4
DELTA = 0.25
KAPPA = 13.88
EPISODE_LEN = 96
OBS_DIM, ACTION_DIM = 15, 2
#: Periods of own injection and offtake appended to the observation in the
#: forecast condition.  Eight quarter hours is two hours, which is about one
#: charge or discharge of a battery whose rating is its capacity over 2.1 h, so
#: it is the horizon over which a plan could differ from a reaction.
FORECAST_PERIODS = 8

#: Learner settings.  Not tuned; a reference score that depended on tuning this
#: file would not be reproducible from the framework later.
HIDDEN = 64
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 1e-3
EPOCHS = 4
LEARNING_RATE = 3e-4
LOG_STD_INIT = -0.5


def augment(obs, state, env_params, horizon):
    """Append the participant's own future injection and offtake.

    **This is built in the experiment and not in the environment.**  §9.4 fixes
    the fifteen channels and the observation is fixed, so a forecast
    channel would be a change to the market specification.  Adding it here
    instead makes the condition an ablation that bears on whether that change is
    warranted, rather than the change itself.  Nothing under
    `powermarketjax/envs` is touched.
    """
    if horizon == 0:
        return obs
    n_periods = env_params.p_pv.shape[0]
    start = jnp.minimum(state.cursor + 1, n_periods - horizon)
    future_pv = jax.lax.dynamic_slice(
        env_params.p_pv, (start, 0), (horizon, env_params.p_pv.shape[1]))
    future_load = jax.lax.dynamic_slice(
        env_params.load, (start, 0), (horizon, env_params.load.shape[1]))
    return jnp.concatenate(
        [obs, future_pv.T, future_load.T], axis=-1).astype(jnp.float32)


def build(n_agents, learner_mask):
    series = load_fluvius_households(n_households=n_agents)
    one_way = math.sqrt(0.85)
    battery = make_battery_bundle(
        n_devices=n_agents, capacity_mwh=0.011, power_mw=0.011 / 2.1,
        eta_charge=one_way, eta_discharge=one_way, soc_min=0.15, soc_max=1.0,
        initial_soc=0.5, dt_hours=DELTA, cycle_cost_per_mwh=0.0)
    params = make_p2p_params(
        p_pv=series.injection, load=series.offtake, battery=battery,
        kappa=np.full(n_agents, KAPPA, np.float32),
        learner_mask=learner_mask, episode_len=EPISODE_LEN)
    return series, params, make_p2p_env(n_agents, PI_EXP, PI_RET, DELTA)


# ------------------------------------------------------------------ the policy

def init_policy(key, obs_dim=OBS_DIM):
    """One shared network: two hidden layers, a Gaussian head and a value head."""
    keys = jax.random.split(key, 5)
    glorot = lambda k, shape: (jax.random.normal(k, shape, jnp.float32)
                               * np.sqrt(2.0 / shape[0]))
    return dict(
        w1=glorot(keys[0], (obs_dim, HIDDEN)), b1=jnp.zeros((HIDDEN,), jnp.float32),
        w2=glorot(keys[1], (HIDDEN, HIDDEN)), b2=jnp.zeros((HIDDEN,), jnp.float32),
        wm=glorot(keys[2], (HIDDEN, ACTION_DIM)) * 0.01,
        bm=jnp.zeros((ACTION_DIM,), jnp.float32),
        wv=glorot(keys[3], (HIDDEN, 1)) * 0.01, bv=jnp.zeros((1,), jnp.float32),
        log_std=jnp.full((ACTION_DIM,), LOG_STD_INIT, jnp.float32))


def init_policy_per_agent(key, n_agents, obs_dim=OBS_DIM):
    """`n_agents` independent copies of `init_policy`, one per participant.

    Every leaf of the shared tree gains a leading ``n_agents`` axis and the
    number of leaves is unchanged, which is the same shape relation
    `powermarketjax.learning.ippo` produces under `per_agent_params=True`.  The
    keys are `n_agents` splits of one key, so participant `j`'s network is drawn
    from split `j` and two participants never start identical.

    The four zero-initialised biases do not depend on the key, and `vmap`
    broadcasts them along the output axis rather than dropping it, so they too
    come back with the leading axis; `tests/tools/test_p2p_external_per_agent_l0`
    asserts that on the scalar count rather than leaving it to this sentence.
    """
    n_agents = int(n_agents)
    return jax.vmap(lambda k: init_policy(k, obs_dim))(
        jax.random.split(key, n_agents))


def forward(policy, obs):
    """`obs` is ``(..., OBS_DIM)``; returns the mean, the log std and the value."""
    hidden = jnp.tanh(obs @ policy["w1"] + policy["b1"])
    hidden = jnp.tanh(hidden @ policy["w2"] + policy["b2"])
    mean = hidden @ policy["wm"] + policy["bm"]
    value = (hidden @ policy["wv"] + policy["bv"])[..., 0]
    return mean, policy["log_std"], value


def forward_per_agent(policy, obs):
    """`forward` with one network per participant, lined up on the agent axis.

    `obs` is ``(..., n_agents, OBS_DIM)`` -- the agent axis is the second to
    last everywhere on this path, because `reset` returns ``(n_agents,
    OBS_DIM)`` and the rollout's `scan` and the batch `vmap` only ever prepend
    axes.  Every parameter leaf carries a leading ``n_agents`` axis.  The agent
    axis is moved to the front, mapped against the parameters, and moved back,
    which is `ippo._apply_per_agent` transcribed onto this file's parameter
    dict.

    Crossing the two layouts is caught by `vmap` rather than by a check here: a
    shared `w1` has a leading axis of `OBS_DIM`, so mapping it against
    `n_agents` observations raises on inconsistent sizes -- except where those
    two numbers coincide, which is why `per_agent` is a keyword the caller
    states and not something inferred from the leaf shapes.
    """
    zz = jnp.moveaxis(obs, -2, 0)
    mean, log_std, value = jax.vmap(forward)(policy, zz)
    return jnp.moveaxis(mean, 0, -2), log_std, jnp.moveaxis(value, 0, -1)


def policy_forward(per_agent):
    """The forward pass for one parameter layout, chosen once at construction.

    Returned rather than branched on inside the rollout so the shared arm is
    `forward` itself and not a wrapper that happens to be the identity.
    """
    return forward_per_agent if per_agent else forward


def distribution(mean, log_std):
    """A tanh-squashed diagonal Gaussian, from `distrax` rather than by hand.

    The log density of a squashed Gaussian is where the hand-written version of
    this file went wrong twice: first by inverting the squash to recover the
    pre-squash sample, which diverges once an action saturates, and then by
    keeping a correction term that `distrax.Transformed` applies for us.  The
    distribution and the two losses below are the parts a published library
    should supply; this project does not build a training framework, and while
    JaxMARL cannot be installed here (it pins `jax<=0.4.38` against this
    repository's 0.10.2), `distrax` and `rlax` require only `jax>=0.7.0` and so
    supply the algorithm's mathematics without disturbing the environment.
    """
    normal = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
    return distrax.Transformed(normal, distrax.Block(distrax.Tanh(), 1))


def log_prob(mean, log_std, raw):
    """Density of the squashed Gaussian, evaluated at the pre-squash sample.

    **The pre-squash sample is what the rollout stores, and inverting the squash
    instead would be a defect rather than a shortcut.**  Once an action saturates,
    and here it does, ``arctanh`` of it diverges and has to be clipped, so the
    recovered value bears no relation to the value that was drawn; the importance
    ratio is then wrong in exactly the region the policy is moving into, and the
    gradient drives it further in.  Measured before the fix: the clipped magnitude
    of the battery command sat at 0.9 of full scale for every setting of the
    entropy bonus and the learning rate that was tried.
    """
    return distribution(mean, log_std).log_prob(raw)


def sample(policy, obs, key, forward_fn=forward):
    mean, log_std, value = forward_fn(policy, obs)
    dist = distribution(mean, log_std)
    action, logp = dist.sample_and_log_prob(seed=key)
    return action, action, logp, value


# ------------------------------------------------------------------- rollouts

def make_rollout(env, env_params, mode, horizon=0, per_agent=False,
                 forward_fn=None):
    """One episode. `mode` selects the arm: learned, random or truthful.

    `per_agent` says which parameter layout `policy` is in, and is used only by
    the two arms that consult it -- `random`, `truthful` and `self_consumption`
    read no parameters at all, so their episodes are the same episodes under
    either setting.

    `forward_fn` overrides the forward pass for a `policy` this module did not
    build.  It exists for `sac_arm`, whose parameter tree is six flax trees
    rather than this file's nine arrays, and whose greedy action is nonetheless
    the same expression -- `tanh(mean)` on a box of `[-1, 1]`.  **When it is
    `None` the expression evaluated is `policy_forward(per_agent)` itself**, the
    same object the two arms have always used, so the PPO arm is not this
    expression with a wrapper around it.
    """
    reset, _, step_auto, _ = env
    _forward = policy_forward(per_agent) if forward_fn is None else forward_fn

    def rollout(policy, key):
        key, sub = jax.random.split(key)
        raw_obs, state = reset(sub, env_params)
        obs = augment(raw_obs, state, env_params, horizon)

        def body(carry, _):
            obs, state, key = carry
            key, a_key, s_key = jax.random.split(key, 3)
            if mode == "learned":
                action, raw, logp, value = sample(policy, obs, a_key, _forward)
            elif mode == "learned_mean":
                # Evaluation at the mean of the distribution rather than a draw
                # from it.  With a state-independent log standard deviation the
                # exploration noise is a policy property at evaluation time as
                # well, and it costs degradation on every period it moves the
                # battery, so reporting only the sampled score confounds the
                # policy with its exploration.
                mean, _, value = _forward(policy, obs)
                action = jnp.tanh(mean)
                logp = jnp.zeros(obs.shape[:-1], jnp.float32)
                raw = mean
            elif mode == "random":
                action = jax.random.uniform(a_key, obs.shape[:-1] + (ACTION_DIM,),
                                            jnp.float32, -1.0, 1.0)
                raw = action
                logp = jnp.zeros(obs.shape[:-1], jnp.float32)
                value = jnp.zeros(obs.shape[:-1], jnp.float32)
            elif mode == "self_consumption":
                # channels 0, 6, 7 of §9.4: rated power, injection, offtake
                rated = jnp.maximum(obs[..., 0], 1e-12)
                surplus = obs[..., 6] - obs[..., 7]
                battery_cmd = jnp.clip(-surplus / rated, -1.0, 1.0)
                # the truthful price for the side the position falls on with the
                # battery idle, which is the rule `baseline_action` applies
                price_cmd = jnp.where(surplus >= 0.0, -1.0, 1.0)
                action = jnp.stack([battery_cmd, price_cmd], axis=-1)
                raw = action
                logp = jnp.zeros(obs.shape[:-1], jnp.float32)
                value = jnp.zeros(obs.shape[:-1], jnp.float32)
            else:                        # truthful: the mask replaces the action
                action = jnp.zeros(obs.shape[:-1] + (ACTION_DIM,), jnp.float32)
                raw = action
                logp = jnp.zeros(obs.shape[:-1], jnp.float32)
                value = jnp.zeros(obs.shape[:-1], jnp.float32)
            nxt, new_state, reward, costs, done, info = step_auto(
                s_key, state, action, env_params)
            # the successor observation is augmented from the successor state
            nxt = augment(nxt, new_state, env_params, horizon)
            terminal = augment(info["terminal_obs"], new_state, env_params,
                               horizon)
            return (nxt, new_state, key), (obs, raw, logp, value, reward,
                                           costs[:, 0], terminal)

        _, out = jax.lax.scan(body, (obs, state, key), None, length=EPISODE_LEN)
        return out

    return rollout


def advantages(reward, value, last_value):
    """Generalised advantage estimation, from `rlax`, per participant.

    The bootstrap value at the end is the one the time limit requires: `done` is
    a truncation, so the return continues past it and the value of the true
    successor observation stands in for the rest.
    """
    values = jnp.concatenate([value, last_value[None]], axis=0)
    discounts = jnp.full_like(reward, GAMMA)
    gae = jax.vmap(
        rlax.truncated_generalized_advantage_estimation,
        in_axes=(1, 1, None, 1), out_axes=1)(
            reward, discounts, GAE_LAMBDA, values)
    return gae, gae + value


def make_update(env, env_params, batch, horizon=0):
    import optax
    rollout = make_rollout(env, env_params, "learned", horizon)
    optimiser = optax.chain(optax.clip_by_global_norm(0.5),
                            optax.adam(LEARNING_RATE))

    def collect(policy, key):
        keys = jax.random.split(key, batch)
        obs, raw, logp, value, reward, cost, terminal = jax.vmap(
            rollout, in_axes=(None, 0))(policy, keys)
        # the successor of the truncated step, which auto-reset overwrote
        _, _, last_value = forward(policy, terminal[:, -1])
        gae, target = jax.vmap(advantages)(reward, value, last_value)
        return dict(obs=obs, raw=raw, logp=logp, gae=gae, target=target,
                    reward=reward, cost=cost)

    def loss_fn(policy, data):
        mean, log_std, value = forward(policy, data["obs"])
        logp = log_prob(mean, log_std, data["raw"])
        adv = data["gae"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        policy_loss = rlax.clipped_surrogate_pg_loss(
            jnp.exp(logp - data["logp"]).ravel(), adv.ravel(), CLIP_EPS)
        value_loss = ((value - data["target"]) ** 2).mean()
        entropy = -logp.mean()
        return policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

    @jax.jit
    def update(policy, opt_state, key):
        key, collect_key = jax.random.split(key)
        data = collect(policy, collect_key)

        def epoch(carry, _):
            policy, opt_state = carry
            grads = jax.grad(loss_fn)(policy, data)
            updates, opt_state = optimiser.update(grads, opt_state)
            return (optax.apply_updates(policy, updates), opt_state), None

        (policy, opt_state), _ = jax.lax.scan(
            epoch, (policy, opt_state), None, length=EPOCHS)
        return policy, opt_state, dict(
            ret=data["reward"].sum(1), cost=data["cost"].sum(1))

    return optimiser, update


def evaluate(env, env_params, mode, policy, key, episodes, horizon=0,
             chunk=64, per_agent=False, forward_fn=None):
    """Episode returns and constraint totals, in chunks so memory stays bounded.

    At 1 200 participants one chunk of 64 episodes already holds 64 x 96 x 1 200
    observations; evaluating 512 in one call would not fit, and an unattended run
    that dies on memory produces nothing.

    `forward_fn` is passed straight through to `make_rollout`; see there.
    """
    rollout = jax.jit(jax.vmap(
        make_rollout(env, env_params, mode, horizon, per_agent, forward_fn),
        in_axes=(None, 0)))
    rewards, costs = [], []
    for start in range(0, episodes, chunk):
        take = min(chunk, episodes - start)
        keys = jax.random.split(jax.random.fold_in(key, start), take)
        _, _, _, _, reward, cost, _ = rollout(policy, keys)
        rewards.append(np.asarray(reward.sum(1)))
        costs.append(np.asarray(cost.sum(1)))
    return np.concatenate(rewards), np.concatenate(costs)


# ------------------------------------------------------------------ statistics

def iqm(values):
    """Interquartile mean: the mean of the middle half of a sample.

    The sample is the set of seeds, one score each.  Pooling seeds with
    evaluation episodes would inflate the sample and narrow the interval below
    what the number of independent runs supports.
    """
    flat = np.sort(np.asarray(values).ravel())
    lo, hi = int(0.25 * len(flat)), int(np.ceil(0.75 * len(flat)))
    hi = max(hi, lo + 1)
    return float(flat[lo:hi].mean())


def bootstrap_ci(values, draws=5000, level=0.95, seed=0):
    rng = np.random.default_rng(seed)
    flat = np.asarray(values).ravel()
    stats = [iqm(rng.choice(flat, len(flat), replace=True)) for _ in range(draws)]
    return (float(np.percentile(stats, 100 * (1 - level) / 2)),
            float(np.percentile(stats, 100 * (1 + level) / 2)))


def stamp():
    return time.strftime("%H:%M:%S")


def run_condition(n_agents, horizon, args, eval_key):
    """Every arm at one population and one observation condition."""
    obs_dim = OBS_DIM + 2 * horizon
    per_seed, per_seed_mean, per_seed_clip = [], [], []
    per_agent_mean = []
    for seed in range(args.seeds):
        _, env_params, env = build(n_agents, np.ones(n_agents, bool))
        optimiser, update = make_update(env, env_params, args.batch, horizon)
        key = jax.random.PRNGKey(seed)
        key, init_key = jax.random.split(key)
        policy = init_policy(init_key, obs_dim)
        opt_state = optimiser.init(policy)
        for iteration in range(args.iterations):
            key, sub = jax.random.split(key)
            policy, opt_state, aux = update(policy, opt_state, sub)
            if not np.isfinite(np.asarray(aux["ret"])).all():
                raise SystemExit(
                    f"non-finite return at N={n_agents} horizon={horizon} "
                    f"seed={seed} iteration={iteration}")
        ret, cost = evaluate(env, env_params, "learned", policy, eval_key,
                             args.eval_episodes, horizon)
        ret_m, cost_m = evaluate(env, env_params, "learned_mean", policy,
                                 eval_key, args.eval_episodes, horizon)
        per_seed.append(float(ret.mean()))
        per_seed_mean.append(float(ret_m.mean()))
        per_seed_clip.append(float(cost_m.mean()))
        per_agent_mean.append(ret_m.mean(0))
        print(f"  [{stamp()}] N={n_agents:5d} h={horizon} seed={seed:2d}  "
              f"sampled {ret.mean():+.4f}  at the mean {ret_m.mean():+.4f}  "
              f"clip {cost_m.mean():7.3f}", flush=True)

    _, env_params, env = build(n_agents, np.ones(n_agents, bool))
    arms = {}
    for label, mode, target_env, target_params in (
            ("random", "random", env, env_params),
            ("self-cons", "self_consumption", env, env_params)):
        ret, cost = evaluate(target_env, target_params, mode,
                             init_policy(jax.random.PRNGKey(0), obs_dim),
                             eval_key, args.eval_episodes, horizon)
        arms[label] = dict(score=float(ret.mean()),
                           episodes=ret.mean(1), clip=float(cost.mean()),
                           per_agent=ret.mean(0))
    _, truth_params, truth_env = build(n_agents, np.zeros(n_agents, bool))
    ret, cost = evaluate(truth_env, truth_params, "truthful",
                         init_policy(jax.random.PRNGKey(0), obs_dim),
                         eval_key, args.eval_episodes, horizon)
    arms["truthful"] = dict(score=float(ret.mean()), episodes=ret.mean(1),
                            clip=float(cost.mean()), per_agent=ret.mean(0))

    # Every row carries the iteration count it was produced at.  `--iterations 0`
    # is the untrained control -- the training loop above does not execute and
    # the evaluation below it does -- and a product whose only record of that is
    # its file name cannot be read back years later.
    rows = []
    for label, info in arms.items():
        lo, hi = bootstrap_ci(info["episodes"], seed=1)
        rows.append(dict(n_agents=n_agents, horizon=horizon, arm=label,
                         iqm=iqm(info["episodes"]), ci_lo=lo, ci_hi=hi,
                         clip=info["clip"], sample="episodes",
                         n_sample=len(info["episodes"]),
                         iterations=args.iterations,
                         untrained_baseline=args.iterations == 0))
    for label, values, clips in (("learned", per_seed, per_seed_clip),
                                 ("learned@mean", per_seed_mean, per_seed_clip)):
        lo, hi = bootstrap_ci(values, seed=2)
        rows.append(dict(n_agents=n_agents, horizon=horizon, arm=label,
                         iqm=iqm(values), ci_lo=lo, ci_hi=hi,
                         clip=float(np.mean(clips)), sample="seeds",
                         n_sample=len(values),
                         iterations=args.iterations,
                         untrained_baseline=args.iterations == 0))
    per_agent = {label: info["per_agent"] for label, info in arms.items()}
    per_agent["learned@mean"] = np.mean(np.stack(per_agent_mean), axis=0)
    return rows, per_agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, nargs="+", default=[16, 100, 1200])
    parser.add_argument("--horizons", type=int, nargs="+", default=[0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--eval-episodes", type=int, default=512)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    print(f"[{stamp()}] device {jax.devices()[0]}  agents {args.agents}  "
          f"horizons {args.horizons}  seeds {args.seeds}  "
          f"iterations {args.iterations}  batch {args.batch}  "
          f"eval episodes {args.eval_episodes}", flush=True)

    rows, per_agent_by_case = [], {}
    for horizon in args.horizons:
        for n_agents in args.agents:
            started = time.perf_counter()
            eval_key = jax.random.PRNGKey(10_000 + n_agents)
            got, per_agent = run_condition(n_agents, horizon, args, eval_key)
            rows.extend(got)
            per_agent_by_case[(horizon, n_agents)] = per_agent
            print(f"  [{stamp()}] N={n_agents} h={horizon} done in "
                  f"{time.perf_counter() - started:.0f} s", flush=True)
            if args.out:
                args.out.write_text(json.dumps(
                    [{k: v for k, v in r.items()} for r in rows], indent=1))

    print(f"\n{'N':>6} {'h':>2} {'arm':<13} {'IQM':>10} {'95% CI':>20} "
          f"{'clip':>9} {'sample':>9}")
    for row in rows:
        print(f"{row['n_agents']:>6} {row['horizon']:>2} {row['arm']:<13} "
              f"{row['iqm']:>+10.4f} [{row['ci_lo']:+8.4f},{row['ci_hi']:+8.4f}] "
              f"{row['clip']:>9.3f} {row['sample']:>9}")

    print("\ngap of the learned policy against the two reference arms:")
    for horizon in args.horizons:
        for n_agents in args.agents:
            got = {r["arm"]: r for r in rows
                   if r["n_agents"] == n_agents and r["horizon"] == horizon}
            print(f"  N={n_agents:5d} h={horizon}  against truthful "
                  f"{got['learned@mean']['iqm'] - got['truthful']['iqm']:+8.4f}"
                  f"   against self-consumption "
                  f"{got['learned@mean']['iqm'] - got['self-cons']['iqm']:+8.4f}")

    # Market power, on a fixed population rather than on population means.  The
    # ladder is nested, so the households of the smallest run appear in every
    # larger one, and comparing their own returns across N holds the population
    # fixed while the market thickens.
    smallest = min(args.agents)
    print(f"\nthe {smallest} households of the smallest run, followed across N "
          f"(population fixed, market thickness varying):")
    from powermarketjax.envs.p2p import load_fluvius_households
    base = load_fluvius_households(n_households=smallest).households.tolist()
    for horizon in args.horizons:
        for arm in ("truthful", "self-cons", "learned@mean"):
            line = []
            for n_agents in args.agents:
                ids = load_fluvius_households(
                    n_households=n_agents).households.tolist()
                where = [ids.index(h) for h in base]
                line.append(f"N={n_agents}: "
                            f"{per_agent_by_case[(horizon, n_agents)][arm][where].mean():+.4f}")
            print(f"  h={horizon} {arm:<13} " + "   ".join(line))

    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
