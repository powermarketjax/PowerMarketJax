"""Constrained against unconstrained learning on the P2P market, with curves.

Not part of the package and not run by CI, for the reasons `preliminary_reference`
states: this project does not build a training framework, JaxMARL cannot be
installed against this repository's `jax` version, and the algorithm's
distribution and losses therefore come from `distrax` and `rlax` while the loop
stays local.  This file adds the arm that file lacks.

**What the two arms are.**  The environment reports a second channel besides the
reward, the magnitude by which the battery command was cut back by the physical
envelope of §3.2, and that channel must never enter the
reward.  The unconstrained arm maximises the reward alone; the constrained arm
maximises the reward less a multiplier times that channel.  Both keep the
reported reward unshaped, so both satisfy that rule, which constrains what
the environment reports and not what an algorithm optimises internally.

**Two recorded findings change how anything quoted from here should be
read.**

*Both arms ran at one shared step size of 1e-4 for a long time, and that value
had never been scanned.*  It is fifty times the optimum of the unconstrained arm
and ten times the optimum of the Lagrangian arm, and the two arms lose 0.46 and
0.18 of return at it.  The conclusion an earlier analysis drew from
those runs -- that unconstrained PPO cannot reach the optimum of its own
objective -- is a property of the unscanned default rather than of the algorithm:
scanned, that arm reaches the best open-loop constant.  Every constant below is
therefore a placeholder the calibration overrides, and a run that leaves them at
these values is not a run any reported number came from.

*Episodes come from a training pool or a held-out pool, never from the whole
panel.*  `make_start_pools` holds out three consecutive days in every fifteen and
admits a training start only if its whole episode misses every held-out period.
The buffer is what removes the leak, because `reset` draws a start at any period
rather than at a day boundary.  `build` returns both environments and `run`
trains on one and scores on the other, so a reported number is a number on days
no update ever saw.

**Four settings differ from `preliminary_reference`, and the first is a defect
rather than a preference.**  Its observations are unscaled, and §9.4's fifteen
channels span five orders of magnitude, so the first layer saturates at
initialisation; see `observation_scale` for the measurement and the consequence.
The other three -- a decayed step size, a decayed exploration width and a
multiplier that ascends on the relative violation -- are what turn a run that
oscillated for its whole length into one that settles, and each carries the
measurement it was chosen from at its own constant.  A score from this file is
therefore not comparable with one from that file, which is why the reference
scores are not restated here.

**`--per-agent-params` gives every participant its own network.**  The switch is
here, on the driver the published learning numbers came from, and not only on
the in-package driver: two arms that differ in the parameter layout AND in which
program produced them are not a contrast in the layout.  The flag name
and its meaning are `powermarketjax.learning.ippo`'s, so the two paths can be
read against each other -- every parameter leaf gains a leading `n_agents` axis,
the number of leaves is unchanged, and `preliminary_reference.forward_per_agent`
lines that axis up against the agent axis of the observation.  Two consequences
are recorded rather than left to be discovered:

* `optax.clip_by_global_norm` bounds the norm of the WHOLE tree, so with one
  network per participant it clips the `n_agents` gradients jointly.  That is a
  property of the optimiser chain and is left in place, for the reason
  `ippo.py`'s docstring gives: replacing it would make the two arms differ twice.
* the exploration width is a schedule this file overwrites before every update,
  and it is written at the shape the policy already carries, so under this flag
  every participant gets its own `log_std` leaf rather than one shared row.

A run under the flag is stamped from the parameter tree the run RETURNED, never
from the boolean this file passed down; "I sent the flag" and "the learner
received it" are two claims and only the second one is worth recording.

    PYTHONPATH=tools/p2p_experiment python -m constrained_baseline \\
        --agents 16 --seeds 5 --iterations 400 --eval-every 5 \\
        --initial-soc 0.5 0.15 --out curves.json

`--eval-every` sets the resolution of the curve, **and it also moves reported
numbers**: `final_eval_return` and `final_eval_cost` are the means of the last
three evaluated curve points (`tail` in `main`), so which iterations those are
depends on `--eval-every`, and the summary table printed at the end of a run
reads those two fields.
"""
import argparse
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import preliminary_reference as R
import sac_arm
from powermarketjax.envs.p2p import make_p2p_env, make_p2p_params
from powermarketjax.resources.battery import make_battery_bundle

#: Constraint limit per episode.  A placeholder: every reported run sets this
#: from the calibration, which scans it per initial state of charge because the
#: value that suits one dock does not suit the other.  What the scans found is
#: that the limit is inert wherever the multiplier reaches zero -- at 2.0 and
#: above at the lower dock the arm returns exactly what the unconstrained arm
#: returns, which is what an inactive constraint must do.
COST_LIMIT = 6.0
#: Step size for the multiplier, on the constraint violation expressed as a
#: multiple of the limit rather than in its own units.  A placeholder, scanned
#: per dock like the limit.  Both ends of its range fail in opposite ways: too
#: small and the multiplier never rises enough to enforce anything, too large and
#: it overshoots and buys the constraint at the cost of the return.
LAMBDA_STEP = 2e-3
#: A backstop only.  It is far above the point at which the surrogate becomes
#: all-constraint: at the unconstrained operating point (return -0.9, constraint
#: total 57) the two terms are already equal at a multiplier of 0.016, and the
#: multiplier never exceeded 0.243 in any run here, so this clip never binds.
LAMBDA_MAX = 50.0
#: A placeholder, and the one that mattered most.  Scanned per arm and per dock:
#: the two arms differ by a factor of five at the same dock, and this value is
#: fifty times what the unconstrained arm wants.  What the step size controls is
#: how far the policy travels from its initialisation, and the step size and the
#: iteration count act only through their product to within a thirtieth of a seed
#: standard deviation, so a run that travels too far lands in the saturated
#: command the envelope refuses, which costs nothing in the reward and is
#: therefore not a place a reward-only gradient leaves.
LEARNING_RATE = 1e-4
#: The exploration width, decayed linearly rather than learned.
#:
#: `preliminary_reference` leaves it a free parameter, and it does not move: it
#: went from -0.579 to -0.603 over 600 iterations there, so the policy explores
#: as widely at the end as at the start.  That makes the training return and the
#: evaluation return two different quantities -- the first is scored under the
#: noise and the second at the mean of the distribution -- and they then move in
#: opposite directions, the unconstrained arm's training return rising from -4.25
#: to -1.20 over the same 300 iterations in which its evaluation return fell from
#: -0.48 to -0.91.  Neither number is wrong, but a run reported by one of them is
#: not described by the other, so the width is decayed to a twentieth of the
#: action range.  That removes the scoring rule as the source of the gap and not
#: the gap itself: at the final iteration the two series still differ by 0.17 to
#: 0.32, of which the scoring rule accounts for 0.008 in the unconstrained arm
#: and 0.116 in the constrained one.  The remainder is the episode sample, since
#: the training return is scored on 16 freshly drawn episodes each iteration and
#: the evaluation return on one fixed set of 64, and that term does not shrink
#: with the schedule.
LOG_STD_START, LOG_STD_END = -0.5, -3.0


def observation_scale(env_params, n_agents):
    """Per-channel divisors for the fifteen channels of §9.4.

    **Without this the network is blind to fourteen of the fifteen channels.**
    The channels are not commensurate: the previous clearing price and the
    degradation cost are tens to hundreds of EUR/MWh, the injection, the offtake
    and the net position are thousandths of a megawatt at household scale, and
    the state of charge and the two efficiencies are fractions -- a span of five
    orders of magnitude.  The first layer is `tanh` with Glorot weights, so a
    channel of order 300 gives a pre-activation of order 100 and saturates every
    unit of that layer at initialisation, which both destroys the information in
    the small channels and leaves almost no gradient anywhere.  Measured on
    2026-08-15 without scaling: the evaluation return did not converge but jumped
    between three discrete plateaus (-0.92, -1.00, -1.27) for the whole 1 500
    iterations, and two of three seeds ended below where they started.

    The divisors are derived from the market parameters rather than estimated
    from rollouts, so the transformation is a fixed function of the scenario and
    a run is reproducible without carrying a calibration artefact.  Channels 0, 6
    and 7 deliberately share one divisor, because the self-consumption heuristic
    forms ``(injection - offtake) / rated power`` from exactly those three and a
    common divisor cancels from that ratio algebraically.

    **It does not cancel in floating point, and the residue is not negligible for
    an individual participant.**  Measured over 64 episodes at 16 participants:
    `truthful` is bit-identical in all 1 024 (episode, participant) returns,
    since it reads no observation at all, but `self_consumption` differs in 240
    of them by up to 1.042e-02 at a state of charge of 0.5, against returns of
    order 0.7.  The reason is the tie rule rather than the arithmetic: that arm
    submits the truthful price, so every participant on a side is tied and §6.4's
    index rule decides the margin, and a last-bit change in the battery command
    moves which participant is marginal.  What moves is the allocation and not
    what the market cleared -- the per-episode community total agrees to
    1.907e-06.  The grand total shows no difference at all, but that is not
    separate evidence: float32 resolves 6.1e-05 at a magnitude of 741, so a
    perturbation of 1.9e-06 could not appear in it.
    """
    battery = env_params.battery
    power = float(np.max(battery.power_max))
    energy = power * R.DELTA
    money = energy * R.PI_RET
    return jnp.asarray([
        power, float(np.max(battery.capacity)), 1.0, 1.0,   # static, §9.4
        R.PI_RET,
        1.0, power, power,                                  # soc, injection, offtake
        power, energy, money,                               # own, previous
        R.PI_RET, energy * n_agents,                        # public, previous
        1.0, 1.0,                                           # calendar
    ], jnp.float32)


def scale_observations(env, scale):
    """`env` with every observation divided by `scale`, including the terminal one.

    Applied to the environment tuple rather than inside `powermarketjax`: §9.4
    fixes the channels and the observation is fixed, so rescaling them in
    the environment would be a change to the market specification.  Here it is a
    property of the learner, and the arms that do not learn are unaffected --
    `truthful` ignores the observation and `self_consumption` is invariant by the
    shared divisor above.
    """
    reset, step, step_auto, spec = env

    def scaled_reset(key, params):
        obs, state = reset(key, params)
        return obs / scale, state

    def scaled_step_auto(key, state, action, params):
        obs, state, reward, costs, done, info = step_auto(
            key, state, action, params)
        info = dict(info, terminal_obs=info["terminal_obs"] / scale)
        return obs / scale, state, reward, costs, done, info

    return scaled_reset, step, scaled_step_auto, spec


#: Held-out evaluation blocks, as a number of consecutive days and the period
#: between blocks.  Blocks rather than isolated days because an episode is
#: exactly one day long, so an isolated held-out day admits exactly one start
#: that lies inside it and 21 such days would give 21 evaluation episodes.
#: Spread over the panel rather than taken as one tail so that both sides of the
#: split cover the same range of seasons; the panel is 209 days, not a year.
EVAL_BLOCK_DAYS, EVAL_EVERY_DAYS = 3, 15


def make_start_pools(n_periods, episode_len, periods_per_day=96,
                     block_days=EVAL_BLOCK_DAYS, every_days=EVAL_EVERY_DAYS):
    """Training and held-out episode starts over one panel.

    A training start is admitted only if its whole episode misses every held-out
    period.  That buffer is what removes the leak, and it is needed because
    ``reset`` draws a start at any period rather than at a day boundary: an
    episode beginning late on the day before a held-out block would otherwise
    read most of that block.  The buffer costs starts, which is reported rather
    than hidden -- 2 565 of 19 969 at the settings above.
    """
    n_days = n_periods // periods_per_day
    held = np.zeros(n_periods, bool)
    for k in range(0, n_days, every_days):
        d0, d1 = k, min(k + block_days, n_days)
        held[d0 * periods_per_day:d1 * periods_per_day] = True
    starts = np.arange(n_periods - episode_len + 1)
    cum = np.concatenate([[0], np.cumsum(held)])
    inside = cum[starts + episode_len] - cum[starts]
    return starts[inside == 0], starts[inside == episode_len]


def restrict_starts(env, allowed):
    """`env` with ``reset`` drawing its start from `allowed` only.

    Applied to the environment tuple in the experiment rather than inside
    `powermarketjax`, for the reason `scale_observations` gives: which episodes a
    run sees is a property of the experiment, and what the environment does
    is fixed.

    Only ``reset`` is wrapped.  The auto-reset inside ``step_auto_reset`` still
    draws over the whole panel, and that is harmless here rather than overlooked:
    `preliminary_reference.make_rollout` scans exactly ``EPISODE_LEN`` steps and
    collects the observation *before* each step, so the observation auto-reset
    produces on the final step is never collected, and the bootstrap that would
    have read it is passed a last value of zero under the terminal boundary of
    section 3.  A rollout longer than one episode would need this revisited.
    """
    reset, step, step_auto, spec = env
    get_obs = spec["get_obs"]
    allowed = jnp.asarray(np.asarray(allowed, np.int32))

    def restricted_reset(key, params):
        draw_key, rest_key = jax.random.split(key)
        _, state = reset(rest_key, params)
        idx = jax.random.randint(draw_key, (), 0, allowed.shape[0])
        state = state.replace(cursor=allowed[idx])
        return get_obs(state, params), state

    return restricted_reset, step, step_auto, spec


def build(n_agents, initial_soc, learner_mask=None):
    """`(params, training environment, held-out environment)`.

    The two environments differ only in which episode starts `reset` may draw,
    and share everything else including the parameters, so an arm trained on one
    and scored on the other sees the same market on unseen days.
    """
    series = R.load_fluvius_households(n_households=n_agents)
    one_way = math.sqrt(0.85)
    battery = make_battery_bundle(
        n_devices=n_agents, capacity_mwh=0.011, power_mw=0.011 / 2.1,
        eta_charge=one_way, eta_discharge=one_way, soc_min=0.15, soc_max=1.0,
        initial_soc=initial_soc, dt_hours=R.DELTA, cycle_cost_per_mwh=0.0)
    mask = np.ones(n_agents, bool) if learner_mask is None else learner_mask
    params = make_p2p_params(
        p_pv=series.injection, load=series.offtake, battery=battery,
        kappa=np.full(n_agents, R.KAPPA, np.float32), learner_mask=mask,
        episode_len=R.EPISODE_LEN)
    env = make_p2p_env(n_agents, R.PI_EXP, R.PI_RET, R.DELTA)
    env = scale_observations(env, observation_scale(params, n_agents))
    train_starts, eval_starts = make_start_pools(series.injection.shape[0],
                                                 R.EPISODE_LEN)
    return params, restrict_starts(env, train_starts), \
        restrict_starts(env, eval_starts)


def make_update(env, env_params, batch, constrained, iterations,
                per_agent=False):
    """One update. With `constrained` false the multiplier is held at zero.

    `per_agent` says which parameter layout `policy` is in.  The forward pass is
    chosen once, HERE, and the same choice reaches the rollout and the loss, so
    a run cannot collect its data under one layout and take its gradient under
    the other.
    """
    import optax
    _forward = R.policy_forward(per_agent)
    rollout = R.make_rollout(env, env_params, "learned", 0, per_agent)
    # Decayed linearly to zero over the run.  The policy is a tanh-squashed
    # Gaussian, so a step taken late is a step into saturation that the next one
    # cannot undo; a constant step size therefore leaves the evaluation return
    # oscillating rather than settling, which is what the unannealed run showed.
    schedule = optax.linear_schedule(LEARNING_RATE, 0.0, iterations * R.EPOCHS)
    optimiser = optax.chain(optax.clip_by_global_norm(0.5),
                            optax.adam(schedule))

    def collect(policy, key):
        keys = jax.random.split(key, batch)
        obs, raw, logp, value, reward, cost, terminal = jax.vmap(
            rollout, in_axes=(None, 0))(policy, keys)
        # No bootstrap past the boundary.  §8 settles the stock left in the
        # battery at the truncated step, so the episode is self-contained and
        # `spec["termination"]` reads "terminal"; bootstrapping the value of
        # `terminal_obs` on top of that would count the stock twice, once in the
        # reward and once in the value.  Passing zero here makes the last
        # temporal difference `r - V`, which is the terminal form.
        del terminal
        last_value = jnp.zeros(value.shape[:1] + value.shape[2:], value.dtype)
        return obs, raw, logp, value, reward, cost, last_value

    def loss_fn(policy, data):
        mean, log_std, value = _forward(policy, data["obs"])
        logp = R.log_prob(mean, log_std, data["raw"])
        adv = data["gae"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        policy_loss = rlax_clip(jnp.exp(logp - data["logp"]), adv)
        value_loss = ((value - data["target"]) ** 2).mean()
        # No entropy bonus: the exploration width is on the schedule above rather
        # than free, so a bonus on it would only fight the schedule.  (The term
        # `preliminary_reference` carries is in any case not an entropy -- it is
        # the negative mean log density of the *stored* actions under the
        # *current* policy, which also rewards moving the mean away from the
        # actions already taken.)
        del log_std
        return policy_loss + R.VALUE_COEF * value_loss

    def rlax_clip(ratio, adv):
        import rlax
        return rlax.clipped_surrogate_pg_loss(ratio.ravel(), adv.ravel(),
                                              R.CLIP_EPS)

    @jax.jit
    def update(policy, opt_state, lam, lam_step, key):
        key, collect_key = jax.random.split(key)
        obs, raw, logp, value, reward, cost, last_value = collect(
            policy, collect_key)
        # The surrogate the policy is optimised against.  The reported reward
        # stays `reward`; `lam` never touches what is measured.
        shaped = reward - lam * cost
        gae, target = jax.vmap(R.advantages)(shaped, value, last_value)
        data = dict(obs=obs, raw=raw, logp=logp, gae=gae, target=target)

        def epoch(carry, _):
            policy, opt_state = carry
            grads = jax.grad(loss_fn)(policy, data)
            updates, opt_state = optimiser.update(grads, opt_state)
            return (optax.apply_updates(policy, updates), opt_state), None

        (policy, opt_state), _ = jax.lax.scan(
            epoch, (policy, opt_state), None, length=R.EPOCHS)

        episode_cost = cost.sum(1).mean()
        # Ascent on the multiplier: it rises while the constraint is violated
        # and falls back to zero once it is met, so the limit is approached from
        # whichever side the policy is on.
        violation = episode_cost / COST_LIMIT - 1.0
        new_lam = jnp.where(
            constrained,
            jnp.clip(lam + lam_step * violation, 0.0, LAMBDA_MAX),
            0.0)
        return policy, opt_state, new_lam, dict(
            ret=reward.sum(1).mean(), cost=episode_cost, lam=lam)

    return optimiser, update


def run(n_agents, initial_soc, constrained, seed, iterations, batch, every,
        per_agent=False, algo="ippo", buffer_size=sac_arm.BUFFER_SIZE,
        utd_ratio=sac_arm.UTD_RATIO):
    """Train on the training starts, score the curve on the held-out ones.

    The environment returned is the held-out one, so a caller that evaluates the
    returned policy against the returned environment is scoring on days no
    update ever saw.

    `per_agent` is the only thing that separates the two parameter layouts: the
    scenario, the pools, the keys, the schedules and the optimiser chain are
    built by the same expressions either way, so a pair of runs differing only
    in this argument differs only in the layout.  The policy that comes back is
    what says which layout actually ran; the caller reads it there.

    `algo` selects the learner.  `ippo` is the default and
    the path every published 04 learning number was produced on, and its body
    below is the body it has always been -- the SAC arm is a different function
    in a different module, not this one with branches sprinkled through it, so
    "the two arms differ in the algorithm" is visible in the file rather than
    argued for.  `buffer_size` and `utd_ratio` belong to SAC alone and are
    refused on the IPPO path rather than silently ignored: a flag that is
    accepted and dropped produces a run whose log and whose product disagree.

    Returns ``(curve, policy, env_eval, params, cfg)``; `cfg` is the `SACConfig`
    that ran, or `None` on the IPPO path, and it is the only place the fitted
    `reward_scale` can be read from.
    """
    if algo not in ("ippo", "sac"):
        raise SystemExit(f"--algo {algo!r} is neither 'ippo' nor 'sac'")
    if algo == "sac":
        return sac_arm.run(n_agents, initial_soc, constrained, seed, iterations,
                           batch, every, per_agent, buffer_size, utd_ratio,
                           build, COST_LIMIT, LAMBDA_STEP, LAMBDA_MAX)
    if buffer_size != sac_arm.BUFFER_SIZE or utd_ratio != sac_arm.UTD_RATIO:
        raise SystemExit(
            "--buffer-size and --utd-ratio are fields of SACConfig; the IPPO "
            "arm has no replay buffer and takes R.EPOCHS gradient steps per "
            "iteration by construction")
    params, env, env_eval = build(n_agents, initial_soc)
    optimiser, update = make_update(env, params, batch, constrained, iterations,
                                    per_agent)
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    policy = (R.init_policy_per_agent(init_key, n_agents) if per_agent
              else R.init_policy(init_key))
    opt_state = optimiser.init(policy)
    lam = jnp.float32(0.0)

    eval_key = jax.random.PRNGKey(20_000 + n_agents)
    curve = []
    if iterations == 0:
        # The untrained control.  The loop below never executes, so without this
        # branch nothing is evaluated and nothing is recorded, and the caller
        # reads an empty curve.  The branch is on the iteration count and not on
        # `curve` being empty, so an empty curve arising any other way still
        # reaches the caller and fails there rather than being reported as the
        # control.  `iteration=-1` says this row is not an iteration that ran,
        # and the two training fields are null rather than a number, because no
        # rollout was collected to average.  The evaluation is the same call the
        # loop makes, on the same held-out environment and the same key, so the
        # number is comparable with `curve[0]`; it differs from it only in that
        # `curve[0]` is scored after one update and this is scored before any.
        ret, cost = R.evaluate(env_eval, params, "learned_mean", policy,
                               eval_key, 64, per_agent=per_agent)
        curve.append(dict(iteration=-1, train_return=None, train_cost=None,
                          lam=float(lam), eval_return=float(ret.mean()),
                          eval_cost=float(cost.mean())))
    for iteration in range(iterations):
        key, sub = jax.random.split(key)
        frac = iteration / max(iterations - 1, 1)
        # Written at the shape the policy already carries rather than at
        # `(R.ACTION_DIM,)`: under `per_agent` that leaf is
        # `(n_agents, ACTION_DIM)`, and a literal shape here would silently
        # collapse it to one row shared by every participant.  On the shared
        # layout the two expressions are the same shape, so this arm is
        # unchanged.
        policy = dict(policy, log_std=jnp.full(
            policy["log_std"].shape,
            LOG_STD_START + frac * (LOG_STD_END - LOG_STD_START), jnp.float32))
        policy, opt_state, lam, aux = update(
            policy, opt_state, lam, jnp.float32(LAMBDA_STEP * (1.0 - frac)), sub)
        row = dict(iteration=iteration, train_return=float(aux["ret"]),
                   train_cost=float(aux["cost"]), lam=float(aux["lam"]))
        if iteration % every == 0 or iteration == iterations - 1:
            ret, cost = R.evaluate(env_eval, params, "learned_mean", policy,
                                   eval_key, 64, per_agent=per_agent)
            row.update(eval_return=float(ret.mean()), eval_cost=float(cost.mean()))
        curve.append(row)
        if not np.isfinite(row["train_return"]):
            raise SystemExit(f"non-finite at iteration {iteration}")
    return curve, policy, env_eval, params, None


def parameter_layout(policy, n_agents, obs_dim=R.OBS_DIM):
    """Which layout the parameter tree that came BACK is in, and how big it is.

    **Read off the returned tree and not off the flag that was passed in.**  A
    flag that is accepted, forwarded and then ignored produces a run that says
    `per_agent_params: true` in its log and carries a shared network in its
    product, and the two are indistinguishable from the caller's side.
    Everything below comes from `policy`.

    **The ALGORITHM is read off the tree too, and by the same rule.**  A run
    that was asked for `--algo sac`, forwarded the string and then built the PPO
    policy would log `sac` and carry nine arrays, and nothing downstream would
    say so; `main` compares what comes back here against what it asked for.  The
    two trees are told apart by their top-level entries -- nine named arrays for
    IPPO, the six of `sac_arm.TOP_LEVEL` for SAC -- which is a property of the
    learner and not of any flag.

    The reference shapes come from `jax.eval_shape`, so the comparison costs no
    random draw and no device work; they are what `init_policy` would produce at
    this observation width.  Neither layout matching is a hard error rather than
    a third value: a tree in neither shape is a tree nobody can place, and
    reporting it as "not per-agent" would file it with the shared runs.
    """
    if sorted(policy) == sorted(sac_arm.TOP_LEVEL):
        return sac_arm.parameter_layout(policy, n_agents)
    ref = jax.eval_shape(lambda k: R.init_policy(k, obs_dim),
                         jax.random.PRNGKey(0))
    shared = {k: tuple(int(d) for d in v.shape) for k, v in ref.items()}
    got = {k: tuple(int(d) for d in jnp.shape(v)) for k, v in policy.items()}
    if sorted(got) != sorted(shared):
        raise SystemExit(
            f"the policy carries leaves {sorted(got)} while `init_policy` "
            f"produces {sorted(shared)} and `sac_arm` produces "
            f"{sorted(sac_arm.TOP_LEVEL)}; the layout cannot be read off a tree "
            f"whose leaves are not the ones this file knows")
    is_shared = got == shared
    is_per_agent = all(got[k] == (int(n_agents),) + shared[k] for k in shared)
    if is_shared == is_per_agent:
        raise SystemExit(
            f"the parameter tree is in neither layout at n_agents={n_agents}: "
            f"shapes {got}, shared reference {shared}")
    return dict(
        algo="ippo",
        per_agent_params=bool(is_per_agent),
        leaves=len(got),
        scalars=int(sum(int(np.prod(v)) if v else 1 for v in got.values())),
        scalars_shared_layout=int(sum(int(np.prod(v)) if v else 1
                                      for v in shared.values())),
        leaf_shapes={k: list(v) for k, v in sorted(got.items())},
        read_from="the policy `run` returned, not the flag passed in")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=3)
    #: Where the seed range starts. Default 0 keeps `--seeds N` meaning "seeds
    #: 0..N-1", which is what every published column was run with. It exists so
    #: one column can be split across cards: `run` builds its whole random
    #: stream from `jax.random.PRNGKey(seed)` and nothing is carried from one
    #: seed to the next, so seeds 3..4 run here are the same runs as seeds 3..4
    #: inside a `--seeds 5` process. That equality was checked, not assumed.
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--initial-soc", type=float, nargs="+",
                        default=[0.5, 0.15])
    parser.add_argument("--out", type=Path, default=None)
    #: One network per participant instead of one shared by all of them.  Same
    #: flag name and same meaning as `run_rl_01.py --per-agent-params`, so the
    #: in-package and out-of-package ablations can be read against each other.
    #: Curves written under this flag are NOT interchangeable with shared ones:
    #: the trees have the same number of leaves and differ only in a leading
    #: axis, so a leaf-for-leaf comparison would line up the wrong things.
    parser.add_argument("--per-agent-params", action="store_true",
                        help="give every participant its own copy of the "
                             "policy and value network. The layout that "
                             "actually ran is stamped into every record from "
                             "the returned parameter tree, not from this flag")
    #: Which learner.  `ippo` is the default and the path
    #: every 04 learning number on disk was produced on.  Same flag name and
    #: same two choices as `run_rl_0{1,2,3}.py --algo`, so the five markets'
    #: SAC columns can be read against each other.  The algorithm that actually
    #: ran is stamped from the returned parameter tree, as the layout is.
    parser.add_argument("--algo", choices=("ippo", "sac"), default="ippo",
                        help="learner: ippo (default, the existing path) or "
                             "sac (off-policy). Stamped as `algo` in every "
                             "record, read off the parameter tree that came "
                             "back rather than off this flag")
    #: Replay capacity in env-steps, SAC only.  Exposed because it is the one
    #: SAC constant that has to move with the population: one env-step costs
    #: 136 bytes PER PARTICIPANT, so the default is 71 MB at 16 participants
    #: and 5.35 GB at 1 200.  A run that leaves it alone reproduces
    #: `hyperparams.SAC_SHARED` exactly, which is the property `run_rl_02.py`
    #: states for its own `--weight-decay`.
    parser.add_argument("--buffer-size", type=int,
                        default=sac_arm.BUFFER_SIZE,
                        help="SAC replay capacity in env-steps "
                             f"(default {sac_arm.BUFFER_SIZE}, SAC_SHARED's)")
    #: Gradient steps per env-step collected, SAC only.  At the default of 1.0
    #: one iteration takes `batch * 96` gradient steps where the IPPO arm takes
    #: `R.EPOCHS = 4`, so the two arms' iteration counts are not comparable and
    #: a SAC run is scheduled on env-steps or on wall clock.
    parser.add_argument("--utd-ratio", type=float, default=sac_arm.UTD_RATIO,
                        help="SAC gradient steps per env-step collected "
                             f"(default {sac_arm.UTD_RATIO}, SAC_SHARED's)")
    args = parser.parse_args()
    #: `--seeds` is the exclusive END of the range, not a count, so
    #: `--seed-start 3 --seeds 2` names no seed at all.  That run used to loop
    #: over nothing and exit 0 with an empty result; refused here instead.
    if args.seeds <= args.seed_start:
        raise SystemExit(
            f"--seed-start {args.seed_start} --seeds {args.seeds} names no seed: "
            f"--seeds is the exclusive end of the range (seeds "
            f"{args.seed_start}..{args.seeds - 1}), so it must exceed --seed-start")

    print(f"[{time.strftime('%H:%M:%S')}] device {jax.devices()[0]}  "
          f"N={args.agents}  seeds={args.seed_start}..{args.seeds - 1}  "
          f"iterations={args.iterations}  "
          f"constraint limit={COST_LIMIT}  initial soc {args.initial_soc}  "
          f"per-agent params requested={bool(args.per_agent_params)}  "
          f"algo requested={args.algo}", flush=True)
    if args.algo == "sac":
        print(f"  SAC: buffer {args.buffer_size} env-steps, utd_ratio "
              f"{args.utd_ratio} -> {int(round(args.utd_ratio * args.batch * R.EPISODE_LEN))} "
              f"gradient steps per iteration against the IPPO arm's "
              f"{R.EPOCHS}; the two iteration counts are not the same quantity",
              flush=True)

    everything = []
    for initial_soc in args.initial_soc:
        for constrained in (False, True):
            label = "lagrangian" if constrained else "unconstrained"
            for seed in range(args.seed_start, args.seeds):
                started = time.perf_counter()
                curve, policy, env, params, cfg = run(
                    args.agents, initial_soc, constrained, seed,
                    args.iterations, args.batch, args.eval_every,
                    args.per_agent_params, args.algo, args.buffer_size,
                    args.utd_ratio)
                #: The layout that ran, read out of what `run` handed back.
                #: Checked against the request rather than assumed equal to it:
                #: this is the assertion that fails if the flag is accepted and
                #: then dropped somewhere between here and `init_policy`.
                layout = parameter_layout(policy, args.agents)
                #: The algorithm that ran, read out of the same tree and by the
                #: same rule.  This is the assertion that fails if `--algo` is
                #: accepted and then dropped between here and the learner.
                if layout["algo"] != args.algo:
                    raise SystemExit(
                        f"--algo={args.algo} was requested and the returned "
                        f"parameter tree is {layout['algo']}'s (top-level "
                        f"entries {sorted(policy)}); the flag did not reach "
                        f"the learner and no number from this run is what it "
                        f"claims")
                if layout["per_agent_params"] != bool(args.per_agent_params):
                    raise SystemExit(
                        f"--per-agent-params={bool(args.per_agent_params)} was "
                        f"requested and the returned policy is in the "
                        f"{'per-agent' if layout['per_agent_params'] else 'shared'} "
                        f"layout ({layout['leaf_shapes']}); the flag did not "
                        f"reach the learner and no number from this run is "
                        f"what it claims")
                if not everything:
                    if cfg is not None:
                        print(f"  SAC reward_scale = {cfg.reward_scale:.6e} "
                              f"(pooled std of the per-agent reward under the "
                              f"truthful action, {args.batch} x "
                              f"{cfg.horizon} sample); one iteration collects "
                              f"{cfg.n_envs * cfg.horizon} env-steps and takes "
                              f"{int(round(cfg.utd_ratio * cfg.n_envs * cfg.horizon))} "
                              f"gradient steps", flush=True)
                    print(f"  parameter layout in force: "
                          f"per_agent_params={layout['per_agent_params']}  "
                          f"{layout['leaves']} leaves  {layout['scalars']} "
                          f"scalars (shared layout would be "
                          f"{layout['scalars_shared_layout']})", flush=True)
                tail = [r for r in curve if "eval_return" in r][-3:]
                everything.append(dict(
                    initial_soc=initial_soc, arm=label, seed=seed, curve=curve,
                    iterations=args.iterations,
                    untrained_baseline=args.iterations == 0,
                    layout=layout,
                    #: the learner that ran, from the tree and not from the
                    #: flag; `hyperparams` is null on the IPPO path, whose
                    #: constants are this file's module-level ones
                    algo=layout["algo"],
                    hyperparams=(None if cfg is None else
                                 {k: (list(v) if isinstance(v, tuple) else v)
                                  for k, v in vars(cfg).items()}),
                    hyperparams_provenance=(None if cfg is None
                                            else sac_arm.PROVENANCE),
                    final_eval_return=float(np.mean([r["eval_return"] for r in tail])),
                    final_eval_cost=float(np.mean([r["eval_cost"] for r in tail])),
                    final_lam=curve[-1]["lam"]))
                print(f"  [{time.strftime('%H:%M:%S')}] soc0={initial_soc} "
                      f"{label:<14} seed={seed}  "
                      f"eval return {everything[-1]['final_eval_return']:+.4f}  "
                      f"eval cost {everything[-1]['final_eval_cost']:8.3f}  "
                      f"lambda {everything[-1]['final_lam']:7.4f}  "
                      f"{time.perf_counter() - started:.0f}s", flush=True)
                if args.out:
                    args.out.write_text(json.dumps(everything))

    # the two arms that do not learn, for the same conditions
    print()
    for initial_soc in args.initial_soc:
        params, _, env = build(args.agents, initial_soc)
        eval_key = jax.random.PRNGKey(20_000 + args.agents)
        sc, scc = R.evaluate(env, params, "self_consumption",
                             R.init_policy(jax.random.PRNGKey(0)), eval_key, 64)
        tparams, _, tenv = build(args.agents, initial_soc,
                                 np.zeros(args.agents, bool))
        tr, trc = R.evaluate(tenv, tparams, "truthful",
                             R.init_policy(jax.random.PRNGKey(0)), eval_key, 64)
        print(f"  soc0={initial_soc}  self-cons {sc.mean():+.4f} "
              f"(cost {scc.mean():.3f})   truthful {tr.mean():+.4f} "
              f"(cost {trc.mean():.3f})")

    print(f"\n{'soc0':>6} {'arm':<14} {'eval return':>12} {'eval cost':>11} "
          f"{'lambda':>9}")
    for initial_soc in args.initial_soc:
        for label in ("unconstrained", "lagrangian"):
            rows = [e for e in everything
                    if e["initial_soc"] == initial_soc and e["arm"] == label]
            print(f"{initial_soc:>6} {label:<14} "
                  f"{np.mean([r['final_eval_return'] for r in rows]):>+12.4f} "
                  f"{np.mean([r['final_eval_cost'] for r in rows]):>11.3f} "
                  f"{np.mean([r['final_lam'] for r in rows]):>9.4f}")

    if args.out:
        args.out.write_text(json.dumps(everything))
        print(f"\ncurves written to {args.out}")


if __name__ == "__main__":
    main()
