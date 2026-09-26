"""The open-loop arms: the constant instrument and the markup grid.

Both need no learning at all, and both are market-agnostic: they consume only
what each market puts in `spec`, so the same code serves markets 01, 02
and 03.

**The constant arm is an instrument, not a baseline.**  It runs on
every benchmark run and answers one question: did this optimisation fail?  A
learning arm that after N iterations is still no better than a single constant
action has failed to optimise; that is a fact about the run, not about the
market.  It also gives the reward-only optimum, without which "does the
constrained arm's multiplier bind" cannot be computed at all -- the P2P market
found its optimal multiplier was zero exactly this way.

**The markup grid is no longer a baseline column** (removed 2026-09-06).  It is swept offline once and reports its best point, **and
whether that best point sits on the grid's upper bound** -- if it does, the
grid was too narrow and the number is a lower bound on what a fixed markup can
do, not the optimum.  The apparatus and its artifacts stay; the results that
rest on them cite them by footnote instead of by column.
"""
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np


def rollout_action(env, params, action, n_envs: int, horizon: int, key,
                   reset_keys=None, reset_states=None,
                   voll: Optional[float] = None,
                   other_shortfall_cost: Optional[float] = None,
                   other_shortfall_cost_key: Optional[str] = None,
                   info_mean_keys: Sequence[str] = ()):
    """Roll out one fixed action and return the metrics a learning arm reports.

    Pure and jit-able: `vmap` over the environment axis, `lax.scan` over the
    horizon, no Python loop on the rollout path.

    `reset_keys` decides what the environment axis contains.  Left `None`, it is
    `n_envs` episodes drawn from `key`, which in all five markets means `n_envs`
    days drawn at random.  Passed, it is one key per environment and the caller
    says what each opens -- `evaluation.open_day` returns exactly such a key for
    a named day, and that is how the other three baselines reach the twelve
    evaluation days.  `key` is still required and still drives the per-step
    keys: `step_auto_reset` hands its key to `reset` when an episode ends, so a
    rollout longer than one episode leaves the pinned days however they were
    pinned.  `markup_grid` refuses that combination rather than this function,
    because the caller that named days is the one that meant them to hold.

    `reset_states` is the third form, added later: a state pytree
    already batched along the environment axis, opened by the caller through the
    market's own `reset_on_day`.  It exists because a key cannot express "the
    first period of day d" in market 03 -- a key fixes the day and leaves the
    offset inside it free, and 265 of 576 pinned half hours were scored on the
    following day.  Given `reset_states`, no `reset` runs here at all; the
    states came out of one.  `reset_keys` and `reset_states` are mutually
    exclusive.

    `reward_sum_per_env` is the per-environment total, kept beside the means so
    a product can be read one day at a time.  An aggregate over twelve days can
    hide and can reverse what the twelve say separately, so both forms are
    reported.

    `voll` and `other_shortfall_cost`: **opt-in, additive, off by default.**
    Left `None` (both), the returned dict is exactly what it has always been --
    every existing caller of `rollout_action` sees no change. Given both, this
    also collects `info["cost"]` (production cost) and `info["shed_mwh"]` from
    every step -- both already published by every market's `step` for exactly
    this purpose (`envs/{day_ahead,real_time,ancillary}/env.py` each say so at
    the `info` site) -- and adds `production_cost_sum_per_env`,
    `shed_mwh_sum_per_env` and `system_cost_sum_per_env` to the result, using
    the identical formula `evaluation.system_cost` uses: production cost plus
    shed energy at `voll` plus `other_shortfall_cost`. `other_shortfall_cost`
    has no default for the same reason `evaluation.system_cost` doesn't: a
    default would silently ship one market's convention (0.0 for day-ahead and
    real-time) to a market where it is the wrong number (ancillary's reserve
    shortfall). Pass `0.0` explicitly where a market has no third term.

    `other_shortfall_cost_key`: for the one market where the third term is not
    a constant. Ancillary's reserve shortfall varies by step, day and level, so
    unlike day-ahead's and real-time's `0.0` it cannot be passed as a fixed
    float; give the `info` key instead (ancillary's `step` already publishes
    `info["volr_cost"]`, already priced in dollars, for exactly this reason --
    see that module's own `info` docstring) and this collects and sums it over
    the horizon the same way `shed_mwh` is summed, then adds it on top of
    `other_shortfall_cost` (still required, still `0.0` where there is nothing
    to add it to). `run_eval_03.py`'s own day-eval loop accumulates the
    identical field the identical way before calling `evaluation.system_cost`.

    `info_mean_keys`: **opt-in, empty by default.**  Each named `info` field is
    collected at every step and returned as `info_mean_<key>`, its mean over the
    horizon and the environment axis with any trailing axes kept (ancillary's
    `reserve_price` comes back as one number per product).  Empty, nothing is
    collected and the traced program is the one it has always been.
    """
    _reset, _step, step_auto_reset, _spec = env
    reset = env[0]
    if reset_keys is not None and reset_states is not None:
        raise ValueError(
            "reset_keys and reset_states both given; they are two ways of "
            "saying what the environment axis contains and one of them would "
            "be silently ignored")
    if reset_states is not None:
        leading = {int(np.asarray(x).shape[0])
                   for x in jax.tree_util.tree_leaves(reset_states)}
        if leading != {int(n_envs)}:
            raise ValueError(
                f"reset_states has leading axis {sorted(leading)} against "
                f"n_envs={n_envs}. The environment axis *is* the pinned "
                f"episodes here, so a mismatch would roll a different set from "
                f"the one the caller named and report it under their name")
        state, step_key = reset_states, key
    elif reset_keys is None:
        keys = jax.random.split(key, n_envs + 1)
        reset_keys, step_key = keys[:n_envs], keys[n_envs]
    else:
        if isinstance(reset_keys, (list, tuple)):
            reset_keys = jnp.stack(list(reset_keys))
        if reset_keys.shape[0] != n_envs:
            raise ValueError(
                f"{reset_keys.shape[0]} reset keys against n_envs={n_envs}. The "
                f"environment axis *is* the pinned episodes here, so a mismatch "
                f"would roll a different set from the one the caller named and "
                f"report it under the name of that one")
        step_key = key
    if reset_states is None:
        _obs, state = jax.vmap(reset, in_axes=(0, None))(reset_keys, params)

    # additive, opt-in only -- see the `voll` / `other_shortfall_cost`
    # paragraph in the docstring. `want_system_cost` is a Python bool fixed at
    # trace time (both arguments are plain floats or None, never jax arrays),
    # so branching on it here does not add a traced conditional to the scan.
    want_system_cost = voll is not None
    if want_system_cost and other_shortfall_cost is None:
        raise ValueError(
            "voll given without other_shortfall_cost: evaluation.system_cost "
            "takes no default for it either, and for the same reason -- pass "
            "0.0 explicitly if this market's clearing objective has no third "
            "term, rather than silently inheriting day-ahead's zero")
    if other_shortfall_cost_key is not None and not want_system_cost:
        raise ValueError(
            "other_shortfall_cost_key given without voll: this key only means "
            "anything once system_cost is being collected at all")

    def one(carry, _):
        state, k = carry
        k, k_env = jax.random.split(k)
        env_keys = jax.random.split(k_env, n_envs)
        _o, nxt, reward, costs, _done, info = jax.vmap(
            step_auto_reset, in_axes=(0, 0, None, None))(
                env_keys, state, action, params)
        out = dict(reward=reward, costs=costs, converged=info["converged"])
        if want_system_cost:
            # `info["cost"]` and `info["shed_mwh"]` are what every market's
            # `step` already publishes for exactly this purpose; only the
            # collection was missing here, not the underlying quantity
            out["prod_cost"] = info["cost"]
            out["shed_mwh"] = info["shed_mwh"]
            if other_shortfall_cost_key is not None:
                out["extra_shortfall"] = info[other_shortfall_cost_key]
        for name in info_mean_keys:
            out["info_" + name] = info[name]
        return (nxt, k), out

    _, traj = jax.lax.scan(one, (state, step_key), None, length=horizon)
    r = traj["reward"]
    result = dict(
        reward_mean=jnp.mean(r),
        reward_per_agent=jnp.mean(r, axis=(0, 1)),
        # summed over the horizon and over every trailing axis -- the agent axis
        # in all five markets -- which leaves one number per environment, i.e.
        # per pinned day when the days are pinned
        reward_sum_per_env=jnp.sum(r, axis=0).reshape(r.shape[1], -1).sum(axis=1),
        costs_mean=jnp.mean(traj["costs"]),
        costs_max=jnp.max(traj["costs"]),
        unconverged_frac=1.0 - jnp.mean(traj["converged"].astype(jnp.float64)),
    )
    if want_system_cost:
        pc = traj["prod_cost"]
        # same reduction shape as reward_sum_per_env, for the same reason: one
        # number per pinned day, agents and horizon both summed away
        production_cost_sum_per_env = jnp.sum(pc, axis=0).reshape(
            pc.shape[1], -1).sum(axis=1)
        shed_mwh_sum_per_env = jnp.sum(traj["shed_mwh"], axis=0)
        # identical arithmetic to `evaluation.system_cost`, applied per
        # environment instead of once over an already-summed day; kept inline
        # rather than calling that function so this stays inside one jnp
        # expression and the per-env vector is never round-tripped through
        # Python floats before `markup_grid` converts the whole dict at once
        result["production_cost_sum_per_env"] = production_cost_sum_per_env
        result["shed_mwh_sum_per_env"] = shed_mwh_sum_per_env
        extra_shortfall_sum_per_env = float(other_shortfall_cost)
        if other_shortfall_cost_key is not None:
            # already priced in dollars (ancillary's `info["volr_cost"]`), so
            # summed over the horizon directly -- no `voll` multiplication here,
            # unlike `shed_mwh_sum_per_env` which is a physical quantity
            extra_shortfall_sum_per_env = (
                jnp.sum(traj["extra_shortfall"], axis=0) + extra_shortfall_sum_per_env)
            result["other_shortfall_cost_sum_per_env"] = extra_shortfall_sum_per_env
        result["system_cost_sum_per_env"] = (
            production_cost_sum_per_env + float(voll) * shed_mwh_sum_per_env
            + extra_shortfall_sum_per_env)
        result["system_cost_mean"] = jnp.mean(result["system_cost_sum_per_env"])
    for name in info_mean_keys:
        result["info_mean_" + name] = jnp.mean(traj["info_" + name], axis=(0, 1))
    return result


def constant_arm(env, params, n_envs: int, horizon: int, key,
                 action=None):
    """The instrument.  Defaults to `spec["baseline_action"]` (truthful cost).

    Passing `action` gives the other constant point the benchmark distinguishes:
    truthful cost is the *zero* of the strategy axis, whereas the best constant
    action is a genuine one-parameter policy.  The two are different arms and
    the first must not be reported as if it were strength-matched.
    """
    if action is None:
        action = env[3]["baseline_action"]
    return rollout_action(env, params, action, n_envs, horizon, key)


def _from_evaluation(*names):
    """Names out of `evaluation`, found whichever way this tools tree was imported.

    Both import shapes exist in the repository and neither can be dropped: the
    drivers put `tools/benchmark` on `sys.path` and import `evaluation` bare,
    while the tests put `tools` on it and import `benchmark.arms`.  Trying both
    is what makes this file usable from either, and the import is deferred to
    call time so that importing `arms` never depends on which one is in force.
    """
    try:
        import evaluation as ev
    except ImportError:                                    # pragma: no cover
        from benchmark import evaluation as ev
    return tuple(getattr(ev, n) for n in names)


def markup_grid(env, params, levels: Sequence[float], n_envs: Optional[int],
                horizon: int, key,
                action_of: Optional[Callable[[float], jnp.ndarray]] = None,
                days: Optional[Sequence[int]] = None,
                day_of_state: Optional[Callable] = None,
                period_of_state: Optional[Callable] = None,
                voll: Optional[float] = None,
                other_shortfall_cost: Optional[float] = None,
                other_shortfall_cost_key: Optional[str] = None):
    """Sweep a one-dimensional grid of constant markup levels on named days.

    `voll` / `other_shortfall_cost` / `other_shortfall_cost_key`: forwarded to
    `rollout_action` unchanged, same opt-in contract -- see that function's
    docstring. Left `None`, every
    row is exactly what it always was; given both, every row also carries
    `system_cost_sum_per_env` and its siblings.

    **`days` is what puts this arm where the other three baselines already are.**
    Every market draws its day inside `reset` and none takes a day as an
    argument, so the honest, constant and optimisation arms all reach the twelve
    evaluation days through `evaluation.open_day`; a grid rolled on days drawn
    at random reports a number from a different sample and is not comparable
    with any of them.  `day_of_state` is required alongside, for the reason
    `open_day` requires it: the read-back is what makes the key search safe, and
    each market carries the day differently.  The indices are into whatever
    series `params` carries -- market 02's evaluation environment holds only the
    held-out days, so there its day 0 is the first held-out day and not the
    first day of the window.

    **One key for the whole grid, not one per level.**  Until 2026-08-27 each
    level was rolled on `fold_in(key, i)` and the winner taken by `argmax` over
    the rows -- estimates from different draws, compared as though they were one
    experiment.  In these markets the draw *is* the day, so that `argmax`
    selected on the days a level happened to be dealt as much as on the level
    itself: sampling on the outcome variable, biased along the very axis being
    reported.  The repair is common random
    numbers: one `key`, one set of days, every level rolled on the identical
    realisation, so the difference between two rows is the level and nothing
    else.

    The other repair available was to roll each level several times and report
    the spread.  It was not taken, and the reason is a measured property rather
    than a preference: `step` consumes no randomness in markets 01, 02 and 03 --
    `day_ahead` and `real_time` each say so in their own docstrings, and
    `ancillary` splits the key and deletes the transition half -- so once `days`
    is passed the rollout is deterministic and the spread would be identically
    zero, at k times the cost.  A market whose transition did draw would need
    the repeats *and* the shared key, which is why the single key is passed here
    rather than removed.

    `horizon` may not exceed `params.episode_len` when days are pinned.  At the
    step where an episode ends, `step_auto_reset` opens a fresh one from the
    step key, so the steps past the first episode would be scored on days nobody
    named while the product still carried the named ones.

    `levels` must be strictly increasing and hold at least two points.  Both are
    about the flag rather than about taste: `at_upper_bound` is computed as
    "the winner is the last element", which means "the top of the grid" only if
    the grid ascends, and a one-point grid returns it as `True` with no content.

    `action_of` maps a level to a full action array.  It is a parameter and not
    an inference from `spec` because the action layouts genuinely differ: the
    ancillary market's last columns are reserve offers, not markups, and filling
    them with the markup level would sweep the wrong axis.  The default fills
    every coordinate, which is right for a pure `kind="markup"` market and wrong
    for one with extra columns -- callers with extra columns must pass their own,
    and the two cases are told apart by reading the market's specification, not
    by a heuristic here.

    Returns `(rows, best, at_upper_bound)` where `rows` is one metrics dict per
    level, `best` is the index of the highest mean reward, and
    `at_upper_bound` says the winner is the last grid point -- in which case
    `best` is a lower bound and not an optimum.  Two different things make it
    true and the caller has to say which: a grid too narrow for the action
    space, or a grid that spans the whole action space and whose optimum sits on
    the endpoint.  The second is market 01 to 03's "maximum bid" column, which is
    the endpoint and was never searched for; comparing
    `levels[-1]` with `spec["action_high"]` is what tells the two apart.
    """
    levels = [float(v) for v in levels]
    if len(levels) < 2:
        raise ValueError(
            f"a grid of {len(levels)} point(s) reports at_upper_bound=True with "
            f"nothing behind it; a sweep needs at least two levels")
    if any(b <= a for a, b in zip(levels, levels[1:])):
        raise ValueError(
            f"levels must be strictly increasing, got {levels}. "
            f"`at_upper_bound` is 'the winner is the last element', which means "
            f"'the top of the grid' only for an ascending grid -- on an unsorted "
            f"one the flag would be reported as if it did")

    spec = env[3]
    if action_of is None:
        shape = spec["action_shape"]
        lo = jnp.asarray(spec["action_low"])
        hi = jnp.asarray(spec["action_high"])
        action_of = lambda v: jnp.clip(jnp.full(shape, v), lo, hi)

    reset_keys = reset_states = None
    if days is None:
        if n_envs is None:
            raise ValueError("pass `days` (one environment per named day) or "
                             "`n_envs` (that many drawn at random); neither was "
                             "given, so the environment axis has no length")
    else:
        if day_of_state is None:
            raise ValueError(
                "`days` needs `day_of_state`: the day is found by searching for "
                "a key that opens it and confirmed by reading it back out of the "
                "state, and only the caller knows how this market carries it")
        days = [int(d) for d in days]
        if n_envs is None:
            n_envs = len(days)
        elif int(n_envs) != len(days):
            raise ValueError(
                f"n_envs={n_envs} against {len(days)} named days: with days "
                f"pinned the environment axis is the days, and any other length "
                f"evaluates something else under their name")
        episode_len = getattr(params, "episode_len", None)
        if episode_len is not None and int(horizon) > int(episode_len):
            raise ValueError(
                f"horizon={horizon} exceeds episode_len={int(episode_len)} while "
                f"days are pinned. The step where the episode ends resets from "
                f"the step key, so every step after it would be rolled on a day "
                f"drawn at random and reported under the named days")
        if "reset_on_day" in spec:
            # This market can be asked for a day, so the grid opens
            # each day at its own first period instead of at whatever period a
            # key search happened to land on.  Market 03 measured what the
            # search cost: 265 of 576 pinned half hours were scored on the day
            # that followed, all of them training days, so the grid's best level
            # and the honest arm it is compared against were not computed on the
            # same periods.
            if period_of_state is None:
                raise ValueError(
                    "this market's `spec` offers `reset_on_day`, so the grid "
                    "opens days at their first period and needs "
                    "`period_of_state` to read the offset back out. Passing it "
                    "is what makes the opening checkable; without it this would "
                    "fall back to the key search and silently keep scoring the "
                    "grid on periods the other arms never saw")
            open_day_start, = _from_evaluation("open_day_start")
            opened = [open_day_start(spec["reset_on_day"], params, d,
                                     day_of_state, period_of_state,
                                     spec["periods_per_day"])[1]
                      for d in days]
            reset_states = jax.tree_util.tree_map(
                lambda *xs: jnp.stack(xs), *opened)
        else:
            open_day, = _from_evaluation("open_day")
            reset_keys = jnp.stack([open_day(env[0], params, d, day_of_state)[0]
                                    for d in days])

    rows = []
    for v in levels:
        m = rollout_action(env, params, action_of(v), n_envs, horizon, key,
                           reset_keys=reset_keys, reset_states=reset_states,
                           voll=voll, other_shortfall_cost=other_shortfall_cost,
                           other_shortfall_cost_key=other_shortfall_cost_key)
        # the days go in every row rather than once beside them: a row is what a
        # product is written from, and a row that cannot state which days it was
        # computed on is not comparable against another one
        rows.append({"level": v, "days": (list(days) if days is not None
                                          else None),
                     **{kk: np.asarray(vv).tolist() for kk, vv in m.items()}})
    best = int(np.argmax([r["reward_mean"] for r in rows]))
    return rows, best, best == len(levels) - 1
