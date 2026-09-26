"""Wiring market 04 into `powermarketjax.learning.ippo`.

Three divergences separate the P2P market from the four the shared learner was
built against, and each is absorbed here rather than in `ippo.py` or in
`envs/p2p/`.  They were measured, not recalled (2026-08-27, CPU).

**One is a false conflict and was fixed in the learner instead**, so it is named
here only to say why it is absent: `_rollout` used to read `info["converged"]`
unconditionally and this market emits no such key -- its clearing is two sorted
orders, two cumulative sums and a comparison reduction, with no solver, no
tolerance and no iteration bound.  `make_ippo` now takes `convergence_key`, and
`make_pooled_ippo` below passes `None`.  Feeding a constant `True` from here was
the alternative and was rejected: `unconverged_frac` would then read 0.0 for a
market where no solver ever ran, which is a property of this adapter wearing the
appearance of a measurement.

**Two is the type of `spec["baseline_action"]`.**  `adapters.unpack_env` records
that the name is shared between markets and the type is not -- an array in the
ancillary market, a function of the state here, because the truthful side of the
offer depends on the current net position -- and it deliberately passes through
whichever it finds rather than normalising the two, since normalising would hide
the question.  `learning.observation_statistics` consumes the key as an array
and therefore cannot drive this market (measured: `TypeError: float() argument
must be a string or a real number, not 'function'`).  Both sides are right, so
the adaptation goes in the one place that knows both:
`baseline_observation_statistics` below runs the market under
``learner_mask`` all False and lets the environment substitute its own truthful
action.

**Three is which episodes a run may see.**  The pool rule itself stays in the
experiment -- `tools/p2p_experiment/constrained_baseline.make_start_pools` is
its one definition and this module consumes the array it returns rather than
restating the rule, so the two cannot drift apart.  What belongs here is the
consequence for a rollout longer than one episode, which the experiment's own
`restrict_starts` names as out of its scope and the shared learner walks
straight into; `make_pooled_ippo` is where that is refused.
"""
from typing import Callable, Dict, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ..learning.ippo import make_ippo

__all__ = ["restrict_starts", "baseline_observation_statistics",
           "make_pooled_ippo"]


def restrict_starts(env, allowed):
    """`env` with ``reset`` drawing its episode start from `allowed` only.

    Measured wording follows `tools/p2p_experiment/constrained_baseline.py`,
    which is where this transformation was first written and where the pool it
    consumes is defined: which episodes a run sees is a property of the
    experiment, and what the environment does is fixed by the market.  This copy exists
    because the shared learner reaches the environment through a different path
    than that file's own rollout does, not because the rule differs.

    **Only ``reset`` is wrapped, and that is not enough on its own.**  The
    auto-reset lives inside `envs/p2p/env.py`'s ``step``, which calls the
    ``reset`` it closed over -- not this one -- so every episode boundary a
    rollout crosses draws a start over the whole panel.  Measured at
    ``episode_len = 96`` and ``horizon = 192``: half the emitted cells belong to
    a second episode and 25.0% of the boundary draws (16 of 64) land outside the
    training pool, 14.1% (9 of 64) squarely inside a held-out block.
    `make_pooled_ippo` is what keeps a rollout from crossing a boundary at all;
    used without it, this function restricts the first episode and nothing else.

    **The recomputed observation is safe here and is not safe in general.**  The
    reset observation is rebuilt from ``spec["get_obs"]``, which is the market's
    own function and knows nothing about wrappers applied outside it.  That is
    correct in this package's path, where no observation transform is wrapped
    around the environment at all -- the standardisation lives inside
    `make_ippo`'s `_norm` and every observation the network sees goes through it.
    Compose this after a wrapper that transforms observations and the reset one
    silently comes back untransformed: measured 2026-08-27 on
    `tools/p2p_experiment`, where `restrict_starts` is applied after
    `scale_observations`, the first observation of every episode is bit-identical
    to the raw one while the other 95 are divided, and the per-channel ratio
    between the reset observation and the next step reproduces the divisor vector
    exactly.  So: this wrapper goes closest to the environment, or the transform
    goes inside the learner.
    """
    reset, step, step_auto_reset, spec = env
    get_obs = spec["get_obs"]
    allowed = jnp.asarray(np.asarray(allowed, np.int32))
    if allowed.ndim != 1 or allowed.shape[0] < 1:
        raise ValueError(
            f"`allowed` must be a non-empty 1-D array of episode starts, got "
            f"shape {allowed.shape}: refused at wiring time rather than at the "
            f"first draw, because `jax.random.randint` over an empty range "
            f"returns 0 without complaining (measured) and what the gather "
            f"after it does with that index depends on whether the caller "
            f"jitted")

    def restricted_reset(key, params):
        """The market's own reset, with the start replaced by a pool draw.

        The key is split first, so the draw and whatever the market's reset
        consumes are independent; everything the market's reset sets other than
        the cursor is kept, and the observation is recomputed from the moved
        state rather than from the one that was drawn.
        """
        draw_key, rest_key = jax.random.split(key)
        _, state = reset(rest_key, params)
        idx = jax.random.randint(draw_key, (), 0, allowed.shape[0])
        state = state.replace(cursor=allowed[idx])
        return get_obs(state, params), state

    return restricted_reset, step, step_auto_reset, spec


def baseline_observation_statistics(env, env_params, key, n_envs: int,
                                    horizon: int):
    """`learning.observation_statistics`, for a market whose baseline is a function.

    Same quantity and same shape: the mean and the standard deviation of every
    observation coordinate under the truthful action, pooled over steps,
    environments and agents, with the deviation floored so a constant coordinate
    does not divide by zero.  The key derivation is the shared function's,
    verbatim including its reuse of ``keys[0]`` as the scan root, so the two
    devices do not answer one question two ways; that reuse is the deliberate
    departure from the explicit-PRNG rule which that function documents.

    **What differs is where the truthful action comes from.**  The shared
    function reads it out of ``spec["baseline_action"]``, which this market
    declares as a function of the state.  Here the market applies it itself:
    ``learner_mask`` all False makes ``step`` replace the submitted action with
    ``baseline_action(p_pv_t, load_t)`` for every agent, so the array passed in
    is provably never read.  It is zeros, and the mask -- not a comment -- is
    what makes that safe; the shape-correct array in a doctored ``spec`` copy is
    the arrangement this one exists to avoid, since nothing there would stop a
    later caller with ``learner_mask`` True from silently training against a
    wrong baseline.
    """
    reset, _step, step_auto_reset, spec = env
    if not bool(np.any(np.asarray(env_params.learner_mask))):
        raise ValueError(
            "env_params.learner_mask is already all False, so this function "
            "cannot tell the mask it sets apart from the one it was handed; "
            "pass the parameters the run trains on")
    params = env_params.replace(
        learner_mask=jnp.zeros_like(env_params.learner_mask))
    ignored = jnp.zeros(tuple(int(d) for d in spec["action_shape"]), jnp.float32)
    keys = jax.random.split(key, n_envs)
    obs, state = jax.vmap(reset, in_axes=(0, None))(keys, params)

    def one(carry, _):
        state, obs, k = carry
        k, k_env = jax.random.split(k)
        env_keys = jax.random.split(k_env, n_envs)
        nxt_obs, nxt_state, *_ = jax.vmap(
            step_auto_reset, in_axes=(0, 0, None, None))(
                env_keys, state, ignored, params)
        return (nxt_state, nxt_obs, k), obs

    _, seen = jax.lax.scan(one, (state, obs, keys[0]), None, length=horizon)
    mean = jnp.mean(seen, axis=(0, 1, 2))
    std = jnp.std(seen, axis=(0, 1, 2))
    return mean, jnp.where(std > 1e-8, std, 1.0)


def unscaled_first_obs(env, obs_std):
    """`env` with the out-of-package defect put back in, on purpose.

    The out-of-package pipeline applies `scale_observations` and then
    `restrict_starts`, and the second recomputes the reset observation from
    `spec["get_obs"]`, which knows nothing about the first.  So the first
    observation of every episode reaches the network raw while the other 95 are
    divided -- measured 2026-08-27, bit-identical to the raw one, with the
    per-channel ratio to the next step reproducing the divisor vector exactly
    (`restrict_starts`'s docstring).

    In this package the standardisation is `make_ippo`'s `_norm`, which computes
    `(obs - obs_mean) / obs_std`, and market 04's driver sets `obs_mean` to zero.
    So handing the reset observation over pre-multiplied by `obs_std` makes
    `_norm` return the raw observation at that one step and changes nothing
    else: point for point the out-of-package situation.

    **This is a defect replicator and must never be anybody's default.**  It
    exists because the evaluation-side injection could only price the defect for
    a policy that was not trained under it, while the out-of-package policies
    were, so the question "implementation difference or that defect" needs the
    defect inside training.  A run under it is not a run of this market.

    One implementation, used by `make_pooled_ippo` for the training path and by
    the driver for the evaluation path, because those two must not be able to
    replicate the defect differently.
    """
    reset, step, step_auto_reset, spec = env

    def reset_with_unscaled_obs(key, params):
        obs, state = reset(key, params)
        return obs * obs_std, state

    return reset_with_unscaled_obs, step, step_auto_reset, spec


def make_pooled_ippo(env: Tuple[Callable, Callable, Callable, Dict],
                     allowed,
                     bounds: Tuple[jnp.ndarray, jnp.ndarray],
                     cfg,
                     obs_mean: jnp.ndarray,
                     obs_std: jnp.ndarray,
                     *,
                     episode_len: int,
                     extra_info_keys: Sequence[str] = (),
                     per_agent_params: bool = False,
                     replicate_external_first_obs_defect: bool = False):
    """`(init, iterate)` for market 04, with every episode drawn from `allowed`.

    Two things stand between the shared learner and a run whose episodes all
    come from one pool, and both are settled here rather than left to a driver
    to remember.

    **A rollout may not cross an episode boundary.**  `restrict_starts` cannot
    reach the auto-reset inside ``step``, so a boundary inside the horizon draws
    the next start over the whole panel -- measured at 25.0% of boundary draws
    outside the training pool.  ``horizon`` is therefore required to equal
    ``episode_len``, which is also the shape the out-of-package experiment runs
    in, and matching it is what the item-by-item comparison against those
    numbers needs.  At that setting the only boundary is the last step of the
    rollout, where `_gae`'s terminal branch multiplies the continuation by
    ``cont = 0``; the value it would have bootstrapped from therefore never
    enters the arithmetic, and that identity is asserted by
    `tests/learning/test_ippo_gae_terminal_l1.py` rather than by this docstring.

    **The carry a previous iteration ended in is that same unrestricted draw.**
    `env_state` and `env_obs` are accepted so the driver loop is the one the
    other markets use, and are then replaced by a fresh pooled reset before
    every iteration.  That is the out-of-package loop's arrangement as well --
    each update collects a fresh batch of episodes drawn from the training pool
    -- and it is done here so that forgetting it is not possible.

    `convergence_key=None` is passed on this market's behalf, so `metrics`
    carries neither `step_converged` nor `unconverged_frac`; the module
    docstring says why a constant was not supplied instead.

    **Which normalisation the 04 runs adopt, and why it is not this module's
    `baseline_observation_statistics`.**  `obs_mean` and `obs_std` are the
    caller's, and the adopted setting for this market is ``obs_mean = 0`` with
    ``obs_std`` the per-channel divisor vector
    `tools/p2p_experiment/constrained_baseline.observation_scale` derives from
    the market parameters, which makes `_norm` the division the out-of-package
    runs already used.  Two reasons, and the first is measured rather than
    preferred (2026-08-27, CPU, real panel, N=16, one 96-step baseline rollout
    over 16 environments):

    * The statistics of a rollout under the truthful action are statistics of a
      run in which the battery never moves.  `soc` therefore has zero variance
      there and is floored to a divisor of 1.0, so the coordinate a battery
      policy is most about would be standardised by a number measured where it
      was constant, against a real range of 0.15 to 1.0.  The floor misses two
      more: `eta_charge` and `eta_discharge` are constants whose float32
      standard deviation comes out at 1.19e-7, above the 1e-8 floor, so they are
      divided by that -- 8.4e6 times smaller than the divisor the scale vector
      gives them.  They stay bounded only because their numerator is exactly
      constant (measured: both come out at exactly +1.0).  Across the fifteen
      channels the two transformations differ by a factor of 1.6e9 end to end.
    * A divisor vector derived from the scenario is reproducible without a
      calibration artefact travelling with the checkpoint, which is the failure
      `tools.benchmark.run_rl_01.write_params` documents and which was found
      on this repository's market 02 archives.

    The cost is recorded rather than hidden: the inputs are not zero-mean, and
    the item-by-item comparison this adapter exists for is what pays for that
    choice -- matching the out-of-package transformation removes one entry from
    the list of differences that separates the two learning arms, and the arms
    that do not learn are unaffected either way, since `truthful` reads no
    observation and `self_consumption` is invariant under the shared divisor of
    channels 0, 6 and 7.

    `replicate_external_first_obs_defect` deliberately reintroduces the defect
    `restrict_starts` documents, for one experiment only; see the comment at its
    branch below.  It defaults off and a run under it is not a run of this
    market.
    """
    episode_len = int(episode_len)
    if cfg.horizon != episode_len:
        raise ValueError(
            f"cfg.horizon={cfg.horizon} must equal episode_len={episode_len}: a "
            f"longer rollout crosses an episode boundary, and the auto-reset "
            f"there draws its start over the whole panel rather than from the "
            f"pool, which puts held-out days into training (measured: 25.0% of "
            f"boundary draws outside the training pool at horizon=192). A "
            f"shorter one would train on a prefix of every episode and never on "
            f"its end, which is where this market settles the stock")
    pooled = restrict_starts(env, allowed)
    if replicate_external_first_obs_defect:
        pooled = unscaled_first_obs(pooled, obs_std)
    pooled_reset = pooled[0]
    init, one_iteration = make_ippo(
        pooled, bounds, cfg, obs_mean, obs_std,
        extra_info_keys=extra_info_keys, per_agent_params=per_agent_params,
        convergence_key=None)

    def iterate(params, tx, opt_state, env_state, env_obs, key, env_params):
        """One PPO iteration, started from a fresh draw out of the pool.

        `env_state` and `env_obs` are deliberately unused: see the docstring
        above.  They stay in the signature so this is a drop-in for the
        `iterate` the other four markets are driven by.
        """
        del env_state, env_obs
        key, k_reset = jax.random.split(key)
        keys = jax.random.split(k_reset, cfg.n_envs)
        env_obs, env_state = jax.vmap(pooled_reset, in_axes=(0, None))(
            keys, env_params)
        return one_iteration(params, tx, opt_state, env_state, env_obs, key,
                             env_params)

    return init, iterate


def make_pooled_sac(env: Tuple[Callable, Callable, Callable, Dict],
                    allowed,
                    bounds: Tuple[jnp.ndarray, jnp.ndarray],
                    cfg,
                    obs_mean: jnp.ndarray,
                    obs_std: jnp.ndarray,
                    *,
                    episode_len: int,
                    per_agent_params: bool = False):
    """`(init, iterate)` for market 04 under SAC, every episode drawn from `allowed`.

    The same two settlements `make_pooled_ippo` makes -- horizon pinned to
    `episode_len` so no rollout crosses an auto-reset, and a fresh pooled reset
    before every iteration in place of the carry -- applied to
    `powermarketjax.learning.sac.make_sac`, whose `init` / `iterate` take the
    same positions as `make_ippo`'s (`iterate(params, tx, learner, env_state,
    env_obs, key, env_params)`), so the driver loop is unchanged.  Added
    2026-09-20 because the SAC columns of this market had only ever run
    on `tools/p2p_experiment/constrained_baseline.py`, which stores no policy,
    so their learned prices could not be read back; this path writes
    `final.params.npz` like the IPPO one.

    `convergence_key=None` for the reason the IPPO wrapper gives.  The defect
    replicator is deliberately not offered here: it belongs to one closed
    experiment on the IPPO path.
    """
    from powermarketjax.learning.sac import make_sac

    episode_len = int(episode_len)
    if cfg.horizon != episode_len:
        raise ValueError(
            f"cfg.horizon={cfg.horizon} must equal episode_len={episode_len} "
            f"(same boundary argument as make_pooled_ippo)")
    pooled = restrict_starts(env, allowed)
    pooled_reset = pooled[0]
    init, one_iteration = make_sac(pooled, bounds, cfg, obs_mean, obs_std,
                                   per_agent_params=per_agent_params,
                                   convergence_key=None)

    def iterate(params, tx, learner, env_state, env_obs, key, env_params):
        del env_state, env_obs
        key, k_reset = jax.random.split(key)
        keys = jax.random.split(k_reset, cfg.n_envs)
        env_obs, env_state = jax.vmap(pooled_reset, in_axes=(0, None))(
            keys, env_params)
        return one_iteration(params, tx, learner, env_state, env_obs, key,
                             env_params)

    return init, iterate
