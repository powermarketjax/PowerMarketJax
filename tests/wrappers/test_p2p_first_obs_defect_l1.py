"""L1: the defect replicator reproduces the out-of-package defect, exactly.

`make_pooled_ippo(..., replicate_external_first_obs_defect=True)` exists for one
experiment: the two learning arms' paired differences sit on opposite sides of
the same reference (+0.030499 in package, -0.0260 out of it), and the
out-of-package pipeline is known to feed the first observation of every episode
to its network unscaled while dividing the other 95
(`restrict_starts`'s docstring).
The evaluation-side injection already priced that defect at -0.391071 of return
-- thirteen times the effect in question -- but only for a policy that was NOT
trained under it, and the out-of-package policies were.  So the question needs
the defect inside training, and that needs the replicator to be the defect
rather than something merely similar.

Two things are checked, and the second is the one that could quietly be wrong.

*What the network sees.*  With the flag on, the observation reaching the policy
at the reset step is the RAW one, and the standardised one at every other step.
Asserted per channel, because the divisor vector spans six and a half orders of
magnitude, so an error on the small channels would be invisible in any norm.

*That the flag is off by default.*  A replicator that leaked into the default
path would silently make every 04 run a run of a defective market, and nothing
downstream would say so.

**What this does not catch**: whether the out-of-package pipeline has only this
defect.  That is section 7's claim, measured there, and not re-derived here.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.envs.p2p import make_p2p_env, make_p2p_params
from powermarketjax.resources.battery import make_battery_bundle
from powermarketjax.wrappers.p2p import (make_pooled_ippo,
                                         restrict_starts,
                                         unscaled_first_obs)

N = 6
N_PERIODS = 40
EPISODE_LEN = 8
PI_EXP, PI_RET = 4.1, 26.11
DELTA = 0.5


def build():
    rng = np.random.default_rng(0)
    battery = make_battery_bundle(
        n_devices=N, dt_hours=DELTA,
        capacity_mwh=rng.uniform(2.0, 10.0, N).tolist(),
        power_mw=rng.uniform(0.5, 3.0, N).tolist())
    params = make_p2p_params(
        p_pv=rng.uniform(0.0, 2.0, (N_PERIODS, N)),
        load=rng.uniform(0.0, 2.0, (N_PERIODS, N)),
        battery=battery, kappa=np.full(N, 3.0),
        learner_mask=np.ones(N, bool), episode_len=EPISODE_LEN)
    return make_p2p_env(N, PI_EXP, PI_RET, DELTA), params


def _obs_std(spec, rng):
    """A divisor vector with the same character as the real one: channels that
    differ by many orders of magnitude, so a per-channel error cannot hide."""
    d = int(spec["obs_dim"])
    return jnp.asarray(10.0 ** rng.uniform(-3.0, 3.0, d), jnp.float32)


def test_the_replicator_returns_the_raw_observation_through_norm():
    """What the network sees at the reset step is the RAW observation, and the
    standardised one at every other step.

    Asserted per channel: the real divisor vector spans six and a half orders of
    magnitude, so an error confined to the small channels would be invisible in
    any norm taken over the vector.
    """
    env, params = build()
    spec = env[3]
    rng = np.random.default_rng(7)
    obs_std = _obs_std(spec, rng)
    allowed = jnp.arange(N_PERIODS - EPISODE_LEN + 1, dtype=jnp.int32)

    plain = restrict_starts(env, allowed)
    defective = unscaled_first_obs(plain, obs_std)

    key = jax.random.PRNGKey(3)
    raw_obs, state_plain = plain[0](key, params)
    handed_over, state_def = defective[0](key, params)

    # `_norm` with obs_mean = 0 is a division by obs_std; this is what each path
    # puts in front of the network at the reset step
    seen_plain = np.asarray(raw_obs / obs_std, np.float64)
    seen_defective = np.asarray(handed_over / obs_std, np.float64)

    np.testing.assert_allclose(seen_defective, np.asarray(raw_obs, np.float64),
                               rtol=1e-5, atol=0)

    # per channel the two differ by exactly the divisor: the check that this is
    # the defect and not merely a perturbation
    # the observation is `(n_agents, obs_dim)` and the divisor is per channel,
    # so the reference is broadcast the same way the division is
    divisor = np.broadcast_to(np.asarray(obs_std, np.float64), seen_plain.shape)
    live = seen_plain != 0.0
    assert live.sum() >= seen_plain.size // 2, (
        "too many entries are exactly zero here for the per-channel check to "
        "have content")
    np.testing.assert_allclose(seen_defective[live] / seen_plain[live],
                               divisor[live], rtol=1e-4, atol=0)

    # and it is the OBSERVATION that moved, not the state the episode starts in
    for a, b in zip(jax.tree_util.tree_leaves(state_plain),
                    jax.tree_util.tree_leaves(state_def)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_it_touches_the_reset_step_only():
    """Steps after the reset must be untouched: the defect is one observation
    out of every 96, and a replicator that scaled all of them would be a
    different experiment giving a much larger effect."""
    env, params = build()
    spec = env[3]
    rng = np.random.default_rng(13)
    obs_std = _obs_std(spec, rng)
    allowed = jnp.arange(N_PERIODS - EPISODE_LEN + 1, dtype=jnp.int32)
    plain = restrict_starts(env, allowed)
    defective = unscaled_first_obs(plain, obs_std)

    key = jax.random.PRNGKey(5)
    action = jax.random.uniform(jax.random.PRNGKey(6), (N, 2), jnp.float32,
                                -1.0, 1.0)
    _o1, s1 = plain[0](key, params)
    _o2, s2 = defective[0](key, params)
    out1 = plain[1](key, s1, action, params)
    out2 = defective[1](key, s2, action, params)
    for a, b in zip(jax.tree_util.tree_leaves(out1),
                    jax.tree_util.tree_leaves(out2)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_the_replicator_is_off_by_default():
    """A defect that became the default would make every run of this market a
    run of a defective market, with nothing downstream saying so."""
    import inspect
    p = inspect.signature(make_pooled_ippo).parameters[
        "replicate_external_first_obs_defect"]
    assert p.default is False
    assert p.kind is inspect.Parameter.KEYWORD_ONLY, (
        "it must be keyword-only: a positional would let a caller switch the "
        "defect on by miscounting arguments")
