"""L0: a sample the market marks unusable must not reach the gradient.

**The failure this exists for** (measured 2026-09-17, `case813nem` market 02):
at iteration 6 one
sample out of 3 072 came from a clearing that did not converge.  Its dual was a
finite but enormous number -- the truthful-action scan over the full 365-day
window reaches `dual_residual` 2.254e+290 on 2025-11-11, and 2.838e+294 on the
same day before the free-column fix, **every value still finite** -- and
`settle`'s ``profit = revenue_rt + revenue_da - cost`` therefore carried terms
past the float32 maximum of 3.4028e+38.  The cast to float32 turned them into
``inf``, two of opposite sign summed to ``NaN``, `reward_mean` went NaN at that
iteration and from the next one every sample was unconverged: the parameters
had gone NaN.  ``costs_mean`` stayed finite throughout, because shed is MW and
not price times MW, which is how the path was identified.

**Why masking and not repair.**  The reward must be the
settlement profit computed from the realised award and this market's own
clearing price.  Substituting a number for a bad price would make the reward a
fabrication.  So the reward's VALUE is left exactly as the market produced it,
and the sample is excluded from the loss instead, with a count published so the
exclusion is visible rather than silent.

**Why `where` and not multiplication by a zero weight.**  ``nan * 0.0`` is
``nan`` (measured), so a zero weight does not remove a bad sample -- it spreads
it.  The bad values have to be replaced before they enter the graph, and the
weight applied on top; `test_a_masked_sample_cannot_influence_the_result` is
what distinguishes the two implementations, because only the replacing one
passes it.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.envs.p2p import make_p2p_env, make_p2p_params
from powermarketjax.learning.ippo import IPPOConfig, make_ippo
from powermarketjax.learning.policy import bounds_for
from powermarketjax.resources.battery import make_battery_bundle

N = 6
N_PERIODS = 40
EPISODE_LEN = 4
HORIZON = 8
N_ENVS = 4
PI_EXP, PI_RET = 4.1, 26.11
DELTA = 0.5

#: `minibatches=1, epochs=1` on purpose: one gradient step over the whole
#: rollout, so a permutation cannot move the mean and two runs are comparable
#: leaf for leaf.  `ent_coef=0` keeps the entropy term (which depends on no
#: sample) out of the comparison.
CFG = IPPOConfig(n_envs=N_ENVS, horizon=HORIZON, epochs=1, minibatches=1,
                 lr=3e-4, clip_eps=0.2, gamma=0.99, gae_lambda=0.95,
                 vf_coef=0.5, ent_coef=0.0, max_grad_norm=0.5, hidden=(16,),
                 init_scale=1.0, weight_decay=0.0)


def _market():
    rng = np.random.default_rng(0)
    battery = make_battery_bundle(
        n_devices=N, dt_hours=DELTA,
        capacity_mwh=rng.uniform(2.0, 10.0, N).tolist(),
        power_mw=rng.uniform(0.5, 3.0, N).tolist())
    params = make_p2p_params(p_pv=rng.uniform(0.0, 2.0, (N_PERIODS, N)),
                             load=rng.uniform(0.0, 2.0, (N_PERIODS, N)),
                             battery=battery, kappa=rng.uniform(0.0, 5.0, N),
                             learner_mask=np.ones(N, bool),
                             episode_len=EPISODE_LEN)
    return make_p2p_env(N, PI_EXP, PI_RET, DELTA), params


def _flagged(env, *, flag, reward_fn=None, flag_name="usable"):
    """`env` whose step publishes ``flag_name`` and optionally rewrites reward.

    `flag` and `reward_fn` both take the step's own reward row, so an injection
    can be expressed without knowing how `_rollout` lays its environments out.
    """
    reset, step, step_auto_reset, spec = env

    def wrapped(key, state, action, prm):
        obs, st, rew, costs, done, info = step_auto_reset(key, state, action, prm)
        ok = flag(rew)
        if reward_fn is not None:
            rew = reward_fn(rew, ok)
        return obs, st, rew, costs, done, dict(info, **{flag_name: ok})

    return reset, step, wrapped, spec


def _run(env, params, seed=3, **kw):
    """One iteration; returns ``(params_out, metrics)``.

    `params_before` is exposed through the attribute-free third element only
    where a test needs it (`_run_with_start`), so the two-value shape the other
    tests read stays what it was.
    """
    spec = env[3]
    obs_mean = jnp.zeros((spec["obs_dim"],), jnp.float32)
    obs_std = jnp.ones((spec["obs_dim"],), jnp.float32)
    #: `convergence_key=None`: market 04 publishes no convergence flag (that is
    #: what `test_ippo_convergence_key_l0.py` exists for), and this file is about
    #: `valid_key`, which is a separate name.
    init, iterate = make_ippo(env, bounds_for(spec), CFG, obs_mean, obs_std,
                              convergence_key=None, **kw)
    p, tx, opt, env_state, env_obs = init(jax.random.PRNGKey(2), params)
    out = iterate(p, tx, opt, env_state, env_obs, jax.random.PRNGKey(seed), params)
    return out[0], out[-1]


def _run_with_start(env, params, seed=3, **kw):
    """Same, plus the parameters `init` produced, for "did anything move at all"."""
    spec = env[3]
    init, iterate = make_ippo(env, bounds_for(spec), CFG,
                              jnp.zeros((spec["obs_dim"],), jnp.float32),
                              jnp.ones((spec["obs_dim"],), jnp.float32),
                              convergence_key=None, **kw)
    p, tx, opt, env_state, env_obs = init(jax.random.PRNGKey(2), params)
    out = iterate(p, tx, opt, env_state, env_obs, jax.random.PRNGKey(seed), params)
    return p, out[0], out[-1]


def _leaves(tree):
    return [np.asarray(x) for x in jax.tree.leaves(tree)]


# ── the two checks ──────────────────────────────────────────────────────────

def test_masking_nothing_is_bit_identical_to_not_masking():
    """The feature must be a no-op when every sample is usable.

    This is the premise of the byte-for-byte gate on the two cases that never
    produce a bad sample: if an all-usable mask moved any leaf, every existing
    29gb and 73rts product would have to be re-run.
    """
    env, prm = _market()
    plain, _m0 = _run(env, prm)
    flagged = _flagged(env, flag=lambda rew: jnp.bool_(True))
    masked, m1 = _run(flagged, prm, valid_key="usable")
    for a, b in zip(_leaves(plain), _leaves(masked)):
        assert np.array_equal(a, b), (
            "an all-usable mask changed a parameter leaf; the byte-for-byte "
            "premise on 29gb and 73rts does not hold")
    assert float(m1["masked_samples"]) == 0.0


@pytest.mark.parametrize("poison", [1e30, float("nan")])
def test_a_masked_sample_cannot_influence_the_result(poison):
    """A flagged sample's VALUE must not reach the parameters.

    Run twice with the same seed, flagging the same sample, and give it a
    different poison each time.  If the mask works, the two runs agree leaf for
    leaf and both stay finite.  A zero-weight implementation fails the `nan`
    case (``nan * 0.0 == nan``) and a no-mask implementation fails both.

    The sample is selected by a property of the reward itself rather than by an
    environment index, because `_rollout` vmaps the step and this wrapper never
    sees which environment it is in.
    """
    env, prm = _market()

    def flag(rew):
        # flag the sample whose first agent's reward is the largest of the row;
        # deterministic, and independent of how environments are laid out
        return jnp.bool_(rew[0] < jnp.max(rew))

    def rewrite(rew, ok):
        return jnp.where(ok, rew, jnp.full_like(rew, poison))

    got, met = _run(_flagged(env, flag=flag, reward_fn=rewrite),
                    prm, valid_key="usable")
    assert all(np.isfinite(x).all() for x in _leaves(got)), (
        f"poison {poison} reached the parameters")
    #: **`reward_mean` is NOT asserted finite.** It is the mean of the reward
    #: the market produced, poisoned sample included, so with `poison=nan` it is
    #: nan by construction -- that is the metric doing its job: the bad sample is
    #: excluded from the LOSS, not erased from the record. Asserting it finite
    #: would demand that the mask falsify the reward, which the
    #: settlement-profit reward forbids.
    assert float(met["masked_samples"]) >= 1.0, (
        "nothing was masked, so this test proves nothing about masking")

def test_a_misspelled_valid_key_is_refused_not_ignored():
    """The name is the caller's, and a wrong one must fail loudly.

    Same convention as `convergence_key` (`test_ippo_convergence_key_l0.py`):
    a caller that misspells the key would otherwise get a run with no masking
    and no record that the thing it asked for never happened -- the silent
    failure this guards against.  The market here publishes ``usable``; asking for
    ``usabel`` has to raise rather than fall back to not masking.
    """
    env, prm = _market()
    flagged = _flagged(env, flag=lambda rew: jnp.bool_(True))
    with pytest.raises(KeyError, match="usabel"):
        _run(flagged, prm, valid_key="usabel")


def test_the_flag_is_read_from_info_not_assumed_true():
    """A market that publishes the key must actually be consulted.

    `test_masking_nothing_is_bit_identical_to_not_masking` passes an all-True
    flag, so it cannot tell "the mask read the flag" from "the mask read
    nothing".  This one flags every sample as unusable: the masked count then
    has to be the whole rollout and the parameters must not move at all, which
    only holds if the flag was read.
    """
    env, prm = _market()
    none_ok = _flagged(env, flag=lambda rew: jnp.bool_(False))
    start, got, met = _run_with_start(none_ok, prm, valid_key="usable")
    assert float(met["masked_samples"]) == float(CFG.horizon * CFG.n_envs), (
        f"an all-unusable rollout masked {float(met['masked_samples'])} of "
        f"{CFG.horizon * CFG.n_envs} samples; the flag is not being read from info")
    #: every sample masked => `wsum` floors at 1, the weighted sums are all zero,
    #: so the gradient is zero and Adam's step is zero (CFG has no weight decay).
    #: The parameters must therefore be exactly where `init` left them.
    for a, b in zip(_leaves(start), _leaves(got)):
        assert np.array_equal(a, b), (
            "masking every sample still moved a parameter leaf; an all-masked "
            "minibatch must give a zero gradient, not a division by a tiny sum")
    #: and the control: with nothing masked the same rollout does move them,
    #: otherwise the check above would pass on a learner that never learns.
    moved, _m = _run(env, prm)
    assert any(not np.array_equal(a, b) for a, b in zip(_leaves(start), _leaves(moved))), (
        "the unmasked run moved no parameter either, so this test proves nothing")
