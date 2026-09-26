"""L0: the convergence flag is a name the caller supplies, not one this module knows.

`_rollout` used to read ``info["converged"]`` unconditionally.
Four of the five markets publish it -- each runs an iterative solve and each
derives the flag from its own residual at its own tolerance -- and market 04
does not: its clearing is two sorted orders, two cumulative sums, a comparison
reduction and a gather, with no solver, no tolerance and no iteration bound.
Wiring 04 in therefore failed with a bare ``KeyError: 'converged'`` at the line
that read it (measured 2026-08-27, CPU).

**Why the repair is here and not in an adapter.**  `make_ippo`'s own docstring
for `extra_info_keys` states the criterion: a market-specific name appearing in
this shared module is itself the warning sign, because a name that also exists
in another market would silently pick up that market's quantity.  `converged`
was such a name.  The alternative was for the adapter to publish a constant
``True``; that was rejected because `unconverged_frac` would then report 0.0 for
a market where no solver ever ran -- a property of the adapter wearing the
appearance of a measurement.

**What must not move.**  The default is still the name the four solver-backed
markets use, so every driver on disk keeps recording the two entries it records
today; `test_named_path_is_bit_identical_to_the_old_arithmetic` is the check
that naming it changes nothing else.
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

CFG = IPPOConfig(n_envs=N_ENVS, horizon=HORIZON, epochs=1, minibatches=1,
                 lr=3e-4, clip_eps=0.2, gamma=0.99, gae_lambda=0.95,
                 vf_coef=0.5, ent_coef=0.0, max_grad_norm=0.5, hidden=(16,),
                 init_scale=1.0, weight_decay=0.0)


def _market():
    """Market 04 and its parameters: the one market that publishes no flag."""
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


def _with_flag(env, name="converged"):
    """`env` with one boolean `info` entry added under `name`, nothing else.

    A measuring device, not a proposal: it exists so that one market can be run
    down both paths and the arithmetic compared.  It is what this file argues
    against doing in a real adapter.
    """
    reset, step, step_auto_reset, spec = env

    def flagged(key, state, action, prm):
        obs, st, rew, costs, done, info = step_auto_reset(key, state, action, prm)
        return obs, st, rew, costs, done, dict(info, **{name: jnp.bool_(True)})

    return reset, step, flagged, spec


def _run(env, params, **kw):
    """One iteration; returns the metrics dict."""
    spec = env[3]
    obs_mean = jnp.zeros((spec["obs_dim"],), jnp.float32)
    obs_std = jnp.ones((spec["obs_dim"],), jnp.float32)
    init, iterate = make_ippo(env, bounds_for(spec), CFG, obs_mean, obs_std, **kw)
    carry = init(jax.random.PRNGKey(2), params)
    p, tx, opt, env_state, env_obs = carry
    return iterate(p, tx, opt, env_state, env_obs, jax.random.PRNGKey(3), params)[-1]


def test_a_market_without_the_flag_fails_loudly_under_the_default():
    """The default still asks for it, and the refusal names what IS published.

    The old failure was a bare ``KeyError: 'converged'`` raised by a subscript,
    which says neither who asked nor what the market does publish.  Both halves
    are asserted, because a message that merely mentions the key would leave a
    caller in exactly the position this ticket started from.
    """
    env, params = _market()
    with pytest.raises(KeyError) as excinfo:
        _run(env, params)
    message = str(excinfo.value)
    assert "convergence_key" in message, message
    assert "clearing_price" in message and "terminal_obs" in message, (
        f"the refusal must list the keys this market's step does return, so "
        f"the caller can see there is no convergence flag to name: {message}")


def test_none_removes_both_entries_rather_than_defaulting_them():
    """Under `None` the two entries are absent -- not zero, not a constant.

    Absence is the point: a caller that records `unconverged_frac` fails by name
    against a market that cannot produce it, instead of writing 0.0 into a
    column that reads as "the solver converged".
    """
    env, params = _market()
    metrics = _run(env, params, convergence_key=None)
    assert "unconverged_frac" not in metrics and "step_converged" not in metrics, (
        f"convergence_key=None must not produce them; got {sorted(metrics)}")
    # everything else the drivers record is still there
    for name in ("reward_mean", "costs_mean", "step_reward", "step_action",
                 "step_cursor", "reward_per_agent"):
        assert name in metrics, (
            f"{name} disappeared with the flag: {sorted(metrics)}")


def test_named_path_is_bit_identical_to_the_old_arithmetic():
    """Naming the flag changes what is reported and nothing that is computed.

    Both runs are the same market on the same key, one under the default and
    one under `None`; every metric they share must agree bit for bit, because
    the flag enters no expression that feeds the update.  Bit equality rather
    than a tolerance: a tolerance here would pass for a change that moved the
    update, which is the failure this asserts against.
    """
    env, params = _market()
    named = _run(_with_flag(env), params)
    unnamed = _run(_with_flag(env), params, convergence_key=None)
    assert set(named) - set(unnamed) == {"step_converged", "unconverged_frac"}, (
        f"the two paths differ in more than the flag: "
        f"{sorted(set(named) ^ set(unnamed))}")
    for name in sorted(unnamed):
        a, b = np.asarray(named[name]), np.asarray(unnamed[name])
        assert a.shape == b.shape and np.array_equal(a, b), (
            f"{name} moved between the two paths: max|diff| "
            f"{np.max(np.abs(a - b)) if a.shape == b.shape else 'shape'}")


def test_the_name_is_the_callers_and_a_wrong_one_is_refused():
    """A caller may name any entry, and a name no market publishes raises.

    The second half is the one that matters: `extra_info_keys` documents that a
    borrowed name would silently pick up another market's quantity, and the
    same exposure arrives with this parameter, so the failure has to be loud.
    """
    env, params = _market()
    metrics = _run(_with_flag(env, "solve_ok"), params, convergence_key="solve_ok")
    assert float(metrics["unconverged_frac"]) == 0.0
    assert np.asarray(metrics["step_converged"]).shape == (HORIZON, N_ENVS)

    with pytest.raises(KeyError, match="convergence_key"):
        _run(_with_flag(env, "solve_ok"), params, convergence_key="converged")
