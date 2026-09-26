"""L1: a settled boundary is paid for once, not once plus a continuation value.

`ippo._gae` has always stated the criterion -- bootstrap at the
boundary if and only if the environment leaves a state quantity there unpriced
-- and, before this file existed, ended it with "a market that settles a terminal
stock at the boundary instead (declared by ``spec["termination"] == "terminal"``)
must **not** be bootstrapped, and nothing here reads that field".  Market 04 is
the one market that declares it: `envs/p2p/env.py` adds a terminal leg to
`reward` on the step where `done` fires, paying for the energy left in the
battery.  `_gae` now reads the field; this file is what says it reads it
correctly.

**The assertion, and why it needs no GAE formula of its own.**  Under the
terminal rule the last temporal difference is ``delta_T = r_T - V(s_T)`` and the
recursion is cut, so the value target is

    ret_T = gae_T + V(s_T) = (r_T - V(s_T)) + V(s_T) = r_T

exactly -- independent of `gamma` and of `gae_lambda`.  So the check is
``ret == reward`` on every step where `done` is set, compared against the
environment's own reward rather than against anything recomputed here.

**What this file measured, and where it corrected the note that asked for it.**
The note that asked for this file says in its own
words that this is a judgement and not a measurement.  Its conclusion holds --
the boundary must not be bootstrapped -- and two halves of its stated mechanism
did not survive being measured:

* The value that used to be bootstrapped at 04's boundary is **not**
  `info["terminal_obs"]`.  `_rollout` emits the observation `step` returns, and
  that one is already past the auto-reset merge, so what entered `delta` was the
  first observation of the *next* episode -- a state holding `initial_soc`, not
  the stock just sold.  It was never the same energy priced twice; it was
  another episode's state value added on top of a settled boundary.
* At a boundary inside the horizon the bootstrap is the **smaller** of the two
  errors.  On the fixture below, `gamma * V` contributes mean|.| 4.84e-01
  against 1.40e+01 for the recursion running on across the boundary -- 3.4% of
  the total.  It is visible alone only at the last step of the rollout, where
  the recursion term is structurally zero: +4.96e-01 against a terminal leg of
  7.72e+00, i.e. 6.4%.

**Two things this file has to build, and both are prerequisites for wiring market 04
into the learner rather than artefacts of testing.**  04 is not wired into `make_ippo` today
(`powermarketjax/wrappers/` holds only an `__init__.py`), and wiring it hits two
walls before the boundary question arises at all:

* `_rollout` reads ``info["converged"]`` unconditionally and 04's `step` does
  not publish it, so the environment is wrapped here with that one key added.
* `learning.observation_statistics` drives its rollout with
  ``spec["baseline_action"]``, which in 04 is a *function of the state* and not
  an action array (`adapters.py` says so and leaves the divergence alone).  The
  statistics are therefore taken from a rollout with ``learner_mask`` all false,
  which makes the environment substitute the truthful action itself.

Neither wrapper touches the boundary arithmetic.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.envs.p2p import make_p2p_env, make_p2p_params
from powermarketjax.learning.ippo import IPPOConfig, make_ippo
from powermarketjax.learning.policy import bounds_for
from powermarketjax.resources.battery import make_battery_bundle

N = 8
N_PERIODS = 40
#: `horizon` is a whole number of episodes, so `done` fires on a known step in
#: every environment and the check is not left to chance about whether the
#: phenomenon it needs occurred at all.  The fixture asserts it below.
EPISODE_LEN = 4
HORIZON = 12
N_ENVS = 4
PI_EXP, PI_RET = 4.1, 26.11
DELTA = 0.5

#: Relative tolerance on the identity, on a market whose whole state is float32.
#: Measured 2026-08-27, CPU (`JAX_PLATFORMS=cpu`), this fixture: under the
#: masked recursion of a generated done-mask variant of `ippo`
#: -- for 04 every `done` IS a termination, so that variant implements the
#: terminal rule -- the residual is at most 9.5e-7 on a reward scale of
#: mean |r| = 1.05e1, i.e. about 1e-7 relative.  Under `ippo._gae` as it stands
#: the same quantity reaches 9.06e+1.  So this tolerance sits three orders above
#: the float32 noise it must not fail on and nine orders below the defect it
#: must catch.
ATOL = 1e-5
RTOL = 1e-5


def _fixture_arrays():
    rng = np.random.default_rng(0)
    battery = make_battery_bundle(
        n_devices=N, dt_hours=DELTA,
        capacity_mwh=rng.uniform(2.0, 10.0, N).tolist(),
        power_mw=rng.uniform(0.5, 3.0, N).tolist())
    return dict(p_pv=rng.uniform(0.0, 2.0, (N_PERIODS, N)),
                load=rng.uniform(0.0, 2.0, (N_PERIODS, N)),
                battery=battery, kappa=rng.uniform(0.0, 5.0, N),
                episode_len=EPISODE_LEN)


def _internals(iterate):
    """``(_rollout, _gae)`` out of `make_ippo`'s closure.

    `make_ippo` returns ``(init, iterate)`` and `iterate` folds the value target
    straight into an optimiser step: it reports `pg_loss`, `vf_loss`, the
    entropy and the per-step series, and neither `adv` nor `ret`.  The quantity
    this file is about is therefore not observable from the public return, and
    the alternative -- exporting it from `ippo.py` -- would be editing the
    object under test.  A rename upstream raises here with the names actually
    found rather than measuring something else.
    """
    cells = dict(zip(iterate.__code__.co_freevars, iterate.__closure__ or ()))
    missing = [n for n in ("_rollout", "_gae") if n not in cells]
    assert not missing, (
        f"make_ippo's `iterate` no longer closes over {missing}; its free "
        f"variables are {sorted(cells)}.  This test reads the rollout and the "
        f"GAE from that closure because `iterate` returns neither `adv` nor "
        f"`ret`; renaming them means this check has to be re-pointed, not "
        f"deleted")
    return cells["_rollout"].cell_contents, cells["_gae"].cell_contents


@pytest.fixture(scope="module")
def rolled():
    """One rollout of market 04 through `make_ippo`, plus the GAE it produces.

    Returns a dict of numpy arrays; every entry comes out of the environment or
    out of `_gae`, none is recomputed here.
    """
    kw = _fixture_arrays()
    params = make_p2p_params(learner_mask=np.ones(N, bool), **kw)
    truthful = make_p2p_params(learner_mask=np.zeros(N, bool), **kw)
    reset, step, step_auto_reset, spec = make_p2p_env(N, PI_EXP, PI_RET, DELTA)

    assert spec["termination"] == "terminal", (
        f"this file checks the rule for a settled boundary and market 04 is the "
        f"only market that declares one; its spec now reads "
        f"{spec['termination']!r}")

    def step_ac(key, state, action, prm):
        """`step_auto_reset` with `info['converged']` added, and nothing else."""
        obs, st, rew, costs, done, info = step_auto_reset(key, state, action, prm)
        return obs, st, rew, costs, done, dict(info, converged=jnp.bool_(True))

    env = (reset, step, step_ac, spec)

    # observation statistics under the truthful action (module docstring)
    keys = jax.random.split(jax.random.PRNGKey(1), N_ENVS)
    obs0, state0 = jax.vmap(reset, in_axes=(0, None))(keys, truthful)
    zero = jnp.zeros((N_ENVS,) + tuple(spec["action_shape"]), jnp.float32)

    def one(carry, _):
        state, obs, k = carry
        k, k_env = jax.random.split(k)
        nxt_obs, nxt_state, *_ = jax.vmap(step_ac, in_axes=(0, 0, 0, None))(
            jax.random.split(k_env, N_ENVS), state, zero, truthful)
        return (nxt_state, nxt_obs, k), obs

    _, seen = jax.lax.scan(one, (state0, obs0, keys[0]), None, length=HORIZON)
    obs_mean = jnp.mean(seen, axis=(0, 1, 2))
    obs_std = jnp.std(seen, axis=(0, 1, 2))
    obs_std = jnp.where(obs_std > 1e-8, obs_std, 1.0)

    cfg = IPPOConfig(n_envs=N_ENVS, horizon=HORIZON, epochs=1, minibatches=1,
                     lr=3e-4, clip_eps=0.2, gamma=0.99, gae_lambda=0.95,
                     vf_coef=0.5, ent_coef=0.0, max_grad_norm=0.5,
                     hidden=(64, 64), init_scale=1.0, weight_decay=0.0)
    init, iterate = make_ippo(env, bounds_for(spec), cfg, obs_mean, obs_std,
                              extra_info_keys=("terminal_stock_value",))
    _rollout, _gae = _internals(iterate)

    p, _tx, _opt, env_state, env_obs = init(jax.random.PRNGKey(2), params)
    _s, _o, _k, traj, last_value = jax.jit(_rollout)(
        p, env_state, env_obs, jax.random.PRNGKey(3), params)
    adv, ret = jax.jit(_gae)(traj, last_value)

    # The same trajectory through a `_gae` built from a spec that differs in one
    # field, so the other arm of the branch is measured on identical inputs.
    _init_t, iterate_t = make_ippo((reset, step, step_ac,
                                    dict(spec, termination="truncation")),
                                   bounds_for(spec), cfg, obs_mean, obs_std,
                                   extra_info_keys=("terminal_stock_value",))
    _roll_t, _gae_t = _internals(iterate_t)
    _adv_t, ret_t = jax.jit(_gae_t)(traj, last_value)

    out = dict(done=np.asarray(traj["done"]),
               ret_as_truncation=np.asarray(ret_t),
               reward=np.asarray(traj["reward"]),
               value=np.asarray(traj["value"]),
               residual=np.asarray(traj["extra_terminal_stock_value"]),
               adv=np.asarray(adv), ret=np.asarray(ret),
               last_value=np.asarray(last_value),
               gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)

    # Non-triviality, as fixture preconditions rather than as a separate test:
    # the rule being checked has content only where a settled boundary and a
    # non-zero stock both occur.
    fired = np.flatnonzero(out["done"][:, 0])
    assert fired.tolist() == [3, 7, 11], (
        f"`done` fired on steps {fired.tolist()} of {HORIZON}, not on every "
        f"{EPISODE_LEN}th; with no settled boundary inside the rollout this "
        f"check would pass vacuously")
    assert out["done"].all(axis=1)[fired].all(), (
        "`done` did not fire on the same step in every environment; the fixture "
        "resets them together so that the boundary steps are known")
    at = out["done"][..., None] & np.ones_like(out["reward"], bool)
    assert (out["residual"][at] > 0.0).mean() > 0.5, (
        f"only {(out['residual'][at] > 0.0).mean():.1%} of boundary cells carry "
        f"a non-zero terminal stock, so the leg this rule is about is barely "
        f"present; the identity would hold for the wrong reason")
    return out


def _report(d):
    """The magnitudes, so a failure says what it found and not only that it did."""
    at = d["done"][..., None] & np.ones_like(d["reward"], bool)
    excess = (d["ret"] - d["reward"])[at]
    residual = d["residual"][at]
    nxt_v = np.concatenate([d["value"][1:], d["last_value"][None]], axis=0)
    nxt_g = np.concatenate([d["adv"][1:], np.zeros_like(d["adv"][:1])], axis=0)
    boot = (d["gamma"] * nxt_v)[at]
    prop = (d["gamma"] * d["gae_lambda"] * nxt_g)[at]
    lines = [
        f"boundary cells: {at.sum()} of {d['reward'].size}; "
        f"reward scale mean|r| = {np.abs(d['reward'][at]).mean():.4e}",
        f"terminal stock paid in `reward`: mean {residual.mean():+.4e}, "
        f"max {residual.max():+.4e}, non-zero in "
        f"{(residual > 0).mean():.1%} of boundary cells",
        f"excess ret-reward: mean {excess.mean():+.4e}, mean|.| "
        f"{np.abs(excess).mean():.4e}, min {excess.min():+.4e}, max "
        f"{excess.max():+.4e}, positive in {(excess > 0).mean():.1%} of cells",
        f"  = gamma*V(s_after_reset)  mean {boot.mean():+.4e}, mean|.| "
        f"{np.abs(boot).mean():.4e}",
        f"  + gamma*lambda*gae_next   mean {prop.mean():+.4e}, mean|.| "
        f"{np.abs(prop).mean():.4e}",
        f"mean|excess| / mean(residual) = "
        f"{np.abs(excess).mean() / residual.mean():.4f}",
    ]
    for t in np.flatnonzero(d["done"][:, 0]):
        m = np.zeros_like(d["done"], bool)
        m[t] = True
        m = m[..., None] & np.ones_like(d["reward"], bool)
        e = (d["ret"] - d["reward"])[m]
        lines.append(
            f"  boundary t={t:2d}: excess mean {e.mean():+.4e} mean|.| "
            f"{np.abs(e).mean():.4e} | gamma*V {(d['gamma'] * nxt_v)[m].mean():+.4e}"
            f" | gamma*lam*gae {(d['gamma'] * d['gae_lambda'] * nxt_g)[m].mean():+.4e}"
            f" | residual {d['residual'][m].mean():+.4e}")
    return "\n".join(lines)


def test_settled_boundary_value_target_is_the_reward_alone(rolled):
    """``ret == reward`` on every step where `done` fires, for a `terminal` market.

    The left side is `_gae`'s value target and the right side is the reward the
    environment returned; nothing on either side is a re-derivation of the GAE
    recursion, and the identity holds for every `gamma` and every `gae_lambda`.
    """
    d = rolled
    at = d["done"][..., None] & np.ones_like(d["reward"], bool)
    ret, reward = d["ret"][at], d["reward"][at]
    off = np.abs(ret - reward) > (ATOL + RTOL * np.abs(reward))
    assert not off.any(), (
        f"{off.sum()} of {off.size} boundary cells carry a value target that is "
        f"not the reward of that step.  `spec['termination']` is 'terminal', so "
        f"the stock left in the battery is already paid for by the terminal leg "
        f"inside `reward`; a continuation value added on top of it is a second "
        f"payment for the end of the episode.\n" + _report(d))


def test_a_truncation_market_still_bootstraps_the_same_boundary(rolled):
    """The other arm, on the same rollout: the branch branches, both ways.

    Without this, a change that masked every market would pass the check above
    while silently rewriting the four markets that declare `"truncation"` -- and
    those are the ones whose published numbers may not move.  So the same
    trajectory is run through a `_gae` built from a spec differing in one field,
    and the boundary must still carry a continuation value.

    The bit-identity of the truncation arm against the code before this branch
    existed is not asserted here: it is guaranteed by construction, because that
    arm's five lines are character-for-character what they were and are reached
    by a Python `if` rather than by a multiplication by a mask that happens to be
    one.  The measured half of that claim is the smoke comparison in the ticket.
    """
    d = rolled
    at = d["done"][..., None] & np.ones_like(d["reward"], bool)
    gap = np.abs(d["ret_as_truncation"] - d["reward"])[at]
    assert gap.max() > 1.0, (
        f"declared as a truncation, the same boundary came back with a value "
        f"target at most {gap.max():.4e} away from the reward of the step -- "
        f"i.e. indistinguishable from the terminal arm, whose residual on this "
        f"fixture is order 1e-6.  The branch is not selecting.\n" + _report(d))


def _minimal_ippo(termination):
    """`make_ippo` on market 04 with `spec["termination"]` replaced or removed.

    Nothing is rolled out: the field is read at construction, so these two calls
    return before any environment work happens.
    """
    reset, step, step_auto_reset, spec = make_p2p_env(N, PI_EXP, PI_RET, DELTA)
    spec = dict(spec)
    if termination is None:
        del spec["termination"]
    else:
        spec["termination"] = termination
    cfg = IPPOConfig(n_envs=1, horizon=1, epochs=1, minibatches=1, lr=3e-4,
                     clip_eps=0.2, gamma=0.99, gae_lambda=0.95, vf_coef=0.5,
                     ent_coef=0.0, max_grad_norm=0.5, hidden=(8,),
                     init_scale=1.0, weight_decay=0.0)
    return make_ippo((reset, step, step_auto_reset, spec), bounds_for(spec),
                     cfg, jnp.zeros(spec["obs_dim"]), jnp.ones(spec["obs_dim"]))


def test_a_market_that_declares_no_boundary_rule_is_refused():
    """The loud half: a missing field is an error, never a default.

    Defaulting is what makes this class of defect invisible -- the default any
    reader would reach for is `False`, i.e. the bootstrap arm, which is exactly
    the arm that is wrong for a market whose declaration nobody has read.  So the
    absence has to stop the wiring rather than pick a branch.
    """
    with pytest.raises(KeyError, match="termination"):
        _minimal_ippo(None)


def test_a_boundary_rule_this_module_cannot_act_on_is_refused():
    """The other loud half: an unrecognised value does not fall through.

    `spec["termination"]` has three values in this tree and the third,
    `"termination+truncation"` (a CartPole control environment outside the package),
    keeps the bootstrap arm deliberately -- `done` conflates the two kinds of
    ending there, so neither arm is right and the repair needs
    `info["terminated"]`.  A fourth value must therefore stop at construction
    instead of quietly inheriting whichever arm happens to be the fallback.
    """
    ok, _ = _minimal_ippo("termination+truncation")   # the control, still built
    assert callable(ok)
    with pytest.raises(ValueError, match="absorbing"):
        _minimal_ippo("absorbing")
