"""L0 for the contract between a market's `spec` and the IPPO harness.

**Why this file exists at all.**  Two defects were found by writing the RL driver
and by nothing else, after three full-suite runs had passed: the real-time market
published no action bounds (`bounds_for` raised `KeyError: 'action_low'`), and
the markup action space is one-dimensional while the harness read `act_dim` off
the last axis of the bounds, so a 66-unit market produced a (66, 66) action and
the offer map rejected it with a broadcast error.  Neither is visible from
inside one market: the first lives in a `spec`, the second in the *agreement*
between a `spec` and `learning/`, and no test touched both sides.

**What the contract is.**  Axis 0 of `action_shape` is the agent axis in every
market; everything after it is one agent's action.  That is what lets IPPO share
one policy across agents, and until now it was an unstated assumption
rather than a checked one.

**Coverage, stated rather than implied.**  The three wholesale markets are
covered here.  The P2P and local-flexibility markets are *not*: their fixtures
and cases belong to other lines and building them here would make this file fail
for reasons that have nothing to do with the contract.  Both already satisfy it
(each declares a two-dimensional action), so the gap is in the guard, not in
those markets -- but it is a gap, and reading "every market" into this file
would be wrong.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("optax", reason="powermarketjax.learning needs the rl extra")

from powermarketjax.case import load_case
from powermarketjax.learning.adapters import unpack_env
from powermarketjax.learning.ippo import IPPOConfig, make_ippo
from powermarketjax.learning.policy import bounds_for

T = 4
K = 1
CAP_SCALE = 0.6        # adopted scenario (2026-08-17)
RAMP_SCALE = 1.0       # adopted scenario (2026-08-17)
MARKUP_MAX = 2.0


@pytest.fixture(scope="module", autouse=True)
def x64():
    prev, prev_mm = jax.config.jax_enable_x64, jax.config.jax_default_matmul_precision
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    yield
    jax.config.update("jax_enable_x64", prev)
    jax.config.update("jax_default_matmul_precision", prev_mm)


def _day_ahead():
    from powermarketjax.envs.day_ahead import load_commitment, load_gb_demand, make_env
    env, spec = make_env(load_case("29gb"), load_commitment(n_periods=T),
                         load_gb_demand(), n_segments=K, kind="markup",
                         markup_max=MARKUP_MAX, cap_scale=CAP_SCALE,
                         ramp_scale=RAMP_SCALE)
    return unpack_env((env, spec)), 0


def _real_time():
    from powermarketjax.envs.day_ahead import load_gb_demand
    from powermarketjax.envs.real_time import load_da_position
    from powermarketjax.envs.real_time.demand import load_gb_demand_half_hourly
    from powermarketjax.envs.real_time.env import make_env as make_rt_env
    hh, _ = load_gb_demand_half_hourly()
    fc, _a, _d = load_gb_demand()
    env, spec = make_rt_env(load_case("29gb"), load_da_position(), hh, fc,
                            n_segments=K, markup_max=MARKUP_MAX,
                            cap_scale=CAP_SCALE, ramp_scale=RAMP_SCALE)
    return unpack_env((env, spec)), 0


def _ancillary():
    from powermarketjax.envs.ancillary.env import make_ancillary_env
    built = make_ancillary_env(load_case("29gb"), (1.0 / 6.0, 0.5), 250.0,
                               (0.020, 0.050), 50.0, n_segments=K,
                               cap_scale=CAP_SCALE, ramp_scale=RAMP_SCALE,
                               kind="markup", markup_max=MARKUP_MAX)
    four = unpack_env(built)
    return four, int(four[3]["n_prod"])


MARKETS = {"day_ahead": _day_ahead, "real_time": _real_time, "ancillary": _ancillary}


@pytest.mark.parametrize("name", sorted(MARKETS))
def test_market_publishes_bounds_shaped_like_its_action(name):
    """`bounds_for` builds, and what it builds is shaped like the action.

    The real-time market failed the first half of this outright -- it published
    `action_shape` and nothing else -- so this is not a restatement of the
    types.  The second half is the part a market can get wrong quietly.
    """
    four, reserve = MARKETS[name]()
    spec = four[3]
    low, high = bounds_for(spec, reserve_columns=reserve)
    shape = tuple(int(d) for d in spec["action_shape"])
    assert low.shape == shape, f"{name}: bounds {low.shape} vs action {shape}"
    assert high.shape == shape
    assert np.all(np.asarray(high) >= np.asarray(low))


@pytest.mark.parametrize("name", sorted(MARKETS))
def test_axis_zero_of_the_action_is_the_agent_axis(name):
    """The assumption IPPO shares parameters over, asserted instead of assumed."""
    four, _ = MARKETS[name]()
    spec = four[3]
    shape = tuple(int(d) for d in spec["action_shape"])
    assert shape, f"{name}: action_shape is empty"
    assert shape[0] == int(spec["n_agents"]), (
        f"{name}: action_shape {shape} does not start with n_agents="
        f"{spec['n_agents']}")


def test_ippo_emits_an_action_the_market_accepts():
    """One real rollout through the one-dimensional market.

    This is the test the (66, 66) defect needed: the shapes alone look fine on
    both sides, and only feeding the policy's action into `step` shows that the
    harness and the offer map disagree about which axis is which.  Day-ahead is
    used because it is the market whose action is one-dimensional -- on a
    two-dimensional market this passes with the old code as well.

    `n_envs` and `horizon` are both two, not one: with one element "take the
    mean over the batch" and "take the first element" coincide, and a shape
    error that only appears when an axis has length greater than one would slip
    through.
    """
    four, _ = _day_ahead()
    _reset, _step, _sar, spec = four
    from powermarketjax.envs.day_ahead import load_commitment, load_gb_demand, make_env
    env, _s = make_env(load_case("29gb"), load_commitment(n_periods=T),
                       load_gb_demand(), n_segments=K, kind="markup",
                       markup_max=MARKUP_MAX, cap_scale=CAP_SCALE,
                       ramp_scale=RAMP_SCALE)
    params = env.make_params(episode_len=2)

    shape = tuple(int(d) for d in spec["action_shape"])
    assert len(shape) == 1, ("this test is about the one-dimensional case; "
                             f"day-ahead now declares {shape}")

    cfg = IPPOConfig(n_envs=2, horizon=2, epochs=1, minibatches=1, lr=3e-4,
                     clip_eps=0.2, gamma=0.99, gae_lambda=0.95, vf_coef=0.5,
                     ent_coef=0.0, max_grad_norm=0.5, hidden=(8,), weight_decay=0.0, init_scale=1.0)
    obs, _st = _reset(jax.random.PRNGKey(0), params)
    obs_dim = int(obs.shape[-1])
    init, iterate = make_ippo(four, bounds_for(spec), cfg,
                              obs_mean=jnp.zeros(obs_dim),
                              obs_std=jnp.ones(obs_dim))
    p_, tx, opt, st, eobs = init(jax.random.PRNGKey(1), params)
    _p, _opt, _st, _obs, _k, metrics = iterate(p_, tx, opt, st, eobs,
                                               jax.random.PRNGKey(2), params)

    # non-triviality: an all-NaN reward would let every shape assertion above
    # hold while the rollout produced nothing usable
    assert np.all(np.isfinite(np.asarray(metrics["reward_mean"]))), metrics
    assert np.asarray(metrics["reward_per_agent"]).shape == (int(spec["n_agents"]),)


def test_a_market_whose_axis_zero_is_not_the_agent_axis_is_refused():
    """The loud half of the contract.

    A market that declared the axes the other way round would previously train a
    shared policy over the wrong axis and run to completion, which is the class
    of failure this whole file exists to remove.
    """
    four, _ = _day_ahead()
    reset, step, sar, spec = four
    bad = dict(spec)
    bad["action_shape"] = (3, int(spec["n_agents"]))
    cfg = IPPOConfig(n_envs=1, horizon=1, epochs=1, minibatches=1, lr=3e-4,
                     clip_eps=0.2, gamma=0.99, gae_lambda=0.95, vf_coef=0.5,
                     ent_coef=0.0, max_grad_norm=0.5, hidden=(8,), weight_decay=0.0, init_scale=1.0)
    with pytest.raises(ValueError, match="agent axis"):
        make_ippo((reset, step, sar, bad),
                  (jnp.zeros((3, int(spec["n_agents"]))),
                   jnp.ones((3, int(spec["n_agents"])))),
                  cfg, obs_mean=jnp.zeros(3), obs_std=jnp.ones(3))


def test_extra_info_keys_refuses_a_key_the_market_does_not_return():
    """The loud half of `extra_info_keys`, with a measured bite.

    The names are supplied by the caller so that no market-specific name sits in
    the shared module (a name that also exists in another market's `info` would
    silently pick up that market's quantity -- `reserve_price` exists in both
    real-time and ancillary, which is not hypothetical).  The cost of that design
    is that a typo reaches the rollout, and an unchecked typo produces a run that
    recorded nothing it was asked to record, which looks exactly like a normal
    run.
    """
    four, _ = _day_ahead()
    _reset, _step, _sar, spec = four
    from powermarketjax.envs.day_ahead import load_commitment, load_gb_demand, make_env
    env, _s = make_env(load_case("29gb"), load_commitment(n_periods=T),
                       load_gb_demand(), n_segments=K, kind="markup",
                       markup_max=MARKUP_MAX, cap_scale=CAP_SCALE,
                       ramp_scale=RAMP_SCALE)
    params = env.make_params(episode_len=2)
    cfg = IPPOConfig(n_envs=2, horizon=2, epochs=1, minibatches=1, lr=3e-4,
                     clip_eps=0.2, gamma=0.99, gae_lambda=0.95, vf_coef=0.5,
                     ent_coef=0.0, max_grad_norm=0.5, hidden=(8,), weight_decay=0.0, init_scale=1.0)
    obs, _st = _reset(jax.random.PRNGKey(0), params)
    obs_dim = int(obs.shape[-1])

    # the control: a key this market does return goes through
    init, iterate = make_ippo(four, bounds_for(spec), cfg,
                              obs_mean=jnp.zeros(obs_dim),
                              obs_std=jnp.ones(obs_dim),
                              extra_info_keys=("shed_mwh",))
    p_, tx, opt, st, eobs = init(jax.random.PRNGKey(1), params)
    out = iterate(p_, tx, opt, st, eobs, jax.random.PRNGKey(2), params)
    assert "step_shed_mwh" in out[-1], sorted(out[-1])

    # the bite: one character different, and it must fail rather than record
    # nothing.  The message has to name the key, because the caller's next
    # question is always "then what is it called here?"
    init2, iterate2 = make_ippo(four, bounds_for(spec), cfg,
                                obs_mean=jnp.zeros(obs_dim),
                                obs_std=jnp.ones(obs_dim),
                                extra_info_keys=("shed_mwhs",))
    p2, tx2, opt2, st2, eobs2 = init2(jax.random.PRNGKey(1), params)
    with pytest.raises(KeyError, match="shed_mwhs"):
        iterate2(p2, tx2, opt2, st2, eobs2, jax.random.PRNGKey(2), params)
