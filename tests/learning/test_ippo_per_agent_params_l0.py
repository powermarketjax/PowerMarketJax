"""L0 for `per_agent_params`: one network per agent, wired to the right agent.

**What a shape test cannot catch here.**  Giving every agent its own copy of the
architecture only moves one thing: an axis.  Every assertion about shapes passes
whether the parameter axis is lined up against the agent axis of the observation
or against some other axis of the same length, and on a market whose `obs_dim`
happens to equal `n_agents` even a `vmap` would accept the wrong pairing.  The
test that bites is therefore behavioural: knock out agent `j`'s output layer and
require that agent `j` and no other agent moves.

**The other half is the shared path.**  `per_agent_params=False` is what every
archive in the repository was produced on, so the two paths are also checked to
agree where they must: `n_agents` copies of one network must reproduce, bit for
bit, the action the shared network gives, and must reproduce its loss terms.
Where they are allowed to disagree is the optimiser, and that is stated rather
than asserted away -- `optax.clip_by_global_norm` bounds the whole pytree, so
with one network per agent it clips the agents jointly.  That is a property of
the optimiser chain and is documented in `ippo.py`; this file pins the forward
pass, which is where the axis bug would live.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("optax", reason="powermarketjax.learning needs the rl extra")

from powermarketjax.case import load_case
from powermarketjax.learning.adapters import unpack_env
from powermarketjax.learning.ippo import (IPPOConfig, make_greedy_action,
                                          make_ippo)
from powermarketjax.learning.policy import bounds_for

T = 4
K = 1
CAP_SCALE = 0.6
RAMP_SCALE = 1.0
MARKUP_MAX = 2.0


@pytest.fixture(scope="module", autouse=True)
def x64():
    prev, prev_mm = jax.config.jax_enable_x64, jax.config.jax_default_matmul_precision
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    yield
    jax.config.update("jax_enable_x64", prev)
    jax.config.update("jax_default_matmul_precision", prev_mm)


def _cfg(**over):
    base = dict(n_envs=2, horizon=2, epochs=1, minibatches=1, lr=3e-4,
                clip_eps=0.2, gamma=0.99, gae_lambda=0.95, vf_coef=0.5,
                ent_coef=0.01, max_grad_norm=0.5, hidden=(8, 8),
                init_scale=1.0, weight_decay=0.0)
    base.update(over)
    return IPPOConfig(**base)


def _day_ahead():
    from powermarketjax.envs.day_ahead import (load_commitment, load_gb_demand,
                                               make_env)
    env, spec = make_env(load_case("29gb"), load_commitment(n_periods=T),
                         load_gb_demand(), n_segments=K, kind="markup",
                         markup_max=MARKUP_MAX, cap_scale=CAP_SCALE,
                         ramp_scale=RAMP_SCALE)
    four = unpack_env((env, spec))
    return four, bounds_for(four[3]), env.make_params(episode_len=2), 0


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
    four = unpack_env((env, spec))
    return four, bounds_for(four[3]), env.make_params(episode_len=4), 0


def _ancillary():
    from powermarketjax.envs.ancillary.env import (AncillaryParams,
                                                   make_ancillary_env)
    from powermarketjax.envs.day_ahead.demand import load_gb_demand
    from tests.envs.ancillary.test_clearing_l0 import FIXTURE
    days = 2
    case = load_case("29gb")
    fx = np.load(FIXTURE, allow_pickle=True)
    idx = fx["day_index"]
    _, actual, _ = load_gb_demand()
    u = np.repeat(fx["commitment"][:days], 2, axis=2).transpose(0, 2, 1).reshape(-1, 66)
    demand = np.repeat(actual[idx[:days]], 2, axis=1).reshape(-1)
    pmin = np.asarray(case.unit_p_min, np.float64)
    pmax = np.asarray(case.unit_p_max, np.float64)
    # the schedule has to serve demand, or the market is indifferent to every
    # offer and this file passes while testing nothing (the operating-point
    # precondition `tests/envs/ancillary/test_env_l0.py` records)
    room = np.maximum(((pmax - pmin) * u).sum(1, keepdims=True), 1.0)
    frac = np.clip((demand[:, None] - (pmin * u).sum(1, keepdims=True)) / room,
                   0.0, 1.0)
    q_da = (pmin + frac * (pmax - pmin)) * u
    four = unpack_env(make_ancillary_env(
        case, (1.0 / 6.0, 0.5), 250.0, (0.050, 0.050), 50.0, n_segments=K,
        cap_scale=CAP_SCALE, ramp_scale=RAMP_SCALE, period_hours=0.5,
        kind="markup", markup_max=MARKUP_MAX))
    spec = four[3]
    prm = AncillaryParams(
        demand=jnp.asarray(demand), forecast=jnp.asarray(demand * 1.02),
        commitment=jnp.asarray(u), q_da=jnp.asarray(q_da),
        lmp_da=jnp.full((len(demand), int(case.n_nodes)), 40.0),
        learner_mask=jnp.ones(66, bool), episode_len=4)
    return four, bounds_for(spec, reserve_columns=int(spec["n_prod"])), prm, \
        int(spec["n_prod"])


MARKETS = {"day_ahead": _day_ahead, "real_time": _real_time,
           "ancillary": _ancillary}


def _built(name, per_agent, cfg=None):
    four, bounds, prm, _reserve = MARKETS[name]()
    obs, _st = four[0](jax.random.PRNGKey(0), prm)
    obs_dim = int(obs.shape[-1])
    cfg = cfg or _cfg()
    init, iterate = make_ippo(four, bounds, cfg,
                              obs_mean=jnp.zeros(obs_dim, obs.dtype),
                              obs_std=jnp.ones(obs_dim, obs.dtype),
                              per_agent_params=per_agent)
    return four, bounds, prm, cfg, obs_dim, init, iterate


@pytest.mark.parametrize("name", sorted(MARKETS))
def test_params_gain_an_agent_axis_and_keep_the_tree(name):
    """Same pytree, one leading axis wider, and the copies are not clones.

    The last clause is the non-triviality: `n_agents` identical copies would
    satisfy every shape assertion here while being the shared policy wearing a
    larger pytree, and would keep doing so through training only if the keys
    were not actually split.
    """
    four, bounds, prm, cfg, obs_dim, init_s, _ = _built(name, False)
    _f, _b, _p, _c, _o, init_p, _it = _built(name, True)
    n_agents = int(four[3]["n_agents"])

    shared, *_ = init_s(jax.random.PRNGKey(1), prm)
    per, *_ = init_p(jax.random.PRNGKey(1), prm)

    ts = jax.tree_util.tree_structure(shared)
    tp = jax.tree_util.tree_structure(per)
    assert ts == tp, f"{name}: per-agent params are a different tree: {tp} vs {ts}"

    ls = jax.tree_util.tree_leaves(shared)
    lp = jax.tree_util.tree_leaves(per)
    for a, b in zip(ls, lp):
        assert b.shape == (n_agents,) + a.shape, (
            f"{name}: leaf {a.shape} became {b.shape}, expected a leading "
            f"{n_agents}")

    # not clones: at least one leaf must differ between agent 0 and agent 1
    spread = max(float(jnp.abs(b[0] - b[1]).max()) for b in lp if b.shape[0] > 1)
    assert spread > 0.0, (
        f"{name}: every agent got the same weights, so this is the shared "
        f"policy in a wider pytree, not {n_agents} networks")


@pytest.mark.parametrize("name", sorted(MARKETS))
def test_iterate_jits_and_the_pytree_is_stable(name):
    """L0 proper: one `jit`, two chained iterations, structure unchanged.

    Two iterations rather than one: `jit` caches on the pytree structure, so a
    structure that changed inside `_update` would recompile on the second call
    or raise, and neither shows up in a single call.
    """
    four, bounds, prm, cfg, obs_dim, init, iterate = _built(name, True)
    n_agents = int(four[3]["n_agents"])
    p, tx, opt, st, eobs = init(jax.random.PRNGKey(1), prm)
    before = jax.tree_util.tree_structure(p)
    shapes = [x.shape for x in jax.tree_util.tree_leaves(p)]

    step = jax.jit(iterate, static_argnums=(1,))
    key = jax.random.PRNGKey(2)
    for _ in range(2):
        key, k = jax.random.split(key)
        p, opt, st, eobs, _k, metrics = step(p, tx, opt, st, eobs, k, prm)

    assert jax.tree_util.tree_structure(p) == before, f"{name}: tree moved"
    assert [x.shape for x in jax.tree_util.tree_leaves(p)] == shapes
    assert np.all(np.isfinite(np.asarray(metrics["reward_mean"]))), metrics
    assert np.asarray(metrics["reward_per_agent"]).shape == (n_agents,)
    assert np.asarray(metrics["step_action"]).shape[:2] == (cfg.horizon,
                                                            cfg.n_envs)


@pytest.mark.parametrize("name", sorted(MARKETS))
def test_each_agent_is_served_by_its_own_network(name):
    """The bite: knock out one agent's head, and only that agent may move.

    Under shared parameters the same edit moves every agent, so this is the
    assertion that separates "the parameter axis is lined up with the agent
    axis" from "the parameter axis is lined up with something else of the same
    length".  Nothing about the shapes distinguishes those two.
    """
    four, bounds, prm, cfg, obs_dim, init, _it = _built(name, True)
    spec = four[3]
    n_agents = int(spec["n_agents"])
    reserve = MARKETS[name]()[3]
    greedy = make_greedy_action(spec, bounds, cfg,
                                obs_mean=jnp.zeros(obs_dim),
                                obs_std=jnp.ones(obs_dim),
                                per_agent_params=True)
    p, *_ = init(jax.random.PRNGKey(1), prm)
    obs, _st = four[0](jax.random.PRNGKey(0), prm)

    base = np.asarray(greedy(p, obs), np.float64)
    j = n_agents // 2
    # Dense_2 is the action head; zeroing agent j's copy of it pins agent j's
    # pre-squash mean at its bias, which the squash then maps to a fixed
    # action.  Every other agent's head is untouched.
    hit = jax.tree_util.tree_map(lambda x: x, p)
    hit["params"]["Dense_2"]["kernel"] = \
        hit["params"]["Dense_2"]["kernel"].at[j].set(0.0)
    hit["params"]["Dense_2"]["bias"] = \
        hit["params"]["Dense_2"]["bias"].at[j].set(0.5)
    moved = np.asarray(greedy(hit, obs), np.float64)

    d = np.abs(moved - base).reshape(n_agents, -1).max(axis=1)
    assert d[j] > 0.0, (
        f"{name}: zeroing agent {j}'s head changed nothing, so the edit did "
        f"not reach the network that serves it")
    others = np.delete(d, j)
    assert others.max() == 0.0, (
        f"{name}: {int((others > 0).sum())} other agents moved when only agent "
        f"{j}'s head was edited (max {others.max():.3e}); the parameter axis is "
        f"not the agent axis")


@pytest.mark.parametrize("name", sorted(MARKETS))
def test_tiling_the_shared_network_reproduces_the_shared_forward(name):
    """`n_agents` copies of one network must be the shared network, to one ulp.

    This is the equivalence the whole option rests on: if it fails, the two arms
    are not the same function of the same weights and every comparison between a
    shared baseline and a per-agent run is comparing two things at once.

    **One ulp and not bit-for-bit, because the arithmetic is not the same
    arithmetic.**  The shared arm contracts one ``(size * n_agents, obs_dim)``
    matmul; the per-agent arm contracts `n_agents` of them under `vmap`, and the
    two reduction orders round differently.  Measured 2026-08-27 on CPU at
    float64: day-ahead and ancillary agree bit for bit, real-time differs by
    2.220e-16 on an action of magnitude 1.804, which is exactly one ulp there.
    The bound is therefore written as one ulp AT THE VALUE, which is the
    smallest bound that can hold at all, rather than as a constant somebody
    picked.
    """
    four, bounds, prm, cfg, obs_dim, init_s, _ = _built(name, False)
    spec = four[3]
    n_agents = int(spec["n_agents"])
    shared, *_ = init_s(jax.random.PRNGKey(1), prm)
    tiled = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (n_agents,) + x.shape), shared)

    obs, _st = four[0](jax.random.PRNGKey(0), prm)
    kw = dict(obs_mean=jnp.zeros(obs_dim), obs_std=jnp.ones(obs_dim))
    a = np.asarray(make_greedy_action(spec, bounds, cfg, **kw)(shared, obs),
                   np.float64)
    b = np.asarray(make_greedy_action(spec, bounds, cfg, per_agent_params=True,
                                      **kw)(tiled, obs), np.float64)
    assert a.shape == b.shape, f"{name}: {a.shape} vs {b.shape}"
    d = np.abs(a - b)
    bound = np.spacing(np.abs(a))
    assert np.all(d <= bound), (
        f"{name}: tiled per-agent action differs from the shared action by "
        f"{d.max():.3e}, which is {int(np.max(d / np.maximum(bound, 1e-300)))} "
        f"ulp at the action's own magnitude; the two arms are not the same "
        f"function of the same weights")


@pytest.mark.parametrize("name", sorted(MARKETS))
def test_tiling_reproduces_the_shared_loss_terms(name):
    """The same equivalence one level up, on the terms that have a scale.

    One epoch and one minibatch, so the reported `aux` is a single evaluation at
    the parameters handed in rather than an average over updates.

    **The six terms do not share a tolerance, and saying so is the point.**
    `vf_loss`, `entropy` and `reward_mean` are O(1) quantities built by sums
    that do not cancel, so they take a relative bound; measured 2026-08-27 the
    worst of the three is `reward_mean` on day-ahead at 1.06e-07 relative, which
    is a one-ulp difference in the sampled action amplified through the clearing
    LP.  `pg_loss`, `approx_kl` and `clip_frac` are the opposite: with the
    behaviour policy and the scored policy being the same weights, `ratio` is
    one to rounding and the advantages are standardised to mean zero, so all
    three are near-total cancellations of O(1) terms and land at 1e-09 to 1e-17.
    Their RELATIVE difference is therefore meaningless -- measured up to 1.54 on
    ancillary, on a `pg_loss` of 9.6e-17 -- and they take an absolute bound
    instead, measured worst 8.9e-10.
    """
    cfg = _cfg(epochs=1, minibatches=1)
    four, bounds, prm, _c, obs_dim, init_s, iter_s = _built(name, False, cfg)
    _f, _b, _p, _c2, _o, init_p, iter_p = _built(name, True, cfg)
    n_agents = int(four[3]["n_agents"])

    shared, tx_s, opt_s, st_s, obs_s = init_s(jax.random.PRNGKey(1), prm)
    tiled = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (n_agents,) + x.shape), shared)
    _p2, tx_p, _o2, st_p, obs_p = init_p(jax.random.PRNGKey(1), prm)

    k = jax.random.PRNGKey(3)
    m_s = iter_s(shared, tx_s, opt_s, st_s, obs_s, k, prm)[-1]
    m_p = iter_p(tiled, tx_p, tx_p.init(tiled), st_p, obs_p, k, prm)[-1]

    for term in ("vf_loss", "entropy", "reward_mean"):
        x, y = float(m_s[term]), float(m_p[term])
        assert np.isclose(x, y, rtol=1e-6, atol=0.0), (
            f"{name}: {term} is {x!r} shared and {y!r} tiled per-agent, "
            f"relative {abs(x - y) / abs(x):.3e}; a ratio of {n_agents} would "
            f"be the entropy sum going unnormalised")
    for term in ("pg_loss", "approx_kl", "clip_frac"):
        x, y = float(m_s[term]), float(m_p[term])
        assert abs(x - y) <= 1e-8, (
            f"{name}: {term} is {x!r} shared and {y!r} tiled per-agent, "
            f"absolute {abs(x - y):.3e}; these three cancel to near zero and "
            f"are bounded absolutely for the reason in the docstring")

    # The measured bite for the one term the per-agent path had to renormalise:
    # `entropy` sums over every axis of `log_std`, which grows an agent axis
    # here.  Without the division `ippo.py` performs, this same comparison would
    # be off by exactly `n_agents` -- 93.65 against 1.419 on the two markup
    # markets, measured 2026-08-27 -- so the check above is not vacuous.
    unnormalised = float(m_p["entropy"]) * n_agents
    assert not np.isclose(unnormalised, float(m_s["entropy"]), rtol=1e-6), (
        f"{name}: entropy times n_agents={n_agents} is still equal to the "
        f"shared entropy, so this market cannot tell a normalised sum from an "
        f"unnormalised one and the assertion above has no content here")


def test_a_shared_pytree_is_refused_by_the_per_agent_arm():
    """Handing shared parameters to the per-agent arm must fail, not broadcast.

    The two arms take pytrees that differ only in a leading axis, so nothing in
    the types stops a caller from crossing them.  `vmap` refuses because the
    leading axis of a kernel is `obs_dim` rather than `n_agents`; the guard is
    real but it is `vmap`'s, so it is pinned here rather than assumed, and
    `ippo.py` records that it would not fire on a market where those two
    numbers coincide.
    """
    four, bounds, prm, cfg, obs_dim, init_s, _ = _built("day_ahead", False)
    spec = four[3]
    shared, *_ = init_s(jax.random.PRNGKey(1), prm)
    assert obs_dim != int(spec["n_agents"]), (
        "this market no longer separates obs_dim from n_agents, so the "
        "mismatch this test relies on would not be visible")
    greedy = make_greedy_action(spec, bounds, cfg,
                                obs_mean=jnp.zeros(obs_dim),
                                obs_std=jnp.ones(obs_dim),
                                per_agent_params=True)
    obs, _st = four[0](jax.random.PRNGKey(0), prm)
    with pytest.raises(ValueError):
        greedy(shared, obs)
