"""L0 for `learning/sac.py`: jit, pytree stability, the replay FIFO, the two
parameter layouts, and the boundary rule -- on the three wholesale markets.

Mirrors `test_ippo_per_agent_params_l0.py` where SAC makes the same decision:
the per-agent bite is the same knock-out test (edit agent `j`'s actor head and
require that agent `j` alone moves), and the shared/tiled equivalence is the
same one-ulp bound on the greedy action.  What is SAC's own is tested on its
own terms: the buffer's FIFO cursor wraps and `filled` saturates at capacity;
the soft-Bellman target masks the continuation at a settled boundary and keeps
it at a truncated one; the temperature is one scalar shared or one per agent.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("optax", reason="powermarketjax.learning needs the rl extra")

from powermarketjax.learning.sac import (SACConfig, _nets, _q_per_agent,
                                         make_sac, make_sac_greedy_action,
                                         reward_statistics)
from tests.learning.test_ippo_per_agent_params_l0 import MARKETS


@pytest.fixture(scope="module", autouse=True)
def x64():
    prev, prev_mm = jax.config.jax_enable_x64, jax.config.jax_default_matmul_precision
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    yield
    jax.config.update("jax_enable_x64", prev)
    jax.config.update("jax_default_matmul_precision", prev_mm)


def _cfg(**over):
    base = dict(n_envs=2, horizon=2, buffer_size=8, batch_size=4, utd_ratio=1.0,
                gamma=0.99, tau=0.005, policy_lr=3e-4, q_lr=1e-3, alpha_lr=1e-3,
                init_alpha=0.2, hidden=(8, 8), log_std_min=-5.0,
                log_std_max=2.0, reward_scale=1.0)
    base.update(over)
    return SACConfig(**base)


def _built(name, per_agent, cfg=None):
    four, bounds, prm, _reserve = MARKETS[name]()
    obs, _st = four[0](jax.random.PRNGKey(0), prm)
    obs_dim = int(obs.shape[-1])
    cfg = cfg or _cfg()
    init, iterate = make_sac(four, bounds, cfg,
                             obs_mean=jnp.zeros(obs_dim, obs.dtype),
                             obs_std=jnp.ones(obs_dim, obs.dtype),
                             per_agent_params=per_agent)
    return four, bounds, prm, cfg, obs_dim, init, iterate


@pytest.mark.parametrize("per_agent", [False, True])
@pytest.mark.parametrize("name", sorted(MARKETS))
def test_iterate_jits_and_the_pytree_is_stable(name, per_agent):
    """One `jit`, three chained iterations, structure unchanged, buffer wraps.

    Three iterations of `n_envs * horizon = 4` env-steps into a buffer of 8:
    after two the buffer is full, and the third wraps the cursor.  `filled`
    must saturate at 8 rather than keep counting, or `_sample` would draw from
    rows that were never written.
    """
    four, bounds, prm, cfg, obs_dim, init, iterate = _built(name, per_agent)
    n_agents = int(four[3]["n_agents"])
    p, tx, learner, st, eobs = init(jax.random.PRNGKey(1), prm)
    before = jax.tree_util.tree_structure(p)
    shapes = [x.shape for x in jax.tree_util.tree_leaves(p)]
    lstruct = jax.tree_util.tree_structure(learner)

    step = jax.jit(iterate, static_argnums=(1,))
    key = jax.random.PRNGKey(2)
    filled = []
    for _ in range(3):
        key, k = jax.random.split(key)
        p, learner, st, eobs, _k, metrics = step(p, tx, learner, st, eobs, k, prm)
        filled.append(int(learner["buffer"]["filled"]))

    assert jax.tree_util.tree_structure(p) == before, f"{name}: params tree moved"
    assert jax.tree_util.tree_structure(learner) == lstruct, f"{name}: learner tree moved"
    assert [x.shape for x in jax.tree_util.tree_leaves(p)] == shapes
    assert filled == [4, 8, 8], filled
    assert int(learner["buffer"]["cursor"]) == 4, "cursor did not wrap modulo capacity"
    for term in ("q_loss", "actor_loss", "alpha_loss", "entropy", "alpha",
                 "reward_mean"):
        assert np.isfinite(float(metrics[term])), (term, metrics[term])
    assert np.asarray(metrics["reward_per_agent"]).shape == (n_agents,)
    assert np.asarray(metrics["step_action"]).shape[:2] == (cfg.horizon, cfg.n_envs)
    alpha_shape = np.shape(p["log_alpha"])
    assert alpha_shape == ((n_agents,) if per_agent else ()), alpha_shape


@pytest.mark.parametrize("name", sorted(MARKETS))
def test_each_agent_is_served_by_its_own_actor(name):
    """Knock out agent `j`'s actor mean head; only agent `j` may move."""
    four, bounds, prm, cfg, obs_dim, init, _it = _built(name, True)
    spec = four[3]
    n_agents = int(spec["n_agents"])
    greedy = make_sac_greedy_action(spec, bounds, cfg, obs_mean=jnp.zeros(obs_dim),
                                    obs_std=jnp.ones(obs_dim),
                                    per_agent_params=True)
    p, *_ = init(jax.random.PRNGKey(1), prm)
    obs, _st = four[0](jax.random.PRNGKey(0), prm)
    base = np.asarray(greedy(p, obs), np.float64)
    j = n_agents // 2
    # Dense_2 is the mean head (Dense_3 the log_std head); pinning its bias
    # moves agent j's pre-squash mean and nothing else
    hit = jax.tree_util.tree_map(lambda x: x, p)
    hit["actor"]["params"]["Dense_2"]["kernel"] = \
        hit["actor"]["params"]["Dense_2"]["kernel"].at[j].set(0.0)
    hit["actor"]["params"]["Dense_2"]["bias"] = \
        hit["actor"]["params"]["Dense_2"]["bias"].at[j].set(0.5)
    moved = np.asarray(greedy(hit, obs), np.float64)
    d = np.abs(moved - base).reshape(n_agents, -1).max(axis=1)
    assert d[j] > 0.0, f"{name}: editing agent {j}'s head changed nothing"
    others = np.delete(d, j)
    assert others.max() == 0.0, (
        f"{name}: {int((others > 0).sum())} other agents moved (max "
        f"{others.max():.3e}); the parameter axis is not the agent axis")


@pytest.mark.parametrize("name", sorted(MARKETS))
def test_tiling_the_shared_actor_reproduces_the_shared_action(name):
    """`n_agents` copies of one actor must give the shared action, to one ulp."""
    four, bounds, prm, cfg, obs_dim, init_s, _ = _built(name, False)
    spec = four[3]
    n_agents = int(spec["n_agents"])
    shared, *_ = init_s(jax.random.PRNGKey(1), prm)
    tiled = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (n_agents,) + jnp.shape(x)), shared)
    obs, _st = four[0](jax.random.PRNGKey(0), prm)
    kw = dict(obs_mean=jnp.zeros(obs_dim), obs_std=jnp.ones(obs_dim))
    a = np.asarray(make_sac_greedy_action(spec, bounds, cfg, **kw)(shared, obs),
                   np.float64)
    b = np.asarray(make_sac_greedy_action(spec, bounds, cfg, per_agent_params=True,
                                          **kw)(tiled, obs), np.float64)
    assert a.shape == b.shape
    d = np.abs(a - b)
    # Tolerance is a ULP count, not a pinned number: CPU gives 0 to 1
    # ulp; on GPU (2026-09-07, real_time, full test run) the same pair
    # differed by 4.441e-16 at |a| ~ 1.3, i.e. 2 ulp, from a different
    # reduction order in the tiled matmul.  4 ulp keeps the bite (a wrong actor
    # is off by ~1e-1, fourteen orders above) while covering both platforms.
    assert np.all(d <= 4 * np.spacing(np.abs(a))), (
        f"{name}: tiled action differs from shared by {d.max():.3e}")


def test_settled_boundary_is_masked_and_truncated_is_not():
    """The boundary rule, read off the metrics of a fabricated market.

    A one-agent, one-coordinate market whose `step` returns a constant reward
    and `done = True` at every step, with `q_target` pinned far from zero via
    the critic's output bias.  Under `terminal` the target is `r / scale`
    exactly; under `truncation` it carries `gamma * (q_next - alpha * logp)`.
    The two must differ, and the `terminal` one must equal the reward.  The
    market is fabricated because no real market lets the continuation be read
    in isolation from the clearing.
    """
    from powermarketjax.learning.ippo import _BOOTSTRAP_AT_DONE

    class St:
        def __init__(self, cursor):
            self.cursor = cursor
    jax.tree_util.register_pytree_node(
        St, lambda s: ((s.cursor,), None), lambda _, c: St(c[0]))

    obs_dim, n_agents = 3, 1
    reward_value = 2.0

    def reset(key, prm):
        return jnp.ones((n_agents, obs_dim)), St(jnp.asarray(0))

    def step(key, state, action, prm):
        obs = jnp.ones((n_agents, obs_dim))
        return (obs, St(state.cursor + 1), jnp.full((n_agents,), reward_value),
                jnp.zeros(()), jnp.asarray(True), {"converged": jnp.asarray(True)})

    targets = {}
    for rule in ("terminal", "truncation"):
        assert rule in _BOOTSTRAP_AT_DONE
        spec = dict(n_agents=n_agents, action_shape=(n_agents, 1),
                    action_low=0.0, action_high=1.0, termination=rule,
                    baseline_action=jnp.zeros((n_agents, 1)))
        bounds = (jnp.zeros((n_agents, 1)), jnp.ones((n_agents, 1)))
        cfg = _cfg(n_envs=1, horizon=1, buffer_size=1, batch_size=1,
                   reward_scale=4.0, gamma=0.5)
        init, iterate = make_sac((reset, step, step, spec), bounds, cfg,
                                 obs_mean=jnp.zeros(obs_dim),
                                 obs_std=jnp.ones(obs_dim))
        p, tx, learner, st, eobs = init(jax.random.PRNGKey(1), None)
        # pin both target critics at a large constant so the continuation is
        # unmistakable when it is present
        for q in ("q1_target", "q2_target"):
            p[q]["params"]["Dense_2"]["kernel"] = jnp.zeros_like(
                p[q]["params"]["Dense_2"]["kernel"])
            p[q]["params"]["Dense_2"]["bias"] = jnp.full_like(
                p[q]["params"]["Dense_2"]["bias"], 100.0)
        *_, m = jax.jit(iterate, static_argnums=(1,))(p, tx, learner, st, eobs,
                                                     jax.random.PRNGKey(2), None)
        targets[rule] = float(m["target_mean"])
    assert np.isclose(targets["terminal"], reward_value / 4.0, atol=1e-12), targets
    # 0.5 * (100 - alpha * logp): far from 0.5 whatever logp is at this scale
    assert targets["truncation"] > 10.0, targets


@pytest.mark.parametrize("name", sorted(MARKETS))
def test_reward_statistics_is_a_positive_scalar(name):
    four, _bounds, prm, _r = MARKETS[name]()
    s = reward_statistics(four, prm, jax.random.PRNGKey(0), n_envs=2, horizon=2)
    assert np.shape(s) == ()
    assert float(s) > 0.0


def test_reserve_box_comes_from_the_market_and_bounds_only_those_columns():
    """`bounds_for` takes the reserve box off the spec, and only that market's.

    The box is the ancillary market's own (`envs/ancillary/action.py`), so this
    asserts three things a caller-supplied box could not give: the ends are the
    market's two constants rather than round numbers, the energy columns are
    untouched, and a spec that declares reserve columns without a box raises
    instead of falling back to +-inf.  Falling back is the failure this guards:
    an archive trained against an invented box looks exactly like one trained
    against the market's.

    `to_action` then maps a large pre-squash draw inside the box, and a run of
    `make_sac` on the ancillary fixture keeps its critic finite over an
    iteration whose updates would otherwise start the divergence measured
    on 2026-09-03 (the full divergence needs 3 072 updates on the real
    fixture; this asserts the bounded path is finite, not that the unbounded
    one diverges here).
    """
    from powermarketjax.learning.policy import bounds_for, to_action
    from powermarketjax.envs.ancillary.action import RESERVE_ACTION_LOW
    four, _bounds, prm, reserve = MARKETS["ancillary"]()
    spec = four[3]
    lo, hi = bounds_for(spec, reserve_columns=reserve)
    # the ends are derived from this market's constants, not restated here
    want_hi = float(np.log(np.expm1(float(spec["volr"])
                                    / float(spec["pi_scale"]))))
    assert np.all(np.asarray(lo)[..., -reserve:] == RESERVE_ACTION_LOW)
    assert np.all(np.asarray(hi)[..., -reserve:] == want_hi)
    # an offer at the top of the box reaches volr and does not pass it
    top = float(jax.nn.softplus(jnp.asarray(want_hi, jnp.float64))
                * float(spec["pi_scale"]))
    assert top <= float(spec["volr"])
    assert float(spec["volr"]) - top < 1e-9, top
    # the energy columns keep the energy map's own bounds
    assert np.all(np.asarray(lo)[..., :-reserve] == float(spec["action_low"]))
    assert np.all(np.asarray(hi)[..., :-reserve] == float(spec["action_high"]))
    a = to_action(jnp.full(lo.shape, 40.0), lo, hi)
    assert float(np.max(np.asarray(a)[..., -reserve:])) <= want_hi
    # a market that publishes reserve columns without their box is an error,
    # not a market with unbounded ones
    stripped = {k: v for k, v in spec.items()
                if k not in ("reserve_low", "reserve_high")}
    with pytest.raises(KeyError):
        bounds_for(stripped, reserve_columns=reserve)
    with pytest.raises(ValueError):
        bounds_for({**spec, "reserve_low": 5.0, "reserve_high": -20.0},
                   reserve_columns=reserve)

    obs, _st = four[0](jax.random.PRNGKey(0), prm)
    obs_dim = int(obs.shape[-1])
    # the critic's scale as the driver fits it; at `reward_scale=1.0` this
    # fixture's rewards are of order 1e4 and `q_loss` would be large for a
    # reason that has nothing to do with the box
    scale = float(reward_statistics(four, prm, jax.random.PRNGKey(3), 2, 2))
    cfg = _cfg(n_envs=2, horizon=2, buffer_size=8, batch_size=4, utd_ratio=8.0,
               reward_scale=scale)
    init, iterate = make_sac(four, (lo, hi), cfg, obs_mean=jnp.zeros(obs_dim, obs.dtype),
                             obs_std=jnp.ones(obs_dim, obs.dtype))
    p, tx, learner, st, eobs = init(jax.random.PRNGKey(1), prm)
    step = jax.jit(iterate, static_argnums=(1,))
    p, learner, st, eobs, _k, m = step(p, tx, learner, st, eobs, jax.random.PRNGKey(2), prm)
    # finite, not small: on this two-day fixture with 8-unit layers and a
    # buffer of 8 the critic's first 32 updates are a transient (measured
    # 5.6e6 at a 2 x 2 reward scale), and a magnitude bound here would be a
    # number picked to pass.  The divergence-versus-box comparison was made
    # on the real fixture; what this asserts is that the
    # bounded path runs and every sampled reserve action stays inside the box.
    assert np.isfinite(float(m["q_loss"])), float(m["q_loss"])
    acts = np.asarray(m["step_action"])[..., -reserve:]
    assert acts.max() <= want_hi and acts.min() >= RESERVE_ACTION_LOW


# ---------------------------------------------------------------------------
# The four invariants found unguarded on review (2026-09-06).  Each one is stated
# with the reading it takes on a deliberately broken copy of `sac.py`, measured
# before the test was committed; the broken copies live nowhere, the
# numbers are what they produced.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("per_agent", [False, True])
@pytest.mark.parametrize("name", sorted(MARKETS))
def test_each_agent_q_reads_only_its_own_row(name, per_agent):
    """Perturb agent `j`'s observation or action; only agent `j`'s Q may move.

    The learner's design starts from "independent learners, no centralised
    critic", and `test_each_agent_is_served_by_its_own_actor` holds only the
    actor to it.  **A centralised critic passes every other L0 in this file**:
    the shapes are unchanged, the pytree is unchanged, the buffer still wraps,
    the boundary rule still holds.  Measured on two broken copies, both on
    `ancillary`: a `SoftQ` that appends the agent-mean of its own input moves
    65 of the 65 other agents' Q by up to 1.413e-03 on the shared path, and a
    `_q_per_agent` fed the agent-mean of `z` and `a` moves 65 of 65 by up to
    1.069e-02 on the per-agent path.  The module as committed moves exactly
    one column, by exactly zero on the others.
    """
    four, bounds, prm, cfg, obs_dim, init, _it = _built(name, per_agent)
    n_agents, act_dim, _shape, _low, _high, _actor, qnet = _nets(four[3], bounds,
                                                                 cfg)
    p, *_ = init(jax.random.PRNGKey(1), prm)
    apply = partial(_q_per_agent, qnet) if per_agent else qnet.apply
    k = jax.random.PRNGKey(3)
    z = jax.random.normal(k, (3, n_agents, obs_dim))
    a = 0.1 * jax.random.normal(jax.random.fold_in(k, 1), (3, n_agents, act_dim))
    base = np.asarray(apply(p["q1"], z, a), np.float64)
    assert base.shape == (3, n_agents), base.shape
    j = n_agents // 2
    for what, zz, aa in (("action", z, a.at[:, j, :].add(0.7)),
                         ("observation", z.at[:, j, :].add(0.7), a)):
        d = np.abs(np.asarray(apply(p["q1"], zz, aa), np.float64) - base).max(axis=0)
        assert d[j] > 0.0, f"{name}/{what}: agent {j}'s own row moved nothing"
        others = np.delete(d, j)
        assert others.max() == 0.0, (
            f"{name}/{what}: {int((others > 0).sum())} of {others.size} other "
            f"agents' Q moved (max {others.max():.3e}); this critic is not "
            f"independent")


@pytest.mark.parametrize("per_agent", [False, True])
@pytest.mark.parametrize("name", sorted(MARKETS))
def test_target_entropy_is_one_agents_action_dimension(name, per_agent):
    """Recover the target entropy from the metrics and pin it to `act_dim`.

    `alpha_lr = 0` freezes the temperature, so `alpha` in the metrics is
    `init_alpha` itself; `utd_ratio` is set so the iteration takes exactly one
    gradient step, so the metrics are that step's own values and not a mean
    over steps.  The shared path then reports ``alpha_loss = alpha * (entropy
    + d)`` and the per-agent path ``alpha_loss = n_agents * alpha * (entropy +
    d)``, which gives `d` in closed form.

    `d` must be **one agent's** action dimension.  A learner that took the
    joint action of every agent would target `n_agents * d` and nothing else
    here would notice: measured on a copy with ``target_entropy =
    -float(n_agents * act_dim)``, this recovers 198.0 where `ancillary` asks
    for 3.0 (66 agents times a 3-column action).
    """
    four, bounds, prm, cfg, obs_dim, init, iterate = _built(
        name, per_agent, _cfg(alpha_lr=0.0, utd_ratio=0.25))
    n_agents, act_dim = _nets(four[3], bounds, cfg)[:2]
    p, tx, learner, st, eobs = init(jax.random.PRNGKey(1), prm)
    *_, m = jax.jit(iterate, static_argnums=(1,))(p, tx, learner, st, eobs,
                                                  jax.random.PRNGKey(2), prm)
    alpha, entropy = float(m["alpha"]), float(m["entropy"])
    assert alpha == pytest.approx(cfg.init_alpha, rel=1e-12), (
        f"{name}: alpha_lr=0 must leave the temperature at init_alpha")
    d = float(m["alpha_loss"]) / (alpha * (n_agents if per_agent else 1)) - entropy
    assert d == pytest.approx(float(act_dim), rel=1e-9), (
        f"{name}/per_agent={per_agent}: the temperature is chasing a target "
        f"entropy of -{d:.6g}, not -{act_dim} (one agent's action dimension)")


@pytest.mark.parametrize("name", sorted(MARKETS))
def test_each_agents_temperature_answers_to_its_own_entropy(name):
    """On the per-agent path the temperatures must separate after one step.

    They start equal (`init_alpha` for every agent), and `_alpha_loss` takes
    the batch mean **per agent** before summing, so agent `j`'s gradient sees
    agent `j`'s samples alone and the values separate on the first update.
    Measured on a copy that pools the mean over the agent axis as well, all 66
    temperatures stay bit-identical at -1.60843791 (unique count 1), which is
    the whole of what an independent learner's own entropy target buys.

    **The assertion is "more than one", not "all distinct", and that is
    measured rather than cautious**: as committed, `ancillary` separates all 66
    (spread 2.0e-03) but `day_ahead` gives 63 distinct values with a spread of
    3.2e-12, because after one Adam step the size of the move is governed by
    `eps` and three agents land on the same float.
    """
    four, bounds, prm, cfg, obs_dim, init, iterate = _built(
        name, True, _cfg(utd_ratio=0.25))
    p, tx, learner, st, eobs = init(jax.random.PRNGKey(1), prm)
    before = np.asarray(p["log_alpha"], np.float64)
    assert np.unique(before).size == 1, "the agents did not start equal"
    p2, *_ = jax.jit(iterate, static_argnums=(1,))(p, tx, learner, st, eobs,
                                                   jax.random.PRNGKey(2), prm)
    after = np.asarray(p2["log_alpha"], np.float64)
    assert np.unique(after).size > 1, (
        f"{name}: all {after.size} temperatures are still bit-identical after "
        f"an update, so every agent's alpha is answering to the same pooled "
        f"entropy")


def test_the_replay_sample_never_draws_an_unwritten_row():
    """Sample far more rows than were written; the target must ignore the rest.

    A fabricated market with a constant reward and a settled boundary, so the
    soft-Bellman target of a written row is `reward / reward_scale` exactly and
    nothing else in the arithmetic can produce that number.  One iteration
    writes one env-step into a buffer of eight and then draws 64 rows, so seven
    eighths of the capacity is still the zero fill `init` allocated.

    `_sample` draws on `[0, filled)`, and `_push` writes from the cursor
    contiguously modulo capacity, so `filled` is exactly the number of written
    rows and the zero fill is unreachable.  Measured on a copy drawing on
    `[0, buffer_size)` instead, `target_mean` comes back 0.154449 against the
    2.0 this asserts: the unwritten rows carry `reward = 0` and `done = False`,
    so they are not even masked out.
    """
    from powermarketjax.learning.ippo import _BOOTSTRAP_AT_DONE
    assert _BOOTSTRAP_AT_DONE["terminal"] is False

    class Cst:
        def __init__(self, cursor):
            self.cursor = cursor
    jax.tree_util.register_pytree_node(
        Cst, lambda s: ((s.cursor,), None), lambda _, c: Cst(c[0]))

    obs_dim, n_agents, reward_value = 3, 1, 2.0

    def reset(key, prm):
        return jnp.ones((n_agents, obs_dim)), Cst(jnp.asarray(0))

    def step(key, state, action, prm):
        return (jnp.ones((n_agents, obs_dim)), Cst(state.cursor + 1),
                jnp.full((n_agents,), reward_value), jnp.zeros(()),
                jnp.asarray(True), {"converged": jnp.asarray(True)})

    spec = dict(n_agents=n_agents, action_shape=(n_agents, 1), action_low=0.0,
                action_high=1.0, termination="terminal",
                baseline_action=jnp.zeros((n_agents, 1)))
    bounds = (jnp.zeros((n_agents, 1)), jnp.ones((n_agents, 1)))
    cfg = _cfg(n_envs=1, horizon=1, buffer_size=8, batch_size=64,
               utd_ratio=1.0, reward_scale=1.0)
    init, iterate = make_sac((reset, step, step, spec), bounds, cfg,
                             obs_mean=jnp.zeros(obs_dim),
                             obs_std=jnp.ones(obs_dim))
    p, tx, learner, st, eobs = init(jax.random.PRNGKey(1), None)
    _p, learner, _st, _o, _k, m = jax.jit(iterate, static_argnums=(1,))(
        p, tx, learner, st, eobs, jax.random.PRNGKey(2), None)
    assert int(learner["buffer"]["filled"]) == 1, "one env-step should be in"
    assert float(m["target_mean"]) == pytest.approx(reward_value, rel=1e-12), (
        f"target_mean {float(m['target_mean']):.6g} against {reward_value}: "
        f"the sample reached rows that were never written")


def test_every_market_declares_a_termination_the_mask_table_knows():
    """Pin each market's declared boundary rule and the table's two answers.

    `make_sac` rejects a value the table does not know, but it cannot reject a
    market that declares the **wrong** known value: swapping `truncation` for
    `terminal` on a wholesale market silently cuts every step's continuation
    value, leaving the target at `reward / reward_scale`, with no error, no
    shape change and no other L0 in this file firing.  Measured by handing this
    body a copy of the `ancillary` spec with `termination = "terminal"`: it
    fails on that market and on nothing else.

    Markets 04 and 05 are not covered here because this file's fixtures are the
    three wholesale markets; their declarations belong with their own env
    tests.
    """
    from powermarketjax.learning.ippo import _BOOTSTRAP_AT_DONE
    assert _BOOTSTRAP_AT_DONE["truncation"] is True
    assert _BOOTSTRAP_AT_DONE["terminal"] is False
    declared = {"day_ahead": "truncation", "real_time": "truncation",
                "ancillary": "truncation"}
    for name in sorted(MARKETS):
        spec = MARKETS[name]()[0][3]
        got = spec["termination"]
        assert got in _BOOTSTRAP_AT_DONE, (
            f"{name} declares termination={got!r}, which the mask table does "
            f"not know")
        assert got == declared[name], (
            f"{name} declares termination={got!r}, not {declared[name]!r}; if "
            f"the market really changed, every SAC and IPPO reading taken "
            f"under the old rule is on a different boundary arithmetic")
