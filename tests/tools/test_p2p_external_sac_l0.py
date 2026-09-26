"""L0 for `--algo sac` on the OUT-OF-PACKAGE market 04 driver.

`tools/p2p_experiment/constrained_baseline.py` is the program every published 04
learning number came from, and until now it had one learner.  SAC shared /
SAC per-agent are two of the eight columns every market owes, accepted on the
condition "test it thoroughly first, wire it in only if it works".  The second
learner is `tools/p2p_experiment/sac_arm.py`, and it is added on THAT driver
rather than only in the package: two arms differing in the algorithm AND in
which program produced them are not a contrast in the algorithm.

**The load-bearing group is the first one.**  The risk this file exists to
retire is not "SAC crashes" -- a crash announces itself.  It is that market 04's
SAC column turns out to be a different quantity from markets 01 to 03's, because
the out-of-package learner was written to look like the in-package one instead
of being it.  So the first group builds `powermarketjax.learning.sac.make_sac`
on market 04 and asserts the two parameter trees are **bit-identical**, and
pins the eleven constants against `hyperparams.SAC_SHARED` field for field.  A
transcription that drifted in a width, a bound or an initialiser fails there and
nowhere else: every other property in this file is satisfied by any competent
SAC.

The remaining groups, and what each would catch:

* the tree says which learner ran.  `constrained_baseline.parameter_layout`
  reads the algorithm off the returned tree and never off `--algo`, for the
  reason it reads the layout off the tree: a flag accepted, forwarded
  and dropped produces a run whose log and whose product disagree, and nothing
  downstream can tell;
* the two layouts are the same relation SAC's own per-agent path produces --
  every leaf gains one leading axis AND the temperature stops being a scalar,
  because `sac.py` gives each independent learner "its own entropy target to
  meet".  A tree with a per-agent actor and a shared temperature is in neither
  layout and is refused rather than filed with one of them;
* the two ALGORITHMS do not compute the same action, which shapes alone cannot
  see -- the two trees do not even have the same leaves, so only a forward pass
  can fail here;
* the IPPO arm is where it was.  Its numbers are not this file's to check
  against history (no test can reach a previous commit); what is checked is that
  the new arguments at their defaults are the path that was always taken, and
  that the SAC-only flags are refused rather than silently ignored on it.

**Measured judgement power** (2026-09-09, CPU, `jax` 0.10.2, `--agents 3/4`).
Against a copy of the two modules in which `run` accepts `algo="sac"`, forwards
it and then builds the PPO policy anyway -- the flag wired in and dropped -- the
tests below go red.
"""
import dataclasses
import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# The out-of-package drivers reach `distrax`/`rlax`/`optax` through the `rl`
# extra, which CI's `dev` install does not carry; a bare import would turn an
# absent optional dependency into a failure rather than a skip.
pytest.importorskip("optax", reason="the out-of-package driver needs the rl extra")
pytest.importorskip("distrax", reason="preliminary_reference builds its policy on distrax")
pytest.importorskip("rlax", reason="the losses come from rlax")

import powermarketjax                                             # noqa: E402

#: The repository, located from the INSTALLED package rather than from this
#: file's own path.  The judgement-power measurement runs this file from a copy
#: outside `tests/`, where a `parents[2]` would resolve to a directory with no
#: `tools/` in it and the two imports below would fail for a reason that has
#: nothing to do with the injection under test -- which is what happened on the
#: first attempt (a script's own location is not the repository).
REPO = pathlib.Path(powermarketjax.__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "benchmark"))
sys.path.insert(0, str(REPO / "tools" / "p2p_experiment"))
import constrained_baseline as CB                                # noqa: E402
import preliminary_reference as R                                # noqa: E402
import sac_arm                                                   # noqa: E402
from hyperparams import SAC_PROVENANCE, SAC_SHARED               # noqa: E402

from powermarketjax.learning.policy import bounds_for, to_action  # noqa: E402
from powermarketjax.learning.sac import (SACConfig, _nets,        # noqa: E402
                                         make_sac)


def raw_env(n_agents):
    """The market's own tuple, unwrapped: what `make_sac` is written against."""
    return R.make_p2p_env(n_agents, R.PI_EXP, R.PI_RET, R.DELTA)


def tiny_cfg(n_envs=1, horizon=8, buffer_size=256, utd_ratio=0.25,
             reward_scale=1.0):
    """A config small enough to run in a test and shaped like a real one.

    Only the four fields that set the SIZE of the run move; the eleven CleanRL
    constants are `config_for`'s, which is what
    `test_the_constants_are_SAC_SHARED_field_for_field` pins.
    """
    return sac_arm.config_for(n_envs, reward_scale, buffer_size, utd_ratio,
                              episode_len=horizon)


def package_cfg(n_envs=1, horizon=8, buffer_size=256, reward_scale=1.0):
    """`hyperparams.SAC_SHARED` with only its four `ours` fields moved.

    **This is what the package side of every parity test below is built from,
    and it is NOT `sac_arm.config_for`.**  Handing `make_sac` the
    out-of-package config would compare a drifted constant against itself: the
    first version of this file did exactly that, and an injected
    `LOG_STD_MIN = -20` went undetected by the parity test because both sides
    read the same drifted value.  Markets 01 to 03 read `SAC_SHARED`, so that
    is the object market 04 has to be checked against.
    """
    return dataclasses.replace(SAC_SHARED, n_envs=n_envs, horizon=horizon,
                               buffer_size=buffer_size,
                               reward_scale=reward_scale)


def leaf_shapes(tree):
    return sac_arm.leaf_shapes(tree)


def synthetic_obs(n_agents, seed=0):
    """One observation per participant, `(n_agents, OBS_DIM)`."""
    return jax.random.normal(jax.random.PRNGKey(seed),
                             (n_agents, R.OBS_DIM), jnp.float32)


# ------------------------------------ the learner is the package's, not a copy

@pytest.mark.parametrize("per_agent", [False, True])
def test_the_parameter_tree_is_the_packages_own_tree(per_agent):
    """`sac_arm.init_params` and `sac.make_sac`'s `init` agree BIT FOR BIT.

    Both are given the key `make_sac` would hand its own `_init_nets`, which is
    the first half of one split, so the comparison is of the initialisers and
    not of two key streams.  The observation the two are shaped on differs --
    `make_sac` resets the market, this file passes zeros -- and that is the
    point: flax's initialisers read the SHAPE of their input and the key, so a
    difference here would mean a width, a bound or an initialiser drifted in
    transcription, which is the one failure a smoke run cannot show.

    The market 04 spec goes into `make_sac` unwrapped, so the bounds, the action
    layout and both networks come from the package's own `_nets`.  The two
    configs are built from opposite ends -- ours from `sac_arm`'s constants,
    the package's from `SAC_SHARED` -- and asserted equal first, so a drifted
    constant fails here as a config mismatch instead of drifting on both sides
    at once.
    """
    n_agents = 4
    env = raw_env(n_agents)
    env_params, _train, _eval = CB.build(n_agents, 0.5)
    ours_cfg = sac_arm.config_for(1, 1.0, 256, SAC_SHARED.utd_ratio,
                                  episode_len=96)
    pkg_cfg = package_cfg(n_envs=1, horizon=96, buffer_size=256)
    assert ours_cfg == pkg_cfg, (
        f"the out-of-package config and `SAC_SHARED` disagree:\n  ours {ours_cfg}"
        f"\n  SAC_SHARED {pkg_cfg}")

    key = jax.random.PRNGKey(7)
    init, _iterate = make_sac(env, bounds_for(env[3]), pkg_cfg,
                              jnp.zeros((R.OBS_DIM,), jnp.float32),
                              jnp.ones((R.OBS_DIM,), jnp.float32),
                              per_agent_params=per_agent,
                              convergence_key=None)
    package_params, _tx, _learner, _st, _obs = init(key, env_params)

    key_net, _key_reset = jax.random.split(key)
    ours = sac_arm.init_params(key_net, env, ours_cfg, per_agent)

    assert leaf_shapes(ours) == leaf_shapes(package_params), (
        "the out-of-package tree and `make_sac`'s tree are not the same shape")
    for path, leaf in jax.tree_util.tree_flatten_with_path(ours)[0]:
        want = package_params
        for step in path:
            want = want[step.key if hasattr(step, "key") else step.idx]
        np.testing.assert_array_equal(
            np.asarray(leaf), np.asarray(want),
            err_msg=f"{jax.tree_util.keystr(path)} differs from `make_sac`'s")


def test_the_constants_are_SAC_SHARED_field_for_field():
    """The eleven CleanRL values, against `hyperparams.SAC_SHARED` itself.

    Imported and compared rather than re-read from CleanRL: `sac_arm` repeats
    the values because `tools/benchmark/hyperparams.py` is not on this driver's
    path and carries markets 01 to 03's `n_envs` and `horizon`, and a repeated
    constant with no test is a second declaration that can drift.  The four
    fields NOT compared are the four `SAC_PROVENANCE` files under "ours", and
    the assertion below names them so the exclusion is a decision and not a
    gap.
    """
    from_source = set(SAC_PROVENANCE["from_source"])
    ours = set(SAC_PROVENANCE["ours"])
    fields = {f.name for f in SACConfig.__dataclass_fields__.values()}
    assert from_source | ours == fields, (
        "SAC_PROVENANCE no longer partitions SACConfig; this test's exclusion "
        f"list is stale: {sorted(fields - (from_source | ours))}")
    assert ours == {"n_envs", "horizon", "buffer_size", "reward_scale"}

    mine = sac_arm.config_for(batch=8, reward_scale=1.0)
    for field in sorted(from_source):
        assert getattr(mine, field) == getattr(SAC_SHARED, field), (
            f"`sac_arm.{field.upper()}` is {getattr(mine, field)!r} where "
            f"SAC_SHARED carries {getattr(SAC_SHARED, field)!r}; market 04's "
            f"SAC column would not be markets 01 to 03's SAC column")
    assert len(from_source) == 11


@pytest.mark.parametrize("per_agent", [False, True])
def test_the_actor_output_is_the_packages_actor_output(per_agent):
    """Both heads, bit for bit, against an actor built from `SAC_SHARED`.

    **This is the test that sees the constants the parameter tree cannot.**
    `log_std_min` and `log_std_max` are read inside `SACActor.__call__`, not by
    any initialiser, so a drift in either leaves every leaf of the tree
    bit-identical and changes only the exploration width the learner is allowed
    to ask for.  Measured on the first version of this file, which had no such
    test: an injected `LOG_STD_MIN = -20` passed all 22 tests but one, and that
    one was red for an unrelated import.

    `log_std` is therefore compared and not just `mean` -- the greedy action
    reads `mean` alone, so a test written on the action alone would be blind
    here too.
    """
    n_agents = 4
    env = raw_env(n_agents)
    spec = env[3]
    ours_cfg = tiny_cfg()
    pkg_cfg = package_cfg()
    _n, _d, _s, _low, _high, pkg_actor, _q = _nets(spec, bounds_for(spec),
                                                   pkg_cfg)
    policy = sac_arm.init_params(jax.random.PRNGKey(2), env, ours_cfg,
                                 per_agent)
    obs = synthetic_obs(n_agents, seed=5) * 3.0

    mean, log_std, value = sac_arm.greedy_forward(env, ours_cfg, per_agent)(
        policy, obs)
    if per_agent:
        want_mean, want_log_std = jax.vmap(pkg_actor.apply)(policy["actor"],
                                                            obs)
    else:
        want_mean, want_log_std = pkg_actor.apply(policy["actor"], obs)

    np.testing.assert_array_equal(np.asarray(mean), np.asarray(want_mean))
    np.testing.assert_array_equal(np.asarray(log_std),
                                  np.asarray(want_log_std))
    assert float(np.asarray(log_std).min()) >= pkg_cfg.log_std_min
    assert float(np.asarray(log_std).max()) <= pkg_cfg.log_std_max
    np.testing.assert_array_equal(np.asarray(value),
                                  np.zeros((n_agents,), np.float32))


def test_the_action_box_and_the_entropy_target_come_from_the_market():
    """`bounds_for` on market 04's own spec, and CleanRL's target from it.

    The box is `[-1, 1]` on both coordinates because `envs/p2p/env.py` publishes
    it, not because this file chose it, and `target_entropy` is minus the action
    dimension that box implies.  A learner that declared its own box would be a
    second declaration of the action space, which is what `bounds_for`'s
    docstring refuses.
    """
    n_agents = 4
    env = raw_env(n_agents)
    cfg = tiny_cfg()
    n, act_dim, act_shape, low, high, _actor, _q = sac_arm.env_layout(env, cfg)
    assert (n, act_dim, act_shape) == (n_agents, 2, (n_agents, 2))
    np.testing.assert_array_equal(np.asarray(low), -np.ones((n_agents, 2)))
    np.testing.assert_array_equal(np.asarray(high), np.ones((n_agents, 2)))
    assert -float(act_dim) == -2.0


def test_the_greedy_action_agrees_with_to_action_to_half_an_ulp():
    """What the market sees at evaluation, against `make_sac_greedy_action`'s.

    The driver's `learned_mean` arm submits `tanh(mean)`; the package's greedy
    action is `to_action(mean, -1, 1)`, i.e. `-1 + (tanh(mean) + 1)`.  Equal in
    exact arithmetic, NOT equal in float32 -- the round trip through `+1` and
    `-1` drops the low bits of a small `tanh`.  The bound is asserted and the
    equality is not, because the equality is false: measured 2026-09-09 on CPU,
    64.75 % of 32 768 draws are bit-identical and none of the 119 draws with
    `|mean| < 0.01` are.

    `tanh` is what runs, and deliberately: it is the line the PPO arm's
    evaluation already used, so the two 04 arms are scored by the same
    expression and differ in the learner alone.
    """
    n_agents = 4
    low, high = bounds_for(raw_env(n_agents)[3])
    mean = jax.random.normal(jax.random.PRNGKey(0),
                             (4096, n_agents, 2), jnp.float32) * 2.0
    driver = np.asarray(jnp.tanh(mean))
    package = np.asarray(to_action(mean, low, high))
    worst = float(np.abs(driver - package).max())
    assert worst < 1.2e-07, (
        f"the driver's evaluation action and `make_sac_greedy_action`'s differ "
        f"by {worst:.3e}, more than one float32 ulp at magnitude one; that is "
        f"no longer a rounding difference")
    assert np.mean(driver == package) < 1.0, (
        "the two are now bit-identical everywhere, so the docstring in "
        "`sac_arm.greedy_forward` recording that they are not is stale")


# ------------------------------------------- the tree says which learner ran

def test_the_two_learners_do_not_share_a_single_top_level_entry():
    """`sorted(tree)` is the discriminator, so the two sets must be disjoint.

    `constrained_baseline.parameter_layout` dispatches on it.  If the PPO tree
    ever gained a leaf named `actor`, a `--algo ippo` run could be read back as
    SAC, and the mis-read would be invisible: both branches return a dict with
    the same keys.
    """
    ppo = set(R.init_policy(jax.random.PRNGKey(0)))
    sac = set(sac_arm.TOP_LEVEL)
    assert sac == {"actor", "log_alpha", "q1", "q1_target", "q2", "q2_target"}
    assert ppo & sac == set(), f"the two trees share {sorted(ppo & sac)}"
    assert sorted(sac_arm.TOP_LEVEL) == list(sac_arm.TOP_LEVEL), (
        "TOP_LEVEL is compared against `sorted(policy)` and is not sorted")


@pytest.mark.parametrize("per_agent", [False, True])
def test_parameter_layout_reads_the_algorithm_off_the_tree(per_agent):
    """The stamp is a property of the tree, and no flag is consulted.

    `parameter_layout` takes the tree and the participant count and nothing
    else, so there is no flag available to it to be right by accident.
    """
    n_agents = 3
    env = raw_env(n_agents)
    tree = sac_arm.init_params(jax.random.PRNGKey(0), env, tiny_cfg(), per_agent)
    got = CB.parameter_layout(tree, n_agents)
    assert got["algo"] == "sac"
    assert got["per_agent_params"] is per_agent
    assert got["top_level"] == sorted(sac_arm.TOP_LEVEL)
    assert got["read_from"].startswith("the policy")

    ppo_tree = (R.init_policy_per_agent(jax.random.PRNGKey(0), n_agents)
                if per_agent else R.init_policy(jax.random.PRNGKey(0)))
    assert CB.parameter_layout(ppo_tree, n_agents)["algo"] == "ippo"


def test_parameter_layout_refuses_a_tree_in_neither_sac_layout():
    """A per-agent actor with a shared temperature is not a third value.

    `sac.py`: the temperature is per agent on that path "because each
    independent learner has its own entropy target to meet".  A tree that moved
    the actor and not the temperature has one of the two and would be filed
    with the per-agent runs by any check that read the actor alone, so the
    readback compares every leaf and raises.
    """
    n_agents = 3
    env = raw_env(n_agents)
    tree = dict(sac_arm.init_params(jax.random.PRNGKey(0), env, tiny_cfg(),
                                    True))
    assert tree["log_alpha"].shape == (n_agents,)
    tree["log_alpha"] = jnp.asarray(jnp.log(0.2), jnp.float32)
    with pytest.raises(SystemExit):
        CB.parameter_layout(tree, n_agents)


# --------------------------------------------------- the two parameter layouts

def test_the_per_agent_layout_multiplies_every_leaf_and_the_temperature():
    """One leading axis on every leaf, the same relation `ippo` produces.

    The scalar count is asserted as an EXACT multiple.  Both networks are dense
    with zero-initialised biases, which do not depend on the mapped key, so a
    `vmap` that broadcast their output axis away would leave a tree that still
    looks per-agent at a glance and whose ratio would fall just short of
    `n_agents`; at these widths that near miss is 3.9782 against 4.0000, which
    an inequality would not separate.
    """
    n_agents = 4
    env = raw_env(n_agents)
    cfg = tiny_cfg()
    shared = sac_arm.init_params(jax.random.PRNGKey(0), env, cfg, False)
    per_agent = sac_arm.init_params(jax.random.PRNGKey(0), env, cfg, True)

    s, p = leaf_shapes(shared), leaf_shapes(per_agent)
    assert sorted(s) == sorted(p)
    for path in s:
        assert p[path] == (n_agents,) + s[path], path

    lay_s = CB.parameter_layout(shared, n_agents)
    lay_p = CB.parameter_layout(per_agent, n_agents)
    assert lay_s["leaves"] == lay_p["leaves"] == 33
    assert lay_s["log_alpha_shape"] == [] and lay_p["log_alpha_shape"] == [n_agents]
    assert lay_p["scalars"] == n_agents * lay_s["scalars"]
    assert lay_s["scalars"] == 353_545, (
        "the shared tree changed size; at hidden (256, 256), obs_dim 15 and "
        "act_dim 2 it is 70 916 for the actor, 70 657 for each of four critics "
        "and one for log_alpha")


def test_each_participant_gets_its_own_actor_and_not_its_neighbours():
    """Row `j` of the output comes from leaf slice `j`, checked one at a time.

    The reference is built by the OTHER path: the shared `actor.apply` on one
    sliced-out network and one observation row, with no `vmap` anywhere.  A
    per-agent forward that used participant 0's network for everybody produces
    arrays of exactly the right shape and fails only here.
    """
    n_agents = 4
    env = raw_env(n_agents)
    cfg = tiny_cfg()
    _n, _d, _shape, _low, _high, actor, _q = sac_arm.env_layout(env, cfg)
    policy = sac_arm.init_params(jax.random.PRNGKey(1), env, cfg, True)
    obs = synthetic_obs(n_agents)
    mean, log_std, _v = sac_arm.greedy_forward(env, cfg, True)(policy, obs)

    for j in range(n_agents):
        one = jax.tree_util.tree_map(lambda leaf: leaf[j], policy["actor"])
        want_mean, want_log_std = actor.apply(one, obs[j])
        np.testing.assert_allclose(np.asarray(mean[j]), np.asarray(want_mean),
                                   rtol=0, atol=1e-6)
        np.testing.assert_allclose(np.asarray(log_std[j]),
                                   np.asarray(want_log_std), rtol=0, atol=1e-6)
        for k in range(n_agents):
            if k == j:
                continue
            other = jax.tree_util.tree_map(lambda leaf: leaf[k], policy["actor"])
            wrong, _ = actor.apply(other, obs[j])
            assert not np.allclose(np.asarray(mean[j]), np.asarray(wrong),
                                   rtol=0, atol=1e-6), (j, k)


# ----------------------------------------- the two algorithms are not one arm

def test_the_two_algorithms_give_different_actions_on_the_same_key():
    """Same init key, same observation, and not one component agrees.

    `tanh(mean)` is what the `learned_mean` arm submits, so this is the quantity
    the market would see and not an intermediate.  An `--algo` that was accepted
    and then dropped lands here with a difference of exactly zero.
    """
    n_agents = 4
    env = raw_env(n_agents)
    key = jax.random.PRNGKey(0)
    obs = synthetic_obs(n_agents)
    ppo = np.asarray(jnp.tanh(R.forward(R.init_policy(key), obs)[0]))
    sac = np.asarray(jnp.tanh(sac_arm.greedy_forward(env, tiny_cfg(), False)(
        sac_arm.init_params(key, env, tiny_cfg(), False), obs)[0]))
    assert ppo.shape == sac.shape == (n_agents, R.ACTION_DIM)
    assert not np.any(ppo == sac), (
        "some action component is bit-identical across the two learners, which "
        "is what a flag that never reached the learner looks like")
    assert np.abs(ppo - sac).max() > 1e-4, (
        "the two learners differ by less than 1e-4 in every component; PPO's "
        "mean head is scaled by 0.01 and SAC's is flax's default, so a "
        "difference this small would mean the assertion above reads float noise")


# ------------------------------------------------------- the learner's conventions

def test_one_iteration_moves_each_participants_temperature_on_its_own():
    """The per-agent temperature ends up with `n_agents` DIFFERENT values.

    `sac.py`'s `_alpha_loss` takes the batch mean per agent and sums the agents,
    "so the gradient on `log_alpha[j]` sees agent `j`'s samples only".  A
    transcription that reduced over the agent axis first would leave one number
    broadcast into `n_agents` slots -- the right shape, the wrong quantity --
    and only an inequality between the entries can see it.
    """
    n_agents = 3
    env_params, env, _eval = CB.build(n_agents, 0.5)
    cfg = tiny_cfg(n_envs=1, horizon=8, buffer_size=64, utd_ratio=0.5)
    init, iterate = sac_arm.make_learner(env, env_params, cfg, False,
                                         CB.COST_LIMIT, CB.LAMBDA_MAX, True)
    policy, learner = init(jax.random.PRNGKey(0))
    start = np.asarray(policy["log_alpha"])
    assert start.shape == (n_agents,)
    assert np.all(start == start[0]), "the temperatures did not start equal"

    policy, learner, _lam, _aux = iterate(policy, learner, jnp.float32(0.0),
                                          jnp.float32(CB.LAMBDA_STEP),
                                          jax.random.PRNGKey(1))
    end = np.asarray(policy["log_alpha"])
    assert not np.any(end == start), "no participant's temperature moved"
    assert len(set(end.tolist())) == n_agents, (
        f"the {n_agents} temperatures are not distinct after one iteration "
        f"({end.tolist()}); one entropy target was met on behalf of all of them")

    shared_init, shared_iter = sac_arm.make_learner(
        env, env_params, cfg, False, CB.COST_LIMIT, CB.LAMBDA_MAX, False)
    shared_policy, shared_learner = shared_init(jax.random.PRNGKey(0))
    assert shared_policy["log_alpha"].shape == ()


def test_the_targets_start_equal_to_the_critics_and_then_lag_them():
    """Polyak, from the two ends that do not need `tau` restated here.

    `make_sac` starts the targets AT the critics (`q1_target=nets["q1"]`), so
    equality at init is exact.  After the updates the target has to have moved
    and to have moved LESS than the critic; a transcription that forgot the
    Polyak line leaves the first assertion true and the second false, and one
    that copied the critic into the target each step leaves them equal forever.
    """
    n_agents = 3
    env_params, env, _eval = CB.build(n_agents, 0.5)
    cfg = tiny_cfg(n_envs=1, horizon=8, buffer_size=64, utd_ratio=0.5)
    init, iterate = sac_arm.make_learner(env, env_params, cfg, False,
                                         CB.COST_LIMIT, CB.LAMBDA_MAX, False)
    policy, learner = init(jax.random.PRNGKey(0))
    flat = lambda t: np.concatenate(
        [np.asarray(x).ravel() for x in jax.tree_util.tree_leaves(t)])
    q0, t0 = flat(policy["q1"]), flat(policy["q1_target"])
    np.testing.assert_array_equal(q0, t0)

    policy, learner, _lam, _aux = iterate(policy, learner, jnp.float32(0.0),
                                          jnp.float32(CB.LAMBDA_STEP),
                                          jax.random.PRNGKey(1))
    q1, t1 = flat(policy["q1"]), flat(policy["q1_target"])
    moved_critic = float(np.abs(q1 - q0).max())
    moved_target = float(np.abs(t1 - t0).max())
    assert moved_critic > 0.0, "the critic did not move at all"
    assert moved_target > 0.0, "the target never moved; the Polyak line is gone"
    assert moved_target < moved_critic, (
        f"the target moved {moved_target:.3e} against the critic's "
        f"{moved_critic:.3e}; a target that keeps up with its critic is not a "
        f"target network")


def test_the_buffer_is_fifo_and_the_boundary_fires_once_per_episode():
    """One iteration writes `n_envs * horizon` rows, and `done` lands on the last.

    Market 04 is `spec["termination"] == "terminal"`, so `_cont` masks the
    boundary instead of bootstrapping it.  That is only the right arithmetic if
    `done` marks the last period of each episode and nothing else, which is
    what the index list below asserts -- a rollout that ran past a boundary
    would put a `done` in the middle and would be continuing from the
    unrestricted auto-reset draw.
    """
    n_agents = 3
    env_params, env, _eval = CB.build(n_agents, 0.5)
    cfg = tiny_cfg(n_envs=2, horizon=R.EPISODE_LEN, buffer_size=1024,
                   utd_ratio=1.0 / 192)
    init, iterate = sac_arm.make_learner(env, env_params, cfg, False,
                                         CB.COST_LIMIT, CB.LAMBDA_MAX, False)
    policy, learner = init(jax.random.PRNGKey(0))
    assert int(learner["buffer"]["filled"]) == 0
    assert int(learner["buffer"]["cursor"]) == 0

    policy, learner, _lam, aux = iterate(policy, learner, jnp.float32(0.0),
                                         jnp.float32(CB.LAMBDA_STEP),
                                         jax.random.PRNGKey(1))
    buf = learner["buffer"]
    per_iter = cfg.n_envs * cfg.horizon
    assert int(buf["filled"]) == per_iter == 192
    assert int(buf["cursor"]) == per_iter
    done = np.flatnonzero(np.asarray(buf["done"]))
    assert done.tolist() == [R.EPISODE_LEN - 1, 2 * R.EPISODE_LEN - 1], (
        f"`done` fired at {done.tolist()}, not once at the end of each of the "
        f"{cfg.n_envs} episodes")
    assert int(aux["buffer_filled"]) == per_iter


def test_a_reward_scale_that_was_never_fitted_is_refused():
    """`SAC_SHARED` ships `nan` as a sentinel, and a run must not train on it.

    `make_sac` refuses it for market 01's reason -- the scale is fitted per run
    -- and the out-of-package learner refuses it in the same place and with the
    same words, so a market 04 run cannot be the one that slips through.
    """
    n_agents = 3
    env_params, env, _eval = CB.build(n_agents, 0.5)
    for bad in (float("nan"), 0.0, -1.0):
        with pytest.raises(ValueError, match="reward_scale"):
            sac_arm.make_learner(env, env_params, tiny_cfg(reward_scale=bad),
                                 False, CB.COST_LIMIT, CB.LAMBDA_MAX, False)


def test_the_fitted_reward_scale_is_the_truthful_spread_and_is_frozen():
    """A positive finite float, fitted from the truthful arm, floored like `sac`'s.

    Not a value assertion: the number depends on the panel and the participant
    count and pinning it here would pin the scenario.  What is pinned is that it
    is a Python float (so `SACConfig` freezes it and it cannot move with the
    parameters), that it is the truthful rollout's spread and not a constant,
    and that the same key gives the same number.
    """
    n_agents = 3
    truth_params, truth_env, _e = CB.build(n_agents, 0.5,
                                           np.zeros(n_agents, bool))
    key = jax.random.PRNGKey(11)
    one = sac_arm.reward_scale_for(truth_env, truth_params, key, 2)
    two = sac_arm.reward_scale_for(truth_env, truth_params, key, 2)
    assert isinstance(one, float) and one == two
    assert np.isfinite(one) and one > 0.0
    assert one != 1.0, (
        "the fitted scale is exactly the floor `sac.reward_statistics` applies "
        "when the truthful reward is constant, which on this market it is not")


# ------------------------------------------------------- the IPPO arm is where it was

def test_the_ippo_arm_is_unchanged_by_the_new_arguments():
    """The defaults ARE the path that was always taken, and it never enters `sac_arm`.

    Two claims, and the second is the one nothing else would catch.  A `run`
    that fell through to `sac_arm` for `algo="ippo"` would still return a tree
    and a curve; here `sac_arm.run` is replaced by a raise, so the IPPO arm
    completing at all is the assertion.

    What this test canNOT do is compare against the previous commit, which is
    where the real gate on this arm lives; that comparison was run by hand on
    2026-09-09.
    """
    n_agents = 3
    plain = CB.run(n_agents, 0.5, False, 0, 1, 2, 1, False)

    def refuse(*a, **k):
        raise AssertionError("the IPPO arm called into `sac_arm.run`")

    saved = sac_arm.run
    try:
        sac_arm.run = refuse
        explicit = CB.run(n_agents, 0.5, False, 0, 1, 2, 1, False, "ippo",
                          sac_arm.BUFFER_SIZE, sac_arm.UTD_RATIO)
    finally:
        sac_arm.run = saved

    assert plain[4] is None and explicit[4] is None, (
        "the IPPO path handed back a SACConfig")
    assert plain[0] == explicit[0], "the curve moved"
    for k in plain[1]:
        np.testing.assert_array_equal(
            np.asarray(plain[1][k]), np.asarray(explicit[1][k]),
            err_msg=f"parameter leaf {k} moved")
    assert CB.parameter_layout(plain[1], n_agents)["algo"] == "ippo"


def test_run_hands_back_the_learner_it_was_asked_for():
    """`constrained_baseline.run` under both algorithms, read off what came back.

    The guard in `main` is not the only place this can be caught, and it should
    not be: a driver that dispatched on `--algo` correctly and a `main` that
    checked the result are two properties, and removing either one leaves a run
    that can lie about its learner.  This asserts the first without going
    through `main`, so the two are red separately.

    One training iteration on one episode: not a score, the smallest thing that
    puts `--algo` through `run`, the reward-scale fit, the replay write and the
    gradient scan.
    """
    n_agents = 3
    ippo = CB.run(n_agents, 0.5, False, 0, 1, 2, 1, False, "ippo")
    sac = CB.run(n_agents, 0.5, False, 0, 1, 1, 1, False, "sac", 1024,
                 sac_arm.UTD_RATIO)

    assert sorted(ippo[1]) == sorted(R.init_policy(jax.random.PRNGKey(0)))
    assert sorted(sac[1]) == sorted(sac_arm.TOP_LEVEL), (
        f"--algo sac came back with {sorted(sac[1])}; the flag did not reach "
        f"the learner")
    assert CB.parameter_layout(ippo[1], n_agents)["algo"] == "ippo"
    assert CB.parameter_layout(sac[1], n_agents)["algo"] == "sac"

    assert ippo[4] is None, "the IPPO path handed back a SACConfig"
    assert isinstance(sac[4], SACConfig), (
        "the SAC path handed back no config, so the fitted reward_scale has "
        "nowhere to be read from")
    assert np.isfinite(sac[4].reward_scale) and sac[4].reward_scale > 0.0
    assert sac[4].buffer_size == 1024 and sac[4].n_envs == 1


def test_the_sac_only_flags_are_refused_on_the_ippo_path():
    """Accepted and dropped is the failure; refused is the behaviour.

    A `--buffer-size` that the IPPO arm swallowed would produce a log saying one
    thing and a run doing another, which is the same defect `--algo` is stamped
    from the tree to avoid.
    """
    with pytest.raises(SystemExit, match="SACConfig"):
        CB.run(3, 0.5, False, 0, 0, 2, 1, False, "ippo", 1024,
               sac_arm.UTD_RATIO)
    with pytest.raises(SystemExit, match="SACConfig"):
        CB.run(3, 0.5, False, 0, 0, 2, 1, False, "ippo", sac_arm.BUFFER_SIZE,
               0.5)
    with pytest.raises(SystemExit, match="neither"):
        CB.run(3, 0.5, False, 0, 0, 2, 1, False, "ppo")


# --------------------------------------------------------------- end to end

@pytest.mark.parametrize("per_agent", [False, True])
def test_the_cli_flag_runs_end_to_end_and_stamps_the_product(
        tmp_path, monkeypatch, per_agent):
    """`--algo sac` through `main`, and the stamps land in the JSON.

    One iteration on one episode: not a score, the smallest thing that puts the
    flag through `run`, the reward-scale fit, the replay write, the gradient
    scan, the readback guard in `main` and the record on disk.  The buffer is
    cut to 1 024 env-steps so the test does not allocate 4.5 MB per participant.

    What is asserted on the record is what a reader needs to tell this run from
    a PPO one WITHOUT trusting the flag: `algo`, the six top-level entries, the
    temperature's shape, and the fitted `reward_scale` that no constant supplies.
    """
    out = tmp_path / "curves.json"
    argv = ["constrained_baseline", "--agents", "3", "--seeds", "1",
            "--iterations", "1", "--batch", "1", "--eval-every", "1",
            "--initial-soc", "0.5", "--algo", "sac", "--buffer-size", "1024",
            "--out", str(out)]
    if per_agent:
        argv.append("--per-agent-params")
    monkeypatch.setattr(sys, "argv", argv)
    CB.main()

    records = json.loads(out.read_text())
    assert records, "main wrote no record"
    for record in records:
        assert record["algo"] == "sac"
        layout = record["layout"]
        assert layout["algo"] == "sac"
        assert layout["per_agent_params"] is per_agent
        assert layout["top_level"] == sorted(sac_arm.TOP_LEVEL)
        assert layout["leaves"] == 33
        assert layout["log_alpha_shape"] == ([3] if per_agent else [])
        assert layout["scalars"] == (3 if per_agent else 1) * 353_545
        assert layout["read_from"].startswith("the policy")

        hp = record["hyperparams"]
        assert hp["buffer_size"] == 1024 and hp["n_envs"] == 1
        assert hp["horizon"] == R.EPISODE_LEN
        assert hp["hidden"] == [256, 256]
        assert np.isfinite(hp["reward_scale"]) and hp["reward_scale"] > 0.0
        assert record["hyperparams_provenance"]["tuned_per_market"] is False
        assert record["hyperparams_provenance"]["from_source"], (
            "the product does not say which constants are CleanRL's")

        curve = record["curve"]
        assert len(curve) == 1
        for field in ("q_loss", "q_mean", "target_mean", "actor_loss",
                      "alpha_loss", "alpha", "entropy", "buffer_filled"):
            assert field in curve[0], f"the SAC curve does not report {field}"
        assert curve[0]["buffer_filled"] == R.EPISODE_LEN
        assert np.isfinite(curve[0]["eval_return"])


def test_the_ippo_product_still_says_ippo_and_carries_no_hyperparams():
    """The other side of the stamp, so `algo` is not a field only SAC writes.

    A stamp present on one arm and absent on the other is read as "old record"
    rather than as "PPO run", which is the reading that files the two arms
    together.
    """
    n_agents = 3
    plain = CB.run(n_agents, 0.5, False, 0, 0, 2, 1, False)
    assert plain[4] is None
    assert CB.parameter_layout(plain[1], n_agents)["algo"] == "ippo"
