# Written for this repository on 2026-09-09 -- no upstream counterpart.
"""L0 for `--algo sac` in `tools/flex_experiment/concentration_baseline.py`.

The sibling file `test_flex_per_agent_params_l0.py` covers the parameter layout
on the IPPO path and its docstring says why a shape test cannot carry that
column.  This file covers the second learner, and the same rule applies twice
over: the two learners hold trees with different KEYS, so crossing them raises,
but within one learner the two layouts differ only in a leading axis and cross
silently.  Every check that could pass on a crossed axis is therefore
behavioural.

**Three things here are not tests of the wiring but records of what the wiring
found**, and they are tests so that a later change cannot quietly move them:

* `test_this_market_squashes_into_its_own_box` -- 05 was the last of the five
  markets declaring an unbounded action box, so `policy.to_action` and
  `sac._q_action` were the identity here and the critic read an unbounded
  coordinate.  `policy.bounds_for` records that this is what diverged SAC on the
  ancillary market, and the SAC section of the driver records the measurement
  showing 05 diverged the same way.  `envs/local_flexibility/action.py`
  published the box on 2026-09-10; what this test guards is
  unchanged -- that the critic's coordinate is the market's declaration and not
  the learner's -- and what it records is now that the box reaches both maps.
* `test_a_saturated_policy_reads_as_the_incdec_arm` -- because §9.3's action map
  is flat past 128, any diverged policy with `incdec`'s sign pattern returns
  `incdec`'s numbers **bit for bit**.  The end-to-end smoke did exactly that, and
  without this test a reader would take it for the learner discovering that
  strategy.
* `test_annealed_log_std_kills_the_entropy_gradient` -- the measurement behind
  the decision not to feed 05's annealed `log_std` to SAC.  The gradient is not
  small, it is bit-zero, and that is what the test asserts.

**The scenario is the real one**, cell 0 of the two-by-two, read from the SwissDN
parquet, so `n_agent`, `obs_dim` and the observations are the driver's own.
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "flex_experiment"))

pytest.importorskip("optax", reason="concentration_baseline's update needs optax")
pytest.importorskip("flax", reason="the package's SAC networks are flax modules")

# Bracketed for the reason `test_flex_per_agent_params_l0.py` gives: importing
# the driver turns `jax_enable_x64` and `jax_default_matmul_precision` on at
# module scope, which is right for the driver and wrong for the session.
_PREV_X64 = jax.config.jax_enable_x64
_PREV_MATMUL = jax.config.jax_default_matmul_precision

import concentration_baseline as cb  # noqa: E402
from powermarketjax.envs.local_flexibility.action import (  # noqa: E402
    ACTION_SATURATION)

jax.config.update("jax_enable_x64", _PREV_X64)
jax.config.update("jax_default_matmul_precision", _PREV_MATMUL)

from powermarketjax.learning.policy import log_prob as squashed_log_prob  # noqa: E402
from powermarketjax.learning.policy import to_action  # noqa: E402
from powermarketjax.learning.sac import _q_action  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def x64():
    """The driver's own numerics, for this module only.  Every tolerance below
    was measured under x64 with the highest matmul precision."""
    prev = jax.config.jax_enable_x64
    prev_mm = jax.config.jax_default_matmul_precision
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    yield
    jax.config.update("jax_enable_x64", prev)
    jax.config.update("jax_default_matmul_precision", prev_mm)


#: One episode is enough for every check here that runs a rollout, and keeping it
#: at one is what keeps this file an L0: the driver's own iteration collects
#: `BATCH` = 64.
N_EPISODES = 2

#: The reward scale a real run fits per cell from the truthful arm.  Fixed here
#: instead of fitted, because fitting it costs a 64-episode rollout (measured
#: 41 s on CPU) and **no check in this file reads it**: it divides the critic's
#: regression target and nothing below regresses.  The value is the one measured
#: on cell `2040p_2040c`, 2026-09-09, so the configuration these tests build is
#: the configuration a run of that cell builds.
REWARD_SCALE = 0.219206


@pytest.fixture(scope="module")
def cell():
    """`(env, params, n, scale, obs)`, cell 0, built once.

    `obs` is one period's observation through the driver's own scaling and
    clipping, so its trailing axis is the `obs_dim` the network reads.
    """
    env, params, n = cb.scenario(*cb.CELLS[0], split="train")
    scale = cb.obs_scale(env, params, n)
    _, state = env[0](jax.random.PRNGKey(0), params)
    obs = jnp.clip(env[3]["get_obs"](state, params) / scale,
                   -cb.OBS_CLIP, cb.OBS_CLIP)
    return env, params, n, scale, obs


@pytest.fixture(scope="module")
def cfg():
    """`SAC_SHARED` at 05's batch, which is what a run builds."""
    conf, _prov = cb.sac_config(REWARD_SCALE)
    return conf


@pytest.fixture(scope="module")
def policies(cell, cfg):
    """`(shared, per_agent, n)`: two SAC layouts from the same seed."""
    env, _params, n, _scale, _obs = cell
    obs_dim = env[3]["obs_dim"]
    key = jax.random.PRNGKey(0)
    return (cb.init_sac_policy(key, obs_dim, n, cfg),
            cb.init_sac_policy(key, obs_dim, n, cfg, per_agent=True), n)


# ------------------------------------------------------ the flag is off by ---
def test_the_existing_path_is_the_default(cell):
    """Every keyword this column added defaults to the behaviour that was there.

    The IPPO arm's bit-identity is a property of the whole driver and is checked
    by running it, not here; what is checked here
    is the one thing a signature can carry -- that no caller written before
    `--algo` has to be edited to keep its behaviour.
    """
    import inspect
    assert inspect.signature(cb.evaluate).parameters["algo"].default == "ippo"
    assert inspect.signature(cb.evaluate).parameters["sac_cfg"].default is None
    assert inspect.signature(cb.evaluate).parameters["greedy"].default is False
    for fn in (cb.make_sac_rollout, cb.make_sac_update):
        p = inspect.signature(fn).parameters
        assert p["per_agent"].default is False, fn.__name__
        # `None` is "this market's box", i.e. `bounds_for(spec)`.  A learner
        # that chose its own would be declaring the action space a second time,
        # which `policy.bounds_for` refuses; the keyword exists for the
        # diagnostic in the SAC section and not for runs.
        assert p["bounds"].default is None, fn.__name__


def test_the_two_learners_hold_different_trees(cell, policies):
    """An IPPO tree in the SAC path raises rather than producing numbers.

    This is the one crossing the driver gets for free: the trees do not merely
    differ in an axis, they differ in their keys, so the wrong `--algo` fails on
    the first lookup.  Crossing the two LAYOUTS within SAC is the silent one and
    is covered by the knockout test below.
    """
    env, _params, n, _scale, obs = cell
    shared, _pa, _n = policies
    ippo = cb.init_policy(jax.random.PRNGKey(0), env[3]["obs_dim"])
    assert set(shared) == {"actor", "q1", "q2", "q1_target", "q2_target",
                           "log_alpha"}
    assert set(ippo) & set(shared) == set()
    with pytest.raises(KeyError):
        cb.sac_apply(cb.sac_config(REWARD_SCALE)[0])[0](ippo["actor"], obs)


# ------------------------------------------------------------ the layouts ---
def test_the_flag_gives_every_leaf_the_agent_axis(policies):
    """Shapes follow the participant count and the temperature multiplies by it."""
    shared, pa, n = policies
    s_leaves = jax.tree_util.tree_flatten_with_path(shared)[0]
    p_leaves = jax.tree_util.tree_flatten_with_path(pa)[0]
    assert [k for k, _ in s_leaves] == [k for k, _ in p_leaves]
    for (k, a), (_, b) in zip(s_leaves, p_leaves):
        assert b.shape == (n,) + a.shape, jax.tree_util.keystr(k)
    # `sac.py`: the temperature is "one scalar on the shared path and one per
    # agent on the per-agent path, because each independent learner has its own
    # entropy target to meet".
    assert shared["log_alpha"].shape == ()
    assert pa["log_alpha"].shape == (n,)
    n_s = sum(int(np.prod(v.shape)) for _k, v in s_leaves)
    n_p = sum(int(np.prod(v.shape)) for _k, v in p_leaves)
    # Measured 2026-09-09, CPU, cell `2040p_2040c`: 365 323 scalars against
    # 8 767 752, i.e. exactly n = 24 times, over 33 leaves in both layouts.
    assert (len(s_leaves), n_p) == (33, n * n_s)
    # Recorded rather than repaired: flax's `param_dtype` defaults to float32
    # while this driver runs under x64, so the networks are float32 and
    # `log_alpha` -- taken as `jnp.log(init_alpha)` under x64 -- is float64.
    # The package's SAC has the same mixture on every market.
    assert str(shared["log_alpha"].dtype) == "float64"
    assert {str(v.dtype) for k, v in s_leaves
            if "log_alpha" not in jax.tree_util.keystr(k)} == {"float32"}


#: How far `n` copies of one actor may drift from the shared forward.  Derived
#: from the contraction, not pinned to what came out: the hidden layer contracts
#: 256 terms, the parameters are float32, and `log_std` leaves `tanh` through an
#: affine map of slope ``(log_std_max - log_std_min) / 2 = 3.5``.  So the bound
#: is 256 float32 ULP of the quantity's own magnitude for `mean`, and 256 ULP of
#: a magnitude-one `tanh` output times 3.5 for `log_std`.  Measured that day on
#: cell `2040p_2040c`, seed 0: `mean` 2.46e-07 against a bound of 1.4e-5 and a
#: signal of 0.474, `log_std` 7.15e-07 against a bound of 1.07e-4 and a signal
#: of 2.11.
_EPS32 = float(np.spacing(np.float32(1.0)))
MEAN_ABS = 256 * _EPS32 * 0.5
LOG_STD_ABS = 256 * _EPS32 * 3.5


def test_replicating_one_actor_reproduces_the_shared_forward(cell, policies, cfg):
    """`n` copies of one actor must give the shared answer, on both heads.

    The axis check in the direction the knockout cannot see: it fails if the
    parameter axis is mapped against anything but the agent axis, because then
    the copies would not line up with the observations that produced the shared
    answer.  The gap that would open is the one the next test measures -- the two
    layouts sit 1.03 apart in `mean`, six orders above the bounds here.
    """
    _env, _params, n, _scale, obs = cell
    shared, _pa, _n = policies
    act_s, _ = cb.sac_apply(cfg)
    act_p, _ = cb.sac_apply(cfg, per_agent=True)
    rep = jax.tree.map(lambda a: jnp.broadcast_to(a, (n,) + a.shape),
                       shared["actor"])
    m_s, l_s = act_s(shared["actor"], obs)
    m_p, l_p = act_p(rep, obs)
    assert float(np.abs(np.asarray(m_p) - np.asarray(m_s)).max()) <= MEAN_ABS
    assert float(np.abs(np.asarray(l_p) - np.asarray(l_s)).max()) <= LOG_STD_ABS


def test_the_two_layouts_are_not_the_same_learner(cell, policies, cfg):
    """Same key, same observation, different action -- by more than a rounding.

    If this were near zero the column would be measuring nothing.  Measured
    2026-09-09, CPU, x64, cell `2040p_2040c`, seed 0: max |Δmean| = **1.027** in
    the unsquashed R^3 coordinates of §9.3, against the 1.4e-5 the replication
    bound above allows -- five orders apart, so neither check can be passing on
    the other's account.  The floor is set three orders below the measurement so
    it fails on a collapse rather than on noise.
    """
    _env, _params, _n, _scale, obs = cell
    shared, pa, _n2 = policies
    a = np.asarray(cb.sac_apply(cfg)[0](shared["actor"], obs)[0])
    b = np.asarray(cb.sac_apply(cfg, per_agent=True)[0](pa["actor"], obs)[0])
    assert a.shape == b.shape
    assert float(np.abs(a - b).max()) > 1e-3


def test_crossing_the_layouts_is_refused(cell, policies, cfg):
    """A shared actor in the per-agent path raises instead of giving numbers.

    `obs_dim` is 23 and `n_agent` 24 on this cell (34 on cells 2 and 3), so
    `vmap` sees inconsistent sizes.  The guard is `vmap`'s, not the driver's, and
    on a market where those two numbers agreed nothing would raise -- which is
    why the check that bites is the knockout below.
    """
    _env, _params, _n, _scale, obs = cell
    shared, _pa, _n2 = policies
    with pytest.raises(ValueError):
        cb.sac_apply(cfg, per_agent=True)[0](shared["actor"], obs)


@pytest.mark.parametrize("j", [0, 7, 23])
def test_knocking_out_one_agent_moves_that_agent_and_no_other(cell, policies,
                                                              cfg, j):
    """Which agent reads which network, asked behaviourally.

    Agent `j`'s output layer is zeroed, so its `mean` becomes zero and its
    `log_std` the midpoint of the box.  If the parameter axis were mapped against
    some other axis, either nobody would move or everybody would.  Measured
    2026-09-09 at j = 7 on cell `2040p_2040c`: the knocked-out agent moves
    **8.54e-01** in policy coordinates while every other agent moves **exactly
    0.0** -- not "below a threshold", bit-identical -- so the 1e-9 cut below does
    no work and any positive cut would do.
    """
    _env, _params, n, _scale, obs = cell
    _shared, pa, _n = policies
    assert j < n
    act_p, _ = cb.sac_apply(cfg, per_agent=True)
    out = pa["actor"]["params"]["Dense_2"]
    hit = dict(pa["actor"], params=dict(
        pa["actor"]["params"],
        Dense_2=dict(kernel=out["kernel"].at[j].set(0.0),
                     bias=out["bias"].at[j].set(0.0))))
    before = np.asarray(act_p(pa["actor"], obs)[0])
    after = np.asarray(act_p(hit, obs)[0])
    delta = np.abs(after - before).max(axis=-1)
    assert np.where(delta > 1e-9)[0].tolist() == [j]
    assert float(np.delete(delta, j).max()) == 0.0


def test_the_temperature_reduction_sums_over_agents(cell):
    """The reduction gate: `sac.py` sums the agents, it does not mean them.

    With `n` copies of one temperature and one batch of log-probabilities, the
    per-agent gradients must sum to `n` times the shared gradient -- that is what
    a **sum** over the agent axis implies and a mean does not.  Measured
    2026-09-09 on cell `2040p_2040c`: shared 1.200467e+00, per-agent sum
    2.881121e+01, ratio **24.000000 = n**.  A mean over agents would put the
    ratio at 1, which is the factor the second assertion refuses.
    """
    _env, _params, n, _scale, _obs = cell
    logp = jax.random.normal(jax.random.PRNGKey(9), (32, n)) - 3.0
    la_s = jnp.asarray(jnp.log(0.2))
    la_p = jnp.full((n,), jnp.log(0.2))
    g_s = float(jax.grad(cb.make_sac_alpha_loss(-3.0))(la_s, logp))
    g_p = np.asarray(jax.grad(cb.make_sac_alpha_loss(-3.0, per_agent=True))(
        la_p, logp))
    assert abs(g_p.sum() / g_s - n) < 1e-9, g_p.sum() / g_s
    assert abs(g_p.sum() / g_s - 1.0) > 1.0, "a sum over agents reads as a mean"
    # And each agent's temperature sees only its own samples: the per-agent
    # gradients are not all equal, because the columns of `logp` are not.
    assert float(g_p.max() - g_p.min()) > 1e-3, g_p


# ------------------------------------------------------------ the rollout ---
@pytest.fixture(scope="module")
def sac_traj(cell, policies, cfg):
    """One SAC rollout of one episode, mode 0, and the policy that produced it."""
    env, params, n, scale, _obs = cell
    shared, _pa, _n = policies
    roll = cb.make_sac_rollout(env, params, scale, n, cfg)
    return roll(shared, jax.random.PRNGKey(11), 0), shared


def test_the_reference_arms_are_bit_identical_across_the_two_learners(cell,
                                                                      policies,
                                                                      cfg):
    """Modes 1 to 4 discard the network, so `--algo` may not move them.

    This is the check that the two arms of one cell differ in the learner and in
    nothing else: the four reference arms are the market's, and every ordering
    §16 reports is measured against them.  It also pins the fifteen shared
    positions of the two rollouts to each other -- a slot that slipped would show
    up here as a difference in a quantity nobody meant to change.

    Measured 2026-09-09 on cell `2040p_2040c` with two episodes: reward, the
    curtailment channel, volume, price, spend, both requirement counts, the
    solver's `mu`, convergence and the overload channel are **bit-identical** for
    every one of the four modes, while mode 0 -- where the network is read --
    differs by **4.91e+01** in reward.  So the check is not passing because the
    two rollouts happen to agree everywhere.
    """
    env, params, n, scale, _obs = cell
    shared, _pa, _n = policies
    roll_i = cb.make_rollout(env, params, scale, n, cb.LOG_STD_FINAL)
    roll_s = cb.make_sac_rollout(env, params, scale, n, cfg)
    dummy = cb.init_policy(jax.random.PRNGKey(0), env[3]["obs_dim"])
    keys = jax.random.split(jax.random.PRNGKey(11), N_EPISODES)
    run = lambda r, p, m: jax.vmap(r, in_axes=(None, 0, None))(p, keys, m)
    #: reward, shed, volume, price, spend, req_th, req_v, mu, converged, overload
    SHARED_SLOTS = (4, 5, 6, 7, 8, 9, 11, 12, 13, 14)
    for mode in (1, 2, 3, 4):
        a, b = run(roll_i, dummy, mode), run(roll_s, shared, mode)
        for i in SHARED_SLOTS:
            assert np.array_equal(np.asarray(a[i]), np.asarray(b[i])), (mode, i)
    a0, b0 = run(roll_i, dummy, 0), run(roll_s, shared, 0)
    assert float(np.abs(np.asarray(a0[4]) - np.asarray(b0[4])).max()) > 1.0


def test_terminal_obs_is_the_successor_observation(cell, sac_traj):
    """What the replay buffer's `next_obs` is, checked against the environment.

    `env.py` computes ``terminal_obs = where(done, _get_obs(next_state), obs)``
    where `obs` is already the observation AFTER the automatic reset, so on an
    interior step both branches are the successor and on the last step it is the
    successor the reset overwrote.  Reading that `obs` as the step's INPUT
    observation instead -- which is how it reads if only the one line is read --
    would put the buffer's `next_obs` one period early and make the Bellman
    target regress on the wrong transition.

    Measured 2026-09-09 on cell `2040p_2040c`: the two readings are **0.0** and
    **20.0** apart respectively, and 20.0 is the full width of the clipping box,
    so nothing about this is a tolerance.
    """
    _env, _params, _n, scale, _obs = cell
    (obs, _pre, _lp, _v, _r, *_rest), _pol = sac_traj
    term, nxt, done = _rest[5], _rest[10], _rest[11]
    scaled = np.asarray(jnp.clip(term / scale, -cb.OBS_CLIP, cb.OBS_CLIP))
    obs = np.asarray(obs)
    assert np.array_equal(scaled[:-1], obs[1:])
    assert float(np.abs(scaled[:-1] - obs[:-1]).max()) > 1.0
    # `next_obs` is exactly that, scaled and clipped the way `obs` is.
    assert np.array_equal(np.asarray(nxt), scaled)
    # And the boundary is where the scan says it is: one `done`, on the last
    # period, because `EPISODE_LEN` is `params.episode_len` and `reset` puts
    # `step_in_episode` at zero.
    assert np.asarray(done).tolist() == [False] * (cb.EPISODE_LEN - 1) + [True]


def test_the_value_slot_is_empty_on_the_sac_path(sac_traj):
    """Position 3 is IPPO's value estimate; SAC has no state-value head.

    Filled with zeros rather than left out, so `evaluate` reads the two rollouts
    by the same positions.  Asserted so that a later change cannot start putting
    something there and have `evaluate` silently consume it as a value.
    """
    (_obs, _pre, lp, value, *_rest), _pol = sac_traj
    assert np.asarray(value).shape == np.asarray(lp).shape
    assert not np.asarray(value).any()


# ------------------------------------------- what this market's box implies ---
def test_this_market_squashes_into_its_own_box(cell):
    """05 publishes a finite box, so both maps squash on all three coordinates.

    `policy.to_action` and `sac._q_action` mask on `isfinite(low) & isfinite
    (high)`.  Until 2026-09-10 this market published ``+-inf`` -- alone among
    the five: 01 and 02 bound the markup at ``[1, markup_max]``, 03 publishes
    its reserve box, 04 declares ``[-1, 1]`` -- and both maps were therefore the
    identity here, which is the one thing that reached `sac._q_action`'s
    pass-through branch on the five markets as they stand.  That branch is
    unreached again.

    That was not a curiosity.  `policy.bounds_for` records that an unbounded
    coordinate is what diverged SAC on the ancillary market, and the driver's SAC
    section records the measurement showing 05 diverged the same way -- `q_mean`
    from 1.07e2 to 6.38e8 inside one iteration, against 26.5 under a finite box.
    The box `envs/local_flexibility/action.py` publishes is
    ``+-ACTION_SATURATION``, the frame this market's own reference strategies
    already sit on, and this asserts it arrives at both maps.

    The discrimination is the measured distance from the identity on one draw:
    `to_action` moves the action by 124.36 and `_q_action`, which returns
    `tanh(pre)` rather than the box, by 9.17.  Both were exactly zero before.
    """
    env, _params, n, _scale, _obs = cell
    low, high = cb.sac_bounds(env[3])
    assert np.isfinite(np.asarray(low)).all()
    assert np.isfinite(np.asarray(high)).all()
    assert float(low.min()) == -ACTION_SATURATION
    assert float(high.max()) == ACTION_SATURATION
    assert low.shape == (n, 3) == high.shape
    pre = jax.random.normal(jax.random.PRNGKey(5), (n, 3)) * 3.0
    mapped = np.asarray(to_action(pre, low, high))
    assert np.all(np.abs(mapped) < ACTION_SATURATION)
    assert float(np.max(np.abs(mapped - np.asarray(pre)))) > 100.0
    # the critic's view is `tanh` of the coordinate, so it is inside (-1, 1)
    q_seen = np.asarray(_q_action(pre, low, high))
    assert np.all(np.abs(q_seen) < 1.0)
    assert float(np.max(np.abs(q_seen - np.asarray(pre)))) > 1.0


def test_a_saturated_policy_reads_as_the_incdec_arm(cell, policies, cfg):
    """Why a diverged SAC reports `incdec`'s return, bit for bit.

    §9.3's action map is flat past 128 -- that is why `INCDEC_ACTION` is written
    at 128 -- so a policy whose output has run away in `incdec`'s sign pattern
    produces `incdec`'s trajectory exactly.  The end-to-end smoke of `--algo sac`
    did exactly that (`ret` 5.2531909943 against `incdec` 5.2531909943), and
    without this test that reads as the learner discovering the strategy §8
    leaves in place.

    Measured 2026-09-09 on cell `2040p_2040c`, two episodes: bit-identical
    rewards at magnitudes 1e2, 1e3 and 1e5, i.e. the map has already saturated at
    the value `INCDEC_ACTION` uses.
    """
    env, params, n, scale, _obs = cell
    shared, _pa, _n = policies
    keys = jax.random.split(jax.random.PRNGKey(11), N_EPISODES)
    roll_i = cb.make_rollout(env, params, scale, n, cb.LOG_STD_FINAL)
    dummy = cb.init_policy(jax.random.PRNGKey(0), env[3]["obs_dim"])
    incdec = jax.vmap(roll_i, in_axes=(None, 0, None))(dummy, keys, 4)[4]
    # A constant actor: the output layer's kernel zeroed and its bias set to
    # `incdec`'s signs at `mag`, divided by ACTION_GAIN so the ACTION is `mag`.
    out = shared["actor"]["params"]["Dense_2"]
    signs = jnp.sign(jnp.asarray(cb.INCDEC_ACTION))
    roll_g = cb.make_sac_rollout(env, params, scale, n, cfg, greedy=True)
    for mag in (128.0, 1e5):
        actor = dict(shared["actor"], params=dict(
            shared["actor"]["params"],
            Dense_2=dict(kernel=jnp.zeros_like(out["kernel"]),
                         bias=jnp.broadcast_to(signs * mag / cb.ACTION_GAIN,
                                               out["bias"].shape))))
        got = jax.vmap(roll_g, in_axes=(None, 0, None))(
            dict(shared, actor=actor), keys, 0)[4]
        assert np.array_equal(np.asarray(got), np.asarray(incdec)), mag


# ------------------------------------------------------- the two judgements ---
def test_annealed_log_std_kills_the_entropy_gradient_on_an_unbounded_box(
        cell, policies, cfg):
    """Why 05's annealed `log_std` is NOT SAC's exploration width -- and what
    publishing a box did to that reason.

    SAC's actor loss is ``mean(alpha * log pi - Q)``.  With **nothing
    squashed**, which is what this market declared until 2026-09-10, the
    log-density under ``pre = mean + exp(log_std) * eps`` is
    ``sum(-0.5 eps^2 - log_std - 0.5 log 2pi)``: a `log_std` handed in by the
    training loop makes the actor's parameters vanish from it and the entropy
    term has **no gradient**, so SAC's temperature would autotune against a
    quantity its actor cannot move.  With the package's state-dependent head the
    term is live.

    Measured 2026-09-09, CPU, x64, cell `2040p_2040c`, seed 0, over the actor's
    73 478 parameters: with the head, max |g| = **4.534** and 32 122
    coordinates are non-zero; with 05's `LOG_STD_FINAL` handed in, max |g| =
    **exactly 0.0** and **0** coordinates are non-zero.  That is bit-zero, not
    small, which is why the assertion is an equality -- and it is asserted at
    the box the measurement was taken on, supplied explicitly through the
    diagnostic keyword `sac_bounds` keeps for exactly this.

    **The premise is gone and the decision built on it has not been retaken.**
    The published box makes `policy.log_prob` add its `tanh` correction, the
    correction is a function of `pre`, and `pre` is a function of the actor's
    parameters -- so under this market's own box the gradient of the handed-in
    constant branch is no longer zero, which the second half asserts.  Whether
    that reopens the decision not to feed 05's anneal to SAC is a question
    about the learner rather than about the market, so it is recorded here and
    not answered here.
    """
    _env, _params, n, _scale, obs = cell
    shared, _pa, _n = policies
    act, _ = cb.sac_apply(cfg)
    eps = jax.random.normal(jax.random.PRNGKey(3), (n, 3))
    unbounded = (jnp.full((n, 3), -jnp.inf), jnp.full((n, 3), jnp.inf))
    published = cb.sac_bounds(_env[3])

    def with_head(actor_params, low, high):
        mean, log_std = act(actor_params, obs)
        pre = mean + jnp.exp(log_std) * eps
        return jnp.mean(squashed_log_prob(pre, mean, log_std, low, high))

    def with_annealed_constant(actor_params, low, high):
        c = jnp.float32(cb.LOG_STD_FINAL)
        mean, _log_std = act(actor_params, obs)
        pre = mean + jnp.exp(c) * eps
        return jnp.mean(squashed_log_prob(pre, mean,
                                          jnp.broadcast_to(c, mean.shape),
                                          low, high))

    flat = lambda g: np.concatenate([np.asarray(v).ravel()
                                     for v in jax.tree_util.tree_leaves(g)])
    grad = lambda f, box: flat(jax.grad(f)(shared["actor"], *box))
    live = grad(with_head, unbounded)
    dead = grad(with_annealed_constant, unbounded)
    assert live.size == dead.size
    assert float(np.abs(dead).max()) == 0.0
    assert int((dead != 0).sum()) == 0
    assert float(np.abs(live).max()) > 1e-2, float(np.abs(live).max())
    # and the same branch under the box this market now publishes
    revived = grad(with_annealed_constant, published)
    assert float(np.abs(revived).max()) > 1e-2, float(np.abs(revived).max())


def test_the_boundary_is_masked_not_bootstrapped(cell, cfg, sac_traj):
    """05's driver settles the boundary, so the Q target must not bootstrap it.

    `ippo._BOOTSTRAP_AT_DONE` keys on `spec["termination"]`, 05 declares
    `"truncation"`, and that entry selects a bootstrap -- its comment names 05
    among the markets where "nothing is settled at the boundary".  True of the
    environment, false of this driver: `terminal_settlement` prices the carry
    into the last period's reward, which is `make_update`'s own reason for a zero
    bootstrap on the PPO side.  The learner spec therefore reports
    `bootstrap_at_done: false` against a spec that says `truncation`, and this
    test pins that disagreement so it cannot be "fixed" by reading the table.

    **The size of what the mask removes, at the initial parameters**: measured
    2026-09-09 on cell `2040p_2040c`, ``gamma * (min Q - alpha * log pi)`` at the
    successor of the done step has mean -0.174 and max |.| 1.354, against a
    scaled reward at that step of mean 91.7 and max |.| 425.6.  So at
    initialisation the term is under half a percent of the reward it would be
    added to, and this check has **little discriminating power there**; the term
    grows with the critic, which reached `q_mean` 26.5 after 1 536 updates in the
    bounded control.  What the test asserts is therefore the structure, and the
    magnitude is recorded rather than asserted.
    """
    env, params, n, scale, _obs = cell
    (_o, _p, _lp, _v, *_rest), policy = sac_traj
    assert env[3]["termination"] == "truncation"
    # The mask itself, and the stamp that is computed by asking it.
    like = jnp.zeros((2, 3))
    assert float(cb.sac_continuation(jnp.ones((2,), jnp.bool_), like).max()) == 0.0
    assert float(cb.sac_continuation(jnp.zeros((2,), jnp.bool_), like).min()) == 1.0
    _init, _upd, spec = cb.make_sac_update(env, params, scale, n, cfg)
    assert spec["bootstrap_at_done"] is False
    assert spec["algo"] == "sac"
    assert spec["annealed_log_std_used"] is False
    # `sac.py:254`'s `-dim(A)`, taken literally.  `act_dim` is read off the
    # action shape rather than written as `-3.0` so that a market that grew a
    # fourth column would not keep a stale literal; the **width** of the box
    # does not enter, and `test_the_entropy_target_does_not_scale_with_the_box`
    # is what holds that.
    assert spec["target_entropy"] == -float(env[3]["action_shape"][-1])
    assert spec["updates_per_iteration"] == cb.BATCH * cb.EPISODE_LEN
    assert (spec["n_envs"], spec["horizon"]) == (cb.BATCH, cb.EPISODE_LEN)
    # The box the critic's coordinate is read in, stamped because it is the
    # quantity `bounds_for` names as the reason an unbounded coordinate leaves
    # SAC without a fixed point.  Finite since 2026-09-10.
    assert (spec["action_low"], spec["action_high"]) \
        == (-ACTION_SATURATION, ACTION_SATURATION)


def test_one_iteration_runs_in_both_layouts(cell, cfg):
    """The L0 itself: jit, vmap and the replay carry, once, in each layout.

    Run at a small `n_envs` and a small `utd_ratio` -- the full configuration
    takes 109 s per iteration on CPU -- because what is under test here is that
    the iteration compiles and its carry keeps its pytree structure, not what it
    learns.  The buffer's dtypes are asserted because they are read from the
    rollout by `eval_shape` rather than written down: the observation is float32
    while the driver runs under x64, and a buffer typed by hand would store a
    float64 copy or narrow silently, neither of which announces itself.
    """
    import dataclasses
    env, params, n, scale, _obs = cell
    small = dataclasses.replace(cfg, n_envs=2, buffer_size=256, batch_size=8,
                                utd_ratio=4 / (2 * cb.EPISODE_LEN))
    for per_agent in (False, True):
        policy = cb.init_sac_policy(jax.random.PRNGKey(0), env[3]["obs_dim"], n,
                                    small, per_agent=per_agent)
        init_state, update, spec = cb.make_sac_update(
            env, params, scale, n, small, per_agent=per_agent)
        state = init_state(policy)
        assert spec["updates_per_iteration"] == 4
        assert {k: str(v.dtype) for k, v in state["buffer"].items()} == dict(
            obs="float32", pre="float32", reward="float32",
            next_obs="float32", done="bool", cursor="int32", filled="int32")
        out, state2, stats = update(policy, state, jax.random.PRNGKey(1),
                                    cb.LOG_STD_FINAL)
        assert jax.tree_util.tree_structure(out) == \
            jax.tree_util.tree_structure(policy)
        assert jax.tree_util.tree_structure(state2) == \
            jax.tree_util.tree_structure(state)
        assert int(state2["buffer"]["filled"]) == 2 * cb.EPISODE_LEN
        for k in cb.SAC_DIAGNOSTICS:
            assert np.isfinite(float(stats[k])), (per_agent, k)
        # The temperature moved off its initial value on the first iteration,
        # which is what "autotuned from the first update" means (`SACConfig`).
        assert float(stats["alpha"]) != cfg.init_alpha

def test_the_entropy_target_does_not_scale_with_the_box(cell, policies, cfg):
    """The target is `sac.py`'s ``-dim(A)``, and the box's width does not enter.

    **The rule, and why it is not the other one.**  `sac.py:254` is what 01,
    02, 03 and 04 run under, and none of their boxes is ``[-1,1]`` either: the
    per-coordinate correction ``log(0.5 w)`` is -0.6931 on 01 and 02, +2.5255
    on 03's reserve columns and exactly 0 on 04.  Scaling the target by the box
    would make 05 the only market whose target moves with its own action space,
    and 1812.05905v2 table 1 gives the *value* ``-dim(A)`` rather than a rule
    for rescaling it.  Between 2026-09-10 and 2026-09-11 this market did scale
    it; that was reverted, and this test is what keeps it reverted.

    **The bite is the second assertion**: the same call under a box sixteen
    times narrower has to return the same target.  A target computed from the
    box fails it however the formula is written.

    **The known cost of the literal value, recorded rather than asserted**, so
    that a reader meeting a decaying `alpha` does not read it as a defect.  A
    squashed policy's entropy scale really does move with the box --
    `policy.log_prob` subtracts ``log(0.5 (high - low))`` per squashed
    coordinate, measured on this box as exactly ``3 log 128 = 14.55609`` at
    every `log_std` from -5 to +2 -- so at the actor's *initial* parameters the
    attainable entropy is ``[0.98, 16.46]`` and ``-3`` lies 3.98 below it,
    which leaves the temperature's constraint slack and `alpha` decaying.
    **That is a transient rather than the fixed point.**  ``-3`` becomes
    attainable once ``|mean|`` reaches about 2, and reading C-AO, on a remote
    A10G over the full 300 iterations, finds the run settling there: entropy
    steady at -2.9 on its target, `alpha` in ``[1.2e-3, 1.2e-2]``, `q_loss` in
    ``[1.63e3, 9.71e3]``, no non-finite values, `eval` rising 1.84 to 28.87.
    The scaled target held its own value instead by letting the multiplier
    diverge -- `alpha` to 306 and `q_loss` to 4.8e8, also with no non-finite
    values, so that was a diverging Lagrange multiplier and not an overflow.
    """
    env, params, n, scale, _obs = cell
    act_dim = int(env[3]["action_shape"][-1])

    _i, _u, published = cb.make_sac_update(env, params, scale, n, cfg)
    assert published["target_entropy"] == -float(act_dim)

    # the bite: a box sixteen times narrower, and the same target out of it
    narrow = (jnp.full((n, act_dim), -8.0), jnp.full((n, act_dim), 8.0))
    _i2, _u2, other = cb.make_sac_update(env, params, scale, n, cfg,
                                         bounds=narrow)
    assert other["target_entropy"] == published["target_entropy"]
    # and the control really is a different box, or the line above is vacuous
    assert (other["action_low"], other["action_high"]) \
        != (published["action_low"], published["action_high"])
