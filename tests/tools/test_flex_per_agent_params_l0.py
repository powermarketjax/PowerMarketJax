# Written for this repository on 2026-09-09 -- no upstream counterpart.
"""L0 for `--per-agent-params` in `tools/flex_experiment/concentration_baseline.py`.

**This driver had no tests at all** (`grep -rl concentration_baseline tests/`
returned nothing on 2026-09-09), so this file is the first, and it covers the
one thing the flag adds rather than the whole driver.

**What a shape test cannot catch here.**  Giving every aggregator its own copy
of the network moves exactly one thing: an axis.  Every assertion about shapes
passes whether the parameter axis is lined up against the agent axis of the
observation or against some other axis of the same length, and on a scenario
whose `obs_dim` equalled `n_agents` even `vmap` would accept the wrong pairing.
On this feeder they never coincide -- `obs_dim` is 23 in all four cells while
`n_agents` is 24 for the 2040 placement and 34 for the 2050 one -- but that is a
coincidence of the data and not a guard anyone designed.  The check that bites
is therefore behavioural: knock out agent `j`'s output head and require that
agent `j` and no other agent moves.

**The reductions are gated too, and that is the point of the ticket rather than
a bonus.**  05's learning loop is written in the driver and does not call
`powermarketjax/learning/ippo.py`, so nothing but a test stops the two from
drifting into two different meanings of "per-agent parameters".  The identity
`test_shared_gradient_is_the_sum_over_agents` asserts is the one a **mean** over
the agent axis implies and a **sum** over it does not: with `n_agents` copies of
one network, the shared gradient is the sum over agents of the per-agent
gradients.  Under a sum the two sides would differ by a factor of `n_agents`,
which is 24 here, and the test measures that factor rather than asserting it.

**The scenario is the real one.**  `cb.scenario(*CELLS[0])` reads the SwissDN
parquet the way the other Swiss tests in this repository do, so `n_agents`,
`obs_dim` and the observations are the driver's own and not stand-ins.
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

# **The import is bracketed, and that is not tidiness.**  `concentration_baseline`
# turns `jax_enable_x64` and `jax_default_matmul_precision` on at module scope,
# which is right for the driver and wrong for the session: measured 2026-09-09,
# importing it unguarded turned `tests/tools/test_runtime_stamp_l0.py` red in a
# `pytest tests/tools` run while both files pass alone, because that file asserts
# the session runs with x64 OFF -- its own docstring says a session that turns
# x64 on "would pass while testing nothing".  So the two settings are restored
# here and re-applied for this module's tests only, by the fixture below, which
# is the arrangement `tests/learning/test_ippo_per_agent_params_l0.py` uses.
_PREV_X64 = jax.config.jax_enable_x64
_PREV_MATMUL = jax.config.jax_default_matmul_precision

import concentration_baseline as cb  # noqa: E402

jax.config.update("jax_enable_x64", _PREV_X64)
jax.config.update("jax_default_matmul_precision", _PREV_MATMUL)


@pytest.fixture(scope="module", autouse=True)
def x64():
    """The driver's own numerics, for this module only.

    Every tolerance in this file was measured under x64 with the highest matmul
    precision, which is what the driver runs on; under float32 they would mean
    something else.
    """
    prev = jax.config.jax_enable_x64
    prev_mm = jax.config.jax_default_matmul_precision
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    yield
    jax.config.update("jax_enable_x64", prev)
    jax.config.update("jax_default_matmul_precision", prev_mm)

#: Two episodes is enough to build a `data` block with every axis populated and
#: is what keeps this file an L0: the driver's own `update` collects BATCH = 64.
N_EPISODES = 2


@pytest.fixture(scope="module")
def cell():
    """Cell 0 of the two-by-two, built once: `(env, params, n, scale, obs)`.

    `obs` is one period's observation taken through the driver's own scaling and
    clipping, so its trailing axis is the `obs_dim` the network actually reads.
    """
    env, params, n = cb.scenario(*cb.CELLS[0], split="train")
    scale = cb.obs_scale(env, params, n)
    _, state = env[0](jax.random.PRNGKey(0), params)
    obs = jnp.clip(env[3]["get_obs"](state, params) / scale,
                   -cb.OBS_CLIP, cb.OBS_CLIP)
    return env, params, n, scale, obs


@pytest.fixture(scope="module")
def policies(cell):
    """`(shared, per_agent, n)`: two layouts from the same seed."""
    env, _params, n, _scale, _obs = cell
    obs_dim = env[3]["obs_dim"]
    key = jax.random.PRNGKey(0)
    return (cb.init_policy(key, obs_dim),
            cb.init_policy_per_agent(key, obs_dim, n),
            n)


def test_flag_off_leaves_the_parameter_tree_alone(cell, policies):
    """The shared layout is the one every product on disk was produced on."""
    env, _params, n, _scale, _obs = cell
    shared, _pa, _n = policies
    obs_dim, h = env[3]["obs_dim"], cb.HIDDEN
    want = dict(w1=(obs_dim, h), b1=(h,), w2=(h, h), b2=(h,),
                mean=(h, 3), mean_b=(3,), value=(h, 1), value_b=(1,))
    assert {k: v.shape for k, v in shared.items()} == want
    # No leaf carries a leading agent axis, which is the whole difference
    # between the two layouts.
    assert all(v.shape[0] != n or k == "w1" for k, v in shared.items()), \
        "a shared leaf leads with the agent count; the layouts are not distinct"
    # The three constructors the driver threads the flag through default to the
    # shared path, so every caller written before the flag keeps its behaviour.
    import inspect
    for fn in (cb.make_rollout, cb.make_update, cb.evaluate, cb.make_loss_fn):
        param = inspect.signature(fn).parameters["per_agent"]
        assert param.default is False, fn.__name__


def test_flag_on_gives_every_leaf_the_agent_axis(policies):
    """Shapes follow the participant count, and the scalars multiply by it."""
    shared, pa, n = policies
    assert set(pa) == set(shared)
    for k in shared:
        assert pa[k].shape == (n,) + shared[k].shape, k
    n_shared = sum(int(np.prod(v.shape)) for v in shared.values())
    n_pa = sum(int(np.prod(v.shape)) for v in pa.values())
    # Measured 2026-09-09, CPU, cell `2040p_2040c`: 5 956 scalars against
    # 142 944, i.e. exactly n = 24 times.
    assert n_pa == n * n_shared


def test_the_two_layouts_are_not_the_same_policy(cell, policies):
    """Same key, same observation, different action -- by more than a rounding.

    If this were near zero the column would be measuring nothing: the treatment
    is supposed to start the run from a different point in parameter space.
    """
    _env, _params, _n, _scale, obs = cell
    shared, pa, _n2 = policies
    a = np.asarray(cb.forward(shared, obs)[0]) * cb.ACTION_GAIN
    b = np.asarray(cb.forward_per_agent(pa, obs)[0]) * cb.ACTION_GAIN
    assert a.shape == b.shape
    gap = float(np.max(np.abs(a - b)))
    # Measured 2026-09-09, CPU, x64, cell `2040p_2040c`, seed 0: max |Δaction|
    # = **1.070e-01** in the unsquashed R^3 coordinates of §9.3, and the agent
    # that moves LEAST still moves 1.790e-02, so this is every agent and not one
    # outlier.  For scale, the shared action itself spans [-2.52e-02, 3.10e-02]
    # on this observation, so the two layouts are further apart than the shared
    # policy's own range.  The floor is set two orders below the measurement so
    # it fails on a collapse rather than on noise; the units are the action's
    # own, not a percentage.
    assert gap > 1e-3, gap


#: How far the two heads may drift when one network is replicated `n` times.
#: **The action head does not drift at all and the value head does**, which is a
#: measurement rather than a design: measured 2026-09-09, CPU, x64,
#: `jax_default_matmul_precision="highest"`, cell `2040p_2040c`, seed 0, the mean
#: is equal in every bit while the value differs in 18 of its 24 entries by at
#: most **4.34e-18, i.e. 5 ULP** at that magnitude.  The cause is the shape of
#: the two contractions under `vmap`: the value head contracts 64 hidden units
#: into 1 output, and XLA reassociates that batched matrix-vector product
#: differently from the unbatched one, while the (64, 3) product of the action
#: head keeps its order.  The bound is therefore derived from the contraction
#: length -- 64 terms, so 64 ULP -- and not pinned to the 5 that were measured.
VALUE_HEAD_ULP = 64


def test_replicating_one_network_reproduces_the_shared_forward(cell, policies):
    """`n` copies of one network must give the shared answer.

    This is the axis check in the direction the knockout cannot see: it fails if
    the parameter axis is mapped against anything but the agent axis, because
    then the copies would not line up with the observations that produced the
    shared answer.  A crossed axis moves the output by the spread of the
    replicated network over agents, and that spread was measured at 1.66e-01
    against the 5.6e-17 the band below allows -- fifteen orders apart, so the
    tolerance is nowhere near wide enough to swallow the defect it is guarding.
    """
    _env, _params, n, _scale, obs = cell
    shared, _pa, _n = policies
    rep = jax.tree.map(lambda a: jnp.broadcast_to(a, (n,) + a.shape), shared)
    m_s, v_s = cb.forward(shared, obs)
    m_p, v_p = cb.forward_per_agent(rep, obs)
    # The action head: no tolerance, the same weights in the same order.
    assert np.array_equal(np.asarray(m_s), np.asarray(m_p))
    # The value head: a few ULP of reassociation, bounded by the contraction
    # length rather than by the number that came out.
    v_s, v_p = np.asarray(v_s), np.asarray(v_p)
    ulps = np.abs(v_p - v_s) / np.spacing(np.abs(v_s))
    assert ulps.max() <= VALUE_HEAD_ULP, ulps.max()


@pytest.mark.parametrize("j", [0, 7, 23])
def test_knocking_out_one_agent_moves_that_agent_and_no_other(cell, policies, j):
    """The check a shape assertion cannot make: which agent reads which network.

    Agent `j`'s output head is zeroed, so its mean becomes the bias alone.  If
    the parameter axis were mapped against some other axis, either nobody would
    move or everybody would.
    """
    _env, _params, n, _scale, obs = cell
    _shared, pa, _n = policies
    assert j < n
    hit = dict(pa)
    hit["mean"] = pa["mean"].at[j].set(0.0)
    hit["mean_b"] = pa["mean_b"].at[j].set(0.0)
    before = np.asarray(cb.forward_per_agent(pa, obs)[0])
    after = np.asarray(cb.forward_per_agent(hit, obs)[0])
    moved = np.where(np.abs(after - before).max(axis=-1) > 1e-9)[0]
    assert moved.tolist() == [j], moved.tolist()
    # The two sides of the discrimination, measured 2026-09-09 at j = 7 on cell
    # `2040p_2040c`: the knocked-out agent moves 3.95e-02 in action units while
    # every other agent moves **exactly 0.0** -- not "below the threshold",
    # bit-identical.  So the 1e-9 above is not doing any work, and the assertion
    # would still hold at any positive cut.
    delta = np.abs(after - before).max(axis=-1)
    assert float(np.delete(delta, j).max()) == 0.0


def test_crossing_the_layouts_is_refused(cell, policies):
    """A shared tree in the per-agent path raises instead of producing numbers.

    `obs_dim` is 23 and `n_agents` is 24 on this cell (34 on cells 2 and 3), so
    `vmap` sees inconsistent sizes.  The guard is `vmap`'s, not the driver's, and
    it is recorded here because it is what makes the flag safe to be a keyword --
    on a market where those two numbers agreed, nothing would raise.
    """
    _env, _params, _n, _scale, obs = cell
    shared, _pa, _n2 = policies
    with pytest.raises(ValueError):
        cb.forward_per_agent(shared, obs)


def test_shared_gradient_is_the_sum_over_agents(cell, policies):
    """The reduction gate: the loss means over the agent axis, it does not sum.

    Built on the driver's own rollout and its own `advantages`, so the two sides
    are two paths through one loss rather than one path against a recomputed
    expectation.  The parameters the loss is evaluated at are perturbed away from
    the ones that collected the data, so `ratio` is not identically one and the
    clipped surrogate is exercised rather than sitting at its kink.
    """
    env, params, n, scale, _obs = cell
    shared, _pa, _n = policies
    log_std = cb.LOG_STD_FINAL
    roll = cb.make_rollout(env, params, scale, n, log_std)
    keys = jax.random.split(jax.random.PRNGKey(3), N_EPISODES)
    obs, raw, lp, value, reward, *_ = jax.vmap(
        roll, in_axes=(None, 0, None))(shared, keys, 0)
    gae, target = jax.vmap(cb.advantages)(reward, value,
                                          jnp.zeros_like(value[:, -1]))
    data = dict(obs=obs, raw=raw, logp=lp, gae=gae, target=target,
                ret_scale=jnp.maximum(jnp.abs(target).mean(), 1.0))

    # Off the collection point, so `ratio != 1`.
    noise = jax.random.normal(jax.random.PRNGKey(11), ())
    at = jax.tree.map(lambda a: a + 0.01 * noise * jnp.ones_like(a), shared)
    rep = jax.tree.map(lambda a: jnp.broadcast_to(a, (n,) + a.shape), at)

    g_shared = jax.grad(cb.make_loss_fn(per_agent=False))(at, data, log_std)
    g_pa = jax.grad(cb.make_loss_fn(per_agent=True))(rep, data, log_std)
    summed = jax.tree.map(lambda a: a.sum(0), g_pa)

    flat_s = np.concatenate([np.asarray(g_shared[k]).ravel()
                             for k in sorted(g_shared)])
    flat_p = np.concatenate([np.asarray(summed[k]).ravel()
                             for k in sorted(summed)])
    denom = np.maximum(np.abs(flat_s), 1e-30)
    rel = float(np.max(np.abs(flat_p - flat_s) / denom))
    # The two sides differ only in the order the same terms are added, so the
    # gap is float64 reassociation and not a modelling tolerance.  Measured
    # 2026-09-09, CPU, x64, `jax_default_matmul_precision="highest"`, cell
    # `2040p_2040c`, 2 episodes, gradient norm 9.00: **max relative deviation
    # 2.383e-12** over 5 956 coordinates (max absolute 7.77e-16).  The bound is
    # set nearly four orders above that, so it fails on a reduction that changed
    # and not on a different machine's summation order.
    assert rel < 1e-8, rel
    # And the number the gate bites with: a sum over the agent axis instead of a
    # mean would put the two sides a factor of n = 24 apart, which is 12 orders
    # of magnitude outside the bound above.
    assert not np.allclose(flat_p, n * flat_s, rtol=1e-8), \
        "the gate cannot tell a mean over agents from a sum over agents"
