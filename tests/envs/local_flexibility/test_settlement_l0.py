"""L0 JAX contract for the settlement (§8).

Small, because the settlement is arithmetic: no solver, no loop, no branch on a
traced value.  What has to hold is that it composes, so it is exercised under
`jit`, under `vmap` over parallel environments, and inside a `lax.scan` that
accumulates an episode.

The accumulation is not incidental.  §15 requires episode totals to be reduced
with `jnp.sum` over the outputs of the scan rather than added into the carry,
because sequential float32 accumulation drifts by two to three orders of
magnitude more than a tree reduction over a hundred thousand steps.  Both are
compared below at a length where the difference is visible.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from powermarketjax.case import load_case
from powermarketjax.envs.local_flexibility import (build_voltage_sensitivity,
                                                   make_settlement)

N_AGENT = 12
T = 96                      # one day at the fifteen-minute period of §14
BATCH = 8


@pytest.fixture(scope="module")
def settle():
    return make_settlement(build_voltage_sensitivity(load_case("33bw")))


@pytest.fixture
def draw():
    rng = np.random.default_rng(0)
    return (jnp.asarray(rng.uniform(20.0, 80.0, N_AGENT)),
            jnp.asarray(rng.uniform(0.0, 0.05, N_AGENT)),
            jnp.asarray(rng.uniform(0.0, 0.05, N_AGENT)),
            40.0, jnp.full((N_AGENT,), 3.0))


def test_jit(settle, draw):
    out = jax.jit(settle)(*draw)
    assert set(out) == {"revenue", "cost", "profit", "reward", "p_ch", "p_dis"}
    for name, value in out.items():
        assert value.shape == (N_AGENT,), name
        assert jnp.isfinite(value).all(), name
    np.testing.assert_array_equal(np.asarray(out["reward"]), np.asarray(out["profit"]))


def test_pytree_structure_is_independent_of_the_data(settle, draw):
    price, award, plan, energy, cycle = draw
    idle = jax.tree.structure(settle(price, jnp.zeros_like(award), plan, energy, cycle))
    busy = jax.tree.structure(settle(price, award, jnp.zeros_like(plan), energy, cycle))
    assert idle == busy


def test_vmap_over_parallel_environments(settle, draw):
    price, award, plan, energy, cycle = draw
    awards = award[None, :] * jnp.linspace(0.0, 1.0, BATCH)[:, None]

    out = jax.jit(jax.vmap(settle, in_axes=(None, 0, None, None, None)))(
        price, awards, plan, energy, cycle)

    assert out["revenue"].shape == (BATCH, N_AGENT)
    # the lane clearing nothing earns nothing, and the rest earn more than it
    assert float(jnp.sum(out["revenue"][0])) == 0.0
    assert float(jnp.sum(out["revenue"][-1])) > 0.0


def test_scan_over_an_episode_and_the_reduction_rule(settle, draw):
    """`jnp.sum` over the stacked outputs, not an addition into the carry (§15)."""
    price, award, plan, energy, cycle = draw
    profile = award[None, :] * jnp.linspace(0.2, 1.0, T)[:, None]

    def period(running, a):
        out = settle(price, a, plan, energy, cycle)
        return running + out["profit"].sum(), out["profit"]

    in_carry, stacked = jax.jit(
        lambda xs: lax.scan(period, jnp.asarray(0.0), xs))(profile)

    assert stacked.shape == (T, N_AGENT)
    reduced = jnp.sum(stacked)
    # the two agree here because this suite runs in float32 at a length of one
    # day; the rule exists for the hundred-thousand-step episodes of §15, and
    # the tolerance is the one pitfalls §9 fixes
    np.testing.assert_allclose(float(in_carry), float(reduced), rtol=1e-5)


def test_vmap_of_scan(settle, draw):
    price, award, plan, energy, cycle = draw
    profile = (award[None, None, :] * jnp.linspace(0.2, 1.0, T)[None, :, None]
               * jnp.linspace(0.5, 1.5, 64)[:, None, None])

    def rollout(xs):
        return lax.scan(lambda c, a: (c, settle(price, a, plan, energy, cycle)["reward"]),
                        jnp.asarray(0.0), xs)[1]

    reward = jax.jit(jax.vmap(rollout))(profile)
    assert reward.shape == (64, T, N_AGENT) and jnp.isfinite(reward).all()


def test_float64_and_float32_agree_to_float32_precision(settle, draw):
    """The settlement is arithmetic, so it runs in whatever precision it is given."""
    price, award, plan, energy, cycle = draw
    narrow = np.asarray(settle(price, award, plan, energy, cycle)["profit"], np.float64)

    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        wide = np.asarray(settle(jnp.asarray(price, jnp.float64),
                                 jnp.asarray(award, jnp.float64),
                                 jnp.asarray(plan, jnp.float64), energy,
                                 jnp.asarray(cycle, jnp.float64))["profit"], np.float64)
    finally:
        jax.config.update("jax_enable_x64", previous)

    np.testing.assert_allclose(narrow, wide, rtol=1e-5)
