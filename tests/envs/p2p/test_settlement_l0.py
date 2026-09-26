"""L0 for the P2P settlement: the JAX contract."""
import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.envs.p2p import make_clearing, make_settlement
from tests.envs.p2p.reference import population

PI_EXP, PI_RET = 4.1, 26.11
N = 16


def _one(seed=0, n=N, kappa_value=2.5):
    rng = np.random.default_rng(seed)
    price, q_sell, q_buy = population(rng, n, PI_EXP, PI_RET, tie_grid=2)
    clear, _ = make_clearing(n, PI_EXP, PI_RET)
    out = jax.jit(clear)(jnp.asarray(price), jnp.asarray(q_sell),
                         jnp.asarray(q_buy))
    throughput = jnp.asarray(rng.uniform(0.0, 1.0, n).astype(np.float32))
    kappa = jnp.full((n,), jnp.float32(kappa_value))
    return (jnp.asarray(q_sell), jnp.asarray(q_buy), out["award_sell"],
            out["award_buy"], out["clearing_price"], kappa, throughput)


@pytest.fixture
def settle():
    return make_settlement(PI_EXP, PI_RET)


def test_rejects_a_bad_tariff_pair():
    with pytest.raises(ValueError):
        make_settlement(PI_RET, PI_EXP)


def test_jit_matches_eager(settle):
    args = _one()
    chex.assert_trees_all_equal(settle(*args), jax.jit(settle)(*args))


def test_shapes_and_dtypes(settle):
    out = jax.jit(settle)(*_one())
    assert set(out) == {"revenue", "cost", "profit", "reward",
                        "degradation_cost"}
    for key, value in out.items():
        assert value.shape == (N,), key
        assert value.dtype == jnp.float32, key


def test_reward_is_profit(settle):
    out = jax.jit(settle)(*_one())
    np.testing.assert_array_equal(np.asarray(out["reward"]),
                                  np.asarray(out["profit"]))


def test_vmap_lanes_are_bit_identical(settle):
    args = _one()
    batch = 8
    tiled = tuple(jnp.broadcast_to(a, (batch,) + a.shape)
                  if a.ndim else jnp.full((batch,), a) for a in args)
    out = jax.jit(jax.vmap(settle))(*tiled)
    for key, value in out.items():
        assert value.shape == (batch, N)
        assert bool(jnp.all(value == value[0])), key


def test_runs_under_scan(settle):
    horizon = 24
    stacked = [jnp.stack([a for a in
                          (_one(seed=s)[i] for s in range(horizon))])
               for i in range(7)]

    def body(carry, xs):
        money = settle(*xs)
        return carry + money["profit"].sum(), money["reward"]

    total, rewards = jax.jit(
        lambda xs: jax.lax.scan(body, jnp.float32(0.0), xs))(tuple(stacked))
    assert rewards.shape == (horizon, N)
    assert bool(jnp.isfinite(total))


def test_no_nan_on_an_empty_market(settle):
    zeros = jnp.zeros((N,), jnp.float32)
    out = jax.jit(settle)(zeros, zeros, zeros, zeros,
                          jnp.float32(0.5 * (PI_EXP + PI_RET)), zeros, zeros)
    for key, value in out.items():
        assert bool(jnp.all(value == 0.0)), key
