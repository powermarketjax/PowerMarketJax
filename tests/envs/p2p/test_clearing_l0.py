"""L0 for the P2P clearing operator: the JAX contract.

jit, vmap with lane-by-lane identity, fixed shapes and dtypes independent of the
submissions, a full episode under `lax.scan` with no Python loop, and no NaN
anywhere in the output.  Nothing here checks that the auction is right; that is
`test_clearing_l1.py`.

This market runs in float32 and does **not** enable x64, the opposite
of the day-ahead operator.  Other modules in this repository switch the global
flag on, so these tests assert the dtype of the output rather than assuming the
ambient configuration.
"""
import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.envs.p2p import make_clearing
from tests.envs.p2p.reference import population

PI_EXP, PI_RET = 4.1, 26.11
N = 24


@pytest.fixture
def clear():
    fn, _ = make_clearing(N, PI_EXP, PI_RET)
    return fn


@pytest.fixture
def inputs():
    rng = np.random.default_rng(0)
    price, q_sell, q_buy = population(rng, N, PI_EXP, PI_RET, tie_grid=2)
    return jnp.asarray(price), jnp.asarray(q_sell), jnp.asarray(q_buy)


def test_spec_shapes_are_static():
    _, spec = make_clearing(N, PI_EXP, PI_RET)
    assert spec["n_agents"] == N
    assert spec["n_candidates"] == 2 * N + 1      # §6.2, independent of inputs
    assert spec["dtype"] is jnp.float32


def test_rejects_a_bad_tariff_pair_at_construction():
    # §3.3 is checked where it still can be: inside `jit` it cannot.
    with pytest.raises(ValueError):
        make_clearing(N, PI_RET, PI_EXP)
    with pytest.raises(ValueError):
        make_clearing(N, 5.0, 5.0)
    with pytest.raises(ValueError):
        make_clearing(N, -1.0, 5.0)


def test_jit_matches_eager(clear, inputs):
    eager = clear(*inputs)
    compiled = jax.jit(clear)(*inputs)
    chex.assert_trees_all_equal(eager, compiled)


def test_output_shapes_and_dtypes(clear, inputs):
    out = jax.jit(clear)(*inputs)
    for key in ("award_sell", "award_buy"):
        assert out[key].shape == (N,)
    for key in ("traded_volume", "clearing_price", "price_interval_lo", "price_interval_hi"):
        assert out[key].shape == ()
    for value in out.values():
        assert value.dtype == jnp.float32


def test_vmap_lanes_are_bit_identical(clear, inputs):
    price, q_sell, q_buy = inputs
    batch = 16
    tiled = tuple(jnp.tile(a, (batch, 1)) for a in (price, q_sell, q_buy))
    out = jax.jit(jax.vmap(clear))(*tiled)
    for key, value in out.items():
        assert value.shape[0] == batch
        assert bool(jnp.all(value == value[0])), key


def test_vmap_cost_does_not_depend_on_batch_content(clear):
    # Different lanes, one compilation: the shape of the candidate set is
    # 2n+1 whatever the submissions are, so nothing here is data dependent.
    rng = np.random.default_rng(1)
    batch = 8
    cols = [population(rng, N, PI_EXP, PI_RET, tie_grid=2) for _ in range(batch)]
    stacked = tuple(jnp.stack([jnp.asarray(c[i]) for c in cols]) for i in range(3))
    out = jax.jit(jax.vmap(clear))(*stacked)
    assert out["award_sell"].shape == (batch, N)
    assert bool(jnp.all(jnp.isfinite(out["clearing_price"])))


def test_runs_under_scan_without_a_python_loop(clear):
    """A full episode as `lax.scan`, which the environment contract requires."""
    horizon = 48
    rng = np.random.default_rng(2)
    cols = [population(rng, N, PI_EXP, PI_RET, tie_grid=2) for _ in range(horizon)]
    xs = tuple(jnp.stack([jnp.asarray(c[i]) for c in cols]) for i in range(3))

    def body(carry, x):
        out = clear(*x)
        return carry + out["traded_volume"], out["clearing_price"]

    total, prices = jax.jit(
        lambda xs: jax.lax.scan(body, jnp.float32(0.0), xs))(xs)
    assert prices.shape == (horizon,)
    assert bool(jnp.isfinite(total))
    assert bool(jnp.all((prices >= PI_EXP - 1e-4) & (prices <= PI_RET + 1e-4)))


def test_no_nan_on_degenerate_inputs(clear):
    """Setup-time validation cannot stop a NaN introduced at run time.

    The three cases below are the ones where a division or an empty reduction
    could appear: nobody submits anything, everybody is on one side, and every
    submission is tied.
    """
    zeros = jnp.zeros((N,), jnp.float32)
    ones = jnp.ones((N,), jnp.float32)
    flat = jnp.full((N,), jnp.float32(0.5 * (PI_EXP + PI_RET)))
    for price, q_sell, q_buy in [
        (flat, zeros, zeros),                       # empty market
        (flat, ones, zeros),                        # sellers only
        (flat, zeros, ones),                        # buyers only
        (jnp.full((N,), jnp.float32(PI_EXP)), ones, ones),   # everything tied
    ]:
        out = jax.jit(clear)(price, q_sell, q_buy)
        for key, value in out.items():
            assert bool(jnp.all(jnp.isfinite(value))), key


def test_grad_does_not_produce_nan(clear, inputs):
    """A zero gradient is expected here (§9.5); a NaN one is not.

    The reward is piecewise constant in the submitted price, so the pathwise
    derivative is zero almost everywhere.  What this checks is that the masks
    and the `where` sentinels do not manufacture a NaN on the backward pass,
    which is how a `where` around an undefined branch usually fails.
    """
    price, q_sell, q_buy = inputs
    g = jax.jit(jax.grad(lambda p: clear(p, q_sell, q_buy)["clearing_price"]))
    assert bool(jnp.all(jnp.isfinite(g(price))))
