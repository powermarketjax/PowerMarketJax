"""L0 JAX contract for the clearing operator (§6).

`jit`, `vmap` over parallel environments, `lax.scan` over the periods of an
episode, a pytree that does not depend on the data, and the float64 guard.

The development case is used throughout: the contract is about shapes and
tracing, not about the network, and the primary case would put a 573 by 3274
dense factorisation inside every one of these.

Batch determinism is checked because the interior point method runs a fixed
number of Newton steps, which is what makes the batch cost independent of batch
content and the lanes bit-identical.  A data-dependent `while_loop` would break
both, and that is what disqualified an external solver for this path.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from powermarketjax.case import load_case
from powermarketjax.envs.local_flexibility import build_voltage_sensitivity
from powermarketjax.envs.local_flexibility.clearing import make_clearing

CASE = "33bw"
N_AGENT = 6
T = 4
BATCH = 8


@pytest.fixture
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture
def built(x64):
    case = load_case(CASE)
    sens = build_voltage_sensitivity(case)
    rng = np.random.default_rng(0)
    load_bus = np.flatnonzero(np.asarray(case.node_pd) > 0)
    load_bus = load_bus[load_bus != sens.slack]
    agent_bus = np.sort(rng.choice(load_bus, N_AGENT, replace=False))

    clear, spec = make_clearing(case, sens, agent_bus, max_iter=60)
    load = np.maximum(np.asarray(case.node_pd, np.float64) / case.base_mva * 2.0, 0.0)
    q_load = np.asarray(case.node_qd, np.float64) / case.base_mva * 2.0
    args = (jnp.asarray(rng.uniform(20.0, 80.0, N_AGENT)),
            jnp.full((N_AGENT,), 0.05), jnp.asarray(-load),
            jnp.asarray(-q_load), jnp.asarray(load))
    return clear, spec, args


def test_float32_construction_is_refused():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        case = load_case(CASE)
        sens = build_voltage_sensitivity(case)
        with pytest.raises(RuntimeError, match="float64"):
            make_clearing(case, sens, np.array([1, 2]))
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_jit(built):
    clear, spec, args = built
    out = jax.jit(clear)(*args)

    assert out["award"].shape == (spec["n_agent"],)
    assert out["shed"].shape == (spec["n_bus"],)
    assert out["z"].shape == ()
    for name in ("award", "shed", "z", "mu", "dual_residual"):
        assert out[name].dtype == jnp.float64, name
        assert jnp.isfinite(out[name]).all(), name


def test_requirement_travels_with_the_award(built):
    """§4's signals come out of the same call, since both read one baseline."""
    clear, _, args = built
    out = jax.jit(clear)(*args)
    for key in ("req_v", "req_th", "req_v_max", "req_th_count", "v_sq", "flow"):
        assert key in out


def test_pytree_structure_is_independent_of_the_data(built):
    clear, _, args = built
    price, qty, p_inj, q_inj, load = args
    quiet = jax.tree.structure(clear(price, qty, 0.05 * p_inj, 0.05 * q_inj, load))
    stressed = jax.tree.structure(clear(price, qty, 2.0 * p_inj, 2.0 * q_inj, load))
    assert quiet == stressed


def test_vmap_over_parallel_environments(built):
    clear, spec, args = built
    price, qty, p_inj, q_inj, load = args
    prices = price[None, :] * jnp.linspace(0.5, 1.5, BATCH)[:, None]

    batched = jax.jit(jax.vmap(clear, in_axes=(0, None, None, None, None)))
    out = batched(prices, qty, p_inj, q_inj, load)

    assert out["award"].shape == (BATCH, spec["n_agent"])
    assert jnp.isfinite(out["award"]).all()
    assert float(jnp.max(out["mu"])) < 1e-5


def test_identical_lanes_are_bit_identical(built):
    """A fixed trip count makes the batch cost independent of its content."""
    clear, _, args = built
    price, qty, p_inj, q_inj, load = args
    repeated = jnp.broadcast_to(price, (BATCH, price.shape[0]))

    out = jax.jit(jax.vmap(clear, in_axes=(0, None, None, None, None)))(
        repeated, qty, p_inj, q_inj, load)

    award = np.asarray(out["award"])
    np.testing.assert_array_equal(award, np.broadcast_to(award[0], award.shape))


def test_scan_over_an_episode(built):
    clear, spec, args = built
    price, qty, p_inj, q_inj, load = args
    profile = jnp.linspace(1.0, 2.0, T)[:, None] * p_inj[None, :]

    def period(total, p):
        out = clear(price, qty, p, 0.3 * p, load)
        return total + out["z"], out["award"]

    total, awards = jax.jit(
        lambda xs: lax.scan(period, jnp.asarray(0.0), xs))(profile)

    assert awards.shape == (T, spec["n_agent"])
    assert jnp.isfinite(total) and float(total) > 0.0


def test_vmap_of_scan(built):
    """The rollout shape of §12, at a batch small enough for a dense solve."""
    clear, spec, args = built
    price, qty, p_inj, q_inj, load = args
    profile = (jnp.linspace(1.0, 2.0, T)[None, :, None] * p_inj[None, None, :]
               * jnp.linspace(0.9, 1.1, 4)[:, None, None])

    def rollout(xs):
        return lax.scan(lambda c, p: (c, clear(price, qty, p, 0.3 * p, load)["z"]),
                        jnp.asarray(0.0), xs)[1]

    z = jax.jit(jax.vmap(rollout))(profile)
    assert z.shape == (4, T) and jnp.isfinite(z).all()
