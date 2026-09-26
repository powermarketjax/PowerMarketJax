"""L0 JAX contract for the voltage sensitivity matrices (§3.2).

`build_voltage_sensitivity` is setup-time numpy and is deliberately not
jittable: it walks a tree in Python and raises on a malformed feeder.  What has
to satisfy the JAX contract is its **output**, since §6 reads these matrices
inside the compiled clearing problem.  This module therefore checks the
expressions of §3.2 under `jit`, under `vmap` over parallel environments, and
under `lax.scan` over the periods of an episode.

It also pins the dtype boundary.  The matrices are float64 on the numpy side,
and they silently become float32 the moment they enter JAX with `x64`
disabled.  That is the same trap the day-ahead clearing guards against with a
construction-time check, and the clearing operator of this market
will need the same guard, so the behaviour is recorded here rather than
assumed.

No global configuration is mutated here.  Switching `jax_enable_x64` inside a
test leaks into every module that runs after it, which is how the day-ahead
operator was once handed float32 duals.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from powermarketjax.case import load_case
from powermarketjax.envs.local_flexibility import build_voltage_sensitivity

CASE = "33bw"
T = 6
BATCH = 8


@pytest.fixture(scope="module")
def sens():
    return build_voltage_sensitivity(load_case(CASE))


def voltage_and_flow(R, X, A, p_inj, q_inj):
    """§3.2 in the form §6 consumes: squared voltage magnitudes and line flows."""
    return 1.0 + 2.0 * (R @ p_inj + X @ q_inj), -A @ p_inj


def test_outputs_are_float64_and_finite(sens):
    for name in ("R", "X", "A", "r", "x"):
        array = getattr(sens, name)
        assert array.dtype == np.float64, name
        assert np.isfinite(array).all(), name
    assert sens.line_index.dtype == np.int64


def test_construction_is_deterministic():
    """Two builds of one case agree bit for bit; the tree walk fixes no arbitrary order."""
    first, second = (build_voltage_sensitivity(load_case(CASE)) for _ in range(2))
    np.testing.assert_array_equal(first.R, second.R)
    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.A, second.A)


def test_entering_jax_downcasts_without_x64(sens):
    """The float64 assembly survives into JAX only when `x64` is on.

    Recorded rather than asserted one way: the suite runs with either setting
    depending on which modules preceded it, and the point is that the dtype
    follows the flag rather than the numpy array.
    """
    dtype = jnp.asarray(sens.R).dtype
    expected = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    assert dtype == expected


def test_jit(sens):
    R, X, A = (jnp.asarray(a) for a in (sens.R, sens.X, sens.A))
    p = jnp.asarray(np.linspace(-0.02, 0.02, sens.n_bus))
    q = 0.3 * p

    compiled = jax.jit(voltage_and_flow)
    v_sq, flow = compiled(R, X, A, p, q)

    assert v_sq.shape == (sens.n_bus,) and flow.shape == (sens.n_line,)
    assert jnp.isfinite(v_sq).all() and jnp.isfinite(flow).all()
    np.testing.assert_allclose(np.asarray(v_sq),
                               1.0 + 2.0 * (sens.R @ np.asarray(p) + sens.X @ np.asarray(q)),
                               rtol=1e-5)


def test_vmap_over_parallel_environments(sens):
    R, X, A = (jnp.asarray(a) for a in (sens.R, sens.X, sens.A))
    p = jnp.asarray(np.linspace(-0.02, 0.02, sens.n_bus))[None, :] * jnp.linspace(
        0.5, 1.5, BATCH)[:, None]

    batched = jax.jit(jax.vmap(voltage_and_flow, in_axes=(None, None, None, 0, 0)))
    v_sq, flow = batched(R, X, A, p, 0.3 * p)

    assert v_sq.shape == (BATCH, sens.n_bus) and flow.shape == (BATCH, sens.n_line)
    assert jnp.isfinite(v_sq).all()
    # lanes differ, so the batch axis is not being broadcast away
    assert float(jnp.abs(v_sq[0] - v_sq[-1]).max()) > 0.0


def test_scan_over_an_episode(sens):
    """A fixed-length episode with no Python loop, which is what §12 requires of `step`."""
    R, X, A = (jnp.asarray(a) for a in (sens.R, sens.X, sens.A))
    profile = jnp.asarray(np.linspace(-0.02, 0.02, sens.n_bus))[None, :] * jnp.linspace(
        0.8, 1.2, T)[:, None]

    def period(carry, p):
        v_sq, flow = voltage_and_flow(R, X, A, p, 0.3 * p)
        return carry + jnp.min(v_sq), (v_sq, flow)

    total, (v_sq, flow) = jax.jit(
        lambda xs: lax.scan(period, jnp.asarray(0.0, xs.dtype), xs))(profile)

    assert v_sq.shape == (T, sens.n_bus) and flow.shape == (T, sens.n_line)
    assert jnp.isfinite(total) and jnp.isfinite(v_sq).all()


def test_vmap_of_scan_runs_many_environments(sens):
    """`vmap` composed with `lax.scan`, the shape the rollout of §12 runs in."""
    R, X, A = (jnp.asarray(a) for a in (sens.R, sens.X, sens.A))
    base = jnp.asarray(np.linspace(-0.02, 0.02, sens.n_bus))
    profile = base[None, None, :] * jnp.linspace(0.8, 1.2, T)[None, :, None] \
        * jnp.linspace(0.5, 1.5, 128)[:, None, None]

    def rollout(xs):
        def period(carry, p):
            v_sq, _ = voltage_and_flow(R, X, A, p, 0.3 * p)
            return carry + jnp.min(v_sq), jnp.min(v_sq)
        return lax.scan(period, jnp.asarray(0.0, xs.dtype), xs)[1]

    worst = jax.jit(jax.vmap(rollout))(profile)
    assert worst.shape == (128, T) and jnp.isfinite(worst).all()
