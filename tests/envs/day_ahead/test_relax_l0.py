"""L0 JAX contract for the relaxed-commitment operator (§16).

`jit`; `vmap` over parallel environments; a fixed trip count, so the batch cost
does not track the batch's hardest element; and float64, which the operator
refuses to run without.

T = 4 rather than 24 for the same reason as `test_clearing_l0`: nothing here is
about the horizon, and the wider block of step 1 makes T = 24 slow on CPU.

One property is checked that step 3 has no counterpart for. Step 1's output is
consumed by a rounding, so what has to be stable under `vmap` is not `u` to
machine precision but `u > 0`, and that is asserted directly rather than through
the float it comes from.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.day_ahead.clearing import segment_costs
from powermarketjax.envs.day_ahead.relax import make_relax

T = 4
K = 1
CAP_SCALE = 0.6        # adopted scenario (2026-08-17)
RAMP_SCALE = 1.0       # adopted scenario (2026-08-17); registered rates undiscounted


@pytest.fixture(scope="module", autouse=True)
def x64():
    """Restored on the way out, which is not decoration.

    This module turned x64 on and left it on until 2026-08-24, which changes the
    numerics of every test that runs after it in the same process.  Collection
    order here is fixed, so that set was the same every run: the leak was a
    silent and repeatable change of precision, not a source of flakiness.
    """
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture(scope="module")
def setup(x64):
    case = load_case("29gb")
    relax, spec = make_relax(case, T, n_segments=K, cap_scale=CAP_SCALE,
                             ramp_scale=RAMP_SCALE)
    n_u = int(case.n_units)
    _, cost = segment_costs(case, K)
    offer = jnp.asarray(np.repeat(cost[:, :, None], T, axis=2))
    demand = jnp.full((T,), 27_542.0)
    # a boundary the operator itself could have produced: every unit off, so the
    # (MU)/(MD) initial conditions are the day-0 ones and no unit is mid-window
    p_init = jnp.zeros(n_u)
    u_prev = jnp.zeros(n_u)
    up_time = jnp.zeros(n_u)
    down_time = jnp.asarray(np.maximum(np.asarray(case.unit_min_down_time), 1),
                            dtype=jnp.float64)
    return relax, spec, (offer, demand, p_init, u_prev, up_time, down_time)


def test_jit_and_shapes(setup):
    relax, spec, args = setup
    out = jax.jit(relax)(*args)
    assert out["u"].shape == (spec["n_units"], T)
    assert out["p"].shape == (spec["n_units"], T)
    assert out["shed"].shape == (T, spec["n_buses"])
    for k in ("u", "p", "shed", "obj", "mu", "dual_residual"):
        assert jnp.isfinite(out[k]).all(), k


def test_u_within_bounds(setup):
    relax, _, args = setup
    u = jax.jit(relax)(*args)["u"]
    assert float(u.min()) >= 0.0 and float(u.max()) <= 1.0


def test_converges(setup):
    """`mu` is the gate: the clearing's discipline applies here too, and the
    calibration measured a hard transition rather than a gradual one -- at 60
    Newton steps `mu` is 1e3 and the rounded commitment is wrong by two thirds of
    its cells, at 120 it is 1e-11 and exact."""
    relax, _, args = setup
    assert float(jax.jit(relax)(*args)["mu"]) < 1e-6


def test_vmap_matches_serial(setup):
    relax, _, args = setup
    offer, demand, p_init, u_prev, up_time, down_time = args
    N = 3
    dem = demand[None, :] * (1.0 + 0.01 * jnp.arange(N)[:, None])
    batched = jax.jit(jax.vmap(relax, in_axes=(None, 0, None, None, None, None)))(
        offer, dem, p_init, u_prev, up_time, down_time)
    single = jax.jit(relax)
    for i in range(N):
        one = single(offer, dem[i], p_init, u_prev, up_time, down_time)
        # the rounding is what step 2 consumes, so it is the thing that has to
        # agree; the float it comes from is checked too, but loosely
        assert np.array_equal(np.asarray(batched["u"][i] > 0.0),
                              np.asarray(one["u"] > 0.0))
        np.testing.assert_allclose(batched["u"][i], one["u"], atol=1e-9)


def test_requires_float64():
    case = load_case("29gb")
    # restored to what was there rather than forced back to True: forcing the
    # literal is what made this module a leak even though it looked balanced
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="float64"):
            make_relax(case, T, n_segments=K, cap_scale=CAP_SCALE,
                       ramp_scale=RAMP_SCALE)
    finally:
        jax.config.update("jax_enable_x64", previous)
