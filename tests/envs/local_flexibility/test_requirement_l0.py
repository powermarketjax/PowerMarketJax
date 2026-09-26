"""L0 JAX contract for the published requirement (§4).

`publish` runs inside the compiled `step`, so unlike the setup-time assembly of
`sensitivity` it has to satisfy the contract itself: `jit`, `vmap` over
parallel environments, `lax.scan` over the periods of an episode, and a pytree
whose structure does not depend on the data.

The construction-time float64 check is exercised here too.  It is not a
formality: the requirement is the right-hand side the clearing problem is
assembled against, and under float32 it stays finite and plausible while being
wrong, which is the failure mode already recorded for the day-ahead duals.

x64 is set by an autouse fixture that restores the previous value rather than
at import time, because other modules in this repository switch it off globally.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from powermarketjax.case import load_case
from powermarketjax.envs.local_flexibility import (build_voltage_sensitivity,
                                                   make_requirement)

CASE = "33bw"
T = 6
BATCH = 8
KEYS = {"v_sq", "flow", "req_v", "req_th",
        "req_v_max", "req_v_count", "req_th_max", "req_th_count"}


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
    load = np.asarray(case.node_pd, np.float64) / case.base_mva
    return case, sens, make_requirement(case, sens), load


def test_float32_construction_is_refused():
    """The guard fires at construction, where it can still be acted on."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        case = load_case(CASE)
        sens = build_voltage_sensitivity(case)
        with pytest.raises(RuntimeError, match="float64"):
            make_requirement(case, sens)
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_outputs_are_float64_and_finite(built):
    _, _, publish, load = built
    out = publish(-load * 2.0, -load * 0.6)

    assert set(out) == KEYS
    for name, value in out.items():
        if name.endswith("_count"):
            assert jnp.issubdtype(value.dtype, jnp.integer), name
            continue
        assert value.dtype == jnp.float64, name
        assert jnp.isfinite(value).all(), name


def test_jit(built):
    case, sens, publish, load = built
    compiled = jax.jit(publish)
    out = compiled(-load * 2.0, -load * 0.6)
    reference = publish(-load * 2.0, -load * 0.6)

    assert out["v_sq"].shape == (sens.n_bus,)
    assert out["flow"].shape == (sens.n_line,)
    assert out["req_v"].shape == (sens.n_bus,)
    assert out["req_th"].shape == (sens.n_line,)
    for key in KEYS:
        np.testing.assert_allclose(np.asarray(out[key]), np.asarray(reference[key]),
                                   rtol=0.0, atol=0.0)


def test_pytree_structure_is_independent_of_the_data(built):
    """A period with no requirement and one with a deep one must agree in structure.

    `lax.scan` carries one structure across every period, so a publisher whose
    output shape depended on how many indices were violated could not be
    scanned at all.
    """
    _, _, publish, load = built
    quiet = jax.tree.structure(publish(-load * 0.1, np.zeros_like(load)))
    stressed = jax.tree.structure(publish(-load * 4.0, np.zeros_like(load)))
    assert quiet == stressed


def test_vmap_over_parallel_environments(built):
    _, sens, publish, load = built
    scales = jnp.linspace(0.5, 3.0, BATCH)[:, None]
    p_inj = -jnp.asarray(load)[None, :] * scales

    out = jax.jit(jax.vmap(publish))(p_inj, 0.3 * p_inj)

    assert out["req_v"].shape == (BATCH, sens.n_bus)
    assert out["req_th"].shape == (BATCH, sens.n_line)
    assert out["req_v_count"].shape == (BATCH,)
    # the requirement grows with load, so the lanes are genuinely separate
    counts = np.asarray(out["req_v_count"])
    assert counts[0] < counts[-1]


def test_scan_over_an_episode(built):
    _, sens, publish, load = built
    profile = -jnp.asarray(load)[None, :] * jnp.linspace(1.0, 3.0, T)[:, None]

    def period(worst, p_inj):
        out = publish(p_inj, 0.3 * p_inj)
        return jnp.maximum(worst, out["req_v_max"]), out["req_v_count"]

    worst, counts = jax.jit(
        lambda xs: lax.scan(period, jnp.asarray(0.0), xs))(profile)

    assert counts.shape == (T,) and jnp.isfinite(worst)
    assert int(counts[0]) <= int(counts[-1])


def test_vmap_of_scan_runs_many_environments(built):
    """The shape the rollout of §12 runs in."""
    _, _, publish, load = built
    profile = (-jnp.asarray(load)[None, None, :]
               * jnp.linspace(1.0, 3.0, T)[None, :, None]
               * jnp.linspace(0.6, 1.4, 128)[:, None, None])

    def rollout(xs):
        return lax.scan(lambda c, p: (c, publish(p, 0.3 * p)["req_th_max"]),
                        jnp.asarray(0.0), xs)[1]

    worst = jax.jit(jax.vmap(rollout))(profile)
    assert worst.shape == (128, T) and jnp.isfinite(worst).all()
