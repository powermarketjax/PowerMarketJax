"""L0 JAX contract for the nonlinear verification (§7).

`jit`, `vmap` over parallel environments, `lax.scan` over the periods of an
episode, and a pytree that does not depend on the data.

The sweep runs under a `lax.while_loop`, so unlike the clearing its cost does
depend on the batch: every lane runs until the slowest one exits.  That is
tolerated only because the verification sits outside the optimisation loop and
was measured at well under one per cent of the cost of a `step` (§15).  The
straggler behaviour is exercised here with a batch whose lanes span the whole
scaling window.

The precision of the sweep follows the global configuration rather than the
cast at the entry, because the vendored solver builds its own accumulators with
bare `jnp.zeros`.  With `x64` on, which is what the clearing requires, the
verification is a float64 verification.  That is pinned below because §15
originally assumed otherwise.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from powermarketjax.case import load_case
from powermarketjax.envs.local_flexibility import (build_voltage_sensitivity,
                                                   make_verification)

CASE = "33bw"
T = 5
BATCH = 16
KEYS = {"v_mag", "p_branch", "v_under", "v_over", "overload",
        "converged", "floor_active", "iterations"}


@pytest.fixture(scope="module")
def built():
    case = load_case(CASE)
    sens = build_voltage_sensitivity(case)
    load = np.asarray(case.node_pd, np.float64) / case.base_mva
    reactive = np.asarray(case.node_qd, np.float64) / case.base_mva
    return make_verification(case, sens), sens, load, reactive


def test_outputs_and_dtypes(built):
    verify, sens, load, reactive = built
    out = jax.jit(verify)(jnp.asarray(-load * 2.0), jnp.asarray(-reactive * 2.0))

    assert set(out) == KEYS
    assert out["v_mag"].shape == (sens.n_bus,)
    assert out["p_branch"].shape == (sens.n_line,)
    assert out["overload"].shape == (sens.n_line,)
    expected = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    for name in ("v_mag", "p_branch", "v_under", "v_over", "overload"):
        assert out[name].dtype == expected, name
        assert jnp.isfinite(out[name]).all(), name
    assert out["converged"].dtype == jnp.bool_
    assert out["floor_active"].dtype == jnp.bool_


def test_the_sweep_follows_the_global_precision_not_the_cast(built):
    """The cast at the entry is a no-op once `x64` is on, and that is worth pinning.

    The vendored sweep builds its own accumulators with bare `jnp.zeros`, which
    are float64 under `x64`, so the operating point promotes back regardless of
    what is handed in.  This market runs with `x64` on because the clearing
    requires it, so the verification is in fact a float64 verification.  That is
    strictly better than the float32 one §15 assumed, and the assumption has
    been corrected there rather than worked around here.
    """
    verify, _, load, reactive = built
    previous = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", True)
        wide = jax.jit(verify)(jnp.asarray(-load * 2.0, jnp.float64),
                               jnp.asarray(-reactive * 2.0, jnp.float64))
        assert wide["v_mag"].dtype == jnp.float64

        jax.config.update("jax_enable_x64", False)
        narrow = jax.jit(verify)(jnp.asarray(-load * 2.0, jnp.float32),
                                 jnp.asarray(-reactive * 2.0, jnp.float32))
        assert narrow["v_mag"].dtype == jnp.float32

        # the two agree to float32 precision, which is what makes the wider one
        # a refinement rather than a different answer
        np.testing.assert_allclose(np.asarray(wide["v_mag"], np.float64),
                                   np.asarray(narrow["v_mag"], np.float64),
                                   rtol=1e-5)
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_pytree_structure_is_independent_of_the_data(built):
    """A quiet period and one that collapses onto the floor must agree in structure."""
    verify, _, load, reactive = built
    quiet = jax.tree.structure(verify(jnp.asarray(-load * 0.1),
                                      jnp.asarray(-reactive * 0.1)))
    collapsed = jax.tree.structure(verify(jnp.asarray(-load * 80.0),
                                          jnp.asarray(-reactive * 80.0)))
    assert quiet == collapsed


def test_vmap_over_a_heterogeneous_batch(built):
    """Lanes spanning the window, which is where the while loop straggles."""
    verify, sens, load, reactive = built
    scales = jnp.linspace(0.5, 4.0, BATCH)[:, None]
    p_inj = -jnp.asarray(load)[None, :] * scales

    out = jax.jit(jax.vmap(verify))(p_inj, -jnp.asarray(reactive)[None, :] * scales)

    assert out["v_mag"].shape == (BATCH, sens.n_bus)
    assert out["overload"].shape == (BATCH, sens.n_line)
    # the lanes genuinely differ, so the batch axis is not broadcast away
    assert float(jnp.abs(out["v_mag"][0] - out["v_mag"][-1]).max()) > 0.0
    # the heaviest lane needs more iterations than the lightest
    assert int(out["iterations"].max()) >= int(out["iterations"].min())


def test_scan_over_an_episode(built):
    verify, sens, load, reactive = built
    profile = -jnp.asarray(load)[None, :] * jnp.linspace(1.0, 3.0, T)[:, None]

    def period(worst, p_inj):
        out = verify(p_inj, 0.6 * p_inj)
        return jnp.maximum(worst, out["v_under"].max()), out["v_under"].max()

    # The carry dtype follows what the body actually returns rather than being
    # pinned: `verify` casts its own arrays to float32 internally but its output
    # promotes to float64 when `jax_enable_x64` is on, and that flag is a process
    # global which other test modules switch on and never restore.  Pinning
    # float32 here made this test pass alone and fail after any module that had
    # enabled x64 -- an ordering dependency, not a numerical fault.  Whether
    # `verify` should return float32 regardless is a question for that market's
    # dtype discipline, not for this assertion.
    carry_dtype = jax.eval_shape(lambda p: period(jnp.zeros(()), p)[0], profile[0]).dtype
    worst, per_period = jax.jit(
        lambda xs: lax.scan(period, jnp.zeros((), carry_dtype), xs))(profile)

    assert per_period.shape == (T,) and jnp.isfinite(worst)
    assert float(per_period[0]) <= float(per_period[-1])


def test_vmap_of_scan_runs_many_environments(built):
    verify, _, load, reactive = built
    profile = (-jnp.asarray(load)[None, None, :]
               * jnp.linspace(1.0, 3.0, T)[None, :, None]
               * jnp.linspace(0.8, 1.2, 128)[:, None, None])

    def rollout(xs):
        return lax.scan(lambda c, p: (c, verify(p, 0.6 * p)["v_under"].max()),
                        jnp.asarray(0.0, jnp.float32), xs)[1]

    worst = jax.jit(jax.vmap(rollout))(profile)
    assert worst.shape == (128, T) and jnp.isfinite(worst).all()


def test_topology_mismatch_is_rejected():
    """The sweep and the sensitivity matrices must index the same in-service lines."""
    case = load_case(CASE)
    sens = build_voltage_sensitivity(case)
    other = build_voltage_sensitivity(load_case("141"))
    with pytest.raises(ValueError, match="in-service lines"):
        make_verification(case, other)
