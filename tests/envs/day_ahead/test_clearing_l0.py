"""L0 JAX contract for the day-ahead clearing operator.

jit, vmap, pytree stability, no NaN, and callability from inside lax.scan.
Domain correctness is L1 (test_clearing_l1.py).

These need float64: the solver runs in float64 throughout, and in
float32 the IPM diverges outright (LMPs of order 1e248 were observed while
`mu` still looked converged).

x64 is set by an autouse fixture rather than at import time because **other
modules in this suite turn it off globally** -- `test_bfs_3phase_power_flow`
and `test_bfs_3phase_eq` both call `jax.config.update("jax_enable_x64", False)`
inside their fixtures, and whichever module runs last wins.  A module-level
update here would be silently undone by them and would equally silently break
them.  The fixture restores the previous value on the way out.
"""
import jax
import jax.numpy as jnp
import jax.tree_util as tu
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.day_ahead import make_clearing, segment_costs

T = 4          # small horizon: L0 checks the contract, not the economics
K = 2
CAP_SCALE = 0.6        # adopted scenario (2026-08-17)
RAMP_SCALE = 1.0       # adopted scenario (2026-08-17); registered rates undiscounted
#: The scenario ramp factor until 2026-08-17, kept as a run point because one
#: test below needs a *surplus* that the ramp-down limit cannot unwind, and at
#: RAMP_SCALE it can: a unit may shed 0.7 p_max in one period, so starting the
#: whole fleet at p_max is no longer an impossible input.  The test asserts the
#: infeasibility is present rather than assuming this constant delivers it.
RAMP_SURPLUS = 0.25
DEMAND = 33466.0


@pytest.fixture(scope="module", autouse=True)
def x64():
    """float64 for this module only; see the module docstring."""
    prev = jax.config.jax_enable_x64
    prev_mm = jax.config.jax_default_matmul_precision
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    yield
    jax.config.update("jax_enable_x64", prev)
    jax.config.update("jax_default_matmul_precision", prev_mm)


@pytest.fixture(scope="module")
def case(x64):
    return load_case("29gb")


@pytest.fixture(scope="module")
def built(case):
    clear, spec = make_clearing(case, T, n_segments=K, cap_scale=CAP_SCALE,
                                ramp_scale=RAMP_SCALE)
    _, cost = segment_costs(case, K)
    n_u = spec["n_units"]
    args = (jnp.broadcast_to(jnp.asarray(cost)[:, :, None], (n_u, K, T)),
            jnp.ones((n_u, T)),
            jnp.full((T,), DEMAND),
            jnp.asarray(np.asarray(case.unit_p_min, np.float64)))
    return clear, spec, args


def test_shapes(built):
    clear, spec, args = built
    out = clear(*args)
    assert out["award"].shape == (spec["n_units"], T)
    assert out["lmp"].shape == (T, spec["n_buses"])
    assert out["shed"].shape == (T, spec["n_buses"])
    assert out["mu"].shape == ()
    assert out["dual_residual"].shape == ()
    # the three duals of §7, which §8's identity is assembled from
    assert out["line_dual_up"].shape == (T, spec["n_l"])
    assert out["line_dual_dn"].shape == (T, spec["n_l"])
    assert out["shed_dual"].shape == (T, spec["n_buses"])


#: Tolerances for jit-vs-eager, derived from measurement rather than guessed
#: (§16 requires the derivation be recorded).  XLA fuses and reorders under jit,
#: so agreement is not bit-exact.  Measured on this configuration: `lmp` differs
#: by 4.6e-11 relative, the primal quantities `award`/`shed` by 2.3e-7 relative
#: -- an interior point method converges its primal more slowly than its duals.
#: Budgets are set an order of magnitude above the measured values.
JIT_RTOL = {"lmp": 1e-9, "mu": 1e-3, "dual_residual": 1e-3}
JIT_RTOL_PRIMAL = 1e-5


def test_jit_matches_eager(built):
    clear, _, args = built
    a, b = clear(*args), jax.jit(clear)(*args)
    for k in a:
        np.testing.assert_allclose(
            np.asarray(a[k]), np.asarray(b[k]),
            rtol=JIT_RTOL.get(k, JIT_RTOL_PRIMAL), atol=1e-6, err_msg=k)


def test_pytree_structure_stable(built):
    """Structure must not depend on the values -- a de-committed fleet, a
    congested run and a shedding run all have to give the same treedef."""
    clear, spec, args = built
    offer, u, demand, p_init = args
    variants = [
        args,
        (offer, u.at[: spec["n_units"] // 2].set(0.0), demand, p_init),
        (offer, u, demand * 3.0, p_init),            # forces shed
    ]
    defs = {tu.tree_structure(clear(*v)) for v in variants}
    assert len(defs) == 1


def test_vmap_over_environments(built):
    """Offers and commitment vary per lane; the structure is shared."""
    clear, spec, args = built
    offer, u, demand, p_init = args
    B = 4
    offers = jnp.stack([offer * (1.0 + 0.1 * i) for i in range(B)])
    us = jnp.stack([u.at[: i * 5].set(0.0) for i in range(B)])
    # p_init has to match each lane's commitment: a unit that is off must already
    # be at zero, or the ramp-down limit cannot get it there in one period and
    # the LP is infeasible.  Not a solver limitation -- the input is impossible.
    p_inits = jnp.stack([p_init * us[i, :, 0] for i in range(B)])
    f = jax.jit(jax.vmap(clear, in_axes=(0, 0, None, 0)))
    out = f(offers, us, demand, p_inits)
    assert out["lmp"].shape == (B, T, spec["n_buses"])
    assert out["shed_dual"].shape == (B, T, spec["n_buses"])
    assert out["line_dual_up"].shape == (B, T, spec["n_l"])
    assert float(jnp.max(out["mu"])) < 1e-8
    # each lane must equal its own solo solve
    for i in range(B):
        solo = clear(offers[i], us[i], demand, p_inits[i])
        np.testing.assert_allclose(np.asarray(out["lmp"][i]),
                                   np.asarray(solo["lmp"]), rtol=1e-7, atol=1e-6)


def test_vmap_identical_inputs_give_identical_lanes(built):
    """The test MPAX failed: replicating one problem must not change its answer.

    MPAX returned batch-size-dependent, non-converged duals here, which is why
    it was excluded.  A fixed trip count
    makes the lanes bit-identical.
    """
    clear, _, args = built
    B = 8
    f = jax.jit(jax.vmap(clear, in_axes=(0, None, None, None)))
    out = f(jnp.stack([args[0]] * B), *args[1:])
    for i in range(1, B):
        np.testing.assert_array_equal(np.asarray(out["lmp"][i]),
                                      np.asarray(out["lmp"][0]))


def test_infeasible_input_shows_up_in_mu(case):
    """An impossible input must not look converged.

    `mu` is the contract the operator puts on the caller, so it has to separate the
    two cases by a wide margin.  The impossible input here is a surplus: every
    unit starts the day at `p_max`, and the ramp-down limit holds first-period
    generation above demand.  Shed offsets a shortfall and never a surplus (§13),
    so the LP has no solution and mu goes to ~1e8 or beyond against ~1e-11 when
    converged.  The returned LMPs are garbage, so `mu` is the only thing standing
    between that and a reward signal.

    A de-committed unit handed a non-zero `p_init` used to serve as the
    impossible input.  It no longer is: (RMP) carries a shut-down allowance of
    `p_min` (§6.3), so switching a unit off from `p_min` is exactly what that
    allowance permits.

    This test builds its own operator at RAMP_SURPLUS instead of using `built`.
    At the adopted ramp_scale = 1.0 the fleet can unwind from p_max fast enough
    that the surplus is *feasible* -- measured mu 1.2e-11, i.e. the impossible
    input became an ordinary one and this gate would have been asserting nothing
    about `mu`.  The threshold is not lowered; the run point is constructed and
    the infeasibility is asserted up front.
    """
    clear_s, spec_s = make_clearing(case, T, n_segments=K, cap_scale=CAP_SCALE,
                                    ramp_scale=RAMP_SURPLUS)
    p_min = np.asarray(spec_s["p_min"], np.float64)
    p_max = np.asarray(spec_s["p_max"], np.float64)
    ramp_dn = np.asarray(case.unit_ramp_down, np.float64) * p_max * RAMP_SURPLUS
    # first-period generation cannot go below this if every unit starts at p_max
    floor = np.maximum(p_min, p_max - ramp_dn).sum()
    assert floor > DEMAND, (
        f"the surplus is not infeasible at ramp_scale={RAMP_SURPLUS}: the "
        f"first-period floor is {floor:.0f} MW against a demand of {DEMAND:.0f} "
        "MW, so this test would be checking mu on a feasible LP")

    _, cost = segment_costs(case, K)
    n_u = spec_s["n_units"]
    offer = jnp.broadcast_to(jnp.asarray(cost)[:, :, None], (n_u, K, T))
    u = jnp.ones((n_u, T))
    demand = jnp.full((T,), DEMAND)
    out = clear_s(offer, u, demand, jnp.asarray(p_max))
    assert float(out["mu"]) > 1e8
    assert float(clear_s(offer, u, demand, jnp.asarray(p_min))["mu"]) < 1e-8


def test_no_nan_anywhere(built):
    clear, spec, args = built
    offer, u, demand, p_init = args
    for v in [args,
              (offer, jnp.zeros_like(u), demand, jnp.zeros_like(p_init)),
              (offer, u, demand * 5.0, p_init)]:              # deep scarcity
        out = clear(*v)
        for k, arr in out.items():
            assert np.isfinite(np.asarray(arr)).all(), f"non-finite in {k}"


def test_runs_inside_scan(built):
    """The rollout driver is lax.scan, so the operator must compose with it
    rather than being driven by a Python loop."""
    clear, _, args = built
    offer, u, demand, p_init = args

    def body(carry, _):
        out = clear(offer, u, demand, carry)
        return out["award"][:, -1], out["lmp"][-1]

    final, stacked = jax.jit(lambda p: jax.lax.scan(body, p, None, length=3))(p_init)
    assert stacked.shape == (3, args[0].shape[0] * 0 + clear(*args)["lmp"].shape[1])
    assert np.isfinite(np.asarray(final)).all()


def test_converges(built):
    """mu is the convergence check the caller is responsible for."""
    clear, _, args = built
    assert float(clear(*args)["mu"]) < 1e-8
