"""L2 numerical equivalence: JAX implementation against the numpy reference.

The reference runs the *same* algorithm by a deliberately
different route (explicit dense constraint matrix, bounds folded into an
explicit G, plain dense KKT factorisation against the implementation's
vectorised assembly, matrix-free products and block-tridiagonal sweep).  So a
disagreement here is an implementation error, not a modelling one.

§16 requires two things of this layer, both learned from this repository's
existing equivalence tests: the reference must be independently written rather
than the implementation's own output frozen, and **the tolerance must be derived
from a measured error with the derivation recorded here**.

Measured over the five configurations below (CPU, float64), worst case:

    lmp             6.2e-5 absolute, 1.8e-7 relative
    award           1.6e-10 MW
    shed, per bus   **2.1 MW**
    shed, per period total   ~1e-8 MW

The two routes converge to the same optimum by different floating-point paths,
so they agree to roughly the interior point method's own dual accuracy rather
than to round-off.  Budgets below are set an order of magnitude above the
measured values, and must NOT be tightened below the ~4e-3 $/MWh GPU
cross-process drift measured on GPU if this is ever run there.

**Per-bus shed is deliberately not compared.**  Its 2.1 MW disagreement is not
an error: when several buses shed simultaneously they all price at VOLL, so the
split of a given total between them is not unique and both solutions are
optimal.  Only the per-period total is determined, and that agrees to 1e-8.
Asserting on the split would be asserting on an arbitrary choice -- the kind of
over-tight equivalence check §16 warns against.

The reference is O(n^3) dense, so the horizon here is short on purpose; T=24
belongs in `tools/lp_bench`, not in CI.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.day_ahead import make_clearing, segment_costs

from . import reference

#: Derived from the measurements in the module docstring, one order of magnitude
#: of headroom.  Not round-off: two different factorisation paths through the
#: same iteration.
ATOL_LMP = 1e-4      # measured 6.2e-5
RTOL_LMP = 1e-6      # measured 1.8e-7
ATOL_MW = 1e-6       # measured 1.6e-10 on award


@pytest.fixture(scope="module", autouse=True)
def x64():
    prev, prev_mm = jax.config.jax_enable_x64, jax.config.jax_default_matmul_precision
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    yield
    jax.config.update("jax_enable_x64", prev)
    jax.config.update("jax_default_matmul_precision", prev_mm)


@pytest.fixture(scope="module")
def case(x64):
    return load_case("29gb")


def _compare(case, T, K, cap_scale, ramp_scale, demand, off_units=0, markup=1.0):
    _, cost = segment_costs(case, K)
    # (n_units, K, T): offers are per delivery period (§5).  `markup` may be a
    # scalar or one factor per period, and a period-varying one is the only
    # kind that can detect a wrong axis order in the cost vector.
    offer = cost[:, :, None] * np.broadcast_to(np.asarray(markup, np.float64), (T,))
    n_u = len(np.asarray(case.unit_p_min))
    u = np.ones((n_u, T))
    if off_units:
        u[:off_units] = 0.0
    p_init = np.asarray(case.unit_p_min, np.float64) * u[:, 0]
    dem = np.full(T, demand)

    clear, _ = make_clearing(case, T, n_segments=K, cap_scale=cap_scale,
                             ramp_scale=ramp_scale)
    got = jax.jit(clear)(jnp.asarray(offer), jnp.asarray(u), jnp.asarray(dem),
                         jnp.asarray(p_init))
    want = reference.clear(case, offer, u, dem, p_init, cap_scale, ramp_scale)

    assert float(got["mu"]) < 1e-8, "implementation did not converge"
    assert want["mu"] < 1e-8, "reference did not converge"
    return got, want


#: `cap_scale` and `ramp_scale` are the third and fourth positional fields.  They
#: are the adopted scenario (0.6 / 1.0 since 2026-08-17) everywhere except
#: where a row says otherwise, and the two rows that say otherwise say so on
#: purpose: `uncongested` needs a network that never binds and `shedding` needs
#: the ramp out of the way so that the shortfall it is named for is the demand
#: level and not the ramp.  Written positionally these values match no search for
#: `cap_scale=`, which is how the scenario sweep of 2026-08-17 first missed them;
#: they are listed in the module docstring's scenario line for that reason.
CONFIGS = [
    pytest.param(2, 1, 0.6, 1.0, 33466.0, 0, 1.0, id="base"),
    pytest.param(2, 1, 1.0, 2.0, 33466.0, 0, 1.0, id="uncongested"),
    pytest.param(3, 2, 0.6, 1.0, 33466.0, 0, 1.3, id="segmented-marked-up"),
    pytest.param(2, 1, 0.6, 1.0, 33466.0, 20, 1.0, id="de-committed"),
    pytest.param(2, 1, 0.6, 2.0, 95000.0, 0, 1.0, id="shedding"),
    pytest.param(3, 2, 0.6, 1.0, 33466.0, 0, np.array([1.0, 1.6, 0.8]),
                 id="period-varying-offer"),
]


@pytest.mark.parametrize("T,K,cap_scale,ramp_scale,demand,off_units,markup", CONFIGS)
def test_matches_numpy_reference(case, T, K, cap_scale, ramp_scale, demand,
                                 off_units, markup):
    got, want = _compare(case, T, K, cap_scale, ramp_scale, demand, off_units, markup)
    np.testing.assert_allclose(np.asarray(got["lmp"]), want["lmp"],
                               rtol=RTOL_LMP, atol=ATOL_LMP, err_msg="lmp")
    np.testing.assert_allclose(np.asarray(got["award"]), want["award"],
                               rtol=1e-6, atol=ATOL_MW, err_msg="award")
    # total only; the split between simultaneously-shedding buses is degenerate
    np.testing.assert_allclose(np.asarray(got["shed"]).sum(1), want["shed"].sum(1),
                               rtol=1e-6, atol=1e-4, err_msg="shed total")


def test_reference_is_not_a_transcription(case):
    """Guards the property the L2 comparison relies on: the reference builds an explicit
    constraint matrix, which the implementation never does.  If someone later
    'simplifies' the reference by calling the implementation, this fails."""
    _, cost = segment_costs(case, 1)
    n_u = len(np.asarray(case.unit_p_min))
    lp = reference.build_lp(case, np.broadcast_to(cost[:, :, None], cost.shape + (2,)),
                            np.ones((n_u, 2)), np.full(2, 33466.0),
                            np.asarray(case.unit_p_min, np.float64),
                            cap_scale=0.6, ramp_scale=1.0)
    assert lp["A_ub"].ndim == 2 and lp["A_ub"].shape[0] > 0
    assert np.count_nonzero(lp["A_ub"]) > 0
