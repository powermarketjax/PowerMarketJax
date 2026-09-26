"""L2 numerical equivalence for the relaxed-commitment operator.

The reference is HiGHS through `tools/commitment/precommit.py`, which builds the
same linear program from a different description: sparse, ordered by variable
type rather than block per period, and with the shut-down indicator carried as a
variable of its own under (LOG) rather than eliminated.  Two descriptions of one
problem is the point -- a transcription of the same assembly would reproduce a
sign error in it.

What is compared is not only the objective.  The operator exists to feed step 2,
so the **rounded** commitment is compared cell by cell, and the objective of the
fixed-commitment SCED that rounding leads to is compared as well: an operator
that agreed on the objective while rounding differently would be useless here and
this asserts against exactly that.

The (MU)/(MD) rows are dropped from the reference too, since that is the problem
this operator solves.
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import linprog

from powermarketjax.case import load_case
from powermarketjax.envs.day_ahead.clearing import segment_costs
from powermarketjax.envs.day_ahead.relax import (ROUND_EPS, make_relax,
                                                 round_commitment)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from commitment import precommit as PC          # noqa: E402

T = 4
K = 1
CAP_SCALE = 0.6        # adopted scenario (2026-08-17)
RAMP_SCALE = 1.0       # adopted scenario (2026-08-17); registered rates undiscounted
KW = dict(cap_scale=CAP_SCALE, ramp_scale=RAMP_SCALE, K=K, delta_h=1.0)


def _drop_mu_md(lp):
    """(MU)/(MD) are the only rows touching no `g` and no `s` column.

    Identified structurally rather than by row arithmetic, so the test does not
    depend on the order `precommit.build` happens to assemble its rows in.
    """
    A = lp["A_ub"].tocsr()[:, :lp["off_u"]].tocsr()
    keep = np.diff(A.indptr) > 0
    out = dict(lp)
    out["A_ub"] = lp["A_ub"].tocsr()[keep]
    out["b_ub"] = lp["b_ub"][keep]
    return out


@pytest.fixture(scope="module", autouse=True)
def x64():
    """Restored on the way out; see `test_relax_l0.py` for what it cost.

    This module turned x64 on and left it on until 2026-08-24.
    """
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture(scope="module")
def case(x64):
    return load_case("29gb")


@pytest.mark.parametrize("total_demand", [27_542.0, 33_466.0])
def test_matches_highs(case, total_demand):
    demand = np.full(T, total_demand)
    boundary = PC.first_boundary(case, float(demand[0]), **KW)

    lp = _drop_mu_md(PC.build(case, demand, **boundary, **KW))
    r = linprog(lp["c"], A_ub=lp["A_ub"], b_ub=lp["b_ub"], A_eq=lp["A_eq"],
                b_eq=lp["b_eq"], bounds=np.stack([lp["lo"], lp["hi"]], 1),
                method="highs")
    assert r.status == 0, r.message
    ref = PC.unpack(lp, np.asarray(r.x))

    relax, _ = make_relax(case, T, n_segments=K, cap_scale=CAP_SCALE,
                          ramp_scale=RAMP_SCALE)
    _, cost = segment_costs(case, K)
    out = jax.jit(relax)(
        jnp.asarray(np.repeat(cost[:, :, None], T, axis=2)),
        jnp.asarray(demand), jnp.asarray(boundary["p_init"]),
        jnp.asarray(boundary["u_prev"]),
        jnp.asarray(boundary["up_time"].astype(float)),
        jnp.asarray(boundary["down_time"].astype(float)))

    assert float(out["mu"]) < 1e-6, "not converged; the comparison below is void"
    np.testing.assert_allclose(float(out["obj"]), float(r.fun), rtol=1e-9)
    np.testing.assert_allclose(np.asarray(out["u"]), ref["u"], atol=1e-7)

    rounded = (np.asarray(out["u"]) > 1e-6).astype(float)
    rounded_ref = (ref["u"] > 0.0).astype(float)
    assert np.array_equal(rounded, rounded_ref), (
        f"rounded commitment differs in {int(np.abs(rounded - rounded_ref).sum())} "
        f"of {rounded.size} unit-periods")

    # and the schedule that rounding leads to, which is what the market clears on
    third = PC.solve(PC.build(case, demand, u_fixed=rounded, **boundary, **KW))
    third_ref = PC.solve(PC.build(case, demand, u_fixed=rounded_ref, **boundary, **KW))
    np.testing.assert_allclose(third["obj"], third_ref["obj"], rtol=1e-12)


def test_rounding_tolerance_is_load_bearing(case):
    """`ROUND_EPS` separates two populations, and dropping it breaks the market.

    Two assertions, and the second is the one with teeth.  First, the tolerance
    sits strictly between the largest ``u`` the simplex leaves at zero and the
    smallest it leaves above zero, which is what makes it a solver tolerance
    rather than an economic threshold.  Second, the literal ``u > 0`` rounding rule
    still commits an order more unit-periods than the schedule is -- so a test
    that only checked `round_commitment` would pass against an implementation
    that had silently reverted to the specification's wording.

    The second assertion used to read ``every`` unit-period, and the structured
    block assembly broke it: summation order changed, and six of the 160 cells
    the simplex leaves at zero now come out as exact zeros, so the literal form
    commits 258 of 264 rather than 264.  Re-measured rather than relaxed, as the
    old wording asked: the gap this tolerance sits in **widened**, from 3.1e16 to
    5.7e16 (zero side 1.1e-18 against a positive side of 6.5e-2), the rounded
    commitment stayed identical to the simplex cell by cell, and ``mu`` did not
    move.  What must fail here is a solver whose zeros are exact enough to make
    the literal threshold accidentally right, which would commit the 104 the
    simplex does.
    """
    demand = np.full(T, 33_466.0)
    boundary = PC.first_boundary(case, float(demand[0]), **KW)

    lp = _drop_mu_md(PC.build(case, demand, **boundary, **KW))
    r = linprog(lp["c"], A_ub=lp["A_ub"], b_ub=lp["b_ub"], A_eq=lp["A_eq"],
                b_eq=lp["b_eq"], bounds=np.stack([lp["lo"], lp["hi"]], 1),
                method="highs")
    assert r.status == 0
    u_ref = PC.unpack(lp, np.asarray(r.x))["u"]

    relax, _ = make_relax(case, T, n_segments=K, cap_scale=CAP_SCALE,
                          ramp_scale=RAMP_SCALE)
    _, cost = segment_costs(case, K)
    u = np.asarray(jax.jit(relax)(
        jnp.asarray(np.repeat(cost[:, :, None], T, axis=2)),
        jnp.asarray(demand), jnp.asarray(boundary["p_init"]),
        jnp.asarray(boundary["u_prev"]),
        jnp.asarray(boundary["up_time"].astype(float)),
        jnp.asarray(boundary["down_time"].astype(float)))["u"])

    zero_side = u[u_ref <= 0.0].max()
    pos_side = u[u_ref > 0.0].min()
    assert zero_side < ROUND_EPS < pos_side, (
        f"tolerance outside the gap: {zero_side:.2e} .. {pos_side:.2e}")

    rounded = np.asarray(round_commitment(jnp.asarray(u)))
    assert np.array_equal(rounded, (u_ref > 0.0).astype(float))

    naive = (u > 0.0).astype(float)
    assert naive.sum() >= 2 * rounded.sum(), (
        f"the specification's literal `u > 0` commits {naive.sum():.0f} of "
        f"{naive.size}, no longer far above the {rounded.sum():.0f} of the "
        "schedule; if the solver has changed to one with exact zeros, ROUND_EPS's "
        "justification needs re-measuring rather than this assertion relaxing")
