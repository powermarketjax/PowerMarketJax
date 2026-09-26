"""``BFSResult.converged`` must not be True on a floored (unsolved) point.

Written for this repository on 2026-08-05 -- no upstream counterpart.

The failure mode: the forward sweep floors V² at ``_MIN_V_SQ_FLOOR`` (0.25,
i.e. 0.5 p.u.). A node pinned on that floor has ``ΔV² == 0`` by construction,
so ``max|ΔV²| < tol`` is satisfied and upstream reported ``converged=True`` on
operating points the solver never solved. Measured on case33bw before the fix:
5x and 10x load both gave ``converged=True`` with ``v_min`` exactly 0.5000, and
50x load gave ``converged=True`` with ``v_max = 4.52`` p.u.

This matters because the P2P and local-flexibility markets need voltage
feasibility checks, and agents there are specifically rewarded for pushing the
feeder toward its limits.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.case import create_case33bw
from powermarketjax.physics import (
    MIN_V_PU_FLOOR,
    prepare_bfs,
    bfs_power_flow,
)


@pytest.fixture(scope="module")
def case33():
    return create_case33bw()


@pytest.fixture(scope="module")
def topo33(case33):
    return prepare_bfs(case33)


@pytest.fixture(scope="module")
def base_load(case33):
    """case33bw's own load, in p.u. Total is ~3.7 MW on a 100 MVA base."""
    p = jnp.asarray(case33.node_pd, jnp.float32) / case33.base_mva
    q = jnp.asarray(case33.node_qd, jnp.float32) / case33.base_mva
    return p, q


def _solve(topo, base_load, scale):
    p, q = base_load
    return jax.jit(lambda pl, ql: bfs_power_flow(topo, pl, ql))(p * scale, q * scale)


class TestFloorInvalidatesConvergence:

    @pytest.mark.parametrize("scale", [0.0, 0.5, 1.0, 2.0])
    def test_normal_loading_converges_and_floor_is_inactive(
        self, topo33, base_load, scale,
    ):
        """Up to 2x nameplate load the solution is genuine: converged, no floor."""
        r = _solve(topo33, base_load, scale)
        assert bool(r.converged), f"scale={scale} should converge"
        assert not bool(r.floor_active), f"scale={scale} must not hit the floor"
        assert float(r.v_mag.min()) > MIN_V_PU_FLOOR

    @pytest.mark.parametrize("scale", [5.0, 10.0, 20.0, 50.0])
    def test_overload_never_reports_convergence(self, topo33, base_load, scale):
        """Loads that collapse voltage onto the floor must report failure.

        These are exactly the cases upstream got wrong: at 5x and 10x it
        returned converged=True with v_min pinned at 0.5.
        """
        r = _solve(topo33, base_load, scale)
        assert bool(r.floor_active), (
            f"scale={scale}: expected the voltage floor to be active "
            f"(v_min={float(r.v_mag.min()):.4f})"
        )
        assert not bool(r.converged), (
            f"scale={scale}: converged must be False while the floor is active "
            f"(v_min={float(r.v_mag.min()):.4f}) -- this is the false-convergence "
            f"regression"
        )

    def test_floored_result_is_pinned_at_the_documented_floor(
        self, topo33, base_load,
    ):
        """When the floor engages, v_min sits exactly on MIN_V_PU_FLOOR."""
        r = _solve(topo33, base_load, 10.0)
        np.testing.assert_allclose(
            float(r.v_mag.min()), MIN_V_PU_FLOOR, atol=1e-4,
        )

    def test_converged_implies_floor_inactive(self, topo33, base_load):
        """The invariant, swept across the whole loading range."""
        for scale in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]:
            r = _solve(topo33, base_load, scale)
            if bool(r.converged):
                assert not bool(r.floor_active), (
                    f"scale={scale}: converged=True with floor_active=True"
                )


class TestFlagsUnderJaxTransforms:

    def test_flags_are_scalars_under_jit(self, topo33, base_load):
        r = _solve(topo33, base_load, 1.0)
        assert np.asarray(r.converged).shape == ()
        assert np.asarray(r.floor_active).shape == ()
        assert np.asarray(r.converged).dtype == np.bool_
        assert np.asarray(r.floor_active).dtype == np.bool_

    def test_flags_batch_under_vmap(self, topo33, base_load):
        """A mixed batch must report per-element flags, not a collapsed one."""
        p, q = base_load
        scales = jnp.array([1.0, 1.0, 10.0, 10.0], jnp.float32)
        batch_p = p[None, :] * scales[:, None]
        batch_q = q[None, :] * scales[:, None]
        vfn = jax.jit(jax.vmap(lambda pl, ql: bfs_power_flow(topo33, pl, ql)))
        r = vfn(batch_p, batch_q)

        assert np.asarray(r.converged).shape == (4,)
        assert np.asarray(r.floor_active).shape == (4,)
        np.testing.assert_array_equal(
            np.asarray(r.converged), np.array([True, True, False, False]),
        )
        np.testing.assert_array_equal(
            np.asarray(r.floor_active), np.array([False, False, True, True]),
        )

    def test_result_pytree_structure_is_static(self, topo33, base_load):
        """Adding floor_active must not make the pytree structure load-dependent."""
        light = _solve(topo33, base_load, 1.0)
        heavy = _solve(topo33, base_load, 10.0)
        assert (jax.tree_util.tree_structure(light)
                == jax.tree_util.tree_structure(heavy))


class TestSweepNumericsUnchanged:
    """The fix must change only the reported flags, never the physics.

    Reference values are the analytic invariants of the sweeps; if a future
    edit to the floor logic perturbs the voltages or flows, these fail.
    """

    def test_slack_voltage_and_zero_load_profile(self, topo33, case33, base_load):
        p_zero = jnp.zeros((case33.n_nodes,), jnp.float32)
        r = bfs_power_flow(topo33, p_zero, p_zero)
        assert bool(r.converged)
        assert not bool(r.floor_active)
        np.testing.assert_allclose(np.asarray(r.v_mag), 1.0, atol=1e-4)
        np.testing.assert_allclose(np.asarray(r.p_branch), 0.0, atol=1e-6)

    def test_nominal_solution_matches_known_profile(self, topo33, base_load):
        """case33bw at nameplate load: v_min ~0.9133 p.u., converged in <=6 iters.

        Measured 2026-08-05 on jax 0.10.2 / float32. This pins the sweep output
        so the floor-detection change cannot silently move the solution.
        """
        r = _solve(topo33, base_load, 1.0)
        assert bool(r.converged)
        np.testing.assert_allclose(float(r.v_mag.min()), 0.9133, atol=2e-3)
        np.testing.assert_allclose(float(r.v_mag[0]), 1.0, atol=1e-4)
        assert int(r.iterations) <= 6
        assert float(r.p_loss.sum()) > 0.0

    def test_no_nan_anywhere_even_when_floored(self, topo33, base_load):
        """A floored result must still be NaN-free, since it enters obs."""
        for scale in [10.0, 50.0]:
            r = _solve(topo33, base_load, scale)
            for leaf in jax.tree_util.tree_leaves(r):
                arr = np.asarray(leaf)
                if arr.dtype.kind == "f":
                    assert not np.isnan(arr).any()
                    assert not np.isinf(arr).any()
