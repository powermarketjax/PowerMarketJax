"""L1 domain correctness for the voltage sensitivity matrices (§3.2).

Two hand-worked feeders, agreement with the independently routed reference,
the structural invariants the matrices must carry, and one physical check
against the nonlinear power flow.

The two hand feeders are not redundant.  A chain pins the accumulation along a
path; a star pins the **absence** of accumulation between two buses that share
nothing but the substation.  An implementation that formed the outer product of
path resistances instead of intersecting the paths passes the chain and fails
the star, and that error leaves the clearing problem feasible with every
reported quantity finite, which is exactly the failure mode §16 names.

Which error each check actually catches was established by mutation rather than
asserted (2026-08-09).  Replacing the shared path by ``min`` of the two path
resistances, which is right on a chain and wrong on a star, fails 11 of these
tests and **passes the chain**; swapping `R` and `X` fails 12; negating `R`
fails 18.  Transposing `R` fails nothing, and that is not a gap: `R` is
symmetric, so the transpose is the same matrix.
"""
import types

import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.local_flexibility import build_voltage_sensitivity
from powermarketjax.physics import bfs_power_flow, prepare_bfs

from .reference import downstream_by_descent, sensitivity_by_intersection

#: `case33bw` is the development case and `case533mt_hi` the primary one
#: (revised 2026-08-09).  `case533mt_lo` is excluded from this market
#: because it is a net exporter, but its topology is identical to the primary
#: case, so it is still a legitimate case to assemble matrices from.
CASES = ("33bw", "533mt_hi", "533mt_lo", "141", "123_1ph", "118zh")


def feeder(from_to, r, x, n_bus, status=None, line_cap=None, base_mva=1.0):
    """Minimal stand-in for `CaseData` carrying only what §3.2 reads."""
    frm, to = zip(*from_to)
    n_line = len(frm)
    return types.SimpleNamespace(
        n_nodes=n_bus,
        slack_bus_idx=0,
        base_mva=base_mva,
        line_from_idx=np.array(frm),
        line_to_idx=np.array(to),
        line_r=np.array(r, np.float64),
        line_x=np.array(x, np.float64),
        line_cap=np.full(n_line, 1e6) if line_cap is None else np.array(line_cap, np.float64),
        line_status=None if status is None else np.array(status),
    )


def test_line_ratings_are_converted_to_per_unit():
    """`line_cap` is MVA and everything else here is per unit; the conversion happens once.

    Assembling (LIM) against the raw rating would leave the limit larger than
    the flows by the base power, so the constraint never binds and the market
    becomes voltage driven with nothing reporting it.
    """
    case = feeder([(0, 1), (1, 2)], [0.1, 0.3], [0.2, 0.4], 3,
                  line_cap=[8.0, 4.0], base_mva=16.0)
    sens = build_voltage_sensitivity(case)

    np.testing.assert_allclose(sens.p_max, [0.5, 0.25], rtol=1e-12)
    assert sens.base_mva == 16.0

    primary = build_voltage_sensitivity(load_case("533mt_hi"))
    raw = np.asarray(load_case("533mt_hi").line_cap, np.float64)[primary.line_index]
    np.testing.assert_allclose(primary.p_max * primary.base_mva, raw, rtol=1e-12)
    # the flows this feeder carries are order one per unit, so a rating left in
    # MVA would sit an order of magnitude above anything it must bind on
    assert primary.p_max.max() < 10.0 < raw.max()


def test_chain_feeder_hand_worked():
    """0 -- 1 -- 2, so bus 2 accumulates both lines and shares line 0 with bus 1."""
    sens = build_voltage_sensitivity(feeder([(0, 1), (1, 2)], [0.1, 0.3], [0.2, 0.4], 3))

    np.testing.assert_allclose(sens.R, [[0.0, 0.0, 0.0],
                                        [0.0, 0.1, 0.1],
                                        [0.0, 0.1, 0.4]])
    np.testing.assert_allclose(sens.X, [[0.0, 0.0, 0.0],
                                        [0.0, 0.2, 0.2],
                                        [0.0, 0.2, 0.6]])
    # line 0 supplies buses 1 and 2; line 1 supplies bus 2 only
    np.testing.assert_allclose(sens.A, [[0.0, 1.0, 1.0],
                                        [0.0, 0.0, 1.0]])


def test_star_feeder_hand_worked():
    """0 -- 1 and 0 -- 2, so the two leaves share no line and R[1, 2] is zero."""
    sens = build_voltage_sensitivity(feeder([(0, 1), (0, 2)], [0.1, 0.3], [0.2, 0.4], 3))

    np.testing.assert_allclose(sens.R, [[0.0, 0.0, 0.0],
                                        [0.0, 0.1, 0.0],
                                        [0.0, 0.0, 0.3]])
    np.testing.assert_allclose(sens.X, [[0.0, 0.0, 0.0],
                                        [0.0, 0.2, 0.0],
                                        [0.0, 0.0, 0.4]])
    np.testing.assert_allclose(sens.A, [[0.0, 1.0, 0.0],
                                        [0.0, 0.0, 1.0]])


def test_out_of_service_lines_are_removed():
    """A chain plus an open tie back to the substation stays a chain."""
    chain = feeder([(0, 1), (1, 2)], [0.1, 0.3], [0.2, 0.4], 3)
    with_tie = feeder([(0, 1), (1, 2), (0, 2)], [0.1, 0.3, 0.7], [0.2, 0.4, 0.9], 3,
                      status=[1, 1, 0])

    np.testing.assert_allclose(build_voltage_sensitivity(with_tie).R,
                               build_voltage_sensitivity(chain).R)
    assert build_voltage_sensitivity(with_tie).line_index.tolist() == [0, 1]


def test_non_radial_input_is_rejected():
    """A closed tie leaves a cycle, and dropping it silently would model another network."""
    closed = feeder([(0, 1), (1, 2), (0, 2)], [0.1, 0.3, 0.7], [0.2, 0.4, 0.9], 3)
    with pytest.raises(ValueError, match="spanning tree"):
        build_voltage_sensitivity(closed)


def test_disconnected_input_is_rejected():
    """A cycle on one component and a second component detached from the substation.

    The line count alone passes here, four lines for five buses, so this is the
    case the count check cannot reach and the traversal must.
    """
    split = feeder([(0, 1), (1, 2), (0, 2), (3, 4)],
                   [0.1, 0.3, 0.5, 0.7], [0.2, 0.4, 0.6, 0.8], 5)
    with pytest.raises(ValueError, match="unreachable"):
        build_voltage_sensitivity(split)


@pytest.mark.parametrize("case_id", CASES)
def test_structural_invariants(case_id):
    """Symmetry, non-negativity, the diagonal dominating its row, and a zero slack row.

    `R` is a Gram matrix of the path indicator against positive line
    resistances, so it is symmetric positive semi-definite by construction.
    The diagonal dominates because a shared path is a subset of a full path.
    """
    sens = build_voltage_sensitivity(load_case(case_id))
    R, X = sens.R, sens.X

    np.testing.assert_allclose(R, R.T, atol=0.0)
    np.testing.assert_allclose(X, X.T, atol=0.0)
    assert (R >= 0.0).all() and (X >= 0.0).all()
    # a shared path cannot exceed either full path
    assert (R <= np.minimum.outer(R.diagonal(), R.diagonal()) + 1e-12).all()
    # the substation shares no line with anything, itself included
    assert not R[sens.slack].any() and not X[sens.slack].any()
    assert np.linalg.eigvalsh(R).min() > -1e-12


@pytest.mark.parametrize("case_id", CASES)
def test_matches_independent_path_intersection(case_id):
    """§16's independent path-sum, in full on the small cases and strided on the large ones.

    The comparison is exact, and that is an observation rather than a
    guarantee: the two routes sum the same float64 values in different orders,
    one inside a BLAS product and one over a sorted subset, so they agree to
    the last bit on every vendored case but are not obliged to.  Should a
    future case disagree, the fix is a tolerance derived from the measured
    error, not a loosened one chosen to make it pass.
    """
    case = load_case(case_id)
    sens = build_voltage_sensitivity(case)

    if sens.n_bus <= 150:
        R_ref, X_ref = sensitivity_by_intersection(case)
        np.testing.assert_allclose(sens.R, R_ref, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(sens.X, X_ref, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(sens.A, downstream_by_descent(case), atol=0.0)
    else:
        # strided rather than sampled: the repository bans `np.random`, and a
        # stride spreads over the feeder without introducing a second notion of
        # randomness into a suite whose PRNG contract is explicit keys
        rows, cols = range(0, sens.n_bus, 7), range(0, sens.n_bus, 5)
        pairs = [(n, m) for n in rows for m in cols]
        R_ref, X_ref = sensitivity_by_intersection(case, pairs)
        got_R = np.array([sens.R[n, m] for n, m in pairs])
        got_X = np.array([sens.X[n, m] for n, m in pairs])
        np.testing.assert_allclose(got_R, R_ref, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(got_X, X_ref, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("case_id", ("33bw", "533mt_hi"))
def test_precision_is_lost_in_the_arithmetic_not_in_the_storage(case_id):
    """What float64 assembly buys, stated as a measurement rather than an assumption.

    Every array of a `CaseData` is float32, the line parameters included, so no
    assembly route recovers precision in the **data**.  Upcasting the vendored
    topology and multiplying in float64 therefore reproduces this module bit for
    bit.  What float64 buys is the **arithmetic**: the same product carried out
    in float32 costs order 1e-7 relative, over paths of up to 23 lines and a
    product of 533-square matrices, and that error alone would consume the L2
    tolerance budget, which must be derived from measurement.
    """
    case = load_case(case_id)
    sens = build_voltage_sensitivity(case)
    topo = prepare_bfs(case)

    assert np.asarray(case.line_r).dtype == np.float32

    P32 = np.asarray(topo.path_matrix)
    scale = np.abs(sens.R).max()

    upcast = (P32.astype(np.float64) * np.asarray(topo.r_pu, np.float64)) @ P32.astype(np.float64).T
    np.testing.assert_allclose(upcast, sens.R, rtol=0.0, atol=0.0)

    in_float32 = (P32 * np.asarray(topo.r_pu)) @ P32.T
    err = np.abs(in_float32 - sens.R).max() / scale
    assert 1e-9 < err < 1e-6, f"expected float32 arithmetic error, got {err:.3g}"


@pytest.mark.parametrize("case_id", ("33bw", "533mt_hi"))
def test_linearised_voltage_tracks_the_nonlinear_solution(case_id):
    """§3.2 against the sweep at the registered load, where the linearisation is tight.

    Measured against mutations: this is the only check here that catches `R`
    and `X` swapped without help from the reference, since the swap leaves both
    matrices symmetric, non-negative and diagonally dominant.  It also catches
    a sign error and the shared-path error, so it is the one check that would
    still have teeth if the reference were to share an error with the
    implementation.  The one-sided gap is §3.3's, so the test pins the
    direction of the error and not only its size.
    """
    case = load_case(case_id)
    sens = build_voltage_sensitivity(case)

    p_pu = np.asarray(case.node_pd, np.float64) / case.base_mva
    q_pu = np.asarray(case.node_qd, np.float64) / case.base_mva
    v_sq_lin = 1.0 + 2.0 * (sens.R @ (-p_pu) + sens.X @ (-q_pu))

    res = bfs_power_flow(topo := prepare_bfs(case),
                         np.float32(p_pu), np.float32(q_pu))
    assert bool(res.converged) and not bool(res.floor_active)
    v_lin = np.sqrt(v_sq_lin)
    v_bfs = np.asarray(res.v_mag, np.float64)

    assert np.abs(v_lin - v_bfs).max() < 5e-3
    # §3.3: dropping the loss terms understates the drop, so the linearised
    # magnitude is an overestimate, one-sided rather than random
    assert (v_lin - v_bfs).min() > -1e-6
    assert topo.n_lines == sens.n_line


@pytest.mark.parametrize("case_id", CASES)
def test_flow_expression_reproduces_the_sweep_at_light_load(case_id):
    """(FL) of §3.2: flow on a line is the negated sum of the injections it supplies.

    Compared against the sweep at one hundredth of the registered load, where
    the dropped loss terms are four orders of magnitude below the flows.
    """
    case = load_case(case_id)
    sens = build_voltage_sensitivity(case)
    scale = 0.01

    p_pu = np.asarray(case.node_pd, np.float64) / case.base_mva * scale
    q_pu = np.asarray(case.node_qd, np.float64) / case.base_mva * scale
    flow_lin = sens.A @ p_pu                       # -A @ (-p) for load-only injection

    res = bfs_power_flow(prepare_bfs(case), np.float32(p_pu), np.float32(q_pu))
    flow_bfs = np.asarray(res.p_branch, np.float64)
    denominator = max(np.abs(flow_bfs).max(), 1e-9)
    assert np.abs(flow_lin - flow_bfs).max() / denominator < 1e-3
