# Written for this repository on 2026-08-15 -- no upstream counterpart.
"""Feeder 459_0: the physics check the source pays for, and the band it does not supply.

This case ships with something the vendored cases do not: **the source's own
power-flow solution**.  That makes the ohm-to-per-unit conversion testable rather
than arguable, which matters because this dataset has already produced one unit
coincidence that read as evidence.

**The criterion is the voltages, not `converged`.**  BFS floors V-squared at 0.25
and a floored fixed point still satisfies the residual test
(a vendored-code pitfall).  A conversion wrong by sqrt(3) or 3 -- the
shape of the known trap -- could put buses on that floor and the solver would
report success against a pinned constant.  So `test_converged_cannot_tell_the_
scalings_apart` asserts that `converged` is `True` even at a scaling that is two
orders of magnitude off: it pins, as a failing assertion rather than a docstring
sentence, the fact that this flag carries no information here.  If someone ever
makes `converged` discriminating, that test goes red, and what should then be
re-read is every place that relies on not trusting it.

**The band has no default and that is the point.**  A default would let the market
produce numbers on an uncalibrated voltage band, and those numbers look ordinary.
"""
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.case.cases.distribution.case459_0 import (
    BASE_CASE_VOLTAGE_PU, create_case459_0)
from powermarketjax.physics.bfs_power_flow import bfs_power_flow, prepare_bfs

#: Measured 2026-08-15 with this repository's own BFS, `base_kv = 20`, the power
#: factor of 0.9 the source's sidecar derives, and the slack held at the
#: published slack voltage.  The published voltages span 0.0196, so this is 2.4%
#: of the spread.  It is one-sided: see the residual test.
MAX_ABS_DV = 4.69e-4


@pytest.fixture(scope="module")
def case():
    """Any band will do here: none of these tests reads the voltage limits.

    That is why this file can run before §18 settles one.
    """
    return create_case459_0(node_v_min=0.9, node_v_max=1.1)


@pytest.fixture(scope="module")
def published():
    return np.asarray(BASE_CASE_VOLTAGE_PU, dtype=float)


def solve(case, published, r_scale=1.0):
    topo = prepare_bfs(case.replace(
        line_r=np.asarray(case.line_r) * r_scale,
        line_x=np.asarray(case.line_x) * r_scale))
    return bfs_power_flow(
        topo,
        p_load_pu=np.asarray(case.node_pd) / case.base_mva,
        q_load_pu=np.asarray(case.node_qd) / case.base_mva,
        v_slack=float(published[int(case.slack_bus_idx)]))


def test_the_topology_is_the_radial_feeder_the_source_describes(case):
    assert (case.n_nodes, case.n_lines) == (129, 128)
    assert case.n_lines == case.n_nodes - 1                  # radial
    assert case.n_units == 0                                 # substation only
    assert np.asarray(case.node_pd).min() >= 0.0             # no negative injection
    assert float(np.asarray(case.node_pd).sum()) == pytest.approx(13.719738, abs=1e-6)


def test_the_market_bfs_reproduces_the_published_base_case(case, published):
    """The conversion is verified against the source's own solution, not argued."""
    v = np.asarray(solve(case, published).v_mag, dtype=float)
    assert np.abs(v - published).max() < MAX_ABS_DV
    assert (v > 0.5 + 1e-9).all()                            # nothing on the floor


def test_the_residual_is_one_sided_which_is_the_reactive_assumption(case, published):
    """Every bus sits low, and by more where the drop is larger.

    That is the signature of slightly too much reactive load, which is expected:
    the 0.9 power factor is the *minimum* of the source's own quotient, so it is
    a lower bound.  A power factor near 0.92 fits better and is deliberately not
    used -- fitting it would spend this base case as a validation target.  Line
    charging, which the BFS does not model, points the same way and is untested.
    """
    d = np.asarray(solve(case, published).v_mag, dtype=float) - published
    assert (d <= 1e-12).all()                                # none high
    assert (d < 0).sum() == 128                              # all but the slack
    assert np.corrcoef(d, 1.0 - published)[0, 1] < -0.99


@pytest.mark.parametrize("scale", [np.sqrt(3), 3.0, 1 / np.sqrt(3), 1 / 3])
def test_a_wrong_conversion_is_visible_in_the_voltages(case, published, scale):
    """Each alternative scaling is at least an order of magnitude worse."""
    v = np.asarray(solve(case, published, r_scale=scale).v_mag, dtype=float)
    assert np.abs(v - published).max() > 10 * MAX_ABS_DV


def test_converged_cannot_tell_the_scalings_apart(case, published):
    """`converged` is True at a scaling that is 91x off, so it is not a criterion.

    Asserted rather than described: if this ever goes red, `converged` has become
    discriminating and every place that relies on not trusting it should be
    re-read.
    """
    good = solve(case, published)
    bad = solve(case, published, r_scale=3.0)
    assert bool(good.converged) and bool(bad.converged)
    bad_v = np.asarray(bad.v_mag, dtype=float)
    assert np.abs(bad_v - published).max() > 20 * MAX_ABS_DV


def test_the_choice_of_base_does_not_move_the_voltage_solution(case, published):
    """r_pu * p_pu = r_ohm * p_mw / base_kv^2, so the base cancels -- *if* r, x
    and the loads are all moved onto the new base together.

    This is why the base-case reproduction does not pin `base_mva`, and it is
    also why `create_case459_0` takes no `base_mva` argument: `_LINES` already
    carries r and x on the generated base, so changing the base alone would
    leave the two inconsistent.  The invariance being asserted is the consistent
    rescaling, which is the true statement; the loose one ("the base does not
    matter") would have hidden that trap.
    """
    worst = []
    for k in (0.01, 0.1, 1.0):
        scaled = case.replace(line_r=np.asarray(case.line_r) * k,
                              line_x=np.asarray(case.line_x) * k,
                              base_mva=case.base_mva * k)
        v = np.asarray(solve(scaled, published).v_mag, dtype=float)
        worst.append(np.abs(v - published).max())
    # 1e-6: measured spread 1.9e-8 over a 100x range of bases, 2026-08-15, GPU
    # (3x RTX 4500 Ada), x64 off.  The BFS runs in float32, so the residual
    # spread is round-off and not a difference in the solution; a tolerance at
    # the measured value would be asserting the round-off, not the invariance.
    assert max(worst) - min(worst) < 1e-6


def test_load_case_names_the_missing_declaration_instead_of_inventing_one():
    """Registered so the failure says which declaration is absent.

    Unregistered it would say "Unknown case", which reads as a typo.
    """
    for name in ("459_0", "swissdn_459_0"):
        with pytest.raises(ValueError, match="no registered voltage band"):
            load_case(name)


def test_the_voltage_band_has_no_default():
    with pytest.raises(TypeError):
        create_case459_0()
