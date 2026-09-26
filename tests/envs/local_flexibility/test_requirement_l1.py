"""L1 domain correctness for the published requirement (§4).

A hand-worked feeder pins the arithmetic, and two definitional properties pin
the meaning.  The definitional ones matter more: §4 states what each
requirement *is*, namely the injection that removes the violation, so the test
injects it and checks the violation is gone.  That cannot be passed by a second
transcription of the same formula, which is what a reference computation of
`req` would be.

The remaining checks are the two §16 asks for by name, that each requirement is
zero at exactly the indices respecting their bound, and the rule that no
aggregate may be a sum.
"""
import types

import jax
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.local_flexibility import (build_voltage_sensitivity,
                                                   make_requirement)

CASES = ("33bw", "533mt_hi", "141", "118zh")


@pytest.fixture(autouse=True)
def x64():
    """Save, set and restore, as the day-ahead suite does.

    x64 is not set at import time because other modules in this repository
    switch it off globally, and a bare `update` here would leak into whatever
    runs next.
    """
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def feeder(from_to, r, x, n_bus, v_min, line_cap, base_mva=1.0):
    """Minimal stand-in for `CaseData` carrying only what §3.2 and §4 read."""
    frm, to = zip(*from_to)
    return types.SimpleNamespace(
        n_nodes=n_bus, slack_bus_idx=0, base_mva=base_mva,
        line_from_idx=np.array(frm), line_to_idx=np.array(to),
        line_r=np.array(r, np.float64), line_x=np.array(x, np.float64),
        line_cap=np.array(line_cap, np.float64), line_status=None,
        node_v_min=np.array(v_min, np.float64),
    )


CHAIN = dict(from_to=[(0, 1), (1, 2)], r=[0.1, 0.3], x=[0.2, 0.4], n_bus=3,
             v_min=[1.0, 0.95, 0.95], line_cap=[0.8, 10.0])


def test_chain_feeder_hand_worked():
    """0 -- 1 -- 2 carrying 0.5 per unit at each leaf, worked out by hand.

    Line 0 carries both loads and is rated below that, so the thermal
    requirement is non-zero there and zero on line 1.
    """
    case = feeder(**CHAIN)
    sens = build_voltage_sensitivity(case)
    out = jax.jit(make_requirement(case, sens))(np.array([0.0, -0.5, -0.5]),
                                                np.zeros(3))

    np.testing.assert_allclose(out["v_sq"], [1.0, 0.8, 0.5], rtol=1e-12)
    np.testing.assert_allclose(out["flow"], [1.0, 0.5], rtol=1e-12)
    # (0.95^2 - 0.8) / (2 * 0.1)  and  (0.95^2 - 0.5) / (2 * 0.4)
    np.testing.assert_allclose(out["req_v"], [0.0, 0.5125, 0.503125], rtol=1e-12)
    np.testing.assert_allclose(out["req_th"], [0.2, 0.0], atol=1e-15)
    assert int(out["req_v_count"]) == 2 and int(out["req_th_count"]) == 1


@pytest.mark.parametrize("case_id", CASES)
def test_injecting_the_voltage_requirement_removes_the_violation(case_id):
    """§4 defines `req_v` as the injection at that bus which, acting alone, removes it.

    Injecting exactly that amount at exactly that bus must land the squared
    voltage magnitude on the bound, to solver-free arithmetic precision.  A
    formula off by a factor, a missing two, or the wrong diagonal fails here
    while still producing a plausible non-negative vector.
    """
    case = load_case(case_id)
    sens = build_voltage_sensitivity(case)
    publish = make_requirement(case, sens)

    scale = 2.0
    p_inj = -np.asarray(case.node_pd, np.float64) / case.base_mva * scale
    q_inj = -np.asarray(case.node_qd, np.float64) / case.base_mva * scale
    base = publish(p_inj, q_inj)
    req_v = np.asarray(base["req_v"])

    violated = np.flatnonzero(req_v > 0.0)
    assert violated.size, f"{case_id} has no violation at scale {scale}"
    v_lo_sq = np.asarray(case.node_v_min, np.float64) ** 2

    for bus in violated[:: max(1, violated.size // 20)]:
        lifted = p_inj.copy()
        lifted[bus] += req_v[bus]
        v_sq = np.asarray(publish(lifted, q_inj)["v_sq"])
        np.testing.assert_allclose(v_sq[bus], v_lo_sq[bus], rtol=1e-9)


@pytest.mark.parametrize("case_id", CASES)
def test_injecting_the_thermal_requirement_removes_the_overload(case_id):
    """§4 defines `req_th` in megawatts because (FL) makes it deliverable one for one.

    One per unit injected anywhere downstream of a line reduces its flow by one
    per unit, so injecting the requirement at any single downstream bus lands
    the flow on the rating.  The same injection at a bus that is not downstream
    must leave the flow untouched, which is what makes locational value against
    a thermal need binary rather than graded.
    """
    case = load_case(case_id)
    sens = build_voltage_sensitivity(case)
    publish = make_requirement(case, sens)

    scale = 3.0
    p_inj = -np.asarray(case.node_pd, np.float64) / case.base_mva * scale
    q_inj = -np.asarray(case.node_qd, np.float64) / case.base_mva * scale
    base = publish(p_inj, q_inj)
    req_th = np.asarray(base["req_th"])

    overloaded = np.flatnonzero(req_th > 0.0)
    if not overloaded.size:
        pytest.skip(f"{case_id} carries no thermal overload at scale {scale}")

    for line in overloaded[:: max(1, overloaded.size // 10)]:
        downstream = np.flatnonzero(sens.A[line] > 0.0)
        upstream = np.flatnonzero(sens.A[line] == 0.0)

        lifted = p_inj.copy()
        lifted[downstream[0]] += req_th[line]
        np.testing.assert_allclose(np.asarray(publish(lifted, q_inj)["flow"])[line],
                                   sens.p_max[line], rtol=1e-9)

        elsewhere = p_inj.copy()
        elsewhere[upstream[upstream != sens.slack][0]] += req_th[line]
        np.testing.assert_allclose(np.asarray(publish(elsewhere, q_inj)["flow"])[line],
                                   np.asarray(base["flow"])[line], rtol=1e-12)


@pytest.mark.parametrize("case_id", CASES)
def test_each_requirement_is_zero_exactly_where_its_bound_holds(case_id):
    """The check §16 asks for by name, in both of its halves."""
    case = load_case(case_id)
    sens = build_voltage_sensitivity(case)
    publish = make_requirement(case, sens)

    p_inj = -np.asarray(case.node_pd, np.float64) / case.base_mva * 2.0
    q_inj = -np.asarray(case.node_qd, np.float64) / case.base_mva * 2.0
    out = publish(p_inj, q_inj)

    v_lo_sq = np.asarray(case.node_v_min, np.float64) ** 2
    respects_voltage = np.asarray(out["v_sq"]) >= v_lo_sq
    respects_voltage[sens.slack] = True          # §3.1: not a constrained bus
    np.testing.assert_array_equal(np.asarray(out["req_v"]) == 0.0, respects_voltage)

    respects_rating = np.asarray(out["flow"]) <= sens.p_max
    np.testing.assert_array_equal(np.asarray(out["req_th"]) == 0.0, respects_rating)


@pytest.mark.parametrize("case_id", CASES)
def test_substation_is_never_a_requirement(case_id):
    """Its row of `R` is zero, so no injection moves it and §4 defines its need as zero.

    Left as a division it would be zero over zero.  The repair of §13 also
    makes both of its registered bounds one, so a naive test against the bound
    would report a violation at every period.
    """
    case = load_case(case_id)
    sens = build_voltage_sensitivity(case)
    publish = make_requirement(case, sens)

    p_inj = -np.asarray(case.node_pd, np.float64) / case.base_mva * 3.0
    out = publish(p_inj, np.zeros_like(p_inj))

    assert float(np.asarray(out["req_v"])[sens.slack]) == 0.0
    np.testing.assert_allclose(np.asarray(out["v_sq"])[sens.slack], 1.0, rtol=1e-12)
    assert np.isfinite(np.asarray(out["req_v"])).all()


@pytest.mark.parametrize("case_id", CASES)
def test_aggregates_are_extremes_and_counts_not_sums(case_id):
    """§4 forbids a total, and the measured factor for the voltage sum is twenty-six."""
    case = load_case(case_id)
    sens = build_voltage_sensitivity(case)
    out = make_requirement(case, sens)(
        -np.asarray(case.node_pd, np.float64) / case.base_mva * 2.0,
        -np.asarray(case.node_qd, np.float64) / case.base_mva * 2.0)

    req_v, req_th = np.asarray(out["req_v"]), np.asarray(out["req_th"])
    assert float(out["req_v_max"]) == req_v.max()
    assert int(out["req_v_count"]) == int((req_v > 0.0).sum())
    assert float(out["req_th_max"]) == req_th.max()
    assert int(out["req_th_count"]) == int((req_th > 0.0).sum())
    # the aggregate must not be the sum, which on this case is far larger
    if req_v.sum() > 0.0:
        assert float(out["req_v_max"]) < req_v.sum()


def test_zero_path_resistance_is_rejected():
    """`req_v` divides by the path resistance, so a resistanceless bus is undefined."""
    case = feeder([(0, 1), (1, 2)], [0.0, 0.0], [0.2, 0.4], 3,
                  v_min=[1.0, 0.95, 0.95], line_cap=[10.0, 10.0])
    sens = build_voltage_sensitivity(case)
    with pytest.raises(ValueError, match="path resistance"):
        make_requirement(case, sens)
