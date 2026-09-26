"""L1 domain correctness for the clearing operator (§6).

Feasibility of the cleared solution, the bound §16 asks for against the
pure-curtailment solution, monotonicity of an award in its own price, the
accounting identity behind the objective value, and the two traps that make
this market's linear program different from the day-ahead one.

Those two traps are why the case list matters here.  `case33bw` registers no
line ratings, so `p_max` is the 1e6 sentinel and (LIM) is slack by four orders
of magnitude: **a sign error in (LIM) is undetectable on it**, and one was
found on the primary case against HiGHS after passing every check on the
development case.  Any test that means to exercise (LIM) has to run on a
533-bus case.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.local_flexibility import build_voltage_sensitivity
from powermarketjax.envs.local_flexibility.clearing import (OFF_EPS, VOLL,
                                                            make_clearing)

#: `case33bw` cannot exercise (LIM); `case533mt_hi` is the primary case and the
#: only one here whose ratings bind (§13).
CASES = (("33bw", 8), ("533mt_hi", 40), ("141", 20))
KAPPA = 2.0
TOL = 1e-7          # solver tolerance; `mu` lands near 1e-10 on these problems


@pytest.fixture(autouse=True)
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def scenario(case_id, n_agent, kappa=KAPPA, seed=1, qty=0.02, margin=0.0):
    """One period on one case: network, population, offers and baseline."""
    case = load_case(case_id)
    sens = build_voltage_sensitivity(case)
    rng = np.random.default_rng(seed)

    load_bus = np.flatnonzero(np.asarray(case.node_pd) > 0)
    load_bus = load_bus[load_bus != sens.slack]
    agent_bus = np.sort(rng.choice(load_bus, n_agent, replace=False))

    clear, spec = make_clearing(case, sens, agent_bus, voltage_margin=margin)
    load = np.maximum(np.asarray(case.node_pd, np.float64) / case.base_mva * kappa, 0.0)
    q_load = np.asarray(case.node_qd, np.float64) / case.base_mva * kappa
    return dict(case=case, sens=sens, clear=jax.jit(clear), spec=spec,
                price=rng.uniform(20.0, 80.0, n_agent),
                qty_max=np.full(n_agent, qty),
                p_inj=-load, q_inj=-q_load, load=load)


def run(s):
    return s["clear"](s["price"], s["qty_max"], s["p_inj"], s["q_inj"], s["load"])


@pytest.mark.parametrize("case_id,n_agent", CASES)
def test_cleared_solution_is_feasible(case_id, n_agent):
    """(VLO), (VHI), (LIM), (CAP) and (SHED) on the solution, to solver tolerance."""
    s = scenario(case_id, n_agent)
    out = run(s)
    sens, spec, case = s["sens"], s["spec"], s["case"]
    bounded = spec["bounded"]

    award, shed = np.asarray(out["award"]), np.asarray(out["shed"])
    assert float(out["mu"]) < 1e-6, f"unconverged solve, mu={float(out['mu']):.3g}"

    # (CAP) and (SHED)
    assert (award >= -TOL).all() and (award <= s["qty_max"] + TOL).all()
    assert (shed >= -TOL).all() and (shed <= s["load"] + TOL).all()

    # the operating point the solution implies
    p_inj = s["p_inj"].copy()
    np.add.at(p_inj, spec["agent_bus"], award)
    p_inj = p_inj + shed
    pd = np.asarray(case.node_pd, np.float64)
    phi = np.where(pd > 0.0, np.asarray(case.node_qd, np.float64) / np.where(pd > 0.0, pd, 1.0), 0.0)
    q_inj = s["q_inj"] + phi * shed

    v_sq = 1.0 + 2.0 * (sens.R @ p_inj + sens.X @ q_inj)
    flow = -sens.A @ p_inj

    v_lo = np.asarray(case.node_v_min, np.float64)[bounded]
    v_hi = np.asarray(case.node_v_max, np.float64)[bounded]
    assert (v_sq[bounded] >= v_lo ** 2 - TOL).all()          # (VLO)
    assert (v_sq[bounded] <= v_hi ** 2 + TOL).all()          # (VHI)
    assert (np.abs(flow) <= sens.p_max + TOL).all()          # (LIM)


@pytest.mark.parametrize("case_id,n_agent", CASES)
def test_objective_is_the_accounting_identity(case_id, n_agent):
    """`z` must be the as-bid cost plus the curtailment cost, in dollars.

    This is what ties the settlement of §8 to the optimal value, and it is the
    check that catches the per-unit to megawatt factor going missing: without
    it every quantity stays finite and the identity is wrong by `base_mva`.
    """
    s = scenario(case_id, n_agent)
    out = run(s)
    delta, base = s["spec"]["period_hours"], s["spec"]["base_mva"]

    as_bid = delta * base * float(np.dot(s["price"], np.asarray(out["award"])))
    curtail = delta * base * VOLL * float(np.asarray(out["shed"]).sum())
    np.testing.assert_allclose(float(out["z"]), as_bid + curtail, rtol=1e-9)


@pytest.mark.parametrize("case_id,n_agent", CASES)
def test_optimum_does_not_exceed_the_pure_curtailment_solution(case_id, n_agent):
    """§16's bound: curtailing everything is feasible, so the optimum is no dearer.

    A sign error in the sensitivity matrices violates this while leaving every
    reported quantity finite.
    """
    s = scenario(case_id, n_agent)
    out = run(s)
    delta, base = s["spec"]["period_hours"], s["spec"]["base_mva"]
    pure_curtailment = delta * base * VOLL * float(s["load"].sum())
    assert float(out["z"]) <= pure_curtailment + 1e-6


@pytest.mark.parametrize("case_id,n_agent", CASES)
def test_award_is_monotone_in_its_own_price(case_id, n_agent):
    """§16: lowering one offer price, all else equal, must not reduce its award."""
    s = scenario(case_id, n_agent)
    base_award = np.asarray(run(s)["award"])

    target = int(np.argmax(s["price"]))
    cheaper = s["price"].copy()
    cheaper[target] *= 0.25
    lowered = np.asarray(s["clear"](cheaper, s["qty_max"], s["p_inj"],
                                    s["q_inj"], s["load"])["award"])

    assert lowered[target] >= base_award[target] - TOL


def test_thermal_limit_binds_on_the_primary_case():
    """(LIM) must actually be reachable, or the sign of its rows is untested.

    The development case cannot serve here: its ratings are the 1e6 sentinel,
    so both line rows are slack by four orders of magnitude and a swapped pair
    of right-hand sides passes unnoticed.  That is not hypothetical; it is how
    the swap in this operator was found.
    """
    s = scenario("533mt_hi", 40, kappa=3.0)
    out = run(s)

    over_at_baseline = int(np.asarray(out["req_th"] > 0.0).sum())
    assert over_at_baseline > 0, "no thermal need, so this test proves nothing"

    sens, spec = s["sens"], s["spec"]
    p_inj = s["p_inj"].copy()
    np.add.at(p_inj, spec["agent_bus"], np.asarray(out["award"]))
    flow = -sens.A @ (p_inj + np.asarray(out["shed"]))
    assert (np.abs(flow) <= sens.p_max + TOL).all()
    # the clearing had to move the flow, not merely respect a slack bound
    assert (np.abs(flow) > sens.p_max - 1e-3).any()


def test_a_positive_safety_margin_stays_feasible():
    """The substation is excluded from (VLO) and (VHI), and this is why.

    Its registered bounds are both one after §13's repair and its squared
    voltage magnitude is identically one, so imposing (VLO) there with a
    positive margin would demand more than one from a bus fixed at one and the
    program would be infeasible for no physical reason.
    """
    s = scenario("533mt_hi", 40, margin=0.005)
    out = run(s)
    assert float(out["mu"]) < 1e-6
    assert np.isfinite(np.asarray(out["award"])).all()

    tighter = scenario("533mt_hi", 40, margin=0.02)
    assert float(run(tighter)["z"]) >= float(out["z"]) - 1e-6


def test_offers_displace_curtailment_when_they_are_cheap():
    """Curtailment is the backstop, so an offer below VOLL per unit of relief wins."""
    dear = scenario("533mt_hi", 40, kappa=2.5)
    dear["price"] = np.full(len(dear["price"]), 5.0)
    cheap_out = dear["clear"](dear["price"], dear["qty_max"], dear["p_inj"],
                              dear["q_inj"], dear["load"])

    none_offered = dear["clear"](dear["price"], np.zeros_like(dear["qty_max"]),
                                 dear["p_inj"], dear["q_inj"], dear["load"])

    assert float(np.asarray(cheap_out["award"]).sum()) > 0.0
    assert float(np.asarray(cheap_out["shed"]).sum()) < \
        float(np.asarray(none_offered["shed"]).sum())
    assert float(cheap_out["z"]) < float(none_offered["z"])


def test_zero_quantity_offers_receive_exactly_nothing():
    """The `OFF_EPS` box keeps the interior alive; its phantom must not escape."""
    s = scenario("533mt_hi", 40)
    qty = s["qty_max"].copy()
    qty[::3] = 0.0
    award = np.asarray(s["clear"](s["price"], qty, s["p_inj"], s["q_inj"],
                                  s["load"])["award"])

    assert (award[::3] == 0.0).all()
    assert (award <= qty + TOL).all()
    assert OFF_EPS < 1e-6          # the phantom is far below anything reported


def test_buses_that_net_to_generation_are_not_curtailed():
    """The cases give a net injection, so a bus that nets to generation sheds nothing.

    Nineteen buses of the primary case are in that position.  Passing their
    negative net injection through as a curtailment bound would make the box
    run from zero to a negative number, which is infeasible.
    """
    case = load_case("533mt_hi")
    sens = build_voltage_sensitivity(case)
    clear, _ = make_clearing(case, sens, np.array([10, 20, 30]))

    raw = np.asarray(case.node_pd, np.float64) / case.base_mva * KAPPA
    assert (raw < 0.0).any(), "case no longer has net-generating buses"

    out = jax.jit(clear)(np.full(3, 40.0), np.full(3, 0.02), -raw,
                         -np.asarray(case.node_qd, np.float64) / case.base_mva * KAPPA,
                         raw)
    shed = np.asarray(out["shed"])
    assert (shed[raw < 0.0] == 0.0).all()
    assert float(out["mu"]) < 1e-6
