"""L1 domain correctness for the settlement (§8).

The money-balance identity against the clearing's optimal value, a hand-worked
example pinning the arithmetic, the property that the reward follows the award
rather than the offer, and the syntax-tree check that `voll` reaches no
settlement expression.

The identity is the strongest check available here and it is still weaker than
its day-ahead counterpart, which is worth stating plainly. Pay-as-bid has no
pricing formula, so there is no dual to get wrong and nothing for a finite
difference to catch. What the identity does pin is the three errors §8 names:
paying on the offered quantity instead of the cleared one, counting a cleared
quantity twice, and letting the cost of curtailment reach a settlement
expression.
"""
import ast
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.local_flexibility import (build_voltage_sensitivity,
                                                   make_settlement)
from powermarketjax.envs.local_flexibility import settlement as settlement_mod
from powermarketjax.envs.local_flexibility.clearing import VOLL, make_clearing

KAPPA = 1.5
N_AGENT = 40
DELTA = 0.25
#: Arbitrary, and arbitrary on purpose: §14 records that this market has no
#: exogenous price series, so the cost basis is an argument rather than a
#: registered value and no result here depends on its level.
ENERGY_PRICE = 40.0
CYCLE_COST = 3.0


@pytest.fixture(autouse=True)
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture(scope="module")
def market():
    case = load_case("533mt_hi")
    sens = build_voltage_sensitivity(case)
    load = np.asarray(case.node_pd, np.float64) / case.base_mva
    reactive = np.asarray(case.node_qd, np.float64) / case.base_mva
    buses = np.flatnonzero(load > 0.0)
    buses = buses[buses != sens.slack]
    weights = load[buses] / load[buses].sum()
    return case, sens, load, reactive, buses, weights


def clear_one(market, seed=0, kappa=KAPPA):
    case, sens, load, reactive, buses, weights = market
    rng = np.random.default_rng(seed)
    agent_bus = np.sort(rng.choice(buses, N_AGENT, replace=False, p=weights))
    clear, spec = make_clearing(case, sens, agent_bus, period_hours=DELTA)
    price = rng.uniform(20.0, 80.0, N_AGENT)
    out = jax.jit(clear)(price, np.full(N_AGENT, 0.02), -load * kappa,
                         -reactive * kappa, np.maximum(load * kappa, 0.0))
    return price, out, spec, sens


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_money_balance_against_the_optimal_value(market, seed):
    """§8: what participants receive is the optimum less the curtailment term.

    With no residual, which is what distinguishes this from a uniform-price
    market: no congestion rent, no uplift, nothing collected that is not paid
    out.
    """
    price, out, spec, sens = clear_one(market, seed)
    settle = jax.jit(make_settlement(sens, period_hours=DELTA))
    money = settle(price, out["award"], jnp.zeros(N_AGENT),
                   ENERGY_PRICE, jnp.full(N_AGENT, CYCLE_COST))

    curtailment = DELTA * sens.base_mva * VOLL * float(np.asarray(out["shed"]).sum())
    paid_out = float(np.asarray(money["revenue"]).sum())
    np.testing.assert_allclose(paid_out, float(out["z"]) - curtailment, rtol=1e-9)


def test_paying_on_the_offered_quantity_breaks_the_identity(market):
    """The identity has teeth against the first error §8 names.

    Substituting the offered quantity for the cleared one is the mistake that a
    reader of §5 could plausibly make, and it leaves every reported figure
    finite and positive.
    """
    price, out, spec, sens = clear_one(market, seed=0)
    settle = make_settlement(sens, period_hours=DELTA)

    offered = np.full(N_AGENT, 0.02)
    assert float(np.asarray(out["award"]).sum()) < offered.sum(), \
        "every offer cleared in full, so this substitution changes nothing"

    wrong = settle(price, offered, jnp.zeros(N_AGENT), ENERGY_PRICE,
                   jnp.full(N_AGENT, CYCLE_COST))
    curtailment = DELTA * sens.base_mva * VOLL * float(np.asarray(out["shed"]).sum())
    assert abs(float(np.asarray(wrong["revenue"]).sum())
               - (float(out["z"]) - curtailment)) > 1.0


def test_hand_worked_arithmetic(market):
    """Two aggregators, one discharging and one charging less than it planned."""
    _, sens, _, _, _, _ = market
    settle = make_settlement(sens, period_hours=DELTA)
    base = DELTA * sens.base_mva

    price = jnp.asarray([50.0, 30.0])
    award = jnp.asarray([0.04, 0.01])
    plan = jnp.asarray([0.01, 0.03])        # agent 0 discharges, agent 1 charges less
    money = settle(price, award, plan, 40.0, jnp.asarray([3.0, 3.0]))

    np.testing.assert_allclose(np.asarray(money["p_dis"]), [0.03, 0.0], atol=1e-15)
    np.testing.assert_allclose(np.asarray(money["p_ch"]), [0.0, 0.02], atol=1e-15)
    np.testing.assert_allclose(np.asarray(money["revenue"]),
                               [base * 50.0 * 0.04, base * 30.0 * 0.01], rtol=1e-12)
    # agent 0 buys nothing back and pays degradation on 0.03 of throughput;
    # agent 1 buys 0.02 at 40 and pays degradation on the same 0.02
    np.testing.assert_allclose(np.asarray(money["cost"]),
                               [base * 3.0 * 0.03,
                                base * (40.0 * 0.02 + 3.0 * 0.02)], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(money["profit"]),
                               np.asarray(money["revenue"]) - np.asarray(money["cost"]),
                               rtol=1e-12)
    np.testing.assert_array_equal(np.asarray(money["reward"]),
                                  np.asarray(money["profit"]))


def test_exactly_one_of_the_two_battery_powers_is_positive(market):
    """§9.5 splits the net injection; a period cannot both charge and discharge."""
    _, sens, _, _, _, _ = market
    settle = make_settlement(sens, period_hours=DELTA)
    rng = np.random.default_rng(3)
    award = jnp.asarray(rng.uniform(0.0, 0.05, 64))
    plan = jnp.asarray(rng.uniform(0.0, 0.05, 64))

    money = settle(jnp.full(64, 50.0), award, plan, 40.0, jnp.full(64, 3.0))
    p_ch, p_dis = np.asarray(money["p_ch"]), np.asarray(money["p_dis"])

    assert (p_ch >= 0.0).all() and (p_dis >= 0.0).all()
    assert (p_ch * p_dis == 0.0).all()
    np.testing.assert_allclose(p_dis - p_ch, np.asarray(award) - np.asarray(plan),
                               rtol=1e-12)


def test_revenue_follows_the_award_and_the_participant_s_own_price(market):
    """Doubling one price doubles that participant's revenue and no one else's.

    The award is held fixed here on purpose: this is the settlement's property,
    not the clearing's, and mixing the two would let a change in the accepted
    set hide a change in the payment rule.
    """
    _, sens, _, _, _, _ = market
    settle = make_settlement(sens, period_hours=DELTA)
    price = jnp.asarray([50.0, 30.0, 70.0])
    award = jnp.asarray([0.04, 0.01, 0.02])
    args = (jnp.zeros(3), 40.0, jnp.full(3, 3.0))

    base = np.asarray(settle(price, award, *args)["revenue"])
    doubled = np.asarray(settle(price.at[1].multiply(2.0), award, *args)["revenue"])

    np.testing.assert_allclose(doubled[1], 2.0 * base[1], rtol=1e-12)
    np.testing.assert_allclose(np.delete(doubled, 1), np.delete(base, 1), rtol=1e-12)


def test_an_unawarded_participant_receives_nothing_and_pays_only_for_its_plan(market):
    """A participant that clears nothing still charges as it planned, and pays for it."""
    _, sens, _, _, _, _ = market
    settle = make_settlement(sens, period_hours=DELTA)
    money = settle(jnp.asarray([50.0]), jnp.asarray([0.0]), jnp.asarray([0.03]),
                   40.0, jnp.asarray([3.0]))

    assert float(np.asarray(money["revenue"])[0]) == 0.0
    assert float(np.asarray(money["p_dis"])[0]) == 0.0
    np.testing.assert_allclose(np.asarray(money["p_ch"]), [0.03], rtol=1e-12)
    assert float(np.asarray(money["profit"])[0]) < 0.0


def test_voll_appears_in_no_settlement_expression():
    """`voll` is a coefficient of the clearing objective and never
    a settlement price.  Checked on the syntax tree so the module docstring may
    say so in prose; string constants are excluded for the same reason.
    """
    tree = ast.parse(pathlib.Path(settlement_mod.__file__).read_text())
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    used |= {n.name for n in ast.walk(tree) if isinstance(n, ast.alias)}
    offenders = sorted(n for n in used if "voll" in n.lower())
    assert not offenders, f"voll reached a settlement expression: {offenders}"


def test_no_feasibility_quantity_appears_in_a_settlement_expression():
    """`cost` and `costs` differ by a letter and by their channel.

    A violation depth or a curtailed quantity reaching `cost` would travel into
    `reward` through `profit`, which the market layer forbids and which the
    repository has recorded as an actual conflict once.
    """
    tree = ast.parse(pathlib.Path(settlement_mod.__file__).read_text())
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    banned = ("shed", "overload", "v_under", "v_over", "violation", "penalty")
    offenders = sorted(n for n in used if any(b in n.lower() for b in banned))
    assert not offenders, f"a feasibility quantity reached the settlement: {offenders}"
