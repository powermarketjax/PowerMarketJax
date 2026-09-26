"""L1 domain correctness for the day-ahead settlement (§8).

Three kinds of check, in order of how much they can catch.

*Independent references.*  A hand-worked two-bus day whose every term -- energy,
no-load, start-up, revenue at two different prices -- is computable on paper, and
a `scipy.integrate.quad` of the marginal-cost curve on the real case.  The
quadrature is the check that matters most: the MATPOWER convention would give a
finite, plausible cost that no aggregate identity notices (§4, §12).

*The money-balance identity of §8, split per agent.*  The identity itself is
already an L1 check on the clearing; what settlement adds is that the sum of the
agent revenues reproduces the payment side of it after regrouping units by bus.
That is what a wrong `bus(i)` lookup or a transposed `lmp` breaks, and it is
invisible in any single-bus test.

*Absence of `voll`.*  It is forbidden in any settlement expression and
until settlement existed there was nothing to check.  Checked on the syntax tree,
since a comment mentioning VOLL -- and settlement.py has one -- must not trip it.

x64 per the L0 module docstring: other modules in this suite turn it off.
"""
import ast
import pathlib
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from powermarketjax.case import load_case
from powermarketjax.envs.day_ahead import make_clearing, make_settlement, segment_costs
from powermarketjax.envs.day_ahead import settlement as settlement_mod

T = 4
K = 1
CAP_SCALE = 0.6        # adopted scenario; real ratings never bind on case29gb
#: Since 2026-08-17: registered ramp rates undiscounted, so this factor is
#: the identity.  The 2 572 MW of first-period shedding this module used to see
#: at ramp_scale = 0.25 was a property of that factor, not of settlement; see
#: the re-measured value in the RAMP_OFF note below.
RAMP_SCALE = 1.0
#: "ramp effectively off", as in the L1 clearing module.  What this is contrasted
#: against changed with the scenario: at ramp_scale = 0.25 the first period could
#: not ramp up from p_init in time and 2 572 MW was shed, pricing that period at
#: VOLL through the balance dual and making every unit hugely profitable.  At the
#: adopted ramp_scale = 1.0 that mechanism is no longer reached.  Re-measured
#: 2026-08-17 at cap 0.6, T = 4, demand 33 466 MW: period-0 shed is 2 571.8 MW at
#: ramp 0.25 and **0.0 MW** at ramp 1.0, and the highest LMP falls from 10 000
#: (VOLL) to 90.12 \$/MWh.  So RAMP_OFF and RAMP_SCALE now differ only in whether
#: a ramp row can bind at all, not in whether load is shed -- and no test in this
#: module may rely on the shedding regime without constructing it.
RAMP_OFF = 2.0
DEMAND = 33466.0       # MW; see the note below
#: 33 466 MW was the median of the `Actual` column that §14 has since discarded
#: as the wrong basis (2026-08-09).  It is kept as the scenario level because
#: every tolerance in this module is calibrated at it, and it remains a GB
#: demand level: it sits at the 78.5th percentile of the realised series §14
#: now uses, whose median is 27 542 MW.


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


def _clear_and_settle(case, ramp_scale):
    clear, spec = make_clearing(case, T, n_segments=K, cap_scale=CAP_SCALE,
                                ramp_scale=ramp_scale)
    _, cost = segment_costs(case, K)
    u = jnp.ones((spec["n_units"], T))
    p_init = jnp.asarray(spec["p_min"]) * u[:, 0]
    offer = jnp.broadcast_to(jnp.asarray(cost)[:, :, None], cost.shape + (T,))
    out = jax.jit(clear)(offer, u, jnp.full((T,), DEMAND), p_init)
    assert float(out["mu"]) < 1e-8, "solve did not converge; prices unusable"
    money = make_settlement(case)(out["award"], out["lmp"], u, u[:, -1])
    return out, spec, u, money


@pytest.fixture(scope="module")
def cleared(case):
    """One cleared day on the primary case at the registered scenario, plus its
    settlement.  It sheds in the first period, so the shed-bound dual is live and
    the money-balance check below exercises that term."""
    return _clear_and_settle(case, RAMP_SCALE)


@pytest.fixture(scope="module")
def cleared_unstressed(case):
    """The same day without the ramp coupling, so nothing sheds and prices stay
    in the 14-121 \\$/MWh band rather than reaching VOLL."""
    return _clear_and_settle(case, RAMP_OFF)


# --------------------------------------------------------------------------
# Money balance, split per agent
# --------------------------------------------------------------------------

def test_agent_revenues_reproduce_the_payment_side(cleared, case):
    """Sum of agent revenues == sum over buses of price times generation.

    The two sides group the same money differently -- by unit and by bus -- so a
    wrong `bus(i)` or a transposed `lmp` shows up here and nowhere else.
    """
    out, spec, _, money = cleared
    lmp, award = np.asarray(out["lmp"]), np.asarray(out["award"])
    gen = np.zeros((T, spec["n_buses"]))
    np.add.at(gen, (slice(None), spec["unit_bus"]), award.T)
    np.testing.assert_allclose(float(np.asarray(money["revenue"]).sum()),
                               float((lmp * gen).sum()), rtol=1e-12)


def test_payments_do_not_exceed_charges(cleared, case):
    """§8: charges minus payments is the congestion plus shed-bound rent, and
    the congestion part is nonnegative, so load pays at least what generation
    earns.  This is the settlement's half of the identity already checked on the
    clearing; the duals it would take to close it exactly are not returned by
    `clear`."""
    out, spec, _, money = cleared
    lmp, shed = np.asarray(out["lmp"]), np.asarray(out["shed"])
    demand = spec["demand_share"][None, :] * DEMAND
    charges = float((lmp * (demand - shed)).sum())
    payments = float(np.asarray(money["revenue"]).sum())
    assert charges - payments > -1e-6 * max(abs(charges), 1.0)


def test_profit_is_revenue_minus_cost(cleared):
    out, _, _, money = cleared
    np.testing.assert_allclose(np.asarray(money["profit"]),
                               np.asarray(money["revenue"]) - np.asarray(money["cost"]),
                               rtol=1e-12)
    np.testing.assert_array_equal(np.asarray(money["reward"]),
                                  np.asarray(money["profit"]))


def test_no_make_whole_payment(cleared_unstressed):
    """§8 omits uplift deliberately, so a committed unit can lose money over the
    day.  Measured on the unstressed day: 61 of the 66 units do, the worst by
    786 555 \\$.  Those are the units held at p_min whose true marginal cost is
    above the clearing price, and which pay no-load on top of that.

    Not checked on the shedding day: there the balance dual is VOLL for a whole
    period and every unit clears an enormous profit, which would make this pass
    for the wrong reason.
    """
    _, _, _, money = cleared_unstressed
    assert (np.asarray(money["profit"]) < 0).sum() > 0.5 * len(money["profit"])


# --------------------------------------------------------------------------
# Cost against an independent quadrature
# --------------------------------------------------------------------------

def test_energy_cost_is_the_integral_of_marginal_cost(case, cleared):
    """Quadrature of MC(p) = a p^2 + b p + c, unit by unit, period by period.

    Independent of both the implementation and its closed form.  Under the
    MATPOWER reading (TC = a p^2 + b p + c) the two disagree by orders of
    magnitude while both stay finite, which is the failure §4 warns about.
    """
    out, _, u, _ = cleared
    award = np.asarray(out["award"])
    a = np.asarray(case.unit_cost_a, np.float64)
    b = np.asarray(case.unit_cost_b, np.float64)
    c = np.asarray(case.unit_cost_c, np.float64)

    # settle with no-load and start-up removed, isolating the energy term
    bare = types.SimpleNamespace(
        unit_p_min=case.unit_p_min, unit_node_idx=case.unit_node_idx,
        unit_cost_a=a, unit_cost_b=b, unit_cost_c=c,
        unit_no_load_cost=np.zeros_like(a), unit_startup_cost=np.zeros_like(a))
    energy = np.asarray(make_settlement(bare)(
        out["award"], out["lmp"], u, u[:, -1])["cost"])

    quadrature = np.array([
        sum(quad(lambda p: a[i] * p ** 2 + b[i] * p + c[i], 0.0, award[i, t])[0]
            for t in range(T))
        for i in range(len(a))])
    np.testing.assert_allclose(energy, quadrature, rtol=1e-9)

    matpower = np.array([
        sum(a[i] * award[i, t] ** 2 + b[i] * award[i, t] + c[i] for t in range(T))
        for i in range(len(a))])
    assert np.abs(matpower - quadrature).max() > 1.0, "the two conventions must differ"


# --------------------------------------------------------------------------
# Hand-worked two-bus day (§16 uses the same example to pin the price)
# --------------------------------------------------------------------------

def _two_bus():
    """The §16 two-bus example, with no-load and start-up costs added.

    Flat marginal costs (a = b = 0) so that TC(p) = c p exactly and every term
    below is arithmetic: cheap unit at bus 0 offering 10 \\$/MWh, dear unit at
    bus 1 offering 50, all 60 MW of load at bus 1, and a line rated 40 so the
    dear unit must cover the remaining 20 MW.
    """
    return types.SimpleNamespace(
        n_nodes=2,
        unit_p_min=np.array([0.0, 0.0]),
        unit_p_max=np.array([100.0, 100.0]),
        unit_cost_a=np.array([0.0, 0.0]),
        unit_cost_b=np.array([0.0, 0.0]),
        unit_cost_c=np.array([10.0, 50.0]),
        unit_no_load_cost=np.array([1.0, 2.0]),      # $/h
        unit_startup_cost=np.array([100.0, 200.0]),  # $
        unit_node_idx=np.array([0, 1]),
        PTDF=np.array([[0.0, -1.0]]),
        line_cap=np.array([40.0]),
        node_pd=np.array([0.0, 1.0]),
        unit_ramp_up=np.array([1.0, 1.0]),
        unit_ramp_down=np.array([1.0, 1.0]),
    )


def _clear_two_bus(n_periods, u, markup=1.0, period_hours=1.0):
    case = _two_bus()
    clear, _ = make_clearing(case, n_periods, n_segments=1, cap_scale=1.0,
                             ramp_scale=100.0, period_hours=period_hours)
    _, cost = segment_costs(case, 1)
    offer = jnp.broadcast_to(jnp.asarray(cost)[:, :, None] * markup,
                             cost.shape + (n_periods,))
    out = jax.jit(clear)(offer, u, jnp.full((n_periods,), 60.0), jnp.zeros(2))
    assert float(out["mu"]) < 1e-8
    return case, out


def test_two_bus_settlement_by_hand():
    """One period, both units starting from cold.

    award = [40, 20] and lmp = [10, 50] (§16).  Then
        unit 0: revenue 10*40 = 400, energy 10*40 = 400, no-load 1, start-up 100
        unit 1: revenue 50*20 = 1000, energy 50*20 = 1000, no-load 2, start-up 200
    """
    u = jnp.ones((2, 1))
    case, out = _clear_two_bus(1, u)
    money = make_settlement(case)(out["award"], out["lmp"], u, jnp.zeros(2))
    np.testing.assert_allclose(np.asarray(money["revenue"]), [400.0, 1000.0], atol=1e-4)
    np.testing.assert_allclose(np.asarray(money["cost"]), [501.0, 1202.0], atol=1e-4)
    np.testing.assert_allclose(np.asarray(money["profit"]), [-101.0, -202.0], atol=1e-4)


def test_start_up_is_not_charged_to_a_unit_already_on():
    """Same day, but both units were on at the end of the previous one."""
    u = jnp.ones((2, 1))
    case, out = _clear_two_bus(1, u)
    money = make_settlement(case)(out["award"], out["lmp"], u, jnp.ones(2))
    np.testing.assert_allclose(np.asarray(money["profit"]), [-1.0, -2.0], atol=1e-4)


def test_start_up_charged_once_per_start():
    """Three periods, the cheap unit off in the middle one and restarting.

    Periods 1 and 3 clear as above; in period 2 the cheap unit is off, the dear
    unit covers all 60 MW, the line carries nothing and both buses price at 50.

        unit 0: revenue 400 + 0 + 400,  energy 10*80, no-load 1*2, start-up 100
        unit 1: revenue 1000 + 3000 + 1000, energy 50*100, no-load 2*3, none

    The tolerance is 0.1 \\$ rather than exact because of OFF_EPS: the
    de-committed unit keeps a 1e-3 MW phantom segment, which is zeroed out of
    `award` but not out of the balance, so the dear unit clears 59.999 MW in the
    middle period.  At 50 \\$/MWh that is 0.05 \\$ on both its revenue and its
    energy cost, and exactly zero on its profit.
    """
    u = jnp.array([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    case, out = _clear_two_bus(3, u)
    np.testing.assert_allclose(np.asarray(out["award"]),
                               [[40.0, 0.0, 40.0], [20.0, 60.0, 20.0]], atol=2e-3)
    money = make_settlement(case)(out["award"], out["lmp"], u, jnp.ones(2))
    np.testing.assert_allclose(np.asarray(money["revenue"]), [800.0, 5000.0], atol=0.1)
    np.testing.assert_allclose(np.asarray(money["cost"]), [902.0, 5006.0], atol=0.1)
    np.testing.assert_allclose(np.asarray(money["profit"]), [-102.0, -6.0], atol=1e-3)


def test_reward_follows_the_award_and_the_true_cost_not_the_offer():
    """Both units offer at twice their true cost.

    The merit order is unchanged, so the award is still [40, 20], but the prices
    double to [20, 100].  Revenue therefore doubles while cost does not move:
    profit is +299 and +798 where truthful bidding lost 101 and 202.  A
    settlement that read the offer, or that valued the award at the offer price,
    could not produce these numbers.
    """
    u = jnp.ones((2, 1))
    case, out = _clear_two_bus(1, u, markup=2.0)
    np.testing.assert_allclose(np.asarray(out["award"])[:, 0], [40.0, 20.0], atol=1e-5)
    np.testing.assert_allclose(np.asarray(out["lmp"])[0], [20.0, 100.0], atol=1e-4)
    money = make_settlement(case)(out["award"], out["lmp"], u, jnp.zeros(2))
    np.testing.assert_allclose(np.asarray(money["revenue"]), [800.0, 2000.0], atol=1e-3)
    np.testing.assert_allclose(np.asarray(money["cost"]), [501.0, 1202.0], atol=1e-3)
    np.testing.assert_allclose(np.asarray(money["profit"]), [299.0, 798.0], atol=1e-3)


def test_period_length_scales_the_rate_terms_only():
    """At delta = 2 h everything charged per hour doubles; start-up does not.

    Start-up is a \\$ figure per start (§3.2), the other three are rates, so this
    separates them: profit goes from -101 to 2*(400 - 400 - 1) - 100 = -102.
    """
    u = jnp.ones((2, 1))
    case, out = _clear_two_bus(1, u, period_hours=2.0)
    money = make_settlement(case, period_hours=2.0)(out["award"], out["lmp"], u,
                                                    jnp.zeros(2))
    np.testing.assert_allclose(np.asarray(money["revenue"]), [800.0, 2000.0], atol=1e-3)
    np.testing.assert_allclose(np.asarray(money["cost"]), [902.0, 2204.0], atol=1e-3)
    np.testing.assert_allclose(np.asarray(money["profit"]), [-102.0, -204.0], atol=1e-3)


def test_agents_owning_several_units_add_up():
    """A `unit_to_agent` that puts both units under one agent."""
    u = jnp.ones((2, 1))
    case, out = _clear_two_bus(1, u)
    args = (out["award"], out["lmp"], u, jnp.zeros(2))
    one = make_settlement(case, unit_to_agent=np.array([0, 0]))(*args)
    per_unit = make_settlement(case)(*args)
    for k in one:
        assert one[k].shape == (1,)
        np.testing.assert_allclose(np.asarray(one[k])[0],
                                   float(np.asarray(per_unit[k]).sum()), rtol=1e-12)


# --------------------------------------------------------------------------
# voll must not appear in a settlement expression
# --------------------------------------------------------------------------

def test_voll_appears_in_no_settlement_expression():
    """`voll` is a cost parameter of the clearing objective and
    never a settlement price.  Checked on the syntax tree, so that the module
    docstring may say so in prose without tripping the check; string constants
    are excluded for the same reason.
    """
    tree = ast.parse(pathlib.Path(settlement_mod.__file__).read_text())
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    used |= {n.name for n in ast.walk(tree) if isinstance(n, ast.alias)}
    offenders = sorted(n for n in used if "voll" in n.lower())
    assert not offenders, f"voll reached a settlement expression: {offenders}"
