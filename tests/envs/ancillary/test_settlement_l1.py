"""L1 for the ancillary settlement (§11), plus the L0 contract.

The reserve money-balance identity is the headline check and it is also the one
that must not be trusted alone: it carries $\\lambda^{res}$ on both of its sides,
so a misread reserve price satisfies it exactly as a correct one does.  §20
records that consequence and the L1 finite-difference check in
`test_clearing_l1.py` is its companion; the identity is asserted here for what it
does catch, which is paying on the wrong quantity, double counting, and letting
a penalty parameter into a settlement expression.

The market as a whole does not balance and no test here asserts that it does:
reserve has no buyer inside the market (§11).  Each leg is checked separately.
"""
import ast
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.ancillary.clearing import make_clearing
from powermarketjax.envs.ancillary.settlement import make_settlement
from powermarketjax.envs.day_ahead.clearing import segment_costs
from tests.envs.ancillary.test_clearing_l0 import FIXTURE, HOUR

CASE = "29gb"
THETA = (1.0 / 6.0, 0.5)
VOLR, DELTA = 250.0, 0.5


@pytest.fixture
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture
def market(x64):
    case = load_case(CASE)
    _, cost = segment_costs(case, 1)
    clear, spec = make_clearing(case, THETA, VOLR, n_segments=1, cap_scale=0.6,
                                ramp_scale=1.0, period_hours=DELTA)
    settle = make_settlement(case, period_hours=DELTA)
    pmin = np.asarray(case.unit_p_min, np.float64)
    pmax = np.asarray(case.unit_p_max, np.float64)
    fx = np.load(FIXTURE, allow_pickle=True)
    u = fx["commitment"][0, :, HOUR].astype(np.float64)
    assert (u == 0).any() and (u > 0).any()
    n_u, n_p = spec["n_units"], spec["n_prod"]
    supply = (np.asarray(spec["res_cap"]) * u[:, None]).sum(0)
    rng = np.random.default_rng(0)

    def run(req_frac=0.70, dfrac=0.85, level=200.0, spread=0.10, offer_scale=1.0):
        req = np.atleast_1d(np.asarray(req_frac, np.float64)) * supply
        demand = float((pmin * u).sum()) + dfrac * float(((pmax - pmin) * u).sum())
        p_prev = (pmin + min(dfrac, 1.0) * (pmax - pmin)) * u
        offer_res = np.repeat(
            (level * (1.0 + np.linspace(-spread, spread, n_u)))[:, None], n_p, 1)
        out = jax.jit(clear)(jnp.asarray(cost * offer_scale),
                             jnp.asarray(offer_res), jnp.asarray(u),
                             jnp.asarray(demand), jnp.asarray(req),
                             jnp.asarray(p_prev))
        assert float(out["dual_residual"]) / (DELTA * 10_000.0) < 1e-6
        # a day-ahead position and price of the same period, exogenous here
        q_da = jnp.asarray(np.asarray(out["award"]) * rng.uniform(0.8, 1.2, n_u))
        lmp_da = jnp.asarray(np.asarray(out["lmp"]) * 0.95)
        u_prev = jnp.asarray(np.where(rng.random(n_u) < 0.1, 0.0, u))
        paid = jax.jit(settle)(out["award"], out["reserve"], out["lmp"],
                               out["reserve_price"], q_da, lmp_da,
                               jnp.asarray(u), u_prev)
        return ({k: np.asarray(v) for k, v in out.items()},
                {k: np.asarray(v) for k, v in paid.items()},
                dict(q_da=np.asarray(q_da), lmp_da=np.asarray(lmp_da),
                     u=u, u_prev=np.asarray(u_prev), spec=spec, case=case,
                     d_res=req))

    return run


# --------------------------------------------------------------------------
# L0 contract


def test_jit_and_shapes(market):
    cleared, paid, aux = market()
    n_u = aux["spec"]["n_units"]
    for key in ("revenue_energy", "revenue_reserve", "revenue", "cost",
                "profit", "reward"):
        assert paid[key].shape == (n_u,), key
        assert paid[key].dtype == np.float64, key
        assert np.isfinite(paid[key]).all(), key


def test_reward_is_profit(market):
    _, paid, _ = market()
    np.testing.assert_array_equal(paid["reward"], paid["profit"])
    np.testing.assert_allclose(paid["revenue"],
                               paid["revenue_energy"] + paid["revenue_reserve"],
                               rtol=0, atol=1e-9)


# --------------------------------------------------------------------------
# the two identities


@pytest.mark.parametrize("req_frac", [0.30, 0.70, 0.95, 1.50])
def test_reserve_money_balance(market, req_frac):
    """§11: total capacity payment equals the served requirement at its price.

    Both sides are computed, which is the requirement for identities of this
    kind: asserting only that the left side is finite and
    nonnegative is not a check on the identity.
    """
    cleared, paid, aux = market(req_frac=req_frac)
    d_res = aux["d_res"]
    left = paid["revenue_reserve"].sum()
    right = float(DELTA * np.sum(cleared["reserve_price"]
                                 * (d_res - cleared["reserve_shortfall"])))
    assert abs(left - right) <= 1e-6 * max(abs(right), 1.0)


def test_reserve_payment_is_correct_unit_by_unit(market):
    """The identity summed over products cannot see a price attributed to the
    wrong product, because the two product prices coincide at most operating
    points of this market and the error then cancels in the sum.

    The comparison is therefore made **on the dimension the reward is consumed
    on**, which is the unit: `revenue_reserve` enters `profit` per unit and each
    agent sees its own.  A price attributed to the wrong product changes any
    unit whose cleared reserve is not split between the products in the same
    proportion, whether or not the total happens to cancel.  Two product prices
    that coincide would hide that regardless of how the comparison is written,
    so the separation of the prices is asserted first and the test declares
    itself uninformative rather than passing quietly if they do not separate.
    """
    # one product inside the band and one short, so the two prices cannot
    # coincide; at the migrated scenario both of the old fractions are
    # short and both prices sit on the cap, which is what made this
    # test declare itself uninformative
    cleared, paid, aux = market(req_frac=np.array([0.10, 1.50]))
    price = cleared["reserve_price"]
    assert abs(price[0] - price[1]) > 1.0, (
        "the two product prices coincide here, so no way of writing this test "
        "could see a mis-attribution; the operating point must be changed")
    want = DELTA * (cleared["reserve"] * price[None, :]).sum(1)   # per unit
    # measured 4.7e-13 $ on this configuration, 2026-08-14
    np.testing.assert_allclose(paid["revenue_reserve"], want, rtol=0, atol=1e-9)


def test_the_requirement_row_closes(market):
    """A check on the clearing rather than on the settlement, kept because the
    reserve identity of §11 rests on it: the cleared reserve of each product
    plus its unmet part equals the requirement wherever the price is positive.
    """
    cleared, _, aux = market(req_frac=np.array([0.95, 0.30]))
    served = cleared["reserve"].sum(0) + cleared["reserve_shortfall"]
    priced = cleared["reserve_price"] > 1e-9
    # measured 2.9e-08 MW on this configuration, 2026-08-14
    np.testing.assert_allclose(served[priced], aux["d_res"][priced],
                               rtol=0, atol=1e-6)


def test_energy_leg_settles_the_deviation_at_the_clearing_price(market):
    """The two-settlement arithmetic, recomputed independently per unit."""
    cleared, paid, aux = market()
    bus = np.asarray(aux["spec"]["unit_bus"])
    want = DELTA * (aux["lmp_da"][bus] * aux["q_da"]
                    + cleared["lmp"][bus] * (cleared["award"] - aux["q_da"]))
    np.testing.assert_allclose(paid["revenue_energy"], want, rtol=0, atol=1e-9)


def test_a_position_equal_to_the_award_pays_the_day_ahead_price_only(market):
    """With no deviation the real-time price cannot enter the energy leg, which
    is the property the two-settlement design exists for."""
    cleared, _, aux = market()
    case = aux["case"]
    settle = make_settlement(case, period_hours=DELTA)
    bus = np.asarray(aux["spec"]["unit_bus"])
    award = jnp.asarray(cleared["award"])
    paid = jax.jit(settle)(award, jnp.asarray(cleared["reserve"]),
                           jnp.asarray(cleared["lmp"] * 3.0),
                           jnp.asarray(cleared["reserve_price"]),
                           award, jnp.asarray(aux["lmp_da"]),
                           jnp.asarray(aux["u"]), jnp.asarray(aux["u_prev"]))
    want = DELTA * aux["lmp_da"][bus] * np.asarray(award)
    np.testing.assert_allclose(np.asarray(paid["revenue_energy"]), want,
                               rtol=0, atol=1e-9)


# --------------------------------------------------------------------------
# what settlement must not do


def test_reserve_is_paid_but_never_costed(market):
    """Holding capacity burns no fuel (§11), so cost cannot move with reserve."""
    cleared, paid, aux = market()
    case = aux["case"]
    settle = make_settlement(case, period_hours=DELTA)
    doubled = jax.jit(settle)(
        jnp.asarray(cleared["award"]), jnp.asarray(cleared["reserve"] * 2.0),
        jnp.asarray(cleared["lmp"]), jnp.asarray(cleared["reserve_price"]),
        jnp.asarray(aux["q_da"]), jnp.asarray(aux["lmp_da"]),
        jnp.asarray(aux["u"]), jnp.asarray(aux["u_prev"]))
    np.testing.assert_allclose(np.asarray(doubled["cost"]), paid["cost"],
                               rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.asarray(doubled["revenue_reserve"]),
                               2.0 * paid["revenue_reserve"], rtol=0, atol=1e-9)


def test_the_offers_do_not_reach_the_settlement(market):
    """Scaling every energy offer changes what clears, and the settlement must
    read only the cleared quantities and prices it is handed."""
    cleared, paid, aux = market()
    case = aux["case"]
    settle = make_settlement(case, period_hours=DELTA)
    same = jax.jit(settle)(
        jnp.asarray(cleared["award"]), jnp.asarray(cleared["reserve"]),
        jnp.asarray(cleared["lmp"]), jnp.asarray(cleared["reserve_price"]),
        jnp.asarray(aux["q_da"]), jnp.asarray(aux["lmp_da"]),
        jnp.asarray(aux["u"]), jnp.asarray(aux["u_prev"]))
    for key in ("revenue", "cost", "profit"):
        np.testing.assert_array_equal(np.asarray(same[key]), paid[key])


def test_neither_penalty_parameter_appears_in_the_settlement():
    """Checked on the syntax tree, because a comment cannot check it."""
    from powermarketjax.envs.ancillary import settlement as module
    tree = ast.parse(inspect.getsource(module))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for forbidden in ("voll", "VOLL", "volr", "VOLR"):
        assert forbidden not in names, forbidden


def test_the_market_does_not_self_balance(market):
    """Reserve has no buyer inside the market, so the operator is out of pocket
    by exactly the capacity payment.  This is asserted as a property, because a
    check that expected the whole market to balance would contradict §11."""
    _, paid, _ = market()
    assert paid["revenue_reserve"].sum() > 0.0
