"""L1 domain correctness for the joint clearing operator (§7, §9).

Two groups.  The primal group checks that what comes back satisfies the
constraints the specification writes, including the three rows this market adds
to the real-time clearing.  The price group checks the duals, and its central
member is the finite difference on the reserve price, which §20 records as the
only check that catches a wrong price readout: both money-balance identities
hold under a misread dual, for the same reason the day-ahead money balance held
under a broken price formula.

The finite difference carries two preconditions rather than a hand-picked step.
It asserts that the solve is clean, because a solve whose stationarity residual
has stalled returns a feasible but suboptimal point whose objective is wrong by
enough to swamp the signal; and it asserts that the value function is locally
linear, by taking the difference at two steps and requiring them to agree, so
that a step large enough to cross a kink is caught rather than silently
measuring something else.  Both come from measurement.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.ancillary.clearing import OFF_EPS, VOLL, make_clearing
from powermarketjax.envs.day_ahead.clearing import segment_costs
from tests.envs.ancillary.test_clearing_l0 import FIXTURE, HOUR

CASE = "29gb"
THETA = (1.0 / 6.0, 0.5)
VOLR = 250.0
#: Requirement as a fraction of what the committed fleet may sell, for the tests
#: that need the price to sit **inside** the band between zero and the cap.
#:
#: It is a scenario-dependent choice and it moved once already: 0.70 was in the
#: band at `ramp_scale = 0.25` and `VOLR = 1000`, and at `ramp_scale = 1.00` the
#: same fraction is four times the absolute requirement while the cap is four
#: times lower, so it lands on the cap instead and the band tests stop testing a
#: band.  Three conditions have to hold at once for a fraction to be usable
#: here, and all three are asserted by the tests that use it rather than assumed:
#: the requirement is met (no shortage), the price is strictly below `VOLR`
#: (the cap is not what is being observed), and the price is strictly above zero
#: (there is something to observe at all).
#:
#: **The band is not reachable by moving the requirement alone at this cell**,
#: which is why the energy loading is part of the run point rather than left at
#: the file's default.  Measured at `cap 0.6 / ramp 1.00 / VOLR 250`, sweeping
#: the requirement at `dfrac = 0.85` gives a price of 2e-14 up to 0.25 of
#: deliverable reserve and 250.0 from 0.26 onwards -- it steps from free to the
#: cap with nothing in between, because a generous `ramp_scale` makes reserve
#: almost free until the capacity constraint binds, and then it is not expensive
#: but unavailable.  At `dfrac = 0.60` the fleet has room, the requirement can
#: be pushed to 0.87 of deliverable reserve without a shortage, and the price
#: lands at 18.59 -- inside the band and four orders below the cap.
#: **Per product, not one number for both.**  The two products have different
#: deliverable reserve (they differ by response time), so a single fraction of
#: `supply` applied to both puts one of them in shortage while the other is
#: still free -- and the shortage pins that product's price at the cap, which is
#: the thing these tests are trying not to observe.  The pair below is where the
#: adopted requirement rule lands at `dfrac = 0.60`: it is `beta = (0.15, 0.20)`
#: of the period's demand expressed as fractions of deliverable reserve.
#: Half-width of the reserve offer spread, the same value the L2 layer runs at
#: and for the same reason: adjacent providers must differ or the duals are in
#: the region this market documents as untrustworthy.
SPREAD = 0.10
BAND_DFRAC = 0.60
BAND_REQ_FRAC = np.array([0.874, 0.389])
BETA = np.array([0.020, 0.050])
DELTA = 0.5
#: Normalised stationarity residual above which the solve is not to be trusted.
#: Clean solves sit at 1e-16 to 1e-14 and the measured harmful ones at 3e-5 and
#: above, so this sits in the gap with four orders of margin on each side.
CLEAN = 1e-6
#: Taken from the L0 module rather than restated here, which is what the L2
#: module already does.  This file carried its own copy of both constants, and
#: the copy is how the scenario migration left it clearing at `cap_scale = 0.6`
#: on a commitment solved at 0.4: the migration moved every value it could find
#: and moved the L0 path, while this one sat behind an identical-looking
#: definition in another file.  A second definition of a scenario input is a
#: second thing to migrate, and nothing fails when only one of them moves.


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
    pmin = np.asarray(case.unit_p_min, np.float64)
    pmax = np.asarray(case.unit_p_max, np.float64)
    fx = np.load(FIXTURE, allow_pickle=True)
    u = fx["commitment"][0, :, HOUR].astype(np.float64)
    assert (u == 0).any(), \
        "this operating point commits every unit, so the de-committed assertions " \
        "would be assertions about the empty set"
    assert (u > 0).any(), "this operating point commits no unit"

    def run(dfrac=0.85, requirement=None, offer_res=None, offer_scale=1.0):
        demand = float((pmin * u).sum()) + dfrac * float(((pmax - pmin) * u).sum())
        d_res = BETA * demand if requirement is None else np.asarray(requirement)
        res = np.zeros((spec["n_units"], spec["n_prod"])) if offer_res is None \
            else np.asarray(offer_res)
        out = jax.jit(clear)(jnp.asarray(cost * offer_scale), jnp.asarray(res),
                             jnp.asarray(u), jnp.asarray(demand),
                             jnp.asarray(d_res),
                             # the previous period cannot have run a unit above
                             # its capacity, so the scenario knob is clipped
                             # here rather than handed to (RMP) as an
                             # infeasible boundary
                             jnp.asarray((pmin + min(dfrac, 1.0)
                                          * (pmax - pmin)) * u))
        return {k: np.asarray(v) for k, v in out.items()}, demand, d_res

    cnorm = DELTA * max(float(cost.max()), VOLL)
    # what the committed fleet may sell under (RA); requirements are set as
    # fractions of it, because the three price regimes are delimited by it
    supply = (np.asarray(spec["res_cap"]) * u[:, None]).sum(0)
    return dict(case=case, clear=clear, spec=spec, u=u, pmin=pmin, pmax=pmax,
                cost=cost, run=run, cnorm=cnorm, supply=supply)


def _residual(out, cnorm):
    return float(out["dual_residual"]) / cnorm


# --------------------------------------------------------------------------
# primal


def test_power_balance(market):
    out, demand, _ = market["run"]()
    served = out["award"].sum() + out["shed"].sum()
    assert abs(served - demand) < 1e-6 * demand


def test_capacity_sharing_holds_on_every_unit(market):
    """(CS): output plus all reserve fits under committed capacity."""
    out, _, _ = market["run"]()
    used = out["award"] + out["reserve"].sum(1)
    room = market["pmax"] * market["u"]
    assert np.max(used - room) <= 1e-6


def test_reserve_eligibility_holds_per_product(market):
    """(RA): a unit sells no more than it can ramp within the response time."""
    out, _, _ = market["run"]()
    cap = market["spec"]["res_cap"] * market["u"][:, None]
    assert np.max(out["reserve"] - cap) <= 1e-6


def test_requirement_row_holds(market):
    """(RD): cleared reserve plus the unmet part covers the requirement."""
    out, _, d_res = market["run"]()
    assert np.min(out["reserve"].sum(0) + out["reserve_shortfall"] - d_res) >= -1e-6


def test_de_committed_units_sell_nothing(market):
    """The OFF_EPS phantom quantities never leave the operator.

    The comparisons here are bitwise, which the tolerance discipline otherwise
    forbids on continuous quantities.  It is exact by construction: the operator
    multiplies all three reported quantities by `u`, and `u` is exactly 0 or 1.
    """
    out, _, _ = market["run"]()
    off = market["u"] == 0
    assert np.abs(out["award"][off]).max() == 0.0
    assert np.abs(out["reserve"][off]).max() == 0.0
    assert np.abs(out["capacity_dual"][off]).max() == 0.0


def test_output_lies_between_the_committed_bounds(market):
    out, _, _ = market["run"]()
    on = market["u"] > 0
    assert np.min(out["award"][on] - market["pmin"][on]) >= -1e-6
    assert np.max(out["award"][on] - market["pmax"][on]) <= 1e-6


def test_line_flows_are_within_their_ratings(market):
    out, demand, _ = market["run"]()
    spec, case = market["spec"], market["case"]
    share = spec["demand_share"]
    inj = np.zeros(spec["n_buses"])
    np.add.at(inj, spec["unit_bus"], out["award"])
    net = inj - (share * demand - out["shed"])
    flow = spec["PTDF"] @ net
    rating = np.asarray(case.line_cap, np.float64) * spec["cap_scale"]
    assert np.max(np.abs(flow) - rating) <= 1e-6 * rating.max()


def test_shed_never_exceeds_the_load_at_its_bus(market):
    out, demand, _ = market["run"]()
    load = market["spec"]["demand_share"] * demand
    assert np.max(out["shed"] - np.maximum(load, OFF_EPS)) <= 1e-6


# --------------------------------------------------------------------------
# prices


def test_reserve_price_lies_between_zero_and_volr(market):
    for dfrac, mult in ((0.60, 0.3), (0.85, 0.7), (0.85, 3.0)):
        out, _, d_res = market["run"](dfrac=dfrac,
                                      requirement=mult * market["supply"])
        lam = out["reserve_price"]
        assert np.min(lam) >= -1e-9
        assert np.max(lam) <= VOLR + 1e-6


def test_the_cap_is_reached_exactly_when_the_requirement_is_unmet(market):
    """§7: lambda_res = VOLR with equality exactly when s_res > 0."""
    seen_short = seen_met = False
    for mult in (BAND_REQ_FRAC, 0.20, 1.5, 3.0):
        out, _, _ = market["run"](requirement=mult * market["supply"])
        assert _residual(out, market["cnorm"]) < CLEAN
        for j in range(market["spec"]["n_prod"]):
            short = out["reserve_shortfall"][j] > 1e-6
            at_cap = out["reserve_price"][j] > VOLR - 1e-6
            # A requirement exactly at what the fleet can deliver is a legitimate
            # degenerate state: nothing is short while the price is free anywhere
            # in [marginal cost, VOLR], and the equivalence has no content there.
            # Assert that no test point sits on that edge, so that the choice of
            # `mult` is a checked precondition rather than luck.
            off_edge = (out["reserve_shortfall"][j] > 1e-3
                        or out["reserve_price"][j] < VOLR - 1e-3)
            assert off_edge, (
                f"point (mult={mult}, product {j}) sits on the degenerate edge "
                "where the requirement exactly exhausts deliverable reserve; the "
                "equivalence below is undefined there")
            assert short == at_cap, (mult, j, out["reserve_shortfall"], out["reserve_price"])
            seen_short |= short
            seen_met |= not short
    assert seen_short and seen_met, "the case distinction was never exercised"


def test_a_slack_requirement_prices_at_zero(market):
    """Reserve costs nothing when it displaces no energy and no offer prices it."""
    out, _, _ = market["run"](dfrac=0.60, requirement=np.array([10.0, 10.0]))
    assert _residual(out, market["cnorm"]) < CLEAN
    assert np.max(out["reserve_price"]) <= 1e-9


def test_energy_price_never_exceeds_voll(market):
    out, _, _ = market["run"]()
    assert np.max(out["lmp"]) <= VOLL + 1e-6


def test_energy_price_is_voll_where_load_is_shed(market):
    """Where shed is strictly interior its box duals vanish, so stationarity
    forces the price at that bus to VOLL exactly.  Which bus carries the shed is
    degenerate under a uniform VOLL, so the assertion is made on whichever buses
    do, not on a bus picked in advance."""
    out, demand, _ = market["run"](dfrac=1.30)
    load = market["spec"]["demand_share"] * demand
    interior = (out["shed"] > 1e-6) & (out["shed"] < load - 1e-6)
    assert interior.any(), "no bus shed an interior amount, the case is not exercised"
    assert np.allclose(out["lmp"][interior], VOLL, rtol=0, atol=1e-3)


def test_reserve_price_matches_the_derivative_of_the_objective(market):
    """§20's finite difference, with the two preconditions the measurements
    require: a clean solve, and a locally linear value function."""
    run = market["run"]
    base_req = BAND_REQ_FRAC * market["supply"]
    out, _, _ = run(dfrac=BAND_DFRAC, requirement=base_req)
    assert _residual(out, market["cnorm"]) < CLEAN
    lam = out["reserve_price"]
    assert np.max(lam) > 1e-3, "the priced band was not reached"
    assert np.max(lam) < VOLR - 1e-6, "the cap is binding, this is not the band"

    for j in range(market["spec"]["n_prod"]):
        slopes = []
        for eps in (5.0, 10.0):
            z = []
            for sign in (+1.0, -1.0):
                step = base_req.copy()
                step[j] += sign * eps
                side, _, _ = run(dfrac=BAND_DFRAC, requirement=step)
                assert _residual(side, market["cnorm"]) < CLEAN
                z.append(float(side["z"]))
            slopes.append((z[0] - z[1]) / (2.0 * eps) / DELTA)
        # local linearity: the two steps must agree before either is believed
        assert abs(slopes[0] - slopes[1]) <= 1e-6 * max(abs(slopes[0]), 1.0), \
            f"value function is not linear over the step used: {slopes}"
        assert abs(slopes[0] - lam[j]) <= 1e-6 * max(abs(slopes[0]), 1.0), \
            f"product {j}: finite difference {slopes[0]} against price {lam[j]}"


def test_capacity_dual_is_nonnegative_and_prices_committed_units_only(market):
    out, _, _ = market["run"]()
    assert np.min(out["capacity_dual"]) >= -1e-9
    assert np.abs(out["capacity_dual"][market["u"] == 0]).max() == 0.0


def test_raising_every_reserve_offer_together_raises_the_price(market):
    """A judgement on the price rather than on the quantity: with the same
    requirement, a uniform offer floor must appear in the clearing price."""
    n_u, n_p = market["spec"]["n_units"], market["spec"]["n_prod"]
    req = BAND_REQ_FRAC * market["supply"]
    # A *uniform* floor is an exact tie across all 66 providers, which is the
    # profile this market's duals are documented to be unreliable on -- and at
    # the band run point they are: a flat 5.0 leaves the normalised residual at
    # 1.9e-04, two orders above the cleanliness gate below.  The floor is
    # therefore given the same +-10% spread the L2 layer uses, which keeps the
    # quantity under test (a floor raises the price) and removes the degeneracy
    # that has nothing to do with it.
    tilt = (1.0 + np.linspace(-SPREAD, SPREAD, n_u))[:, None]
    low, _, _ = market["run"](dfrac=BAND_DFRAC, requirement=req,
                              offer_res=np.repeat(5.0 * tilt, n_p, axis=1))
    high, _, _ = market["run"](dfrac=BAND_DFRAC, requirement=req,
                               offer_res=np.repeat(60.0 * tilt, n_p, axis=1))
    assert _residual(low, market["cnorm"]) < CLEAN
    assert _residual(high, market["cnorm"]) < CLEAN
    assert np.max(low["reserve_price"]) < VOLR - 1e-6, \
        "the requirement is short even at the low offer, so the cap hides the effect"
    assert np.all(high["reserve_price"] >= low["reserve_price"] + 40.0)
