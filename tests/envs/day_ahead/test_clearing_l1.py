"""L1 domain correctness for the day-ahead clearing operator.

Power balance, capacity, ramp, line limits, the money-balance identity of §8,
VOLL pricing of shed load, and a hand-worked two-bus example pinning the sign of
the congestion term.

That last one is not optional.  A sign error in §7 is invisible in aggregate
money balance -- the identity holds either way -- and invisible on any run that
does not congest, which is every run on real `case29gb` ratings.  It would
invert the reward at every congested bus.  Exactly that error was once caught
by comparing against a finite difference.

**This file is the price side of the L1 split.**  Every
assertion here reads a dual: the pricing formula against a finite difference, the
money-balance identity term by term, the sign of the congestion component, and
the cap that keeps a fully shed bus from pricing above $\\mathrm{VOLL}$.  The
primal side and the settlement are in `test_env_l1.py`, and a new assertion
belongs on whichever side its quantity comes from, not in whichever file is open.
The split exists because the primal side is blind to a price error: the power
balance, the capacity bounds and the ramp limits all hold whatever the duals say.

Each side carries its own discriminating injection, since a group without one
cannot be told from a group of vacuous assertions.  On this side there are two:
deleting the shed-bound term of §7, which puts the finite-difference check at
15 553.96 against a derivative of 10 000; and rotating the congestion rent across
periods, which preserves every total by construction and so proves that the
summed form of the identity is blind rather than merely unlucky.

x64 per the L0 module docstring: other modules in this suite turn it off.
"""
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.day_ahead import OFF_EPS, VOLL, make_clearing, segment_costs

T = 4
K = 1
CAP_SCALE = 0.6        # adopted scenario; real ratings never bind on case29gb
#: Since 2026-08-17 the registered ramp rates are taken undiscounted, so
#: this factor is now the identity.  At Delta = 1 h that leaves the ramp rows
#: almost never binding and the day-ahead inter-temporal coupling is commitment
#: alone -- which is why the two tests below that are *about* a binding ramp
#: build their own run point at RAMP_BINDING instead of using this value.
RAMP_SCALE = 1.0
#: The scenario ramp factor this module used until 2026-08-17, kept as a run
#: point rather than a scenario: at RAMP_SCALE the start-up allowance of §6.3 is
#: not needed by any unit on case29gb, so a test asserting the allowance works
#: would pass while checking nothing.  A test needing that phenomenon has to
#: construct it, and each such test asserts up front that the
#: phenomenon is present rather than trusting this constant.
RAMP_BINDING = 0.25
#: "ramp effectively off" for the tests that want to isolate something else.
#: 2.0 already exceeds (p_max - p_min) / (0.7 p_max) ~ 1.15, so no ramp row can
#: bind.  Do NOT reach for a huge value: at 1e6 the ramp right-hand side is
#: ~6e9 against ~1e2 elsewhere, and that seven-order spread alone stalls the
#: IPM at mu = 2.2e-6 instead of 1e-11.
RAMP_OFF = 2.0
DEMAND = 33466.0       # MW; see the note below
#: The same level given the periods a shape.  A flat demand plus a ramp that no
#: longer binds makes the four periods four identical solves, and then the
#: per-period form of the §8 identity carries no more information than the summed
#: form: rotating a rent between identical periods changes nothing, so the
#: injection that is supposed to demonstrate the difference cannot.  Measured
#: 2026-08-17 at cap 0.6 / ramp 1.0, spread of the per-period congestion rent:
#: flat 0.0000, +-5% 29 041, +-15% 59 970, +-30% 441 272 -- and the injection
#: bites at every profiled level, at none of them flat.  +-15% is used because it
#: is an ordinary intra-day shape rather than the smallest value that works.
DEMAND_SHAPED = DEMAND * np.array([0.85, 1.00, 1.15, 1.05])
#: 33 466 MW was the median of the `Actual` column that §14 has since discarded
#: as the wrong basis (2026-08-09).  It is kept as the scenario level because
#: every tolerance in this module is calibrated at it, and it remains a GB
#: demand level: it sits at the 78.5th percentile of the realised series §14
#: now uses, whose median is 27 542 MW.
#: A level at which some bus sheds its entire load while carrying a unit, which
#: is the only configuration where the shed-bound term of §8 is non-zero.  It is
#: not a GB demand level and is not meant to be one: it is the scenario parameter
#: that makes one term of the identity observable.
#:
#: 60 000 MW did that at cap_scale 0.4, and stopped doing it when the scenario
#: loosened the network on 2026-08-17.  Re-measured that day at ramp_scale 1.0,
#: buses fully shed while carrying a unit, against demand:
#:
#:     cap 0.6   60k 0   70k 0   80k 0   95k 0     <- the adopted network, never
#:     cap 0.5   60k 0   70k 0   80k 1   95k 1
#:     cap 0.4   60k 0   70k 1   80k 1   95k 2
#:
#: So the phenomenon needs a network tighter than the adopted one at any demand
#: this module is willing to solve, and the run point below is constructed rather
#: than inherited.  cap 0.5 is chosen over 0.4 because it is the *smaller*
#: departure from the adopted 0.6 and still carries the larger shed-bound term
#: (2.17e-2 of the identity against 5.08e-3 at cap 0.4 / 80 000).
DEMAND_FULL_SHED = 80000.0
CAP_FULL_SHED = 0.5
#: Tolerances for the money-balance identity of §8, asserted per period.  The
#: relative bound governs the periods that carry rent and the absolute bound
#: governs the ones that carry none, where both sides are zero to rounding and a
#: relative comparison is meaningless.  Measured 2026-08-15 over both scenarios of
#: this module: worst relative 1.85e-14 on a period with rent, worst absolute
#: 1.21e-07 on a period without.  Rounded up by three orders and by one.
IDENTITY_RTOL = 1e-11
IDENTITY_ATOL = 1e-6


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


def _run(case, cap_scale=CAP_SCALE, demand=DEMAND, u=None, ramp_scale=RAMP_SCALE):
    clear, spec = make_clearing(case, T, n_segments=K, cap_scale=cap_scale,
                                ramp_scale=ramp_scale)
    _, cost = segment_costs(case, K)
    n_u = spec["n_units"]
    u = jnp.ones((n_u, T)) if u is None else u
    p_min = jnp.asarray(spec["p_min"])
    p_init = p_min * u[:, 0]                  # consistent with the commitment
    offer = jnp.broadcast_to(jnp.asarray(cost)[:, :, None], (n_u, K, T))
    # a scalar means "the same level every period"; a length-T vector gives the
    # periods a shape, which one test below needs and the rest do not care about
    dem = jnp.broadcast_to(jnp.asarray(demand, float), (T,))
    out = jax.jit(clear)(offer, u, dem, p_init)
    assert float(out["mu"]) < 1e-8, "solve did not converge; duals unusable"
    return out, spec, u, p_init


@pytest.fixture(scope="module")
def base(case):
    return _run(case)


def test_power_balance(base):
    """Generation plus shed equals demand, every period."""
    out, spec, _, _ = base
    served = np.asarray(out["award"]).sum(0) + np.asarray(out["shed"]).sum(1)
    np.testing.assert_allclose(served, DEMAND, rtol=1e-9)


def test_award_within_committed_capacity(base):
    out, spec, u, _ = base
    award = np.asarray(out["award"])
    u_np = np.asarray(u)
    lo = spec["p_min"][:, None] * u_np
    hi = spec["p_max"][:, None] * u_np
    assert (award >= lo - 1e-6).all()
    assert (award <= hi + 1e-6).all()


def test_decommitted_units_produce_nothing(case):
    """(CAP) has no rows of its own -- a de-committed unit is zeroed by
    construction, so this is checking the construction, not a constraint."""
    clear, spec = make_clearing(case, T, n_segments=K, cap_scale=CAP_SCALE,
                                ramp_scale=RAMP_SCALE)
    n_u = spec["n_units"]
    u = jnp.ones((n_u, T)).at[::3].set(0.0)
    out, _, u_used, _ = _run(case, u=u)
    off = np.asarray(u_used)[:, 0] == 0
    assert off.sum() > 0
    np.testing.assert_array_equal(np.asarray(out["award"])[off], 0.0)


def test_ramp_limits_respected(base, case):
    out, spec, _, p_init = base
    award = np.asarray(out["award"])
    ramp_up = (np.asarray(case.unit_ramp_up, np.float64) * spec["p_max"] * RAMP_SCALE)
    ramp_dn = (np.asarray(case.unit_ramp_down, np.float64) * spec["p_max"] * RAMP_SCALE)
    delta = np.diff(np.hstack([np.asarray(p_init)[:, None], award]), axis=1)
    assert delta.max() <= ramp_up.max() + 1e-6
    assert (delta - ramp_up[:, None]).max() < 1e-6
    assert (-delta - ramp_dn[:, None]).max() < 1e-6


def test_commitment_may_switch_under_binding_ramp(case):
    """§6.3: (RMP) carries a start-up allowance of p_min and a shut-down
    allowance of p_max.

    Every clearing test above holds the commitment constant, and that is what
    hid the following for as long as it was hidden: `case29gb` has
    p_min / p_max ~ 0.2 while ramp_scale = 0.25 allows 0.175 p_max per period,
    so a unit switching on jumps further than the plain ramp limit permits.
    Without the allowance the LP is infeasible for **any** switching commitment
    -- measured mu 6.5e264 for one unit shutting down and 5.9e285 for one
    starting -- and step 2 of the three-step clearing (rounding) produces
    nothing but switching commitments.

    This test runs at RAMP_BINDING and not at the scenario's RAMP_SCALE, because
    the scenario stopped producing the phenomenon on 2026-08-17: at ramp_scale =
    1.0 the plain limit is 0.7 p_max against a p_min of ~0.2 p_max, so no unit
    needs the allowance and the assertions below would hold for a clearing
    operator that granted none.  The gate is not widened and the run point is not
    the scenario's; the precondition is asserted instead of assumed.
    """
    clear, spec = make_clearing(case, T, n_segments=K, cap_scale=CAP_SCALE,
                                ramp_scale=RAMP_BINDING)
    n_u = spec["n_units"]
    # the phenomenon this test is about, asserted rather than trusted: without a
    # start-up allowance some unit's p_min exceeds what its ramp row permits
    _ramp_up_chk = (np.asarray(case.unit_ramp_up, np.float64)
                    * spec["p_max"] * RAMP_BINDING)
    _needs_allowance = int((spec["p_min"] > _ramp_up_chk + 1e-9).sum())
    assert _needs_allowance > 0, (
        "no unit needs the start-up allowance at ramp_scale="
        f"{RAMP_BINDING}, so this test would check nothing; it must construct a "
        "run point where the plain ramp limit forbids switching on")
    _, cost = segment_costs(case, K)
    offer = jnp.broadcast_to(jnp.asarray(cost)[:, :, None], (n_u, K, T))
    u = jnp.ones((n_u, T)).at[0, 2:].set(0.0).at[1, :2].set(0.0)   # one stops, one starts
    p_init = jnp.asarray(spec["p_min"]) * u[:, 0]
    out = jax.jit(clear)(offer, u, jnp.full((T,), DEMAND), p_init)
    assert float(out["mu"]) < 1e-8, f"switching commitment infeasible: mu={float(out['mu']):.2e}"

    award, u_np = np.asarray(out["award"]), np.asarray(u)
    p_min, p_max = spec["p_min"], spec["p_max"]
    ramp_up = np.asarray(case.unit_ramp_up, np.float64) * p_max * RAMP_BINDING
    ramp_dn = np.asarray(case.unit_ramp_down, np.float64) * p_max * RAMP_BINDING
    u_prev = np.hstack([(np.asarray(p_init) > 0).astype(float)[:, None], u_np[:, :-1]])
    start = np.maximum(u_np - u_prev, 0.0) * p_min[:, None]
    stop = np.maximum(u_prev - u_np, 0.0) * p_max[:, None]
    delta = np.diff(np.hstack([np.asarray(p_init)[:, None], award]), axis=1)
    # tolerance is OFF_EPS, not 1e-6: the ramp rows act on the LP variables,
    # which keep a phantom of up to OFF_EPS for a de-committed unit, while
    # `award` has that phantom multiplied away.  The two therefore differ by up
    # to OFF_EPS across a shut-down, which is why the constant-commitment ramp
    # test above can use 1e-6 and this one cannot.
    tol = OFF_EPS + 1e-6
    assert (delta - (ramp_up[:, None] + start)).max() < tol
    assert (-delta - (ramp_dn[:, None] + stop)).max() < tol
    # the allowance is granted only where the commitment actually switches
    steady = (start == 0) & (stop == 0)
    assert steady.any()
    plain = np.maximum(ramp_up, ramp_dn)[:, None].repeat(T, 1)
    assert (np.abs(delta)[steady] - plain[steady]).max() < tol


def test_high_output_unit_can_shut_down_the_same_day(case):
    """§6.3: the shut-down allowance is p_max, so a unit running high at the day
    boundary can be de-committed in the first period.

    With the tighter allowance of p_min this was infeasible rather than
    expensive, and the failure was reachable from ordinary play: the dispatch
    that leaves a unit high depends on the offers, so an exogenous commitment
    computed before bidding breaks once an agent bids away from cost.  Measured
    on a 59-day chained rollout, 11 day transitions failed at a markup of 2.0
    and the reward came back NaN.
    """
    clear, spec = make_clearing(case, T, n_segments=K, cap_scale=CAP_SCALE,
                                ramp_scale=RAMP_SCALE)
    n_u = spec["n_units"]
    _, cost = segment_costs(case, K)
    offer = jnp.broadcast_to(jnp.asarray(cost)[:, :, None], (n_u, K, T))
    p_min, p_max = spec["p_min"], spec["p_max"]
    ramp_dn = np.asarray(case.unit_ramp_down, np.float64) * p_max * RAMP_SCALE
    # units that cannot reach zero in one period under the plain ramp limit
    stuck = np.where(p_max > p_min + ramp_dn + 1e-9)[0]
    assert len(stuck) > 20, f"case lost its high-output units: {len(stuck)}"
    victim = int(stuck[np.argmax(p_max[stuck] - p_min[stuck] - ramp_dn[stuck])])

    u = jnp.ones((n_u, T)).at[victim, :].set(0.0)      # off all day
    # Only the victim enters high, so that what this test exercises is one unit
    # shutting down and not a fleet-wide surplus.  The figure this comment used
    # to give for the fleet-wide case was 0.825 of installed capacity, which is
    # 1 - 0.7 * 0.25 and therefore belonged to the retired ramp_scale = 0.25; at
    # the adopted 1.0 it is 1 - 0.7 = 0.30, about 24.7 GW against a 33.5 GW
    # demand, i.e. no longer a surplus and no longer infeasible (measured
    # 2026-08-17, mu 1.2e-11).  `test_clearing_l0` therefore constructs that
    # infeasible input at its own RAMP_SURPLUS rather than at the scenario.
    p_init = jnp.asarray(p_min).at[victim].set(p_max[victim])
    out = jax.jit(clear)(offer, u, jnp.full((T,), DEMAND), p_init)
    assert float(out["mu"]) < 1e-8, (
        f"unit {victim} could not shut down from p_max={p_max[victim]:.0f} with "
        f"ramp_dn={ramp_dn[victim]:.0f}: mu={float(out['mu']):.2e}")
    np.testing.assert_array_equal(np.asarray(out["award"])[victim], 0.0)


def test_line_flows_within_rating(base, case):
    """Flows respect the scaled rating; shed is what absorbs any shortfall."""
    out, spec, u, _ = base
    PTDF, F = spec["PTDF"], np.asarray(case.line_cap, np.float64) * CAP_SCALE
    award, shed = np.asarray(out["award"]), np.asarray(out["shed"])
    demand = spec["demand_share"][None, :] * DEMAND
    gen = np.zeros((T, spec["n_buses"]))
    np.add.at(gen, (slice(None), spec["unit_bus"]), award.T)
    flow = (gen + shed - demand) @ PTDF.T
    assert np.abs(flow).max() <= F.max() + 1e-6
    assert (np.abs(flow) - F[None, :]).max() < 1e-6


def _identity(out, spec, case, demand, cap_scale=CAP_SCALE):
    """Both sides of §8, plus the pieces a caller needs to judge the scenario.

    Left side is payments minus charges.  Right side is assembled from the duals
    of the same solve, `-sum (mu+ + mu-) F - sum rho P`, so the comparison is a
    check and not a regrouping of one side.  An earlier version of this module
    computed only the left side and named the difference `rent`, which asserts
    nothing about the identity: a sign error in the `rho . P` term survived that
    for as long as it existed (found 2026-08-11 by the real-time market line).

    Everything is returned **per period** as well as summed, because the identity
    holds period by period and the sum hides where it does not.  The test layering
    requires the itemised side to be a dimension the reward is consumed on, and
    the reward of §8 is consumed over units and periods; the network makes the bus
    axis the wrong one, since congestion rent is not attributable to a bus, so the
    period axis is the one available here.  A price moved from one period to
    another leaves every total intact.
    """
    lmp, shed = np.asarray(out["lmp"]), np.asarray(out["shed"])
    award = np.asarray(out["award"])
    up, dn = np.asarray(out["line_dual_up"]), np.asarray(out["line_dual_dn"])
    rho = np.asarray(out["shed_dual"])
    F = np.asarray(case.line_cap, np.float64) * cap_scale
    # `demand` is a scalar level or a per-period vector, matching `_run`
    dem_t = np.broadcast_to(np.asarray(demand, float).reshape(-1), (T,))
    d = spec["demand_share"][None, :] * dem_t[:, None]
    gen = np.zeros((T, spec["n_buses"]))
    np.add.at(gen, (slice(None), spec["unit_bus"]), award.T)
    injection = gen + shed - d                      # P of §3.1
    payments_t, charges_t = (lmp * gen).sum(1), (lmp * (d - shed)).sum(1)
    congestion_t = -((up + dn) * F[None, :]).sum(1)
    shed_bound_t = (rho * injection).sum(1)
    carries_unit = np.zeros(spec["n_buses"], bool)
    carries_unit[np.asarray(spec["unit_bus"])] = True
    fully_shed = ((shed > d - 1e-6) & (d > 1e-9)).any(0)
    return types.SimpleNamespace(
        lhs_t=payments_t - charges_t, congestion_t=congestion_t,
        shed_bound_t=shed_bound_t,
        lhs=(payments_t - charges_t).sum(), congestion=congestion_t.sum(),
        shed_bound=shed_bound_t.sum(),
        payments=payments_t.sum(), charges=charges_t.sum(),
        fully_shed_with_a_unit=int((fully_shed & carries_unit).sum()))


def test_money_balance(base, case):
    """§8: payments minus charges equal the congestion rent and the shed-bound rent.

    On this scenario the shed-bound term is 7.3e-19 of the identity, because no
    bus sheds its entire load and `rho` is therefore zero everywhere, so what
    this case pins is the congestion half.  Measured residual 2.3e-14 relative;
    the tolerance is three orders above it.  The scenario that does exercise the
    other half is the next test, and it is a separate one precisely because this
    one cannot see it.
    """
    out, spec, _, _ = base
    idt = _identity(out, spec, case, DEMAND)
    scale = max(abs(idt.lhs), 1.0)
    # §7 states all three duals are nonnegative; the identity would also fail if
    # they were returned negated, but that failure names the wrong culprit
    for name in ("line_dual_up", "line_dual_dn", "shed_dual"):
        assert np.asarray(out[name]).min() > -1e-9, f"{name} broke the §7 sign convention"
    assert abs(idt.shed_bound) / scale < 1e-12, "scenario changed; rho is no longer idle"
    # per period, not only summed: a rent attributed to the wrong period leaves
    # the total intact (see `_identity` for why the period axis)
    np.testing.assert_allclose(idt.lhs_t, idt.congestion_t - idt.shed_bound_t,
                               rtol=IDENTITY_RTOL, atol=IDENTITY_ATOL)
    # congestion rent is non-negative: load pays at least what generation earns
    assert idt.charges - idt.payments > -1e-6 * scale

    # What the per-period form buys over the summed one, demonstrated rather than
    # asserted: rotating the congestion rent across periods is an attribution
    # error, and the sum cannot see it because rotation preserves a sum.
    #
    # This runs on DEMAND_SHAPED and not on `base`.  On `base` the four periods
    # are identical solves -- flat demand, and since 2026-08-17 a ramp that does
    # not bind -- so rotation is the identity map and the demonstration is empty.
    # The scenario is not the thing under test here, so the shape is added rather
    # than the assertion dropped.
    out_s, spec_s, _, _ = _run(case, demand=DEMAND_SHAPED)
    idt_s = _identity(out_s, spec_s, case, DEMAND_SHAPED)
    scale_s = max(abs(idt_s.lhs), 1.0)
    np.testing.assert_allclose(idt_s.lhs_t, idt_s.congestion_t - idt_s.shed_bound_t,
                               rtol=IDENTITY_RTOL, atol=IDENTITY_ATOL)
    # the rotation is detectable only if it moves a period by more than the
    # tolerance that period is compared under; anything less and the assertion
    # below would pass for a reason that has nothing to do with attribution
    spread = float(np.ptp(idt_s.congestion_t))
    detectable = IDENTITY_ATOL + IDENTITY_RTOL * float(np.abs(idt_s.congestion_t).max())
    assert spread > detectable, (
        f"the periods carry rent within {spread:.3e} of each other against a "
        f"detection floor of {detectable:.3e}, so rotating it cannot be told from "
        "not rotating it and the injection below proves nothing; give the periods "
        "a shape")
    rotated = np.roll(idt_s.congestion_t, 1)
    assert abs(rotated.sum() - idt_s.congestion_t.sum()) < IDENTITY_ATOL * scale_s, \
        "the rotation is supposed to leave the total untouched"
    assert not np.allclose(idt_s.lhs_t, rotated - idt_s.shed_bound_t,
                           rtol=IDENTITY_RTOL, atol=IDENTITY_ATOL), \
        "the per-period form failed to see a rent moved between periods"


def test_money_balance_when_a_bus_sheds_its_whole_load(case):
    """§8: the shed-bound term carries a minus sign, and this scenario shows it.

    `rho` is non-zero only at a bus shedding its entire load, and the term it
    enters is `rho . P`, so it vanishes unless such a bus also carries a unit
    and therefore has a non-zero net injection.  At the module demand no bus
    reaches its shed bound at all; at 60 000 MW one does and it carries a unit.

    Discriminating power, measured 2026-08-11 at cap 0.4 / ramp 0.25 / 60 000 MW
    and `mu` 2.4e-11: the term was 1.6e-2 of the identity, the documented minus
    sign left a residual of 1.3e-15, and flipping it to plus left 3.3e-2.

    Re-measured 2026-08-17 at the run point below (cap 0.5 / ramp 1.0 / 80 000 MW,
    `mu` 1.99e-11): the term is **2.17e-2** of the identity, the minus sign leaves
    **1.19e-16** and the plus sign **1.10e-2**.  Fourteen orders separate them, so
    this assertion still stands between §8 and a sign error, and it does so with a
    slightly larger term than the run point it replaced.

    The run point is constructed, not inherited.  At the adopted cap_scale 0.6 no
    demand this module is willing to solve reaches a shed bound at all -- measured
    0 fully-shed buses at 60 000 / 70 000 / 80 000 / 95 000 MW -- so the network is
    tightened for this test only, to the smallest departure from 0.6 that restores
    the phenomenon.  See the note on DEMAND_FULL_SHED for the scan.

    Demand is not pushed arbitrarily far.  At cap 0.4 / ramp 0.25, 63 000 MW and
    above stopped converging (`mu` 1.5e-4 to 1.2e-3) and stopped being
    reproducible: two runs of one input gave different `mu` and disagreed on how
    many buses reach the shed bound.  That ceiling was a property of that run
    point and had to be re-checked here rather than assumed: at cap 0.5 / ramp 1.0
    / 80 000 MW four repeats in one process agreed bit for bit on `mu`, on the
    identity's left side and on the shed total.  `_run` asserts convergence, which
    keeps this test off that ground either way.
    """
    out, spec, _, _ = _run(case, cap_scale=CAP_FULL_SHED, demand=DEMAND_FULL_SHED)
    idt = _identity(out, spec, case, DEMAND_FULL_SHED, cap_scale=CAP_FULL_SHED)
    scale = max(abs(idt.lhs), 1.0)
    assert idt.fully_shed_with_a_unit >= 1, "scenario no longer reaches a shed bound"
    assert abs(idt.shed_bound) / scale > 1e-3, "shed-bound term too small to have teeth"
    np.testing.assert_allclose(idt.lhs_t, idt.congestion_t - idt.shed_bound_t,
                               rtol=IDENTITY_RTOL, atol=IDENTITY_ATOL)


def test_uncongested_prices_are_uniform(case):
    """With real ratings nothing binds, so every bus sees the same price."""
    out, _, _, _ = _run(case, cap_scale=1.0, ramp_scale=RAMP_OFF)
    lmp = np.asarray(out["lmp"])
    assert np.ptp(lmp, axis=1).max() < 1e-6


def test_congestion_separates_prices(base):
    """At the scenario's cap_scale the network binds and prices must spread out.

    Re-measured 2026-08-17 at cap 0.6 / ramp 1.0: four of 396 line-periods carry a
    congestion dual and the widest per-period LMP spread is 77.20 \\$/MWh.  At the
    retired cap 0.4 / ramp 0.25 it was ten line-periods and 105.97 \\$/MWh, so
    loosening the network cut the congestion roughly in half without removing it.
    """
    out, _, _, _ = base
    assert np.ptp(np.asarray(out["lmp"]), axis=1).max() > 1.0


def test_shed_is_priced_at_voll(case):
    """A bus shedding strictly inside its bound is priced at VOLL: one more MW
    of demand is one more MW shed."""
    out, spec, _, _ = _run(case, demand=95_000.0, ramp_scale=RAMP_OFF)
    lmp, shed = np.asarray(out["lmp"]), np.asarray(out["shed"])
    demand = spec["demand_share"][None, :] * 95_000.0
    interior = (shed > 1e-6) & (shed < demand - 1e-6)
    assert interior.any(), "scenario did not shed"
    np.testing.assert_allclose(lmp[interior], VOLL, rtol=1e-6)


def test_offer_is_per_period(case):
    """§5: a unit may bid one period differently from another.

    With ramp effectively off the periods are independent, so marking up every
    offer in period 0 alone must raise the price of period 0 and leave the other
    periods where the day-flat offer put them.  A day-flat offer cannot detect a
    transposed period axis in the cost vector; this can.

    The untouched periods move by 9.3e-16 relative rather than not at all: one
    interior point method solves all $T$ periods together, so the barrier and
    the regularisation couple them numerically even where ramp does not couple
    them physically.
    """
    clear, spec = make_clearing(case, T, n_segments=K, cap_scale=1.0,
                                ramp_scale=RAMP_OFF)
    _, cost = segment_costs(case, K)
    n_u = spec["n_units"]
    u = jnp.ones((n_u, T))
    args = (u, jnp.full((T,), DEMAND), jnp.asarray(spec["p_min"]))
    flat = jnp.broadcast_to(jnp.asarray(cost)[:, :, None], (n_u, K, T))
    bumped = flat.at[:, :, 0].multiply(1.5)

    base_lmp = np.asarray(jax.jit(clear)(flat, *args)["lmp"])
    got_lmp = np.asarray(jax.jit(clear)(bumped, *args)["lmp"])
    assert got_lmp[0, 0] > base_lmp[0, 0] + 1.0, "period 0 did not reprice"
    np.testing.assert_allclose(got_lmp[1:], base_lmp[1:], rtol=1e-12)


def test_price_equals_the_offer_of_a_unit_when_nothing_binds(case):
    """§16: with no line binding and no bus shedding, the congestion and
    shed-bound terms of §7 both vanish and the price is the offer of the
    marginal unit.  `test_uncongested_prices_are_uniform` only pins the prices
    to each other; this pins them to a number the market produced."""
    out, spec, _, _ = _run(case, cap_scale=1.0, ramp_scale=RAMP_OFF)
    assert np.asarray(out["shed"]).max() < 1e-6, "scenario shed; §7 term is live"
    _, offer = segment_costs(case, K)
    lmp = np.asarray(out["lmp"])[:, 0]                # uniform, so any bus
    gap = np.abs(lmp[:, None] - np.asarray(offer)[None, :, 0]).min(1)
    assert gap.max() < 1e-6, f"no unit offers at the clearing price: {gap.max()}"


# --------------------------------------------------------------------------
# Finite difference (§16).  The money-balance identity is blind to the pricing
# formula -- it held to 1e-16 both before and after the shed-bound term was
# added -- so this is the only check that can catch an error in §7.
# --------------------------------------------------------------------------

FD_ON = 16          # units committed; the configuration §16 records
FD_EPS = 1.0        # MW, matching tools/lp_bench
#: Measured 2026-08-09 over the three probe buses of the scenario below: worst
#: |finite difference - lmp| is 7.9e-9 \$/MWh on CPU and 6.3e-8 on GPU.  The
#: tolerance is far looser because the floor is set at the GPU
#: inter-process drift, about 4e-3, which has not been remeasured under float64.
FD_ATOL = 1e-2


def _case_with_nodal_demand(case, node_mw):
    """`case` with its demand split replaced, so that one bus can be bumped.

    `clear` takes total system demand and splits it by a share fixed at build
    time, so a single-bus perturbation has to enter through that share.  Setting
    `node_pd` to the wanted megawatts and passing their sum as the demand
    reproduces them exactly, since `make_clearing` normalises `node_pd`.
    """
    return types.SimpleNamespace(
        n_nodes=case.n_nodes, unit_p_min=case.unit_p_min, unit_p_max=case.unit_p_max,
        unit_cost_a=case.unit_cost_a, unit_cost_b=case.unit_cost_b,
        unit_cost_c=case.unit_cost_c, unit_node_idx=case.unit_node_idx,
        PTDF=case.PTDF, line_cap=case.line_cap,
        node_pd=np.asarray(node_mw, np.float64),
        unit_ramp_up=case.unit_ramp_up, unit_ramp_down=case.unit_ramp_down)


def _clear_one_period(case_like, demand, u, cap_scale):
    """One period, and the objective recomputed from the cleared result.

    The objective is the as-bid cost of the accepted segments plus VOLL times
    the shed.  The must-run block carries no offer price in this formulation
    (§19) and does not depend on demand either way, so it drops out of the
    derivative taken below.
    """
    clear, spec = make_clearing(case_like, 1, n_segments=K, cap_scale=cap_scale,
                                ramp_scale=RAMP_OFF)
    _, offer = segment_costs(case_like, K)
    u_np = np.asarray(u)
    out = jax.jit(clear)(jnp.asarray(offer)[:, :, None], u, jnp.array([demand]),
                         jnp.asarray(spec["p_min"]) * u[:, 0])
    assert float(out["mu"]) < 1e-8, "solve did not converge; duals unusable"
    award, shed = np.asarray(out["award"])[:, 0], np.asarray(out["shed"])[0]
    obj = float(np.asarray(offer)[:, 0] @ (award - spec["p_min"] * u_np[:, 0])) \
        + VOLL * float(shed.sum())
    return out, spec, obj


@pytest.fixture(scope="module")
def scarce(case):
    """16 of 66 units committed at the scenario's cap_scale, so one solve carries
    both a congested bus and a fully-shed one.

    Re-measured 2026-08-17 at cap 0.6: 20 buses shed, 19 of them to their bound,
    27 978 MW in total, and prices span 105.30 to 10 000 \\$/MWh.  At the retired
    cap 0.4 it was 22 buses, 21 to their bound, 28 949 MW, and 32.95 to 10 000.
    The lower end of the span moved because the looser network lets the cheap buses
    reach more of the load; the fully-shed buses this fixture exists to provide are
    still there, which is what the tests consuming it need."""
    n_u = len(np.asarray(case.unit_p_min))
    u = jnp.zeros((n_u, 1)).at[:FD_ON].set(1.0)
    out, spec, obj = _clear_one_period(case, DEMAND, u, CAP_SCALE)
    return out, spec, obj, u


def test_finite_difference_matches_lmp(case, scarce):
    """§16: probe one fully-shed bus, the cheapest bus and the dearest served
    bus, and compare the price against the derivative of the objective.

    This is what catches the shed-bound term of §7.  Measured 2026-08-09 by
    deleting `rho` from the price: this scenario then reports 15 553.96 \\$/MWh
    at the fully-shed bus against a derivative of 10 000, and every other test
    in this module, the money-balance identity included, still passes.
    """
    out, spec, obj0, u = scarce
    lmp, shed = np.asarray(out["lmp"])[0], np.asarray(out["shed"])[0]
    demand = spec["demand_share"] * DEMAND
    full = np.where((shed > demand - 1e-6) & (demand > 1e-6))[0]
    served = np.where((shed < 1e-6) & (demand > 1e-6))[0]
    assert len(full) > 0 and len(served) > 1, "scenario lost its shed or its load"
    order = served[np.argsort(lmp[served])]
    probe = [int(full[0]), int(order[0]), int(order[-1])]

    for n in probe:
        bumped = demand.copy()
        bumped[n] += FD_EPS
        _, _, obj1 = _clear_one_period(_case_with_nodal_demand(case, bumped),
                                       float(bumped.sum()), u, CAP_SCALE)
        np.testing.assert_allclose((obj1 - obj0) / FD_EPS, lmp[n], atol=FD_ATOL)


def test_fully_shed_bus_is_not_priced_above_voll(scarce):
    """§16: once a bus sheds its entire load, one more megawatt of demand is one
    more megawatt shed, so its price is VOLL and not more.  The shed-bound term
    of §7 is what holds it there; without it the congestion component pushes the
    price above VOLL, which is the visible symptom §7 describes."""
    out, spec, _, _ = scarce
    lmp, shed = np.asarray(out["lmp"])[0], np.asarray(out["shed"])[0]
    demand = spec["demand_share"] * DEMAND
    full = (shed > demand - 1e-6) & (demand > 1e-6)
    assert full.sum() > 0, "scenario did not shed a bus in full"
    np.testing.assert_allclose(lmp[full], VOLL, atol=FD_ATOL)
    assert lmp.max() <= VOLL + FD_ATOL


# --------------------------------------------------------------------------
# Hand-worked two-bus example (§16).  Must congest, or it proves nothing.
# --------------------------------------------------------------------------

def _two_bus(line_cap):
    """Two buses, one line, one cheap unit at bus 0 and one dear unit at bus 1.

    PTDF with bus 0 as reference: injecting 1 MW at bus 1 sends -1 MW along the
    line 0->1, so PTDF = [[0, -1]].  Marginal costs are constants (a = b = 0),
    so segment cost is exactly c.
    """
    return types.SimpleNamespace(
        n_nodes=2,
        unit_p_min=np.array([0.0, 0.0]),
        unit_p_max=np.array([100.0, 100.0]),
        unit_cost_a=np.array([0.0, 0.0]),
        unit_cost_b=np.array([0.0, 0.0]),
        unit_cost_c=np.array([10.0, 50.0]),      # $/MWh, flat
        unit_node_idx=np.array([0, 1]),
        PTDF=np.array([[0.0, -1.0]]),
        line_cap=np.array([float(line_cap)]),
        node_pd=np.array([0.0, 1.0]),            # all load at bus 1
        unit_ramp_up=np.array([1.0, 1.0]),       # fraction of p_max per hour
        unit_ramp_down=np.array([1.0, 1.0]),
    )


def _clear_two_bus(line_cap, demand=60.0):
    case = _two_bus(line_cap)
    clear, spec = make_clearing(case, 1, n_segments=1, cap_scale=1.0, ramp_scale=100.0)
    _, cost = segment_costs(case, 1)
    out = jax.jit(clear)(jnp.asarray(cost)[:, :, None], jnp.ones((2, 1)),
                         jnp.array([demand]), jnp.zeros(2))
    assert float(out["mu"]) < 1e-8
    return out


def test_two_bus_uncongested():
    """Cheap unit serves all 60 MW; both buses price at its offer."""
    out = _clear_two_bus(line_cap=200.0)
    np.testing.assert_allclose(np.asarray(out["award"])[:, 0], [60.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(np.asarray(out["lmp"])[0], [10.0, 10.0], atol=1e-6)


def test_two_bus_congestion_sign():
    """The sign test.  Rating 40 < 60, so the line binds.

    Hand-worked: the cheap unit can deliver only 40 MW to bus 1, so the dear
    unit must supply the remaining 20.  One more MW at bus 0 comes from the
    cheap unit locally (10 $/MWh); one more at bus 1 must come from the dear
    unit (50 $/MWh).  Hence lmp = [10, 50].

    Checking §7 term by term: lmp[n] = lambda - (mu+ - mu-) PTDF[0, n].
    lmp[0] = lambda = 10 since PTDF[0,0] = 0, and lmp[1] = 10 + (mu+ - mu-) = 50
    gives mu+ = 40 with the upper limit binding.  Drop one negation and lmp[1]
    becomes 10 - 40 = -30: the dear bus would price *below* the cheap one and
    the reward at every congested bus would invert.
    """
    out = _clear_two_bus(line_cap=40.0)
    award, lmp = np.asarray(out["award"])[:, 0], np.asarray(out["lmp"])[0]
    np.testing.assert_allclose(award, [40.0, 20.0], atol=1e-5)
    np.testing.assert_allclose(lmp, [10.0, 50.0], atol=1e-4)
    assert lmp[1] > lmp[0], "congestion component has the wrong sign"


def test_two_bus_price_spread_equals_shadow_price():
    """The spread is exactly the line's shadow price, 40 $/MWh here."""
    out = _clear_two_bus(line_cap=40.0)
    lmp = np.asarray(out["lmp"])[0]
    np.testing.assert_allclose(lmp[1] - lmp[0], 40.0, atol=1e-4)
