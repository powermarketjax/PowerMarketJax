"""L2 numerical equivalence against the numpy reference.

Every comparison here runs at a reserve offer separation measured to leave both
solves trustworthy, and **asserts that precondition rather than assuming it**.
Below it the duals of this LP are unreliable in both, so the two would agree or
disagree for reasons that have nothing to do with either being written correctly.

Tolerances are derived from measurement, not chosen: they are the observed
deviation of this pair over the configurations below, rounded up.  Writing them
by feel is a known failure mode of this repository's L2 layer, and it caught this
file too: the first draft asserted 1e-6 on every quantity and failed on five of
eleven tests, because the reserve award among near-tied providers legitimately
differs by half a megawatt.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.ancillary.clearing import (MAX_ITER, REG_COEF,
                                                    make_clearing)
from powermarketjax.envs.day_ahead.clearing import segment_costs
from tests.envs.ancillary.reference import clear_reference
from tests.envs.ancillary.test_clearing_l0 import FIXTURE, HOUR

CASE = "29gb"
THETA = (1.0 / 6.0, 0.5)
VOLR = 250.0
DELTA = 0.5
CAP_SCALE, RAMP_SCALE = 0.6, 1.0
DUAL_START = "cost_norm"
#: Half-width of the reserve offer spread.  What governs trustworthiness is the
#: **pairwise** gap between adjacent offers, not the width of the range: with 66
#: providers spread linearly over +-10% the neighbours differ by 0.31%, and that
#: is measured clean here (both residuals at 1e-11, prices agreeing to 3.5e-8).
#: The same construction at +-1% leaves neighbours 0.031% apart, and at these
#: operating points that is *not* clean -- the implementation residual reaches
#: 1.7e-6 and per-unit reserve revenue moves by 150 $.  So this file runs at
#: +-10% and asserts the precondition rather than assuming it.
SEPARATION = 0.10
#: A profile with no separation at all, at an operating point measured to leave
#: the solve untrustworthy; used by the last test.
#:
#: **Both coordinates were once written in units that move with the scenario,
#: and the run point drifted out of the phenomenon when they did.**  `level` was
#: the absolute 900, chosen while `VOLR` was 1000 so that a tied offer sat below
#: the cap and therefore set the price; at `VOLR = 250` an offer of 900 is above
#: the cap, the price is pinned at `VOLR`, and a tie cannot reach the price at
#: all.  `req_frac` was 0.30 of what the committed fleet may sell, chosen at
#: `ramp_scale = 0.25`; at 1.00 the same fraction is four times the absolute
#: requirement and sits deep in shortage, where the price is again pinned at
#: `VOLR`.  Either drift alone makes the clearing come back clean, and a clean
#: answer here is indistinguishable at a glance from the limitation having been
#: fixed -- which is the reading that was actually taken before this was found.
#:
#: So `level` is a fraction of `VOLR` and `req_frac` is low enough to stay out
#: of shortage on both scenarios, and the test asserts that it did rather than
#: trusting these numbers to keep meaning what they meant.  Measured 2026-08-17
#: at `cap 0.6 / ramp 1.00 / VOLR 250` over 768 tied run points: of the 473 not
#: in shortage a tied profile alarms on 288, worst normalised residual 9.84e-02
#: against this test's 1e-6 gate, while all 295 shortage run points are clean at
#: 8.13e-11.
#: Chosen by sweeping the grid rather than by picking a plausible pair: over
#: `level_frac` in {0.2, 0.5, 0.8, 0.95} x `req_frac` in {0.05, 0.1, 0.2, 0.3,
#: 0.5} at this file's own cell, 19 of the 20 combinations alarm and one does
#: not -- (0.8, 0.10), at 9.35e-07 against the 1e-6 gate.  A pair picked for
#: looking reasonable had a one-in-twenty chance of being the quiet one, and the
#: first pair tried was it.
#:
#: **Re-measured 2026-08-24 and the "19 of 20" does not reproduce at any
#: scenario, while the argument it supports comes out stronger.**  Same grid,
#: same `dfrac` and `spread` as `TIED_CASE`, CPU:
#:
#:     cap 0.4 / ramp 0.25    alarm 20   quiet 0   in shortage 0
#:     cap 0.6 / ramp 0.50    alarm 11   quiet 8   in shortage 1
#:     cap 0.6 / ramp 1.00    alarm  7   quiet 5   in shortage 8   <- adopted
#:
#: None of the three is 19, and (0.8, 0.10) at 9.35e-07 was not found; that
#: reading belongs to a run whose cell is not recoverable from this comment.
#: **The inference survives and gets sharper: at the adopted scenario 5 of 20 are
#: quiet, so picking a plausible-looking pair has a one-in-four chance of landing
#: on a quiet one, not one in twenty.**  Sweeping was the right method by a wider
#: margin than the sentence above claims.
#:
#: **And the count itself is not stable on GPU**: three freshly started processes
#: at the adopted scenario give 9, 9 and 10 alarms.  That is the same per-process
#: kernel selection documented in
#: `test_a_tied_offer_profile_reports_itself_untrustworthy`, here reaching a
#: **count** rather than a single residual -- so this grid must be read on CPU as
#: well.  Found by `powermarketjax-f7`, who read the numbers below against that
#: test's measurement and asked whether a count could inherit the same spread.
#:
#: **(0.8, 0.20) is taken over the widest-margin pair because it is the one that
#: survives the migrated scenario as well**, and surviving means three things at
#: once, of which the margin is only the first.  At `cap 0.6 / ramp 1.00 /
#: VOLR 250` on this same cell the pair still alarms (2.92e-05 against the
#: gate -- **and that number is a CPU reading, which is the second reason the
#: test below pins the platform: measured 2026-08-24 the same quantity takes
#: four different values per process on GPU, so 2.92e-05 was one of four and
#: nothing here said so. Pinning makes this recorded margin the quantity's
#: single value rather than one draw from it.** Reasoning due to
#: `powermarketjax-f7`), the run point is still out of shortage, **and the numpy
#: reference's
#: KKT solve still completes**.  That last one is not automatic: at this cell
#: and scenario the reference raises `LinAlgError` on four of the six pairs
#: measured, including every pair at a tied level of 50, so a pair chosen on
#: margin alone would leave this test un-runnable after the migration rather
#: than merely red -- and un-runnable is an error, which a gate looking for
#: failures does not see.  The old scenario's margin here is 3.15e-04, two
#: orders above the gate rather than the four the widest pair gives; that is the
#: price of the property.
#: **Moved from (0.8, 0.20) to (0.6, 0.15) on 2026-08-25, when `REG_COEF` went to
#: 1e-20.**  Not a relaxation and not a preference: at 1e-20 the numpy reference's
#: KKT solve raises `LinAlgError: Singular matrix` at the old point, which is the
#: un-runnable failure this test's docstring warns about -- and un-runnable is an
#: error a gate looking for failures does not see.
#:
#: Re-derived by sweeping `level_frac` x `req_frac` = 8 x 6 = 48 combinations
#: against all four conditions this test needs (level < VOLR; the reference
#: completes; out of shortage; the tied profile alarms).  **Exactly one survives**,
#: and its margin is enormous rather than marginal:
#:
#:     (0.60, 0.15)   residual 1.612e-02 = 16 120x the 1e-6 gate
#:                    shortfall 8.07e-09 MW
#:
#: The other 47 fail on one of: the reference going singular (every combination at
#: `level_frac >= 0.8` with `req_frac <= 0.20`), or the run point sitting in
#: shortage (every `req_frac >= 0.30`), or the residual falling below the gate.
#: **`feat/ancillary-services` independently picked the same point** when it made
#: this change on its own line (`c3e8a49`), reporting 1.61e-02 and 8.07e-09 -- two
#: separate derivations on two different versions of this file agreeing to three
#: significant figures.
TIED_LEVEL_FRAC = 0.6
TIED_CASE = dict(req_frac=0.15, dfrac=0.85, level=TIED_LEVEL_FRAC * VOLR,
                 spread=0.0)


@pytest.fixture
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture
def pair(x64):
    case = load_case(CASE)
    _, cost = segment_costs(case, 1)
    clear, spec = make_clearing(case, THETA, VOLR, n_segments=1,
                                cap_scale=CAP_SCALE, ramp_scale=RAMP_SCALE,
                                period_hours=DELTA)
    pmin = np.asarray(case.unit_p_min, np.float64)
    pmax = np.asarray(case.unit_p_max, np.float64)
    fx = np.load(FIXTURE, allow_pickle=True)
    u = fx["commitment"][0, :, HOUR].astype(np.float64)
    assert (u == 0).any() and (u > 0).any()
    n_u, n_p = spec["n_units"], spec["n_prod"]
    supply = (np.asarray(spec["res_cap"]) * u[:, None]).sum(0)

    def both(req_frac=0.70, dfrac=0.85, level=200.0, spread=SEPARATION):
        demand = float((pmin * u).sum()) + dfrac * float(((pmax - pmin) * u).sum())
        p_prev = (pmin + min(dfrac, 1.0) * (pmax - pmin)) * u
        d_res = req_frac * supply
        offer_res = level * (1.0 + np.linspace(-spread, spread, n_u))[:, None]
        offer_res = np.repeat(offer_res, n_p, axis=1)
        got = jax.jit(clear)(jnp.asarray(cost), jnp.asarray(offer_res),
                             jnp.asarray(u), jnp.asarray(demand),
                             jnp.asarray(d_res), jnp.asarray(p_prev))
        want = clear_reference(case, cost, offer_res, u, demand, d_res, p_prev,
                               theta=THETA, volr=VOLR, cap_scale=CAP_SCALE,
                               ramp_scale=RAMP_SCALE, period_hours=DELTA,
                               max_iter=MAX_ITER, dual_start=DUAL_START,
                               reg_coef=REG_COEF)
        return {k: np.asarray(v) for k, v in got.items()}, want

    return both, spec


#: **Scenario of every measured value in this block**: `cap_scale = 0.4`,
#: `ramp_scale = 0.25`, and the commitment fixture built on 2023-07-05 plus 59
#: consecutive days.  **All three changed on 2026-08-16** (to 0.6, 0.5, and four
#: fifteen-day seasonal windows), and `ramp_scale` changed again on 2026-08-17
#: to 1.00 -- the registered ramp rate at full value, no longer discounted.
#:
#: **That re-measure was done on 2026-08-24 and both numbers are kept**, the old
#: one beside the new, because the difference between them is the direct evidence
#: of what the scenario change bought.  The new one was taken on the **full 3x3
#: grid** (see the next paragraph) and at **both** `reg_coef` values -- 1e-18, in
#: force when this was measured, and 1e-20, which had been decided and which
#: went into force on 2026-08-25 -- so each
#: bound below is justified against the worse of the two and does not have to
#: move when that change lands.  **Every bound held**; none was relaxed.  The
#: tightest is `reserve_shortfall`, 5.8e-02 against 1e-1, a factor of 1.7.
#:
#: A tolerance's stamp needs the scenario as well as the value, the date and the
#: platform.  This block is why: it carried the value, the date and the platform,
#: and none of those told a reader that the operating points underneath had moved.
#:
#: Every tolerance below carries the value it was measured at and the date, so
#: that a number taken from measurement does not look like a number written by
#: feel.
#:
#: **The grid.**  This block used to say "the nine configurations this file runs
#: (three requirement levels by three offer levels)", which reads as a cross
#: product.  It was not one: the tests parametrize **one axis at a time** -- three
#: `req_frac` at `level` 200 plus three `level` at `req_frac` 0.70, which is five
#: distinct points, not nine, and the corners were never visited.  Measured on
#: 2026-08-24, that costs a factor of **1.8 to 2.7** on the worst value for
#: `award`, `reserve_shortfall`, `lmp`, `reserve`, `z` and energy revenue, and
#: 826x on `shed`: the one-at-a-time grid does not contain the worst point.
#:
#: The bounds below are now stamped with the worst over the **full 3x3**, so they
#: are conservative with respect to what the tests visit rather than equal to it.
#: And the corner the full grid found -- `req_frac 0.30 / level 5.0`, worst for
#: seven of the ten quantities -- is now one of the points the tests do run, per
#: the rule that a release condition is tried on the hardest sample.  What is
#: still not run is the full cross product: nine `pair` builds per test at ~1.15 s
#: each roughly doubles this file, and the measurement says the two points left
#: out (`0.95/5`, `0.95/900`) are worst for nothing.
#: Each entry: `bound  # was <old>, 0.4/0.25 grid-of-5, 2026-08-14 | now <new>,
#: 0.6/1.00 full 3x3, worse of reg 1e-18 and 1e-20, 2026-08-24, CPU`.
ATOL = dict(
    award=1e-9,               # was 4.5e-12 | now 8.2e-12  (122x margin)
    shed=1e-10,               # was 2.6e-13 | now 1.4e-17
    reserve_shortfall=1e-1,   # was 3.8e-02 | now 5.8e-02  (1.7x -- the tightest)
    lmp=1e-6,                 # was 3.1e-08 | now 5.9e-08  (17x)
    reserve_price=1e-6,       # was 3.5e-08 | now 3.5e-08  (28x)
    #: **Not a comparison tolerance any more**, and the old stamp read like one.
    #: The elementwise assertion it belonged to was replaced by the (CS) property
    #: test below, which uses this only as the threshold for "the multiplier is
    #: positive".  The elementwise difference itself is now 3.1e+01 at reg 1e-18
    #: and 1.9e+01 at 1e-20, and that is the dual being non-unique, not an error
    #: -- see that test's docstring.  So this number is a small positive floor,
    #: and nothing below depends on its exact value.
    capacity_dual=1e-6,       # was 4.5e-08 as a comparison bound, 2026-08-14
)
#: was 6.8e-16 relative, 2026-08-14 | now 6.5e-16, full 3x3, 2026-08-24
Z_RTOL = 1e-12
#: Reserve award tolerance against the numpy reference.
#:
#: The reserve **quantity** is not compared tightly, and that is deliberate: the
#: award among near-tied providers sits on a face the two solvers may pick
#: different points of, and the measured spread is 0.52 MW while the money it
#: carries agrees to 6.9e-06 $.  The judgement is taken on the money, per the
#: rule that a quantity difference is only an error once it is a revenue
#: difference.
#:
#: **What the measurement behind it is about.**  0.52 MW was measured against
#: `reference_ipm`, 2026-08-14, at `cap_scale = 0.4` / `ramp_scale = 0.25` on the
#: 60-day window; **all three changed on 2026-08-16 and this pair is awaiting
#: re-measure**.  It is a statement about **two interior
#: point methods agreeing with each other**, not about how determined the
#: reserve award is.  The same quantity against HiGHS is **24.79 MW** (measured
#: 2026-08-15 on the C8 sweep, CPU), **48 times larger** -- so agreement between
#: two implementations of the same algorithm is far closer than agreement
#: between either and a simplex, which is the quantitative form of the blind
#: spot of a reference that shares the method under
#: test.  The 24.79 MW carries no money: the objective agrees to 1e-15 relative
#: and the per-unit revenue to 4.6e-06 dollars, so it is a different point on
#: the same optimal face and the permitting condition (objective also agrees)
#: holds.
#:
#: This bound is not wrong -- it guards what an L2 against this reference can
#: guard.  It simply must not be read as "the reserve award is determined to
#: half a megawatt".
#: **The re-measure this comment asked for, done 2026-08-24: 0.056 MW**, on the
#: adopted scenario over the full 3x3 at the worse of the two `reg_coef` -- an
#: order of magnitude *tighter* than the 0.52 MW measured on the retired one, so
#: the scenario change made the two solvers agree better here, not worse.  The
#: bound stays 5.0: it is not a measurement of how determined the award is (the
#: 24.79 MW against HiGHS is the number for that), and tightening it to the
#: measured value would turn a guard against a wrong face into a tripwire on
#: which point of the right face gets picked.
RESERVE_MW_ATOL = 5.0     # was 0.52 MW, 2026-08-14 | now 0.056 MW, 2026-08-24, CPU
REVENUE_ATOL = dict(
    reserve=1e-4,          # was 6.9e-06 $ | now 1.2e-05 $, 2026-08-24  (8x)
    energy=1e-3,           # was 3.0e-05 $ | now 1.6e-04 $, 2026-08-24  (6x)
)


def _clean(got, want):
    """Both solves must be trustworthy before their outputs are compared."""
    scale = DELTA * 10_000.0
    assert float(got["dual_residual"]) / scale < 1e-6, "implementation not clean"
    assert float(want["dual_residual"]) / scale < 1e-6, "reference not clean"


#: The five `(req_frac, level)` the comparison tests run.  The first three are
#: the original one-at-a-time row; the last two are the corners the 2026-08-24
#: full-3x3 measurement found to be worst -- `0.30/5.0` for seven of the ten
#: quantities, `0.30/900.0` for `award`.  Both were outside the old grid, which
#: is why its numbers were 1.8x to 2.7x low.  Not the full cross product: the two
#: points left out are worst for nothing, and nine builds per test at ~1.15 s
#: would roughly double this file.
#:
#: **Measured proof that the two added points bite, not just widen.**  Set
#: `reserve_shortfall` to 2e-2 -- between the old grid's worst (1.7e-02) and the
#: full grid's (3.4e-02) -- and exactly one case fails: `[0.3-5.0]`, the corner.
#: The three original points pass.  So the addition is what would catch a bound
#: chosen from the old measurement, which is the whole reason for it.
RUN_POINTS = [(0.30, 200.0), (0.70, 200.0), (0.95, 200.0),
              (0.30, 5.0), (0.30, 900.0)]


@pytest.mark.parametrize("req_frac,level", RUN_POINTS)
def test_primal_quantities_match_the_reference(pair, req_frac, level):
    both, _ = pair
    got, want = both(req_frac=req_frac, level=level)
    _clean(got, want)
    for key in ("award", "shed", "reserve_shortfall"):
        np.testing.assert_allclose(got[key], want[key], rtol=0,
                                   atol=ATOL[key], err_msg=key)
    np.testing.assert_allclose(got["reserve"], want["reserve"], rtol=0,
                               atol=RESERVE_MW_ATOL, err_msg="reserve")


@pytest.mark.parametrize("req_frac,level", RUN_POINTS)
def test_prices_match_the_reference(pair, req_frac, level):
    both, _ = pair
    got, want = both(req_frac=req_frac, level=level)
    _clean(got, want)
    for key in ("lmp", "reserve_price"):
        np.testing.assert_allclose(got[key], want[key], rtol=0,
                                   atol=ATOL[key], err_msg=key)


@pytest.mark.parametrize("req_frac", [0.30, 0.70, 0.95])
def test_capacity_dual_satisfies_complementary_slackness(pair, req_frac):
    """(CS)'s multiplier is checked by a property, because it is not unique.

    **Comparing it elementwise against the reference is comparing a choice, not
    a quantity.**  At `cap 0.6 / ramp 1.00` the two solves disagree on it by up
    to 40.95 in absolute terms and 87% in relative terms on 16 of 66 units,
    while every quantity that reaches money agrees to machine precision on the
    same run points: `award` to 8.1e-16 relative, the objective to 4.9e-16 (bit
    identical at one requirement fraction), `lmp` and `reserve_price` to 1e-10.
    Two different duals beside one primal solution and one objective is the
    definition of a non-unique dual, not of an implementation error, and the
    scenario made more units sit on their (CS) boundary.

    `capacity_dual` reaches no settlement expression -- `settle()` does not take
    it -- so this is not a money check; it is in the observation, so it is not
    nothing either.

    **The property holds for any valid dual solution**: a unit with a positive
    multiplier on (CS) must have that row tight.  (CS) is
    ``sum_k g + sum_j r <= (p_max - p_min) * u``, so its slack is
    ``p_max * u - award - sum_j reserve``.  What this still catches is what the
    elementwise assertion was really protecting: a wrong row index or a flipped
    sign puts the multiplier on units whose row is slack, and that violates the
    property immediately (measured: offsetting the (CS) block by one unit
    breaks it on 12 units, negating it on 18).

    The non-emptiness precondition is not decoration: on a run point where no
    unit is at its capacity boundary every implication below is vacuously true,
    and a test that passes by having nothing to check is the failure mode this
    file has already recorded once.
    """
    both, spec = pair
    got, want = both(req_frac=req_frac)
    _clean(got, want)
    pmax = np.asarray(spec["p_max"], np.float64)
    u = np.asarray(got["award"]) > 0.0

    for name, out in (("implementation", got), ("reference", want)):
        dual = np.asarray(out["capacity_dual"])
        slack = pmax * u - np.asarray(out["award"]) - np.asarray(out["reserve"]).sum(1)
        binding = dual > ATOL["capacity_dual"]
        assert binding.any(), (
            f"{name}: no unit carries a positive (CS) multiplier at req_frac "
            f"{req_frac}, so this test would pass without checking anything; "
            f"choose a run point where the capacity constraint binds")
        worst = float(np.max(np.abs(slack[binding]))) if binding.any() else 0.0
        assert worst <= RESERVE_MW_ATOL, (
            f"{name}: a unit carries a positive (CS) multiplier while its row "
            f"is slack by {worst:.3e} MW, which violates complementary "
            f"slackness; the multiplier is on the wrong row or has the wrong "
            f"sign")


def test_objective_matches_the_reference(pair):
    both, _ = pair
    got, want = both()
    _clean(got, want)
    assert abs(got["z"] - want["z"]) <= Z_RTOL * abs(want["z"])


def test_revenue_matches_the_reference(pair):
    """The judgement the quantity comparison defers to: money, not megawatts.

    Comparing money instead of megawatts is only legitimate under one extra
    condition, and this test asserts it rather than relying on another test to:
    **the objective must agree too**.  Objective agreeing plus revenue agreeing
    means the megawatt difference lies on an equivalent optimal face and carries
    no money.  Objective agreeing while revenue does not would mean the
    megawatt difference does carry money, and then it must fail.  Without the
    objective assertion here, relaxing it elsewhere would silently remove the
    licence for this comparison.
    """
    both, spec = pair
    got, want = both()
    _clean(got, want)
    assert abs(got["z"] - want["z"]) <= Z_RTOL * abs(want["z"]), (
        "the objective disagrees, so a megawatt difference cannot be excused "
        "by the money agreeing")
    bus = np.asarray(spec["unit_bus"])
    for name, f in (("reserve", lambda o: DELTA * (o["reserve"] * o["reserve_price"]).sum(1)),
                    ("energy", lambda o: DELTA * (o["award"] * o["lmp"][bus]))):
        np.testing.assert_allclose(f(got), f(want), rtol=0,
                                   atol=REVENUE_ATOL[name], err_msg=name)


@pytest.mark.parametrize("level", [5.0, 200.0, 900.0])
def test_agreement_holds_across_the_offer_level(pair, level):
    """The separation is what matters, not the level; both are checked."""
    both, _ = pair
    got, want = both(level=level)
    _clean(got, want)
    np.testing.assert_allclose(got["reserve_price"], want["reserve_price"],
                               rtol=0, atol=ATOL["reserve_price"])
    np.testing.assert_allclose(got["reserve"], want["reserve"], rtol=0,
                               atol=RESERVE_MW_ATOL)


def test_a_tied_offer_profile_reports_itself_untrustworthy(pair):
    """Below the separation threshold the duals are not trustworthy, and this
    asserts that the solve *says so* rather than returning a quiet wrong price.

    **This test is expected to fail if the solver is ever improved**, and that
    failure is the point: it pins a limitation that holds today, so that the
    limitation disappearing becomes an event rather than a silent benefit.  If
    it goes red, do not relax it.  Re-measure the separation threshold with a
    separation sweep, then update the separation figure in the specification.

    **Two preconditions come first, because a clean answer has two causes and
    only one of them is "the limitation is gone".**  The other is that the run
    point has left the region where a tie can affect anything: in shortage, and
    for any offer above the cap, the reserve price is pinned at `VOLR` by (RD)
    rather than set by an offer, so tied offers are irrelevant and the solve is
    clean for a reason that says nothing about separation.  That is what
    happened once already -- see `TIED_CASE`.  Asserting the preconditions turns
    that failure mode from a quiet pass into a loud one.

    **The corroboration this docstring used to offer no longer reproduces, and
    that is recorded rather than deleted.**  It read: at these tied run points
    the reserve price has been measured at 267.38 \\$/MWh while `VOLR` was 250,
    and since section 7 makes `lambda_res <= VOLR` an identity, a price above the
    cap is a dual that cannot be believed at all rather than a large number
    needing a threshold.  That argument is still the right shape -- an identity
    violation needs no gate -- but **measured 2026-08-24 at the `TIED_CASE`
    below, `max(lambda_res)` is 200.098 on CPU and 199.99998 on GPU against
    `VOLR` 250, so the identity holds and there is no violation to point at.**
    The 267.38 belongs to an earlier configuration.  So the 1e-6 gate is
    currently the only criterion here, which is exactly why the third cause
    below had to be dealt with rather than absorbed.

    **This runs on CPU, and that is a scope restriction rather than a
    relaxation.**  The gate is unchanged.  What changed is where the quantity is
    read, because on GPU it is not reproducible between processes.  Measured
    2026-08-24, `residual` at this run point, one value per freshly started
    process:

        CPU, 8 processes    2.91841526621084325e-05  all eight, bit-identical
        GPU, 8 processes    1.71023055674228891e-04  x3
                            1.44640074704568626e-05  x2
                            9.38629136848589902e-06  x1
                            4.84402562517516325e-07  x2   <-- below the gate

    Inside one process five repeats are bit-identical on both, so this is not
    run-to-run noise; it is per-process kernel selection landing on one of four
    discrete outcomes.  **Two of eight fall below 1e-6, i.e. this test was red
    about a quarter of the time on GPU** -- and the full-suite run that first
    surfaced it reported `4.844025625175163e-07`, the same value to every digit
    and on a different card, which is what identifies the mechanism as
    per-process selection rather than anything about the device.

    The flakiness is not new and nothing here introduced it; it was found by
    running the suite twice on two cards and comparing the failure sets,
    which is the only way a 25% flake shows up as anything other than
    a mystery.  Pinning the platform is the right response for a
    quantity that is reproducible only inside one environment: the property
    asserted -- that a fully tied profile leaves a dual residue at this run
    point -- is a property of the algorithm and the run point, not of the GPU.
    On the pinned platform the margin is 29x.
    """
    both, spec = pair

    # Precondition one is checked before the solve because it needs no solve,
    # and because it is the *cause* of precondition two rather than an
    # independent condition: an offer above the cap is never bought -- paying
    # the VOLR shortfall penalty is cheaper -- so `level >= VOLR` produces a
    # shortage, and the shortage assertion would fire first and report the
    # symptom while this reports the cause.  Measured: at VOLR 250 a tied level
    # of 900 leaves 307.7 MW unmet even at req_frac 0.02.
    assert TIED_CASE["level"] < spec["volr"], (
        f"the tied offer level {TIED_CASE['level']} is at or above VOLR "
        f"{spec['volr']}, so no reserve is bought at it and the price is the "
        f"cap rather than an offer; express the level as a fraction of VOLR")

    # Read on CPU: see the docstring.  `default_device` is set around the solve
    # rather than the fixture so that only this test's device changes.
    with jax.default_device(jax.devices("cpu")[0]):
        got, _want = both(**TIED_CASE)
    # precondition two: the requirement is met, so the price is set by an offer
    # rather than pinned at the cap by a shortage
    shortfall = float(np.asarray(got["reserve_shortfall"]).sum())
    assert shortfall <= 1e-6, (
        f"this run point is in shortage ({shortfall:.3e} MW unmet), so the "
        f"reserve price is pinned at VOLR and a tied profile cannot reach it; "
        f"the test would pass or fail for a reason unrelated to separation. "
        f"req_frac is a fraction of what the committed fleet may sell, which "
        f"moves with ramp_scale -- lower it until this holds")

    residual = float(got["dual_residual"]) / (DELTA * float(spec["voll"]))
    assert residual > 1e-6, (
        "a fully tied reserve offer profile solved cleanly; the separation "
        "threshold this market documents may no longer hold, see the docstring")
