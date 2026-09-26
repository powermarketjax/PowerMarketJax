"""L2 numerical equivalence for the two-settlement rule.

Same arithmetic, two routes: the implementation vectorises §8 and the reference
loops over units, periods, buses and lines.  **The solve is not covered here and
that is deliberate** -- this market's clearing *is* the day-ahead operator at
`T = 1`, so the day-ahead L2 covers it against its own reference, and a second
copy would be a transcription rather than an independent route (the local
flexibility market makes the same declaration).

**Every tolerance below is derived from a measured error**, with the derivation
beside it.  The repository has a recorded failure of the opposite habit: its
first equivalence layer declared tolerances four to five orders of magnitude
looser than the errors they bounded, which makes a comparison that cannot fail.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.real_time import load_da_position
from powermarketjax.envs.real_time.boundary import make_boundary
from powermarketjax.envs.real_time.clearing import make_rt_clearing
from powermarketjax.envs.real_time.demand import load_gb_demand_half_hourly
from powermarketjax.envs.real_time.settlement import make_settlement
from powermarketjax.envs.day_ahead.clearing import segment_costs

from . import reference

#: Split onto separate lines: as a tuple unpacking this was invisible to both
#: census forms (no `cap_scale=` to find, and `^CAP_SCALE` misses a line that
#: starts with `DELTA`).
DELTA = 0.5
#: The adopted scenario, stated rather than read back from the
#: fixture so the two remain independent statements.
CAP_SCALE = 0.60
RAMP_SCALE = 1.00
#: `_seasons` is the expand-phase name; batch 3 renames it back.
CHAIN = "step1prime_seasons"

#: Relative bound on the settlement legs.  **Re-measured 2026-08-17 on the
#: adopted scenario** (`cap_scale` 0.60, `ramp_scale` 1.00, four seasonal
#: segments, VOLL 10000, CPU, float64), over this module's five operating points,
#: taken where `|reference| > 1`:
#:
#:     revenue_da   0.000e+00      bit-identical
#:     revenue_rt   0.000e+00      bit-identical
#:     cost         3.800e-16
#:     profit       1.340e-15      about six ULP
#:
#: The first measurement is kept because the bound was set from it and the
#: structure it records still holds: 2026-08-15, eight operating points,
#: `cap_scale` 0.4 / `ramp_scale` 0.25 / old sixty consecutive days, RTX 4500
#: Ada, `jax` 0.10.2 -- `cost` 2.300e-16 and `profit` 9.382e-16.  The scenario
#: change moved the worst by well under an order and left the two revenue legs
#: bit-identical, so the bound stands rather than being re-derived.
#:
#: The two revenue legs agree **exactly**, because each is a sum of products that
#: both routes evaluate in the same order; `cost` does not, because it carries
#: the cubic integral of §3.2 whose terms the two accumulate differently, and
#: `profit` inherits that.  The bound sits two orders above the worst.
#:
#: **The first version of this constant carried an invented measurement** -- a
#: figure written into the comment before the sweep was run, which came out an
#: order and a half from the truth.  It is the same failure this file's own
#: preamble warns about, committed in the act of warning about it, and the fix is
#: mechanical rather than attitudinal: run the sweep, then write the number.
MONEY_RTOL = 1e-13

#: Relative bound on the money-balance terms -- looser than the settlement legs
#: by five orders because these terms carry the `PTDF` products, which the
#: implementation forms as one matrix multiply and the reference accumulates line
#: by line.
#:
#: **Re-measured 2026-08-17 on the adopted scenario** (same run point as above),
#: on the two points of `MONEY_BALANCE_POINTS`, taken where `|reference| > 1`:
#: worst 7.723e-13 (`left`; `right` and `rent` 1.622e-13, `shed_term` exactly 0).
#: Originally measured worst 1.650e-11 on the old scenario over eight points, and
#: the bound was set ~60x above that.  It now sits ~1300x above the worst, which
#: is looser than intended rather than tighter: **the margin is recorded, not
#: silently enjoyed**, because a bound that has drifted this far from its
#: measurement stops discriminating.  Tightening it is a separate decision from
#: this migration and is not taken here.
#:
#: The comparison is `rtol` beside `atol=1e-6`, and on periods with no binding
#: line the terms are ~1e-17, so `atol` is what carries those points -- the
#: relative figures above are meaningful only where the quantity is non-trivial,
#: which is why `MONEY_BALANCE_POINTS` selects for exactly that.
BALANCE_RTOL = 1e-9


@pytest.fixture(scope="module", autouse=True)
def x64():
    prev = jax.config.jax_enable_x64, jax.config.jax_default_matmul_precision
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    yield
    jax.config.update("jax_enable_x64", prev[0])
    jax.config.update("jax_default_matmul_precision", prev[1])


@pytest.fixture(scope="module")
def rig(x64):
    pos = load_da_position(chain=CHAIN)
    # no `make_env` here to enforce it, so the agreement is asserted directly
    assert (pos["meta"]["cap_scale"], pos["meta"]["ramp_scale"]) == \
        (CAP_SCALE, RAMP_SCALE), (
            f"position built at cap {pos['meta']['cap_scale']} / ramp "
            f"{pos['meta']['ramp_scale']}, but this module clears at "
            f"{CAP_SCALE} / {RAMP_SCALE}")
    case = load_case(pos["meta"]["case"])
    hh, _d = load_gb_demand_half_hourly()
    clear, _s = make_rt_clearing(case, n_segments=1, cap_scale=CAP_SCALE,
                                 ramp_scale=RAMP_SCALE, period_hours=DELTA)
    boundary, _b = make_boundary(case, n_segments=1, cap_scale=CAP_SCALE,
                                 period_hours=DELTA)
    settle, mb = make_settlement(case, period_hours=DELTA, cap_scale=CAP_SCALE)
    _w, cost = segment_costs(case, 1)
    return pos, case, hh, jax.jit(clear), boundary, settle, mb, jnp.asarray(cost)[:, :, None]


def _one_period(rig, day, t_rt):
    pos, case, hh, clear, boundary, settle, mb, offer = rig
    h = t_rt // 2
    di = int(np.asarray(pos["day_index"])[day])
    u = jnp.asarray(pos["u"][day][:, h:h + 1], jnp.float64)
    q_da = jnp.asarray(pos["q_da"][day][:, h:h + 1], jnp.float64)
    lmp_da = jnp.asarray(pos["lmp_da"][day][h:h + 1], jnp.float64)
    s_da = jnp.asarray(pos["s_da"][day][h:h + 1], jnp.float64)
    d_da = jnp.asarray(pos["d_da"][day][h:h + 1], jnp.float64)
    demand = jnp.asarray([float(hh[di, t_rt])])
    p_prev, _ = boundary(u, demand)
    out = clear(offer, u, demand, p_prev)
    share = np.asarray(pos["d_da"][day][h]) / float(np.asarray(pos["d_da"][day][h]).sum())
    demand_bus = jnp.asarray(share[None, :] * float(demand[0]))
    return out, u, q_da, lmp_da, s_da, d_da, demand_bus


@pytest.mark.parametrize("day,t_rt", [(0, 0), (10, 3), (16, 3), (48, 20), (4, 30)])
def test_settlement_matches_the_reference(rig, day, t_rt):
    """The two legs and the profit, elementwise against the loop version."""
    pos, case, hh, clear, boundary, settle, mb, offer = rig
    out, u, q_da, lmp_da, s_da, d_da, demand_bus = _one_period(rig, day, t_rt)
    assert float(out["mu"]) < 1e-6

    mine = settle(out["award"], out["lmp"], u, jnp.zeros(len(case.unit_p_min)),
                  q_da, lmp_da)
    theirs = reference.settle(case, out["award"], out["lmp"], u,
                              np.zeros(len(case.unit_p_min)), q_da, lmp_da, DELTA)
    for key in ("revenue_da", "revenue_rt", "cost", "profit"):
        np.testing.assert_allclose(np.asarray(mine[key]), theirs[key],
                                   rtol=MONEY_RTOL, atol=1e-9, err_msg=key)


#: Chosen for the congestion rent, not inherited.  Under the old sixty
#: consecutive days `(10, 3)` and `(16, 3)` both carried a binding line; under
#: the four seasonal segments they carry none, and the rent term is then exactly
#: zero -- the precondition at the end of this test fails rather than the test
#: passing on a comparison of two zeros.  Scanned 45 candidate periods on the
#: adopted window (cap 0.60 / ramp 1.00 / VOLL 10000, 2026-08-17): four have both
#: `|left| > 1` and `|rent| > 1`, and these two are the pair on different days.
#: `(40, 3)` carries rent 48.1 $, `(48, 40)` carries 7.8 $.
MONEY_BALANCE_POINTS = [(40, 3), (48, 40)]


@pytest.mark.parametrize("day,t_rt", MONEY_BALANCE_POINTS)
def test_money_balance_matches_the_reference(rig, day, t_rt):
    """Both sides of §8's identity, and the split into rent and shed term.

    Comparing the two sides against each other is the *domain* check and lives in
    L1; what this adds is that the two implementations of each side agree, which
    is the only thing that separates "the identity holds" from "both routes make
    the same mistake".
    """
    pos, case, hh, clear, boundary, settle, mb, offer = rig
    out, u, q_da, lmp_da, s_da, d_da, demand_bus = _one_period(rig, day, t_rt)

    mine = mb(out["award"], out["lmp"], out["shed"], demand_bus, q_da, s_da, d_da,
              out["line_dual_up"], out["line_dual_dn"], out["shed_dual"])
    theirs = reference.money_balance(
        case, out["award"], out["lmp"], out["shed"], demand_bus, q_da, s_da, d_da,
        out["line_dual_up"], out["line_dual_dn"], out["shed_dual"], DELTA, CAP_SCALE)
    for key in ("left", "right", "rent", "shed_term"):
        a, b = float(mine[key]), float(theirs[key])
        np.testing.assert_allclose(a, b, rtol=BALANCE_RTOL, atol=1e-6, err_msg=key)
    # and the pieces are not all zero, or the agreement says nothing
    assert abs(float(mine["left"])) > 1.0 and abs(float(mine["rent"])) > 1.0


def test_the_reference_disagrees_when_the_implementation_is_wrong(rig):
    """L2 must be able to fail, so a deliberate perturbation is compared too.

    Without this the agreement above could hold because both routes read the same
    inputs and return them unchanged; here the reference is given a `q_da` the
    implementation did not see, and the legs must part.
    """
    pos, case, hh, clear, boundary, settle, mb, offer = rig
    # Run point chosen for price separation, not at random: `profit` depends on
    # `q_da` only through the coefficient `(lmp_da - lmp_rt)`, so wherever the
    # two prices agree this injection is dead no matter how large `q_da` is.
    # The previous point (day 10, period 3) had |dlmp| = 1.5e-11 and the halving
    # of a 7081 MW position moved profit by 2.24e-08 $ -- the test passed only
    # because that noise sat one decade above `np.allclose`'s tolerance, so
    # whether it went red depended on XLA's algorithm choice under memory
    # pressure rather than on the implementation. Day 16 period 18 separates the
    # prices by congestion, which this scenario has structurally (2873/2880
    # cells with a line binding); a scarcity-driven point separates them further
    # but depends on shed occurring, and successive scenario fixes have removed
    # shed more than once.
    out, u, q_da, lmp_da, s_da, d_da, demand_bus = _one_period(rig, 16, 18)
    # What must be non-zero is the money the injection can move, not the
    # quantity it perturbs.  Guarding `max|q_da| > 0` does not do it: `q_da` was
    # 7081 MW at the dead run point.
    coef = float(np.abs(np.asarray(lmp_da) - np.asarray(out["lmp"])).max())
    movable = coef * float(np.abs(np.asarray(q_da)).max())
    #: Measured 1.278872e+06 $ on 2026-08-19, CPU float64, cap 0.60 / ramp 1.00,
    #: four-season 4 x 15 day window, at day 16 period 18.  The bound is three
    #: decades below that and still twelve decades above the `np.allclose`
    #: tolerance at this profit scale, so it admits ordinary drift in the run
    #: point while refusing a point where the channel has closed.
    MOVABLE_FLOOR = 1.0e3
    assert movable > MOVABLE_FLOOR, (
        f"profit is insensitive to q_da at this run point: price gap {coef:.3e} "
        f"$/MWh over max |q_da| {float(np.abs(np.asarray(q_da)).max()):.1f} MW "
        f"moves at most {movable:.3e} $, below the {MOVABLE_FLOOR:.0e} $ floor. "
        "The injection cannot bite here, so this test would assert that two "
        "near-identical numbers differ and would pass or fail on float noise")
    mine = settle(out["award"], out["lmp"], u, jnp.zeros(len(case.unit_p_min)),
                  q_da, lmp_da)
    theirs = reference.settle(case, out["award"], out["lmp"], u,
                              np.zeros(len(case.unit_p_min)),
                              np.asarray(q_da) * 0.5, lmp_da, DELTA)
    assert not np.allclose(np.asarray(mine["profit"]), theirs["profit"],
                           rtol=MONEY_RTOL, atol=1e-9)
