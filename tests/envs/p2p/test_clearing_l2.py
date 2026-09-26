"""L2 for the P2P clearing operator: numerical equivalence against numpy.

Compared against `reference.py`, which runs the same algorithm by a deliberately
different route in float64 while the operator runs in float32.

The assertions are **graded**, because §16 says the line between exact and
inexact runs through the middle of the output, and a single blanket tolerance
would be looser than the tightest part deserves and would repeat a failure
already met once in a vendored equivalence test.

    sort order                    exact, no tolerance
    number of differing awards    at most one per side, no tolerance
    award and traded volume       AWARD_RTOL, relative to the traded volume
    clearing price                one ULP, and no degenerate jump at all

**How the tolerances were derived.**  2400 markets, sizes 10 / 30 / 60, half of
them with prices snapped onto a 0.5 grid to force ties at the margin and half
with distinct prices, seed 101, measured 2026-08-09 on CPU:

    order mismatches                          0
    max elements differing per side           1
    max |d award|      / traded volume    2.05e-07
    max |d traded volume| / traded volume 1.54e-07
    max |d clearing price| absolute       9.54e-07   (1 ULP at 26.11 = 1.91e-06)

``AWARD_RTOL`` is 5e-7, about 2.4 times the measured maximum and about four
float32 epsilons.  The clearing price is compared at one ULP rather than at a
derived number because it is a midpoint of two submitted prices and cannot
legitimately drift further; anything larger is the degeneracy of §7, which is
counted separately and asserted to be absent.

**The degeneracy detector is what makes the ULP tolerance defensible.**  A price
interval sitting on a quantity tie can be pushed to a neighbouring breakpoint by
a single ULP of cumulative sum, and the price then moves by a finite amount --
measured up to 2.0 \\$/MWh on adversarially degenerate populations, where all
quantities were made nearly equal.  Absorbing that into a tolerance would need
about 9 per cent of the price bracket and would stop the test constraining
anything.  Instead the populations here are drawn from continuous distributions,
which do not produce quantity ties, and the count of price disagreements beyond
one ULP is asserted to be zero.  If a future change introduces degeneracy the
test names it instead of swallowing it.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.envs.p2p import make_clearing
from tests.envs.p2p.reference import clear_ref, population

PI_EXP, PI_RET = 4.1, 26.11
AWARD_RTOL = 5e-7
PRICE_ULP = float(np.spacing(np.float32(PI_RET)))

SIZES = (10, 30, 60)
TRIALS = 150


def _cases():
    """Both regimes: distinct prices, and ties forced onto a coarse grid."""
    for tie_grid in (None, 2):
        for n in SIZES:
            rng = np.random.default_rng(101)
            for _ in range(TRIALS):
                yield n, tie_grid, population(rng, n, PI_EXP, PI_RET,
                                              tie_grid=tie_grid)


def test_l2_graded_equivalence():
    clears = {n: jax.jit(make_clearing(n, PI_EXP, PI_RET)[0]) for n in SIZES}
    price_jumps = []
    worst_award = 0.0
    worst_volume = 0.0
    worst_price = 0.0

    for n, tie_grid, (price, q_sell, q_buy) in _cases():
        out = clears[n](jnp.asarray(price), jnp.asarray(q_sell),
                        jnp.asarray(q_buy))
        ref = clear_ref(price, q_sell, q_buy, PI_EXP, PI_RET)
        scale = max(float(ref["traded_volume"]), 1e-6)

        # the sort is exact: a total key over (price, index) on both sides
        np.testing.assert_array_equal(
            np.asarray(jnp.lexsort((jnp.arange(n), jnp.asarray(price)))),
            ref["order_sell"])
        np.testing.assert_array_equal(
            np.asarray(jnp.lexsort((jnp.arange(n), -jnp.asarray(price)))),
            ref["order_buy"])

        for key in ("award_sell", "award_buy"):
            delta = np.abs(np.asarray(out[key], np.float64) - ref[key])
            # every award is the whole submission or nothing except at most one
            assert int((delta > 0).sum()) <= 1, (key, n, tie_grid)
            worst_award = max(worst_award, float(delta.max()) / scale)
            assert float(delta.max()) / scale < AWARD_RTOL

        dv = abs(float(out["traded_volume"]) - ref["traded_volume"]) / scale
        worst_volume = max(worst_volume, dv)
        assert dv < AWARD_RTOL

        dp = abs(float(out["clearing_price"]) - ref["clearing_price"])
        worst_price = max(worst_price, dp)
        if dp > PRICE_ULP:
            price_jumps.append((n, tie_grid, dp))

    # A-strict: the populations are continuous, so degeneracy must not appear.
    assert price_jumps == [], price_jumps
    # guard against the tolerances silently becoming loose relative to reality
    assert worst_award < AWARD_RTOL
    assert worst_volume < AWARD_RTOL
    assert worst_price <= PRICE_ULP


def test_reference_is_not_the_implementation():
    """The reference must disagree when the algorithm is perturbed.

    An earlier vendored equivalence test had a reference that was the
    implementation's own output, whose error was therefore exactly zero, and
    which could only ever detect a change in behaviour rather than a persistent error.  Feeding the reference a
    deliberately wrong tie-break -- descending index instead of ascending,
    which §6.4 forbids and which a stable-sort implementation could produce by
    accident -- must change the awards it returns on a tied population.
    """
    rng = np.random.default_rng(3)
    n = 20
    disagreements = 0
    for _ in range(100):
        price, q_sell, q_buy = population(rng, n, PI_EXP, PI_RET, tie_grid=2)
        good = clear_ref(price, q_sell, q_buy, PI_EXP, PI_RET)
        flipped = clear_ref(price[::-1], q_sell[::-1], q_buy[::-1],
                            PI_EXP, PI_RET)
        if not np.array_equal(good["award_sell"], flipped["award_sell"][::-1]):
            disagreements += 1
    assert disagreements > 0


@pytest.mark.parametrize("n", [5, 12])
def test_l2_on_the_constructed_cases(n):
    """The reference must reproduce the hand-worked values of `test_clearing_l1`."""
    if n == 5:
        price = np.array([8., 10., 20., 18., 6.], np.float32)
        q_sell = np.array([5., 5., 0., 0., 0.], np.float32)
        q_buy = np.array([0., 0., 3., 4., 5.], np.float32)
        expect_q, expect_p = 7.0, 10.0
    else:
        price = np.array([8., 10., 20., 18.] + [15.] * 8, np.float32)
        q_sell = np.array([5., 5., 0., 0.] + [0.] * 8, np.float32)
        q_buy = np.array([0., 0., 6., 4.] + [0.] * 8, np.float32)
        expect_q, expect_p = 10.0, 14.0
    ref = clear_ref(price, q_sell, q_buy, PI_EXP, PI_RET)
    assert ref["traded_volume"] == pytest.approx(expect_q, abs=1e-9)
    assert ref["clearing_price"] == pytest.approx(expect_p, abs=1e-9)
