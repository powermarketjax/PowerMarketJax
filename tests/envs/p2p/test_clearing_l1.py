"""L1 for the P2P clearing operator: domain correctness (§17).

Four constructed cases and four sampled property groups.  The constructed cases
are hand-worked in the docstrings below, because a sign error in the award or an
off-by-one in the search for the marginal position survives every aggregate
identity untouched (§17) and only an example with known values catches it.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.envs.p2p import make_clearing
from tests.envs.p2p.reference import population

PI_EXP, PI_RET = 4.1, 26.11


def run(price, q_sell, q_buy, pi_exp=PI_EXP, pi_ret=PI_RET):
    clear, _ = make_clearing(len(price), pi_exp, pi_ret)
    return jax.jit(clear)(jnp.asarray(np.asarray(price, np.float32)),
                          jnp.asarray(np.asarray(q_sell, np.float32)),
                          jnp.asarray(np.asarray(q_buy, np.float32)))


# --------------------------------------------------------------------------
# constructed cases (§17)
# --------------------------------------------------------------------------

def test_truthful_baseline_supply_scarce():
    """§4 truthful bidding, supply the scarcer side: the price is `pi_ret`.

    Every seller asks `pi_exp` and every buyer bids `pi_ret`, so both sides are
    entirely tied and §6.4 decides the awards by index alone.  Supply totals 4
    against demand 9, so the buy side is not exhausted, `D_plus` is a real
    submission at `pi_ret`, and the sell side is exhausted so `S_plus` takes the
    `pi_ret` sentinel.  Both bounds are then `pi_ret`.
    """
    price = [PI_EXP] * 3 + [PI_RET] * 3
    out = run(price, [1., 2., 1., 0., 0., 0.], [0., 0., 0., 3., 3., 3.])
    assert float(out["traded_volume"]) == pytest.approx(4.0, abs=1e-6)
    assert float(out["clearing_price"]) == np.float32(PI_RET)
    # index priority under a full tie: the two lowest indices fill the volume
    np.testing.assert_allclose(np.asarray(out["award_buy"]), [0, 0, 0, 3, 1, 0],
                               atol=1e-6)


def test_truthful_baseline_demand_scarce():
    """The mirror case: demand is scarcer, so the price is `pi_exp`."""
    price = [PI_EXP] * 3 + [PI_RET] * 3
    out = run(price, [3., 3., 3., 0., 0., 0.], [0., 0., 0., 1., 2., 1.])
    assert float(out["traded_volume"]) == pytest.approx(4.0, abs=1e-6)
    assert float(out["clearing_price"]) == np.float32(PI_EXP)
    np.testing.assert_allclose(np.asarray(out["award_sell"]), [3, 1, 0, 0, 0, 0],
                               atol=1e-6)


def test_no_trade():
    """Highest bid below lowest ask: nothing trades and the price sits between.

    Asks are 20, 22, 24 and bids 14, 12, 10.  `Q* = 0`, so both curves are
    undefined there and take their sentinels, `S* = pi_exp` and `D* = pi_ret`.
    The rejected marginals are real submissions: `S+ = 20`, `D+ = 14`.  The
    interval is [14, 20] and the price is 17.
    """
    out = run([20., 22., 24., 10., 12., 14.],
              [1., 1., 1., 0., 0., 0.], [0., 0., 0., 1., 1., 1.])
    assert float(out["traded_volume"]) == 0.0
    assert float(out["award_sell"].sum()) == 0.0
    assert float(out["award_buy"].sum()) == 0.0
    assert float(out["price_interval_lo"]) == 14.0
    assert float(out["price_interval_hi"]) == 20.0
    assert float(out["clearing_price"]) == 17.0


def test_partial_fill_on_one_side():
    """`Q*` on a buy-side breakpoint, so exactly one seller is filled in part.

    Sellers  i0 (8, 5) and i1 (10, 5)     -> cumulative 5, 10
    Buyers   i2 (20, 3), i3 (18, 4), i4 (6, 5) -> cumulative 3, 7, 12

    The candidate 10 fails because the marginal bid there is 6 against an ask
    of 10, so `Q* = 7`, which is a buy-side breakpoint.  Every buyer is
    therefore full or empty and i1 takes 7 - 5 = 2 of the 5 it offered.  At
    `Q* = 7` the four marginals are `S* = 10`, `D* = 18`, `S+ = 10` (the first
    sell breakpoint above 7 belongs to i1 itself, which is what collapses the
    interval, §7) and `D+ = 6`.  The interval is the point {10}.
    """
    out = run([8., 10., 20., 18., 6.],
              [5., 5., 0., 0., 0.], [0., 0., 3., 4., 5.])
    assert float(out["traded_volume"]) == pytest.approx(7.0, abs=1e-6)
    np.testing.assert_allclose(np.asarray(out["award_sell"]), [5, 2, 0, 0, 0],
                               atol=1e-6)
    np.testing.assert_allclose(np.asarray(out["award_buy"]), [0, 0, 3, 4, 0],
                               atol=1e-6)
    assert float(out["price_interval_lo"]) == 10.0
    assert float(out["price_interval_hi"]) == 10.0
    assert float(out["clearing_price"]) == 10.0


def test_interval_of_positive_width_and_both_sentinels():
    """`Q*` a breakpoint of both sides: nobody is partial and the midpoint acts.

    Sellers  i0 (8, 5) and i1 (10, 5)   -> cumulative 5, 10
    Buyers   i2 (20, 6) and i3 (18, 4)  -> cumulative 6, 10

    Both sides are exhausted at `Q* = 10`, so both rejected marginals fall back
    on the §7 sentinels: `S+ = pi_ret` and `D+ = pi_exp`, and neither binds.
    With `S* = 10` and `D* = 18` the interval is [10, 18] and (PRC) publishes
    its midpoint 14.  This is the one shape in which the midpoint rule decides
    anything, and §7 measures it at about one period in forty.
    """
    out = run([8., 10., 20., 18.], [5., 5., 0., 0.], [0., 0., 6., 4.])
    assert float(out["traded_volume"]) == pytest.approx(10.0, abs=1e-6)
    np.testing.assert_allclose(np.asarray(out["award_sell"]), [5, 5, 0, 0], atol=1e-6)
    np.testing.assert_allclose(np.asarray(out["award_buy"]), [0, 0, 6, 4], atol=1e-6)
    assert float(out["price_interval_lo"]) == 10.0
    assert float(out["price_interval_hi"]) == 18.0
    assert float(out["clearing_price"]) == 14.0


def test_empty_and_one_sided_markets():
    """The three shapes in which the sentinels are the only values available."""
    mid = 0.5 * (PI_EXP + PI_RET)
    price = [mid] * 5
    empty = run(price, [0.] * 5, [0.] * 5)
    assert float(empty["traded_volume"]) == 0.0
    # both curves undefined everywhere: the interval is the whole §5 bracket
    assert float(empty["price_interval_lo"]) == np.float32(PI_EXP)
    assert float(empty["price_interval_hi"]) == np.float32(PI_RET)

    sellers = run(price, [1., 2., 3., 0., 0.], [0.] * 5)
    assert float(sellers["traded_volume"]) == 0.0
    assert float(sellers["price_interval_lo"]) == np.float32(PI_EXP)   # D+ sentinel
    assert float(sellers["price_interval_hi"]) == pytest.approx(mid, abs=1e-5)  # S+ real

    buyers = run(price, [0.] * 5, [1., 2., 3., 0., 0.])
    assert float(buyers["traded_volume"]) == 0.0
    assert float(buyers["price_interval_lo"]) == pytest.approx(mid, abs=1e-5)  # D+ real
    assert float(buyers["price_interval_hi"]) == np.float32(PI_RET)    # S+ sentinel


# --------------------------------------------------------------------------
# sampled properties (§17)
# --------------------------------------------------------------------------

N_PROP = 30
TRIALS = 300


def _populations(seed=0, n=N_PROP, trials=TRIALS, tie_grid=2):
    rng = np.random.default_rng(seed)
    return [population(rng, n, PI_EXP, PI_RET, tie_grid=tie_grid)
            for _ in range(trials)]


def test_awards_never_exceed_submissions_and_sum_to_the_volume():
    clear, _ = make_clearing(N_PROP, PI_EXP, PI_RET)
    cj = jax.jit(clear)
    for price, q_sell, q_buy in _populations():
        out = cj(jnp.asarray(price), jnp.asarray(q_sell), jnp.asarray(q_buy))
        aw_s = np.asarray(out["award_sell"])
        aw_b = np.asarray(out["award_buy"])
        q = float(out["traded_volume"])
        assert (aw_s >= 0).all() and (aw_b >= 0).all()
        assert (aw_s <= q_sell + 1e-6).all()
        assert (aw_b <= q_buy + 1e-6).all()
        assert aw_s.sum() == pytest.approx(q, rel=1e-5, abs=1e-6)
        assert aw_b.sum() == pytest.approx(q, rel=1e-5, abs=1e-6)


def test_at_most_one_participant_in_the_market_is_partially_filled():
    """§6.3, in the tight form: one in the market, not one per side.

    `Q*` is drawn from the breakpoints, so on whichever side supplied it the
    straddling participant is filled exactly to its own breakpoint.  The
    comparison carries a float32 tolerance because a cumulative sum can leave
    a fully filled participant a fraction of an ULP short (§16).
    """
    clear, _ = make_clearing(N_PROP, PI_EXP, PI_RET)
    cj = jax.jit(clear)
    tol = 1e-5
    for price, q_sell, q_buy in _populations():
        out = cj(jnp.asarray(price), jnp.asarray(q_sell), jnp.asarray(q_buy))
        aw_s = np.asarray(out["award_sell"])
        aw_b = np.asarray(out["award_buy"])
        partial = int(((aw_s > tol) & (aw_s < q_sell - tol)).sum()
                      + ((aw_b > tol) & (aw_b < q_buy - tol)).sum())
        assert partial <= 1, (partial, price, q_sell, q_buy)


def test_price_interval_is_non_empty_and_contains_the_price():
    clear, _ = make_clearing(N_PROP, PI_EXP, PI_RET)
    cj = jax.jit(clear)
    for price, q_sell, q_buy in _populations():
        out = cj(jnp.asarray(price), jnp.asarray(q_sell), jnp.asarray(q_buy))
        lo, hi, lam = (float(out["price_interval_lo"]), float(out["price_interval_hi"]),
                       float(out["clearing_price"]))
        assert lo <= hi + 1e-6
        assert lo - 1e-6 <= lam <= hi + 1e-6
        assert PI_EXP - 1e-4 <= lam <= PI_RET + 1e-4


def test_accepted_and_rejected_submissions_lie_on_the_right_side_of_the_price():
    """Every accepted ask at or below the price, every rejected ask at or above.

    Checked on `award` and `price` directly, which is why the four marginal
    submissions are not returned by `clear`.
    """
    clear, _ = make_clearing(N_PROP, PI_EXP, PI_RET)
    cj = jax.jit(clear)
    tol = 1e-4
    for price, q_sell, q_buy in _populations():
        out = cj(jnp.asarray(price), jnp.asarray(q_sell), jnp.asarray(q_buy))
        aw_s, aw_b = np.asarray(out["award_sell"]), np.asarray(out["award_buy"])
        lam = float(out["clearing_price"])
        assert (price[aw_s > 1e-6] <= lam + tol).all()
        assert (price[aw_b > 1e-6] >= lam - tol).all()
        # a rejected seller is one that submitted and was awarded nothing
        rejected_s = (q_sell > 1e-6) & (aw_s <= 1e-6)
        rejected_b = (q_buy > 1e-6) & (aw_b <= 1e-6)
        assert (price[rejected_s] >= lam - tol).all()
        assert (price[rejected_b] <= lam + tol).all()


def test_permutation_equivariance_without_ties():
    """§16, in the form that actually holds: exact when no two prices are equal.

    With distinct prices the sorted order is a function of the prices alone, so
    permuting the participants permutes the whole result and nothing else moves,
    bit for bit.  Ties are excluded here on purpose and covered by the test
    below: §6.4 breaks them by **index**, so relabelling the participants
    genuinely reorders the tied ones, and the claim in §16 that the order does
    not depend on the arrangement of the input means it does not depend on it
    beyond the index rule, not that the index rule has no effect.
    """
    clear, _ = make_clearing(N_PROP, PI_EXP, PI_RET)
    cj = jax.jit(clear)
    rng = np.random.default_rng(5)
    for price, q_sell, q_buy in _populations(seed=5, trials=200, tie_grid=None):
        assert len(np.unique(price)) == N_PROP     # the premise of this test
        base = cj(jnp.asarray(price), jnp.asarray(q_sell), jnp.asarray(q_buy))
        perm = rng.permutation(N_PROP)
        inv = np.empty(N_PROP, int)
        inv[perm] = np.arange(N_PROP)
        moved = cj(jnp.asarray(price[perm]), jnp.asarray(q_sell[perm]),
                   jnp.asarray(q_buy[perm]))
        assert float(moved["clearing_price"]) == float(base["clearing_price"])
        assert float(moved["traded_volume"]) == float(base["traded_volume"])
        np.testing.assert_array_equal(np.asarray(moved["award_sell"])[inv],
                                      np.asarray(base["award_sell"]))
        np.testing.assert_array_equal(np.asarray(moved["award_buy"])[inv],
                                      np.asarray(base["award_buy"]))


def test_ties_move_the_awards_but_not_the_price():
    """§6.4: a tie leaves the identity of the winners open, never the price.

    Relabelling the participants reorders whoever is tied at the margin, so the
    awards may land on different people and the traded volume may move in its
    last bits, because the sorted quantities are then summed in a different
    order.  The clearing price and the interval around it must not move at all:
    they are read off submitted prices, and reordering equal prices cannot
    change which value is read.
    """
    clear, _ = make_clearing(N_PROP, PI_EXP, PI_RET)
    cj = jax.jit(clear)
    rng = np.random.default_rng(6)
    moved_awards = 0
    for price, q_sell, q_buy in _populations(seed=6, trials=200, tie_grid=2):
        base = cj(jnp.asarray(price), jnp.asarray(q_sell), jnp.asarray(q_buy))
        perm = rng.permutation(N_PROP)
        inv = np.empty(N_PROP, int)
        inv[perm] = np.arange(N_PROP)
        out = cj(jnp.asarray(price[perm]), jnp.asarray(q_sell[perm]),
                 jnp.asarray(q_buy[perm]))
        assert float(out["clearing_price"]) == float(base["clearing_price"])
        assert float(out["price_interval_lo"]) == float(base["price_interval_lo"])
        assert float(out["price_interval_hi"]) == float(base["price_interval_hi"])
        assert float(out["traded_volume"]) == pytest.approx(
            float(base["traded_volume"]), rel=1e-6, abs=1e-6)
        if not np.array_equal(np.asarray(out["award_sell"])[inv],
                              np.asarray(base["award_sell"])):
            moved_awards += 1
    # the point of the test is that this happens and costs nothing in price
    assert moved_awards > 0


def test_zero_quantity_participants_change_nothing():
    """§6.1 and the padding rule of §14, at the two precisions they hold at.

    Appending participants that submit nothing leaves the price **bit
    identical** and awards them exactly zero, because a zero-quantity
    participant leaves `Q[j] == Q[j-1]` and can satisfy neither side of the
    interval condition.  The quantities move by float32 rounding only: the
    cumulative sum is a parallel scan whose association tree depends on the
    length of the array, so lengthening it perturbs the last bits (§16).
    """
    rng = np.random.default_rng(7)
    n = 20
    for _ in range(200):
        price, q_sell, q_buy = population(rng, n, PI_EXP, PI_RET, tie_grid=2)
        clear_n, _ = make_clearing(n, PI_EXP, PI_RET)
        base = jax.jit(clear_n)(jnp.asarray(price), jnp.asarray(q_sell),
                                jnp.asarray(q_buy))

        m = int(rng.integers(1, 8))
        pad_price = rng.uniform(PI_EXP, PI_RET, m).astype(np.float32)
        clear_m, _ = make_clearing(n + m, PI_EXP, PI_RET)
        padded = jax.jit(clear_m)(
            jnp.asarray(np.concatenate([price, pad_price])),
            jnp.asarray(np.concatenate([q_sell, np.zeros(m, np.float32)])),
            jnp.asarray(np.concatenate([q_buy, np.zeros(m, np.float32)])))

        assert float(padded["clearing_price"]) == float(base["clearing_price"])
        assert float(padded["price_interval_lo"]) == float(base["price_interval_lo"])
        assert float(padded["price_interval_hi"]) == float(base["price_interval_hi"])
        assert float(np.asarray(padded["award_sell"])[n:].sum()) == 0.0
        assert float(np.asarray(padded["award_buy"])[n:].sum()) == 0.0
        np.testing.assert_allclose(np.asarray(padded["award_sell"])[:n],
                                   np.asarray(base["award_sell"]),
                                   rtol=1e-6, atol=1e-6)
        assert float(padded["traded_volume"]) == pytest.approx(
            float(base["traded_volume"]), rel=1e-6, abs=1e-6)
