"""L1 for the P2P settlement: §8, the two money-balance identities, and the
check that the reward is computed from awards rather than from submissions.

**Neither identity is taken relative to the aggregate profit.**  §16 says why:
it is a difference of two nearly equal sums and passes through zero, so a
tolerance relative to it has to be loosened by orders of magnitude to survive
the cases where the two sides almost cancel.
Each identity is measured against its own flow, which is not the same scale for
the two: the internal one against the money moving inside the market and the
external one against the money crossing to the grid.  Measured 2026-08-09, 1500
markets at each of three sizes, float32 throughout: the external identity
reached 1.8e-4 relative to the aggregate profit and 2.8e-7 relative to the gross
external flow, for the same absolute error of 3.1e-5, and the internal one
reached 3.3e-7 against the internal flow.  The tolerance below is 1e-6, about
three times the larger of the two measured maxima.

There is no episode in this file -- `EnvState` and `step` are tested elsewhere --
so the episode accumulation rule does not bind
and the identities are checked per period, where §16 says they are checked far
tighter than the 1e-5 an episode total would allow.
"""
import ast
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.envs.p2p import make_clearing, make_settlement
from powermarketjax.envs.p2p import settlement as settlement_module
from tests.envs.p2p.reference import population

PI_EXP, PI_RET = 4.1, 26.11
BALANCE_RTOL = 1e-6


_CLEARERS = {}


def _clearer(n):
    """One compiled operator per size; rebuilding it per trial recompiles."""
    if n not in _CLEARERS:
        _CLEARERS[n] = jax.jit(make_clearing(n, PI_EXP, PI_RET)[0])
    return _CLEARERS[n]


def _market(rng, n, kappa_scale=3.0):
    price, q_sell, q_buy = population(rng, n, PI_EXP, PI_RET, tie_grid=2)
    out = _clearer(n)(jnp.asarray(price), jnp.asarray(q_sell),
                      jnp.asarray(q_buy))
    throughput = rng.uniform(0.0, 1.0, n).astype(np.float32)
    kappa = rng.uniform(0.0, kappa_scale, n).astype(np.float32)
    return price, q_sell, q_buy, out, kappa, throughput


def _balances(q_sell, q_buy, out, kappa, throughput, money):
    """Each identity against **its own** scale, which is not the same one.

    The internal identity is money moving between the two sides inside the
    market, so it is measured against that flow, ``lambda * sum(award_sell)``.
    The external identity is money crossing to the grid, so it is measured
    against the gross external flow.  Using the external scale for both was
    tried and fails: when almost everything clears locally the external flow
    approaches zero while the internal one does not, and the ratio diverges for
    the same reason §16 rejects the aggregate profit as a denominator.
    """
    lam = float(out["clearing_price"])
    aw_s = np.asarray(out["award_sell"], np.float64)
    aw_b = np.asarray(out["award_buy"], np.float64)
    q_ex = q_sell.astype(np.float64) - aw_s
    q_im = q_buy.astype(np.float64) - aw_b
    c_deg = kappa.astype(np.float64) * throughput.astype(np.float64)

    internal = abs(lam * aw_s.sum() - lam * aw_b.sum())
    internal_scale = max(lam * aw_s.sum(), 1e-9)
    external = abs(float(np.asarray(money["profit"], np.float64).sum())
                   - (PI_EXP * q_ex.sum() - PI_RET * q_im.sum() - c_deg.sum()))
    external_scale = max(PI_EXP * q_ex.sum() + PI_RET * q_im.sum()
                         + c_deg.sum(), 1e-9)
    return internal / internal_scale, external / external_scale, q_ex, q_im


@pytest.mark.parametrize("kappa_scale", [0.0, 3.0])
def test_both_money_balance_identities(kappa_scale):
    """Both identities, at kappa = 0 and kappa > 0.

    The internal identity does not involve kappa at all -- the clearing price
    cancels out of the aggregate -- so running only the kappa = 0 case would
    leave the degradation term of the external identity untested.
    """
    settle = make_settlement(PI_EXP, PI_RET)
    sj = jax.jit(settle)
    worst_internal = worst_external = 0.0
    for n in (10, 30, 60):
        rng = np.random.default_rng(31)
        for _ in range(200):
            price, q_sell, q_buy, out, kappa, thr = _market(rng, n, kappa_scale)
            money = sj(jnp.asarray(q_sell), jnp.asarray(q_buy),
                       out["award_sell"], out["award_buy"],
                       out["clearing_price"], jnp.asarray(kappa),
                       jnp.asarray(thr))
            internal, external, _, _ = _balances(
                q_sell, q_buy, out, kappa, thr, money)
            worst_internal = max(worst_internal, internal)
            worst_external = max(worst_external, external)
    assert worst_internal < BALANCE_RTOL, worst_internal
    assert worst_external < BALANCE_RTOL, worst_external


def test_residual_quantities_are_never_negative():
    """(AWD) never awards more than was submitted, so both residuals are >= 0.

    Zero tolerance: this is a structural property of the minimum in (AWD), not
    a numerical one, and it was satisfied in every one of 4500 markets measured
    while deriving the tolerances above.
    """
    settle = make_settlement(PI_EXP, PI_RET)
    for n in (10, 30, 60):
        rng = np.random.default_rng(32)
        for _ in range(200):
            price, q_sell, q_buy, out, kappa, thr = _market(rng, n)
            money = jax.jit(settle)(
                jnp.asarray(q_sell), jnp.asarray(q_buy), out["award_sell"],
                out["award_buy"], out["clearing_price"], jnp.asarray(kappa),
                jnp.asarray(thr))
            _, _, q_ex, q_im = _balances(q_sell, q_buy, out, kappa, thr, money)
            assert (q_ex >= 0).all()
            assert (q_im >= 0).all()
            assert not ((q_ex > 1e-9) & (q_im > 1e-9)).any()


def test_reward_is_not_derivable_from_the_submitted_prices():
    """Shift every submitted price by a constant: awards must not move at all.

    This is the P2P form of the day-ahead check that multiplies the offers and
    watches the awards stay put.  Adding the same constant to every submission
    leaves ``D_t - S_t`` unchanged at every candidate, so the crossing point,
    the awards and the traded volume are bit-identical, while the clearing
    price shifts by exactly that constant.  Profit must therefore change only
    through ``lambda * award``, by an amount that can be predicted in closed
    form.  Any path that read a submitted price instead of an award would move
    the awards or break the prediction.

    Prices are drawn well separated so that the shift cannot reorder two
    submissions or flip an admissibility comparison; with ties this would be a
    test of float32 rounding rather than of the settlement.
    """
    n, shift = 12, 3.0
    rng = np.random.default_rng(33)
    settle = make_settlement(PI_EXP, PI_RET)
    clear, _ = make_clearing(n, PI_EXP, PI_RET)
    cj, sj = jax.jit(clear), jax.jit(settle)

    for _ in range(50):
        # leave room above for the shift, and keep the prices distinct
        price = np.sort(rng.choice(np.arange(6.0, 20.0, 0.5), n, replace=False)
                        ).astype(np.float32)
        rng.shuffle(price)
        net = rng.normal(0.0, 1.0, n).astype(np.float32)
        q_sell = (np.maximum(net, 0.0) * 0.5).astype(np.float32)
        q_buy = (np.maximum(-net, 0.0) * 0.5).astype(np.float32)
        kappa = rng.uniform(0.0, 3.0, n).astype(np.float32)
        thr = rng.uniform(0.0, 1.0, n).astype(np.float32)

        base = cj(jnp.asarray(price), jnp.asarray(q_sell), jnp.asarray(q_buy))
        moved = cj(jnp.asarray(price + np.float32(shift)),
                   jnp.asarray(q_sell), jnp.asarray(q_buy))

        np.testing.assert_array_equal(np.asarray(moved["award_sell"]),
                                      np.asarray(base["award_sell"]))
        np.testing.assert_array_equal(np.asarray(moved["award_buy"]),
                                      np.asarray(base["award_buy"]))
        assert float(moved["traded_volume"]) == float(base["traded_volume"])
        assert float(moved["clearing_price"]) == pytest.approx(
            float(base["clearing_price"]) + shift, abs=1e-4)

        args = (jnp.asarray(q_sell), jnp.asarray(q_buy), base["award_sell"],
                base["award_buy"], None, jnp.asarray(kappa), jnp.asarray(thr))
        m0 = sj(*args[:4], base["clearing_price"], *args[5:])
        m1 = sj(*args[:4], moved["clearing_price"], *args[5:])

        # the whole change in profit is `shift * (award_sell - award_buy)`
        predicted = shift * (np.asarray(base["award_sell"], np.float64)
                             - np.asarray(base["award_buy"], np.float64))
        actual = (np.asarray(m1["profit"], np.float64)
                  - np.asarray(m0["profit"], np.float64))
        np.testing.assert_allclose(actual, predicted, atol=1e-3)


def test_the_constraint_channel_quantity_never_enters_settlement():
    """`clip` must not appear in `settlement.py`, checked on the syntax tree.

    The same check the day-ahead settlement carries for `voll`: a comment
    cannot enforce this and a numerical test cannot see it, because a clip
    added to `cost` would produce a finite, plausible number.  The market layer
    forbids a feasibility quantity from reaching `reward`, and `profit` reaches
    `reward` through `cost`.
    """
    tree = ast.parse(inspect.getsource(settlement_module))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree)
              if isinstance(node, ast.Attribute)}
    names |= {arg.arg for node in ast.walk(tree)
              if isinstance(node, ast.arg) for arg in [node]}
    for forbidden in ("clip", "cost_sum", "cost_soc_clip", "voll"):
        assert forbidden not in names, forbidden
