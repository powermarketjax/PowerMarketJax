"""L0 and L1 for the two reference pricing rules of §11.

They are baselines and not the mechanism, so they carry the same JAX
contract as everything else in this package but no `award` and no
`clearing_price`.  The two layers are kept in one file because the whole module
is two sums and a handful of `jnp.where`, and splitting eight assertions across
two files would say more about the convention than about the code.
"""
import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.envs.p2p import make_baseline_pricing

PI_EXP, PI_RET = 4.1, 26.11
MID = 0.5 * (PI_EXP + PI_RET)
N = 20


@pytest.fixture
def price_baselines():
    fn, _ = make_baseline_pricing(PI_EXP, PI_RET)
    return jax.jit(fn)


def _split(net):
    net = np.asarray(net, np.float32)
    return (jnp.asarray(np.maximum(net, 0.0)),
            jnp.asarray(np.maximum(-net, 0.0)))


def _population(rng, n=N):
    net = rng.normal(0.0, 1.0, n).astype(np.float32)
    net[rng.random(n) < 0.15] = 0.0
    return _split(net)


# --------------------------------------------------------------- L0
def test_rejects_a_bad_tariff_pair():
    with pytest.raises(ValueError):
        make_baseline_pricing(PI_RET, PI_EXP)


def test_jit_vmap_and_scan(price_baselines):
    fn, _ = make_baseline_pricing(PI_EXP, PI_RET)
    rng = np.random.default_rng(0)
    q_sell, q_buy = _population(rng)
    # Close, not equal, and only for the one field with contractible
    # arithmetic: `(pi_ret - pi_exp) * SDR + pi_exp` is a multiply-add that XLA
    # fuses under `jit` and evaluates in two rounded steps eagerly, measured at
    # 4.8e-7 absolute and 1e-7 relative.  `action.py` carries the same note for
    # the same reason; everything else here agrees bit for bit.
    eager, compiled = fn(q_sell, q_buy), price_baselines(q_sell, q_buy)
    chex.assert_trees_all_close(eager, compiled, rtol=1e-6, atol=1e-6)
    for key in eager:
        if key != "sdr_price_sell":
            assert float(eager[key]) == float(compiled[key]), key

    out = price_baselines(q_sell, q_buy)
    for key, value in out.items():
        assert value.shape == (), key
        assert value.dtype == jnp.float32, key

    batch = 8
    tiled = tuple(jnp.broadcast_to(a, (batch, N)) for a in (q_sell, q_buy))
    vout = jax.jit(jax.vmap(fn))(*tiled)
    for key, value in vout.items():
        assert value.shape == (batch,)
        assert bool(jnp.all(value == value[0])), key

    horizon = 32
    cols = [_population(rng) for _ in range(horizon)]
    xs = (jnp.stack([c[0] for c in cols]), jnp.stack([c[1] for c in cols]))
    _, prices = jax.jit(lambda xs: jax.lax.scan(
        lambda c, x: (c, fn(*x)["sdr_price_buy"]), jnp.float32(0.0), xs))(xs)
    assert prices.shape == (horizon,)
    assert bool(jnp.all(jnp.isfinite(prices)))


def test_no_nan_on_the_degenerate_aggregates(price_baselines):
    """Both rules divide by an aggregate that the empty and one-sided markets
    leave at exactly zero, and a `where` around an unguarded quotient still
    evaluates the branch not taken."""
    zeros = jnp.zeros((N,), jnp.float32)
    ones = jnp.ones((N,), jnp.float32)
    for q_sell, q_buy in ((zeros, zeros), (ones, zeros), (zeros, ones)):
        out = price_baselines(q_sell, q_buy)
        for key, value in out.items():
            assert bool(jnp.isfinite(value)), key


# --------------------------------------------------------------- L1
def test_the_rules_do_not_read_a_submitted_price():
    """The argument that they are not the mechanism, checked on the signature
    rather than on samples.

    Both rules price from the aggregates alone, so an agent whose quantity is
    fixed by its net position (§5) has no lever on the price.  The strongest
    form of that claim is that the submitted prices are not an argument at all,
    which is why this module takes only the two quantity vectors.
    """
    import inspect
    from powermarketjax.envs.p2p import baseline_pricing
    fn, _ = make_baseline_pricing(PI_EXP, PI_RET)
    assert list(inspect.signature(fn).parameters) == ["q_sell", "q_buy"]
    src = inspect.getsource(baseline_pricing)
    tree = __import__("ast").parse(src)
    names = {n.id for n in __import__("ast").walk(tree)
             if isinstance(n, __import__("ast").Name)}
    assert "price" not in names


def test_hand_worked_values():
    """§11 verbatim, at the four points where the branches meet or turn."""
    fn, _ = make_baseline_pricing(PI_EXP, PI_RET)
    f = jax.jit(fn)
    one, zero = jnp.ones((4,), jnp.float32), jnp.zeros((4,), jnp.float32)

    # SDR = 0: no supply at all, so both sides price at the retail tariff
    out = f(zero, one)
    assert float(out["sdr_price_sell"]) == pytest.approx(PI_RET, abs=1e-4)
    assert float(out["sdr_price_buy"]) == pytest.approx(PI_RET, abs=1e-4)

    # SDR = 1: the two branches agree, and both give the export price
    out = f(one, one)
    assert float(out["sdr_price_sell"]) == pytest.approx(PI_EXP, abs=1e-4)
    assert float(out["sdr_price_buy"]) == pytest.approx(PI_EXP, abs=1e-4)
    assert float(out["mmr_price_sell"]) == pytest.approx(MID, abs=1e-4)
    assert float(out["mmr_price_buy"]) == pytest.approx(MID, abs=1e-4)

    # SDR = 1/3, worked by hand from (SDR):
    #   sell = 4.1 * 26.11 / ((26.11 - 4.1)/3 + 4.1) = 107.051 / 11.4367
    #   buy  = sell/3 + 26.11 * 2/3
    out = f(jnp.asarray([1., 0., 0., 0.], jnp.float32),
            jnp.asarray([2., 1., 0., 0.], jnp.float32))
    sell = PI_EXP * PI_RET / ((PI_RET - PI_EXP) / 3.0 + PI_EXP)
    assert float(out["sdr_price_sell"]) == pytest.approx(sell, rel=1e-5)
    assert float(out["sdr_price_buy"]) == pytest.approx(
        sell / 3.0 + PI_RET * 2.0 / 3.0, rel=1e-5)
    # (MMR) with demand 3 against supply 1: sellers at mid, buyers absorb the
    # shortfall of 2 bought at the retail tariff
    assert float(out["mmr_price_sell"]) == pytest.approx(MID, abs=1e-4)
    assert float(out["mmr_price_buy"]) == pytest.approx(
        (MID * 1.0 + 2.0 * PI_RET) / 3.0, rel=1e-5)

    # empty market: §11 declares the mid rate on both sides for both rules
    out = f(zero, zero)
    for key in ("sdr_price_sell", "sdr_price_buy",
                "mmr_price_sell", "mmr_price_buy"):
        assert float(out[key]) == pytest.approx(MID, abs=1e-4), key


def test_prices_stay_in_the_bracket_and_buy_is_never_below_sell(price_baselines):
    rng = np.random.default_rng(1)
    for _ in range(500):
        out = price_baselines(*_population(rng))
        for key in ("sdr_price_sell", "sdr_price_buy",
                    "mmr_price_sell", "mmr_price_buy"):
            v = float(out[key])
            assert PI_EXP - 1e-4 <= v <= PI_RET + 1e-4, (key, v)
        assert float(out["sdr_price_buy"]) >= float(out["sdr_price_sell"]) - 1e-4
        assert float(out["mmr_price_buy"]) >= float(out["mmr_price_sell"]) - 1e-4


def test_both_rules_are_budget_balanced_against_the_grid(price_baselines):
    """The §8 external identity, in the form each rule takes.

    Every participant transacts all it submitted, so buyers pay
    ``price_buy * demand`` and sellers receive ``price_sell * supply``; the
    difference must be exactly what the grid charges for the shortfall or pays
    for the surplus.  This is the check that catches a transposed branch, which
    the bracket test above would not.
    """
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(500):
        q_sell, q_buy = _population(rng)
        out = price_baselines(q_sell, q_buy)
        supply, demand = float(jnp.sum(q_sell)), float(jnp.sum(q_buy))
        residual = (PI_RET * max(demand - supply, 0.0)
                    - PI_EXP * max(supply - demand, 0.0))
        scale = max(PI_RET * demand + PI_RET * supply, 1e-9)
        for sell_key, buy_key in (("sdr_price_sell", "sdr_price_buy"),
                                  ("mmr_price_sell", "mmr_price_buy")):
            paid = float(out[buy_key]) * demand
            received = float(out[sell_key]) * supply
            worst = max(worst, abs(paid - received - residual) / scale)
    assert worst < 1e-6, worst


def test_the_sdr_price_falls_as_supply_rises(price_baselines):
    """"Price inversely proportional to SDR" is the premise §11 attributes to
    the source; a transposed branch would break the monotonicity and nothing
    else in this file would notice."""
    # four buyers of one unit each, so total demand is four and a per-agent
    # supply of `v` puts the ratio at exactly `v`
    demand = jnp.ones((4,), jnp.float32)
    prices = []
    for v in np.linspace(0.0, 2.0, 41, dtype=np.float32):
        out = price_baselines(jnp.full((4,), v, jnp.float32), demand)
        prices.append(float(out["sdr_price_buy"]))
    assert all(b <= a + 1e-5 for a, b in zip(prices, prices[1:]))
    assert prices[0] == pytest.approx(PI_RET, abs=1e-4)
    assert prices[-1] == pytest.approx(PI_EXP, abs=1e-4)


def test_the_reported_ratio_is_the_true_one(price_baselines):
    out = price_baselines(jnp.full((4,), 2.0, jnp.float32),
                          jnp.ones((4,), jnp.float32))
    assert float(out["sdr"]) == pytest.approx(2.0, rel=1e-6)
    out = price_baselines(jnp.ones((4,), jnp.float32),
                          jnp.zeros((4,), jnp.float32))
    assert float(out["sdr"]) > 1e6            # zero demand, not a clamped 1.0
