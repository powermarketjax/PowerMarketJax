"""numpy reference for the P2P clearing operator.

This answers "is the JAX implementation written correctly", which is a different
question from "is the mechanism right".  The latter is answered by the
hand-worked cases and the two money-balance identities in `test_clearing_l1.py`
and `test_settlement_l1.py`.  This file runs the **same** algorithm, so the two
can be compared elementwise.

To be worth anything it must not be a transcription of the implementation with
`jnp` swapped for `np`; an earlier vendored equivalence test was found
whose reference was the implementation's own output
and whose error was therefore exactly zero.  It takes a deliberately different
route to the same mathematics:

* the two orders come from Python's ``sorted`` with an explicit ``(price,
  index)`` tuple key, which states §6.4's total order directly, where the
  implementation reaches it through ``jnp.lexsort``;
* the cumulative quantities are accumulated in a Python loop, sequentially,
  where the implementation uses a parallel scan;
* the two step functions are evaluated by scanning the sorted order for the one
  position whose interval contains the point, where the implementation builds a
  ``(2n+1, n)`` comparison matrix and reduces it;
* the sentinels are explicit ``if`` branches, where the implementation masks.

What is shared is the algorithm itself and only that: the same sort order, the
same candidate set, the same admissibility test, the same (AWD), the same
sentinel rule.  An elementwise comparison requires that; a reference running a different
algorithm could not be compared elementwise.

It computes in float64 against the implementation's float32, so the comparison
measures the algorithm and the precision at once.  §16 says which parts of the
output that leaves exact: the orders and the clearing price are, the awards are
not, and the criterion is that at most one participant per side disagrees.

Kept in `tests/` because it exists only to be compared against.
"""
import numpy as np


def clear_ref(price, q_sell, q_buy, pi_exp, pi_ret):
    """Clear one period the long way.  All inputs 1-D of equal length."""
    price = np.asarray(price, np.float64)
    q_sell = np.asarray(q_sell, np.float64)
    q_buy = np.asarray(q_buy, np.float64)
    pi_exp, pi_ret = float(pi_exp), float(pi_ret)
    n = len(price)

    # §6.1 and §6.4: the sort key is the pair, so the order does not depend on
    # the stability of `sorted` -- which happens to be stable, and must not be
    # relied upon (§16).
    order_sell = sorted(range(n), key=lambda i: (price[i], i))
    order_buy = sorted(range(n), key=lambda i: (-price[i], i))

    p_s = [price[i] for i in order_sell]
    q_s = [q_sell[i] for i in order_sell]
    p_b = [price[i] for i in order_buy]
    q_b = [q_buy[i] for i in order_buy]

    cum_s, run = [], 0.0
    for v in q_s:
        run += v
        cum_s.append(run)
    cum_b, run = [], 0.0
    for v in q_b:
        run += v
        cum_b.append(run)
    prev_s = [0.0] + cum_s[:-1]
    prev_b = [0.0] + cum_b[:-1]

    def curve(x, prev, cum, prices, sentinel):
        """§6.1: the price where prev[j] < x <= cum[j], the sentinel if none."""
        for j in range(n):
            if prev[j] < x <= cum[j]:
                return prices[j]
        return sentinel

    # §6.2: the largest admissible breakpoint, x = 0 admitted by convention
    total = min(cum_s[-1], cum_b[-1])
    traded_volume = 0.0
    for x in [0.0] + cum_s + cum_b:
        if x > total:
            continue
        if curve(x, prev_b, cum_b, p_b, pi_ret) >= curve(x, prev_s, cum_s, p_s, pi_exp):
            if x > traded_volume:
                traded_volume = x

    # §6.3 (AWD)
    award_sell = np.zeros(n)
    award_buy = np.zeros(n)
    for j in range(n):
        award_sell[order_sell[j]] = min(max(traded_volume - prev_s[j], 0.0), q_s[j])
        award_buy[order_buy[j]] = min(max(traded_volume - prev_b[j], 0.0), q_b[j])

    # §7 (PRC).  A missing lower-bound contributor is pi_exp and a missing
    # upper-bound contributor is pi_ret; written out as four branches here.
    s_star = curve(traded_volume, prev_s, cum_s, p_s, pi_exp)
    d_star = curve(traded_volume, prev_b, cum_b, p_b, pi_ret)
    s_plus = pi_ret
    for j in range(n):
        if cum_s[j] > traded_volume:
            s_plus = p_s[j]
            break
    d_plus = pi_exp
    for j in range(n):
        if cum_b[j] > traded_volume:
            d_plus = p_b[j]
            break

    price_interval_lo = max(s_star, d_plus)
    price_interval_hi = min(d_star, s_plus)
    return dict(order_sell=np.array(order_sell), order_buy=np.array(order_buy),
                award_sell=award_sell, award_buy=award_buy,
                traded_volume=traded_volume, clearing_price=0.5 * (price_interval_lo + price_interval_hi),
                price_interval_lo=price_interval_lo, price_interval_hi=price_interval_hi)


def population(rng, n, pi_exp, pi_ret, tie_grid=None, zero_frac=0.15):
    """A random population, returned as the float32 arrays `clear` consumes.

    ``tie_grid`` snaps a random subset of the prices onto a coarse grid, which
    is how ties at the margin are produced; ``zero_frac`` is the share of
    participants whose net position is exactly zero, which is what exercises
    §6.1's claim that a zero-quantity participant is never read.
    """
    price = rng.uniform(pi_exp, pi_ret, n).astype(np.float32)
    if tie_grid:
        k = rng.integers(0, n + 1)
        if k:
            pick = rng.choice(n, k, replace=False)
            price[pick] = np.round(price[pick] * tie_grid) / tie_grid
        price = np.clip(price, pi_exp, pi_ret).astype(np.float32)
    net = rng.normal(0.0, 1.0, n).astype(np.float32)
    net[rng.random(n) < zero_frac] = 0.0
    q_sell = (np.maximum(net, 0.0) * 0.5).astype(np.float32)
    q_buy = (np.maximum(-net, 0.0) * 0.5).astype(np.float32)
    return price, q_sell, q_buy


def step_ref(action, soc, p_pv, load, battery, kappa, pi_exp, pi_ret, delta):
    """One whole period in numpy, for the step-by-step comparison of §17 (L3).

    `clear_ref` above answers whether the auction is implemented correctly.  This
    answers a question one layer out: whether the *sequence* is -- whether the
    submission handed to the auction at period ``t`` is the one the state at
    ``t`` implies, and whether the state at ``t+1`` is the one this period's
    deliverable power implies.  An off-by-one in the cursor, a state of charge
    that fails to carry, or a reset that does not restore it are all invisible to
    `clear_ref`, because each of those still clears its own inputs correctly.

    It takes the same deliberately different route as the reference above.  The
    envelope of §3.2 is evaluated from the two closed forms in that section with
    an explicit branch per device in a Python loop, where the implementation
    calls the vendored batched `compute_feasible_power_batch`; the advance is
    (SOC) written out, where the implementation calls `update_soc_batch`.  Only
    the mathematics is shared, which is what is required of a reference
    that is to be compared elementwise.

    Everything is float64 against the implementation's float32, so the
    comparison measures the algorithm and the precision at once, as at L2.

    Returns ``(submission, cleared, money, soc_next)``.
    """
    n = len(soc)
    rated = np.asarray(battery.power_max, np.float64)
    capacity = np.asarray(battery.capacity, np.float64)
    soc_min = np.asarray(battery.soc_min, np.float64)
    soc_max = np.asarray(battery.soc_max, np.float64)
    eta_c = np.asarray(battery.eta_charge, np.float64)
    eta_d = np.asarray(battery.eta_discharge, np.float64)

    alpha = np.clip(np.asarray(action, np.float64), -1.0, 1.0)
    p_desired = alpha[:, 0] * rated

    # §3.2, one device at a time and with the two bounds written out
    p_signed = np.empty(n)
    for i in range(n):
        if p_desired[i] >= 0.0:
            head = (soc[i] - soc_min[i]) * capacity[i] * eta_d[i] / delta
            limit = min(max(head, 0.0), rated[i])
            p_signed[i] = min(p_desired[i], limit)
        else:
            head = (soc_max[i] - soc[i]) * capacity[i] / (eta_c[i] * delta)
            limit = min(max(head, 0.0), rated[i])
            p_signed[i] = -min(-p_desired[i], limit)

    p_dis = np.maximum(p_signed, 0.0)
    p_ch = np.maximum(-p_signed, 0.0)
    net = np.asarray(p_pv, np.float64) + p_dis - np.asarray(load, np.float64) - p_ch
    submission = dict(
        price=pi_exp + 0.5 * (1.0 + alpha[:, 1]) * (pi_ret - pi_exp),
        q_sell=delta * np.maximum(net, 0.0),
        q_buy=delta * np.maximum(-net, 0.0),
        net_position=net,
        p_signed=p_signed,
        clip=np.abs(p_desired - p_signed) / np.maximum(rated, 1e-12),
        throughput=delta * (p_ch + p_dis),
    )

    cleared = clear_ref(submission["price"], submission["q_sell"],
                        submission["q_buy"], pi_exp, pi_ret)

    # §8, with the two residuals written out
    q_ex = submission["q_sell"] - cleared["award_sell"]
    q_im = submission["q_buy"] - cleared["award_buy"]
    c_deg = np.asarray(kappa, np.float64) * submission["throughput"]
    revenue = cleared["clearing_price"] * cleared["award_sell"] + pi_exp * q_ex
    cost = cleared["clearing_price"] * cleared["award_buy"] + pi_ret * q_im + c_deg
    money = dict(revenue=revenue, cost=cost, profit=revenue - cost)

    # (SOC) of §3.2, then the defensive clip the implementation also applies
    delta_soc = delta / capacity * (eta_c * p_ch - p_dis / eta_d)
    soc_next = np.clip(soc + delta_soc, soc_min, soc_max)
    return submission, cleared, money, soc_next
