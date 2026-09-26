"""P2P clearing operator: submissions in, awards and one clearing price out.

Array in, array out, pure.  This is a double auction: sellers submit offers and
buyers submit bids, the two aggregate curves are built as sorted step
functions, and the market clears where they cross.  Two sorted orders, two
cumulative sums, the crossing point, the rationing at the margin, and the
midpoint of the price interval the crossing leaves.

There is no solver, no callback, no Python loop and no data-dependent shape.
Every choice is a comparison whose result multiplies a value -- the
``jnp.where`` form, which unlike ``lax.cond`` costs nothing under ``vmap``.

Ties in price are broken by ascending participant index, stated as a total
``jnp.lexsort`` key rather than left to the stability of the sorting routine.

The crossing point is found by dense comparison against all ``2 n_agents +
1`` breakpoints rather than by binary search: the dense form is one wide
reduction against a chain of ``log n`` dependent kernels, and it is the
faster of the two at the participant counts this market is sized for.

The curves are evaluated on ``Q[j-1] < x <= Q[j]``.  The set of ``j``
satisfying that condition is empty exactly where the curve is undefined --
at ``x = 0`` on the accepted side, and beyond the last breakpoint on the
rejected side -- so each such case takes a sentinel rather than a patch.  The
four marginal values that bound the price obey one sentinel rule: a missing
lower-bound contributor takes ``pi_exp`` and a missing upper-bound
contributor takes ``pi_ret``, the neutral element of the reduction it feeds.

``clearing_price`` is a midpoint of two exact prices, so adding participants
who submit nothing leaves it unchanged.  The awards are not exact, because
``jnp.cumsum`` is a parallel scan whose association tree depends on the
length of the array, so the same addition can move ``traded_volume`` in the
last bits; at most one participant per side can disagree, since every other
award is the whole submission or nothing.

**Two diagnostic pricing rules sit beside the market's own, behind
`pricing_rule`, and neither is the mechanism.**  This package forbids
reconstructing a price after the clearing by a heuristic, and that rule is not
loosened here: the default is untouched and bit-identical, the alternatives
change nothing about the matching, the awards or ``traded_volume``, and each
is a *stated pricing rule of the same double auction* rather than a fit to the
outcome.  They exist for one measurement -- whether the reversal of the
single-deviator premium above 256 households is caused by the price collapsing
onto an end of the tariff bracket -- and a run under either of them is not a run of this
market.  What each does, and what it costs:

``award-consistent-midpoint``
    The market's own rule and the default.  ``[lo, hi]`` is exactly the set of
    prices consistent with the awards, so when one participant clears partly
    its own quote is forced on both ends and the interval is a point.  Under
    truthful submissions every seller is at ``pi_exp`` and every buyer at
    ``pi_ret``, so that point is an end of the bracket: measured at 95.65% of
    periods at 1200 households and 99.97% at 16.

``marginal-accepted-midpoint``
    The half-way point between the marginal accepted offer and the marginal
    accepted bid, which is the k = 1/2 double auction.  It gives the midpoint
    of the bracket -- 203.2 EUR/MWh on the adopted tariff -- in *every* period
    a truthful population clears, instead of in 4.35% of them.  It is not
    award-consistent: the participant clearing partly is priced away from its
    own quote.  A deviator that is the marginal accepted quote on its side
    still moves the price, by half of what it moves its own quote, so the
    premium this diagnostic measures is not zero by construction.

``tariff-mid-rate``
    The midpoint of the bracket itself, held constant.  This is the mid-market
    rate `baseline_pricing` already computes, used here as the price rather
    than reported beside it.  **It is vacuous as a deviation diagnostic and is
    provided as the control that shows it**: the price does not read any
    submission, so with the quantity fixed by the net position no submission
    can move it, exactly as `baseline_pricing`'s own docstring states, and the
    only thing a deviation can still do is price itself out of an award it
    wanted.  The premium under it is therefore zero by construction, which is a
    fact about the rule and not about the population size.
"""
from typing import Callable, Dict, Tuple

import chex
import jax
import jax.numpy as jnp

#: The three pricing rules `make_clearing` accepts.  Named rather than left to
#: a string comparison at the call site so that a typo is a construction-time
#: failure and so that a product can record which one was in force.
PRICING_RULES = ("award-consistent-midpoint", "marginal-accepted-midpoint",
                 "tariff-mid-rate")


def make_clearing(
    n_agents: int,
    pi_exp: float,
    pi_ret: float,
    *,
    pricing_rule: str = "award-consistent-midpoint",
) -> Tuple[Callable, Dict]:
    """Build the clearing operator for one population and tariff pair.

    Returns ``(clear, spec)`` where ``clear(price, q_sell, q_buy)`` is pure and
    jittable:

        price   (n_agents,)  submitted price, \\$/MWh, inside [pi_exp, pi_ret]
                             by construction of the action map
        q_sell  (n_agents,)  energy offered for sale, MWh; zero for a buyer
        q_buy   (n_agents,)  energy bid for, MWh; zero for a seller

    and returning a dict of ``award_sell`` and ``award_buy``, both
    ``(n_agents,)``; ``traded_volume``, ``clearing_price``,
    ``price_interval_lo`` and ``price_interval_hi``, all scalars.  The last two
    are the endpoints of the price interval the crossing leaves, and
    ``clearing_price`` is their midpoint.

    ``pricing_rule`` selects which of `PRICING_RULES` produces
    ``clearing_price``; the default is the market's own and the other two are
    diagnostics the module docstring describes.  It is keyword-only and it is
    echoed in ``spec["pricing_rule"]``, so a caller reads back what took effect
    rather than what it meant to pass.  ``price_interval_lo`` and
    ``price_interval_hi`` are the award-consistent interval under every rule,
    so a diagnostic run still reports the interval its price left.
    """
    if not 0.0 <= pi_exp < pi_ret:
        raise ValueError(
            f"the export price must satisfy 0 <= pi_exp < pi_ret, got {pi_exp=} {pi_ret=}")
    if n_agents < 1:
        raise ValueError(f"n_agents must be positive, got {n_agents}")
    if pricing_rule not in PRICING_RULES:
        raise ValueError(
            f"pricing_rule={pricing_rule!r} is not one of {list(PRICING_RULES)}; "
            f"the price this market reports has to come from a rule that was "
            f"named, and an unrecognised name would otherwise select the "
            f"default silently")

    exp = jnp.float32(pi_exp)
    ret = jnp.float32(pi_ret)
    mid = jnp.float32(0.5 * (pi_exp + pi_ret))
    idx = jnp.arange(n_agents)
    zero = jnp.zeros((1,), jnp.float32)

    def _curve(x, q_prev, q_cum, price_sorted, sentinel):
        """One aggregate curve, domain included.

        Returns the submitted price attached to ``x`` on one side, for every
        ``x`` at once, or ``sentinel`` where the curve is undefined.  ``x`` is
        ``(m,)`` and the rest ``(n_agents,)``; the mask is ``(m, n_agents)``.  A
        participant carrying zero quantity leaves ``q_cum[j] == q_prev[j]``, so
        no ``x`` can satisfy the strict lower and weak upper bound at once and
        that column is never selected, without a filtering step.
        """
        m = (q_prev[None, :] < x[:, None]) & (x[:, None] <= q_cum[None, :])
        return jnp.where(m.any(axis=1), price_sorted[jnp.argmax(m, axis=1)],
                         sentinel)

    def _pick(mask, price_sorted, sentinel):
        """First true position of a 1-D mask, or the sentinel if none is."""
        return jnp.where(mask.any(), price_sorted[jnp.argmax(mask)], sentinel)

    def clear(price: chex.Array, q_sell: chex.Array,
              q_buy: chex.Array) -> Dict[str, chex.Array]:
        """Clear one period of the double auction.

        Consumes the three ``(n_agents,)`` vectors the factory docstring lists
        and returns the six entries it names: ``award_sell`` and ``award_buy``
        per participant, and the scalars ``traded_volume``, ``clearing_price``,
        ``price_interval_lo`` and ``price_interval_hi``.

        Both sides quote, which is what makes this a double auction rather than
        a one-sided one: ``price`` carries a seller's offer and a buyer's bid in
        the same vector, told apart by which of ``q_sell`` and ``q_buy`` the
        participant submitted, and both are sorted into merit order below.
        ``award_sell`` and ``award_buy`` are therefore two cleared quantities
        and not one, and each summed on its own equals ``traded_volume``, the
        energy that changed hands **inside** the market.  What a participant
        submitted and did not have awarded is not lost -- `settlement` takes it
        to the grid at the outside option -- and none of it counts here.
        """
        # Ascending for offers and descending for bids, ties by index.  lexsort
        # takes the last key as primary; negating the price is exact in IEEE754,
        # and -0.0 still compares equal to 0.0 so a submission at pi_exp = 0
        # still falls through to the index.
        order_sell = jnp.lexsort((idx, price))
        order_buy = jnp.lexsort((idx, -price))

        p_s, qs = price[order_sell], q_sell[order_sell]
        p_b, qb = price[order_buy], q_buy[order_buy]

        # A running maximum on top of the scan: float32 `jnp.cumsum` on CPU is a
        # parallel scan and is not monotone in the last bits -- a run of
        # zero-quantity participants after the last positive one can end 1-2 ULP
        # *below* it.  Then `traded_volume = cum[-1]` sits under the last
        # positive breakpoint, `s_plus`/`d_plus` pick a marginal that is not
        # there, and the price interval comes out inverted ([pi_ret, pi_exp],
        # midpoint 203.2 on the adopted tariff): measured 2026-09-26 in 264 of
        # 6 144 truthful periods at 1 200 households against `step_ref`.  Where
        # the scan is already monotone this is the identity, bit for bit.
        cum_s = jax.lax.cummax(jnp.cumsum(qs))
        cum_b = jax.lax.cummax(jnp.cumsum(qb))
        prev_s = jnp.concatenate([zero, cum_s[:-1]])       # Q_{j-1}, Q_0 = 0
        prev_b = jnp.concatenate([zero, cum_b[:-1]])

        # The breakpoints of the two step functions, together with zero.
        # Fixed length 2 n_agents + 1, independent of the submissions.
        cands = jnp.concatenate([zero, cum_s, cum_b])

        supply = _curve(cands, prev_s, cum_s, p_s, exp)
        demand = _curve(cands, prev_b, cum_b, p_b, ret)
        admissible = (cands <= jnp.minimum(cum_s[-1], cum_b[-1])) \
            & (demand >= supply)
        # x = 0 is admissible by construction -- both curves are undefined
        # there, so they take their sentinels and pi_ret >= pi_exp holds -- so
        # the maximum always exists and no empty-set case arises.
        traded_volume = jnp.max(jnp.where(admissible, cands, 0.0))

        # (AWD): fill each order from its best submission onwards.  A
        # participant entirely below traded_volume gets all it submitted, one
        # entirely above gets nothing, and exactly one per side straddles.  A
        # zero-quantity participant gets zero from the second argument of the
        # minimum, again without a special case.
        aw_s = jnp.minimum(jnp.maximum(traded_volume - prev_s, 0.0), qs)
        aw_b = jnp.minimum(jnp.maximum(traded_volume - prev_b, 0.0), qb)
        award_sell = jnp.zeros(n_agents, jnp.float32).at[order_sell].set(aw_s)
        award_buy = jnp.zeros(n_agents, jnp.float32).at[order_buy].set(aw_b)

        # (PRC).  Accepted marginals use the same interval condition as the
        # curves; rejected marginals are the first breakpoint strictly above
        # traded_volume.  Sentinels follow the one rule: a missing lower-bound
        # contributor is pi_exp, a missing upper-bound contributor is pi_ret.
        s_star = _pick((prev_s < traded_volume) & (traded_volume <= cum_s), p_s, exp)
        d_star = _pick((prev_b < traded_volume) & (traded_volume <= cum_b), p_b, ret)
        s_plus = _pick(cum_s > traded_volume, p_s, ret)
        d_plus = _pick(cum_b > traded_volume, p_b, exp)

        # The crossing of the two aggregate curves need not be a point: any
        # price in [lo, hi] is consistent with the same awards, the same
        # non-uniqueness an LP optimal dual has.  The market rule takes the
        # midpoint, so `clearing_price` is a point *chosen inside* the
        # interval the mechanism produced, not a price fitted to the outcome
        # afterwards.  The width is a diagnostic and both endpoints go to
        # `info` for it.
        price_interval_lo = jnp.maximum(s_star, d_plus)
        price_interval_hi = jnp.minimum(d_star, s_plus)

        # Selected in Python at construction, so the default path is the same
        # expression it has always been rather than a `jnp.where` over three
        # branches that happens to fold to it.
        if pricing_rule == "award-consistent-midpoint":
            clearing_price = 0.5 * (price_interval_lo + price_interval_hi)
        elif pricing_rule == "marginal-accepted-midpoint":
            clearing_price = 0.5 * (s_star + d_star)
        else:
            clearing_price = jnp.broadcast_to(mid, jnp.shape(traded_volume))

        return dict(award_sell=award_sell, award_buy=award_buy,
                    traded_volume=traded_volume,
                    clearing_price=clearing_price,
                    price_interval_lo=price_interval_lo, price_interval_hi=price_interval_hi)

    spec = dict(n_agents=n_agents, n_candidates=2 * n_agents + 1,
                pi_exp=float(pi_exp), pi_ret=float(pi_ret), dtype=jnp.float32,
                pricing_rule=pricing_rule,
                is_market_pricing_rule=(
                    pricing_rule == "award-consistent-midpoint"))
    return clear, spec
