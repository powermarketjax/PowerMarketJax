"""Reference pricing rules: supply-demand ratio and mid-market rate.

Array in, array out, pure.  These are **baselines, not the mechanism** --
nothing in `clearing.py`, `settlement.py` or `action.py` imports this module.
It exists so that a run can report what the same submissions would have been
priced at under the two rules the literature uses most, at the cost of two
sums and a few `jnp.where`.

Both rules price from the **aggregates alone**.  Neither takes the submitted
prices as an argument: with the quantity fixed by the net position, an agent
has no lever on the price at all under either rule, the game between agents
collapses, and what is left is a set of independent storage problems.  The
signature is the claim.

Neither rule produces a clearing price in the double-auction sense.  A
uniform-price auction pays one price to both sides; these pay the sides
differently and recover the difference from the grid, so each returns a pair
and the names carry the side.  There is no award either: every participant
transacts all it submitted and the community residual is pooled, so a
comparison against the auction is a comparison of prices and of money, never
of quantities.

Two conventions belong to these rules rather than to the sources, which do
not state them: a zero total demand is treated as the oversupply branch, a
zero total supply against positive demand as the ratio being zero, and a
market where nothing at all is submitted returns the mid rate on both sides.
"""
from typing import Callable, Dict, Tuple

import chex
import jax.numpy as jnp

#: Floor on the denominators.  Both rules divide by an aggregate that is
#: exactly zero in the empty and one-sided markets; the value never changes
#: the result since every quotient it guards is selected away by
#: `jnp.where`, and it exists only so that branch cannot produce a NaN.
AGG_EPS = 1e-12


def make_baseline_pricing(pi_exp: float, pi_ret: float) -> Tuple[Callable, Dict]:
    """Build the two reference pricing rules for one tariff pair.

    Returns ``(price_baselines, spec)`` where ``price_baselines(q_sell, q_buy)``
    is pure and jittable and takes the same two submitted-quantity vectors
    the auction consumes:

        q_sell  (n_agents,)  energy offered for sale, MWh
        q_buy   (n_agents,)  energy bid for, MWh

    and returns a dict of scalars: ``sdr`` and ``mid_rate``, plus
    ``sdr_price_sell`` / ``sdr_price_buy`` and ``mmr_price_sell`` /
    ``mmr_price_buy``.  Every price lies in ``[pi_exp, pi_ret]`` and the buy
    price of each rule is never below its sell price.
    """
    if not 0.0 <= pi_exp < pi_ret:
        raise ValueError(
            f"the export price must satisfy 0 <= pi_exp < pi_ret, got {pi_exp=} {pi_ret=}")

    exp = jnp.float32(pi_exp)
    ret = jnp.float32(pi_ret)
    mid = jnp.float32(0.5 * (pi_exp + pi_ret))
    span = jnp.float32(pi_ret - pi_exp)

    def price_baselines(q_sell: chex.Array,
                        q_buy: chex.Array) -> Dict[str, chex.Array]:
        """Price the same submitted quantities under (SDR) and (MMR).

        Consumes the two ``(n_agents,)`` submitted-quantity vectors in MWh
        and returns the six scalars the factory docstring names: the two
        diagnostics ``sdr`` and ``mid_rate``, and a sell/buy price for each
        rule.  Nothing per participant comes back, because neither rule
        awards anything -- every submission transacts in full and the
        community residual is traded with the grid.

        The submitted prices are not an argument here and no ``award`` is
        returned, so what a run gets from this is a price comparison and a
        money comparison against the auction, never a quantity comparison.
        The six values are reported through ``info`` and no market quantity
        is computed from them.
        """
        supply = jnp.sum(q_sell)
        demand = jnp.sum(q_buy)
        empty = (supply <= 0.0) & (demand <= 0.0)

        # ---- (SDR).  Ratio undefined at zero demand; that case is declared
        # to be the oversupply branch, which is where a zero denominator
        # would otherwise send it anyway.
        oversupply = (demand <= 0.0) | (supply > demand)
        # reported as the true ratio, floored only to keep it finite: a zero
        # demand shows up as a very large number rather than as a clamped 1.0,
        # because a diagnostic that hides the case it was added to reveal is
        # worse than no diagnostic
        sdr = supply / jnp.maximum(demand, AGG_EPS)
        # the formula below is only ever selected on the scarce branch, where
        # the ratio is at most one; clamping keeps the branch not taken finite
        ratio = jnp.minimum(sdr, 1.0)

        sell_scarce = exp * ret / jnp.maximum(span * ratio + exp, AGG_EPS)
        sdr_sell = jnp.where(oversupply, exp, sell_scarce)
        sdr_buy = jnp.where(oversupply, exp,
                            sell_scarce * ratio + ret * (1.0 - ratio))

        # ---- (MMR).  The short side is priced at the mid rate and the
        # long side absorbs the residual traded with the grid.
        shortage = jnp.maximum(demand - supply, 0.0)
        surplus = jnp.maximum(supply - demand, 0.0)
        buy_short = (mid * supply + shortage * ret) / jnp.maximum(demand, AGG_EPS)
        sell_long = (mid * demand + surplus * exp) / jnp.maximum(supply, AGG_EPS)

        under = demand > supply
        over = supply > demand
        mmr_sell = jnp.where(over, sell_long, mid)
        mmr_buy = jnp.where(under, buy_short, mid)

        # the empty market has no aggregate to price from; the mid rate is
        # used on both sides rather than letting a guarded quotient decide
        pick = lambda x: jnp.where(empty, mid, x)
        return dict(sdr=jnp.where(empty, jnp.float32(0.0), sdr), mid_rate=mid,
                    sdr_price_sell=pick(sdr_sell), sdr_price_buy=pick(sdr_buy),
                    mmr_price_sell=pick(mmr_sell), mmr_price_buy=pick(mmr_buy))

    spec = dict(pi_exp=float(pi_exp), pi_ret=float(pi_ret),
                mid_rate=float(0.5 * (pi_exp + pi_ret)), dtype=jnp.float32)
    return price_baselines, spec
