"""P2P settlement: awards and the clearing price in, profit per agent out.

Array in, array out, pure.  It converts the output of `clearing` into the
reward, and it is the only place where money is computed.

    q_ex_i    = q_sell_i - award_sell_i          energy exported to the grid
    q_im_i    = q_buy_i  - award_buy_i           energy imported from it
    revenue_i = lambda * award_sell_i + pi_exp * q_ex_i
    cost_i    = lambda * award_buy_i  + pi_ret * q_im_i + kappa_i * thr_i
    profit_i  = revenue_i - cost_i

Three properties of that arithmetic are load bearing.

**The reward is computed from `award`.** Not from the submission, not from an
expectation.  The submitted price never appears below; it has already done
its work inside the clearing, and what reaches here is the price the
mechanism produced.

**`clip` appears nowhere.**  It is a `costs`-channel quantity that nobody
pays, so it has no place in any expression here.  Battery degradation goes
the other way: a participant really does pay it, so it enters `cost` and
through it the reward.  The action map hands the two over separately, and
this module only ever sees `throughput`.

**`kappa` is a runtime argument with no default.**  A default of zero here
would let a caller study a market without degradation while believing
otherwise.  It is a runtime argument rather than a closure constant because
it may be swept across runs, and closing it over would recompile on every
value.

Profit is negative for a participant that consumes more than it produces over
the period: such a participant has a bill and not an income, and maximising
profit for it means minimising that bill.  There is no make-whole payment.

The two money-balance identities this settlement must satisfy are **not**
returned; a settlement that reported its own residual would be checking
itself.  Any external check of them should compare relative to the gross
money flow, never relative to the aggregate profit, which is a difference of
two nearly equal sums and passes through zero.
"""
from typing import Callable

import chex
import jax.numpy as jnp


def make_settlement(pi_exp: float, pi_ret: float) -> Callable:
    """Build the settlement function for one tariff pair.

    Returns ``settle(q_sell, q_buy, award_sell, award_buy, clearing_price,
    kappa, throughput)``, pure and jittable:

        q_sell, q_buy            (n_agents,)  submitted quantities, MWh
        award_sell, award_buy    (n_agents,)  cleared quantities, MWh, from
                                              `clearing`
        clearing_price           scalar       \\$/MWh, from `clearing`
        kappa                    (n_agents,)  degradation price, \\$/MWh of
                                              throughput; no default
        throughput               (n_agents,)  energy through the battery, MWh,
                                              from the action map

    and returning a dict of ``revenue``, ``cost``, ``profit``, ``reward`` and
    ``degradation_cost``,
    each ``(n_agents,)`` in \\$ per period.  ``reward`` is ``profit``; the two
    names are kept apart so that scaling or normalising the learning signal
    touches only the RL quantity.

    ``period_hours`` does not appear anywhere in this module.  Every quantity
    reaching it is already an energy in MWh, the action map having done all
    three conversions, so there is no unit of time here to get wrong and no
    second place where ``delta`` could disagree with itself.
    """
    if not 0.0 <= pi_exp < pi_ret:
        raise ValueError(
            f"the export price must satisfy 0 <= pi_exp < pi_ret, got {pi_exp=} {pi_ret=}")

    exp = jnp.float32(pi_exp)
    ret = jnp.float32(pi_ret)

    def settle(q_sell: chex.Array, q_buy: chex.Array, award_sell: chex.Array,
               award_buy: chex.Array, clearing_price: chex.Array,
               kappa: chex.Array, throughput: chex.Array):
        """Settle one period: ``revenue``, ``cost``, ``profit``, ``reward``
        and ``degradation_cost``.

        Consumes the seven arguments the factory docstring lists and returns
        those five ``(n_agents,)`` vectors, ``reward`` being ``profit``.

        ``degradation_cost`` is ``kappa * throughput``, already a term of
        ``cost`` above.  It is returned separately because a caller that wanted
        it had to multiply ``kappa`` by a throughput obtained elsewhere, which
        is a second implementation of a product this function has already
        formed, and the two could then disagree about which ``kappa`` was in
        force.

        **Whether it is the whole of this market's system cost is a judgement
        about where the system boundary sits, not a measurement.**  Taking the
        modelled participants as the boundary it is: there is no generation and
        no load shedding here, and the auction leg is a transfer between two
        participants that nets to zero across them.  The two residual legs are
        transfers with the grid, which is outside that boundary and is not part
        of the double auction this market clears, so under
        the baseline matrix's definition -- production cost,
        plus shedding at the value of lost load, plus the remaining priced
        terms **of the market's own clearing objective** -- they do not enter.
        A reader who draws the boundary to include the grid would have to add
        them and would get a different quantity.

        The two residual lines below are the outside option: the pair (retail
        tariff, export price) that prices what a participant does when it
        does not trade inside the market.  A surplus the auction did not
        award is exported at ``pi_exp`` and a deficit it did not award is
        imported at ``pi_ret``; the grid absorbs and supplies any quantity at
        those two, so nothing is ever left unserved and no residual needs a
        price of its own.  Neither member of the pair is a settlement price
        -- they price the trade that did not happen -- and neither residual
        is part of ``traded_volume``, which counts only what cleared.  The
        pair must come from one jurisdiction and one period: that is a
        property of the two floats this factory closed over, and there is
        nothing here that could check it.
        """
        # Both residuals are non-negative because (AWD) never awards more than
        # was submitted, and at most one of the two is positive because a
        # participant submits on one side only.
        q_ex = q_sell - award_sell
        q_im = q_buy - award_buy

        c_deg = kappa * throughput
        revenue = clearing_price * award_sell + exp * q_ex
        cost = clearing_price * award_buy + ret * q_im + c_deg
        profit = revenue - cost
        return dict(revenue=revenue, cost=cost, profit=profit, reward=profit,
                    degradation_cost=c_deg)

    return settle


def make_terminal_settlement(pi_exp: float) -> Callable:
    """Returns ``settle_terminal(soc, battery)``, the leg applied at truncation.

    Separate from `make_settlement` because it is the only money term that is not
    a function of an award: it prices a stock rather than a flow, it fires on one
    step of the episode rather than on every step, and it needs the battery
    rather than the auction.  Keeping it apart also leaves the signature of
    `settle` unchanged for the callers that only clear and settle a period.
    """
    exp = jnp.float32(pi_exp)

    def settle_terminal(soc: chex.Array, battery) -> chex.Array:
        """Value of the stock left in the battery when the episode is cut.

        Energy above the lower bound of the state-of-charge range is
        delivered at the discharge efficiency and valued at the export
        price, which the grid makes available to a household in every
        period, so this is the price at which the stock can always be
        disposed of and therefore a lower bound on what carrying it forward
        is worth.

        Without this leg the stock is valued at zero, which is below any
        price a household faces, and a fixed-length episode then pays for
        ending empty.

        No degradation is charged: degradation prices throughput, and no
        energy moves through the converter here, so charging it would price
        a cycle that did not happen.
        """
        deliverable = ((soc - battery.soc_min) * battery.capacity
                       * battery.eta_discharge)
        return exp * deliverable

    return settle_terminal
