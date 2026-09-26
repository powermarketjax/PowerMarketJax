"""Local flexibility settlement: pay-as-bid, and its money-balance identity.

Array in, array out, pure, jittable.  It converts the output of `clearing`
into the reward, and it is the only place in this market where money is
computed.

    revenue_i = delta * price_i * award_i
    cost_i    = delta * [ energy_price * p_ch_i + cycle_cost_i * (p_ch_i + p_dis_i) ]
    profit_i  = revenue_i - cost_i

Four properties of that arithmetic are load bearing.  Payment is the
participant's own submitted price: there is no uniform price in this market
and no dual is consulted, so nothing is reconstructed after the clearing,
and two participants cleared at the same bus are paid differently if they
offered differently.  The reward is computed from `award`, never from the
offered quantity or an expectation.  `voll` appears nowhere: it is a
coefficient of the clearing objective and of no settlement expression, and
the curtailment a period incurs is a `costs`-channel quantity instead.
`cost` carries only what a participant actually pays -- energy bought when
charging and degradation on everything that passes through the battery, no
feasibility penalty and no cost for serving the participant's own load --
and nothing is charged for the energy discharged, since it was paid for in
the period it was stored.

For any solution of the clearing problem the settlement satisfies

    sum_i revenue_i = Z_t - delta * base_mva * voll * sum_n shed_n

with no residual term: no congestion rent, no uplift, nothing collected that
is not paid out.  This is a weak check against pricing errors, since
pay-as-bid has no pricing formula to get wrong, but it is exactly the right
check against paying on the offered quantity rather than the cleared one,
counting a cleared quantity twice, or letting curtailment reach a settlement
expression.

Per unit in, dollars out: `award` and the battery powers are per unit and
the prices are in dollars per megawatt hour, so every money expression
carries the base power.

The exogenous energy price has no source in this repository, so it is an
argument rather than a registered value.  Until it is settled, `revenue` and
the identity above are meaningful and `profit` is not.
"""
from typing import Callable, Dict

import chex
import jax.numpy as jnp

from .sensitivity import VoltageSensitivity


def make_settlement(sens: VoltageSensitivity,
                    period_hours: float = 0.25) -> Callable:
    """Build the settlement for one network.

    Args:
        sens: network constants, read for the base power only.
        period_hours: the period length in hours.

    Returns:
        ``settle(price, award, plan, energy_price, cycle_cost) -> dict``, pure
        and jittable.  ``price`` is in dollars per megawatt hour, ``award`` and
        ``plan`` are per unit, ``energy_price`` is a scalar in dollars per
        megawatt hour and ``cycle_cost`` is per aggregator in the same unit.
        Returns ``revenue``, ``cost``, ``profit`` and ``reward``, each
        ``(n_agent,)`` in dollars, together with the realised battery powers
        ``p_ch`` and ``p_dis`` that the state of charge update needs.
    """
    money = period_hours * sens.base_mva

    def settle(price: chex.Array, award: chex.Array, plan: chex.Array,
               energy_price: chex.Array, cycle_cost: chex.Array
               ) -> Dict[str, chex.Array]:
        """Settle one period pay-as-bid.

        Args:
            price: ``(n_agent,)`` the participant's own submitted offer
                price in \\$/MWh, which is what it is paid; no uniform price
                exists in this market and no dual is consulted.
            award: ``(n_agent,)`` cleared quantity, per unit.  Every money
                term below is built from this and never from ``qty_max``.
            plan: ``(n_agent,)`` planned charging of the baseline, per unit.
            energy_price: scalar exogenous energy price, \\$/MWh.
            cycle_cost: ``(n_agent,)`` degradation cost per MWh of throughput.

        Returns:
            A dict of ``(n_agent,)`` arrays.  ``revenue``, ``cost``, ``profit``
            and ``reward`` are in dollars, ``reward`` being ``profit`` itself;
            ``p_ch`` and ``p_dis`` are the realised charging and discharging
            powers in per unit, which the state of charge update consumes.
            ``cost`` carries only what the participant pays; no feasibility
            penalty is in it, those being `costs`-channel quantities assembled
            in `env`.
        """
        # The net battery injection splits into its two halves; exactly
        # one of them is non-zero, and an award below the planned charging is
        # delivered by charging less rather than by discharging
        p_dis = jnp.maximum(award - plan, 0.0)
        p_ch = jnp.maximum(plan - award, 0.0)

        revenue = money * price * award
        cost = money * (energy_price * p_ch + cycle_cost * (p_ch + p_dis))
        profit = revenue - cost

        return dict(revenue=revenue, cost=cost, profit=profit, reward=profit,
                    p_ch=p_ch, p_dis=p_dis)

    return settle
