"""Ancillary settlement: cleared quantities and prices in, profit per agent out.

Array in, array out, pure.  Two legs settle here and they are structurally
different, which is the whole content of this module.

    revenue_energy_i = delta * [ lmp_da[bus(i)] * q_da_i
                               + lmp[bus(i)] * (award_i - q_da_i) ]
    revenue_res_i    = delta * sum_j lambda_res[j] * reserve[i, j]
    cost_i           = delta * TC_i(award_i) + delta * NL_i * u_i + S_i * v_i
    profit_i         = revenue_energy_i + revenue_res_i - cost_i

**The energy leg is a two-settlement rule**, the same arithmetic the real-time
balancing market uses: the day-ahead schedule is paid at the day-ahead price
and only the deviation from it is settled at the price of this clearing.

**Reserve is paid but not charged, and the market therefore does not balance.**
Energy payments net against energy charges because load pays inside the same
clearing.  The requirement is an obligation the operator imposes, not a bid a
consumer submits, so the capacity payment is an expenditure recovered outside
the market.  Each leg balances separately and nothing here should be expected
to make the whole balance; a check that asserted it would be asserting
something the mechanism denies.

**Holding capacity burns no fuel.**  Reserve appears in no cost term.  The cost
of the period is the energy cost of the realised output plus the no-load cost of
a committed unit plus a start-up cost when the exogenous commitment switches the
unit on, and nothing else.

**Neither `voll` nor `volr` appears.**  Both are cost parameters of the clearing
objective and of no settlement expression.  A shed bus prices at the value of
lost load through `lmp`, which is the mechanism's own dual rather than a
penalty pasted on afterwards, and an unmet requirement prices at `volr` through
`reserve_price` for the same reason.

**The reward is computed from cleared quantities.**  Neither offer appears
below; both have already done their work inside the clearing.

Agents are an array axis.  `unit_to_agent` maps units onto the `N` agents and
defaults to one agent per unit; everything returned is `(N,)`.
"""
from typing import Callable, Optional

import chex
import jax
import jax.numpy as jnp
import numpy as np

from powermarketjax.physics.power_flow import compute_generation_cost


def make_settlement(
    case,
    unit_to_agent: Optional[np.ndarray] = None,
    period_hours: float = 0.5,
) -> Callable:
    """Build the settlement function for one case and agent partition.

    Returns ``settle(award, reserve, lmp, reserve_price, q_da, lmp_da, u,
    u_prev)``, pure and jittable:

        award          (n_units,)            cleared output, from `clearing`
        reserve        (n_units, n_prod)     cleared reserve, from `clearing`
        lmp            (n_buses,)            real-time price, from `clearing`
        reserve_price  (n_prod,)             product prices, from `clearing`
        q_da           (n_units,)            day-ahead schedule of this period
        lmp_da         (n_buses,)            day-ahead price of this period
        u              (n_units,)            the fixed commitment that cleared
        u_prev         (n_units,)            commitment of the previous period,
                                             which is what decides who pays to
                                             start

    and returns a dict of ``revenue_energy``, ``revenue_reserve``, ``revenue``,
    ``cost``, ``profit`` and ``reward``, each ``(n_agents,)`` in \\$ for the
    period.  ``reward`` is ``profit``; the two names are kept apart so that
    scaling the learning signal touches only the RL quantity.
    """
    unit_bus = jnp.asarray(np.asarray(case.unit_node_idx, np.int64))
    cost_a = jnp.asarray(np.asarray(case.unit_cost_a, np.float64))
    cost_b = jnp.asarray(np.asarray(case.unit_cost_b, np.float64))
    cost_c = jnp.asarray(np.asarray(case.unit_cost_c, np.float64))
    no_load = jnp.asarray(np.asarray(case.unit_no_load_cost, np.float64))
    startup = jnp.asarray(np.asarray(case.unit_startup_cost, np.float64))

    n_units = len(np.asarray(case.unit_p_min))
    if unit_to_agent is None:
        unit_to_agent = np.arange(n_units)
    unit_to_agent = np.asarray(unit_to_agent, np.int64)
    n_agents = int(unit_to_agent.max()) + 1
    agent_of = jnp.asarray(unit_to_agent)

    def settle(award: chex.Array, reserve: chex.Array, lmp: chex.Array,
               reserve_price: chex.Array, q_da: chex.Array, lmp_da: chex.Array,
               u: chex.Array, u_prev: chex.Array) -> dict:
        """Settle one period and return the money per agent.

        The eight arguments and the returned keys are listed in
        `make_settlement`'s ``Returns``.  ``q_da`` and ``lmp_da`` are the frozen
        day-ahead position of this period and ``u`` and ``u_prev`` the exogenous
        commitment: all four are data, not the output of a day-ahead clearing
        run here.

        The energy, reserve and no-load terms carry ``period_hours``, since the
        prices are \\$/MWh and the no-load cost is \\$/h; the start-up cost is a
        per-start amount and does not.
        """
        price = lmp[unit_bus]
        price_da = lmp_da[unit_bus]
        # the day-ahead schedule settles at the day-ahead price and only the
        # deviation from it at this clearing's price
        revenue_energy = period_hours * (price_da * q_da
                                         + price * (award - q_da))
        revenue_reserve = period_hours * jnp.sum(reserve_price[None, :] * reserve,
                                                 axis=1)

        energy = period_hours * jax.vmap(compute_generation_cost)(
            award[:, None], cost_a, cost_b, cost_c)
        idle = period_hours * no_load * u
        start = startup * jnp.maximum(u - u_prev, 0.0)
        cost = energy + idle + start

        profit = revenue_energy + revenue_reserve - cost
        per_agent = lambda x: jax.ops.segment_sum(x, agent_of,
                                                  num_segments=n_agents)
        out = dict(revenue_energy=per_agent(revenue_energy),
                   revenue_reserve=per_agent(revenue_reserve),
                   cost=per_agent(cost), profit=per_agent(profit),
                   # the three cost components separately, because the claim
                   # this market makes about its negative profit is that the
                   # no-load cost alone exceeds the shortfall, and a total
                   # cannot support that claim.  Reported rather than left to a
                   # caller to rebuild from the case, which would be a second
                   # implementation of the same three lines.
                   cost_energy=per_agent(energy), cost_noload=per_agent(idle),
                   cost_startup=per_agent(start))
        out["revenue"] = out["revenue_energy"] + out["revenue_reserve"]
        out["reward"] = out["profit"]
        return out

    return settle
