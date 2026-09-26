"""Day-ahead settlement: cleared schedule and prices in, profit per agent out.

Array in, array out, pure.  It converts the output of `clearing` into the
reward, and it is the only place where money is computed.

    revenue_i = sum_t  delta * lmp[bus(i), t] * award[i, t]
    cost_i    = sum_t [ delta * TC_i(award[i, t]) + delta * NL_i * u[i, t] ]
              + sum_t  S_i * v[i, t]
    profit_i  = revenue_i - cost_i

Three properties of that arithmetic are load bearing.

**The reward is computed from `award`.** Not from the offer, not from an
expectation.  The offer never appears below; it has already done its work
inside the clearing.

**`voll` appears nowhere.**  `VOLL * s` is a term of the clearing objective and
of no settlement expression.  A shed bus prices at the value of lost load
through `lmp` -- that is the mechanism's own dual, not a penalty pasted on
afterwards -- and the shed energy itself is a `costs`-channel quantity, not a
`reward`-channel one.

**Cost goes through `physics.power_flow.compute_generation_cost`.**
`unit_cost_a/b/c` are coefficients of the *marginal* cost curve, not MATPOWER
total-cost ones, so the total cost is the integral `(a/3)p^3 + (b/2)p^2 + cp`.
Writing `a p^2 + b p + c` instead yields a finite, plausible-looking number that
is wrong by orders of magnitude.

There is **no make-whole payment**.  A unit committed by the caller can end
the day with negative profit -- no-load and start-up costs are charged whether
or not the uniform price recovers them -- and living with that is part of the
decision problem, not a defect to be patched in settlement.

Agents are an array axis.  `unit_to_agent` maps units onto the `N` agents and
defaults to one agent per unit; everything returned is `(N,)`.
"""
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from powermarketjax.physics.power_flow import compute_generation_cost


def make_settlement(
    case,
    unit_to_agent: Optional[np.ndarray] = None,
    period_hours: float = 1.0,
) -> Callable:
    """Build the settlement function for one case and agent partition.

    Returns ``settle(award, lmp, commitment, commitment_status)``, pure and
    jittable:

        award              (n_units, n_periods)  cleared quantity, from `clearing`
        lmp                (n_periods, n_buses)  prices, from `clearing`
        commitment         (n_units, n_periods)  the same fixed `u` that cleared
        commitment_status  (n_units,)            the day-boundary carry:
                                                 commitment in the last period of
                                                 the previous day, which is what
                                                 decides whether period 1 pays to
                                                 start

    and returning a dict of ``revenue``, ``cost``, ``profit`` and ``reward``,
    each ``(n_agents,)`` in \\$ per market day.  ``reward`` is ``profit``; the two
    names are kept apart so that scaling or normalising the learning signal
    touches only the RL quantity.
    """
    unit_bus = jnp.asarray(np.asarray(case.unit_node_idx, np.int64))
    cost_a = jnp.asarray(np.asarray(case.unit_cost_a, np.float64))
    cost_b = jnp.asarray(np.asarray(case.unit_cost_b, np.float64))
    cost_c = jnp.asarray(np.asarray(case.unit_cost_c, np.float64))
    no_load = jnp.asarray(np.asarray(case.unit_no_load_cost, np.float64))
    startup = jnp.asarray(np.asarray(case.unit_startup_cost, np.float64))

    n_units = len(np.asarray(case.unit_p_min))
    if unit_to_agent is None:
        unit_to_agent = np.arange(n_units)          # default: one agent per unit
    unit_to_agent = np.asarray(unit_to_agent, np.int64)
    n_agents = int(unit_to_agent.max()) + 1
    agent_of = jnp.asarray(unit_to_agent)

    def settle(award, lmp, commitment, commitment_status):
        """Settle one market day: cleared schedule and prices in, money out.

        The four arguments are the ones `make_settlement` documents above, and all
        four have to come from the **same** clearing: the price is read at each
        unit's own bus, so a schedule paired with another day's prices settles a
        market that never cleared.

        Returns a dict of ``revenue``, ``cost``, ``profit``, ``reward`` and the
        three components ``energy_cost``, ``no_load_cost`` and ``startup_cost``,
        each ``(n_agents,)`` in \\$ per market day.  The three components sum to
        ``cost``, and ``reward`` is ``profit``.
        """
        price = lmp.T[unit_bus]                              # (n_units, T)
        revenue = period_hours * jnp.sum(price * award, axis=1)

        # vmapped over units, so each call integrates one unit's marginal cost
        # over its own T outputs and the function's internal sum is the sum
        # over periods
        energy = period_hours * jax.vmap(compute_generation_cost)(
            award, cost_a, cost_b, cost_c)
        idle = period_hours * no_load * jnp.sum(commitment, axis=1)
        # v[i, t] = max(u[i, t] - u[i, t-1], 0), with u[i, 0] reaching back into
        # the previous day; a unit already on at the boundary does not pay to
        # start again
        prev = jnp.concatenate([commitment_status[:, None], commitment[:, :-1]], 1)
        start = startup * jnp.sum(jnp.maximum(commitment - prev, 0.0), axis=1)

        cost = energy + idle + start
        profit = revenue - cost
        per_agent = lambda x: jax.ops.segment_sum(x, agent_of, num_segments=n_agents)
        revenue, cost, profit = per_agent(revenue), per_agent(cost), per_agent(profit)
        # the three components are returned alongside the total they sum to, so
        # a caller asking "how much of this is no-load?" gets the quantity this
        # function actually charged rather than a second evaluation elsewhere
        return dict(revenue=revenue, cost=cost, profit=profit, reward=profit,
                    energy_cost=per_agent(energy), no_load_cost=per_agent(idle),
                    startup_cost=per_agent(start))

    return settle
