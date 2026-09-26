"""Two-settlement rule: position and clearing in, profit per agent out.

Array in, array out, pure.  The day-ahead schedule is a financially binding
contract paid at the day-ahead price whether or not the unit produces it, and
**only the deviation** is settled at the real-time price:

    revenue_da_i = sum_t  delta * lmp_da[bus(i), t] * q_da[i, t]
    revenue_rt_i = sum_t  delta * lmp[bus(i), t] * (p[i, t] - q_da[i, t])
    cost_i       = sum_t [ delta * TC_i(p[i, t]) + delta * NL_i * u[i, t] ]
                 + sum_t  S_i * v[i, t]
    profit_i     = revenue_da_i + revenue_rt_i - cost_i

`revenue_da`, the no-load term and the start-up term do not depend on the action
-- schedule, prices and commitment are all data -- but are kept, so that
`profit` is the delivery period's real economic quantity.

The reward is computed from the **realised award**, never from an offer or an
expectation.  `voll` appears nowhere: `VOLL * s` is a term of the clearing
objective and of no settlement expression.  Cost goes through
`compute_generation_cost` because `unit_cost_a/b/c` are *marginal*-cost
coefficients, and there is no make-whole payment, so a unit committed day-ahead
can end a period in the red.

**`period_hours` multiplies here and only here.**  `envs.day_ahead.clearing`'s
objective carries no factor of `delta`, so its duals are already \\$/MWh and this
module applies `delta` exactly once.  `envs.day_ahead.relax` uses the opposite
convention, and at this market's `delta = 0.5` a price taken from the wrong one
is out by a factor of two.

`money_balance` assembles **both sides** of the money-balance identity for the
real-time leg.  It is separate from `settle` because nothing in the environment
consumes it: it exists to be checked against.
"""
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from powermarketjax.physics.power_flow import compute_generation_cost


def make_settlement(
    case,
    unit_to_agent: Optional[np.ndarray] = None,
    period_hours: float = 0.5,
    cap_scale: float = 1.0,
) -> Tuple[Callable, Callable]:
    """Build `(settle, money_balance)` for one case and agent partition.

    Args:
        case: a `CaseData`.
        unit_to_agent: `(n_units,)` map onto the `N` agents; one agent per unit
            by default.
        period_hours: `delta`, 0.5 h for this market.  Declared rather than
            defaulted at the call sites that matter, since it is the factor that
            silently halves or doubles every settled amount.
        cap_scale: the same scenario parameter the clearing was built with.  The
            identity prices line headroom against `F_l`, so a `cap_scale` that
            disagrees with the clearing's makes the right-hand side describe a
            different network.

    Returns:
        ``settle(award, lmp, commitment, commitment_status, q_da, lmp_da)``
        returning ``revenue_da``, ``revenue_rt``, ``cost``, ``profit`` and
        ``reward``, each ``(n_agents,)`` in \\$; and ``money_balance(...)``,
        which returns both sides of the money-balance identity for the
        real-time leg.
    """
    unit_bus_np = np.asarray(case.unit_node_idx, np.int64)
    unit_bus = jnp.asarray(unit_bus_np)
    cost_a = jnp.asarray(np.asarray(case.unit_cost_a, np.float64))
    cost_b = jnp.asarray(np.asarray(case.unit_cost_b, np.float64))
    cost_c = jnp.asarray(np.asarray(case.unit_cost_c, np.float64))
    no_load = jnp.asarray(np.asarray(case.unit_no_load_cost, np.float64))
    startup = jnp.asarray(np.asarray(case.unit_startup_cost, np.float64))
    PTDF = jnp.asarray(np.asarray(case.PTDF, np.float64))
    F = jnp.asarray(np.asarray(case.line_cap, np.float64) * cap_scale)

    n_units, n_buses = len(unit_bus_np), int(case.n_nodes)
    if unit_to_agent is None:
        unit_to_agent = np.arange(n_units)              # one agent per unit
    unit_to_agent = np.asarray(unit_to_agent, np.int64)
    n_agents = int(unit_to_agent.max()) + 1
    agent_of = jnp.asarray(unit_to_agent)

    # (n_buses, n_units), so that a per-unit quantity sums onto its bus
    bus_of_unit = jnp.asarray(
        (unit_bus_np[None, :] == np.arange(n_buses)[:, None]).astype(np.float64))

    def settle(award, lmp, commitment, commitment_status, q_da, lmp_da):
        """Settle one set of periods and return the money per agent.

        Args:
            award: `(n_units, T)` cleared output in MW, from the clearing.
            lmp: `(T, n_buses)` real-time price in \\$/MWh, the duals of the same
                solve that produced ``award``.
            commitment: `(n_units, T)` fixed commitment of those periods.
            commitment_status: `(n_units,)` commitment of the period *before*
                the first one, which is what decides who pays to start.
            q_da: `(n_units, T)` day-ahead schedule.
            lmp_da: `(T, n_buses)` day-ahead price.  Together with ``q_da`` and
                ``commitment`` this is the frozen day-ahead position: exogenous
                data read from a fixture, never the return value of a
                day-ahead clearing run here.

        Returns:
            A dict of ``revenue_da``, ``revenue_rt``, ``cost``, ``profit`` and
            ``reward``, each ``(n_agents,)`` in \\$ for the whole set of periods.
            ``reward`` is ``profit``; the two names are kept apart so that
            scaling the learning signal touches only the RL quantity.  Nothing
            returned here is a `costs` channel -- the shed energy that feeds
            that vector is formed in `envs.real_time.env`.  The energy and
            no-load terms carry ``period_hours``; the start-up term is a
            per-start amount and does not.
        """
        price = lmp.T[unit_bus]                                  # (n_units, T)
        price_da = lmp_da.T[unit_bus]

        revenue_da = period_hours * jnp.sum(price_da * q_da, axis=1)
        # only the deviation is settled at the real-time price
        revenue_rt = period_hours * jnp.sum(price * (award - q_da), axis=1)

        energy = period_hours * jax.vmap(compute_generation_cost)(
            award, cost_a, cost_b, cost_c)
        idle = period_hours * no_load * jnp.sum(commitment, axis=1)
        # v[i, t] = max(u[i, t] - u[i, t-1], 0); the commitment switches within
        # the day at the boundaries where the day-ahead schedule starts or stops
        # a unit, so start-up is charged here and not only at the day edge
        prev = jnp.concatenate([commitment_status[:, None], commitment[:, :-1]], 1)
        start = startup * jnp.sum(jnp.maximum(commitment - prev, 0.0), axis=1)

        cost = energy + idle + start
        profit = revenue_da + revenue_rt - cost
        seg = lambda x: jax.ops.segment_sum(x, agent_of, num_segments=n_agents)
        return dict(revenue_da=seg(revenue_da), revenue_rt=seg(revenue_rt),
                    cost=seg(cost), profit=seg(profit), reward=seg(profit))

    def money_balance(award, lmp, shed, demand_bus, q_da, s_da, d_da,
                      line_dual_up, line_dual_dn, shed_dual):
        """Both sides of the money-balance identity for the real-time leg.

        Args mirror one period-set of the clearing output plus the position:
        ``shed``/``demand_bus``/``s_da``/``d_da`` are ``(T, n_buses)`` and the
        three duals come from the **same solve** that produced ``lmp``.

        Returns ``left``, ``right``, their difference, and the two right-hand
        terms separately, because the identity is checked by comparing the two
        sides and the split is what says which term carries it.
        """
        # net injections, formed the same way on both sides
        gen = award.T @ bus_of_unit.T                            # (T, n_buses)
        gen_da = q_da.T @ bus_of_unit.T
        P = gen + shed - demand_bus
        P_da = gen_da + s_da - d_da
        f_da = P_da @ PTDF.T                                     # (T, n_lines)

        # left: payment to generators for deviations, minus the charge to load
        # for deviations of *served* load, so shed in either solve is in neither
        price = lmp.T[unit_bus]
        pay_gen = period_hours * jnp.sum(price * (award - q_da))
        served_dev = (demand_bus - shed) - (d_da - s_da)
        charge_load = period_hours * jnp.sum(lmp * served_dev)
        left = pay_gen - charge_load

        # right: congestion rent on the deviation flows, plus the shed-bound term
        rent = period_hours * jnp.sum(line_dual_up * (F[None, :] - f_da)
                                      + line_dual_dn * (F[None, :] + f_da))
        shed_term = period_hours * jnp.sum(shed_dual * (P - P_da))
        right = -rent - shed_term
        return dict(left=left, right=right, difference=left - right,
                    rent=rent, shed_term=shed_term)

    return settle, money_balance
