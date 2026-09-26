"""P2P action map: raw action in, submission out.

Array in, array out, pure.  It turns the two-component action into the
price-quantity pair the double auction consumes.

    alpha_bat    -> p_signed, p_ch, p_dis, clip     battery feasible envelope
    p_pv, load   -> net_position
    net_position -> q_sell, q_buy                   scaled by `period_hours`
    alpha_price  -> price in [pi_exp, pi_ret]

The action space is ``Box(-1, 1)`` of shape ``(n_agents, 2)``, and a raw action
outside it is clipped onto it here, once, at entry.  Every submission is then
legal by construction and none is ever repaired after being built.

The price map is affine rather than smooth so that ``alpha_price = -1`` and
``+1`` land exactly on ``pi_exp`` and ``pi_ret``, the truthful submissions and
the reference strategy of this market.  A smooth map buys no gradient in
exchange: a participant's price reaches the clearing price only while that
participant is marginal, and elsewhere the price is set by other submissions
and the pathwise derivative is exactly zero.

``clip`` is computed from the clipped action rather than the raw one, so it
measures one thing only: how far the state of charge fell short of delivering
the commanded power.  From the raw action an ``alpha_bat`` of 2 would report a
full clip on an untouched battery, and a CMDP algorithm would penalise the
range of the policy output rather than a physical infeasibility.

The battery arrives as a ``BatteryBundle`` used **as a parameter container
only**: this module reads its arrays and calls ``compute_feasible_power_batch``,
and never calls ``BatteryBundle.step``, which reduces the per-device clip and
degradation cost to scalars with ``jnp.sum`` and carries a single static
degradation price.  Building the bundle through ``make_battery_bundle`` is what
rejects a zero or non-finite capacity at setup time.
"""
from typing import Callable, Dict, Tuple

import chex
import jax.numpy as jnp

from powermarketjax.resources.battery import (
    compute_feasible_power_batch, update_soc_batch)

#: Denominator floor for the normalisation by rated power.  A heterogeneous
#: population may be padded with devices of ``power_max=0``, and 0/0 would put a
#: NaN on the `costs` channel of a participant that submitted nothing.
RATED_EPS = 1e-6


def make_action_map(
    n_agents: int,
    pi_exp: float,
    pi_ret: float,
    period_hours: float,
) -> Tuple[Callable, Dict]:
    """Build the action map for one population, tariff pair and period length.

    Returns ``(act_map, spec)`` where ``act_map`` is pure and jittable:

        action   (n_agents, 2)  raw action; column 0 is the battery command and
                                column 1 the price command, both nominally in
                                [-1, 1] and clipped onto it here
        soc      (n_agents,)    state of charge at the start of the period
        p_pv     (n_agents,)    photovoltaic output, MW, exogenous
        load     (n_agents,)    own load, MW, exogenous
        battery                 ``BatteryBundle`` read as parameters only

    and returning a dict of ``price``, ``q_sell``, ``q_buy``, ``net_position``,
    ``p_signed``, ``p_ch``, ``p_dis``, ``throughput`` and ``clip``, each
    ``(n_agents,)`` float32.  ``throughput`` is ``period_hours * (p_ch +
    p_dis)``, the energy in MWh that passed through the battery, which the
    settlement multiplies by the degradation price; ``clip`` is the
    constraint-channel quantity.

    ``pi_exp`` and ``pi_ret`` are Python floats, held constant over an episode
    and checked here because the check is impossible inside `jit`.
    ``period_hours`` is the period length :math:`\\Delta` and converts MW into
    the MWh the auction trades.
    """
    if not 0.0 <= pi_exp < pi_ret:
        # The strict inequality leaves room for a local trade to beat the grid
        # for both sides at once.  Equal tariffs collapse the price interval
        # onto a point and the market has nothing to divide; reversed ones put
        # the clearing price outside the bracket.
        raise ValueError(
            f"the export price must satisfy 0 <= pi_exp < pi_ret, got {pi_exp=} {pi_ret=}")
    if period_hours <= 0.0:
        raise ValueError(f"period_hours must be positive, got {period_hours}")

    exp = jnp.float32(pi_exp)
    ret = jnp.float32(pi_ret)
    delta = jnp.float32(period_hours)

    def act_map(action: chex.Array, soc: chex.Array, p_pv: chex.Array,
                load: chex.Array, battery) -> Dict[str, chex.Array]:
        """Turn one period's raw action into what each participant submits.

        Consumes the five arguments the factory docstring lists and returns the
        nine ``(n_agents,)`` float32 vectors it names there.  The battery is read
        as parameters; the state of charge it implies for the next period is
        `make_soc_advance`'s job, not this one.

        **Only the price is a decision.**  The side of the market is read off
        the sign of ``net_position``: a participant in surplus offers all of it
        as ``q_sell``, one in deficit bids for all of it as ``q_buy``, so exactly
        one of the two is non-zero for any participant and neither is chosen.
        The battery command moves the net position, and through it the side and
        the quantity, but it cannot withhold -- there is no curtailment action
        and no flexible load in this market -- so the price command carries the
        whole submission strategy.
        """
        alpha = jnp.clip(action, -1.0, 1.0)                    # onto Box(-1, 1)
        alpha_bat, alpha_price = alpha[:, 0], alpha[:, 1]

        rated = battery.power_max
        safe_rated = jnp.maximum(rated, RATED_EPS)
        p_desired = alpha_bat * rated

        # feasible-power envelope of the battery; positive is discharge
        p_signed = compute_feasible_power_batch(
            soc, p_desired, rated, battery.capacity,
            battery.soc_min, battery.soc_max,
            battery.eta_charge, battery.eta_discharge, period_hours)

        p_dis = jnp.maximum(p_signed, 0.0)
        p_ch = jnp.maximum(-p_signed, 0.0)
        clip = jnp.abs(p_desired - p_signed) / safe_rated

        net = p_pv + p_dis - load - p_ch          # signed; positive is surplus
        q_sell = delta * jnp.maximum(net, 0.0)
        q_buy = delta * jnp.maximum(-net, 0.0)

        # Written as an interpolation rather than as `exp + w * span`: the two
        # agree in exact arithmetic, but only this form returns `exp` and `ret`
        # *exactly* at the endpoints in float32.
        w = 0.5 * (1.0 + alpha_price)
        price = (1.0 - w) * exp + w * ret

        # Degradation is charged on energy through the battery, so this
        # MW-to-MWh conversion belongs beside the ones on `q_sell` and
        # `q_buy`; `settlement.py` then contains no unit of time at all.
        throughput = delta * (p_ch + p_dis)

        return dict(price=price, q_sell=q_sell, q_buy=q_buy, net_position=net,
                    p_signed=p_signed, p_ch=p_ch, p_dis=p_dis,
                    throughput=throughput, clip=clip)

    spec = dict(n_agents=n_agents, action_shape=(n_agents, 2),
                action_low=-1.0, action_high=1.0,
                pi_exp=float(pi_exp), pi_ret=float(pi_ret),
                period_hours=float(period_hours), dtype=jnp.float32)
    return act_map, spec


def make_soc_advance(period_hours: float) -> Callable:
    """Build the state-of-charge advance for one period length.

    Returns ``advance(soc, p_signed, battery)``, pure and jittable, where
    ``p_signed`` is the deliverable power ``act_map`` already produced.  It is a
    thin wrapper over ``resources.battery.update_soc_batch`` and adds nothing to
    the physics.

    The advance is kept out of ``act_map`` because it is the transition and not
    the submission: ``act_map`` answers what a participant brings to the
    auction, and this answers what the battery looks like afterwards.

    ``update_soc_batch`` ends with a clip onto the state-of-charge bounds, so
    the value returned here satisfies them whether or not the feasible-power
    envelope did.
    """
    if period_hours <= 0.0:
        raise ValueError(f"period_hours must be positive, got {period_hours}")

    def advance(soc: chex.Array, p_signed: chex.Array, battery) -> chex.Array:
        """State of charge the next period starts from.

        ``soc`` and ``p_signed`` are ``(n_agents,)``, ``p_signed`` positive for
        discharge as ``act_map`` returned it, and ``battery`` is the parameter
        container.  Returns the ``(n_agents,)`` state of charge, a fraction of
        capacity rather than an energy, over the ``period_hours`` closed over
        above.
        """
        return update_soc_batch(
            soc, p_signed, battery.capacity,
            battery.eta_charge, battery.eta_discharge,
            battery.soc_min, battery.soc_max, period_hours)

    return advance
