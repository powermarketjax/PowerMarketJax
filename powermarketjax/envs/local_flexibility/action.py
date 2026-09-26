"""Local flexibility action map: raw action in, submission out.

Array in, array out, pure, jittable.  It turns the three-component raw action
into the flexibility offer the clearing consumes, and it is the only place the
deliverability rule is enforced.

    alpha_pi  -> price     = c_rep * (1 + softplus(alpha_pi))       $/MWh
    alpha_q   -> qty_max   = sigmoid(alpha_q) * q_phys              MW
    alpha_ch  -> plan      = sigmoid(alpha_ch) * charge headroom    MW

**Megawatts out, not per unit.**  The battery ratings this map is built from
are in MW and the clearing problem is per unit, so the conversion happens once
in `env.py` at the call to `clear`.

`qty_max <= q_phys` holds by construction and nothing is repaired afterwards:
`plan` is a fraction of the charging headroom and `q_phys = plan + deliverable
discharge`.  Those two envelope terms are the ones
`compute_feasible_power_batch` computes, written out here because `q_phys`
itself is needed and not only a clipped power.

**The map clips nothing and the action space is a finite box, and those two
are not in tension.**  The map is defined on all of R^3 -- softplus and sigmoid
are what bound the submission, and nothing is repaired afterwards -- while
`[-ACTION_SATURATION, +ACTION_SATURATION]` is the frame in which both of those
functions have already saturated -- on every end but the one named below -- so
every submission this market can act on is reachable inside the box.  The
non-learner's baseline `(-128, +128, -128)` is a corner of the box rather than
a point outside it, and its three saturation
points are exact in float32 -- `exp(-128)`
falls below the smallest subnormal and `exp(+128)` overflows, whichever library
computes the two functions -- so the baseline submission is *exactly* the
truthful price, the full deliverable quantity and zero planned charging.

**The box is declared here rather than by each learner.**  An action space that
two algorithms each declare for itself is one quantity with two defaults;
`learning/policy.py:bounds_for` reads `spec["action_low"]` and
`spec["action_high"]` and refuses to invent them.  A box is needed because
SAC's critic reads the raw coordinate while its actor maximises the critic, so
an unbounded coordinate leaves the critic with no fixed point: measured
2026-09-09 on cell `2040p_2040c`, `q_loss` reached 9.0e18 inside the first
iteration against 3.2e3 under a finite box (the SAC section of
`tools/flex_experiment/concentration_baseline.py`).  `envs/ancillary/action.py`
answered the same failure the same way, and 05 was the last of the five markets
still declaring `+-inf`.

**What the box costs, stated rather than implied.**  On five of the six ends,
nothing at all: at `alpha_pi = -128`, `alpha_q = -+128` and `alpha_ch = -+128`
every one of `price`, `qty_max`, `plan` and `q_phys` is bit-identical to what
the map returns at `-+1e6`, so no action outside the box produces a submission
the box cannot (measured 2026-09-10 over twelve participants,
`tests/envs/local_flexibility/test_action_l1.py`).  The sixth is `alpha_pi =
+128`, where softplus is the identity and there is no saturation to reach: the
price is exactly `129 c_rep` and a larger action prices higher still.  **The
box therefore loses submissions, and no outcomes.**  What carries that is not
the price coordinate but the quantity one: `alpha_q = -128` offers exactly zero
megawatts, (CAP) then holds every award at zero, and an offer of nothing clears
nothing whatever its price -- so refusing to sell is inside the box on a
coordinate where the box is exact.  At the adopted Swiss configuration the
price route is available too, since `129 c_rep` is 19 334.8 $/MWh against a
`VOLL` of 10 000 at `c_rep = 149.88`; that is a fact about the scenario rather
than about the box, and it does not hold for `c_rep < VOLL / 129 = 77.5`
$/MWh, which is why the argument above is made on the quantity.

**`c_rep` is a replacement cost, not an accounting cost.**  It charges
degradation on the megawatt hour delivered and on the energy that replaces it,
and it charges the replacement energy at the exogenous price, both inflated by
the round-trip efficiency because more than one MWh has to be stored to deliver
one.  With `energy_price` unset there is no origin for the markup.
"""
from typing import Callable, Dict, Tuple

import chex
import jax
import jax.numpy as jnp

#: Both ends of the action box, on all three coordinates.  **Not a new number:
#: it is the coordinate already fixed for this market's baseline,
#: and for the property the box needs.**  128 is a power of two, exactly
#: representable, and below every saturation threshold of both ways of writing
#: softplus and sigmoid, so the ends are exact whichever library evaluates
#: them.  Those thresholds are *not* implementation independent and are not
#: what the box rests on: measured 2026-09-10, float32, jax 0.10.2, CPU
#: backend, `jax.nn.softplus` and `jax.nn.sigmoid` are exactly zero from -88
#: downwards and `jax.nn.sigmoid` exactly one from +17 upwards, while the
#: bare formulas need about -104 and -95.
#:
#: **What the learner actually uses, so the width is not asserted blind.**
#: Measured 2026-09-10 on cell `2040p_2040c`, seed 1, per-agent parameters, over
#: the first 30 iterations of the IPPO arm and 1 105 920 submitted actions:
#: the largest coordinate the
#: market was handed is 12.43 in absolute value -- 11.62 on the price column,
#: 11.82 on the quantity, 12.43 on the planned charge -- which is 10.3 times
#: inside this box and does not reach even `+-17`, where the two sigmoids have
#: already saturated.  That is a prefix of a 300 iteration run rather than the
#: whole of one, and the noise term contracts along the anneal
#: (`4 exp(-0.5) = 2.43` down to `4 exp(-3) = 0.199`), so what could still grow
#: is the mean; over those 30 iterations it does not.
#:
#: One scalar pair covers all three coordinates because one scalar pair is what
#: `bounds_for` publishes, so the box is symmetric by construction and not by
#: choice.  What the symmetry costs a learner that squashes into it is
#: resolution rather than reach.  Under `tanh` the two sigmoid coordinates are
#: unsaturated only for `|pre| < atanh(17/128) = 0.134` on this backend, and
#: the truthful price is reached from `pre < -atanh(88/128) = -0.843` here or
#: `-atanh(104/128) = -1.134` under the bare formula -- both
#: inside ordinary exploration, so nothing is out of reach; what is lost is
#: that most of the `tanh` range lands on the flats.
ACTION_SATURATION = 128.0


def make_action_map(n_agent: int, period_hours: float) -> Tuple[Callable, Dict]:
    """Build the action map for one population and period length.

    Args:
        n_agent: number of aggregators; fixes the action shape in ``spec``.
        period_hours: $\\Delta$, the period length in hours.

    Returns:
        ``(act_map, spec)`` where ``act_map`` is pure and jittable:

            action       (n_agent, 3)  raw action, unbounded
            soc          (n_agent,)    state of charge at the start of the period
            energy_price ()            exogenous energy price, \\$/MWh
            battery                    ``BatteryBundle`` read as parameters only
            cycle_cost   (n_agent,)    degradation cost per MWh of throughput

        returning ``price`` in \\$/MWh, ``qty_max``, ``plan``, ``q_phys`` and
        ``charge_headroom`` in MW, and ``c_rep`` in \\$/MWh, each
        ``(n_agent,)``.
    """
    if period_hours <= 0.0:
        raise ValueError(f"period_hours must be positive, got {period_hours}")

    delta = jnp.float32(period_hours)

    def act_map(action: chex.Array, soc: chex.Array, energy_price: chex.Array,
                battery, cycle_cost: chex.Array) -> Dict[str, chex.Array]:
        """Turn one period's raw action into a flexibility offer.

        Args:
            action: ``(n_agent, 3)``, unbounded.  Column 0 is the markup over
                the replacement cost, column 1 the fraction of the deliverable
                quantity actually offered, column 2 the fraction of the
                charging headroom the participant plans to take.
            soc: ``(n_agent,)`` state of charge at the start of the period.
            energy_price: scalar exogenous energy price, \\$/MWh.
            battery: ``BatteryBundle``, read for its registered ratings only.
            cycle_cost: ``(n_agent,)`` degradation cost per MWh of throughput.

        Returns:
            A dict of ``(n_agent,)`` arrays.  ``price`` and ``c_rep`` are in
            \\$/MWh; ``qty_max``, ``plan``, ``q_phys`` and ``charge_headroom``
            are in MW.  ``qty_max`` is the offered quantity of the
            `flex_offer`, that is, a deviation from the baseline operating
            point rather than an energy quantity, and ``plan`` is the charging
            that enters that baseline.
        """
        alpha_pi, alpha_q, alpha_ch = action[:, 0], action[:, 1], action[:, 2]

        # Degradation on the delivered MWh and on the one that replaces it,
        # plus the replacement energy, both inflated by the round trip
        eta_rt = battery.eta_charge * battery.eta_discharge
        c_rep = cycle_cost * (1.0 + 1.0 / eta_rt) + energy_price / eta_rt
        price = c_rep * (1.0 + jax.nn.softplus(alpha_pi))

        # Both bracketed quantities are non-negative without a guard:
        # `update_soc_batch` keeps `soc` inside its own bounds and
        # `make_battery_bundle` rejects a negative rating, a non-positive
        # capacity and an efficiency outside (0, 1] at construction.
        headroom = jnp.minimum(
            battery.power_max,
            battery.capacity * (battery.soc_max - soc)
            / (battery.eta_charge * delta))
        plan = jax.nn.sigmoid(alpha_ch) * headroom

        # The charging that can be forgone plus the discharging that can be
        # sustained, the second bounded by whichever of the rating and the
        # stored energy binds first
        q_phys = plan + jnp.minimum(
            battery.power_max,
            battery.eta_discharge * battery.capacity
            * (soc - battery.soc_min) / delta)
        qty_max = jax.nn.sigmoid(alpha_q) * q_phys

        return dict(price=price, qty_max=qty_max, plan=plan, q_phys=q_phys,
                    c_rep=c_rep, charge_headroom=headroom)

    spec = dict(n_agent=n_agent, action_shape=(n_agent, 3),
                action_low=-ACTION_SATURATION, action_high=ACTION_SATURATION,
                period_hours=float(period_hours), dtype=jnp.float32)
    return act_map, spec
