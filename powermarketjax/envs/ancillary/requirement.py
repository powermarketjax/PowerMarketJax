"""Reserve requirement: a declared fraction of the forecast demand.

    d_res[j] = beta[j] * sum_n d_forecast[n]

The rule reads the forecast, not the realised demand.  The requirement is
published before offers are submitted, when only the forecast exists; and
because realised demand does not move the forecast, the energy price formula is
unaffected.  Binding the rule to the realisation would instead add
``sum_j beta[j] * lambda_res[j]`` to every energy price.

The fractions have no defensible default and are required arguments.  Their
usable range is a property of the committed fleet, not of this rule: too small
and the requirement is met without moving the energy dispatch at all, so the
reserve price is zero; too large and the fleet cannot deliver it and the price
sits at VOLR in every period.
"""
from typing import Sequence

import chex
import jax.numpy as jnp
import numpy as np


def make_requirement(betas: Sequence[float]):
    """Build the requirement rule for a declared pair of fractions.

    Args:
        betas: one fraction per product, in product order.

    Returns:
        ``requirement(demand_forecast) -> (n_prod,)`` in MW, pure and jittable,
        where ``demand_forecast`` is the published forecast of the period, either
        per bus or already totalled.
    """
    beta = np.asarray(betas, np.float64)
    if beta.ndim != 1 or len(beta) == 0:
        raise ValueError(f"betas must be a non-empty 1-D sequence, got {beta!r}")
    if (beta < 0.0).any():
        raise ValueError(f"a requirement fraction cannot be negative: {beta!r}")
    if (beta > 1.0).any():
        # A fraction above one asks for more reserve than the demand it is a
        # fraction of, which no ramp-capability check could pass; catching it
        # here beats diagnosing a market that is short in every period.
        raise ValueError(f"a requirement fraction above 1.0 is a scenario "
                         f"error, not a scarce market: {beta!r}")
    beta_j = jnp.asarray(beta)

    def requirement(demand_forecast: chex.Array) -> chex.Array:
        """`(n_prod,)` requirement in MW for one period's published forecast.

        The sum is over whatever axis the forecast arrives on, so a per-bus
        vector and an already totalled scalar give the same answer.
        """
        return beta_j * jnp.sum(demand_forecast)

    return requirement
