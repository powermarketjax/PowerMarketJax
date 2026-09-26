"""The published requirement: baseline operating point in, two signals out.

Array in, array out, pure, jittable.  Given the baseline injections of one
period, it returns the linearised operating point and the two quantities
published on it,

    req_v[n]  = max(v_lo[n]^2 - v_sq[n], 0) / (2 R[n, n])      power at bus n
    req_th[l] = max(flow[l] - p_max[l], 0)                     power on line l

Both are signals, never constraints: the clearing problem enforces the
voltage bounds and the line ratings themselves, and publishing both is what
makes this a reverse auction rather than a redispatch instruction.  A
participant told only about voltage is told about part of what it is paid to
relieve.

A megawatt reaches the two differently, which is why they are not one signal
with two rows.  One megawatt injected downstream of a line reduces its flow
by exactly one megawatt and one injected elsewhere does nothing, so `req_th`
is already an active power and locational value against it is binary; `req_v`
instead scales with the shared path resistance, so locational value against
it is graded.  Both are powers, and like everything else here they are per
unit -- `env` multiplies by `base_mva` on the way into the observation, where
the other power columns are in MW.

Neither may be summed over its index: relieving one bus relieves its
neighbours, and lines on one path are nested, so both sums count the same
injection more than once.  The aggregates returned here are therefore the
largest requirement of each kind and the number of indices carrying one.

Everything is per unit; the line ratings were converted once in
`sensitivity`.  The substation bus is a special case: its row of `R` and `X`
is zero, so its squared voltage is exactly the reference and its requirement
is defined as zero rather than computed to avoid a division by zero, and its
registered bounds are both one, so it is not a constrained bus.
"""
from typing import Callable, Dict

import chex
import jax
import jax.numpy as jnp
import numpy as np

from .sensitivity import VoltageSensitivity


def make_requirement(case, sens: VoltageSensitivity) -> Callable:
    """Build the requirement publisher for one case.

    Args:
        case: a `CaseData`, read for `node_v_min` only.
        sens: the network constants from `build_voltage_sensitivity`.

    Returns:
        ``publish(p_inj, q_inj) -> dict``, pure and jittable, where the two
        arguments are the baseline active and reactive injections in per unit
        with a positive value representing injection into the feeder.
        The result carries ``v_sq`` and ``flow`` for the operating point,
        ``req_v`` and ``req_th`` for the two published requirements, and the
        four aggregates: an extreme and a count of the indices carrying a
        requirement, for each of the two.

    Raises:
        ValueError: if a bus other than the substation has zero path
            resistance, which leaves ``req_v`` undefined there.
    """
    if not jax.config.jax_enable_x64:
        # The published requirement is also the right-hand side the clearing
        # problem is assembled against, which needs float64.  Building it
        # here under float32 does not merely lose accuracy, it loses it where
        # the formula is most sensitive: `req_v` divides the voltage
        # shortfall by twice the path resistance, which reaches 1e-9 on a
        # vendored case, so a float32 rounding error of 1e-7 in the squared
        # voltage magnitude leaves the published requirement wrong by
        # megawatts while staying finite and plausible.
        raise RuntimeError(
            "the local flexibility requirement requires float64: set "
            "jax.config.update('jax_enable_x64', True) before building it")

    v_lo = np.asarray(case.node_v_min, np.float64).copy()
    v_lo_sq = v_lo ** 2

    diag = sens.R.diagonal()
    offending = np.flatnonzero(diag <= 0.0)
    offending = offending[offending != sens.slack]
    if offending.size:
        raise ValueError(
            "the voltage requirement divides by the path resistance, "
            f"which is zero at buses {offending.tolist()}; those buses have no "
            "resistive path to the substation and the requirement is undefined "
            "there")

    # The substation entry is zero rather than infinite: its requirement is
    # defined as zero, and folding that into the coefficient keeps the
    # runtime free of a guarded division.  The smallest path resistance
    # across the vendored cases is 1e-9, so this coefficient can reach 5e8:
    # a violation of 1e-9 per unit squared is then reported as 0.5 MW, which
    # is the formula's own sensitivity, not a defect of this implementation.
    coefficient = np.zeros_like(diag)
    live = np.arange(len(diag)) != sens.slack
    coefficient[live] = 1.0 / (2.0 * diag[live])

    v_lo_sq_j = jnp.asarray(v_lo_sq)
    coefficient_j = jnp.asarray(coefficient)
    R_j, X_j, A_j = jnp.asarray(sens.R), jnp.asarray(sens.X), jnp.asarray(sens.A)
    p_max_j = jnp.asarray(sens.p_max)

    def publish(p_inj: chex.Array, q_inj: chex.Array) -> Dict[str, chex.Array]:
        """The linearised operating point and the two requirements published on it.

        Args:
            p_inj: ``(n_bus,)`` active injection, per unit, positive into the
                feeder, so a load-only bus is negative.
            q_inj: ``(n_bus,)`` reactive injection, same sign convention.

        Returns:
            A dict, everything per unit as the module docstring states:

            ``v_sq`` ``(n_bus,)`` squared voltage magnitude of the linearised
                model; ``flow`` ``(n_line,)`` line flow, positive downstream,
                that is, away from the substation.
            ``req_v`` ``(n_bus,)`` and ``req_th`` ``(n_line,)``, the two
                published requirements.  Both are signals, never constraints,
                and neither may be summed over its own index.  ``req_th`` is
                one-sided in the flow direction: a reversed flow past the
                rating leaves it zero, while the clearing's thermal
                constraint is imposed against both signs.
            ``req_v_max``, ``req_v_count``, ``req_th_max``, ``req_th_count``,
                the four aggregates: an extreme and a count of the indices
                carrying a requirement, for each of the two, which is what
                remains available once summing is ruled out.
        """
        v_sq = 1.0 + 2.0 * (R_j @ p_inj + X_j @ q_inj)
        flow = -A_j @ p_inj

        req_v = jnp.maximum(v_lo_sq_j - v_sq, 0.0) * coefficient_j
        # Non-negativity is load-bearing outside this file.  `env.py` reduces
        # this over a bus's downstream path as `max(path_mask * req_th)`,
        # with `path_mask` in {0, 1}, so an excluded line contributes 0 and
        # 0 cannot outrank a real value.  Were this ever redefined as a
        # signed margin rather than an overload depth, that reduction would
        # silently return 0 instead of the largest value, with no error.
        req_th = jnp.maximum(flow - p_max_j, 0.0)

        return dict(
            v_sq=v_sq,
            flow=flow,
            req_v=req_v,
            req_th=req_th,
            # summing is forbidden, so the aggregates are an extreme and a count
            req_v_max=jnp.max(req_v),
            req_v_count=jnp.sum(req_v > 0.0),
            req_th_max=jnp.max(req_th),
            req_th_count=jnp.sum(req_th > 0.0),
        )

    return publish
