"""Verification of the cleared schedule against the nonlinear power flow.

Array in, array out, pure, jittable.  The cleared quantities and the directed
curtailment define a physical operating point; this substitutes that point
into the radial power flow of `physics.bfs_power_flow` and reports how far
the real operating point misses the registered limits, tested against the
bounds without the safety margin.

The award does not change.  A cleared quantity is financially binding, so
settlement is untouched by whatever this finds, and the depth of any
violation travels on the `costs` channel rather than the reward.  This
measures the modelling error of the linearised clearing; it does not repair
it, and awards are never adjusted afterwards, since that would break the
settlement's money balance.  The one admissible response is the safety
margin in `clearing`, set before clearing.

Convergence is not admissibility, and no flag here conflates them.
`converged` says the iteration reached a genuine fixed point, which the
vendored solver defines as the tolerance being met and the voltage floor
being inactive -- a node pinned on that floor has zero increment and would
otherwise be mistaken for a solution.  `floor_active` says whether the floor
was engaged at all.  The violation depths say whether the point is
admissible; a converged solve can still be inadmissible, so the caller reads
the depths, never the flag.

The cast at the entry is a no-op in the configuration this market runs in:
the vendored solver's own accumulators are float64 whenever `x64` is on, and
the clearing requires `x64`.  It is kept because it states the intent at the
boundary.
"""
from typing import Callable, Dict

import chex
import jax.numpy as jnp
import numpy as np

from powermarketjax.physics import bfs_power_flow, prepare_bfs

from .sensitivity import VoltageSensitivity

#: Iterations allowed to the sweep.  The vendored default; on these feeders it
#: converges in three to six, and the ceiling only matters on the operating
#: points that collapse onto the voltage floor.
MAX_SWEEP = 100


def make_verification(case, sens: VoltageSensitivity,
                      max_iter: int = MAX_SWEEP) -> Callable:
    """Build the verification for one case.

    Args:
        case: a `CaseData`; read for the registered voltage bounds and the
            reactive-to-active demand ratio.
        sens: network constants, used for the line ratings and the agent
            placement is supplied per call.
        max_iter: iteration ceiling for the sweep.

    Returns:
        ``verify(p_inj, q_inj) -> dict``, pure and jittable, taking the **net**
        per-unit injections of the cleared operating point, positive into the
        feeder.  Returns the nonlinear operating point, the three violation
        depths that travel on the `costs` channel, and the solver flags that
        travel on `info`.
    """
    topo = prepare_bfs(case)
    if topo.n_lines != sens.n_line:
        raise ValueError(
            f"the sweep topology carries {topo.n_lines} lines and the "
            f"sensitivity matrices {sens.n_line}; they must index the same "
            "in-service lines")

    v_lo = jnp.asarray(np.asarray(case.node_v_min, np.float64), jnp.float32)
    v_hi = jnp.asarray(np.asarray(case.node_v_max, np.float64), jnp.float32)
    p_max = jnp.asarray(sens.p_max, jnp.float32)

    def verify(p_inj: chex.Array, q_inj: chex.Array) -> Dict[str, chex.Array]:
        """Sweep the cleared operating point and measure it against the registered limits.

        Args:
            p_inj: ``(n_bus,)`` net active injection of the cleared point, per
                unit, positive into the feeder; `cleared_injection` builds it.
            q_inj: ``(n_bus,)`` net reactive injection, same convention.

        Returns:
            A dict carrying the nonlinear operating point, the three checks and
            the three flags:

            ``v_mag`` ``(n_bus,)`` and ``p_branch`` ``(n_line,)``, the point
                the sweep returned, in per unit; whether it is a fixed point is
                a separate question the flags below answer.
            ``v_under`` / ``v_over`` ``(n_bus,)``, depth below `node_v_min` and
                above `node_v_max`, per unit.  A positive entry means the sweep
                found that bus outside the registered band at an operating
                point the linearised clearing held inside it, so it measures
                the linearisation error the safety margin did not cover.  The
                award is not revised on account of it; the depth travels on
                `costs`.
            ``overload`` ``(n_line,)``, depth of ``|p_branch|`` above the
                registered rating, per unit, read the same way.
            ``converged`` / ``floor_active`` / ``iterations``, what the sweep
                itself did.  ``converged`` false means the iteration never
                reached a fixed point, so the three depths above measure
                nothing and are not a verdict either way; it is never a
                feasibility criterion, and admissibility is read off the depths
                alone.
        """
        # the sweep takes loads, positive for consumption; the cast states the
        # boundary and binds only when x64 is off (see the module docstring)
        swept = bfs_power_flow(topo,
                               jnp.asarray(-p_inj, jnp.float32),
                               jnp.asarray(-q_inj, jnp.float32),
                               max_iter=max_iter)

        v_mag = swept.v_mag
        return dict(
            v_mag=v_mag,
            p_branch=swept.p_branch,
            # the three quantities sent to the cost channel
            v_under=jnp.maximum(v_lo - v_mag, 0.0),
            v_over=jnp.maximum(v_mag - v_hi, 0.0),
            overload=jnp.maximum(jnp.abs(swept.p_branch) - p_max, 0.0),
            # diagnostics, for `info`; none of these is a feasibility criterion
            converged=swept.converged,
            floor_active=swept.floor_active,
            iterations=swept.iterations,
        )

    return verify


def cleared_injection(sens: VoltageSensitivity, agent_bus, phi,
                      p_inj_base, q_inj_base, award, shed):
    """Net injection of the cleared operating point, per unit.

    The baseline plus the awards at the buses that made them plus the
    curtailment, with curtailment removing reactive load at the registered
    ratio as well.  Kept beside the verification because getting the
    reactive term wrong leaves the sweep converging on a slightly different
    network than the one the clearing solved, and the difference looks like
    linearisation error.

    Args:
        sens: network constants.  The body does not read them; the parameter
            names the network the other arguments are indexed against.
        agent_bus: ``(n_agent,)`` bus of each aggregator.  The awards are
            scattered with an add, so two aggregators at one bus accumulate.
        phi: ``(n_bus,)`` reactive-to-active demand ratio, from the clearing
            operator's ``spec`` rather than recomputed.
        p_inj_base, q_inj_base: ``(n_bus,)`` baseline injections, per unit.
        award: ``(n_agent,)`` cleared quantities, per unit.
        shed: ``(n_bus,)`` directed curtailment, per unit; it adds to the net
            injection because curtailing removes load.

    Returns:
        ``(p_inj, q_inj)``, both ``(n_bus,)`` per unit and positive into the
        feeder, which is what `verify` takes.
    """
    p = p_inj_base.at[jnp.asarray(agent_bus)].add(award) + shed
    return p, q_inj_base + jnp.asarray(phi) * shed
