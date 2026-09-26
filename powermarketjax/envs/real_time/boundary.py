"""The episode's opening carry `p_prev`.

`p_prev` is the realised dispatch of the period before the one being cleared.
It is the only physical coupling between steps, so the first step of an episode
has nothing to read it from and it has to be constructed: clear that first period
*on its own*, with the ramp rows made non-binding, and take its dispatch.

**It must be built at this market's resolution.**  The per-period ramp allowance
is halved at `delta = 0.5 h` while an hourly starting point inherited from the
day-ahead schedule is not, so the opening period would have to absorb a step the
ramp rate forbids.

The offers used are the true-cost ones, matching the basis the day-ahead position
was built on: the boundary stands for the system running on its schedule when the
episode opened, and no agent has acted yet.
"""
from typing import Callable, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np

from powermarketjax.envs.day_ahead.clearing import segment_costs

from .clearing import MAX_ITER, make_rt_clearing

#: `ramp_scale` for the boundary solve, large enough that the ramp rows cannot
#: bind.  The rows still exist -- the operator's shape is fixed -- so this is how
#: "there is no previous period" is expressed against an array-shaped clearing.
RAMP_FREE = 1.0e4


def make_boundary(
    case,
    n_segments: int = 1,
    cap_scale: float = 1.0,
    period_hours: float = 0.5,
    max_iter: int = MAX_ITER,
    monitored_lines=None,
    spec_out=None,
    lowrank_free=None,
    lu_batching: str = "auto",
    reg_coef=None,
    stop_tol=None,
) -> Tuple[Callable, chex.Array]:
    """Build `boundary(u, demand) -> p_prev` for one case.

    Args:
        case: a `CaseData`.
        n_segments: offer segments, matching the market's.
        cap_scale: the market's line-rating scale.  The boundary is a dispatch of
            the same network, so it must see the same ratings.
        period_hours: `delta`; the boundary is built at the market's own
            resolution, never at the day-ahead market's.
        max_iter: Newton steps.
        monitored_lines: the line set this boundary's own clearing carries.
            **The boundary is a second clearing operator**, and it produces the
            dispatch the first period starts from, so leaving it on the dense
            route while the market runs low-rank opens the episode at the other
            route's vertex.  `envs.real_time.env.make_env` passes its own value
            here and refuses the pair if they disagree.
        spec_out: if a dict is given, the clearing `spec` this boundary was
            built with is written into it.  It exists so a caller can read the
            line set **off the operator** rather than trust that its argument
            arrived -- a flag that reached one operator and not the other is
            invisible in every product.  The return arity is
            unchanged because eight call sites unpack exactly two values.

    Returns:
        ``(boundary, offer)``.  ``boundary(u, demand)`` is pure and jittable and
        returns ``(p_prev, out)``: the dispatch of that period, ``(n_units,)`` in
        MW, and the clearing's own output dict.  ``offer`` is the true-cost offer
        this construction uses, exposed so a caller can check that the first step
        was given the same one.
    """
    clear, _spec = make_rt_clearing(case, n_segments=n_segments,
                                    cap_scale=cap_scale, ramp_scale=RAMP_FREE,
                                    period_hours=period_hours, max_iter=max_iter,
                                    monitored_lines=monitored_lines,
                                    lowrank_free=lowrank_free, lu_batching=lu_batching,
                                    reg_coef=reg_coef, stop_tol=stop_tol)
    if spec_out is not None:
        spec_out.clear()
        spec_out.update(_spec)
    _width, cost = segment_costs(case, n_segments)
    offer = jnp.asarray(cost)[:, :, None]                    # (n_units, K, 1)
    n_units = len(np.asarray(case.unit_p_min))

    def boundary(u, demand):
        """`u` is `(n_units, 1)` and `demand` is `(1,)`, both for that period.

        Returns ``(p_prev, out)``: the dispatch of that single period in MW,
        and the clearing's own output dict, so that a caller can read `mu` and
        judge whether the opening carry came from a converged solve.
        """
        # `p_init` is unused in substance: with `RAMP_FREE` the ramp rows cannot
        # bind whatever it is.  Zero also sets `u_prev` to all-off in the
        # operator's period-0 convention, which grants every committed unit the
        # start-up allowance and so cannot make the solve infeasible from below.
        out = clear(offer, u, demand, jnp.zeros(n_units))
        return out["award"][:, 0], out

    return boundary, offer
