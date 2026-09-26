"""This market's clearing: the day-ahead operator at `T = 1`.

**No clearing code is added here.**  The real-time clearing is the
fixed-commitment SCED of `envs.day_ahead.clearing` restricted to a single
period.  What this module holds is the part that is not inherited: the Newton
budget, calibrated against this market's own consumed quantities rather than
copied from the day-ahead market's.

Both quantities this market consumes, `lmp` and `award`, are continuous and
reach money with nothing in between -- the commitment is exogenous, so no
rounding step absorbs solver error -- which is why the budget is calibrated on
the per-agent profit the settlement produces.
"""
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from powermarketjax.envs.day_ahead.clearing import make_clearing

#: Newton steps for this market's single-period clearing, **not** the day-ahead
#: market's 60.  Judged on per-agent profit: fewer steps leave a period's
#: profit measurably wrong.
MAX_ITER = 70

#: The dual start this market's calibration was measured at.  Recorded rather
#: than passed: `envs.day_ahead.clearing` fixes it internally.
DUAL_START = "cost_norm"

#: Units the low-rank route keeps in its pivoted block on this market's shape
#: (`kkt_lowrank`, "Free columns").  In a shed period the units off both
#: bounds and off their ramp are the marginal ones: 3-4 in each of the five
#: `case813nem` periods measured 2026-09-16, but
#: over a 48 x 64 true-value rollout (2026-09-17)
#: 16 of the 564 periods re-solved densely were left at a dual residual of
#: 2e-6..5e-3 with 8 units in the block, and all 16 fall to the dense route's
#: 1e-10..5e-9 with 32; more shed columns (813) do not help, so it is the unit
#: count.  32 costs 24 more block columns on a 539-wide block.
LOWRANK_FREE_UNITS = 32

#: A bus counts as behind a monitored line when its PTDF on that line exceeds
#: this; the buses behind one radial line share one price and are the columns
#: that go dual-degenerate together when that region sheds.
BEHIND_LINE_PTDF = 0.5


def lowrank_free_for(case, monitored_lines, n_units: int = LOWRANK_FREE_UNITS) -> Tuple[int, int]:
    """``(units, shed columns)`` for `envs.day_ahead.clearing.make_clearing`'s
    ``lowrank_free`` on this market's shape: the largest set of buses behind
    any one monitored line, so that the region that sheds behind a binding
    line fits in the pivoted block whichever line it is.  On `case813nem`
    with the rated lines that is 523 (line 10); the five failing periods of
    2026-09-16 shed behind line 738, 109 buses.  ``None`` for the line set
    means every line is monitored, which takes the dense route, where the
    sizing is unused.
    """
    if monitored_lines is None:
        return (0, 0)
    PTDF = np.asarray(case.PTDF, np.float64)
    mon = np.unique(np.asarray(monitored_lines, np.int64))
    behind = int((np.abs(PTDF[mon]) > BEHIND_LINE_PTDF).sum(1).max()) if mon.size else 0
    return (min(n_units, len(np.asarray(case.unit_p_min))), min(behind, PTDF.shape[1]))


def lowrank_free_all(case) -> Tuple[int, int]:
    """Every column free: ``(n_units, n_buses)``.  The sizing this market
    defaults to on its own shape since 2026-09-17: the
    arrowhead solve makes the block's width a linear cost, so nothing is
    eliminated through ``H0^-1`` and no region can be left out -- the sizing
    rule, the bridge-line count and the leftover-mass stamp all fall away.
    """
    return (len(np.asarray(case.unit_p_min)), int(np.asarray(case.PTDF).shape[1]))


def default_lowrank_free(case, monitored_lines, n_lookahead: int = 1,
                         n_segments: int = 1) -> Tuple[int, int]:
    """What `make_rt_clearing` sizes the block to when not told: all columns
    on the arrowhead shape (``n_lookahead * n_segments == 1``), `lowrank_free_for`
    on any other, ``(0, 0)`` on the dense route.  `tools/benchmark/run_rl_02.py`
    reads the same function, so the value it expects is the value in effect.
    """
    if monitored_lines is None:
        return (0, 0)
    if n_lookahead * n_segments == 1:
        return lowrank_free_all(case)
    return lowrank_free_for(case, monitored_lines)


def make_rt_clearing(
    case,
    n_segments: int = 1,
    cap_scale: float = 1.0,
    ramp_scale: float = 1.0,
    period_hours: float = 0.5,
    max_iter: int = MAX_ITER,
    n_lookahead: int = 1,
    monitored_lines=None,
    lowrank_free: Optional[Tuple[int, int]] = None,
    lu_batching: str = "auto",
    reg_coef: Optional[float] = None,
    stop_tol: Optional[Tuple[float, float]] = None,
) -> Tuple[Callable, Dict]:
    """The day-ahead clearing over this market's decision window.

    Returns ``(clear, spec)`` exactly as `envs.day_ahead.clearing.make_clearing`
    does; ``clear(offer, u, demand, p_init)`` takes ``n_lookahead`` periods of
    ``demand`` and the previous period's realised dispatch as ``p_init``.

    Args:
        n_lookahead: periods solved together.  Larger values solve
            ``[t, t+n_lookahead-1]`` and the environment realises only ``t``.
            `MAX_ITER` was calibrated at 1 and does not carry over: `T > 1`
            sends the KKT from dense to block-tridiagonal, which is a different
            solve rather than a longer one.
        monitored_lines: which line limits the LP carries, passed straight to
            `envs.day_ahead.clearing.make_clearing` -- ``None`` enforces every
            line and takes the dense KKT route, which is what every archive on
            disk was produced on and what this argument defaults to.  An index
            array enforces those lines only and takes the low-rank route.  **On
            ``case813nem`` the two routes agree on price and on each period's
            total but not on the per-unit dispatch**: measured 2026-09-16 over
            18 days, ``|dlmp|`` 4.93e-07 \$/MWh against ``|dq|`` up to 62.39 MW
            on 34 of 151 units, the redistribution a degenerate LP is free to
            make between vertices of the same optimal face.  So a claim that
            consumes per-unit dispatch has to stay on one route throughout.
        lowrank_free: the low-rank route's pivoted-block sizing; ``None``
            takes `default_lowrank_free`: every column on this market's own
            shape, `lowrank_free_for` with a lookahead.  On this
            market's shape the plain Schur route solves a shed period behind
            a binding line to a residual of 1e1..4e1 against the dense
            route's 1e-12 (five `case813nem` periods, 2026-09-16), so the
            sizing is not optional here; ``(0, 0)`` reproduces that defect.
        reg_coef, stop_tol: passed straight to `envs.day_ahead.clearing.make_clearing`
            and stamped there as ``spec["reg_coef"]`` / ``spec["stop_tol"]``; ``None``
            for both is the operator as it always was.
        lu_batching: passed straight to `envs.day_ahead.clearing.make_clearing`;
            ``"auto"`` is the arrowhead solve on this market's own shape.
    """
    if n_lookahead < 1:
        raise ValueError(f"n_lookahead must be at least 1, got {n_lookahead}")
    if lowrank_free is None:
        lowrank_free = default_lowrank_free(case, monitored_lines, n_lookahead, n_segments)
    return make_clearing(case, n_lookahead, n_segments=n_segments,
                         cap_scale=cap_scale, ramp_scale=ramp_scale,
                         period_hours=period_hours, max_iter=max_iter,
                         monitored_lines=monitored_lines, lowrank_free=lowrank_free,
                         lu_batching=lu_batching,
                         reg_coef=reg_coef, stop_tol=stop_tol)
