"""Local flexibility clearing operator: a linear program.

Array in, array out, pure, jittable.  Given the offers of one period and the
baseline operating point, it returns the quantity cleared for every
aggregator and the load the operator curtails at every bus.  There is no
uniform clearing price: this is a pay-as-bid market, every accepted offer is
paid its own submitted price, so the duals of this program enter no
settlement expression and are not returned.

    min  Δ Σ_i π_i q_i  +  Δ VOLL Σ_n s_n
    s.t. (VLO) (VHI) on every bus except the substation,
         (LIM) on every line, (CAP) on every offer, (SHED) on every bus

A linear program because acceptance is divisible; no MILP solver runs inside
`jit` on an accelerator.  Per unit throughout except `price`, with the
objective's `base_mva` factor putting the optimal value `z` in dollars.

Two safety margins pull the voltage bounds and the line ratings inward: both
compensate for the linearised flow and voltage-drop terms understating their
true values, so a bus or line cleared exactly onto its bound can be over that
bound once the full nonlinear network is swept.  Verification always tests
against the registered limits, so whatever a margin fails to cover is
reported rather than hidden.

(VLO)/(VHI) skip the substation bus: its row of `R` and `X` is zero, so its
squared voltage is identically the reference, held there by the transmission
system; imposing (VLO) with a positive margin would demand more than one from
a bus fixed at one, making the program infeasible for no physical reason.

Directed curtailment is bounded by the load actually present: the cases
register a net injection per bus, so a bus that nets to generation has
nothing to curtail -- the data does not resolve gross load behind embedded
generation.

No equality row: power balance is substituted out by (FL), and the solver is
asked for `n_eq=0` directly.  The interior point method itself is imported
from `solvers.ipm`, which knows nothing about electricity markets and reaches
the Newton system through two pluggable interfaces.
"""
from typing import Callable, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np

from powermarketjax.solvers import ipm

from .requirement import make_requirement
from .sensitivity import VoltageSensitivity

#: Value of lost load, the penalty price on directed curtailment, \$/MWh.
#: Not yet rechecked against this market's own cost basis: it is the ceiling
#: on what the market can charge, and the offer prices here are built from a
#: different cost basis pending the exogenous price series.
VOLL = 10_000.0

#: Newton steps.  This market consults no dual, so the count is calibrated
#: against the primal solution's convergence alone rather than against dual
#: accuracy.  Below the adopted value the solve is measurably incomplete:
#: awards and the objective have not settled.  The caller must still read
#: `mu`, which is what actually detects a solve that did not converge.
MAX_ITER = 80

#: Slack kept on any box whose two bounds would otherwise coincide (zero
#: offer, or a bus with no load), which would destroy the strict interior
#: the interior point method needs.  The resulting phantom quantities are at
#: most `OFF_EPS` and are zeroed out of `award` and `shed` before they leave.
OFF_EPS = 1e-9


def make_clearing(
    case,
    sens: VoltageSensitivity,
    agent_bus: np.ndarray,
    voltage_margin: float = 0.0,
    thermal_margin: float = 0.0,
    period_hours: float = 0.25,
    voll: float = VOLL,
    max_iter: int = MAX_ITER,
    dual_start: str = "unit",
) -> Tuple[Callable, Dict]:
    """Build the clearing operator for one case and one aggregator population.

    Args:
        case: a `CaseData`; read for the voltage bounds and the demand ratio.
        sens: network constants from `build_voltage_sensitivity`.
        agent_bus: ``(n_agent,)`` bus index of each aggregator.  Placement is
            supplied by the caller; this operator does not generate it.
        voltage_margin: per-unit voltage margin, pulling both voltage bounds
            inward.
        thermal_margin: fraction of each line rating, pulling (LIM) inward.
            Relative rather than absolute because the ratings on the primary
            case span two orders of magnitude.  Both default to zero, which
            applies no margin.
        period_hours: the period length in hours.
        voll: penalty price on directed curtailment, \\$/MWh.
        max_iter: Newton steps; see `MAX_ITER` for why the caller must still
            read `mu`.

    Returns:
        ``(clear, spec)``.  ``clear(price, qty_max, p_inj, q_inj, load)`` is
        pure and jittable, with every array per unit except ``price`` in
        \\$/MWh, and returns a dict carrying ``award``, ``shed``, the
        published requirement, the objective value ``z``, and the solver
        diagnostics ``mu`` and ``dual_residual``.
    """
    if not jax.config.jax_enable_x64:
        # Under float32 the interior point method returns a finite,
        # plausible-looking solution whose complementarity gap is meaningless.
        raise RuntimeError(
            "the local flexibility clearing requires float64: set "
            "jax.config.update('jax_enable_x64', True) before building it")

    agent_bus = np.asarray(agent_bus, np.int64)
    n_agent, n_bus, n_line = len(agent_bus), sens.n_bus, sens.n_line
    n_var = n_agent + n_bus

    # the substation is not a constrained bus
    bounded = np.setdiff1d(np.arange(n_bus), [sens.slack])
    n_bounded = len(bounded)

    v_lo = np.asarray(case.node_v_min, np.float64)[bounded] + voltage_margin
    v_hi = np.asarray(case.node_v_max, np.float64)[bounded] - voltage_margin
    if (v_lo > v_hi).any():
        raise ValueError(
            f"voltage_margin={voltage_margin} closes the band at "
            f"{int((v_lo > v_hi).sum())} buses")
    if not 0.0 <= thermal_margin < 1.0:
        raise ValueError(
            f"thermal_margin={thermal_margin} must lie in [0, 1); it is a "
            "fraction of each line rating, not a per-unit power")

    # The dropped loss terms understate the flow just as they understate the
    # voltage drop, so a line the program clears onto its rating can be over
    # that rating once the full network is swept, and no value of the voltage
    # margin covers it.
    p_max_margin = sens.p_max * (1.0 - thermal_margin)

    # reactive-to-active demand ratio; curtailment removes both.  It goes out
    # in `spec` because `cleared_injection` needs exactly this vector, and
    # recomputing it in the environment layer would be a second copy of a
    # rule that has to agree with this one for the sweep to verify the
    # network the clearing solved.
    pd = np.asarray(case.node_pd, np.float64)
    qd = np.asarray(case.node_qd, np.float64)
    phi = np.where(pd > 0.0, qd / np.where(pd > 0.0, pd, 1.0), 0.0)

    # sensitivity of the squared voltage magnitude to each variable
    dv_dq = 2.0 * sens.R[np.ix_(bounded, agent_bus)]            # (n_bounded, n_agent)
    dv_ds = 2.0 * (sens.R[bounded] + sens.X[bounded] * phi[None, :])
    dv = np.hstack([dv_dq, dv_ds])                               # (n_bounded, n_var)

    # sensitivity of the line flow to each variable
    df = np.hstack([sens.A[:, agent_bus], sens.A])               # (n_line, n_var)

    identity = np.eye(n_var)
    G = np.vstack([-dv, dv, -df, df, identity, -identity])
    row0 = dict(vlo=0, vhi=n_bounded, lim_lo=2 * n_bounded,
                lim_hi=2 * n_bounded + n_line, box_hi=2 * n_bounded + 2 * n_line)
    m = G.shape[0]

    spec = dict(n_agent=n_agent, n_bus=n_bus, n_line=n_line, n=n_var, m=m,
                bounded=bounded, agent_bus=agent_bus, row0=row0, phi=phi,
                voltage_margin=voltage_margin, thermal_margin=thermal_margin,
                period_hours=period_hours,
                voll=voll, base_mva=sens.base_mva)

    publish = make_requirement(case, sens)
    solve = ipm.make_solver(n_var, m, n_eq=0, max_iter=max_iter,
                            # valid choices are in `ipm.DUAL_STARTS`
                            dual_start=dual_start)

    G_j = jnp.asarray(G)
    v_lo_sq_j, v_hi_sq_j = jnp.asarray(v_lo ** 2), jnp.asarray(v_hi ** 2)
    p_max_j = jnp.asarray(p_max_margin)
    bounded_j = jnp.asarray(bounded)
    money = period_hours * sens.base_mva          # per-unit power to dollars
    voll_j = jnp.asarray(voll)
    # `n_eq=0` is passed to the solver; these are the empty equality blocks its
    # signature still takes.  A row of zeros here would be a different thing
    # entirely and would break the solve silently.
    A_empty, b_empty = jnp.zeros((0, n_var)), jnp.zeros((0,))

    def clear(price, qty_max, p_inj, q_inj, load):
        """Clear one period's offers against the voltage bounds and the ratings.

        Args:
            price: ``(n_agent,)`` submitted offer price, \\$/MWh.
            qty_max: ``(n_agent,)`` offered quantity, per unit; the right-hand
                side of (CAP), and a deviation from the baseline rather than an
                energy quantity.
            p_inj: ``(n_bus,)`` baseline active injection, per unit, positive
                into the feeder.
            q_inj: ``(n_bus,)`` baseline reactive injection, same convention.
            load: ``(n_bus,)`` load available for directed curtailment, per
                unit; the right-hand side of (SHED).

        Returns:
            A dict carrying ``award`` ``(n_agent,)`` and ``shed`` ``(n_bus,)``
            in per unit, the objective value ``z`` in dollars, the solver
            diagnostics ``mu``, ``dual_residual`` and ``primal_residual``, and
            the whole result of `requirement.publish` on the baseline point
            (``v_sq``, ``flow``, the two requirements and their four
            aggregates).

            No price is returned and none is formed: this is a pay-as-bid
            market, so each accepted offer is paid its own submitted price,
            this program decides quantities only, and its duals reach no
            settlement expression.  ``mu`` is what says whether the solve
            converged; nothing else does.
        """
        # (SHED) bounds curtailment by the load present at the bus, and the
        # vendored cases register a net injection per bus, not a gross load:
        # a bus that nets to generation has nothing in the data saying how
        # much load sits underneath it, so the sheddable quantity there is
        # taken as zero rather than invented.  Without this the box at such a
        # bus runs from zero to a negative number and the program is
        # infeasible.
        load = jnp.maximum(load, 0.0)
        signals = publish(p_inj, q_inj)
        v_sq_base = signals["v_sq"][bounded_j]
        flow_base = signals["flow"]

        hi = jnp.maximum(jnp.concatenate([qty_max, load]), OFF_EPS)
        h = jnp.concatenate([
            v_sq_base - v_lo_sq_j,          # (VLO): -dv x <= v_base^2 - v_lo^2
            v_hi_sq_j - v_sq_base,          # (VHI):  dv x <= v_hi^2 - v_base^2
            # (LIM).  The flow is  P_l = flow_base - df x, so the upper rating
            # gives  -df x <= p_max - flow_base  and the lower one gives
            # df x <= p_max + flow_base.  Swapping these two right-hand sides
            # is invisible on any case whose ratings are the 1e6 sentinel,
            # since both rows are then slack by four orders of magnitude.
            p_max_j - flow_base,
            p_max_j + flow_base,
            hi,                             # box upper
            jnp.zeros(n_var),               # box lower, x >= 0
        ])
        c = money * jnp.concatenate([price, jnp.full((n_bus,), voll_j)])

        x0 = 0.25 * hi
        x, _lam, _nu, mu, r1, r3 = solve(c, A_empty, b_empty, G_j, h, x0)

        # drop the phantom quantities the OFF_EPS boxes admit
        award = jnp.where(qty_max > 0.0, x[:n_agent], 0.0)
        shed = jnp.where(load > 0.0, x[n_agent:], 0.0)
        return dict(award=award, shed=shed, z=jnp.dot(c, x),
                    mu=mu, dual_residual=r1, primal_residual=r3, **signals)

    return clear, spec
