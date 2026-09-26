"""numpy reference for the two-settlement rule.

This answers "is the settlement written correctly", which is a different question
from "is the mechanism right".  It runs the **same** arithmetic, so the two can be
compared elementwise.

**Only the settlement is written here.**  The clearing needs no new reference:
this market's operator *is* the day-ahead one at `T = 1`, so
`tests/envs/day_ahead/reference.py` covers it at a different configuration, and a
second copy would be a transcription rather than an independent route.

To be worth anything the settlement reference must not be the implementation with
`jnp` swapped for `np`.  It therefore takes a different route to the same
numbers:

* the implementation forms per-unit price rows by fancy-indexing `lmp.T` with the
  bus vector and sums with `jnp.sum` over an axis; this **loops over units and
  periods** and reads each unit's bus one at a time;
* the implementation aggregates onto agents with `segment_sum`; this accumulates
  into a Python list of per-agent totals;
* the implementation forms the money-balance identity with matrix products
  against `PTDF` and a bus-incidence matrix; this walks the lines and the buses
  and sums the terms one at a time.

What is shared is the arithmetic of §8 itself, and only that.
"""
import numpy as np


def _total_cost(p, a, b, c):
    """Integral of the marginal cost curve (§3.2), one unit, one period.

    `unit_cost_a/b/c` are **marginal**-cost coefficients, not MATPOWER total-cost
    ones, so the total is `(a/3)p^3 + (b/2)p^2 + cp`.  Writing `a p^2 + b p + c`
    gives a finite, plausible number that is wrong by orders of magnitude, and
    the repository documented these fields the other way until 2026-08-05.
    """
    return a / 3.0 * p ** 3 + b / 2.0 * p ** 2 + c * p


def settle(case, award, lmp, commitment, commitment_status, q_da, lmp_da,
           period_hours, unit_to_agent=None):
    """§8's two legs, by loops.  Shapes match the implementation's."""
    award = np.asarray(award, np.float64)
    lmp = np.asarray(lmp, np.float64)
    u = np.asarray(commitment, np.float64)
    u_prev = np.asarray(commitment_status, np.float64)
    q_da = np.asarray(q_da, np.float64)
    lmp_da = np.asarray(lmp_da, np.float64)

    n_units, n_periods = award.shape
    bus = np.asarray(case.unit_node_idx, np.int64)
    a = np.asarray(case.unit_cost_a, np.float64)
    b = np.asarray(case.unit_cost_b, np.float64)
    c = np.asarray(case.unit_cost_c, np.float64)
    nl = np.asarray(case.unit_no_load_cost, np.float64)
    su = np.asarray(case.unit_startup_cost, np.float64)

    if unit_to_agent is None:
        unit_to_agent = np.arange(n_units)
    unit_to_agent = np.asarray(unit_to_agent, np.int64)
    n_agents = int(unit_to_agent.max()) + 1

    rev_da = np.zeros(n_agents)
    rev_rt = np.zeros(n_agents)
    cost = np.zeros(n_agents)
    for i in range(n_units):
        n = int(bus[i])
        g = unit_to_agent[i]
        for t in range(n_periods):
            rev_da[g] += period_hours * lmp_da[t, n] * q_da[i, t]
            rev_rt[g] += period_hours * lmp[t, n] * (award[i, t] - q_da[i, t])
            cost[g] += period_hours * _total_cost(award[i, t], a[i], b[i], c[i])
            cost[g] += period_hours * nl[i] * u[i, t]
            was_on = u_prev[i] if t == 0 else u[i, t - 1]
            cost[g] += su[i] * max(u[i, t] - was_on, 0.0)
    profit = rev_da + rev_rt - cost
    return dict(revenue_da=rev_da, revenue_rt=rev_rt, cost=cost, profit=profit,
                reward=profit)


def money_balance(case, award, lmp, shed, demand_bus, q_da, s_da, d_da,
                  line_dual_up, line_dual_dn, shed_dual, period_hours, cap_scale):
    """Both sides of §8's identity for the real-time leg, by loops."""
    award, lmp = np.asarray(award, np.float64), np.asarray(lmp, np.float64)
    shed, demand_bus = np.asarray(shed, np.float64), np.asarray(demand_bus, np.float64)
    q_da, s_da = np.asarray(q_da, np.float64), np.asarray(s_da, np.float64)
    d_da = np.asarray(d_da, np.float64)
    mu_up = np.asarray(line_dual_up, np.float64)
    mu_dn = np.asarray(line_dual_dn, np.float64)
    rho = np.asarray(shed_dual, np.float64)

    bus = np.asarray(case.unit_node_idx, np.int64)
    PTDF = np.asarray(case.PTDF, np.float64)
    F = np.asarray(case.line_cap, np.float64) * cap_scale
    n_units, n_periods = award.shape
    n_buses, n_lines = int(case.n_nodes), PTDF.shape[0]

    # net injections, one bus at a time
    P = np.zeros((n_periods, n_buses))
    P_da = np.zeros((n_periods, n_buses))
    for t in range(n_periods):
        for i in range(n_units):
            P[t, int(bus[i])] += award[i, t]
            P_da[t, int(bus[i])] += q_da[i, t]
        for n in range(n_buses):
            P[t, n] += shed[t, n] - demand_bus[t, n]
            P_da[t, n] += s_da[t, n] - d_da[t, n]

    left = 0.0
    for t in range(n_periods):
        for i in range(n_units):
            left += period_hours * lmp[t, int(bus[i])] * (award[i, t] - q_da[i, t])
        for n in range(n_buses):
            served = (demand_bus[t, n] - shed[t, n]) - (d_da[t, n] - s_da[t, n])
            left -= period_hours * lmp[t, n] * served

    rent = 0.0
    shed_term = 0.0
    for t in range(n_periods):
        for l in range(n_lines):
            f_da = sum(PTDF[l, n] * P_da[t, n] for n in range(n_buses))
            rent += period_hours * (mu_up[t, l] * (F[l] - f_da)
                                    + mu_dn[t, l] * (F[l] + f_da))
        for n in range(n_buses):
            shed_term += period_hours * rho[t, n] * (P[t, n] - P_da[t, n])
    right = -rent - shed_term
    return dict(left=left, right=right, difference=left - right,
                rent=rent, shed_term=shed_term)
