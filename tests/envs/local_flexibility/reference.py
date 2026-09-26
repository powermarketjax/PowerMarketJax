"""numpy reference for the local flexibility market.

This answers "is the implementation written correctly", which is a different
question from "is the mechanism right".  It runs the **same** mathematics, so
the two can be compared elementwise.

To be worth anything it must not be a transcription of the implementation with
one array library swapped for another.  What the two share is the spanning
tree, and they must: on a connected graph with one fewer line than buses the
tree is unique, so there is no second route to it and an error there is caught
by the connectivity and count checks rather than by this comparison.  What the
two do differently is everything after the tree:

* the implementation walks **up** from every bus to the substation to fill a
  path indicator matrix, then obtains `R` and `X` as one product
  ``P diag(r) P'``, so the shared path is never formed explicitly and falls out
  of the product;
* this reference carries the path set **down** from the substation as it
  descends, then intersects the two sets of every bus pair and sums the shared
  resistance directly, with no matrix product anywhere.

The two agree only if the shared-path set is right, which is the error §16
singles out: a wrong shared-path set leaves the clearing problem feasible and
every reported quantity finite.  Replacing the shared path by the smaller of
the two path resistances, which is right on a chain and wrong wherever the
feeder branches, was measured to fail this comparison on all six distribution
cases (2026-08-09).

The clearing assembly and the settlement are covered the same way: the
implementation stacks vectorised blocks and this writes the rows one at a time
out of §6, the implementation vectorises §8 and this loops over participants.

**The solve itself is not covered, deliberately.**  A reference is worth
comparing against only when it reaches the same mathematics by a different
route, and the day-ahead reference achieves that by assembling row by row, by
forming `G` explicitly and by factoring densely.  The last two are unavailable
here, since §15 puts this market on the dense default path, so a solver
reference would differ from the implementation in nothing but the array
library.  The Newton loop is therefore taken from `tests/solvers/reference_ipm.py`
rather than rewritten: one numpy solver for both markets, covered by the
day-ahead L2 against the same `ipm.py`.  It
used to be imported from the day-ahead reference, which is where the trip-count
mismatch recorded in `clear` came from.

Kept in `tests/` because it exists only to be compared against.
"""
import numpy as np

from powermarketjax.envs.local_flexibility.clearing import MAX_ITER
from tests.solvers.reference_ipm import solve_ipm


def paths_by_descent(case):
    """Path from the substation to each bus, as a set of in-service line ids.

    Depth-first from the root, carrying the path down.  Returns
    ``(paths, r, x, line_index)`` where ``paths[n]`` is a set of positions into
    the in-service arrays.
    """
    n_bus = int(case.n_nodes)
    slack = int(case.slack_bus_idx)
    r = np.asarray(case.line_r, np.float64)
    x = np.asarray(case.line_x, np.float64)
    frm = np.asarray(case.line_from_idx, np.int64)
    to = np.asarray(case.line_to_idx, np.int64)
    line_index = np.arange(len(frm), dtype=np.int64)
    if case.line_status is not None:
        live = np.asarray(case.line_status) > 0
        r, x, frm, to, line_index = r[live], x[live], frm[live], to[live], line_index[live]

    incident: list = [[] for _ in range(n_bus)]
    for line, (a, b) in enumerate(zip(frm, to)):
        incident[a].append((b, line))
        incident[b].append((a, line))

    paths: list = [None] * n_bus
    paths[slack] = frozenset()
    stack = [slack]
    while stack:
        node = stack.pop()
        for neighbour, line in incident[node]:
            if paths[neighbour] is None:
                paths[neighbour] = paths[node] | {line}
                stack.append(neighbour)

    if any(p is None for p in paths):
        unreached = [n for n, p in enumerate(paths) if p is None]
        raise ValueError(f"buses unreachable from the substation: {unreached}")
    return paths, r, x, line_index


def sensitivity_by_intersection(case, pairs=None):
    """``R`` and ``X`` by intersecting the two paths of every bus pair.

    Args:
        case: a `CaseData`.
        pairs: optional iterable of ``(n, m)`` to evaluate instead of the full
            matrix.  The 533-bus cases have 284 089 pairs, and a subsample is
            enough to compare against once the small cases have been compared
            in full.

    Returns:
        ``(R, X)`` full matrices when ``pairs`` is None, otherwise two 1-D
        arrays holding one entry per requested pair.
    """
    paths, r, x, _ = paths_by_descent(case)

    if pairs is not None:
        pairs = list(pairs)
        R = np.empty(len(pairs), np.float64)
        X = np.empty(len(pairs), np.float64)
        for k, (n, m) in enumerate(pairs):
            shared = sorted(paths[n] & paths[m])
            R[k] = r[shared].sum()
            X[k] = x[shared].sum()
        return R, X

    n_bus = int(case.n_nodes)
    R = np.zeros((n_bus, n_bus), np.float64)
    X = np.zeros((n_bus, n_bus), np.float64)
    for n in range(n_bus):
        for m in range(n, n_bus):
            shared = sorted(paths[n] & paths[m])
            R[n, m] = R[m, n] = r[shared].sum()
            X[n, m] = X[m, n] = x[shared].sum()
    return R, X


def downstream_by_descent(case):
    """``A[l, m] = 1`` when bus ``m`` lies downstream of line ``l``.

    Read straight off the path sets: bus ``m`` is downstream of line ``l``
    exactly when ``l`` lies on the path to ``m``.
    """
    paths, r, _, _ = paths_by_descent(case)
    A = np.zeros((len(r), int(case.n_nodes)), np.float64)
    for bus, path in enumerate(paths):
        for line in path:
            A[line, bus] = 1.0
    return A


# ---------------------------------------------------------------------------
# The clearing problem of §6 and the settlement of §8
# ---------------------------------------------------------------------------

def build_lp(case, sens, agent_bus, price, qty_max, p_inj, q_inj,
             load, voltage_margin=0.0, thermal_margin=0.0,
             period_hours=0.25, voll=10_000.0, off_eps=1e-9):
    """Assemble §6 explicitly, **row by row**, as a canonical linear program.

    The implementation forms the constraint blocks as vectorised products
    against the sensitivity matrices and stacks four of them.  This walks the
    buses and the lines in Python and writes one row at a time, reading each
    coefficient out of §6 rather than out of a block expression.  The variable
    order is the implementation's, ``[q ; s]``, because the comparison is
    elementwise on ``x``.

    Everything else is shared deliberately: the same `off_eps`, the same
    starting point rule, the same Mehrotra loop.  An L2 reference is one
    algorithm reached by two routes, not two algorithms.
    """
    n_agent, n_bus, n_line = len(agent_bus), sens.n_bus, sens.n_line
    n = n_agent + n_bus
    slack = sens.slack

    pd = np.asarray(case.node_pd, np.float64)
    qd = np.asarray(case.node_qd, np.float64)
    phi = np.where(pd > 0.0, qd / np.where(pd > 0.0, pd, 1.0), 0.0)

    v_lo = np.asarray(case.node_v_min, np.float64) + voltage_margin
    v_hi = np.asarray(case.node_v_max, np.float64) - voltage_margin
    p_max = sens.p_max * (1.0 - thermal_margin)

    v_sq = 1.0 + 2.0 * (sens.R @ p_inj + sens.X @ q_inj)
    flow = -sens.A @ p_inj

    rows, rhs = [], []
    for bus in range(n_bus):
        if bus == slack:                       # §3.1: not a constrained bus
            continue
        coefficient = np.zeros(n)
        for k, agent in enumerate(agent_bus):
            coefficient[k] = 2.0 * sens.R[bus, agent]
        for other in range(n_bus):
            coefficient[n_agent + other] = 2.0 * (sens.R[bus, other]
                                                  + phi[other] * sens.X[bus, other])
        rows.append(-coefficient)              # (VLO)
        rhs.append(v_sq[bus] - v_lo[bus] ** 2)
        rows.append(coefficient)               # (VHI)
        rhs.append(v_hi[bus] ** 2 - v_sq[bus])

    for line in range(n_line):
        coefficient = np.zeros(n)
        for k, agent in enumerate(agent_bus):
            coefficient[k] = sens.A[line, agent]
        for other in range(n_bus):
            coefficient[n_agent + other] = sens.A[line, other]
        rows.append(-coefficient)              # (LIM), upper rating
        rhs.append(p_max[line] - flow[line])
        rows.append(coefficient)               # (LIM), lower rating
        rhs.append(p_max[line] + flow[line])

    # the row order above is bus-major and the implementation's is block-major;
    # they describe the same polyhedron, and only `x` is compared
    hi = np.maximum(np.concatenate([qty_max, np.maximum(load, 0.0)]), off_eps)
    return dict(
        n=n,
        c=period_hours * sens.base_mva * np.concatenate(
            [price, np.full(n_bus, voll)]),
        A_eq=np.zeros((0, n)), b_eq=np.zeros(0),
        A_ub=np.array(rows), b_ub=np.array(rhs),
        lo=np.zeros(n), hi=hi,
    )


def clear(case, sens, agent_bus, price, qty_max, p_inj, q_inj, load, max_iter=MAX_ITER,
          **kw):
    """Assemble and solve, on the shared numpy Mehrotra of `tests/solvers`.

    `max_iter` defaults to **this market's** calibrated 80, matching the operator
    under comparison.  Until the shared solver made the argument mandatory this
    call omitted it and inherited the day-ahead 120, so the reference ran a
    different trip count from the implementation it was certifying.
    """
    lp = build_lp(case, sens, agent_bus, price, qty_max, p_inj, q_inj, load, **kw)
    x, lam, nu, mu, r1, r3 = solve_ipm(lp, 0.25 * lp["hi"], max_iter,
                                       dual_start="unit")

    n_agent = len(agent_bus)
    award = np.where(np.asarray(qty_max) > 0.0, x[:n_agent], 0.0)
    shed = np.where(np.maximum(load, 0.0) > 0.0, x[n_agent:], 0.0)
    return dict(award=award, shed=shed, z=float(lp["c"] @ x), mu=mu,
                dual_residual=r1, primal_residual=r3)


def settle(sens, price, award, plan, energy_price, cycle_cost,
           period_hours=0.25):
    """§8 written out term by term, against the implementation's vector form."""
    money = period_hours * sens.base_mva
    n = len(award)
    revenue = np.empty(n)
    cost = np.empty(n)
    p_ch = np.empty(n)
    p_dis = np.empty(n)
    for i in range(n):
        p_dis[i] = max(award[i] - plan[i], 0.0)
        p_ch[i] = max(plan[i] - award[i], 0.0)
        revenue[i] = money * price[i] * award[i]
        cost[i] = money * (energy_price * p_ch[i]
                           + cycle_cost[i] * (p_ch[i] + p_dis[i]))
    return dict(revenue=revenue, cost=cost, profit=revenue - cost,
                reward=revenue - cost, p_ch=p_ch, p_dis=p_dis)
