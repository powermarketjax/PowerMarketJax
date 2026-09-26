"""Market 02's per-step LP written the way a cvxpy user writes it by default.

Every call builds a fresh `cp.Variable`, the constraint list and a
`cp.Problem`, then `Problem.solve()` with **no arguments**: cvxpy picks the
solver, no `cp.Parameter` / DPP, no `warm_start`, no solver options.  That is
the default idiom; each of those omissions is a default, not a handicap chosen
here.  The LP data come from `ExtClear.data`, so the LP is identical to the
HiGHS arm's (and L2 against the market's `clear` is run on this class too).

Per call we keep: the wall time of `solve()`, `solver_stats.solve_time`
(the solver's own clock), `problem.compilation_time` (cvxpy's own clock of
canonicalization), and `solver_stats.solver_name`.
"""
import time

import cvxpy as cp
import numpy as np

from highs_clear import ExtClear


class CvxClear(ExtClear):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.M = self._A.toarray()[1:1 + self.n_l]
        self.S = self._A.toarray()[1 + self.n_l:]
        self.log = []                          # (solve_wall, solve_time, compilation_time, name)

    def solve_lane(self, i, offer, u, demand, p_init):
        c, hi, rl, ru, d, must = self.data(offer, u, demand, p_init)
        nl = self.n_l
        x = cp.Variable(self.nb)
        cons = [cp.sum(x) == rl[0],
                self.M @ x <= ru[1:1 + nl], self.M @ x >= rl[1:1 + nl],
                self.S @ x <= ru[1 + nl:], self.S @ x >= rl[1 + nl:],
                x >= 0, x <= hi]
        prob = cp.Problem(cp.Minimize(c @ x), cons)
        t = time.perf_counter()
        prob.solve()
        wall = time.perf_counter() - t
        st = prob.solver_stats
        self.log.append((wall, st.solve_time, getattr(prob, "compilation_time", None),
                         st.solver_name))
        ok = prob.status == cp.OPTIMAL
        xv = np.asarray(x.value)
        n_seg = self.n_u * self.K
        #: cvxpy's duals are >= 0 on `<=` / `>=` rows; d obj / d rhs is
        #: -dual on `expr <= rhs`, +dual on `expr >= rhs`, -dual on `==`
        y_eq = -float(cons[0].dual_value)
        y_line = -np.asarray(cons[1].dual_value) + np.asarray(cons[2].dual_value)
        z_up = -np.asarray(cons[6].dual_value)[n_seg:]
        lmp = y_eq + y_line @ self.PTDF + z_up
        award = (must + xv[:n_seg].reshape(self.n_u, self.K).sum(1)) * u
        shed = np.where(d > 0.0, xv[n_seg:], 0.0)
        return award, lmp, shed, float(c @ xv), ok, 0
