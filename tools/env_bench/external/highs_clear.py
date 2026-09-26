"""Market 02's per-step clearing LP solved by HiGHS, batched and hot-started.

The LP is the one `powermarketjax.envs.day_ahead.clearing.make_clearing` builds
at T = 1 (real-time market, `n_lookahead` = 1), restated row by row so that the
same bounds -- including the OFF_EPS slack on de-committed segments and on the
shed bound of a zero-demand bus -- go into both solvers:

    min  c'x                      x = [segments (n_units*K), shed (n_buses)]
    s.t. 1'x = sum(d) - sum(must_run)                                  (E)
         line_rhs - F <= M x <= F + line_rhs                          (L)
         -(rdn + stop + must_run - prev) <= S x <= rup + start - must_run + prev
         0 <= x <= hi

What is done to make it fast:
  1. the constraint matrix is constant across steps and lanes, so every lane
     owns one `highspy.Highs` built once; a step only overwrites costs, column
     bounds and row bounds (three vector calls);
  2. hot start: HiGHS keeps the previous step's simplex basis after bound/cost
     changes, so each step starts from the last optimal basis (dual simplex);
  3. batch: all `n_envs` lanes are solved in one call, spread over a thread
     pool (`threads` workers; highspy releases the GIL inside `run()`), each
     Highs instance single-threaded so the pool is the only parallelism;
  4. output off, no per-step model rebuild, no Python loop over rows.

Prices are the solve's own duals: lmp[n] = y_E + sum_l y_L[l] PTDF[l, n] + z_shed[n]
(z only when the shed column sits at its upper bound d_n), i.e. d obj / d d_n.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import highspy
import numpy as np
from scipy import sparse

from powermarketjax.envs.day_ahead.clearing import OFF_EPS, VOLL, segment_costs


class ExtClear:
    def __init__(self, case, n_segments=1, cap_scale=1.0, ramp_scale=1.0,
                 period_hours=0.5, n_lanes=1, threads=1, presolve="off",
                 warm=True, demand_scale=1.0):
        K = n_segments
        pmin = np.asarray(case.unit_p_min, np.float64)
        pmax = np.asarray(case.unit_p_max, np.float64)
        unit_bus = np.asarray(case.unit_node_idx, np.int64)
        width, _ = segment_costs(case, K)
        n_u, n_b = len(pmin), int(case.n_nodes)
        PTDF = np.asarray(case.PTDF, np.float64)
        F = np.asarray(case.line_cap, np.float64) * cap_scale
        node_pd = np.asarray(case.node_pd, np.float64)
        if node_pd.sum() > 1e-6:
            share = node_pd / node_pd.sum()
        else:
            share = np.zeros(n_b)
            np.add.at(share, np.asarray(case.load_node_idx, np.int64),
                      np.asarray(case.load_d_max, np.float64))
            share = share / share.sum()
        seg_bus = np.zeros((n_b, n_u * K))
        seg_bus[np.repeat(unit_bus, K), np.arange(n_u * K)] = 1.0
        M = np.hstack([PTDF @ seg_bus, PTDF])
        S = np.zeros((n_u, n_u * K + n_b))
        S[np.repeat(np.arange(n_u), K), np.arange(n_u * K)] = 1.0
        UB = np.zeros((n_b, n_u)); UB[unit_bus, np.arange(n_u)] = 1.0
        self.nb = nb = n_u * K + n_b
        A = np.vstack([np.ones((1, nb)), M, S])
        self.n_l = M.shape[0]
        self.nrow = A.shape[0]
        self.K, self.n_u, self.n_b = K, n_u, n_b
        self.pmin, self.pmax, self.width, self.share = pmin, pmax, width, share
        self.PTDF, self.F, self.UB = PTDF, F, UB
        self.rup = np.asarray(case.unit_ramp_up, np.float64) * pmax * period_hours * ramp_scale
        self.rdn = np.asarray(case.unit_ramp_down, np.float64) * pmax * period_hours * ramp_scale
        self.demand_scale = demand_scale     # the L2 injection: 1.01 must go red
        self.warm = warm
        csc = sparse.csc_matrix(A)
        self._A = csc
        self.cidx = np.arange(nb, dtype=np.int32)
        self.ridx = np.arange(self.nrow, dtype=np.int32)
        self.lanes = [self._new(presolve) for _ in range(n_lanes)]
        self.pool = ThreadPoolExecutor(threads) if threads > 1 else None
        self.threads = threads
        self.n_solves = 0
        self.simplex_iters = 0

    def _new(self, presolve):
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        h.setOptionValue("presolve", presolve)
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = self.nb, self.nrow
        lp.col_cost_ = np.zeros(self.nb)
        lp.col_lower_ = np.zeros(self.nb)
        lp.col_upper_ = np.ones(self.nb)
        lp.row_lower_ = np.full(self.nrow, -highspy.kHighsInf)
        lp.row_upper_ = np.full(self.nrow, highspy.kHighsInf)
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = self._A.indptr.astype(np.int32)
        lp.a_matrix_.index_ = self._A.indices.astype(np.int32)
        lp.a_matrix_.value_ = self._A.data
        h.passModel(lp)
        return h

    # ---------------------------------------------------------------- one lane
    def data(self, offer, u, demand, p_init):
        """numpy vectors of one lane's LP; offer (n_u, K), u (n_u,), demand scalar."""
        demand = float(demand) * self.demand_scale
        d = self.share * demand
        must = self.pmin * u
        wbox = np.where(u > 0, self.width, OFF_EPS)
        c = np.concatenate([offer.reshape(-1), np.full(self.n_b, VOLL)])
        hi = np.concatenate([np.repeat(wbox, self.K), np.maximum(d, OFF_EPS)])
        b_eq = d.sum() - must.sum()
        line_rhs = self.PTDF @ (d - self.UB @ must)
        u_prev = (p_init > 0.0).astype(np.float64)
        start = np.maximum(u - u_prev, 0.0) * self.pmin
        stop = np.maximum(u_prev - u, 0.0) * self.pmax
        r_hi = self.rup + start - must + p_init
        r_lo = -(self.rdn + stop + must - p_init)
        rl = np.concatenate([[b_eq], line_rhs - self.F, r_lo])
        ru = np.concatenate([[b_eq], line_rhs + self.F, r_hi])
        return c, hi, rl, ru, d, must

    def solve_lane(self, i, offer, u, demand, p_init):
        h = self.lanes[i]
        c, hi, rl, ru, d, must = self.data(offer, u, demand, p_init)
        if not self.warm:
            h.clearSolver()
        h.changeColsCost(self.nb, self.cidx, c)
        h.changeColsBounds(self.nb, self.cidx, np.zeros(self.nb), hi)
        h.changeRowsBounds(self.nrow, self.ridx, rl, ru)
        h.run()
        ok = h.getModelStatus() == highspy.HighsModelStatus.kOptimal
        sol = h.getSolution()
        x = np.asarray(sol.col_value)
        y = np.asarray(sol.row_dual)
        z = np.asarray(sol.col_dual)
        n_seg = self.n_u * self.K
        at_up = np.asarray([s == highspy.HighsBasisStatus.kUpper
                            for s in h.getBasis().col_status[n_seg:]])
        lmp = y[0] + y[1:1 + self.n_l] @ self.PTDF + np.where(at_up, z[n_seg:], 0.0)
        award = (must + x[:n_seg].reshape(self.n_u, self.K).sum(1)) * u
        shed = np.where(d > 0.0, x[n_seg:], 0.0)
        it = h.getInfo().simplex_iteration_count
        return award, lmp, shed, float(c @ x), ok, it

    # ---------------------------------------------------------------- batched
    def solve_batch(self, offer, u, demand, p_init):
        """offer (B, n_u, K), u (B, n_u), demand (B,), p_init (B, n_u)."""
        B = offer.shape[0]
        f = lambda i: self.solve_lane(i, offer[i], u[i], demand[i], p_init[i])
        res = list(self.pool.map(f, range(B))) if self.pool else [f(i) for i in range(B)]
        self.n_solves += B
        self.simplex_iters += sum(r[5] for r in res)
        award = np.stack([r[0] for r in res]); lmp = np.stack([r[1] for r in res])
        shed = np.stack([r[2] for r in res]); obj = np.array([r[3] for r in res])
        ok = np.array([r[4] for r in res])
        return award, lmp, shed, obj, ok

    # ---------------------------------------------------------------- jax side
    def jax_clear(self):
        """A drop-in for `make_rt_clearing`'s `clear` at T = 1, via pure_callback.

        `vmap_method="broadcast_all"` hands the callback the whole batch at once,
        which is what lets `solve_batch` spread the lanes over the pool.
        """
        import jax
        import jax.numpy as jnp
        n_u, n_b = self.n_u, self.n_b

        def host(offer, u, demand, p_init):
            offer = np.asarray(offer); lead = offer.shape[:-3]
            B = int(np.prod(lead)) if lead else 1
            o = offer.reshape(B, n_u, self.K)
            uu = np.asarray(u).reshape(B, n_u)
            dd = np.broadcast_to(np.asarray(demand), lead + (1,)).reshape(B)
            pp = np.asarray(p_init).reshape(B, n_u)
            award, lmp, shed, obj, ok = self.solve_batch(o, uu, dd, pp)
            #: the consumption control (after rollout_bench --perturb-ulp): move
            #: every returned award and price by `perturb_ulp` float64 ULP, or by a
            #: relative `perturb_rel`.  If the chain consumes this callback's
            #: output, the fingerprint has to move.
            pu, pr = getattr(self, "perturb_ulp", 0), getattr(self, "perturb_rel", 0.0)
            if pu:
                award = award + pu * np.spacing(award); lmp = lmp + pu * np.spacing(lmp)
            if pr:
                award = award * (1.0 + pr); lmp = lmp * (1.0 + pr)
            mu = np.where(ok, 0.0, np.inf)
            return (award.reshape(lead + (n_u, 1)), lmp.reshape(lead + (1, n_b)),
                    shed.reshape(lead + (1, n_b)), mu.reshape(lead),
                    mu.reshape(lead).copy())

        def clear(offer, u, demand, p_init):
            lead = ()
            shapes = (jax.ShapeDtypeStruct((n_u, 1), jnp.float64),
                      jax.ShapeDtypeStruct((1, n_b), jnp.float64),
                      jax.ShapeDtypeStruct((1, n_b), jnp.float64),
                      jax.ShapeDtypeStruct((), jnp.float64),
                      jax.ShapeDtypeStruct((), jnp.float64))
            a, l, s, mu, r = jax.pure_callback(host, shapes, offer, u, demand, p_init,
                                               vmap_method="broadcast_all")
            return dict(award=a, lmp=l, shed=s, mu=mu, dual_residual=r)
        return clear


# ---------------------------------------------------------------- process pool
def _worker(conn, case, args, lanes, core):
    """One process: owns `lanes` Highs instances, pinned to one core."""
    import os
    if core is not None:
        os.sched_setaffinity(0, {core})
    X = ExtClear(case, *args, n_lanes=len(lanes), threads=1)
    while True:
        msg = conn.recv()
        if msg is None:
            break
        offer, u, demand, p_init = msg
        res = [X.solve_lane(i, offer[i], u[i], demand[i], p_init[i]) for i in range(len(lanes))]
        conn.send(res)


class ExtClearProc(ExtClear):
    """ExtClear whose batch is spread over `procs` persistent processes.

    The thread pool does not scale: highspy's per-call Python work holds the
    GIL (measured on 4 cores: 0.316 ms/LP at 1 thread, 0.190 at 4, i.e. 1.7x),
    and the 32-core cell read cores_used 2.2.  Each worker owns a fixed slice
    of the lanes (so each lane's hot-start basis stays in one process), is
    pinned to one core of the affinity, and is fed raw per-lane inputs; the LP
    vectors are built inside the worker.
    """

    def __init__(self, case, n_segments=1, cap_scale=1.0, ramp_scale=1.0,
                 period_hours=0.5, n_lanes=1, procs=1, **_ignored):
        import multiprocessing as mp, os
        super().__init__(case, n_segments, cap_scale, ramp_scale, period_hours,
                         n_lanes=0, threads=1)
        #: workers import jax via the clearing module; keep them off the card
        prev = os.environ.get("JAX_PLATFORMS")
        os.environ["JAX_PLATFORMS"] = "cpu"
        ctx = mp.get_context("spawn")
        cores = sorted(os.sched_getaffinity(0))
        self.slices = [s for s in np.array_split(np.arange(n_lanes), procs) if len(s)]
        self.conns, self.procs = [], []
        for k, sl in enumerate(self.slices):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, daemon=True,
                            args=(b, case, (n_segments, cap_scale, ramp_scale, period_hours),
                                  list(sl), cores[k % len(cores)]))
            p.start()
            self.conns.append(a); self.procs.append(p)
        if prev is None:
            del os.environ["JAX_PLATFORMS"]
        else:
            os.environ["JAX_PLATFORMS"] = prev
        self.threads = len(self.slices)

    def solve_batch(self, offer, u, demand, p_init):
        for c, sl in zip(self.conns, self.slices):
            c.send((offer[sl], u[sl], demand[sl], p_init[sl]))
        res = []
        for c in self.conns:
            res.extend(c.recv())
        self.n_solves += len(res)
        self.simplex_iters += sum(r[5] for r in res)
        award = np.stack([r[0] for r in res]); lmp = np.stack([r[1] for r in res])
        shed = np.stack([r[2] for r in res]); obj = np.array([r[3] for r in res])
        ok = np.array([r[4] for r in res])
        return award, lmp, shed, obj, ok
