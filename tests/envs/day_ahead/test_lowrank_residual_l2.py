"""L2 on the linear-solve layer of the low-rank route (review, 2026-09-16):
the Newton directions it returns must satisfy the UNREDUCED Newton system to
the accuracy the dense route reaches, on real T = 24 iterates.

Why a separate test: `test_lowrank_l2.py` compares the two routes' final
answers to the L2 tolerance, and that comparison is blind to the
solver's own residual as long as the interior point iteration lands on the
same optimum.  Measured 2026-09-16 (CPU): on the pilot's own 813nem T = 24 iterates the
shipped ``N_REFINE = 3`` left a worst SCED residual of 5.9e-9 against the
dense route's 1.9e-10 on the same steps (31x; 6.8e-9 over 36 days), while
every end-to-end number still agreed to tolerance.  ``N_REFINE = 4`` brought
the SCED to 3.0e-10 (1.6x); the relaxation floors at ~1.2e-9 from three steps
on (dense 2.1e-10), a floor of the Schur route itself.

Bounds, derived from those measurements: the SCED's worst residual over the
60 recorded steps must sit below **1e-9** -- three times the 3.0e-10 floor at
four refinement steps and six times below the 5.9e-9 three steps leave, so
losing one step or the refinement altogether turns it red; the relaxation's
must sit below **`MU_TOL`** (1e-8), the level `env` reads convergence at, and
its ratio to the dense route is printed, not asserted, because no refinement
count moves it.  The T = 2 configuration of `test_lowrank_l2.py` does not
reproduce the gap (ratio 2.7 at three steps) and is not used here.

Device: every Newton step's ``(D, reg, rhs)`` is recorded through a callback
wrapped around the production ``factor`` / ``apply`` pair, with the LP
assembled by `make_clearing` / `make_relax` themselves, then each step is
re-solved offline by the low-rank route and its residual ``|K z - rhs|_inf /
|rhs|_inf`` formed matrix-free with the operators the solver uses; the dense
route re-solves the three steps with the worst low-rank residual, for the
printed ratio.  813nem T = 24 on the 7 rated lines, honest offers, u = ones.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from powermarketjax.case import load_case
from powermarketjax.envs.day_ahead import MU_TOL, kkt_lowrank, segment_costs
from powermarketjax.envs.day_ahead import clearing as clearing_mod
from powermarketjax.envs.day_ahead import kkt as kkt_mod
from powermarketjax.envs.day_ahead import relax as relax_mod
from powermarketjax.envs.day_ahead import relax_kkt
from powermarketjax.solvers import ipm

from .test_lowrank_l2 import CASE_813NEM_LINES, PC, x64  # noqa: F401

#: Worst relative residual over the 60 SCED steps; see the module docstring.
SCED_RESIDUAL_MAX = 1e-9
#: Dense re-solves per operator, on the steps with the worst low-rank residual.
N_DENSE_STEPS = 3
T = 24


def _record(monkeypatch, build, call):
    """Build an operator with a recording solver, run it once, return the
    captured ``(spec, ops, lowrank pair, dense pair, steps)``."""
    cap, rec = {}, []
    orig_solver = ipm.make_solver

    def rec_solver(n, m, max_iter, *, kkt=None, **kw):
        factor, apply = kkt

        def fr(D, reg):
            jax.debug.callback(lambda D, r: rec.append(("D", np.asarray(D).copy(), float(r))),
                               D, reg, ordered=True)
            return factor(D, reg)

        def ar(st, rhs):
            jax.debug.callback(lambda r: rec.append(("rhs", np.asarray(r).copy())), rhs, ordered=True)
            return apply(st, rhs)
        return orig_solver(n, m, max_iter, kkt=(fr, ar), **kw)

    def wrap(mod, name, key):
        fn = getattr(mod, name)

        def w(*args, **kw):
            out = fn(*args, **kw)
            cap[key] = (args, out)
            return out
        monkeypatch.setattr(mod, name, w)

    monkeypatch.setattr(ipm, "make_solver", rec_solver)
    wrap(kkt_lowrank, "make_kkt_sced", "lr"); wrap(kkt_lowrank, "make_kkt_relax", "lr")
    wrap(kkt_mod, "make_ops", "ops"); wrap(relax_kkt, "make_ops", "ops")
    op, spec = build()
    out = jax.jit(op)(*call)
    jax.block_until_ready(out)
    assert float(out["mu"]) < 1e-8
    steps = [(rec[i][1], rec[i][2], rec[i + 1][1], rec[i + 2][1]) for i in range(0, len(rec), 3)]
    assert len(steps) == clearing_mod.MAX_ITER or len(steps) == relax_mod.MAX_ITER
    return spec, cap, steps


def _residuals(spec, cap, steps, a_row, dense_pair):
    T, nb, n = spec["T"], spec["nb"], spec["n"]
    Gx, GTy = cap["ops"][1]
    A = np.zeros((T, n))
    for t in range(T):
        A[t, t * nb:(t + 1) * nb] = a_row
    A = jnp.asarray(A)

    @jax.jit
    def resid(D, reg, z, rhs):
        x, nu = z[:n], z[n:]
        Kz = jnp.concatenate([GTy(D * Gx(x)) + reg * x + A.T @ nu, A @ x])
        return jnp.abs(Kz - rhs).max() / jnp.abs(rhs).max()

    def solve_all(pair, idx):
        factor, apply = (jax.jit(f) for f in pair)
        rs = {}
        for i in idx:
            D, reg, ra, rc = steps[i]
            D, ra, rc = jnp.asarray(D), jnp.asarray(ra), jnp.asarray(rc)
            st = factor(D, reg)
            rs[i] = max(float(resid(D, reg, apply(st, ra), ra)), float(resid(D, reg, apply(st, rc), rc)))
        return rs
    lr = solve_all(cap["lr"][1], range(len(steps)))
    worst = sorted(lr, key=lr.get, reverse=True)[:N_DENSE_STEPS]
    dn = solve_all(dense_pair, worst)
    return dict(lowrank=np.asarray([lr[i] for i in range(len(steps))]), worst=worst,
                dense_on_worst=np.asarray([dn[i] for i in worst]),
                lowrank_on_worst=np.asarray([lr[i] for i in worst]))


def _report(res, label):
    lr = res["lowrank"].max()
    print(f"\n{label}: low-rank residual max {lr:.2e} (p50 {np.median(res['lowrank']):.1e}) at steps "
          f"{res['worst']}; dense on those steps {[f'{v:.1e}' for v in res['dense_on_worst']]}; "
          f"ratio on the worst step {res['lowrank_on_worst'][0] / res['dense_on_worst'][0]:.1f}; "
          f"N_REFINE={kkt_lowrank.N_REFINE}")
    return lr


def _inputs():
    case = load_case("813nem")
    pmax = np.asarray(case.unit_p_max, np.float64)
    pmin = np.asarray(case.unit_p_min, np.float64)
    demand = np.full(T, 0.5 * pmax.sum())
    _, cost = segment_costs(case, 1)
    offer = np.repeat(cost[:, :, None], T, axis=2)
    return case, pmin, demand, offer


def test_sced_directions_satisfy_the_unreduced_system(monkeypatch):
    case, pmin, demand, offer = _inputs()
    u = np.ones((len(pmin), T))
    build = lambda: clearing_mod.make_clearing(case, T, n_segments=1, cap_scale=1.0,
                                               ramp_scale=1.0, monitored_lines=CASE_813NEM_LINES)
    call = (jnp.asarray(offer), jnp.asarray(u), jnp.asarray(demand), jnp.asarray(pmin))
    spec, cap, steps = _record(monkeypatch, build, call)
    (_, Mj, Sj), _ = cap["ops"]
    res = _residuals(spec, cap, steps, np.ones(spec["nb"]), kkt_mod.make_kkt(spec, Mj, Sj))
    lr = _report(res, "813nem T=24 SCED, 7 rated lines")
    assert lr <= SCED_RESIDUAL_MAX, (
        f"SCED low-rank residual {lr:.2e} above {SCED_RESIDUAL_MAX:.0e} "
        f"(N_REFINE={kkt_lowrank.N_REFINE}; four steps measured 3.0e-10, three 5.9e-9)")


def test_relax_directions_satisfy_the_unreduced_system(monkeypatch):
    case, _, demand, offer = _inputs()
    boundary = PC.first_boundary(case, float(demand[0]), cap_scale=1.0, ramp_scale=1.0, K=1, delta_h=1.0)
    build = lambda: relax_mod.make_relax(case, T, n_segments=1, cap_scale=1.0, ramp_scale=1.0,
                                         monitored_lines=CASE_813NEM_LINES)
    call = (jnp.asarray(offer), jnp.asarray(demand), jnp.asarray(boundary["p_init"]),
            jnp.asarray(boundary["u_prev"]), jnp.asarray(boundary["up_time"].astype(float)),
            jnp.asarray(boundary["down_time"].astype(float)))
    spec, cap, steps = _record(monkeypatch, build, call)
    (_, ops_j), _ = cap["ops"]
    (_, ops_np), _ = cap["lr"]
    res = _residuals(spec, cap, steps, np.asarray(ops_np["a"]), relax_kkt.make_kkt(spec, ops_j))
    lr = _report(res, "813nem T=24 relax, 7 rated lines")
    assert lr <= MU_TOL, f"relax low-rank residual {lr:.2e} above MU_TOL {MU_TOL:.0e}"
