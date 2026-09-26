"""numpy reference for `powermarketjax.solvers.ipm` (the L2 numerical-equivalence layer).

Mehrotra predictor-corrector, dense, in plain numpy.  It runs the **same**
algorithm as the implementation, which is what lets the two be compared
elementwise; what differs is the route to the Newton system, and that difference
is supplied by the caller rather than by this file:

* the LP arrives as an already-assembled dict, so each market's reference is
  free to assemble it the long way (row by row in a Python loop) where the
  implementation vectorises;
* ``G`` is materialised here as a real matrix, with the bounds folded in
  explicitly, where the day-ahead implementation never forms it;
* the Newton system is factored densely, not by the block tridiagonal sweep.

It lives outside `tests/envs/` for the same reason the implementation does:
before this file existed, `tests/envs/local_flexibility/reference.py` imported
its Newton loop from `tests/envs/day_ahead/reference.py`, and the real-time and
ancillary references would have added two more such reaches.

**`max_iter` is required and has no default.**  That is not tidiness.  While the
default was the day-ahead constant, the local flexibility reference called this
without one and so ran 120 Newton steps against an implementation running 80,
which quietly contradicted the "same fixed trip count" that the L2 comparison
rests on.  Both counts sit on that market's converged plateau, so no L2 test was ever
wrong -- but nothing in the apparatus would have said so.  Each market's
reference now passes its own market's calibration.

`x0` is required for the same reason: the starting point is market-specific
(the day-ahead one is exact on every period's balance; the local flexibility
market has no equality row and its interior is the box alone), and a default
here would silently be one market's.
"""
import warnings

import numpy as np

from powermarketjax.solvers.ipm import REG_COEF, TAU


def solve_ipm(lp, x0, max_iter, dual_start, reg_coef=REG_COEF, tau=TAU,
              freeze_mu=None):
    """Mehrotra predictor-corrector, dense.  Same algorithm as the implementation.

    Args:
        lp: dict with ``c``, ``A_eq``, ``b_eq``, ``A_ub``, ``b_ub``, ``lo``,
            ``hi``, ``n``.  A market with no equality row passes ``A_eq`` with
            zero rows rather than a row of zeros.
        x0: strictly interior starting point; market-specific, see the module
            docstring.
        max_iter: Newton steps; the calling market's own calibration.
        dual_start: mirrors `ipm.DUAL_STARTS`, and must match what the market's
            implementation was calibrated with -- the trip count depends on it,
            so a reference running one start against an implementation running
            the other compares two different calibrations and the L2
            "same fixed trip count" no longer holds.
        freeze_mu: mirrors `ipm.make_solver`'s merit gate (``None`` = off):
            once ``mu`` is below it a step that would raise
            ``max(mu, |r1|_inf / kappa, |r3|_inf / (1 + |h|_inf))`` is refused.
            A reference compared against a frozen implementation must run the
            same gate, for the reason `max_iter` is required.

    Returns:
        ``(x, lam, nu, mu, r1_inf, r3_inf)``, matching `ipm.make_solver`.
        The primal residual is carried here for the same reason as the
        dual one: L2 compares two solvers each complete in itself, and a
        reference missing a field would have to borrow it from the
        implementation it is checking.
    """
    c, A, b = lp["c"], lp["A_eq"], lp["b_eq"]
    n, n_eq = lp["n"], lp["A_eq"].shape[0]
    # fold bounds into G explicitly: the implementation never materialises this
    G = np.vstack([lp["A_ub"], np.eye(n), -np.eye(n)])
    h = np.concatenate([lp["b_ub"], lp["hi"], -lp["lo"]])
    m = G.shape[0]

    x = np.asarray(x0, np.float64)
    s = np.maximum(h - G @ x, 1.0)
    lam = (1.0 if dual_start == "unit" else max(1.0, float(np.abs(c).max()))) / s
    nu = np.zeros(n_eq)
    merit_scale = max(1.0, float(np.abs(c).max()))
    hscale = 1.0 + float(np.abs(h).max())

    def merit(x_, s_, lam_, nu_):
        return max(float(s_ @ lam_) / m,
                   float(np.abs(c + A.T @ nu_ + G.T @ lam_).max()) / merit_scale,
                   float(np.abs(G @ x_ + s_ - h).max()) / hscale)

    def step_len(v, dv):
        neg = dv < 0
        if not neg.any():
            return 1.0
        return min(1.0, tau * np.min(-v[neg] / dv[neg]))

    for _ in range(max_iter):
        mu = float(s @ lam) / m
        D = lam / s
        reg = max(1e-12, reg_coef * D.mean())

        r1 = c + A.T @ nu + G.T @ lam
        r2 = A @ x - b
        r3 = G @ x + s - h

        H = (G * D[:, None]).T @ G + reg * np.eye(n)
        K = np.block([[H, A.T], [A, np.zeros((n_eq, n_eq))]])
        sc = np.sqrt(np.maximum(np.abs(np.diag(K)), 1e-12))
        Ks = (K / sc[:, None]) / sc[None, :]

        def solve_kkt(rhs):
            #: **The exact-zero pivot is a property of this loop, not of the
            #: market.**  There is no convergence exit here -- the
            #: implementation cannot have one (`lax.fori_loop` needs a static
            #: trip count) and this reference deliberately mirrors it, so both
            #: keep stepping long after they have converged.  Past that point
            #: `D = lam / s` is pushed to both extremes by the `1e-14` clamps
            #: below, `reg = max(1e-12, REG_COEF * D.mean())` stays pinned at
            #: its floor (`REG_COEF` is 1e-20), and `H = G' D G + reg I` goes
            #: numerically rank deficient.
            #:
            #: **Only one side of the L2 comparison is allowed to complain
            #: about that, and that asymmetry is the defect.**  The
            #: implementation reaches the same regime through
            #: `jax.scipy.linalg.lu_factor`, which cannot raise under `jit`: it
            #: returns whatever a singular factorisation gives and carries on.
            #: `np.linalg.solve` raises the moment LAPACK meets a pivot that is
            #: exactly 0.0, and whether a pivot lands on 0.0 rather than 1e-300
            #: is decided by the last bits of `G' D G` -- i.e. by the order
            #: BLAS happened to sum in, i.e. by the host's thread count.
            #: Measured 2026-09-09 on the ancillary `TIED_CASE`: 8 of 16 thread
            #: counts raised here (1, 2, 6, 8, 12, 16, 24, 28) while 3, 4, 20,
            #: 32, 48, 64, 96 and 128 solved, with the reading bit-identical
            #: within each count.  A reference that dies where the
            #: implementation continues is not comparing like with like.
            #:
            #: Measured again 2026-09-13 on the same case and host, counting the
            #: raw fallbacks instead of pytest's once-per-location warning
            #: summary (which counts test items, not calls, and so reads `2`
            #: whatever the real number is): the fallback fires **2 to 16 times
            #: per `solve_ipm` call** across those eight counts -- 16 at 1 and
            #: 16 cores, 14 at 28, 6 at 2, 4 at 6 and 8, 2 at 12 and 24 -- and
            #: **at isolated Newton steps in the back half of the plateau, not
            #: from one step onward**: at 8 cores steps 52 and 57 raise while
            #: 53-56 solve, so `H` recovers rather than staying rank deficient.
            #: Both the predictor and the corrector of a raising step always
            #: raise together, never one alone.  One `lstsq` on this 230x230
            #: costs about 3 ms against the 0.3-1.3 s the whole call takes, so
            #: **the cost sits in the per-call price, not in the count**; all
            #: 120 falling back would cost about 0.39 s, the order of the call
            #: itself.
            #:
            #: So carry on too, by the minimum-norm solution.  **This changes
            #: nothing that currently works**: the `try` branch is the previous
            #: line unaltered, so every non-singular solve is bit-for-bit what
            #: it was.  The warning is not decoration -- a reference that
            #: silently switched solvers mid-run would be worse than one that
            #: crashed, because then "the reference completed" would stop
            #: meaning what a reader takes it to mean.
            rhs_scaled = rhs / sc
            try:
                return np.linalg.solve(Ks, rhs_scaled) / sc
            except np.linalg.LinAlgError:
                warnings.warn(
                    "reference IPM: the scaled KKT matrix was singular to "
                    "LAPACK at this Newton step, so the step was taken from "
                    "the minimum-norm least-squares solution instead. The "
                    "comparison this reference feeds is still meaningful only "
                    "if the caller's own cleanliness precondition passes.",
                    RuntimeWarning, stacklevel=2)
                return np.linalg.lstsq(Ks, rhs_scaled, rcond=None)[0] / sc

        w = (-(lam * s) + lam * r3) / s
        za = solve_kkt(np.concatenate([-r1 - G.T @ w, -r2]))
        dxa, dna = za[:n], za[n:]
        Ga = G @ dxa
        dsa, dla = -r3 - Ga, w + D * Ga
        ap, ad = step_len(s, dsa), step_len(lam, dla)

        mu_aff = float((s + ap * dsa) @ (lam + ad * dla)) / m
        sigma = float(np.clip((mu_aff / max(mu, 1e-30)) ** 3, 1e-8, 1.0))

        r4 = lam * s - sigma * mu + dsa * dla
        w2 = (-r4 + lam * r3) / s
        z = solve_kkt(np.concatenate([-r1 - G.T @ w2, -r2]))
        dx, dn = z[:n], z[n:]
        Gd = G @ dx
        ds, dl = -r3 - Gd, w2 + D * Gd
        ap, ad = step_len(s, ds), step_len(lam, dl)

        if not (np.isfinite(dx).all() and np.isfinite(dn).all()):
            continue
        if freeze_mu is not None and mu < freeze_mu:
            xn, sn = x + ap * dx, np.maximum(s + ap * ds, 1e-14)
            ln, nn = np.maximum(lam + ad * dl, 1e-14), nu + ad * dn
            if merit(xn, sn, ln, nn) > merit(x, s, lam, nu):
                continue
        x = x + ap * dx
        s = np.maximum(s + ap * ds, 1e-14)
        lam = np.maximum(lam + ad * dl, 1e-14)
        nu = nu + ad * dn

    return (x, lam, nu, float(s @ lam) / m,
            float(np.abs(c + A.T @ nu + G.T @ lam).max()),
            float(np.abs(G @ x + s - h).max()))
