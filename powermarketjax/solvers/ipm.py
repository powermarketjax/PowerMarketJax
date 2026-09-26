"""Mehrotra predictor-corrector primal-dual interior point method, model-agnostic.

Solves the canonical linear program

    min c'x   s.t.   A x = b ,   G x <= h

and returns the solver's own duals; no price is reconstructed afterwards.

Nothing here knows about electricity markets.  The Newton system is reached
through two pluggable interfaces, so that structure can be exploited without
specialising the algorithm:

    ops = (Gx, GTy)                 matrix-vector products against G
    kkt = (factor, apply)           solve [[G'DG + reg I, A'], [A, 0]] z = r

Both default to dense forms, which suit a single period; `envs.day_ahead.kkt`
supplies block-tridiagonal versions for the multi-period problem.

The trip count is fixed rather than data-dependent, so a batch costs the same
whatever its hardest element is and lanes stay bit-identical under `vmap`.
`max_iter` and `dual_start` have no defaults: a market calibrates the two
together.

A fixed trip count has one failure of its own: on a problem whose Newton
system goes near-singular once the iterate has converged (``D = lam / s``
pushed to both clamps, ``reg`` pinned at its floor), the steps taken after
convergence are computed from garbage directions and the stationarity
residual -- the duals, i.e. the prices -- drifts back up while ``mu`` stays
put.  Measured 2026-09-16 on ``case813nem``'s ancillary clearing: the residual reaches 1e-8 at step ~25
and wanders in 1e-2..1e1 to step 60, on this implementation and on the numpy
reference alike; ``case29gb`` stays at 1e-12.  ``freeze_mu`` is the answer
that keeps the trip count fixed: once ``mu`` is below it, a step is taken only
if it does not raise the merit ``max(mu, |r1|_inf / kappa, |r3|_inf / (1 +
|h|_inf))``, so the iterate is held at the best point the loop reached.
Default ``None`` is the loop as it always was, to the bit.

Three things the gate is not.  It is not on by default anywhere: a market
turns it on by passing ``freeze_mu`` and stamps the value.  It is not a
substitute for iterations: a solve that has not reached its plateau by the
last step is held where it is, not carried further (market 02's mild-tier
cells, 2026-09-17: the two that 200
steps cure, 70 gated steps do not).  And it is not free on every shape: the
merit is not monotone along every true path, and on one of five market-02
all-rows cells the gate froze a solve that would have reached 1.5e-9 at
9.5e-5, moving the price by 2e-3 \$/MWh.  So a market that wants it runs its
own negative control first (market 03 did, on ``case29gb``); market 02 does
not use it.

The fixed count has a second failure, and this one the gate cannot see
because the loop never arrives: the regulariser ``reg = reg_coef *
mean(lam / s)`` is a mean over rows, and a row whose slack sits on
``SLACK_FLOOR`` with a multiplier near its cost weighs 1e18 in it.  On
``case813nem``'s real-time clearing (2026-09-17) 813 shed variables at zero do that,
``reg`` is 3.6e+03 against a `REG_COEF` of 1e-14, and the Newton system is a
proximal one: each step moves ``x`` by ``r1 / reg`` (1.4e-6) with full step
lengths, so the stationarity residual stays at the two tied units' cost gap
(5.0e-3) for as many steps as the loop has, ``mu`` reads 4e-11, and the price
is 0.108 \$/MWh from the all-rows solve.  With ``reg_coef`` two orders lower
the same 70 steps reach 8e-12.  Under any small regulariser both markets'
813nem cells converge by step ~25 and then walk off the plateau if the loop
keeps stepping (45 of the next 70 factorisations singular); under the large
one market 02 never arrives.  ``stop_tol`` is the other half of that recipe:
a data-dependent trip count that stops at the market's gate with ``max_iter``
as the cap, and returns the best iterate seen when it hits the cap.  Default
``None`` is the fixed count, to the bit.  `REG_COEF` keeps its original value;
a market that wants the smaller one passes it and stamps it.

float64 throughout; on GPU `jax_default_matmul_precision` must be `'highest'`.
"""
from typing import Callable, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jax import lax

#: Primal-dual regularisation coefficient; ``reg = REG_COEF * mean(lam / s)``,
#: floored at 1e-12.  It is not scale-free: raising it perturbs the duals,
#: lowering it costs the KKT factorisation its conditioning, and it needs
#: retuning as the problem grows.
REG_COEF = 1e-14

#: Fraction-to-boundary step damping: a step covers at most this fraction of the
#: distance to the boundary, which keeps the slacks and the multipliers strictly
#: positive.  At 1 the iterate lands on the boundary and ``lam / s`` divides by
#: zero; well below 1 the steps shorten and the trip count rises.
TAU = 0.995

#: Floor under the slacks and the multipliers after each step, and hence the
#: floor of ``mu``.  An implicit regulariser rather than a safety net: lowering
#: it drives ``mu`` orders of magnitude lower while making the stationarity
#: residual worse by as much, and dual accuracy is the side that is consumed.
SLACK_FLOOR = 1e-14

#: Dual scale at the starting point: ``lam = kappa / s``.  ``"unit"`` takes
#: ``kappa = 1``, which centres every complementarity product at 1 but starts
#: the multipliers orders of magnitude below where the solution needs them;
#: ``"cost_norm"`` takes ``kappa = max(1, |c|_inf)``, the problem's own dual
#: scale, and reaches those magnitudes in far fewer steps.  There is no default:
#: the trip count a market needs depends on the start it runs, so `max_iter` and
#: `dual_start` are one calibration.
DUAL_STARTS = ("unit", "cost_norm")


def _dense_ops(G: chex.Array):
    """Default ``(Gx, GTy)``: dense matrix-vector products against ``G``."""
    return (lambda v: G @ v), (lambda y: G.T @ y)


def _dense_kkt(n: int, G: chex.Array, A: chex.Array):
    """Default ``(factor, apply)``: form the KKT matrix densely and LU-factor it."""
    n_eq = A.shape[0]

    def factor(D, reg):
        """Factor ``[[G'DG + reg I, A'], [A, 0]]``; returns ``(lu, sc)``."""
        H = (G * D[:, None]).T @ G + reg * jnp.eye(n, dtype=G.dtype)
        K = jnp.concatenate(
            [jnp.concatenate([H, A.T], 1),
             jnp.concatenate([A, jnp.zeros((n_eq, n_eq), G.dtype)], 1)], 0)
        # symmetric diagonal equilibration before factoring; the KKT matrix
        # spans ~30 orders of magnitude by the final iterations
        sc = jnp.sqrt(jnp.maximum(jnp.abs(jnp.diag(K)), jnp.asarray(1e-12, G.dtype)))
        return jsl.lu_factor((K / sc[:, None]) / sc[None, :]), sc

    def apply(state, rhs):
        """Solve one right-hand side, undoing the equilibration on both sides."""
        lu, sc = state
        return jsl.lu_solve(lu, rhs / sc) / sc

    return factor, apply


def make_solver(
    n: int,
    m: int,
    max_iter: int,
    *,
    dual_start: str,
    n_eq: int = 1,
    reg_coef: float = REG_COEF,
    tau: float = TAU,
    slack_floor: float = SLACK_FLOOR,
    ops: Optional[Tuple[Callable, Callable]] = None,
    kkt: Optional[Tuple[Callable, Callable]] = None,
    freeze_mu: Optional[float] = None,
    stop_tol: Optional[Tuple[float, float]] = None,
    report_steps: bool = False,
) -> Callable:
    """Build a jittable solver for a fixed problem shape.

    Args:
        n: number of variables.
        m: number of inequality rows of ``G`` (bounds already folded in).
        max_iter: Newton steps.  No default -- each market calibrates it.
        n_eq: number of equality rows; one balance per period for SCED.
        slack_floor: floor under the slacks and multipliers; see `SLACK_FLOOR`.
        ops: optional ``(Gx, GTy)``; dense products against ``G`` if omitted.
        kkt: optional ``(factor, apply)``; dense factorisation if omitted.
        dual_start: scale of the multipliers at the starting point, one of
            `DUAL_STARTS`.  No default -- it sets the trip count a market needs.
        freeze_mu: ``None`` (the default) runs every step unconditionally, as
            this loop always did.  A float arms the merit gate of the module
            docstring once ``mu`` is below it: from then on a step is taken
            only if it does not raise ``max(mu, |r1|_inf / kappa,
            |r3|_inf / (1 + |h|_inf))``, ``kappa = max(1, |c|_inf)``.  The
            trip count, the returned tuple and the `vmap` behaviour are
            unchanged; a market that turns this on stamps the value it used.
        stop_tol: ``None`` (the default) keeps the fixed trip count.  A pair
            ``(mu_tol, dual_tol)`` makes the trip count data-dependent: the
            loop stops at the first iterate with ``mu < mu_tol`` and
            ``|r1|_inf < dual_tol`` -- the market's own gate, so that a lane
            that has converged takes no further step and cannot walk away
            from its plateau -- and ``max_iter`` becomes the hard cap.  A
            lane that reaches the cap without meeting the tolerance returns
            the iterate with the smallest ``max(mu / mu_tol, |r1|_inf /
            dual_tol)`` it visited (the final one included, last wins on
            ties), which is the point the fixed-count loop would have wanted
            to stop at.  Under `vmap` the batch runs until its slowest lane
            stops and a finished lane's carry is held, so a lane takes the
            same number of steps whatever else is in the batch; its values
            agree with a single-lane run to the batched kernels' rounding
            (1e-12 on `case29gb`, and the fixed-count loop has the same
            difference).  The market
            stamps the pair it used.
        report_steps: append the number of steps taken to the returned
            tuple (``max_iter`` when ``stop_tol`` is ``None``).  Off by
            default so that every existing caller unpacks six values.

    Returns:
        ``solve(c, A, b, G, h, x0) -> (x, lam, nu, mu, r1_inf, r3_inf)``.  ``lam``
        are the duals of ``G x <= h`` (>= 0), ``nu`` those of ``A x = b``.  ``mu``
        is the complementarity gap and is the convergence check: a large value
        means the duals must not be used.  ``G`` is ignored when both ``ops`` and
        ``kkt`` are given.

        ``r1_inf`` is the dual (stationarity) residual and ``r3_inf`` the primal
        one, ``|G x + s - h|`` at the final iterate.  ``s`` is held at or above
        ``slack_floor`` throughout, so ``G x - h = r3 - s``: the inequalities are
        satisfied by construction only while ``r3`` is zero, and ``mu`` does not
        report that.
    """

    if dual_start not in DUAL_STARTS:
        raise ValueError(f"dual_start must be one of {DUAL_STARTS}, got {dual_start!r}")

    def step_len(v, dv):
        """Longest step along ``dv`` that keeps ``v`` strictly positive.

        Only components moving towards zero can bind, so the others take an
        infinite ratio, and the winning ratio is damped by ``tau`` to stop short
        of the boundary.  The inner `jnp.where` puts ``-1`` in the denominator on
        the components that cannot bind, so the discarded branch divides by a
        finite number rather than producing a NaN.
        """
        neg = dv < 0
        ratio = jnp.where(neg, -v / jnp.where(neg, dv, -1.0), jnp.inf)
        return jnp.minimum(1.0, tau * jnp.min(ratio))

    def solve(c, A, b, G, h, x0):
        """Run ``max_iter`` Newton steps on ``min c'x  s.t.  A x = b, G x <= h``.

        Args:
            c: objective coefficients, ``(n,)``.
            A: equality matrix, ``(n_eq, n)``.
            b: equality right-hand side, ``(n_eq,)``.
            G: inequality matrix, ``(m, n)``; read only by the dense defaults,
                so callers that supplied both ``ops`` and ``kkt`` pass a
                placeholder here.
            h: inequality right-hand side, ``(m,)``.
            x0: starting primal point.  The slacks follow from it and the duals
                from the slacks, by the rule `DUAL_STARTS` names; ``nu`` starts
                at zero in ``x0``'s dtype.

        Returns:
            ``(x, lam, nu, mu, r1_inf, r3_inf)``: the primal iterate, the duals
            of ``G x <= h``, the duals of ``A x = b``, the complementarity gap,
            and the infinity norms of the stationarity and primal-inequality
            residuals, all read off the final iterate.  There is no convergence
            flag: the trip count is fixed and nothing is tested.
        """
        Gx, GTy = ops if ops is not None else _dense_ops(G)
        factor, apply = kkt if kkt is not None else _dense_kkt(n, G, A)

        x = x0
        # strictly interior slacks; the naive box-midpoint start stalls with
        # zero step length for many iterations
        s = jnp.maximum(h - Gx(x), 1.0)
        kappa = 1.0 if dual_start == "unit" else jnp.maximum(1.0, jnp.abs(c).max())
        lam = kappa / s
        nu = jnp.zeros((n_eq,), x0.dtype)

        if freeze_mu is not None:
            # the merit gate; built only when asked for, so that the default
            # loop below traces exactly what it traced before this existed
            merit_scale = jnp.maximum(1.0, jnp.abs(c).max())
            hscale = 1.0 + jnp.abs(h).max()

            def merit(x_, s_, lam_, nu_):
                return jnp.maximum(
                    jnp.maximum(jnp.dot(s_, lam_) / m,
                                jnp.abs(c + A.T @ nu_ + GTy(lam_)).max() / merit_scale),
                    jnp.abs(Gx(x_) + s_ - h).max() / hscale)

        def body(_, carry):
            x, s, lam, nu = carry
            mu = jnp.dot(s, lam) / m         # average complementarity product
            D = lam / s                      # barrier weights, the D of G'DG
            reg = jnp.maximum(1e-12, reg_coef * jnp.mean(D))

            # The three KKT residuals at the current iterate: stationarity,
            # equality feasibility, and inequality feasibility with the slack
            # folded in.  `r1` and `r3` are the two the caller receives, re-formed
            # after the loop from the final iterate.
            r1 = c + A.T @ nu + GTy(lam)     # stationarity: cancellation-prone
            r2 = A @ x - b
            r3 = Gx(x) + s - h
            state = factor(D, reg)

            # affine (predictor) direction
            # `w` is what `dlam` collapses to once `ds` and the complementarity
            # row are eliminated: the full direction is `dlam = w + D Gdx`,
            # which is why the reduced right-hand side carries `-G'w` and why
            # `ds` comes back as `-r3 - Gdx`.
            w = (-(lam * s) + lam * r3) / s
            za = apply(state, jnp.concatenate([-r1 - GTy(w), -r2]))
            dxa, dna = za[:n], za[n:n + n_eq]
            Ga = Gx(dxa)
            dsa, dla = -r3 - Ga, w + D * Ga
            ap, ad = step_len(s, dsa), step_len(lam, dla)

            # Mehrotra's adaptive centring: a productive affine step earns a
            # smaller sigma.  A fixed sigma converges to a primal-feasible point
            # with unusable duals.
            mu_aff = jnp.dot(s + ap * dsa, lam + ad * dla) / m
            sigma = jnp.clip((mu_aff / jnp.maximum(mu, 1e-30)) ** 3, 1e-8, 1.0)

            # corrector, reusing the same factorisation.  `r4` is the
            # complementarity residual carrying both of Mehrotra's terms:
            # `-sigma * mu` recentres, and `dsa * dla` cancels the second-order
            # product that the affine direction left behind.
            r4 = lam * s - sigma * mu + dsa * dla
            w2 = (-r4 + lam * r3) / s
            z = apply(state, jnp.concatenate([-r1 - GTy(w2), -r2]))
            dx, dn = z[:n], z[n:n + n_eq]
            Gd = Gx(dx)
            ds, dl = -r3 - Gd, w2 + D * Gd
            ap, ad = step_len(s, ds), step_len(lam, dl)

            # If the factorisation returns a non-finite direction, the step is
            # refused rather than taken.  The trip count is fixed and there is no
            # early exit, so one NaN admitted here would reach every later step
            # and both returned residuals; instead the iterate is held and the
            # remaining steps run on it.
            # `jnp.where` rather than `lax.cond`: under `vmap` both branches of
            # a `lax.cond` execute anyway, so the selection has to be a `where`.
            ok = jnp.isfinite(dx).all() & jnp.isfinite(dn).all()
            if freeze_mu is None:
                return (jnp.where(ok, x + ap * dx, x),
                        jnp.where(ok, jnp.maximum(s + ap * ds, slack_floor), s),
                        jnp.where(ok, jnp.maximum(lam + ad * dl, slack_floor), lam),
                        jnp.where(ok, nu + ad * dn, nu))
            # the gated variant: the same candidate, refused when it would
            # raise the merit once the run is on the plateau.  Armed on `mu`
            # rather than always because the early iterations raise `mu` and
            # the residuals on purpose before bringing them down.
            xn = x + ap * dx
            sn = jnp.maximum(s + ap * ds, slack_floor)
            ln = jnp.maximum(lam + ad * dl, slack_floor)
            nn = nu + ad * dn
            armed = mu < freeze_mu
            ok = ok & (~armed | (merit(xn, sn, ln, nn) <= merit(x, s, lam, nu)))
            return (jnp.where(ok, xn, x), jnp.where(ok, sn, s),
                    jnp.where(ok, ln, lam), jnp.where(ok, nn, nu))

        if stop_tol is None:
            x, s, lam, nu = lax.fori_loop(0, max_iter, body, (x, s, lam, nu))
            steps = jnp.asarray(max_iter, jnp.int32)
        else:
            # the data-dependent trip count: the same `body`, driven by a
            # `while_loop` whose test is the market's gate on the current
            # iterate.  The merit is carried rather than recomputed in the
            # test, so each step pays two extra matrix-vector products, not
            # four; the factorisation is untouched.  Its own name: `merit`
            # above is the freeze gate's, and both flags may be on at once.
            mu_tol, dual_tol = (float(v) for v in stop_tol)

            def stop_merit(s_, lam_, nu_):
                return jnp.maximum(jnp.dot(s_, lam_) / m / mu_tol,
                                   jnp.abs(c + A.T @ nu_ + GTy(lam_)).max() / dual_tol)

            def cond(carry):
                it, _x, _s, _lam, _nu, mer, _best = carry
                return (it < max_iter) & (mer >= 1.0)

            def body_stop(carry):
                it, x_, s_, lam_, nu_, mer, best = carry
                # `<=` so that among equal merits the later iterate wins: a
                # healthy run's final iterate is then its own best
                keep = mer <= best[4]
                best = tuple(jnp.where(keep, new, old)
                             for new, old in zip((x_, s_, lam_, nu_, mer), best))
                x_, s_, lam_, nu_ = body(it, (x_, s_, lam_, nu_))
                return it + 1, x_, s_, lam_, nu_, stop_merit(s_, lam_, nu_), best

            best0 = (x, s, lam, nu, jnp.asarray(jnp.inf, x0.dtype))
            steps, x, s, lam, nu, mer, best = lax.while_loop(
                cond, body_stop,
                (jnp.asarray(0, jnp.int32), x, s, lam, nu, stop_merit(s, lam, nu), best0))
            # out of steps and the last iterate is not the best one seen:
            # hand back the best.  `jnp.where`, not `lax.cond`, for `vmap`.
            worse = (steps >= max_iter) & (mer > best[4])
            x, s, lam, nu = (jnp.where(worse, b, v)
                             for b, v in zip(best[:4], (x, s, lam, nu)))
        # Both residuals are re-formed here from the final iterate rather than
        # carried out of the loop, which is why adding the second one cannot
        # move the first: the loop is untouched and `mu` and `r1_inf` are the
        # same expressions on the same `x, s, lam, nu` as before.
        out = (x, lam, nu, jnp.dot(s, lam) / m,
               jnp.abs(c + A.T @ nu + GTy(lam)).max(),
               jnp.abs(Gx(x) + s - h).max())
        return out + (steps,) if report_steps else out

    return solve
