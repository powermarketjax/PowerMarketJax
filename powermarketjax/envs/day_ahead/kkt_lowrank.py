"""Low-rank Newton system for the multi-period SCED and the relaxed commitment,
for a case whose line limits are enforced on a **monitored** subset of lines.

`kkt.py` and `relax_kkt.py` solve the same block-tridiagonal system with one
dense LU per period of order ``nb + 1``.  The block is dense for one reason
only: the line term ``Mblk' D Mblk`` couples every bus to every other through
the PTDF.  On ``case813nem`` that is a 964-wide block per period, 1.2 GiB per
environment for the SCED and 1.6 GiB for the relaxation, while only 7 of the
1 278 lines carry a finite rating -- the other 1 271 are 1e6 MW and can never
bind (max |PTDF| = 1, installed capacity 39 GW).  Keeping the rows of the never-
binding lines was measured earlier to save 1.17x, because the width of the block
is set by the bus count, not by the row count.

With ``k`` monitored lines the Hessian of the barrier is

    H = H0 + W diag(d) W'                      d = D+ + D-  per (line, period)

where ``W`` has one column per monitored line and period and ``H0`` is what the
remaining rows -- box, ramp, and for the relaxation capacity and the bounds on
the eliminated shut-down indicator -- contribute.  Every one of those rows lives
inside a single unit: it touches that unit's own columns at ``t`` and at ``t-1``
and nothing else.  So in unit-major order ``H0`` is a direct sum of ``n_units``
small block-tridiagonal matrices of order ``q*T`` (``q`` columns per unit and
period: ``K`` for the SCED, ``K+2`` for the relaxation) plus a diagonal over the
shed variables.  The balance rows are ``T`` more columns of the same kind.  The
Newton system is then solved exactly by a Schur complement on the
``r = T*(k+1)`` low-rank columns:

    [[H0, U], [U', -C]] [x; w] = [r_x; -[0; r_nu]]      U = [W sqrt(d), A']
    (U' H0^-1 U + C) w = U' H0^-1 r_x - [0; r_nu]       C = blockdiag(I, 0)
    x = H0^-1 (r_x - U w)

which factors ``n_units`` matrices of order ``q*T`` (batched Cholesky) and one
of order ``r`` (dense LU), and never forms an ``nb``-wide block.  The
capacitance matrix is formed as ``I + sqrt(d) W' H0^-1 W sqrt(d)`` rather than
through ``d^-1``, because ``d`` spans some thirty orders of magnitude by the end
of the interior point iteration and the reciprocal form would be dominated by
the rows that do not bind.

**Same LP, same iterates.**  Nothing here changes the problem or the Newton
direction: `factor`/`apply` return the solution of the same linear system the
dense route solves, to round-off, so `ipm` runs the same predictor-corrector
sequence.  The reduction to monitored rows is the caller's (`clearing`,
`relax`): those modules drop the unmonitored line rows from ``G`` and ``h``, and
the result is the optimum of the full LP whenever the dropped rows cannot bind.

**Iterative refinement.**  The Schur route is exact in arithmetic but not
backward stable: when a unit's ramp binds in consecutive periods its ``H0``
block is singular to round-off in the direction those periods move together,
and ``H0^-1`` amplifies rounding by the reciprocal of that eigenvalue, which
the dense route never does because its LU sees the balance column in the same
matrix (Goldfarb & Scheinberg, Math. Prog. 2004, say the same of Sherman-
Morrison-Woodbury for dense columns).  Measured on real iterates (813nem, T=2,
2026-09-16) the plain solve leaves a
relative residual of 6e-3 at the late steps where the dense route leaves 2e-7.
`apply` therefore refines against the unreduced system, matrix-free, with the
same factors: ``N_REFINE`` steps each cost two small products and one more
Schur solve, and each recovers the factor the plain solve lost.  Both
factorisations are pivoted LU after symmetric diagonal scaling; Cholesky fails
outright on the same round-off-singular blocks.

**Free columns (2026-09-16).**  The refinement above assumes the plain
Schur solve is a contraction, and in a period that sheds load behind a
binding line it is not.  Every shed column of the import-constrained region
is then dual-degenerate -- its price is VOLL, so both box rows have weight
~1e-14 and its only curvature is ``reg`` -- while the binding line's weight
is 1e14..1e18.  The Schur route computes such a column as ``x_f = (r_f - U_f
w) / reg``: two O(VOLL) terms cancel and the difference is divided by
``reg``, so ``x_f`` carries an error of ``eps * VOLL / reg`` on each of the
~110 columns of the region, and the binding line's row of the Newton matrix
multiplies their net flow by ``d_j``.  Measured on `case813nem` at T=1 (5
shed periods of the real-time market,
2026-09-16): plain-solve residual 1e1 to 4e1 against the dense route's
1e-12, and every refinement step multiplies it by ~20 because the correction
is computed by the same solve (spectral radius of ``I - M^-1 K`` measured
6-22).  No step count helps: N_REFINE 0..8 gives 2.7e1 .. 4.1e10.  The
day-ahead shape is not exempt, it just never shed: two T=24 days that shed
230 and 4 047 MW leave the plain route at 7e29 and 3e38.

The fix is the one the dense route applies implicitly through pivoting: the
columns whose own curvature is tiny against the stiff line curvature acting
on them are **not** eliminated through ``H0^-1``.  `factor` ranks every
column by ``sum_j d_j W_ji^2 / H0_ii`` (the ratio the error bound above
scales with), keeps the ``n_free = (units, shed columns)`` highest-ranked
inside a pivoted LU together with the ``r`` Schur columns, and eliminates
only the rest through ``H0^-1``.  With ``n_free = (0, 0)`` -- the default,
and what every day-ahead product was made on -- the code path is the one
above, unchanged.  Sizing: the region behind `case813nem`'s line 738 has 109
buses and is what the five failing periods shed in; the largest region
behind any rated line there (line 10) has 523.  The pivoted block is of
order ``n_free_units * q * T + n_free_shed + r``.

**Arrowhead solve of that block (2026-09-17).**  At ``q * T = 1`` --
the real-time market's shape, one segment and one period -- every unit
block is a scalar, so the pivoted block is diagonal except for its last
``r`` rows and columns: an arrowhead with an ``r``-column border.  A dense
pivoted LU of it spends ``(n_free + r)^3 * 2/3`` flops on zeros, and on
GPU cusolver's batched getrf at order 539 gives no parallel gain across
lanes (2026-09-17, GPU: 346.8 s per RL iteration of 64 envs x 48
steps, 10.5 s on the plain route).  ``lu_batching="arrow"`` solves the
same block exactly through a thin Householder QR of the scaled border,
``R~ = Q T`` (``n_free x r``), and a pivoted LU of the ``(p + r)``-wide core, ``p = min(n_free, r)``,
``[[I, T], [T', -S~]]``: with ``z = Q' x~`` the border rows see only ``z``,
which the core solve produces directly instead of as the difference of two
``O(VOLL / sqrt(reg))`` numbers -- the cancellation the plain route dies
of never appears.  The solution is that of the pivoted LU to round-off;
the work per factorisation is ``O(n_free * r^2)``.

**Local border (2026-09-17).**  The ancillary market's unit block is
of order ``q = K + P`` (segments and reserve products, coupled by the
capacity-sharing row), so at ``T = 1`` the free-column block is block
diagonal rather than diagonal and the arrowhead above does not apply as it
stands.  Pivoting through a scaled LU of each ``q x q`` block is not an
option: when the coupling row binds its weight is 1e14 against ``reg`` for
the columns' own curvature, the scaled block is the all-ones matrix to
round-off (eigenvalues ``-6e-16, -2e-17, 3`` measured on `case813nem`),
and ``H0^-1`` in the ``q - 1`` directions orthogonal to ones is noise.
Every row that couples a unit's columns is instead a border column of its
own unit -- ``sqrt(d) c`` on that unit's ``q`` rows, ``+1`` on the
capacitance diagonal -- which makes the top-left diagonal again in the box
weights alone.  The border is then solved in two levels (`_make_kkt`,
`local_arrow_factor`): a batched full QR of each unit's ``(q, n_loc)``
block changes that unit's frame so its own border couples to the rest only
through ``p = min(q, n_loc)`` pivot rows, the local unknowns are
eliminated exactly through ``(p + n_loc)``-wide pivoted cores whose
contribution to the Schur block is a sum of positive semidefinite terms,
and what remains is the single-border arrowhead of the shape above on the
other rows.  ``ARROW_REFINE`` applies as before, against the three
residuals of the scaled system.  Off this shape (``q * T = 1``) the code
path is the one above, unchanged.
"""
from typing import Callable, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
from jax import lax
from jax.custom_batching import sequential_vmap

#: Refinement steps against the unreduced Newton system in `apply`; see the
#: module docstring.  Each step multiplies the plain solve's relative residual
#: by roughly that residual itself.  Measured on the recorded 813nem T=24
#: iterates (CPU, 2026-09-16): two steps
#: leave 2.1e-7 against the dense route's 1.3e-10, three leave the level the
#: low-rank route was first benchmarked at.  Zero reproduces
#: the plain Schur solve.  Raised 3 -> 4 on review (2026-09-16,
#: 813nem T=24 pilot iterates, CPU): on the
#: SCED three steps left a worst residual of 5.9e-9 against the dense route's
#: 1.9e-10 on the same steps, four leave 3.0e-10; five and six buy nothing
#: more.  The relaxation floors at ~1.2e-9 from three steps on (dense 2.1e-10):
#: that floor is the Schur route's own and no step count reaches below it.
#: Seconds per step are unchanged within timing noise.  The pilot and its
#: seeds/sweep of 2026-09-16 were produced at 3.
N_REFINE = 4


#: How the pivoted LU of the free-column block is batched under `vmap`.
#: ``"sequential"`` factors one lane at a time (`sequential_vmap`):
#: measured 2026-09-16 (CPU, 8 cores, 539-wide block)
#: XLA CPU's batched kernel takes 2 ms per
#: matrix up to 4 lanes and 617 ms per matrix at 8 -- the same cliff that
#: leaves the dense route's 965-wide block unusable under `vmap` on CPU --
#: while the sequential form stays at 2 ms per matrix through 64 lanes and
#: returns the same factors to the bit.  ``"batched"`` is the plain batched
#: kernel: on GPU that is cusolver's batched getrf, and the sequential form
#: there is a serial loop over lanes with no parallel gain (GPU,
#: 2026-09-17: 64 envs x 48 steps at 1 259 s per iteration, 0.355 s x 64 per
#: step).  ``"auto"`` resolves at build time to ``"sequential"`` on the CPU
#: backend and ``"batched"`` elsewhere.  `lu_solve` batches fine on both and
#: is left as it is.
#: ``"arrow"`` does not LU-factor the block at all: it is solved as an
#: arrowhead (module docstring, "Arrowhead solve"), which needs ``q * T = 1``
#: and is refused otherwise.
LU_BATCHINGS = ("auto", "sequential", "batched", "arrow")

#: Refinement passes inside the arrowhead solve (`solve_once_free`, ``arrow``),
#: against the block's own two residuals with the same ``Q`` and core.  The
#: top rows' right-hand side is orders of magnitude above their solution (a
#: shed column's ``r_i / sqrt(reg)`` against a tiny step), and the round-off
#: non-orthogonality of ``Q`` leaks that into the border rows, which the
#: binding lines then multiply by ``|T|``.  Measured 2026-09-17 on a recorded
#: `case813nem` iterate (exact rational
#: arithmetic for the residuals): zero passes leave the border rows at 6e-4
#: against the pivoted LU's 2e-16 and the full Newton system at a relative
#: residual of 27, from which `apply`'s outer refinement diverges; one pass
#: brings the border to the floor of its double-precision evaluation.  Read at
#: trace time, like `N_REFINE`.
ARROW_REFINE = 1

_lu_factor_seq = sequential_vmap(lambda a: jsl.lu_factor(a))


def resolve_lu_batching(lu_batching: str, arrowhead: bool = False) -> str:
    """``"auto"`` -> ``"arrow"`` on the arrowhead shape, else the platform's
    choice of LU batching; the others pass through.

    ``arrowhead`` says whether the block has the arrowhead shape (``q * T =
    1``, the real-time market); ``"arrow"`` asked for explicitly on any other
    shape is refused by `_make_kkt`.  ``"auto"`` picks it there since
    2026-09-17: the same solution as the pivoted LU on the five
    recorded shed periods at
    ``O(n_free * r^2)`` instead of a 539-wide LU per lane.
    """
    if lu_batching not in LU_BATCHINGS:
        raise ValueError(f"lu_batching must be one of {LU_BATCHINGS}, got {lu_batching!r}")
    if lu_batching != "auto":
        return lu_batching
    if arrowhead:
        return "arrow"
    return "sequential" if jax.default_backend() == "cpu" else "batched"


def householder_qr(A: chex.Array, full: bool = False) -> Tuple[chex.Array, chex.Array]:
    """Thin QR of an ``(m, r)`` array by ``p = min(m, r)`` Householder
    reflectors, ``r`` static and small: ``A = Q @ T`` with ``Q`` ``(m, p)``
    orthonormal and ``T`` ``(p, r)`` upper triangular (trapezoidal when the
    border is wider than the block is tall, as on a case whose monitored
    lines outnumber its free columns).  ``full`` returns the square ``Q``
    ``(m, m)`` instead, whose first ``p`` columns are the thin one's (the
    local border of `_make_kkt`'s unit blocks needs the other ``m - p`` to
    change frame); the default is the thin factor as before.

    Written out in ``jnp`` so it batches under `vmap` on any backend as
    elementwise work, and so a column of round-off-zero norm (a monitored
    line with no free column behind it) leaves an identity reflector and a
    zero row of ``T`` rather than a NaN; the pivoted solve of the core
    absorbs the rank deficiency.  The sign of the reflector follows the
    leading entry, the usual choice that keeps ``v`` from cancelling.
    """
    m, r = A.shape
    p = min(m, r)
    R = A
    vs = []
    for j in range(p):
        x = R[j:, j]
        norm = jnp.sqrt(jnp.sum(x * x))
        sign = jnp.where(x[0] >= 0, 1.0, -1.0)
        v = x.at[0].add(sign * norm)
        vn = jnp.sqrt(jnp.sum(v * v))
        v = jnp.where(vn > 0, v / jnp.where(vn > 0, vn, 1.0), 0.0)
        R = R.at[j:, :].add(-2.0 * v[:, None] * (v @ R[j:, :])[None, :])
        vs.append(v)
    T = jnp.triu(R[:p, :])
    Q = jnp.eye(m, m if full else p, dtype=A.dtype)
    for j in reversed(range(p)):
        v = vs[j]
        Q = Q.at[j:, :].add(-2.0 * v[:, None] * (v @ Q[j:, :])[None, :])
    return Q, T


def _local(X: np.ndarray, cols: np.ndarray, name: str) -> np.ndarray:
    """Gather a per-unit row operator ``X`` ``(n_units, nb)`` at each unit's own
    columns ``cols`` ``(n_units, q)`` and check nothing was left behind.

    Scattering the gathered values back has to reproduce ``X`` exactly; if it
    does not, some row reaches a column outside its unit and the direct-sum
    structure this module rests on does not hold for that operator.
    """
    V = np.take_along_axis(X, cols, axis=1)
    back = np.zeros_like(X)
    np.put_along_axis(back, cols, V, axis=1)
    if not np.array_equal(back, X):
        raise AssertionError(
            f"{name} has nonzeros outside its unit's columns; the per-unit "
            "direct sum this solver assumes does not hold")
    return V


def _make_kkt(spec: Dict, cols: np.ndarray, o_s: int, M_mon: np.ndarray,
              a_row: np.ndarray, split: Callable,
              n_free: Tuple[int, int] = (0, 0),
              lu_batching: str = "auto",
              arrow_local: bool = False) -> Tuple[Callable, Callable]:
    """``(factor, apply)`` for `ipm.make_solver`, low-rank in the monitored lines.

    Args:
        spec: the market's spec, already reduced to the monitored lines
            (``n_l`` = number of monitored lines, ``m`` counted on them).
            Only ``T``, ``nb``, ``n_units``, ``n_buses`` and ``n_l`` are read:
            ``n_buses`` is the number of diagonal columns from ``o_s`` on and
            ``n_l`` the number of rows of ``M_mon``, whatever they stand for
            (`make_kkt_ancillary` hands in a view in which the reserve
            shortfall columns count as shed columns and the requirement rows
            as monitored lines; both are exact, see there).
        cols: ``(n_units, q)`` block-local column indices of each unit's own
            variables, in the order the unit-local operators are gathered in.
        o_s: offset of the ``n_buses`` shed columns inside a period block.
        M_mon: ``(k, nb)`` line operator restricted to the monitored lines.
        a_row: ``(nb,)`` the balance row of one period.
        split: ``D -> (d_line (T, k), groups, within, d_box (T, nb))``, where
            ``groups`` is a list of ``(d (T, n_units), own (n_units, q), prev
            (n_units, q))`` -- row ``t`` of that group is ``own . x_t - prev .
            x_{t-1}`` -- and ``within`` a list of ``(d (T, n_units, K), C
            (n_units, K, q))`` for rows that act inside one period only.
        n_free: ``(units, shed columns)`` kept inside the pivoted block instead
            of being eliminated through ``H0^-1``; see the module docstring.
            ``(0, 0)`` is the plain Schur route.  Static: the sizes fix the
            order of the block, the members are chosen per factorisation.
        lu_batching: how that block's LU is batched under `vmap`, one of
            `LU_BATCHINGS`; unused when ``n_free`` is ``(0, 0)``.
        arrow_local: allow the arrowhead solve at ``T = 1`` with ``q > 1``
            by moving every ``within`` row of a free unit into a local
            border (module docstring, "Local border"); off, the arrowhead
            is refused off the ``q * T = 1`` shape as before.
    """
    T, nb, n_u, n_b = spec["T"], spec["nb"], spec["n_units"], spec["n_buses"]
    k = spec["n_l"]
    n = nb * T
    q = cols.shape[1]
    r = T * (k + 1)
    cols_flat = jnp.asarray(cols.reshape(-1))
    M_loc = jnp.asarray(M_mon[:, cols.reshape(-1)].reshape(k, n_u, q))   # (k, n_u, q)
    M_shed = jnp.asarray(M_mon[:, o_s: o_s + n_b])                       # (k, n_b)
    a_loc = jnp.asarray(a_row[cols.reshape(-1)].reshape(n_u, q))
    a_shed = jnp.asarray(a_row[o_s: o_s + n_b])
    I_T = jnp.eye(T)
    Sub = jnp.eye(T, k=-1)                    # Sub[t, t-1] = 1
    # +1 on the diagonal of the capacitance for the line columns, 0 for balance
    cmask = jnp.concatenate([jnp.ones((T, k)), jnp.zeros((T, 1))], 1).ravel()
    n_fu, n_fs = int(n_free[0]), int(n_free[1])
    if not (0 <= n_fu <= n_u and 0 <= n_fs <= T * n_b):
        raise ValueError(f"n_free must lie in [0, {n_u}] x [0, {T * n_b}], got {n_free!r}")
    free = n_fu + n_fs > 0
    qT = q * T
    arrowhead = qT == 1 or (arrow_local and T == 1)
    batching = resolve_lu_batching(lu_batching, arrowhead=arrowhead)
    arrow = batching == "arrow"
    if arrow and not arrowhead:
        raise ValueError(
            f"lu_batching='arrow' needs q * T == 1 (a diagonal unit block); got q={q}, T={T}")
    local = arrow and qT != 1
    #: **On the local-border route the free set is either empty or holds every
    #: unit; it cannot hold only some.**  The `local` branch below zeroes, by
    #: selection, the `H0^-1 U` rows of the units **inside** the free set
    #: (`inf * 0` is NaN, so it is selection and not weighting).  A unit
    #: **outside** the free set whose capacity-sharing row binds is not in that
    #: zeroing: its block is singular to round-off, its `X_unit` rows are inf,
    #: and the capacitance sum turns the whole Newton direction non-finite --
    #: which `ipm`'s `ok` swallows, so nothing shows in the readings (derived
    #: 2026-09-17; measured by injection: 230 non-finite components on 29gb,
    #: and on 813nem at (32,815), 7 of the 60 steps after the stopping point
    #: non-finite).
    #:
    #: `n_fu == n_u` zeroes everything and `n_fu == 0` does not take this path,
    #: so both ends are safe; only the tier in between had no guard.  Refused
    #: statically rather than caught at run time: this is a shape that can be
    #: judged when the operator is built, and should not wait for someone to
    #: read a direction that `ok` swallowed.
    #:
    #: **The criterion hangs on `local`, not on the `arrow_local` argument**,
    #: and the two are not the same thing (an earlier version hung it on the
    #: wrong one, and on 2026-09-17 it was measured to be wrong in both
    #: directions): with `arrow_local=True` and `lu_batching="sequential"`,
    #: `local` is False -- there is no zeroing and so no hole, yet the old
    #: criterion would refuse it (the `(8,523)+sequential` configuration is
    #: exactly this tier); conversely, with `arrow_local=False`,
    #: `lu_batching="arrow"` and `q*T != 1`, `local` is True -- the hole is
    #: there, yet the old criterion let it through.  **The effective value is
    #: read from the line that computes it, not from the argument passed in.**
    if local and 0 < n_fu < n_u:
        raise ValueError(
            f"the local-border arrowhead needs the free set to hold either no unit "
            f"or every unit; got n_free[0]={n_fu} of {n_u}. A unit left outside it "
            f"whose capacity-sharing row binds goes through H0^-1 with a block that "
            f"is singular to round-off, and the direction comes back non-finite")
    lu_factor_aug = _lu_factor_seq if batching == "sequential" else jsl.lu_factor

    def unit_blocks(groups, within, d_box_u, reg):
        """``(n_units, q*T, q*T)`` dense per-unit block-tridiagonal matrices."""
        own = jnp.zeros((T, n_u, q, q))
        cross = jnp.zeros((T, n_u, q, q))        # block (t, t-1); t = 0 unused
        for d, so, sp in groups:
            own = own + d[:, :, None, None] * (so[:, :, None] * so[:, None, :])
            # period t's row also acts on block t-1, and t = 0 has no predecessor
            own = own.at[:-1].add(d[1:, :, None, None] * (sp[:, :, None] * sp[:, None, :]))
            cross = cross - d[:, :, None, None] * (so[:, :, None] * sp[:, None, :])
        for d, C in within:
            own = own + jnp.einsum("tik,ikq,ikp->tiqp", d, C, C)
        own = own + jax.vmap(jax.vmap(jnp.diag))(d_box_u + reg)
        B = (jnp.einsum("tiqp,ts->itqsp", own, I_T)
             + jnp.einsum("tiqp,ts->itqsp", cross, Sub)
             + jnp.einsum("tiqp,ts->isptq", cross, Sub))
        return B.reshape(n_u, q * T, q * T)

    def solve_units(chol, s, b):
        """``H0^-1 b`` on the unit columns: ``(n_units, qT, m)`` in and out.

        ``chol`` is the Cholesky factor of the symmetrically scaled block and
        ``s`` the scaling, so the solve is ``s^-1 (L L')^-1 s^-1 b``.
        """
        def one(lu, sc, bb):
            return jsl.lu_solve(lu, bb / sc[:, None]) / sc[:, None]
        return jax.vmap(one)(chol, s, b)

    def factor(D: chex.Array, reg):
        """Factor the low-rank system for one set of barrier weights.

        Returns the per-unit Cholesky factors and their scaling, the two pieces
        of ``X = H0^-1 U`` (unit rows dense, shed rows block-diagonal in ``t``),
        the columns ``U`` themselves, the shed diagonal, and the LU of the scaled
        capacitance matrix.  None of it depends on a right-hand side.
        """
        d_line, groups, within, d_box = split(D)
        d_box_u = d_box[:, cols_flat].reshape(T, n_u, q)
        d_shed = d_box[:, o_s: o_s + n_b] + reg                        # (T, n_b)

        B = unit_blocks(groups, within, d_box_u, reg)
        s = jnp.sqrt(jax.vmap(jnp.diag)(B))                            # (n_u, qT)
        Bs = B / s[:, :, None] / s[:, None, :]
        chol = jax.vmap(jsl.lu_factor)(Bs)                             # pivoted LU per unit

        sq = jnp.sqrt(d_line)                                          # (T, k)
        U_unit = jnp.concatenate(
            [jnp.einsum("jiq,tj->tiqj", M_loc, sq),
             jnp.broadcast_to(a_loc[None, :, :, None], (T, n_u, q, 1))], -1)  # (T, n_u, q, k+1)
        U_shed = jnp.concatenate(
            [jnp.einsum("jb,tj->tbj", M_shed, sq),
             jnp.broadcast_to(a_shed[None, :, None], (T, n_b, 1))], -1)      # (T, n_b, k+1)

        # X_unit[i, (t, q), (t', j)] = delta_tt' U_unit[t, i, q, j], then H0^-1
        R = jnp.einsum("tiqj,ts->itqsj", U_unit, I_T).reshape(n_u, q * T, r)
        X_unit = solve_units(chol, s, R)                               # (n_u, qT, r)
        X_shed = U_shed / d_shed[:, :, None]                           # (T, n_b, k+1)

        Cap = jnp.einsum("tiqj,itqr->tjr", U_unit,
                         X_unit.reshape(n_u, T, q, r)).reshape(r, r)
        Cap_shed = jnp.einsum("tbj,tbl->tjl", U_shed, X_shed)          # (T, k+1, k+1)
        Cap = Cap + jnp.einsum("tjl,ts->tjsl", Cap_shed, I_T).reshape(r, r)
        S = Cap + jnp.diag(cmask)
        sS = jnp.sqrt(jnp.maximum(jnp.abs(jnp.diag(S)), 1e-300))
        lu = jsl.lu_factor(S / sS[:, None] / sS[None, :])
        if not free:
            return B, chol, s, X_unit, X_shed, U_unit, U_shed, d_shed, lu, sS

        # Free columns: rank every column by the stiff line curvature acting on
        # it over its own curvature, keep the top `n_free` in a pivoted block
        # with the Schur columns, and eliminate only the rest through H0^-1.
        # `R_shed` is the shed rows of U in the (T*n_b, r) layout of `R`.
        R_shed = jnp.einsum("tbj,ts->tbsj", U_shed, I_T).reshape(T * n_b, r)
        score_u = ((R ** 2) * cmask).sum(-1) / jax.vmap(jnp.diag)(B)     # (n_u, qT)
        score_s = ((R_shed ** 2) * cmask).sum(-1) / d_shed.reshape(-1)  # (T*n_b,)
        F_u = (lax.top_k(score_u.max(-1), n_fu)[1] if n_fu
               else jnp.zeros((0,), jnp.int32))
        F_s = (lax.top_k(score_s, n_fs)[1] if n_fs
               else jnp.zeros((0,), jnp.int32))
        mask_u = jnp.ones((n_u,)).at[F_u].set(0.0)
        mask_s = jnp.ones((T * n_b,)).at[F_s].set(0.0).reshape(T, n_b)
        if local:
            # A free unit's own block may be singular to round-off (module
            # docstring, "Local border": the scaled block is the all-ones
            # matrix when its coupling row binds), so its rows of H0^-1 U are
            # inf.  Those rows are masked out of every sum below, but inf * 0
            # is NaN, so they are removed by selection rather than by weight.
            # Measured 2026-09-17 on a `case813nem` cell with a binding reserve
            # price: 31 of 60 Newton directions NaN before this line, none
            # after.
            X_unit = jnp.where(mask_u[:, None, None] > 0, X_unit, 0.0)
        # the capacitance over the eliminated columns only
        Cap_B = jnp.einsum("tiqj,itqr,i->tjr", U_unit,
                           X_unit.reshape(n_u, T, q, r), mask_u).reshape(r, r)
        Cap_B = Cap_B + jnp.einsum("tjl,ts->tjsl",
                                   jnp.einsum("tbj,tbl,tb->tjl", U_shed, X_shed, mask_s),
                                   I_T).reshape(r, r)
        S_B = Cap_B + jnp.diag(cmask)
        # [[B_F, 0, R_F], [0, diag(d_F), R_sF], [R_F', R_sF', -S_B]]
        B_F = jnp.einsum("iab,ij->iajb", B[F_u], jnp.eye(n_fu)).reshape(n_fu * qT, n_fu * qT)
        R_F = R[F_u].reshape(n_fu * qT, r)
        d_F = d_shed.reshape(-1)[F_s]
        R_sF = R_shed[F_s]
        if local:
            # Local-border arrowhead (module docstring, "Local border"): the
            # free units' top-left is diagonal in the box weights alone, and
            # every `within` row of a free unit is a border column of its
            # own unit, sqrt(d) c on that unit's q rows.
            if groups or not within:
                raise ValueError("the local-border arrowhead needs every unit row inside one "
                                 "period and at least one such row: pass ramp rows as `within`")
            d_top = jnp.concatenate([(d_box_u[0] + reg)[F_u].reshape(-1), d_F])   # (n_top,), > 0
            R_top = jnp.concatenate([R_F, R_sF], 0)                                # (n_top, r)
            s_top = jnp.sqrt(d_top)
            R_pre = R_top / s_top[:, None]
            diag_S = jnp.abs(jnp.diag(S_B))
            s_S = jnp.where(diag_S > 0, jnp.sqrt(jnp.maximum(diag_S, 1e-300)),
                            jnp.maximum(jnp.sqrt(jnp.sum(R_pre ** 2, 0)), 1e-300))
            R_t = R_pre / s_S[None, :]
            S_t = S_B / s_S[:, None] / s_S[None, :]
            L = jnp.concatenate(
                [jnp.swapaxes(jnp.sqrt(d[0])[F_u][:, :, None] * C[F_u], 1, 2) for d, C in within],
                -1)                                                                # (n_fu, q, n_loc)
            L_t = L / s_top[: n_fu * q].reshape(n_fu, q)[:, :, None]
            return (B, chol, s, X_unit, X_shed, U_unit, U_shed, d_shed, lu, sS,
                    F_u, F_s, mask_u, mask_s,
                    (s_top, s_S, R_t, S_t, L_t) + local_arrow_factor(L_t, R_t, S_t), None)
        if arrow:
            # [[diag(d_top), R_top], [R_top', -S_B]] scaled to [[I, R~], [R~', -S~]];
            # R~ = Q T, and the core [[I, T], [T', -S~]] takes the pivoted LU.
            d_top = jnp.concatenate([B[F_u, 0, 0], d_F])                  # (n_top,), > 0
            R_top = jnp.concatenate([R_F, R_sF], 0)                       # (n_top, r)
            s_top = jnp.sqrt(d_top)
            # a Schur column with nothing eliminated behind it (every column
            # free) has a zero diagonal in S_B: scale it by its border
            # column's norm instead, so the scaled system stays finite
            R_pre = R_top / s_top[:, None]
            diag_S = jnp.abs(jnp.diag(S_B))
            s_S = jnp.where(diag_S > 0, jnp.sqrt(jnp.maximum(diag_S, 1e-300)),
                            jnp.maximum(jnp.sqrt(jnp.sum(R_pre ** 2, 0)), 1e-300))
            R_t = R_pre / s_S[None, :]
            S_t = S_B / s_S[:, None] / s_S[None, :]
            Q, Tm = householder_qr(R_t)                                   # (n_top, p), (p, r)
            core = jnp.concatenate([
                jnp.concatenate([jnp.eye(Q.shape[1]), Tm], 1),
                jnp.concatenate([Tm.T, -S_t], 1)], 0)
            lu_core = jsl.lu_factor(core)
            return (B, chol, s, X_unit, X_shed, U_unit, U_shed, d_shed, lu, sS,
                    F_u, F_s, mask_u, mask_s, (Q, lu_core, s_top, s_S, R_t, S_t), None)
        z_us = jnp.zeros((n_fu * qT, n_fs))
        M = jnp.concatenate([
            jnp.concatenate([B_F, z_us, R_F], 1),
            jnp.concatenate([z_us.T, jnp.diag(d_F), R_sF], 1),
            jnp.concatenate([R_F.T, R_sF.T, -S_B], 1)], 0)
        sM = jnp.sqrt(jnp.maximum(jnp.abs(jnp.diag(M)), 1e-300))
        lu_aug = lu_factor_aug(M / sM[:, None] / sM[None, :])
        return (B, chol, s, X_unit, X_shed, U_unit, U_shed, d_shed, lu, sS,
                F_u, F_s, mask_u, mask_s, lu_aug, sM)

    def local_arrow_factor(L_t, R_t, S_t):
        """Two-level factorisation of the scaled arrowhead with a local border,

            [[I, L, R], [L', -I, 0], [R', 0, -S]]      L (n_top, m) block-sparse,

        ``L`` holding each free unit's own rows (``n_loc`` columns on its
        ``q`` rows, nothing elsewhere) and ``R`` the ``r`` Schur columns.
        A batched full QR of every unit's ``(q, n_loc)`` block, ``L_i = Q_i
        [T_i; 0]``, changes that unit's frame: its first ``p = min(q,
        n_loc)`` rows are the pivot rows of its own border, the other
        ``q - p`` and the shed rows are the rest, ``N``.  In the new frame
        the ``m`` local unknowns couple to the rest only through ``T_lg =
        (Q' R)[pivot rows]``, so they are eliminated exactly through their
        ``(p + n_loc)``-wide local cores ``[[I, T_i], [T_i', -I]]`` (batched
        pivoted LU; ``Z`` is their solve against ``T_lg``), which adds
        ``T_lg' Z_z`` -- a sum of positive semidefinite terms -- to ``S``.
        The rest is the single-border arrowhead of the ``q * T = 1`` shape:
        a thin QR of ``R`` on the ``N`` rows and one ``(p_N + r)``-wide core.
        Work per factorisation is ``O(n_top * r^2)`` plus ``n_fu`` tiny
        solves; nothing of order ``m`` is ever factored densely.
        """
        n_loc = L_t.shape[-1]
        p = min(q, n_loc)
        Q_loc, T_loc = jax.vmap(lambda a: householder_qr(a, full=True))(L_t)   # (n_fu, q, q), (n_fu, p, n_loc)
        R_u = jnp.einsum("iab,iar->ibr", Q_loc, R_t[: n_fu * q].reshape(n_fu, q, r))   # Q_i' R_i
        T_lg = R_u[:, :p, :]                                                    # (n_fu, p, r)
        R_N = jnp.concatenate([R_u[:, p:, :].reshape(-1, r), R_t[n_fu * q:]], 0)
        Q_N, T_g = householder_qr(R_N)                                          # (n_N, p_N), (p_N, r)
        eye_p = jnp.broadcast_to(jnp.eye(p), (n_fu, p, p))
        eye_l = jnp.broadcast_to(jnp.eye(n_loc), (n_fu, n_loc, n_loc))
        K_loc = jnp.concatenate([jnp.concatenate([eye_p, T_loc], 2),
                                 jnp.concatenate([jnp.swapaxes(T_loc, 1, 2), -eye_l], 2)], 1)
        lu_loc = jax.vmap(jsl.lu_factor)(K_loc)
        Z = jax.vmap(jsl.lu_solve)(lu_loc, jnp.concatenate([T_lg, jnp.zeros((n_fu, n_loc, r))], 1))
        S_eff = S_t + jnp.einsum("ipr,ips->rs", T_lg, Z[:, :p, :])
        core = jnp.concatenate([jnp.concatenate([jnp.eye(Q_N.shape[1]), T_g], 1),
                                jnp.concatenate([T_g.T, -S_eff], 1)], 0)
        return Q_loc, T_lg, Q_N, lu_loc, Z, jsl.lu_factor(core)

    def local_arrow_solve(st, a_t, c_loc, b_t):
        """One solve of the factored local-border arrowhead: ``(x, w_loc,
        w)`` for the scaled right-hand sides of the top, the local and the
        Schur rows.  The top rows come back as ``Q z`` plus the part of the
        right-hand side outside the border's span, so a border row sees only
        ``z`` -- the same property the single-border solve rests on."""
        _s_top, _s_S, _R_t, _S_t, L_t, Q_loc, T_lg, Q_N, lu_loc, Z, lu_core = st
        n_loc = L_t.shape[-1]
        p, p_N, n_rest = min(q, n_loc), Q_N.shape[1], q - min(q, n_loc)
        a2 = jnp.einsum("iab,ia->ib", Q_loc, a_t[: n_fu * q].reshape(n_fu, q))    # Q_i' a_i
        a_N = jnp.concatenate([a2[:, p:].reshape(-1), a_t[n_fu * q:]])
        zw0 = jax.vmap(jsl.lu_solve)(lu_loc, jnp.concatenate([a2[:, :p], c_loc], 1))
        rhs_z = Q_N.T @ a_N
        zw = jsl.lu_solve(lu_core, jnp.concatenate(
            [rhs_z, b_t - jnp.einsum("ipr,ip->r", T_lg, zw0[:, :p])]))
        z_g, w = zw[:p_N], zw[p_N:]
        x_N = Q_N @ z_g + (a_N - Q_N @ rhs_z)
        zw_loc = zw0 - jnp.einsum("ikr,r->ik", Z, w)
        x_u = jnp.einsum("iab,ib->ia", Q_loc, jnp.concatenate(
            [zw_loc[:, :p], x_N[: n_fu * n_rest].reshape(n_fu, n_rest)], 1))
        return jnp.concatenate([x_u.reshape(-1), x_N[n_fu * n_rest:]]), zw_loc[:, p:], w

    def local_arrow_residual(st, a_t, c_loc, b_t, x_t, w_loc, w):
        """The three residuals of the scaled system for the inner refinement."""
        _s_top, _s_S, R_t, S_t, L_t = st[:5]
        x_u = x_t[: n_fu * q].reshape(n_fu, q)
        r_top = (a_t - x_t - R_t @ w).at[: n_fu * q].add(
            -jnp.einsum("iak,ik->ia", L_t, w_loc).reshape(-1))
        r_loc = c_loc - (jnp.einsum("iak,ia->ik", L_t, x_u) - w_loc)
        return r_top, r_loc, b_t - R_t.T @ x_t + S_t @ w

    def split_rhs(rhs):
        """Solver layout -> ``(unit rows (n_u, qT), shed rows (T, n_b), balance (T,))``."""
        r_x = rhs[:n].reshape(T, nb)
        r_unit = jnp.swapaxes(r_x[:, cols_flat].reshape(T, n_u, q), 0, 1).reshape(n_u, q * T)
        return r_unit, r_x[:, o_s: o_s + n_b], rhs[n:]

    def join(x_unit, x_shed, nu):
        """The inverse of `split_rhs`."""
        x = jnp.zeros((T, nb))
        x = x.at[:, cols_flat].set(
            jnp.swapaxes(x_unit.reshape(n_u, T, q), 0, 1).reshape(T, n_u * q))
        x = x.at[:, o_s: o_s + n_b].set(x_shed)
        return jnp.concatenate([x.ravel(), nu])

    def ut(U_unit, U_shed, x_unit, x_shed):
        """``U' x`` as ``(T, k+1)``: the first ``k`` columns are ``sqrt(d) W' x``,
        the last is ``A x``."""
        return (jnp.einsum("tiqj,itq->tj", U_unit, x_unit.reshape(n_u, T, q))
                + jnp.einsum("tbj,tb->tj", U_shed, x_shed))

    def kmul(state, x_unit, x_shed, nu):
        """The unreduced Newton matrix applied to ``(x, nu)``, matrix-free:
        ``[H0 x + W d W' x + A' nu ; A x]`` in the split layout."""
        B, U_unit, U_shed, d_shed = state[0], state[5], state[6], state[7]
        Ux = ut(U_unit, U_shed, x_unit, x_shed)                        # (T, k+1)
        # W d W' x = U [sqrt(d) W' x ; 0], A' nu = U [0 ; nu]
        coef = Ux.at[:, k].set(nu)
        h_unit = (jnp.einsum("ijk,ik->ij", B, x_unit)
                  + jnp.einsum("tiqj,tj->itq", U_unit, coef).reshape(n_u, q * T))
        h_shed = d_shed * x_shed + jnp.einsum("tbj,tj->tb", U_shed, coef)
        return h_unit, h_shed, Ux[:, k]

    def solve_once_plain(state, r_unit, r_shed, r_nu):
        """One Schur solve in the split layout."""
        _, chol, s, X_unit, X_shed, U_unit, U_shed, d_shed, lu, sS = state
        y_unit = solve_units(chol, s, r_unit[:, :, None])[:, :, 0]
        y_shed = r_shed / d_shed
        b = ut(U_unit, U_shed, y_unit, y_shed).at[:, k].add(-r_nu)
        w = jsl.lu_solve(lu, b.ravel() / sS) / sS
        W = w.reshape(T, k + 1)
        x_unit = y_unit - jnp.einsum("itr,r->it", X_unit, w)
        x_shed = y_shed - jnp.einsum("tbj,tj->tb", X_shed, W)
        return x_unit, x_shed, W[:, k]

    def solve_once_free(state, r_unit, r_shed, r_nu):
        """One solve with the free columns kept in the pivoted block.

        The eliminated columns give ``x_B = y_B - X_B w`` as in the plain
        solve; the free columns and ``w`` come out of the pivoted block
        together, which is what keeps a binding line's flow consistent to
        round-off when the free columns' own curvature is only ``reg``.
        """
        (_, chol, s, X_unit, X_shed, U_unit, U_shed, d_shed, _, _,
         F_u, F_s, mask_u, mask_s, lu_aug, sM) = state
        y_unit = solve_units(chol, s, r_unit[:, :, None])[:, :, 0]
        if local:
            # same as in `factor`: a free unit's rows may be inf, and are
            # overwritten by the block solve below in any case
            y_unit = jnp.where(mask_u[:, None] > 0, y_unit, 0.0)
        y_shed = r_shed / d_shed
        # U_B' y_B, the eliminated columns only
        b = (jnp.einsum("tiqj,itq,i->tj", U_unit, y_unit.reshape(n_u, T, q), mask_u)
             + jnp.einsum("tbj,tb,tb->tj", U_shed, y_shed, mask_s)).at[:, k].add(-r_nu)
        rhs_aug = jnp.concatenate([r_unit[F_u].reshape(-1), r_shed.reshape(-1)[F_s], -b.ravel()])
        if local:
            st = lu_aug
            s_top, s_S = st[0], st[1]
            n_top = n_fu * qT + n_fs
            a_t, b_t = rhs_aug[:n_top] / s_top, rhs_aug[n_top:] / s_S
            c_loc = jnp.zeros((n_fu, st[4].shape[-1]))
            x_t, w_loc, w_t = local_arrow_solve(st, a_t, c_loc, b_t)
            for _ in range(ARROW_REFINE):
                dx, dwl, dw = local_arrow_solve(
                    st, *local_arrow_residual(st, a_t, c_loc, b_t, x_t, w_loc, w_t))
                x_t, w_loc, w_t = x_t + dx, w_loc + dwl, w_t + dw
            sol = jnp.concatenate([x_t / s_top, w_t / s_S])
        elif arrow:
            Q, lu_core, s_top, s_S, R_t, S_t = lu_aug
            n_top = n_fu * qT + n_fs

            p = Q.shape[1]

            def arrow_solve(a_t, b_t):
                a1 = Q.T @ a_t
                zw = jsl.lu_solve(lu_core, jnp.concatenate([a1, b_t]))
                return Q @ zw[:p] + (a_t - Q @ a1), zw[p:]

            a_t = rhs_aug[:n_top] / s_top
            b_t = rhs_aug[n_top:] / s_S
            x_t, w_t = arrow_solve(a_t, b_t)
            # refinement inside the block, see `ARROW_REFINE`
            for _ in range(ARROW_REFINE):
                dx, dw = arrow_solve(a_t - x_t - R_t @ w_t, b_t - R_t.T @ x_t + S_t @ w_t)
                x_t, w_t = x_t + dx, w_t + dw
            sol = jnp.concatenate([x_t / s_top, w_t / s_S])
        else:
            sol = jsl.lu_solve(lu_aug, rhs_aug / sM) / sM
        x_Fu = sol[: n_fu * qT].reshape(n_fu, qT)
        x_Fs = sol[n_fu * qT: n_fu * qT + n_fs]
        w = sol[n_fu * qT + n_fs:]
        W = w.reshape(T, k + 1)
        x_unit = (y_unit - jnp.einsum("itr,r->it", X_unit, w)).at[F_u].set(x_Fu)
        x_shed = (y_shed - jnp.einsum("tbj,tj->tb", X_shed, W)).reshape(-1).at[F_s].set(x_Fs)
        return x_unit, x_shed.reshape(T, n_b), W[:, k]

    solve_once = solve_once_free if free else solve_once_plain

    def apply(state, rhs: chex.Array) -> chex.Array:
        """Solve the factored system for one right-hand side.

        ``rhs`` and the result are in the solver's layout, ``[n variable rows ;
        T balance rows]``, period-major; the unit-major regrouping happens here
        and is undone on the way out.  The plain Schur solve is followed by
        `N_REFINE` refinement steps against the unreduced system.
        """
        r_unit, r_shed, r_nu = split_rhs(rhs)
        x_unit, x_shed, nu = solve_once(state, r_unit, r_shed, r_nu)
        for _ in range(N_REFINE):
            h_unit, h_shed, h_nu = kmul(state, x_unit, x_shed, nu)
            d_unit, d_shed_, d_nu = solve_once(state, r_unit - h_unit, r_shed - h_shed, r_nu - h_nu)
            x_unit, x_shed, nu = x_unit + d_unit, x_shed + d_shed_, nu + d_nu
        return join(x_unit, x_shed, nu)

    return factor, apply


def make_kkt_sced(spec: Dict, Mblk_mon: np.ndarray, Sblk: np.ndarray,
                  n_free: Tuple[int, int] = (0, 0),
                  lu_batching: str = "auto") -> Tuple[Callable, Callable]:
    """The SCED of `clearing`, reduced to ``spec["n_l"]`` monitored lines.

    Row order is `kkt.make_ops`' with ``n_l`` monitored lines: ``[lines: 2*n_l
    per period ; ramp: 2*n_units per period ; +I ; -I]``.  A period block is
    ``[n_units*K segment columns ; n_buses shed columns]``, so a unit's own
    columns are its ``K`` segments and the balance row is all ones.
    """
    T, K, n_u, nb = spec["T"], spec["K"], spec["n_units"], spec["nb"]
    n_l, n = spec["n_l"], spec["n"]
    ramp0, n_ramp = spec["ramp_row0"], spec["n_ramp_rows"]
    cols = np.arange(n_u * K).reshape(n_u, K)
    S_loc = jnp.asarray(_local(np.asarray(Sblk), cols, "Sblk"))     # (n_u, K), all ones

    def split(D):
        line = D[: 2 * n_l * T].reshape(T, 2, n_l).sum(1)
        ramp = D[ramp0: ramp0 + n_ramp].reshape(T, 2, n_u).sum(1)
        box0 = ramp0 + n_ramp
        box = D[box0: box0 + n].reshape(T, nb) + D[box0 + n: box0 + 2 * n].reshape(T, nb)
        # both ramp directions are one operator negated, so their weights add
        return line, [(ramp, S_loc, S_loc)], [], box

    return _make_kkt(spec, cols, n_u * K, np.asarray(Mblk_mon), np.ones(nb), split, n_free,
                     lu_batching)


def make_kkt_relax(spec: Dict, ops: Dict,
                   n_free: Tuple[int, int] = (0, 0),
                   lu_batching: str = "auto") -> Tuple[Callable, Callable]:
    """The relaxed commitment of `relax`, reduced to ``spec["n_l"]`` monitored lines.

    Row order is `relax_kkt.make_ops`' with ``n_l`` monitored lines: ``[lines ;
    cap ; ramp up, down ; w-bounds ; +I ; -I]``.  A unit's own columns are its
    ``K`` segments, its ``u`` and its ``v``; the shut-down indicator is not
    eliminated here -- it stays inside the unit's ``q*T`` block, which is small.
    """
    T, K, n_u, nb = spec["T"], spec["K"], spec["n_units"], spec["nb"]
    n_l, n = spec["n_l"], spec["n"]
    o_g, o_s, o_u, o_v = spec["o_g"], spec["o_s"], spec["o_u"], spec["o_v"]
    cols = np.stack([np.concatenate([o_g + i * K + np.arange(K), [o_u + i, o_v + i]])
                     for i in range(n_u)])                             # (n_u, K+2)
    S = {name: jnp.asarray(_local(np.asarray(ops[name]), cols, name))
         for name in ("Sa", "Sb", "Sc", "Sd", "Se", "Sf")}
    # capacity row (i, kappa) touches its own segment and its unit's u
    C = np.asarray(ops["C"])
    C_loc = np.stack([np.take_along_axis(C[i * K: (i + 1) * K], np.broadcast_to(cols[i], (K, K + 2)), axis=1)
                      for i in range(n_u)])                            # (n_u, K, K+2)
    back = np.zeros_like(C)
    for i in range(n_u):
        np.put_along_axis(back[i * K: (i + 1) * K], np.broadcast_to(cols[i], (K, K + 2)), C_loc[i], axis=1)
    if not np.array_equal(back, C):
        raise AssertionError("C has nonzeros outside its unit's columns")
    C_loc = jnp.asarray(C_loc)

    def split(D):
        i = 0
        line = D[i: i + 2 * n_l * T].reshape(T, 2, n_l).sum(1); i += 2 * n_l * T
        cap = D[i: i + n_u * K * T].reshape(T, n_u, K); i += n_u * K * T
        ramp = D[i: i + 2 * n_u * T].reshape(T, 2, n_u); i += 2 * n_u * T
        wbnd = D[i: i + 2 * n_u * T].reshape(T, 2, n_u).sum(1); i += 2 * n_u * T
        box = D[i: i + n].reshape(T, nb) + D[i + n: i + 2 * n].reshape(T, nb)
        # up row: Sa on t, Sb on t-1; down row: -Sc on t, -(-Sd) on t-1; the two
        # w bounds are one operator negated and share a weight
        groups = [(ramp[:, 0], S["Sa"], S["Sb"]),
                  (ramp[:, 1], -S["Sc"], -S["Sd"]),
                  (wbnd, S["Se"], S["Sf"])]
        return line, groups, [(cap, C_loc)], box

    return _make_kkt(spec, cols, o_s, np.asarray(ops["M"]), np.asarray(ops["a"]), split, n_free,
                     lu_batching)


def make_kkt_ancillary(spec: Dict, M_mon: np.ndarray, S: np.ndarray, CS: np.ndarray,
                       RD: np.ndarray, a_row: np.ndarray,
                       n_free: Tuple[int, int] = (0, 0),
                       lu_batching: str = "auto") -> Tuple[Callable, Callable]:
    """The joint energy-reserve clearing of `envs.ancillary.clearing`, reduced
    to ``spec["n_l"]`` monitored lines (2026-09-17).

    Row order is that operator's: ``[lines +; lines -; ramp +; ramp -; CS;
    RD; +I; -I]``, columns ``[g (n_units * K); r (n_units * P); s (n_buses);
    s_res (P)]``, one period.  A unit's own columns are its ``K`` segments
    and its ``P`` reserve columns; the capacity-sharing row (CS) couples them
    -- the one place energy and reserve meet -- so the unit block is of order
    ``q = K + P`` and, with ``reg`` for the only other curvature of an
    interior column, is singular to round-off in the ``q - 1`` directions
    orthogonal to ones whenever the row binds (measured on `case813nem`:
    eigenvalues of the scaled block ``-6e-16, -2e-17, 3``).  That is why the arrowhead here takes
    the local border (`_make_kkt`, ``arrow_local``) rather than a scaled LU
    of the block.

    Two things this shape has that the day-ahead one does not, both handed
    to `_make_kkt` as what they algebraically are.  The ``P`` requirement
    rows (RD) each touch every unit's column of one product and that
    product's shortfall column: a low-rank inequality row, so they are
    appended to the monitored lines (``W = [M; RD]``, ``k = n_l + P``) and
    take a ``+1`` on the capacitance diagonal like a line does.  The ``P``
    shortfall columns are touched by RD and the box only: diagonal columns,
    so they are counted with the shed columns (``n_buses + P`` diagonal
    columns from ``s`` on; the two blocks are adjacent).  The balance row is
    ones on ``g`` and ``s``.  The plain Schur route (``n_free = (0, 0)``)
    eliminates every unit through its ``q x q`` block and is kept as the
    negative control; the market's default is every column free.
    """
    K, P, n_u, n_b = spec["K"], spec["n_prod"], spec["n_units"], spec["n_buses"]
    n, n_l = spec["n"], spec["n_l"]
    col0, row0 = spec["col0"], spec["row0"]
    cols = np.stack([np.concatenate([col0["g"] + i * K + np.arange(K),
                                     col0["r"] + i * P + np.arange(P)])
                     for i in range(n_u)])                                 # (n_u, K + P)
    S_loc = jnp.asarray(_local(np.asarray(S), cols, "S")[:, None, :])     # (n_u, 1, q)
    CS_loc = jnp.asarray(_local(np.asarray(CS), cols, "CS")[:, None, :])  # (n_u, 1, q)
    W = np.vstack([np.asarray(M_mon), np.asarray(RD)])                    # (n_l + P, n)
    if RD.shape != (P, n) or np.any(RD[:, : col0["r"]] != 0) or np.any(RD[:, col0["s"]: col0["s_res"]] != 0):
        raise AssertionError("RD must touch the reserve and shortfall columns only")
    view = dict(T=1, nb=n, n_units=n_u, n_buses=n_b + P, n_l=n_l + P)

    def split(D):
        line = D[: 2 * n_l].reshape(1, 2, n_l).sum(1)
        rd = D[row0["rd"]: row0["rd"] + P].reshape(1, P)
        ramp = (D[row0["ramp_up"]: row0["ramp_up"] + n_u]
                + D[row0["ramp_dn"]: row0["ramp_dn"] + n_u]).reshape(1, n_u, 1)
        cs = D[row0["cs"]: row0["cs"] + n_u].reshape(1, n_u, 1)
        box = (D[row0["box_hi"]: row0["box_hi"] + n]
               + D[row0["box_lo"]: row0["box_lo"] + n]).reshape(1, n)
        # every unit row acts inside the one period: all are `within`, none `groups`
        return jnp.concatenate([line, rd], 1), [], [(ramp, S_loc), (cs, CS_loc)], box

    return _make_kkt(view, cols, col0["s"], W, np.asarray(a_row), split, n_free, lu_batching,
                     arrow_local=True)


def make_ops_ancillary(spec: Dict, M_mon: np.ndarray) -> Tuple[Callable, Callable]:
    """``(Gx, GTy)`` for `ipm.make_solver` on the ancillary clearing's rows,
    without the ``(m, n)`` matrix: the monitored-line block is a ``(n_l, n)``
    product, the unit rows are segment sums, the requirement rows product
    sums, the box the identity.  Same values as the dense products to
    round-off; used on the low-rank route only, the dense route keeps the
    matrix it always had.
    """
    K, P, n_u, n = spec["K"], spec["n_prod"], spec["n_units"], spec["n"]
    n_l, col0, row0 = spec["n_l"], spec["col0"], spec["row0"]
    g0, r0, sr0 = col0["g"], col0["r"], col0["s_res"]
    Mj = jnp.asarray(M_mon)

    def Gx(v):
        lines = Mj @ v
        g = v[g0: g0 + n_u * K].reshape(n_u, K).sum(1)
        rsv = v[r0: r0 + n_u * P].reshape(n_u, P)
        cs = g + rsv.sum(1)
        rd = -rsv.sum(0) - v[sr0: sr0 + P]
        return jnp.concatenate([lines, -lines, g, -g, cs, rd, v, -v])

    def GTy(y):
        y_line = y[: n_l] - y[n_l: 2 * n_l]
        y_ramp = y[row0["ramp_up"]: row0["ramp_up"] + n_u] - y[row0["ramp_dn"]: row0["ramp_dn"] + n_u]
        y_cs = y[row0["cs"]: row0["cs"] + n_u]
        y_rd = y[row0["rd"]: row0["rd"] + P]
        out = Mj.T @ y_line + (y[row0["box_hi"]: row0["box_hi"] + n] - y[row0["box_lo"]: row0["box_lo"] + n])
        out = out.at[g0: g0 + n_u * K].add(jnp.repeat(y_ramp + y_cs, K))
        out = out.at[r0: r0 + n_u * P].add(jnp.repeat(y_cs, P) - jnp.tile(y_rd, n_u))
        return out.at[sr0: sr0 + P].add(-y_rd)

    return Gx, GTy
