"""Block-tridiagonal Newton system for the multi-period SCED, and matrix-free G.

The dense route forms ``H = G'DG`` (m*n^2 flops) and factors an (n+T)^2 matrix.
At T=24, K=3 that is 6.1e11 flops per Newton step and 820 MB for G alone, which
puts one clearing at 27.9 s on GPU.

With the block-per-period variable ordering ``clearing.clear`` builds its
arrays in -- ``x = [block_1, ..., block_T]``, ``block_t = [g[:, :, t] ; s[:,
t]]`` -- energy balance, line limits and box bounds are block-diagonal in
``t``, and **ramp is the only coupling and reaches exactly one period back**.
So

    K = [[H, A'], [A, 0]]   reordered as  [x_1, nu_1, x_2, nu_2, ...]

is block-tridiagonal with T blocks of order ``nb + 1``, ``nb = n_units*K +
n_buses``.  A block Thomas sweep costs T*(nb+1)^3 instead of (n+T)^3 and never
forms G: 1.68 s per clearing, 16.6x faster, with G's 820 MB gone.

Both sweeps are ``lax.scan`` over T with a fixed trip count, so the batch cost
stays independent of batch content -- measured 0.99x for one hard element among
sixteen, against MPAX's 24.25x.

Where each block comes from, given that row layout
(``[lines: 2*n_l per period ; ramp: 2*n_u per period ; +I ; -I]``):

    lines  -> block diagonal:  Mblk' (D+ + D-) Mblk
    box    -> block diagonal:  diag(D_hi + D_lo)
    ramp   -> tridiagonal:     Sblk' W Sblk on (t,t) and (t-1,t-1),
                               -Sblk' W Sblk on (t,t-1) and (t-1,t)
    A      -> one all-ones row per period, block diagonal
"""
from typing import Callable, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jax import lax


def _split_barrier(spec: Dict, D: chex.Array):
    """Slice the (m,) barrier weights into the per-period pieces blocks need."""
    n_l, n_u, nb, T = spec["n_l"], spec["n_units"], spec["nb"], spec["T"]
    n = nb * T
    ramp0, n_ramp = spec["ramp_row0"], spec["n_ramp_rows"]
    line = D[: 2 * n_l * T].reshape(T, 2, n_l)
    ramp = D[ramp0: ramp0 + n_ramp].reshape(T, 2, n_u)
    box0 = ramp0 + n_ramp
    hi = D[box0: box0 + n].reshape(T, nb)
    lo = D[box0 + n: box0 + 2 * n].reshape(T, nb)
    return line.sum(1), ramp.sum(1), hi + lo


def _build_blocks(spec: Dict, Mblk, Sblk, D, reg):
    """Diagonal and sub-diagonal blocks, each (T, nb+1, nb+1)."""
    nb, T = spec["nb"], spec["T"]
    d_line, d_ramp, d_box = _split_barrier(spec, D)

    Hd = jnp.einsum("li,tl,lj->tij", Mblk, d_line, Mblk)
    Hd = Hd + jax.vmap(jnp.diag)(d_box) + reg * jnp.eye(nb)

    # period t's ramp rows act on blocks t and t-1
    R = jnp.einsum("ui,tu,uj->tij", Sblk, d_ramp, Sblk)
    Hd = Hd + R
    Hd = Hd.at[:-1].add(R[1:])
    Off = -R

    one = jnp.ones((T, nb, 1))
    Diag = jnp.concatenate(
        [jnp.concatenate([Hd, one], 2),
         jnp.concatenate([jnp.swapaxes(one, 1, 2), jnp.zeros((T, 1, 1))], 2)], 1)
    Ofull = jnp.concatenate(
        [jnp.concatenate([Off, jnp.zeros((T, nb, 1))], 2),
         jnp.zeros((T, 1, nb + 1))], 1)
    return Diag, Ofull


def make_ops(spec: Dict, Mblk: chex.Array, Sblk: chex.Array) -> Tuple[Callable, Callable]:
    """``(Gx, GTy)`` without ever forming G.  Uses only Mblk and Sblk (tens of kB)."""
    nb, T, n_u, n_l = spec["nb"], spec["T"], spec["n_units"], spec["n_l"]
    n = nb * T
    ramp0, n_ramp = spec["ramp_row0"], spec["n_ramp_rows"]

    def Gx(v):
        """``G @ v``: the ``(n,)`` variable vector in, the ``(m,)`` row vector out.

        Rows come out in the order `clearing` builds ``h`` in -- both line
        directions, both ramp directions, then ``+I`` and ``-I`` for the box.  The
        ramp rows are the one-period difference of each unit's segment sum, which
        is the only term that reaches outside its own period; period 0's
        predecessor is zero because `clearing` moved the day boundary onto the
        right-hand side.
        """
        V = v[:n].reshape(T, nb)
        line = V @ Mblk.T
        Sv = V @ Sblk.T
        dS = Sv - jnp.concatenate([jnp.zeros((1, n_u)), Sv[:-1]], 0)
        return jnp.concatenate([
            jnp.concatenate([line, -line], 1).ravel(),
            jnp.concatenate([dS, -dS], 1).ravel(),
            v, -v])

    def GTy(y):
        """``G' @ y``, the adjoint of `Gx`: ``(m,)`` in, ``(n,)`` out.

        Each row group is sliced back out in the order `Gx` wrote it, and the two
        directions of a group are differenced because they are the same operator
        with opposite signs.
        """
        line = y[: 2 * n_l * T].reshape(T, 2, n_l)
        ramp = y[ramp0: ramp0 + n_ramp].reshape(T, 2, n_u)
        box0 = ramp0 + n_ramp
        out = (line[:, 0] - line[:, 1]) @ Mblk
        w = ramp[:, 0] - ramp[:, 1]
        out = out + w @ Sblk
        # period t's ramp row also touches block t-1, with the opposite sign; the
        # shift is written on the output because the last period has no successor
        out = out.at[:-1].add(-(w[1:] @ Sblk))
        return out.ravel() + y[box0: box0 + n] - y[box0 + n: box0 + 2 * n]

    return Gx, GTy


def make_kkt(spec: Dict, Mblk: chex.Array, Sblk: chex.Array) -> Tuple[Callable, Callable]:
    """``(factor, apply)`` for ``ipm.make_solver``, block-tridiagonal."""
    nb, T = spec["nb"], spec["T"]
    n = nb * T

    def factor(D, reg):
        """Factor the block-tridiagonal KKT matrix for one set of barrier weights.

        Args:
            D: ``(m,)`` barrier weights ``lam / s``, in the row order `Gx` uses.
            reg: primal regularisation, added to every diagonal block.

        Returns:
            The per-period state ``(lu, piv, C, L)``, each stacked over ``t``:
            the LU factors of the modified diagonal block ``M_t = B_t - L_t
            C_{t-1}``, that period's ``C_t = M_t^-1 U_t``, and the sub-diagonal
            block ``L_t`` that the forward substitution in `apply` re-uses.
        """
        Diag, Off = _build_blocks(spec, Mblk, Sblk, D, reg)
        nbp = Diag.shape[-1]
        # `Off[t]` is the (t, t-1) block, so `Lo` is it with period 0's zeroed --
        # period 0 has no predecessor -- and `Up[t]`, the (t, t+1) block, is
        # `Off[t+1]` transposed, the matrix being symmetric.  The last period has
        # no successor, hence the zero block appended.
        Lo = Off.at[0].set(jnp.zeros((nbp, nbp)))
        Up = jnp.concatenate([jnp.swapaxes(Off[1:], 1, 2), jnp.zeros((1, nbp, nbp))], 0)

        # forward sweep; C_t and the LU of M_t do not depend on the rhs, so one
        # Newton step factors once and solves twice (predictor + corrector)
        def fwd(Cp, xs):
            """One period of the block Thomas sweep.  The carry is ``C_{t-1}``."""
            B, L, U = xs
            lu, piv = jsl.lu_factor(B - L @ Cp)
            C = jsl.lu_solve((lu, piv), U)
            return C, (lu, piv, C, L)

        _, state = lax.scan(fwd, jnp.zeros((nbp, nbp)), (Diag, Lo, Up))
        return state

    def apply(state, rhs):
        """Solve the factored system for one right-hand side.

        ``rhs`` and the result are in the solver's own layout, ``[n variable rows ;
        T balance rows]``; the interleaving into ``[x_t ; nu_t]`` blocks that the
        sweep is written in happens here and is undone on the way out, so `ipm`
        never sees the reordering.  ``state`` comes from `factor` and carries no
        right-hand side, which is why the same one serves both the predictor and
        the corrector solve of a Newton step.
        """
        lus, pivs, Cs, Ls = state
        r = jnp.concatenate([rhs[:n].reshape(T, nb), rhs[n:].reshape(T, 1)], 1)

        def fwd(dp, xs):
            """Forward substitution, ``d_t = M_t^-1 (r_t - L_t d_{t-1})``.

            The carry is ``d_{t-1}``, zero before the first period.
            """
            lu, piv, L, ri = xs
            d = jsl.lu_solve((lu, piv), ri - L @ dp)
            return d, d

        _, ds = lax.scan(fwd, jnp.zeros(r.shape[-1]), (lus, pivs, Ls, r))

        def bwd(znext, xs):
            """Back substitution, ``z_t = d_t - C_t z_{t+1}``.

            Runs with ``reverse=True``, so the carry is ``z_{t+1}``, zero after
            the last period.
            """
            C, d = xs
            z = d - C @ znext
            return z, z

        _, zs = lax.scan(bwd, jnp.zeros(r.shape[-1]), (Cs, ds), reverse=True)
        return jnp.concatenate([zs[:, :nb].ravel(), zs[:, nb]])

    return factor, apply
