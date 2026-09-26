"""Block-tridiagonal Newton system for the relaxed commitment, minimum
up-/downtime dropped.

`kkt.py` does this for the fixed-commitment dispatch, where the commitment is a
constant and the block is ``[g ; s]``.  The relaxed commitment carries ``u`` and
the start-up indicator as variables too, and still fits one block-tridiagonal
sweep once two things are done.

**Minimum up-/downtime rows come out of the linear program.**  They are the only
rows that reach further back than one period, and keeping them would force a
bordered rather than a block-tridiagonal scheme.  Dropping them leaves minimum
up-/downtime violations possible in the rounded commitment, which the
environment already measures and reports as a `costs` quantity rather than
enforcing here.

**The shut-down indicator is eliminated.**  The logical identity
``u_t - u_{t-1} = v - w`` lets ``w = v - (u_t - u_{t-1})`` substitute out, taking
the equality row with it.  What is left is one balance row per period, the shape
`solvers.ipm.make_solver` and the sweep below already assume.  The bounds on the
eliminated variable become two ordinary inequality rows, both of which reach
exactly one period back.

Block per period::

    x_t = [ g[:, :, t] ; s[:, t] ; u[:, t] ; v[:, t] ]      nb = n_u*K + n_b + 2*n_u

Row layout, which `_split_barrier` below and the caller must agree on::

    lines  2*n_l per period    block diagonal
    cap    n_u*K per period    block diagonal   g <= width * u
    ramp   2*n_u per period    tridiagonal      the two ramp allowances
    wbnd   2*n_u per period    tridiagonal      0 <= v - (u_t - u_{t-1}) <= 1
    box    +I then -I          block diagonal

The ramp rows are where this differs from `kkt.py` beyond block width.  There the
two directions are the same operator with opposite signs, so one ``Sblk``
suffices.  Here the start-up allowance ``p_min * v`` sits on the up row and the
shut-down allowance ``p_max * w`` on the down row, and after eliminating ``w``
the two directions carry different coefficients on ``u`` as well.  Four operators
are therefore needed rather than one, and the cross-period block is no longer a
scalar multiple of the diagonal one.

**The start-up indicator is eliminated from the factored block as well**, which
is what makes the factored block narrower than the full one.  ``v`` is reached
only by rows that are per-unit -- the two ramp rows, the two bounds on the
eliminated ``w``, and its own box -- while the line rows, the capacity rows and
the balance row all carry a zero coefficient on it.  Three consequences, and the
elimination rests on all three: ``H_vv`` is diagonal, so the pivot is a scalar;
``v_t`` couples only to ``(g, u)_t`` and ``(g, u)_{t-1}``, so the fill-in stays
inside the tridiagonal pattern; and the balance row does not touch ``v``, so the
border stays out of it.  ``g``, ``s`` and ``u`` are all coupled densely through
the PTDF and none of the three holds for them, which is why only ``v`` comes out.

The elimination is exact, and better conditioned than factoring the full block:
smaller factored blocks accumulate less round-off, and the pivot is a
well-conditioned positive diagonal.
"""
from typing import Callable, Dict, Tuple

import chex
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
from jax import lax


def _split_barrier(spec: Dict, D: chex.Array):
    """Slice the (m,) barrier weights into the per-period pieces the blocks need."""
    n_l, n_u, nb, T, K = spec["n_l"], spec["n_units"], spec["nb"], spec["T"], spec["K"]
    n = nb * T
    i = 0
    line = D[i: i + 2 * n_l * T].reshape(T, 2, n_l); i += 2 * n_l * T
    cap = D[i: i + n_u * K * T].reshape(T, n_u * K); i += n_u * K * T
    ramp = D[i: i + 2 * n_u * T].reshape(T, 2, n_u); i += 2 * n_u * T
    wbnd = D[i: i + 2 * n_u * T].reshape(T, 2, n_u); i += 2 * n_u * T
    hi = D[i: i + n].reshape(T, nb); i += n
    lo = D[i: i + n].reshape(T, nb)
    return line.sum(1), cap, ramp, wbnd, hi + lo


def _make_struct(spec: Dict, ops: Dict) -> Dict:
    """Per-unit support of the sparse operators, read off the operators themselves.

    ``C`` and ``Sa``..``Sf`` carry two or three nonzeros per row -- a unit's own
    ``g`` segments, its ``u`` and its ``v`` -- so forming ``S' D S`` as a dense
    (nb, nb) contraction does nb/q times the arithmetic it needs.  Scattering the
    per-unit q*q blocks instead is exact: the blocks agree with the dense
    contraction to machine precision, with identical sparsity, and cost far less
    to assemble than the one operator that is genuinely dense (``M``, through
    the PTDF).

    The support is extracted from the operators and checked here rather than
    written out by hand.  A hand-written column list is the one way this can go
    wrong quietly -- it would drift from the block layout `relax.make_relax`
    builds, and every downstream number would still look plausible.
    """
    n_u, K, nb = spec["n_units"], spec["K"], spec["nb"]
    o_g, o_u, o_v = spec["o_g"], spec["o_u"], spec["o_v"]
    q = K + 2

    # Both index into the **columns of one period block**, i.e. positions in
    # ``0..nb-1`` under the layout `relax.make_relax` fixes with ``o_g``..``o_v``.
    # ``cols[i]`` is unit ``i``'s K segment columns followed by its ``u`` and its
    # ``v``, so it is the support of one row of Sa..Sf.  ``rows_c[j]`` is the
    # support of capacity row ``j``: its own ``g`` column and the ``u`` of the unit
    # that segment belongs to, ``j // K``.
    cols = np.stack([np.concatenate([o_g + i * K + np.arange(K), [o_u + i, o_v + i]])
                     for i in range(n_u)]).astype(np.int32)          # (n_u, q)
    rows_c = np.stack([np.array([o_g + j, o_u + j // K])
                       for j in range(n_u * K)]).astype(np.int32)    # (n_u*K, 2)

    def pick(name, idx):
        """Gather ``ops[name]`` at ``idx``, and check nothing was left behind.

        ``idx`` is one set of block-local column indices per row of the operator,
        so the result is ``(n_rows, idx.shape[1])`` of nonzeros.  Scattering them
        back has to reproduce the operator bit for bit; if it does not, that
        operator has a nonzero outside the support assumed here and the structured
        assembly would drop it silently.
        """
        X = np.asarray(ops[name], np.float64)
        V = np.take_along_axis(X, idx, axis=1)
        back = np.zeros_like(X)
        np.put_along_axis(back, idx, V, axis=1)
        if not np.array_equal(back, X):
            raise AssertionError(
                f"{name} has nonzeros outside the per-unit support; the block "
                "layout changed and the structured assembly no longer applies")
        return V

    S = {k: pick(k, cols) for k in ("Sa", "Sb", "Sc", "Sd", "Se", "Sf")}
    Vc = pick("C", rows_c)

    nr = nb - n_u                       # retained columns [g ; s ; u]; v is the tail
    if o_v != nr:
        raise AssertionError(
            f"v is not the tail of the block (o_v={o_v}, nr={nr}); the reduced "
            "assembly slices on that and would silently mix columns")
    for k in ("Sb", "Sd", "Sf"):        # the t-1 operators must not reach v
        if np.any(S[k][:, q - 1]):
            raise AssertionError(f"{k} is nonzero on v; the pivot is no longer diagonal")
    M = np.asarray(ops["M"], np.float64)
    if np.any(M[:, nr:]):
        raise AssertionError("M is nonzero on v; the line rows would enter the pivot")

    def outer(x, y):
        """Row-wise outer products: two ``(n_rows, q)`` arrays -> ``(n_rows, q, q)``.

        One row's contribution to ``S' D S`` with the barrier weight factored out,
        so `_build_blocks` only has to scale these by ``D`` and sum them.
        """
        return jnp.asarray(x[:, :, None] * y[:, None, :])

    def scatter_ix(idx, a, b):
        """Flat ``(row, col)`` index pair for scattering one ``(a, b)`` block per row.

        ``idx`` holds block-local column indices, so the pair returned addresses
        the ``(nr, nr)`` reduced Hessian and
        ``H.at[:, r, c].add(blocks.reshape(T, -1))`` drops each row's block at that
        row's own columns.  Repeated indices accumulate under ``.add``, which is
        what lets several rows land on the same entry -- the ``u`` column that a
        unit's K capacity rows share, for one -- rather than overwrite it.
        """
        r = np.broadcast_to(idx[:, :, None], (idx.shape[0], a, b)).reshape(-1)
        c = np.broadcast_to(idx[:, None, :], (idx.shape[0], a, b)).reshape(-1)
        return jnp.asarray(r), jnp.asarray(c)

    cols_r = cols[:, :q - 1]                                         # (n_u, K+1)
    # What the returned index maps address.  ``cols_r`` and ``cols_r_flat`` are
    # ``cols`` with the ``v`` column dropped, i.e. the retained columns of a unit;
    # ``ix_r`` scatters the per-unit ramp and w-bound blocks and ``ix_c`` the
    # per-segment capacity blocks, both into the ``(nr, nr)`` reduced Hessian;
    # ``ix_d`` is that Hessian's diagonal, where the box term and ``reg`` land.
    # The ``on_t`` / ``on_p`` / ``cross`` triples are the (t,t), (t-1,t-1) and
    # (t,t-1) outer products of the ramp-up, ramp-down and w-bound operators, in
    # that order within each triple.
    return dict(
        q=q, qr=q - 1, nr=nr,
        cols_r=jnp.asarray(cols_r), cols_r_flat=jnp.asarray(cols_r.reshape(-1)),
        ix_r=scatter_ix(cols_r, q - 1, q - 1),
        ix_c=scatter_ix(rows_c, 2, 2),
        ix_d=jnp.arange(nr),
        M_r=jnp.asarray(M[:, :nr]),     # the line term needs retained columns only
        cap=outer(Vc, Vc),
        on_t=(outer(S["Sa"], S["Sa"]), outer(S["Sc"], S["Sc"]), outer(S["Se"], S["Se"])),
        on_p=(outer(S["Sb"], S["Sb"]), outer(S["Sd"], S["Sd"]), outer(S["Sf"], S["Sf"])),
        cross=(outer(S["Sa"], S["Sb"]), outer(S["Sc"], S["Sd"]), outer(S["Se"], S["Sf"])),
    )


def _build_blocks(spec: Dict, ops: Dict, st: Dict, D, reg):
    """Reduced diagonal and sub-diagonal blocks, each (T, nr+1, nr+1), plus what
    `apply` needs to fold the right-hand side and recover ``dv``.

    Eliminating ``v`` with the diagonal pivot ``d = H_vv`` per unit and period::

        dv_t     = (r_v - B' dr_t - E dr_{t-1}) / d
        A_t     -= B d^-1 B'          Off_t   -= B d^-1 E
        A_{t-1} -= E' d^-1 E

    where ``B`` holds the ``v`` column on the ``(g, u)_t`` rows and ``E`` the ``v``
    row on the ``(g, u)_{t-1}`` columns, both K+1 numbers per unit.  Every update
    is per-unit, so the elimination costs O(n_units * (K+1)^2) per period against
    the (nr/nb)^3 it removes from the factorisation.
    """
    T, nr, qr = spec["T"], st["nr"], st["qr"]
    d_line, d_cap, d_ramp, d_wbnd, d_box = _split_barrier(spec, D)
    du, dd = d_ramp[:, 0], d_ramp[:, 1]              # (T, n_u) up / down rows
    dw = d_wbnd.sum(1)                               # both w bounds hit the same pair

    def per_unit(blocks):
        """The three per-unit q*q contributions, summed."""
        return (du[:, :, None, None] * blocks[0]
                + dd[:, :, None, None] * blocks[1]
                + dw[:, :, None, None] * blocks[2])              # (T, n_u, q, q)

    OT, OP, CR = per_unit(st["on_t"]), per_unit(st["on_p"]), -per_unit(st["cross"])

    # Only OT reaches (v, v): Sb, Sd and Sf are zero there, checked in _make_struct
    inv = 1.0 / (OT[:, :, qr, qr] + d_box[:, nr:] + reg)         # (T, n_u)
    Bv = OT[:, :, :qr, qr]                                       # (T, n_u, qr)
    Ev = CR[:, :, qr, :qr]                                       # (T, n_u, qr)
    sc = inv[:, :, None, None]
    a_own = OT[:, :, :qr, :qr] - sc * (Bv[..., None] * Bv[:, :, None, :])
    a_prev = OP[:, :, :qr, :qr] - sc * (Ev[..., None] * Ev[:, :, None, :])
    off = CR[:, :, :qr, :qr] - sc * (Bv[..., None] * Ev[:, :, None, :])

    rr, cr = st["ix_r"]
    rc, cc = st["ix_c"]
    Hd = jnp.einsum("li,tl,lj->tij", st["M_r"], d_line, st["M_r"])
    Hd = Hd.at[:, rc, cc].add((d_cap[:, :, None, None] * st["cap"]).reshape(T, -1))
    Hd = Hd.at[:, st["ix_d"], st["ix_d"]].add(d_box[:, :nr] + reg)

    # period t's ramp and w-bound rows act on blocks t and t-1
    Hd = Hd.at[:, rr, cr].add(a_own.reshape(T, -1))
    Hd = Hd.at[:-1, rr, cr].add(a_prev[1:].reshape(T - 1, -1))  # period 0's rows have no t-1
    # Off[t] is the (t, t-1) block
    Off = jnp.zeros((T, nr, nr), Hd.dtype).at[:, rr, cr].add(off.reshape(T, -1))

    col = jnp.broadcast_to(ops["a"][:nr][None, :, None], (T, nr, 1))
    Diag = jnp.concatenate(
        [jnp.concatenate([Hd, col], 2),
         jnp.concatenate([jnp.swapaxes(col, 1, 2), jnp.zeros((T, 1, 1), Hd.dtype)], 2)], 1)
    Ofull = jnp.concatenate(
        [jnp.concatenate([Off, jnp.zeros((T, nr, 1), Hd.dtype)], 2),
         jnp.zeros((T, 1, nr + 1), Hd.dtype)], 1)
    return Diag, Ofull, inv, Bv, Ev


def make_ops(spec: Dict, ops: Dict) -> Tuple[Callable, Callable]:
    """``(Gx, GTy)`` without ever forming G."""
    nb, T, n_u, n_l, K = spec["nb"], spec["T"], spec["n_units"], spec["n_l"], spec["K"]
    n = nb * T
    M, C = ops["M"], ops["C"]
    Sa, Sb, Sc, Sd, Se, Sf = (ops[k] for k in ("Sa", "Sb", "Sc", "Sd", "Se", "Sf"))

    def _shift(Y):
        """Y[t-1], with a zero row in front."""
        return jnp.concatenate([jnp.zeros((1, Y.shape[1])), Y[:-1]], 0)

    def Gx(v):
        """``G @ v``: the ``(n,)`` variable vector in, the ``(m,)`` row vector out.

        Rows come out in the order the module docstring tabulates and `relax`
        builds ``h`` in -- both line directions, capacity, both ramp directions,
        both bounds on the eliminated ``w``, then ``+I`` and ``-I`` for the box.
        Unlike `kkt`, the two ramp directions are **not** one operator negated:
        the up row pairs ``Sa`` on ``t`` with ``Sb`` on ``t-1`` and the down row
        ``Sc`` with ``Sd``, because the start-up and shut-down ramp allowances
        differ.
        """
        V = v[:n].reshape(T, nb)
        line = V @ M.T
        cap = V @ C.T
        up = V @ Sa.T - _shift(V @ Sb.T)
        dn = -(V @ Sc.T) + _shift(V @ Sd.T)
        w0 = V @ Se.T - _shift(V @ Sf.T)
        w1 = -(V @ Se.T) + _shift(V @ Sf.T)
        return jnp.concatenate([
            jnp.concatenate([line, -line], 1).ravel(),
            cap.ravel(),
            jnp.concatenate([up, dn], 1).ravel(),
            jnp.concatenate([w0, w1], 1).ravel(),
            v, -v])

    def GTy(y):
        """``G' @ y``, the adjoint of `Gx`: ``(m,)`` in, ``(n,)`` out.

        The row groups are sliced back out in the order `Gx` wrote them, with the
        running offset ``i`` doing the same arithmetic `_split_barrier` does on the
        barrier weights.
        """
        i = 0
        line = y[i: i + 2 * n_l * T].reshape(T, 2, n_l); i += 2 * n_l * T
        cap = y[i: i + n_u * K * T].reshape(T, n_u * K); i += n_u * K * T
        ramp = y[i: i + 2 * n_u * T].reshape(T, 2, n_u); i += 2 * n_u * T
        wbnd = y[i: i + 2 * n_u * T].reshape(T, 2, n_u); i += 2 * n_u * T
        box_hi = y[i: i + n]; box_lo = y[i + n: i + 2 * n]

        out = (line[:, 0] - line[:, 1]) @ M + cap @ C
        du, dd = ramp[:, 0], ramp[:, 1]
        dw = wbnd[:, 0] - wbnd[:, 1]
        out = out + du @ Sa - dd @ Sc + dw @ Se
        # the same rows also touch block t-1, with the paired operator
        back = -(du @ Sb) + dd @ Sd - dw @ Sf
        out = out.at[:-1].add(back[1:])
        return out.ravel() + box_hi - box_lo

    return Gx, GTy


def make_kkt(spec: Dict, ops: Dict) -> Tuple[Callable, Callable]:
    """``(factor, apply)`` for ``solvers.ipm.make_solver``, block-tridiagonal.

    ``v`` is eliminated inside, so the factored blocks are ``nr+1`` wide, but the
    interface is unchanged: ``apply`` takes and returns full-length vectors in the
    caller's variable layout, and the solver never sees the reduction.
    """
    nb, T, n_u = spec["nb"], spec["T"], spec["n_units"]
    n = nb * T
    st = _make_struct(spec, ops)
    nr, qr = st["nr"], st["qr"]
    cols_r, cols_flat = st["cols_r"], st["cols_r_flat"]

    def factor(D, reg):
        """Factor the reduced block-tridiagonal KKT matrix, ``v`` eliminated first.

        Args:
            D: ``(m,)`` barrier weights ``lam / s``, in the row order `Gx` uses.
            reg: primal regularisation, added to every diagonal block.

        Returns:
            `kkt`'s per-period state ``(lu, piv, Cm, L)`` -- the LU factors of the
            modified diagonal block ``M_t = B_t - L_t Cm_{t-1}``, that period's
            ``Cm_t = M_t^-1 U_t`` and its sub-diagonal block -- with ``(inv, Bv,
            Ev)`` appended: the reciprocal of the diagonal ``v`` pivot and the two
            coupling vectors `apply` needs to fold the right-hand side and to
            recover ``dv`` afterwards.  None of it depends on a right-hand side, so
            one Newton step factors once and applies twice, for the predictor and
            the corrector direction.
        """
        Diag, Off, inv, Bv, Ev = _build_blocks(spec, ops, st, D, reg)
        nbp = Diag.shape[-1]
        # `Off[t]` is the (t, t-1) block: period 0 has no predecessor, and `Up[t]`,
        # the (t, t+1) block, is `Off[t+1]` transposed since the matrix is
        # symmetric -- the last period gets a zero block for the same reason.
        Lo = Off.at[0].set(jnp.zeros((nbp, nbp), Diag.dtype))
        Up = jnp.concatenate([jnp.swapaxes(Off[1:], 1, 2),
                              jnp.zeros((1, nbp, nbp), Diag.dtype)], 0)

        def fwd(Cp, xs):
            """One period of the block Thomas sweep.  The carry is ``Cm_{t-1}``."""
            B, L, U = xs
            lu, piv = jsl.lu_factor(B - L @ Cp)
            Cm = jsl.lu_solve((lu, piv), U)
            return Cm, (lu, piv, Cm, L)

        _, state = lax.scan(fwd, jnp.zeros((nbp, nbp), Diag.dtype), (Diag, Lo, Up))
        return state + (inv, Bv, Ev)

    def apply(state, rhs):
        """Solve the reduced system for one right-hand side and expand ``dv`` back.

        ``rhs`` and the result are in the caller's full layout, ``[nb*T variable
        rows ; T balance rows]``, so the solver never sees the elimination.  The
        ``v`` rows of the right-hand side are folded into the retained ones first
        -- at period ``t`` through ``Bv`` and at ``t-1`` through ``Ev`` -- then the
        reduced blocks are swept forwards and backwards, and ``dv`` is recovered
        from the retained solution by the formula in `_build_blocks`.
        """
        lus, pivs, Cs, Ls, inv, Bv, Ev = state
        X = rhs[:n].reshape(T, nb)
        r_r, r_v = X[:, :nr], X[:, nr:]

        # fold the eliminated rows into the retained ones, at t and at t-1
        z = r_v * inv
        r_r = r_r.at[:, cols_flat].add(-(z[:, :, None] * Bv).reshape(T, -1))
        r_r = r_r.at[:-1, cols_flat].add(-(z[:, :, None] * Ev)[1:].reshape(T - 1, -1))
        r = jnp.concatenate([r_r, rhs[n:].reshape(T, 1)], 1)

        def fwd(dp, xs):
            """Forward substitution, ``d_t = M_t^-1 (r_t - L_t d_{t-1})``.

            The carry is ``d_{t-1}``, zero before the first period.
            """
            lu, piv, L, ri = xs
            d = jsl.lu_solve((lu, piv), ri - L @ dp)
            return d, d

        _, ds = lax.scan(fwd, jnp.zeros(r.shape[-1], r.dtype), (lus, pivs, Ls, r))

        def bwd(znext, xs):
            """Back substitution, ``z_t = d_t - Cm_t z_{t+1}``.

            Runs with ``reverse=True``, so the carry is ``z_{t+1}``, zero after
            the last period.
            """
            Cm, d = xs
            z = d - Cm @ znext
            return z, z

        _, zs = lax.scan(bwd, jnp.zeros(r.shape[-1], r.dtype), (Cs, ds), reverse=True)
        dr = zs[:, :nr]

        # recover `dv` one unit and period at a time; period 0 has no `t-1` block,
        # hence the zero row in front of `prev`.  It is written back at the tail of
        # each block, which is where `v` sits, and the balance rows follow -- the
        # caller's layout again.
        own = dr[:, cols_r]                                       # (T, n_u, qr)
        prev = jnp.concatenate([jnp.zeros((1, n_u, qr), dr.dtype), dr[:-1][:, cols_r]], 0)
        dv = inv * (r_v - (Bv * own).sum(-1) - (Ev * prev).sum(-1))
        return jnp.concatenate([jnp.concatenate([dr, dv], 1).ravel(), zs[:, nr]])

    return factor, apply
