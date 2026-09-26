# Vendored from PowerZooJax. Do not edit to track upstream;
# this file is now this repository's code.
# Source        : powerzoojax/envs/grid/bfs_3phase_power_flow.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# NOT verbatim. Beyond the package rename (powerzoojax -> powermarketjax, in
# both import statements and path strings): docstrings and comments were
# added on 2026-08-20 for API documentation. No executable line was changed.
# Upstream licence: MIT, Copyright (c) 2026 PowerZooJax Contributors.
"""Three-phase backward/forward sweep (BFS) power flow — pure JAX.

Solves unbalanced **radial (tree-topology) distribution networks** with
single-phase laterals via the BIBC/BCBV/DLF formulation, working directly
with complex voltages and currents. This preserves per-phase imbalance,
unlike the single-phase ``bfs_power_flow`` module, which works on real
``v_sq``.

Conventions (following Kersting, *Distribution System Modeling and Analysis*):
    - Branch direction: from sending bus (closer to slack) to receiving bus.
    - K matrix: incidence matrix, K[line, bus] = +1 for sending, −1 for
      receiving (reference-bus column removed to form Gamma).
    - BIBC = −inv(Gamma^T): Bus-Injection to Branch-Current matrix.
    - BCBV: Branch-Current to Bus-Voltage matrix (embeds branch impedances).
    - DLF = BCBV @ BIBC: Direct Load Flow matrix.
    - Loads are positive for consumption: S_load = P + jQ.

Setup-time (NumPy, once): ``build_3phase_topology`` → ``ThreePhaseTopoData``.
Runtime (pure JAX, JIT-compilable): ``bfs_3phase_power_flow`` →
``BFS3PhResult``.

Default dtype is float32; enable ``jax.config.update('jax_enable_x64', True)``
if convergence residuals approach 1e-6 on long or highly unbalanced feeders.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import jax.lax as lax
import chex
from flax import struct


@struct.dataclass
class ThreePhaseTopoData:
    """Precomputed 3-phase radial topology (built once at setup time).

    Every runtime vector is flattened bus-major / phase-minor: entry ``3*i + p``
    is phase ``p`` (a, b, c) of the ``i``-th **non-reference** bus, taken in
    ascending bus index with ``ref_bus`` deleted. Branch-indexed vectors use the
    same layout over the line list. The two indexings share a length because a
    radial tree has ``n_lines == n_nodes - 1``, which is also what makes the
    BIBC inversion in ``build_3phase_topology`` well posed.

    Impedances and voltages are p.u. Complex matrices are stored as separate
    real and imaginary parts because the runtime solver works in real
    arithmetic (``_complex_matmul``).
    """
    DLF_real: chex.Array       # (3*n_lines, 3*n_lines)  # n_lines == n_nodes-1 for radial tree — real part of DLF
    DLF_imag: chex.Array       # (3*n_lines, 3*n_lines)  # n_lines == n_nodes-1 for radial tree — imag part of DLF
    BIBC_real: chex.Array      # (3*n_lines, 3*n_lines)  # n_lines == n_nodes-1 for radial tree
    BIBC_imag: chex.Array      # (3*n_lines, 3*n_lines)  # n_lines == n_nodes-1 for radial tree — pure real, imag=0
    V_ref_3ph_real: chex.Array # (3*(n_nodes-1),) == (3*n_lines,) for radial tree — real part of tiled reference voltage
    V_ref_3ph_imag: chex.Array # (3*(n_nodes-1),) == (3*n_lines,) for radial tree — imag part of tiled reference voltage
    n_nodes: int = struct.field(pytree_node=False, default=0)
    n_lines: int = struct.field(pytree_node=False, default=0)
    from_nodes: chex.Array = None  # (n_lines,) int32
    to_nodes: chex.Array = None    # (n_lines,) int32
    Z_diag_r: chex.Array = None    # (n_lines, 3) diagonal R
    Z_diag_x: chex.Array = None    # (n_lines, 3) diagonal X
    line_phase_mask: chex.Array = None   # (n_lines, 3) float32: 1.0 if phase energized
    line_phase_count: chex.Array = None  # (n_lines,) float32: energized phase count (≥1)


@struct.dataclass
class BFS3PhResult:
    """Output of 3-phase BFS power flow solver.

    Bus quantities (``V_real`` / ``V_imag`` / ``v_mag``) cover the non-reference
    buses only — the reference bus is held at the reference voltage and never
    appears in the state. Branch quantities are per line-phase. Both follow the
    bus-major / phase-minor layout of ``ThreePhaseTopoData`` and are p.u.
    """
    V_real: chex.Array         # (3*n_lines,) real part of bus voltages
    V_imag: chex.Array         # (3*n_lines,) imag part
    v_mag: chex.Array          # (3*n_lines,) voltage magnitudes
    I_branch_real: chex.Array  # (3*n_lines,) branch current real
    I_branch_imag: chex.Array  # (3*n_lines,) branch current imag
    P_branch: chex.Array       # (3*n_lines,) branch active power
    Q_branch: chex.Array       # (3*n_lines,) branch reactive power
    converged: chex.Array      # bool scalar
    iterations: chex.Array     # int32 scalar


# ---------------------------------------------------------------------------
# Setup-time functions (NumPy)
# ---------------------------------------------------------------------------

def build_3phase_topology(
    n_nodes: int,
    from_nodes: np.ndarray,
    to_nodes: np.ndarray,
    Z_3ph_pu: np.ndarray,
    ref_bus: int = 0,
    v_ref_mag: float = 1.0,
) -> ThreePhaseTopoData:
    """Build 3-phase topology with BIBC/BCBV/DLF matrices.

    Args:
        n_nodes: Number of buses.
        from_nodes: (n_lines,) from-bus indices.
        to_nodes: (n_lines,) to-bus indices.
        Z_3ph_pu: (n_lines, 3, 3) complex impedance matrices per branch [p.u.].
        ref_bus: Reference (slack) bus index.
        v_ref_mag: Reference voltage magnitude.

    Returns:
        ThreePhaseTopoData with dense JAX arrays.
    """
    n_lines = len(from_nodes)
    from_nodes = np.asarray(from_nodes, dtype=int)
    to_nodes = np.asarray(to_nodes, dtype=int)
    Z_3ph_pu = np.asarray(Z_3ph_pu, dtype=complex)

    # Incidence matrix K (n_lines, n_nodes)
    # Convention: K[line, from_bus] = +1, K[line, to_bus] = -1
    # (direction points from sending to receiving relative to slack).
    # The test below flips a line whose to-bus index is at or below ``ref_bus``;
    # for the usual ``ref_bus = 0`` that fires only for a line listed as
    # pointing into the reference bus. No spanning-tree search happens here
    # (unlike ``build_radial_topology`` in `physics.bfs_power_flow`), so the
    # line list must already be oriented away from the reference bus.
    K = np.zeros((n_lines, n_nodes))
    for i in range(n_lines):
        if to_nodes[i] <= ref_bus:
            K[i, from_nodes[i]] = -1
            K[i, to_nodes[i]] = 1
        else:
            K[i, from_nodes[i]] = 1
            K[i, to_nodes[i]] = -1

    # Remove reference bus column → Gamma
    # Gamma is (n_lines, n_nodes-1) and its transpose is inverted just below, so
    # it has to be square. This is where the radial assumption actually bites: a
    # meshed input has n_lines > n_nodes-1 and ``np.linalg.inv`` raises rather
    # than silently solving some other network.
    Gamma = np.delete(K, ref_bus, axis=1)

    # BIBC_0 = -inv(Gamma^T)
    # Single-phase Bus-Injection to Branch-Current: BIBC_0[line, bus] is nonzero
    # exactly when that bus lies downstream of that line.
    BIBC_0 = -np.linalg.inv(Gamma.T)

    # Expand to 3-phase: Kronecker product with I_3
    # Each scalar entry becomes a 3×3 diagonal block, so the phases share the
    # topology but do not mix here; inter-phase coupling enters only through the
    # off-diagonal terms of ``Z_3ph_pu``, which reach the solve via BCBV.
    BIBC = np.kron(BIBC_0, np.eye(3))

    # BCBV matrix (complex)
    # BCBV[bus i, line j] = Z_j when line j lies on the path from the reference
    # bus to bus i, which is exactly the sparsity pattern of BIBC_0.T (nonzero
    # where bus i is downstream of line j); the radial bijection between
    # non-reference buses and lines is what lets one index serve for both. The
    # reshape/swapaxes is a (n_lines, n_lines, 3, 3) *view* of BCBV, so the
    # blockwise assignment writes through to it.
    BCBV = np.zeros((3 * n_lines, 3 * n_lines), dtype=complex)
    BCBV_view = BCBV.reshape(n_lines, 3, n_lines, 3).swapaxes(1, 2)
    rows_b, cols_bb = np.nonzero(BIBC_0.T)
    BCBV_view[rows_b, cols_bb] = Z_3ph_pu[cols_bb]

    # DLF = BCBV @ BIBC
    # One matrix from bus injection currents straight to bus voltage drops, so
    # the runtime iteration is ``V = V_ref + DLF @ I_inj``.
    DLF = BCBV @ BIBC

    # Reference voltage (balanced 3-phase)
    # Phases a/b/c at 0°, −120°, +120°, tiled onto every non-reference bus: this
    # is both the boundary condition and the flat start of the iteration.
    V_ref_3ph = v_ref_mag * np.exp(1j * np.deg2rad([0, -120, 120]))
    Vr_n = np.tile(V_ref_3ph, n_lines)

    # Extract diagonal impedances for loss computation
    Z_diag = np.diagonal(Z_3ph_pu, axis1=1, axis2=2)

    # Phase energization mask: a phase is present when its self-impedance |Z_ii| > 0
    # Carried for callers that need to know which phases a lateral energizes
    # (single-phase laterals leave the absent phases at zero impedance); the
    # solver below does not read either field.
    phase_mask = (np.abs(Z_diag) > 1e-12).astype(np.float32)  # (n_lines, 3)
    phase_count = np.maximum(phase_mask.sum(axis=1), 1.0).astype(np.float32)

    return ThreePhaseTopoData(
        DLF_real=jnp.array(DLF.real, dtype=jnp.float32),
        DLF_imag=jnp.array(DLF.imag, dtype=jnp.float32),
        BIBC_real=jnp.array(BIBC.real, dtype=jnp.float32),
        BIBC_imag=jnp.array(BIBC.imag, dtype=jnp.float32),
        V_ref_3ph_real=jnp.array(Vr_n.real, dtype=jnp.float32),
        V_ref_3ph_imag=jnp.array(Vr_n.imag, dtype=jnp.float32),
        n_nodes=n_nodes,
        n_lines=n_lines,
        from_nodes=jnp.array(from_nodes, dtype=jnp.int32),
        to_nodes=jnp.array(to_nodes, dtype=jnp.int32),
        Z_diag_r=jnp.array(Z_diag.real, dtype=jnp.float32),
        Z_diag_x=jnp.array(Z_diag.imag, dtype=jnp.float32),
        line_phase_mask=jnp.array(phase_mask, dtype=jnp.float32),
        line_phase_count=jnp.array(phase_count, dtype=jnp.float32),
    )


# ---------------------------------------------------------------------------
# Runtime functions (pure JAX, JIT-compilable)
# ---------------------------------------------------------------------------

def _complex_matmul(Mr, Mi, xr, xi):
    """(M_real + j*M_imag) @ (x_real + j*x_imag) using real arithmetic."""
    yr = Mr @ xr - Mi @ xi
    yi = Mr @ xi + Mi @ xr
    return yr, yi


def bfs_3phase_power_flow(
    topo: ThreePhaseTopoData,
    P_3ph_pu: chex.Array,
    Q_3ph_pu: chex.Array,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> BFS3PhResult:
    """Run 3-phase BFS power flow via lax.while_loop.

    Conventions:
        P_3ph_pu, Q_3ph_pu are LOADS per bus-phase (positive = consumption).
        Internally: S_inj = -(P + jQ), I_inj = conj(S_inj / V).

    Args:
        topo: ThreePhaseTopoData from build_3phase_topology.
        P_3ph_pu: (3*n_lines,) or (n_lines, 3) active load [p.u.].
        Q_3ph_pu: (3*n_lines,) or (n_lines, 3) reactive load [p.u.].
        max_iter: Max iterations.
        tol: Convergence tolerance on max |ΔV|.

    Returns:
        BFS3PhResult with voltages, currents, branch powers, convergence info.

        ``converged`` is the fixed-point test ``max |V_new - V| < tol`` over all
        bus-phases, evaluated on the update that produced the returned
        voltages; it is False when the loop stopped at ``max_iter``. It is not a
        feasibility test — voltage and thermal admissibility remain the
        caller's to check.
    """
    # Flatten to (3*n_lines,)
    # A (n_lines, 3) input flattens bus-major / phase-minor, which is the layout
    # of ``topo.V_ref_3ph_*``; an already-flat input passes through unchanged.
    P_flat = P_3ph_pu.reshape(-1)
    Q_flat = Q_3ph_pu.reshape(-1)

    # S_inj = -(P + jQ)
    S_inj_real = -P_flat
    S_inj_imag = -Q_flat

    # ``eps`` guards the |V|² division only. At 1e-12 p.u.² it sits far below
    # any voltage a feeder reaches, so it does not pin the iterate the way the
    # single-phase solver's V² floor does — hence no floor flag on the result.
    eps = jnp.float32(1e-12)
    tol_arr = jnp.float32(tol)

    # State: (V_real, V_imag, iteration, converged)
    def _cond(state):
        _, _, it, conv = state
        return jnp.logical_and(it < max_iter, jnp.logical_not(conv))

    def _body(state):
        Vr, Vi, it, _ = state

        V_mag_sq = Vr ** 2 + Vi ** 2
        V_mag_sq = jnp.maximum(V_mag_sq, eps)

        # I_inj = conj(S_inj / V) = conj((Sr + j*Si) / (Vr + j*Vi))
        # S/V = ((Sr*Vr + Si*Vi) + j*(Si*Vr - Sr*Vi)) / |V|^2
        # conj(S/V) = ((Sr*Vr + Si*Vi) - j*(Si*Vr - Sr*Vi)) / |V|^2
        I_inj_real = (S_inj_real * Vr + S_inj_imag * Vi) / V_mag_sq
        I_inj_imag = -(S_inj_imag * Vr - S_inj_real * Vi) / V_mag_sq

        # delta_V = DLF @ I_inj
        dV_r, dV_i = _complex_matmul(
            topo.DLF_real, topo.DLF_imag, I_inj_real, I_inj_imag)

        # Each bus voltage is rebuilt from the reference plus the accumulated
        # drops, so nothing survives from the previous iterate except through
        # the injection currents ``I_inj`` computed above.
        V_new_r = topo.V_ref_3ph_real + dV_r
        V_new_i = topo.V_ref_3ph_imag + dV_i

        # Residual: the largest complex-magnitude voltage change over all
        # bus-phases [p.u.] — a fixed-point test on the update, not a power
        # mismatch.
        diff_r = V_new_r - Vr
        diff_i = V_new_i - Vi
        max_diff = jnp.max(jnp.sqrt(diff_r ** 2 + diff_i ** 2))
        converged = max_diff < tol_arr

        return (V_new_r, V_new_i, it + 1, converged)

    init = (topo.V_ref_3ph_real, topo.V_ref_3ph_imag, jnp.int32(0), jnp.bool_(False))
    V_r_f, V_i_f, iters, conv = lax.while_loop(_cond, _body, init)

    # Final: recompute injection current and branch current
    V_mag_sq = jnp.maximum(V_r_f ** 2 + V_i_f ** 2, eps)
    I_inj_r = (S_inj_real * V_r_f + S_inj_imag * V_i_f) / V_mag_sq
    I_inj_i = -(S_inj_imag * V_r_f - S_inj_real * V_i_f) / V_mag_sq

    # I_branch = BIBC @ I_inj
    Ib_r, Ib_i = _complex_matmul(
        topo.BIBC_real, topo.BIBC_imag, I_inj_r, I_inj_i)

    # Branch power: S_branch = V_from * conj(I_branch)
    # NOTE: Uses receiving-end bus voltages as approximation.  The resulting
    # P_branch underestimates sending-end power by omitting I²R losses.
    # For thermal-limit enforcement, use |I_branch| directly instead of
    # |S_branch|, since branch currents are accurate.
    P_branch = V_r_f * Ib_r + V_i_f * Ib_i
    Q_branch = V_i_f * Ib_r - V_r_f * Ib_i

    v_mag = jnp.sqrt(jnp.maximum(V_r_f ** 2 + V_i_f ** 2, 0.0))

    return BFS3PhResult(
        V_real=V_r_f,
        V_imag=V_i_f,
        v_mag=v_mag,
        I_branch_real=Ib_r,
        I_branch_imag=Ib_i,
        P_branch=P_branch,
        Q_branch=Q_branch,
        converged=conv,
        iterations=iters,
    )
