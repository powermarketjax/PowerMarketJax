# Vendored from PowerZooJax. Do not edit to track upstream;
# this file is now this repository's code.
# Source        : powerzoojax/envs/grid/bfs_power_flow.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# NOT verbatim. Beyond the package rename (powerzoojax -> powermarketjax, in
# both import statements and path strings), on 2026-08-05:
#   - ``converged`` additionally requires the V² floor to be inactive.
#     Upstream reported converged=True whenever max|ΔV²| < tol, which a node
#     pinned on the floor satisfies by construction (ΔV² == 0), so overloaded
#     feeders were reported as solved.
#   - ``BFSResult.floor_active`` added as a required field. It carried a
#     ``= None`` default until 2026-08-06, which made the pytree 8 leaves or
#     9 depending on how it was built and broke ``lax.cond`` across the two
#     shapes; the solver is the only construction site, here and upstream,
#     so the default protected nothing.
#   - ``MIN_V_PU_FLOOR`` exported so callers need not hard-code 0.5.
#   - Docstrings and comments were added on 2026-08-20 for API documentation;
#     no executable line was changed.
#
# The loop exit criterion is unchanged, so the numerics are unchanged: only
# the reported flags differ. Verified against upstream on case33bw at 1x / 5x
# / 10x / 50x load (2026-08-05): identical iteration counts, and v_sq /
# p_branch agreeing to <= 1 ULP of float32 (6e-8), which is the
# jax 0.6.2 -> 0.10.2 XLA difference, not this change.
# Upstream licence: MIT, Copyright (c) 2026 PowerZooJax Contributors.
"""Backward/forward sweep (BFS) power flow — pure JAX.

Solves the DistFlow equations on voltage-squared (``v_sq``) for **radial
(tree-topology) distribution networks**: branch flows are accumulated
leaf-to-root in the backward sweep, then voltages are rebuilt root-to-leaf in
the forward sweep. Typically converges in 1-2 iterations on physical cases.

Assumptions:
  - The network is a connected spanning tree rooted at the slack bus.
  - Branch direction is from sending (closer to slack) to receiving (leaf).
  - ``build_radial_topology`` raises if any node is unreachable.

Setup-time (NumPy, once): ``build_radial_topology`` / ``prepare_bfs`` →
``BFSTopoData``. Runtime (pure JAX, JIT-compilable): ``backward_sweep``,
``forward_sweep``, ``bfs_power_flow`` → ``BFSResult``. Loads, voltages, flows
and losses are all p.u.

Voltage-squared is floored at ``0.25`` (0.5 p.u.) inside both sweeps to keep
the loss denominators away from zero. A node pinned on that floor has
``ΔV² == 0`` and would pass the raw tolerance test, so ``converged``
additionally requires the floor to be inactive; ``floor_active`` reports it on
its own. Neither flag is a feasibility check.
"""

from __future__ import annotations

import collections
from typing import Tuple

import numpy as np
import jax.numpy as jnp
import jax.lax as lax
import chex
from flax import struct

from powermarketjax.case.case_data import CaseData


# ---------------------------------------------------------------------------
# Numerical guard constants
# ---------------------------------------------------------------------------

# Minimum voltage squared used as denominator guard in loss calculations.
# 0.25 corresponds to 0.5 p.u. — an extreme undervoltage that prevents
# division by zero while still being physically plausible.
_MIN_V_SQ_GUARD: float = 0.25

# Floor for voltage-squared results in the forward sweep.  Same physical
# interpretation as above: prevents negative / near-zero squared voltages
# from destabilising the iteration.
_MIN_V_SQ_FLOOR: float = 0.25

# Public form of the same floor, in volts p.u. Exported so callers do not have
# to hard-code 0.5 when reasoning about a floored solution.
MIN_V_PU_FLOOR: float = 0.5

# Threshold for deciding that a node has landed *on* the floor rather than
# merely near it. The 1e-5 margin is well above float32 round-off at V² ≈ 0.25
# (eps ≈ 3e-8) and far below any voltage a healthy feeder reaches, so it cannot
# misfire on a genuine solution.
_FLOOR_DETECT_V_SQ: float = _MIN_V_SQ_FLOOR * (1.0 + 1e-5)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@struct.dataclass
class BFSTopoData:
    """Precomputed radial topology data (built once at env construction).

    All matrices are dense JAX arrays for JIT/vmap compatibility.
    """
    path_matrix: chex.Array       # (n_nodes, n_lines) float32
    downstream_matrix: chex.Array # (n_lines, n_nodes) float32
    r_pu: chex.Array              # (n_lines,) float32
    x_pu: chex.Array              # (n_lines,) float32
    sending_node_idx: chex.Array  # (n_lines,) int32
    receiving_node_idx: chex.Array # (n_lines,) int32
    n_nodes: int = struct.field(pytree_node=False, default=0)
    n_lines: int = struct.field(pytree_node=False, default=0)


@struct.dataclass
class BFSResult:
    """Output of BFS power flow solver."""
    v_sq: chex.Array       # (n_nodes,) voltage squared [p.u.^2]
    v_mag: chex.Array      # (n_nodes,) voltage magnitude [p.u.]
    p_branch: chex.Array   # (n_lines,) active branch flow [p.u.]
    q_branch: chex.Array   # (n_lines,) reactive branch flow [p.u.]
    p_loss: chex.Array     # (n_lines,) active branch loss [p.u.]
    q_loss: chex.Array     # (n_lines,) reactive branch loss [p.u.]
    converged: chex.Array  # bool scalar — see below
    iterations: chex.Array # int32 scalar
    floor_active: chex.Array  # bool scalar — see below

    # ``converged`` is not just ``max|ΔV²| < tol``: the forward sweep floors V²
    # at ``_MIN_V_SQ_FLOOR``, and a node pinned on that floor has ΔV²
    # identically zero, so the tolerance test alone would report success on an
    # operating point the solver never solved. ``converged`` therefore also
    # requires ``not floor_active``.
    #
    # ``floor_active`` is True when any node sits on the voltage floor
    # (V ≤ MIN_V_PU_FLOOR), which separates "voltage collapsed onto the floor"
    # from "did not converge within max_iter".
    #
    # Neither flag is a feasibility check. ``converged=True`` says only that the
    # power flow solved; whether the result is admissible (e.g. 0.9 ≤ V ≤ 1.1)
    # is a separate test the caller must perform.


# ---------------------------------------------------------------------------
# Setup-time functions (NumPy)
# ---------------------------------------------------------------------------

def build_radial_topology(
    n_nodes: int,
    from_nodes: np.ndarray,
    to_nodes: np.ndarray,
    r_pu: np.ndarray,
    x_pu: np.ndarray,
    slack_bus_id: int = 0,
) -> BFSTopoData:
    """Build radial topology from graph data via BFS.

    Args:
        n_nodes: Number of buses.
        from_nodes: (n_lines,) from-bus index for each line.
        to_nodes: (n_lines,) to-bus index for each line.
        r_pu: (n_lines,) resistance [p.u.].
        x_pu: (n_lines,) reactance [p.u.].
        slack_bus_id: Root bus index.

    Returns:
        BFSTopoData with dense path/downstream matrices as JAX arrays.
    """
    n_lines = len(from_nodes)
    from_nodes = np.asarray(from_nodes, dtype=int)
    to_nodes = np.asarray(to_nodes, dtype=int)
    r_pu = np.asarray(r_pu, dtype=float)
    x_pu = np.asarray(x_pu, dtype=float)

    # Build adjacency list. Undirected: the ``from``/``to`` orientation of the
    # input line list carries no meaning, the tree orientation is recomputed by
    # the search below.
    adj: dict = {i: [] for i in range(n_nodes)}
    for i in range(n_lines):
        adj[from_nodes[i]].append((to_nodes[i], i))
        adj[to_nodes[i]].append((from_nodes[i], i))

    # BFS to build spanning tree
    parent = np.full(n_nodes, -1, dtype=int)
    parent_line = np.full(n_nodes, -1, dtype=int)
    visited = np.zeros(n_nodes, dtype=bool)
    queue = collections.deque([slack_bus_id])
    visited[slack_bus_id] = True

    while queue:
        node = queue.popleft()
        for neighbor, line_idx in adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                parent[neighbor] = node
                parent_line[neighbor] = line_idx
                queue.append(neighbor)

    if visited.sum() < n_nodes:
        unreachable = np.where(~visited)[0]
        raise ValueError(f"Unreachable nodes from slack bus {slack_bus_id}: {unreachable}")

    # Sending/receiving node direction in the tree
    # ``parent[from_nodes] == to_nodes`` means the line is listed pointing at
    # its own parent, i.e. towards the root, so its two ends are swapped;
    # otherwise the listed orientation already points away from the root.
    sending_nodes = np.where(parent[from_nodes] == to_nodes, to_nodes, from_nodes)
    receiving_nodes = np.where(sending_nodes == from_nodes, to_nodes, from_nodes)

    # Path matrix: path_matrix[node, line] = 1 if line is on root→node path
    # Only spanning-tree lines are ever marked. A line outside the tree (an
    # extra loop closure) keeps an all-zero column, so it carries no flow and
    # causes no voltage drop — a meshed input is solved as if that line were
    # open. The one topology error raised is an unreachable node, above.
    path_matrix = np.zeros((n_nodes, n_lines), dtype=np.float32)
    for node in range(n_nodes):
        curr = node
        while parent[curr] != -1:
            line = parent_line[curr]
            if line != -1:
                path_matrix[node, line] = 1.0
            curr = parent[curr]

    # Downstream matrix = path_matrix.T
    # Transposing turns "line is on the root→node path" into "node lies
    # downstream of line", which is the identity both sweeps rely on:
    # ``downstream_matrix @ x`` sums a nodal quantity over each line's subtree,
    # and ``path_matrix @ y`` accumulates a per-line quantity along the path
    # from the root down to each node.
    downstream_matrix = path_matrix.T.copy()

    return BFSTopoData(
        path_matrix=jnp.array(path_matrix),
        downstream_matrix=jnp.array(downstream_matrix),
        r_pu=jnp.array(r_pu, dtype=jnp.float32),
        x_pu=jnp.array(x_pu, dtype=jnp.float32),
        sending_node_idx=jnp.array(sending_nodes, dtype=jnp.int32),
        receiving_node_idx=jnp.array(receiving_nodes, dtype=jnp.int32),
        n_nodes=n_nodes,
        n_lines=n_lines,
    )


def prepare_bfs(case: CaseData) -> BFSTopoData:
    """Build BFS topology data from a CaseData with distribution parameters.

    Inactive lines (status=0, e.g. normally-open tie switches) are excluded
    so that the BFS spanning tree reflects the actual operating topology.

    Args:
        case: CaseData with line_r, line_x, line_from_idx, line_to_idx.

    Returns:
        BFSTopoData ready for JIT-compiled BFS solver.

    Note:
        The returned topology indexes the *filtered* line list, so its line
        positions stop matching ``case.line_*`` as soon as any line is out of
        service. Anything indexed by line (ratings, sensitivity matrices) must
        be filtered the same way before being used alongside a ``BFSTopoData``.
    """
    r_pu = np.asarray(case.line_r)
    x_pu = np.asarray(case.line_x)
    from_idx = np.asarray(case.line_from_idx, dtype=int)
    to_idx = np.asarray(case.line_to_idx, dtype=int)

    if case.line_status is not None:
        active = np.asarray(case.line_status) > 0
        r_pu = r_pu[active]
        x_pu = x_pu[active]
        from_idx = from_idx[active]
        to_idx = to_idx[active]

    slack = int(case.slack_bus_idx)

    return build_radial_topology(
        n_nodes=case.n_nodes,
        from_nodes=from_idx,
        to_nodes=to_idx,
        r_pu=r_pu,
        x_pu=x_pu,
        slack_bus_id=slack,
    )


# ---------------------------------------------------------------------------
# Runtime functions (pure JAX, JIT-compilable)
# ---------------------------------------------------------------------------

def backward_sweep(
    topo: BFSTopoData,
    p_load_pu: chex.Array,
    q_load_pu: chex.Array,
    v_sq: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """Backward sweep: branch flows from downstream loads + losses.

    Args:
        topo: BFSTopoData with downstream_matrix, r_pu, x_pu, etc.
        p_load_pu: (n_nodes,) active load [p.u.].
        q_load_pu: (n_nodes,) reactive load [p.u.].
        v_sq: (n_nodes,) voltage squared from previous iteration.

    Returns:
        (p_branch, q_branch) in p.u.
    """
    n_nodes = p_load_pu.shape[0]

    # Base branch flow = sum of downstream loads
    # Loads are positive for consumption and p.u. on the case base, so a
    # positive branch flow runs from the sending node towards the leaves.
    p_branch = topo.downstream_matrix @ p_load_pu
    q_branch = topo.downstream_matrix @ q_load_pu

    # Line losses from previous iteration voltages
    # ``i_sq`` is |I|² = (P² + Q²)/V² taken at the sending end, with the voltage
    # of the previous forward sweep — that lag is what makes the scheme iterate.
    v_sending_sq = jnp.maximum(v_sq[topo.sending_node_idx], _MIN_V_SQ_GUARD)
    i_sq = (p_branch ** 2 + q_branch ** 2) / v_sending_sq
    p_loss = topo.r_pu * i_sq
    q_loss = topo.x_pu * i_sq

    # Losses at receiving nodes, propagated upstream
    # Each line's loss is deposited at its receiving node, then aggregated with
    # the same downstream operator, which charges it to every line between that
    # node and the root — one pass suffices because the topology is a tree.
    p_loss_at_node = jnp.zeros(n_nodes).at[topo.receiving_node_idx].add(p_loss)
    q_loss_at_node = jnp.zeros(n_nodes).at[topo.receiving_node_idx].add(q_loss)

    p_branch = p_branch + topo.downstream_matrix @ p_loss_at_node
    q_branch = q_branch + topo.downstream_matrix @ q_loss_at_node

    return p_branch, q_branch


def forward_sweep(
    topo: BFSTopoData,
    p_branch: chex.Array,
    q_branch: chex.Array,
    v_sq: chex.Array,
    v_slack: float = 1.0,
) -> chex.Array:
    """Forward sweep: voltage drops from root to leaves (DistFlow).

    Args:
        topo: BFSTopoData.
        p_branch: (n_lines,) active branch flow [p.u.].
        q_branch: (n_lines,) reactive branch flow [p.u.].
        v_sq: (n_nodes,) voltage squared from previous iteration.
        v_slack: Slack bus voltage magnitude [p.u.].

    Returns:
        v_sq_new: (n_nodes,) updated voltage squared.
    """
    v_sending_sq = jnp.maximum(v_sq[topo.sending_node_idx], _MIN_V_SQ_GUARD)

    # DistFlow drop across one line, in p.u.²:
    #     ΔV² = 2(r·P + x·Q) − (r² + x²)(P² + Q²)/V²_sending
    # ``term1`` is the linear drop, ``term2`` the loss correction.
    z_sq = topo.r_pu ** 2 + topo.x_pu ** 2
    s_sq = p_branch ** 2 + q_branch ** 2

    term1 = 2.0 * (topo.r_pu * p_branch + topo.x_pu * q_branch)
    term2 = z_sq * s_sq / v_sending_sq
    delta_v_sq = term1 - term2

    # Every node is rebuilt from the slack by accumulating the per-line drops
    # along its root→node path, so the incoming ``v_sq`` enters only through the
    # loss denominator above and never as the base of the update.
    total_drop = topo.path_matrix @ delta_v_sq
    v_sq_new = v_slack ** 2 - total_drop

    # Floored here; ``BFSResult.floor_active`` is what reports it to the caller.
    return jnp.maximum(v_sq_new, _MIN_V_SQ_FLOOR)


def bfs_power_flow(
    topo: BFSTopoData,
    p_load_pu: chex.Array,
    q_load_pu: chex.Array,
    v_slack: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> BFSResult:
    """Run BFS power flow via lax.while_loop (JIT-compilable).

    Args:
        topo: BFSTopoData from build_radial_topology or prepare_bfs.
        p_load_pu: (n_nodes,) active load [p.u.].
        q_load_pu: (n_nodes,) reactive load [p.u.].
        v_slack: Slack bus voltage [p.u.].
        max_iter: Maximum BFS iterations.
        tol: Convergence tolerance on max|ΔV²|.

    Returns:
        BFSResult with voltages, branch flows, losses, convergence info.
        ``converged`` and ``floor_active`` are documented on that class;
        neither is a feasibility check, so voltage admissibility (e.g.
        ``0.9 ≤ v_mag ≤ 1.1``) remains a separate check the caller must
        perform.
    """
    n_nodes = p_load_pu.shape[0]
    n_lines = topo.r_pu.shape[0]
    tol_arr = jnp.float32(tol)

    # Flat start at 1 p.u.²; the first forward sweep re-anchors every node to
    # ``v_slack``. These take the JAX default dtype, so the whole loop carry is
    # float64 when ``jax_enable_x64`` is on and float32 otherwise, while ``tol``
    # is pinned to float32 either way.
    v_sq0 = jnp.ones(n_nodes)
    p_br0 = jnp.zeros(n_lines)
    q_br0 = jnp.zeros(n_lines)

    # State: (v_sq, p_branch, q_branch, iteration, settled)
    #
    # ``settled`` is the raw fixed-point test ``max|ΔV²| < tol`` and is the
    # sole loop exit criterion. The floor check stays out of it: folding it in
    # would exit as soon as any node touched the floor, which assumes the
    # floor can never engage transiently and then recover — an assumption
    # this loop does not make. A floored fixed point satisfies ``settled``
    # trivially (ΔV² == 0 once pinned), so ``settled`` alone would misreport
    # convergence; the floor test therefore runs once after the loop, where it
    # cannot affect the iterates.
    def _cond(state):
        v_sq, p_br, q_br, it, settled = state
        return jnp.logical_and(it < max_iter, jnp.logical_not(settled))

    def _body(state):
        v_sq, _p_br, _q_br, it, _settled = state

        p_branch, q_branch = backward_sweep(topo, p_load_pu, q_load_pu, v_sq)
        v_sq_new = forward_sweep(topo, p_branch, q_branch, v_sq, v_slack)

        max_diff = jnp.max(jnp.abs(v_sq_new - v_sq))
        settled = max_diff < tol_arr

        return (v_sq_new, p_branch, q_branch, it + 1, settled)

    init = (v_sq0, p_br0, q_br0, jnp.int32(0), jnp.bool_(False))
    v_sq_f, p_br_f, q_br_f, iters, settled_f = lax.while_loop(_cond, _body, init)

    # The two reported flags, both formed after the loop so that neither can
    # alter the iterates. The test is over the whole vector: one node on the
    # floor is enough to raise ``floor_active_f`` and to withhold ``converged``.
    floor_active_f = jnp.min(v_sq_f) <= _FLOOR_DETECT_V_SQ
    conv = jnp.logical_and(settled_f, jnp.logical_not(floor_active_f))

    # Compute final losses
    # Recomputed from the final voltages; the losses formed inside the loop
    # belong to the previous iterate and are not carried out of it.
    v_sending_sq = jnp.maximum(v_sq_f[topo.sending_node_idx], _MIN_V_SQ_GUARD)
    i_sq = (p_br_f ** 2 + q_br_f ** 2) / v_sending_sq
    p_loss = topo.r_pu * i_sq
    q_loss = topo.x_pu * i_sq

    return BFSResult(
        v_sq=v_sq_f,
        v_mag=jnp.sqrt(jnp.maximum(v_sq_f, 0.0)),
        p_branch=p_br_f,
        q_branch=q_br_f,
        p_loss=p_loss,
        q_loss=q_loss,
        converged=conv,
        iterations=iters,
        floor_active=floor_active_f,
    )
