# Vendored from PowerZooJax. Do not edit to track upstream;
# this file is now this repository's code.
# Source        : powerzoojax/case/case_matrices.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
# Upstream licence: MIT, Copyright (c) 2026 PowerZooJax Contributors.
"""
Case Matrix Computations

Power system matrix calculations:
- PTDF (Power Transfer Distribution Factor) — NumPy/CPU, one-time init
- Adjacency matrix
- Degree matrix
- Laplacian matrix
- Node-to-unit/load mapping matrices

PTDF uses NumPy because it is computed once at initialization (not in training
loops), so CPU LAPACK is faster than GPU + JIT compilation overhead.
Adjacency/degree/Laplacian/mapping functions are JIT-compiled JAX.

Usage:
    >>> from powermarketjax.case.case_matrices import compute_ptdf
    >>> PTDF = compute_ptdf(line_from_idx, line_to_idx, line_x, n_nodes, slack_idx=0)
"""

from functools import partial
import numpy as np  # For CPU-only operations (linalg.inv not supported on Metal)
import jax
import jax.numpy as jnp
import chex


def compute_ptdf(
    line_from_idx,
    line_to_idx,
    line_x,
    n_nodes: int,
    slack_idx: int = 0
):
    """Compute Power Transfer Distribution Factor (PTDF) matrix.

    PTDF relates nodal power injections to line power flows:
        P_line = PTDF @ P_injection

    Uses DC power flow approximation (NumPy/CPU, LAPACK inversion).
    Called once at initialization — not in training loops.

    Args:
        line_from_idx: From-node indices (n_lines,), JAX or NumPy array
        line_to_idx: To-node indices (n_lines,), JAX or NumPy array
        line_x: Line reactance values (n_lines,)
        n_nodes: Number of nodes
        slack_idx: Slack bus index (default 0)

    Returns:
        PTDF: JAX array (n_lines, n_nodes)

    Example:
        >>> PTDF = compute_ptdf(
        ...     line_from_idx=np.array([0, 0, 1, 2]),
        ...     line_to_idx=np.array([1, 2, 2, 3]),
        ...     line_x=np.array([0.1, 0.2, 0.15, 0.1]),
        ...     n_nodes=4,
        ... )
        >>> line_flows = PTDF @ node_injections
    """
    line_from_idx = np.asarray(line_from_idx)
    line_to_idx = np.asarray(line_to_idx)
    line_x = np.asarray(line_x)

    n_lines = len(line_from_idx)

    # Build incidence matrix A
    A = np.zeros((n_lines, n_nodes))
    A[np.arange(n_lines), line_from_idx] = 1.0
    A[np.arange(n_lines), line_to_idx] = -1.0

    # Line susceptance diagonal matrix
    b_line = -1.0 / line_x
    B_line = np.diag(b_line)

    # Node admittance matrix
    B_node = A.T @ B_line @ A

    # Remove slack bus row/col
    idx_no_slack = np.concatenate([
        np.arange(slack_idx),
        np.arange(slack_idx + 1, n_nodes)
    ])
    B_reduced = B_node[np.ix_(idx_no_slack, idx_no_slack)]
    B_reduced_inv = np.linalg.inv(B_reduced)

    # Expand back to full size (slack row/col stays zero)
    X = np.zeros((n_nodes, n_nodes))
    X[np.ix_(idx_no_slack, idx_no_slack)] = B_reduced_inv

    PTDF = B_line @ A @ X
    return jnp.array(PTDF)


@partial(jax.jit, static_argnums=(2,))
def compute_adjacency_matrix(
    line_from_idx: chex.Array,
    line_to_idx: chex.Array,
    n_nodes: int
) -> chex.Array:
    """Compute adjacency matrix of the network graph.
    
    A[i,j] = 1 if there is a line between node i and j.
    Matrix is symmetric (undirected graph).
    
    Args:
        line_from_idx: From-node indices (n_lines,)
        line_to_idx: To-node indices (n_lines,)
        n_nodes: Number of nodes
        
    Returns:
        A: (n_nodes, n_nodes) adjacency matrix
    """
    A = jnp.zeros((n_nodes, n_nodes))
    
    # Add edges in both directions (symmetric)
    A = A.at[line_from_idx, line_to_idx].set(1.0)
    A = A.at[line_to_idx, line_from_idx].set(1.0)
    
    return A


@jax.jit
def compute_degree_matrix(adjacency: chex.Array) -> chex.Array:
    """Compute degree matrix from adjacency matrix.
    
    D[i,i] = sum of row i of adjacency matrix (node degree).
    
    Args:
        adjacency: (n_nodes, n_nodes) adjacency matrix
        
    Returns:
        D: (n_nodes, n_nodes) diagonal degree matrix
    """
    degrees = jnp.sum(adjacency, axis=1)
    return jnp.diag(degrees)


@jax.jit
def compute_laplacian_matrix(adjacency: chex.Array) -> chex.Array:
    """Compute normalized Laplacian matrix.
    
    L = I - D^(-1/2) @ A @ D^(-1/2)
    
    Args:
        adjacency: (n_nodes, n_nodes) adjacency matrix
        
    Returns:
        L: (n_nodes, n_nodes) normalized Laplacian
    """
    # Add self-loops
    n = adjacency.shape[0]
    A_self = adjacency + jnp.eye(n)
    
    # Compute D^(-1/2)
    degrees = jnp.sum(A_self, axis=1)
    D_inv_sqrt = jnp.diag(jnp.power(degrees + 1e-10, -0.5))
    
    # Normalized Laplacian
    L = jnp.eye(n) - D_inv_sqrt @ A_self @ D_inv_sqrt
    
    return L


@partial(jax.jit, static_argnums=(2, 3))
def compute_node_element_map(
    element_node_idx: chex.Array,
    element_ids: chex.Array,
    n_nodes: int,
    n_elements: int
) -> chex.Array:
    """Compute node-to-element mapping matrix.
    
    Creates a (n_nodes, n_elements) binary matrix where
    M[node_i, element_j] = 1 if element_j is connected to node_i.
    
    Used for nodes_units_map and nodes_loads_map.
    
    Args:
        element_node_idx: Node index for each element (n_elements,)
        element_ids: Element indices (n_elements,), 0-indexed
        n_nodes: Number of nodes
        n_elements: Number of elements
        
    Returns:
        M: (n_nodes, n_elements) mapping matrix
    """
    M = jnp.zeros((n_nodes, n_elements))
    element_indices = jnp.arange(n_elements)
    M = M.at[element_node_idx, element_indices].set(1.0)
    return M


def build_case_matrices(
    node_ids: chex.Array,
    unit_bus_ids: chex.Array,
    load_bus_ids: chex.Array,
    line_from: chex.Array,
    line_to: chex.Array,
    line_x: chex.Array,
    slack_bus_id: float = None
) -> dict:
    """Build all case matrices from raw data.
    
    This is a convenience function that computes:
    - Internal node indices
    - PTDF matrix
    - Node-unit mapping
    - Node-load mapping
    - Adjacency matrix
    
    Args:
        node_ids: Node IDs (n_nodes,)
        unit_bus_ids: Unit bus IDs (n_units,)
        load_bus_ids: Load bus IDs (n_loads,)
        line_from: Line from-bus IDs (n_lines,)
        line_to: Line to-bus IDs (n_lines,)
        line_x: Line reactance (n_lines,)
        slack_bus_id: Slack bus ID (default: first node)
        
    Returns:
        Dict with computed matrices and indices
    """
    n_nodes = len(node_ids)
    n_units = len(unit_bus_ids)
    n_loads = len(load_bus_ids)
    n_lines = len(line_from)
    
    # Create ID to index mapping
    # Assumes node_ids are unique and sorted
    id_to_idx = {float(nid): i for i, nid in enumerate(node_ids)}
    
    # Convert bus IDs to internal indices
    unit_node_idx = jnp.array([id_to_idx[float(bid)] for bid in unit_bus_ids], dtype=jnp.int32)
    load_node_idx = jnp.array([id_to_idx[float(bid)] for bid in load_bus_ids], dtype=jnp.int32)
    line_from_idx = jnp.array([id_to_idx[float(bid)] for bid in line_from], dtype=jnp.int32)
    line_to_idx = jnp.array([id_to_idx[float(bid)] for bid in line_to], dtype=jnp.int32)
    
    # Slack bus index
    if slack_bus_id is None:
        slack_bus_idx = 0
    else:
        slack_bus_idx = id_to_idx[float(slack_bus_id)]
    
    PTDF = compute_ptdf(line_from_idx, line_to_idx, line_x, n_nodes, slack_bus_idx)
    
    # Compute mapping matrices
    nodes_units_map = compute_node_element_map(unit_node_idx, jnp.arange(n_units), n_nodes, n_units)
    nodes_loads_map = compute_node_element_map(load_node_idx, jnp.arange(n_loads), n_nodes, n_loads)
    
    # Compute adjacency
    adjacency = compute_adjacency_matrix(line_from_idx, line_to_idx, n_nodes)
    
    return {
        'unit_node_idx': unit_node_idx,
        'load_node_idx': load_node_idx,
        'line_from_idx': line_from_idx,
        'line_to_idx': line_to_idx,
        'slack_bus_idx': slack_bus_idx,
        'PTDF': PTDF,
        'nodes_units_map': nodes_units_map,
        'nodes_loads_map': nodes_loads_map,
        'adjacency': adjacency,
    }
