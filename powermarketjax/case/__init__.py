# Vendored from PowerZooJax. Do not edit to track upstream;
# this file is now this repository's code.
# Source        : powerzoojax/case/__init__.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Trimmed to this repository's scope -- NOT verbatim.
#   Removed: case_info (CaseInfo/print_case/print_summary) and case_plotter
#   (CasePlotter/plot_case). Both are CPU display helpers; case_plotter would
#   add matplotlib + networkx dependencies for no benchmark purpose.
#   Added 2026-08-21: `create_case73rts` and `create_case813nem`, two cases
#   upstream does not have; see `cases/transmission/__init__.py`.
# Upstream licence: MIT, Copyright (c) 2026 PowerZooJax Contributors.
"""Built-in case library.

This package is the setup-time entrypoint for benchmark cases. It provides:
- `CaseData`, the JAX-native array container used by env params
- a registry-backed `load_case()` / `list_cases()` interface
- built-in transmission and distribution cases
- matrix helpers such as PTDF and graph-derived case matrices

The case layer is intentionally separate from the market layer: load and
inspect cases here, then pass the resulting `CaseData` into the market
environments under `powermarketjax.envs`.

Quick start:
    >>> from powermarketjax.case import load_case, list_cases
    >>> case = load_case("5")
    >>> metas = list_cases(grid_type="distribution")
"""

import warnings

# Core data structure (GPU)
from powermarketjax.case.case_data import (CaseData, scale_min_output,
                                           validate_case_data)

# Matrix computations
from powermarketjax.case.case_matrices import (
    compute_ptdf,
    compute_adjacency_matrix,
    compute_degree_matrix,
    compute_laplacian_matrix,
    build_case_matrices,
)

# Built-in cases — aggregated from transmission/ + distribution/
from powermarketjax.case.cases import (
    create_case5,
    create_case14,
    create_case33bw,
    create_case118,
    create_case118zh,
    create_case123,
    create_case123_1ph,
    create_case141,
    create_case300,
    create_case533mt_hi,
    create_case533mt_lo,
    create_case1354pegase,
    create_case2383wp,
    create_case29gb,
    create_case552gb,
    create_case73rts,
    create_case813nem,
)

# Case adapter (for converting raw cases)
from powermarketjax.case.case_adapter import case_to_jax, convert_case

# Registry
from powermarketjax.case._registry import CaseMeta, list_cases, get_meta


def load_case(case_id: str = "5", *, grid_type: str = None) -> CaseData:
    """Load a built-in case by ID.

    If *grid_type* is given and the loaded case's metadata does not match,
    a :class:`UserWarning` is emitted.

    Args:
        case_id: Case identifier (e.g. ``"5"``, ``"33bw"``, ``"118"``).
        grid_type: Optional ``"transmission"`` or ``"distribution"``.

    Returns:
        CaseData with all arrays populated.
    """
    from powermarketjax.case._registry import get_registry

    key = str(case_id).lower().replace("case", "")
    reg = get_registry()

    if key not in reg:
        available = sorted(set(reg.keys()))
        raise ValueError(f"Unknown case '{case_id}'. Available: {available}")

    meta = reg[key]

    if grid_type and meta.grid_type and meta.grid_type != grid_type:
        warnings.warn(
            f"Case '{meta.name}' has grid_type='{meta.grid_type}', "
            f"but grid_type='{grid_type}' was requested.",
            UserWarning,
            stacklevel=2,
        )

    return meta.factory()


__all__ = [
    # Core
    "CaseData",
    "validate_case_data",
    "scale_min_output",
    # Matrix computations
    "compute_ptdf",
    "compute_adjacency_matrix",
    "compute_degree_matrix",
    "compute_laplacian_matrix",
    "build_case_matrices",
    # Built-in cases (native JAX)
    "create_case5",
    "create_case14",
    "create_case33bw",
    "create_case118",
    "create_case118zh",
    "create_case123",
    "create_case123_1ph",
    "create_case141",
    "create_case300",
    "create_case533mt_hi",
    "create_case533mt_lo",
    "create_case1354pegase",
    "create_case2383wp",
    "create_case29gb",
    "create_case552gb",
    "create_case73rts",
    "create_case813nem",
    "load_case",
    "list_cases",
    "get_meta",
    "CaseMeta",
    # Case adapter (raw -> JAX)
    "case_to_jax",
    "convert_case",
]
