# Vendored from PowerZooJax. Do not edit to track upstream;
# this file is now this repository's code.
# Source        : powerzoojax/case/case_adapter.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
# Upstream licence: MIT, Copyright (c) 2026 PowerZooJax Contributors.
"""
Case Adapter: Convert PowerZoo Cases to JAX CaseData

Two paths — both end at the same ``build_case_from_tables()`` core:

Path A (standalone):  case .py files define tables → build_case_from_tables()
Path B (runtime):     PowerZoo ClearCase instance → case_to_jax() → build_case_from_tables()

Usage:
    >>> # Path B — from live PowerZoo object
    >>> from powerzoo.case import load_case
    >>> from powermarketjax.case import case_to_jax
    >>> case_data = case_to_jax(load_case("5"))
    >>>
    >>> # Path A — standalone (no PowerZoo dependency)
    >>> from powermarketjax.case import load_case
    >>> case_data = load_case("5")
"""

# Re-export from the builder module so existing imports keep working
from powermarketjax.case.case_builder import (  # noqa: F401
    build_case_from_tables,
    case_to_jax,
)

convert_case = case_to_jax
