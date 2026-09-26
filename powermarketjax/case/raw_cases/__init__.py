# Vendored from PowerZooJax. Do not edit to track upstream;
# this file is now this repository's code.
# Source        : powerzoojax/case/raw_cases/__init__.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
# Upstream licence: MIT, Copyright (c) 2026 PowerZooJax Contributors.
"""
Raw Cases: Original OOP-style case definitions and SCUC JSON helpers.

``case14.json`` and ``scuc_json_to_case_units.py`` are mirrored under
``PowerZoo/powerzoo/case/raw_cases/``; keep both copies identical.

These cases use the same DataFrame-oriented layout as PowerZoo (via the
local ``powermarketjax.case.dataframe.DataFrame`` helper). They can be converted to JAX format
using the case_adapter.

Example:
    >>> from powermarketjax.case.raw_cases import Case5
    >>> from powermarketjax.case import case_to_jax
    >>> 
    >>> case = Case5()
    >>> case_data = case_to_jax(case)
"""

from powermarketjax.case.raw_cases.case5 import Case5
from powermarketjax.case.raw_cases.case33bw import Case33bw
from powermarketjax.case.raw_cases.case118 import Case118

__all__ = ['Case5', 'Case33bw', 'Case118']
