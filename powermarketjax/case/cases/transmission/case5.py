# Vendored from PowerZooJax. Do not edit to track upstream;
# this file is now this repository's code.
# Source        : powerzoojax/case/cases/transmission/case5.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
# Upstream licence: MIT, Copyright (c) 2026 PowerZooJax Contributors.
"""Case5: IEEE 5-Bus Test System

Grid type: transmission
5 buses, 5 generators, 6 branches, 5 loads
Base MVA: 100.0

GPU memory (RTX 4500 Ada 24GB, measured):
    CaseData: <0.01 MB | DCOPFSetup: <0.01 MB
    ACPF vmap: <0.01 MB/env | DC vmap: negligible
    Max vmap on 24GB: 65536+ envs (any solver)
"""

from powermarketjax.case.case_builder import build_case_from_tables
from powermarketjax.case.case_data import CaseData


def create_case5() -> CaseData:
    """Create Case5 (5 buses, transmission)."""

    _nodes = [
        ["id", "type", "Pd", "Qd", "Gs", "Bs", "x", "y"],
        [1, 3, 0, 0, 0, 0, 0, 0],
        [2, 1, 300, 98.6, 0, 0, 3, 0],
        [3, 2, 300, 98.6, 0, 0, 4.5, 1],
        [4, 2, 400, 131.5, 0, 0, 2, 2],
        [5, 2, 0, 0, 0, 0, 0, 2],
    ]

    _units = [
        ["id", "bus_id", "mc_a", "mc_b", "mc_c", "p_max", "p_min",
         "Pg", "Qg", "Vg", "Qmax", "Qmin"],
        [1, 1, 0, 0, 14, 40, 5, 40, 0, 1.06, 30, -30],
        [2, 1, 0, 0, 15, 170, 10, 170, 0, 1.06, 127.5, -127.5],
        [3, 3, 0, 0, 30, 520, 20, 323.49, 0, 1.01, 390, -390],
        [4, 4, 0, 0, 40, 200, 10, 0, 0, 1.01, 150, -150],
        [5, 5, 0, 0, 10, 600, 20, 466.51, 0, 1.01, 450, -450],
    ]

    _lines = [
        ["id", "from", "to", "r", "x", "b", "floor", "cap", "status"],
        [1, 1, 2, 0.0094, 0.0281, 0.0211, -400, 400, 1],
        [2, 1, 4, 0.0101, 0.0304, 0.0228, -1e+06, 1e+06, 1],
        [3, 1, 5, 0.0021, 0.0064, 0.0048, -1e+06, 1e+06, 1],
        [4, 2, 3, 0.0036, 0.0108, 0.0081, -1e+06, 1e+06, 1],
        [5, 3, 4, 0.0099, 0.0297, 0.0223, -1e+06, 1e+06, 1],
        [6, 4, 5, 0.0099, 0.0297, 0.0223, -240, 240, 1],
    ]

    _loads = [
        ["id", "bus_id", "mc_a", "mc_b", "mc_c", "d_max", "d_min"],
        [1, 1, 0, 0, 0, 0, 0],
        [2, 2, 0, 0, 0, 500, 0],
        [3, 3, 0, 0, 0, 600, 0],
        [4, 4, 0, 0, 0, 400, 0],
        [5, 5, 0, 0, 0, 0, 0],
    ]

    return build_case_from_tables(
        nodes_cols=_nodes[0], nodes_data=_nodes[1:],
        units_cols=_units[0], units_data=_units[1:],
        lines_cols=_lines[0], lines_data=_lines[1:],
        loads_cols=_loads[0], loads_data=_loads[1:],
        base_mva=100.0,
    )
