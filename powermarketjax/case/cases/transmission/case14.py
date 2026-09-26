# Vendored from PowerZooJax. Do not edit to track upstream;
# this file is now this repository's code.
# Source        : powerzoojax/case/cases/transmission/case14.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
# Upstream licence: MIT, Copyright (c) 2026 PowerZooJax Contributors.
"""Case14: IEEE 14-Bus Test System

Grid type: transmission
14 buses, 5 generators, 20 branches, 14 loads
Base MVA: 100.0

Unit table uses the same UC / economic columns as ``case118`` (``type``, ``ramp_*``,
``init_*``, ``min_*``).  Thermal and PQ data follow MATPOWER ``case14.m``; UC scalars
and marginal-cost coefficients ``mc_*`` are derived from ``raw_cases/case14.json``
(mirrored under PowerZoo ``powerzoo/case/raw_cases/case14.json``) via
``raw_cases/scuc_json_to_case_units.py`` (same script in both repos) — single aggregate
startup cost per unit (max of JSON tiers), ramps as MW/h → fraction/h of ``p_max``,
min up/down in hours → steps at ``delta_t_hours=0.5``.

GPU memory (RTX 4500 Ada 24GB, measured):
    CaseData: <0.01 MB | ACPFSetup: <0.01 MB
    ACPF vmap: ~0.01 MB/env | DC vmap: negligible
    Max vmap on 24GB: 65536+ envs (any solver)
"""

from powermarketjax.case.case_builder import build_case_from_tables
from powermarketjax.case.case_data import CaseData


def create_case14() -> CaseData:
    """Create Case14 (14 buses, transmission)."""

    _nodes = [
        ["id", "type", "Pd", "Qd", "Gs", "Bs", "Vm", "Va", "baseKV", "Vmax", "Vmin"],
        [1, 3, 0, 0, 0, 0, 1.06, 0, 0, 1.06, 0.94],
        [2, 2, 21.7, 12.7, 0, 0, 1.045, -4.98, 0, 1.06, 0.94],
        [3, 2, 94.2, 19, 0, 0, 1.01, -12.72, 0, 1.06, 0.94],
        [4, 1, 47.8, -3.9, 0, 0, 1.019, -10.33, 0, 1.06, 0.94],
        [5, 1, 7.6, 1.6, 0, 0, 1.02, -8.78, 0, 1.06, 0.94],
        [6, 2, 11.2, 7.5, 0, 0, 1.07, -14.22, 0, 1.06, 0.94],
        [7, 1, 0, 0, 0, 0, 1.062, -13.37, 0, 1.06, 0.94],
        [8, 2, 0, 0, 0, 0, 1.09, -13.36, 0, 1.06, 0.94],
        [9, 1, 29.5, 16.6, 0, 19, 1.056, -14.94, 0, 1.06, 0.94],
        [10, 1, 9, 5.8, 0, 0, 1.051, -15.1, 0, 1.06, 0.94],
        [11, 1, 3.5, 1.8, 0, 0, 1.057, -14.79, 0, 1.06, 0.94],
        [12, 1, 6.1, 1.6, 0, 0, 1.055, -15.07, 0, 1.06, 0.94],
        [13, 1, 13.5, 5.8, 0, 0, 1.05, -15.16, 0, 1.06, 0.94],
        [14, 1, 14.9, 5, 0, 0, 1.036, -16.04, 0, 1.06, 0.94],
    ]

    # Columns aligned with case118 / case29gb UC tables (see case_builder.build_case_from_tables).
    _units = [
        ["id", "bus_id", "Pg", "Qg", "Qmax", "Qmin", "Vg", "mBase", "status", "Pmax", "Pmin", "mc_a", "mc_b", "mc_c", "p_max", "p_min", "type", "ramp_up", "ramp_down", "init_start_up_cost", "keep_time", "init_power", "init_state", "min_up_time", "min_down_time", "init_no_load_cost"],
        [1, 1, 232.4, -16.9, 10, 0, 1.06, 100, 1, 332.4, 0, 0.0, 0.05541301, 34.216168, 332.4, 0, "gas", 0.6953069, 0.6953069, 34238.45, 48.0, 230.74888, 1.0, 2.0, 2.0, 0.0],
        [2, 2, 40, 42.4, 50, -40, 1.045, 100, 1, 140, 0, 0.00897295, -1.143378, 67.97416, 140, 0, "gas", 0.66157144, 0.66157144, 11382.84, 48.0, 0.0, 0.0, 2.0, 2.0, 0.0],
        [3, 3, 0, 23.4, 40, 0, 1.01, 100, 1, 100, 0, 0.0357393, -2.380081, 72.64597, 100, 0, "gas", 0.6621, 0.6621, 5235.41, 48.0, 0.0, 0.0, 2.0, 2.0, 0.0],
        [4, 6, 0, 12.2, 24, -6, 1.07, 100, 1, 100, 0, 0.03130176, -2.3915932, 78.5705, 100, 0, "gas", 0.6694, 0.6694, 5272.29, 48.0, 0.0, 0.0, 8.0, 8.0, 0.0],
        [5, 8, 0, 17.4, 24, -6, 1.09, 100, 1, 100, 0, 0.03348683, -2.4855613, 76.65385, 100, 0, "gas", 0.6504, 0.6504, 7950.56, 48.0, 0.0, 0.0, 2.0, 2.0, 0.0],
    ]

    _lines = [
        ["id", "from", "to", "r", "x", "b", "rateA", "ratio", "angle", "status", "floor", "cap"],
        [1, 1, 2, 0.01938, 0.05917, 0.0528, 0, 0, 0, 1, -1e+06, 1e+06],
        [2, 1, 5, 0.05403, 0.22304, 0.0492, 0, 0, 0, 1, -1e+06, 1e+06],
        [3, 2, 3, 0.04699, 0.19797, 0.0438, 0, 0, 0, 1, -1e+06, 1e+06],
        [4, 2, 4, 0.05811, 0.17632, 0.034, 0, 0, 0, 1, -1e+06, 1e+06],
        [5, 2, 5, 0.05695, 0.17388, 0.0346, 0, 0, 0, 1, -1e+06, 1e+06],
        [6, 3, 4, 0.06701, 0.17103, 0.0128, 0, 0, 0, 1, -1e+06, 1e+06],
        [7, 4, 5, 0.01335, 0.04211, 0, 0, 0, 0, 1, -1e+06, 1e+06],
        [8, 4, 7, 0, 0.20912, 0, 0, 0.978, 0, 1, -1e+06, 1e+06],
        [9, 4, 9, 0, 0.55618, 0, 0, 0.969, 0, 1, -1e+06, 1e+06],
        [10, 5, 6, 0, 0.25202, 0, 0, 0.932, 0, 1, -1e+06, 1e+06],
        [11, 6, 11, 0.09498, 0.1989, 0, 0, 0, 0, 1, -1e+06, 1e+06],
        [12, 6, 12, 0.12291, 0.25581, 0, 0, 0, 0, 1, -1e+06, 1e+06],
        [13, 6, 13, 0.06615, 0.13027, 0, 0, 0, 0, 1, -1e+06, 1e+06],
        [14, 7, 8, 0, 0.17615, 0, 0, 0, 0, 1, -1e+06, 1e+06],
        [15, 7, 9, 0, 0.11001, 0, 0, 0, 0, 1, -1e+06, 1e+06],
        [16, 9, 10, 0.03181, 0.0845, 0, 0, 0, 0, 1, -1e+06, 1e+06],
        [17, 9, 14, 0.12711, 0.27038, 0, 0, 0, 0, 1, -1e+06, 1e+06],
        [18, 10, 11, 0.08205, 0.19207, 0, 0, 0, 0, 1, -1e+06, 1e+06],
        [19, 12, 13, 0.22092, 0.19988, 0, 0, 0, 0, 1, -1e+06, 1e+06],
        [20, 13, 14, 0.17093, 0.34802, 0, 0, 0, 0, 1, -1e+06, 1e+06],
    ]

    _loads = [
        ["id", "bus_id", "Pd", "Qd", "mc_a", "mc_b", "mc_c", "d_max", "d_min"],
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 21.7, 12.7, 0, 0, 0, 21.7, 21.7],
        [3, 3, 94.2, 19, 0, 0, 0, 94.2, 94.2],
        [4, 4, 47.8, -3.9, 0, 0, 0, 47.8, 47.8],
        [5, 5, 7.6, 1.6, 0, 0, 0, 7.6, 7.6],
        [6, 6, 11.2, 7.5, 0, 0, 0, 11.2, 11.2],
        [7, 7, 0, 0, 0, 0, 0, 0, 0],
        [8, 8, 0, 0, 0, 0, 0, 0, 0],
        [9, 9, 29.5, 16.6, 0, 0, 0, 29.5, 29.5],
        [10, 10, 9, 5.8, 0, 0, 0, 9, 9],
        [11, 11, 3.5, 1.8, 0, 0, 0, 3.5, 3.5],
        [12, 12, 6.1, 1.6, 0, 0, 0, 6.1, 6.1],
        [13, 13, 13.5, 5.8, 0, 0, 0, 13.5, 13.5],
        [14, 14, 14.9, 5, 0, 0, 0, 14.9, 14.9],
    ]

    return build_case_from_tables(
        nodes_cols=_nodes[0], nodes_data=_nodes[1:],
        units_cols=_units[0], units_data=_units[1:],
        lines_cols=_lines[0], lines_data=_lines[1:],
        loads_cols=_loads[0], loads_data=_loads[1:],
        base_mva=100.0,
    )
