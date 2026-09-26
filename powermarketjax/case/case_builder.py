# Vendored from PowerZooJax. Do not edit to track upstream;
# this file is now this repository's code.
# Source        : powerzoojax/case/case_builder.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
# Upstream licence: MIT, Copyright (c) 2026 PowerZooJax Contributors.
"""
Case Builder: Construct CaseData from tabular data.

Provides ``build_case_from_tables()`` — the single entry-point that both
hand-written case files and the runtime adapter (``case_to_jax``) share.

The table format mirrors PowerZoo's (columns + rows), so case .py files
read like MATPOWER data, not like 150-line JAX array dumps.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import jax.numpy as jnp

from powermarketjax.case.case_data import CaseData
from powermarketjax.case.case_matrices import build_case_matrices


def _col(table: np.ndarray, columns: List[str], name: str) -> np.ndarray:
    """Extract a column from a 2-D table by header name."""
    return table[:, columns.index(name)]


def _col_or(table: np.ndarray, columns: List[str], name: str,
            default: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Extract a column if present, else *default*."""
    if name in columns:
        return table[:, columns.index(name)]
    return default


# ── fuel-type string → int mapping (matches CaseData.unit_fuel_type) ────
FUEL_MAP = {
    "nuclear": 1, "coal": 2, "gas": 3, "oil": 4,
    "hydro": 5, "wind": 6, "solar": 7,
}


def build_case_from_tables(
    nodes_cols: List[str],
    nodes_data: Sequence,
    units_cols: List[str],
    units_data: Sequence,
    lines_cols: List[str],
    lines_data: Sequence,
    loads_cols: List[str],
    loads_data: Sequence,
    *,
    base_mva: float = 100.0,
    slack_bus_id: Optional[float] = None,
) -> CaseData:
    """Build a :class:`CaseData` from four column+data table pairs.

    The table format is identical to PowerZoo's ``DataFrame(columns, data)``:
    each table is a list of column names plus a list-of-lists of row data.

    All heavy lifting (PTDF, mapping matrices, index conversion) is handled
    internally — callers only supply the raw numbers.

    Args:
        nodes_cols / nodes_data:  Node (bus) table.
            Required columns: ``id``.
            Optional: ``type``, ``Pd``, ``Qd``, ``Gs``, ``Bs``,
            ``Vmin``, ``Vmax``, ``x``, ``y``,
            ``Pd_A``/``Qd_A``/``Pd_B``/``Qd_B``/``Pd_C``/``Qd_C`` (3-phase).
        units_cols / units_data:  Generator table.
            Required: ``id``, ``bus_id``, ``mc_a``, ``mc_b``, ``mc_c``,
            ``p_max``, ``p_min``.
        lines_cols / lines_data:  Branch table.
            Required: ``id``, ``from``, ``to``, ``x``, ``floor``, ``cap``.
            Optional: ``r``, ``b``, ``ratio``, ``angle``, ``status``, ``rateA``.
        loads_cols / loads_data:  Load table.
            Required: ``id``, ``bus_id``, ``d_max``, ``d_min``.
        base_mva:  System base (default 100).
        slack_bus_id:  Slack bus ID. ``None`` → first type-3 node or first node.

    Returns:
        Fully-populated :class:`CaseData`.
    """
    def _strip_string_cols(cols, data):
        """Remove string columns, returning (clean_cols, numeric_array, {col: [str_values]})."""
        str_cols = {}
        if not data:
            return cols, np.empty((0, len(cols))), str_cols
        for ci, c in enumerate(cols):
            if any(isinstance(row[ci], str) for row in data):
                str_cols[c] = [row[ci] for row in data]
        if not str_cols:
            return cols, np.array(data, dtype=np.float64), str_cols
        keep = [i for i, c in enumerate(cols) if c not in str_cols]
        clean_cols = [cols[i] for i in keep]
        clean_data = [[row[i] for i in keep] for row in data]
        return clean_cols, np.array(clean_data, dtype=np.float64), str_cols

    nodes_cols, nodes, _ = _strip_string_cols(nodes_cols, list(nodes_data))
    units_cols, units, units_str = _strip_string_cols(units_cols, list(units_data))
    lines_cols, lines, _ = _strip_string_cols(lines_cols, list(lines_data))
    loads_cols, loads, _ = _strip_string_cols(loads_cols, list(loads_data))

    # Fuel-type string → int encoding
    unit_fuel_type_arr = None
    if "type" in units_str:
        raw_types = units_str["type"]
        unit_fuel_type_arr = np.array(
            [FUEL_MAP.get(str(t).strip().lower(), 0) for t in raw_types], dtype=np.float64
        )

    n_nodes, n_units, n_lines, n_loads = (
        len(nodes), len(units), len(lines), len(loads),
    )

    # ── Nodes ────────────────────────────────────────────────────────────
    node_ids = jnp.array(_col(nodes, nodes_cols, "id"), dtype=jnp.float32)
    node_x = jnp.array(_col_or(nodes, nodes_cols, "x", np.zeros(n_nodes)), dtype=jnp.float32)
    node_y = jnp.array(_col_or(nodes, nodes_cols, "y", np.zeros(n_nodes)), dtype=jnp.float32)

    node_type_arr = _col_or(nodes, nodes_cols, "type")
    node_type = jnp.array(node_type_arr, dtype=jnp.int32) if node_type_arr is not None else None

    node_pd = jnp.array(_col_or(nodes, nodes_cols, "Pd", np.zeros(n_nodes)), dtype=jnp.float32)
    node_qd = jnp.array(_col_or(nodes, nodes_cols, "Qd", np.zeros(n_nodes)), dtype=jnp.float32)
    node_gs = jnp.array(_col_or(nodes, nodes_cols, "Gs", np.zeros(n_nodes)), dtype=jnp.float32)
    node_bs = jnp.array(_col_or(nodes, nodes_cols, "Bs", np.zeros(n_nodes)), dtype=jnp.float32)
    node_v_min = jnp.array(_col_or(nodes, nodes_cols, "Vmin", np.full(n_nodes, 0.94)), dtype=jnp.float32)
    node_v_max = jnp.array(_col_or(nodes, nodes_cols, "Vmax", np.full(n_nodes, 1.06)), dtype=jnp.float32)

    # Three-phase
    node_pd_a = _safe_jnp(_col_or(nodes, nodes_cols, "Pd_A"))
    node_qd_a = _safe_jnp(_col_or(nodes, nodes_cols, "Qd_A"))
    node_pd_b = _safe_jnp(_col_or(nodes, nodes_cols, "Pd_B"))
    node_qd_b = _safe_jnp(_col_or(nodes, nodes_cols, "Qd_B"))
    node_pd_c = _safe_jnp(_col_or(nodes, nodes_cols, "Pd_C"))
    node_qd_c = _safe_jnp(_col_or(nodes, nodes_cols, "Qd_C"))

    if node_pd_a is not None:
        node_pd = node_pd_a + node_pd_b + node_pd_c
        node_qd = node_qd_a + node_qd_b + node_qd_c

    # ── Units ────────────────────────────────────────────────────────────
    unit_ids = jnp.array(_col(units, units_cols, "id"), dtype=jnp.float32)
    unit_bus_ids = jnp.array(_col(units, units_cols, "bus_id"), dtype=jnp.float32)
    unit_cost_a = jnp.array(_col(units, units_cols, "mc_a"), dtype=jnp.float32)
    unit_cost_b = jnp.array(_col(units, units_cols, "mc_b"), dtype=jnp.float32)
    unit_cost_c = jnp.array(_col(units, units_cols, "mc_c"), dtype=jnp.float32)

    # p_max / p_min — PowerZoo uses both naming conventions
    if "p_max" in units_cols:
        unit_p_max = jnp.array(_col(units, units_cols, "p_max"), dtype=jnp.float32)
        unit_p_min = jnp.array(_col(units, units_cols, "p_min"), dtype=jnp.float32)
    elif "Pmax" in units_cols:
        unit_p_max = jnp.array(_col(units, units_cols, "Pmax"), dtype=jnp.float32)
        unit_p_min = jnp.array(_col(units, units_cols, "Pmin"), dtype=jnp.float32)
    else:
        unit_p_max = jnp.zeros(n_units)
        unit_p_min = jnp.zeros(n_units)

    unit_pg = _safe_jnp(_col_or(units, units_cols, "Pg"))
    unit_qg = _safe_jnp(_col_or(units, units_cols, "Qg"))
    unit_q_max = _safe_jnp(_col_or(units, units_cols, "Qmax"))
    unit_q_min = _safe_jnp(_col_or(units, units_cols, "Qmin"))
    unit_vg = _safe_jnp(_col_or(units, units_cols, "Vg"))
    unit_mbase = _safe_jnp(_col_or(units, units_cols, "mBase"))
    unit_status = _safe_jnp(_col_or(units, units_cols, "status"))

    # UC fields
    unit_ramp_up = _safe_jnp(_col_or(units, units_cols, "ramp_up"))
    unit_ramp_down = _safe_jnp(_col_or(units, units_cols, "ramp_down"))
    unit_min_up_time = _safe_jnp(_col_or(units, units_cols, "min_up_time"))
    unit_min_down_time = _safe_jnp(_col_or(units, units_cols, "min_down_time"))
    unit_init_power = _safe_jnp(_col_or(units, units_cols, "init_power"))
    unit_init_state = _safe_jnp(_col_or(units, units_cols, "init_state"))
    unit_startup_cost = _safe_jnp(_col_or(units, units_cols, "init_start_up_cost"))
    unit_no_load_cost = _safe_jnp(_col_or(units, units_cols, "init_no_load_cost"))
    unit_keep_time = _safe_jnp(_col_or(units, units_cols, "keep_time"))
    unit_fuel_type = _safe_jnp(unit_fuel_type_arr)  # string→int done during table parse

    # ── Lines ────────────────────────────────────────────────────────────
    line_ids = jnp.array(_col(lines, lines_cols, "id"), dtype=jnp.float32)
    line_from = jnp.array(_col(lines, lines_cols, "from"), dtype=jnp.float32)
    line_to = jnp.array(_col(lines, lines_cols, "to"), dtype=jnp.float32)

    if "x" in lines_cols:
        line_x = jnp.array(_col(lines, lines_cols, "x"), dtype=jnp.float32)
    else:
        line_x = jnp.ones(n_lines) * 0.01

    floor_arr = _col(lines, lines_cols, "floor")
    cap_arr = _col(lines, lines_cols, "cap")
    floor_arr = np.where(floor_arr == 0, -1e6, floor_arr)
    cap_arr = np.where(cap_arr == 0, 1e6, cap_arr)
    line_floor = jnp.array(floor_arr, dtype=jnp.float32)
    line_cap = jnp.array(cap_arr, dtype=jnp.float32)

    line_r = _safe_jnp(_col_or(lines, lines_cols, "r"))
    line_b = _safe_jnp(_col_or(lines, lines_cols, "b"))
    line_ratio = _safe_jnp(_col_or(lines, lines_cols, "ratio"))
    line_angle = _safe_jnp(_col_or(lines, lines_cols, "angle"))
    line_status = _safe_jnp(_col_or(lines, lines_cols, "status"))
    line_rate_a = _safe_jnp(_col_or(lines, lines_cols, "rateA"))

    # ── Loads ────────────────────────────────────────────────────────────
    load_ids = jnp.array(_col(loads, loads_cols, "id"), dtype=jnp.float32)
    load_bus_ids = jnp.array(_col(loads, loads_cols, "bus_id"), dtype=jnp.float32)
    load_d_max = jnp.array(_col(loads, loads_cols, "d_max"), dtype=jnp.float32)
    load_d_min = jnp.array(_col(loads, loads_cols, "d_min"), dtype=jnp.float32)

    # ── Slack bus ────────────────────────────────────────────────────────
    if slack_bus_id is None:
        if node_type is not None:
            slack_mask = np.array(node_type_arr) == 3
            if slack_mask.any():
                slack_bus_id = float(np.array(nodes[:, nodes_cols.index("id")])[np.argmax(slack_mask)])
        if slack_bus_id is None:
            slack_bus_id = float(nodes[0, nodes_cols.index("id")])

    # ── Compute matrices ─────────────────────────────────────────────────
    matrices = build_case_matrices(
        node_ids=node_ids,
        unit_bus_ids=unit_bus_ids,
        load_bus_ids=load_bus_ids,
        line_from=line_from,
        line_to=line_to,
        line_x=line_x,
        slack_bus_id=slack_bus_id,
    )

    # ── Assemble CaseData ────────────────────────────────────────────────
    return CaseData(
        n_nodes=n_nodes, n_lines=n_lines, n_units=n_units, n_loads=n_loads,
        node_ids=node_ids, node_x=node_x, node_y=node_y,
        unit_ids=unit_ids, unit_bus_ids=unit_bus_ids,
        unit_p_min=unit_p_min, unit_p_max=unit_p_max,
        unit_cost_a=unit_cost_a, unit_cost_b=unit_cost_b, unit_cost_c=unit_cost_c,
        line_ids=line_ids, line_from=line_from, line_to=line_to,
        line_x=line_x, line_cap=line_cap, line_floor=line_floor,
        load_ids=load_ids, load_bus_ids=load_bus_ids,
        load_d_max=load_d_max, load_d_min=load_d_min,
        PTDF=matrices['PTDF'],
        nodes_units_map=matrices['nodes_units_map'],
        nodes_loads_map=matrices['nodes_loads_map'],
        unit_node_idx=matrices['unit_node_idx'],
        load_node_idx=matrices['load_node_idx'],
        line_from_idx=matrices['line_from_idx'],
        line_to_idx=matrices['line_to_idx'],
        slack_bus_idx=matrices['slack_bus_idx'],
        # AC line
        line_r=line_r, line_b=line_b,
        line_ratio=line_ratio, line_angle=line_angle, line_status=line_status,
        line_rate_a=line_rate_a,
        # AC unit
        unit_q_min=unit_q_min, unit_q_max=unit_q_max,
        unit_pg=unit_pg, unit_qg=unit_qg, unit_vg=unit_vg,
        unit_status=unit_status, unit_mbase=unit_mbase,
        # AC node
        node_type=node_type, node_pd=node_pd, node_qd=node_qd,
        node_gs=node_gs, node_bs=node_bs,
        node_v_min=node_v_min, node_v_max=node_v_max,
        base_mva=base_mva,
        # UC
        unit_ramp_up=unit_ramp_up, unit_ramp_down=unit_ramp_down,
        unit_min_up_time=unit_min_up_time, unit_min_down_time=unit_min_down_time,
        unit_init_power=unit_init_power, unit_init_state=unit_init_state,
        unit_startup_cost=unit_startup_cost, unit_no_load_cost=unit_no_load_cost,
        unit_keep_time=unit_keep_time, unit_fuel_type=unit_fuel_type,
        # Three-phase
        node_pd_a=node_pd_a, node_qd_a=node_qd_a,
        node_pd_b=node_pd_b, node_qd_b=node_qd_b,
        node_pd_c=node_pd_c, node_qd_c=node_qd_c,
    )


def _safe_jnp(arr) -> Optional[jnp.ndarray]:
    """Convert to jnp.float32 if not None."""
    if arr is None:
        return None
    return jnp.array(arr, dtype=jnp.float32)


def case_to_jax(case: Any, *, slack_bus_id: Optional[float] = None) -> CaseData:
    """Convert a PowerZoo ``ClearCase`` instance to :class:`CaseData`.

    This is **Direction B** — runtime conversion from a live PowerZoo object.
    Handles all column naming variants (``p_max``/``Pmax``, ``Pd``/``Pd_A``, etc.).

    Args:
        case:  A PowerZoo ``ClearCase`` (or compatible) instance with
               ``.nodes``, ``.units``, ``.lines``, ``.loads`` DataFrames.
        slack_bus_id:  Override slack bus. ``None`` → auto-detect.

    Returns:
        CaseData ready for GPU training.
    """
    import pandas as pd

    def _df_cols(df):
        return [c for c in df.columns if c != "#id"]

    def _df_data(df, cols):
        return df[cols].values.tolist()

    base_mva = getattr(case, "baseMVA", 100.0)

    # Add missing buses referenced by units/lines but absent from nodes
    nodes_df = _ensure_referenced_buses(case)
    lines_df = case.lines
    # Three-phase cases: derive r/x from line_config + length
    if hasattr(case, "line_config") and "config_name" in case.lines.columns:
        lines_df = _compute_3ph_impedance(case)

    n_cols = _df_cols(nodes_df)
    u_cols = _df_cols(case.units)
    l_cols = _df_cols(lines_df)
    d_cols = _df_cols(case.loads)

    return build_case_from_tables(
        nodes_cols=n_cols, nodes_data=_df_data(nodes_df, n_cols),
        units_cols=u_cols, units_data=_df_data(case.units, u_cols),
        lines_cols=l_cols, lines_data=_df_data(lines_df, l_cols),
        loads_cols=d_cols, loads_data=_df_data(case.loads, d_cols),
        base_mva=base_mva,
        slack_bus_id=slack_bus_id,
    )


def _ensure_referenced_buses(case):
    """Add missing buses that are referenced by units/lines but absent from nodes."""
    import pandas as pd

    node_ids = set(float(x) for x in case.nodes["id"].values)
    referenced = set()
    for bid in case.units["bus_id"].values:
        referenced.add(float(bid))
    for bid in case.lines["from"].values:
        referenced.add(float(bid))
    for bid in case.lines["to"].values:
        referenced.add(float(bid))

    missing = sorted(referenced - node_ids)
    if not missing:
        return case.nodes

    cols = [c for c in case.nodes.columns if c != "#id"]
    extra_rows = []
    for mid in missing:
        row = {c: 0.0 for c in cols}
        row["id"] = mid
        if "type" in cols:
            row["type"] = 3.0  # slack for missing substation bus
        extra_rows.append(row)

    extra_df = pd.DataFrame(extra_rows)[cols]
    merged = pd.concat([extra_df, case.nodes[cols]], ignore_index=True)
    merged = merged.sort_values("id").reset_index(drop=True)
    return merged


def _compute_3ph_impedance(case):
    """Derive r/x columns for three-phase lines from line_config impedance matrices."""
    import pandas as pd

    lc = case.line_config
    z_cols = ["Z11", "Z22", "Z33"]
    z_diag = {}
    for cid in lc["id"].values:
        idx = lc.index[lc["id"] == cid][0]
        z_diag[int(cid)] = [lc.at[idx, c] for c in z_cols]

    configs = case.lines["config_name"].values
    lengths = np.array(case.lines["length"].values, dtype=np.float64)

    r_arr = np.zeros(len(case.lines))
    x_arr = np.zeros(len(case.lines))
    for i, (cn, ln) in enumerate(zip(configs, lengths)):
        zs = z_diag.get(int(cn), [0j, 0j, 0j])
        best = max(zs, key=lambda z: abs(z))
        r_arr[i] = np.real(best) * ln
        x_arr[i] = np.imag(best) * ln
    x_arr = np.where(x_arr > 0, x_arr, 1e-4)

    lines_df = case.lines.copy()
    lines_df["r"] = r_arr
    lines_df["x"] = x_arr
    if "b" not in lines_df.columns:
        lines_df["b"] = 0.0
    return lines_df
