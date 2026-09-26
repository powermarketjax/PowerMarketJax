# Vendored from PowerZooJax. Do not edit to track upstream;
# this file is now this repository's code.
# Source        : powerzoojax/case/case_data.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# NOT verbatim. Beyond the package rename (powerzoojax -> powermarketjax, in
# both import statements and path strings): the docstrings of unit_cost_a /
# unit_cost_b / unit_cost_c and of gen_cost_coeffs were corrected on
# 2026-08-05. Upstream documented them under MATPOWER's *total cost*
# convention, which is wrong -- they are *marginal cost* polynomial
# coefficients. On 2026-08-07 the docstrings of unit_ramp_up / unit_ramp_down
# were corrected the same way: upstream documented them as [MW/step] when they
# are a fraction of p_max per hour. No code or data changed.
# On 2026-08-24 the trailing comment on unit_fuel_type was replaced: upstream
# documented the encoding as "(0=unknown,1=nuclear,2=coal,3=gas)", stopping at
# 3, while `case_builder.FUEL_MAP` defines 4=oil, 5=hydro, 6=wind, 7=solar and
# the built cases actually carry 0,1,2,3,4,5,7 (case552gb has 1 301 units at
# code 7). Comment text only, no field, no encoding and no data changed.
# Upstream licence: MIT, Copyright (c) 2026 PowerZooJax Contributors.
"""
CaseData: Pure JAX Arrays for GPU Training

This module defines the core data structure for power system cases.
All data is stored as JAX arrays for efficient GPU computation.

Design Principles:
1. CaseData is a frozen dataclass (immutable)
2. All arrays are JAX arrays (jnp.ndarray)
3. No Python objects or DataFrame - pure numerical data
4. Can be passed through JIT-compiled functions

Usage:
    >>> case_data = create_case5()
    >>> # Use in JIT-compiled training loop
    >>> @jax.jit
    ... def train_step(case_data, state):
    ...     line_flows = case_data.PTDF @ node_injection
    ...     return line_flows
"""

from typing import NamedTuple, Optional
import jax.numpy as jnp
import chex
from flax import struct


@struct.dataclass
class CaseData:
    """Power system case data (pure JAX arrays).
    
    This is the core data structure used in training.
    All data is stored as JAX arrays for GPU acceleration.
    
    Note: This is a frozen dataclass - all fields are immutable.
    Use `case_data.replace(field=new_value)` to create modified copies.
    
    Note: ``name`` was removed because Python ``str`` cannot be traced by
    JAX JIT.  If you need a display name, store it alongside the CaseData.
    
    Attributes:
        n_nodes: Number of buses/nodes (problem dimension).
        n_lines: Number of transmission lines.
        n_units: Number of generators.
        n_loads: Number of loads.
        node_ids: Node IDs (for reference); shape ``(n_nodes,)``.
        node_x: Node x coordinates for plotting; shape ``(n_nodes,)``.
        node_y: Node y coordinates for plotting; shape ``(n_nodes,)``.
        unit_ids: Unit IDs; shape ``(n_units,)``.
        unit_bus_ids: Bus ID where each unit is connected; shape ``(n_units,)``.
        unit_p_min: Minimum power output [MW]; shape ``(n_units,)``.
        unit_p_max: Maximum power output [MW]; shape ``(n_units,)``.
        unit_cost_a: Quadratic coefficient of the **marginal cost** curve
            [$/MW³h]; shape ``(n_units,)``. See the marginal-cost convention
            note below -- this is NOT a MATPOWER total-cost coefficient.
        unit_cost_b: Linear coefficient of the **marginal cost** curve
            [$/MW²h]; shape ``(n_units,)``.
        unit_cost_c: Constant term of the **marginal cost** curve [$/MWh],
            i.e. marginal cost at zero output; shape ``(n_units,)``.

            **Marginal-cost convention (read before computing any cost).**
            ``unit_cost_a/b/c`` are coefficients of the MARGINAL cost curve::

                MC(p) = a·p² + b·p + c            [$/MWh]
                TC(p) = (a/3)p³ + (b/2)p² + c·p   [$/h]

            MATPOWER instead stores total-cost coefficients, where
            ``TC = a·p² + b·p + c`` and ``MC = 2a·p + b``. **Converting is
            the caller's job when importing from MATPOWER.** The built-in
            cases already store MC coefficients: ``case_builder`` reads them
            from columns named ``mc_a`` / ``mc_b`` / ``mc_c``, and
            ``physics.power_flow.compute_generation_cost`` integrates them
            with the TC formula above.

            Writing ``cost = a·p² + b·p + c`` therefore yields neither MC nor
            TC, and gives a dimensionally wrong number that no existing test
            catches. Since ``reward = revenue - cost``,
            that error lands directly in the reward signal. Sanity check:
            ``case5`` has ``a = b = 0`` and ``c ∈ [10, 40]``, i.e. flat
            marginal costs of 10-40 $/MWh.
        line_ids: Line IDs; shape ``(n_lines,)``.
        line_from: From bus ID; shape ``(n_lines,)``.
        line_to: To bus ID; shape ``(n_lines,)``.
        line_x: Line reactance [p.u.]; shape ``(n_lines,)``.
        line_cap: Branch thermal rating [MVA], from MATPOWER ``rateA``.
            **Convention:** In DC mode this is treated as an active-power limit [MW]
            (``Q≈0`` approximation). In AC PF / ACOPF it is used as an apparent-power
            limit [MVA] with thermal magnitude ``√(P²+Q²)``. Do not treat this field
            as pure MW on AC paths when checking thermal limits.
        line_floor: Line capacity lower bound [MW]; shape ``(n_lines,)``.
        load_ids: Load IDs; shape ``(n_loads,)``.
        load_bus_ids: Bus ID where each load is connected; shape ``(n_loads,)``.
        load_d_max: Maximum demand [MW]; shape ``(n_loads,)``.
        load_d_min: Minimum demand [MW]; shape ``(n_loads,)``.
        PTDF: Power Transfer Distribution Factor; shape ``(n_lines, n_nodes)``.
        nodes_units_map: Node-to-unit mapping; shape ``(n_nodes, n_units)``.
        nodes_loads_map: Node-to-load mapping; shape ``(n_nodes, n_loads)``.
        unit_node_idx: Internal node index for each unit (0-indexed); shape ``(n_units,)``.
        load_node_idx: Internal node index for each load; shape ``(n_loads,)``.
        line_from_idx: Internal from-node index for lines; shape ``(n_lines,)``.
        line_to_idx: Internal to-node index for lines; shape ``(n_lines,)``.
        slack_bus_idx: Slack bus internal index (default 0).
    """
    # Dimensions
    n_nodes: int = 0
    n_lines: int = 0
    n_units: int = 0
    n_loads: int = 0
    
    # Node data
    node_ids: chex.Array = None      # (n_nodes,) float or int
    node_x: chex.Array = None        # (n_nodes,) float
    node_y: chex.Array = None        # (n_nodes,) float
    
    # Unit data
    unit_ids: chex.Array = None      # (n_units,)
    unit_bus_ids: chex.Array = None  # (n_units,)
    unit_p_min: chex.Array = None    # (n_units,)
    unit_p_max: chex.Array = None    # (n_units,)
    # MARGINAL cost curve MC(p) = a·p² + b·p + c [$/MWh], NOT MATPOWER total
    # cost. TC(p) = (a/3)p³ + (b/2)p² + c·p. See the class docstring.
    unit_cost_a: chex.Array = None   # (n_units,) quadratic term of MC
    unit_cost_b: chex.Array = None   # (n_units,) linear term of MC
    unit_cost_c: chex.Array = None   # (n_units,) MC at zero output [$/MWh]
    
    # Line data
    line_ids: chex.Array = None      # (n_lines,)
    line_from: chex.Array = None     # (n_lines,)
    line_to: chex.Array = None       # (n_lines,)
    line_x: chex.Array = None        # (n_lines,) reactance
    line_cap: chex.Array = None      # (n_lines,)
    line_floor: chex.Array = None    # (n_lines,)
    
    # Load data
    load_ids: chex.Array = None      # (n_loads,)
    load_bus_ids: chex.Array = None  # (n_loads,)
    load_d_max: chex.Array = None    # (n_loads,)
    load_d_min: chex.Array = None    # (n_loads,)
    
    # Pre-computed matrices
    PTDF: chex.Array = None          # (n_lines, n_nodes)
    nodes_units_map: chex.Array = None  # (n_nodes, n_units)
    nodes_loads_map: chex.Array = None  # (n_nodes, n_loads)
    
    # Internal indices (0-based)
    unit_node_idx: chex.Array = None    # (n_units,) int
    load_node_idx: chex.Array = None    # (n_loads,) int
    line_from_idx: chex.Array = None    # (n_lines,) int
    line_to_idx: chex.Array = None      # (n_lines,) int
    
    # Configuration
    slack_bus_idx: int = 0

    # ========== AC fields (optional, backward-compatible) ==========

    # AC line data (n_lines,)
    line_r: chex.Array = None           # Resistance [p.u.]
    line_b: chex.Array = None           # Total charging susceptance [p.u.]
    line_ratio: chex.Array = None       # Transformer tap ratio (0 = regular line → treated as 1)
    line_angle: chex.Array = None       # Phase shift [degrees]
    line_status: chex.Array = None      # 1 = active, 0 = inactive

    # AC unit data (n_units,)
    unit_q_min: chex.Array = None       # Min reactive power [MVAr]
    unit_q_max: chex.Array = None       # Max reactive power [MVAr]
    unit_pg: chex.Array = None          # Scheduled active power [MW]
    unit_qg: chex.Array = None          # Scheduled reactive power [MVAr]
    unit_vg: chex.Array = None          # Voltage setpoint [p.u.]

    # AC node data (n_nodes,)
    node_type: chex.Array = None        # 1=PQ, 2=PV, 3=Slack (MATPOWER convention)
    node_pd: chex.Array = None          # Active load at bus [MW]
    node_qd: chex.Array = None          # Reactive load at bus [MVAr]
    node_gs: chex.Array = None          # Bus shunt conductance [MW at 1 p.u. V]
    node_bs: chex.Array = None          # Bus shunt susceptance [MVAr at 1 p.u. V]
    node_v_min: chex.Array = None       # Min voltage magnitude [p.u.]
    node_v_max: chex.Array = None       # Max voltage magnitude [p.u.]

    # AC line rating (n_lines,)
    line_rate_a: chex.Array = None       # Long-term thermal rating [MVA]

    # AC unit operational (n_units,)
    unit_status: chex.Array = None       # 1 = in-service, 0 = out-of-service
    unit_mbase: chex.Array = None        # Machine base [MVA]

    # ========== UC fields (optional, backward-compatible) ==========

    # Unit commitment data (n_units,)
    # Ramp limits are a FRACTION OF p_max PER HOUR, not MW.  MW/step is
    # ramp_up * p_max * delta_t_hours.  All 66 case29gb units carry 0.700
    # despite p_max spanning 12-7832 MW, which only makes sense as a fraction;
    # upstream comparison_tso.py:206 converts it exactly that way.  Read as
    # MW/step the whole 66-unit fleet could move 46 MW/h against a median GB
    # hourly demand swing of 1269 MW -- the LP stays feasible and still
    # returns an LMP, it just describes a grid that cannot follow load.
    unit_ramp_up: chex.Array = None      # Ramp-up limit [fraction of p_max per hour]
    unit_ramp_down: chex.Array = None    # Ramp-down limit [fraction of p_max per hour]
    unit_min_up_time: chex.Array = None  # Minimum on time [hours]
    unit_min_down_time: chex.Array = None # Minimum off time [hours]
    unit_init_power: chex.Array = None   # Initial power output [MW]
    unit_init_state: chex.Array = None   # Initial on/off (1=on, 0=off)
    unit_startup_cost: chex.Array = None # Startup cost [$]
    unit_no_load_cost: chex.Array = None # No-load cost [$/h]
    unit_keep_time: chex.Array = None    # Initial keep-time [hours]
    # Fuel type int, encoded by `case_builder.FUEL_MAP`: 0=unknown, 1=nuclear,
    # 2=coal, 3=gas, 4=oil, 5=hydro, 6=wind, 7=solar.  The codes above 3 are not
    # hypothetical: `case73rts` carries oil, `case552gb` carries hydro and solar,
    # `case813nem` both.  Upstream's comment stopped at 3.
    unit_fuel_type: chex.Array = None

    # ========== Three-phase fields (optional, backward-compatible) ==========

    # Per-phase node loads (n_nodes,) — only populated for three-phase cases
    node_pd_a: chex.Array = None         # Phase-A active load [MW]
    node_qd_a: chex.Array = None         # Phase-A reactive load [MVAr]
    node_pd_b: chex.Array = None         # Phase-B active load [MW]
    node_qd_b: chex.Array = None         # Phase-B reactive load [MVAr]
    node_pd_c: chex.Array = None         # Phase-C active load [MW]
    node_qd_c: chex.Array = None         # Phase-C reactive load [MVAr]

    # System
    base_mva: float = 100.0

    # Distribution topology (populated by BFS setup)
    path_matrix: chex.Array = None       # (n_lines, n_nodes) BFS path indicator
    downstream_matrix: chex.Array = None # (n_lines, n_nodes) downstream node indicator
    sending_node_idx: chex.Array = None  # (n_lines,) sending-end bus index
    receiving_node_idx: chex.Array = None # (n_lines,) receiving-end bus index

    @property
    def gen_cost_coeffs(self) -> chex.Array:
        """Backward-compatible generator cost coefficients alias.

        Older smoke scripts expect ``case.gen_cost_coeffs`` to expose the
        per-generator cost triplets. The canonical storage remains
        ``unit_cost_a`` / ``unit_cost_b`` / ``unit_cost_c``.

        These are **marginal cost** coefficients, ``MC(p) = a·p² + b·p + c``,
        not MATPOWER total-cost coefficients. See the class docstring.

        Returns:
            ``(n_units, 3)`` array of ``[a, b, c]`` per unit.
        """
        return jnp.stack(
            [self.unit_cost_a, self.unit_cost_b, self.unit_cost_c], axis=-1
        )


class CaseArrays(NamedTuple):
    """Raw arrays for case definition (before computing matrices).
    
    Used as intermediate format when defining cases.
    """
    nodes: jnp.ndarray   # (n_nodes, n_cols) - id, x, y, ...
    units: jnp.ndarray   # (n_units, n_cols) - id, bus_id, costs, limits, ...
    lines: jnp.ndarray   # (n_lines, n_cols) - id, from, to, x, limits, ...
    loads: jnp.ndarray   # (n_loads, n_cols) - id, bus_id, limits, ...


def validate_case_data(case_data: CaseData) -> bool:
    """Shape/length smoke check for CaseData. **Not** a data validator.

    Checks the lengths of ``*_ids``, the UC unit arrays and the three-phase
    node arrays against the corresponding ``n_*``, plus the PTDF shape.

    Does **not** check: any value (NaN, inf, negative capacity, zero
    reactance, ``unit_p_min > unit_p_max``), any index range
    (``line_from_idx >= n_nodes`` passes, and jnp then silently clamps the
    lookup to the last element), or the lengths of the data arrays
    themselves (``unit_p_max``, ``line_cap``, ``load_d_max``,
    ``unit_node_idx``, ``line_from_idx``, ``nodes_units_map``, ...).
    Measured 2026-08-06: 4 of 20 corruptions caught. Do not use this as a
    gate on data the market layer depends on.

    Args:
        case_data: CaseData to validate

    Returns:
        True if the checked shapes agree.

    Raises:
        AssertionError: If a checked shape disagrees. These are ``assert``
            statements, so ``python -O`` strips them and the function then
            catches nothing at all.
    """
    # Check dimensions
    if case_data.node_ids is not None:
        assert len(case_data.node_ids) == case_data.n_nodes, \
            f"node_ids length {len(case_data.node_ids)} != n_nodes {case_data.n_nodes}"
    
    if case_data.unit_ids is not None:
        assert len(case_data.unit_ids) == case_data.n_units, \
            f"unit_ids length {len(case_data.unit_ids)} != n_units {case_data.n_units}"
    
    if case_data.line_ids is not None:
        assert len(case_data.line_ids) == case_data.n_lines, \
            f"line_ids length {len(case_data.line_ids)} != n_lines {case_data.n_lines}"
    
    if case_data.load_ids is not None:
        assert len(case_data.load_ids) == case_data.n_loads, \
            f"load_ids length {len(case_data.load_ids)} != n_loads {case_data.n_loads}"
    
    # Check PTDF shape
    if case_data.PTDF is not None:
        assert case_data.PTDF.shape == (case_data.n_lines, case_data.n_nodes), \
            f"PTDF shape {case_data.PTDF.shape} != ({case_data.n_lines}, {case_data.n_nodes})"

    # Check UC unit arrays (n_units,) when present
    _uc_unit_fields = [
        'unit_ramp_up', 'unit_ramp_down', 'unit_min_up_time',
        'unit_min_down_time', 'unit_init_power', 'unit_init_state',
        'unit_startup_cost', 'unit_no_load_cost', 'unit_keep_time',
        'unit_fuel_type', 'unit_status', 'unit_mbase',
    ]
    for fname in _uc_unit_fields:
        arr = getattr(case_data, fname, None)
        if arr is not None:
            assert len(arr) == case_data.n_units, \
                f"{fname} length {len(arr)} != n_units {case_data.n_units}"

    # Check three-phase node arrays (n_nodes,) when present
    _3ph_node_fields = [
        'node_pd_a', 'node_qd_a', 'node_pd_b', 'node_qd_b',
        'node_pd_c', 'node_qd_c',
    ]
    for fname in _3ph_node_fields:
        arr = getattr(case_data, fname, None)
        if arr is not None:
            assert len(arr) == case_data.n_nodes, \
                f"{fname} length {len(arr)} != n_nodes {case_data.n_nodes}"

    return True


def scale_min_output(case, scale: float):
    """The registered case at a different minimum-generation level.

    The third scenario scale, beside `cap_scale` and `ramp_scale`.  It exists
    because `case73rts` carries an aggregate `sum p_min / sum p_max` of 0.464
    against 0.274 for `case813nem` and 0.200 for `case29gb`, and that ratio --
    not the demand floor -- is what makes its chained commitment infeasible: a
    commitment sized for the day's peak cannot shut down to the trough, because
    the committed minimum alone exceeds it.

    **A scale, not an edited data file.**  `load_case` resolves a name and the
    products record only that name, so a case edited in place yields products
    indistinguishable by name, by derived path and by meta from products built
    on the registered data.  Scaling per run and recording the multiplier is what
    `cap_scale` and `ramp_scale` already do, and it leaves RTS-96 as RTS-96.

    **It scales the case, not each operator.**  `segment_costs`, `make_relax`,
    `make_clearing` and `precommit.build` all read `unit_p_min` for themselves,
    and one of them silently not receiving a threaded parameter would run the
    unscaled problem while the log reported the scaled one.  Scaling once makes
    that unrepresentable.

    `unit_p_max` deliberately does not move: capacity is not a
    minimum-generation property.  The dispatchable range `p_max - p_min` widens
    accordingly and `segment_costs` re-integrates the true cost curve over it,
    which is the intended consequence and not a side effect.
    """
    if scale == 1.0:
        return case
    return case.replace(unit_p_min=case.unit_p_min * scale)
