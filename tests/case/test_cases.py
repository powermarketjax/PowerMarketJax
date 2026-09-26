# Vendored test from PowerZooJax.
# Source        : tests/case/test_cases.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# NOT verbatim. Beyond the package rename (powerzoojax -> powermarketjax, in
# both import statements and path strings), two tests were changed on 2026-08-15
# when the SwissDN feeder 459_0 was registered with a factory that raises:
# test_load_all_cases now skips cases carrying `requires_declaration` (and a new
# test_cases_needing_a_declaration_say_so_instead_of_defaulting asserts those
# raise), and test_list_cases_filter_grid_type was restated as the filter's own
# invariants instead of the hardcoded counts 8 and 7 -- the count was never the
# property under test, and any case added under either grid type broke it.
# On 2026-08-21 `case73rts` and `case813nem` were added to ALL_CASES and to the
# two unit-commitment parametrisations, so the L0 contract and the UC-field
# checks cover them on the same footing as `case29gb`.
"""
Unit tests for all built-in power system cases.

L0: JAX contract tests (structure, JIT compat, validation)
L1: Physics sanity tests (limits, PTDF properties, topology)
"""
import pytest
import jax
import jax.numpy as jnp

from powermarketjax.case import (
    load_case, list_cases, validate_case_data,
    create_case5, create_case14, create_case33bw,
    create_case118, create_case118zh, create_case123, create_case123_1ph,
    create_case141, create_case300,
    create_case533mt_hi, create_case533mt_lo,
    create_case1354pegase, create_case2383wp,
    create_case29gb, create_case552gb,
    create_case73rts, create_case813nem,
)

# ============================================================
# Fixtures
# ============================================================

ALL_CASES = [
    ("5", create_case5),
    ("14", create_case14),
    ("33bw", create_case33bw),
    ("118", create_case118),
    ("118zh", create_case118zh),
    ("123", create_case123),
    ("123_1ph", create_case123_1ph),
    ("141", create_case141),
    ("300", create_case300),
    ("533mt_hi", create_case533mt_hi),
    ("533mt_lo", create_case533mt_lo),
    ("1354pegase", create_case1354pegase),
    ("2383wp", create_case2383wp),
    ("29gb", create_case29gb),
    ("552gb", create_case552gb),
    ("73rts", create_case73rts),
    ("813nem", create_case813nem),
]

_case_cache = {}

def _get_case(case_id, factory):
    if case_id not in _case_cache:
        _case_cache[case_id] = factory()
    return _case_cache[case_id]


@pytest.fixture(params=ALL_CASES, ids=[c[0] for c in ALL_CASES])
def case_pair(request):
    """Yields (case_id, CaseData) for each built-in case."""
    case_id, factory = request.param
    return case_id, _get_case(case_id, factory)


# ============================================================
# L0: JAX Contract / Structural Tests
# ============================================================

class TestL0Validation:
    """validate_case_data() passes for every case."""

    def test_validate_passes(self, case_pair):
        _, case = case_pair
        assert validate_case_data(case)

    def test_dimensions_positive(self, case_pair):
        _, case = case_pair
        assert case.n_nodes > 0
        assert case.n_lines > 0
        assert case.n_units > 0
        assert case.n_loads > 0

    def test_node_ids_shape(self, case_pair):
        _, case = case_pair
        assert case.node_ids is not None
        assert case.node_ids.shape == (case.n_nodes,)

    def test_unit_ids_shape(self, case_pair):
        _, case = case_pair
        assert case.unit_ids is not None
        assert case.unit_ids.shape == (case.n_units,)

    def test_line_ids_shape(self, case_pair):
        _, case = case_pair
        assert case.line_ids is not None
        assert case.line_ids.shape == (case.n_lines,)

    def test_load_ids_shape(self, case_pair):
        _, case = case_pair
        assert case.load_ids is not None
        assert case.load_ids.shape == (case.n_loads,)

    def test_ptdf_shape(self, case_pair):
        _, case = case_pair
        assert case.PTDF is not None
        assert case.PTDF.shape == (case.n_lines, case.n_nodes)

    def test_maps_shape(self, case_pair):
        _, case = case_pair
        assert case.nodes_units_map is not None
        assert case.nodes_units_map.shape == (case.n_nodes, case.n_units)
        assert case.nodes_loads_map is not None
        assert case.nodes_loads_map.shape == (case.n_nodes, case.n_loads)

    def test_idx_arrays_shape(self, case_pair):
        _, case = case_pair
        assert case.unit_node_idx.shape == (case.n_units,)
        assert case.load_node_idx.shape == (case.n_loads,)
        assert case.line_from_idx.shape == (case.n_lines,)
        assert case.line_to_idx.shape == (case.n_lines,)

    def test_required_unit_fields(self, case_pair):
        _, case = case_pair
        assert case.unit_bus_ids is not None
        assert case.unit_p_min is not None
        assert case.unit_p_max is not None
        assert case.unit_cost_a is not None
        assert case.unit_cost_b is not None
        assert case.unit_cost_c is not None

    def test_required_line_fields(self, case_pair):
        _, case = case_pair
        assert case.line_from is not None
        assert case.line_to is not None
        assert case.line_x is not None
        assert case.line_cap is not None
        assert case.line_floor is not None

    def test_required_load_fields(self, case_pair):
        _, case = case_pair
        assert case.load_bus_ids is not None
        assert case.load_d_max is not None
        assert case.load_d_min is not None

    def test_backward_compat_gen_cost_coeffs_alias(self):
        case = _get_case("5", create_case5)
        coeffs = case.gen_cost_coeffs
        assert coeffs.shape == (case.n_units, 3)
        assert jnp.allclose(coeffs[:, 0], case.unit_cost_a)
        assert jnp.allclose(coeffs[:, 1], case.unit_cost_b)
        assert jnp.allclose(coeffs[:, 2], case.unit_cost_c)


class TestL0JITCompat:
    """CaseData can be used in JIT-compiled functions."""

    def test_ptdf_matmul_jit(self, case_pair):
        _, case = case_pair
        injection = jnp.zeros(case.n_nodes)
        @jax.jit
        def compute_flows(ptdf, inj):
            return ptdf @ inj
        result = compute_flows(case.PTDF, injection)
        assert result.shape == (case.n_lines,)

    def test_unit_cost_jit(self, case_pair):
        _, case = case_pair
        @jax.jit
        def total_cost(p, a, b, c):
            return jnp.sum(a * p**2 + b * p + c)
        p = jnp.ones(case.n_units)
        cost = total_cost(p, case.unit_cost_a, case.unit_cost_b, case.unit_cost_c)
        assert jnp.isfinite(cost)


# ============================================================
# L1: Physics Sanity Tests
# ============================================================

class TestL1Physics:
    """Basic physical constraint checks."""

    def test_p_limits(self, case_pair):
        """unit p_max >= p_min for all generators."""
        _, case = case_pair
        assert jnp.all(case.unit_p_max >= case.unit_p_min), \
            "Some units have p_max < p_min"

    def test_d_limits(self, case_pair):
        """load d_max >= d_min for all loads."""
        _, case = case_pair
        assert jnp.all(case.load_d_max >= case.load_d_min), \
            "Some loads have d_max < d_min"

    def test_slack_bus_valid(self, case_pair):
        _, case = case_pair
        assert 0 <= case.slack_bus_idx < case.n_nodes

    def test_line_x_nonzero(self, case_pair):
        """All line reactances must be nonzero (required for PTDF)."""
        _, case = case_pair
        assert jnp.all(case.line_x != 0), \
            f"Zero reactances found (would cause division by zero in PTDF)"

    def test_ptdf_slack_column_near_zero(self, case_pair):
        """PTDF column at slack bus should be approximately zero."""
        _, case = case_pair
        slack_col = case.PTDF[:, case.slack_bus_idx]
        assert jnp.allclose(slack_col, 0.0, atol=1e-5), \
            f"PTDF slack column max abs: {float(jnp.abs(slack_col).max())}"

    def test_node_maps_binary(self, case_pair):
        """Mapping matrices should be binary (0 or 1)."""
        _, case = case_pair
        for name, m in [("units_map", case.nodes_units_map),
                        ("loads_map", case.nodes_loads_map)]:
            unique_vals = jnp.unique(m)
            assert jnp.all((unique_vals == 0) | (unique_vals == 1)), \
                f"{name} has non-binary values: {unique_vals}"

    def test_unit_node_idx_in_range(self, case_pair):
        _, case = case_pair
        assert jnp.all(case.unit_node_idx >= 0)
        assert jnp.all(case.unit_node_idx < case.n_nodes)

    def test_load_node_idx_in_range(self, case_pair):
        _, case = case_pair
        assert jnp.all(case.load_node_idx >= 0)
        assert jnp.all(case.load_node_idx < case.n_nodes)

    def test_line_idx_in_range(self, case_pair):
        _, case = case_pair
        assert jnp.all(case.line_from_idx >= 0)
        assert jnp.all(case.line_from_idx < case.n_nodes)
        assert jnp.all(case.line_to_idx >= 0)
        assert jnp.all(case.line_to_idx < case.n_nodes)

    def test_line_cap_ge_floor(self, case_pair):
        _, case = case_pair
        assert jnp.all(case.line_cap >= case.line_floor), \
            "Some lines have cap < floor"


# ============================================================
# L0: Case-Specific Tests (UC, Three-Phase)
# ============================================================

class TestCaseSpecific:
    """Tests for cases with UC or three-phase fields."""

    @pytest.mark.parametrize("case_id,factory", [
        ("29gb", create_case29gb),
        ("118", create_case118),
        ("14", create_case14),
        ("73rts", create_case73rts),
        ("813nem", create_case813nem),
    ])
    def test_uc_fields_present(self, case_id, factory):
        case = _get_case(case_id, factory)
        assert case.unit_ramp_up is not None, f"{case_id}: missing unit_ramp_up"
        assert case.unit_ramp_down is not None, f"{case_id}: missing unit_ramp_down"
        assert case.unit_min_up_time is not None, f"{case_id}: missing unit_min_up_time"
        assert case.unit_min_down_time is not None, f"{case_id}: missing unit_min_down_time"
        assert case.unit_init_power is not None, f"{case_id}: missing unit_init_power"
        assert case.unit_init_state is not None, f"{case_id}: missing unit_init_state"
        assert case.unit_startup_cost is not None, f"{case_id}: missing unit_startup_cost"
        assert case.unit_no_load_cost is not None, f"{case_id}: missing unit_no_load_cost"

    @pytest.mark.parametrize("case_id,factory", [
        ("29gb", create_case29gb),
        ("118", create_case118),
        ("14", create_case14),
        ("73rts", create_case73rts),
        ("813nem", create_case813nem),
    ])
    def test_uc_field_shapes(self, case_id, factory):
        case = _get_case(case_id, factory)
        n = case.n_units
        for field in ['unit_ramp_up', 'unit_ramp_down', 'unit_min_up_time',
                       'unit_min_down_time', 'unit_init_power', 'unit_init_state',
                       'unit_startup_cost', 'unit_no_load_cost']:
            arr = getattr(case, field)
            assert arr.shape == (n,), f"{case_id}.{field}: shape {arr.shape} != ({n},)"

    def test_case29gb_fuel_type(self):
        case = _get_case("29gb", create_case29gb)
        assert case.unit_fuel_type is not None
        assert case.unit_fuel_type.shape == (case.n_units,)
        # Fuel types: 1=nuclear, 2=coal, 3=gas
        unique = set(int(x) for x in jnp.unique(case.unit_fuel_type))
        assert unique <= {1, 2, 3}, f"Unexpected fuel types: {unique}"

    def test_case123_three_phase_fields(self):
        case = _get_case("123", create_case123)
        for field in ['node_pd_a', 'node_qd_a', 'node_pd_b', 'node_qd_b',
                       'node_pd_c', 'node_qd_c']:
            arr = getattr(case, field)
            assert arr is not None, f"Case123 missing {field}"
            assert arr.shape == (case.n_nodes,), f"{field} shape mismatch"

    def test_case123_phase_sum_equals_total(self):
        """node_pd should equal sum of per-phase loads."""
        case = _get_case("123", create_case123)
        total_pd = case.node_pd_a + case.node_pd_b + case.node_pd_c
        total_qd = case.node_qd_a + case.node_qd_b + case.node_qd_c
        assert jnp.allclose(case.node_pd, total_pd, atol=1e-6)
        assert jnp.allclose(case.node_qd, total_qd, atol=1e-6)


# ============================================================
# load_case() integration test
# ============================================================

class TestLoadCase:

    def test_load_all_cases(self):
        """load_case(id) works for every case that can be built without a declaration.

        Filtered on `requires_declaration` rather than on a case name: a case
        whose zero-argument factory raises (because the source supplies no
        voltage band, say) is registered on purpose, so that `load_case` reports
        *which* declaration is missing instead of "Unknown case". Skipping it by
        name would have to be repeated for the next such case.
        """
        for meta in list_cases():
            if meta.requires_declaration:
                continue
            case = load_case(meta.name)
            assert case.n_nodes > 0, f"load_case('{meta.name}') returned empty case"

    def test_cases_needing_a_declaration_say_so_instead_of_defaulting(self):
        """The other half of the filter above: those cases must raise, loudly."""
        for meta in list_cases():
            if not meta.requires_declaration:
                continue
            with pytest.raises(ValueError):
                load_case(meta.name)

    def test_load_case_invalid(self):
        with pytest.raises(ValueError, match="Unknown case"):
            load_case("nonexistent_999")

    def test_load_case_strip_prefix(self):
        """Case prefix is stripped: load_case('case5') == load_case('5')."""
        c1 = load_case("5")
        c2 = load_case("case5")
        assert c1.n_nodes == c2.n_nodes

    def test_load_case_grid_type_warning(self):
        """Requesting wrong grid_type emits UserWarning."""
        import warnings as w
        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            load_case("5", grid_type="distribution")
        assert any(issubclass(c.category, UserWarning) for c in caught)

    def test_list_cases_nonempty(self):
        cases = list_cases()
        assert len(cases) >= 15

    def test_list_cases_filter_grid_type(self):
        """What this asserts is the filter, not how many cases of each kind exist.

        It read `len(trans) == 8` and `len(dist) == 7` until 2026-08-15, when
        registering feeder 459_0 turned the second one red at 8. The count was
        never the property under test: any case added under either grid type
        breaks it, and the breakage says nothing about `list_cases`. Restated as
        the filter's own invariants -- everything returned carries the type,
        nothing carrying the type is left out, and the two partition the whole --
        it moves with the registry instead of against it.
        """
        trans = list_cases(grid_type="transmission")
        dist = list_cases(grid_type="distribution")
        assert all(m.grid_type == "transmission" for m in trans)
        assert all(m.grid_type == "distribution" for m in dist)
        every = list_cases()
        assert {m.name for m in trans} == {m.name for m in every
                                           if m.grid_type == "transmission"}
        assert {m.name for m in dist} == {m.name for m in every
                                          if m.grid_type == "distribution"}
        assert len(trans) + len(dist) == len(every)

    def test_list_cases_filter_phase(self):
        three_phase = list_cases(phase="3")
        assert len(three_phase) >= 1
        assert all(m.phase == "3" for m in three_phase)

    def test_list_cases_filter_min_buses(self):
        large = list_cases(min_buses=300)
        assert all(m.bus_count >= 300 for m in large)
