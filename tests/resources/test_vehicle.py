# Vendored test from PowerZooJax.
# Source        : tests/resource/test_vehicle.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
"""L0 + L1 tests for VehicleEnv — Electric Vehicle resource.

L0: JAX contracts — JIT, vmap, pytree
L1: SOC dynamics, commute scheduling, availability, sign convention
"""

import pytest
import jax
import jax.numpy as jnp

from powermarketjax.resources.vehicle import (
    VehicleEnv, VehicleState, VehicleParams,
    make_vehicle_params, MAX_TRIPS,
)


# ========================== Fixtures ==========================

@pytest.fixture
def env():
    return VehicleEnv()


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def ev_params():
    """Standard EV: 60 kWh, 7 kW charger, single trip 8am-6pm, 60-min steps."""
    return make_vehicle_params(
        E_max_kWh=60.0,
        soc_init=0.8,
        soc_min=0.1,
        soc_max=0.95,
        soc_departure_min=0.8,
        p_charge_max_kW=7.0,
        p_discharge_max_kW=7.0,
        eta_charge=0.95,
        eta_discharge=0.95,
        delta_t_minutes=60.0,
    )


@pytest.fixture
def ev_multi_params():
    """EV with 3 trips, 60-min resolution."""
    return make_vehicle_params(
        E_max_kWh=60.0,
        soc_init=0.8,
        p_charge_max_kW=7.0,
        p_discharge_max_kW=7.0,
        commute_schedule=[
            {'departure': 8.0, 'arrival': 9.0, 'energy_kWh': 10.0},
            {'departure': 12.0, 'arrival': 13.0, 'energy_kWh': 5.0},
            {'departure': 18.0, 'arrival': 19.0, 'energy_kWh': 10.0},
        ],
        delta_t_minutes=60.0,
    )


# ========================== L0 JAX Contracts ==========================

class TestVehicleJaxContracts:

    def test_reset_jit(self, env, ev_params, key):
        obs, state = env.reset(key, ev_params)
        assert obs.shape == (9,)
        assert state.soc.shape == ()

    def test_step_jit(self, env, ev_params, key):
        _, state = env.reset(key, ev_params)
        obs, s2, r, costs, d, info = env.step(key, state, jnp.float32(0.0), ev_params)
        assert obs.shape == (9,)
        assert costs.shape == (1,)

    def test_pytree_stable(self, env, ev_params, key):
        _, s0 = env.reset(key, ev_params)
        _, s1, *_ = env.step(key, s0, jnp.float32(0.0), ev_params)
        assert jax.tree_util.tree_structure(s0) == jax.tree_util.tree_structure(s1)

    def test_vmap(self, env, ev_params, key):
        _, state = env.reset(key, ev_params)
        b = 4
        states = jax.tree_util.tree_map(lambda x: jnp.stack([x] * b), state)
        keys = jax.random.split(key, b)
        actions = jnp.array([3.0 / 7.0, -3.0 / 7.0, 0.0, 1.0])

        def f(k, s, a):
            return env.step(k, s, a, ev_params)

        obs_b, _, _, costs_b, _, _ = jax.vmap(f)(keys, states, actions)
        assert obs_b.shape == (b, 9)
        assert costs_b.shape == (b, 1)


# ========================== L0 make_vehicle_params ==========================

class TestMakeVehicleParams:

    def test_kwh_to_mwh_conversion(self):
        p = make_vehicle_params(E_max_kWh=60.0)
        assert p.capacity_mwh == pytest.approx(0.06)

    def test_kw_to_mw_conversion(self):
        p = make_vehicle_params(p_charge_max_kW=7.0, p_discharge_max_kW=11.0)
        assert p.p_charge_max_mw == pytest.approx(0.007)
        assert p.p_discharge_max_mw == pytest.approx(0.011)

    def test_trips_sorted(self):
        p = make_vehicle_params(commute_schedule=[
            {'departure': 18.0, 'arrival': 19.0, 'energy_kWh': 10.0},
            {'departure': 8.0, 'arrival': 9.0, 'energy_kWh': 10.0},
        ])
        assert float(p.trip_departures[0]) < float(p.trip_departures[1])

    def test_default_single_trip(self):
        p = make_vehicle_params()
        assert p.n_trips == 1
        assert float(p.trip_departures[0]) == 8.0
        assert float(p.trip_arrivals[0]) == 18.0

    def test_unused_slots_sentinel(self):
        p = make_vehicle_params()
        assert float(p.trip_departures[1]) == float('inf')
        assert float(p.trip_arrivals[1]) == float('inf')


# ========================== L1 Sign Convention ==========================

class TestEVSignConvention:

    def test_charge_negative(self, env, ev_params, key):
        _, state = env.reset(key, ev_params)
        _, state, *_ = env.step(key, state, jnp.float32(-5.0 / 7.0), ev_params)
        assert float(state.current_p_mw) < 0

    def test_discharge_positive(self, env, ev_params, key):
        _, state = env.reset(key, ev_params)
        _, state, *_ = env.step(key, state, jnp.float32(5.0 / 7.0), ev_params)
        assert float(state.current_p_mw) > 0

    def test_idle_zero(self, env, ev_params, key):
        _, state = env.reset(key, ev_params)
        _, state, *_ = env.step(key, state, jnp.float32(0.0), ev_params)
        assert float(state.current_p_mw) == pytest.approx(0.0)


# ========================== L1 SOC Dynamics ==========================

class TestEVSOCDynamics:

    def test_charge_increases_soc(self, env, ev_params, key):
        _, state = env.reset(key, ev_params)
        soc0 = float(state.soc)
        _, state, *_ = env.step(key, state, jnp.float32(-5.0 / 7.0), ev_params)
        assert float(state.soc) > soc0

    def test_discharge_decreases_soc(self, env, ev_params, key):
        _, state = env.reset(key, ev_params)
        soc0 = float(state.soc)
        _, state, *_ = env.step(key, state, jnp.float32(5.0 / 7.0), ev_params)
        assert float(state.soc) < soc0

    def test_charge_energy_accounting(self, env, ev_params, key):
        """G2V: energy stored = |P| × Δt × η_c."""
        _, state = env.reset(key, ev_params)
        soc0 = float(state.soc)
        action_norm = -5.0 / 7.0  # maps to -0.005 MW (5 kW charge)
        _, state, *_ = env.step(key, state, jnp.float32(action_norm), ev_params)
        p_mw = 0.005
        dt = ev_params.delta_t_hours
        expected_delta_soc = p_mw * dt * ev_params.eta_charge / ev_params.capacity_mwh
        actual_delta_soc = float(state.soc) - soc0
        assert actual_delta_soc == pytest.approx(expected_delta_soc, rel=1e-3)

    def test_discharge_energy_accounting(self, env, ev_params, key):
        """V2G: energy from cell = P × Δt / η_d."""
        _, state = env.reset(key, ev_params)
        soc0 = float(state.soc)
        action_norm = 5.0 / 7.0  # maps to 0.005 MW
        _, state, *_ = env.step(key, state, jnp.float32(action_norm), ev_params)
        p_mw = 0.005
        dt = ev_params.delta_t_hours
        expected_delta_soc = p_mw * dt / (ev_params.eta_discharge * ev_params.capacity_mwh)
        actual_delta_soc = soc0 - float(state.soc)
        assert actual_delta_soc == pytest.approx(expected_delta_soc, rel=1e-3)


# ========================== L1 Availability / Commute ==========================

class TestAvailability:

    def test_starts_at_home(self, env, ev_params, key):
        _, state = env.reset(key, ev_params)
        assert float(state.is_home) == 1.0
        assert float(state.time_of_day) == 0.0

    def test_vehicle_departs(self, env, ev_params, key):
        """Departure at 8:00. After 8 steps, time_of_day=8.0.
        Step 9 sees time_of_day=8.0 at start → departure fires."""
        _, state = env.reset(key, ev_params)
        for _ in range(9):
            _, state, *_ = env.step(key, state, jnp.float32(0.0), ev_params)
        assert float(state.is_home) == pytest.approx(0.0), "Vehicle should have departed"

    def test_departure_consumes_energy(self, env, ev_params, key):
        """Trip energy deducted from SOC at departure."""
        _, state = env.reset(key, ev_params)
        soc_before = float(state.soc)
        for _ in range(9):  # departure fires on step 9
            _, state, *_ = env.step(key, state, jnp.float32(0.0), ev_params)
        # Default trip: 15 kWh = 0.015 MWh, capacity = 0.06 MWh → loss = 0.25
        expected_loss = 0.015 / ev_params.capacity_mwh
        assert float(state.soc) < soc_before
        assert (soc_before - float(state.soc)) >= expected_loss - 0.02

    def test_vehicle_arrives(self, env, ev_params, key):
        """Vehicle arrives at home after arrival time."""
        _, state = env.reset(key, ev_params)
        # Depart on step 9, then arrive when time_of_day hits 18.0
        for _ in range(9):
            _, state, *_ = env.step(key, state, jnp.float32(0.0), ev_params)
        assert float(state.is_home) == 0.0, "Should have departed"
        # Need enough steps to reach arrival=18.0;
        # After 9 steps time_of_day=9.0; arrival checked at start of each step.
        # Step 10: time=9.0→10.0, step 18: time=17.0→18.0, step 19: time=18.0 → arrival
        for _ in range(10):
            _, state, *_ = env.step(key, state, jnp.float32(0.0), ev_params)
        assert float(state.is_home) == pytest.approx(1.0), "Vehicle should be home after arrival"

    def test_no_charge_while_away(self, env, ev_params, key):
        """Power is zero when vehicle not at home."""
        _, state = env.reset(key, ev_params)
        # Advance past departure (step 9 triggers departure)
        for _ in range(9):
            _, state, *_ = env.step(key, state, jnp.float32(0.0), ev_params)
        assert float(state.is_home) == 0.0, "Should be away"
        soc_before = float(state.soc)
        _, state, *_ = env.step(key, state, jnp.float32(-1.0), ev_params)
        assert float(state.current_p_mw) == pytest.approx(0.0)
        assert float(state.soc) == pytest.approx(soc_before)

    def test_multi_trip_departures_and_arrivals(self, env, ev_multi_params, key):
        """Multiple departures and arrivals in a day."""
        _, state = env.reset(key, ev_multi_params)
        departures = 0
        arrivals = 0
        was_home = True
        for _ in range(24):
            _, state, *_ = env.step(key, state, jnp.float32(0.0), ev_multi_params)
            now_home = float(state.is_home) > 0.5
            if was_home and not now_home:
                departures += 1
            if not was_home and now_home:
                arrivals += 1
            was_home = now_home
        assert departures >= 2
        assert arrivals >= 2


# ========================== L1 SOC Constraints ==========================

class TestEVSOCConstraints:

    def test_soc_never_below_min(self, env, ev_params, key):
        _, state = env.reset(key, ev_params)
        for _ in range(100):
            _, state, *_ = env.step(key, state, jnp.float32(1.0), ev_params)
        assert float(state.soc) >= ev_params.soc_min - 1e-6

    def test_soc_never_above_max(self, env, ev_params, key):
        _, state = env.reset(key, ev_params)
        for _ in range(100):
            _, state, *_ = env.step(key, state, jnp.float32(-1.0), ev_params)
        assert float(state.soc) <= ev_params.soc_max + 1e-6


# ========================== L1 Observation ==========================

class TestEVObs:

    def test_obs_shape_dtype(self, env, ev_params, key):
        obs, _ = env.reset(key, ev_params)
        assert obs.shape == (9,)
        assert obs.dtype == jnp.float32

    def test_obs_soc_in_range(self, env, ev_params, key):
        obs, _ = env.reset(key, ev_params)
        assert 0.0 <= float(obs[0]) <= 1.0

    def test_obs_is_home_binary(self, env, ev_params, key):
        obs, _ = env.reset(key, ev_params)
        is_home = float(obs[2])
        assert is_home in [0.0, 1.0]


# ========================== L1 Reset ==========================

class TestEVReset:

    def test_reset_initial_state(self, env, ev_params, key):
        obs, state = env.reset(key, ev_params)
        assert float(state.soc) == pytest.approx(ev_params.soc_init)
        assert float(state.is_home) == 1.0
        assert float(state.time_of_day) == 0.0
        assert int(state.time_step) == 0


# ========================== L1 Departure SOC Violation ==========================

class TestDepartureSOCViolation:
    """Verify that cost_departure_soc penalises under-charged departures."""

    def test_no_penalty_when_soc_sufficient(self, env, key):
        """High initial SOC → no violation at departure."""
        params = make_vehicle_params(
            soc_init=0.9,
            soc_departure_min=0.8,
            delta_t_minutes=60.0,
        )
        _, state = env.reset(key, params)
        # Step to departure (default 8am).  reset tod=0; after 9 steps,
        # the 9th step checks tod=8.0 which hits departure window [8,9).
        for _ in range(9):
            _, state, _, costs, _, info = env.step(key, state, jnp.array([0.0]), params)
        assert float(info["cost_departure_soc"]) == pytest.approx(0.0)
        assert float(costs[0]) == pytest.approx(0.0)

    def test_penalty_when_soc_low(self, env, key):
        """Drain SOC via V2G, then observe penalty at departure time."""
        params = make_vehicle_params(
            E_max_kWh=60.0,
            soc_init=0.5,          # already below soc_departure_min
            soc_departure_min=0.8,
            delta_t_minutes=60.0,
        )
        _, state = env.reset(key, params)
        # 9 idle steps → step 9 checks tod=8.0  (departure window)
        for _ in range(9):
            _, state, _, costs, _, info = env.step(key, state, jnp.array([0.0]), params)
        # SOC 0.5 < 0.8 → violation = 0.3
        assert float(info["cost_departure_soc"]) == pytest.approx(0.3, abs=0.05)
        assert float(costs[0]) > 0.0
        assert float(info["cost_sum"]) > 0.0
