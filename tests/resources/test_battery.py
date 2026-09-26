# Vendored test from PowerZooJax.
# Source        : tests/resource/test_battery.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
"""L0 + L1 tests for BatteryEnv — BESS pure-JAX resource.

L0: JAX contracts — JIT, vmap, pytree structure stability
L1: Physics correctness — SOC dynamics, efficiency, sign convention, power clamping
"""

import pytest
import jax
import jax.numpy as jnp

from powermarketjax.resources.battery import (
    BatteryEnv, BatteryState, BatteryParams,
    compute_feasible_power, update_soc, make_battery_params,
)


# ========================== Fixtures ==========================

@pytest.fixture
def env():
    return BatteryEnv()


@pytest.fixture
def default_params():
    """Standard 50 MWh / 20 MW, η_rt=0.95 (sqrt legs), 30-min steps."""
    return make_battery_params(
        capacity_mwh=50.0, power_mw=20.0, eta_roundtrip=0.95,
        soc_min=0.1, soc_max=0.9, initial_soc=0.5,
        delta_t_hours=0.5, steps_per_day=48, max_steps=48,
    )


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


# ========================== L0 JAX Contracts ==========================

class TestBatteryJaxContracts:
    """JIT compilation, vmap, pytree stability."""

    def test_reset_jit(self, env, default_params, key):
        """reset() is JIT-compiled and returns correct shapes."""
        obs, state = env.reset(key, default_params)
        assert obs.shape == (6,)
        assert state.soc.shape == ()

    def test_step_jit(self, env, default_params, key):
        """step() is JIT-compiled."""
        obs, state = env.reset(key, default_params)
        action = jnp.float32(0.25)
        obs2, state2, reward, costs, done, info = env.step(
            key, state, action, default_params
        )
        assert obs2.shape == (6,)
        assert reward.shape == ()
        assert costs.shape == (1,)

    def test_pytree_structure_stable(self, env, default_params, key):
        """BatteryState pytree structure doesn't change across reset/step."""
        _, state0 = env.reset(key, default_params)
        _, state1, *_ = env.step(key, state0, jnp.float32(0.25), default_params)
        tree0 = jax.tree_util.tree_structure(state0)
        tree1 = jax.tree_util.tree_structure(state1)
        assert tree0 == tree1

    def test_vmap_step(self, env, default_params, key):
        """Batched step across multiple states."""
        _, state = env.reset(key, default_params)

        # Create a batch of states by stacking
        batch_size = 4
        states = jax.tree_util.tree_map(
            lambda x: jnp.stack([x] * batch_size), state
        )
        keys = jax.random.split(key, batch_size)
        actions = jnp.array([0.25, -0.25, 0.5, -0.5])

        def single_step(k, s, a):
            return env.step(k, s, a, default_params)

        obs_batch, state_batch, r_batch, c_batch, d_batch, _ = jax.vmap(single_step)(
            keys, states, actions
        )
        assert obs_batch.shape == (batch_size, 6)
        assert state_batch.soc.shape == (batch_size,)
        assert c_batch.shape == (batch_size, 1)

    def test_jit_recompile_stable(self, env, default_params, key):
        """Multiple calls don't trigger recompilation (same pytree structure)."""
        _, state = env.reset(key, default_params)
        for p in [0.25, -0.25, 0.0]:
            _, state, *_ = env.step(key, state, jnp.float32(p), default_params)
        # Just confirming no errors


# ========================== L0 make_battery_params ==========================

class TestMakeBatteryParams:

    def test_symmetric_efficiency_decomposition(self):
        """η=0.81 → η_c = η_d = 0.9."""
        p = make_battery_params(eta_roundtrip=0.81)
        assert p.eta_charge == pytest.approx(0.9, abs=1e-4)
        assert p.eta_discharge == pytest.approx(0.9, abs=1e-4)

    def test_explicit_eta_overrides(self):
        """Explicit η_c, η_d override round-trip shorthand."""
        p = make_battery_params(eta_roundtrip=0.50, eta_charge=0.95, eta_discharge=0.90)
        assert p.eta_charge == pytest.approx(0.95)
        assert p.eta_discharge == pytest.approx(0.90)

    def test_partial_explicit_eta(self):
        """Only η_c given → η_d falls back to sqrt(eta_roundtrip)."""
        p = make_battery_params(eta_roundtrip=0.80, eta_charge=0.95)
        assert p.eta_charge == 0.95
        import math
        assert p.eta_discharge == pytest.approx(math.sqrt(0.80))

    def test_default_one_way_095(self):
        p = make_battery_params()
        assert p.eta_charge == pytest.approx(0.95)
        assert p.eta_discharge == pytest.approx(0.95)

    def test_cycle_cost_default_zero(self):
        p = make_battery_params()
        assert p.cycle_cost_per_mwh == pytest.approx(0.0)

    def test_cycle_cost_passthrough(self):
        p = make_battery_params(cycle_cost_per_mwh=10.0)
        assert p.cycle_cost_per_mwh == pytest.approx(10.0)


# ========================== L1 Cycle energy throughput cost ==========================

class TestCycleCostInfo:
    """`costs[0]` = |feasible_P| * delta_t_hours * cycle_cost_per_mwh."""

    def test_default_zero_cycle_cost(self, env, key):
        params = make_battery_params(cycle_cost_per_mwh=0.0, power_mw=20.0)
        _, state = env.reset(key, params)
        _, state2, _, costs, _, info = env.step(key, state, jnp.float32(1.0), params)
        assert float(costs[0]) == pytest.approx(0.0)
        assert float(info["cost_sum"]) == pytest.approx(0.0)
        assert float(state2.current_p_mw) > 0

    def test_cycle_cost_matches_formula(self, env, key):
        rate = 10.0  # $/MWh
        dt = 0.5
        params = make_battery_params(
            cycle_cost_per_mwh=rate,
            delta_t_hours=dt,
            power_mw=20.0,
            initial_soc=0.5,
        )
        _, state = env.reset(key, params)
        _, state2, _, costs, _, info = env.step(key, state, jnp.float32(1.0), params)
        p = float(state2.current_p_mw)
        expected = abs(p) * dt * rate
        assert float(costs[0]) == pytest.approx(expected, rel=1e-5)
        assert float(info["cost_cycle_throughput"]) == pytest.approx(expected, rel=1e-5)

    def test_step_with_cycle_cost_info_keys(self, env, key):
        """BatteryEnv.step is already @jit; ensure info dict still exposes cost."""
        params = make_battery_params(cycle_cost_per_mwh=5.0, max_steps=8)
        _, state = env.reset(key, params)
        _, _, _, costs, _, info = env.step(key, state, jnp.float32(-0.25), params)
        assert "cost_sum" in info
        assert "cost_cycle_throughput" in info
        assert "cost_action_clip" in info
        assert float(costs[0]) >= 0.0


# ========================== L1 Sign Convention ==========================

class TestSignConvention:

    def test_discharge_positive(self, env, default_params, key):
        _, state = env.reset(key, default_params)
        _, state, *_ = env.step(key, state, jnp.float32(0.5), default_params)
        assert float(state.current_p_mw) > 0

    def test_charge_negative(self, env, default_params, key):
        _, state = env.reset(key, default_params)
        _, state, *_ = env.step(key, state, jnp.float32(-0.5), default_params)
        assert float(state.current_p_mw) < 0

    def test_idle_zero(self, env, default_params, key):
        _, state = env.reset(key, default_params)
        _, state, *_ = env.step(key, state, jnp.float32(0.0), default_params)
        assert float(state.current_p_mw) == pytest.approx(0.0)


# ========================== L1 SOC Dynamics ==========================

class TestSOCDynamics:

    def test_discharge_reduces_soc(self, env, default_params, key):
        _, state = env.reset(key, default_params)
        soc_before = float(state.soc)
        _, state, *_ = env.step(key, state, jnp.float32(0.5), default_params)
        assert float(state.soc) < soc_before

    def test_charge_increases_soc(self, env, default_params, key):
        _, state = env.reset(key, default_params)
        soc_before = float(state.soc)
        _, state, *_ = env.step(key, state, jnp.float32(-0.5), default_params)
        assert float(state.soc) > soc_before

    def test_discharge_energy_conservation(self, env, default_params, key):
        """Discharge: battery loses P * dt / (η_d * cap) of SOC."""
        _, state = env.reset(key, default_params)
        soc_before = float(state.soc)
        action_norm = 0.5  # maps to 10 MW discharge
        _, state, *_ = env.step(key, state, jnp.float32(action_norm), default_params)
        p_mw = 10.0
        dt_h = default_params.delta_t_hours
        expected = p_mw * dt_h / (default_params.eta_discharge * default_params.capacity_mwh)
        actual = soc_before - float(state.soc)
        assert actual == pytest.approx(expected, rel=1e-5)

    def test_charge_energy_conservation(self, env, default_params, key):
        """Charge: battery gains |P| * dt * η_c / cap of SOC."""
        _, state = env.reset(key, default_params)
        soc_before = float(state.soc)
        action_norm = -0.5  # maps to -10 MW (charge)
        _, state, *_ = env.step(key, state, jnp.float32(action_norm), default_params)
        p_mw = 10.0
        dt_h = default_params.delta_t_hours
        expected = p_mw * dt_h * default_params.eta_charge / default_params.capacity_mwh
        actual = float(state.soc) - soc_before
        assert actual == pytest.approx(expected, rel=1e-5)

    def test_asymmetric_efficiency(self, env, key):
        """η_c ≠ η_d: same |P| yields different SOC changes for charge vs discharge."""
        params = make_battery_params(
            capacity_mwh=100.0, power_mw=50.0,
            eta_charge=0.95, eta_discharge=0.90,
            initial_soc=0.5,
        )
        _, state = env.reset(key, params)
        # Charge: -0.2 maps to -10 MW (10/50=0.2)
        soc0 = float(state.soc)
        _, state_c, *_ = env.step(key, state, jnp.float32(-0.2), params)
        delta_charge = float(state_c.soc) - soc0

        # Discharge: 0.2 maps to 10 MW
        _, state_d, *_ = env.step(key, state, jnp.float32(0.2), params)
        delta_discharge = soc0 - float(state_d.soc)

        # charge_delta < discharge_delta (charging stores less per MW, discharging drains more per MW)
        assert delta_charge < delta_discharge


# ========================== L1 Power Clamping ==========================

class TestPowerClamp:

    def test_clip_above_max(self, env, default_params, key):
        _, state = env.reset(key, default_params)
        _, state, *_ = env.step(key, state, jnp.float32(5.0), default_params)
        assert float(state.current_p_mw) <= default_params.power_mw + 1e-9

    def test_clip_below_min(self, env, default_params, key):
        _, state = env.reset(key, default_params)
        _, state, *_ = env.step(key, state, jnp.float32(-5.0), default_params)
        assert float(state.current_p_mw) >= -default_params.power_mw - 1e-9


# ========================== L1 SOC Constraints ==========================

class TestSOCConstraints:

    def test_cannot_discharge_below_soc_min(self, env, key):
        """Near soc_min, discharge should be limited."""
        params = make_battery_params(
            capacity_mwh=10.0, power_mw=100.0,
            soc_min=0.2, soc_max=0.9, initial_soc=0.21,
        )
        _, state = env.reset(key, params)
        _, state, *_ = env.step(key, state, jnp.float32(1.0), params)
        assert float(state.soc) >= params.soc_min - 1e-6

    def test_cannot_charge_above_soc_max(self, env, key):
        """Near soc_max, charge should be limited."""
        params = make_battery_params(
            capacity_mwh=10.0, power_mw=100.0,
            soc_min=0.1, soc_max=0.8, initial_soc=0.79,
        )
        _, state = env.reset(key, params)
        _, state, *_ = env.step(key, state, jnp.float32(-1.0), params)
        assert float(state.soc) <= params.soc_max + 1e-6

    def test_feasible_power_at_soc_min(self):
        """At soc_min, discharge should be ~0."""
        params = make_battery_params(
            capacity_mwh=50.0, power_mw=20.0,
            soc_min=0.1, soc_max=0.9, initial_soc=0.1,
        )
        fp = compute_feasible_power(jnp.float32(0.1), jnp.float32(20.0), params)
        assert float(fp) == pytest.approx(0.0, abs=1e-6)

    def test_feasible_power_at_soc_max(self):
        """At soc_max, charge should be ~0."""
        params = make_battery_params(
            capacity_mwh=50.0, power_mw=20.0,
            soc_min=0.1, soc_max=0.9, initial_soc=0.9,
        )
        fp = compute_feasible_power(jnp.float32(0.9), jnp.float32(-20.0), params)
        assert float(fp) == pytest.approx(0.0, abs=1e-6)

    def test_soc_bounded_random_200_steps(self, env, key):
        """200 random steps — SOC in [soc_min, soc_max]."""
        params = make_battery_params(
            capacity_mwh=50.0, power_mw=20.0,
            soc_min=0.1, soc_max=0.9, initial_soc=0.5,
        )
        _, state = env.reset(key, params)
        subkeys = jax.random.split(key, 200)
        for i in range(200):
            action = jax.random.uniform(subkeys[i], minval=-1.0, maxval=1.0)
            _, state, *_ = env.step(subkeys[i], state, action, params)
            soc = float(state.soc)
            assert params.soc_min - 1e-6 <= soc <= params.soc_max + 1e-6, \
                f"SOC {soc} out of bounds at step {i}"


# ========================== L1 Observation ==========================

class TestObservation:

    def test_obs_shape_and_dtype(self, env, default_params, key):
        obs, _ = env.reset(key, default_params)
        assert obs.shape == (6,)
        assert obs.dtype == jnp.float32

    def test_obs_soc_in_range(self, env, default_params, key):
        obs, _ = env.reset(key, default_params)
        soc = float(obs[0])
        assert 0.0 <= soc <= 1.0

    def test_obs_after_discharge(self, env, default_params, key):
        _, state = env.reset(key, default_params)
        obs, *_ = env.step(key, state, jnp.float32(0.5), default_params)
        p_norm = float(obs[1])
        assert p_norm > 0

    def test_obs_after_charge(self, env, default_params, key):
        _, state = env.reset(key, default_params)
        obs, *_ = env.step(key, state, jnp.float32(-0.5), default_params)
        p_norm = float(obs[1])
        assert p_norm < 0


# ========================== L1 Reset ==========================

class TestReset:

    def test_reset_initial_soc(self, env, default_params, key):
        obs, state = env.reset(key, default_params)
        assert float(state.soc) == pytest.approx(default_params.initial_soc)
        assert float(state.current_p_mw) == pytest.approx(0.0)
        assert int(state.time_step) == 0

    def test_reset_after_steps(self, env, default_params, key):
        _, state = env.reset(key, default_params)
        _, state, *_ = env.step(key, state, jnp.float32(0.5), default_params)
        obs, state2 = env.reset(key, default_params)
        assert float(state2.soc) == pytest.approx(default_params.initial_soc)
        assert int(state2.time_step) == 0


# ========================== L1 Spaces ==========================

class TestSpaces:

    def test_action_space(self, env, default_params):
        space = env.action_space(default_params)
        assert space.shape == (1,)
        assert float(space.low[0]) == pytest.approx(-1.0)
        assert float(space.high[0]) == pytest.approx(1.0)

    def test_observation_space(self, env, default_params):
        space = env.observation_space(default_params)
        assert space.shape == (6,)


# ========================== L1 Done Flag ==========================

class TestDone:

    def test_done_at_max_steps(self, env, key):
        params = make_battery_params(max_steps=3)
        _, state = env.reset(key, params)
        for i in range(3):
            _, state, _, _, done, _ = env.step(key, state, jnp.float32(0.0), params)
        assert bool(done)

    def test_not_done_before_max(self, env, key):
        params = make_battery_params(max_steps=10)
        _, state = env.reset(key, params)
        _, state, _, _, done, _ = env.step(key, state, jnp.float32(0.0), params)
        assert not bool(done)
