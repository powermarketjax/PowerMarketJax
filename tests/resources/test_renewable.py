# Vendored test from PowerZooJax.
# Source        : tests/resource/test_renewable.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
"""L0 + L1 tests for RenewableEnv, SolarEnv, WindEnv.

L0: JAX contracts — JIT, vmap, pytree structure
L1: Profile lookup, curtailment, non-negative output
"""

import pytest
import jax
import jax.numpy as jnp

from powermarketjax.resources.renewable import (
    RenewableEnv, SolarEnv, WindEnv, RenewableState, RenewableParams,
)


# ========================== Fixtures ==========================

@pytest.fixture
def env():
    return RenewableEnv()


@pytest.fixture
def solar_env():
    return SolarEnv()


@pytest.fixture
def wind_env():
    return WindEnv()


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


@pytest.fixture
def default_params():
    """Flat 50% CF profile, 48 steps."""
    return RenewableEnv().default_params()


@pytest.fixture
def solar_params():
    return SolarEnv().default_params()


@pytest.fixture
def custom_params():
    """Custom explicit profile for precise testing."""
    profiles = jnp.array([0.0, 0.2, 0.5, 0.8, 1.0, 0.7, 0.3, 0.0], dtype=jnp.float32)
    return RenewableParams(
        capacity_mw=100.0,
        profiles=profiles,
        allow_curtailment=1,
        max_steps=8,
        steps_per_day=8,
        delta_t_hours=3.0,
    )


# ========================== L0 JAX Contracts ==========================

class TestRenewableJaxContracts:

    def test_reset_jit(self, env, default_params, key):
        obs, state = env.reset(key, default_params)
        assert obs.shape == (4,)
        assert state.capacity_factor.shape == ()

    def test_step_jit(self, env, default_params, key):
        _, state = env.reset(key, default_params)
        obs, state2, r, costs, d, info = env.step(
            key, state, jnp.float32(0.0), default_params
        )
        assert obs.shape == (4,)
        assert costs.shape == (2,)

    def test_pytree_structure_stable(self, env, default_params, key):
        _, s0 = env.reset(key, default_params)
        _, s1, *_ = env.step(key, s0, jnp.float32(0.0), default_params)
        assert jax.tree_util.tree_structure(s0) == jax.tree_util.tree_structure(s1)

    def test_vmap_step(self, env, default_params, key):
        _, state = env.reset(key, default_params)
        batch = 4
        states = jax.tree_util.tree_map(lambda x: jnp.stack([x] * batch), state)
        keys = jax.random.split(key, batch)
        actions = jnp.array([-1.0, -0.5, 0.0, 1.0])

        def f(k, s, a):
            return env.step(k, s, a, default_params)

        obs_b, _, _, costs_b, _, _ = jax.vmap(f)(keys, states, actions)
        assert obs_b.shape == (batch, 4)
        assert costs_b.shape == (batch, 2)


# ========================== L1 Profile Lookup ==========================

class TestProfileLookup:

    def test_first_step_reads_index_0(self, env, custom_params, key):
        _, state = env.reset(key, custom_params)
        # step 0: profile[0] = 0.0, action=+1 → no curtailment
        obs, state, *_ = env.step(key, state, jnp.float32(1.0), custom_params)
        assert float(state.capacity_factor) == pytest.approx(0.0)
        assert float(state.current_p_mw) == pytest.approx(0.0)

    def test_step_4_reads_peak(self, env, custom_params, key):
        _, state = env.reset(key, custom_params)
        # Advance 4 steps (reads indices 0,1,2,3), step 5 reads index 4 = 1.0
        for _ in range(4):
            _, state, *_ = env.step(key, state, jnp.float32(1.0), custom_params)
        _, state, *_ = env.step(key, state, jnp.float32(1.0), custom_params)
        assert float(state.capacity_factor) == pytest.approx(1.0)
        assert float(state.current_p_mw) == pytest.approx(100.0)

    def test_profile_wraps(self, env, custom_params, key):
        """After n_steps, profile wraps back to index 0."""
        _, state = env.reset(key, custom_params)
        # 8-step profile, run 9 times → step 9 reads index 8 % 8 = 0
        for _ in range(8):
            _, state, *_ = env.step(key, state, jnp.float32(1.0), custom_params)
        _, state, *_ = env.step(key, state, jnp.float32(1.0), custom_params)
        # Index 0 of custom profile = 0.0
        assert float(state.capacity_factor) == pytest.approx(0.0)


# ========================== L1 Curtailment ==========================

class TestCurtailment:

    def test_no_curtailment_full_output(self, env, custom_params, key):
        """action=+1 → curtailment=0 → full capacity × CF."""
        _, state = env.reset(key, custom_params)
        # Move to step that reads index 3 (CF=0.8)
        for _ in range(3):
            _, state, *_ = env.step(key, state, jnp.float32(1.0), custom_params)
        _, state, *_ = env.step(key, state, jnp.float32(1.0), custom_params)
        assert float(state.current_p_mw) == pytest.approx(100.0 * 0.8, abs=1e-4)

    def test_full_curtailment_zero(self, env, custom_params, key):
        """action=-1 → curtailment=1 → zero output."""
        _, state = env.reset(key, custom_params)
        for _ in range(3):
            _, state, *_ = env.step(key, state, jnp.float32(1.0), custom_params)
        _, state, *_ = env.step(key, state, jnp.float32(-1.0), custom_params)
        assert float(state.current_p_mw) == pytest.approx(0.0, abs=1e-6)

    def test_partial_curtailment(self, env, custom_params, key):
        """action=0 → curtailment=0.5 → half of max."""
        _, state = env.reset(key, custom_params)
        for _ in range(3):
            _, state, *_ = env.step(key, state, jnp.float32(1.0), custom_params)
        _, state, *_ = env.step(key, state, jnp.float32(0.0), custom_params)
        expected = 100.0 * 0.8 * (1.0 - 0.5)
        assert float(state.current_p_mw) == pytest.approx(expected, abs=1e-3)

    def test_curtailment_clamped(self, env, default_params, key):
        """action < -1 → curtailment clipped to 1 (full curtailment)."""
        _, state = env.reset(key, default_params)
        obs, state, *_ = env.step(key, state, jnp.float32(-5.0), default_params)
        assert float(state.current_p_mw) == pytest.approx(0.0, abs=1e-6)

    def test_negative_curtailment_clamped(self, env, default_params, key):
        """action > 1 → curtailment clipped to 0 (no curtailment)."""
        _, state = env.reset(key, default_params)
        obs, state, *_ = env.step(key, state, jnp.float32(5.0), default_params)
        # Full output (no curtailment)
        expected = default_params.capacity_mw * default_params.profiles[0]
        assert float(state.current_p_mw) == pytest.approx(float(expected), abs=1e-3)

    def test_allow_curtailment_off(self, env, key):
        """With allow_curtailment=0, curtailment action is ignored."""
        profiles = jnp.full((8,), 0.5, dtype=jnp.float32)
        params = RenewableParams(
            capacity_mw=100.0, profiles=profiles,
            allow_curtailment=0, max_steps=8, steps_per_day=8,
        )
        _, state = env.reset(key, params)
        _, state, *_ = env.step(key, state, jnp.float32(-1.0), params)  # full curtail action
        # Curtailment disabled → full output
        assert float(state.current_p_mw) == pytest.approx(50.0, abs=1e-3)


# ========================== L1 Non-negative Output ==========================

class TestNonNegativeOutput:

    def test_output_always_nonneg(self, env, default_params, key):
        """Renewable output ≥ 0 for all normalized action levels."""
        _, state = env.reset(key, default_params)
        for a in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            _, state_c, *_ = env.step(key, state, jnp.float32(a), default_params)
            assert float(state_c.current_p_mw) >= -1e-9


# ========================== L1 Observation ==========================

class TestRenewableObs:

    def test_obs_shape_dtype(self, env, default_params, key):
        obs, _ = env.reset(key, default_params)
        assert obs.shape == (4,)
        assert obs.dtype == jnp.float32

    def test_cf_in_range(self, env, default_params, key):
        obs, _ = env.reset(key, default_params)
        cf = float(obs[0])
        assert 0.0 <= cf <= 1.0

    def test_p_norm_in_range(self, env, default_params, key):
        _, state = env.reset(key, default_params)
        obs, *_ = env.step(key, state, jnp.float32(1.0), default_params)
        p_norm = float(obs[1])
        assert 0.0 <= p_norm <= 1.0


# ========================== L1 Subclass Defaults ==========================

class TestSubclassDefaults:

    def test_solar_profile_peak_midday(self, solar_env, solar_params, key):
        """Solar profile peaks around midday (~step 24 of 48)."""
        cf_noon = float(solar_params.profiles[24])
        cf_midnight = float(solar_params.profiles[0])
        assert cf_noon > cf_midnight
        assert cf_midnight == pytest.approx(0.0, abs=1e-5)
        assert cf_noon > 0.5

    def test_wind_profile_nonzero_night(self, wind_env, key):
        """Wind can produce at night."""
        wp = wind_env.default_params()
        cf_midnight = float(wp.profiles[0])
        assert cf_midnight > 0.0

    def test_solar_env_name(self, solar_env):
        assert solar_env.name == "SolarEnv"

    def test_wind_env_name(self, wind_env):
        assert wind_env.name == "WindEnv"


# ========================== L1 Time Step ==========================

class TestTimeStep:

    def test_time_step_advances(self, env, default_params, key):
        _, state = env.reset(key, default_params)
        assert int(state.time_step) == 0
        _, state, *_ = env.step(key, state, jnp.float32(0.0), default_params)
        assert int(state.time_step) == 1
        _, state, *_ = env.step(key, state, jnp.float32(0.0), default_params)
        assert int(state.time_step) == 2


# ========================== Reset: CF = profiles[0] ==========================

@pytest.fixture
def params_q():
    profiles = jnp.full((8,), 0.5, dtype=jnp.float32)
    return RenewableParams(
        capacity_mw=100.0,
        profiles=profiles,
        enable_q_control=True,
        s_rated_mva=120.0,
        max_steps=8,
        steps_per_day=8,
    )


class TestResetCfFix:

    def test_custom_profile_first_step_zero_reset_obs_cf(self, env, custom_params, key):
        obs, _ = env.reset(key, custom_params)
        assert float(obs[0]) == pytest.approx(0.0, abs=1e-6)

    def test_custom_profile_first_step_half_reset_obs_cf(self, env, key):
        profiles = jnp.full((8,), 0.5, dtype=jnp.float32)
        params = RenewableParams(
            capacity_mw=100.0,
            profiles=profiles,
            max_steps=8,
            steps_per_day=8,
        )
        obs, _ = env.reset(key, params)
        assert float(obs[0]) == pytest.approx(0.5, abs=1e-6)

    def test_solar_default_reset_obs_cf_matches_midnight_profile(
        self, solar_env, solar_params, key
    ):
        obs, _ = solar_env.reset(key, solar_params)
        assert float(obs[0]) == pytest.approx(float(solar_params.profiles[0]), abs=1e-5)

    def test_wind_default_reset_obs_cf_matches_midnight_profile(self, wind_env, key):
        wind_params = wind_env.default_params()
        obs, _ = wind_env.reset(key, wind_params)
        assert float(obs[0]) == pytest.approx(float(wind_params.profiles[0]), abs=1e-5)


# ========================== Q control (enable_q_control) ==========================

class TestQControl:

    def test_spaces_p_only(self, env, default_params):
        assert env.action_space(default_params).shape == (1,)
        assert env.observation_space(default_params).shape == (4,)

    def test_spaces_with_q(self, env, params_q):
        assert env.action_space(params_q).shape == (2,)
        assert env.observation_space(params_q).shape == (5,)

    def test_step_q_zero_when_disabled(self, env, default_params, key):
        _, state = env.reset(key, default_params)
        _, state2, *_ = env.step(key, state, jnp.float32(0.0), default_params)
        assert float(state2.current_q_mvar) == pytest.approx(0.0, abs=1e-6)

    def test_pq_circle_satisfied(self, env, params_q, key):
        _, state = env.reset(key, params_q)
        s = float(params_q.s_rated_mva)
        for a_p, a_q in [(1.0, 1.0), (1.0, -1.0), (-1.0, 0.5), (0.0, -0.3)]:
            _, state, *_ = env.step(
                key, state, jnp.array([a_p, a_q], dtype=jnp.float32), params_q
            )
            p = float(state.current_p_mw)
            q = float(state.current_q_mvar)
            assert p * p + q * q <= s * s + 1e-3

    def test_p_zero_night_q_max_equals_s_rated(self, env, key):
        profiles = jnp.zeros((8,), dtype=jnp.float32)
        params = RenewableParams(
            capacity_mw=100.0,
            profiles=profiles,
            enable_q_control=True,
            s_rated_mva=120.0,
            max_steps=8,
            steps_per_day=8,
        )
        _, state = env.reset(key, params)
        _, state2, *_ = env.step(
            key, state, jnp.array([1.0, 1.0], dtype=jnp.float32), params
        )
        assert float(state2.current_p_mw) == pytest.approx(0.0, abs=1e-5)
        assert float(state2.current_q_mvar) == pytest.approx(120.0, abs=1e-3)

    def test_full_output_s_equals_capacity_q_clipped_to_zero(self, env, key):
        profiles = jnp.ones((8,), dtype=jnp.float32)
        params = RenewableParams(
            capacity_mw=100.0,
            profiles=profiles,
            enable_q_control=True,
            s_rated_mva=100.0,
            max_steps=8,
            steps_per_day=8,
        )
        _, state = env.reset(key, params)
        _, state2, *_ = env.step(
            key, state, jnp.array([1.0, 1.0], dtype=jnp.float32), params
        )
        assert float(state2.current_p_mw) == pytest.approx(100.0, abs=1e-3)
        assert float(state2.current_q_mvar) == pytest.approx(0.0, abs=1e-4)

    def test_full_output_s_larger_q_at_q_max_clip(self, env, key):
        profiles = jnp.ones((8,), dtype=jnp.float32)
        s = 120.0
        params = RenewableParams(
            capacity_mw=100.0,
            profiles=profiles,
            enable_q_control=True,
            s_rated_mva=s,
            max_steps=8,
            steps_per_day=8,
        )
        _, state = env.reset(key, params)
        _, state2, *_ = env.step(
            key, state, jnp.array([1.0, 1.0], dtype=jnp.float32), params
        )
        q_exp = (s ** 2 - 100.0 ** 2) ** 0.5
        assert float(state2.current_p_mw) == pytest.approx(100.0, abs=1e-3)
        assert float(state2.current_q_mvar) == pytest.approx(q_exp, abs=1e-3)

    def test_obs_q_norm_in_unit_range(self, env, params_q, key):
        _, state = env.reset(key, params_q)
        for a_p, a_q in [(1.0, 0.0), (0.5, 0.8), (-0.2, -1.0)]:
            obs, state, *_ = env.step(
                key, state, jnp.array([a_p, a_q], dtype=jnp.float32), params_q
            )
            qn = float(obs[2])
            assert -1.0 - 1e-5 <= qn <= 1.0 + 1e-5

    def test_pytree_structure_stable_q_mode(self, env, params_q, key):
        _, s0 = env.reset(key, params_q)
        _, s1, *_ = env.step(
            key, s0, jnp.array([0.0, 0.5], dtype=jnp.float32), params_q
        )
        assert jax.tree_util.tree_structure(s0) == jax.tree_util.tree_structure(s1)

    def test_vmap_step_q_mode(self, env, params_q, key):
        _, state = env.reset(key, params_q)
        batch = 4
        states = jax.tree_util.tree_map(lambda x: jnp.stack([x] * batch), state)
        keys = jax.random.split(key, batch)
        actions = jnp.array(
            [[1.0, 1.0], [-1.0, 0.0], [0.0, -0.5], [0.5, 0.25]], dtype=jnp.float32
        )

        def f(k, s, a):
            return env.step(k, s, a, params_q)

        obs_b, _, _, costs_b, _, _ = jax.vmap(f)(keys, states, actions)
        assert obs_b.shape == (batch, 5)
        assert costs_b.shape == (batch, 2)
