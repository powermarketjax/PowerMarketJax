# Vendored test from PowerZooJax.
# Source        : tests/resource/test_flexload.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
"""L0 + L1 tests for FlexLoadEnv — Demand Response resource.

L0: JAX contracts — JIT, vmap, pytree
L1: Curtailment, shifting, buffer, energy conservation, cost signals
"""

import pytest
import jax
import jax.numpy as jnp

from powermarketjax.resources.flexload import (
    FlexLoadEnv, FlexLoadState, FlexLoadParams,
    _buffer_push, _buffer_pop, _add_to_buffer,
)


# ========================== Fixtures ==========================

@pytest.fixture
def env():
    return FlexLoadEnv()


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def default_params():
    """10 MW curtail, 10 MW shift, 4-step horizon."""
    return FlexLoadParams(
        curtail_cap_mw=10.0,
        shift_cap_mw=10.0,
        shift_horizon=4,
        max_steps=48,
    )


def _act(c=0.0, s=0.0):
    """Helper: create 2-D action array [curtail_frac, shift_frac]."""
    return jnp.array([c, s], dtype=jnp.float32)


# ========================== L0 JAX Contracts ==========================

class TestFlexLoadJaxContracts:

    def test_reset_jit(self, env, default_params, key):
        obs, state = env.reset(key, default_params)
        assert obs.shape == (8,)
        assert state.curtailed_mw.shape == ()

    def test_step_jit(self, env, default_params, key):
        _, state = env.reset(key, default_params)
        obs, s2, r, costs, d, info = env.step(key, state, _act(0.5, 0.0), default_params)
        assert obs.shape == (8,)
        assert costs.shape == (3,)

    def test_pytree_stable(self, env, default_params, key):
        _, s0 = env.reset(key, default_params)
        _, s1, *_ = env.step(key, s0, _act(), default_params)
        assert jax.tree_util.tree_structure(s0) == jax.tree_util.tree_structure(s1)

    def test_vmap(self, env, default_params, key):
        _, state = env.reset(key, default_params)
        b = 4
        states = jax.tree_util.tree_map(lambda x: jnp.stack([x] * b), state)
        keys = jax.random.split(key, b)
        actions = jnp.array([
            [0.5, 0.0],  # curtail only
            [0.0, 0.5],  # shift only
            [0.0, 0.0],  # idle
            [0.8, 0.2],  # both
        ], dtype=jnp.float32)

        def f(k, s, a):
            return env.step(k, s, a, default_params)

        obs_b, _, _, costs_b, _, _ = jax.vmap(f)(keys, states, actions)
        assert obs_b.shape == (b, 8)
        assert costs_b.shape == (b, 3)


# ========================== L1 Curtailment ==========================

class TestCurtailment:

    def test_curtail_positive_injection(self, env, default_params, key):
        """Curtailment → positive current_p_mw(load reduction = injection)."""
        _, state = env.reset(key, default_params)
        _, state, *_ = env.step(key, state, _act(c=0.5), default_params)
        assert float(state.current_p_mw) > 0
        assert float(state.curtailed_mw) == pytest.approx(5.0)

    def test_curtail_clipped(self, env, default_params, key):
        _, state = env.reset(key, default_params)
        _, state, *_ = env.step(key, state, _act(c=5.0), default_params)
        assert float(state.curtailed_mw) <= default_params.curtail_cap_mw + 1e-9

    def test_curtail_no_buffer(self, env, default_params, key):
        """Curtailment does not add to buffer."""
        _, state = env.reset(key, default_params)
        _, state, *_ = env.step(key, state, _act(c=0.5), default_params)
        assert int(state.buffer_size) == 0

    def test_curtail_current_p_mw(self, env, default_params, key):
        """current_p_mw= curtail + released (no prior buffer → released=0)."""
        _, state = env.reset(key, default_params)
        _, state, *_ = env.step(key, state, _act(c=0.8), default_params)
        assert float(state.current_p_mw) == pytest.approx(8.0, abs=1e-5)


# ========================== L1 Demand Shifting ==========================

class TestDemandShifting:

    def test_shift_adds_to_buffer(self, env, default_params, key):
        _, state = env.reset(key, default_params)
        _, state, *_ = env.step(key, state, _act(s=0.6), default_params)
        assert int(state.buffer_size) == default_params.shift_horizon
        assert float(state.shift_out_mw) == pytest.approx(6.0)

    def test_shift_distributes_evenly(self, env, default_params, key):
        """Shifted demand spread evenly over shift_horizon steps in buffer."""
        _, state = env.reset(key, default_params)
        _, state, *_ = env.step(key, state, _act(s=0.8), default_params)
        per_step = 8.0 / default_params.shift_horizon  # 2.0
        for i in range(default_params.shift_horizon):
            idx = (int(state.buffer_head) + i) % 64
            assert float(state.deferred_buffer[idx]) == pytest.approx(per_step, abs=1e-5)

    def test_shift_clipped(self, env, default_params, key):
        _, state = env.reset(key, default_params)
        _, state, *_ = env.step(key, state, _act(s=5.0), default_params)
        assert float(state.shift_out_mw) <= default_params.shift_cap_mw + 1e-9

    def test_shift_current_p_mw(self, env, default_params, key):
        """First shift, no prior buffer → current_p_mw = shift_out - released = 6 - 0 = 6."""
        _, state = env.reset(key, default_params)
        _, state, *_ = env.step(key, state, _act(s=0.6), default_params)
        # shift_out is positive injection (load deferred), then released later as negative
        assert float(state.current_p_mw) == pytest.approx(6.0, abs=1e-4)

    def test_deferred_released_in_subsequent_steps(self, env, default_params, key):
        """Deferred energy trickles back over shift_horizon steps."""
        _, state = env.reset(key, default_params)
        _, state, *_ = env.step(key, state, _act(s=0.8), default_params)  # shift=8
        released_total = 0.0
        for _ in range(default_params.shift_horizon):
            _, state, *_ = env.step(key, state, _act(), default_params)
            # On idle: current_p = 0 + 0 - released = -released (released is negative injection)
            released_total += float(-state.current_p_mw)  # released shows as negative p
        assert released_total == pytest.approx(8.0, abs=0.2)


# ========================== L1 Energy Conservation ==========================

class TestFlexEnergyConservation:

    def test_shift_then_release_net_zero(self, env, default_params, key):
        """Total current_p_mw sums to ~0 over a full shift-release cycle."""
        _, state = env.reset(key, default_params)
        _, state, *_ = env.step(key, state, _act(s=1.0), default_params)
        total_p = float(state.current_p_mw)
        for _ in range(default_params.shift_horizon + 2):
            _, state, *_ = env.step(key, state, _act(), default_params)
            total_p += float(state.current_p_mw)
        assert total_p == pytest.approx(0.0, abs=0.2)

    def test_curtailment_net_positive(self, env, default_params, key):
        """Curtailment produces net positive injection over time."""
        _, state = env.reset(key, default_params)
        _, state, *_ = env.step(key, state, _act(c=0.5), default_params)
        total = float(state.current_p_mw)
        for _ in range(5):
            _, state, *_ = env.step(key, state, _act(), default_params)
            total += float(state.current_p_mw)
        assert total > 0


# ========================== L1 Simultaneous Action ==========================

class TestSimultaneousAction:

    def test_both_curtail_and_shift(self, env, default_params, key):
        """Both actions can be non-zero simultaneously (soft complementarity)."""
        _, state = env.reset(key, default_params)
        _, state, _, costs, _, info = env.step(key, state, _act(c=0.5, s=0.3), default_params)
        assert float(state.curtailed_mw) == pytest.approx(5.0)
        assert float(state.shift_out_mw) == pytest.approx(3.0)
        # Cost signal for simultaneous activation
        assert float(info["cost_simultaneous"]) > 0
        assert float(costs[2]) > 0.0


# ========================== L1 Idle ==========================

class TestIdle:

    def test_idle_zero_curtail_shift(self, env, default_params, key):
        _, state = env.reset(key, default_params)
        _, state, *_ = env.step(key, state, _act(), default_params)
        assert float(state.curtailed_mw) == pytest.approx(0.0)
        assert float(state.shift_out_mw) == pytest.approx(0.0)

    def test_idle_releases_deferred(self, env, default_params, key):
        """Idle after shift → current_p_mw = -released (negative, load returns)."""
        _, state = env.reset(key, default_params)
        _, state, *_ = env.step(key, state, _act(s=0.8), default_params)
        _, state, *_ = env.step(key, state, _act(), default_params)
        # Released demand appears as negative current_p
        assert float(state.current_p_mw) < 0


# ========================== L1 Observation ==========================

class TestFlexObs:

    def test_obs_shape_dtype(self, env, default_params, key):
        obs, _ = env.reset(key, default_params)
        assert obs.shape == (8,)
        assert obs.dtype == jnp.float32

    def test_obs_after_curtail(self, env, default_params, key):
        _, state = env.reset(key, default_params)
        obs, *_ = env.step(key, state, _act(c=0.5), default_params)
        assert float(obs[0]) > 0  # curtail_norm > 0
        assert float(obs[1]) == pytest.approx(0.0)  # shift_out_norm = 0

    def test_obs_after_shift(self, env, default_params, key):
        _, state = env.reset(key, default_params)
        obs, *_ = env.step(key, state, _act(s=0.5), default_params)
        assert float(obs[0]) == pytest.approx(0.0)  # curtail_norm = 0
        assert float(obs[1]) > 0  # shift_out_norm > 0

    def test_obs_shift_in_after_release(self, env, default_params, key):
        """After shift + idle step: shift_in_norm > 0."""
        _, state = env.reset(key, default_params)
        _, state, *_ = env.step(key, state, _act(s=0.8), default_params)
        obs, *_ = env.step(key, state, _act(), default_params)
        assert float(obs[2]) > 0  # shift_in_norm

    def test_obs_buffer_fill_and_energy(self, env, default_params, key):
        """After shift: buffer_fill_ratio and buffer_energy_norm > 0."""
        _, state = env.reset(key, default_params)
        obs, state, *_ = env.step(key, state, _act(s=0.8), default_params)
        assert float(obs[3]) > 0  # buffer_fill_ratio
        assert float(obs[4]) > 0  # buffer_energy_norm

    def test_obs_bounded(self, env, default_params, key):
        """All obs values in their expected ranges.
        price_norm (obs[-1]) is in [-1, 2]; time features in [-1, 1]; others in [0, 1].
        """
        _, state = env.reset(key, default_params)
        obs, *_ = env.step(key, state, _act(s=1.0), default_params)
        for v in obs:
            assert -1.0 - 1e-6 <= float(v) <= 2.0 + 1e-6


# ========================== L1 Cost Signals ==========================

class TestCostSignals:

    def test_cost_curtailment(self, env, default_params, key):
        _, state = env.reset(key, default_params)
        _, _, _, costs, _, info = env.step(key, state, _act(c=0.5), default_params)
        assert float(info["cost_curtailment"]) > 0
        assert float(costs[0]) > 0.0

    def test_cost_shift_discomfort(self, env, default_params, key):
        _, state = env.reset(key, default_params)
        _, state, *_ = env.step(key, state, _act(s=0.5), default_params)
        # After buffering, next step has holding cost
        _, _, _, costs, _, info = env.step(key, state, _act(), default_params)
        assert float(info["cost_shift_discomfort"]) > 0
        assert float(costs[1]) > 0.0

    def test_cost_simultaneous(self, env, default_params, key):
        _, state = env.reset(key, default_params)
        _, _, _, costs, _, info = env.step(key, state, _act(c=0.3, s=0.3), default_params)
        assert float(info["cost_simultaneous"]) > 0
        assert float(costs[2]) > 0.0

    def test_no_cost_on_idle(self, env, default_params, key):
        _, state = env.reset(key, default_params)
        _, _, _, costs, _, info = env.step(key, state, _act(), default_params)
        assert float(info["cost_curtailment"]) == pytest.approx(0.0)
        assert float(info["cost_simultaneous"]) == pytest.approx(0.0)
        assert float(info["cost_sum"]) == pytest.approx(float(jnp.sum(costs)))


# ========================== L1 Buffer Operations ==========================

class TestBufferOperations:

    def test_push_and_pop(self):
        buf = jnp.zeros(8, dtype=jnp.float32)
        head = jnp.int32(0)
        size = jnp.int32(0)

        buf, size = _buffer_push(buf, head, size, jnp.float32(3.0), 8)
        assert int(size) == 1
        assert float(buf[0]) == 3.0

        val, buf, head, size = _buffer_pop(buf, head, size, 8)
        assert float(val) == 3.0
        assert int(size) == 0

    def test_pop_empty_returns_zero(self):
        buf = jnp.zeros(8, dtype=jnp.float32)
        val, _, _, _ = _buffer_pop(buf, jnp.int32(0), jnp.int32(0), 8)
        assert float(val) == 0.0

    def test_add_to_buffer_spread(self):
        buf = jnp.zeros(8, dtype=jnp.float32)
        head = jnp.int32(0)
        size = jnp.int32(0)
        buf, size = _add_to_buffer(buf, head, size, jnp.float32(12.0), 4, 8)
        assert int(size) == 4
        for i in range(4):
            assert float(buf[i]) == pytest.approx(3.0)

    def test_consecutive_shifts_superimpose(self):
        """Two consecutive _add_to_buffer calls must superimpose on overlapping
        time slots, NOT queue after each other (regression for tail-append bug).
        """
        buf = jnp.zeros(8, dtype=jnp.float32)
        head = jnp.int32(0)
        size = jnp.int32(0)
        horizon = 4

        # First shift: 8 MW / 4 slots = 2 MW per slot → positions 0,1,2,3
        buf, size = _add_to_buffer(buf, head, size, jnp.float32(8.0), horizon, 8)
        assert int(size) == 4

        # Simulate one pop (head advances)
        val, buf, head, size = _buffer_pop(buf, head, size, 8)
        assert float(val) == pytest.approx(2.0)
        # head=1, size=3, buffer=[0,2,2,2,0,...]

        # Second shift: 8 MW / 4 = 2 MW → should ADD to positions 1,2,3,4
        buf, size = _add_to_buffer(buf, head, size, jnp.float32(8.0), horizon, 8)
        assert int(size) == 4  # max(3, 4)

        # Verify overlap: positions 1,2,3 should be 2+2=4; position 4 = 2
        assert float(buf[1]) == pytest.approx(4.0)
        assert float(buf[2]) == pytest.approx(4.0)
        assert float(buf[3]) == pytest.approx(4.0)
        assert float(buf[4]) == pytest.approx(2.0)


# ========================== L1 Price Signal ==========================

class TestPriceSignal:

    def test_price_norm_reflects_lmp(self, env, default_params, key):
        """price_norm (obs[-1]) must change when lmp changes between steps."""
        _, state = env.reset(key, default_params)
        obs_low, *_ = env.step(key, state, _act(), default_params, lmp=20.0)
        obs_high, *_ = env.step(key, state, _act(), default_params, lmp=80.0)
        assert float(obs_low[-1]) < float(obs_high[-1])

    def test_price_norm_zero_when_lmp_zero(self, env, default_params, key):
        """Default lmp=0 → price_norm=0."""
        _, state = env.reset(key, default_params)
        obs, *_ = env.step(key, state, _act(), default_params)
        assert float(obs[-1]) == pytest.approx(0.0)

    def test_lmp_stored_in_state(self, env, default_params, key):
        """current_lmp in state reflects the lmp passed to step."""
        _, state = env.reset(key, default_params)
        _, state, _, _, _, info = env.step(key, state, _act(), default_params, lmp=55.0)
        assert float(state.current_lmp) == pytest.approx(55.0)
        assert float(info["current_lmp"]) == pytest.approx(55.0)

    def test_lmp_clipped_in_obs(self, env, default_params, key):
        """price_norm is clipped to [-1, 2] even for extreme LMP values."""
        _, state = env.reset(key, default_params)
        obs_neg, *_ = env.step(key, state, _act(), default_params, lmp=-9999.0)
        obs_huge, *_ = env.step(key, state, _act(), default_params, lmp=9999.0)
        assert float(obs_neg[-1]) == pytest.approx(-1.0)
        assert float(obs_huge[-1]) == pytest.approx(2.0)

    def test_negative_lmp_visible_in_obs(self, env, default_params, key):
        """Negative LMP (e.g. wind curtailment) must produce negative price_norm."""
        _, state = env.reset(key, default_params)
        obs, *_ = env.step(key, state, _act(), default_params, lmp=-30.0)
        assert float(obs[-1]) < 0.0


# ========================== L1 Info Fields ==========================

class TestInfoFields:

    def test_info_keys(self, env, default_params, key):
        """info dict must contain exactly the documented keys."""
        _, state = env.reset(key, default_params)
        _, _, _, costs, _, info = env.step(key, state, _act(c=0.3, s=0.2), default_params)
        assert set(info.keys()) == {
            "cost_sum", "cost_curtailment", "cost_shift_discomfort",
            "cost_simultaneous", "current_lmp",
        }
        assert costs.shape == (3,)

    def test_no_buffer_overflow_count(self, env, default_params, key):
        """buffer_overflow_count was removed — must not appear in info."""
        _, state = env.reset(key, default_params)
        _, _, _, _, _, info = env.step(key, state, _act(), default_params)
        assert "buffer_overflow_count" not in info


# ========================== L1 Reset ==========================

class TestFlexReset:

    def test_reset_clears_state(self, env, default_params, key):
        _, state = env.reset(key, default_params)
        _, state, *_ = env.step(key, state, _act(s=0.8), default_params)
        _, state2 = env.reset(key, default_params)
        assert float(state2.curtailed_mw) == 0.0
        assert float(state2.shift_out_mw) == 0.0
        assert int(state2.buffer_size) == 0
        assert int(state2.time_step) == 0
