# Vendored test from PowerZooJax.
# Source        : tests/data/test_nonstationary.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
"""Tests for NonstationarySampler — D3 non-stationary episode sampling."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.data.nonstationary import (
    EpisodeConfig,
    NonstationarySampler,
    apply_drift,
)


@pytest.fixture(scope="module")
def sampler():
    return NonstationarySampler(
        total_steps=10080,      # ~7 months × 48 steps/day
        total_episodes=500,
        max_steps=48,
        drift_rate=0.0003,
        rolling_sigma_steps=720.0,
    )


@pytest.fixture(scope="module")
def key():
    return jax.random.PRNGKey(42)


# ==================== Sampler basics ====================

class TestNonstationarySampler:

    def test_sample_returns_episode_config(self, sampler, key):
        cfg = sampler.sample(0, key)
        assert isinstance(cfg, EpisodeConfig)
        assert isinstance(cfg.episode_start, int)
        assert isinstance(cfg.drift_factor, float)

    def test_episode_start_in_bounds(self, sampler, key):
        """episode_start should be within [0, total_steps - max_steps]."""
        for idx in [0, 100, 250, 499]:
            k = jax.random.PRNGKey(idx)
            cfg = sampler.sample(idx, k)
            assert 0 <= cfg.episode_start <= sampler.max_start

    def test_drift_factor_increases(self, sampler, key):
        """Drift factor should increase with episode_idx."""
        cfg0 = sampler.sample(0, key)
        cfg100 = sampler.sample(100, key)
        cfg499 = sampler.sample(499, key)
        assert cfg0.drift_factor < cfg100.drift_factor < cfg499.drift_factor

    def test_drift_factor_formula(self, sampler, key):
        """drift_factor = 1.0 + 0.0003 * episode_idx."""
        for idx in [0, 100, 250, 499]:
            cfg = sampler.sample(idx, key)
            expected = 1.0 + 0.0003 * idx
            np.testing.assert_allclose(cfg.drift_factor, expected, atol=1e-8)

    def test_different_keys_different_starts(self, sampler):
        """Different PRNG keys should produce different episode_start values."""
        starts = set()
        for i in range(20):
            k = jax.random.PRNGKey(i)
            cfg = sampler.sample(250, k)
            starts.add(cfg.episode_start)
        # With 20 different keys, we should see more than 1 unique start
        assert len(starts) > 1

    def test_rolling_center_moves(self, sampler, key):
        """Episode starts should tend to be later for higher episode_idx."""
        early_starts = []
        late_starts = []
        for i in range(50):
            k = jax.random.PRNGKey(i)
            early_starts.append(sampler.sample(10, k).episode_start)
            late_starts.append(sampler.sample(490, k).episode_start)
        # On average, late episodes should start later in the window
        assert np.mean(late_starts) > np.mean(early_starts)


# ==================== Modes ====================

class TestSamplerModes:

    def test_iid_mode_no_drift(self, sampler, key):
        cfg = sampler.sample(100, key, mode="iid")
        assert cfg.drift_factor == 1.0

    def test_iid_mode_valid_start(self, sampler, key):
        cfg = sampler.sample(100, key, mode="iid")
        assert 0 <= cfg.episode_start <= sampler.max_start

    def test_drift_shock_mode(self, sampler, key):
        cfg = sampler.sample(100, key, mode="drift_shock")
        assert cfg.drift_factor == 1.25

    def test_fixed_mode(self, sampler, key):
        cfg = sampler.sample(100, key, mode="fixed")
        assert cfg.episode_start == 0
        assert cfg.drift_factor == 1.0

    def test_invalid_mode_raises(self, sampler, key):
        with pytest.raises(ValueError, match="Unknown mode"):
            sampler.sample(0, key, mode="invalid")


# ==================== Batch ====================

class TestSamplerBatch:

    def test_sample_batch_size(self, sampler, key):
        batch = sampler.sample_batch(0, 10, key)
        assert len(batch) == 10

    def test_sample_batch_increasing_drift(self, sampler, key):
        batch = sampler.sample_batch(0, 10, key)
        drifts = [cfg.drift_factor for cfg in batch]
        # Drift should strictly increase with episode index
        for i in range(1, len(drifts)):
            assert drifts[i] > drifts[i - 1]


# ==================== apply_drift ====================

class TestApplyDrift:

    def test_drift_factor_1_noop(self):
        p = jnp.ones((48, 33), dtype=jnp.float32)
        q = jnp.ones((48, 33), dtype=jnp.float32) * 0.5
        p_d, q_d = apply_drift(p, q, 1.0)
        np.testing.assert_allclose(p_d, p)
        np.testing.assert_allclose(q_d, q)

    def test_drift_scales_proportionally(self):
        p = jnp.ones((48, 33), dtype=jnp.float32) * 2.0
        q = jnp.ones((48, 33), dtype=jnp.float32) * 1.0
        p_d, q_d = apply_drift(p, q, 1.1)
        np.testing.assert_allclose(p_d, 2.2, atol=1e-5)
        np.testing.assert_allclose(q_d, 1.1, atol=1e-5)

    def test_drift_preserves_shape(self):
        p = jnp.zeros((48, 33), dtype=jnp.float32)
        q = jnp.zeros((48, 33), dtype=jnp.float32)
        p_d, q_d = apply_drift(p, q, 1.5)
        assert p_d.shape == (48, 33)
        assert q_d.shape == (48, 33)
