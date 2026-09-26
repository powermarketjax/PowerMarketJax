# Vendored test from PowerZooJax.
# Source        : tests/test_spaces.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
"""Tests for powermarketjax/spaces.py — L0 JAX contract + basic correctness."""

import jax
import jax.numpy as jnp
import pytest

from powermarketjax.spaces import (
    Box, Discrete, MultiDiscrete, MultiBinary,
    make_box, make_discrete, make_multi_discrete, make_multi_binary,
)


# ==================== Box ====================

class TestBox:
    def test_create(self):
        space = Box(low=jnp.zeros(3), high=jnp.ones(3), shape=(3,))
        assert space.shape == (3,)

    def test_sample_shape(self):
        space = Box(low=jnp.zeros(4), high=jnp.ones(4), shape=(4,))
        key = jax.random.PRNGKey(0)
        sample = space.sample(key)
        assert sample.shape == (4,)

    def test_sample_bounds(self):
        low = jnp.array([-1.0, 0.0])
        high = jnp.array([1.0, 2.0])
        space = Box(low=low, high=high, shape=(2,))
        key = jax.random.PRNGKey(42)
        sample = space.sample(key)
        assert jnp.all(sample >= low)
        assert jnp.all(sample <= high)

    def test_contains(self):
        space = Box(low=jnp.zeros(2), high=jnp.ones(2), shape=(2,))
        assert space.contains(jnp.array([0.5, 0.5]))
        assert not space.contains(jnp.array([1.5, 0.5]))

    def test_sample_deterministic(self):
        space = Box(low=jnp.zeros(3), high=jnp.ones(3), shape=(3,))
        s1 = space.sample(jax.random.PRNGKey(0))
        s2 = space.sample(jax.random.PRNGKey(0))
        assert jnp.allclose(s1, s2)

    def test_jit_compatible(self):
        space = Box(low=jnp.zeros(3), high=jnp.ones(3), shape=(3,))
        key = jax.random.PRNGKey(0)
        jitted = jax.jit(lambda s, k: s.sample(k))
        result = jitted(space, key)
        assert result.shape == (3,)

    def test_static_fields_not_leaves(self):
        space = Box(low=jnp.zeros(2), high=jnp.ones(2), shape=(2,))
        leaves = jax.tree_util.tree_leaves(space)
        # Only low and high should be leaves
        assert len(leaves) == 2


# ==================== Discrete ====================

class TestDiscrete:
    def test_create(self):
        space = Discrete(n=5)
        assert space.n == 5

    def test_sample_range(self):
        space = Discrete(n=10)
        key = jax.random.PRNGKey(0)
        sample = space.sample(key)
        assert 0 <= int(sample) < 10

    def test_contains(self):
        space = Discrete(n=3)
        assert space.contains(jnp.int32(2))
        assert not space.contains(jnp.int32(3))

    def test_jit_compatible(self):
        space = Discrete(n=5)
        key = jax.random.PRNGKey(0)
        jitted = jax.jit(lambda s, k: s.sample(k))
        result = jitted(space, key)
        assert 0 <= int(result) < 5

    def test_static_fields_not_leaves(self):
        space = Discrete(n=5)
        leaves = jax.tree_util.tree_leaves(space)
        assert len(leaves) == 0


# ==================== MultiDiscrete ====================

class TestMultiDiscrete:
    def test_create(self):
        space = MultiDiscrete(nvec=jnp.array([3, 4, 5]), n_dims=3, shape=(3,))
        assert space.nvec.shape == (3,)

    def test_sample_shape(self):
        space = MultiDiscrete(nvec=jnp.array([3, 4]), n_dims=2, shape=(2,))
        sample = space.sample(jax.random.PRNGKey(0))
        assert sample.shape == (2,)

    def test_jit_compatible(self):
        space = MultiDiscrete(nvec=jnp.array([3, 4, 5]), n_dims=3, shape=(3,))
        key = jax.random.PRNGKey(0)
        jitted = jax.jit(lambda s, k: s.sample(k))
        result = jitted(space, key)
        assert result.shape == (3,)

    def test_static_fields_not_leaves(self):
        space = MultiDiscrete(nvec=jnp.array([3, 4, 5]), n_dims=3, shape=(3,))
        leaves = jax.tree_util.tree_leaves(space)
        # Only nvec should be a leaf
        assert len(leaves) == 1


# ==================== MultiBinary ====================

class TestMultiBinary:
    def test_sample_shape(self):
        space = MultiBinary(n=5, shape=(5,))
        sample = space.sample(jax.random.PRNGKey(0))
        assert sample.shape == (5,)

    def test_sample_values(self):
        space = MultiBinary(n=4, shape=(4,))
        sample = space.sample(jax.random.PRNGKey(0))
        assert jnp.all((sample == 0) | (sample == 1))

    def test_jit_compatible(self):
        space = MultiBinary(n=5, shape=(5,))
        key = jax.random.PRNGKey(0)
        jitted = jax.jit(lambda s, k: s.sample(k))
        result = jitted(space, key)
        assert result.shape == (5,)
        assert jnp.all((result == 0) | (result == 1))


# ==================== make_box ====================

class TestMakeBox:
    def test_scalar_bounds(self):
        space = make_box(-1.0, 1.0, shape=(3,))
        assert space.shape == (3,)
        assert jnp.allclose(space.low, jnp.full(3, -1.0))
        assert jnp.allclose(space.high, jnp.full(3, 1.0))

    def test_array_bounds(self):
        low = jnp.array([0.0, -1.0])
        high = jnp.array([1.0, 2.0])
        space = make_box(low, high)
        assert space.shape == (2,)

    def test_invalid_bounds_raises(self):
        with pytest.raises(ValueError, match="low must be <= high"):
            make_box(1.0, -1.0, shape=(3,))


# ==================== Factory functions ====================

class TestMakeDiscrete:
    def test_basic(self):
        space = make_discrete(5)
        assert space.n == 5

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError, match="n must be > 0"):
            make_discrete(0)


class TestMakeMultiDiscrete:
    def test_basic(self):
        space = make_multi_discrete([3, 4, 5])
        assert space.n_dims == 3
        assert space.shape == (3,)

    def test_invalid_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            make_multi_discrete([])

    def test_invalid_zero_element_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            make_multi_discrete([3, 0, 5])


class TestMakeMultiBinary:
    def test_basic(self):
        space = make_multi_binary(4)
        assert space.n == 4
        assert space.shape == (4,)

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError, match="n must be > 0"):
            make_multi_binary(0)
