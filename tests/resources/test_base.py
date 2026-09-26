# Vendored test from PowerZooJax.
# Source        : tests/resource/test_base.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
"""L0 + L1 tests for base resource utilities — ResourceState, ResourceParams,
time_features.

L0: JAX contracts (jit, vmap, pytree)
L1: Correctness of utility functions
"""

import pytest
import jax
import jax.numpy as jnp

from powermarketjax.resources.base import (
    ResourceState, ResourceParams, time_features,
)


# ========================== L0 JAX Contracts ==========================

class TestResourceStatePytree:
    """ResourceState is a valid pytree and flax struct."""

    def test_can_create(self):
        s = ResourceState(
            current_p_mw=jnp.float32(0.0),
            current_q_mvar=jnp.float32(0.0),
            time_step=jnp.int32(0),
        )
        assert s.current_p_mw.shape == ()

    def test_pytree_round_trip(self):
        s = ResourceState(
            current_p_mw=jnp.float32(1.0),
            current_q_mvar=jnp.float32(2.0),
            time_step=jnp.int32(5),
        )
        leaves = jax.tree_util.tree_leaves(s)
        assert len(leaves) == 3
        # Can reconstruct
        treedef = jax.tree_util.tree_structure(s)
        s2 = treedef.unflatten(leaves)
        assert float(s2.current_p_mw) == pytest.approx(1.0)

    def test_replace(self):
        s = ResourceState(
            current_p_mw=jnp.float32(0.0),
            current_q_mvar=jnp.float32(0.0),
            time_step=jnp.int32(0),
        )
        s2 = s.replace(current_p_mw=jnp.float32(5.0))
        assert float(s2.current_p_mw) == 5.0
        assert float(s.current_p_mw) == 0.0  # original unchanged (immutable)


class TestResourceParamsPytree:

    def test_default_values(self):
        p = ResourceParams()
        assert p.bus_id == -1
        assert p.delta_t_hours == 0.5
        assert p.steps_per_day == 48
        assert p.max_steps == 48


# ========================== L0 time_features ==========================

class TestTimeFeaturesJit:

    def test_jit_compiles(self):
        s, c = time_features(jnp.int32(0), 48)
        assert s.shape == ()
        assert c.shape == ()

    def test_vmap_over_time_step(self):
        steps = jnp.arange(48, dtype=jnp.int32)
        sin_vals, cos_vals = jax.vmap(lambda t: time_features(t, 48))(steps)
        assert sin_vals.shape == (48,)
        assert cos_vals.shape == (48,)


# ========================== L1 time_features ==========================

class TestTimeFeaturesCorrectness:

    def test_step_0(self):
        s, c = time_features(jnp.int32(0), 48)
        assert float(s) == pytest.approx(0.0, abs=1e-6)
        assert float(c) == pytest.approx(1.0, abs=1e-6)

    def test_quarter_day(self):
        """At step 12 of 48, phase = π/2 → sin=1, cos=0."""
        s, c = time_features(jnp.int32(12), 48)
        assert float(s) == pytest.approx(1.0, abs=1e-5)
        assert float(c) == pytest.approx(0.0, abs=1e-5)

    def test_half_day(self):
        """At step 24 of 48, phase = π → sin=0, cos=-1."""
        s, c = time_features(jnp.int32(24), 48)
        assert float(s) == pytest.approx(0.0, abs=1e-5)
        assert float(c) == pytest.approx(-1.0, abs=1e-5)

    def test_full_day_wraps(self):
        """At step 48 of 48, phase = 2π → same as step 0."""
        s, c = time_features(jnp.int32(48), 48)
        s0, c0 = time_features(jnp.int32(0), 48)
        assert float(s) == pytest.approx(float(s0), abs=1e-5)
        assert float(c) == pytest.approx(float(c0), abs=1e-5)
