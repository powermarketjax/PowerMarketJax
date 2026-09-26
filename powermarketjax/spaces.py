# Vendored from PowerZooJax. Do not edit to track upstream;
# this file is now this repository's code.
# Source        : powerzoojax/envs/spaces.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
# Upstream licence: MIT, Copyright (c) 2026 PowerZooJax Contributors.
"""JAX-compatible action and observation space definitions (Gymnax/JaxMARL conventions).

All ``shape``, ``dtype``, ``n`` fields are marked ``pytree_node=False`` so they
become part of the PyTree *structure* (static), not traced leaves.  This is
required for correct JIT compilation when a Space object is passed as a
function argument.
"""

from typing import Tuple
import jax
import jax.numpy as jnp
import numpy as np
import chex
from flax import struct


@struct.dataclass
class Space:
    """Base class for spaces."""
    shape: Tuple[int, ...] = struct.field(pytree_node=False, default=())
    dtype: jnp.dtype = struct.field(pytree_node=False, default=jnp.float32)

    def sample(self, key: chex.PRNGKey) -> chex.Array:
        """Sample from the space."""
        raise NotImplementedError

    def contains(self, x: chex.Array) -> chex.Array:
        """Check if x is in the space."""
        raise NotImplementedError


@struct.dataclass
class Box(Space):
    """Continuous box space bounded by [low, high]."""
    low: chex.Array = None
    high: chex.Array = None
    shape: Tuple[int, ...] = struct.field(pytree_node=False, default=())
    dtype: jnp.dtype = struct.field(pytree_node=False, default=jnp.float32)

    def sample(self, key: chex.PRNGKey) -> chex.Array:
        return jax.random.uniform(
            key, shape=self.shape, minval=self.low, maxval=self.high, dtype=self.dtype
        )

    def contains(self, x: chex.Array) -> chex.Array:
        return jnp.all((x >= self.low) & (x <= self.high))


@struct.dataclass
class Discrete(Space):
    """Discrete space with n elements {0, 1, ..., n-1}."""
    n: int = struct.field(pytree_node=False, default=1)
    shape: Tuple[int, ...] = struct.field(pytree_node=False, default=())
    dtype: jnp.dtype = struct.field(pytree_node=False, default=jnp.int32)

    def sample(self, key: chex.PRNGKey) -> chex.Array:
        return jax.random.randint(key, shape=(), minval=0, maxval=self.n, dtype=self.dtype)

    def contains(self, x: chex.Array) -> chex.Array:
        return jnp.all((x >= 0) & (x < self.n))


@struct.dataclass
class MultiDiscrete(Space):
    """Multi-discrete space; each dimension has its own n (nvec: (n_dims,) int array)."""
    nvec: chex.Array = None
    n_dims: int = struct.field(pytree_node=False, default=0)
    shape: Tuple[int, ...] = struct.field(pytree_node=False, default=())
    dtype: jnp.dtype = struct.field(pytree_node=False, default=jnp.int32)

    def sample(self, key: chex.PRNGKey) -> chex.Array:
        keys = jax.random.split(key, self.n_dims)
        return jax.vmap(
            lambda k, n: jax.random.randint(k, shape=(), minval=0, maxval=n)
        )(keys, self.nvec).astype(self.dtype)

    def contains(self, x: chex.Array) -> chex.Array:
        return jnp.all((x >= 0) & (x < self.nvec))


@struct.dataclass
class MultiBinary(Space):
    """Multi-binary space; n binary {0, 1} dimensions."""
    n: int = struct.field(pytree_node=False, default=1)
    shape: Tuple[int, ...] = struct.field(pytree_node=False, default=())
    dtype: jnp.dtype = struct.field(pytree_node=False, default=jnp.int32)

    def sample(self, key: chex.PRNGKey) -> chex.Array:
        return jax.random.randint(key, shape=(self.n,), minval=0, maxval=2, dtype=self.dtype)

    def contains(self, x: chex.Array) -> chex.Array:
        return jnp.all((x == 0) | (x == 1))


def make_box(low, high, shape=None, dtype=jnp.float32) -> Box:
    """Create a Box space, broadcasting scalar low/high to shape if needed.

    NOTE: This is a setup-time helper — do NOT call inside JIT-compiled code
    (it uses ``jnp.broadcast_to`` and Python shape inference).
    """
    low = jnp.asarray(low, dtype=dtype)
    high = jnp.asarray(high, dtype=dtype)
    if shape is None:
        shape = low.shape if low.shape else high.shape if high.shape else ()
    if low.shape != shape:
        low = jnp.broadcast_to(low, shape)
    if high.shape != shape:
        high = jnp.broadcast_to(high, shape)
    # Validate bounds at setup time
    if not np.all(np.asarray(low) <= np.asarray(high)):
        raise ValueError(f"make_box: low must be <= high, got low={low}, high={high}")
    return Box(low=low, high=high, shape=shape, dtype=dtype)


def make_discrete(n: int, dtype=jnp.int32) -> Discrete:
    """Create a Discrete space with validation."""
    if n <= 0:
        raise ValueError(f"make_discrete: n must be > 0, got {n}")
    return Discrete(n=n, dtype=dtype)


def make_multi_discrete(nvec, dtype=jnp.int32) -> MultiDiscrete:
    """Create a MultiDiscrete space with validation."""
    nvec = jnp.asarray(nvec, dtype=jnp.int32)
    if nvec.ndim != 1 or nvec.shape[0] == 0:
        raise ValueError(f"make_multi_discrete: nvec must be 1-D and non-empty, got shape {nvec.shape}")
    if not np.all(np.asarray(nvec) > 0):
        raise ValueError(f"make_multi_discrete: all nvec elements must be > 0, got {nvec}")
    n_dims = int(nvec.shape[0])
    return MultiDiscrete(nvec=nvec, n_dims=n_dims, shape=(n_dims,), dtype=dtype)


def make_multi_binary(n: int, dtype=jnp.int32) -> MultiBinary:
    """Create a MultiBinary space with validation."""
    if n <= 0:
        raise ValueError(f"make_multi_binary: n must be > 0, got {n}")
    return MultiBinary(n=n, shape=(n,), dtype=dtype)
