"""Smoke tests: the package imports, JAX jit works, and GPU is reachable if present.

These are deliberately domain-free. They exist so that the documented "how to run" block
is executable truth from day one, and so every later change has a
self-correction loop to run against.
"""

import jax
import jax.numpy as jnp
import pytest

import powermarketjax


def test_import_and_version():
    assert powermarketjax.__version__ == "0.0.1"


def test_jit_executes_and_keeps_float32():
    out = jax.jit(lambda x: x @ x)(jnp.eye(4))
    assert jnp.allclose(out, jnp.eye(4))
    assert out.dtype == jnp.float32


def test_jit_output_lands_on_gpu_when_available():
    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("no GPU")
    out = jax.jit(lambda x: x @ x)(jnp.eye(4))
    assert all(d.platform == "gpu" for d in out.devices())
