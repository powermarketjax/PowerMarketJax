# Vendored test from PowerZooJax.
# Source        : tests/resource/conftest.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
"""Shared fixtures for resource environment tests."""

import pytest
import jax
import jax.numpy as jnp

from powermarketjax.resources.battery import BatteryEnv, BatteryParams, make_battery_params
from powermarketjax.resources.renewable import (
    RenewableEnv, SolarEnv, WindEnv, RenewableParams,
)
from powermarketjax.resources.vehicle import VehicleEnv, VehicleParams, make_vehicle_params
from powermarketjax.resources.flexload import FlexLoadEnv, FlexLoadParams


@pytest.fixture
def key():
    """Deterministic PRNG key."""
    return jax.random.PRNGKey(42)
