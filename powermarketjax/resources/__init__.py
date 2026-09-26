# The device modules in this package are verbatim copies of
# powerzoojax/envs/resource/{base,battery,renewable,flexload,vehicle,diesel}.py;
# env_base.py is a verbatim copy of powerzoojax/envs/base.py. Upstream's
# resource/__init__.py also exports a device this repository does not vendor
# (it belongs to a different product line), so this file's exports below are
# written fresh rather than copied.
"""Controllable DER assets: batteries, PV/wind, flexible loads, EVs, diesel.

Two parallel APIs per device:

- **Single-device Env** — gymnax-style, inherits ``Environment`` from
  ``env_base``. Unit-level RL on one device.
- **Bundle (SoA)** — a struct-of-arrays container of N homogeneous devices.
  Bundles do **not** inherit ``Environment``; they implement the duck-typed
  ``reset / step / observe`` protocol of ``ResourceBundle``.

``env_base.Environment`` is the resource layer's single-agent base class,
not the base class for the multi-agent market envs under
``powermarketjax.envs``.
"""

from powermarketjax.resources.base import (
    ResourceState,
    ResourceParams,
    ResourceBundleState,
    ResourceBundle,
    time_features,
)
from powermarketjax.resources.battery import (
    BatteryState,
    BatteryParams,
    BatteryEnv,
    make_battery_params,
    compute_feasible_power,
    update_soc,
    compute_feasible_power_batch,
    update_soc_batch,
    BatteryBundleState,
    BatteryBundle,
    make_battery_bundle,
)
from powermarketjax.resources.renewable import (
    RenewableState,
    RenewableParams,
    RenewableEnv,
    SolarEnv,
    WindEnv,
    RenewableBundleState,
    RenewableBundle,
    make_renewable_bundle,
)
from powermarketjax.resources.flexload import (
    FlexLoadState,
    FlexLoadParams,
    FlexLoadEnv,
    FlexLoadBundleState,
    FlexLoadBundle,
    make_flexload_bundle,
)
from powermarketjax.resources.vehicle import (
    VehicleState,
    VehicleParams,
    VehicleEnv,
    make_vehicle_params,
)
from powermarketjax.resources.diesel import (
    DieselParams,
    compute_dg_power,
    compute_dg_fuel_cost,
    compute_dg_emissions,
    DieselBundleState,
    DieselBundle,
    make_diesel_bundle,
)

__all__ = [
    # Base
    "ResourceState",
    "ResourceParams",
    "ResourceBundleState",
    "ResourceBundle",
    "time_features",
    # Battery
    "BatteryState",
    "BatteryParams",
    "BatteryEnv",
    "make_battery_params",
    "compute_feasible_power",
    "update_soc",
    "compute_feasible_power_batch",
    "update_soc_batch",
    "BatteryBundleState",
    "BatteryBundle",
    "make_battery_bundle",
    # Renewable
    "RenewableState",
    "RenewableParams",
    "RenewableEnv",
    "SolarEnv",
    "WindEnv",
    "RenewableBundleState",
    "RenewableBundle",
    "make_renewable_bundle",
    # Flexible load
    "FlexLoadState",
    "FlexLoadParams",
    "FlexLoadEnv",
    "FlexLoadBundleState",
    "FlexLoadBundle",
    "make_flexload_bundle",
    # Vehicle
    "VehicleState",
    "VehicleParams",
    "VehicleEnv",
    "make_vehicle_params",
    # Diesel
    "DieselParams",
    "compute_dg_power",
    "compute_dg_fuel_cost",
    "compute_dg_emissions",
    "DieselBundleState",
    "DieselBundle",
    "make_diesel_bundle",
]
