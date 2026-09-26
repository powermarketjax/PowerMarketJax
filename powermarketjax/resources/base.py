# Vendored from PowerZooJax. Do not edit to track upstream;
# this file is now this repository's code.
# Source        : powerzoojax/envs/resource/base.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# NOT verbatim. Beyond the package rename (powerzoojax -> powermarketjax, in
# both import statements and path strings): the module docstring was condensed
# on 2026-08-21 for API documentation, and two of its statements were corrected
# to this repository at the same time -- the single-device Env inherits
# Environment from `resources.env_base`, not `envs.base`, and the list of DER
# assets no longer names AI data centres, which this repository has no module
# for. No executable line was changed.
# Upstream licence: MIT, Copyright (c) 2026 PowerZooJax Contributors.
"""Resource layer base classes.

Controllable DER assets — batteries, EVs, PV / wind, diesel, flexible
loads — sitting below the grid and market envs. Every resource module
exposes one or both of two parallel APIs:

  - **Single-device Env** : a Gymnax-style environment for unit-level
    RL training. Inherits ``Environment`` from ``resources.env_base``
    and uses ``ResourceState`` + ``ResourceParams`` (defined here) as
    its pytree base. ``step`` returns the CMDP tuple
    ``(obs, state, reward, costs, done, info)`` with ``costs`` produced
    by ``stack_costs(...)``.

  - **Bundle (SoA)** : a struct-of-arrays container of N homogeneous
    devices, attached to a host env via ``params.resources``. Bundles do
    NOT inherit ``Environment`` — they implement the duck-typed
    ``reset / step / observe`` protocol documented on ``ResourceBundle``
    below, and ``step`` returns ``(state, p_inject_mw, q_inject_mvar,
    obs_slice, cost_info)`` with ``cost_info`` as a *dict*. A caller
    wiring a Bundle into a CMDP training loop stacks costs itself.

Conventions across every resource:
    Sign     : current_p_mw > 0 ⇒ injection (discharge / generation)
               current_p_mw < 0 ⇒ absorption (charge / load)
               current_q_mvar follows the same sign rule when present.
    Reward   : 0. Resource envs are CMDP-only — task reward is supplied
               by the host env or a training wrapper.
    Costs    : a small named vector of nonnegative scalars per step,
               ordered by each subclass's ``constraint_names``.

``ResourceState`` / ``ResourceParams`` do **not** inherit from
``EnvState`` / ``EnvParams``. They share ``time_step``, ``max_steps`` and
``delta_t_hours``, but ``ResourceState`` omits ``done`` — each resource
subclass adds it alongside its own domain-specific fields.
"""

import chex
from flax import struct

from powermarketjax.resources.env_base import time_features  # canonical definition; re-exported via __init__.py


@struct.dataclass
class ResourceState:
    """Base state for all resource environments.

    Attributes:
        current_p_mw: Active power injection [MW]. Positive = inject to grid, negative = draw.
        current_q_mvar: Reactive power [MVAr]. Usually 0 for DER resources.
        time_step: Current simulation step (int32).
    """
    current_p_mw: chex.Array   # float32 scalar — active power injection [MW]
    current_q_mvar: chex.Array   # float32 scalar — reactive power [MVAr]
    time_step: chex.Array   # int32 scalar


@struct.dataclass
class ResourceParams:
    """Base parameters for all resource environments.

    Attributes:
        bus_id: Connection bus ID (int32). -1 = unregistered sentinel.
        delta_t_hours: Time step duration in hours.
        steps_per_day: Number of steps per day (for time encoding).
        max_steps: Maximum steps per episode.

    All fields are marked ``pytree_node=False`` — they are static
    compile-time constants under ``jax.jit`` and will not be traced.
    """
    bus_id: int = struct.field(pytree_node=False, default=-1)
    delta_t_hours: float = struct.field(pytree_node=False, default=0.5)
    steps_per_day: int = struct.field(pytree_node=False, default=48)
    max_steps: int = struct.field(pytree_node=False, default=48)


# ============ ResourceBundle Protocol (duck-typed, no ABC) ============

@struct.dataclass
class ResourceBundleState:
    """Marker base for bundle state pytrees (documentation only).

    Each concrete bundle (``BatteryBundle``, ``RenewableBundle``, ...)
    defines its own state ``@struct.dataclass``.  This base carries no
    fields and exists for type-hint documentation.
    """
    pass


class ResourceBundle:
    """Protocol documentation for resource bundles (duck-typed, no ABC).

    A *bundle* groups N homogeneous devices (batteries, renewables, ...)
    into one struct-of-arrays container.  It is a ``@struct.dataclass`` with:

    **Static fields** (``pytree_node=False``, compile-time constants)::

        n_devices            : int    — number of devices
        per_device_action_dim: int    — action dim per device (battery=1)
        per_device_obs_dim   : int    — obs dim per device   (battery=2)
        dt_hours             : float  — time-step duration [h]

    **Traced leaves** (``jnp.ndarray``, JAX-traced)::

        bus_idx    : int32  (n_devices,)   — grid node index per device
        <device-specific SoA arrays>       — e.g. power_max, capacity, ...

    **Derived properties**::

        action_dim = n_devices * per_device_action_dim
        obs_dim    = n_devices * per_device_obs_dim

    **Methods**::

        reset(key) -> BundleState
        step(state, action, ctx) -> (new_state, p_inject_mw, q_inject_mvar,
                                      obs_slice, cost_info)
        observe(state, ctx) -> obs_slice

    ``obs_slice`` uses **device-major** layout::

        obs_slice[i * per_device_obs_dim : (i+1) * per_device_obs_dim]

    gives device *i*'s observation.  A single-agent caller consumes the flat
    vector; a multi-agent one reshapes via ``per_device_*_dim``.
    """
    pass
