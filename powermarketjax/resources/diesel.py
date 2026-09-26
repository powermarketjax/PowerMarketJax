# Vendored from PowerZooJax. Do not edit to track upstream;
# this file is now this repository's code.
# Source        : powerzoojax/envs/resource/diesel.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# NOT verbatim. Beyond the package rename (powerzoojax -> powermarketjax, in
# both import statements and path strings): docstrings and comments were
# added on 2026-08-20 for API documentation. No executable line was changed.
# Upstream licence: MIT, Copyright (c) 2026 PowerZooJax Contributors.
"""Diesel Generator (DG) — pure-JAX implementation.

A grid-connected diesel generator modelled as a *memoryless*, linear
controllable injection: at each step the agent picks a normalised
setpoint and the DG instantaneously produces that fraction of rated
power. No ramp rate, no startup/shutdown logic, no quadratic fuel
curve, no SOC-like state — DieselBundleState only carries the last-step
output so the host env can rebuild its obs slice without re-stepping.

Sign convention:
    p_inj ≥ 0  (DG never absorbs from the grid; Q = 0 in this benchmark)

Model:
    p_dg      = dg_norm · p_max                              (default)
    fuel_cost = p_dg · dt · fuel_cost_per_mwh                [$]
    carbon_kg = p_dg · dt · 1e3 · emission_factor[kg/kWh]    [kgCO2]

Optional minimum-loading constraint (real gensets should run ≥ ~30 % of
rated load to avoid wet-stacking engine damage). When ``p_min_norm > 0``:
    dg_norm < p_min_norm/2          → DG OFF (p = 0)
    p_min_norm/2 ≤ dg_norm < p_min  → snap up to p_min_norm · p_max
    dg_norm ≥ p_min_norm            → linear (dg_norm · p_max)
Introduces a discontinuity at dg_norm = p_min_norm/2, so on-policy
methods (PPO/SAC) are preferred over deterministic-policy ones.

Control problem (CMDP):
    Action      : a ∈ [0, 1] per device.
    Reward      : 0 — define externally (e.g. negative fuel + carbon).
    Costs       : *none* in the cost vector (cost_sum = 0). Fuel cost
                  and CO2 are surfaced via the info dict (``fuel_cost``,
                  ``carbon_kg``) — DG is treated as a controllable
                  injection, not a CMDP-constrained resource.
    Observation : 2-D per device [p_norm, 1 − p_norm]; the second field
                  is just the available headroom and is fully redundant
                  with the first, kept for the ResourceBundle protocol.
    Termination : owned by the host env (no DieselEnv standalone class).

Why no standalone Gymnax env: a memoryless DG is uninteresting to train
as a single-agent RL problem (optimal policy is myopic). Always attach
via DieselBundle to a host grid / microgrid env. ``DieselParams`` and
the ``compute_dg_*`` pure helpers are exposed as a thin Python-side
single-device API for callers who don't need the SoA layout.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

# Single-device pure helpers (consumed by both DieselBundle and Python mirror)
@struct.dataclass
class DieselParams:
    """Static parameters for a single diesel generator.

    Attributes:
        p_dg_max_mw: Nameplate capacity [MW].
        fuel_cost_per_mwh: Variable fuel cost [$/MWh].
        emission_factor: Carbon intensity [kgCO2 / kWh_e].

    All fields are ``pytree_node=False`` — compile-time constants under JIT.
    """
    p_dg_max_mw: float = struct.field(pytree_node=False, default=0.6)
    fuel_cost_per_mwh: float = struct.field(pytree_node=False, default=300.0)
    emission_factor: float = struct.field(pytree_node=False, default=0.80)


def compute_dg_power(
    desired_norm: chex.Array,
    p_dg_max_mw: float,
) -> chex.Array:
    """Compute DG active power from normalised setpoint in [0, 1]."""
    return jnp.clip(
        jnp.asarray(desired_norm, dtype=jnp.float32),
        jnp.float32(0.0),
        jnp.float32(1.0),
    ) * jnp.float32(p_dg_max_mw)


def compute_dg_fuel_cost(
    p_dg_mw: chex.Array,
    dt_h: float,
    fuel_cost_per_mwh: float,
) -> chex.Array:
    """Fuel cost [$] for one time step."""
    return p_dg_mw * jnp.float32(dt_h) * jnp.float32(fuel_cost_per_mwh)


def compute_dg_emissions(
    p_dg_mw: chex.Array,
    dt_h: float,
    emission_factor: float,
) -> chex.Array:
    """CO2 emissions [kgCO2] for one time step.

    MW × h = MWh; × 1e3 → kWh_e; × emission_factor[kg/kWh] → kgCO2.
    """
    return p_dg_mw * jnp.float32(dt_h) * jnp.float32(1e3) * jnp.float32(emission_factor)

# Bundle (SoA, for grid / microgrid env resource attachment)
@struct.dataclass
class DieselBundleState:
    """State for a DieselBundle — last-step DG output.

    Attributes:
        p_dg_mw: Last-step active output [MW], shape ``(n_devices,)``.
    """
    p_dg_mw: chex.Array  # (n_devices,)


@struct.dataclass
class DieselBundle:
    """Struct-of-arrays diesel-generator bundle for grid env attachment.

    Groups N DGs with identical action/obs dimensions.  All quantities are
    in MW; the host env is responsible for unit conversion.  Action per
    device is a single scalar in ``[0, 1]`` mapped to ``[0, p_max]``.

    Attributes:
        n_devices: Number of devices.
        per_device_action_dim: 1.
        per_device_obs_dim: 2 (p_norm, dg_margin_norm).
        dt_hours: Time-step duration [h].
        bus_idx: Internal bus index per device, shape (n_devices,) int32.
            For busless mode (microgrid), defaults to ``arange(n_devices)``.
        p_max: Rated power per device [MW], shape (n_devices,).
        fuel_cost_per_mwh: Fuel cost [$/MWh], shape (n_devices,).
        emission_factor: kgCO2 / kWh_e, shape (n_devices,).
        p_min_norm: Minimum loading fraction in ``[0, 1)`` per device, shape
            (n_devices,).  Default 0.0 reproduces the unconstrained
            (linear-from-0) behaviour.  When > 0 a soft deadband at
            ``p_min_norm / 2`` is applied: requested setpoints below the
            deadband shut the generator OFF (p = 0); setpoints above the
            deadband are clamped to ``[p_min_norm, 1]``.  Real diesel gensets
            should not run below ~30% of rated load (wet stacking damages the
            engine), so a typical opt-in value is ``p_min_norm = 0.3``.
    """

    # Static fields
    n_devices: int = struct.field(pytree_node=False)
    per_device_action_dim: int = struct.field(pytree_node=False, default=1)
    per_device_obs_dim: int = struct.field(pytree_node=False, default=2)
    dt_hours: float = struct.field(pytree_node=False, default=0.5)

    # Traced SoA leaves
    bus_idx: chex.Array = None         # int32 (n_devices,)
    p_max: chex.Array = None           # float32 (n_devices,) MW
    fuel_cost_per_mwh: chex.Array = None  # float32 (n_devices,)
    emission_factor: chex.Array = None    # float32 (n_devices,) kg/kWh
    p_min_norm: chex.Array = None      # float32 (n_devices,) ∈ [0, 1)

    @property
    def action_dim(self) -> int:
        """Width of the flat action vector for the whole bundle, not per device.

        One setpoint per device (``per_device_action_dim = 1``), so this is
        just ``n_devices``.
        """
        return self.n_devices * self.per_device_action_dim

    @property
    def obs_dim(self) -> int:
        """Width of the flat ``obs_slice`` for the whole bundle, not per device.

        Two channels per device ([p_norm, dg_margin_norm]), device-major:
        device *i* occupies slots ``[2i, 2i+2)``.
        """
        return self.n_devices * self.per_device_obs_dim

    def reset(self, key: chex.PRNGKey) -> DieselBundleState:
        """Initial bundle state — DG starts idle (p=0)."""
        return DieselBundleState(p_dg_mw=jnp.zeros((self.n_devices,), dtype=jnp.float32))

    def step(
        self,
        state: DieselBundleState,
        action: chex.Array,
        ctx: Dict[str, Any],
    ) -> Tuple[DieselBundleState, chex.Array, chex.Array, chex.Array, Dict[str, chex.Array]]:
        """Run one step for all N DGs.

        Args:
            state: Current bundle state.
            action: (n_devices,) in [0, 1] — DG normalised setpoints.
            ctx: Grid-provided context (unused, reserved for future use).

        Returns:
            (new_state, p_inject_mw, q_inject_mvar=0, obs_slice, cost_info).
            ``p_inject_mw`` is ≥ 0 in MW — a DG only injects.  ``cost_info``
            reports ``cost_sum = 0``: fuel is money the operator actually pays
            (a cost that reaches ``reward`` through ``profit``, not a CMDP
            constraint entry) and leaves separately as ``fuel_cost`` [$ for
            this step], next to ``carbon_kg`` [kgCO2 for this step].
        """
        a = jnp.clip(jnp.asarray(action, dtype=jnp.float32).reshape(self.n_devices),
                     jnp.float32(0.0), jnp.float32(1.0))

        # Optional minimum-loading soft constraint:
        #   - requested below deadband (p_min_norm/2) → DG OFF (p = 0)
        #   - requested above deadband → clamp to [p_min_norm, 1]
        # When p_min_norm == 0 (default) this collapses to a no-op (p = a).
        deadband = self.p_min_norm * jnp.float32(0.5)
        on_mask = a > deadband
        effective_norm = jnp.where(
            on_mask,
            jnp.maximum(a, self.p_min_norm),
            jnp.float32(0.0),
        )
        p_dg = effective_norm * self.p_max

        # Reactive power: DGs don't provide Q in this benchmark
        q_inj = jnp.zeros_like(p_dg)

        # Observation: per-device (p_norm, dg_margin_norm)
        # ``safe_p`` floors the divisor so a device rated 0 MW yields p_norm=0
        # instead of a NaN.
        safe_p = jnp.maximum(self.p_max, jnp.float32(1e-9))
        p_norm = p_dg / safe_p
        dg_margin = jnp.float32(1.0) - p_norm
        obs_per = jnp.stack([p_norm, dg_margin], axis=-1)
        obs_slice = obs_per.reshape(-1)

        # Economics — vectorised across N devices
        # MW x h x [$/MWh] = $; MW x h x 1e3 [kWh/MWh] x [kg/kWh] = kgCO2.
        # Both are summed over devices, so they are bundle totals, not (N,).
        fuel_cost = jnp.sum(p_dg * jnp.float32(self.dt_hours) * self.fuel_cost_per_mwh)
        carbon_kg = jnp.sum(
            p_dg * jnp.float32(self.dt_hours) * jnp.float32(1e3) * self.emission_factor
        )

        new_state = DieselBundleState(p_dg_mw=p_dg)
        return new_state, p_dg, q_inj, obs_slice, {
            "cost_sum": jnp.float32(0.0),
            "cost": jnp.float32(0.0),  # deprecated compatibility alias
            "fuel_cost": fuel_cost,
            "carbon_kg": carbon_kg,
        }

    def observe(self, state: DieselBundleState, ctx: Dict[str, Any]) -> chex.Array:
        """Rebuild obs_slice from state (used by host env's _get_obs / reset)."""
        safe_p = jnp.maximum(self.p_max, jnp.float32(1e-9))
        p_norm = state.p_dg_mw / safe_p
        dg_margin = jnp.float32(1.0) - p_norm
        obs_per = jnp.stack([p_norm, dg_margin], axis=-1)
        return obs_per.reshape(-1)


def make_diesel_bundle(
    case: Optional[Any] = None,
    bus_ids: Optional[Sequence[int]] = None,
    *,
    n_devices: Optional[int] = None,
    p_max_mw: float = 0.6,
    fuel_cost_per_mwh: float = 300.0,
    emission_factor: float = 0.80,
    p_min_norm: float = 0.0,
    dt_hours: float = 0.5,
) -> DieselBundle:
    """Create a DieselBundle.

    Two modes:

    Grid mode (``case`` and ``bus_ids`` provided):
        ``bus_idx`` is resolved through ``case.node_ids`` exactly like
        ``make_battery_bundle``.  ``n_devices = len(bus_ids)``.

    Busless mode (``case=None, bus_ids=None``):
        Used by behind-the-meter host envs with no explicit topology.
        ``n_devices`` must be supplied explicitly (default 1).
        ``bus_idx = arange(n_devices)`` and the host env sums injections
        rather than scattering by index.

    Args:
        case: Grid case data (optional, for grid mode).
        bus_ids: External node IDs (optional, for grid mode).
        n_devices: Number of DGs (busless mode; ignored if ``bus_ids`` given).
        p_max_mw: Nameplate capacity [MW]. Scalar or per-device sequence.
        fuel_cost_per_mwh: Variable fuel cost [$/MWh].
        emission_factor: Carbon intensity [kgCO2 / kWh_e].
        p_min_norm: Minimum loading fraction in ``[0, 1)``.  Default 0.0
            keeps the legacy unconstrained linear behaviour.  Set to e.g.
            0.3 to model real diesel gensets that should not run below
            ~30 % of rated load (wet stacking).
        dt_hours: Time-step duration [h].

    Returns:
        DieselBundle ready to be attached to grid/microgrid params.
    """
    import numpy as np_cpu

    if bus_ids is not None:
        if case is None:
            raise ValueError("bus_ids requires case for index resolution")
        bus_ids = list(bus_ids)
        n = len(bus_ids)
        if n == 0:
            raise ValueError("bus_ids must be non-empty")
        node_ids = np_cpu.asarray(case.node_ids)
        internal_idx = np_cpu.empty(n, dtype=np_cpu.int32)
        for i, bid in enumerate(bus_ids):
            matches = np_cpu.where(node_ids == bid)[0]
            if len(matches) == 0:
                raise ValueError(
                    f"bus_ids[{i}]={bid} not found in case.node_ids. "
                    f"Available: {node_ids.tolist()}"
                )
            internal_idx[i] = int(matches[0])
        bus_idx = jnp.asarray(internal_idx, dtype=jnp.int32)
    else:
        n = int(n_devices) if n_devices is not None else 1
        if n <= 0:
            raise ValueError(f"n_devices must be >= 1, got {n}")
        bus_idx = jnp.arange(n, dtype=jnp.int32)

    def _broadcast(val):
        """Tile a scalar to ``(n,)`` float32, or pass a length-``n`` sequence through."""
        if hasattr(val, "__len__") and not isinstance(val, str):
            arr = np_cpu.asarray(val, dtype=np_cpu.float32)
            if arr.shape[0] != n:
                raise ValueError(f"Expected length {n}, got {arr.shape[0]}")
            return jnp.asarray(arr)
        return jnp.full((n,), float(val), dtype=jnp.float32)

    return DieselBundle(
        n_devices=n,
        per_device_action_dim=1,
        per_device_obs_dim=2,
        dt_hours=float(dt_hours),
        bus_idx=bus_idx,
        p_max=_broadcast(p_max_mw),
        fuel_cost_per_mwh=_broadcast(fuel_cost_per_mwh),
        emission_factor=_broadcast(emission_factor),
        p_min_norm=_broadcast(p_min_norm),
    )
