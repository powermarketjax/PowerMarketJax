# Vendored from PowerZooJax. Do not edit to track upstream;
# this file is now this repository's code.
# Source        : powerzoojax/envs/resource/battery.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# NOT verbatim. Beyond the package rename (powerzoojax -> powermarketjax, in
# both import statements and path strings): setup-time validation was added to
# make_battery_bundle on 2026-08-05, rejecting parameters that divide by zero
# in update_soc (capacity_mwh, eta_discharge) or are physically impossible.
# Extended 2026-08-06, after the 2026-08-05 version was found to let every
# non-finite value through (NaN compares False against every bound, so it
# reached soc as NaN -- the failure the block existed to prevent; inf was
# silent the other way, freezing soc while p_inject stayed unconstrained).
# ``_require`` now folds ~isfinite into the same predicate. Runtime numerics
# are unchanged.
# Docstrings and comments were added on 2026-08-20 for API documentation; no
# executable line was changed.
# Upstream licence: MIT, Copyright (c) 2026 PowerZooJax Contributors.
"""Battery Energy Storage System (BESS) — pure-JAX implementation.

A grid-connected battery whose only stateful quantity is SOC. Each step
the agent commands a grid-side power [MW], clipped to the SOC-feasible
envelope, then SOC is integrated with one-way efficiencies on each leg:
discharging loses P/η_d per hour, charging gains |P|·η_c per hour. The
envelope is state-dependent — the feasible discharge / charge limits
move with SOC every step.

Sign convention:
    current_p_mw > 0 ⇒ discharge / inject to grid
    current_p_mw < 0 ⇒ charge   / draw from grid

Control problem (CMDP):
    Action      : a ∈ [-1, 1], scaled to [-power_mw, power_mw], then
                  clipped to the SOC-feasible envelope (cost_action_clip
                  reports the truncation).
    Reward      : 0. Define the task externally in a wrapper.
    Costs       : (cycle_throughput,) = |P|·dt·cycle_cost_per_mwh [$], a
                  soft proxy for cell aging. cost_action_clip lives in
                  info only — feasibility is hard-clipped, not penalised.
    Termination : t ≥ max_steps (auto-resets).

Two public APIs share this physics:

``BatteryEnv``    — standalone Gymnax-style env. 1-D action, 6-D obs
                    [soc, p_norm, p_dis_max_norm, p_chg_max_norm,
                     sin t, cos t]. Costs as a stacked vector;
                    cost_action_clip in info.

``BatteryBundle`` — SoA resource attached to a host env that owns the
                    episode loop, time features, and reward.
                      P-only (default) : 1-D action, obs [soc, p_norm].
                      P+Q  (enabled)   : 2-D action, obs [soc, p_norm, q_norm],
                                         P-priority PQ-circle projection
                                         Q_max = sqrt(max(S_rated² − P², 0)),
                                         Q energetically free (ideal inverter).
                    Costs as a *dict* (cost_sum, cost_cycle,
                    cost_soc_clip), not a stacked vector.

Defaults: η_c = η_d = 0.95. ``make_battery_params(eta_roundtrip=...)``
splits a round-trip value as sqrt() across missing sides.
"""

import math
from functools import partial
from typing import Any, Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as tu
from flax import struct

from powermarketjax.resources.env_base import (
    Environment,
    denormalize_action,
    stack_costs,
)
from powermarketjax.spaces import Box
from powermarketjax.resources.base import (
    ResourceState, ResourceParams, time_features,
)

_DEFAULT_ONEWAY_ETA = 0.95


def _resolve_battery_efficiencies(
    eta_charge: Optional[float],
    eta_discharge: Optional[float],
    eta_roundtrip: Optional[float],
) -> Tuple[float, float]:
    """Resolve one-way charge/discharge efficiencies from any combination of
    ``eta_charge``, ``eta_discharge`` and ``eta_roundtrip``."""
    if eta_roundtrip is None:
        ec = _DEFAULT_ONEWAY_ETA if eta_charge is None else float(eta_charge)
        ed = _DEFAULT_ONEWAY_ETA if eta_discharge is None else float(eta_discharge)
        if not (0.0 < ec <= 1.0):
            raise ValueError(f"eta_charge must be in (0, 1], got {ec}")
        if not (0.0 < ed <= 1.0):
            raise ValueError(f"eta_discharge must be in (0, 1], got {ed}")
        return ec, ed

    if not (0.0 < eta_roundtrip <= 1.0):
        raise ValueError(f"eta_roundtrip must be in (0, 1], got {eta_roundtrip}")

    eta_rt_sqrt = math.sqrt(eta_roundtrip)

    if eta_charge is not None and eta_discharge is not None:
        ec, ed = float(eta_charge), float(eta_discharge)
        if not (0.0 < ec <= 1.0):
            raise ValueError(f"eta_charge must be in (0, 1], got {ec}")
        if not (0.0 < ed <= 1.0):
            raise ValueError(f"eta_discharge must be in (0, 1], got {ed}")
        return ec, ed

    if eta_charge is not None:
        ec = float(eta_charge)
        if not (0.0 < ec <= 1.0):
            raise ValueError(f"eta_charge must be in (0, 1], got {ec}")
        return ec, eta_rt_sqrt

    if eta_discharge is not None:
        ed = float(eta_discharge)
        if not (0.0 < ed <= 1.0):
            raise ValueError(f"eta_discharge must be in (0, 1], got {ed}")
        return eta_rt_sqrt, ed

    return eta_rt_sqrt, eta_rt_sqrt


# State / Params
@struct.dataclass
class BatteryState(ResourceState):
    """Battery state extending ResourceState.

    Attributes:
        soc: State of charge [0, 1].
        done: Episode termination flag.
    """
    soc: chex.Array        # float32 scalar [0, 1]
    done: chex.Array       # bool scalar


@struct.dataclass
class BatteryParams(ResourceParams):
    """Battery parameters extending ResourceParams.

    Attributes:
        capacity_mwh: Energy capacity [MWh].
        power_mw: Maximum charge/discharge power [MW].
        eta_charge: One-way charging efficiency [0, 1].
        eta_discharge: One-way discharging efficiency [0, 1].
        soc_min: Minimum allowed SOC.
        soc_max: Maximum allowed SOC.
        initial_soc: SOC at reset.

    All scalar fields are ``pytree_node=False`` — static under JIT.
    """
    capacity_mwh: float = struct.field(pytree_node=False, default=50.0)
    power_mw: float = struct.field(pytree_node=False, default=20.0)
    eta_charge: float = struct.field(pytree_node=False, default=0.95)
    eta_discharge: float = struct.field(pytree_node=False, default=0.95)
    soc_min: float = struct.field(pytree_node=False, default=0.1)
    soc_max: float = struct.field(pytree_node=False, default=0.9)
    initial_soc: float = struct.field(pytree_node=False, default=0.5)
    cycle_cost_per_mwh: float = struct.field(pytree_node=False, default=0.0)


# Pure Functions
@jax.jit
def compute_feasible_power(
    soc: chex.Array,
    desired_power: chex.Array,
    params: BatteryParams,
) -> chex.Array:
    """Compute feasible grid-side power respecting SOC and power limits.

    Discharge (P > 0):
        max_discharge = (soc - soc_min) * capacity * η_d / dt
    Charge (P < 0):
        max_charge = (soc_max - soc) * capacity / (η_c * dt)

    Both limits are **grid-side** [MW], and that is what fixes the placement of
    the efficiencies.  ``(soc - soc_min) * capacity`` is energy sitting in the
    cells, so on its way out it is *multiplied* by η_d; conversely a grid-side
    charge only deposits η_c of itself, so the cell-side headroom
    ``(soc_max - soc) * capacity`` is *divided* by η_c to become a grid-side
    limit.  Both are then capped at the rated ``power_mw``.
    """
    dt = params.delta_t_hours

    # Max discharge limited by available energy
    max_discharge = jnp.clip(
        (soc - params.soc_min) * params.capacity_mwh * params.eta_discharge / dt,
        0.0, params.power_mw)

    # Max charge limited by available capacity
    max_charge = jnp.clip(
        (params.soc_max - soc) * params.capacity_mwh / (params.eta_charge * dt),
        0.0, params.power_mw)

    # Clip: positive side by discharge limit, negative side by charge limit
    feasible = jnp.clip(desired_power, -max_charge, max_discharge)
    return feasible


@jax.jit
def update_soc(
    soc: chex.Array,
    power: chex.Array,
    params: BatteryParams,
) -> chex.Array:
    """Update SOC based on grid-side power (Coulomb counting with losses).

    Discharge (P > 0): Δsoc = -P * dt / (η_d * capacity)
    Charge   (P < 0): Δsoc = -P * dt * η_c / capacity   (P is negative, so -P is positive)

    ``power`` is grid-side [MW], so the two efficiencies sit on the opposite
    sides from ``compute_feasible_power``: discharging *divides* by η_d (the
    cells give up more than the grid receives) and charging *multiplies* by
    η_c (the cells keep less than the grid supplies).  The closing clip to
    [soc_min, soc_max] is defensive — that bound is the one the power envelope
    already enforces.
    """
    dt = params.delta_t_hours

    # Discharge branch: battery loses more energy than grid receives
    delta_discharge = -power * dt / (params.eta_discharge * params.capacity_mwh)

    # Charge branch: battery gains less energy than grid supplies
    delta_charge = -power * dt * params.eta_charge / params.capacity_mwh

    delta_soc = jnp.where(power >= 0, delta_discharge, delta_charge)
    new_soc = jnp.clip(soc + delta_soc, params.soc_min, params.soc_max)
    return new_soc


# Environment Class
class BatteryEnv(Environment):
    """Battery Energy Storage System environment.

    Action: scalar in [-1, 1], denormalized internally to [-power_mw, power_mw] MW.
        Positive = discharge (inject to grid), negative = charge (draw from grid).
    Observation: 6-D [soc, p_norm, headroom_norm, charge_room_norm, time_sin, time_cos].
    """

    # RL Interface Methods
    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: BatteryParams
    ) -> Tuple[chex.Array, BatteryState]:
        """Start a fresh episode: SOC at ``initial_soc``, power at 0 MW.

        Deterministic — ``key`` is accepted for the ``Environment`` contract
        but never drawn from.  Randomised initial SOC exists only on the SoA
        side (``BatteryBundle.randomize_initial_soc``).
        """
        state = BatteryState(
            current_p_mw=jnp.float32(0.0),
            current_q_mvar=jnp.float32(0.0),
            time_step=jnp.int32(0),
            soc=jnp.float32(params.initial_soc),
            done=jnp.bool_(False),
        )
        obs = self._get_obs(state, params)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: BatteryState,
        action: chex.Array,
        params: BatteryParams,
    ) -> Tuple[chex.Array, BatteryState, chex.Array, chex.Array, chex.Array, Dict[str, Any]]:
        """Apply one grid-side power command and integrate SOC over ``delta_t_hours``.

        The command is denormalised to ±``power_mw`` and then hard-clipped to
        the SOC-feasible envelope; the truncated magnitude is reported in
        ``info["cost_action_clip"]`` and is not charged to ``costs``.

        Args:
            key: Consumed only to derive the auto-reset key; the transition
                itself is deterministic.
            action: Scalar in [-1, 1]. Positive = discharge (inject to grid),
                negative = charge (draw from grid).

        Returns:
            ``(obs, state, reward, costs, done, info)``.  ``reward`` is always
            0.0 — resource envs are CMDP-only and the task reward comes from
            the host env or a wrapper.  ``costs`` is the 1-vector ordered by
            ``constraint_names``; ``info`` additionally carries
            ``cost_action_clip``, which never enters ``costs``.
        """

        # Denormalize action from [-1, 1] to [-power_mw, power_mw]
        action = jnp.float32(action).reshape(())
        desired = denormalize_action(
            action, -params.power_mw, params.power_mw)

        # Compute feasible power respecting SOC constraints
        feasible = compute_feasible_power(state.soc, desired, params)

        # Update SOC
        new_soc = update_soc(state.soc, feasible, params)

        # Next state
        new_time = state.time_step + 1
        done = new_time >= params.max_steps
        new_state = BatteryState(
            current_p_mw=feasible,
            current_q_mvar=jnp.float32(0.0),
            time_step=new_time,
            soc=new_soc,
            done=done,
        )

        # Auto-reset: when done, swap in fresh reset state
        # The split is the PRNG contract of ``Environment.step``: the reset
        # draw must not reuse the key that drove the transition, or the new
        # episode's initial state correlates with the last step's randomness.
        # The swap is ``jnp.where`` rather than ``lax.cond`` because under vmap
        # both branches execute anyway, and both are cheap here.
        _, k_reset = jax.random.split(key)
        _, reset_state = self.reset(k_reset, params)
        final_state = tu.tree_map(
            lambda n, r: jnp.where(done, r, n), new_state, reset_state)
        final_obs = self._get_obs(final_state, params)

        reward = jnp.float32(0.0)

        # Normalized clipping magnitude: how far the raw command exceeded the
        # feasible range.  Lets RL detect "my action was silently clipped" and
        # learn to stay within physical limits instead of relying on the guard.
        safe_p = jnp.maximum(params.power_mw, 1e-6)  # floors the divisor below
        cost_action_clip = jnp.abs(desired - feasible) / safe_p
        # |P| [MW] x Δt [h] x cycle_cost_per_mwh [$/MWh] = $ charged this step.
        cycle_cost = jnp.abs(feasible) * params.delta_t_hours * params.cycle_cost_per_mwh
        costs = stack_costs(cycle_cost)
        info = {
            "cost_cycle_throughput": cycle_cost,
            "cost_sum": cycle_cost,
            "cost_action_clip": cost_action_clip,
        }
        return final_obs, final_state, reward, costs, done, info

    # Spaces & Observation
    def _get_obs(self, state: BatteryState, params: BatteryParams) -> chex.Array:
        """6-D observation: [soc, p_norm, p_discharge_max_norm, p_charge_max_norm, time_sin, time_cos].

        ``p_discharge_max_norm``, ``p_charge_max_norm`` ∈ [0, 1]:
            SOC-feasible maximum discharge / charge power, normalised by
            ``power_mw``.  Both the rated-power cap and the SOC energy-availability
            constraint are accounted for.

            - ``p_discharge_max_norm = min(max_p_discharge, power_mw) / power_mw``
              where ``max_p_discharge = (soc - soc_min) * capacity * η_d / Δt``
            - ``p_charge_max_norm = min(max_p_charge, power_mw) / power_mw``
              where ``max_p_charge = (soc_max - soc) * capacity / (η_c * Δt)``
        """
        safe_p = jnp.maximum(params.power_mw, 1e-6)
        dt = params.delta_t_hours
        p_norm = state.current_p_mw / safe_p

        # SOC-feasible max discharge (grid-side) capped at rated power
        max_p_discharge = jnp.maximum(0.0, state.soc - params.soc_min) * params.capacity_mwh * params.eta_discharge / dt
        p_discharge_max_norm = jnp.minimum(max_p_discharge, params.power_mw) / safe_p

        # SOC-feasible max charge (grid-side magnitude) capped at rated power
        max_p_charge = jnp.maximum(0.0, params.soc_max - state.soc) * params.capacity_mwh / (params.eta_charge * dt)
        p_charge_max_norm = jnp.minimum(max_p_charge, params.power_mw) / safe_p

        t_sin, t_cos = time_features(state.time_step, params.steps_per_day)
        return jnp.stack([state.soc, p_norm, p_discharge_max_norm, p_charge_max_norm, t_sin, t_cos])

    def observation_space(self, params: BatteryParams) -> Box:
        """Box over the 6-D vector built by ``_get_obs``.

        In order: ``soc`` ∈ [0, 1]; ``p_norm`` ∈ [-1, 1], the grid-side power
        divided by ``power_mw`` (positive = discharge); the two SOC-feasible
        headroom channels ``p_discharge_max_norm`` / ``p_charge_max_norm``
        ∈ [0, 1] (see ``_get_obs`` for how they are formed); and the
        time-of-day pair ``time_sin`` / ``time_cos`` ∈ [-1, 1].  Every channel
        is dimensionless.
        """
        low = jnp.array([0.0, -1.0, 0.0, 0.0, -1.0, -1.0], dtype=jnp.float32)
        high = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
        return Box(low=low, high=high, shape=(6,), dtype=jnp.float32)

    def action_space(self, params: BatteryParams) -> Box:
        """Box([-1, 1]) of shape (1,) — normalised grid-side power setpoint.

        ``+1`` is ``+power_mw`` (full discharge, injection to the grid), ``-1``
        is ``-power_mw`` (full charge, draw from the grid), linear through 0
        via ``denormalize_action``.  The envelope is symmetric because a
        battery has one rating for both directions; ``VehicleEnv`` rates the
        two sides separately and maps its action piecewise instead.
        """
        return Box(
            low=jnp.array([-1.0], dtype=jnp.float32),
            high=jnp.array([1.0], dtype=jnp.float32),
            shape=(1,), dtype=jnp.float32,
        )

    # Status & Diagnostics
    @property
    def name(self) -> str:
        """Registration name, pinned to a literal rather than the class name."""
        return "BatteryEnv"

    def default_params(self) -> BatteryParams:
        """Stock 50 MWh / 20 MW unit: η = 0.95 each way, SOC held in [0.1, 0.9]."""
        return BatteryParams()

    def constraint_names(self, params: BatteryParams) -> tuple[str, ...]:
        """Names of the ``costs`` entries, in the order ``step`` stacks them.

        ``cycle_throughput`` — |P| x Δt x ``cycle_cost_per_mwh``, in $ per
        step, a soft proxy for cell ageing.  It is money the owner pays, yet
        the resource layer carries it in the CMDP constraint vector; a market
        caller that wants it as the cost that reaches ``reward`` through
        ``profit`` has to move it across itself.  ``cycle_cost_per_mwh``
        defaults to 0, so the entry is 0 until set.
        """
        return ("cycle_throughput",)


def make_battery_params(
    bus_id: int = -1,
    capacity_mwh: float = 50.0,
    power_mw: float = 20.0,
    eta_charge: Optional[float] = None,
    eta_discharge: Optional[float] = None,
    eta_roundtrip: Optional[float] = None,
    soc_min: float = 0.1,
    soc_max: float = 0.9,
    initial_soc: float = 0.5,
    delta_t_hours: float = 0.5,
    steps_per_day: int = 48,
    max_steps: int = 48,
    cycle_cost_per_mwh: float = 0.0,
) -> BatteryParams:
    """Build ``BatteryParams``, resolving efficiencies from ``eta_charge`` /
    ``eta_discharge`` / ``eta_roundtrip``.

    Args:
        bus_id: Connection bus ID for grid attachment. -1 = unregistered sentinel.
            Set this when the params are attached to a grid-level host env's
            resource list.
        capacity_mwh: Energy capacity [MWh].
        power_mw: Maximum charge/discharge power [MW].

    Omitted one-way efficiencies default to 0.95 unless ``eta_roundtrip`` is set,
    in which case missing sides use ``sqrt(eta_roundtrip)``.
    """
    _eta_c, _eta_d = _resolve_battery_efficiencies(
        eta_charge, eta_discharge, eta_roundtrip
    )

    return BatteryParams(
        bus_id=bus_id,
        capacity_mwh=capacity_mwh,
        power_mw=power_mw,
        eta_charge=_eta_c,
        eta_discharge=_eta_d,
        soc_min=soc_min,
        soc_max=soc_max,
        initial_soc=initial_soc,
        delta_t_hours=delta_t_hours,
        steps_per_day=steps_per_day,
        max_steps=max_steps,
        cycle_cost_per_mwh=float(cycle_cost_per_mwh),
    )


# Vectorized SOC utilities for Grid Env resource attachment
# Bundle (SoA, for Grid Env resource attachment)
@struct.dataclass
class BatteryBundleState:
    """State for a BatteryBundle — SOC of all N devices."""
    soc: chex.Array  # (n_devices,)


@struct.dataclass
class BatteryBundle:
    """Struct-of-arrays battery bundle for grid env attachment.

    Groups N batteries with identical action/obs dimensions.
    Works in MW throughout; the grid env handles unit conversion.

    When ``enable_q_control=False`` (default, P-only mode):
        Action per device: [P_norm]          Obs per device: [soc, p_norm]
    When ``enable_q_control=True`` (P+Q mode):
        Action per device: [P_norm, Q_norm]  Obs per device: [soc, p_norm, q_norm]
        PQ circle constraint: P² + Q² ≤ S_rated² (P-priority projection).
    """

    # Static (pytree_node=False)
    n_devices: int = struct.field(pytree_node=False)
    per_device_action_dim: int = struct.field(pytree_node=False, default=1)
    per_device_obs_dim: int = struct.field(pytree_node=False, default=2)
    dt_hours: float = struct.field(pytree_node=False, default=0.5)
    enable_q_control: bool = struct.field(pytree_node=False, default=False)

    # Traced leaves — SoA, shape (n_devices,)
    bus_idx: chex.Array = None       # int32 (n_devices,)
    power_max: chex.Array = None     # (n_devices,) MW
    s_rated: chex.Array = None       # (n_devices,) MVA — inverter apparent power rating
    capacity: chex.Array = None      # (n_devices,) MWh
    soc_min: chex.Array = None       # (n_devices,)
    soc_max: chex.Array = None       # (n_devices,)
    eta_charge: chex.Array = None    # (n_devices,)
    eta_discharge: chex.Array = None  # (n_devices,)
    initial_soc: chex.Array = None   # (n_devices,)
    randomize_initial_soc: bool = struct.field(pytree_node=False, default=False)
    soc_init_low: float = struct.field(pytree_node=False, default=0.3)
    soc_init_high: float = struct.field(pytree_node=False, default=0.7)
    cycle_cost_per_mwh: float = struct.field(pytree_node=False, default=0.0)

    @property
    def action_dim(self) -> int:
        """Width of the flat action vector for the whole bundle, not per device.

        ``per_device_action_dim`` is 1 in P-only mode and 2 with
        ``enable_q_control``; device *i* owns slots
        ``[i * per_device_action_dim, (i+1) * per_device_action_dim)``.
        """
        return self.n_devices * self.per_device_action_dim

    @property
    def obs_dim(self) -> int:
        """Width of the flat ``obs_slice`` for the whole bundle, not per device.

        ``per_device_obs_dim`` is 2 ([soc, p_norm]) in P-only mode and 3
        ([soc, p_norm, q_norm]) with ``enable_q_control``; same device-major
        layout as ``action_dim``.
        """
        return self.n_devices * self.per_device_obs_dim

    def reset(self, key: chex.PRNGKey) -> BatteryBundleState:
        """Return initial bundle state (deterministic or uniform-random SOC)."""
        if self.randomize_initial_soc:
            unif = jax.random.uniform(
                key,
                shape=self.initial_soc.shape,
                dtype=jnp.float32,
                minval=self.soc_init_low,
                maxval=self.soc_init_high,
            )
            soc = jnp.clip(unif, self.soc_min, self.soc_max)
            return BatteryBundleState(soc=soc)
        return BatteryBundleState(soc=self.initial_soc)

    def step(
        self,
        state: BatteryBundleState,
        action: chex.Array,
        ctx: Dict[str, Any],
    ) -> Tuple[BatteryBundleState, chex.Array, chex.Array, chex.Array, Dict[str, chex.Array]]:
        """Run one step for all N devices.

        Args:
            state: Current bundle state.
            action: (action_dim,) in [-1, 1].
                P-only: one scalar per device.
                P+Q: [P_norm, Q_norm] per device (device-major, interleaved).
            ctx: Grid-provided context (unused by battery, reserved).

        Returns:
            (new_state, p_inject_mw, q_inject_mvar, obs_slice, cost_info)
        """

        # Denormalize actions
        if self.enable_q_control:
            act2d = action.reshape(self.n_devices, 2)
            p_desired = jnp.clip(act2d[:, 0], -1.0, 1.0) * self.power_max
            q_desired = jnp.clip(act2d[:, 1], -1.0, 1.0) * self.s_rated
        else:
            p_desired = jnp.clip(action, -1.0, 1.0) * self.power_max

        # P feasibility (SOC constraints)
        feasible_p = compute_feasible_power_batch(
            state.soc, p_desired, self.power_max, self.capacity,
            self.soc_min, self.soc_max, self.eta_charge, self.eta_discharge,
            self.dt_hours)

        # Q: PQ circle headroom after P is finalized (P-priority)
        if self.enable_q_control:
            q_max = jnp.sqrt(jnp.maximum(self.s_rated ** 2 - feasible_p ** 2, 0.0))
            feasible_q = jnp.clip(q_desired, -q_max, q_max)
        else:
            feasible_q = jnp.zeros_like(feasible_p)

        # SOC update (only P affects stored energy; Q is "free")
        new_soc = update_soc_batch(
            state.soc, feasible_p, self.capacity,
            self.eta_charge, self.eta_discharge,
            self.soc_min, self.soc_max, self.dt_hours)
        new_state = state.replace(soc=new_soc)

        # Observation
        # ``safe_p`` floors the divisor of p_norm and of cost_clip below: a
        # device rated 0 MW is legal here (it is how a heterogeneous
        # population is padded), and would otherwise divide by zero.
        safe_p = jnp.maximum(self.power_max, 1e-6)
        p_norm = feasible_p / safe_p
        if self.enable_q_control:
            safe_s = jnp.maximum(self.s_rated, 1e-6)
            q_norm = feasible_q / safe_s
            obs_per = jnp.stack([new_soc, p_norm, q_norm], axis=-1)
        else:
            obs_per = jnp.stack([new_soc, p_norm], axis=-1)
        obs_slice = obs_per.reshape(-1)

        # Clipping cost (normalized)
        cost_clip = jnp.sum(jnp.abs(p_desired - feasible_p) / safe_p)
        if self.enable_q_control:
            safe_s = jnp.maximum(self.s_rated, 1e-6)
            cost_clip = cost_clip + jnp.sum(jnp.abs(q_desired - feasible_q) / safe_s)

        # Cycle degradation cost
        # Σ|P| [MW] x dt [h] x [$/MWh] = $ over the whole bundle this step.
        cycle_cost = jnp.sum(jnp.abs(feasible_p)) * self.dt_hours * self.cycle_cost_per_mwh

        # cost_sum mixes a real $ cost (cycle_cost) with a dimensionless clip
        # magnitude (cost_clip); read cost_cycle / cost_soc_clip separately
        # from the dict below to keep the two channels apart.
        cost_sum = cost_clip + cycle_cost

        return new_state, feasible_p, feasible_q, obs_slice, {
            "cost_soc_clip": cost_clip,
            "cost_cycle": cycle_cost,
            "cost_sum": cost_sum,
        }

    def observe(self, state: BatteryBundleState, ctx: Dict[str, Any]) -> chex.Array:
        """Rebuild obs_slice from state without stepping (for _get_obs / reset).

        ``BatteryBundleState`` holds only SOC, so the ``p_norm`` (and
        ``q_norm``) channels come back as zeros rather than the last-step
        power.  That is exact right after ``reset``; between steps use the
        ``obs_slice`` that ``step`` returns instead.
        """
        p_norm = jnp.zeros_like(self.power_max)
        if self.enable_q_control:
            q_norm = jnp.zeros_like(self.s_rated)
            obs_per = jnp.stack([state.soc, p_norm, q_norm], axis=-1)
        else:
            obs_per = jnp.stack([state.soc, p_norm], axis=-1)
        return obs_per.reshape(-1)


def make_battery_bundle(
    case=None,
    bus_ids=None,
    *,
    n_devices=None,
    power_mw=20.0,
    capacity_mwh=50.0,
    s_rated_mva=None,
    enable_q_control: bool = False,
    soc_min=0.1,
    soc_max=0.9,
    eta_charge=0.95,
    eta_discharge=0.95,
    initial_soc=0.5,
    dt_hours=0.5,
    randomize_initial_soc: bool = False,
    soc_init_low: float = 0.3,
    soc_init_high: float = 0.7,
    cycle_cost_per_mwh: float = 0.0,
) -> BatteryBundle:
    """Create a BatteryBundle.

    Two construction modes:

    Grid mode (``case`` and ``bus_ids`` both provided):
        Resolves ``bus_idx`` through ``case.node_ids``.  ``n_devices``
        equals ``len(bus_ids)``.  Used by grid-level host envs with an
        explicit topology.

    Busless mode (``case=None, bus_ids=None``):
        For behind-the-meter host envs where no power flow exists.
        ``n_devices`` must be supplied (default 1).
        ``bus_idx = arange(n_devices)`` and the host env sums injections
        directly rather than scattering by index.

    All broadcasting and index lookup is done in NumPy at init time (CPU),
    then converted to jnp arrays.  NOT called inside JIT.

    Args:
        case: CaseData instance with ``node_ids`` attribute (optional).
        bus_ids: External node IDs (optional).
        n_devices: Number of batteries (busless mode; ignored if ``bus_ids``).
        power_mw: Max power per device [MW].  Scalar or Sequence[float].
        capacity_mwh: Energy capacity per device [MWh].
        s_rated_mva: Inverter apparent power rating [MVA] per device.
            Defaults to ``power_mw`` (inverter sized to active power rating).
            Set to e.g. ``1.1 * power_mw`` for extra reactive headroom.
        enable_q_control: If True, action space includes reactive power
            (per_device_action_dim=2, per_device_obs_dim=3) with PQ circle
            constraint.  Default False preserves P-only behavior.
        soc_min / soc_max: SOC bounds.
        eta_charge / eta_discharge: One-way efficiencies.
        initial_soc: Initial SOC (used when ``randomize_initial_soc`` is False).
        dt_hours: Time-step duration [h].
        randomize_initial_soc: If True, sample initial SOC per device from
            ``Uniform(soc_init_low, soc_init_high)`` at each reset (training).
        soc_init_low / soc_init_high: Bounds for random initial SOC.

    Returns:
        BatteryBundle with (n_devices,) arrays.
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
        bus_idx_arr = jnp.asarray(internal_idx, dtype=jnp.int32)
    else:
        n = int(n_devices) if n_devices is not None else 1
        if n <= 0:
            raise ValueError(f"n_devices must be >= 1, got {n}")
        bus_idx_arr = jnp.arange(n, dtype=jnp.int32)

    def _broadcast(val):
        """Tile a scalar to ``(n,)`` float32, or pass a length-``n`` sequence through."""
        if hasattr(val, '__len__') and not isinstance(val, str):
            arr = np_cpu.asarray(val, dtype=np_cpu.float32)
            if arr.shape[0] != n:
                raise ValueError(f"Expected length {n}, got {arr.shape[0]}")
            return jnp.asarray(arr)
        return jnp.full((n,), float(val), dtype=jnp.float32)

    s_rated_val = _broadcast(s_rated_mva if s_rated_mva is not None else power_mw)

    # --- Setup-time validation --------------------------------------------
    # ``update_soc`` divides by ``capacity`` and by ``eta_discharge``: a zero
    # in either silently yields NaN SOC while p_inject stays clean, which then
    # rides the whole lax.scan trajectory into obs -> policy -> gradients.
    # Reject it here, where a Python raise is legal. Pad a heterogeneous
    # device population with a small positive capacity and power_mw=0, not
    # capacity_mwh=0.
    def _require(arr, name, *, lo, hi=None, lo_strict=True):
        """Raise if any element is non-finite or out of bounds (``> lo``, ``<= hi``).

        ``hi=None`` leaves the upper end open; ``lo_strict=False`` relaxes the
        lower end to ``>= lo``, which is what lets a device be rated 0 MW.
        The error names the first offending device index.
        """
        a = np_cpu.asarray(arr, dtype=np_cpu.float64)
        # NaN compares False against every bound below, so ~isfinite must be
        # explicit in the predicate; inf passes those same bounds and would
        # freeze soc while leaving p_inject unconstrained.
        bad = ~np_cpu.isfinite(a)
        bad = bad | ((a <= lo) if lo_strict else (a < lo))
        if hi is not None:
            bad = bad | (a > hi)
        offenders = np_cpu.where(bad)[0]
        if offenders.size:
            i = int(offenders[0])
            bound = f"finite and in ({lo}, {hi}]" if hi is not None else (
                f"finite and > {lo}" if lo_strict else f"finite and >= {lo}")
            raise ValueError(
                f"make_battery_bundle: {name} must be {bound}, got "
                f"{a[i]} at device index {i}. "
                f"To model a device with no storage, use a small positive "
                f"capacity_mwh with power_mw=0 -- capacity_mwh=0 divides by "
                f"zero in update_soc and yields NaN SOC."
            )

    _require(_broadcast(capacity_mwh), "capacity_mwh", lo=0.0)
    _require(_broadcast(eta_charge), "eta_charge", lo=0.0, hi=1.0)
    _require(_broadcast(eta_discharge), "eta_discharge", lo=0.0, hi=1.0)
    _require(_broadcast(power_mw), "power_mw", lo=0.0, lo_strict=False)
    _require(s_rated_val, "s_rated_mva", lo=0.0, lo_strict=False)
    if not np_cpu.isfinite(dt_hours) or float(dt_hours) <= 0.0:
        raise ValueError(
            f"make_battery_bundle: dt_hours must be finite and > 0, got {dt_hours}")
    soc_lo = np_cpu.asarray(_broadcast(soc_min), dtype=np_cpu.float64)
    soc_hi = np_cpu.asarray(_broadcast(soc_max), dtype=np_cpu.float64)
    if np_cpu.any(soc_lo > soc_hi):
        raise ValueError(
            f"make_battery_bundle: soc_min must be <= soc_max, got "
            f"soc_min={soc_lo.tolist()}, soc_max={soc_hi.tolist()}"
        )

    return BatteryBundle(
        n_devices=n,
        per_device_action_dim=2 if enable_q_control else 1,
        per_device_obs_dim=3 if enable_q_control else 2,
        dt_hours=float(dt_hours),
        enable_q_control=enable_q_control,
        bus_idx=bus_idx_arr,
        power_max=_broadcast(power_mw),
        s_rated=s_rated_val,
        capacity=_broadcast(capacity_mwh),
        soc_min=_broadcast(soc_min),
        soc_max=_broadcast(soc_max),
        eta_charge=_broadcast(eta_charge),
        eta_discharge=_broadcast(eta_discharge),
        initial_soc=_broadcast(initial_soc),
        randomize_initial_soc=randomize_initial_soc,
        soc_init_low=float(soc_init_low),
        soc_init_high=float(soc_init_high),
        cycle_cost_per_mwh=float(cycle_cost_per_mwh),
    )


# Vectorized SOC utilities for Grid Env resource attachment
def compute_feasible_power_batch(
    soc: chex.Array,
    desired_power: chex.Array,
    power_max: chex.Array,
    capacity: chex.Array,
    soc_min: chex.Array,
    soc_max: chex.Array,
    eta_charge: chex.Array,
    eta_discharge: chex.Array,
    dt: float,
) -> chex.Array:
    """Vectorized feasibility clipping for a batch of batteries.

    All array arguments have shape ``(n_batteries,)``.  Safe for ``n_batteries=0``
    (empty arrays propagate through all operations and return empty arrays).

    Sign convention (matches ``compute_feasible_power``):
        positive = discharge (inject to grid)
        negative = charge (draw from grid)

    Args:
        soc: Current SOC, shape (n_batteries,).
        desired_power: Desired grid-side power [MW], shape (n_batteries,).
        power_max: Rated power limit [MW], shape (n_batteries,).
        capacity: Energy capacity [MWh], shape (n_batteries,).
        soc_min: Minimum SOC, shape (n_batteries,).
        soc_max: Maximum SOC, shape (n_batteries,).
        eta_charge: One-way charging efficiency, shape (n_batteries,).
        eta_discharge: One-way discharging efficiency, shape (n_batteries,).
        dt: Time step duration [hours] (scalar).

    Returns:
        Feasible power [MW], shape (n_batteries,).
    """
    max_discharge = jnp.clip(
        (soc - soc_min) * capacity * eta_discharge / dt,
        0.0, power_max,
    )
    max_charge = jnp.clip(
        (soc_max - soc) * capacity / (eta_charge * dt),
        0.0, power_max,
    )
    return jnp.clip(desired_power, -max_charge, max_discharge)


def update_soc_batch(
    soc: chex.Array,
    power: chex.Array,
    capacity: chex.Array,
    eta_charge: chex.Array,
    eta_discharge: chex.Array,
    soc_min: chex.Array,
    soc_max: chex.Array,
    dt: float,
) -> chex.Array:
    """Vectorized SOC update for a batch of batteries.

    All array arguments have shape ``(n_batteries,)``.  Safe for ``n_batteries=0``.

    Args:
        soc: Current SOC, shape (n_batteries,).
        power: Actual grid-side power [MW] (feasible), shape (n_batteries,).
        capacity: Energy capacity [MWh], shape (n_batteries,).
        eta_charge: One-way charging efficiency, shape (n_batteries,).
        eta_discharge: One-way discharging efficiency, shape (n_batteries,).
        soc_min: Minimum SOC, shape (n_batteries,).
        soc_max: Maximum SOC, shape (n_batteries,).
        dt: Time step duration [hours] (scalar).

    Returns:
        Updated SOC, shape (n_batteries,).
    """
    delta_discharge = -power * dt / (eta_discharge * capacity)
    delta_charge = -power * dt * eta_charge / capacity
    delta = jnp.where(power >= 0, delta_discharge, delta_charge)
    return jnp.clip(soc + delta, soc_min, soc_max)
