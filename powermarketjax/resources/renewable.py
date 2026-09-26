# Vendored from PowerZooJax. Do not edit to track upstream;
# this file is now this repository's code.
# Source        : powerzoojax/envs/resource/renewable.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# NOT verbatim. Beyond the package rename (powerzoojax -> powermarketjax, in
# both import statements and path strings): docstrings and comments were
# added on 2026-08-20 for API documentation. No executable line was changed.
# Upstream licence: MIT, Copyright (c) 2026 PowerZooJax Contributors.
"""Renewable energy resources (Solar / Wind) — pure-JAX implementation.

An inverter-interfaced renewable generator (PV or wind) driven by an
exogenous capacity-factor profile. Each step the available active power
is ``cap·CF(t)``; the agent's only choice on the active side is *how
much to spill* (curtailment), and optionally how much reactive power
to inject through the inverter:

    p_out = cap · CF · (1 − curtailment)            curtailment ∈ [0, 1]
    q_out = clip(q_desired, ±sqrt(S_rated² − p²))   PQ-circle, P-priority

The system is memoryless modulo the time index into the profile (no
SOC, no buffer, no ramp limit). Real-world reasons to curtail are
over-supply / negative prices, congestion management, and grid
stability — the agent has to learn when the marginal value of an extra
MW is negative.

Sign convention:
    p_out ≥ 0   (renewables only inject; never absorb)
    q_out signed, bounded by the PQ circle when Q control is enabled

Action mapping (curtailment scalar a ∈ [−1, 1]):
    +1 → 0% curtailment (full MPPT output)
     0 → 50% curtailment
    −1 → 100% curtailment (zero output)

Two public APIs share this physics:

``RenewableEnv`` — Gymnax-style single-device env. Action 1-D (curtail)
    or 2-D [curtail, q] when ``enable_q_control``. Obs 4-D (or 5-D with
    Q): [cf, p_norm, (q_norm,) sin t, cos t]. Costs returned as a
    stacked vector. ``SolarEnv`` / ``WindEnv`` only override the default
    profile (bell curve / stylised diurnal); physics is identical.

``RenewableBundle`` — SoA resource attached to a grid env. Two static
    flags choose the per-device schema:
        enable_curtailment / enable_q_control ∈ {True, False}
    giving action_dim ∈ {0, 1, 2} and obs_dim ∈ {1, 3, 4}. The
    (False, False) mode is "exogenous PV": the device follows its
    profile, agent has no control — used by behind-the-meter host envs
    where PV is purely exogenous. A separate soft flag
    ``allow_curtailment`` zeros out the curtailment action at runtime
    without changing tensor shapes (useful for curriculum / A-B sweeps).
    Costs returned as a *dict* — host env aggregates.

Control problem (CMDP):
    Reward      : 0 — define externally.
    Costs       : (curtailment, q_clip), with a unit asymmetry between
                  the two APIs that callers should be aware of:
                    RenewableEnv     cost_curtailment = cap·cf·curtail [MW]
                                     (lost-capacity proxy, no $ scaling)
                    RenewableBundle  cost_curtailment = curtail·dt·$/MWh [$]
                                     (default $/MWh = 0, so 0 unless set)
                  cost_q_clip = ‖a_q·s_rated − q_out‖ / s_rated [dimless]
    Termination : Env auto-resets at t ≥ max_steps; Bundle is host-owned.
"""

from functools import partial
from typing import Tuple, Dict, Any, Optional, Sequence

import jax
import jax.numpy as jnp
import jax.tree_util as tu
import chex
from flax import struct

from powermarketjax.resources.env_base import Environment, denormalize_action, stack_costs
from powermarketjax.spaces import Box
from powermarketjax.resources.base import (
    ResourceState, ResourceParams, time_features,
)


# State / Params
@struct.dataclass
class RenewableState(ResourceState):
    """Renewable resource state.

    Attributes:
        capacity_factor: Current capacity factor [0, 1] before curtailment.
        done: Episode termination flag.
    """
    capacity_factor: chex.Array   # float32 scalar [0, 1]
    done: chex.Array              # bool scalar


@struct.dataclass
class RenewableParams(ResourceParams):
    """Renewable resource parameters.

    Attributes:
        capacity_mw: Installed nameplate capacity [MW].
        profiles: Pre-loaded capacity factor time series, shape (n_steps,), normalized [0, 1].
        allow_curtailment: Whether curtailment action is honored (bool-like int).
        enable_q_control: If True, action is 2-D [curtailment_norm, q_norm] and obs
            includes q_norm.  PQ circle constraint is applied (P-priority).
        s_rated_mva: Inverter apparent power rating [MVA].  Only used when
            ``enable_q_control=True``.  Defaults to ``capacity_mw``.

    Scalar fields are ``pytree_node=False`` — static under JIT.
    ``profiles`` remains a traced pytree leaf (array data).
    """
    capacity_mw: float = struct.field(pytree_node=False, default=100.0)
    profiles: chex.Array = struct.field(default_factory=lambda: jnp.full((48,), 0.5, dtype=jnp.float32))
    allow_curtailment: int = struct.field(pytree_node=False, default=1)
    enable_q_control: bool = struct.field(pytree_node=False, default=False)
    s_rated_mva: float = struct.field(pytree_node=False, default=100.0)


# Environment Class
class RenewableEnv(Environment):
    """Renewable (Solar/Wind) environment.

    Action: scalar in [-1, 1], mapped internally to curtailment [0, 1].
    Observation: 4-D [capacity_factor, p_norm, time_sin, time_cos].
    """

    # RL Interface Methods
    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: RenewableParams
    ) -> Tuple[chex.Array, RenewableState]:
        """Start a fresh episode at ``profiles[0]``, with P = Q = 0 MW.

        Deterministic — ``key`` is accepted for the ``Environment`` contract
        but never drawn from; the capacity-factor profile is exogenous and
        fixed in ``params``.  ``time_step`` starts at 0, which is also the
        profile index ``step`` will read on the first call.
        """
        state = RenewableState(
            current_p_mw=jnp.float32(0.0),
            current_q_mvar=jnp.float32(0.0),
            time_step=jnp.int32(0),
            capacity_factor=params.profiles[0],
            done=jnp.bool_(False),
        )
        obs = self._get_obs(state, params)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: RenewableState,
        action: chex.Array,
        params: RenewableParams,
    ) -> Tuple[chex.Array, RenewableState, chex.Array, chex.Array, chex.Array, Dict[str, Any]]:
        """Spill part of the available output for one period, and optionally set Q.

        Args:
            key: Consumed only to derive the auto-reset key; the transition is
                deterministic given ``params.profiles``.
            action: The curtailment scalar alone, or ``[a_curtail, a_q]`` when
                ``params.enable_q_control``.  Both in [-1, 1]; see
                ``action_space`` for what the two ends mean.

        Returns:
            ``(obs, state, reward, costs, done, info)``.  ``reward`` is always
            0.0 — resource envs are CMDP-only and the task reward comes from
            the host env or a wrapper.  ``costs`` is
            ``(cost_curtailment, cost_q_clip)`` in the order of
            ``constraint_names``; ``cost_curtailment`` is in MW here rather
            than $, unlike ``RenewableBundle`` (module docstring).
        """

        # Parse action: P-only or P+Q depending on enable_q_control
        if params.enable_q_control:
            act = jnp.clip(jnp.asarray(action, dtype=jnp.float32).reshape((2,)), -1.0, 1.0)
            a_curtail = act[0]
            a_q = act[1]
        else:
            a_curtail = jnp.clip(jnp.float32(action).reshape(()), -1.0, 1.0)

        # Map curtailment action [-1, 1] to curtailment fraction [0, 1]
        # Convention: +1 = no curtailment (full output), -1 = full curtailment
        curtailment = (1.0 - a_curtail) / 2.0

        # Honor allow_curtailment flag (0 = disabled, 1 = enabled)
        curtailment = jnp.where(params.allow_curtailment, curtailment, 0.0)

        # Look up capacity factor from profiles
        n_steps = params.profiles.shape[0]
        idx = state.time_step % n_steps
        cf = params.profiles[idx]

        # Compute active power output
        output_p = params.capacity_mw * cf * (1.0 - curtailment)

        # Compute reactive power: PQ circle constraint, P-priority (matches BatteryBundle)
        # The inner maximum keeps the radicand non-negative: when the active
        # output already exceeds the inverter rating (s_rated_mva may be set
        # below capacity_mw) there is no reactive headroom, and q_max is 0
        # rather than NaN.
        if params.enable_q_control:
            q_max = jnp.sqrt(jnp.maximum(params.s_rated_mva ** 2 - output_p ** 2, 0.0))
            output_q = jnp.clip(a_q * params.s_rated_mva, -q_max, q_max)
        else:
            output_q = jnp.float32(0.0)

        new_time = state.time_step + 1
        done = new_time >= params.max_steps

        new_state = RenewableState(
            current_p_mw=output_p,
            current_q_mvar=output_q,
            time_step=new_time,
            capacity_factor=cf,
            done=done,
        )

        # Auto-reset: when done, swap in fresh reset state
        # The split is the PRNG contract of ``Environment.step``: the reset
        # draw must not reuse the key that drove the transition, or the new
        # episode's initial state correlates with the last step's randomness.
        # ``jnp.where`` rather than ``lax.cond`` — under vmap both branches run
        # regardless, and both are cheap here.
        _, k_reset = jax.random.split(key)
        _, reset_state = self.reset(k_reset, params)
        final_state = tu.tree_map(
            lambda n, r: jnp.where(done, r, n), new_state, reset_state)
        final_obs = self._get_obs(final_state, params)

        reward = jnp.float32(0.0)
        # Spilled active power [MW], not $ — no price is applied on this side.
        cost_curtailment = params.capacity_mw * cf * curtailment
        if params.enable_q_control:
            cost_q_clip = jnp.abs(a_q * params.s_rated_mva - output_q) / jnp.maximum(
                params.s_rated_mva, 1e-6
            )
        else:
            cost_q_clip = jnp.float32(0.0)
        costs = stack_costs(cost_curtailment, cost_q_clip)
        info = {
            "cost_curtailment": cost_curtailment,
            "cost_q_clip": cost_q_clip,
            "cost_sum": jnp.sum(costs),
        }
        return final_obs, final_state, reward, costs, done, info

    # Spaces & Observation
    def _get_obs(self, state: RenewableState, params: RenewableParams) -> chex.Array:
        """Observation vector.

        P-only (enable_q_control=False): 4-D [capacity_factor, p_norm, time_sin, time_cos].
        P+Q   (enable_q_control=True):  5-D [capacity_factor, p_norm, q_norm, time_sin, time_cos].
        """
        p_norm = state.current_p_mw / jnp.maximum(params.capacity_mw, 1e-6)
        t_sin, t_cos = time_features(state.time_step, params.steps_per_day)
        if params.enable_q_control:
            q_norm = state.current_q_mvar / jnp.maximum(params.s_rated_mva, 1e-6)
            return jnp.stack([state.capacity_factor, p_norm, q_norm, t_sin, t_cos])
        return jnp.stack([state.capacity_factor, p_norm, t_sin, t_cos])

    def observation_space(self, params: RenewableParams) -> Box:
        """Box over the vector built by ``_get_obs`` — 4-D, or 5-D with Q control.

        In order: ``capacity_factor`` ∈ [0, 1], the profile value *before*
        curtailment; ``p_norm`` ∈ [0, 1], active output over ``capacity_mw``
        (no negative side — renewables only inject); ``q_norm`` ∈ [-1, 1] over
        ``s_rated_mva``, present only when ``enable_q_control``; and
        ``time_sin`` / ``time_cos`` ∈ [-1, 1].  Every channel is
        dimensionless.
        """
        if params.enable_q_control:
            low = jnp.array([0.0, 0.0, -1.0, -1.0, -1.0], dtype=jnp.float32)
            high = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
            return Box(low=low, high=high, shape=(5,), dtype=jnp.float32)
        low = jnp.array([0.0, 0.0, -1.0, -1.0], dtype=jnp.float32)
        high = jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
        return Box(low=low, high=high, shape=(4,), dtype=jnp.float32)

    def action_space(self, params: RenewableParams) -> Box:
        """Box([-1, 1]) — curtailment, plus a reactive scalar when enabled.

        Slot 0 commands curtailment through ``curtailment = (1 - a) / 2``:
        ``+1`` spills nothing (full MPPT output), ``0`` spills half, ``-1``
        spills everything.  Slot 1 exists only when
        ``params.enable_q_control``; it is reactive power in units of
        ``s_rated_mva`` (positive = injecting VArs), clipped afterwards to the
        P-priority PQ circle.
        """
        if params.enable_q_control:
            return Box(
                low=jnp.array([-1.0, -1.0], dtype=jnp.float32),
                high=jnp.array([1.0, 1.0], dtype=jnp.float32),
                shape=(2,), dtype=jnp.float32,
            )
        return Box(
            low=jnp.array([-1.0], dtype=jnp.float32),
            high=jnp.array([1.0], dtype=jnp.float32),
            shape=(1,), dtype=jnp.float32,
        )

    # Status & Diagnostics
    @property
    def name(self) -> str:
        """Registration name, pinned to a literal rather than the class name."""
        return "RenewableEnv"

    def default_params(self) -> RenewableParams:
        """100 MW device on a flat capacity factor of 0.5 for 48 steps."""
        return RenewableParams()

    def constraint_names(self, params: RenewableParams) -> tuple[str, ...]:
        """Names of the ``costs`` entries, in the order ``step`` stacks them.

        ``curtailment`` — ``capacity_mw x cf x curtail``, the active power
        spilled this period, in MW; a lost-capacity proxy carrying no price
        (``RenewableBundle`` meters the same decision in $ instead).
        ``q_clip`` — ``|a_q x s_rated - q_out| / s_rated``, dimensionless, how
        far the PQ circle cut the reactive command back.  Both are ``costs``
        entries: neither is money anyone pays, and neither enters ``reward``.
        The tuple has length 2 whatever ``enable_q_control`` is; with Q
        control off the second entry is identically 0.
        """
        return ("curtailment", "q_clip")


class SolarEnv(RenewableEnv):
    """Solar PV — subclass with bell-curve default profile."""

    @property
    def name(self) -> str:
        """Registration name, pinned to a literal rather than the class name."""
        return "SolarEnv"

    def default_params(self) -> RenewableParams:
        """100 MW PV on a 48-step half-hourly bell curve, dark before 06:00 and after 18:00."""

        # Bell-curve solar profile: peak at noon (step 24 of 48), zero at night
        steps = jnp.arange(48, dtype=jnp.float32)
        hour = steps / 48.0 * 24.0
        profiles = jnp.clip(jnp.sin(jnp.pi * (hour - 6.0) / 12.0), 0.0, 1.0)
        return RenewableParams(capacity_mw=100.0, profiles=profiles)


class WindEnv(RenewableEnv):
    """Wind turbine — subclass with variable default profile."""

    @property
    def name(self) -> str:
        """Registration name, pinned to a literal rather than the class name."""
        return "WindEnv"

    def default_params(self) -> RenewableParams:
        """100 MW turbine on a deterministic two-harmonic profile about cf = 0.3."""

        # Wind: moderate, somewhat random-looking but deterministic profile
        steps = jnp.arange(48, dtype=jnp.float32)
        hour = steps / 48.0 * 24.0
        profiles = 0.3 + 0.2 * jnp.sin(2.0 * jnp.pi * hour / 24.0) + 0.1 * jnp.cos(4.0 * jnp.pi * hour / 24.0)
        profiles = jnp.clip(profiles, 0.0, 1.0)
        return RenewableParams(capacity_mw=100.0, profiles=profiles)

# RenewableBundle — grid-attachable SoA bundle.
# Mirrors the BatteryBundle / FlexLoadBundle interface so a grid-level host
# env can treat all DER bundles uniformly.  Implements active curtailment +
# reactive power support on the same PQ-circle / P-priority rule used by
# BatteryBundle.


@struct.dataclass
class RenewableBundleState:
    """State for a RenewableBundle — time index + last-step outputs.

    Attributes:
        t: Internal time step (int32 scalar).  Used to index ``profiles``.
        cf: Last-step capacity factor, shape ``(n_devices,)``.
        p_mw: Last-step active output [MW], shape ``(n_devices,)``.
        q_mvar: Last-step reactive output [MVAr], shape ``(n_devices,)``.
        curtail_frac: Last-step curtailment fraction ∈ [0,1], shape ``(n_devices,)``.
    """
    t: chex.Array          # int32 scalar
    cf: chex.Array         # (n_devices,) float32
    p_mw: chex.Array       # (n_devices,) float32
    q_mvar: chex.Array     # (n_devices,) float32
    curtail_frac: chex.Array  # (n_devices,) float32


@struct.dataclass
class RenewableBundle:
    """Struct-of-arrays renewable bundle for grid env attachment.

    Groups ``N`` inverter-interfaced renewable generators (PV or wind)
    driven by pre-loaded capacity-factor profiles.  Supports active
    curtailment and reactive power support with an inverter PQ-circle
    constraint (P-priority projection, matching ``BatteryBundle``).

    Action / observation dimensions are controlled by two static flags:

    ``enable_curtailment`` (default ``True``)
        When ``True``, the action vector includes a per-device curtailment
        scalar in ``[-1, 1]`` (``+1`` = no curtailment, ``-1`` = full
        curtailment).  The observation vector includes a ``curtail_norm``
        channel.
    ``enable_q_control`` (default ``True``)
        When ``True``, the action vector includes a per-device reactive
        scalar in ``[-1, 1]`` (scaled by ``s_rated`` then PQ-circle clipped
        with P-priority).  The observation vector includes a ``q_norm``
        channel.

    Resulting per-device dimensions:

    +--------------------+----------------+-------------+-------------------+
    | (curtail, Q)       | action_dim     | obs_dim     | obs layout         |
    +--------------------+----------------+-------------+-------------------+
    | (True,  True)      | 2 (default)    | 4 (default) | cf, p_norm, q, c  |
    | (True,  False)     | 1              | 3           | cf, p_norm, c     |
    | (False, True)      | 1              | 3           | cf, p_norm, q     |
    | (False, False)     | 0              | 1           | cf                |
    +--------------------+----------------+-------------+-------------------+

    The ``(False, False)`` mode is used by behind-the-meter host envs where
    PV is purely exogenous.

    Action layout (when present): ``[curtail, q]`` per device — that is,
    ``[curtail_0, q_0, curtail_1, q_1, ...]``; if only one of curtail / Q
    is enabled the corresponding scalar fills slot 0.
    """

    # Static (pytree_node=False)
    n_devices: int = struct.field(pytree_node=False)
    per_device_action_dim: int = struct.field(pytree_node=False, default=2)
    per_device_obs_dim: int = struct.field(pytree_node=False, default=4)
    dt_hours: float = struct.field(pytree_node=False, default=0.5)
    allow_curtailment: bool = struct.field(pytree_node=False, default=True)
    enable_curtailment: bool = struct.field(pytree_node=False, default=True)
    enable_q_control: bool = struct.field(pytree_node=False, default=True)

    # Traced leaves — SoA (n_devices,)
    bus_idx: chex.Array = None      # int32 (n_devices,)
    capacity_mw: chex.Array = None  # float32 (n_devices,)
    s_rated: chex.Array = None      # float32 (n_devices,) — inverter MVA rating
    profiles: chex.Array = None     # float32 (T, n_devices) — capacity factors [0,1]
    curtail_cost_per_mwh: chex.Array = None  # float32 (n_devices,)

    @property
    def action_dim(self) -> int:
        """Width of the flat action vector for the whole bundle, not per device.

        ``per_device_action_dim`` is the count of enabled controls (0, 1 or 2 —
        see the class docstring's table), so this is 0 in the exogenous
        ``(False, False)`` mode.  Device-major: device *i* owns slots
        ``[i * per_device_action_dim, (i+1) * per_device_action_dim)``.
        """
        return self.n_devices * self.per_device_action_dim

    @property
    def obs_dim(self) -> int:
        """Width of the flat ``obs_slice`` for the whole bundle, not per device.

        ``per_device_obs_dim`` is 1, 3 or 4 depending on the same two flags;
        ``observe`` lists the channels in order.  Same device-major layout as
        ``action_dim``.
        """
        return self.n_devices * self.per_device_obs_dim

    def reset(self, key: chex.PRNGKey) -> RenewableBundleState:
        """Return the initial state — t=0, outputs at profile[0]."""
        cf0 = self.profiles[0]
        zero = jnp.zeros(self.n_devices, dtype=jnp.float32)
        return RenewableBundleState(
            t=jnp.int32(0),
            cf=cf0,
            p_mw=zero,
            q_mvar=zero,
            curtail_frac=zero,
        )

    def step(
        self,
        state: RenewableBundleState,
        action: chex.Array,
        ctx: Dict[str, Any],
    ) -> Tuple[RenewableBundleState, chex.Array, chex.Array, chex.Array, Dict[str, chex.Array]]:
        """Run one step for all N devices.

        Args:
            state: Current bundle state.
            action: ``(action_dim,)`` in ``[-1, 1]``.  Layout depends on the
                static ``enable_curtailment`` / ``enable_q_control`` flags
                (see class docstring).  Empty array when both flags False.
            ctx: Grid-provided context (unused; reserved).

        Returns:
            ``(new_state, p_inject_mw, q_inject_mvar, obs_slice, cost_info)``
        """
        a = jnp.asarray(action, dtype=jnp.float32)

        # Parse action by static flags (Python-level if at trace time).
        if self.enable_curtailment and self.enable_q_control:
            act2d = jnp.clip(a.reshape(self.n_devices, 2), -1.0, 1.0)
            a_curtail = act2d[:, 0]
            a_q = act2d[:, 1]
        elif self.enable_curtailment:
            a_curtail = jnp.clip(a.reshape(self.n_devices), -1.0, 1.0)
            a_q = jnp.zeros((self.n_devices,), dtype=jnp.float32)
        elif self.enable_q_control:
            a_curtail = jnp.ones((self.n_devices,), dtype=jnp.float32)
            a_q = jnp.clip(a.reshape(self.n_devices), -1.0, 1.0)
        else:
            a_curtail = jnp.ones((self.n_devices,), dtype=jnp.float32)
            a_q = jnp.zeros((self.n_devices,), dtype=jnp.float32)

        # Map [-1, 1] → curtailment fraction [0, 1] (+1 = no curtail, -1 = full)
        curtail = (1.0 - a_curtail) / 2.0
        curtail = jnp.where(self.allow_curtailment, curtail, jnp.zeros_like(curtail))

        # Look up capacity factor at current device time step
        T = self.profiles.shape[0]
        t_idx = state.t % T
        cf = self.profiles[t_idx]                    # (n_devices,)

        # Active output (MW) ≥ 0
        p_max = self.capacity_mw * cf                 # (n_devices,)
        p_out = p_max * (1.0 - curtail)               # (n_devices,)

        # Reactive headroom (PQ circle, P-priority)
        q_max = jnp.sqrt(jnp.maximum(self.s_rated ** 2 - p_out ** 2, 0.0))
        q_desired = a_q * self.s_rated
        q_out = jnp.clip(q_desired, -q_max, q_max)

        new_state = RenewableBundleState(
            t=state.t + 1,
            cf=cf,
            p_mw=p_out,
            q_mvar=q_out,
            curtail_frac=curtail,
        )
        obs_slice = self.observe(new_state, ctx)

        # Cost signals
        safe_cap = jnp.maximum(self.capacity_mw, 1e-6)
        safe_s = jnp.maximum(self.s_rated, 1e-6)

        # Curtailed MWh (opportunity cost)
        # MW x h x [$/MWh] = $ per device this period; unlike ``RenewableEnv``,
        # which reports the spilled MW itself and attaches no price.
        curtail_mw = p_max - p_out                    # (n_devices,)
        cost_curtail = curtail_mw * self.dt_hours * self.curtail_cost_per_mwh

        # Reactive clipping cost — how far q_desired was reduced (normalised)
        cost_q_clip = jnp.sum(jnp.abs(q_desired - q_out) / safe_s)

        # cost_sum mixes a real $ cost (cost_curtail) with a dimensionless
        # clip magnitude (cost_q_clip); read cost_curtailment / cost_q_clip
        # from the dict below to keep the two channels separate.
        cost_sum = jnp.sum(cost_curtail) + cost_q_clip
        cost_info: Dict[str, chex.Array] = {
            "cost_sum": cost_sum,
            "cost": cost_sum,  # deprecated compatibility alias
            "cost_curtailment": jnp.sum(cost_curtail),
            "cost_q_clip": cost_q_clip,
        }
        return new_state, p_out, q_out, obs_slice, cost_info

    def observe(
        self,
        state: RenewableBundleState,
        ctx: Dict[str, Any],
    ) -> chex.Array:
        """Flat obs slice of shape ``(obs_dim,)``.

        Per-device channels included (in order):
          ``cf`` always; ``p_norm`` if any control enabled; ``q_norm`` if
          ``enable_q_control``; ``curtail_norm`` if ``enable_curtailment``.
        See class docstring for the resulting dimensions.
        """
        cf_part = jnp.clip(state.cf, 0.0, 1.0)
        if not (self.enable_curtailment or self.enable_q_control):

            # No-control mode: cf only (per-device shape becomes scalar).
            return cf_part.reshape(-1)

        safe_cap = jnp.maximum(self.capacity_mw, 1e-6)
        p_norm = jnp.clip(state.p_mw / safe_cap, 0.0, 1.0)
        parts = [cf_part, p_norm]
        if self.enable_q_control:
            safe_s = jnp.maximum(self.s_rated, 1e-6)
            parts.append(jnp.clip(state.q_mvar / safe_s, -1.0, 1.0))
        if self.enable_curtailment:
            parts.append(jnp.clip(state.curtail_frac, 0.0, 1.0))
        per_device = jnp.stack(parts, axis=1)
        return per_device.reshape(-1)


def _default_pv_profile(T: int) -> jnp.ndarray:
    """Bell-curve PV capacity factor profile of length ``T`` (peak at noon)."""
    steps = jnp.arange(T, dtype=jnp.float32)
    hour = steps / jnp.float32(T) * 24.0
    shape = jnp.clip(jnp.sin(jnp.pi * (hour - 6.0) / 12.0), 0.0, 1.0)
    return shape


def make_renewable_bundle(
    case=None,
    bus_ids: Optional[Sequence[int]] = None,
    *,
    n_devices: Optional[int] = None,
    capacity_mw=0.20,
    s_rated_mva: Optional[float] = None,
    profiles: Optional[chex.Array] = None,
    max_steps: int = 48,
    allow_curtailment: bool = True,
    enable_curtailment: bool = True,
    enable_q_control: bool = True,
    curtail_cost_per_mwh=0.0,
    dt_hours: float = 0.5,
) -> RenewableBundle:
    """Create a ``RenewableBundle``.

    Two construction modes:

    Grid mode (``case`` and ``bus_ids`` both provided):
        ``bus_idx`` resolved through ``case.node_ids``;
        ``n_devices = len(bus_ids)``.  Used by grid-level host envs with
        an explicit topology.

    Busless mode (``case=None, bus_ids=None``):
        For behind-the-meter host envs with no explicit topology.
        ``n_devices`` must be supplied (default 1).
        ``bus_idx = arange(n_devices)`` and the host env sums injections.

    Args:
        case: ``CaseData`` instance with a ``node_ids`` attribute (optional).
        bus_ids: External node IDs (optional).
        n_devices: Number of devices (busless mode; ignored if ``bus_ids``).
        capacity_mw: Nameplate capacity [MW].  Scalar or length-``n`` sequence.
        s_rated_mva: Inverter apparent power rating [MVA].  Defaults to
            ``1.1 * capacity_mw`` to leave reactive headroom at full output.
        profiles: Capacity-factor time series, shape ``(T, n)`` or ``(T,)``.
            If ``None``, a synthetic bell-curve PV profile of length
            ``max_steps`` is broadcast to every device.
        max_steps: Used only when ``profiles is None`` to build the default.
        allow_curtailment: Soft flag.  If False, the curtailment action (if
            any) is ignored and every device runs at its MPPT point.
        enable_curtailment: Hard flag.  When False, the curtailment scalar
            is removed from the action / observation vectors entirely.
        enable_q_control: Hard flag.  When False, the reactive-power scalar
            is removed from the action / observation vectors entirely.
        curtail_cost_per_mwh: Opportunity cost for curtailed energy ``[$/MWh]``.
        dt_hours: Time-step duration.

    Returns:
        ``RenewableBundle`` with ``(n_devices,)`` SoA arrays.
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

    def _broadcast_float(val):
        """Tile a scalar to ``(n,)`` float32, or pass a length-``n`` sequence through."""
        if hasattr(val, "__len__") and not isinstance(val, str):
            arr = np_cpu.asarray(val, dtype=np_cpu.float32)
            if arr.shape[0] != n:
                raise ValueError(f"Expected length {n}, got {arr.shape[0]}")
            return jnp.asarray(arr)
        return jnp.full((n,), float(val), dtype=jnp.float32)

    capacity_arr = _broadcast_float(capacity_mw)
    if s_rated_mva is None:
        s_rated_arr = capacity_arr * 1.1
    else:
        s_rated_arr = _broadcast_float(s_rated_mva)

    if profiles is None:
        shape = _default_pv_profile(max_steps)                  # (T,)
        profiles_arr = jnp.broadcast_to(shape[:, None], (int(max_steps), n))
    else:
        profiles_arr = jnp.asarray(profiles, dtype=jnp.float32)
        if profiles_arr.ndim == 1:
            profiles_arr = jnp.broadcast_to(profiles_arr[:, None], (profiles_arr.shape[0], n))
        elif profiles_arr.ndim != 2 or profiles_arr.shape[1] != n:
            raise ValueError(
                f"profiles must have shape (T,) or (T, {n}), got {profiles_arr.shape}"
            )

    # Compute action/obs dims from the new hard flags
    a_dim = int(enable_curtailment) + int(enable_q_control)
    if enable_curtailment or enable_q_control:

        # cf + p_norm + (q_norm) + (curtail_norm)
        o_dim = 2 + int(enable_q_control) + int(enable_curtailment)
    else:
        o_dim = 1  # cf only (busless / exogenous PV mode)

    return RenewableBundle(
        n_devices=n,
        per_device_action_dim=a_dim,
        per_device_obs_dim=o_dim,
        dt_hours=float(dt_hours),
        allow_curtailment=bool(allow_curtailment),
        enable_curtailment=bool(enable_curtailment),
        enable_q_control=bool(enable_q_control),
        bus_idx=bus_idx_arr,
        capacity_mw=capacity_arr,
        s_rated=s_rated_arr,
        profiles=profiles_arr,
        curtail_cost_per_mwh=_broadcast_float(curtail_cost_per_mwh),
    )
