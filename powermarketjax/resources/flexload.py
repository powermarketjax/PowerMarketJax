# Vendored from PowerZooJax. Do not edit to track upstream;
# this file is now this repository's code.
# Source        : powerzoojax/envs/resource/flexload.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# NOT verbatim. Beyond the package rename (powerzoojax -> powermarketjax, in
# both import statements and path strings): docstrings and comments were
# added on 2026-08-20 for API documentation. No executable line was changed.
# Upstream licence: MIT, Copyright (c) 2026 PowerZooJax Contributors.
"""FlexLoad — Flexible Load Demand Response — pure-JAX implementation.

A single-bus demand-response aggregator that, at every step, splits its
flexibility between two physically distinct modes:

  - Curtail (a0): drop load now, irreversibly. Energy is lost and the
    consumer eats a discomfort cost.
  - Shift  (a1): defer load to the future. Energy is conserved — every
    MW pushed out *must* return as shift_in, spread evenly over the
    next shift_horizon steps, and pays a holding cost while it sits.

The deferred-release buffer is the only stateful dynamic, and the only
thing that couples decisions across time; shift_in is not a control
input, it is the forced consequence of past shift_out calls.

Control problem (CMDP):
    Observation : (curtail, shift_out, shift_in, buf_fill, buf_energy,
                   sin t, cos t, price)  — buffer state plus exogenous
                   time/price signals.
    Action      : (a0, a1) ∈ [0, 1]², scaled by curtail_cap_mw and
                   shift_cap_mw respectively.
    Reward      : 0.  FlexLoadEnv emits no scalar reward; supply it
                   externally (market revenue, grid-level objective,
                   outer-loop dispatch signal, ...).
    Costs       : (curtailment, shift_discomfort, simultaneous_activation),
                   stacked as a vector for constrained-RL training.
    Termination : t ≥ max_steps (auto-resets to a fresh episode).

Dynamics (time-coupling lives entirely in deferred_buffer):
    on shift_out :  buf[t : t+H] += a1·shift_cap_mw / H
    on every step:  shift_in_t = buf[head_t]; 
                    buf[head_t] := 0
                    head ← (head + 1) mod max_buffer
    grid sees    :  dP_t = curtail_t + shift_out_t − shift_in_t
                    (generator-positive: dP > 0 ⇒ net load reduction)

Step execution order:
    1. Pop head of deferred_buffer → shift_in (forced release)
    2. Scale action → (curtail_mw, shift_out_mw); clip to capacities
    3. Push shift_out into the next shift_horizon buffer slots
    4. dP = curtail + shift_out − shift_in
    5. t ← t + 1; auto-reset on done

Notes:
    - Ring buffer length is fixed at DEFAULT_MAX_BUFFER (compile-time
      constant required by lax.scan); shift_horizon ≤ DEFAULT_MAX_BUFFER.
    - LMP is not stored in params — pass it to step(lmp=...) each call,
      from a market clearing result or an exogenous RTP/ToU profile.
    - complementarity_penalty discourages co-activating both modes;
      real DR programs treat curtail and shift as alternative responses
      to a single dispatch signal, not as additive ones.
"""

from functools import partial
from typing import Tuple, Dict, Any

import jax
import jax.numpy as jnp
import jax.tree_util as tu
import chex
from flax import struct

from powermarketjax.resources.env_base import Environment, stack_costs
from powermarketjax.spaces import Box
from powermarketjax.resources.base import (
    ResourceState, ResourceParams, time_features,
)

DEFAULT_MAX_BUFFER = 64  # Default ring buffer capacity


# State / Params
@struct.dataclass
class FlexLoadState(ResourceState):
    """FlexLoad state.

    Attributes:
        curtailed_mw: Current step curtailment [MW].
        shift_out_mw: Current step shift-out amount [MW].
        shift_in_mw: Current step shift-in (released deferred demand) [MW].
        deferred_buffer: Ring buffer of per-step deferred release amounts.
        buffer_head: Read pointer into deferred_buffer.
        buffer_size: Number of valid entries in buffer.
        current_lmp: Current locational marginal price [$/MWh].
        done: Episode termination flag.
    """
    curtailed_mw: chex.Array     # float32 scalar
    shift_out_mw: chex.Array     # float32 scalar
    shift_in_mw: chex.Array      # float32 scalar
    deferred_buffer: chex.Array  # (max_buffer_size,) float32
    buffer_head: chex.Array      # int32 scalar (read pointer)
    buffer_size: chex.Array      # int32 scalar (number of valid entries)
    current_lmp: chex.Array      # float32 scalar
    done: chex.Array             # bool scalar


@struct.dataclass
class FlexLoadParams(ResourceParams):
    """FlexLoad parameters.

    Attributes:
        curtail_cap_mw: Maximum curtailment capacity [MW].
        shift_cap_mw: Maximum shift capacity per step [MW].
        shift_horizon: Number of future steps over which deferred demand is released.
        curtail_cost_per_mwh: Discomfort/compensation cost for curtailment [$/MWh].
        shift_cost_per_mwh: Holding cost rate for buffered deferred demand [$/MWh].
        complementarity_penalty: Penalty for co-activating curtailment and shift [$/MW].
        price_ref: Reference LMP for price normalization in observation [$/MWh].

    All scalar fields are ``pytree_node=False`` — static under JIT.
    Ring buffer capacity is fixed at DEFAULT_MAX_BUFFER (compile-time constant
    required by lax.scan). Do not change at runtime.

    LMP is NOT stored in params. Pass it as ``lmp`` to ``step()`` each call —
    either from a market clearing result or from an exogenous price profile.
    """
    curtail_cap_mw: float = struct.field(pytree_node=False, default=10.0)
    shift_cap_mw: float = struct.field(pytree_node=False, default=10.0)
    shift_horizon: int = struct.field(pytree_node=False, default=4)
    curtail_cost_per_mwh: float = struct.field(pytree_node=False, default=50.0)
    shift_cost_per_mwh: float = struct.field(pytree_node=False, default=10.0)
    complementarity_penalty: float = struct.field(pytree_node=False, default=100.0)
    price_ref: float = struct.field(pytree_node=False, default=100.0)


# Ring Buffer Pure Functions
def _buffer_push(
    buffer: chex.Array,
    head: chex.Array,
    size: chex.Array,
    value: chex.Array,
    max_size: int,
) -> Tuple[chex.Array, chex.Array]:
    """Push a value to the tail of the ring buffer."""
    tail = (head + size) % max_size
    buffer = buffer.at[tail].set(value)
    new_size = jnp.minimum(size + 1, max_size)
    return buffer, new_size


def _buffer_pop(
    buffer: chex.Array,
    head: chex.Array,
    size: chex.Array,
    max_size: int,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """Pop a value from the head of the ring buffer.

    Returns (value, buffer, new_head, new_size). Returns 0.0 if empty.
    """
    is_empty = size <= 0
    value = jnp.where(is_empty, 0.0, buffer[head])

    # Zero out popped slot
    buffer = jnp.where(is_empty, buffer, buffer.at[head].set(0.0))
    new_head = jnp.where(is_empty, head, (head + 1) % max_size)
    new_size = jnp.where(is_empty, size, size - 1)
    return value, buffer, new_head, new_size


def _add_to_buffer(
    buffer: chex.Array,
    head: chex.Array,
    size: chex.Array,
    shift_mw: chex.Array,
    shift_horizon: chex.Array,
    max_size: int,
) -> Tuple[chex.Array, chex.Array]:
    """Spread deferred demand evenly over shift_horizon future steps.

    Physical semantics: "add per_step MW to each of the next shift_horizon
    time slots starting from head". This is positional superimposition —
    concurrent shifts overlap correctly on the same future slots.

    Uses fixed-length scan over max_size with masking (vmap/JIT safe).

    Slot ``(head + k) % max_size`` holds the MW to be released ``k`` pops from
    now, so passing the head *after* this step's pop makes offset 0 the next
    step's release.  ``new_size`` is not a count of non-zero slots — concurrent
    shifts share slots — but a bound on how far ahead of ``head`` the filled
    region reaches, which is what ``_buffer_pop`` and ``_buffer_energy`` read
    it as.
    """
    per_step = shift_mw / jnp.maximum(shift_horizon, 1)

    def scan_fn(buf, i):
        """Add ``per_step`` MW to offset ``i`` from head, for ``i < shift_horizon``.

        The carry is the whole ring buffer; ``i`` sweeps all ``max_size``
        offsets so the traced length is a compile-time constant, and the mask
        makes the tail of that sweep a no-op.  ``.add`` rather than ``.set``
        so shifts issued on different steps superimpose on shared slots.
        """
        should_add = i < shift_horizon
        pos = (head + i) % max_size
        buf = jnp.where(should_add, buf.at[pos].add(per_step), buf)
        return buf, None

    buffer, _ = jax.lax.scan(scan_fn, buffer, jnp.arange(max_size))
    new_size = jnp.minimum(jnp.maximum(size, shift_horizon), max_size)
    return buffer, new_size


def _buffer_energy(buffer: chex.Array, head: chex.Array, size: chex.Array, max_size: int) -> chex.Array:
    """Sum all valid entries in the ring buffer."""
    indices = jnp.arange(max_size)
    valid = indices < size

    # Actual positions in the circular buffer
    positions = (head + indices) % max_size
    return jnp.sum(jnp.where(valid, buffer[positions], 0.0))


# Environment Class
class FlexLoadEnv(Environment):
    """Flexible Load demand-response environment.

    Action: 2-D in [0, 1] × [0, 1] (unit-scaled):
        action[0] : curtailment fraction × curtail_cap_mw
        action[1] : shift-out fraction × shift_cap_mw

    Observation: 8-D [curtail_norm, shift_out_norm, shift_in_norm,
                      buffer_fill_ratio, buffer_energy_norm,
                      time_sin, time_cos, price_norm].
    """

    # RL Interface Methods
    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: FlexLoadParams
    ) -> Tuple[chex.Array, FlexLoadState]:
        """Start a fresh episode: nothing curtailed or shifted, empty buffer.

        Deterministic — ``key`` is accepted for the ``Environment`` contract
        but never drawn from.  ``current_lmp`` starts at 0 $/MWh because the
        price is an argument of ``step`` rather than a parameter, so the first
        observation reports 0 until a step supplies one.
        """
        state = FlexLoadState(
            current_p_mw=jnp.float32(0.0),
            current_q_mvar=jnp.float32(0.0),
            time_step=jnp.int32(0),
            curtailed_mw=jnp.float32(0.0),
            shift_out_mw=jnp.float32(0.0),
            shift_in_mw=jnp.float32(0.0),
            deferred_buffer=jnp.zeros(DEFAULT_MAX_BUFFER, dtype=jnp.float32),
            buffer_head=jnp.int32(0),
            buffer_size=jnp.int32(0),
            current_lmp=jnp.float32(0.0),
            done=jnp.bool_(False),
        )
        obs = self._get_obs(state, params)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: FlexLoadState,
        action: chex.Array,
        params: FlexLoadParams,
        lmp: chex.Array = 0.0,
    ) -> Tuple[chex.Array, FlexLoadState, chex.Array, chex.Array, chex.Array, Dict[str, Any]]:
        """Run one environment step.

        Args:
            lmp: Locational marginal price for this step [$/MWh]. Pass the
                market-cleared price (market participation) or the current value
                from an exogenous RTP/ToU profile. Defaults to 0.0.
        """
        lmp = jnp.asarray(lmp, dtype=jnp.float32)

        # Step 1: Release deferred demand from previous steps
        released, buffer, new_head, new_size = _buffer_pop(
            state.deferred_buffer, state.buffer_head, state.buffer_size,
            DEFAULT_MAX_BUFFER,
        )

        # Step 2: Parse 2-D action → physical MW (unit-scaled: [0,1] × capacity)
        action = jnp.asarray(action, dtype=jnp.float32).reshape(2)
        a0 = jnp.clip(action[0], 0.0, 1.0)
        a1 = jnp.clip(action[1], 0.0, 1.0)
        curtail_mw = a0 * params.curtail_cap_mw
        shift_out_mw = a1 * params.shift_cap_mw

        # Step 3: clip to capacity (redundant after scaling, but safe)
        curtail_mw = jnp.clip(curtail_mw, 0.0, params.curtail_cap_mw)
        shift_out_mw = jnp.clip(shift_out_mw, 0.0, params.shift_cap_mw)

        # Step 4: Buffer shift-out demand for future release
        # The push starts at ``new_head``, the pointer *after* this step's pop,
        # so offset k carries the MW released k+1 steps from now and the first
        # instalment returns at the very next step.  Energy is conserved: every
        # MW pushed here comes back as ``shift_in``, unlike curtailment.
        has_shift = shift_out_mw > 0.0
        buffer_shifted, size_shifted = _add_to_buffer(
            buffer, new_head, new_size, shift_out_mw,
            params.shift_horizon, DEFAULT_MAX_BUFFER,
        )
        buffer_after = jnp.where(has_shift, buffer_shifted, buffer)
        size_after = jnp.where(has_shift, size_shifted, new_size)

        # Step 5: Net injection (generator-positive convention)
        current_p = curtail_mw + shift_out_mw - released

        new_time = state.time_step + 1
        done = new_time >= params.max_steps

        new_state = FlexLoadState(
            current_p_mw=current_p,
            current_q_mvar=jnp.float32(0.0),
            time_step=new_time,
            curtailed_mw=curtail_mw,
            shift_out_mw=shift_out_mw,
            shift_in_mw=released,
            deferred_buffer=buffer_after,
            buffer_head=new_head,
            buffer_size=size_after,
            current_lmp=lmp,
            done=done,
        )

        # Auto-reset: when done, swap in fresh reset state
        # The split is the PRNG contract of ``Environment.step``: the reset
        # draw must not reuse the key that drove the transition, or the new
        # episode's initial state correlates with the last step's randomness.
        # The buffer is part of the state, so this also discards any energy
        # still owed at the episode boundary.
        _, k_reset = jax.random.split(key)
        _, reset_state = self.reset(k_reset, params)
        final_state = tu.tree_map(
            lambda n, r: jnp.where(done, r, n), new_state, reset_state)
        final_obs = self._get_obs(final_state, params)

        reward = jnp.float32(0.0)

        # CMDP cost signals
        # Both energy terms are MW x h x [$/MWh] = $ for this step.  The
        # holding charge is levied on the *whole outstanding buffer* every
        # step, so one deferral is charged again on every step until the
        # buffer drains, on a balance that shrinks as slots are released — it
        # is a rent on the obligation, not a one-off fee.
        dt_h = params.delta_t_hours
        cost_curtailment = curtail_mw * dt_h * params.curtail_cost_per_mwh
        buf_total = _buffer_energy(buffer_after, new_head, size_after, DEFAULT_MAX_BUFFER)
        cost_shift_discomfort = buf_total * dt_h * params.shift_cost_per_mwh
        cost_simultaneous = jnp.minimum(curtail_mw, shift_out_mw) * params.complementarity_penalty
        costs = stack_costs(
            cost_curtailment, cost_shift_discomfort, cost_simultaneous
        )

        info = {
            "cost_curtailment": cost_curtailment,
            "cost_shift_discomfort": cost_shift_discomfort,
            "cost_simultaneous": cost_simultaneous,
            "cost_sum": jnp.sum(costs),
            "current_lmp": lmp,
        }
        return final_obs, final_state, reward, costs, done, info

    # Spaces & Observation
    def _get_obs(self, state: FlexLoadState, params: FlexLoadParams) -> chex.Array:
        """8-D observation: [curtail_norm, shift_out_norm, shift_in_norm,
        buffer_fill_ratio, buffer_energy_norm, time_sin, time_cos, price_norm]."""
        c_norm = state.curtailed_mw / jnp.maximum(params.curtail_cap_mw, 1e-6)
        s_out_norm = state.shift_out_mw / jnp.maximum(params.shift_cap_mw, 1e-6)
        s_in_norm = state.shift_in_mw / jnp.maximum(params.shift_cap_mw, 1e-6)

        buf_fill = state.buffer_size / jnp.maximum(params.shift_horizon, 1)

        buf_energy = _buffer_energy(
            state.deferred_buffer, state.buffer_head, state.buffer_size,
            DEFAULT_MAX_BUFFER,
        )
        buf_energy_norm = buf_energy / jnp.maximum(
            params.shift_cap_mw * params.shift_horizon, 1e-6)

        t_sin, t_cos = time_features(state.time_step, params.steps_per_day)

        price_norm = jnp.clip(
            state.current_lmp / jnp.maximum(params.price_ref, 1e-6), -1.0, 2.0)

        return jnp.stack([
            jnp.clip(c_norm, 0.0, 1.0),
            jnp.clip(s_out_norm, 0.0, 1.0),
            jnp.clip(s_in_norm, 0.0, 1.0),
            jnp.clip(buf_fill, 0.0, 1.0),
            jnp.clip(buf_energy_norm, 0.0, 1.0),
            t_sin, t_cos,
            price_norm,
        ])

    def observation_space(self, params: FlexLoadParams) -> Box:
        """Box over the 8-D vector built by ``_get_obs``.

        In order, every channel dimensionless: ``curtail_norm``,
        ``shift_out_norm``, ``shift_in_norm`` ∈ [0, 1] — the three MW
        quantities over their caps (``curtail_cap_mw`` for the first,
        ``shift_cap_mw`` for the other two); ``buffer_fill_ratio`` ∈ [0, 1],
        valid buffer entries over ``shift_horizon``; ``buffer_energy_norm``
        ∈ [0, 1], the queued per-step releases summed [MW] over
        ``shift_cap_mw x shift_horizon``; ``time_sin`` / ``time_cos``
        ∈ [-1, 1]; and ``price_norm``, the LMP over ``price_ref``, the one
        channel with asymmetric bounds — clipped to [-1, 2], so it admits a
        price up to twice reference and down to minus reference.
        """
        low = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0], dtype=jnp.float32)
        high = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0], dtype=jnp.float32)
        return Box(low=low, high=high, shape=(8,), dtype=jnp.float32)

    def action_space(self, params: FlexLoadParams) -> Box:
        """Box([0, 1]²) — the two flexibility modes as fractions of their caps.

        Slot 0 is curtail: ``a0 x curtail_cap_mw`` MW dropped now and never
        served.  Slot 1 is shift-out: ``a1 x shift_cap_mw`` MW deferred now and
        returned in full over the next ``shift_horizon`` steps.  The lower
        bound is 0, not -1 as on the generator-style devices: neither mode has
        a negative direction, and shift-*in* is not a control at all — it is
        what past shift-outs oblige.
        """
        return Box(
            low=jnp.zeros(2, dtype=jnp.float32),
            high=jnp.ones(2, dtype=jnp.float32),
            shape=(2,), dtype=jnp.float32,
        )

    # Status & Diagnostics
    @property
    def name(self) -> str:
        """Registration name, pinned to a literal rather than the class name."""
        return "FlexLoadEnv"

    def constraint_names(self, params: FlexLoadParams) -> tuple[str, ...]:
        """Names of the ``costs`` entries, in the order ``step`` stacks them.

        ``curtailment`` — ``curtail_mw x Δt x curtail_cost_per_mwh``, $ this
        step, the compensation owed for load dropped outright.
        ``shift_discomfort`` — outstanding buffer ``x Δt x
        shift_cost_per_mwh``, $ this step, a holding charge re-levied every
        step until the buffer drains.  ``simultaneous_activation`` —
        ``min(curtail_mw, shift_out_mw) x complementarity_penalty`` ([MW] x
        [$/MW]), which prices no transaction; it exists to discourage
        co-activating two modes that real DR programmes treat as alternative
        responses to one dispatch signal.  All three are ``costs`` entries and
        never enter ``reward``, whatever their units.
        """
        return ("curtailment", "shift_discomfort", "simultaneous_activation")

    def default_params(self) -> FlexLoadParams:
        """Stock aggregator: 10 MW curtailable, 10 MW shiftable per step, 4-step return."""
        return FlexLoadParams()


# FlexLoadBundle — grid-attachable SoA bundle
# Mirrors BatteryBundle interface so a grid-level host env can treat all
# bundles uniformly.

@struct.dataclass
class FlexLoadBundleState:
    """State for a FlexLoadBundle — one ring buffer per device.

    Attributes:
        curtailed_mw: Current-step curtailment [MW], shape (n_devices,).
        shift_out_mw: Current-step shift-out [MW], shape (n_devices,).
        shift_in_mw: Current-step released deferred demand [MW], shape (n_devices,).
        deferred_buffer: Per-device ring buffers, shape (n_devices, DEFAULT_MAX_BUFFER).
        buffer_head: Read pointer per device, shape (n_devices,) int32.
        buffer_size: Valid entries per device, shape (n_devices,) int32.
    """
    curtailed_mw: chex.Array    # (n_devices,) float32
    shift_out_mw: chex.Array    # (n_devices,) float32
    shift_in_mw: chex.Array     # (n_devices,) float32
    deferred_buffer: chex.Array  # (n_devices, DEFAULT_MAX_BUFFER) float32
    buffer_head: chex.Array     # (n_devices,) int32
    buffer_size: chex.Array     # (n_devices,) int32


def _device_step(
    buf: chex.Array,
    head: chex.Array,
    size: chex.Array,
    action: chex.Array,
    curtail_cap: chex.Array,
    shift_cap: chex.Array,
    shift_horizon: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """Step one FlexLoad device (pure function, vmappable).

    Args:
        buf: Ring buffer, shape (DEFAULT_MAX_BUFFER,).
        head: Read pointer (int32 scalar).
        size: Valid entries in buffer (int32 scalar).
        action: 2-D action [curtail_frac, shift_frac] in [0, 1].
        curtail_cap: Curtailment capacity [MW] (scalar).
        shift_cap: Shift capacity [MW] (scalar).
        shift_horizon: Number of future steps for deferred release (int32 scalar).

    Returns:
        (new_buf, new_head, new_size, curtail_mw, shift_out_mw, shift_in_mw)
        All scalars or (DEFAULT_MAX_BUFFER,) buf.
    """

    # 1. Release deferred demand (shift_in)
    released, buf, new_head, new_size = _buffer_pop(buf, head, size, DEFAULT_MAX_BUFFER)

    # 2. Parse action
    a0 = jnp.clip(action[0], 0.0, 1.0)
    a1 = jnp.clip(action[1], 0.0, 1.0)
    curtail_mw = a0 * curtail_cap
    shift_out_mw = a1 * shift_cap

    # 3. Buffer shift-out
    buf_shifted, size_shifted = _add_to_buffer(
        buf, new_head, new_size, shift_out_mw, shift_horizon, DEFAULT_MAX_BUFFER,
    )
    has_shift = shift_out_mw > 0.0
    buf_after = jnp.where(has_shift, buf_shifted, buf)
    size_after = jnp.where(has_shift, size_shifted, new_size)

    return buf_after, new_head, size_after, curtail_mw, shift_out_mw, released


# Vectorized over n_devices
_batch_device_step = jax.vmap(
    _device_step,
    in_axes=(0, 0, 0, 0, 0, 0, 0),
    out_axes=(0, 0, 0, 0, 0, 0),
)


@struct.dataclass
class FlexLoadBundle:
    """Struct-of-arrays FlexLoad bundle for grid env attachment.

    Groups N FlexLoad devices with identical action/obs dimensions.
    Implements the ResourceBundle duck-typed protocol used by
    grid-level host envs.

    Action per device : [curtail_frac, shift_frac]  ∈ [0, 1]²
    Obs per device    : [curtail_norm, shift_out_norm, shift_in_norm,
                         buf_fill_ratio, buf_energy_norm]  (5 dims)
    """

    # Static (pytree_node=False)
    n_devices: int = struct.field(pytree_node=False)
    per_device_action_dim: int = struct.field(pytree_node=False, default=2)
    per_device_obs_dim: int = struct.field(pytree_node=False, default=5)
    dt_hours: float = struct.field(pytree_node=False, default=0.5)

    # Traced leaves — SoA, shape (n_devices,)
    bus_idx: chex.Array = None           # int32 (n_devices,)
    curtail_cap_mw: chex.Array = None    # float32 (n_devices,) MW
    shift_cap_mw: chex.Array = None      # float32 (n_devices,) MW
    shift_horizon: chex.Array = None     # int32 (n_devices,) steps
    curtail_cost_per_mwh: chex.Array = None   # float32 (n_devices,)
    shift_cost_per_mwh: chex.Array = None     # float32 (n_devices,)

    @property
    def action_dim(self) -> int:
        """Width of the flat action vector for the whole bundle, not per device.

        Two controls per device ([curtail_frac, shift_frac]), device-major:
        device *i* owns slots ``[2i, 2i+2)``.
        """
        return self.n_devices * self.per_device_action_dim

    @property
    def obs_dim(self) -> int:
        """Width of the flat ``obs_slice`` for the whole bundle, not per device.

        Five channels per device (``observe`` lists them), same device-major
        layout: device *i* occupies slots ``[5i, 5i+5)``.
        """
        return self.n_devices * self.per_device_obs_dim

    def reset(self, key: chex.PRNGKey) -> FlexLoadBundleState:
        """Return zero-initialised bundle state."""
        return FlexLoadBundleState(
            curtailed_mw=jnp.zeros(self.n_devices, dtype=jnp.float32),
            shift_out_mw=jnp.zeros(self.n_devices, dtype=jnp.float32),
            shift_in_mw=jnp.zeros(self.n_devices, dtype=jnp.float32),
            deferred_buffer=jnp.zeros(
                (self.n_devices, DEFAULT_MAX_BUFFER), dtype=jnp.float32
            ),
            buffer_head=jnp.zeros(self.n_devices, dtype=jnp.int32),
            buffer_size=jnp.zeros(self.n_devices, dtype=jnp.int32),
        )

    def step(
        self,
        state: FlexLoadBundleState,
        action: chex.Array,
        ctx: Dict[str, Any],
    ) -> Tuple[
        FlexLoadBundleState,
        chex.Array,  # p_inject_mw  (n_devices,)
        chex.Array,  # q_inject_mvar (n_devices,) — always 0
        chex.Array,  # obs_slice     (obs_dim,)
        Dict[str, chex.Array],
    ]:
        """Run one step for all N FlexLoad devices.

        Args:
            state: Current bundle state.
            action: (action_dim,) in [0, 1], device-major layout
                ``[curtail_0, shift_0, curtail_1, shift_1, ...]``.
            ctx: Grid-provided context (unused; reserved for future use).

        Returns:
            (new_state, p_inject_mw, q_inject_mvar, obs_slice, cost_info)
        """
        act2d = jnp.clip(
            jnp.asarray(action, dtype=jnp.float32).reshape(self.n_devices, 2),
            0.0, 1.0,
        )

        (
            new_buf,
            new_head,
            new_size,
            curtail_mw,
            shift_out_mw,
            shift_in_mw,
        ) = _batch_device_step(
            state.deferred_buffer,
            state.buffer_head,
            state.buffer_size,
            act2d,
            self.curtail_cap_mw,
            self.shift_cap_mw,
            self.shift_horizon,
        )

        # Net power injection: load reduction = positive injection equivalent
        p_inject_mw = curtail_mw + shift_out_mw - shift_in_mw
        q_inject_mvar = jnp.zeros(self.n_devices, dtype=jnp.float32)

        new_state = FlexLoadBundleState(
            curtailed_mw=curtail_mw,
            shift_out_mw=shift_out_mw,
            shift_in_mw=shift_in_mw,
            deferred_buffer=new_buf,
            buffer_head=new_head,
            buffer_size=new_size,
        )

        obs_slice = self.observe(new_state, ctx)

        # Cost signals
        dt = self.dt_hours
        cost_curtail = curtail_mw * dt * self.curtail_cost_per_mwh  # (n_devices,)

        # Sum buffered energy per device for shift discomfort
        def _buf_energy_i(buf_i, head_i, size_i):
            """Bind ``max_size`` so ``_buffer_energy`` can be vmapped over the device axis."""
            return _buffer_energy(buf_i, head_i, size_i, DEFAULT_MAX_BUFFER)

        buf_totals = jax.vmap(_buf_energy_i)(new_buf, new_head, new_size)  # (n_devices,)
        cost_shift = buf_totals * dt * self.shift_cost_per_mwh  # (n_devices,)

        # Both components stay per-device in the dict below; only ``cost_sum``
        # is reduced to a bundle total.  Both are in $, so this sum stays
        # inside one channel — unlike ``BatteryBundle.step``.
        cost_sum = jnp.sum(cost_curtail + cost_shift)
        cost_info: Dict[str, chex.Array] = {
            "cost_sum": cost_sum,
            "cost": cost_sum,  # deprecated compatibility alias
            "cost_curtailment": cost_curtail,
            "cost_shift_discomfort": cost_shift,
        }

        return new_state, p_inject_mw, q_inject_mvar, obs_slice, cost_info

    def observe(
        self,
        state: FlexLoadBundleState,
        ctx: Dict[str, Any],
    ) -> chex.Array:
        """Return flat obs slice of shape (obs_dim,) = (n_devices * 5,).

        Per-device obs layout:
            [curtail_norm, shift_out_norm, shift_in_norm,
             buf_fill_ratio, buf_energy_norm]
        """
        c_norm = state.curtailed_mw / jnp.maximum(self.curtail_cap_mw, 1e-6)
        s_out_norm = state.shift_out_mw / jnp.maximum(self.shift_cap_mw, 1e-6)
        s_in_norm = state.shift_in_mw / jnp.maximum(self.shift_cap_mw, 1e-6)

        horizon_f = jnp.asarray(self.shift_horizon, dtype=jnp.float32)
        buf_fill = state.buffer_size.astype(jnp.float32) / jnp.maximum(horizon_f, 1.0)

        def _buf_energy_i(buf_i, head_i, size_i):
            """Bind ``max_size`` so ``_buffer_energy`` can be vmapped over the device axis."""
            return _buffer_energy(buf_i, head_i, size_i, DEFAULT_MAX_BUFFER)

        buf_totals = jax.vmap(_buf_energy_i)(
            state.deferred_buffer, state.buffer_head, state.buffer_size
        )
        max_buf_energy = jnp.maximum(self.shift_cap_mw * horizon_f, 1e-6)
        buf_energy_norm = buf_totals / max_buf_energy

        # Stack per-device vectors, then flatten
        per_device = jnp.stack(
            [
                jnp.clip(c_norm, 0.0, 1.0),
                jnp.clip(s_out_norm, 0.0, 1.0),
                jnp.clip(s_in_norm, 0.0, 1.0),
                jnp.clip(buf_fill, 0.0, 1.0),
                jnp.clip(buf_energy_norm, 0.0, 1.0),
            ],
            axis=1,
        )  # (n_devices, 5)
        return per_device.reshape(-1)  # (obs_dim,)


def make_flexload_bundle(
    case,
    bus_ids,
    *,
    curtail_cap_mw=0.10,
    shift_cap_mw=0.10,
    shift_horizon=4,
    curtail_cost_per_mwh=50.0,
    shift_cost_per_mwh=10.0,
    dt_hours: float = 0.5,
) -> FlexLoadBundle:
    """Create a FlexLoadBundle by resolving bus_ids through case.node_ids.

    All index lookup is done in NumPy at init time (CPU), then converted to
    jnp arrays.  NOT called inside JIT.

    Args:
        case: CaseData instance with ``node_ids`` attribute.
        bus_ids: External node IDs (Sequence[int]).
        curtail_cap_mw: Max curtailment per device [MW].  Scalar or Sequence.
        shift_cap_mw: Max shift-out per device [MW].  Scalar or Sequence.
        shift_horizon: Deferred release horizon in steps.  Scalar or Sequence.
        curtail_cost_per_mwh: Curtailment discomfort cost [$/MWh].  Scalar or Seq.
        shift_cost_per_mwh: Shift holding cost [$/MWh].  Scalar or Seq.
        dt_hours: Time-step duration [h].

    Returns:
        FlexLoadBundle with (n_devices,) SoA arrays.
    """
    import numpy as np_cpu

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

    def _broadcast_float(val):
        """Tile a scalar to ``(n,)`` float32, or pass a length-``n`` sequence through."""
        if hasattr(val, '__len__') and not isinstance(val, str):
            arr = np_cpu.asarray(val, dtype=np_cpu.float32)
            if arr.shape[0] != n:
                raise ValueError(f"Expected length {n}, got {arr.shape[0]}")
            return jnp.asarray(arr)
        return jnp.full((n,), float(val), dtype=jnp.float32)

    def _broadcast_int(val):
        """Same as ``_broadcast_float`` but int32 — ``shift_horizon`` counts steps."""
        if hasattr(val, '__len__') and not isinstance(val, str):
            arr = np_cpu.asarray(val, dtype=np_cpu.int32)
            if arr.shape[0] != n:
                raise ValueError(f"Expected length {n}, got {arr.shape[0]}")
            return jnp.asarray(arr)
        return jnp.full((n,), int(val), dtype=jnp.int32)

    return FlexLoadBundle(
        n_devices=n,
        per_device_action_dim=2,
        per_device_obs_dim=5,
        dt_hours=float(dt_hours),
        bus_idx=jnp.asarray(internal_idx, dtype=jnp.int32),
        curtail_cap_mw=_broadcast_float(curtail_cap_mw),
        shift_cap_mw=_broadcast_float(shift_cap_mw),
        shift_horizon=_broadcast_int(shift_horizon),
        curtail_cost_per_mwh=_broadcast_float(curtail_cost_per_mwh),
        shift_cost_per_mwh=_broadcast_float(shift_cost_per_mwh),
    )
