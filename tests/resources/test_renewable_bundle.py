# Vendored test from PowerZooJax.
# Source        : tests/resource/test_renewable_bundle.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim except: import paths, and the single DistGridEnv integration test at the end (see note there).
"""R2 — RenewableBundle (PV inverter: active curtail + reactive support).

Test coverage:
  L0  JAX contract:  jit / vmap / pytree / scan rollout / pytree_structure
  L1  Physics:       curtailment reduces P, Q-circle constraint, no-curtail path,
                     profiles indexed correctly, obs shape, cost signals
  L1  make_renewable_bundle: bus_idx resolution, shape broadcasting
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.case import create_case33bw
from powermarketjax.resources.renewable import (
    RenewableBundle,
    RenewableBundleState,
    make_renewable_bundle,
)


# ========================== Fixtures ==========================


@pytest.fixture(scope="module")
def case33():
    return create_case33bw()


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


def _simple_bundle(case33, *, n=3, allow_curtailment=True, T=12) -> RenewableBundle:
    """Create a small 3-device bundle on feeder ends of case33bw."""
    return make_renewable_bundle(
        case33,
        bus_ids=[18, 22, 33],
        capacity_mw=0.20,
        s_rated_mva=0.25,
        max_steps=T,
        allow_curtailment=allow_curtailment,
        curtail_cost_per_mwh=10.0,
        dt_hours=0.5,
    )


# ========================== L0 — JAX Contract ==========================


def test_pytree_structure(case33):
    """RenewableBundle and RenewableBundleState are valid JAX pytrees."""
    bundle = _simple_bundle(case33)
    state = bundle.reset(jax.random.PRNGKey(0))

    leaves_b, treedef_b = jax.tree_util.tree_flatten(bundle)
    leaves_s, treedef_s = jax.tree_util.tree_flatten(state)
    assert len(leaves_b) > 0
    assert len(leaves_s) > 0


def test_static_fields_unchanged_under_jit(case33):
    """Static fields (n_devices, per_device_action_dim, …) survive JIT round-trip."""
    bundle = _simple_bundle(case33)

    @jax.jit
    def _reset(key):
        return bundle.reset(key)

    state = _reset(jax.random.PRNGKey(0))
    assert bundle.n_devices == 3
    assert bundle.per_device_action_dim == 2
    assert bundle.per_device_obs_dim == 4


def test_jit_step(case33, key):
    """step() is JIT-compilable."""
    bundle = _simple_bundle(case33)
    state = bundle.reset(key)
    act = jnp.zeros(bundle.action_dim)

    step_jit = jax.jit(lambda s, a: bundle.step(s, a, {}))
    out1 = step_jit(state, act)
    out2 = step_jit(state, act)
    np.testing.assert_allclose(
        np.asarray(out1[1]), np.asarray(out2[1])
    )  # p_inject deterministic


def test_vmap_step(case33, key):
    """step() vmaps over a batch of states."""
    bundle = _simple_bundle(case33)
    keys = jax.random.split(key, 4)
    states = jax.vmap(lambda k: bundle.reset(k))(keys)
    acts = jnp.ones((4, bundle.action_dim), dtype=jnp.float32)

    batched = jax.vmap(lambda s, a: bundle.step(s, a, {}))
    new_states, p_inj, q_inj, obs, cost_info = batched(states, acts)
    assert p_inj.shape == (4, bundle.n_devices)
    assert q_inj.shape == (4, bundle.n_devices)
    assert obs.shape == (4, bundle.obs_dim)


def test_scan_rollout(case33, key):
    """Bundle step can drive a lax.scan rollout."""
    bundle = _simple_bundle(case33, T=12)
    state0 = bundle.reset(key)
    act = jnp.zeros(bundle.action_dim)

    def scan_fn(carry, _xs):
        s = carry
        new_s, p_inj, q_inj, obs, cost = bundle.step(s, act, {})
        return new_s, p_inj

    _, p_traj = jax.lax.scan(scan_fn, state0, None, length=12)
    assert p_traj.shape == (12, bundle.n_devices)


def test_pytree_reconstruct(case33, key):
    """Reconstructing from leaves produces identical outputs."""
    bundle = _simple_bundle(case33)
    state = bundle.reset(key)
    leaves, treedef = jax.tree_util.tree_flatten(state)
    state2 = jax.tree_util.tree_unflatten(treedef, leaves)
    act = jnp.zeros(bundle.action_dim)
    _, p1, _, _, _ = bundle.step(state, act, {})
    _, p2, _, _, _ = bundle.step(state2, act, {})
    np.testing.assert_allclose(np.asarray(p1), np.asarray(p2))


# ========================== L1 — Physics ==========================


def test_no_curtailment_full_output(case33, key):
    """Action +1 (no curtailment) → P ≈ capacity × CF."""
    bundle = _simple_bundle(case33)
    state = bundle.reset(key)
    # action[0::2] = +1 (no curtail), action[1::2] = 0 (Q=0)
    act = jnp.zeros(bundle.action_dim).at[0::2].set(1.0)
    _, p_inj, _, _, _ = bundle.step(state, act, {})
    # CF at t=0 from bell-curve profile: around 0 for step 0 (midnight)
    # Just verify P ≥ 0
    assert float(jnp.min(p_inj)) >= -1e-6


def test_full_curtailment_zero_output(case33, key):
    """Action -1 (full curtailment) → P ≈ 0."""
    bundle = _simple_bundle(case33, allow_curtailment=True)
    # Use flat profiles = constant 0.5 for a clear test
    profiles = jnp.full((12, 3), 0.5, dtype=jnp.float32)
    bundle_flat = make_renewable_bundle(
        case33,
        bus_ids=[18, 22, 33],
        capacity_mw=0.20,
        profiles=profiles,
        allow_curtailment=True,
    )
    state = bundle_flat.reset(key)
    act = jnp.full(bundle_flat.action_dim, -1.0)  # full curtailment
    _, p_inj, _, _, _ = bundle_flat.step(state, act, {})
    np.testing.assert_allclose(np.asarray(p_inj), 0.0, atol=1e-6)


def test_partial_curtailment(case33, key):
    """Partial curtailment produces output strictly between 0 and full."""
    profiles = jnp.full((12, 3), 0.8, dtype=jnp.float32)
    bundle = make_renewable_bundle(
        case33,
        bus_ids=[18, 22, 33],
        capacity_mw=0.20,
        profiles=profiles,
        allow_curtailment=True,
    )
    state = bundle.reset(key)
    full_act = jnp.zeros(bundle.action_dim).at[0::2].set(1.0)   # no curtail
    half_act = jnp.zeros(bundle.action_dim).at[0::2].set(0.0)   # 50% curtail
    _, p_full, _, _, _ = bundle.step(state, full_act, {})
    _, p_half, _, _, _ = bundle.step(state, half_act, {})
    # p_half should be strictly less than p_full
    assert float(jnp.min(p_full - p_half)) > 1e-6


def test_allow_curtailment_false_ignores_curtailment(case33, key):
    """allow_curtailment=False: active output at MPPT regardless of action."""
    profiles = jnp.full((12, 3), 0.5, dtype=jnp.float32)
    bundle_nc = make_renewable_bundle(
        case33,
        bus_ids=[18, 22, 33],
        capacity_mw=0.20,
        profiles=profiles,
        allow_curtailment=False,
    )
    bundle_c = make_renewable_bundle(
        case33,
        bus_ids=[18, 22, 33],
        capacity_mw=0.20,
        profiles=profiles,
        allow_curtailment=True,
    )
    s_nc = bundle_nc.reset(key)
    s_c = bundle_c.reset(key)
    act_curtail = jnp.full(bundle_nc.action_dim, -1.0)
    _, p_nc, _, _, _ = bundle_nc.step(s_nc, act_curtail, {})
    _, p_c, _, _, _ = bundle_c.step(s_c, act_curtail, {})
    # no-curtail bundle should output MPPT (0.20 MW × 0.5 CF = 0.10 MW)
    expected = 0.20 * 0.5
    np.testing.assert_allclose(np.asarray(p_nc), expected, atol=1e-5)
    # curtail-enabled bundle should output 0
    np.testing.assert_allclose(np.asarray(p_c), 0.0, atol=1e-5)


def test_q_circle_constraint(case33, key):
    """Reactive output stays inside PQ circle: P² + Q² ≤ S_rated²."""
    profiles = jnp.full((12, 3), 0.6, dtype=jnp.float32)
    bundle = make_renewable_bundle(
        case33,
        bus_ids=[18, 22, 33],
        capacity_mw=0.20,
        s_rated_mva=0.22,
        profiles=profiles,
        allow_curtailment=True,
    )
    state = bundle.reset(key)
    # Push max Q with half curtailment
    act = jnp.zeros(bundle.action_dim).at[0::2].set(0.0).at[1::2].set(1.0)
    _, p_inj, q_inj, _, _ = bundle.step(state, act, {})
    apparent = jnp.sqrt(p_inj ** 2 + q_inj ** 2)
    s_rated_broadcast = jnp.asarray([0.22, 0.22, 0.22])
    assert float(jnp.max(apparent - s_rated_broadcast)) <= 1e-5


def test_q_zero_when_action_zero(case33, key):
    """Q-action = 0 → Q injection = 0."""
    bundle = _simple_bundle(case33)
    state = bundle.reset(key)
    act = jnp.zeros(bundle.action_dim)
    _, _, q_inj, _, _ = bundle.step(state, act, {})
    np.testing.assert_allclose(np.asarray(q_inj), 0.0, atol=1e-6)


def test_q_positive_and_negative(case33, key):
    """Positive Q action gives positive Q; negative gives negative."""
    profiles = jnp.full((12, 3), 0.5, dtype=jnp.float32)
    bundle = make_renewable_bundle(
        case33,
        bus_ids=[18, 22, 33],
        capacity_mw=0.10,
        s_rated_mva=0.20,
        profiles=profiles,
    )
    state = bundle.reset(key)
    act_pos = jnp.zeros(bundle.action_dim).at[1::2].set(1.0)
    act_neg = jnp.zeros(bundle.action_dim).at[1::2].set(-1.0)
    _, _, q_pos, _, _ = bundle.step(state, act_pos, {})
    _, _, q_neg, _, _ = bundle.step(state, act_neg, {})
    assert float(jnp.min(q_pos)) > 0.0
    assert float(jnp.max(q_neg)) < 0.0


def test_profile_indexed_correctly(case33, key):
    """Output P follows the profile: higher CF → higher P at same curtailment."""
    profiles_lo = jnp.full((12, 3), 0.2, dtype=jnp.float32)
    profiles_hi = jnp.full((12, 3), 0.8, dtype=jnp.float32)
    bundle_lo = make_renewable_bundle(
        case33, bus_ids=[18, 22, 33], capacity_mw=0.20, profiles=profiles_lo
    )
    bundle_hi = make_renewable_bundle(
        case33, bus_ids=[18, 22, 33], capacity_mw=0.20, profiles=profiles_hi
    )
    s_lo = bundle_lo.reset(key)
    s_hi = bundle_hi.reset(key)
    act = jnp.zeros(bundle_lo.action_dim).at[0::2].set(1.0)  # no curtail
    _, p_lo, _, _, _ = bundle_lo.step(s_lo, act, {})
    _, p_hi, _, _, _ = bundle_hi.step(s_hi, act, {})
    assert float(jnp.min(p_hi - p_lo)) > 1e-5


def test_obs_shape(case33, key):
    """observe() returns correct shape (n_devices * 4,)."""
    bundle = _simple_bundle(case33)
    state = bundle.reset(key)
    act = jnp.zeros(bundle.action_dim)
    _, _, _, obs, _ = bundle.step(state, act, {})
    assert obs.shape == (bundle.obs_dim,)
    assert bundle.obs_dim == bundle.n_devices * 4


def test_curtailment_cost_signal(case33, key):
    """Curtailment cost is non-zero when output is curtailed."""
    profiles = jnp.full((12, 3), 0.5, dtype=jnp.float32)
    bundle = make_renewable_bundle(
        case33,
        bus_ids=[18, 22, 33],
        capacity_mw=0.20,
        profiles=profiles,
        allow_curtailment=True,
        curtail_cost_per_mwh=10.0,
    )
    state = bundle.reset(key)
    act_full = jnp.zeros(bundle.action_dim).at[0::2].set(1.0)   # no curtail
    act_curt = jnp.full(bundle.action_dim, -1.0)                 # full curtail
    _, _, _, _, cost_full = bundle.step(state, act_full, {})
    _, _, _, _, cost_curt = bundle.step(state, act_curt, {})
    # Curtailment cost should be larger when curtailing
    assert float(cost_curt["cost_curtailment"]) > float(cost_full["cost_curtailment"])
    assert float(cost_curt["cost"]) > 0.0


def test_p_always_nonnegative(case33, key):
    """Renewable P output is never negative."""
    bundle = _simple_bundle(case33)
    state = bundle.reset(key)
    for a0 in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        act = jnp.full(bundle.action_dim, a0)
        _, p_inj, _, _, _ = bundle.step(state, act, {})
        assert float(jnp.min(p_inj)) >= -1e-6


# ========================== L1 — make_renewable_bundle ==========================


def test_make_bundle_bus_idx_resolution(case33):
    """bus_ids are mapped to correct internal indices via case.node_ids."""
    import numpy as np_cpu
    bundle = make_renewable_bundle(case33, bus_ids=[18, 22, 33], capacity_mw=0.10)
    node_ids = np_cpu.asarray(case33.node_ids)
    for i, bid in enumerate([18, 22, 33]):
        expected_idx = int(np_cpu.where(node_ids == bid)[0][0])
        assert int(bundle.bus_idx[i]) == expected_idx


def test_make_bundle_scalar_broadcast(case33):
    """Scalar capacity_mw is broadcast to (n_devices,) correctly."""
    bundle = make_renewable_bundle(case33, bus_ids=[10, 20, 30], capacity_mw=0.15)
    np.testing.assert_allclose(np.asarray(bundle.capacity_mw), 0.15, atol=1e-6)
    assert bundle.capacity_mw.shape == (3,)


def test_make_bundle_per_device_capacity(case33):
    """Per-device capacity array is accepted."""
    bundle = make_renewable_bundle(
        case33, bus_ids=[10, 20, 30], capacity_mw=[0.10, 0.15, 0.20]
    )
    np.testing.assert_allclose(np.asarray(bundle.capacity_mw), [0.10, 0.15, 0.20], atol=1e-6)


def test_make_bundle_profiles_1d_broadcast(case33):
    """1-D profile is broadcast to (T, n_devices)."""
    prof = jnp.linspace(0.0, 1.0, 12)
    bundle = make_renewable_bundle(case33, bus_ids=[10, 20], profiles=prof)
    assert bundle.profiles.shape == (12, 2)


def test_make_bundle_profiles_2d(case33):
    """2-D profile (T, n_devices) is accepted as-is."""
    prof = jnp.ones((12, 2), dtype=jnp.float32) * 0.7
    bundle = make_renewable_bundle(case33, bus_ids=[10, 20], profiles=prof)
    assert bundle.profiles.shape == (12, 2)


def test_make_bundle_invalid_bus_raises(case33):
    """Non-existent bus_id raises ValueError."""
    with pytest.raises(ValueError, match="not found"):
        make_renewable_bundle(case33, bus_ids=[9999])


def test_make_bundle_empty_bus_raises(case33):
    """Empty bus_ids raises ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        make_renewable_bundle(case33, bus_ids=[])


def test_make_bundle_action_obs_dims(case33):
    """Bundle exposes correct action_dim and obs_dim for n devices."""
    bundle = make_renewable_bundle(case33, bus_ids=[10, 20, 30, 5, 15])
    assert bundle.n_devices == 5
    assert bundle.action_dim == 5 * 2   # per_device_action_dim = 2
    assert bundle.obs_dim == 5 * 4     # per_device_obs_dim = 4


def test_make_bundle_s_rated_default(case33):
    """Default s_rated = 1.1 * capacity_mw."""
    bundle = make_renewable_bundle(case33, bus_ids=[10, 20], capacity_mw=0.20)
    np.testing.assert_allclose(np.asarray(bundle.s_rated), 0.20 * 1.1, atol=1e-5)


# ========================== L1 — DistGridEnv Integration ==========================
#
# REMOVED when vendoring (2026-08-05). Upstream had one test here,
# ``test_renewable_bundle_in_dist_grid_changes_voltage``, which attaches the
# bundle to ``powerzoojax.envs.grid.dist.DistGridEnv`` and asserts that
# reactive injection moves bus voltages. ``DistGridEnv`` is a grid *environment*
# and is deliberately not vendored (solvers yes, env classes no). The other 26 tests in this file are verbatim.
#
# The physics it covered is not lost: bundle Q output is asserted in the unit
# tests above, and voltage response to injection is covered by
# tests/physics/test_bfs_power_flow.py.
