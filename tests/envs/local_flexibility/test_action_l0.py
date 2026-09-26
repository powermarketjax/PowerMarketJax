"""L0 JAX contract for the local flexibility action map (§9.3).

`jit`, `vmap`, and the float32 dtype discipline of the environment layer.  The dtype checks are
not decoration here: every test in this repository runs with `jax_enable_x64`
on, so a constant written without a dtype turns the whole submission float64
without changing a single number, and
the submission is what `EnvState` and the clearing are built from.

**The action box is checked here too**, because it is published by this
market's `spec` and read by `learning/policy.py:bounds_for`, and the file that
guards that contract for the wholesale markets says in its own docstring that
it does not build this one.  The map's `spec` needs no case, so the contract is
checkable from this side without one.

**The price is compared within fusion rather than bitwise, and the quantities
are compared bitwise.**  `price = c_rep * (1 + softplus(alpha))` is a
multiply-add, which XLA fuses into an FMA under `jit` and under `vmap` but not
in the eager path; measured here at 7.6e-6 \\$/MWh, 6.8e-8 relative, on one of
eight participants.  That is the same 1-ULP fusion the P2P action map records,
and it reaches only the two money-like leaves: the four megawatt quantities are
sums, minima and products that no reassociation touches, so they are equal to
the last bit and are asserted that way.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.envs.local_flexibility import make_action_map
from powermarketjax.resources.battery import make_battery_bundle

N = 8
DELTA = 0.25
BATCH = 16
KEYS = ("price", "qty_max", "plan", "q_phys", "c_rep", "charge_headroom")
#: Leaves XLA fuses into an FMA (module docstring); everything else is bitwise.
FUSED = ("price", "c_rep")


def compare(left, right, key):
    if key in FUSED:
        np.testing.assert_allclose(np.asarray(left), np.asarray(right),
                                   rtol=1e-6, err_msg=key)
    else:
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right),
                                      err_msg=key)


@pytest.fixture(autouse=True)
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture
def bundle():
    rng = np.random.default_rng(0)
    return make_battery_bundle(
        n_devices=N, dt_hours=DELTA,
        capacity_mwh=rng.uniform(0.2, 2.0, N).tolist(),
        power_mw=rng.uniform(0.05, 0.5, N).tolist(),
        eta_charge=0.95, eta_discharge=0.93)


def inputs(seed=0):
    rng = np.random.default_rng(seed)
    return (jnp.asarray(rng.normal(0.0, 2.0, (N, 3)), jnp.float32),
            jnp.asarray(rng.uniform(0.15, 0.85, N), jnp.float32),
            jnp.float32(rng.uniform(20.0, 90.0)),
            jnp.asarray(rng.uniform(2.0, 20.0, N), jnp.float32))


def test_shapes_and_jit_matches_eager(bundle):
    act_map, spec = make_action_map(N, DELTA)
    action, soc, price_e, cycle = inputs()
    eager = act_map(action, soc, price_e, bundle, cycle)
    jitted = jax.jit(act_map, static_argnums=())(action, soc, price_e, bundle, cycle)
    for key in KEYS:
        assert eager[key].shape == (N,)
        compare(eager[key], jitted[key], key)
    assert spec["action_shape"] == (N, 3)


def test_dtypes_are_float32_under_x64(bundle):
    """Float32 in, float32 out, with `jax_enable_x64` on."""
    act_map, _ = make_action_map(N, DELTA)
    out = jax.jit(act_map)(*inputs()[:3], bundle, inputs()[3])
    for key in KEYS:
        assert out[key].dtype == jnp.float32, key


def test_vmap_over_environments(bundle):
    act_map, _ = make_action_map(N, DELTA)
    action, soc, price_e, cycle = inputs()
    batched_action = jnp.broadcast_to(action, (BATCH, N, 3))
    batched_soc = jnp.broadcast_to(soc, (BATCH, N))
    batched_price = jnp.full((BATCH,), price_e, jnp.float32)
    out = jax.jit(jax.vmap(act_map, in_axes=(0, 0, 0, None, None)))(
        batched_action, batched_soc, batched_price, bundle, cycle)
    single = act_map(action, soc, price_e, bundle, cycle)
    for key in KEYS:
        assert out[key].shape == (BATCH, N)
        compare(out[key][0], single[key], key)


def test_period_hours_must_be_positive():
    with pytest.raises(ValueError, match="period_hours"):
        make_action_map(N, 0.0)


def test_the_spec_publishes_a_finite_box_and_bounds_for_reads_it():
    """The action space is finite, is this market's own, and reaches the harness.

    Three claims a box chosen by a learner could not make.  The ends are
    finite, so `policy.to_action` squashes instead of passing the coordinate
    through -- which is the point: SAC's critic reads the raw coordinate and
    its actor maximises the critic, so an unbounded one leaves the critic with
    no fixed point (`policy.bounds_for` records that failure; the SAC section
    of `tools/flex_experiment/concentration_baseline.py` records this market's
    own measurement of it, `q_loss` 9.0e18 inside one iteration).  They are
    `ACTION_SATURATION`, this market's own constant, and not a round number.
    And they are the frame the non-learner's baseline already sits on, so
    `BASELINE_ACTION` is a corner of the action space rather than a point
    outside it -- which is what lets a learner that squashes submit the
    truthful reference exactly, unlike the ancillary market, whose truthful
    action lies below its box by design.

    The other side of the discrimination is the same call: on this draw the
    identity map and the box differ by 124.35, so "the box was read" is not a
    property of the draw being small.
    """
    pytest.importorskip("optax",
                        reason="powermarketjax.learning needs the rl extra")
    from powermarketjax.learning.policy import bounds_for, to_action
    from powermarketjax.envs.local_flexibility.action import ACTION_SATURATION
    from powermarketjax.envs.local_flexibility.env import BASELINE_ACTION

    spec = make_action_map(N, DELTA)[1]
    assert np.isfinite(spec["action_low"]) and np.isfinite(spec["action_high"])
    assert (spec["action_low"], spec["action_high"]) \
        == (-ACTION_SATURATION, ACTION_SATURATION)
    low, high = bounds_for(spec)
    assert low.shape == (N, 3) == high.shape
    assert np.all(np.asarray(low) == -ACTION_SATURATION)
    assert np.all(np.asarray(high) == ACTION_SATURATION)
    # the baseline is a corner of the box, coordinate by coordinate
    assert np.all(np.abs(np.asarray(BASELINE_ACTION)) == ACTION_SATURATION)
    pre = jax.random.normal(jax.random.PRNGKey(5), (N, 3)) * 3.0
    mapped = np.asarray(to_action(pre, low, high))
    assert np.all(np.abs(mapped) < ACTION_SATURATION)
    assert float(np.max(np.abs(mapped - np.asarray(pre)))) > 100.0
