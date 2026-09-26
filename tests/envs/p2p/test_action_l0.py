"""L0 for the P2P action map: the JAX contract."""
import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.envs.p2p import make_action_map
from powermarketjax.resources.battery import make_battery_bundle

PI_EXP, PI_RET, DELTA = 4.1, 26.11, 0.5
N = 12


@pytest.fixture
def battery():
    return make_battery_bundle(n_devices=N, power_mw=2.0, capacity_mwh=4.0)


@pytest.fixture
def act():
    fn, _ = make_action_map(N, PI_EXP, PI_RET, DELTA)
    return fn


@pytest.fixture
def inputs():
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    action = jax.random.uniform(k1, (N, 2), jnp.float32, -1.0, 1.0)
    soc = jax.random.uniform(k2, (N,), jnp.float32, 0.2, 0.8)
    p_pv = jax.random.uniform(k3, (N,), jnp.float32, 0.0, 3.0)
    load = jnp.float32(1.5) * jnp.ones((N,), jnp.float32)
    return action, soc, p_pv, load


def test_rejects_a_bad_tariff_pair_and_period(battery):
    with pytest.raises(ValueError):
        make_action_map(N, PI_RET, PI_EXP, DELTA)
    with pytest.raises(ValueError):
        make_action_map(N, PI_EXP, PI_RET, 0.0)


def test_jit_matches_eager(act, inputs, battery):
    """Close, not equal: the price interpolation is a multiply-add.

    `price` is ``(1 - w) * pi_exp + w * pi_ret``, and XLA contracts that into a
    fused multiply-add under `jit` while the eager path evaluates it in two
    rounded steps, which was measured at 9.5e-7 absolute and 4.2e-8 relative.
    Everything else agrees bit for bit.  The clearing operator is compared with
    exact equality in `test_clearing_l0.py` because it contains no arithmetic
    that can be contracted -- only comparisons, gathers and cumulative sums.
    """
    eager = act(*inputs, battery)
    compiled = jax.jit(act)(*inputs, battery)
    chex.assert_trees_all_close(eager, compiled, rtol=1e-6, atol=1e-6)
    for key in eager:
        if key != "price":
            np.testing.assert_array_equal(np.asarray(eager[key]),
                                          np.asarray(compiled[key]))


def test_shapes_and_dtypes(act, inputs, battery):
    out = jax.jit(act)(*inputs, battery)
    for key, value in out.items():
        assert value.shape == (N,), key
        assert value.dtype == jnp.float32, key


def test_vmap_lanes_are_bit_identical(act, inputs, battery):
    batch = 8
    tiled = tuple(jnp.broadcast_to(a, (batch,) + a.shape) for a in inputs)
    out = jax.jit(jax.vmap(act, in_axes=(0, 0, 0, 0, None)))(*tiled, battery)
    for key, value in out.items():
        assert value.shape == (batch, N)
        assert bool(jnp.all(value == value[0])), key


def test_runs_under_scan(act, battery):
    """The soc advance is not this module's job, so the carry is trivial here.

    What this checks is that nothing in the map depends on a Python-level value
    that `lax.scan` would refuse to trace.
    """
    horizon = 32
    key = jax.random.PRNGKey(1)
    actions = jax.random.uniform(key, (horizon, N, 2), jnp.float32, -1.0, 1.0)
    soc = jnp.full((N,), 0.5, jnp.float32)
    p_pv = jnp.full((N,), 2.0, jnp.float32)
    load = jnp.full((N,), 1.0, jnp.float32)

    def body(carry, action):
        out = act(action, carry, p_pv, load, battery)
        return carry, out["clip"].sum()

    _, clips = jax.jit(
        lambda a: jax.lax.scan(body, soc, a))(actions)
    assert clips.shape == (horizon,)
    assert bool(jnp.all(jnp.isfinite(clips)))


def test_no_nan_with_zero_rated_power(act):
    """§14 pads with `power_mw=0`, which would put 0/0 on the `costs` channel.

    A padded participant has no rated power at all, so the normalisation in
    §9.5 divides by zero unless the denominator is floored.  It must report a
    clip of zero rather than a NaN, and a NaN here would ride the trajectory
    into the gradients.
    """
    padded = make_battery_bundle(n_devices=N, power_mw=0.0, capacity_mwh=1e-3)
    out = jax.jit(act)(jnp.ones((N, 2), jnp.float32),
                       jnp.full((N,), 0.5, jnp.float32),
                       jnp.zeros((N,), jnp.float32),
                       jnp.zeros((N,), jnp.float32), padded)
    for key, value in out.items():
        assert bool(jnp.all(jnp.isfinite(value))), key
    assert float(out["clip"].sum()) == 0.0


def test_no_nan_on_extreme_actions(act, inputs, battery):
    _, soc, p_pv, load = inputs
    for magnitude in (1e3, 1e6):
        for sign in (-1.0, 1.0):
            action = jnp.full((N, 2), jnp.float32(sign * magnitude))
            out = jax.jit(act)(action, soc, p_pv, load, battery)
            for key, value in out.items():
                assert bool(jnp.all(jnp.isfinite(value))), (key, sign, magnitude)


def test_soc_advance_under_jit_vmap_and_scan(battery):
    """`make_soc_advance` carries the same JAX contract as the action map."""
    from powermarketjax.envs.p2p import make_soc_advance
    act, _ = make_action_map(N, PI_EXP, PI_RET, DELTA)
    advance = make_soc_advance(DELTA)

    soc = jnp.full((N,), 0.5, jnp.float32)
    p = jnp.linspace(-1.0, 1.0, N, dtype=jnp.float32)
    chex.assert_trees_all_equal(advance(soc, p, battery),
                                jax.jit(advance)(soc, p, battery))
    assert jax.jit(advance)(soc, p, battery).shape == (N,)
    assert jax.jit(advance)(soc, p, battery).dtype == jnp.float32

    batch = 8
    out = jax.jit(jax.vmap(advance, in_axes=(0, 0, None)))(
        jnp.broadcast_to(soc, (batch, N)), jnp.broadcast_to(p, (batch, N)), battery)
    assert out.shape == (batch, N)
    assert bool(jnp.all(out == out[0]))

    # a real rollout: the action map and the advance chained under one scan
    horizon = 64
    actions = jax.random.uniform(jax.random.PRNGKey(4), (horizon, N, 2),
                                 jnp.float32, -1.0, 1.0)
    p_pv = jnp.full((N,), 2.0, jnp.float32)
    load = jnp.full((N,), 1.0, jnp.float32)

    def body(carry, action):
        out = act(action, carry, p_pv, load, battery)
        return advance(carry, out["p_signed"], battery), out["clip"].sum()

    final, clips = jax.jit(lambda a: jax.lax.scan(body, soc, a))(actions)
    assert clips.shape == (horizon,)
    assert bool(jnp.all(jnp.isfinite(final)))
    assert bool(jnp.all((final >= battery.soc_min - 1e-6)
                        & (final <= battery.soc_max + 1e-6)))
