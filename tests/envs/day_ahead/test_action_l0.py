"""L0 JAX contract for the day-ahead action map.

jit, vmap, pytree stability, no NaN, and callability from inside lax.scan.
Domain correctness is L1 (test_action_l1.py).

x64 is set by an autouse fixture rather than at import time because other
modules in this suite turn it off globally (see the L0 clearing module).  The
map does not need float64, but it feeds the operator that does.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.day_ahead import make_offer_map

T = 4
K = 3
MARKUP_MAX = 3.0


@pytest.fixture(scope="module", autouse=True)
def x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


@pytest.fixture(scope="module")
def case(x64):
    return load_case("29gb")


@pytest.fixture(scope="module")
def n_units(case):
    return len(np.asarray(case.unit_p_min))


def test_full_offer_shape_and_spec(case, n_units):
    offer_map, spec = make_offer_map(case, K, T, kind="full")
    assert spec["shape"] == (n_units, K, T)
    assert spec["low"] == -np.inf and spec["high"] == np.inf
    offer = offer_map(jnp.zeros(spec["shape"], jnp.float64))
    assert offer.shape == (n_units, K, T)
    assert offer.dtype == jnp.float64
    assert np.isfinite(np.asarray(offer)).all()


def test_markup_shape_and_spec(case, n_units):
    offer_map, spec = make_offer_map(case, K, T, kind="markup",
                                     markup_max=MARKUP_MAX)
    assert spec["shape"] == (n_units,)
    assert (spec["low"], spec["high"]) == (1.0, MARKUP_MAX)
    offer = offer_map(jnp.full((n_units,), 1.5, jnp.float64))
    assert offer.shape == (n_units, K, T)
    assert np.isfinite(np.asarray(offer)).all()


def test_jit_matches_eager(case, n_units):
    offer_map, spec = make_offer_map(case, K, T, kind="full")
    action = jax.random.normal(jax.random.PRNGKey(0), spec["shape"], jnp.float64)
    np.testing.assert_allclose(np.asarray(offer_map(action)),
                               np.asarray(jax.jit(offer_map)(action)), rtol=1e-13)


def test_vmap_over_batch(case, n_units):
    """Batched mapping equals per-lane mapping, lane by lane."""
    offer_map, spec = make_offer_map(case, K, T, kind="full")
    actions = jax.random.normal(jax.random.PRNGKey(1), (3,) + spec["shape"],
                                jnp.float64)
    out = jax.vmap(offer_map)(actions)
    for i in range(3):
        np.testing.assert_allclose(np.asarray(out[i]),
                                   np.asarray(offer_map(actions[i])), rtol=1e-13)


def test_inside_lax_scan(case, n_units):
    """One day is one scan step, and the action changes from day to day."""
    offer_map, spec = make_offer_map(case, K, T, kind="markup",
                                     markup_max=MARKUP_MAX)
    markups = jnp.linspace(1.0, MARKUP_MAX, 5, dtype=jnp.float64)

    def day(carry, alpha):
        offer = offer_map(jnp.full((n_units,), alpha))
        return carry, offer.mean()

    _, means = jax.lax.scan(day, 0.0, markups)
    assert means.shape == (5,)
    # the mean offer is linear in the multiplier, so it is strictly increasing
    assert (np.diff(np.asarray(means)) > 0).all()


def test_rejects_unknown_kind_and_missing_bound(case):
    with pytest.raises(ValueError, match="markup_max"):
        make_offer_map(case, K, T, kind="markup")
    with pytest.raises(ValueError, match="kind"):
        make_offer_map(case, K, T, kind="scalar")
