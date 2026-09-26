"""L0 JAX contract for the joint clearing operator (§7).

`jit`, `vmap` over parallel environments, `lax.scan` over the periods of an
episode, a pytree that does not depend on the data, the float64 guard, and the
constructor guards on the two parameters that have no safe default.

Batch determinism is checked for the same reason the other two clearing
operators check it: the interior point method runs a fixed number of Newton
steps, which is what makes the batch cost independent of batch content and the
lanes bit-identical.  A data-dependent `while_loop` would break both.

The operating point is a real one, taken from the day-ahead pre-commitment
fixture, because this market's diagnostics were measured to depend on it: a
previous dispatch set to the committed minimum makes (RMP) bind on everything
and trivialises the problem, which is how one earlier measurement went wrong.
"""
from pathlib import Path

import jax
import jax.extend
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from powermarketjax.case import load_case
from powermarketjax.envs.ancillary.clearing import MAX_ITER, make_clearing
from powermarketjax.envs.day_ahead.clearing import segment_costs

CASE = "29gb"
THETA = (1.0 / 6.0, 0.5)
VOLR = 250.0
BETA = np.array([0.020, 0.050])
BATCH = 8
T = 4


@pytest.fixture
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


#: Resolved from this file rather than from the working directory: which fixture
#: a relative path finds depends on where pytest was invoked, which is the same
#: family of hazard as the interpreter resolving the package by script location.
#: **The filename is a scenario setting point, and it is the one that no search
#: for scenario constants can find**: it carries `cap_scale` and `ramp_scale` in
#: a string, so it matches neither a value pattern, nor a symbol name, nor a
#: tuple unpacking, nor an arithmetic literal.  Migrating the constants above
#: without migrating this path leaves every test in this package clearing a
#: 0.6 / 1.00 market on top of a 0.4 / 0.25 commitment -- the exact defect the
#: cross-check in an early training driver now refuses, found
#: there first and here second.
FIXTURE = (Path(__file__).resolve().parents[2] / "fixtures"
           / "day_ahead_commitment_29gb_T24_relax_seasons.npz")
#: Hour 12 of day 0 is used because it carries **both** committed and
#: de-committed units, which several assertions below need in order to be
#: non-empty.  The two guards in `_operating_point` are that reason in
#: executable form: a rebuilt fixture that made this hour all-on would turn
#: those assertions into statements about the empty set, which pass while
#: testing nothing.
HOUR = 12


def _operating_point(case):
    """A committed hour of the fixture, with a previous dispatch that leaves room."""
    pmin = np.asarray(case.unit_p_min, np.float64)
    pmax = np.asarray(case.unit_p_max, np.float64)
    fx = np.load(FIXTURE, allow_pickle=True)
    u = fx["commitment"][0, :, HOUR].astype(np.float64)
    assert (u == 0).any(), \
        "this operating point commits every unit, so the de-committed assertions " \
        "would be assertions about the empty set"
    assert (u > 0).any(), "this operating point commits no unit"
    dfrac = 0.85
    demand = float((pmin * u).sum()) + dfrac * float(((pmax - pmin) * u).sum())
    p_prev = (pmin + dfrac * (pmax - pmin)) * u
    return u, demand, p_prev


@pytest.fixture
def built(x64):
    case = load_case(CASE)
    _, cost = segment_costs(case, 1)
    clear, spec = make_clearing(case, THETA, VOLR, n_segments=1, cap_scale=0.6,
                                ramp_scale=1.0, period_hours=0.5)
    u, demand, p_prev = _operating_point(case)
    args = (jnp.asarray(cost), jnp.zeros((spec["n_units"], spec["n_prod"])),
            jnp.asarray(u), jnp.asarray(demand),
            jnp.asarray(BETA * demand), jnp.asarray(p_prev))
    return clear, spec, args


def test_float32_construction_is_refused():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="float64"):
            make_clearing(load_case(CASE), THETA, VOLR)
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_volr_at_or_above_voll_is_refused(x64):
    """§7 derives the ordering from the one every operator applies."""
    case = load_case(CASE)
    for bad in (10_000.0, 20_000.0, 0.0, -1.0):
        with pytest.raises(ValueError, match="volr"):
            make_clearing(case, THETA, bad)


def test_response_times_are_validated(x64):
    case = load_case(CASE)
    for bad in ((), (0.0, 0.5), (-1.0,)):
        with pytest.raises(ValueError, match="theta|response time"):
            make_clearing(case, bad, VOLR)


def test_jit(built):
    clear, spec, args = built
    out = jax.jit(clear)(*args)

    n_u, n_b, n_p = spec["n_units"], spec["n_buses"], spec["n_prod"]
    assert out["award"].shape == (n_u,)
    assert out["reserve"].shape == (n_u, n_p)
    assert out["shed"].shape == (n_b,)
    assert out["lmp"].shape == (n_b,)
    assert out["reserve_price"].shape == (n_p,)
    assert out["capacity_dual"].shape == (n_u,)
    assert out["reserve_shortfall"].shape == (n_p,)
    for name, value in out.items():
        assert value.dtype == jnp.float64, name
        assert jnp.isfinite(value).all(), name


def test_shape_matches_the_layout_of_the_specification(built):
    """§16: N_var = N_unit (K + N_prod) + N_bus + N_prod, one balance row."""
    _, spec, _ = built
    n_u, n_b, n_p, K = (spec["n_units"], spec["n_buses"], spec["n_prod"], spec["K"])
    assert spec["n"] == n_u * (K + n_p) + n_b + n_p
    assert spec["m"] == 2 * spec["n_l"] + 3 * n_u + n_p + 2 * spec["n"]
    assert spec["n_eq"] == 1


def test_pytree_structure_is_independent_of_the_data(built):
    clear, _, args = built
    offer, offer_res, u, demand, d_res, p_prev = args
    slack = jax.tree.structure(clear(offer, offer_res, u, 0.5 * demand,
                                     0.1 * d_res, p_prev))
    tight = jax.tree.structure(clear(offer, offer_res, u, demand,
                                     3.0 * d_res, p_prev))
    assert slack == tight


def test_vmap_over_parallel_environments(built):
    clear, spec, args = built
    offer, offer_res, u, demand, d_res, p_prev = args
    offers = offer[None] * jnp.linspace(0.8, 1.6, BATCH)[:, None, None]

    batched = jax.jit(jax.vmap(clear, in_axes=(0, None, None, None, None, None)))
    out = batched(offers, offer_res, u, demand, d_res, p_prev)

    assert out["award"].shape == (BATCH, spec["n_units"])
    assert out["reserve"].shape == (BATCH, spec["n_units"], spec["n_prod"])
    assert jnp.isfinite(out["award"]).all()
    assert float(jnp.max(out["mu"])) < 1e-5


def test_identical_lanes_are_bit_identical(built):
    """A fixed trip count makes the batch cost independent of its content."""
    clear, _, args = built
    offer, offer_res, u, demand, d_res, p_prev = args
    repeated = jnp.broadcast_to(offer, (BATCH,) + offer.shape)

    out = jax.jit(jax.vmap(clear, in_axes=(0, None, None, None, None, None)))(
        repeated, offer_res, u, demand, d_res, p_prev)

    for key in ("award", "reserve", "lmp", "reserve_price"):
        value = np.asarray(out[key])
        np.testing.assert_array_equal(
            value, np.broadcast_to(value[0], value.shape), err_msg=key)


def test_repeated_calls_are_bit_identical(built):
    clear, _, args = built
    first = jax.jit(clear)(*args)
    second = jax.jit(clear)(*args)
    for key in ("award", "reserve", "lmp", "reserve_price", "z"):
        np.testing.assert_array_equal(np.asarray(first[key]),
                                      np.asarray(second[key]), err_msg=key)


def test_scan_over_the_periods_of_an_episode(built):
    clear, _, args = built
    offer, offer_res, u, demand, d_res, p_prev = args

    def step(carry, factor):
        out = clear(offer, offer_res, u, demand * factor, d_res, carry)
        return out["award"], out["reserve_price"]

    final, prices = lax.scan(step, p_prev, jnp.linspace(0.98, 1.02, T))
    assert final.shape == p_prev.shape
    assert prices.shape == (T, len(THETA))
    assert jnp.isfinite(prices).all()


def test_exactly_one_clearing_per_call(built):
    """Counted as the environment layer's acceptance criteria define it: scans of the traced
    function whose trip count equals the solver iteration limit.  Counting every
    scan or searching for a while loop both give the wrong number."""
    clear, _, args = built
    jaxpr = jax.make_jaxpr(clear)(*args)
    trips = []

    def walk(closed):
        for eqn in closed.eqns:
            if eqn.primitive.name == "scan":
                trips.append(eqn.params["length"])
            for sub in jax.extend.core.jaxprs_in_params(eqn.params):
                walk(sub)

    walk(jaxpr.jaxpr)
    assert sum(1 for t in trips if t == MAX_ITER) == 1, trips
