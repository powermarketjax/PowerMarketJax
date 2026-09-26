"""L1 domain correctness for the day-ahead action map (§9.3).

The rule this module exists to keep is §5's: offers are non-decreasing in the
segment index.  §12 says the action map is what enforces it and that no repair
step may exist anywhere, so the tests here are about the property holding for
*every* action rather than for a sampled few, and about the map staying smooth,
since a repair is exactly what a smooth map does not do.

The case that matters is `case118` at K=3, where four units have a negative raw
segment cost.  Scaling the increments by those raw costs would send the curve
downwards, and §19 records that nothing downstream detects a non-monotone offer:
the balance dual stays the true derivative of the objective either way, so the
money-balance identity and the finite-difference check both pass.

x64 per the L0 module docstring: other modules in this suite turn it off.
"""
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.day_ahead import (make_clearing, make_offer_map, segment_costs,
                                           truthful_action)

T = 3
MARKUP_MAX = 3.0
#: Actions far outside anything a trained policy would emit.  The point is that
#: monotonicity is a property of the map, not of the policy.
EXTREMES = [-1e3, -30.0, -1.0, 0.0, 1.0, 30.0, 1e3]
#: Below this, `softplus` and its derivative are both exactly zero in float64
#: (measured: 9.9e-305 at -700, 0.0 at -740).  That is the exponential
#: underflowing, not a repair, but it does mean an action driven that far below
#: zero stops producing gradient.
SOFTPLUS_FLOOR = -700.0


@pytest.fixture(scope="module", autouse=True)
def x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


@pytest.fixture(scope="module")
def case(x64):
    return load_case("29gb")


def _monotone(offer):
    """Worst decrease along the segment axis; zero or positive means monotone.

    K=1 leaves nothing to compare, and that is the default segment count, so it
    is a case worth running rather than excluding."""
    d = np.diff(np.asarray(offer), axis=1)
    return float(d.min()) if d.size else 0.0


@pytest.mark.parametrize("K", [1, 2, 3, 5, 10])
def test_full_offer_is_monotone_at_every_extreme(case, K):
    offer_map, spec = make_offer_map(case, K, T, kind="full")
    for value in EXTREMES:
        offer = offer_map(jnp.full(spec["shape"], value, jnp.float64))
        assert _monotone(offer) >= 0.0, f"alpha={value}, K={K}"


@pytest.mark.parametrize("K", [2, 3, 5, 10])
def test_full_offer_is_monotone_under_random_actions(case, K):
    offer_map, spec = make_offer_map(case, K, T, kind="full")
    action = 20.0 * jax.random.normal(jax.random.PRNGKey(7), spec["shape"],
                                      jnp.float64)
    assert _monotone(offer_map(action)) >= 0.0


@pytest.mark.parametrize("K", [3, 5, 10])
def test_monotone_on_the_case_with_negative_segment_costs(K):
    """`case118`: 4 of 54 units have a negative raw segment cost at K=3.

    Scaling the increments by the raw cost instead of the envelope makes the
    offer descend on exactly those units, which is the failure this map is
    built to avoid and which no downstream check would report.
    """
    case = load_case("118")
    _, raw = segment_costs(case, K, monotone=False)
    assert (raw < 0).any(), "case118 lost its negative segment costs"

    offer_map, spec = make_offer_map(case, K, T, kind="full")
    for value in (-1e3, 0.0, 5.0):
        assert _monotone(offer_map(jnp.full(spec["shape"], value, jnp.float64))) >= 0.0

    # and the raw-cost version really would fail, so the test above has teeth
    envelope = np.maximum.accumulate(raw, axis=1)
    naive = envelope[:, :1, None] + np.cumsum(
        np.asarray(jax.nn.softplus(jnp.zeros((raw.shape[0], K, T)))) * raw[:, :, None],
        axis=1)
    assert np.diff(naive, axis=1).min() < 0.0


def test_markup_of_one_is_the_truthful_offer(case):
    """alpha = 1 reproduces the envelope exactly, which is the offer the
    clearing tests bid.  This is what ties the map to the verified operator."""
    K = 3
    offer_map, _ = make_offer_map(case, K, T, kind="markup", markup_max=MARKUP_MAX)
    n_units = len(np.asarray(case.unit_p_min))
    _, truthful = segment_costs(case, K)          # the monotone envelope
    got = np.asarray(offer_map(jnp.ones((n_units,), jnp.float64)))
    for t in range(T):
        np.testing.assert_array_equal(got[:, :, t], truthful)


def test_markup_scales_the_offer_and_the_clearing_price(case):
    """A uniform markup on every unit raises every offer by that factor, and
    with no line binding and no bus shedding it raises the price by it too."""
    K, alpha = 1, 1.4
    n_units = len(np.asarray(case.unit_p_min))
    offer_map, _ = make_offer_map(case, K, T, kind="markup", markup_max=MARKUP_MAX)
    truthful = offer_map(jnp.ones((n_units,), jnp.float64))
    marked = offer_map(jnp.full((n_units,), alpha, jnp.float64))
    np.testing.assert_allclose(np.asarray(marked), alpha * np.asarray(truthful),
                               rtol=1e-13)

    clear, spec = make_clearing(case, T, n_segments=K, cap_scale=1.0, ramp_scale=2.0)
    args = (jnp.ones((n_units, T)), jnp.full((T,), 33466.0),
            jnp.asarray(spec["p_min"]))
    base = jax.jit(clear)(truthful, *args)
    up = jax.jit(clear)(marked, *args)
    assert float(base["mu"]) < 1e-8 and float(up["mu"]) < 1e-8
    assert np.asarray(base["shed"]).max() < 1e-6
    np.testing.assert_allclose(np.asarray(up["lmp"]), alpha * np.asarray(base["lmp"]),
                               rtol=1e-9)


def test_offer_increases_with_the_action(case):
    K = 3
    offer_map, spec = make_offer_map(case, K, T, kind="full")
    low = np.asarray(offer_map(jnp.full(spec["shape"], -1.0, jnp.float64)))
    high = np.asarray(offer_map(jnp.full(spec["shape"], 1.0, jnp.float64)))
    assert (high > low).all()


def test_map_is_smooth_so_a_repair_would_show(case):
    """The derivative of every offer with respect to its own action is strictly
    positive, down to the point where the exponential itself underflows.

    A clip, a `maximum`, or any other repair step (§12) would flatten the
    gradient over a whole region, which is the region a policy has to cross
    while learning.  `softplus` never does; it only runs out of float64 below
    about -740, which is what `SOFTPLUS_FLOOR` records.
    """
    K = 3
    offer_map, spec = make_offer_map(case, K, T, kind="full")

    def first_segment(action):
        return offer_map(action)[0, 0, 0]

    for value in (SOFTPLUS_FLOOR, -50.0, 0.0, 50.0):
        g = jax.grad(first_segment)(jnp.full(spec["shape"], value, jnp.float64))
        assert float(g[0, 0, 0]) > 0.0, f"gradient vanished at alpha={value}"


def test_construction_refuses_a_negative_envelope():
    """No case in the repository has a negative first segment cost, so the
    envelope stays nonnegative and the increments keep their sign.  A case that
    broke that would produce a descending offer with nothing to report it, so
    the map refuses to build instead."""
    bad = types.SimpleNamespace(
        unit_p_min=np.array([0.0]), unit_p_max=np.array([100.0]),
        unit_cost_a=np.array([0.0]), unit_cost_b=np.array([0.0]),
        unit_cost_c=np.array([-5.0]))          # marginal cost below zero
    with pytest.raises(ValueError, match="negative"):
        make_offer_map(bad, 2, T, kind="full")


def test_envelope_distortion_matches_the_recorded_measurement(case):
    """§19 chooses K against this: the envelope never lowers an offer, and on
    `case29gb` it touches 14 of 66 units at K=2, 43 at K=3 and all 66 at K=5,
    raising offers by up to 22, 34 and 45 \\$/MWh at K=3, 5 and 10."""
    expected_units = {2: 14, 3: 43, 5: 66}
    for K, n_touched in expected_units.items():
        _, raw = segment_costs(case, K, monotone=False)
        envelope = np.maximum.accumulate(raw, axis=1)
        assert (envelope >= raw).all()
        assert int((envelope > raw + 1e-9).any(1).sum()) == n_touched, f"K={K}"

    for K, gap in {3: 22.0, 5: 34.0, 10: 45.0}.items():
        _, raw = segment_costs(case, K, monotone=False)
        lift = (np.maximum.accumulate(raw, axis=1) - raw).max()
        assert gap - 1.0 <= lift <= gap + 1.0, f"K={K}: envelope lifts {lift:.1f}"


# --------------------------------------------------------------------------
# The truthful action.  `learner_mask` fixes part of the
# population to true-cost bidding, and it does that by handing those agents an
# *action*, so the action that reproduces the envelope has to exist and be exact.
# --------------------------------------------------------------------------

def test_truthful_action_reproduces_the_envelope(case):
    """The offer of the truthful action is the envelope itself, to float64.

    This is not the obvious action, and the obvious guess -- zero -- is wrong by
    69%: `softplus(0)` is 0.693, so `alpha = 0` prices the first segment at
    1.693 times cost.  The map has no finite action that returns a zero
    increment, and the first segment therefore needs the underflow sentinel
    `ZERO_INCREMENT`; measured here at K = 1, 2, 3 and 5.
    """
    for K in (1, 2, 3, 5):
        offer_map, _ = make_offer_map(case, K, T, kind="full")
        got = np.asarray(offer_map(truthful_action(case, K, T, kind="full")))
        _, envelope = segment_costs(case, K)
        err = np.abs(got - np.asarray(envelope)[:, :, None]).max()
        assert err < 1e-13, f"K={K}: truthful offer is out by {err:.3e} $/MWh"
        assert _monotone(got) >= 0.0

    offer_map, _ = make_offer_map(case, 3, T, kind="markup", markup_max=2.0)
    got = np.asarray(offer_map(truthful_action(case, 3, T, kind="markup")))
    _, envelope = segment_costs(case, 3)
    np.testing.assert_array_equal(got, np.broadcast_to(
        np.asarray(envelope)[:, :, None], got.shape))


def test_zero_action_is_not_truthful(case):
    """The discriminating half of the test above.

    If `truthful_action` returned zeros -- or anything else finite for the first
    segment -- the offer would sit strictly above cost, and a non-learning agent
    would be marking up while the population was described as truthful.
    """
    offer_map, _ = make_offer_map(case, 2, T, kind="full")
    naive = np.asarray(offer_map(jnp.zeros((66, 2, T))))
    _, envelope = segment_costs(case, 2)
    ratio = (naive[:, 0, 0] / np.asarray(envelope)[:, 0]).max()
    assert ratio > 1.6, f"zero action prices the first segment at {ratio:.3f} x cost"


def test_truthful_action_on_a_case_with_negative_segment_costs():
    """`case118` at K=3 is where the envelope is flat over whole stretches.

    A flat stretch needs a zero increment, which is the same underflow the first
    segment needs, so this checks the branch rather than assuming it: four of the
    54 units have a negative raw segment cost there, and the envelope's flat runs
    are what an inverse softplus cannot express directly.
    """
    case118 = load_case("118")
    K = 3
    _, raw = segment_costs(case118, K, monotone=False)
    envelope = np.maximum.accumulate(raw, axis=1)
    flat = (np.diff(envelope, axis=1) <= 0).sum()
    assert flat > 0, "case118 K=3 should have flat envelope stretches"
    offer_map, _ = make_offer_map(case118, K, T, kind="full")
    got = np.asarray(offer_map(truthful_action(case118, K, T, kind="full")))
    err = np.abs(got - envelope[:, :, None]).max()
    assert err < 1e-13, f"truthful offer is out by {err:.3e} $/MWh"
