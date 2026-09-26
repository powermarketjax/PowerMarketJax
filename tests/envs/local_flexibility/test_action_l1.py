"""L1 domain correctness for the local flexibility action map (§9.3, §5, §9.5).

Four groups.

**The action box loses one submission and no outcome.**  `spec` publishes
`-+ACTION_SATURATION` on all three coordinates, so a learner that squashes into
it cannot submit what the map returns outside; on five of the six ends there is
nothing outside to submit, and the sixth is answered on the quantity coordinate
rather than on the price.

**§5 holds for every real action, without a repair.**  The offered quantity
never exceeds the deliverable quantity, the planned charging never exceeds the
headroom, and neither is negative -- for actions far outside any range a policy
would produce, since the map is defined on all of R^3 and nothing clips it.

**The non-learner's baseline is reached exactly.**  `(-128, +128, -128)` has
to give the replacement cost itself, the full deliverable quantity and zero
planned charging, bitwise: it is the non-learning bidder of the whole market,
and a baseline that only approaches truthfulness would make every measurement
against it approximate.

**The envelope agrees with the vendored one.**  §5 and §9.3 write the battery
envelope out in closed form and `compute_feasible_power_batch` computes the
same thing, so clipping a realised power against the vendored envelope must be
a no-op.  That comparison is the reason the closed form is allowed to exist
here at all, and it is what would catch an efficiency on the wrong side of a
division.  The tolerance is not zero even so: at the upper edge the realised
power is `(plan + discharge) - plan`, and that reassociation may cost one
float32 rounding.  Measured over 50 action draws at exactly that edge it costs
none -- the difference is 0.0 MW -- so the bound asserted, four float32 epsilon
of the largest deliverable quantity (3.7e-7 MW here), is derived from the
arithmetic rather than from an observed failure, and a violation of it would
mean the two envelopes disagree in substance.

**The replacement cost is §9.5's, not the settlement's.**  `c_rep` charges
degradation twice and the replacement energy once, both inflated by the round
trip; getting either inflation wrong leaves every number finite and the markup
measured against the wrong basis.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.envs.local_flexibility import make_action_map
from powermarketjax.envs.local_flexibility.action import ACTION_SATURATION
from powermarketjax.envs.local_flexibility.env import BASELINE_ACTION
from powermarketjax.resources.battery import (compute_feasible_power_batch,
                                              make_battery_bundle)

N = 12
DELTA = 0.25


@pytest.fixture(autouse=True)
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture(scope="module")
def bundle():
    rng = np.random.default_rng(1)
    return make_battery_bundle(
        n_devices=N, dt_hours=DELTA,
        capacity_mwh=rng.uniform(0.2, 2.0, N).tolist(),
        power_mw=rng.uniform(0.05, 0.5, N).tolist(),
        soc_min=0.1, soc_max=0.9, eta_charge=0.94, eta_discharge=0.92)


@pytest.fixture(scope="module")
def act_map():
    return make_action_map(N, DELTA)[0]


def soc_and_cost(seed=0):
    rng = np.random.default_rng(seed)
    return (jnp.asarray(rng.uniform(0.1, 0.9, N), jnp.float32),
            jnp.asarray(rng.uniform(2.0, 20.0, N), jnp.float32))


@pytest.mark.parametrize("spread", [1.0, 5.0, 50.0])
def test_submission_satisfies_section_5(act_map, bundle, spread):
    """(CAP)'s bound, the charging headroom, and non-negativity."""
    soc, cycle = soc_and_cost()
    rng = np.random.default_rng(int(spread))
    action = jnp.asarray(rng.normal(0.0, spread, (N, 3)), jnp.float32)
    out = jax.jit(act_map)(action, soc, jnp.float32(60.0), bundle, cycle)

    assert (np.asarray(out["qty_max"]) >= 0.0).all()
    assert (np.asarray(out["plan"]) >= 0.0).all()
    assert (np.asarray(out["qty_max"]) <= np.asarray(out["q_phys"])).all()
    assert (np.asarray(out["plan"]) <= np.asarray(out["charge_headroom"])).all()
    # §5: what can be forgone plus what can be discharged
    discharge = np.minimum(
        np.asarray(bundle.power_max),
        np.asarray(bundle.eta_discharge) * np.asarray(bundle.capacity)
        * (np.asarray(soc) - np.asarray(bundle.soc_min)) / DELTA)
    np.testing.assert_allclose(np.asarray(out["q_phys"]),
                               np.asarray(out["plan"]) + discharge, rtol=1e-6)


def test_baseline_action_saturates_exactly(act_map, bundle):
    """The baseline: truthful price, nothing withheld, no positioning."""
    soc, cycle = soc_and_cost()
    action = jnp.tile(jnp.asarray(BASELINE_ACTION, jnp.float32), (N, 1))
    out = jax.jit(act_map)(action, soc, jnp.float32(60.0), bundle, cycle)

    np.testing.assert_array_equal(np.asarray(out["price"]),
                                  np.asarray(out["c_rep"]))
    np.testing.assert_array_equal(np.asarray(out["qty_max"]),
                                  np.asarray(out["q_phys"]))
    assert (np.asarray(out["plan"]) == 0.0).all()
    # and the deliverable quantity is then the discharge term alone
    assert (np.asarray(out["q_phys"]) > 0.0).all()


def test_price_is_strictly_above_cost_away_from_saturation(act_map, bundle):
    """§9.3: softplus is positive, so offering below cost is outside the map."""
    soc, cycle = soc_and_cost()
    rng = np.random.default_rng(3)
    action = jnp.asarray(rng.uniform(-20.0, 20.0, (N, 3)), jnp.float32)
    out = jax.jit(act_map)(action, soc, jnp.float32(60.0), bundle, cycle)
    assert (np.asarray(out["price"]) > np.asarray(out["c_rep"])).all()


def test_replacement_cost_matches_section_9_5(act_map, bundle):
    soc, cycle = soc_and_cost()
    price_e = 73.5
    out = act_map(jnp.zeros((N, 3), jnp.float32), soc, jnp.float32(price_e),
                  bundle, cycle)
    eta_rt = np.asarray(bundle.eta_charge) * np.asarray(bundle.eta_discharge)
    expected = np.asarray(cycle) * (1.0 + 1.0 / eta_rt) + price_e / eta_rt
    np.testing.assert_allclose(np.asarray(out["c_rep"]), expected, rtol=1e-6)


@pytest.mark.parametrize("fraction", [0.0, 0.37, 1.0])
def test_realised_power_lies_inside_the_vendored_envelope(act_map, bundle,
                                                          fraction):
    """The clip of `compute_feasible_power_batch` is a no-op on this map.

    `fraction` is the share of the offered quantity that clears, so 1.0 is the
    upper edge where the reassociation costs one rounding (module docstring).
    """
    soc, cycle = soc_and_cost(5)
    rng = np.random.default_rng(7)
    action = jnp.asarray(rng.normal(0.0, 3.0, (N, 3)), jnp.float32)
    out = jax.jit(act_map)(action, soc, jnp.float32(60.0), bundle, cycle)

    award = fraction * out["qty_max"]
    p_signed = award - out["plan"]
    clipped = compute_feasible_power_batch(
        soc, p_signed, bundle.power_max, bundle.capacity, bundle.soc_min,
        bundle.soc_max, bundle.eta_charge, bundle.eta_discharge, DELTA)

    tolerance = 4.0 * np.finfo(np.float32).eps * float(
        np.max(np.abs(np.asarray(out["q_phys"]))))
    np.testing.assert_allclose(np.asarray(clipped), np.asarray(p_signed),
                               rtol=0.0, atol=tolerance)


def test_each_component_moves_only_what_section_9_3_says(act_map, bundle):
    """Price in the first, quantity in the second, charging in the third.

    The third moves the deliverable quantity as well, and that is §5 rather
    than a leak: planned charging is part of what can be delivered.
    """
    soc, cycle = soc_and_cost(9)
    base = jnp.zeros((N, 3), jnp.float32)
    out0 = act_map(base, soc, jnp.float32(60.0), bundle, cycle)

    raised_price = act_map(base.at[:, 0].add(1.0), soc, jnp.float32(60.0),
                           bundle, cycle)
    assert (np.asarray(raised_price["price"]) > np.asarray(out0["price"])).all()
    np.testing.assert_array_equal(np.asarray(raised_price["qty_max"]),
                                  np.asarray(out0["qty_max"]))

    raised_qty = act_map(base.at[:, 1].add(1.0), soc, jnp.float32(60.0),
                         bundle, cycle)
    assert (np.asarray(raised_qty["qty_max"]) > np.asarray(out0["qty_max"])).all()
    np.testing.assert_array_equal(np.asarray(raised_qty["price"]),
                                  np.asarray(out0["price"]))

    raised_plan = act_map(base.at[:, 2].add(1.0), soc, jnp.float32(60.0),
                          bundle, cycle)
    assert (np.asarray(raised_plan["plan"]) > np.asarray(out0["plan"])).all()
    assert (np.asarray(raised_plan["q_phys"]) > np.asarray(out0["q_phys"])).all()


def test_empty_and_full_battery_are_the_two_degenerate_offers(act_map, bundle):
    """At `soc_min` only forgone charging is deliverable; at `soc_max` none is
    plannable, so the offer is the discharge alone."""
    _, cycle = soc_and_cost(11)
    action = jnp.zeros((N, 3), jnp.float32)

    empty = act_map(action, bundle.soc_min, jnp.float32(60.0), bundle, cycle)
    np.testing.assert_array_equal(np.asarray(empty["q_phys"]),
                                  np.asarray(empty["plan"]))

    full = act_map(action, bundle.soc_max, jnp.float32(60.0), bundle, cycle)
    assert (np.asarray(full["charge_headroom"]) == 0.0).all()
    assert (np.asarray(full["plan"]) == 0.0).all()
    assert (np.asarray(full["q_phys"]) > 0.0).all()


#: How far outside the box the comparison action is taken.  The map is defined
#: on all of R^3, so what the box has to answer is what a learner would have
#: submitted at an action the box refuses; 1e6 is four orders past the box and
#: exactly representable in float32.
BEYOND_THE_BOX = 1e6


def test_the_box_ends_submit_what_the_unbounded_map_submits(act_map, bundle):
    """Five of the six ends lose nothing, and the sixth loses no outcome.

    The four submitted quantities at the two corners of the box are compared
    with the same four at `-+BEYOND_THE_BOX`, **bitwise**, because the claim is
    exactness and not closeness: softplus and sigmoid have saturated by `-+128`
    because `exp(-128)` underflows and `exp(+128)` overflows float32, and that
    reason is the floating point format rather than a library's choice of formula.

    The one end that is not a saturation is `alpha_pi = +ACTION_SATURATION`,
    where softplus is the identity: the price there is exactly
    `(1 + ACTION_SATURATION) c_rep` and an action outside the box prices higher
    still, so the box does lose submissions.  It loses no outcome, and what
    carries that is the **quantity** coordinate rather than the price:
    `alpha_q = -ACTION_SATURATION` offers exactly zero megawatts, and an offer
    of nothing clears nothing at any price -- asserted on the clearing itself
    in `test_env_l1.py`.  The price route is available too at the adopted
    scenario, where `129 c_rep` is 19 334.8 $/MWh against a `VOLL` of 10 000,
    but that is a fact about `c_rep` and not about the box, so it is not what
    is asserted here.

    **The bite, measured 2026-09-10 on this fixture.**  Under a box of `-+1`
    all seven comparisons below fail rather than pass: `price` at the floor is
    off by 33.58 $/MWh (31.3% of the replacement cost), `qty_max` at the floor
    by 0.138 MW against an exact zero, `qty_max` at the ceiling by 0.297 MW
    (46.2%) and `plan` at the ceiling by 0.132 MW (26.9%).  The smallest of the
    seven gaps is 0.132 MW, which is what the closing assertion bites at, so
    the equalities above are not vacuous at the width they are asserted at.
    """
    soc, cycle = soc_and_cost()
    f = jax.jit(act_map)

    def corner(v):
        action = jnp.tile(jnp.asarray((v, v, v), jnp.float32), (N, 1))
        return f(action, soc, jnp.float32(60.0), bundle, cycle)

    S = ACTION_SATURATION
    exact = ("qty_max", "plan", "q_phys")
    lo_end, lo_far = corner(-S), corner(-BEYOND_THE_BOX)
    hi_end, hi_far = corner(+S), corner(+BEYOND_THE_BOX)
    for key in exact + ("price",):
        np.testing.assert_array_equal(np.asarray(lo_end[key]),
                                      np.asarray(lo_far[key]), err_msg=key)
    for key in exact:
        np.testing.assert_array_equal(np.asarray(hi_end[key]),
                                      np.asarray(hi_far[key]), err_msg=key)
    # the end that is not a saturation, asserted as the exact multiple it is
    np.testing.assert_allclose(np.asarray(hi_end["price"]),
                               (1.0 + S) * np.asarray(hi_end["c_rep"]),
                               rtol=1e-6)
    # the coordinate that carries declining to sell, at the box's own floor
    assert (np.asarray(lo_end["qty_max"]) == 0.0).all()
    # nothing here is degenerate: there is a quantity to withhold
    assert (np.asarray(hi_end["q_phys"]) > 0.0).all()

    # the other side of the discrimination: the same seven comparisons at +-1
    narrow_lo, narrow_hi = corner(-1.0), corner(+1.0)
    gaps = [float(np.abs(np.asarray(narrow_lo[k])
                         - np.asarray(lo_far[k])).max())
            for k in exact + ("price",)]
    gaps += [float(np.abs(np.asarray(narrow_hi[k])
                          - np.asarray(hi_far[k])).max()) for k in exact]
    assert min(gaps) > 0.1, gaps
