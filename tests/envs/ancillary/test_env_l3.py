"""L3: step-by-step comparison against the reference under one action sequence.

L2 compares one clearing at a time on inputs a test constructs.  What it cannot
see is the part of the environment that carries state: the previous dispatch
(RMP) reads, the requirement built from the forecast of the period the cursor
points at, the commitment gathered for that period, and the day-ahead position
the settlement nets against.  A wrong index or a stale carry leaves every single
clearing correct and the trajectory wrong, which is exactly the gap this layer
exists to close.

The reference is driven from the environment's own state rather than from a
second copy of the bookkeeping, because a second copy would be the same
bookkeeping written twice and would agree with the first for the wrong reason.
What is independent here is the clearing and the settlement arithmetic; what is
shared is only the state the environment reports.

There is no L4.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.ancillary.clearing import MAX_ITER, REG_COEF
from powermarketjax.envs.ancillary.env import AncillaryParams, make_ancillary_env
from powermarketjax.envs.ancillary.requirement import make_requirement
from powermarketjax.envs.day_ahead.action import make_offer_map
from powermarketjax.envs.day_ahead.clearing import segment_costs
from tests.envs.ancillary.reference import clear_reference
from tests.envs.ancillary.test_env_l0 import (BETA, CASE, DELTA, EPISODE,
                                              PI_SCALE, THETA, VOLR, _action,
                                              _build)

STEPS = 6
#: measured over the six steps below, 2026-08-15
ATOL = dict(award=1e-9, reserve=5.0, lmp=1e-6, reserve_price=1e-6,
            reserve_revenue=1e-4)


@pytest.fixture
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def test_a_trajectory_matches_the_reference_step_by_step(x64):
    case = load_case(CASE)
    _, cost = segment_costs(case, 1)
    (reset, step, _, spec), params = _build()
    energy_map, _ = make_offer_map(case, 1, 1, kind="markup", markup_max=2.0)
    requirement = make_requirement(BETA)
    n_prod = spec["n_prod"]
    bus = np.asarray(spec["clearing_spec"]["unit_bus"])

    _, state = jax.jit(reset)(jax.random.PRNGKey(0), params)
    # a sequence that varies, so a stale carry cannot pass by coincidence
    actions = [_action(energy=1.0 + 0.1 * t, reserve=0.3 + 0.05 * t, spread=0.2)
               for t in range(STEPS)]

    seen_reserve = False
    for t, action in enumerate(actions):
        cursor = int(state.cursor)
        p_prev = np.asarray(state.p_prev)
        obs, state, reward, costs, done, info = jax.jit(step)(
            jax.random.PRNGKey(100 + t), state, action, params)
        assert bool(info["converged"]), f"step {t} did not solve"

        # rebuild the same period's inputs from the state the step started at
        # the markup map takes one multiplier per unit, shape (n_units,)
        offer = np.asarray(energy_map(jnp.asarray(action)[:, 0]))[:, :, 0]
        offer_res = np.asarray(jax.nn.softplus(action[:, 1:]) * PI_SCALE)
        u = np.asarray(params.commitment[cursor])
        demand = float(params.demand[cursor])
        d_res = np.asarray(requirement(params.forecast[cursor]))
        want = clear_reference(case, offer, offer_res, u, demand, d_res, p_prev,
                               theta=THETA, volr=VOLR, cap_scale=0.6,
                               ramp_scale=1.0, period_hours=DELTA,
                               max_iter=MAX_ITER, dual_start="cost_norm",
                               reg_coef=REG_COEF)

        np.testing.assert_allclose(np.asarray(info["lmp"]), want["lmp"],
                                   rtol=0, atol=ATOL["lmp"], err_msg=f"lmp t={t}")
        np.testing.assert_allclose(np.asarray(info["reserve_price"]),
                                   want["reserve_price"], rtol=0,
                                   atol=ATOL["reserve_price"],
                                   err_msg=f"reserve_price t={t}")
        # the award is what the next period's (RMP) reads, so a mismatch here
        # is what would drift the trajectory
        np.testing.assert_allclose(np.asarray(state.award_prev)
                                   if not bool(done) else want["award"],
                                   want["award"], rtol=0, atol=ATOL["award"],
                                   err_msg=f"award t={t}")
        # money, per the rule that a quantity difference is only an error once
        # it is a revenue difference.  Both sides are taken from their own
        # solver: mixing the price of one with the quantity of the other would
        # leave the quantity of the environment untested.
        rev = DELTA * (want["reserve"] * want["reserve_price"]).sum(1)
        got_rev = DELTA * (np.asarray(info["reserve"])
                           * np.asarray(info["reserve_price"])[None, :]).sum(1)
        np.testing.assert_allclose(got_rev, rev, rtol=0,
                                   atol=ATOL["reserve_revenue"],
                                   err_msg=f"reserve revenue t={t}")
        np.testing.assert_allclose(np.asarray(info["reserve"]), want["reserve"],
                                   rtol=0, atol=ATOL["reserve"],
                                   err_msg=f"reserve t={t}")
        seen_reserve |= float(want["reserve"].sum()) > 1.0

    assert seen_reserve, "no reserve cleared in any step, the comparison is empty"


def test_the_carry_advances_rather_than_repeating(x64):
    """A trajectory whose carry never moves would pass the comparison above."""
    (reset, step, _, _), params = _build()
    _, state = jax.jit(reset)(jax.random.PRNGKey(0), params)
    cursors, carries = [], []
    for t in range(min(STEPS, EPISODE)):
        cursors.append(int(state.cursor))
        carries.append(np.asarray(state.p_prev).copy())
        _, state, *_ = jax.jit(step)(jax.random.PRNGKey(200 + t), state,
                                     _action(energy=1.0 + 0.1 * t), params)
    assert cursors == list(range(cursors[0], cursors[0] + len(cursors)))
    moved = [not np.array_equal(carries[i], carries[i + 1])
             for i in range(len(carries) - 1)]
    assert all(moved), f"the dispatch carry repeated between steps: {moved}"
