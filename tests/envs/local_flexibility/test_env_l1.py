"""L1 domain correctness for the local flexibility environment layer (item 37).

The operators each carry their own acceptance, so what is checked here is the
wiring this layer adds -- and that wiring contains the two pieces of new domain
arithmetic in the market, so this is not a routing test only.

**§14's spatial mapping and §3.1's injection.**  The feeder total of a period
is allocated to buses in proportion to the registered load vector, and the
photovoltaic output and the planned charging of each aggregator enter at its
own bus.  Both are recomputed here from the case and the series, independently
of the closure, and the published requirement they produce is compared against
the one the step reports.  An error in either leaves every quantity finite: a
transposed allocation, a factor of `base_mva`, or a planned charging that never
reaches the baseline would all clear, settle and report.

**§9.5's transition.**  The state of charge is recomputed from the award and
the planned charging with the analytic form of §9.5 and **without** the clip
`update_soc_batch` ends with, because that clip absorbs exactly the excursion a
wrong envelope would produce and an assertion against the clipped value can
never fail.

**§8's identity, through `costs` and `info`.**  Total payment equals the
clearing's optimal value less what curtailment was charged at `VOLL`, and the
curtailment term is read off the `costs` channel rather than recomputed, which
ties the three outputs together: a `costs` column in the wrong unit breaks the
identity here even though the settlement's own L1 passes.

**The baseline, the truncation and the empty period.**  An all-False
`learner_mask` must make the step ignore its action entirely; `terminal_obs`
must be the true successor observation rather than the restarted one; and a
period in which nothing clears must report an average price of exactly zero
rather than a NaN that would spread across a `vmap` batch.

`case33bw` carries no line ratings, so one test runs on `case533mt_hi`: the
(LIM) rows are invisible on the development case, and a reversed right-hand
side there was caught only on the primary one.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from powermarketjax.case import load_case
from powermarketjax.envs.local_flexibility import (build_voltage_sensitivity,
                                                   draw_agent_buses,
                                                   make_action_map,
                                                   make_local_flex_env,
                                                   make_local_flex_params)
from powermarketjax.envs.local_flexibility.clearing import make_clearing
from powermarketjax.envs.local_flexibility.verification import (
    cleared_injection, make_verification)
from powermarketjax.resources.battery import make_battery_bundle

CASE = "33bw"
N = 6
N_PERIODS = 20
EPISODE_LEN = 4
KAPPA = 2.0
DELTA = 0.25
VOLL = 10_000.0


@pytest.fixture(scope="module", autouse=True)
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def build(case_id=CASE, n_agent=N, kappa=KAPPA, n_periods=N_PERIODS,
          episode_len=EPISODE_LEN, seed=0, pv_scale=0.05, margins=(0.0, 0.0),
          **extra):
    case = load_case(case_id)
    sens = build_voltage_sensitivity(case)
    agent_bus = draw_agent_buses(case, sens, n_agent, seed=seed)
    env = make_local_flex_env(case, sens, agent_bus,
                              voltage_margin=margins[0],
                              thermal_margin=margins[1], period_hours=DELTA)
    rng = np.random.default_rng(seed)
    total = float(np.asarray(case.node_pd, np.float64).sum())
    battery = make_battery_bundle(
        n_devices=n_agent, dt_hours=DELTA,
        capacity_mwh=rng.uniform(0.2, 1.0, n_agent).tolist(),
        power_mw=rng.uniform(0.05, 0.3, n_agent).tolist())
    params = make_local_flex_params(
        load_series=total * rng.uniform(0.9, 1.1, n_periods),
        pv=rng.uniform(0.0, pv_scale, (n_periods, n_agent)),
        energy_price=rng.uniform(30.0, 90.0, n_periods),
        battery=battery, cycle_cost=rng.uniform(5.0, 20.0, n_agent),
        load_scale=kappa, learner_mask=np.ones(n_agent, bool),
        episode_len=episode_len, **extra)
    return env, params, case, sens, agent_bus


@pytest.fixture(scope="module")
def built():
    return build()


def some_action(key, n_agent=N):
    return jax.random.normal(key, (n_agent, 3), jnp.float32)


def recompute(case, sens, agent_bus, params, state, action, kappa=KAPPA,
              margins=(0.0, 0.0), monitor=False):
    """One step, assembled here from the case and the series (module doc).

    The assembly is numpy, which is what makes it a second route rather than
    the same expression twice, and it is why the comparisons carry a tolerance:
    numpy and XLA differ by one float32 ULP on the division that converts an
    aggregator's injection into per unit (measured 1.5e-8 on the operand, 2.9e-8
    relative on the published requirement).  Replacing the numpy scatter by a
    `jnp` one makes the same comparison bitwise, which is the measurement
    behind that statement and also the reason not to do it.
    """
    n_agent = len(agent_bus)
    act_map, _ = make_action_map(n_agent, DELTA)
    clear, _ = make_clearing(case, sens, agent_bus, voltage_margin=margins[0],
                             thermal_margin=margins[1], period_hours=DELTA)
    cursor = int(state.cursor)
    base = float(sens.base_mva)

    pd = np.asarray(case.node_pd, np.float64)
    qd = np.asarray(case.node_qd, np.float64)
    scale = kappa * float(np.asarray(params.load_series)[cursor]) / pd.sum()
    bus_load = scale * pd / base
    bus_q = scale * qd / base

    sub = act_map(action, state.soc, jnp.asarray(params.energy_price)[cursor],
                  params.battery, params.cycle_cost)
    p_base = -bus_load.copy()
    # `monitor` drops the planned charging out of the baseline and nothing
    # else, which is the whole of what the switch is supposed to do
    plan = 0.0 if monitor else np.asarray(sub["plan"])
    np.add.at(p_base, agent_bus,
              (np.asarray(params.pv)[cursor] - plan) / base)

    out = clear(sub["price"], np.asarray(sub["qty_max"]) / base, p_base,
                -bus_q, bus_load)
    return sub, out, bus_load, scale


def test_allocation_injection_and_requirement_match_an_outside_assembly(built):
    """§14 and §3.1, recomputed from the case rather than from the closure."""
    (reset, step, _, _), params, case, sens, agent_bus = built
    key = jax.random.PRNGKey(0)
    _, state = reset(key, params)
    action = some_action(key)

    _, state2, _, _, _, info = jax.jit(step)(key, state, action, params)
    _, out, _, scale = recompute(case, sens, agent_bus, params, state, action)

    for name in ("req_v", "req_th"):
        # rtol from the float32 ULP boundary of `recompute`, three orders
        # above the 2.9e-8 measured there and far below any assembly error
        np.testing.assert_allclose(np.asarray(info[name]),
                                   np.asarray(out[name]), rtol=1e-5, atol=1e-9,
                                   err_msg=name)
    np.testing.assert_allclose(
        np.asarray(state2.award_prev),
        np.asarray(out["award"]) * float(sens.base_mva), rtol=1e-5, atol=1e-9)
    # the own load an aggregator observes is its bus's share of the feeder total
    pd = np.asarray(case.node_pd, np.float64)
    np.testing.assert_allclose(np.asarray(state2.load_prev),
                               scale * pd[agent_bus], rtol=1e-5)


def test_requirement_fields_are_the_published_signals(built):
    """§9.4: own bus, largest on the path to the substation, feeder extremes.

    **And in MW**, which is the point of the `base_mva` factor below.  `info`
    carries the operator's per-unit vectors straight out of `requirement`,
    while the observation is a row of MW -- `award_prev` and `volume_prev` are
    scaled, `pv_prev` and `load_prev` never left MW -- so a requirement column
    left per unit would sit in the same row a hundred times too small on this
    case and be read against the award as if it were the same quantity.  The
    preconditions are what stop the factor from being invisible: at
    ``base_mva`` of one it would cancel, and at an all-zero requirement it
    would multiply nothing.

    **The thermal half is vacuous on this case and is asserted to be**, not
    quietly carried: `case33bw` has the 1e6 sentinel for every rating (§13),
    so ``req_th`` is identically zero here and a wrong factor on it would be
    zero either way.  The MW assertion for that half lives in
    `test_primary_case_one_step_at_the_adopted_configuration`, which runs on
    `case533mt_hi` where the ratings are real.
    """
    (reset, step, _, _), params, case, sens, agent_bus = built
    key = jax.random.PRNGKey(1)
    _, state = reset(key, params)
    _, state2, _, _, _, info = jax.jit(step)(key, state, some_action(key),
                                             params)

    base = float(sens.base_mva)
    assert base != 1.0, base
    req_v, req_th = np.asarray(info["req_v"]), np.asarray(info["req_th"])
    assert req_v.max() > 0.0, req_v.max()
    assert not req_th.any(), (
        "the thermal requirement is no longer identically zero on this case, "
        "so the vacuous half recorded above has stopped being vacuous and the "
        "assertions here should be extended rather than left as they are")

    np.testing.assert_allclose(np.asarray(state2.req_v_own),
                               base * req_v[agent_bus], rtol=1e-6)
    on_path = (sens.A[:, agent_bus] * req_th[:, None]).max(axis=0)
    np.testing.assert_allclose(np.asarray(state2.req_th_own), base * on_path,
                               rtol=1e-6, atol=1e-7)
    assert float(state2.req_v_max) == pytest.approx(base * req_v.max(), rel=1e-6)
    assert int(state2.req_v_count) == int((req_v > 0.0).sum())
    assert int(state2.req_th_count) == int((req_th > 0.0).sum())


def test_monitor_baseline_removes_the_planned_charge_from_what_is_procured():
    """The `monitor_baseline` switch, against the same outside assembly as §14 above.

    The switch is one line in `env.py` and two things have to be true of it,
    not one.  **What it changes**: the requirement the operator publishes is
    the one raised by a feeder on which nobody planned to charge, so it is
    recomputed here from the case and the series with the planned charging
    left out, and compared against what the step reports.  **What it must not
    change**: the planned charge itself still happens, and it still reaches
    the feeder.  The sweep is re-run here on the **real** baseline -- the one
    that carries the charge -- with the award the monitored clearing produced,
    and the violation depths it returns are compared against the ones the step
    reports.  That is the assertion that separates a monitored *operator* from
    a monitored *network*: without it, an implementation that let the switch
    shave the physics as well passes everything else here, which was measured
    before this paragraph was written.  `plan_total` is
    asserted bitwise equal to the unmonitored run, and the state of charge is
    recomputed from §9.5 with **that unchanged plan and the monitored award**,
    which is the precise statement -- the award does move, and the transition
    follows it.  A version that dropped the charge from the physics as well
    would satisfy the first half of this test and fail that recomputation, and
    it is the mistake worth a test.

    **The operating point is chosen so the criterion can fail.**  At the module
    `KAPPA` of 2.0 the voltage requirement is far larger than this fleet can
    offer, so every offer clears under either baseline and the award is pinned
    to the offered quantity: the money assertion below would hold there for a
    switch wired to nothing.  At 1.2 the award tracks the requirement, and the
    preconditions state that rather than trust it.
    """
    kappa = 1.2
    # (price, quantity, planned charge).  A price far below the floor's
    # saturation offers at the replacement cost, the quantity is nearly all
    # of what is deliverable, and the charge is near the top of the headroom.
    charging = jnp.tile(jnp.asarray([-30.0, 5.0, 5.0], jnp.float32), (N, 1))
    key = jax.random.PRNGKey(0)

    (reset, step, _, _), off, case, sens, agent_bus = build(kappa=kappa)
    _, on, *_ = build(kappa=kappa, monitor_baseline=True)
    assert not bool(off.monitor_baseline) and bool(on.monitor_baseline)

    _, state = reset(key, off)
    _, s_off, _, _, _, i_off = jax.jit(step)(key, state, charging, off)
    _, s_on, _, _, _, i_on = jax.jit(step)(key, state, charging, on)

    # preconditions: the arm positions, and the award is not on its ceiling
    sub_off, out_off, _, _ = recompute(case, sens, agent_bus, off, state,
                                       charging, kappa=kappa)
    assert float(i_off["plan_total"]) > 0.0
    awarded = float(np.sum(np.asarray(out_off["award"]))) * float(sens.base_mva)
    assert float(np.sum(np.asarray(sub_off["qty_max"]))) > awarded > 0.0, (
        "the award is at the offered quantity, so the requirement is not what "
        "sets it here and this operating point cannot discriminate")

    # what the switch changes, against the outside assembly
    _, out_on, _, _ = recompute(case, sens, agent_bus, on, state, charging,
                                kappa=kappa, monitor=True)
    for name in ("req_v", "req_th"):
        np.testing.assert_allclose(np.asarray(i_on[name]),
                                   np.asarray(out_on[name]), rtol=1e-5,
                                   atol=1e-9, err_msg=name)
    np.testing.assert_allclose(
        np.asarray(s_on.award_prev),
        np.asarray(out_on["award"]) * float(sens.base_mva), rtol=1e-5,
        atol=1e-9)
    assert float(np.sum(np.asarray(i_on["req_v"]))) < 0.5 * float(
        np.sum(np.asarray(i_off["req_v"])))
    assert float(jnp.sum(s_on.payment_prev)) < 0.5 * float(
        jnp.sum(s_off.payment_prev))

    # what it must not change: the charge is still planned, and the transition
    # is §9.5 on that plan against the award the monitored clearing produced
    assert float(i_on["plan_total"]) == float(i_off["plan_total"])
    sub_on, _, _, _ = recompute(case, sens, agent_bus, on, state, charging,
                                kappa=kappa, monitor=True)
    np.testing.assert_array_equal(np.asarray(sub_on["plan"]),
                                  np.asarray(sub_off["plan"]))
    base = float(sens.base_mva)
    award_mw = np.asarray(out_on["award"], np.float64) * base
    plan_mw = np.asarray(sub_on["plan"], np.float64)
    battery = on.battery
    expected = np.asarray(state.soc, np.float64) + DELTA / np.asarray(
        battery.capacity, np.float64) * (
            np.asarray(battery.eta_charge, np.float64)
            * np.maximum(plan_mw - award_mw, 0.0)
            - np.maximum(award_mw - plan_mw, 0.0)
            / np.asarray(battery.eta_discharge, np.float64))
    np.testing.assert_allclose(np.asarray(s_on.soc, np.float64), expected,
                               rtol=1e-5, atol=1e-7)

    # the sweep sees the feeder the participants actually created: real
    # baseline, monitored award
    _, clearing_spec = make_clearing(case, sens, agent_bus, voltage_margin=0.0,
                                     thermal_margin=0.0, period_hours=DELTA)
    cursor = int(state.cursor)
    pd = np.asarray(case.node_pd, np.float64)
    scale = kappa * float(np.asarray(off.load_series)[cursor]) / pd.sum()
    bus_load = scale * pd / base
    p_real = -bus_load.copy()
    np.add.at(p_real, agent_bus,
              (np.asarray(off.pv)[cursor] - np.asarray(sub_on["plan"])) / base)
    p_cleared, q_cleared = cleared_injection(
        sens, jnp.asarray(agent_bus), jnp.asarray(clearing_spec["phi"]),
        jnp.asarray(p_real, jnp.float32),
        jnp.asarray(-scale * np.asarray(case.node_qd, np.float64) / base,
                    jnp.float32),
        jnp.asarray(out_on["award"], jnp.float32),
        jnp.asarray(out_on["shed"], jnp.float32))
    swept = make_verification(case, sens)(p_cleared, q_cleared)
    for name in ("v_under", "v_over"):
        np.testing.assert_allclose(np.asarray(i_on[name]),
                                   np.asarray(swept[name]), rtol=1e-4,
                                   atol=1e-7, err_msg=name)
    # and the precondition that makes that comparison worth making: the
    # monitored feeder is left in a worse state than the specified one, which
    # is the cost of the monitor and the only channel that shows it
    assert float(np.sum(np.asarray(i_on["v_under"]))) > 2.0 * float(
        np.sum(np.asarray(i_off["v_under"]))) > 0.0

    assert not np.allclose(np.asarray(s_on.soc), np.asarray(s_off.soc)), (
        "the two runs dispatch identically, so nothing above distinguishes "
        "a switch that changed what is procured from one wired to nothing")


def test_monitor_baseline_changes_nothing_for_an_arm_that_plans_no_charge():
    """The control, and it is the half that says what the switch *is*.

    The test above shows the payment falls.  On its own that is also what a
    switch that simply pays less would show.  Here the arm plans exactly zero
    charge -- ``-128`` is the saturation the action map documents as exact in
    float32 -- so there is nothing positioned for a monitor to remove, and
    every reported quantity has to be bitwise identical with the switch on and
    off.  A monitor that also shaved the baseline it was not asked to shave
    would fail here and pass there.

    It carries one more equality, and that one is the mechanism rather than a
    guard: with the switch on, **an arm that plans a charge is paid exactly
    what an arm that plans none is paid**.  That is the claim `§8`'s exposure
    is worth measuring against, stated as an identity between two runs instead
    of as a percentage.
    """
    kappa = 1.2
    passive = jnp.tile(jnp.asarray([-30.0, 5.0, -128.0], jnp.float32), (N, 1))
    charging = jnp.tile(jnp.asarray([-30.0, 5.0, 5.0], jnp.float32), (N, 1))
    key = jax.random.PRNGKey(0)

    (reset, step, _, _), off, *_ = build(kappa=kappa)
    _, on, *_ = build(kappa=kappa, monitor_baseline=True)
    _, state = reset(key, off)

    _, p_off, _, _, _, i_p_off = jax.jit(step)(key, state, passive, off)
    _, p_on, _, _, _, i_p_on = jax.jit(step)(key, state, passive, on)
    assert float(i_p_off["plan_total"]) == 0.0, (
        "the control arm is positioning after all, so it is not a control")
    for name in ("award_prev", "payment_prev", "profit_prev"):
        np.testing.assert_array_equal(np.asarray(getattr(p_on, name)),
                                      np.asarray(getattr(p_off, name)), name)
    np.testing.assert_array_equal(np.asarray(i_p_on["req_v"]),
                                  np.asarray(i_p_off["req_v"]))

    _, c_on, _, _, _, _ = jax.jit(step)(key, state, charging, on)
    np.testing.assert_allclose(np.asarray(c_on.payment_prev),
                               np.asarray(p_on.payment_prev), rtol=1e-6,
                               atol=1e-9)


def test_money_balance_identity_ties_costs_info_and_state(built):
    """§8, with the curtailment term read off the `costs` channel."""
    (reset, step, _, spec), params, *_ = built
    key = jax.random.PRNGKey(2)
    _, state = reset(key, params)
    _, state2, _, costs, _, info = jax.jit(step)(key, state, some_action(key),
                                                 params)

    shed_mwh = float(np.asarray(costs)[0, 0])
    paid = float(np.sum(np.asarray(state2.payment_prev, np.float64)))
    expected = float(info["z"]) - spec["voll"] * shed_mwh
    assert paid == pytest.approx(expected, rel=1e-5, abs=1e-6)
    # all three columns are system quantities copied along the agent axis
    assert np.allclose(np.asarray(costs), np.asarray(costs)[0])


def test_costs_are_the_three_quantities_of_section_16(built):
    (reset, step, _, spec), params, *_ = built
    key = jax.random.PRNGKey(3)
    _, state = reset(key, params)
    _, _, _, costs, _, info = jax.jit(step)(key, state, some_action(key), params)
    costs = np.asarray(costs)[0]

    v_depth = float(np.sum(np.asarray(info["v_under"]) + np.asarray(info["v_over"])))
    assert costs[1] == pytest.approx(v_depth, rel=1e-5, abs=1e-9)
    assert costs[2] == pytest.approx(float(np.sum(np.asarray(info["overload"]))),
                                     rel=1e-5, abs=1e-9)
    assert spec["cost_names"] == ("shed_energy_mwh", "voltage_violation_sum_pu",
                                  "thermal_violation_sum_pu")
    assert (costs >= 0.0).all()


def test_reward_is_the_settlement_profit_and_carries_no_penalty(built):
    """§9.5's cost, recomputed; no violation depth and no curtailment in it."""
    (reset, step, _, _), params, case, sens, agent_bus = built
    key = jax.random.PRNGKey(4)
    _, state = reset(key, params)
    action = some_action(key)
    _, state2, reward, costs, _, _ = jax.jit(step)(key, state, action, params)

    sub, out, _, _ = recompute(case, sens, agent_bus, params, state, action)
    base = float(sens.base_mva)
    award = np.asarray(out["award"], np.float64)
    plan = np.asarray(sub["plan"], np.float64) / base
    p_dis = np.maximum(award - plan, 0.0)
    p_ch = np.maximum(plan - award, 0.0)
    price_e = float(np.asarray(params.energy_price)[int(state.cursor)])
    cycle = np.asarray(params.cycle_cost, np.float64)
    money = DELTA * base
    revenue = money * np.asarray(sub["price"], np.float64) * award
    cost = money * (price_e * p_ch + cycle * (p_ch + p_dis))

    np.testing.assert_allclose(np.asarray(reward, np.float64), revenue - cost,
                               rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(state2.payment_prev, np.float64),
                               revenue, rtol=1e-5, atol=1e-6)
    # the feasibility quantities of the period are non-trivial and none of them
    # appears in the reward above
    assert float(np.asarray(costs)[0].sum()) > 0.0


def test_soc_follows_section_9_5_without_the_vendored_clip(built):
    """The discriminating form: recompute (SOC), do not read the clipped value."""
    (reset, step, _, _), params, case, sens, agent_bus = built
    key = jax.random.PRNGKey(5)
    _, state = reset(key, params)
    action = some_action(key)
    _, state2, *_ = jax.jit(step)(key, state, action, params)

    sub, out, _, _ = recompute(case, sens, agent_bus, params, state, action)
    base = float(sens.base_mva)
    award_mw = np.asarray(out["award"], np.float64) * base
    plan_mw = np.asarray(sub["plan"], np.float64)
    p_dis = np.maximum(award_mw - plan_mw, 0.0)
    p_ch = np.maximum(plan_mw - award_mw, 0.0)

    battery = params.battery
    expected = np.asarray(state.soc, np.float64) + DELTA / np.asarray(
        battery.capacity, np.float64) * (
            np.asarray(battery.eta_charge, np.float64) * p_ch
            - p_dis / np.asarray(battery.eta_discharge, np.float64))
    np.testing.assert_allclose(np.asarray(state2.soc, np.float64), expected,
                               rtol=1e-5, atol=1e-7)
    lo = np.asarray(battery.soc_min, np.float64)
    hi = np.asarray(battery.soc_max, np.float64)
    assert (expected >= lo - 1e-6).all() and (expected <= hi + 1e-6).all()


def test_all_false_mask_ignores_the_action_bitwise(built):
    """The baseline is an action, and it replaces the policy's."""
    (reset, step, _, _), params, *_ = built
    masked = params.replace(learner_mask=jnp.zeros((N,), bool))
    key = jax.random.PRNGKey(6)
    _, state = reset(key, masked)

    first = jax.jit(step)(key, state, some_action(key), masked)
    second = jax.jit(step)(key, state, 50.0 * some_action(
        jax.random.PRNGKey(60)), masked)
    for a, b in zip(jax.tree_util.tree_leaves(first[:4]),
                    jax.tree_util.tree_leaves(second[:4])):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_baseline_offers_truthfully_and_withholds_nothing(built):
    """The three components of the baseline, read off the submission itself."""
    (reset, step, _, _), params, case, sens, agent_bus = built
    masked = params.replace(learner_mask=jnp.zeros((N,), bool))
    key = jax.random.PRNGKey(7)
    _, state = reset(key, masked)

    from powermarketjax.envs.local_flexibility.env import baseline_action
    sub, _, _, _ = recompute(case, sens, agent_bus, masked, state,
                             baseline_action(N))
    np.testing.assert_array_equal(np.asarray(sub["price"]),
                                  np.asarray(sub["c_rep"]))
    np.testing.assert_array_equal(np.asarray(sub["qty_max"]),
                                  np.asarray(sub["q_phys"]))
    assert (np.asarray(sub["plan"]) == 0.0).all()


def test_terminal_obs_is_the_true_successor_observation():
    """`done` is a truncation, so the bootstrap target is the successor of the
    step that was cut, pinned against the same transition under a longer
    episode where no reset interferes."""
    (reset, step, _, _), params, *_ = build(episode_len=2, n_periods=12)
    long_env, long_params, *_ = build(episode_len=6, n_periods=12)
    key = jax.random.PRNGKey(8)
    _, state = reset(key, params)
    action = some_action(key)

    obs_a, state_a, *_ = jax.jit(step)(key, state, action, params)
    _, _, _, _, done, info = jax.jit(step)(key, state_a, some_action(key),
                                           params)
    assert bool(done)

    long_step = long_env[1]
    _, state_b, *_ = jax.jit(long_step)(key, state, action, long_params)
    obs_b, _, _, _, done_b, _ = jax.jit(long_step)(key, state_b,
                                                   some_action(key),
                                                   long_params)
    assert not bool(done_b)
    np.testing.assert_allclose(np.asarray(info["terminal_obs"]),
                               np.asarray(obs_b), rtol=1e-6, atol=1e-9)


def test_done_restarts_the_state(built):
    (reset, step, _, _), params, *_ = built
    key = jax.random.PRNGKey(9)
    _, state = reset(key, params)
    for _ in range(EPISODE_LEN):
        _, state, _, _, done, info = jax.jit(step)(key, state,
                                                   some_action(key), params)
    assert bool(done)
    assert int(state.step_in_episode) == 0
    np.testing.assert_array_equal(np.asarray(state.soc),
                                  np.asarray(params.battery.initial_soc,
                                             np.float32))
    for name in ("award_prev", "payment_prev", "profit_prev", "pv_prev",
                 "load_prev", "req_v_own", "req_th_own"):
        assert not np.any(np.asarray(getattr(state, name))), name
    assert float(state.volume_prev) == 0.0
    assert float(state.price_avg_prev) == 0.0
    # and the truncation reported the observation the reset then overwrote
    assert not np.array_equal(np.asarray(info["terminal_obs"]),
                              np.asarray(reset(key, params)[0]))


def test_idle_feeder_clears_nothing_and_reports_no_price(built):
    """A feeder inside its limits procures nothing.

    What it leaves is interior-point dust rather than exact zeros -- the
    analytic centre gives every variable a small positive value -- so the
    reported average price is the ratio of two negligible numbers and not the
    exact zero the environment specifies.  Measured here at 1e-19 MW and 3e-7 \\$/MWh.
    The exact case is the next test.
    """
    (reset, step, _, _), params, *_ = build(kappa=1.0, pv_scale=0.0)
    key = jax.random.PRNGKey(10)
    _, state = reset(key, params)
    _, _, reward, costs, _, info = jax.jit(step)(key, state,
                                                 some_action(key), params)
    assert float(info["traded_volume"]) == pytest.approx(0.0, abs=1e-6)
    assert abs(float(info["price_avg"])) < 1e-6
    assert np.isfinite(np.asarray(reward)).all()
    assert float(np.asarray(costs)[0, 0]) == pytest.approx(0.0, abs=1e-9)


def test_empty_population_hits_the_zero_volume_guard(built):
    """The reachable exact zero: batteries at `soc_min` under the baseline
    offer nothing at all, so the average price is 0/0 without the guard."""
    (reset, step, _, _), params, *_ = built
    masked = params.replace(learner_mask=jnp.zeros((N,), bool))
    key = jax.random.PRNGKey(13)
    _, state = reset(key, masked)
    empty = state.replace(soc=jnp.asarray(params.battery.soc_min, jnp.float32))

    _, state2, reward, _, _, info = jax.jit(step)(key, empty,
                                                  some_action(key), masked)
    assert float(info["traded_volume"]) == 0.0
    assert float(info["price_avg"]) == 0.0
    assert float(state2.price_avg_prev) == 0.0
    assert np.isfinite(np.asarray(reward)).all()


def test_load_scale_is_swept_without_rebuilding(built):
    """kappa lives in params, and it is what creates the need."""
    (reset, step, _, _), params, *_ = built
    key = jax.random.PRNGKey(11)
    _, state = reset(key, params)

    light = params.replace(load_scale=jnp.float32(1.0))
    heavy = params.replace(load_scale=jnp.float32(2.5))
    step_jit = jax.jit(step)
    _, _, _, _, _, info_light = step_jit(key, state, some_action(key), light)
    _, _, _, _, _, info_heavy = step_jit(key, state, some_action(key), heavy)

    assert float(info_light["req_v_max"]) < float(info_heavy["req_v_max"])
    assert int(info_light["req_v_count"]) <= int(info_heavy["req_v_count"])


def test_the_real_demand_series_feeds_the_environment():
    """§14's series, loaded and run: what is missing is data, not wiring.

    The photovoltaic array is the zero fallback §14 names and the price is a
    constant placeholder, so **no market result may be quoted from this test**
    -- it asserts shapes, finiteness and convergence only.  What it does say is
    that the loader's output goes into `make_local_flex_params` without an
    adapter, which is the claim `data.py` exists to support.
    """
    from powermarketjax.envs.local_flexibility import (AEDT_WINDOW,
                                                       load_substation_demand)

    load_mw, index = load_substation_demand(
        "Mona Vale 33_11kV", start=AEDT_WINDOW[0], end="2024-10-07 12:45")
    n_periods = len(index)
    assert n_periods == 96                       # one local day, 15 minutes

    case = load_case(CASE)
    sens = build_voltage_sensitivity(case)
    agent_bus = draw_agent_buses(case, sens, N, seed=0)
    reset, step, _, _ = make_local_flex_env(case, sens, agent_bus,
                                            period_hours=DELTA)
    rng = np.random.default_rng(0)
    battery = make_battery_bundle(
        n_devices=N, dt_hours=DELTA,
        capacity_mwh=rng.uniform(0.2, 1.0, N).tolist(),
        power_mw=rng.uniform(0.05, 0.3, N).tolist())
    params = make_local_flex_params(
        load_series=load_mw, pv=np.zeros((n_periods, N)),
        energy_price=np.full(n_periods, 60.0), battery=battery,
        cycle_cost=np.full(N, 10.0), load_scale=1.5,
        learner_mask=np.ones(N, bool), episode_len=4)

    key = jax.random.PRNGKey(14)
    _, state = reset(key, params)

    def body(carry, k):
        _, carry, reward, costs, done, info = step(k, carry,
                                                   some_action(k), params)
        return carry, (reward, costs, info["mu"])

    _, (reward, costs, mu) = jax.jit(lambda s, k: lax.scan(body, s, k))(
        state, jax.random.split(key, 4))
    assert np.isfinite(np.asarray(reward)).all()
    assert (np.asarray(mu) < 1e-6).all()
    assert (np.asarray(costs) >= 0.0).all()


def test_primary_case_one_step_at_the_adopted_configuration():
    """The one test on `case533mt_hi`: the development case cannot exercise
    (LIM) at all, since its ratings are the 1e6 sentinel (§13).  The adopted
    configuration -- 40 aggregators, kappa = 1.5, both margins.
    """
    (reset, step, _, spec), params, case, sens, agent_bus = build(
        case_id="533mt_hi", n_agent=40, kappa=1.5, n_periods=6, episode_len=3,
        seed=3, margins=(0.002, 0.02))
    key = jax.random.PRNGKey(12)
    _, state = reset(key, params)
    _, state2, reward, costs, _, info = jax.jit(step)(key, state,
                                                      some_action(key, 40),
                                                      params)

    assert float(info["mu"]) < 1e-6, "unconverged solve on the primary case"
    assert np.isfinite(np.asarray(reward)).all()
    # the ratings are real here, so (LIM) is a live constraint rather than a
    # sentinel: some line carries a finite rating and the thermal requirement
    # is defined against it
    assert np.isfinite(sens.p_max).all() and (sens.p_max < 1e3).any()

    # ...and therefore the one place the thermal requirement's unit can be
    # checked: `case33bw` reports it as identically zero, so
    # `test_requirement_fields_are_the_published_signals` records that half as
    # vacuous and leaves it here.  `info` is per unit out of `requirement`, the
    # observation is MW.
    base = float(sens.base_mva)
    req_th = np.asarray(info["req_th"])
    assert base != 1.0 and req_th.max() > 0.0, (base, req_th.max())
    assert float(state2.req_th_max) == pytest.approx(base * req_th.max(),
                                                     rel=1e-6)
    on_path = (sens.A[:, agent_bus] * req_th[:, None]).max(axis=0)
    np.testing.assert_allclose(np.asarray(state2.req_th_own), base * on_path,
                               rtol=1e-6, atol=1e-7)

    paid = float(np.sum(np.asarray(state2.payment_prev, np.float64)))
    expected = float(info["z"]) - spec["voll"] * float(np.asarray(costs)[0, 0])
    assert paid == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_offering_zero_quantity_clears_nothing_whatever_the_price(built):
    """The coordinate that carries declining to sell, at the box's own floor.

    `envs/local_flexibility/action.py` publishes a finite action box and loses
    exactly one submission by doing so: a price above `(1 + ACTION_SATURATION)
    c_rep`, since softplus does not saturate above.  The argument that it loses
    no *outcome* rests on this step and not on that price.  At
    `alpha_q = -ACTION_SATURATION` the offered quantity is exactly zero, (CAP)
    then holds the award at zero, and the price the offer carries changes
    nothing -- so refusing to sell is inside the box, on a coordinate where the
    box is bit-exact.

    Asserted at **both** ends of the price coordinate, because the claim is
    about the quantity: if it held only at the high price it would be the
    price route in disguise, and that route needs
    `c_rep >= VOLL / (1 + ACTION_SATURATION) = 77.52` $/MWh, which is a fact
    about the scenario rather than about the box.  On this fixture `c_rep` runs
    below that for part of the population, so the two are genuinely separated
    here.

    `test_clearing_l1.py::test_zero_quantity_offers_receive_exactly_nothing`
    already holds the clearing's half of this -- hand the program a zero
    `qty_max` and the award is exactly zero rather than the `OFF_EPS` phantom.
    What is new here is the other half: that the box's own floor on `alpha_q`
    is what produces that zero, end to end through the action map, and that the
    price column does not enter.

    The bite is the same step with `alpha_q` at the box's ceiling instead: the
    award is then not zero, so "nothing cleared" is a property of the offered
    quantity and not of this fixture having nothing to clear.
    """
    (reset, step, _, spec), params, *_ = built
    key = jax.random.PRNGKey(0)
    _, state = reset(key, params)
    saturation = float(-np.asarray(spec["action_low"]))

    def award_at(alpha_pi, alpha_q):
        # `alpha_ch` at the floor throughout, so nothing is planned either and
        # the participant is idle: a planned charge is bought and paid for, so
        # a non-zero one puts a cost in `reward` that has nothing to do with
        # what cleared.
        action = jnp.tile(
            jnp.asarray((alpha_pi, alpha_q, -saturation), jnp.float32), (N, 1))
        _, state2, reward, _, _, _ = jax.jit(step)(key, state, action, params)
        return np.asarray(state2.award_prev), np.asarray(reward)

    for alpha_pi in (-saturation, +saturation):
        award, reward = award_at(alpha_pi, -saturation)
        assert (award == 0.0).all(), (alpha_pi, award)
        assert (reward == 0.0).all(), (alpha_pi, reward)
    # the other side: at the ceiling of the same coordinate something clears,
    # so the zero above is the offered quantity and not the fixture
    award, _reward = award_at(-saturation, +saturation)
    assert float(award.max()) > 0.0, award
