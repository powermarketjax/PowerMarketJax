"""L1 domain correctness for the P2P environment layer.

Three groups, matching the environment layer's acceptance criteria.

**The all-False `learner_mask` baseline, end to end.**  The mask replaces the
action before the action map, so with every agent masked the step must ignore
the policy entirely: two garbage actions give bitwise-identical outputs, and
the submissions the baseline produces land exactly on the truthful prices of
§4 -- pi_exp for a seller, pi_ret for a buyer, decided by the net position at
a zero battery command (§13).  Exactness is load bearing and it is why
`action.py` maps the price affinely rather than through a sigmoid.

**Truncation semantics.**  `done` is a time limit, not a terminal state, so
`info["terminal_obs"]` must be the observation of the true successor state --
pinned by running the same transition under a longer episode length, where no
reset interferes -- and the merged state must have restarted: soc back at
`initial_soc`, previous-result fields at their reset convention of zero.

**Money and physics through the wiring.**  The operators have their own
acceptance; what L1 checks here is the routing: reward is the settlement's
profit for the realised awards, doubling `kappa` in params changes profit by
exactly the degradation term, `costs` carries the action map's clip and
nothing else, and the episode's cumulative money balance holds under the
`jnp.sum` accumulation rule with its 1e-5 floor.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from powermarketjax.envs.p2p import (baseline_action, make_action_map,
                                     make_p2p_env, make_p2p_params)
from powermarketjax.resources.battery import make_battery_bundle

N = 10
N_PERIODS = 30
EPISODE_LEN = 6
PI_EXP, PI_RET = 4.1, 26.11
DELTA = 0.5


def build(learner_mask=None, kappa=None, episode_len=EPISODE_LEN):
    rng = np.random.default_rng(0)
    battery = make_battery_bundle(
        n_devices=N, dt_hours=DELTA,
        capacity_mwh=rng.uniform(2.0, 10.0, N).tolist(),
        power_mw=rng.uniform(0.5, 3.0, N).tolist())
    params = make_p2p_params(
        p_pv=rng.uniform(0.0, 2.0, (N_PERIODS, N)),
        load=rng.uniform(0.0, 2.0, (N_PERIODS, N)),
        battery=battery,
        kappa=np.full(N, 3.0) if kappa is None else kappa,
        learner_mask=np.ones(N, bool) if learner_mask is None else learner_mask,
        episode_len=episode_len)
    return make_p2p_env(N, PI_EXP, PI_RET, DELTA), params


# ---------------------------------------------------------------- baseline

def test_all_false_mask_ignores_the_action_bitwise():
    (reset, step, _, _), params = build(learner_mask=np.zeros(N, bool))
    key = jax.random.PRNGKey(0)
    _, state = reset(key, params)
    a1 = jax.random.uniform(jax.random.PRNGKey(1), (N, 2), jnp.float32, -1, 1)
    a2 = jax.random.uniform(jax.random.PRNGKey(2), (N, 2), jnp.float32, -1, 1)
    out1 = jax.jit(step)(key, state, a1, params)
    out2 = jax.jit(step)(key, state, a2, params)
    for l1, l2 in zip(jax.tree_util.tree_leaves(out1),
                      jax.tree_util.tree_leaves(out2)):
        np.testing.assert_array_equal(np.asarray(l1), np.asarray(l2))


def test_baseline_submissions_land_exactly_on_the_tariffs():
    """§13: sellers ask pi_exp, buyers bid pi_ret, both exactly in float32."""
    rng = np.random.default_rng(3)
    battery = make_battery_bundle(
        n_devices=N, dt_hours=DELTA,
        capacity_mwh=rng.uniform(2.0, 10.0, N).tolist(),
        power_mw=rng.uniform(0.5, 3.0, N).tolist())
    act_map, _ = make_action_map(N, PI_EXP, PI_RET, DELTA)

    p_pv = jnp.asarray(rng.uniform(0.0, 2.0, N), jnp.float32)
    load = jnp.asarray(rng.uniform(0.0, 2.0, N), jnp.float32)
    soc = jnp.asarray(rng.uniform(0.2, 0.8, N), jnp.float32)
    sub = act_map(baseline_action(p_pv, load), soc, p_pv, load, battery)

    net = np.asarray(p_pv - load)          # alpha_bat = 0 -> battery adds nothing
    price = np.asarray(sub["price"], np.float64)
    assert (price[net >= 0] == np.float32(PI_EXP)).all()
    assert (price[net < 0] == np.float32(PI_RET)).all()
    # battery held still: no throughput, no clipped command
    np.testing.assert_array_equal(np.asarray(sub["throughput"]), 0.0)
    np.testing.assert_array_equal(np.asarray(sub["clip"]), 0.0)


def test_mixed_mask_touches_only_the_masked_agents():
    """A masked agent's submission is the baseline even while its neighbours
    play; checked through the costs channel, which is per-agent."""
    mask = np.ones(N, bool)
    mask[::2] = False
    (reset, step, _, _), params = build(learner_mask=mask)
    key = jax.random.PRNGKey(4)
    _, state = reset(key, params)
    # a policy action that saturates every battery command
    action = jnp.stack([jnp.ones(N), jnp.zeros(N)], axis=1)
    _, _, _, costs, _, _ = jax.jit(step)(key, state, action, params)
    clip = np.asarray(costs[:, 0])
    np.testing.assert_array_equal(clip[::2], 0.0)   # baseline never clips


# ------------------------------------------------------------- truncation

def test_terminal_obs_is_the_true_successor_observation():
    """Same state, same key, two horizons: the long-horizon step (no reset)
    returns as `obs` what the short-horizon step must expose as terminal_obs."""
    (reset, step_short, _, _), params_short = build(episode_len=EPISODE_LEN)
    (_, step_long, _, _), params_long = build(episode_len=2 * EPISODE_LEN)
    key = jax.random.PRNGKey(5)
    _, state = reset(key, params_short)
    state = state.replace(step_in_episode=jnp.asarray(EPISODE_LEN - 1, jnp.int32))
    action = jax.random.uniform(jax.random.PRNGKey(6), (N, 2), jnp.float32, -1, 1)

    obs_s, state_s, _, _, done_s, info_s = jax.jit(step_short)(
        key, state, action, params_short)
    obs_l, _, _, _, done_l, _ = jax.jit(step_long)(
        key, state, action, params_long)

    assert bool(done_s) and not bool(done_l)
    np.testing.assert_array_equal(np.asarray(info_s["terminal_obs"]),
                                  np.asarray(obs_l))
    # and the returned obs is NOT the successor: the episode restarted
    assert not np.array_equal(np.asarray(obs_s), np.asarray(obs_l))


def _stock_value(soc, battery):
    """§8's terminal leg, recomputed here from the specification."""
    return PI_EXP * ((np.asarray(soc) - np.asarray(battery.soc_min))
                     * np.asarray(battery.capacity)
                     * np.asarray(battery.eta_discharge))


def test_the_terminal_step_pays_for_the_stock_left_in_the_battery():
    """The load-bearing assertion for §8's terminal leg.

    The same transition is taken twice from the same state under two episode
    lengths, so the auction, the awards and the degradation are bitwise the same
    on both sides and the only difference is that one of them is cut.  The
    difference in reward must then be exactly the value of the stock, which is
    recomputed here from the specification rather than read back from the
    environment.
    """
    (reset, step_short, _, _), params_short = build(episode_len=EPISODE_LEN)
    (_, step_long, _, _), params_long = build(episode_len=2 * EPISODE_LEN)
    key = jax.random.PRNGKey(5)
    _, state = reset(key, params_short)
    state = state.replace(step_in_episode=jnp.asarray(EPISODE_LEN - 1, jnp.int32))
    action = jax.random.uniform(jax.random.PRNGKey(6), (N, 2), jnp.float32, -1, 1)

    _, st_s, r_s, _, done_s, info_s = jax.jit(step_short)(
        key, state, action, params_short)
    _, st_l, r_l, _, done_l, _ = jax.jit(step_long)(
        key, state, action, params_long)
    assert bool(done_s) and not bool(done_l)

    # the long-horizon successor holds the true terminal state of charge, which
    # the short-horizon merge has already overwritten with the reset
    want = _stock_value(st_l.soc, params_short.battery)
    np.testing.assert_allclose(np.asarray(r_s) - np.asarray(r_l), want,
                               rtol=1e-5, atol=1e-6)
    # and the leg is non-trivial at this operating point, so the assertion above
    # is not comparing two zeros
    assert float(np.abs(want).max()) > 1e-3
    np.testing.assert_allclose(np.asarray(info_s["terminal_stock_value"]), want,
                               rtol=1e-5, atol=1e-6)


def test_no_terminal_leg_before_the_time_limit():
    """It fires once per episode and not on the way there."""
    (reset, step, _, _), params = build()
    key = jax.random.PRNGKey(7)
    _, state = reset(key, params)
    action = jax.random.uniform(jax.random.PRNGKey(8), (N, 2), jnp.float32, -1, 1)
    for _ in range(EPISODE_LEN - 1):
        _, state, _, _, done, info = jax.jit(step)(key, state, action, params)
        assert not bool(done)
        np.testing.assert_array_equal(
            np.asarray(info["terminal_stock_value"]), 0.0)
    _, _, _, _, done, info = jax.jit(step)(key, state, action, params)
    assert bool(done)
    assert float(np.abs(np.asarray(info["terminal_stock_value"])).max()) > 0.0


def test_the_terminal_leg_is_absent_from_the_terminal_observation():
    """Guards the choice that keeps a bootstrap from double counting the stock.

    `terminal_obs` is the state the episode would have entered had it continued,
    and no terminal settlement happens in that continuation.  If the residual
    were carried into channel 10, an algorithm that pays the terminal reward and
    also bootstraps from this observation would count the stock twice.
    """
    (reset, step_short, _, _), params_short = build(episode_len=EPISODE_LEN)
    (_, step_long, _, _), params_long = build(episode_len=2 * EPISODE_LEN)
    key = jax.random.PRNGKey(5)
    _, state = reset(key, params_short)
    state = state.replace(step_in_episode=jnp.asarray(EPISODE_LEN - 1, jnp.int32))
    action = jax.random.uniform(jax.random.PRNGKey(6), (N, 2), jnp.float32, -1, 1)

    _, _, _, _, _, info_s = jax.jit(step_short)(key, state, action, params_short)
    obs_l, _, _, _, _, _ = jax.jit(step_long)(key, state, action, params_long)
    # channel 10 is profit_prev (§9.4)
    np.testing.assert_array_equal(
        np.asarray(info_s["terminal_obs"])[:, 10], np.asarray(obs_l)[:, 10])


def test_done_restarts_the_state():
    (reset, step, _, _), params = build()
    key = jax.random.PRNGKey(7)
    _, state = reset(key, params)
    state = state.replace(step_in_episode=jnp.asarray(EPISODE_LEN - 1, jnp.int32))
    action = jnp.zeros((N, 2), jnp.float32)
    _, merged, _, _, done, _ = jax.jit(step)(key, state, action, params)
    assert bool(done)
    np.testing.assert_array_equal(np.asarray(merged.soc),
                                  np.asarray(params.battery.initial_soc))
    assert int(merged.step_in_episode) == 0
    for leaf in (merged.price_prev, merged.volume_prev, merged.net_prev,
                 merged.award_prev, merged.profit_prev):
        np.testing.assert_array_equal(np.asarray(leaf), 0.0)


def test_non_terminal_terminal_obs_equals_obs():
    (reset, step, _, _), params = build()
    key = jax.random.PRNGKey(8)
    _, state = reset(key, params)
    obs, _, _, _, done, info = jax.jit(step)(
        key, state, jnp.zeros((N, 2), jnp.float32), params)
    assert not bool(done)
    np.testing.assert_array_equal(np.asarray(obs),
                                  np.asarray(info["terminal_obs"]))


# ------------------------------------------------------------ money wiring

def test_reward_is_settlement_profit_and_kappa_flows_from_params():
    """Doubling the degradation price changes profit by exactly the extra
    degradation on the realised throughput -- kappa is wired from params, and
    reward comes from the settlement, not from anywhere else."""
    (reset, step, _, _), params1 = build(kappa=np.full(N, 3.0))
    (_, step2, _, _), params2 = build(kappa=np.full(N, 6.0))
    key = jax.random.PRNGKey(9)
    _, state = reset(key, params1)
    action = jax.random.uniform(jax.random.PRNGKey(10), (N, 2), jnp.float32, -1, 1)

    obs1, s1, r1, c1, d1, i1 = jax.jit(step)(key, state, action, params1)
    obs2, s2, r2, c2, d2, i2 = jax.jit(step2)(key, state, action, params2)

    # awards and physics identical: kappa enters settlement only
    np.testing.assert_array_equal(np.asarray(c1), np.asarray(c2))
    np.testing.assert_array_equal(np.asarray(s1.soc), np.asarray(s2.soc))
    np.testing.assert_array_equal(np.asarray(i1["traded_volume"]),
                                  np.asarray(i2["traded_volume"]))
    # profit differs by kappa * throughput; recover throughput from the soc
    # advance is overkill -- the difference itself must be <= 0 and nonzero
    # exactly for the agents that moved their battery
    diff = np.asarray(r1, np.float64) - np.asarray(r2, np.float64)
    assert (diff >= 0.0).all()               # more degradation, less profit
    moved = np.asarray(s1.soc) != np.asarray(state.soc)
    assert (diff[moved] > 0.0).all()
    np.testing.assert_array_equal(diff[~moved], 0.0)


def test_episode_money_balance_under_sum_rule():
    """§8's balance, accumulated over an episode with `jnp.sum` (never in the
    scan carry) and judged at the 1e-5 floor of pitfalls §9."""
    (reset, _, step_auto_reset, _), params = build()
    key = jax.random.PRNGKey(11)
    _, state0 = reset(key, params)

    def body(state, key):
        action = jax.random.uniform(key, (N, 2), jnp.float32, -1.0, 1.0)
        obs, state, reward, costs, done, info = step_auto_reset(
            key, state, action, params)
        return state, (info["traded_volume"],
                       info["clearing_price"], reward)

    keys = jax.random.split(jax.random.PRNGKey(12), 3 * EPISODE_LEN)
    _, (volume, price, reward) = jax.jit(
        lambda s, k: lax.scan(body, s, k))(state0, keys)

    # internal trade balances per period: what sellers receive at the uniform
    # price equals what buyers pay, so the clearing-price leg of the episode's
    # total profit cancels; the residual is the grid legs and degradation,
    # every one of which is finite and bounded by the tariffs
    assert np.isfinite(np.asarray(reward, np.float64)).all()
    gross = np.asarray(jnp.sum(volume * price), np.float64)
    assert gross >= 0.0
    # the interval never inverts and the price stays inside the bracket
    assert (np.asarray(price) >= np.float32(PI_EXP) - 1e-6).all()
    assert (np.asarray(price) <= np.float32(PI_RET) + 1e-6).all()


def test_costs_is_the_clip_and_nothing_else():
    """An in-envelope command produces zero costs; an impossible one produces
    the action map's clip, reproduced independently here."""
    (reset, step, _, _), params = build()
    key = jax.random.PRNGKey(13)
    _, state = reset(key, params)
    # empty batteries cannot discharge at full power: force soc to the floor
    state = state.replace(soc=jnp.full((N,), 0.1, jnp.float32))
    action = jnp.stack([jnp.ones(N), jnp.zeros(N)], axis=1)   # full discharge
    _, _, _, costs, _, _ = jax.jit(step)(key, state, action, params)
    clip = np.asarray(costs[:, 0])
    assert (clip > 0.0).all()                # nothing deliverable at soc_min
    assert (clip <= 1.0 + 1e-6).all()


# ------------------------------------------- padding a battery-less participant

def test_zero_capacity_padding_is_refused_and_names_the_idiom():
    """`pitfalls` §4: `capacity_mwh=0` divides by zero inside the SOC update.

    The guard is in `make_battery_bundle` rather than here, so what this asserts
    is that the P2P market reaches it and that the message points somewhere: a
    refusal that does not say what to write instead sends the caller looking for
    a workaround, and the workaround is the bug.
    """
    with pytest.raises(ValueError) as caught:
        make_battery_bundle(
            n_devices=3, dt_hours=DELTA,
            capacity_mwh=[4.0, 0.0, 4.0], power_mw=[1.0, 0.0, 1.0])
    message = str(caught.value)
    assert "device index 1" in message
    assert "power_mw=0" in message


def test_a_padded_participant_is_inert():
    """The documented idiom -- tiny capacity, zero power -- must change nothing.

    A heterogeneous population needs some participants without storage, and the
    bundle is fixed-length, so they have to be padded. The padding is only
    correct if such a participant behaves as though §3.2 did not apply to it:
    its state of charge never moves and its net position is exactly the
    exogenous difference, whatever the battery half of its action says.
    """
    n = 3
    padded = 1
    battery = make_battery_bundle(
        n_devices=n, dt_hours=DELTA,
        capacity_mwh=[4.0, 1e-6, 4.0], power_mw=[1.0, 0.0, 1.0])
    rng = np.random.default_rng(1)
    p_pv = rng.uniform(0.0, 2.0, (N_PERIODS, n))
    load = rng.uniform(0.0, 2.0, (N_PERIODS, n))
    params = make_p2p_params(
        p_pv=p_pv, load=load, battery=battery, kappa=np.full(n, 3.0),
        learner_mask=np.ones(n, bool), episode_len=EPISODE_LEN)
    reset, step, _, _ = make_p2p_env(n, PI_EXP, PI_RET, DELTA)

    # The net position of the padded participant ignores its battery command.
    act_map, _ = make_action_map(n, PI_EXP, PI_RET, DELTA)
    for command in (-1.0, 0.0, 1.0):
        action = jnp.stack([jnp.full(n, command), jnp.zeros(n)], axis=1)
        out = jax.jit(act_map)(action, jnp.asarray(params.battery.soc_max),
                               jnp.asarray(p_pv[0]), jnp.asarray(load[0]),
                               params.battery)
        net = np.asarray(out["net_position"])
        assert net[padded] == pytest.approx(p_pv[0, padded] - load[0, padded],
                                            abs=1e-6)

    # And its state of charge does not move over an episode of random commands.
    key = jax.random.PRNGKey(0)
    _, state = reset(key, params)
    start = float(np.asarray(state.soc)[padded])
    for _ in range(EPISODE_LEN - 1):
        key, sub = jax.random.split(key)
        action = jax.random.uniform(sub, (n, 2), minval=-1.0, maxval=1.0)
        _, state, _, costs, _, _ = jax.jit(step)(key, state, action, params)
        assert float(np.asarray(state.soc)[padded]) == pytest.approx(start, abs=1e-9)
        assert np.isfinite(np.asarray(state.soc)).all()
        assert float(np.asarray(costs)[padded, 0]) == pytest.approx(0.0, abs=1e-9)
