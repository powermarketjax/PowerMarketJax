"""L1 domain correctness for the real-time environment (§17).

Scoped to what is **this environment's** rather than the clearing's: the clearing
is the day-ahead operator at `T = 1` and its own L1 already covers power balance,
line limits and the box, so what is checked here is the sequential coupling, the
two-settlement reward, the `costs` channel and the truncation contract.
"""
import ast
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.day_ahead import load_gb_demand
from powermarketjax.envs.real_time import load_da_position
from powermarketjax.envs.real_time import env as env_module
from powermarketjax.envs.real_time.demand import T_RT, load_gb_demand_half_hourly
from powermarketjax.envs.real_time.env import COST_NAMES, make_env

EPISODE_LEN = 6
MARKUP_MAX = 2.0

#: The adopted scenario, stated rather than read back from the
#: fixture: reading the scales out of it makes any comparison pass by
#: construction.
#: **`real_time.make_env` does not check them**: the refusal comparing a
#: fixture's `meta` against the requested scales lives in `day_ahead.make_env`,
#: while this market's constructor defaults `cap_scale` to 1.0 and validates
#: nothing.  The assertion in the fixture below is the whole guard.
CAP_SCALE = 0.60
RAMP_SCALE = 1.00
#: `_seasons` is the expand-phase name; batch 3 renames it back, and
#: `grep -rn "_seasons"` is that batch's acceptance criterion.
CHAIN = "step1prime_seasons"


@pytest.fixture(scope="module", autouse=True)
def x64():
    prev = jax.config.jax_enable_x64, jax.config.jax_default_matmul_precision
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    yield
    jax.config.update("jax_enable_x64", prev[0])
    jax.config.update("jax_default_matmul_precision", prev[1])


@pytest.fixture(scope="module")
def built(x64):
    pos = load_da_position(chain=CHAIN)
    assert (pos["meta"]["cap_scale"], pos["meta"]["ramp_scale"]) == \
        (CAP_SCALE, RAMP_SCALE), (
            f"position built at cap {pos['meta']['cap_scale']} / ramp "
            f"{pos['meta']['ramp_scale']}, but this module builds the "
            f"environment at {CAP_SCALE} / {RAMP_SCALE}")
    case = load_case(pos["meta"]["case"])
    hh, _d = load_gb_demand_half_hourly()
    forecast, _a, _days = load_gb_demand()
    env, spec = make_env(case, pos, hh, forecast, n_segments=1,
                         markup_max=MARKUP_MAX, cap_scale=CAP_SCALE,
                         ramp_scale=RAMP_SCALE)
    return env, spec, env.make_params(episode_len=EPISODE_LEN), case


def _roll(env, params, n, key=0):
    """Run `n` steps at the truthful action, returning the per-step records."""
    action = env.truthful_action()
    _obs, state = env.reset(jax.random.PRNGKey(key), params)
    out = []
    step = jax.jit(env.step)
    for _ in range(n):
        obs, nxt, reward, costs, done, info = step(
            jax.random.PRNGKey(0), state, action, params)
        out.append((state, nxt, reward, costs, done, info))
        state = nxt
    return out


def test_the_physical_carry_is_the_realised_dispatch(built):
    """§9.2: `p_prev` of the next step is the award of this one, exactly.

    This is the only coupling between steps, so if it were recomputed or shifted
    the episode would silently become a sequence of independent periods -- which
    would still converge, still settle, and still look like a market.
    """
    env, spec, params, _case = built
    for state, nxt, _r, _c, _d, info in _roll(env, params, EPISODE_LEN - 1):
        np.testing.assert_array_equal(np.asarray(nxt.p_prev),
                                      np.asarray(nxt.award_prev))
        assert float(info["mu"]) < env_module.MU_TOL


def _day_that_switches(params, n_days, steps):
    """First day whose first `steps` periods contain a commitment change."""
    for day in range(n_days):
        base = day * T_RT
        moved = 0
        for t in range(steps):
            cur = base + t
            now = np.asarray(params.u_da[cur], np.float64)
            prev = np.asarray(params.u_da[max(cur - 1, 0)], np.float64)
            moved += int(np.abs(now - prev).sum())
        if moved:
            return day
    return None


def _seed_opening_day(env, params, day, limit=5000):
    """A `_roll` seed whose `reset` opens `day`.

    `reset` draws the day itself, so the day is selected by choosing the key
    rather than by patching `state.cursor`: the opening carry is built inside
    `reset` from the day it drew, and a patched cursor would leave the carry
    belonging to a different day -- which is precisely the quantity the bound
    below is a statement about.
    """
    for i in range(limit):
        _obs, state = env.reset(jax.random.PRNGKey(i), params)
        if int(state.cursor) == day * T_RT:
            return i
    raise AssertionError(f"no key in {limit} draws opened day {day}")


def test_the_ramp_limit_is_respected_between_consecutive_steps(built):
    """(RMP) reaches the previous step, including the switching allowances (§6.3).

    Checked on the realised trajectory rather than on the constraint rows, so it
    tests what the environment actually produced.  The allowances are part of the
    bound: a unit the schedule starts may jump to its committed minimum, and one
    the schedule stops may fall from anywhere, so both are added where the
    commitment changes.
    """
    env, spec, params, case = built
    p_max = np.asarray(case.unit_p_max, np.float64)
    p_min = np.asarray(case.unit_p_min, np.float64)
    delta = spec["period_hours"]
    up = np.asarray(case.unit_ramp_up, np.float64) * p_max * delta * spec["ramp_scale"]
    dn = np.asarray(case.unit_ramp_down, np.float64) * p_max * delta * spec["ramp_scale"]

    # The operating point is built rather than drawn.  Under the old sixty
    # consecutive days a randomly drawn day happened to switch; under the four
    # seasonal segments it does not, and the allowance half of the bound then
    # goes unexercised while the test still reports green.  Six of the sixty days
    # switch within an episode, so the run point is chosen to be one of them and
    # the choice is asserted to have worked -- an empty check is reported, never
    # passed.
    day = _day_that_switches(params, spec["n_days"], EPISODE_LEN - 1)
    assert day is not None, (
        "no day in this window switches commitment within an episode, so the "
        "allowance half of (RMP) is unreachable and this test is vacuous here; "
        "report that rather than widening the bound")
    seed = _seed_opening_day(env, params, day)

    switched = 0
    for state, nxt, _r, _c, _d, _i in _roll(env, params, EPISODE_LEN - 1, seed):
        t = int(state.cursor)
        u_now = np.asarray(params.u_da[t], np.float64)
        u_prev = np.asarray(params.u_da[max(t - 1, 0)], np.float64)
        start = np.maximum(u_now - u_prev, 0.0)
        stop = np.maximum(u_prev - u_now, 0.0)
        switched += int((start + stop).sum())
        move = np.asarray(nxt.p_prev, np.float64) - np.asarray(state.p_prev, np.float64)
        assert (move <= up + p_min * start + 1e-3).all()
        assert (-move <= dn + p_max * stop + 1e-3).all()
    # the run point was chosen for this; if it stops holding the check has gone
    # empty and must be seen, not skipped
    assert switched > 0, (
        f"day {day} was chosen because it switches within an episode, but the "
        f"rolled trajectory saw none")


def test_reward_is_the_settlement_of_the_realised_award(built):
    """Reward comes from `award`, never from the offer.

    Recomputed here from the settlement's own pieces, which `info` reports, so a
    reward assembled from anything else disagrees.
    """
    env, spec, params, _case = built
    for _s, _n, reward, _c, _d, info in _roll(env, params, 3):
        rebuilt = (np.asarray(info["revenue_da"]) + np.asarray(info["revenue_rt"])
                   - np.asarray(info["cost"]))
        np.testing.assert_allclose(np.asarray(reward), rebuilt, rtol=1e-5, atol=1e-3)


def test_costs_carry_shed_only_and_voll_never_reaches_reward(built):
    """The `costs` channel is physical, and `VOLL * s` is not revenue.

    Two halves.  The numeric half checks that the column is the shed energy the
    clearing produced.  The syntactic half checks that `VOLL` cannot reach the
    reward at all, on the syntax tree of the settlement module, because a comment
    cannot enforce it and the day-ahead market carries the same check.
    """
    env, spec, params, _case = built
    assert COST_NAMES == ("shed_mwh",)
    for _s, _n, _r, costs, _d, info in _roll(env, params, 3):
        col = np.asarray(costs)[:, 0]
        assert (col == col[0]).all(), "the shed column is a system quantity"
        np.testing.assert_allclose(col[0], float(info["shed_mwh"]), rtol=1e-9)
        assert col[0] >= 0.0

    tree = ast.parse((Path(env_module.__file__).parent / "settlement.py").read_text())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not {n for n in names if "voll" in n.lower()}


def test_the_shed_column_uses_a_physical_floor(built):
    """An absolute floor, because a clean period leaves ~1e-20.

    Asserted on the value rather than on the constant, so that raising the floor
    without meaning to shows up here.
    """
    env, spec, params, _case = built
    for _s, _n, _r, costs, _d, _i in _roll(env, params, 4):
        v = float(np.asarray(costs)[0, 0])
        assert v == 0.0 or v >= env_module.SHED_FLOOR, (
            f"shed {v:.3e} sits between zero and the floor, so the column is "
            f"reporting solver noise as curtailment")


def test_terminal_obs_is_the_observation_the_episode_ended_in(built):
    """`done` is truncation, so the returned `obs` is the *new* episode's.

    Without `terminal_obs` a learner bootstrapping the truncated step reads a
    value for a state that never followed the one it acted in.  The day-ahead
    market recorded this as a functional gap; this asserts it is closed here.
    """
    env, spec, params, _case = built
    action = env.truthful_action()
    _obs, state = env.reset(jax.random.PRNGKey(0), params)
    auto = jax.jit(env.step_auto_reset)
    seen_done = False
    for _ in range(EPISODE_LEN):
        obs, state, _r, _c, done, info = auto(jax.random.PRNGKey(0), state, action, params)
        if bool(done):
            seen_done = True
            term, fresh = np.asarray(info["terminal_obs"]), np.asarray(obs)
            assert term.shape == fresh.shape
            assert not np.allclose(term, fresh), (
                "terminal_obs equals the post-reset observation, so the "
                "truncation carries no information")
    assert seen_done, "the episode never truncated"
    assert spec["termination"] == "truncation"
