"""L1 for `arms.markup_grid`, which had no test and no caller of any kind.

Three defects were repaired on 2026-08-27 and this module is what says so:

1. the sweep had no way to name its days, so it rolled on days drawn at random
   while the other three baselines reach the twelve evaluation days through
   `evaluation.open_day`;
2. each level was rolled on `fold_in(key, i)` and the winner taken by `argmax`
   over the rows -- and in these markets the draw *is* the day, so the winner
   was decided by the days a level happened to be dealt as much as by the level
   (the sampling rule has to be external to the quantity being reported);
3. `at_upper_bound` was computed as "the winner is the last element", which
   means "the top of the grid" only if the grid ascends.

The first half of this module runs a market whose optimum is known by
construction -- profit is an explicit function of the markup and the day, with
no solver anywhere -- because "the sweep finds the best level" cannot be checked
against a market whose best level nobody knows.  **The pre-fix rule is run on
the same instance** (`_prefix_markup_grid`, the code as it stood at `90c1538`)
and is required to pick a different level: without that half the tests would say
what the repaired code does and nothing about what was repaired.

The second half runs the real day-ahead market, because the toy shares no code
with `reset` and can say nothing about whether the pinned keys open the days
they name.  There the reference is built by a different route from the thing
under test: `markup_grid` reaches its answer through `vmap` and `lax.scan`,
while the reference opens each day on its own and steps it unbatched.
"""
import pathlib
import sys
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# `powermarketjax.learning` imports the IPPO module, which needs `optax` from the
# `rl` extra while CI installs `dev`, so a bare import would turn an absent
# optional dependency into a failure.
pytest.importorskip("optax", reason="powermarketjax.learning needs the rl extra")

from powermarketjax.case import load_case
from powermarketjax.envs.day_ahead import load_commitment, load_gb_demand, make_env
from powermarketjax.learning.adapters import unpack_env

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
from benchmark import arms                                       # noqa: E402
from benchmark.evaluation import open_day                        # noqa: E402

# ---------------------------------------------------------------------------
# the toy market
# ---------------------------------------------------------------------------

N_UNITS = 3
N_DAYS = 6
PEAK = 1.40                 # the markup this toy market pays most for
CURV = 1.0                  # $ per squared markup unit

#: Per-day profit offsets in $.  Their spread is what makes a *per-level* day
#: draw decide the winner: the curvature between two neighbouring levels near
#: the peak is 0.0025 $ (one 0.05 step), and these differ by up to 13 $, four
#: orders above it.  That ratio is the whole mechanism of defect 2, and the test
#: asserts it rather than assuming it.
DAY_PROFIT = (0.0, 3.0, -2.0, 8.0, 1.0, -5.0)

#: the truthful reserve offer of the ancillary layout, pre-softplus (its own
#: driver holds the two reserve columns at -800.0, which softplus sends to zero)
BASE_RESERVE = -800.0
#: $ per unit of reserve-offer deviation.  Large on purpose: sweeping the markup
#: level across the reserve columns is not a small error, it is a different
#: experiment, and the toy has to make that visible in the reward.
WRONG_AXIS_PENALTY = 100.0

#: 1.00 to 2.00 in steps of 0.10 -- the action space of markets 01 to 03 with
#: the peak on a grid point.  Coarser than the sweep this arm will really run
#: (0.05) for one reason only: every level here costs one XLA compile of the
#: scan, and the tests below run the grid fourteen times.
LEVELS = tuple(round(1.00 + 0.10 * i, 2) for i in range(11))     # 1.00 .. 2.00

#: seeds the two selection rules are compared over.  Four rather than one
#: because the defect is that the answer moves with the seed, and a single seed
#: cannot show a thing moving.
N_CONTROL_SEEDS = 4


#: the toy episode: one step, as market 01 evaluates (`run_eval_01.py`)
TOY_EPISODE_LEN = 1


class ToyParams(NamedTuple):
    """`params` with the `episode_len` the horizon guard reads.

    A NamedTuple because that is a pytree JAX already knows how to carry through
    `vmap` and `lax.scan`; the toy transition reads nothing out of it, so the
    single field is there for the guard and for nothing else.
    """

    episode_len: int = TOY_EPISODE_LEN


def _toy_market(peak=PEAK, layout="markup", n_days=N_DAYS,
                offsets=DAY_PROFIT):
    """A market whose profit is an explicit function of the markup and the day.

    `reset` draws the day from its key and the transition draws nothing, which
    is the property markets 01, 02 and 03 all have and the property defect 2
    turns into a bias.  `layout="ancillary"` gives the `(n_units, 3)` action
    whose last two columns are reserve offers rather than markups.
    """
    offs = jnp.asarray(offsets, jnp.float64)
    shape = (N_UNITS,) if layout == "markup" else (N_UNITS, 3)
    baseline = (jnp.ones(shape, jnp.float64) if layout == "markup"
                else jnp.asarray(np.column_stack(
                    [np.ones(N_UNITS), np.full(N_UNITS, BASE_RESERVE),
                     np.full(N_UNITS, BASE_RESERVE)])))

    def reset(key, params):
        day = jax.random.randint(key, (), 0, n_days, dtype=jnp.int32)
        state = dict(day=day, step=jnp.zeros((), jnp.int32))
        return jnp.zeros((1,), jnp.float64), state

    def step_auto_reset(key, state, action, params):
        action = jnp.asarray(action, jnp.float64)
        level = jnp.mean(action[..., 0] if action.ndim > 1 else action)
        penalty = (0.0 if action.ndim == 1 else
                   WRONG_AXIS_PENALTY * jnp.mean(
                       jnp.abs(action[:, 1:] - BASE_RESERVE)))
        r = -CURV * (level - peak) ** 2 + offs[state["day"]] - penalty
        nxt = dict(day=state["day"], step=state["step"] + 1)
        done = nxt["step"] >= TOY_EPISODE_LEN
        _o, fresh = reset(key, params)
        nxt = jax.tree_util.tree_map(lambda a, b: jnp.where(done, a, b),
                                     fresh, nxt)
        info = dict(converged=jnp.ones((), bool))
        return (jnp.zeros((1,), jnp.float64), nxt,
                jnp.full((N_UNITS,), r), jnp.zeros((N_UNITS, 1)), done, info)

    spec = dict(n_agents=N_UNITS, action_shape=shape, action_low=1.0,
                action_high=2.0, baseline_action=baseline, kind="markup")
    return (reset, step_auto_reset, step_auto_reset, spec)


def _prefix_markup_grid(env, params, levels, n_envs, horizon, key,
                        action_of=None):
    """The selection rule as it stood before 2026-08-27, kept as the control.

    Copied from `tools/benchmark/arms.py` at `90c1538`: one `fold_in(key, i)`
    per level, `argmax` over the rows that come back.  It calls today's
    `rollout_action` through its unpinned path, which this repair did not touch,
    so this is the old rule running on the current rollout rather than a
    re-implementation of both.

    Checked against the file itself rather than trusted: on 2026-08-27 the
    `markup_grid` of `90c1538`, loaded from `git show` outside this checkout,
    returned 1.9, 1.6, 1.7, 1.5 on seeds 0 to 3 of the market below, and this
    function returned the same four.
    """
    spec = env[3]
    if action_of is None:
        lo, hi = jnp.asarray(spec["action_low"]), jnp.asarray(spec["action_high"])
        action_of = lambda v: jnp.clip(jnp.full(spec["action_shape"], v), lo, hi)
    rows = []
    for i, v in enumerate(levels):
        k = jax.random.fold_in(key, i)
        m = arms.rollout_action(env, params, action_of(float(v)), n_envs,
                                horizon, k)
        rows.append({"level": float(v), "reward_mean": float(m["reward_mean"])})
    best = int(np.argmax([r["reward_mean"] for r in rows]))
    return rows, best, best == len(levels) - 1


def _days_dealt(reset, params, key, i, n_envs):
    """The days `fold_in(key, i)` deals -- derivation copied from the rollout.

    Copied rather than re-derived for the reason the sibling module gives: a
    reference that split keys differently would disagree for a legitimate reason
    and the test would be measuring key handling.
    """
    keys = jax.random.split(jax.random.fold_in(key, i), n_envs + 1)
    _o, state = jax.vmap(reset, in_axes=(0, None))(keys[:n_envs], params)
    return np.asarray(state["day"])


@pytest.fixture(scope="module", autouse=True)
def x64():
    """float64 for this module; the day-ahead clearing refuses to build without it."""
    prev, prev_mm = jax.config.jax_enable_x64, jax.config.jax_default_matmul_precision
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    yield
    jax.config.update("jax_enable_x64", prev)
    jax.config.update("jax_default_matmul_precision", prev_mm)


def test_the_grid_finds_a_known_interior_optimum(x64):
    """The repaired sweep lands on the level this market really pays most for.

    `PEAK` is interior by construction, which is what makes this able to fail:
    an implementation that always returned the top of the grid would pass a test
    whose optimum sat on the boundary.  The rows are checked against the closed
    form as well as the winner, because "it found the peak" also holds for a
    sweep that reported one number twenty-one times.
    """
    env = _toy_market()
    params, key = ToyParams(), jax.random.PRNGKey(0)
    days = list(range(N_DAYS))

    rows, best, at_top = arms.markup_grid(
        env, params, LEVELS, None, 1, key, days=days,
        day_of_state=lambda s: int(s["day"]))

    assert rows[best]["level"] == pytest.approx(PEAK), (
        f"the sweep chose {rows[best]['level']} where this market's optimum is "
        f"{PEAK} by construction")
    assert at_top is False, "the constructed optimum is interior, not the top"
    assert rows[best]["days"] == days, "a row cannot state which days it ran on"

    # every level saw all six days, so the day term is the same constant in every
    # row and what is left is the curvature
    mean_off = float(np.mean(DAY_PROFIT))
    for r in rows:
        assert r["reward_mean"] == pytest.approx(
            -CURV * (r["level"] - PEAK) ** 2 + mean_off, abs=1e-9)
    # the per-day totals are the third thing a product is written from: three
    # units each paid the period's profit
    np.testing.assert_allclose(
        np.asarray(rows[best]["reward_sum_per_env"]),
        N_UNITS * (np.asarray(DAY_PROFIT) - CURV * (rows[best]["level"] - PEAK) ** 2),
        atol=1e-9)


def test_the_rule_that_was_replaced_lets_the_seed_pick_the_winner(x64):
    """The control, without which the test above says nothing about the repair.

    Both rules run on the same four seeds, the same grid and the same six days'
    worth of environments.  The repaired rule returns one level on all eight,
    because the days are pinned and this market's transition draws nothing; the
    pre-fix rule returns several, and which one it returns is decided by the days
    that level happened to be dealt -- asserted directly, so that a seed where it
    happens to land near the peak still shows the mechanism.

    **The distance is not the point and is not asserted as one.** Measured
    2026-08-27 on this grid, seeds 0 to 7: the pre-fix rule chose
    1.9, 1.6, 1.7, 1.5, 1.3, 1.3, 1.8, 2.0 -- seven distinct levels and the true
    peak on none of the eight.  On a 0.05 grid the same seed 0 chose 1.45, one
    step from the peak: an instance where the defect is fully present and the
    answer still looks reasonable.  That is the harder case for a reader, so it
    is recorded rather than tuned away by choosing a seed.
    """
    env, params = _toy_market(), ToyParams()
    days = list(range(N_DAYS))
    day_of = lambda s: int(s["day"])

    old_picks, new_picks = [], []
    for s in range(N_CONTROL_SEEDS):
        key = jax.random.PRNGKey(s)
        _r, b_old, _t = _prefix_markup_grid(env, params, LEVELS, len(days), 1, key)
        old_picks.append(LEVELS[b_old])
        _r, b_new, _t = arms.markup_grid(env, params, LEVELS, None, 1, key,
                                         days=days, day_of_state=day_of)
        new_picks.append(LEVELS[b_new])

    assert set(new_picks) == {PEAK}, (
        f"the repaired sweep moved with the seed: {new_picks}")
    assert len(set(old_picks)) >= 3, (
        f"the pre-fix rule returned {set(old_picks)} over {N_CONTROL_SEEDS} "
        f"seeds; on this instance it is not visibly seed-dependent and the "
        f"control has no content")
    assert sum(p == PEAK for p in old_picks) <= N_CONTROL_SEEDS // 3

    # why it moves: on seed 0 its winner is exactly the level dealt the best days
    key = jax.random.PRNGKey(0)
    dealt = [float(np.mean(np.asarray(DAY_PROFIT)[
        _days_dealt(env[0], params, key, i, len(days))]))
        for i in range(len(LEVELS))]
    day_spread = max(dealt) - min(dealt)
    curv_spread = CURV * max((v - PEAK) ** 2 for v in LEVELS)
    assert day_spread > 5.0 * curv_spread, (
        f"the days dealt spread {day_spread:.3f} $ against a curvature range of "
        f"{curv_spread:.3f} $; this instance cannot show the bias")
    _r, b_old0, _t = _prefix_markup_grid(env, params, LEVELS, len(days), 1, key)
    assert b_old0 == int(np.argmax(dealt)), (
        "the pre-fix winner is not the level that was dealt the best days, so "
        "this instance is not demonstrating the mechanism it was built for")


def test_at_upper_bound_is_reported_and_is_not_always_the_same_answer(x64):
    """The flag, both ways round.

    An interior optimum must give `False` and an optimum at or above the top of
    the grid must give `True`.  Only the pair pins it: a flag hard-wired either
    way passes one of these two.
    """
    params, key = ToyParams(), jax.random.PRNGKey(0)
    days = list(range(N_DAYS))
    day_of = lambda s: int(s["day"])

    _r, best_in, top_in = arms.markup_grid(
        _toy_market(peak=PEAK), params, LEVELS, None, 1, key, days=days,
        day_of_state=day_of)
    # a market that pays most above the grid: the sweep's answer is a lower
    # bound and the flag is what says so
    _r, best_up, top_up = arms.markup_grid(
        _toy_market(peak=3.0), params, LEVELS, None, 1, key, days=days,
        day_of_state=day_of)

    assert (top_in, top_up) == (False, True)
    assert best_in == LEVELS.index(PEAK) and best_up == len(LEVELS) - 1


def test_the_action_reaching_the_market_is_the_one_action_of_built(x64):
    """Market 03's layout: the default sweeps the wrong axis, and it shows.

    The ancillary action's last two columns are reserve offers held at
    `BASE_RESERVE`, and the default `action_of` fills *every* coordinate with the
    level and clips the result into `[action_low, action_high]` -- which moves
    the reserve columns by 801 units while the sweep reports a markup.  The toy
    prices that deviation, so "which `action_of` was used" is visible in the
    number rather than only in the caller's intent.
    """
    env = _toy_market(layout="ancillary")
    params, key = ToyParams(), jax.random.PRNGKey(0)
    days, day_of = list(range(N_DAYS)), lambda s: int(s["day"])
    baseline = np.asarray(env[3]["baseline_action"])

    def markup_column_only(v):
        a = np.array(baseline, copy=True)
        a[:, 0] = v
        return jnp.asarray(a)

    rows, best, _top = arms.markup_grid(env, params, LEVELS, None, 1, key,
                                        action_of=markup_column_only, days=days,
                                        day_of_state=day_of)
    assert rows[best]["level"] == pytest.approx(PEAK)

    wrong, wrong_best, _t = arms.markup_grid(env, params, LEVELS, None, 1, key,
                                             days=days, day_of_state=day_of)
    # the default is not merely a different answer, it is a different experiment:
    # the penalty it collects is orders above the curvature the sweep is meant
    # to be resolving
    assert (rows[best]["reward_mean"] - wrong[wrong_best]["reward_mean"]
            > 100.0), (
        "the default `action_of` scored the same as the market's own layout, so "
        "this test cannot tell whether `action_of` reaches the environment")


@pytest.mark.parametrize("kwargs, match", [
    (dict(levels=(1.0, 1.5, 1.2), days=[0, 1]), "strictly increasing"),
    (dict(levels=(1.5,), days=[0, 1]), "at least two levels"),
    (dict(levels=LEVELS, days=[0, 1], n_envs=3), "against 2 named days"),
    (dict(levels=LEVELS, days=[0, 1], horizon=2), "exceeds episode_len"),
    (dict(levels=LEVELS, days=[0, 1], day_of_state=None), "needs `day_of_state`"),
    (dict(levels=LEVELS), "no length"),
])
def test_the_grid_refuses_the_shapes_that_would_report_a_different_run(
        x64, kwargs, match):
    """Each guard refuses loudly rather than reporting something else.

    Every one of these is a way of producing a number that looks like the one
    asked for: an unsorted grid mislabels the boundary flag, a one-point grid
    raises it vacuously, a mismatched `n_envs` or an over-long horizon rolls days
    nobody named, and a missing `day_of_state` would leave the day unverified.
    """
    env = _toy_market()
    call = dict(levels=LEVELS, n_envs=None, horizon=1,
                day_of_state=lambda s: int(s["day"]))
    call.update(kwargs)
    with pytest.raises(ValueError, match=match):
        arms.markup_grid(env, ToyParams(), call["levels"], call["n_envs"],
                         call["horizon"], jax.random.PRNGKey(0),
                         days=call.get("days"),
                         day_of_state=call["day_of_state"])


def test_rollout_action_refuses_a_reset_key_count_that_is_not_the_axis(x64):
    """The same rule one level down, where the keys meet the `vmap` axis."""
    env = _toy_market()
    keys = jnp.stack([jax.random.PRNGKey(i) for i in range(3)])
    with pytest.raises(ValueError, match="reset keys against n_envs"):
        arms.rollout_action(env, ToyParams(), jnp.ones((N_UNITS,)), 2, 1,
                            jax.random.PRNGKey(0), reset_keys=keys)


def test_it_still_compiles_and_the_days_come_back_in_the_order_named(x64):
    """L0 for both rollout paths, and the order the environment axis is in.

    A training-side driver wraps `rollout_action` in
    `jax.jit`, so the pinned path has to compile as well as the unpinned one --
    it takes an argument the other does not, and a stacked key array is not the
    thing `jax.random.split` returns.  The per-environment totals are asserted
    against a **deliberately unsorted** day list, because "one number per day"
    is worth nothing to a product unless the numbers are in the order the days
    were named.
    """
    env, params = _toy_market(), ToyParams()
    action = jnp.full((N_UNITS,), PEAK)
    days = [3, 1, 0]
    reset_keys = jnp.stack([open_day(env[0], params, d,
                                     lambda st: int(st["day"]))[0] for d in days])

    unpinned = jax.jit(lambda a, k: arms.rollout_action(
        env, params, a, N_DAYS, 1, k))
    pinned = jax.jit(lambda a, k, r: arms.rollout_action(
        env, params, a, len(days), 1, k, reset_keys=r))

    assert np.isfinite(float(unpinned(action, jax.random.PRNGKey(0))["reward_mean"]))
    got = pinned(action, jax.random.PRNGKey(0), reset_keys)
    np.testing.assert_allclose(
        np.asarray(got["reward_sum_per_env"]),
        N_UNITS * np.asarray([DAY_PROFIT[d] for d in days]), atol=1e-9)


# ---------------------------------------------------------------------------
# the real day-ahead market
# ---------------------------------------------------------------------------

T = 4
K = 1
CAP_SCALE = 0.6        # adopted scenario (2026-08-17); as in tests/learning/test_arms_l1.py
RAMP_SCALE = 1.0       # adopted scenario (2026-08-17); registered rates undiscounted
MARKUP_MAX = 2.0


@pytest.fixture(scope="module")
def built(x64):
    case = load_case("29gb")
    fixture = load_commitment(n_periods=T)
    env, spec = make_env(case, fixture, load_gb_demand(), n_segments=K,
                         kind="markup", markup_max=MARKUP_MAX,
                         cap_scale=CAP_SCALE, ramp_scale=RAMP_SCALE)
    four = unpack_env((env, spec))
    # `episode_len=1` is market 01's evaluation setting (`run_eval_01.py`), one
    # day per episode, which is also what lets the horizon stay inside it
    return four, env.make_params(episode_len=1), four[3]


def _reference_day(four, params, action, day, step_key):
    """One named day opened and stepped on its own -- no `vmap`, no `scan`."""
    reset, _step, step_auto_reset, _spec = four
    key, state = open_day(reset, params, day, lambda st: int(st.cursor))
    _o, _s, reward, _c, _d, info = step_auto_reset(step_key, state, action,
                                                   params)
    return np.asarray(reward, np.float64), info


def test_the_pinned_days_are_the_days_the_market_actually_opened(built):
    """The reward reported for each pinned day is that day's own reward.

    This is the property the whole column rests on: the other three baselines are
    computed one named day at a time, and a grid whose rows came from other days
    would be compared against them as though they were the same sample.

    The reference reaches each day through `open_day` and steps it unbatched,
    while `markup_grid` reaches all of them through `vmap` over the pinned keys,
    so agreement is not a re-implementation agreeing with itself.
    """
    four, params, spec = built
    days = [0, 3]
    key = jax.random.PRNGKey(0)
    levels = (1.2, 1.6)

    rows, best, at_top = arms.markup_grid(
        four, params, levels, None, 1, key,
        days=days, day_of_state=lambda st: int(st.cursor))

    # the step keys `rollout_action` hands the two environments, derivation
    # copied from it (this market's `step` consumes no key, so nothing here
    # depends on the copy being faithful -- it is copied so that a market that
    # started consuming one would be caught by this test rather than by a report)
    k, k_env = jax.random.split(key)
    env_keys = jax.random.split(k_env, len(days))

    for row, level in zip(rows, levels):
        action = jnp.full((spec["n_agents"],), float(level))
        per_day = [_reference_day(four, params, action, d, env_keys[i])[0].sum()
                   for i, d in enumerate(days)]
        # Not exact, and the reason is the reduction rather than the wiring:
        # `reward` is float32 (§15 keeps the state in float32 while the clearing
        # runs in float64) and the two sides sum sixty-six agents in different
        # orders.  Measured 2026-08-27 on CPU, `case29gb` T=4, days 0 and 3,
        # levels 1.2 and 1.6: the largest disagreement of the four numbers is
        # 5.74e-08 relative, under half of one float32 ULP (1.19e-07), so
        # `rtol=1e-6` leaves a factor of about seventeen.
        np.testing.assert_allclose(np.asarray(row["reward_sum_per_env"]),
                                   np.asarray(per_day), rtol=1e-6, atol=0.0)

    # non-triviality: the two days must differ, or "the right days were opened"
    # would hold for any two days at all
    got = np.asarray(rows[0]["reward_sum_per_env"])
    assert abs(got[0] - got[1]) > 1e-6, (
        f"the two pinned days returned the same profit ({got}); this fixture "
        f"cannot tell one day from another")
    assert isinstance(best, int) and isinstance(at_top, bool)


def test_pinning_the_days_is_what_makes_the_sweep_reproducible(built):
    """With days pinned the answer does not depend on `key`; without, it does.

    The first half is the measured claim the docstring's choice rests on -- these
    markets draw nothing in `step`, so pinning the day pins the whole rollout,
    which is why repeats would report a spread of identically zero.  The second
    half is what makes the first half judge anything: on the same grid without
    days, two keys give two different numbers, so an implementation that ignored
    `key` altogether could not pass both.
    """
    four, params, _spec = built
    days = [1, 4]
    day_of = lambda st: int(st.cursor)
    levels = (1.2, 1.6)

    a, _b, _t = arms.markup_grid(four, params, levels, None, 1,
                                 jax.random.PRNGKey(0), days=days,
                                 day_of_state=day_of)
    b, _b2, _t2 = arms.markup_grid(four, params, levels, None, 1,
                                   jax.random.PRNGKey(7), days=days,
                                   day_of_state=day_of)
    assert [r["reward_mean"] for r in a] == [r["reward_mean"] for r in b], (
        "two keys gave two answers on pinned days: some market on this path now "
        "consumes randomness in `step`, and one shared key no longer removes "
        "the sampling noise -- the sweep needs repeats and a reported spread")

    c, _b3, _t3 = arms.markup_grid(four, params, levels, 2, 1,
                                   jax.random.PRNGKey(0))
    d, _b4, _t4 = arms.markup_grid(four, params, levels, 2, 1,
                                   jax.random.PRNGKey(7))
    assert [r["reward_mean"] for r in c] != [r["reward_mean"] for r in d], (
        "unpinned, two different keys drew the same days; this fixture cannot "
        "see a key effect at all, so the half above proves nothing")
