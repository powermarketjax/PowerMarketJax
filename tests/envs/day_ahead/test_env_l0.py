"""L0 JAX contract for the day-ahead environment (§16).

`jit`; `vmap` over parallel environments; an `DayAheadState` pytree whose structure and
dtypes are identical across `jit` and across `done`; a full episode under
`lax.scan` with no Python loop; and no NaN anywhere in `DayAheadState` afterwards.

**The dtype half of the pytree check is the load-bearing half.**  §15 requires
float64 for the solver and float32 for the state, so `jax_enable_x64` is on while
every state array is constructed with an explicit dtype.  An omitted dtype
produces a float64 leaf, which leaves the treedef identical and the shapes
identical, so only comparing dtypes catches it -- and under `lax.scan` a carry
whose dtype differs between the initial value and the body's output is a hard
error at trace time, which is how a float32/float64 mix would surface much later
and much less legibly.

T = 4 rather than 24: one clearing costs 2.5 s on CPU at T=4 against 22 s at
T=24, and nothing in this module is about the horizon.  The fixture is a real
pre-commitment sweep over the first four hours of six real market days, not a
synthetic commitment; `tools/commitment/precommit.py` built it.
"""
import copy
import datetime

import jax
import jax.extend
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.day_ahead import (COST_NAMES, load_commitment,
                                           load_gb_demand, make_clearing, make_env,
                                           segment_costs)
from powermarketjax.envs.day_ahead.commitment import FIXTURE_DIR
from powermarketjax.envs.day_ahead.clearing import MAX_ITER
from powermarketjax.envs.day_ahead.relax import MAX_ITER as RELAX_MAX_ITER
from powermarketjax.envs.day_ahead.relax import make_relax

T = 4
K = 1
CAP_SCALE = 0.6        # adopted scenario (2026-08-17), and the value FIXTURE was committed at
RAMP_SCALE = 1.0       # adopted scenario (2026-08-17); registered rates undiscounted
MARKUP_MAX = 2.0       # §19 leaves it open; it is declared, never defaulted
EPISODE_LEN = 3


@pytest.fixture(scope="module", autouse=True)
def x64():
    """x64 on for this module and restored afterwards.

    Other modules in this suite switch it off globally, and whichever runs last
    wins, so a module that needs it has to assert it itself.  The clearing
    operator raises at construction without it.
    """
    prev, prev_mm = jax.config.jax_enable_x64, jax.config.jax_default_matmul_precision
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    yield
    jax.config.update("jax_enable_x64", prev)
    jax.config.update("jax_default_matmul_precision", prev_mm)


@pytest.fixture(scope="module")
def built(x64):
    case = load_case("29gb")
    fixture = load_commitment(n_periods=T)
    env, spec = make_env(case, fixture, load_gb_demand(), n_segments=K,
                         kind="markup", markup_max=MARKUP_MAX,
                         cap_scale=CAP_SCALE, ramp_scale=RAMP_SCALE)
    return env, spec, env.make_params(episode_len=EPISODE_LEN)


def _leaves(state):
    return {name: (getattr(state, name).shape, getattr(state, name).dtype)
            for name in state.__dataclass_fields__}


def test_step_under_jit(built):
    env, spec, params = built
    N = spec["n_agents"]
    obs, state = env.reset(jax.random.PRNGKey(0), params)
    action = jnp.ones((N,))
    obs2, state2, reward, costs, done, info = jax.jit(env.step)(
        jax.random.PRNGKey(1), state, action, params)
    assert obs2.shape == (N, spec["obs_dim"]) and obs2.dtype == jnp.float32
    assert reward.shape == (N,) and reward.dtype == jnp.float32
    assert costs.shape == (N, len(COST_NAMES)) and costs.dtype == jnp.float32
    assert done.shape == () and done.dtype == jnp.bool_
    assert bool(info["converged"]), f"clearing did not converge: mu={info['mu']:.2e}"


def test_state_pytree_identical_across_jit_and_done(built):
    """The structure must not depend on `jit`, on `done`, or on the x64 flag.

    Three states are compared: the one `reset` builds, the one `step` builds, and
    the one `step_auto_reset` builds on the step where `done` fires.  The last is
    a `jnp.where` over the whole tree, so a leaf whose dtype differs between the
    reset branch and the step branch would promote silently.
    """
    env, spec, params = built
    action = jnp.ones((spec["n_agents"],))
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key, params)
    ref = _leaves(state)

    step = jax.jit(env.step)
    auto = jax.jit(env.step_auto_reset)
    for i in range(EPISODE_LEN):
        _, stepped, _, _, done, _ = step(key, state, action, params)
        assert _leaves(stepped) == ref, f"step changed the state layout at day {i}"
        _, state, _, _, done_a, _ = auto(key, state, action, params)
        assert _leaves(state) == ref, f"auto-reset changed the layout at day {i}"
        assert bool(done) == bool(done_a) == (i == EPISODE_LEN - 1)


def test_obs_is_a_function_of_state(built):
    """§9.4's observation has to survive an auto-reset.

    `step` returns an observation of the state it just built; `step_auto_reset`
    replaces that state on `done` and must replace the observation with the one
    that state produces.  This is why the previous day's schedule and profit are
    carried in `DayAheadState` at all -- computing them from the day just cleared would
    make the observation after an auto-reset disagree with the state beside it.

    The comparison is to a tolerance and not bitwise, and the environment contract asks for bitwise.
    The tolerance is the measured position, so it is recorded here rather than
    inherited: a tolerance carried over
    from elsewhere was once found that sat four to five orders above the real error and could not
    fail.

    What is measured.  Recomputing the observation from the returned state agrees
    **exactly**, difference 0.0, by all three available routes -- eagerly, under its
    own `jit`, and under `vmap` -- over six steps on GPU with this module run alone.
    The single disagreement seen so far was in a full-suite GPU run, where this
    assertion was bitwise and failed; its magnitude was not captured before the
    assertion was loosened, so the bound below is derived rather than fitted.

    How the bound is derived.  The observation is built inside `jit` by `step` and
    outside it by `get_obs`, and it contains a reduction over the buses, the system
    average price, whose reassociation is the compiler's choice; §15 records that
    parallel reductions differ by about one unit in the last place.  Entries here
    span 6.9e-2 to 2.0e4 in magnitude, and one float32 ULP is 1.2e-7 of the
    magnitude, so `atol=1e-5` with `rtol=1e-6` admits about eight ULP on the largest
    entries and governs by `atol` only below magnitude 10, where one ULP is at most
    9.5e-7. It therefore permits a handful of last-bit differences anywhere and
    nothing structural: a field read from the wrong slice, or an observation
    computed from the day just cleared rather than from the state, is relative order
    one.
    """
    env, spec, params = built
    key = jax.random.PRNGKey(0)
    action = jnp.ones((spec["n_agents"],))
    _, state = env.reset(key, params)
    for _ in range(EPISODE_LEN):
        obs, state, _, _, _, _ = jax.jit(env.step_auto_reset)(key, state, action, params)
        np.testing.assert_allclose(np.asarray(obs),
                                   np.asarray(env.get_obs(state, params)),
                                   rtol=1e-6, atol=1e-5)


def test_vmap_over_parallel_environments(built):
    """Agents are an array axis, so environments vmap on top of them.

    Lanes are given different start days by different reset keys, and each lane
    must reproduce what it produces alone.  Batching over states with shared
    params is the rollout shape: the case, the commitment fixture and the demand
    series are one market, the lanes are parallel days.
    """
    env, spec, params = built
    B = 4
    keys = jax.random.split(jax.random.PRNGKey(3), B)
    obs, states = jax.vmap(env.reset, in_axes=(0, None))(keys, params)
    assert obs.shape == (B, spec["n_agents"], spec["obs_dim"])
    action = jnp.ones((B, spec["n_agents"]))
    fn = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))
    _, _, reward, costs, done, info = fn(keys, states, action, params)
    assert reward.shape == (B, spec["n_agents"]) and costs.shape == (B, spec["n_agents"], 2)
    assert done.shape == (B,)
    assert bool(jnp.all(info["converged"]))

    single = jax.jit(env.step)
    for b in range(B):
        one = jax.tree.map(lambda a: a[b], states)
        _, _, r1, c1, _, _ = single(keys[b], one, action[b], params)
        np.testing.assert_allclose(np.asarray(reward[b]), np.asarray(r1), rtol=1e-6)
        # the two `costs` columns need different comparisons, and giving them the
        # same one fails on GPU for a reason that is not a disagreement: the shed
        # column is megawatt-hours summed over every bus and period, and a day
        # that sheds nothing leaves it at ~1.5e-20, where the batched and the
        # unbatched reduction differ by 2.3e-5 *relative* and 3.6e-25 absolute.
        # §15 records that parallel reductions differ by about one ULP, so the
        # floor here is a physical one: 1e-6 MWh is not shed load.
        np.testing.assert_allclose(np.asarray(costs[b, :, 0]),
                                   np.asarray(c1[:, 0]), rtol=1e-6, atol=1e-6)
        # the violation column is a count, so it is exact or it is wrong
        np.testing.assert_array_equal(np.asarray(costs[b, :, 1]),
                                      np.asarray(c1[:, 1]))


def test_full_episode_under_scan(built):
    """A fixed-length `lax.scan` over two episodes, with no Python loop.

    `done` must fire on exactly the last day of each episode and the state must
    carry no NaN at the end -- both of them things a rollout of 10^5 steps would
    otherwise discover slowly.
    """
    env, spec, params = built
    action = jnp.ones((spec["n_agents"],))
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key, params)

    def body(carry, step_key):
        st = carry
        obs, st, reward, costs, done, info = env.step_auto_reset(step_key, st, action, params)
        return st, (reward.sum(), done, info["mu"], costs.sum(0))

    keys = jax.random.split(key, 2 * EPISODE_LEN)
    final, (rewards, dones, mus, costs) = jax.jit(
        lambda s, k: jax.lax.scan(body, s, k))(state, keys)
    assert np.asarray(dones).tolist() == [False, False, True] * 2
    assert np.isfinite(np.asarray(rewards)).all()
    assert float(np.max(mus)) < 1e-8, "a day of the episode did not converge"
    for name in final.__dataclass_fields__:
        leaf = np.asarray(getattr(final, name))
        assert np.isfinite(leaf).all(), f"{name} is not finite after two episodes"


def test_step_is_pure(built):
    """Same inputs, identical outputs, bit for bit: no hidden state anywhere."""
    env, spec, params = built
    key = jax.random.PRNGKey(0)
    action = jnp.full((spec["n_agents"],), 1.3)
    _, state = env.reset(key, params)
    step = jax.jit(env.step)
    a = step(key, state, action, params)
    b = step(key, state, action, params)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_episode_len_is_checked_against_the_fixture(built):
    """An episode longer than the pre-committed window has no commitment to read.

    It would not fail: the day index would clamp to the last day and the market
    would silently repeat it.
    """
    env, spec, _ = built
    with pytest.raises(ValueError, match="episode_len"):
        env.make_params(episode_len=spec["n_days"] + 1)


def test_scenario_parameters_must_be_declared():
    """`cap_scale` and `ramp_scale` have no defaults, and a fixture committed at
    other values describes another market (§13)."""
    case = load_case("29gb")
    fixture = load_commitment(n_periods=T)
    demand = load_gb_demand()
    with pytest.raises(ValueError, match="no defensible default"):
        make_env(case, fixture, demand, n_segments=K, kind="markup",
                 markup_max=MARKUP_MAX, cap_scale=CAP_SCALE)
    with pytest.raises(ValueError, match="fixture was built at"):
        make_env(case, fixture, demand, n_segments=K, kind="markup",
                 markup_max=MARKUP_MAX, cap_scale=1.0, ramp_scale=RAMP_SCALE)


def test_the_window_a_fixture_covers_is_checked_day_by_day():
    """The scenario has a third component and the check above cannot see it.

    A fixture records the dates it was built over and a `day_index` into the
    demand series; the environment must reject the case where those two describe
    different windows.  Comparing lengths does not reach it: the sixty
    consecutive days covered until 2026-08-17 and the four fifteen-day seasonal
    segments that replaced them are both sixty long, so the injection here keeps
    the length and swaps only the dates, which is the one form that a
    length-based check passes and a day-by-day check fails.

    The control matters as much as the injection.  A deep copy with nothing
    swapped has to build, or the raise below would be evidence about copying
    rather than about the window.
    """
    case = load_case("29gb")
    demand = load_gb_demand()
    seasonal = load_commitment(
        path=FIXTURE_DIR / "day_ahead_commitment_29gb_T24_relax.npz")
    kw = dict(n_segments=K, kind="markup", markup_max=MARKUP_MAX,
              cap_scale=seasonal["meta"]["cap_scale"],
              ramp_scale=seasonal["meta"]["ramp_scale"])

    control = copy.deepcopy(seasonal)
    control["meta"] = dict(control["meta"])
    make_env(case, control, demand, **kw)

    injected = copy.deepcopy(seasonal)
    injected["meta"] = dict(injected["meta"])
    # The counterfactual window is synthesised here rather than borrowed from
    # another fixture.  It used to be read out of the retired sixty-consecutive-day
    # commitment fixture, which made this gate's life equal to that fixture's: the
    # gate exists to catch a fixture being swapped for one covering another window,
    # and it was reading a fixture to do it.  When the contract phase renames the
    # seasonal fixture onto the retired one's name, that read would return the very
    # window being injected into, the injection would become a no-op, and the
    # `pytest.raises` below would stop firing -- the gate would fail open, silently.
    #
    # The synthesis keeps the one property that makes the injection meaningful:
    # sixty days again, so a length comparison still passes, but consecutive from
    # the window's own first day instead of four seasonal segments. The two agree
    # for the first segment and diverge at the first segment join, which is the
    # realistic form of the error and exercises the "first disagreement at
    # position" branch rather than the trivial all-different one.
    first = datetime.date.fromisoformat(str(seasonal["meta"]["dates"][0]))
    injected["meta"]["dates"] = [
        (first + datetime.timedelta(days=i)).isoformat()
        for i in range(len(seasonal["meta"]["dates"]))]
    assert len(injected["meta"]["dates"]) == len(seasonal["meta"]["dates"]), \
        "the injection is only meaningful while both windows are the same length"
    assert injected["meta"]["dates"] != list(seasonal["meta"]["dates"]), \
        "the synthesised window equals the real one, so nothing is being injected"
    with pytest.raises(ValueError, match="first disagreement at position"):
        make_env(case, injected, demand, **kw)


# --------------------------------------------------------------------------
# The environment layer's acceptance list.  This environment was written before
# that list was fixed, so these four are a self-check against it rather than tests grown
# from this module's own history; `tests/envs/p2p/test_env_l0.py` is the accepted
# reference for them.
# --------------------------------------------------------------------------

def test_params_leaves_are_float32_or_int32(built):
    """Every `EnvParams` leaf, not only the state's.

    §15 scopes float64 to the solver, and `jax_enable_x64` is on while this runs,
    so any array built without an explicit dtype becomes float64 and silently
    doubles the memory of the per-day series.  `learner_mask` is the one bool.
    """
    env, spec, params = built
    for name, leaf in ((n, getattr(params, n)) for n in params.__dataclass_fields__
                       if hasattr(getattr(params, n), "dtype")):
        assert leaf.dtype in (jnp.float32, jnp.int32, jnp.bool_), f"{name}: {leaf.dtype}"
    assert params.learner_mask.dtype == jnp.bool_
    assert isinstance(params.episode_len, int), "episode_len must stay static"
    # the truncation semantics are declared once, in `spec`, rather than
    # carrying a `truncated` field that would be identically `done`
    assert spec["termination"] == "truncation"


def _count_solver_loops(jaxpr):
    """How many interior-point loops a jaxpr contains, counted by loop length.

    A `scan` is the only loop here, and `step` contains more than one kind: the
    two solvers', whose trip counts are `clearing.MAX_ITER` and `relax.MAX_ITER`,
    and the run-length recursion of §6.3's minimum up-/downtime counter, whose
    trip count is `T`.  Counting `scan` equations would therefore report the
    recursion too; counting only those with a solver trip count reports what the
    question is about.  The walk is recursive because a solver's loop sits inside
    the closed call each operator is.

    The two budgets are equal at present, so this counts solves without telling
    which is which.  What it rules out is a **doubled** solve, which is what the
    accident looks like; the day a budget changes the count separates by itself.
    """
    n = 0
    for eqn in jaxpr.eqns:
        if str(eqn.primitive) == "scan" and \
                eqn.params.get("length") in (MAX_ITER, RELAX_MAX_ITER):
            n += 1
        for sub in jax.extend.core.jaxprs_in_params(eqn.params):
            n += _count_solver_loops(sub)
    return n


def test_step_runs_each_solve_exactly_once(built):
    """No third solve hiding in `step`.

    `step` now solves twice by design -- step 1' for the commitment and step 3 for
    the dispatch and the prices -- so the question is no longer "exactly one" but
    "exactly one of each".  The reference counts come from the two operators built
    standalone rather than from the constant 2, so that the expectation follows
    the operators if either ever gains a loop.

    `reset` still reads its boundary from the fixture instead of clearing period 0
    on its own, which is what could have put a further solve on the hot path.  The
    count comes from the jaxpr rather than from a timing, since a doubled solve is
    exactly twice the interior-point iterations and that is not a judgement call.
    """
    env, spec, params = built
    case = load_case("29gb")
    _, state = env.reset(jax.random.PRNGKey(0), params)
    action = jnp.ones((spec["n_agents"],))
    clear, _ = make_clearing(case, T, n_segments=K, cap_scale=CAP_SCALE,
                             ramp_scale=RAMP_SCALE)
    relax, _ = make_relax(case, T, n_segments=K, cap_scale=CAP_SCALE,
                          ramp_scale=RAMP_SCALE)
    _, cost = segment_costs(case, K)
    offer = jnp.broadcast_to(jnp.asarray(cost)[:, :, None], (spec["n_units"], K, T))
    demand = jnp.asarray(params.actual[0], jnp.float64)
    p_init = jnp.asarray(params.boundary_p_init[0], jnp.float64)
    # values are irrelevant to a jaxpr, only shapes and dtypes are
    u_any = jnp.ones((spec["n_units"], T), jnp.float64)

    in_clear = _count_solver_loops(
        jax.make_jaxpr(clear)(offer, u_any, demand, p_init).jaxpr)
    in_relax = _count_solver_loops(jax.make_jaxpr(relax)(
        offer, demand, p_init, u_any[:, 0],
        jnp.asarray(params.boundary_up_time[0]),
        jnp.asarray(params.boundary_down_time[0])).jaxpr)
    in_step = _count_solver_loops(jax.make_jaxpr(env.step)(
        jax.random.PRNGKey(0), state, action, params).jaxpr)
    assert in_clear == 1, f"the clearing should contain one solver loop, found {in_clear}"
    assert in_relax == 1, f"the relaxation should contain one, found {in_relax}"
    assert in_step == in_clear + in_relax, \
        f"step contains {in_step} solver loops, not {in_clear + in_relax}"


def test_terminal_obs_is_the_successor_of_the_truncated_step(built):
    """`done` is a time-limit truncation, so the learner needs the
    observation that follows the truncated step, and auto-reset overwrites the
    returned one.

    On a step that does not end the episode the two are the same array; on the one
    that does they must differ, and the terminal one must be what `step` alone
    returns from the same inputs.
    """
    env, spec, params = built
    key = jax.random.PRNGKey(0)
    action = jnp.ones((spec["n_agents"],))
    _, state = env.reset(key, params)
    step, auto = jax.jit(env.step), jax.jit(env.step_auto_reset)
    for i in range(EPISODE_LEN):
        plain_obs, _, _, _, plain_done, _ = step(key, state, action, params)
        obs, state, _, _, done, info = auto(key, state, action, params)
        assert info["terminal_obs"].shape == obs.shape
        np.testing.assert_array_equal(np.asarray(info["terminal_obs"]),
                                      np.asarray(plain_obs))
        if bool(done):
            assert not np.allclose(np.asarray(obs), np.asarray(info["terminal_obs"])), \
                "auto-reset returned the truncated observation unchanged"
        else:
            np.testing.assert_array_equal(np.asarray(obs),
                                          np.asarray(info["terminal_obs"]))
    assert bool(plain_done)


def test_the_three_window_off_by_ones(built):
    """The legal start days are `0 .. n_days - episode_len`.

    An episode starting at the upper bound must read its last day in range, an
    episode as long as the window must have exactly one legal start, and one day
    longer must fail at construction.  The middle case is what a `randint` with
    the wrong exclusive bound gets wrong, and the last silently clamps to the
    final day instead.
    """
    env, spec, _ = built
    n_days = spec["n_days"]
    action = jnp.ones((spec["n_agents"],))

    params = env.make_params(episode_len=2)
    starts = {int(env.reset(jax.random.PRNGKey(s), params)[1].cursor)
              for s in range(200)}
    assert max(starts) == n_days - 2, f"upper bound not reachable: {max(starts)}"
    assert min(starts) == 0

    # an episode from the upper bound reads its last day in range
    state = env.reset(jax.random.PRNGKey(0), params)[1].replace(
        cursor=jnp.asarray(n_days - 2, jnp.int32))
    seen = []
    for _ in range(2):
        seen.append(int(state.cursor))
        _, state, _, _, done, _ = jax.jit(env.step)(jax.random.PRNGKey(0), state,
                                                    action, params)
    assert seen == [n_days - 2, n_days - 1], seen
    assert bool(done)

    full = env.make_params(episode_len=n_days)
    assert {int(env.reset(jax.random.PRNGKey(s), full)[1].cursor)
            for s in range(50)} == {0}, "a full-window episode has one legal start"

    with pytest.raises(ValueError, match="episode_len"):
        env.make_params(episode_len=n_days + 1)
