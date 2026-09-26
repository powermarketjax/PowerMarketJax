"""L3: the same action sequence, stepped through both routes.

The layering stops here.  **There is no L4**: no full-environment alignment is
attempted, and that boundary is fixed deliberately.

What this adds over L2 is the **chain**.  L2 compares one period at a time with
both sides handed the same inputs; the only thing that couples periods in this
market is `p_prev`, so a defect in the coupling is invisible to L2 and shows up
only when both routes are stepped forward independently.  The opening period's
carry is already pinned by `test_boundary_l1`; what this covers is what happens
after it, including the step where auto-reset fires.

Four conditions, and the first two are the ones that make it a comparison at all:

1. **The reference is given the same float32-rounded action**.  The
   environment's action is float32 and the solver runs in float64; handing the
   reference the unrounded value compares two different inputs and calls the
   difference an error.
2. **The comparison must be able to fail**, and a perturbed run is included to
   show that it does.  Four vacuous assertions have already been found in this
   repository, one of them in this market's own `p_prev` criterion.
3. **Neither side borrows a component from the other.**  Each trajectory is
   produced end to end by its own solver.  The recorded failure is an L3 whose
   revenue was formed as "the environment's price times the reference's award",
   after which the environment's award was never compared at all.
4. **The reference carries its own `p_prev`.**  It is the coupling under test, so
   taking it from the environment would remove exactly what this layer exists to
   check.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.day_ahead import load_gb_demand
from powermarketjax.envs.day_ahead.action import make_offer_map
from powermarketjax.envs.day_ahead.clearing import segment_costs
from powermarketjax.envs.real_time import load_da_position
from powermarketjax.envs.real_time.boundary import RAMP_FREE
from powermarketjax.envs.real_time.clearing import MAX_ITER
from powermarketjax.envs.real_time.demand import load_gb_demand_half_hourly
from powermarketjax.envs.real_time.env import make_env

from tests.envs.day_ahead import reference as da_reference

#: Split onto separate lines: as a tuple unpacking this was invisible to both
#: census forms used on the migration.
DELTA = 0.5
#: The adopted scenario, stated rather than read back from the
#: fixture: reading the scales out of it makes any comparison pass by
#: construction.
#: **`real_time.make_env` does not check them**: the refusal comparing a
#: fixture's `meta` against the requested scales lives in `day_ahead.make_env`,
#: while this market's constructor defaults `cap_scale` to 1.0 and validates
#: nothing.  The assertion in the fixture below is the whole guard.
CAP_SCALE = 0.60
RAMP_SCALE = 1.00
#: `_seasons` is the expand-phase name; batch 3 renames it back.
CHAIN = "step1prime_seasons"
MARKUP_MAX = 2.0
N_STEPS = 5
EPISODE_LEN = 3          # shorter than N_STEPS, so auto-reset fires mid-trajectory

#: Bounds on the step-by-step agreement.  **Set by degeneracy, not by precision**,
#: and the distinction matters for reading them: these are four orders looser than
#: the L2 bounds in this same directory, and the reason is not that the trajectory
#: is computed less carefully.
#:
#: Measured 2026-08-15 over three action seeds, five steps each (RTX 4500 Ada,
#: float64 solve, `jax` 0.10.2), at `cap_scale` 0.4 / `ramp_scale` 0.25 over the
#: old sixty consecutive days, against awards of order 7 040 MW:
#:
#:     seed 0   |dAward|  1.5e-4  9.2e-5  2.7e-4  2.5e-4  1.3e-4
#:     seed 1   |dAward|  2.2e-4  4.3e-4  2.1e-4  6.3e-1   5.8e-1
#:     seed 2   |dAward|  1.5e-4  1.1e-4  2.0e-4  5.6e-5  1.1e-4
#:     worst |dLmp| 1.55e-1 $/MWh, on the same step of seed 1
#:
#: **The difference does not accumulate along the chain** -- it fluctuates around
#: 1e-4 and returns to it after the excursion, which is what distinguishes
#: per-solve degeneracy from a carry that amplifies error.  The excursion on seed
#: 1 is one step where the two routes settle on different points of a degenerate
#: optimal face; ties are the norm on these cases.
#:
#: **Re-measured 2026-08-17 on the adopted scenario** (`cap_scale` 0.60,
#: `ramp_scale` 1.00, four seasonal segments, VOLL 10000, CPU, float64):
#:
#:     seed 0   |dAward|  1.1e-4  2.0e-5  1.2e-4  1.4e-4  1.4e-4
#:     seed 1   |dAward|  4.7e-5  1.1e-4  9.7e-5  2.0e-4  1.2e-4
#:     seed 2   |dAward|  1.8e-4  1.2e-4  1.7e-4  1.0e-4  1.8e-4
#:     worst |dLmp| 3.69e-06 $/MWh
#:
#: **The excursion is absent on this scenario**, and the bounds are therefore now
#: ~1e4x the worst award difference rather than ~3x.  They are **not** tightened
#: here: the excursion is a property of landing on a degenerate optimal face, and
#: three seeds not landing on one is not evidence that no seed will.  Tightening
#: to the observed worst would make the bound fail the first time a tie recurs,
#: which is the ordinary case on these instances.  The margin is recorded so the
#: next reader knows it is deliberate rather than unexamined.
#:
#: What makes the bounds still meaningful is the **separation from what the layer
#: is for**: breaking the carry moves the award by 3.525e3 MW
#: (`test_the_comparison_can_fail`), three orders above this bound.  Measured
#: 2026-08-24 on the adopted scenario, CPU, float64, and **bit-identical on
#: action seeds 0, 1 and 2** -- the separation is set by the ramp headroom the
#: dropped carry releases rather than by the offers, so sweeping seeds does not
#: probe it.
#:
#: **This sentence read "order 6e3 MW" and "four orders" until 2026-08-24.**
#: That figure belongs to the retired scenario of the first table: `ramp_scale`
#: went 0.25 -> 1.00, which loosens the ramp, so dropping the carry now moves
#: the award less.  The re-measure of 2026-08-17 updated the agreement tables
#: above and left this separation figure at its old scenario's value, which is
#: the failure mode the block below the tolerances warns about in words.
AWARD_ATOL = 2.0         # MW
LMP_ATOL = 0.5           # $/MWh

#: Measured magnitude of the perturbation the guard injects, MW.  Stated so the
#: guard compares against a number rather than a multiple of the bound above --
#: tying the two together would let a loosened bound silently weaken the guard.
BROKEN_CARRY_MW = 1e3


@pytest.fixture(scope="module", autouse=True)
def x64():
    prev = jax.config.jax_enable_x64, jax.config.jax_default_matmul_precision
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    yield
    jax.config.update("jax_enable_x64", prev[0])
    jax.config.update("jax_default_matmul_precision", prev[1])


@pytest.fixture(scope="module")
def rig(x64):
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
    offer_map, _as = make_offer_map(case, 1, 1, kind="markup", markup_max=MARKUP_MAX)
    return env, spec, case, env.make_params(episode_len=EPISODE_LEN), offer_map


def _actions(spec, seed=0):
    """A fixed, non-truthful action sequence, **rounded to float32 once**.

    Non-truthful on purpose: at the truthful action the offers are the true costs
    and both routes solve a problem whose optimum is insensitive to the action
    map, so an error in the map would not show.  The rounding happens here and the
    same array goes to both sides (condition 1).
    """
    rng = np.random.default_rng(seed)
    return [jnp.asarray(rng.uniform(0.0, 1.0, spec["action_shape"]), jnp.float32)
            for _ in range(N_STEPS)]


def _env_trajectory(env, params, actions):
    """Step the environment, recording what each step cleared.

    **`step` is used and the reset is composed here, rather than reading the
    result out of `step_auto_reset`.**  On the step where `done` fires,
    auto-reset replaces the whole state, so `award_prev` and `lmp_prev` come back
    as the fresh episode's zeros -- the results of the step that just happened are
    not reachable from the state it returns.  That is precisely the situation
    `terminal_obs` exists for, and the first version of this harness read the
    zeroed values and reported a 6 424 MW disagreement against the reference.

    `test_auto_reset_matches_the_composed_step` below checks that composing them
    here reproduces what `step_auto_reset` does, so nothing is lost by splitting.
    """
    _obs, state = env.reset(jax.random.PRNGKey(0), params)
    step = jax.jit(env.step)
    rec = []
    for a in actions:
        cursor, p_prev = int(state.cursor), np.asarray(state.p_prev, np.float64)
        _o, nxt, _r, _c, done, info = step(jax.random.PRNGKey(0), state, a, params)
        rec.append(dict(cursor=cursor, p_prev=p_prev, done=bool(done),
                        award=np.asarray(nxt.award_prev, np.float64),
                        lmp=np.asarray(nxt.lmp_prev, np.float64),
                        mu=float(info["mu"])))
        if bool(done):
            _o, state = env.reset(jax.random.PRNGKey(0), params)
        else:
            state = nxt
    return rec


def _reference_trajectory(case, params, actions, offer_map, opening_cursor,
                          episode_len, n_days):
    """The same sequence through the numpy reference, carrying its own `p_prev`.

    The reference reproduces the environment's own opening construction -- clear
    the first period with the ramp rows slack -- rather than reading `p_prev`
    from the environment, so the carry is produced independently on both sides.
    """
    def period(cursor):
        u = np.asarray(params.u_da[cursor], np.float64)[:, None]
        d = np.asarray([float(params.demand_actual[cursor])])
        return u, d

    def clear(offer, u, d, p_init, ramp_scale):
        return da_reference.clear(case, offer, u, d, p_init, CAP_SCALE, ramp_scale,
                                  period_hours=DELTA, max_iter=MAX_ITER)

    # the opening carry uses the **true-cost** offer, not the action's: the
    # boundary stands for "the system was running on its schedule when the
    # episode opened", and `make_boundary` is built that way.  Using the action's
    # offer here disagreed by 7 380 MW -- a difference in what the two routes
    # were asked, not in how they answered.
    _w, seg_cost = segment_costs(case, 1)
    true_offer = np.asarray(seg_cost, np.float64)[:, :, None]

    def open_carry(cursor):
        u, d = period(cursor)
        return clear(true_offer, u, d, np.zeros(len(case.unit_p_min)),
                     RAMP_FREE)["award"][:, 0]

    cursor = opening_cursor
    p_prev = open_carry(cursor)

    rec, step_in_episode = [], 0
    for a in actions:
        u, d = period(cursor)
        offer = np.asarray(offer_map(a), np.float64)
        out = clear(offer, u, d, p_prev, RAMP_SCALE)
        step_in_episode += 1
        done = step_in_episode >= episode_len
        rec.append(dict(cursor=cursor, award=out["award"][:, 0], lmp=out["lmp"][0],
                        done=done))
        if done:
            # auto-reset: the environment restarts at the same drawn day, so the
            # reference restarts the same way and rebuilds its own carry
            cursor = opening_cursor
            step_in_episode = 0
            p_prev = open_carry(cursor)
        else:
            p_prev = out["award"][:, 0]
            cursor += 1
    return rec


def test_trajectories_agree_step_by_step(rig):
    """The chain, not one period: both routes carry their own `p_prev` forward."""
    env, spec, case, params, offer_map = rig
    actions = _actions(spec)
    mine = _env_trajectory(env, params, actions)
    theirs = _reference_trajectory(case, params, actions, offer_map,
                                   mine[0]["cursor"], EPISODE_LEN, spec["n_days"])

    assert any(r["done"] for r in mine), "auto-reset never fired in this trajectory"
    worst_a = worst_l = 0.0
    for k, (a, b) in enumerate(zip(mine, theirs)):
        assert a["cursor"] == b["cursor"], f"step {k}: the two routes diverged in time"
        assert a["mu"] < 1e-6, f"step {k}: the environment's clearing did not converge"
        worst_a = max(worst_a, float(np.abs(a["award"] - b["award"]).max()))
        worst_l = max(worst_l, float(np.abs(a["lmp"] - b["lmp"]).max()))
    assert worst_a <= AWARD_ATOL, f"award worst {worst_a:.3e} MW"
    assert worst_l <= LMP_ATOL, f"lmp worst {worst_l:.3e} $/MWh"


def test_the_comparison_can_fail(rig):
    """Condition 2: a perturbed reference must part from the environment.

    The perturbation is in the **carry**, not in the offers, because the carry is
    what this layer exists to compare: a reference that reset its `p_prev` to zero
    each step would still clear, still converge, and still produce plausible
    prices -- which is the failure L2 cannot see.
    """
    env, spec, case, params, offer_map = rig
    actions = _actions(spec)
    mine = _env_trajectory(env, params, actions)

    cursor = mine[0]["cursor"]
    worst = 0.0
    for k, a in enumerate(actions):
        u = np.asarray(params.u_da[cursor], np.float64)[:, None]
        d = np.asarray([float(params.demand_actual[cursor])])
        out = da_reference.clear(case, np.asarray(offer_map(a), np.float64), u, d,
                                 np.zeros(len(case.unit_p_min)),   # the broken carry
                                 CAP_SCALE, RAMP_SCALE, period_hours=DELTA,
                                 max_iter=MAX_ITER)
        worst = max(worst, float(np.abs(mine[k]["award"] - out["award"][:, 0]).max()))
        cursor = mine[k]["cursor"] + 1 if not mine[k]["done"] else mine[0]["cursor"]
    assert worst > BROKEN_CARRY_MW, (
        f"dropping the carry moved the award by only {worst:.3e} MW, so this "
        f"trajectory cannot tell a broken coupling from a working one and the "
        f"agreement above is not evidence")


def test_auto_reset_matches_the_composed_step(rig):
    """`step_auto_reset` must equal `step` followed by `reset` on the done step.

    The trajectory harness composes the two by hand so that the final step's
    results stay readable; this asserts the composition is what the environment
    does, so that convenience does not quietly change what is being compared.
    """
    env, spec, case, params, offer_map = rig
    actions = _actions(spec)
    _obs, state = env.reset(jax.random.PRNGKey(0), params)
    step, auto = jax.jit(env.step), jax.jit(env.step_auto_reset)
    for a in actions:
        _o, nxt, _r, _c, done, _i = step(jax.random.PRNGKey(0), state, a, params)
        _oa, auto_state, _ra, _ca, done_a, _ia = auto(
            jax.random.PRNGKey(0), state, a, params)
        assert bool(done) == bool(done_a)
        expected = (env.reset(jax.random.PRNGKey(0), params)[1] if bool(done) else nxt)
        for name in expected.__dataclass_fields__:
            np.testing.assert_allclose(
                np.asarray(getattr(auto_state, name)),
                np.asarray(getattr(expected, name)), rtol=0, atol=0, err_msg=name)
        state = expected


def test_the_reference_never_reads_the_environments_state(rig):
    """Condition 3 and 4, on the source rather than on the numbers.

    A trajectory comparison that borrows a component silently narrows to the
    components it did not borrow; the recorded instance formed revenue as the
    environment's price times the reference's award, and the environment's award
    then went uncompared.  Checked syntactically because it cannot be checked
    numerically -- borrowed values agree perfectly.
    """
    import ast
    from pathlib import Path
    src = Path(__file__).read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "_reference_trajectory")
    names = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    for forbidden in ("award_prev", "lmp_prev", "p_prev_from_env", "step_auto_reset"):
        assert forbidden not in names, (
            f"the reference trajectory reads `{forbidden}` from the environment")
    assert "p_prev" in {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
