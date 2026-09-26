# Written for this repository on 2026-08-15 -- no upstream counterpart.
"""A full episode of the Swiss configuration of §14.

The Australian layers above cover the mechanism.  What this module covers is
that the **second configuration** reaches the environment at all, and it exists
because every piece of that configuration differs from the one those layers
run on: the period is an hour rather than a quarter, the placement and the
population come from the data rather than from a draw, the photovoltaic array
is per bus rather than a regional series times a capacity, the price is an
annual scalar rather than a series, and the case carries a declared voltage
band rather than a registered one.  A defect in any of those joins leaves every
reported quantity finite, which is why the assertions below are on the shapes
and the invariants rather than on a smoke test that merely returns.

**The duty cycle is the number this module exists to produce, and it overturned
the screen that stood in for it.**  Before the case was registered, how often
the market has anything to do could only be estimated from the registered peak
and the published line utilisation; that screen gave 7.4% of periods at what
was then the candidate operating point.  Measured here from what the clearing
publishes, the same point gives **0.3%** -- the screen is high by more than an
order of magnitude, because it ignores the photovoltaic output that reduces net
withdrawal at exactly the hours the load peaks.  The scaling factor below was
moved on that measurement, and the direction is worth keeping: **both screens
this market used were wrong in opposite directions and photovoltaic output is
the reason for both**, the duty-cycle screen high and the reachability screen
low.

**$c^{\\mathrm{cyc}}$ is undecided** (§18, and item 3 of the note's final
section), so it is declared here as a test constant and **nothing in this
module asserts on a quantity that depends on its value**: it enters the offer
price through §9.5, so a test that pinned reward magnitude would be pinning an
undecided parameter.  What is asserted is structure, which it does not move.
"""
import numpy as np
import pytest
from jax import lax
import jax
import jax.numpy as jnp

from powermarketjax.case.cases.distribution.case459_0 import (
    EN50160_MV_V_MAX, EN50160_MV_V_MIN, create_case459_0)
from powermarketjax.envs.local_flexibility import make_local_flex_env
from powermarketjax.envs.local_flexibility.data import (SWISS_BESS_FILE,
                                                        _default_data_dir,
                                                        load_swiss_flex_series)
from powermarketjax.envs.local_flexibility.env import make_local_flex_params
from powermarketjax.envs.local_flexibility.sensitivity import \
    build_voltage_sensitivity
from powermarketjax.resources.battery import make_battery_bundle

#: The projection year fixes the fleet, its placement and $N^{\mathrm{agent}}$
#: at once (§14), so it is one declared choice and not three.  2050 is the
#: largest population, which is the hardest case for the pytree and dtype
#: checks and the one where the voltage headroom test is tightest.
PROJECTION_YEAR = 2050

#: $\kappa$ multiplying the **series**, whose maximum is 0.852801 of the
#: registered peak, so the registered-peak figure is this divided by that.
#:
#: **Not the calibrated value** -- that is a separate calibration and it has to come from
#: the curtailment share.  This is the smallest value at which the market is
#: not trivial, which is a different and weaker requirement, and it was chosen
#: against a measurement rather than against the screen: at 1.407 the clearing
#: publishes a requirement in 0.3% of periods and 6.2% of episodes, against the
#: 7.4% of periods the registered-peak screen predicted.  **The screen is high
#: by more than an order of magnitude**, because it ignores the photovoltaic
#: output that the clearing sees and that reduces net withdrawal exactly at the
#: hours the load peaks.  Measured over 16 episodes at each value:
#:
#:     kappa (series)  1.407   1.600   1.800   2.000   2.400
#:     periods binding  0.3%    2.3%   27.3%   54.9%   89.3%
#:     episodes any     6.2%   31.2%   75.0%   81.2%  100.0%
#:     shed MWh/step   0.0000  0.0000  0.0118  0.2044  1.5214
KAPPA = 1.800

#: Undecided (§18).  Declared here so the episode runs; see the module
#: docstring for why nothing asserts on a quantity it moves.
CYCLE_COST = 15.0

DELTA = 1.0
EPISODE_LEN = 24


@pytest.fixture(scope="module", autouse=True)
def x64():
    """Restored on the way out, which is not decoration.

    A module that turns `jax_enable_x64` on and leaves it on changes the
    numerics of every test that runs after it in the same process, and one such
    leak made an unrelated assertion fail two directories away.

    **No module under `tests/envs/` leaks it any more.**  This said "one" until
    2026-08-24, when a census found three: `day_ahead/test_relax_l0.py` (two
    functions), `day_ahead/test_relax_l2.py`, and the sibling
    `test_swiss_voltage_headroom_l1.py`, which set it inside a test body.  All
    three were brought onto the shape above the same day, and 42 of 42 functions
    that write the flag now follow it.

    Measured two ways rather than one: an AST census of every write site, and
    running each module alone with a hook that reads the flag after the session.
    The two agree, and the second is the one that decides -- the count from the
    source would have gone on saying three for the one site whose enclosing test
    had been skipped, where the leak was dormant rather than fixed.

    The exposure was **deterministic, not flaky**, which is the sharper claim of
    the two: collection order is fixed here (`pytest` 9.1.1, no ordering plugin
    installed, two `--collect-only` runs byte-identical), so the tests that
    inherited a leaked flag were the same ones every run -- a silent, repeatable
    change of precision rather than a source of nondeterminism.  Which tests
    those were is determinable, everything collected after those two modules in
    the same process, and was never measured.
    """
    previous = (jax.config.jax_enable_x64,
                jax.config.jax_default_matmul_precision)
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    yield
    jax.config.update("jax_enable_x64", previous[0])
    jax.config.update("jax_default_matmul_precision", previous[1])


@pytest.fixture(scope="module")
def built(x64):
    case = create_case459_0(node_v_min=EN50160_MV_V_MIN,
                            node_v_max=EN50160_MV_V_MAX)
    sens = build_voltage_sensitivity(case)
    series = load_swiss_flex_series(projection_year=PROJECTION_YEAR,
                                    tariff_category="C2", tariff_period=2026)
    fleet = (__import__("pandas").read_parquet(_default_data_dir() / SWISS_BESS_FILE)
             .query("projection_year == @PROJECTION_YEAR").sort_values("osmid"))
    nodes = __import__("pandas").read_parquet(
        _default_data_dir() / "SwissDN_459_0_MV_Nodes.parquet")
    bus_of = {str(o): i for i, o in enumerate(nodes["osmid"])}
    agent_bus = np.array([bus_of[str(o)] for o in fleet["osmid"]], np.int32)
    n = len(agent_bus)

    env = make_local_flex_env(case, sens, agent_bus, thermal_margin=0.02,
                              period_hours=DELTA)
    battery = make_battery_bundle(
        n_devices=n, dt_hours=DELTA,
        capacity_mwh=(fleet["capacity_kwh"].to_numpy() / 1000.0).tolist(),
        power_mw=(fleet["nominal_power_kw"].to_numpy() / 1000.0).tolist())
    params = make_local_flex_params(
        load_series=series.load_mw, pv=series.pv,
        energy_price=series.energy_price, battery=battery,
        cycle_cost=np.full(n, CYCLE_COST), load_scale=KAPPA,
        learner_mask=np.ones(n, bool), episode_len=EPISODE_LEN)
    return env, params, n


def test_the_operating_point_is_not_trivial(built):
    """A fixture precondition, asserted before anything is measured on it.

    The test layering requires the non-triviality of an operating point to be a
    precondition rather than prose, in the executable form "perturb an input,
    assert the output moves".  It matters here more than usual: an earlier
    revision of this module ran at a scaling factor where the clearing
    published a requirement in 0.3% of periods, so every assertion below held
    on a market that never cleared anything, and each one of them would have
    kept holding through any defect in the mechanism.
    """
    (reset, _, step_auto, _), params, n = built
    moved = 0
    for key in range(16):
        k = jax.random.PRNGKey(key)
        _, state = reset(k, params)
        act = jax.random.uniform(k, (n, 3), minval=-1.0, maxval=1.0)
        _, _, _, _, _, base = step_auto(k, state, act, params)
        _, _, _, _, _, hi = step_auto(k, state, act.at[:, 0].add(4.0), params)
        moved += (float(np.asarray(base["traded_volume"])) !=
                  float(np.asarray(hi["traded_volume"])))
    print(f"\n  non-triviality: cleared volume moved in {moved}/16 drawn windows")
    assert moved >= 4, (
        f"raising every offer price moved the cleared volume in only {moved} "
        "of 16 drawn windows, so this operating point cannot distinguish "
        "bidding behaviour often enough and the fixture has to be changed")


def _episode(built, key=0):
    (reset, _, step_auto, _), params, n = built
    k = jax.random.PRNGKey(key)
    _, state = reset(k, params)
    keys = jax.random.split(k, EPISODE_LEN)

    def body(carry, kk):
        act = jax.random.uniform(kk, (n, 3), minval=-1.0, maxval=1.0)
        obs, s, reward, costs, done, info = step_auto(kk, carry, act, params)
        return s, (reward, costs, done, info, obs)

    return jax.jit(lambda s, kk: lax.scan(body, s, kk))(state, keys)


def test_a_full_episode_runs_under_scan_with_no_python_loop(built):
    """The JAX contract on the second configuration, not on the first."""
    _, (reward, costs, done, info, obs) = _episode(built)
    n = built[2]
    assert reward.shape == (EPISODE_LEN, n)
    assert costs.shape == (EPISODE_LEN, n, 3)
    assert done.shape == (EPISODE_LEN,)
    assert bool(done[-1]) and not bool(done[:-1].any())
    for name, arr in (("reward", reward), ("costs", costs), ("obs", obs)):
        assert np.isfinite(np.asarray(arr)).all(), name


def test_the_hourly_period_reaches_the_environment(built):
    """$\\Delta$ = 1 h here against 0.25 h in the Australian configuration.

    The period length is closed over at construction and enters the energy of
    every award, so a configuration that silently kept the quarter-hour would
    move every settled quantity by a factor of four while leaving all of them
    finite.
    """
    (_, _, _, spec), _, _ = built
    assert spec["period_hours"] == pytest.approx(DELTA)


def test_the_per_bus_photovoltaic_array_survives_the_join(built):
    """The Swiss series is per bus, so `pv` is a matrix and not an outer product.

    A transposed or mis-indexed join would leave the shapes right only by
    coincidence at this population, so the check is that the array the params
    carry is the one the loader returned, element for element.
    """
    _, params, n = built
    series = load_swiss_flex_series(projection_year=PROJECTION_YEAR,
                                    tariff_category="C2", tariff_period=2026)
    assert series.pv.shape == (8760, n)
    assert np.array_equal(np.asarray(params.pv), series.pv)


def test_the_solver_converges_and_the_sweep_does_not_floor(built):
    """`mu` is the convergence criterion and `converged` never substitutes.

    The power flow's floor flag is separate and is checked with it: a floored
    sweep is a failure rather than a converged solution (pitfalls §2).
    """
    _, (_, _, _, info, _) = _episode(built)
    assert float(np.asarray(info["mu"]).max()) < 1e-6
    assert not bool(np.asarray(info["floor_active"]).any())
    assert bool(np.asarray(info["converged"]).all())


def test_the_voltage_constraints_stay_slack_across_the_episode(built):
    """The realised half of the voltage-headroom test's pair, on the environment's own path.

    That test bounds the corner of the reachable injection box and asserts the
    clearing never approaches either limit.  This is the same statement made
    where the episode actually walks, and it is the half that does not depend
    on the box's injection list being complete -- the list was found short four
    times in one afternoon.
    """
    _, (_, costs, _, info, _) = _episode(built)
    assert float(np.asarray(info["v_under"]).max()) == 0.0
    assert float(np.asarray(info["v_over"]).max()) == 0.0
    # `costs` column 1 is the surviving voltage violation depth, so it carries
    # the same statement through the channel a constrained policy optimises on.
    assert float(np.asarray(costs)[:, :, 1].max()) == 0.0


def test_the_duty_cycle_is_measured_from_the_clearing_not_from_the_screen(built):
    """How often the market has anything to do, replacing the 7.4% screen.

    The screen multiplied the published line utilisation by the peak-normalised
    profile and asked how often any line would leave its rating.  It ignores
    losses, photovoltaic output and planned charging, all three of which the
    clearing sees.  This reports the figure the clearing publishes; the two are
    not required to agree and the point of the test is that the second one
    exists.
    """
    th, v = [], []
    for key in range(32):
        _, (_, _, _, info, _) = _episode(built, key=key)
        th.append(np.asarray(info["req_th_count"]))
        v.append(np.asarray(info["req_v_count"]))
    th, v = np.concatenate(th), np.concatenate(v)
    binding = (th > 0) | (v > 0)
    share = float(binding.mean())
    per_ep = binding.reshape(32, EPISODE_LEN).any(1)
    print(f"\nduty cycle, 32 episodes x {EPISODE_LEN} hourly periods, "
          f"kappa={KAPPA} (series axis):")
    print(f"  periods with a requirement : {share:.1%}  "
          f"(thermal {float((th > 0).mean()):.1%}, voltage {float((v > 0).mean()):.1%})")
    print(f"  episodes with any at all   : {float(per_ep.mean()):.1%}  "
          f"({int(per_ep.sum())}/32)")
    # `reset` draws the start uniformly over the legal window, so 32 keys are
    # 32 windows spread over the year rather than 32 draws of one window.  A
    # single episode from the first key sees January nights and clears nothing,
    # which is a property of where the year starts and not of the market.
    assert 0.0 <= share <= 1.0
    # The voltage family must contribute nothing: the voltage-headroom test measures it slack
    # by six to ten points across the whole window, so a non-zero count here
    # would mean the band reaching the environment is not the declared one.
    assert int(v.max()) == 0
