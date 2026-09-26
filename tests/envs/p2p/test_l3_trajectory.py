"""L3 for the P2P market: step-by-step comparison under one action sequence (§17).

L2 in `test_clearing_l2.py` answers whether the auction is implemented correctly,
by clearing thousands of independent populations against a numpy reference.  What
it cannot answer is whether the *sequence* is right, because every one of those
markets is fed inputs assembled by the test rather than by the previous period.
Three classes of defect survive L2 untouched, and each of them is a plausible way
to get an environment wrong rather than an exotic one:

* the state of charge failing to carry, so that period ``t+1`` sees the battery
  the reset left rather than the one period ``t`` produced;
* the cursor advancing by the wrong amount or being read before it advances, so
  that the auction is run against the wrong row of the exogenous series;
* the observation or the reward being assembled from a different period's
  quantities than the ones that were cleared.

L3 drives the same float32 action sequence through `step` and through
`reference.step_ref`, and compares the submission, the clearing, the money and
the state of charge at every period.  The reference carries its own state of
charge forward, so a carry that is wrong in the implementation shows up as a
divergence that grows rather than as a single mismatched period, which is the
signature worth having.

**The action sequence is rounded to float32 before either side sees it.**
That is a precondition of the comparison: the implementation's action
space is float32, so handing the reference a float64 action would compare two
different problems and attribute the difference to the algorithm.

The tolerances are the L2 ones, per quantity, since the arithmetic is the same
and only its inputs are now produced by the trajectory. `done` is never reached:
the episode is one period longer than the run, because at `done` the
implementation resets from a key this reference does not model, and replicating
that here would couple the reference to the implementation's key handling for no
gain -- `test_env_l1.py` already pins the reset.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.envs.p2p import make_p2p_env, make_p2p_params
from powermarketjax.resources.battery import make_battery_bundle

from .reference import step_ref

N_AGENTS = 12
N_PERIODS = 300
STEPS = 64
EPISODE_LEN = STEPS + 1          # so `done` is never set inside the run
PI_EXP, PI_RET = 73.0, 333.4
DELTA = 0.25
KAPPA = 13.88

#: Same grading as L2, whose derivation is in `test_clearing_l2.py`: the order
#: and the price are exact in exact arithmetic and differ only by float32
#: rounding, the awards are not.  Scaled here to household magnitudes -- an award
#: is thousandths of a megawatt hour where L2 measured at unit scale -- so the
#: award tolerance is relative to the traded volume of the same period.
AWARD_RTOL = 1e-5
PRICE_ATOL = 1e-3               # EUR/MWh, against prices of order 100
SOC_ATOL = 1e-6
MONEY_ATOL = 1e-4               # EUR per agent per period
POWER_ATOL = 1e-8               # MW, household scale


@pytest.fixture(scope="module")
def trajectory():
    """One episode through the implementation, with everything kept per period."""
    rng = np.random.default_rng(7)
    one_way = math.sqrt(0.85)
    battery = make_battery_bundle(
        n_devices=N_AGENTS, dt_hours=DELTA,
        capacity_mwh=rng.uniform(0.008, 0.014, N_AGENTS).tolist(),
        power_mw=rng.uniform(0.003, 0.007, N_AGENTS).tolist(),
        eta_charge=one_way, eta_discharge=one_way,
        soc_min=0.15, soc_max=1.0, initial_soc=0.5, cycle_cost_per_mwh=0.0)
    # Half the premises inject in any period, so both sides are populated and
    # the marginal position moves from period to period.
    injection = (rng.integers(0, 2, (N_PERIODS, N_AGENTS))
                 * rng.uniform(0.0, 6e-3, (N_PERIODS, N_AGENTS)))
    load = rng.uniform(0.0, 6e-3, (N_PERIODS, N_AGENTS))
    kappa = np.full(N_AGENTS, KAPPA, np.float32)
    params = make_p2p_params(
        p_pv=injection, load=load, battery=battery, kappa=kappa,
        learner_mask=np.ones(N_AGENTS, bool), episode_len=EPISODE_LEN)
    reset, step, _, _ = make_p2p_env(N_AGENTS, PI_EXP, PI_RET, DELTA)

    # float32 before either side sees it
    actions = jnp.asarray(
        rng.uniform(-1.0, 1.0, (STEPS, N_AGENTS, 2)), jnp.float32)

    key = jax.random.PRNGKey(3)
    _, state = reset(key, params)
    jstep = jax.jit(step)
    got = []
    for k in range(STEPS):
        cursor = int(state.cursor)
        soc = np.asarray(state.soc, np.float64)
        obs, state, reward, costs, done, info = jstep(
            key, state, actions[k], params)
        assert not bool(done), "the run must not reach the truncation step"
        got.append(dict(cursor=cursor, soc=soc, reward=np.asarray(reward),
                        costs=np.asarray(costs), price=float(info["clearing_price"]),
                        volume=float(info["traded_volume"]),
                        award=np.asarray(state.award_prev),
                        net=np.asarray(state.net_prev),
                        soc_next=np.asarray(state.soc, np.float64)))
    return dict(params=params, battery=battery, kappa=kappa, actions=actions,
                injection=injection, load=load, got=got)


def test_the_cursor_advances_by_one_period_each_step(trajectory):
    """The cheapest of the three defects to introduce and the hardest to see."""
    cursors = [row["cursor"] for row in trajectory["got"]]
    assert cursors == list(range(cursors[0], cursors[0] + STEPS))
    assert 0 <= cursors[0] <= N_PERIODS - EPISODE_LEN


def test_step_by_step_against_the_reference(trajectory):
    """The assertion this file exists for: every period, whole chain.

    The reference carries its own state of charge, so it agrees with the
    implementation only if the carry, the cursor and the assembly all agree.
    """
    battery = trajectory["battery"]
    kappa = np.asarray(trajectory["kappa"], np.float64)
    actions = np.asarray(trajectory["actions"], np.float64)
    got = trajectory["got"]

    soc = np.asarray(got[0]["soc"], np.float64)          # same start, then free
    worst = dict(price=0.0, award=0.0, soc=0.0, money=0.0, power=0.0)

    for k, row in enumerate(got):
        cursor = row["cursor"]
        submission, cleared, money, soc_next = step_ref(
            actions[k], soc, trajectory["injection"][cursor],
            trajectory["load"][cursor], battery, kappa,
            PI_EXP, PI_RET, DELTA)

        # the submission: net position and deliverable power
        worst["power"] = max(worst["power"], float(np.abs(
            row["net"].astype(np.float64) - submission["net_position"]).max()))
        assert np.abs(row["net"].astype(np.float64)
                      - submission["net_position"]).max() < POWER_ATOL, k

        # the clearing: price exactly, award relative to the volume
        worst["price"] = max(worst["price"],
                             abs(row["price"] - cleared["clearing_price"]))
        assert abs(row["price"] - cleared["clearing_price"]) < PRICE_ATOL, k
        volume = max(cleared["traded_volume"], 1e-12)
        # `award_prev` is signed, sell less buy (env.py:91); a participant is on
        # one side only (§5) so this is that side's award with its sign.
        award_ref = cleared["award_sell"] - cleared["award_buy"]
        rel = np.abs(row["award"].astype(np.float64) - award_ref) / volume
        worst["award"] = max(worst["award"], float(rel.max()))
        assert rel.max() < AWARD_RTOL, k

        # the money
        worst["money"] = max(worst["money"], float(np.abs(
            row["reward"].astype(np.float64) - money["profit"]).max()))
        assert np.abs(row["reward"].astype(np.float64)
                      - money["profit"]).max() < MONEY_ATOL, k

        # and the carry, which is what makes this a trajectory test
        worst["soc"] = max(worst["soc"], float(np.abs(
            row["soc_next"] - soc_next).max()))
        assert np.abs(row["soc_next"] - soc_next).max() < SOC_ATOL, k
        soc = soc_next

    # Reported so that a tolerance and the measurement behind it cannot drift
    # apart silently: if these approach the constants above, revisit them.
    print("\nL3 worst deviations over "
          f"{STEPS} periods x {N_AGENTS} agents: {worst}")


def test_the_reference_diverges_when_the_carry_is_broken(trajectory):
    """Guards the test above: it must be able to fail.

    A trajectory comparison that keeps re-seeding the reference from the
    implementation's state each period degenerates into L2 repeated, and would
    pass even with no carry at all.  This feeds the reference a frozen state of
    charge instead and requires the comparison to break, which is what shows the
    carry is genuinely being checked.
    """
    battery = trajectory["battery"]
    kappa = np.asarray(trajectory["kappa"], np.float64)
    actions = np.asarray(trajectory["actions"], np.float64)
    got = trajectory["got"]
    frozen = np.asarray(got[0]["soc"], np.float64)

    diverged = False
    for k, row in enumerate(got):
        submission, _, _, _ = step_ref(
            actions[k], frozen, trajectory["injection"][row["cursor"]],
            trajectory["load"][row["cursor"]], battery, kappa,
            PI_EXP, PI_RET, DELTA)
        if np.abs(row["net"].astype(np.float64)
                  - submission["net_position"]).max() >= POWER_ATOL:
            diverged = True
            break
    assert diverged, (
        "freezing the reference's state of charge did not break the comparison, "
        "so the trajectory test is not actually checking the carry")
