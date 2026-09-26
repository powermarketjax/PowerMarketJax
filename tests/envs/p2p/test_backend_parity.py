"""Cross-backend agreement for the P2P market.

This closes an item the acceptance record carried as "measured on GPU but not
asserted".  It runs one episode of the whole forward chain twice in the same
process, once on each addressable platform, and compares.

**Bitwise agreement across backends is not achievable here and asserting it
would be wrong.**  Measured on 2026-08-14 over 96 steps at 24 agents, every
output differs in the last bits, *including the net position of §3.1* -- which is
elementwise arithmetic and might be expected to be exact -- by up to 7.5e-09 MW.
The cause is not the auction: an accelerator is free to contract a multiply and
an add into one fused instruction and to divide by a different algorithm, so
float32 elementwise arithmetic is not reproducible across platforms in the first
place.  A test written to assert bit-identity would therefore be asserting
something the hardware does not offer, and would either fail or be quietly
weakened later.

**What is asserted instead is the discrete outcome, and that is the criterion
that matters.**  float32 was rejected for the local flexibility market on
exactly this ground: the same input gave *different award sets* on different
processes of the same card, and neither the duality measure nor feasibility
could gate it.  Here the award set is invariant -- which participant clears and
which does not agrees in every cell -- and only the last bits of the quantities
move.  That is the evidence which makes float32 safe for this market, and it is
stronger than what was on record before, because the previous support was that
the forward chain is bit-identical with `jax_enable_x64` on and off, and that is
a same-backend property.

The continuous tolerances below are graded from the measurement rather than set
to one blanket number, in the same style as the L2 tolerances of
`test_clearing_l2.py`, and each carries the value it was derived from.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.envs.p2p import make_p2p_env, make_p2p_params
from powermarketjax.resources.battery import make_battery_bundle

N_AGENTS = 24
N_PERIODS = 400
EPISODE_LEN = 96
PI_EXP, PI_RET = 73.0, 333.4
DELTA = 0.25
KAPPA = 13.88

#: Absolute tolerances, each from the maximum measured on 2026-08-14 over one
#: 96-step episode, with roughly an order of magnitude of headroom.  The awards
#: and the traded volume are energies in MWh at household scale, the state of
#: charge is a fraction, the price and the observation are in EUR/MWh, and the
#: reward and the constraint channel are money and a clipped fraction.
MEASURED = {
    "reward": 5.886e-07,
    "costs": 1.401e-06,
    "clearing_price": 3.052e-05,
    "traded_volume": 1.630e-09,
    "soc": 2.384e-07,
    "award": 1.630e-09,
    "net": 7.451e-09,
    "obs": 3.052e-05,
}
TOLERANCE = {
    "reward": 1e-5,
    "costs": 1e-5,
    "clearing_price": 1e-3,
    "traded_volume": 1e-8,
    "soc": 1e-6,
    "award": 1e-8,
    "net": 1e-7,
    "obs": 1e-3,
}
#: Relative agreement on the awards that actually clear, measured at 8.77e-06.
AWARD_RTOL = 1e-4


def _platforms():
    available = []
    for name in ("cpu", "gpu"):
        try:
            if jax.devices(name):
                available.append(name)
        except RuntimeError:
            pass
    return available


def _rollout_on(platform, key, params, actions, reset, step_auto):
    device = jax.devices(platform)[0]

    def rollout(key, params, actions):
        _, state = reset(key, params)

        def body(state, action):
            obs, state, reward, costs, _, info = step_auto(
                key, state, action, params)
            return state, (reward, costs, info["clearing_price"],
                           info["traded_volume"], state.soc, state.award_prev,
                           state.net_prev, obs)

        _, out = jax.lax.scan(body, state, actions)
        return out

    placed = [jax.device_put(x, device) for x in (key, params, actions)]
    return [np.asarray(x) for x in jax.jit(rollout)(*placed)]


@pytest.fixture(scope="module")
def both_backends():
    platforms = _platforms()
    if len(platforms) < 2:
        pytest.skip(f"needs two addressable platforms, found {platforms}")

    rng = np.random.default_rng(0)
    one_way = math.sqrt(0.85)
    battery = make_battery_bundle(
        n_devices=N_AGENTS, capacity_mwh=0.011, power_mw=0.011 / 2.1,
        eta_charge=one_way, eta_discharge=one_way, soc_min=0.15, soc_max=1.0,
        initial_soc=0.5, dt_hours=DELTA, cycle_cost_per_mwh=0.0)
    # A population where roughly half the premises inject in any period, so both
    # sides of the auction are populated and the marginal position moves about.
    injection = (rng.integers(0, 2, (N_PERIODS, N_AGENTS))
                 * rng.uniform(0, 5e-3, (N_PERIODS, N_AGENTS)))
    params = make_p2p_params(
        p_pv=injection, load=rng.uniform(0, 5e-3, (N_PERIODS, N_AGENTS)),
        battery=battery, kappa=np.full(N_AGENTS, KAPPA, np.float32),
        learner_mask=np.ones(N_AGENTS, bool), episode_len=EPISODE_LEN)
    reset, _, step_auto, _ = make_p2p_env(N_AGENTS, PI_EXP, PI_RET, DELTA)
    actions = jnp.asarray(
        rng.uniform(-1, 1, (EPISODE_LEN, N_AGENTS, 2)), jnp.float32)
    key = jax.random.PRNGKey(0)

    runs = {p: _rollout_on(p, key, params, actions, reset, step_auto)
            for p in platforms}
    names = ("reward", "costs", "clearing_price", "traded_volume", "soc",
             "award", "net", "obs")
    first, second = platforms[0], platforms[1]
    return (dict(zip(names, runs[first])), dict(zip(names, runs[second])),
            first, second)


def test_the_award_set_is_invariant_across_backends(both_backends):
    """The load-bearing assertion: the same participants clear on both.

    float32 was rejected for the local flexibility market because the award
    set moved between processes; if it moved between backends here, a policy
    trained on one would face a different mechanism on the other, and nothing in
    the money-balance identities would show it.
    """
    left, right, first, second = both_backends
    clears_left = left["award"] > 0.0
    clears_right = right["award"] > 0.0
    differing = int((clears_left != clears_right).sum())
    assert differing == 0, (
        f"{differing} of {clears_left.size} (period, agent) cells disagree on "
        f"whether the participant cleared, between {first} and {second}")

    # And on the cells that clear, the quantity agrees relatively, not merely in
    # absolute terms -- a household award is thousandths of a megawatt hour.
    clearing = clears_left | clears_right
    if clearing.any():
        rel = (np.abs(left["award"][clearing] - right["award"][clearing])
               / np.maximum(left["award"][clearing], 1e-12))
        assert rel.max() < AWARD_RTOL


def test_every_output_agrees_to_its_measured_tolerance(both_backends):
    left, right, first, second = both_backends
    for name, tol in TOLERANCE.items():
        worst = float(np.abs(left[name].astype(np.float64)
                             - right[name].astype(np.float64)).max())
        assert worst < tol, (
            f"{name}: {first} against {second} differs by {worst:.3e}, above "
            f"the tolerance {tol:.0e} derived from {MEASURED[name]:.3e}")


def test_bit_identity_is_not_claimed(both_backends):
    """Guards the docstring rather than the code.

    If a future change did make every output bit-identical across backends, the
    tolerances above would be silently pointless, and the module docstring would
    be wrong about why they exist. This records which outputs actually differ so
    that the reason for grading them is visible in a failure rather than only in
    prose.
    """
    left, right, _, _ = both_backends
    differing = [name for name in TOLERANCE
                 if not np.array_equal(left[name], right[name])]
    assert differing, (
        "every output is now bit-identical across backends; the tolerances in "
        "this module and the reasoning in its docstring need revisiting")


def test_the_tied_baseline_is_bitwise_across_backends():
    """The case where the tie rule decides everything, which the fixture above
    does not reach.

    The fixture draws actions uniformly, so submitted prices are distinct and a
    tie at the margin is a measure-zero event.  §17 records that the opposite
    holds at the truthful baseline: every seller submits the export price and
    every buyer the retail tariff, so both sides are entirely tied and §6.4's
    index rule decides the whole award.  That is where a one-bit difference in a
    price could move which participant clears, and it is therefore the case worth
    checking rather than the generic one.

    It is exactly bitwise, and by design rather than by luck.  §9.3 writes the
    price map as an interpolation rather than as an offset plus a scaled span
    precisely so that the endpoints come back as the two tariffs exactly in
    float32; the tie is then broken by `jnp.lexsort` on an integer index, which
    is exact. Measured 2026-08-14: award and price both agree to the last bit,
    against 1.6e-09 and 3.1e-05 in the generic case.
    """
    platforms = _platforms()
    if len(platforms) < 2:
        pytest.skip(f"needs two addressable platforms, found {platforms}")

    rng = np.random.default_rng(0)
    one_way = math.sqrt(0.85)
    battery = make_battery_bundle(
        n_devices=N_AGENTS, capacity_mwh=0.011, power_mw=0.011 / 2.1,
        eta_charge=one_way, eta_discharge=one_way, soc_min=0.15, soc_max=1.0,
        initial_soc=0.5, dt_hours=DELTA, cycle_cost_per_mwh=0.0)
    injection = (rng.integers(0, 2, (N_PERIODS, N_AGENTS))
                 * rng.uniform(0, 5e-3, (N_PERIODS, N_AGENTS)))
    # an all-False mask makes every submission the truthful one (§13)
    params = make_p2p_params(
        p_pv=injection, load=rng.uniform(0, 5e-3, (N_PERIODS, N_AGENTS)),
        battery=battery, kappa=np.full(N_AGENTS, KAPPA, np.float32),
        learner_mask=np.zeros(N_AGENTS, bool), episode_len=EPISODE_LEN)
    reset, _, step_auto, _ = make_p2p_env(N_AGENTS, PI_EXP, PI_RET, DELTA)
    actions = jnp.zeros((EPISODE_LEN, N_AGENTS, 2), jnp.float32)
    key = jax.random.PRNGKey(0)

    runs = {}
    for platform in platforms:
        runs[platform] = _rollout_on(platform, key, params, actions,
                                     reset, step_auto)
    names = ("reward", "costs", "clearing_price", "traded_volume", "soc",
             "award", "net", "obs")
    left = dict(zip(names, runs[platforms[0]]))
    right = dict(zip(names, runs[platforms[1]]))
    for name in ("clearing_price", "award", "net", "traded_volume"):
        np.testing.assert_array_equal(
            left[name], right[name],
            err_msg=f"{name} is not bitwise identical at the tied baseline")
