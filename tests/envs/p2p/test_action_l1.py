"""L1 for the P2P action map: the submission rules of §3.1, §5 and §9.3.

The three groups §17 puts under "on submissions", plus the two properties that
decide whether the `costs` channel means anything.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.envs.p2p import make_action_map, make_soc_advance
from powermarketjax.resources.battery import (
    compute_feasible_power_batch, make_battery_bundle)

PI_EXP, PI_RET, DELTA = 4.1, 26.11, 0.5
N = 16


@pytest.fixture
def battery():
    return make_battery_bundle(n_devices=N, power_mw=2.0, capacity_mwh=4.0,
                               soc_min=0.1, soc_max=0.9)


@pytest.fixture
def act():
    fn, _ = make_action_map(N, PI_EXP, PI_RET, DELTA)
    return jax.jit(fn)


#: The state of charge is a pure input this round -- the advance belongs to RTM
#: item 22 -- so the tests supply it, and they supply it inside the bounds the
#: bundle declares.  Outside them the envelope of §3.2 would be negative and the
#: vendored ``compute_feasible_power_batch`` floors it at zero, which is correct
#: but is a robustness path rather than the domain the market operates on.
SOC_MIN, SOC_MAX = 0.1, 0.9


def _sample(seed, n=N):
    rng = np.random.default_rng(seed)
    return (jnp.asarray(rng.uniform(-1.5, 1.5, (n, 2)).astype(np.float32)),
            jnp.asarray(rng.uniform(SOC_MIN, SOC_MAX, n).astype(np.float32)),
            jnp.asarray(rng.uniform(0.0, 4.0, n).astype(np.float32)),
            jnp.asarray(rng.uniform(0.0, 4.0, n).astype(np.float32)))


def test_at_most_one_side_carries_a_positive_quantity(act, battery):
    for seed in range(40):
        out = act(*_sample(seed), battery)
        q_sell, q_buy = np.asarray(out["q_sell"]), np.asarray(out["q_buy"])
        assert (q_sell >= 0).all() and (q_buy >= 0).all()
        assert not ((q_sell > 0) & (q_buy > 0)).any()


def test_quantities_are_the_net_position_scaled_by_delta(act, battery):
    for seed in range(40):
        out = act(*_sample(seed), battery)
        net = np.asarray(out["net_position"])
        np.testing.assert_allclose(np.asarray(out["q_sell"]),
                                   DELTA * np.maximum(net, 0.0), rtol=1e-6)
        np.testing.assert_allclose(np.asarray(out["q_buy"]),
                                   DELTA * np.maximum(-net, 0.0), rtol=1e-6)


def test_price_lies_in_the_bracket_for_every_real_action(act, battery):
    """§5 holds by construction, including for actions outside the action space.

    This is what the clip at the entry of the map buys.  Without it an
    unbounded Gaussian policy would submit above `pi_ret`, the §7 sentinel
    substitutions would stop being exact, and `lambda` would leave the bracket.
    """
    _, soc, p_pv, load = _sample(0)
    for magnitude in (1.0, 1.5, 10.0, 1e4):
        for sign in (-1.0, 1.0):
            action = jnp.full((N, 2), jnp.float32(sign * magnitude))
            price = np.asarray(act(action, soc, p_pv, load, battery)["price"])
            assert (price >= np.float32(PI_EXP)).all()
            assert (price <= np.float32(PI_RET)).all()


def test_the_endpoints_are_exact(act, battery):
    """`alpha = -1` and `+1` must give `pi_exp` and `pi_ret` bit for bit.

    Those two values are the truthful submissions of §4 and the whole basis of
    the tied baseline of §17, so a rounding error of one ULP here would leave a
    learner unable to reach the reference strategy of this market and would
    break the constructed case that checks it.
    """
    _, soc, p_pv, load = _sample(1)
    lo = act(jnp.full((N, 2), jnp.float32(-1.0)), soc, p_pv, load, battery)
    hi = act(jnp.full((N, 2), jnp.float32(1.0)), soc, p_pv, load, battery)
    assert (np.asarray(lo["price"]) == np.float32(PI_EXP)).all()
    assert (np.asarray(hi["price"]) == np.float32(PI_RET)).all()


def test_price_is_monotone_in_the_price_command(act, battery):
    _, soc, p_pv, load = _sample(2)
    grid = np.linspace(-1.0, 1.0, 21, dtype=np.float32)
    prices = [float(act(jnp.full((N, 2), jnp.float32(a)), soc, p_pv, load,
                        battery)["price"][0]) for a in grid]
    assert all(b >= a for a, b in zip(prices, prices[1:]))


def test_the_battery_command_stays_inside_the_envelope(act, battery):
    """§3.2 checked against the envelope itself, not against the advanced soc.

    `update_soc_batch` ends with a clip onto the state-of-charge bounds
    (`battery.py:804`), which is redundant when the envelope is right and
    silently absorbs the excursion when it is not, so §14 requires this
    criterion to be checked here rather than after the advance.
    """
    for seed in range(40):
        action, soc, p_pv, load = _sample(seed)
        out = act(action, soc, p_pv, load, battery)
        p = np.asarray(out["p_signed"])
        rated = np.asarray(battery.power_max)
        cap = np.asarray(battery.capacity)
        s = np.asarray(soc)
        # the analytic envelope of §3.2, recomputed here independently.  The
        # floor at zero is part of the bound and not a repair: a state of
        # charge already at its lower limit admits no discharge at all.
        max_dis = np.clip((s - np.asarray(battery.soc_min)) * cap
                          * np.asarray(battery.eta_discharge) / DELTA,
                          0.0, rated)
        max_ch = np.clip((np.asarray(battery.soc_max) - s) * cap
                         / (np.asarray(battery.eta_charge) * DELTA),
                         0.0, rated)
        assert (p <= max_dis + 1e-5).all()
        assert (p >= -max_ch - 1e-5).all()
        # the two components split a single signed command
        np.testing.assert_allclose(np.asarray(out["p_dis"]),
                                   np.maximum(p, 0.0), atol=1e-6)
        np.testing.assert_allclose(np.asarray(out["p_ch"]),
                                   np.maximum(-p, 0.0), atol=1e-6)
        assert not ((np.asarray(out["p_dis"]) > 0)
                    & (np.asarray(out["p_ch"]) > 0)).any()


def test_clip_is_zero_exactly_when_the_command_was_deliverable(act, battery):
    """§9.5: the `costs` channel reports physical infeasibility and nothing else."""
    for seed in range(40):
        action, soc, p_pv, load = _sample(seed)
        out = act(action, soc, p_pv, load, battery)
        clipped = np.clip(np.asarray(action)[:, 0], -1.0, 1.0)
        desired = clipped * np.asarray(battery.power_max)
        delivered = np.asarray(out["p_signed"])
        expected = np.abs(desired - delivered) / np.maximum(
            np.asarray(battery.power_max), 1e-6)
        np.testing.assert_allclose(np.asarray(out["clip"]), expected, atol=1e-6)
        assert (np.asarray(out["clip"]) >= 0).all()


def test_clip_does_not_report_the_range_of_the_policy_output(act, battery):
    """An out-of-range action on a battery with headroom must report no clip.

    This is the property that made the entry clip the right choice over letting
    the raw action through: with the raw action, `alpha_bat = 2` would report a
    clip of 1 on a battery that delivered exactly what its rating allows, and a
    CMDP algorithm would treat a policy-scaling artefact as a constraint
    violation.
    """
    soc = jnp.full((N,), 0.5, jnp.float32)          # ample headroom both ways
    zeros = jnp.zeros((N,), jnp.float32)
    for magnitude in (1.0, 2.0, 50.0):
        for sign in (-1.0, 1.0):
            action = jnp.full((N, 2), jnp.float32(sign * magnitude))
            out = act(action, soc, zeros, zeros, battery)
            assert float(np.abs(np.asarray(out["clip"])).max()) == 0.0
            # and the command saturates at the rating rather than exceeding it
            assert float(np.abs(np.asarray(out["p_signed"])).max()) == \
                pytest.approx(float(np.asarray(battery.power_max).max()),
                              rel=1e-6)


def test_clip_is_positive_when_the_state_of_charge_binds(act, battery):
    empty = jnp.full((N,), jnp.float32(0.1))        # at soc_min: cannot discharge
    full = jnp.full((N,), jnp.float32(0.9))         # at soc_max: cannot charge
    zeros = jnp.zeros((N,), jnp.float32)
    discharge = jnp.concatenate(
        [jnp.ones((N, 1), jnp.float32), jnp.zeros((N, 1), jnp.float32)], axis=1)
    charge = discharge.at[:, 0].set(-1.0)
    assert float(act(discharge, empty, zeros, zeros, battery)["clip"].min()) > 0.5
    assert float(act(charge, full, zeros, zeros, battery)["clip"].min()) > 0.5


def test_throughput_is_energy(act, battery):
    for seed in range(20):
        action, soc, p_pv, load = _sample(seed)
        out = act(action, soc, p_pv, load, battery)
        np.testing.assert_allclose(
            np.asarray(out["throughput"]),
            DELTA * (np.asarray(out["p_ch"]) + np.asarray(out["p_dis"])),
            rtol=1e-6, atol=1e-7)
        assert (np.asarray(out["throughput"]) >= 0).all()


#: Derived from measurement, 2026-08-10, CPU: over 4320 periods x 16 agents the
#: unclipped (SOC) left its bounds by at most 2.24e-8 and differed from
#: ``update_soc_batch`` by the same 2.24e-8, which is under half a float32
#: epsilon at a state of charge near 0.5 (5.96e-8).  The tolerance is about
#: 4.5 times that.
SOC_TOL = 1e-7


def _rollout(act, advance, battery, seed, horizon):
    """One episode under `lax.scan`, returning the advanced and analytic soc.

    The analytic value is (SOC) of §3.2 applied to ``p_signed`` **without any
    clip**, which is the quantity the bounds must be asserted on: the clip at
    the end of ``update_soc_batch`` would otherwise make the assertion vacuous.
    """
    rng = np.random.default_rng(seed)
    actions = jnp.asarray(rng.uniform(-1.5, 1.5, (horizon, N, 2)).astype(np.float32))
    pv = jnp.asarray(rng.uniform(0.0, 4.0, (horizon, N)).astype(np.float32))
    ld = jnp.asarray(rng.uniform(0.0, 4.0, (horizon, N)).astype(np.float32))
    soc0 = jnp.asarray(rng.uniform(0.15, 0.85, N).astype(np.float32))

    def body(soc, xs):
        action, p_pv, load = xs
        out = act(action, soc, p_pv, load, battery)
        p = out["p_signed"]
        analytic = soc + jnp.where(
            p >= 0.0,
            -p * DELTA / (battery.eta_discharge * battery.capacity),
            -p * DELTA * battery.eta_charge / battery.capacity)
        return advance(soc, p, battery), (analytic, out["clip"])

    return jax.jit(lambda xs: jax.lax.scan(body, soc0, xs))((actions, pv, ld))


def test_soc_stays_inside_its_bounds_over_an_episode(battery):
    """§3.2 across periods, asserted on the **unclipped** (SOC).

    This is the one criterion the round that built the action map could not
    check, because `soc` was a pure input there.  It is asserted on the
    analytic advance rather than on what `advance` returns, for the reason §14
    gives: `update_soc_batch` ends with a clip onto the bounds, so asserting
    the bounds on its output passes no matter what the envelope does.
    """
    act_fn, _ = make_action_map(N, PI_EXP, PI_RET, DELTA)
    advance = make_soc_advance(DELTA)
    for seed in range(4):
        _, (analytic, _) = _rollout(act_fn, advance, battery, seed, horizon=480)
        a = np.asarray(analytic)
        assert np.isfinite(a).all()
        assert a.min() >= float(np.asarray(battery.soc_min).min()) - SOC_TOL
        assert a.max() <= float(np.asarray(battery.soc_max).max()) + SOC_TOL


def test_the_trailing_clip_never_does_any_work(battery):
    """The clip in `update_soc_batch` absorbs rounding and nothing else.

    This is what makes the test above meaningful.  If the envelope of §3.2 were
    wrong, the analytic advance would leave the bounds and the clip would pull
    it back, so the two would disagree by a finite amount; they agree to within
    the measured rounding, so the clip is redundant here and the previous
    assertion is testing the envelope rather than the clip.

    Stepped in Python rather than under `lax.scan` because the comparison needs
    the state of charge at the start of each period as well as at its end.
    """
    act_fn, _ = make_action_map(N, PI_EXP, PI_RET, DELTA)
    advance = make_soc_advance(DELTA)
    cap = np.asarray(battery.capacity)
    eta_ch = np.asarray(battery.eta_charge)
    eta_dis = np.asarray(battery.eta_discharge)
    worst = 0.0
    for seed in range(3):
        rng = np.random.default_rng(seed)
        soc = jnp.asarray(rng.uniform(0.15, 0.85, N).astype(np.float32))
        for _ in range(240):
            out = act_fn(jnp.asarray(rng.uniform(-1.5, 1.5, (N, 2)).astype(np.float32)),
                         soc,
                         jnp.asarray(rng.uniform(0.0, 4.0, N).astype(np.float32)),
                         jnp.asarray(rng.uniform(0.0, 4.0, N).astype(np.float32)),
                         battery)
            p = np.asarray(out["p_signed"])
            analytic = np.asarray(soc) + np.where(
                p >= 0.0, -p * DELTA / (eta_dis * cap), -p * DELTA * eta_ch / cap)
            soc = advance(soc, out["p_signed"], battery)
            worst = max(worst, float(np.abs(np.asarray(soc) - analytic).max()))
    assert worst < SOC_TOL, worst


def test_soc_advance_rejects_a_bad_period():
    with pytest.raises(ValueError):
        make_soc_advance(0.0)
    with pytest.raises(ValueError):
        make_soc_advance(-1.0)


def test_a_saturating_command_walks_the_soc_to_its_bound_and_holds(battery):
    """Commanding full discharge forever must settle exactly on `soc_min`.

    The envelope shrinks as the state of charge falls, so the command becomes
    undeliverable and `clip` turns positive; what must not happen is the state
    of charge crossing the bound and being pulled back by the trailing clip.
    """
    act_fn, _ = make_action_map(N, PI_EXP, PI_RET, DELTA)
    advance = make_soc_advance(DELTA)
    zeros = jnp.zeros((N,), jnp.float32)
    soc = jnp.full((N,), 0.8, jnp.float32)
    action = jnp.concatenate(
        [jnp.ones((N, 1), jnp.float32), jnp.zeros((N, 1), jnp.float32)], axis=1)
    for _ in range(200):
        out = act_fn(action, soc, zeros, zeros, battery)
        p = np.asarray(out["p_signed"])
        ana = np.asarray(soc) + (-p * DELTA
                                 / (np.asarray(battery.eta_discharge)
                                    * np.asarray(battery.capacity)))
        assert ana.min() >= float(np.asarray(battery.soc_min).min()) - SOC_TOL
        soc = advance(soc, out["p_signed"], battery)
    np.testing.assert_allclose(np.asarray(soc),
                               np.asarray(battery.soc_min), atol=SOC_TOL)
    # at the floor the command is no longer deliverable, so the channel reports it
    out = act_fn(action, soc, zeros, zeros, battery)
    assert float(np.asarray(out["clip"]).min()) > 0.9
