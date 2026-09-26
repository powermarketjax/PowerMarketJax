"""L2 numerical equivalence against the numpy reference.

Same algorithm, two routes.  The implementation stacks vectorised blocks and
the reference writes the rows of §6 one at a time; the implementation
vectorises §8 and the reference loops over participants.  The Newton loop is
shared with the day-ahead reference and is covered by the day-ahead L2, for the
reason argued in `reference.py`.

**Every tolerance below is derived from a measured error and the derivation is
in the comment beside it.**  The repository has a recorded failure of exactly
the opposite habit: its first equivalence layer declared tolerances four to
five orders of magnitude looser than the errors they bounded, which makes a
comparison that cannot fail.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.local_flexibility import (build_voltage_sensitivity,
                                                   make_settlement)
from powermarketjax.envs.local_flexibility.clearing import make_clearing

from . import reference

#: The development case: 41 variables against 573 on the primary one, and the
#: reference assembles its rows in Python.  The primary case is exercised once,
#: at the end, because a sign that only (LIM) can reveal needs real ratings.
DEV, PRIMARY = "33bw", "533mt_hi"
KAPPA = 1.5

#: Derived from a measurement over the ten configurations exercised below,
#: taken on both backends because the repository records a GPU-to-CPU drift as
#: the basis of every comparison tolerance (2026-08-10):
#:
#:     worst relative error in `award`     CPU 1.5e-16   GPU 7.8e-16
#:     worst relative error in `z`         CPU 3.3e-16   GPU 1.5e-15
#:     worst absolute error in `shed`      CPU 1.7e-08   GPU 1.9e-08
#:
#: The two money quantities land within a few units in the last place of
#: float64 over sums of hundreds of terms, and the tolerances below are the GPU
#: figure times five.  A first draft of this module declared 1e-14 and 1e-13,
#: which is sixty and three hundred times the measured error; that is exactly
#: the failure the existing L2 layer was found to have, so it is written down
#: rather than quietly corrected.
#:
#: `shed` is seven orders of magnitude looser and that is degeneracy rather
#: than error: every unit of curtailment carries the same price, so two buses
#: relieving one constraint are interchangeable and the two solvers stop at
#: different points of one optimal face.  The objective is what pins that face,
#: and it agrees.
AWARD_RTOL = 4e-15
OBJECTIVE_RTOL = 8e-15
SHED_ATOL = 1e-7


@pytest.fixture(autouse=True)
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def scenario(case_id, n_agent, seed=0, margin=0.0, thermal=0.0, kappa=KAPPA):
    case = load_case(case_id)
    sens = build_voltage_sensitivity(case)
    rng = np.random.default_rng(seed)

    load = np.asarray(case.node_pd, np.float64) / case.base_mva
    reactive = np.asarray(case.node_qd, np.float64) / case.base_mva
    buses = np.flatnonzero(load > 0.0)
    buses = buses[buses != sens.slack]
    weights = load[buses] / load[buses].sum()
    agent_bus = np.sort(rng.choice(buses, n_agent, replace=False, p=weights))

    return dict(case=case, sens=sens, agent_bus=agent_bus,
                price=rng.uniform(20.0, 80.0, n_agent),
                qty_max=np.full(n_agent, 0.02),
                p_inj=-load * kappa, q_inj=-reactive * kappa,
                load=np.maximum(load * kappa, 0.0),
                voltage_margin=margin, thermal_margin=thermal)


def both(s):
    clear, _ = make_clearing(s["case"], s["sens"], s["agent_bus"],
                             voltage_margin=s["voltage_margin"],
                             thermal_margin=s["thermal_margin"])
    got = jax.jit(clear)(s["price"], s["qty_max"], s["p_inj"], s["q_inj"], s["load"])
    want = reference.clear(s["case"], s["sens"], s["agent_bus"], s["price"],
                           s["qty_max"], s["p_inj"], s["q_inj"], s["load"],
                           voltage_margin=s["voltage_margin"],
                           thermal_margin=s["thermal_margin"])
    return got, want


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_award_matches_the_reference(seed):
    got, want = both(scenario(DEV, 8, seed=seed))
    assert float(got["mu"]) < 1e-6 and want["mu"] < 1e-6
    np.testing.assert_allclose(np.asarray(got["award"]), want["award"],
                               rtol=AWARD_RTOL, atol=1e-15)


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_objective_matches_the_reference(seed):
    got, want = both(scenario(DEV, 8, seed=seed))
    np.testing.assert_allclose(float(got["z"]), want["z"], rtol=OBJECTIVE_RTOL)


def test_curtailment_matches_within_the_degeneracy_it_carries():
    """Looser than the award, and the comment on `SHED_ATOL` says why."""
    got, want = both(scenario(DEV, 4, seed=5, kappa=3.0))
    assert float(np.asarray(got["shed"]).sum()) > 0.0, "no curtailment to compare"
    np.testing.assert_allclose(np.asarray(got["shed"]), want["shed"],
                               atol=SHED_ATOL)


@pytest.mark.parametrize("margin,thermal", ((0.002, 0.0), (0.0, 0.02), (0.002, 0.02)))
def test_both_margins_reach_the_reference(margin, thermal):
    """The margins enter the right-hand sides, which is where a route can diverge."""
    got, want = both(scenario(DEV, 8, margin=margin, thermal=thermal))
    np.testing.assert_allclose(np.asarray(got["award"]), want["award"],
                               rtol=AWARD_RTOL, atol=1e-15)
    np.testing.assert_allclose(float(got["z"]), want["z"], rtol=OBJECTIVE_RTOL)


def test_the_primary_case_matches_where_the_ratings_bind():
    """The development case cannot exercise (LIM); this one can and must."""
    s = scenario(PRIMARY, 12, seed=1, kappa=2.0)
    got, want = both(s)

    flow = -s["sens"].A @ (s["p_inj"] + np.asarray(got["shed"]))
    assert (np.abs(flow) > s["sens"].p_max * 0.99).any(), "no rating near binding"

    np.testing.assert_allclose(np.asarray(got["award"]), want["award"],
                               rtol=AWARD_RTOL, atol=1e-14)
    np.testing.assert_allclose(float(got["z"]), want["z"], rtol=OBJECTIVE_RTOL)


def test_settlement_matches_the_reference():
    """§8 vectorised against §8 written out term by term."""
    s = scenario(DEV, 8)
    got, _ = both(s)
    sens = s["sens"]
    rng = np.random.default_rng(11)
    plan = rng.uniform(0.0, 0.03, len(s["agent_bus"]))
    cycle = rng.uniform(1.0, 5.0, len(s["agent_bus"]))

    mine = make_settlement(sens)(s["price"], got["award"], jnp.asarray(plan),
                                 40.0, jnp.asarray(cycle))
    theirs = reference.settle(sens, s["price"], np.asarray(got["award"]),
                              plan, 40.0, cycle)

    for key in ("revenue", "cost", "profit", "p_ch", "p_dis"):
        # exact: both evaluate the same products and differences on the same
        # float64 values, and neither reduces over an axis
        np.testing.assert_allclose(np.asarray(mine[key]), theirs[key],
                                   rtol=0.0, atol=0.0)
