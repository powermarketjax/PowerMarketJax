"""L1 domain correctness for the verification against the nonlinear power flow (§7).

The check §16 asks for by name is the hard one: a scenario in which the
linearised solution passes and the nonlinear one fails.  It does not have to be
contrived here, because the clearing produces one on its own.  The linear
program drives the squared voltage magnitude exactly onto the bound at every
binding bus, and §3.3 establishes that the linearised magnitude is a one-sided
overestimate, so the swept magnitude at those buses lands below the bound by
the linearisation error.  That is the whole reason §6 carries a safety margin.

The margin is exercised too, in both directions: it removes the violation and
it costs procurement.  §18 calls the second the more expensive error.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.local_flexibility import (build_voltage_sensitivity,
                                                   cleared_injection,
                                                   make_verification)
from powermarketjax.envs.local_flexibility.clearing import make_clearing

KAPPA = 1.5          # the adopted scaling factor
N_AGENT = 40
#: The margin that removes the voltage violation at the adopted configuration,
#: measured here rather than assumed.  §18 still records it as unfixed because
#: it is calibrated at one configuration only.
MARGIN = 0.002
#: The thermal margin that clears the residual overload the voltage margin
#: cannot reach, measured the same way and at the same configuration.  Relative
#: to each line rating rather than absolute, since the ratings span two orders
#: of magnitude.
THERMAL_MARGIN = 0.02


@pytest.fixture(autouse=True)
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture(scope="module")
def network():
    case = load_case("533mt_hi")
    sens = build_voltage_sensitivity(case)
    load = np.asarray(case.node_pd, np.float64) / case.base_mva
    reactive = np.asarray(case.node_qd, np.float64) / case.base_mva
    phi = np.where(load > 0.0, reactive / np.where(load > 0.0, load, 1.0), 0.0)
    buses = np.flatnonzero(load > 0.0)
    buses = buses[buses != sens.slack]
    return case, sens, load, reactive, phi, buses, load[buses] / load[buses].sum()


def cleared(network, margin, seed=0, kappa=KAPPA, thermal=0.0):
    """Clear one period and return the nonlinear check on the resulting point."""
    case, sens, load, reactive, phi, buses, weights = network
    rng = np.random.default_rng(seed)
    agent_bus = np.sort(rng.choice(buses, N_AGENT, replace=False, p=weights))

    clear, _ = make_clearing(case, sens, agent_bus, voltage_margin=margin,
                             thermal_margin=thermal)
    p_base, q_base = -load * kappa, -reactive * kappa
    out = jax.jit(clear)(rng.uniform(20.0, 80.0, N_AGENT),
                         np.full(N_AGENT, 0.02), p_base, q_base,
                         np.maximum(load * kappa, 0.0))

    p_inj, q_inj = cleared_injection(sens, agent_bus, phi, jnp.asarray(p_base),
                                     jnp.asarray(q_base), out["award"], out["shed"])
    checked = jax.jit(make_verification(case, sens))(p_inj, q_inj)
    return out, checked


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_the_linearisation_error_surfaces_as_a_nonlinear_violation(network, seed):
    """§16's constructed violation, which the clearing constructs by itself.

    With no margin the program clears onto the bound, and the sweep then places
    the real magnitude below it.  The violation must be one-sided, since §3.3
    fixes the direction: the linearised model understates the drop.
    """
    case, _, _, _, _, _, _ = network
    out, checked = cleared(network, margin=0.0, seed=seed)

    assert float(out["mu"]) < 1e-6
    assert bool(checked["converged"]) and not bool(checked["floor_active"])

    under = np.asarray(checked["v_under"])
    assert (under > 0.0).sum() > 0, "no violation, so this test proves nothing"
    assert 1e-4 < under.max() < 1e-2

    # one-sided: the sweep lands below the bound, never above it
    assert float(np.asarray(checked["v_over"]).max()) == 0.0


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_the_safety_margin_removes_it_and_costs_procurement(network, seed):
    """The only admissible response of §7, and the price §18 says it carries."""
    loose, unchecked = cleared(network, margin=0.0, seed=seed)
    tight, checked = cleared(network, margin=MARGIN, seed=seed)

    assert float(np.asarray(unchecked["v_under"]).max()) > 0.0
    assert float(np.asarray(checked["v_under"]).max()) == 0.0
    assert float(np.asarray(tight["award"]).sum()) > float(np.asarray(loose["award"]).sum())


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_the_voltage_margin_alone_leaves_the_ratings_overrun(network, seed):
    """Why §6 carries two margins rather than one.

    The voltage margin pulls the voltage bounds inward and nothing else, while
    the same dropped loss terms make the swept flow exceed the linearised one.
    A line the program clears onto its rating is therefore over it in the sweep,
    and no value of the voltage margin changes that: the two overload vectors
    below are equal.
    """
    _, loose = cleared(network, margin=0.0, seed=seed)
    _, tight = cleared(network, margin=MARGIN, seed=seed)

    np.testing.assert_allclose(np.asarray(loose["overload"]),
                               np.asarray(tight["overload"]), rtol=1e-5)
    assert float(np.asarray(tight["overload"]).max()) > 0.0


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_the_thermal_margin_clears_what_the_voltage_margin_cannot(network, seed):
    """The second margin, measured the same way as the first.

    Two per cent of each rating removes the residual overload entirely, and the
    two margins together leave the swept operating point inside both families
    of limit.  The price is small: procurement rises by a few per cent against
    the voltage margin alone.
    """
    both, checked = cleared(network, margin=MARGIN, seed=seed,
                            thermal=THERMAL_MARGIN)
    voltage_only, unchecked = cleared(network, margin=MARGIN, seed=seed)

    assert float(np.asarray(unchecked["overload"]).max()) > 0.0
    assert float(np.asarray(checked["overload"]).max()) == 0.0
    assert float(np.asarray(checked["v_under"]).max()) == 0.0
    assert float(np.asarray(checked["v_over"]).max()) == 0.0
    assert float(np.asarray(both["award"]).sum()) > \
        float(np.asarray(voltage_only["award"]).sum())


def test_a_thermal_margin_outside_the_unit_interval_is_rejected(network):
    """It is a fraction of a rating, not a power, and one is not the other."""
    case, sens, _, _, _, buses, _ = network
    for bad in (-0.1, 1.0, 3.0):
        with pytest.raises(ValueError, match="thermal_margin"):
            make_clearing(case, sens, buses[:3], thermal_margin=bad)


def test_a_collapsed_operating_point_is_not_reported_as_converged(network):
    """`converged` requires the voltage floor to be inactive (pitfalls §2).

    A point driven onto the floor has zero increment and would satisfy a bare
    tolerance test.  The flag must refuse it, and `floor_active` must say so.
    """
    case, sens, load, reactive, _, _, _ = network
    verify = jax.jit(make_verification(case, sens))
    checked = verify(jnp.asarray(-load * 60.0), jnp.asarray(-reactive * 60.0))

    assert bool(checked["floor_active"])
    assert not bool(checked["converged"])
    # and it is still reported as a violation rather than silently accepted
    assert float(np.asarray(checked["v_under"]).max()) > 0.0


def test_convergence_alone_never_certifies_the_point(network):
    """A converged sweep can be inadmissible, which is why no combined flag exists."""
    case, sens, load, reactive, _, _, _ = network
    verify = jax.jit(make_verification(case, sens))
    checked = verify(jnp.asarray(-load * 2.5), jnp.asarray(-reactive * 2.5))

    assert bool(checked["converged"]) and not bool(checked["floor_active"])
    assert float(np.asarray(checked["v_under"]).max()) > 0.0
    assert set(checked) & {"v_under", "v_over", "overload"}
    assert "admissible" not in checked


def test_an_unloaded_feeder_violates_nothing(network):
    """The lower end of the range, where the linearisation is exact enough to agree."""
    case, sens, load, reactive, _, _, _ = network
    verify = jax.jit(make_verification(case, sens))
    checked = verify(jnp.asarray(-load * 0.1), jnp.asarray(-reactive * 0.1))

    assert bool(checked["converged"])
    assert float(np.asarray(checked["v_under"]).max()) == 0.0
    assert float(np.asarray(checked["overload"]).max()) == 0.0


def test_cleared_injection_places_the_award_at_the_offering_bus(network):
    """A misplaced award would look exactly like linearisation error."""
    _, sens, load, reactive, phi, buses, _ = network
    agent_bus = np.array([buses[0], buses[1], buses[0]])       # two share a bus
    award = jnp.asarray([0.01, 0.02, 0.03])
    shed = jnp.zeros(sens.n_bus)

    p_inj, q_inj = cleared_injection(sens, agent_bus, phi, jnp.asarray(-load),
                                     jnp.asarray(-reactive), award, shed)

    expected = -load.copy()
    expected[buses[0]] += 0.04                                  # scatter-add
    expected[buses[1]] += 0.02
    np.testing.assert_allclose(np.asarray(p_inj), expected, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(q_inj), -reactive, rtol=1e-12)


def test_curtailment_removes_reactive_load_at_the_registered_ratio(network):
    """§3.1: directed curtailment removes active and reactive demand together."""
    _, sens, load, reactive, phi, buses, _ = network
    shed = np.zeros(sens.n_bus)
    shed[buses[3]] = 0.05

    p_inj, q_inj = cleared_injection(sens, np.array([buses[0]]), phi,
                                     jnp.asarray(-load), jnp.asarray(-reactive),
                                     jnp.zeros(1), jnp.asarray(shed))

    np.testing.assert_allclose(np.asarray(p_inj)[buses[3]],
                               -load[buses[3]] + 0.05, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(q_inj)[buses[3]],
                               -reactive[buses[3]] + phi[buses[3]] * 0.05, rtol=1e-12)
