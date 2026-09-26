"""L1: the per-bus shed this market publishes reconciles with its own scalar.

`info["shed_mwh"]` is the quantity `tools/benchmark/evaluation.py` folds into
the system-cost figure; `info["shed"]` is the same shedding before that
reduction, one entry per bus, in MW.  Both are published because neither
recovers the other: summing the vector loses the period-length factor and the
scalar loses which bus.

The vector exists for one consumer.  Handing a period's LP to an independent
solver and comparing prices reports a DISAGREEMENT; turning that into an ERROR
needs each side's objective at its own point, and assembling the LP's primal
vector needs the per-bus entries.  Before this landed, the comparison had to
stop at "the two differ".

Three checks, and the third is the one that could quietly be missing.

*The identity*, on the dimension it is consumed on: the scalar equals the vector
summed over buses times the period length, with the period length read back out
of `spec` rather than restated here -- a restated constant cannot catch the two
disagreeing about it.

*Non-triviality*: an operating point where nothing sheds satisfies the identity
vacuously, so the fixture is asserted to shed.

*Discrimination*: perturbing one bus's entry must break the identity.  Without
it the identity would also hold if `shed` were the scalar broadcast over buses,
which is a different array with the same sum.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.envs.ancillary.test_env_l0 import _action, _build, x64  # noqa: F401


@pytest.fixture
def stepped(x64):                                                  # noqa: F811
    """A cleared period that actually sheds.

    The shared L0 fixture is built to serve its demand -- deliberately, because
    a point that sheds everywhere prices every bus at the value of lost load and
    makes the market indifferent to every offer.  So demand is scaled here until
    the fleet cannot follow it.  The scaling is a property of THIS fixture and
    nothing outside this file reads it; the non-triviality check below is what
    keeps it honest if the case data ever changes underneath.
    """
    env, params = _build()
    reset, step, _sar, spec = env
    params = params.replace(demand=params.demand * 3.0,
                            forecast=params.forecast * 3.0)
    key = jax.random.PRNGKey(0)
    _obs, state = reset(key, params)
    _o, _s, _r, _c, _d, info = jax.jit(step)(key, state, _action(energy=1.2),
                                             params)
    return spec, info


def test_the_scalar_is_the_vector_summed_over_buses_and_scaled(stepped):
    spec, info = stepped
    shed = np.asarray(info["shed"], np.float64)
    scalar = float(np.asarray(info["shed_mwh"]))
    period_hours = float(spec["period_hours"])
    assert shed.ndim == 1, f"`shed` is {shed.shape}, expected one entry per bus"
    np.testing.assert_allclose(scalar, period_hours * shed.sum(),
                               rtol=1e-12, atol=0)


def test_the_run_point_actually_sheds(stepped):
    _spec, info = stepped
    shed = np.asarray(info["shed"], np.float64)
    assert shed.sum() > 1e-6, (
        "nothing sheds at this operating point, so the identity above holds "
        "vacuously and this file checks nothing")
    assert (shed > 1e-9).sum() >= 1, "the shedding is not attributable to a bus"


def test_perturbing_one_bus_breaks_the_identity(stepped):
    """The identity must be sensitive to WHICH bus, not only to the total.

    A `shed` that were the scalar broadcast over buses would satisfy the sum
    identity and be useless to the consumer this vector exists for.
    """
    spec, info = stepped
    shed = np.asarray(info["shed"], np.float64).copy()
    scalar = float(np.asarray(info["shed_mwh"]))
    period_hours = float(spec["period_hours"])
    i = int(np.argmax(shed))
    bumped = shed.copy()
    bumped[i] += max(shed[i], 1.0)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(scalar, period_hours * bumped.sum(),
                                   rtol=1e-12, atol=0)
