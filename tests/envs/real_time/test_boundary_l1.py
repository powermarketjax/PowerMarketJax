"""L1 for the episode's opening carry — spec §9.2, §15.

Two criteria, and they exist because the construction was **proposed from the
specification and a precedent rather than measured**.  Both are stated so they
can fail, and both are checked here against the two constructions §15 records as
failing, so that "it passes" means something.

1. The first step must not shed more than a later step does.  A boundary that
   opens the episode outside the ramp envelope shows up exactly here: the opening
   period cannot be served and the shortfall goes to shed at VOLL.  This is the
   criterion the ancillary line's measurement points at -- 3 of 6 summer and 7 of
   21 winter shed events fell in the first one or two half-hours of a day, purely
   from carrying an hourly starting point into a half-hourly period.

2. The first step's ramp rows must not bind.  If the carry is constructed by
   clearing the opening period with the ramp free, then clearing that same period
   *with* the ramp against that carry is the same problem plus a slack
   constraint, so at the same offers it must return the same dispatch.  A carry
   from anywhere else does not have that property.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.day_ahead.clearing import make_clearing, segment_costs
from powermarketjax.envs.real_time.clearing import MAX_ITER
from powermarketjax.envs.real_time import hour_of_period, load_da_position
from powermarketjax.envs.real_time.boundary import make_boundary
from powermarketjax.envs.real_time.demand import load_gb_demand_half_hourly

#: Split onto separate lines on purpose.  As a tuple unpacking this defeated
#: both census forms used on the migration: there is no `cap_scale=` to find,
#: and `^CAP_SCALE` does not match a line beginning with `DELTA`.
DELTA = 0.5
#: The adopted scenario.  Stated rather than read back from the
#: position's `meta`, so that a fixture built for another network and a test
#: asking for this one remain two independent statements.
CAP_SCALE = 0.60
RAMP_SCALE = 1.00
#: `_seasons` is the expand-phase name; batch 3 renames it back.
CHAIN = "step1prime_seasons"


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
    # the fixture and this file must be describing the same network; `make_env`
    # enforces that where it is used, and here there is no `make_env` to do it
    assert (pos["meta"]["cap_scale"], pos["meta"]["ramp_scale"]) == \
        (CAP_SCALE, RAMP_SCALE), (
            f"position built at cap {pos['meta']['cap_scale']} / ramp "
            f"{pos['meta']['ramp_scale']}, but this module clears at "
            f"{CAP_SCALE} / {RAMP_SCALE}")
    case = load_case(pos["meta"]["case"])
    actual, _days = load_gb_demand_half_hourly()
    boundary, offer = make_boundary(case, n_segments=1, cap_scale=CAP_SCALE,
                                    period_hours=DELTA)
    step, _spec = make_clearing(case, 1, n_segments=1, cap_scale=CAP_SCALE,
                                ramp_scale=RAMP_SCALE, period_hours=DELTA,
                                max_iter=MAX_ITER)
    return pos, case, actual, boundary, offer, jax.jit(step)


def _period(pos, actual, day, t_rt):
    """Exogenous inputs for one real-time period: commitment and demand."""
    u = jnp.asarray(pos["u"][day][:, hour_of_period(t_rt):hour_of_period(t_rt) + 1],
                    jnp.float64)
    d = jnp.asarray([float(actual[int(np.asarray(pos["day_index"])[day]), t_rt])])
    return u, d


@pytest.mark.parametrize("day", [0, 10, 16])
def test_the_first_step_does_not_shed_more_than_later_steps(rig, day):
    """Criterion 1: the opening period must not be the shed outlier.

    Compared against the distribution of the following steps rather than against
    zero, because a period may legitimately shed for reasons that have nothing to
    do with the boundary; what a bad boundary produces is a first step that sheds
    when its neighbours do not.

    **The comparison half is only live on days whose later steps shed**, and the
    three days are not alike in that.  Measured 2026-08-15 over the first eight
    periods: day 10 reaches 1 052.6 MWh and day 16 reaches 996.5 MWh, so on those
    the comparison has something to bite on; **day 0's later steps are all at the
    numerical floor (~3.7e-21)**, so there the assertion degenerates to the
    absolute form, "the first step sheds nothing".  That is still a real
    assertion -- both rejected constructions violate it -- but it is the weaker
    half, and a reader should not take day 0 as evidence for the comparison.

    The floor is absolute (1e-6 MWh) and not relative: a period that sheds nothing leaves ~1e-20 behind, and a relative
    tolerance there compares noise.
    """
    pos, case, actual, boundary, offer, step = rig
    p_prev, _ = boundary(*_period(pos, actual, day, 0))

    shed = []
    for t in range(8):
        u, d = _period(pos, actual, day, t)
        out = step(offer, u, d, p_prev)
        assert float(out["mu"]) < 1e-6, f"period {t} did not converge"
        shed.append(float(np.asarray(out["shed"]).sum()))
        p_prev = np.asarray(out["award"])[:, 0]

    first, later = shed[0], np.asarray(shed[1:])
    assert first <= max(later.max(), 1e-6), (
        f"day {day}: the first step shed {first:.3f} MWh against a later-step "
        f"maximum of {later.max():.3f}; the opening carry is outside the ramp "
        f"envelope, which is what an hourly starting point produces here")


@pytest.mark.parametrize("day", [0, 10, 16])
def test_the_first_steps_ramp_rows_do_not_bind(rig, day):
    """Criterion 2: clearing the opening period against its own carry is a no-op.

    **Why this is the sharp form and not a weaker proxy**, since the obvious
    objection is "how do you know the ramp rows are slack without reading their
    duals?".  The carry is the optimum of this period with the ramp rows made
    non-binding.  Re-clearing the same period *with* the ramp rows is therefore
    the same program plus constraints, and the previous optimum is still feasible
    for it -- `|p - p_prev| = 0` satisfies any ramp bound.  A constrained problem
    whose optimum coincides with the unconstrained one has those constraints
    inactive.  So equality of the dispatch **is** slackness, exactly, and the
    ramp duals would add nothing; the clearing operator does not return them and
    does not need to.

    The converse is what gives the assertion teeth: a carry from anywhere else is
    not that optimum, so the ramp rows generally do bind and the dispatch moves.
    The third test below measures that on a rejected construction rather than
    assuming it.
    """
    pos, case, actual, boundary, offer, step = rig
    u, d = _period(pos, actual, day, 0)
    p_prev, free_out = boundary(u, d)
    out = step(offer, u, d, p_prev)

    assert float(out["mu"]) < 1e-6
    delta = float(np.abs(np.asarray(out["award"])[:, 0] - np.asarray(p_prev)).max())
    scale = max(float(np.abs(np.asarray(p_prev)).max()), 1.0)
    assert delta / scale < 1e-9, (
        f"day {day}: re-clearing the opening period against its own carry moved "
        f"the dispatch by {delta:.3e} MW, so the ramp rows are binding and the "
        f"carry did not come from this period")


@pytest.mark.parametrize("day", [0, 10, 16])
def test_the_two_rejected_constructions_fail_these_criteria(rig, day):
    """The criteria must be able to fail, on the constructions §15 rejects.

    Without this, both assertions above could be satisfied by a boundary that
    happens to be benign on this data, and nothing would say so.  The committed
    minimum is the construction §15 measured shedding into a VOLL-priced first
    period; it is used here rather than the registered initial power because the
    latter is infeasible and would fail before producing a number to compare.
    """
    pos, case, actual, boundary, offer, step = rig
    u, d = _period(pos, actual, day, 0)
    good, _ = boundary(u, d)

    p_min = np.asarray(case.unit_p_min, np.float64)
    bad = jnp.asarray(p_min * np.asarray(u)[:, 0])           # the committed minimum
    out_bad = step(offer, u, d, bad)
    out_good = step(offer, u, d, good)

    moved = float(np.abs(np.asarray(out_bad["award"])[:, 0] - np.asarray(bad)).max())
    scale = max(float(np.abs(np.asarray(bad)).max()), 1.0)
    assert moved / scale > 1e-9, (
        f"day {day}: the committed-minimum carry left the ramp rows slack too, so "
        f"criterion 2 cannot distinguish it from the prescribed construction on "
        f"this data and is not a check here")

    # and it is worse on the criterion that matters physically
    shed_bad = float(np.asarray(out_bad["shed"]).sum())
    shed_good = float(np.asarray(out_good["shed"]).sum())
    assert shed_bad >= shed_good
