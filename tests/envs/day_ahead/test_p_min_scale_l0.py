"""`p_min_scale`: the third scenario scale, beside `cap_scale` and `ramp_scale`.

Why it exists.  `case73rts` carries an aggregate minimum-generation ratio of
0.464 (sum p_min over sum p_max) against 0.274 for `case813nem` and 0.200 for
`case29gb`, and that is what makes its chained commitment infeasible: a
commitment sized for the day's peak cannot shut down to the trough, because the
committed minimum alone exceeds it.

Why it is a *scale* and not an edited data file.  `load_case` resolves a name and
`meta["case"]` records only that name, so a case edited in place produces
products that are indistinguishable by name, by path and by meta from products
built on the registered data -- the same failure `refuse_if_scenario_moved`
exists to stop.  `cap_scale` and `ramp_scale` already solve this by multiplying
registered data per run and recording the multiplier; this is the third of them,
and RTS-96 stays RTS-96.

Why it scales the case rather than being threaded through each operator.  Every
consumer of `unit_p_min` -- `segment_costs`, `make_relax`, `make_clearing`,
`precommit.build` -- would need the parameter, and one of them silently not
getting it is exactly the failure mode that is hard to see: the run would report
a scaled scenario while one operator solved the unscaled one.  Scaling the case
once makes that unrepresentable.
"""
import numpy as np
import pytest

from powermarketjax.case import load_case, scale_min_output
from powermarketjax.envs.day_ahead.clearing import segment_costs


@pytest.fixture(scope="module")
def rts():
    return load_case("73rts")


def test_unit_scale_is_the_identity(rts):
    """1.0 must change nothing, so a default run is the registered case."""
    same = scale_min_output(rts, 1.0)
    assert np.array_equal(np.asarray(same.unit_p_min), np.asarray(rts.unit_p_min))
    assert np.array_equal(np.asarray(same.unit_p_max), np.asarray(rts.unit_p_max))


def test_it_scales_p_min_and_leaves_p_max_alone(rts):
    s = scale_min_output(rts, 0.8)
    assert np.allclose(np.asarray(s.unit_p_min),
                       0.8 * np.asarray(rts.unit_p_min), rtol=0, atol=1e-9)
    # capacity is not a minimum-generation property and must not move with it
    assert np.array_equal(np.asarray(s.unit_p_max), np.asarray(rts.unit_p_max))


def test_the_adopted_scale_puts_rts_between_nem_and_its_own_registered_value():
    """The number the run point is chosen at, asserted rather than described."""
    ratio = lambda c: float(np.asarray(c.unit_p_min).sum()
                            / np.asarray(c.unit_p_max).sum())
    rts, nem, gb = load_case("73rts"), load_case("813nem"), load_case("29gb")
    assert ratio(rts) == pytest.approx(0.464, abs=5e-4)
    assert ratio(nem) == pytest.approx(0.274, abs=5e-4)
    assert ratio(gb) == pytest.approx(0.200, abs=5e-4)
    # 0.80 is the adopted value: still the highest of the three, so the
    # correction is conservative rather than one that makes RTS the outlier
    # in the other direction
    assert ratio(scale_min_output(rts, 0.80)) == pytest.approx(0.371, abs=5e-4)
    assert ratio(scale_min_output(rts, 0.80)) > ratio(nem) > ratio(gb)


def test_the_segment_envelope_follows_the_scale(rts):
    """Lowering the minimum widens the dispatchable range; costs re-integrate.

    Checked because `segment_costs` reads `unit_p_min` itself: if the scale did
    not reach it, the LP would carry a scaled minimum with unscaled segment
    widths and the capacity rows would not add up to `p_max`.
    """
    w0, _ = segment_costs(rts, 1)
    w1, _ = segment_costs(scale_min_output(rts, 0.8), 1)
    pmin = np.asarray(rts.unit_p_min, np.float64)
    pmax = np.asarray(rts.unit_p_max, np.float64)
    assert np.allclose(w0, np.maximum(pmax - pmin, 1e-9))
    assert np.allclose(w1, np.maximum(pmax - 0.8 * pmin, 1e-9))
    # p_min * u + sum_k g still reaches exactly p_max at full output
    assert np.allclose(0.8 * pmin + w1, pmax)


def test_precommit_records_the_scale_as_scenario():
    """A product built at another `p_min_scale` is a different market."""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]
                          / "tools" / "commitment"))
    import precommit                                              # noqa: E402
    assert "p_min_scale" in precommit.SCENARIO_KEYS, (
        "two fixtures differing only in p_min_scale would land on one derived "
        "path and overwrite each other in silence")


# ── the ABSENT trap: a new scale must not silently overwrite an old product ──

def _write_fixture(path, meta):
    import json
    np.savez(path, day_index=np.arange(3),
             meta=np.array(json.dumps(meta)))


def _precommit():
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]
                          / "tools" / "commitment"))
    import precommit
    return precommit


BASE_META = dict(mode="relax", case="29gb", n_periods=24, n_segments=1,
                 cap_scale=0.4, ramp_scale=0.25,
                 demand_source="load_gb_demand", demand_kwargs={})


def test_a_prior_without_the_scale_is_read_as_unscaled_not_as_uncomparable(tmp_path):
    """The failure `powermarketjax-ba` named when `p_min_scale` was proposed.

    `refuse_if_scenario_moved` treats a field the prior does not record as ABSENT
    and does not refuse on it, which is right for the demand pair: an unstamped
    non-GB product cannot be read at all, so an absent-and-same-`case` prior can
    only be a `29gb` one paired with `load_gb_demand()`, i.e. the same market.

    `p_min_scale` has no such compensating fact, and it does not need one: unlike
    the demand pairing it has a *known* value for the years before it existed --
    nothing was scaled, so it was 1.0.  Read that way the comparison is exact in
    both directions, and the case this test pins is the one that would otherwise
    fail open: an existing `29gb` product, built before the field, overwritten by
    a `29gb` run at 0.80.  Same case, every other scenario field equal, and a
    genuinely different market.
    """
    precommit = _precommit()
    path = tmp_path / "day_ahead_commitment_29gb_T24_relax.npz"
    _write_fixture(path, BASE_META)                      # built before this field existed

    now = dict(BASE_META, p_min_scale=0.80, day_index=[0, 1, 2])
    with pytest.raises(SystemExit):
        precommit.refuse_if_scenario_moved(path, now, force=False)


def test_the_same_prior_at_the_implied_value_is_not_a_move(tmp_path):
    """The other direction: reading ABSENT as 1.0 must not refuse an unscaled run.

    Without this the fix would trade a silent overwrite for a refusal of every
    default run against every product built before the field -- which is the
    failure the ABSENT branch was added to avoid in the first place.
    """
    precommit = _precommit()
    path = tmp_path / "day_ahead_commitment_29gb_T24_relax.npz"
    _write_fixture(path, BASE_META)
    now = dict(BASE_META, p_min_scale=1.0, day_index=[0, 1, 2])
    precommit.refuse_if_scenario_moved(path, now, force=False)   # passes if it does not raise


def test_the_demand_pair_keeps_its_absent_semantics(tmp_path):
    """`p_min_scale`'s rule must not leak onto the fields it was not argued for.

    The demand pairing has no implied prior value -- a product that does not
    record it does not tell you which series it used -- so it stays ABSENT and
    uncompared, on the external fact named in `refuse_if_scenario_moved`.
    """
    precommit = _precommit()
    assert "p_min_scale" in precommit.IMPLIED_PRIOR
    assert "demand_source" not in precommit.IMPLIED_PRIOR
    assert "demand_kwargs" not in precommit.IMPLIED_PRIOR

    path = tmp_path / "day_ahead_commitment_29gb_T24_relax.npz"
    old = {k: v for k, v in BASE_META.items()
           if k not in ("demand_source", "demand_kwargs")}
    _write_fixture(path, old)
    now = dict(BASE_META, p_min_scale=1.0, day_index=[0, 1, 2])
    precommit.refuse_if_scenario_moved(path, now, force=False)   # ABSENT, not refused
