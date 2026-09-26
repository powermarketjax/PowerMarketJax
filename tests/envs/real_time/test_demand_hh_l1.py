r"""L1 for the realised series at 48 periods a day, one loader per case.

Until 2026-09-09 this market had one realised loader and it was GB's, so every
driver that read its network from a fixture's `meta["case"]` then called
`load_gb_demand_half_hourly()` regardless -- the forecast side of exactly that
defect is what `8f8e704` fixed and what `tests/tools/test_demand_case_pairing_l0`
holds shut.  `case73rts` and `case813nem` now have their own, and this checks the
two properties that make them usable rather than merely present:

    the day set   is the day-ahead loader's, because the position fixture's
                  `day_index` indexes into it
    the values    are the same realised series the day-ahead leg carries, one
                  aggregation step earlier

The second is checked by aggregating this market's output 2:1 and comparing it
with the day-ahead loader's realised leg.  **The two sides are two code paths
over two files, not one implementation against a recomputed expectation**: for
`73rts` the half-hourly leg reads the 30-minute parquet (6:1 from the 5-minute
sources) and the day-ahead leg reads the 60-minute one (12:1 from the same
sources), and the two files are written by separate passes of
`tools/data_prep/rts_gmlc_timeseries.py`.  For `813nem` both legs read one panel
but through separate functions.

**The floor is where they are allowed to differ, and by how much is measured.**
Clipping half-hours and averaging is not the same function as averaging and then
clipping, so `test_the_floor_diverges_by_the_measured_amount` states the size of
that gap rather than hiding it in a tolerance.  Which convention the run points
adopt is an open scenario question; the day-ahead position fixtures were built on
the hourly clip.
"""
import shutil

import numpy as np
import pandas as pd
import pytest

from powermarketjax.envs.day_ahead.demand import (NEM_MAINLAND, RTS_NETTED,
                                                  load_nem_demand,
                                                  load_rts_demand)
from powermarketjax.envs.real_time.demand import (CASE_REALISED, T_RT,
                                                  half_hourly_from_meta,
                                                  load_gb_demand_half_hourly,
                                                  load_nem_demand_half_hourly,
                                                  load_rts_demand_half_hourly)

#: The adopted run points, 2026-09-04.
RTS_FLOOR, NEM_FLOOR = 2500.0, 11500.0

#: How far the 2:1 aggregation of this market's series may sit from the
#: day-ahead leg's before the two are called different series.  Measured
#: 2026-09-09 on CPU with ``floor_mw=None``: `73rts` 3.66e-04 MW over 8 784
#: hours, `813nem` 0 MW exactly over 8 880.  The RTS figure is float32
#: quantization, not disagreement -- one ULP at 6 000 MW is 4.9e-04 MW -- so the
#: constant is set just above it and is **not** a physical tolerance: derived
#: from the float32 output format, per the ULP rule, rather than pinned to a
#: measured run.
AGG_TOL_MW = 1e-3


@pytest.fixture(scope="module")
def rts_unfloored():
    return load_rts_demand_half_hourly(floor_mw=None)


@pytest.fixture(scope="module")
def nem_unfloored():
    return load_nem_demand_half_hourly(floor_mw=None)


def _aggregate(hh):
    """The half-hourly array at the day-ahead market's 24 periods."""
    return hh.astype(np.float64).reshape(len(hh), 24, 2).mean(2)


# ── the two properties that make a realised series usable ────────────────


@pytest.mark.parametrize("hh_loader, hourly_loader, floor, n_days", [
    (load_rts_demand_half_hourly, load_rts_demand, RTS_FLOOR, 366),
    (load_nem_demand_half_hourly, load_nem_demand, NEM_FLOOR, 370),
])
def test_shape_dtype_and_day_set_follow_the_day_ahead_leg(
        hh_loader, hourly_loader, floor, n_days):
    """A day set of its own would silently reindex the position fixture."""
    hh, days = hh_loader(floor_mw=floor)
    _f, hourly, day_ahead_days = hourly_loader(floor_mw=floor)

    assert hh.shape == (n_days, T_RT)
    assert hourly.shape == (n_days, 24)
    # float32 because the EnvState arrays are
    assert hh.dtype == np.float32
    assert not np.isnan(hh).any()
    assert list(days) == list(day_ahead_days)
    assert days.is_monotonic_increasing and days.is_unique


def test_rts_unfloored_series_is_the_day_ahead_one_before_aggregation(rts_unfloored):
    """Two files, two aggregation ratios of the same 5-minute samples."""
    hh, days = rts_unfloored
    _f, hourly, _d = load_rts_demand(floor_mw=None)
    worst = float(np.abs(_aggregate(hh) - hourly.astype(np.float64)).max())
    assert worst < AGG_TOL_MW, f"6:1 then 2:1 differs from 12:1 by {worst:.3e} MW"


def test_nem_unfloored_series_is_the_day_ahead_one_before_aggregation(nem_unfloored):
    hh, days = nem_unfloored
    _f, hourly, _d = load_nem_demand(floor_mw=None)
    worst = float(np.abs(_aggregate(hh) - hourly.astype(np.float64)).max())
    assert worst < AGG_TOL_MW, f"the two legs differ by {worst:.3e} MW"


@pytest.mark.parametrize("hh_loader, hourly_loader, floor, n_hours, max_mw", [
    (load_rts_demand_half_hourly, load_rts_demand, RTS_FLOOR, 290, 145.60),
    (load_nem_demand_half_hourly, load_nem_demand, NEM_FLOOR, 20, 589.00),
])
def test_the_floor_diverges_by_the_measured_amount(
        hh_loader, hourly_loader, floor, n_hours, max_mw):
    """Clip-then-average is not average-then-clip, and this states the size.

    Measured 2026-09-09 on CPU at the adopted floors.  It is not a tolerance to
    be widened: if these move, one of the two legs changed and the run points
    built on the other one have to be re-read.  Hours are counted above
    `AGG_TOL_MW` so that float32 quantization is not mistaken for divergence --
    counting above 1e-6 instead gives 2 183 hours for RTS, which is the same 290
    real ones plus quantization noise.
    """
    hh, days = hh_loader(floor_mw=floor)
    _f, hourly, _d = hourly_loader(floor_mw=floor)
    gap = np.abs(_aggregate(hh) - hourly.astype(np.float64))
    assert int((gap > AGG_TOL_MW).sum()) == n_hours
    assert float(gap.max()) == pytest.approx(max_mw, abs=0.01)


@pytest.mark.parametrize("fixture_name, median_mw", [
    ("rts_unfloored", 40.0),
    ("nem_unfloored", 200.0),
])
def test_the_half_hourly_series_is_not_the_hourly_one_repeated(
        request, fixture_name, median_mw):
    """If it were, this market would have no intraday content to price.

    Measured 2026-09-09: the median absolute deviation of a half-hour from its
    own hour's mean is 49.1 MW for `73rts` and 236.2 MW for `813nem`
    (measured separately from the sources rather than through these loaders).
    The bounds below are set well under those so that a data refresh does not
    trip them; what they refuse is a step function.
    """
    hh, _days = request.getfixturevalue(fixture_name)
    within = hh.astype(np.float64) - np.repeat(_aggregate(hh), 2, axis=1)
    assert float(np.median(np.abs(within))) > median_mw
    assert float(np.abs(within).max()) > 10 * median_mw


# ── the injection: what amplitude actually makes the check above fail ────


def _tainted_parquet_dir(tmp_path, names):
    """A data directory holding only the parquet files a case needs.

    The manifests are not copied: `DataLoader` takes the two directories
    separately, so the registry keeps reading the real ones and only the data
    moves.  Copying the whole parquet directory would move 196 MB.
    """
    from powermarketjax.data import data_loader
    src = data_loader.__file__.rsplit("/", 1)[0] + "/parquet"
    out = tmp_path / "parquet"
    out.mkdir()
    for name in names:
        for suffix in (".parquet", ".json"):
            shutil.copy(f"{src}/{name}{suffix}", out / f"{name}{suffix}")
    return out


#: The injection amplitude the tests below use, and the amplitude at which the
#: check actually turns over.  **Measured, not asserted from the arithmetic**
#: (2026-09-09, CPU): stepping one stored value by delta and re-running
#: `test_*_unfloored_series_is_the_day_ahead_one_before_aggregation`, `73rts`
#: goes red between 1e-3 and 2e-3 MW and `813nem` between 2e-3 and 4e-3 MW.
#: Both thresholds are float32 quantization of the output, ~1e-7 relative at
#: these magnitudes, not a property of the arithmetic in between: the transfer
#: from a stored 30-minute value to the aggregated hour is exactly delta/2 on
#: both paths.  1 MW is used below because it leaves ~2.5 decades of margin;
#: reporting "one bit would do it" would be the false claim, since a change of
#: 1e-3 MW at a 6 817 MW period does *not* turn this check red.
INJECT_MW = 1.0

RTS_HOURLY_FILE = "RTS_GMLC_Load_and_Renewables_2020_60min"
RTS_30MIN_FILE = "RTS_GMLC_RealTime_Load_and_Renewables_2020_30min"
NEM_PANEL_FILE = "AEMO_Forecast_vs_Actual_2025"


def test_a_one_megawatt_error_in_the_rts_thirty_minute_file_is_caught(tmp_path):
    """The control runs first: the copy alone must not turn the check red."""
    data_dir = _tainted_parquet_dir(tmp_path, [RTS_HOURLY_FILE, RTS_30MIN_FILE])
    _f, hourly, _d = load_rts_demand(floor_mw=None)

    def worst():
        hh, _days = load_rts_demand_half_hourly(floor_mw=None, data_dir=data_dir)
        return float(np.abs(_aggregate(hh) - hourly.astype(np.float64)).max())

    assert worst() < AGG_TOL_MW, "the copied files already differ from the tree"

    path = data_dir / f"{RTS_30MIN_FILE}.parquet"
    df = pd.read_parquet(path)
    # the period of peak net load, so the injection cannot be absorbed by a
    # floor -- `floor_mw=None` here, but the point is chosen to stay valid if
    # this test is ever re-run with one
    net = df["load_rt_mw"] - df[[f"{c}_rt_mw" for c in RTS_NETTED]].sum(axis=1)
    i = int(net.idxmax())
    df.loc[i, "load_rt_mw"] = df.loc[i, "load_rt_mw"] + INJECT_MW
    df.to_parquet(path, compression="zstd", index=False)

    assert worst() > AGG_TOL_MW, (
        f"a {INJECT_MW} MW error at period {i} did not reach the check")
    assert worst() == pytest.approx(INJECT_MW / 2, abs=1e-3)


def test_a_one_megawatt_error_in_the_nem_panel_is_caught(tmp_path):
    data_dir = _tainted_parquet_dir(tmp_path, [NEM_PANEL_FILE])
    _f, hourly, _d = load_nem_demand(floor_mw=None)

    def worst():
        hh, _days = load_nem_demand_half_hourly(floor_mw=None, data_dir=data_dir)
        return float(np.abs(_aggregate(hh) - hourly.astype(np.float64)).max())

    assert worst() < AGG_TOL_MW, "the copied panel already differs from the tree"

    path = data_dir / f"{NEM_PANEL_FILE}.parquet"
    df = pd.read_parquet(path)
    mainland = df[df["REGIONID"].isin(NEM_MAINLAND)]
    i = int(mainland["OPERATIONAL_DEMAND"].idxmax())
    df.loc[i, "OPERATIONAL_DEMAND"] = df.loc[i, "OPERATIONAL_DEMAND"] + INJECT_MW
    df.to_parquet(path, compression="zstd", index=False)

    assert worst() > AGG_TOL_MW, (
        f"a {INJECT_MW} MW error at row {i} did not reach the check")
    assert worst() == pytest.approx(INJECT_MW / 2, abs=1e-3)


# ── the dispatcher: one record, two markets ──────────────────────────────


def test_every_case_with_a_demand_pair_has_a_realised_series():
    """The two tables are keyed alike, which `half_hourly_from_meta` relies on."""
    from powermarketjax.envs.day_ahead.demand import CASE_DEMAND
    assert set(CASE_REALISED) == set(CASE_DEMAND)


def test_each_realised_loader_takes_its_day_ahead_sibling_s_pairing_arguments():
    """A fixture carries one demand stamp; both markets have to be able to use it.

    Checked on the signatures rather than by calling, because a mismatch that
    only shows up when a run reaches it would show up as `TypeError` in the
    middle of a training job.
    """
    import inspect
    from powermarketjax.envs.day_ahead.demand import CASE_DEMAND

    for case, (loader, required) in CASE_DEMAND.items():
        params = inspect.signature(CASE_REALISED[case]).parameters
        missing = set(required) - set(params)
        assert not missing, (
            f"{CASE_REALISED[case].__name__} cannot accept {sorted(missing)}, "
            f"which {loader.__name__} pairs on and a {case} fixture records")


def test_a_case_without_a_realised_series_is_refused_not_paired_with_gb():
    with pytest.raises(ValueError, match="118ieee"):
        half_hourly_from_meta({"case": "118ieee"})


def test_an_rts_meta_does_not_resolve_to_gb_half_hours():
    """The failure this table exists to stop, in the form it had until today."""
    meta = {"case": "73rts", "demand_source": "load_rts_demand",
            "demand_kwargs": {"netted": list(RTS_NETTED), "floor_mw": RTS_FLOOR}}
    hh, days = half_hourly_from_meta(meta)
    gb, gb_days = load_gb_demand_half_hourly()
    assert hh.shape != gb.shape and len(days) == 366 and len(gb_days) == 648
    assert float(hh.min()) == pytest.approx(RTS_FLOOR)
    assert float(hh.max()) < float(gb.min())


def test_a_legacy_gb_meta_still_resolves_to_gb_half_hours(capsys):
    """The pre-stamp fixtures are every product this repository has built."""
    hh, days = half_hourly_from_meta({"case": "29gb", "n_periods": 24})
    gb, gb_days = load_gb_demand_half_hourly()
    assert np.array_equal(hh, gb) and list(days) == list(gb_days)
    # and it announces which branch it took, as the day-ahead side does
    assert "load_gb_demand_half_hourly" in capsys.readouterr().out


def test_the_pairing_record_is_read_by_the_day_ahead_validator():
    """One record, one validator: a bad stamp is refused here as it is there."""
    bad = {"case": "73rts", "demand_source": "load_gb_demand",
           "demand_kwargs": {"netted": list(RTS_NETTED), "floor_mw": RTS_FLOOR}}
    with pytest.raises(ValueError, match="demand_source"):
        half_hourly_from_meta(bad)
    bare = {"case": "73rts"}
    with pytest.raises(KeyError, match="demand"):
        half_hourly_from_meta(bare)
