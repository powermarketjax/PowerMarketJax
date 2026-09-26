"""Realised demand at this market's resolution, one loader per case.

    actual_hh, days = load_gb_demand_half_hourly()      # case29gb
    actual_hh, days = load_rts_demand_half_hourly(...)  # case73rts, net of renewables
    actual_hh, days = load_nem_demand_half_hourly(...)  # case813nem, mainland

`(n_days, 48)` float32 megawatts.  Each is the realised leg of the pair its
day-ahead sibling in `envs.day_ahead.demand` returns, at 48 periods a day rather
than 24, and **takes the same pairing arguments as that sibling** -- that is what
lets one demand stamp on a fixture serve both markets, through
`half_hourly_from_meta` below.

The period structure follows the data: the forecast is hourly and the
realisation is native at 30 minutes or finer, so one day-ahead hour covers two
real-time periods.  Where the day-ahead loader aggregates, this one stops one
step earlier; nothing here is upsampled, and a case whose realised series is not
available at 30 minutes has no loader here rather than an interpolated one.

**The day set must match the day-ahead one exactly**, because the position
fixture's `day_index` indexes into it.  The days in each loader are the
intersection of "48 periods present" with that case's day-ahead day set, taken
from the day-ahead loader rather than recomputed.

**`floor_mw` is not the same operation here as it is there, and the difference
is measured.**  The day-ahead loaders clip the hourly series; clipping the
half-hourly series and averaging is not the same function as averaging and then
clipping, so on the periods where the floor binds the two legs of one run
disagree.  The size of that disagreement is in each loader's docstring.  It is
recorded rather than resolved: which of the two the run points adopt is a
scenario question, and the day-ahead position fixtures were built on the hourly
clip.
"""
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from powermarketjax.data.data_loader import DataLoader
from powermarketjax.envs.day_ahead.demand import (
    ACTUAL_SIGNAL, RTS_CLASS_PREFIX, RTS_NETTED, demand_kwargs_from_meta,
    load_gb_demand, load_nem_demand, load_rts_demand, nem_mainland_half_hours)

#: Real-time periods per market day.  Twice the day-ahead market's 24.
T_RT = 48

#: The RTS real-time load column of the 30-minute file.  The 60-minute file maps
#: a column onto `load.rts_rt_mw` and this one onto a distinct name on purpose:
#: `DatasetRegistry.resolve_signals` takes the first manifest that claims a
#: signal and says nothing when two do, so a shared name would make which
#: resolution a caller gets depend on the glob order of the manifest directory.
RTS_LOAD_SIGNAL_30MIN = "load.rts_rt_30min_mw"


def _on_the_day_ahead_days(
    s: pd.Series, day_ahead_days: pd.DatetimeIndex,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """``(actual, days)`` from a datetime-indexed series, on whole market days.

    The day set is the intersection of "48 periods present" with the day-ahead
    day set the caller passes, and the arrays are `(n_days, 48)` float32: the
    state carries float32 and the solver widens what it needs.  Shared by the
    three loaders because the position fixture's `day_index` indexes into this
    day set, so all three have to build it the same way or a fixture would index
    into a different calendar than the one it was written against.
    """
    s = s.sort_index()
    dates = pd.Index(s.index.date)
    counts = s.groupby(dates).transform("size")
    s = s[(counts == T_RT) & dates.isin(day_ahead_days.date)]

    days = pd.DatetimeIndex(sorted(set(s.index.date)))
    if len(s) != len(days) * T_RT:                          # pragma: no cover
        raise ValueError(f"expected {T_RT} periods per day, got "
                         f"{len(s)} points over {len(days)} days")
    return s.to_numpy(np.float32).reshape(len(days), T_RT), days


def load_gb_demand_half_hourly(
    data_dir: Optional[Path] = None,
    manifest_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Load the realised demand at 48 periods per day, on the day-ahead day set.

    Returns ``(actual, days)`` with ``actual`` of shape ``(n_days, 48)`` in
    float32 megawatts: the state carries float32 and the solver widens what it
    needs.
    """
    _forecast, _hourly, day_ahead_days = load_gb_demand(data_dir, manifest_dir)

    loader = DataLoader(data_dir=data_dir, manifest_dir=manifest_dir)
    df = loader.load_signals([ACTUAL_SIGNAL])
    s = df.set_index("datetime")[ACTUAL_SIGNAL].astype(np.float64)
    return _on_the_day_ahead_days(s, day_ahead_days)


# ── case73rts: the 30-minute real-time file ──────────────────────────────


def load_rts_demand_half_hourly(
    netted: Tuple[str, ...] = RTS_NETTED,
    floor_mw: Optional[float] = None,
    data_dir: Optional[Path] = None,
    manifest_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    r"""Realised RTS net load at 48 periods a day, calendar 2020.

    ``netted`` and ``floor_mw`` mean exactly what they mean in
    `load_rts_demand`, which this is the realised leg of, and `floor_mw` has no
    default here for the same reason it has none there.

    **This reads a different file from its day-ahead sibling.** RTS-GMLC ships
    its real-time series at 5 minutes and the 60-minute parquet averages them
    12:1, so a 30-minute series cannot be recovered from it;
    `tools/data_prep/rts_gmlc_timeseries.py --resolution 30min` writes the same
    sources at 6:1 into a second file, five real-time columns and no day-ahead
    ones.  The two files agree where they can be compared: this one aggregated
    2:1 reproduces the hourly file's ``*_rt_mw`` columns to
    2.7e-12 MW, which is the sidecar's
    ``consistency_with_hourly_file``.

    **The 5-minute detail this discards is real**, so the choice of 30 minutes is
    a period length and not a claim that finer is empty: within an hour the
    5-minute regional load has a median range of 39 to 54 MW and reaches 871 MW,
    and no hour of 2020 is flat.  `T_RT` is 48 because the market is; going finer would move it.

    **What the floor costs at this resolution.** At the adopted
    ``floor_mw = 2500`` the floor binds on 9 736 of the
    17 568 half-hours, so the half-hourly clip and the hourly
    clip are not the same series: aggregating this output 2:1 differs from
    `load_rts_demand`'s realised leg on 290 of
    8 784 hours, by at most 145.60 MW.  With
    ``floor_mw=None`` the same comparison agrees to 3.7e-04 MW, which is under
    one float32 ULP at these magnitudes rather than a disagreement: both sides
    are means of the same 5-minute samples and the 6:1-then-2:1 route differs
    from the 12:1 one only in where it rounds.  All of these measured
    2026-09-09, CPU, with ``netted`` = all four classes.

    Returns:
        ``(actual, days)`` with ``actual`` of shape ``(n_days, 48)`` float32 MW.
    """
    # `netted` is validated by the day-ahead loader on the next line rather than
    # again here; one copy of that rule is the point of calling it
    _forecast, _hourly, day_ahead_days = load_rts_demand(
        netted=netted, floor_mw=floor_mw,
        data_dir=data_dir, manifest_dir=manifest_dir)

    loader = DataLoader(data_dir=data_dir, manifest_dir=manifest_dir)
    wanted = [RTS_LOAD_SIGNAL_30MIN]
    wanted += [f"{RTS_CLASS_PREFIX[cls]}_rt_30min_mw" for cls in netted]
    df = loader.load_signals(wanted).set_index("datetime").astype(np.float64)

    actual = df[RTS_LOAD_SIGNAL_30MIN]
    for name in wanted[1:]:
        actual = actual - df[name]
    if floor_mw is not None:
        actual = actual.clip(lower=floor_mw)
    return _on_the_day_ahead_days(actual, day_ahead_days)


# ── case813nem: the panel before its 2:1 aggregation ─────────────────────


def load_nem_demand_half_hourly(
    floor_mw: Optional[float] = None,
    data_dir: Optional[Path] = None,
    manifest_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Realised mainland NEM demand at 48 periods a day.

    ``floor_mw`` means what it means in `load_nem_demand`, which this is the
    realised leg of, and has no default here for the same reason.

    **No new data source.** ``OPERATIONAL_DEMAND`` is native at 30 minutes in the
    AEMO panel this tree already carries; both legs read it through the one
    function `nem_mainland_half_hours`, and `load_nem_demand` averages what comes
    back 2:1 while this takes the realised column as it stands.  So the forecast
    column's gaps drop half-hours here too -- 17 760 of them survive for both
    markets -- which is deliberate: dropping only the realisation's gaps would
    give this loader days its day-ahead sibling does not have.

    **The hourly aggregation is not a repackaging of the same numbers.**  Each
    half-hour differs from its hour's mean by a median of 236.2 MW and by up to
    2 750.0 MW, which is the
    intraday movement market 02 is here to see.

    **What the floor costs at this resolution.** At the adopted
    ``floor_mw = 11500`` the floor binds on 118 of the
    17 760 half-hours; aggregating this output 2:1 then differs
    from `load_nem_demand`'s realised leg on 20 of
    8 880 hours, by at most 589.00 MW.  With
    ``floor_mw=None`` the comparison is exact: the largest difference over the
    8 880 hours is 0 MW.  Measured 2026-09-09, CPU.

    Returns:
        ``(actual, days)`` with ``actual`` of shape ``(n_days, 48)`` float32 MW.
    """
    _forecast, _hourly, day_ahead_days = load_nem_demand(
        floor_mw=floor_mw, data_dir=data_dir, manifest_dir=manifest_dir)

    actual = nem_mainland_half_hours(data_dir, manifest_dir)["actual"]
    if floor_mw is not None:
        actual = actual.clip(lower=floor_mw)
    return _on_the_day_ahead_days(actual, day_ahead_days)


# ── which realised series goes with which case ───────────────────────────

#: The 48-period loader each case is defined against, keyed as `CASE_DEMAND` is
#: and pairing on the same arguments, which `half_hourly_from_meta` relies on.
#: A case absent here has no realised series at this market's resolution and
#: raises rather than falling back to the GB one -- the failure `8f8e704` fixed
#: on the forecast side, which until this table existed was still live here:
#: every driver that read `meta["case"]` for its network then called
#: `load_gb_demand_half_hourly()` unconditionally.
CASE_REALISED = {
    "29gb": load_gb_demand_half_hourly,
    "73rts": load_rts_demand_half_hourly,
    "813nem": load_nem_demand_half_hourly,
}


def half_hourly_from_meta(
    meta: dict,
    data_dir: Optional[Path] = None,
    manifest_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """The realised series at 48 periods that a fixture's `meta` describes.

    `demand_from_meta` for this market: same record, same validation -- the
    pairing is read by `demand_kwargs_from_meta`, so a fixture carries one demand
    stamp and the two markets cannot come to disagree about what it says -- and
    the same refusal of a case that has no series of its own.  What differs is
    only which table the case indexes.

    ``demand_source`` records the day-ahead loader, because that is the function
    the fixture's own series came from; it is not restamped here.  The loader in
    force is printed for the same reason it is there: so the run's log says which
    series it ran on.
    """
    case = meta["case"]
    if case not in CASE_REALISED:
        raise ValueError(
            f"fixture is for case {case!r}, which has no realised series at "
            f"{T_RT} periods a day; it is not paired with GB demand by default")
    kwargs, recorded = demand_kwargs_from_meta(meta)
    loader = CASE_REALISED[case]
    shown = ", ".join(f"{k}={v!r}" for k, v in kwargs.items()) if recorded else ""
    note = "" if recorded else "  -- no recorded pairing, pre-stamp fixture"
    print(f"realised demand: {loader.__name__}({shown}){note}", flush=True)
    return loader(**kwargs, data_dir=data_dir, manifest_dir=manifest_dir)
