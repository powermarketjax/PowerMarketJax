"""Load the exogenous series of the local flexibility market: feeder demand,
photovoltaic output and the exogenous energy price, on one grid.

Setup-time pandas, run once per configuration.  Returns the arrays
``make_local_flex_params`` takes.

    load_mw, pv, price, index = load_local_flex_series("Ashfield 33_11kV")
    params = make_local_flex_params(load_mw, np.outer(pv, pv_capacity_mw),
                                    price, battery, cycle_cost, kappa, mask, H)

Two configurations are supported: an Australian zone-substation feeder on a
15-minute grid, and a Swiss medium-voltage feeder on an hourly grid.

Each loader resolves its series through the manifest registry by signal name
and raises ``MissingSeries`` naming what a manifest would have to provide
when none is registered.  Every series of both configurations is registered
or vendored here, so that path is reached only by a caller pointing
``manifest_dir=`` or ``data_dir=`` somewhere else.

The grid is the demand series' own and must be complete: a gap would rotate
the row-indexed calendar encoding the environment computes from it, so the
loader reindexes onto the regular grid the timestamps imply and raises on a
gap rather than interpolating one away.  The row index means time of day
only if the series starts at local midnight, which the loader checks.
"""
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

from powermarketjax.data.manifest import DatasetManifest
from powermarketjax.data.registry import DatasetRegistry

#: The zone-substation demand series: 15-minute, 175 zone substations,
#: financial year ending April 2025, gaps already imputed by the publisher.
DEMAND_DATASET = "ausgrid_zone_substation_fy25_imputed"
DEMAND_SIGNAL = "load.actual_mw"

#: Signal the photovoltaic series has to provide.  Deliberately not
#: `solar.available_mw`, which other registered datasets already map at a
#: different aggregation level and year: a name no other dataset carries
#: keeps this market's series out of a collision.
PV_SIGNAL = "solar.substation_mw"

#: Signal the exogenous energy price has to provide.  A regional spot price
#: registered under a different name can be passed explicitly via `signal=`.
PRICE_SIGNAL = "market.spot_price_aud_mwh"

#: Period length of the demand series, which fixes the period length to 0.25 h.
PERIOD_HOURS = 0.25

#: Settlement calendar of the demand data, used only to check that the series
#: starts at a local midnight so that the row index means time of day.
DEMAND_TIMEZONE = "Australia/Sydney"

#: The two windows of the demand series that avoid both daylight-saving
#: transitions, each starting at a local midnight and staying inside one UTC
#: offset.  Pass one as ``start=``/``end=``: the summer window is longer and
#: is the season a photovoltaic series matters in, the winter one is where
#: demand peaks.
AEDT_WINDOW = ("2024-10-06 13:00", "2025-04-05 12:45")
AEST_WINDOW = ("2024-04-30 14:00", "2024-10-05 15:45")

#: A second configuration, not a replacement: medium-voltage feeder `459_0`
#: of the SwissDN dataset, hourly, so one `step` is one hour here against
#: fifteen minutes for the Australian series.  The demand table is per bus;
#: this market takes the feeder total, which is what ``aggregate=True`` on
#: the demand loader sums.
SWISS_DEMAND_DATASET = "swissdn_459_0_mv_load"

#: `UTC` and not `Europe/Zurich`: the source's `MM-DD HH:MM:SS` timestamps
#: carry no year and no timezone, so they are already local clock time, and
#: converting them would shift the row index of the calendar encoding by an
#: hour.  The photovoltaic table's `hour_of_day` uses the same convention.
SWISS_TIMEZONE = "UTC"

#: Two Swiss tables are deliberately not registered as manifests, so they are
#: read by filename here.  The photovoltaic one because the loader's tiling
#: would destroy its seasonal structure (see `load_swiss_pv`); the tariff one
#: because it is an annual scalar per category rather than a series, and there
#: is nothing for a time aligner to align.
SWISS_PV_FILE = "SwissDN_459_0_MV_PV_RepDays.parquet"
SWISS_TARIFF_FILE = "ElCom_SwissDN_459_0_Tariff_Annual.parquet"
SWISS_BESS_FILE = "SwissDN_459_0_MV_BESS.parquet"


class MissingSeries(RuntimeError):
    """A series this module requires has no dataset registered to provide it."""


class LocalFlexSeries(NamedTuple):
    """The three exogenous series on one grid, plus that grid.

    ``pv`` carries either shape, and which one depends on what the configuration
    publishes rather than on a preference.  The Australian series is regional, so
    it is ``(n_periods,)`` and the caller multiplies it by the installed capacity
    of each aggregator.  The Swiss one is per bus, so it is already
    ``(n_periods, n_agent)``.  ``make_local_flex_params`` takes the second form
    either way, so the difference does not reach the environment.
    """
    load_mw: np.ndarray           # float32 (n_periods,)  feeder total, MW
    pv: np.ndarray                # float32 (n_periods,) or (n_periods, n_agent)
    energy_price: np.ndarray      # float32 (n_periods,)  per MWh, currency of
                                  #   the configuration: the Australian path
                                  #   loads a \\$/MWh series, the Swiss one an
                                  #   ElCom tariff in CHF/MWh.  Nothing here
                                  #   converts, so a figure taken from one
                                  #   configuration is not comparable with the
                                  #   other's without saying which.
    index: pd.DatetimeIndex       # UTC, regular; step is the series' own


def _infer_step(index: pd.DatetimeIndex, period_hours: Optional[float],
                region: str) -> pd.Timedelta:
    """The grid step, taken from the series unless the caller fixes it.

    Inferring beats a module constant here because the constant was the
    Australian configuration's 0.25 h and the Swiss one is 1 h, and a constant
    that does not match the series does not fail loudly: it turns every hour
    into three absent quarter-hours and reports a gap, which reads as a defect
    in the data rather than as a mismatched period length.  The smallest gap
    between consecutive timestamps is the step whenever rows are missing, since
    a hole only ever widens a difference; duplicates are rejected before this.
    """
    if period_hours is not None:
        return pd.Timedelta(hours=period_hours)
    if len(index) < 2:
        raise ValueError(
            f"region {region!r} carries {len(index)} rows, so no period length "
            "can be inferred; pass period_hours= explicitly")
    return pd.Timedelta(index.to_series().diff().dropna().min())


def _registry(manifest_dir: Optional[Path]) -> DatasetRegistry:
    """The manifest registry, reading its own default directory unless the
    caller names one."""
    return DatasetRegistry(Path(manifest_dir) if manifest_dir else None)


def _default_data_dir() -> Path:
    """Where the parquet files sit: ``powermarketjax/data/parquet``.

    Resolved from the package rather than from the working directory, so a
    loader called from anywhere reads the same files.
    """
    from powermarketjax import data as data_pkg
    return Path(data_pkg.__file__).resolve().parent / "parquet"


def _frame(manifest: DatasetManifest, signal: str,
           data_dir: Optional[Path]) -> pd.DataFrame:
    """Read one manifest's parquet and rename its columns to canonical names.

    The renaming is the manifest's own `index_map` and `column_map`, so this
    reaches the same names `DataLoader.load_signals` would without going
    through the signal index, which cannot express "this dataset and not the
    other one that maps the same signal".

    Args:
        manifest: the resolved manifest; its ``parquet_file`` is read from
            ``data_dir``, which defaults to the packaged parquet directory.
        signal: the canonical name the caller needs after the renaming.
        data_dir: override for where the parquet files live.

    Returns:
        The frame with a UTC ``datetime`` column and whatever other canonical
        columns the manifest maps, ``region`` among them for a multi-region
        dataset.

    Raises:
        MissingSeries: if the parquet file is absent, or if it does not carry
            ``signal`` once renamed.
    """
    path = (Path(data_dir) if data_dir else _default_data_dir()) / manifest.parquet_file
    if not path.exists():
        raise MissingSeries(
            f"manifest '{manifest.name}' is registered but its parquet file is "
            f"absent: {path}")
    frame = pd.read_parquet(path)
    # A mapping whose raw column is absent is skipped rather than raising: a
    # manifest may map more signals than the file carries, and the one that
    # matters is checked by name below.
    rename = {raw: canon
              for raw, canon in {**manifest.index_map, **manifest.column_map}.items()
              if raw in frame.columns}
    frame = frame.rename(columns=rename)
    if signal not in frame.columns:
        raise MissingSeries(
            f"manifest '{manifest.name}' maps {signal} but the parquet does "
            f"not carry it after renaming; columns are {list(frame.columns)}")
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    return frame


def _series(manifest: DatasetManifest, signal: str, region: Optional[str],
            data_dir: Optional[Path]) -> pd.Series:
    """One numeric series indexed by timestamp, optionally filtered by region."""
    frame = _frame(manifest, signal, data_dir)
    if region is not None:
        if "region" not in frame.columns:
            raise ValueError(
                f"dataset '{manifest.name}' carries no region column, so "
                f"region={region!r} cannot be selected")
        selected = frame[frame["region"] == region]
        if selected.empty:
            known = manifest.region_values or sorted(frame["region"].unique())
            raise ValueError(
                f"region {region!r} is not in dataset '{manifest.name}'; it "
                f"carries {len(known)} regions, for example {known[:5]}")
        frame = selected
    return (frame.set_index("datetime")[signal]
            .astype(np.float64).sort_index())


def _resolve(signal: str, dataset: Optional[str], manifest_dir: Optional[Path],
             what: str, options: str) -> DatasetManifest:
    """Find the manifest that provides `signal`, or say what is missing."""
    registry = _registry(manifest_dir)
    if dataset is not None:
        try:
            manifest = registry.get_manifest(dataset)
        except KeyError as exc:
            raise MissingSeries(f"{what}: {exc}") from exc
        if signal not in manifest.signals:
            raise MissingSeries(
                f"{what}: dataset '{dataset}' does not provide {signal}; it "
                f"provides {manifest.signals}")
        return manifest

    candidates = registry.find_by_signal(signal)
    if not candidates:
        raise MissingSeries(
            f"{what}: no registered dataset provides {signal}, and a missing "
            f"series is never filled with a plausible number.  "
            f"{options}  Registering a manifest and a parquet file under "
            f"powermarketjax/data is the whole change needed here.")
    if len(candidates) > 1:
        raise MissingSeries(
            f"{what}: {len(candidates)} datasets provide {signal} "
            f"({[m.name for m in candidates]}); name one with dataset=, since "
            "resolving by signal alone would pick whichever sorts first")
    return candidates[0]


def load_substation_demand(
    region: str,
    *,
    dataset: str = DEMAND_DATASET,
    signal: str = DEMAND_SIGNAL,
    timezone: str = DEMAND_TIMEZONE,
    period_hours: Optional[float] = None,
    aggregate: bool = False,
    start: Optional[str] = None,
    end: Optional[str] = None,
    data_dir: Optional[Path] = None,
    manifest_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Load one zone substation's demand series.

    Args:
        region: zone substation name, e.g. ``"Ashfield 33_11kV"``.  It has no
            default: which feeder the series comes from is a scenario choice
            and travels with any result obtained under it.
        start, end: optional UTC bounds, inclusive, as anything
            ``pd.Timestamp`` accepts.
        aggregate: sum every row sharing a timestamp into one feeder total,
            which is what a dataset registering demand per bus needs -- the
            Swiss table is that shape.  ``region`` is then not read at all and
            is rebound to a label used only in the messages below.
        period_hours: fix the grid step instead of taking it from the
            timestamps; see `_infer_step` for why inferring is the default.
        timezone: the settlement calendar the midnight check at the end is
            made in.  It is a property of the dataset, not of the reader.
        dataset, signal, data_dir, manifest_dir: overrides for a different
            registration of the same series.

    Returns:
        ``(load_mw, index)`` with ``load_mw`` float32 ``(n_periods,)`` in MW
        and ``index`` the regular UTC grid it sits on.  The step is the
        series' own: 15 minutes for the Australian dataset, one hour for the
        Swiss one.

    Raises:
        MissingSeries: if no manifest provides the series.
        ValueError: on an unknown region, an irregular grid, a gap or a
            non-finite value, since each of those would shift every later
            period rather than show up as a market quantity.
    """
    manifest = _resolve(signal, dataset, manifest_dir,
                        "feeder demand series",
                        "The series is the adopted one and it is present.")
    if aggregate:
        frame = _frame(manifest, signal, data_dir)
        series = (frame.groupby("datetime")[signal].sum()
                  .astype(np.float64).sort_index())
        region = f"<all rows of {manifest.name}>"
    else:
        series = _series(manifest, signal, region, data_dir)
    if start is not None:
        series = series[series.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        series = series[series.index <= pd.Timestamp(end, tz="UTC")]
    if series.empty:
        raise ValueError(f"region {region!r} carries no rows in the window "
                         f"[{start}, {end}]")
    if series.index.has_duplicates:
        repeated = series.index[series.index.duplicated()].unique()
        raise ValueError(
            f"region {region!r} carries {len(repeated)} repeated timestamps, "
            f"the first at {repeated[0]}; this dataset was published on local "
            "timestamps and both daylight-saving transitions survived the "
            "conversion (module docstring), so a window spanning one has no "
            f"regular grid.  Use AEDT_WINDOW or AEST_WINDOW")

    step = _infer_step(series.index, period_hours, region)
    grid = pd.date_range(series.index[0], series.index[-1], freq=step)
    on_grid = series.reindex(grid)
    missing = int(on_grid.isna().sum())
    if missing:
        raise ValueError(
            f"region {region!r} is missing {missing} of {len(grid)} "
            f"{step.total_seconds() / 60:.0f}-minute periods between {grid[0]} "
            f"and {grid[-1]}; the environment counts periods by row, so a gap "
            "rotates the calendar encoding rather than showing up as a "
            "market quantity.  Choose another region or another window; do not "
            "interpolate silently here")
    if len(on_grid) != len(series):
        raise ValueError(
            f"region {region!r} carries {len(series)} rows on a grid of "
            f"{len(on_grid)} periods, so its timestamps are not a regular "
            f"{step.total_seconds() / 60:.0f}-minute series")

    local_start = grid[0].tz_convert(timezone)
    if (local_start.hour, local_start.minute) != (0, 0):
        raise ValueError(
            f"the series starts at {local_start} local time rather than at "
            "midnight, so the row index of the calendar encoding would not "
            "mean time of day; trim the window with start= to a local midnight")
    return on_grid.to_numpy(np.float32), grid


def load_pv_series(
    index: pd.DatetimeIndex,
    *,
    dataset: Optional[str] = None,
    signal: str = PV_SIGNAL,
    region: Optional[str] = None,
    upsample: str = "interpolate",
    data_dir: Optional[Path] = None,
    manifest_dir: Optional[Path] = None,
) -> np.ndarray:
    """Load the photovoltaic series onto ``index``.

    Returns float32 ``(len(index),)`` in whatever unit the dataset registers:
    a capacity factor multiplies an installed capacity, an output in MW
    multiplies a share.  The Australian configuration resolves to
    `aemo_nsw1_rooftop_pv`, a 30-minute regional total in MW, so it is the
    second of those and the caller supplies the share.

    A series coarser than the demand grid is interpolated rather than held,
    which is the branch of `_to_grid` this default selects; a finer one is
    averaged down either way.
    """
    manifest = _resolve(
        signal, dataset, manifest_dir, "photovoltaic series",
        "A substation-level series for the same region and year as the demand "
        "data is the coherent route; running with no photovoltaic generation, "
        "stated as load-driven only, is the fallback, and is passed "
        "as an explicit array of zeros rather than defaulted to here.")
    return _to_grid(_series(manifest, signal, region, data_dir), index, upsample)


def load_energy_price(
    index: pd.DatetimeIndex,
    *,
    dataset: Optional[str] = None,
    signal: str = PRICE_SIGNAL,
    region: Optional[str] = None,
    upsample: str = "ffill",
    data_dir: Optional[Path] = None,
    manifest_dir: Optional[Path] = None,
) -> np.ndarray:
    """Load the exogenous energy price onto ``index``, \\$/MWh.

    ``upsample`` defaults to holding the last value rather than interpolating,
    because a settlement price is constant across its own interval and a
    straight line between two of them is a price nobody was charged.  The
    Australian configuration resolves to `aemo_nsw1_dispatch_price`, which is
    finer than the demand grid and so is averaged down rather than held, and
    which goes negative.  A negative energy price inverts the markup map, and
    this series reaches that region.
    """
    manifest = _resolve(
        signal, dataset, manifest_dir, "exogenous energy price",
        "A price series for the same market as the demand data is the coherent "
        "route; the British index vendored here does not pair with an "
        "Australian load series.")
    return _to_grid(_series(manifest, signal, region, data_dir), index, upsample)


def load_swiss_pv(
    index: pd.DatetimeIndex,
    buses: "list[str]",
    *,
    projection_year: int,
    rule: str = "repeat",
    seed: Optional[int] = None,
    timezone: str = SWISS_TIMEZONE,
    data_dir: Optional[Path] = None,
) -> np.ndarray:
    """Photovoltaic output of the Swiss configuration, ``(len(index), len(buses))`` MW.

    The source publishes one representative day per month, 288 hourly values
    per bus and projection year against 8760 for the demand series, so a
    rollout over a month has to obtain the remaining days from the one
    supplied.  Which rule does that is a declared scenario parameter, not a
    property of the data, and it is passed rather than defaulted to silently:

        ``repeat``  every day of a month takes that month's representative
                    day.  Removes the day-to-day variation that makes the
                    published requirement move.
        ``sample``  draw each (day, bus, hour) from a normal centred on the
                    representative value with the standard deviation the
                    source publishes at that hour, truncated at zero.
                    Restores the variation at the cost of a second random
                    draw, so it needs ``seed``.

    This table is deliberately not reachable through the manifest registry:
    ``TimeAligner.align_profile`` tiles a repeatable profile end to end,
    which would turn twelve monthly days into a twelve-day cycle and remove
    the seasonal structure photovoltaic output needs.
    """
    if rule not in ("repeat", "sample"):
        raise ValueError(f"rule must be 'repeat' or 'sample', got {rule!r}")
    if rule == "sample" and seed is None:
        raise ValueError("rule='sample' draws a random variate per (day, bus, "
                         "hour) and therefore needs an explicit seed, which "
                         "travels with any result obtained under it")
    path = (Path(data_dir) if data_dir else _default_data_dir()) / SWISS_PV_FILE
    if not path.exists():
        raise MissingSeries(f"the Swiss photovoltaic table is absent: {path}")
    frame = pd.read_parquet(path)
    frame = frame[frame["projection_year"] == projection_year]
    if frame.empty:
        years = sorted(pd.read_parquet(path)["projection_year"].unique())
        raise ValueError(f"projection year {projection_year} is not published; "
                         f"the table carries {years}")
    missing = sorted(set(buses) - set(frame["osmid"].astype(str)))
    if missing:
        raise ValueError(
            f"{len(missing)} of the {len(buses)} buses asked for carry no "
            f"photovoltaic record in projection year {projection_year}, the "
            f"first being {missing[0]!r}; the equipped set is a property of the "
            "data and a bus outside it has no output rather than zero output")

    # Every period of the window takes the representative value of its (month,
    # hour of day) pair, which is the expansion the source's twelve days have
    # to go through.  `pivot_table` averages any repeated (month, hour, bus)
    # record rather than raising, and the reindex fixes the column order to
    # `buses`, that is, to the aggregator order the caller asked for.
    local = index.tz_convert(timezone)
    key = pd.MultiIndex.from_arrays([local.month, local.hour])
    mean = (frame.pivot_table(index=["month", "hour_of_day"], columns="osmid",
                              values="pv_kw")
            .reindex(columns=[str(b) for b in buses]))
    out = mean.reindex(key).to_numpy(np.float64) / 1000.0     # kW -> MW
    if rule == "sample":
        std = (frame.pivot_table(index=["month", "hour_of_day"], columns="osmid",
                                 values="pv_std_kw")
               .reindex(columns=[str(b) for b in buses])
               .reindex(key).to_numpy(np.float64) / 1000.0)
        # One draw per calendar day, not per period: within a day the source's
        # representative shape is the signal, and redrawing at every hour would
        # turn a smooth profile into noise no photovoltaic plant produces.
        day = local.normalize()
        codes = pd.factorize(day)[0]
        rng = np.random.default_rng(seed)
        shocks = rng.standard_normal((codes.max() + 1, out.shape[1]))
        out = np.maximum(out + shocks[codes] * std, 0.0)
    if not np.isfinite(out).all():
        raise ValueError("the photovoltaic table left non-finite values after "
                         "expansion, so some (month, hour) is not published")
    return out.astype(np.float32)


def load_elcom_tariff(
    index: pd.DatetimeIndex,
    *,
    category: str,
    period: int,
    data_dir: Optional[Path] = None,
) -> np.ndarray:
    """The adopted exogenous energy price of the Swiss configuration, CHF/MWh.

    The published figure is annual, so this broadcasts one number across the
    window rather than aligning a series: the price is constant over an
    episode of any length below a year, and the locational structure becomes
    the only thing an offer prices.

    Takes the energy component and not the total: this market prices the
    energy an aggregator stores, not the network service delivering it, and
    the two differ by roughly a factor of two here.

    Raises on a category the operator does not offer.  ElCom publishes those
    as rows of zeros rather than by omitting them, so selecting one would
    otherwise give a replacement cost of zero and offers priced at zero with
    nothing raising.

    Args:
        index: the grid to broadcast onto; only its length is read.
        category: ElCom consumption category, e.g. ``"C2"``.
        period: the tariff year.

    Returns:
        float32 ``(len(index),)`` carrying the same number in every period.
    """
    path = (Path(data_dir) if data_dir else _default_data_dir()) / SWISS_TARIFF_FILE
    if not path.exists():
        raise MissingSeries(f"the ElCom tariff table is absent: {path}")
    frame = pd.read_parquet(path)
    row = frame[(frame["category"] == category) & (frame["period"] == period)]
    if row.empty:
        cats = sorted(frame["category"].unique())
        yrs = (int(frame["period"].min()), int(frame["period"].max()))
        raise ValueError(
            f"no tariff published for category {category!r} in {period}; the "
            f"table carries categories {cats} over {yrs[0]}-{yrs[1]}")
    value = float(row["energy_chf_mwh"].iloc[0])
    if value <= 0.0:
        raise ValueError(
            f"category {category!r} carries an energy component of {value} in "
            f"{period}, which means this operator does not offer it: ElCom "
            "publishes an unoffered category as a row of zeros rather than "
            "omitting it, and taking it would give a cost basis of zero and "
            "offers priced at zero without anything raising")
    return np.full(len(index), value, dtype=np.float32)


def load_local_flex_series(
    region: str,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    pv_dataset: Optional[str] = None,
    price_dataset: Optional[str] = None,
    pv_region: Optional[str] = None,
    price_region: Optional[str] = None,
    data_dir: Optional[Path] = None,
    manifest_dir: Optional[Path] = None,
) -> LocalFlexSeries:
    """The three series on the demand series' own grid.

    The demand series decides the grid: the period length is taken from it
    and the other two are put onto it, averaged down or held and interpolated
    up as `_to_grid` describes.

    Returns:
        `LocalFlexSeries`: ``load_mw`` the feeder total in MW, ``pv``
        ``(n_periods,)`` regional and in the unit its dataset registers,
        ``energy_price`` in \\$/MWh, and ``index`` the UTC grid all three sit
        on.  ``pv`` has to be turned into the ``(n_periods, n_agent)`` matrix
        `make_local_flex_params` takes by the caller, since the per-aggregator
        capacity is not a property of the series.
    """
    load_mw, index = load_substation_demand(
        region, start=start, end=end, data_dir=data_dir,
        manifest_dir=manifest_dir)
    pv = load_pv_series(index, dataset=pv_dataset, region=pv_region,
                        data_dir=data_dir, manifest_dir=manifest_dir)
    price = load_energy_price(index, dataset=price_dataset, region=price_region,
                              data_dir=data_dir, manifest_dir=manifest_dir)
    return LocalFlexSeries(load_mw=load_mw, pv=pv, energy_price=price,
                           index=index)


def _to_grid(series: pd.Series, index: pd.DatetimeIndex,
             upsample: str) -> np.ndarray:
    """Put a series on ``index``, averaging down and holding or drawing up.

    A series finer than the target grid is averaged into it, which is what a
    15-minute period means for a 5-minute quantity.  A coarser one is either
    held constant across the target periods or interpolated in time, and which
    of the two is right depends on the quantity rather than on the arithmetic,
    so the caller states it.
    """
    if upsample not in ("ffill", "interpolate"):
        raise ValueError(f"upsample must be 'ffill' or 'interpolate', got "
                         f"{upsample!r}")
    step = index[1] - index[0]
    source_step = pd.Series(series.index).diff().median()
    if source_step < step:
        series = series.resample(step).mean()
        source_step = step

    # The span is checked before the reindex, and that is not belt and braces:
    # both fill rules extend the last value forward for as long as they are
    # asked to, so a series that stops inside the window would be silently
    # continued at its final value rather than leave a NaN to be caught below.
    covered = (series.index[0], series.index[-1] + source_step)
    window = (index[0], index[-1] + step)
    if covered[0] > window[0] or covered[1] < window[1]:
        raise ValueError(
            f"the series covers [{covered[0]}, {covered[1]}] and the demand "
            f"window runs over [{window[0]}, {window[1]}]; trim the window "
            "with start=/end= rather than letting the environment run on a "
            "series that stops")

    if upsample == "ffill":
        aligned = series.reindex(index, method="ffill")
    else:
        aligned = (series.reindex(series.index.union(index))
                   .interpolate(method="time").reindex(index))

    missing = int(aligned.isna().sum())
    if missing:
        raise ValueError(
            f"the series leaves {missing} of {len(index)} periods of the "
            f"demand window without a value even after {upsample}")
    return aligned.to_numpy(np.float32)


def load_swiss_flex_series(
    *,
    projection_year: int,
    tariff_category: str,
    tariff_period: int,
    pv_rule: str = "repeat",
    pv_seed: Optional[int] = None,
    buses: "Optional[list[str]]" = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    data_dir: Optional[Path] = None,
    manifest_dir: Optional[Path] = None,
) -> LocalFlexSeries:
    """The three exogenous series of the Swiss configuration.

    Every argument that has no default is a declared scenario parameter and
    travels with any result obtained under it: the projection year fixes the
    battery fleet, its placement and the aggregator count at once, and the
    tariff category and year fix the energy price.  There is no default for
    them because no value of either is more natural than another.

    ``buses`` defaults to the buses carrying a battery in ``projection_year``,
    the aggregator set the data determines, so the photovoltaic matrix comes
    back aligned to the participants rather than to the feeder.

    Returns:
        `LocalFlexSeries` on the demand table's hourly UTC grid: ``load_mw``
        the feeder total in MW, ``pv`` ``(n_periods, len(buses))`` in MW,
        ``energy_price`` constant across the window in CHF/MWh, and ``index``.

        The column order of ``pv`` is the order of ``buses``, which is the
        order the battery table lists them in.  That is the aggregator order
        the environment indexes on, so the same order has to be used when the
        placement and the battery fleet are built, or an aggregator observes a
        neighbour's photovoltaic output.
    """
    load_mw, index = load_substation_demand(
        None, dataset=SWISS_DEMAND_DATASET, timezone=SWISS_TIMEZONE,
        aggregate=True, start=start, end=end,
        data_dir=data_dir, manifest_dir=manifest_dir)
    if buses is None:
        path = (Path(data_dir) if data_dir else _default_data_dir()) / SWISS_BESS_FILE
        if not path.exists():
            raise MissingSeries(f"the Swiss battery table is absent: {path}")
        fleet = pd.read_parquet(path)
        fleet = fleet[fleet["projection_year"] == projection_year]
        if fleet.empty:
            raise ValueError(
                f"projection year {projection_year} registers no battery; the "
                f"table carries "
                f"{sorted(pd.read_parquet(path)['projection_year'].unique())}")
        buses = [str(b) for b in fleet["osmid"]]
    pv = load_swiss_pv(index, buses, projection_year=projection_year,
                       rule=pv_rule, seed=pv_seed, data_dir=data_dir)
    price = load_elcom_tariff(index, category=tariff_category,
                              period=tariff_period, data_dir=data_dir)
    return LocalFlexSeries(load_mw=load_mw, pv=pv, energy_price=price,
                           index=index)
