"""Demand for the day-ahead market: one realised series, one forecast, per case.

Each loader returns the pair the Markov game needs, a realised demand that the
clearing is run against and a day-ahead forecast that the agent observes, as
``(n_days, 24)`` arrays of hourly megawatts, plus the dates::

    forecast, actual, days = load_gb_demand()      # case29gb
    forecast, actual, days = load_rts_demand()     # case73rts, net of renewables
    forecast, actual, days = load_nem_demand()     # case813nem, mainland

The three are not interchangeable and each carries its own pairing argument;
read the function that matches the case.  What they share is the interface and
one property: **the forecast/realisation gap is real data in all three**, never
noise injected on top of a single series.  Where a source could not supply that
honestly the loader says so rather than manufacturing it.

The GB pairing is documented immediately below because it came first.

**The pair is `load.tsd_mw` from NESO and `load.forecast_da_mw` from Elexon,
and that crossing of sources is deliberate.**  The `Actual` column shipped
alongside the forecast in `gb_forecast_actual_demand` is *not* its counterpart:
it is a gross measure that still contains distribution-connected generation,
tracking national demand plus embedded solar and wind rather than transmission
system demand, and its daily shape peaks several hours earlier than GB demand
actually does.  Elexon documents the comparable pairings as INDO with NDF and
ITSDO with TSDF, so a forecast of transmission system demand has to be scored
against transmission system demand, which is `load.tsd_mw`.

**Transmission system demand rather than national demand**, because `TSD`
includes station transformer load, pumped-storage pumping and interconnector
exports, all of which the transmission network of `case29gb` has to carry, and
serving those units with a gross series would ask them to supply load that
rooftop solar is already supplying.

Missing values are not an issue under this pairing: the two series lose no
overlapping half-hours in their common window, so the 2:1 aggregation to hourly
leaves no gaps and every day but the last -- truncated by the end of the
overlap rather than by a gap, and dropped -- carries all 24 hours.

`load.actual_mw` is **not** the name to reach for: three datasets map a column
onto it and the registry resolves it to AEMO, so asking for it returns 5-minute
Australian data with a `region` column and no error.
"""
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from powermarketjax.data.data_loader import DataLoader

#: Realised demand: transmission system demand, NESO, 30 minutes, 2009-2025.
ACTUAL_SIGNAL = "load.tsd_mw"
#: Day-ahead forecast: Elexon, 30 minutes, 2023-07 onwards.
FORECAST_SIGNAL = "load.forecast_da_mw"
#: Periods per market day; every loader here aggregates to this resolution.
T = 24


def load_gb_demand(
    data_dir: Optional[Path] = None,
    manifest_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Load the hourly forecast/realisation pair over the overlap of the two series.

    Returns ``(forecast, actual, days)`` where the arrays are ``(n_days, 24)``
    float32 megawatts and ``days`` are the UTC dates, so ``actual[d]`` is what
    the clearing of day ``d`` is run against and ``forecast[d]`` is what the
    observation shows the agent beforehand.  float32 because the `EnvState`
    arrays are float32; the solver widens what it needs.
    """
    loader = DataLoader(data_dir=data_dir, manifest_dir=manifest_dir)
    series = {}
    for name, signal in (("actual", ACTUAL_SIGNAL), ("forecast", FORECAST_SIGNAL)):
        df = loader.load_signals([signal])
        series[name] = df.set_index("datetime")[signal].astype(np.float64)

    pair = pd.concat(series, axis=1).loc[
        series["forecast"].index.min(): series["actual"].index.max()]
    hourly = pair.resample("1h").mean()          # 2:1, and it absorbs the 4 gaps

    # keep only whole market days: `transform` broadcasts each day's count of
    # complete hours back onto that day's rows, so the mask drops all 24 rows of a
    # day together and the reshape below is guaranteed to be rectangular
    complete = hourly.notna().all(axis=1).groupby(hourly.index.date).transform("sum")
    hourly = hourly[complete == T]
    days = pd.DatetimeIndex(sorted(set(hourly.index.date)))
    shape = (len(days), T)
    return (hourly["forecast"].to_numpy(np.float32).reshape(shape),
            hourly["actual"].to_numpy(np.float32).reshape(shape),
            days)


# ── case73rts: RTS-GMLC net load ─────────────────────────────────────────

#: The renewable classes RTS-GMLC gives as time series, all of which `case73rts`
#: excludes from its unit table and therefore nets off demand by default.
RTS_NETTED = ("hydro", "pv", "rtpv", "wind")

#: Signal-name stem of each renewable class, the part before the vintage suffix.
#: Shared with `envs.real_time.demand`, which appends a different suffix for the
#: 30-minute file: the mapping from class to stem is one fact and two spellings
#: of it would let the two resolutions net different things under one name.
RTS_CLASS_PREFIX = {"hydro": "hydro.rts", "pv": "solar.rts_pv",
                    "rtpv": "solar.rts_rtpv", "wind": "wind.rts"}


def load_rts_demand(
    netted: Tuple[str, ...] = RTS_NETTED,
    floor_mw: Optional[float] = None,
    data_dir: Optional[Path] = None,
    manifest_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    r"""Hourly net-load forecast/realisation pair for `case73rts`, calendar 2020.

    ``netted`` names which of ``hydro``, ``pv``, ``rtpv`` and ``wind`` to subtract.
    The default subtracts all four, which is what makes the case solvable: gross
    demand peaks at 8 192 MW against 8 076 MW of thermal capacity, and the two
    net series this returns peak at 6 019 MW (forecast) and 5 963 MW
    (realisation). `rts_gmlc_to_case` quotes 6 228 MW for the same thing, which
    is the day-ahead net peak *before* the rescaling below, not this output.

    **The day-ahead load is rescaled and the reason is measured, not stylistic.**
    RTS-GMLC's real-time load is its day-ahead load times a constant -- per region
    the ratio has a standard deviation of about 3e-5 -- so paired as shipped the
    realisation is below the forecast in 100% of hours and the market would face a
    one-signed error that is a units artefact rather than a forecast. Multiplying
    the day-ahead load by the measured ratio removes it, after which 53.6% of hours
    have the realisation above the forecast. **Which ratio matters and the two are
    a digit apart**: the code uses the energy ratio ``sum(rt) / sum(da)`` =
    0.9714674, the parquet sidecar's ``rescaling_factor``. The sidecar also carries
    ``system_ratio_mean`` = 0.9716535, the mean of the hourly ratios, and scaling
    by that one instead gives 50.6%. The rescaling touches the load term
    only; the renewable terms are paired as shipped, and since their two vintages
    genuinely differ they are where the forecast error of this pair comes from.
    Net of all four classes the residual has a median relative size of 11.6% with
    57.0% of hours positive -- larger than GB's 6.1%, and concentrated in wind,
    which is what a 45%-renewable system looks like.

    **393 hours of 2020 have negative net load** and nothing is removed by
    default. A thermal-only clearing has no over-generation slack, so the choice
    of what to do belongs to the run that makes it. ``floor_mw`` is that choice
    in its curtailment form: net demand is raised to the floor wherever it falls
    below, which is renewables being curtailed in the low-load hours. It has **no
    default** and is passed explicitly, as `cap_scale` and `ramp_scale` are, so
    that a run cannot inherit it silently.

    What it is for, measured over all 366 days of 2020 at ``ramp_scale = 1.0``
    by a full-year sweep, which is what this paragraph reports:
    with no floor the three-step clearing solves on about a quarter of days, and
    every failure has the committed minimum at the trough period exceeding
    trough demand. The floor has to be chosen jointly with ``cap_scale``, since
    the two failure modes are different -- too low a floor breaks the balance at
    the trough, too tight a ``cap_scale`` breaks the network under the must-run
    block alone. **The adopted demand floor is ``floor_mw = 2500`` with
    ``cap_scale = 0.424``** (2026-09-04;
    it supersedes ``2800``/``0.42``, which the same gates also pass but at a much
    higher floor).

    **It is no longer a pair: 2026-09-05 added ``p_min_scale = 0.80``** as a third
    scenario scale, because the *chained* commitment cannot be built at the
    registered minimum output at all -- a commitment sized for the day's peak
    cannot shut down to the trough.  The floor and
    ``cap_scale`` above did not move.

    The clearing-side numbers this docstring used to quote here -- 366 of 366 days
    clear, nothing shed above 1e-6 MWh, a line bound on every day (7 183
    line-periods), prices between −26.37 and 287.78 \$/MWh, worst ``mu`` 1.29e-11
    -- were measured by that full-year sweep, which clears each day from a
    cold start and therefore **never committed the three units with
    ``min_down_time = 48``** (676.0 MW, 18.1% of the fleet's ``sum p_min``).  They
    are not restated here, because the device that builds the product markets
    01/02/03 consume is the chained one, not the cold-start one that produced
    them.

    **The two knobs are not independent, which is why the floor could come down.**
    At ``cap_scale = 0.420`` zero shed needs ``floor_mw = 2800``; at ``0.424`` it
    needs only 2 500, because the floor fixes the trough balance while
    ``cap_scale`` fixes the network under the must-run block. The floor is at its
    measured minimum: ``2450``/``0.424`` sheds 150.9 MWh over two days and
    ``2400``/``0.424`` additionally leaves 2020-12-23 unsolved, so a 50 MW split
    brackets it. Going the other way, ``0.428`` leaves two days with no congestion
    at either floor -- that collapse is set by ``cap_scale``, not by the floor.

    It costs two things. 36.5% of the day-ahead renewable energy is curtailed at
    2 500 MW (46.1% at 2 800) -- and **that is a property of the rounding rule,
    not of the system**, since `round_commitment` commits every unit the
    relaxation touches and an exact commitment would need a far lower floor, so it
    must not be quoted as a curtailment result. The second cost is easier to miss:
    the floor is applied to both series, so in every hour where both are pinned to
    it the forecast equals the realisation exactly. At 2 500 MW that is **4 499 of
    8 784 hours, 51.2%** (5 348, 60.9%, at 2 800). Over the remaining hours the
    residual is 5.02% with 50.9% positive; over all hours the median is 0.00% with
    24.9% positive, because the median falls inside the pinned majority. Report
    both or neither.

    Returns:
        ``(forecast, actual, days)`` with the arrays ``(n_days, 24)`` float32 MW.
    """
    unknown = set(netted) - set(RTS_NETTED)
    if unknown:
        raise ValueError(f"not RTS-GMLC renewable classes: {sorted(unknown)}")

    loader = DataLoader(data_dir=data_dir, manifest_dir=manifest_dir)
    wanted = ["load.rts_da_mw", "load.rts_rt_mw"]
    for cls in netted:
        prefix = RTS_CLASS_PREFIX[cls]
        wanted += [f"{prefix}_da_mw", f"{prefix}_rt_mw"]
    df = loader.load_signals(wanted).set_index("datetime").astype(np.float64)

    scale = df["load.rts_rt_mw"].sum() / df["load.rts_da_mw"].sum()
    forecast = df["load.rts_da_mw"] * scale
    actual = df["load.rts_rt_mw"]
    for name in wanted[2:]:
        if name.endswith("_da_mw"):
            forecast = forecast - df[name]
        else:
            actual = actual - df[name]

    if floor_mw is not None:
        forecast = forecast.clip(lower=floor_mw)
        actual = actual.clip(lower=floor_mw)

    days = pd.DatetimeIndex(sorted(set(df.index.date)))
    shape = (len(days), T)
    return (forecast.to_numpy(np.float32).reshape(shape),
            actual.to_numpy(np.float32).reshape(shape),
            days)


# ── case813nem: mainland Australian NEM ──────────────────────────────────

#: The four mainland NEM regions. Tasmania is excluded because `case813nem` has
#: no Tasmanian buses -- Basslink is HVDC and the DC model cannot carry it.
NEM_MAINLAND = ("NSW1", "QLD1", "SA1", "VIC1")

#: Lead time of the AEMO forecast vintage taken as the day-ahead one, in hours.
#: The panel in this tree holds exactly one vintage and it is 24 hours ahead for
#: all 89 145 of its rows, so this is a check rather than a selection.
NEM_FORECAST_LEAD_H = 24.0


def nem_mainland_half_hours(
    data_dir: Optional[Path] = None,
    manifest_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """The mainland forecast/realisation pair at the panel's own 30 minutes.

    ``(forecast, actual)`` in megawatts, indexed by target time: the AEMO panel
    filtered to `NEM_MAINLAND` and to the one vintage it holds, pivoted, and
    summed across regions.  `load_nem_demand` averages this 2:1 for its hourly
    pair and `envs.real_time.demand.load_nem_demand_half_hourly` takes the
    realised column as it stands.

    **It is one function because the two markets have to see the same
    half-hours.**  `dropna` is applied across both columns, so a half-hour any
    mainland region is missing from either series is dropped from both markets;
    a second copy of that rule is how the real-time day set would come to differ
    from the day-ahead one that the position fixture's `day_index` indexes into.
    """
    loader = DataLoader(data_dir=data_dir, manifest_dir=manifest_dir)
    panel = loader.load_forecast_panel(
        ["load.forecast_p50_mw", "load.actual_mw"], source="aemo")
    panel = panel[panel["region"].isin(NEM_MAINLAND)]

    lead = (panel["target_time"] - panel["issue_time"]).dt.total_seconds() / 3600.0
    panel = panel[lead == NEM_FORECAST_LEAD_H]
    if panel.empty:
        raise ValueError(
            f"no rows at a {NEM_FORECAST_LEAD_H:g}-hour lead; the panel's vintage "
            f"has changed and the day-ahead one has to be re-identified")

    wide = panel.pivot_table(index="target_time", columns="region",
                             values=["load.forecast_p50_mw", "load.actual_mw"])
    wide = wide.dropna()
    return pd.DataFrame({
        "forecast": wide["load.forecast_p50_mw"].sum(axis=1),
        "actual": wide["load.actual_mw"].sum(axis=1),
    })


def load_nem_demand(
    floor_mw: Optional[float] = None,
    data_dir: Optional[Path] = None,
    manifest_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Hourly forecast/realisation pair for `case813nem`, mainland NEM.

    Both series come from one AEMO panel: ``OPERATIONAL_DEMAND_POE50`` issued 24
    hours before the target interval as the forecast, ``OPERATIONAL_DEMAND`` for
    the same interval as the realisation. That is a like-for-like pair by
    construction, which is what the GB pairing had to cross two sources to get,
    and 30 minutes is the native resolution of both, aggregated 2:1 to hourly here.

    Measured on the mainland total over the panel's 17 760 half-hours: median
    relative gap 1.41%, realisation above forecast in 67.0% of periods. The share
    is not the 50% an unbiased pair would show -- AEMO's POE50 sits below the
    realisation more often than not on this window -- and that is recorded rather
    than corrected, because a correction would be this repository's adjustment to
    an operator's published forecast.

    **Tasmania is excluded** from the total, matching the case. **The fleet is
    2016-17 and the demand is 2025**; `egrimod_nem_to_case` records what that
    costs.

    ``floor_mw`` raises both series where they fall below it, and here it is a
    symptom of that era mismatch rather than curtailment: the 151 scheduled units
    of 2016-17 have 10 725 MW of minimum output between them, while 2025 mainland
    operational demand -- already net of the rooftop PV that has been built since
    -- troughs below that on a handful of spring and late-summer days. The
    three-step clearing then has no feasible dispatch. It has **no default**, for
    the same reason `cap_scale` has none. Measured over the year at
    ``cap_scale = 1.0``, the days that fail all trough below about 11 000 MW, and
    ``floor_mw = 11500`` clears them at ``mu`` 3.7e-11 with no shed, measured by
    the corresponding full-year sweep. Unlike the
    RTS floor this one is cheap, because it binds on a few percent of days rather
    than on 61% of hours; the alternative is simply to drop those days from the
    scenario window and say which.

    Returns:
        ``(forecast, actual, days)`` with the arrays ``(n_days, 24)`` float32 MW.
    """
    hourly = nem_mainland_half_hours(data_dir, manifest_dir).resample("1h").mean()
    if floor_mw is not None:
        hourly = hourly.clip(lower=floor_mw)

    complete = hourly.notna().all(axis=1).groupby(hourly.index.date).transform("sum")
    hourly = hourly[complete == T]
    days = pd.DatetimeIndex(sorted(set(hourly.index.date)))
    shape = (len(days), T)
    return (hourly["forecast"].to_numpy(np.float32).reshape(shape),
            hourly["actual"].to_numpy(np.float32).reshape(shape),
            days)


# ── which demand goes with which case ────────────────────────────────────

#: The loader each case is defined against, and the pairing arguments that
#: loader will not run without.  A driver reads its case at runtime -- from a
#: fixture's ``meta["case"]`` or from its own ``--case`` -- so the demand has to
#: be read from the same place; naming one loader in the source pins the demand
#: while the case moves.  **That mismatch is silent**: serving `case73rts`, 73
#: units with 8 076 MW of thermal capacity, against GB transmission demand of
#: 16 679 to 47 494 MW raises nothing, it just sheds.  A case absent from this
#: table therefore raises rather than falling back to the GB pair.
CASE_DEMAND = {
    "29gb": (load_gb_demand, ()),
    "73rts": (load_rts_demand, ("netted", "floor_mw")),
    "813nem": (load_nem_demand, ("floor_mw",)),
}

#: The one case whose products predate the demand stamp, and so the one case a
#: fixture may omit `demand_source` / `demand_kwargs` for.  Every commitment
#: fixture in `tests/fixtures/` is a `29gb` one carrying 16 to 22 meta keys, none
#: of them about demand; refusing those would refuse every product this
#: repository has built.  The branch is scoped to this case alone and says so on
#: stdout when it fires, so a run that took it is visible in its own log.
LEGACY_DEMAND_CASE = "29gb"


def demand_pairing(case: str, floor_mw=None, netted=None) -> dict:
    """The pairing kwargs a driver states on the command line, as a dict.

    The one place that knows what "flag not given" means, so that the ten
    `--case` drivers and `precommit.py` cannot drift apart on it.  Argparse
    stays in the drivers; this takes values, not a parser, because
    `powermarketjax/` is a library.

    A flag left off is *not* a default: `load_rts_demand` and `load_nem_demand`
    give ``floor_mw`` none on purpose, so omitting it here omits the key and
    `demand_meta` refuses the case that needs one.  ``netted`` reads as an
    exception and is not one -- `load_rts_demand` does have a default for it,
    and this mirrors that default rather than adding a second one, so what ran
    is still what a record says ran.
    """
    given = {}
    if floor_mw is not None:
        given["floor_mw"] = floor_mw
    if netted is not None:
        given["netted"] = list(netted)
    elif case in CASE_DEMAND and "netted" in CASE_DEMAND[case][1]:
        given["netted"] = list(RTS_NETTED)
    return given


def demand_meta(case: str, **kwargs) -> dict:
    """The demand pairing, as the two meta keys whatever is built on it records.

    ``kwargs`` are the pairing arguments of the loader `case` selects and must
    be exactly them: `load_rts_demand` and `load_nem_demand` give ``floor_mw``
    no default deliberately, so that a run cannot inherit one silently, and a
    record that omits it cannot be turned back into the series it describes.
    ``netted`` is normalised to a list because the record is stored as JSON and
    comes back as one, and the writer's output has to equal the reader's input.

    Returns ``{"demand_source": ..., "demand_kwargs": {...}}``.
    `demand_source` is the loader's name and is provenance: it records which
    function ran, so that editing `CASE_DEMAND` later makes the fixtures written
    under the old map refuse to load rather than resolve to a different series.
    """
    if case not in CASE_DEMAND:
        raise ValueError(
            f"no demand series is defined for case {case!r}; add it to "
            f"`CASE_DEMAND` with its pairing arguments. It is not paired with "
            f"GB demand by default, which is the whole point of this table")
    loader, required = CASE_DEMAND[case]
    if set(kwargs) != set(required):
        raise ValueError(
            f"case {case!r} takes its demand from {loader.__name__}, whose "
            f"pairing arguments are {sorted(required)}; got {sorted(kwargs)}")
    out = {k: (list(v) if isinstance(v, tuple) else v) for k, v in kwargs.items()}
    return {"demand_source": loader.__name__, "demand_kwargs": out}


def demand_kwargs_from_meta(meta: dict) -> Tuple[dict, bool]:
    """``(kwargs, recorded)``: the pairing arguments a fixture's `meta` states.

    Dispatches on ``meta["case"]`` and validates what the fixture recorded under
    ``meta["demand_source"]`` / ``meta["demand_kwargs"]`` against `CASE_DEMAND`,
    so a record cannot resolve to a series other than the one it names.  A
    `29gb` meta with no demand keys is the pre-stamp fixtures and comes back as
    ``({}, False)``; any other case without them raises, because what it was
    built against cannot be reconstructed and guessing is the failure this
    check exists to stop.

    ``recorded`` is returned rather than left for the caller to re-derive: it is
    the same question this function already answered, and the callers use it
    only to say in their own log which branch they took.

    **It is separate from `demand_from_meta` because two markets read one
    record.**  The real-time market needs the same pairing at 48 periods a day
    (`envs.real_time.demand.realised_from_meta`), and a fixture carries one
    demand stamp, not one per market; duplicating the validation there is how
    the two would come to disagree about what a record means.
    """
    case = meta["case"]
    if case not in CASE_DEMAND:
        raise ValueError(
            f"fixture is for case {case!r}, which `CASE_DEMAND` has no demand "
            f"series for; it is not paired with GB demand by default")
    loader, required = CASE_DEMAND[case]

    if "demand_kwargs" not in meta:
        if case != LEGACY_DEMAND_CASE:
            raise KeyError(
                f"fixture for case {case!r} records no demand_source / "
                f"demand_kwargs, so which series it was built against cannot be "
                f"read off it; rebuild it with a `tools/commitment/precommit.py` "
                f"that stamps them. The pre-stamp fallback covers "
                f"{LEGACY_DEMAND_CASE!r} only")
        return {}, False

    source = meta.get("demand_source")
    if source != loader.__name__:
        raise ValueError(
            f"fixture for case {case!r} records demand_source={source!r} but "
            f"`CASE_DEMAND` pairs that case with {loader.__name__}; the map "
            f"moved under a product already written")
    kwargs = dict(meta["demand_kwargs"])
    if set(kwargs) != set(required):
        raise ValueError(
            f"fixture for case {case!r} records demand_kwargs {sorted(kwargs)}, "
            f"but {loader.__name__} pairs on {sorted(required)}")
    return kwargs, True


def demand_from_meta(
    meta: dict,
    data_dir: Optional[Path] = None,
    manifest_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """The demand pair a fixture's `meta` describes: ``(forecast, actual, days)``.

    The record is read by `demand_kwargs_from_meta`, so the series is the one
    the fixture was built against rather than the one the caller happened to
    name.  The loader in force is printed rather than assumed, so that the value
    that ran is in the run's own log.
    """
    kwargs, recorded = demand_kwargs_from_meta(meta)
    loader, _required = CASE_DEMAND[meta["case"]]
    if not recorded:
        print(f"demand: {loader.__name__}() -- {LEGACY_DEMAND_CASE} fixture "
              f"predating the demand stamp, no recorded pairing", flush=True)
        return loader(data_dir=data_dir, manifest_dir=manifest_dir)
    shown = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    print(f"demand: {loader.__name__}({shown})", flush=True)
    return loader(**kwargs, data_dir=data_dir, manifest_dir=manifest_dir)


def demand_for_case(
    case: str,
    data_dir: Optional[Path] = None,
    manifest_dir: Optional[Path] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """`demand_from_meta` for a driver that names its case directly.

    Scripts that take a ``--case`` have no fixture meta to read the pairing off,
    so they state it here; the record `demand_meta` builds is what they go
    through, so a driver and a fixture cannot disagree about what a case pairs
    with.
    """
    return demand_from_meta(dict(demand_meta(case, **kwargs), case=case),
                            data_dir=data_dir, manifest_dir=manifest_dir)
