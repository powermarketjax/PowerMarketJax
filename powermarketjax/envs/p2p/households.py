"""The exogenous household series of the P2P market.

Setup-time pandas, run once per configuration.  It returns the two arrays
``make_p2p_params`` takes, on one 15-minute grid:

    series = load_fluvius_households(window=CEST_WINDOW, n_households=16)
    params = make_p2p_params(series.injection, series.offtake, battery,
                             kappa, learner_mask, episode_len)

**`p_pv` is the metered injection and `load` the metered offtake, and that is
a statement about the dataset rather than a convenience.**  A digital meter
has two registers and neither is gross photovoltaic output: the market needs
the net position, and at this meter the net position *is* the difference of
the two, measured rather than reconstructed.  Every market quantity depends
on the pair only through that difference -- `act_map` computes `net = p_pv +
p_dis - load - p_ch` and `baseline_action` reads `p_pv - load` -- so feeding
the two registers in is exact for the clearing, the settlement and the
battery.  What it changes is the meaning of two of the observation channels:
the agent sees its injection and its offtake, not its array output and its
consumption.  The price of the choice is that gross photovoltaic output is
not recoverable, so the array cannot be rescaled to a later vintage and
behind-the-meter self-consumption cannot be separated out.

**Windows are confined to one UTC offset and the loader derives them from the
data.**  The calendar encoding is computed from the row cursor rather than
from the clock, so the row index means local time of day only if the window
starts at a local midnight and holds one offset throughout; a window
spanning a daylight-saving transition rotates local time of day by an hour
against the phase for the rest of it, and no market quantity shows it.  A
local day is clean when it carries exactly 96 periods, the two transition
days are the only ones that do not, and a window is a maximal run of
consecutive clean local days; `single_offset_windows` returns them in the
order they occur.

**The grid is complete, which is why nothing here interpolates.**  A gap
would not be visible in a market quantity -- it would shift every later
period by one and rotate the calendar encoding -- so this module raises on
one instead of filling it.

**Household selection is a round robin over the groups, by identifier.**  The
households come in four publisher-defined groups -- photovoltaic array
alone, plus heat pump, plus home-charging electric vehicle, plus both -- and
taking the first `n` by identifier would take them all from one group,
because the identifier runs group by group.  Interleaving keeps any `n`
balanced across the groups without a seed, and `households=` overrides it
outright -- overriding which households are taken, not the order of the
household axis, which is ascending by identifier on every path.
"""
from pathlib import Path
from typing import NamedTuple, Optional, Sequence, Tuple

import json

import numpy as np
import pandas as pd

from powermarketjax.data.manifest import DatasetManifest
from powermarketjax.data.registry import DatasetRegistry

#: The household dataset: Flemish residential digital meters, calendar year
#: 2024, 15 minutes, the four groups that carry a photovoltaic array.
HOUSEHOLD_DATASET = "fluvius_dm_residential"

#: The two registers.  Neither name is `load.actual_mw`, which six manifests
#: in this repository already map: a signal string is not validated against
#: `data/signals.py`, so a dataset may register names of its own and stay
#: out of a collision the registry would otherwise resolve by index order.
INJECTION_SIGNAL = "load.injection_mw"
OFFTAKE_SIGNAL = "load.offtake_mw"

#: Period length of the household series, which fixes $\Delta$ = 0.25 h for
#: this market.  The period is not pinned by convention here; it follows the
#: data.
PERIOD_HOURS = 0.25
PERIODS_PER_DAY = 96

#: Settlement calendar of the source, used to find the local midnights.
HOUSEHOLD_TIMEZONE = "Europe/Brussels"


class MissingSeries(RuntimeError):
    """The household dataset is registered but its parquet is absent."""


class Window(NamedTuple):
    """A run of local days inside one UTC offset, as UTC bounds (inclusive)."""
    start: pd.Timestamp
    end: pd.Timestamp
    n_periods: int
    utc_offset: pd.Timedelta
    first_local_day: str
    last_local_day: str


class HouseholdSeries(NamedTuple):
    """The two registers as panels, plus the households and the grid."""
    injection: np.ndarray         # float32 (n_periods, N)  MW, fed in as p_pv
    offtake: np.ndarray           # float32 (n_periods, N)  MW, fed in as load
    households: np.ndarray        # int16   (N,)            publisher identifiers
    groups: Tuple[str, ...]       #         (N,)            publisher group names
    index: pd.DatetimeIndex       # UTC, regular, 15 minutes


def _default_data_dir() -> Path:
    """Where the registered parquet files live, when no ``data_dir`` is given.

    The ``parquet`` directory beside `powermarketjax.data`, resolved from the
    package rather than from the working directory, so the loader finds the file
    from anywhere.  Same helper and same location as the local-flexibility
    loader uses.
    """
    from powermarketjax import data as data_pkg
    return Path(data_pkg.__file__).resolve().parent / "parquet"


def _manifest(manifest_dir: Optional[Path]) -> DatasetManifest:
    """The registered manifest of `HOUSEHOLD_DATASET`, by name.

    ``manifest_dir`` selects a registry root, ``None`` the packaged one.  The
    dataset being absent is raised as `MissingSeries` rather than as the
    registry's ``KeyError``, since from here the two failures a caller can act
    on are "not registered" and "registered but the parquet is missing", and
    `_panel` raises the second one under the same type.
    """
    registry = DatasetRegistry(Path(manifest_dir) if manifest_dir else None)
    try:
        return registry.get_manifest(HOUSEHOLD_DATASET)
    except KeyError as exc:
        raise MissingSeries(
            f"dataset '{HOUSEHOLD_DATASET}' is not registered; no manifest "
            f"provides the household meter series this market needs") from exc


def _panel(manifest: DatasetManifest, data_dir: Optional[Path]
           ) -> Tuple[pd.DataFrame, dict]:
    """The parquet with canonical column names, plus its metadata sidecar.

    Read by dataset name and never by signal alone, which is the house rule the
    day-ahead and local-flexibility loaders both state: the signal index cannot
    express "this dataset and not the other one that maps the same name".

    The parquet is a long table, one row per (household, quarter hour), and the
    manifest's ``index_map`` and ``column_map`` rename it to the four columns
    everything below reads: ``datetime``, the UTC start of the 15-minute block;
    ``region``, which is the registry's name for the panel key and here carries
    the publisher's household identifier rather than a geographic region; and
    the two registers `INJECTION_SIGNAL` and `OFFTAKE_SIGNAL`, both in MW.  The
    sidecar is optional and is read only for its ``households`` entries, each
    pairing a household with its publisher group, which is what a group-aware
    selection needs.
    """
    root = Path(data_dir) if data_dir else _default_data_dir()
    path = root / manifest.parquet_file
    if not path.exists():
        raise MissingSeries(
            f"manifest '{manifest.name}' is registered but its parquet file is "
            f"absent: {path}")
    frame = pd.read_parquet(path)
    rename = {raw: canon
              for raw, canon in {**manifest.index_map, **manifest.column_map}.items()
              if raw in frame.columns}
    frame = frame.rename(columns=rename)
    missing = {INJECTION_SIGNAL, OFFTAKE_SIGNAL, "datetime", "region"} - set(frame.columns)
    if missing:
        raise MissingSeries(
            f"manifest '{manifest.name}' does not yield {sorted(missing)} after "
            f"renaming; columns are {list(frame.columns)}")
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)

    meta = {}
    if manifest.metadata_json:
        sidecar = root / manifest.metadata_json
        if sidecar.exists():
            meta = json.loads(sidecar.read_text())
    return frame, meta


def _clean_local_days(index: pd.DatetimeIndex) -> pd.Series:
    """Periods per local day; a clean day carries exactly `PERIODS_PER_DAY`."""
    local = index.tz_convert(HOUSEHOLD_TIMEZONE)
    return pd.Series(1, index=pd.Index(local.date, name="local_day")).groupby(
        level=0).sum()


def single_offset_windows(index: pd.DatetimeIndex) -> Tuple[Window, ...]:
    """Maximal runs of consecutive clean local days, as UTC bounds.

    A daylight-saving transition can only fall on a day that is not
    `PERIODS_PER_DAY` long, so a run of clean days holds one UTC offset by
    construction and no offset arithmetic is needed to find them.
    """
    per_day = _clean_local_days(index)
    clean = per_day == PERIODS_PER_DAY
    local_day = pd.Series(index.tz_convert(HOUSEHOLD_TIMEZONE).date, index=index)

    windows, run = [], []
    for day, is_clean in clean.items():
        if is_clean:
            run.append(day)
            continue
        if run:
            windows.append(run)
        run = []
    if run:
        windows.append(run)

    out = []
    for run in windows:
        stamps = index[local_day.isin(run).to_numpy()]
        out.append(Window(
            start=stamps.min(), end=stamps.max(), n_periods=len(stamps),
            utc_offset=stamps.min().tz_convert(HOUSEHOLD_TIMEZONE).utcoffset(),
            first_local_day=str(run[0]), last_local_day=str(run[-1])))
    return tuple(out)


def _select(households: pd.Index, groups_of: dict, groups: Optional[Sequence[str]],
            n_households: Optional[int], explicit: Optional[Sequence[int]]
            ) -> np.ndarray:
    """Which households to take: explicit list, or a round robin over groups."""
    if explicit is not None:
        chosen = np.asarray(explicit, dtype=np.int64)
        unknown = sorted(set(chosen) - set(households))
        if unknown:
            raise ValueError(f"households not in the dataset: {unknown}")
        # Sorted for the same reason the two branches below are.  The panel's
        # column order is the reshape's, and that is ascending by identifier
        # whatever order the caller wrote; returning the caller's order here
        # would label column j with a different household's identifier, and
        # `kappa`, `learner_mask` and the group names would all then be read
        # against a neighbour's series without any shape changing.
        return np.sort(chosen)

    available = [h for h in households if groups is None or groups_of[h] in groups]
    if groups is not None:
        unknown = set(groups) - set(groups_of.values())
        if unknown:
            raise ValueError(
                f"unknown groups {sorted(unknown)}; the dataset carries "
                f"{sorted(set(groups_of.values()))}")
    if n_households is None:
        return np.asarray(available, dtype=np.int64)
    if not 1 <= n_households <= len(available):
        raise ValueError(
            f"n_households must lie in [1, {len(available)}], got {n_households}")

    # Round robin over the groups in the order they first appear, so any n is
    # balanced across them; the identifier runs group by group, so taking the
    # first n outright would take them all from one group.
    by_group: dict = {}
    for h in available:
        by_group.setdefault(groups_of[h], []).append(h)
    order, cursor = [], 0
    while len(order) < n_households:
        for members in by_group.values():
            if cursor < len(members):
                order.append(members[cursor])
                if len(order) == n_households:
                    break
        cursor += 1
    return np.asarray(sorted(order), dtype=np.int64)


def load_fluvius_households(
    window: Optional[Tuple[str, str]] = None,
    n_households: Optional[int] = None,
    groups: Optional[Sequence[str]] = None,
    households: Optional[Sequence[int]] = None,
    data_dir: Optional[Path] = None,
    manifest_dir: Optional[Path] = None,
) -> HouseholdSeries:
    """Load the two registers as ``(n_periods, N)`` float32 panels in MW.

    ``window`` is a pair of UTC timestamps, inclusive; omit it to take the
    longest single-offset window `single_offset_windows` finds, which for 2024
    is the summer one.  ``groups`` restricts the publisher groups drawn from and
    ``n_households`` how many, ``households`` names them outright.

    Both panels carry the same two axes.  Axis 0 is time, one row per
    15-minute period, ascending and gap-free over the returned ``index``, and
    it is what the row cursor of `env` steps along one period at a time; the
    calendar encoding is derived from that cursor, which is why the window
    has to start at a local midnight and hold one UTC offset.  Axis 1 is the
    household, in ascending publisher-identifier order, which is what the
    reshape below produces from the sorted long table; ``households`` and
    ``groups`` label that axis in the same order, on every selection path.
    ``households=`` therefore chooses **which** households, not the order they
    appear in: a list given in any other order is sorted before it is used, so
    that column ``j`` and label ``j`` are the same household.  Both panels are in
    MW, each value the average power over its quarter hour rather than an
    energy, and `make_p2p_params` takes ``injection`` as ``p_pv`` and
    ``offtake`` as ``load``.
    """
    manifest = _manifest(manifest_dir)
    frame, meta = _panel(manifest, data_dir)

    groups_of = {int(h["household"]): h["group"] for h in meta.get("households", [])}
    ids = pd.Index(sorted(frame["region"].unique()))
    if groups is not None or n_households is not None:
        absent = set(ids) - set(groups_of)
        if absent:
            raise MissingSeries(
                f"the metadata sidecar does not name a group for "
                f"{len(absent)} households, so a group-aware selection cannot "
                f"be made; pass households= instead")
    chosen = _select(ids, groups_of, groups, n_households, households)

    frame = frame[frame["region"].isin(chosen)]
    grid = pd.DatetimeIndex(sorted(frame["datetime"].unique()))
    step = grid.to_series().diff().dropna().unique()
    if len(step) != 1 or step[0] != pd.Timedelta(hours=PERIOD_HOURS):
        raise ValueError(
            f"the household grid is not a uniform {PERIOD_HOURS} h step; the "
            f"differences present are {step}. A gap would shift every later "
            f"period by one and rotate the calendar encoding, "
            f"which no market quantity would show, so it is refused rather "
            f"than filled")

    if window is None:
        candidates = single_offset_windows(grid)
        if not candidates:
            raise ValueError("no window of whole clean local days exists")
        pick = max(candidates, key=lambda w: w.n_periods)
        lo, hi = pick.start, pick.end
    else:
        lo, hi = (pd.Timestamp(w, tz="UTC") for w in window)
        if lo not in grid or hi not in grid:
            raise ValueError(
                f"window bounds must be timestamps of the grid; it runs "
                f"{grid.min()} to {grid.max()} at {PERIOD_HOURS} h")
        local_lo = lo.tz_convert(HOUSEHOLD_TIMEZONE)
        if (local_lo.hour, local_lo.minute) != (0, 0):
            raise ValueError(
                f"a window must start at a local midnight so that the row "
                f"index means time of day; {lo} is "
                f"{local_lo:%H:%M} in {HOUSEHOLD_TIMEZONE}")
        span = grid[(grid >= lo) & (grid <= hi)]
        offsets = {s.tz_convert(HOUSEHOLD_TIMEZONE).utcoffset()
                   for s in (span.min(), span.max())}
        if len(offsets) != 1:
            raise ValueError(
                f"the window spans a daylight-saving transition, offsets "
                f"{sorted(str(o) for o in offsets)}: local time of day would "
                f"rotate by an hour against the row cursor. "
                f"single_offset_windows() lists the windows that do not")

    frame = frame[(frame["datetime"] >= lo) & (frame["datetime"] <= hi)]
    index = pd.DatetimeIndex(sorted(frame["datetime"].unique()))
    expected = len(index) * len(chosen)
    if len(frame) != expected:
        raise ValueError(
            f"the panel is incomplete: {len(frame)} rows for {len(index)} "
            f"periods and {len(chosen)} households, expected {expected}")

    frame = frame.sort_values(["datetime", "region"], kind="stable")
    shape = (len(index), len(chosen))
    return HouseholdSeries(
        injection=frame[INJECTION_SIGNAL].to_numpy(np.float32).reshape(shape),
        offtake=frame[OFFTAKE_SIGNAL].to_numpy(np.float32).reshape(shape),
        households=chosen.astype(np.int16),
        groups=tuple(groups_of.get(int(h), "") for h in chosen),
        index=index)
