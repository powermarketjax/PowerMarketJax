"""L0 and L1 for the P2P household loader (§15).

Two fixtures, for two different jobs.  `tiny` is a synthetic four-household
panel on the real 2024 Flemish grid, written to a temporary directory with its
own manifest: it exercises the window arithmetic, the selection rule and every
refusal without paying for a 42-million-row read each time, and it doubles as
the test that `data_dir`/`manifest_dir` are honoured.  `real` is the registered
dataset, and it carries the assertions that are about the data rather than about
the code -- above all that what the loader hands to `make_p2p_params` has
`p_pv - load` equal to the net position of §3.1.

**The window arithmetic is the load-bearing part.**  The environment computes the
calendar encoding from the row cursor, so a window that starts off a local
midnight or spans a daylight-saving transition silently rotates local time of
day against the phase, and no market quantity shows it.  Both are refused, and
the refusals are asserted rather than assumed.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from powermarketjax.envs.p2p.households import (
    HOUSEHOLD_TIMEZONE, PERIODS_PER_DAY, PERIOD_HOURS, MissingSeries,
    load_fluvius_households, single_offset_windows)

GROUPS = ("enkel_ZP", "WP_met_ZP", "EV_met_ZP", "WP_EV_met_ZP")
#: The three windows of 2024, measured: the two transition days are the only
#: local days that are not 96 periods long, so they split the year into three.
EXPECTED_WINDOWS = (
    ("2024-01-01", "2024-03-30", 8_640, pd.Timedelta(hours=1)),
    ("2024-04-01", "2024-10-26", 20_064, pd.Timedelta(hours=2)),
    ("2024-10-28", "2024-12-31", 6_240, pd.Timedelta(hours=1)),
)
N_TIMESTAMPS = 35_136


def _grid() -> pd.DatetimeIndex:
    """The UTC grid a complete local year of quarter hours implies."""
    start = pd.Timestamp("2024-01-01 00:00", tz=HOUSEHOLD_TIMEZONE).tz_convert("UTC")
    return pd.date_range(start, periods=N_TIMESTAMPS, freq="15min", tz="UTC")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A four-household synthetic panel with its own manifest, one per group."""
    root = tmp_path_factory.mktemp("fluvius_tiny")
    grid = _grid()
    # Three per group, at the foot of each group's 300-block, so that "the
    # first n by identifier" and "a round robin over the groups" disagree --
    # which is the whole reason the selection rule is not the former.
    ids = [1, 2, 3, 601, 602, 603, 1201, 1202, 1203, 1801, 1802, 1803]
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "household": np.tile(np.asarray(ids, np.int16), len(grid)),
        "interval_start": np.repeat(grid.to_numpy(), len(ids)),
        "offtake_mw": rng.uniform(0, 5e-3, len(grid) * len(ids)).astype(np.float32),
        "injection_mw": rng.uniform(0, 5e-3, len(grid) * len(ids)).astype(np.float32),
    })
    frame.to_parquet(root / "tiny.parquet", index=False)
    (root / "tiny.json").write_text(json.dumps({
        "households": [{"household": h, "group": GROUPS[i // 3], "pv": 1}
                       for i, h in enumerate(ids)]}))

    manifests = root / "manifests"
    manifests.mkdir()
    (manifests / "fluvius_dm_residential.json").write_text(json.dumps({
        "name": "fluvius_dm_residential", "source": "fluvius",
        "data_type": "actual_series", "time_mode": "calendar",
        "resolution": "15min", "parquet_file": "tiny.parquet",
        "column_map": {"offtake_mw": "load.offtake_mw",
                       "injection_mw": "load.injection_mw"},
        "index_map": {"interval_start": "datetime", "household": "region"},
        "metadata_json": "tiny.json", "timezone": HOUSEHOLD_TIMEZONE}))
    return {"data_dir": root, "manifest_dir": manifests}


@pytest.fixture(scope="module")
def real():
    return load_fluvius_households(n_households=8)


# --------------------------------------------------------------- L0: windows

def test_windows_are_derived_from_the_data_not_hard_coded():
    windows = single_offset_windows(_grid())
    got = tuple((w.first_local_day, w.last_local_day, w.n_periods, w.utc_offset)
                for w in windows)
    assert got == EXPECTED_WINDOWS
    # The three cover the year less the two transition days, which is the whole
    # cost of the rule: 364 local days out of 366.
    assert sum(w.n_periods for w in windows) == 364 * PERIODS_PER_DAY
    assert all(w.n_periods % PERIODS_PER_DAY == 0 for w in windows)


def test_default_window_is_the_longest(tiny):
    series = load_fluvius_households(**tiny)
    assert len(series.index) == 20_064
    local = series.index.min().tz_convert(HOUSEHOLD_TIMEZONE)
    assert (local.hour, local.minute) == (0, 0)
    assert str(local.date()) == "2024-04-01"


def test_a_window_spanning_a_transition_is_refused(tiny):
    # Starts at a local midnight, ends after the spring transition: legal as a
    # UTC interval, illegal as a row-cursor calendar.
    with pytest.raises(ValueError, match="daylight-saving transition"):
        load_fluvius_households(
            window=("2024-03-29 23:00", "2024-04-02 22:00"), **tiny)


def test_a_window_not_starting_at_local_midnight_is_refused(tiny):
    with pytest.raises(ValueError, match="local midnight"):
        load_fluvius_households(
            window=("2024-04-01 00:00", "2024-04-02 21:45"), **tiny)


def test_window_bounds_must_lie_on_the_grid(tiny):
    with pytest.raises(ValueError, match="timestamps of the grid"):
        load_fluvius_households(
            window=("2024-04-01 00:07", "2024-04-02 21:45"), **tiny)


# ------------------------------------------------------------- L0: selection

@pytest.mark.parametrize("n", [1, 2, 4, 3])
def test_selection_is_balanced_across_groups(tiny, n):
    series = load_fluvius_households(n_households=n, **tiny)
    assert len(series.households) == n
    counts = pd.Series(series.groups).value_counts()
    # Round robin: no group may be ahead of another by more than one.
    assert counts.max() - counts.min() <= 1 if len(counts) > 1 else True
    assert series.injection.shape == (20_064, n)


def test_taking_the_first_n_by_identifier_would_not_be_balanced(tiny):
    """The reason the round robin exists, measured rather than asserted about."""
    everything = load_fluvius_households(**tiny)
    of = dict(zip(everything.households.tolist(), everything.groups))

    chosen = load_fluvius_households(n_households=3, **tiny)
    assert len({of[h] for h in chosen.households.tolist()}) == 3

    naive = sorted(of)[:3]                     # what "first n by id" would give
    assert len({of[h] for h in naive}) == 1


def test_explicit_households_override_the_rule(tiny):
    """``households=`` chooses which households, and the labels follow axis 1.

    The list is given in descending order on purpose, and the assertions are
    made without ``sorted``: a ``sorted(series.households)`` passes whether or
    not the labels agree with the columns, and the failure it lets through is
    silent.  The panel is reshaped from a table sorted by identifier, so column
    ``j`` is the ``j``-th household in ascending order; `kappa`, `learner_mask`
    and the group names are all indexed by that column position, so a label out
    of step with it attaches every one of them to a neighbour's series with no
    shape and no value ever looking wrong.  Hence the columns are checked
    against the source table rather than against each other.
    """
    series = load_fluvius_households(households=[1801, 1], **tiny)
    assert series.households.tolist() == [1, 1801]
    assert series.groups == ("enkel_ZP", "WP_EV_met_ZP")

    # the two columns must differ, or the alignment below is vacuous
    assert not np.array_equal(series.injection[:, 0], series.injection[:, 1])

    frame = pd.read_parquet(Path(tiny["data_dir"]) / "tiny.parquet")
    frame = frame[(frame["interval_start"] >= series.index.min())
                  & (frame["interval_start"] <= series.index.max())]
    for j, h in enumerate(series.households.tolist()):
        own = frame[frame["household"] == h].sort_values("interval_start",
                                                         kind="stable")
        np.testing.assert_array_equal(
            series.injection[:, j],
            own["injection_mw"].to_numpy(np.float32), err_msg=f"injection {h}")
        np.testing.assert_array_equal(
            series.offtake[:, j],
            own["offtake_mw"].to_numpy(np.float32), err_msg=f"offtake {h}")


def test_groups_restrict_the_pool(tiny):
    series = load_fluvius_households(groups=["enkel_ZP"], **tiny)
    assert set(series.groups) == {"enkel_ZP"}


def test_unknown_group_and_bad_count_are_refused(tiny):
    with pytest.raises(ValueError, match="unknown groups"):
        load_fluvius_households(groups=["no_such_group"], **tiny)
    with pytest.raises(ValueError, match="n_households must lie"):
        load_fluvius_households(n_households=99, **tiny)
    with pytest.raises(ValueError, match="not in the dataset"):
        load_fluvius_households(households=[999_999], **tiny)


def test_an_absent_parquet_says_so(tiny, tmp_path):
    with pytest.raises(MissingSeries, match="parquet file is absent"):
        load_fluvius_households(data_dir=tmp_path,
                                manifest_dir=tiny["manifest_dir"])


# ------------------------------------------------------ L1: what the data is

def test_shapes_dtypes_and_grid(real):
    assert real.injection.shape == real.offtake.shape == (20_064, 8)
    assert real.injection.dtype == real.offtake.dtype == np.float32
    assert len(real.households) == len(real.groups) == 8
    steps = real.index.to_series().diff().dropna().unique()
    assert list(steps) == [pd.Timedelta(hours=PERIOD_HOURS)]


def test_both_registers_are_non_negative(real):
    """They are meter registers, not a signed net position."""
    assert (real.injection >= 0.0).all()
    assert (real.offtake >= 0.0).all()


def test_the_difference_is_the_net_position_of_the_parquet(real):
    """`p_pv - load` must be the metered net, not a reconstruction of it."""
    root = Path("powermarketjax/data/parquet")
    frame = pd.read_parquet(root / "Fluvius_DM_Residential_2024_15min.parquet")
    frame = frame[frame["household"].isin(real.households)]
    frame = frame[(frame["interval_start"] >= real.index.min())
                  & (frame["interval_start"] <= real.index.max())]
    frame = frame.sort_values(["interval_start", "household"], kind="stable")
    expected = (frame["injection_mw"].to_numpy(np.float32)
                - frame["offtake_mw"].to_numpy(np.float32)
                ).reshape(real.injection.shape)
    np.testing.assert_array_equal(real.injection - real.offtake, expected)


def test_every_household_taken_carries_an_array(real):
    assert set(real.groups) <= set(GROUPS)
    assert all(g for g in real.groups)


def test_it_feeds_make_p2p_params_and_the_net_survives(real):
    """The end of the chain: the arrays go in and §3.1's net comes out."""
    import math

    from powermarketjax.envs.p2p.env import make_p2p_params
    from powermarketjax.resources.battery import make_battery_bundle

    n = real.injection.shape[1]
    oneway = math.sqrt(0.85)
    bundle = make_battery_bundle(
        n_devices=n, capacity_mwh=0.011, power_mw=0.011 / 2.1,
        eta_charge=oneway, eta_discharge=oneway, soc_min=0.15, soc_max=1.0,
        initial_soc=0.5, dt_hours=PERIOD_HOURS, cycle_cost_per_mwh=0.0)
    params = make_p2p_params(
        p_pv=real.injection, load=real.offtake, battery=bundle,
        kappa=np.zeros(n, np.float32), learner_mask=np.ones(n, bool),
        episode_len=PERIODS_PER_DAY)
    np.testing.assert_array_equal(
        np.asarray(params.p_pv) - np.asarray(params.load),
        real.injection - real.offtake)
    assert int(params.episode_len) == PERIODS_PER_DAY


def test_the_calendar_phase_equals_local_time_of_day(real):
    """What the window rule is for, checked end to end.

    The environment computes the calendar encoding from `cursor` rather than from a
    clock, so it means local time of day only if the series starts at a local
    midnight and holds one UTC offset -- which is what `single_offset_windows`
    exists to guarantee.  Nothing in a market quantity would reveal a rotated
    phase, so the check has to be against the timestamps the loader returned.
    """
    import math

    import jax
    import jax.numpy as jnp

    from powermarketjax.envs.p2p.env import make_p2p_env, make_p2p_params
    from powermarketjax.resources.battery import make_battery_bundle

    n = real.injection.shape[1]
    oneway = math.sqrt(0.85)
    bundle = make_battery_bundle(
        n_devices=n, capacity_mwh=0.011, power_mw=0.011 / 2.1,
        eta_charge=oneway, eta_discharge=oneway, soc_min=0.15, soc_max=1.0,
        initial_soc=0.5, dt_hours=PERIOD_HOURS, cycle_cost_per_mwh=0.0)
    params = make_p2p_params(
        p_pv=real.injection, load=real.offtake, battery=bundle,
        kappa=np.zeros(n, np.float32), learner_mask=np.ones(n, bool),
        episode_len=PERIODS_PER_DAY)
    reset, _, _, spec = make_p2p_env(n, 73.0, 394.3, PERIOD_HOURS)
    get_obs = spec["get_obs"]
    _, state = reset(jax.random.PRNGKey(0), params)

    # Four cursors spread over one local day, plus one deep into the window so
    # that a phase that drifted with the row index would show.
    for cursor in (0, 24, 48, 72, PERIODS_PER_DAY * 137 + 33):
        obs = np.asarray(get_obs(
            state.replace(cursor=jnp.asarray(cursor, jnp.int32)), params))
        local = real.index[cursor].tz_convert(HOUSEHOLD_TIMEZONE)
        fraction = (local.hour * 60 + local.minute) / (24 * 60)
        assert obs[0, -2] == pytest.approx(math.sin(2 * math.pi * fraction), abs=1e-5)
        assert obs[0, -1] == pytest.approx(math.cos(2 * math.pi * fraction), abs=1e-5)
