# Written for this repository on 2026-08-15 -- no upstream counterpart.
"""gamma^v is 0 on feeder 459_0, and this is the check that replaces calibrating it.

`gamma^v` buys linearisation margin for a voltage constraint sitting *at* its
bound.  On this feeder no such constraint exists inside the usable load window,
so every value of it produces the same clearing and a calibration sweep would be
measuring nothing.  The market took the same route once before, at item 36: rather
than tune `compute_feasible_power_batch`, it asserted that clipping with it is a
no-op, turning a redundancy into a criterion.

**Both sides, because the two are worst at opposite ends.**  The available
argument scales load by kappa and finds the voltage family idle until kappa is
several times the usable window. That covers (VLO): more load, more drop. It does
not cover (VHI). The base case has no negative injection anywhere, but batteries
discharge, and injection raises voltage -- so (VHI) *loosens* as kappa grows and
its tightest point sits at **small kappa with deep discharge**, which a kappa
sweep never reaches. Measured here: at zero load with every battery at its rating
and every array at its peak, the feeder rises to 1.0269 p.u. at 2050, which is the
corner the one-sided argument misses. It is still nowhere near 1.1.

**Each injection term needs its scope stated, not just its name.** Discharge at the
battery buses; photovoltaic output at **every** photovoltaic bus, which is a larger
set (29 against 15 in 2030, 47 against 34 in 2050), so arrays on non-battery buses
inject while sitting outside any battery-indexed vector; charging at the battery
buses, which tightens (VLO) instead. Every miss in this round was a scope miss
rather than a missing name -- omitted entirely, then charging, then restricted to
the battery buses -- which is why the terms are written here with their index sets.
The lower corner deliberately has photovoltaic output at zero: that is the
night-time worst case.

**The upper corner sits at zero load.** Peak irradiance does not coincide with an
empty feeder, so it is not a reachable operating point -- but it is the corner of
the box, and a bound evaluated inside the box is not a bound. Starting at kappa
0.5 instead reads 0.010 p.u. too generous.

**Over the reachable box, not over sampled operating points.** "Voltage did not
bind in the runs we made" is weaker than "voltage cannot bind", and only the
second retires a parameter. The clearing chooses inside the box whose corners are
maximum load with everything charging and minimum load with everything
discharging, so bounding those bounds every clearing.

**The conclusion is conditional on the declared band, and that is the point of
recording the legacy number too.** Under EN 50160's +-10% the margin is 0.060 p.u.;
under the 0.95 floor previously assumed it is 0.010 p.u. -- a tenth of the band,
and the same order as the linearisation error gamma^v exists to cover. So
`gamma^v = 0` is a consequence of the band declaration, not an independent fact
about the feeder, and tightening the band would put it back in question.
"""
import numpy as np
import pytest

from powermarketjax.case.cases.distribution.case459_0 import (
    EN50160_MV_V_MAX, EN50160_MV_V_MIN, create_case459_0)
from powermarketjax.physics.bfs_power_flow import bfs_power_flow, prepare_bfs

#: Thermal binds here, from the published apparent-power utilisation.
KAPPA_THERMAL = 1.1166
#: Upper edge of the window the market is usable on.
KAPPA_MAX = 1.2
#: The floor assumed before EN 50160 was declared; kept to show the dependence.
LEGACY_V_MIN = 0.95

#: Measured 2026-08-15, GPU, x64 off (BFS is float32 internally). Recorded as
#: floors the assertions must clear, not as equalities: they move if the case,
#: the battery fleet or the band changes, and each of those should be visible.
#: The (VHI) floor is 0.070 against a measured 0.0731 at the tightest year; an
#: independent LinDistFlow closed form gives 0.0727 on the same corner, so the
#: two solvers agree to 4e-4 and the floor clears both.
MIN_SLACK_VLO = 0.060
MIN_SLACK_VHI = 0.070


@pytest.fixture(scope="module")
def case():
    return create_case459_0(node_v_min=EN50160_MV_V_MIN,
                            node_v_max=EN50160_MV_V_MAX)


@pytest.fixture
def x64():
    """Function-scoped on purpose, and restored on the way out.

    Only the clearing test below needs float64.  The floors at the top of this
    module were measured with x64 **off** because BFS is float32 internally, so
    turning it on for the whole module would move the numbers they were set
    against.  Until 2026-08-24 that test set the flag inline and never restored
    it, which left it on for whatever ran next in the same process.
    """
    import jax
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def battery_fleet(year):
    import pandas as pd
    from pathlib import Path
    import powermarketjax
    pq = Path(powermarketjax.__file__).resolve().parents[1] / \
        "powermarketjax/data/parquet"
    b = pd.read_parquet(pq / "SwissDN_459_0_MV_BESS.parquet")
    b = b[b["projection_year"] == year]
    nodes = pd.read_parquet(pq / "SwissDN_459_0_MV_Nodes.parquet")
    order = {str(o): i for i, o in enumerate(nodes["osmid"].astype(str))}
    idx = np.array([order[str(o)] for o in b["osmid"]], dtype=int)
    return idx, b["nominal_power_kw"].to_numpy(float) / 1000.0


def pv_peak_by_node(year, n_nodes):
    """Peak output per *bus*, over every bus with an array.

    Not `load_swiss_flex_series(...).pv`, whose columns are the battery buses:
    that indexing is right for the participant vector and wrong for a bound.
    """
    import pandas as pd
    from pathlib import Path
    import powermarketjax
    pq = Path(powermarketjax.__file__).resolve().parents[1] / \
        "powermarketjax/data/parquet"
    pv = pd.read_parquet(pq / "SwissDN_459_0_MV_PV_RepDays.parquet")
    pv = pv[pv["projection_year"] == year]
    nodes = pd.read_parquet(pq / "SwissDN_459_0_MV_Nodes.parquet")
    order = {str(o): i for i, o in enumerate(nodes["osmid"].astype(str))}
    out = np.zeros(n_nodes)
    for osmid, mw in (pv.groupby(pv["osmid"].astype(str))["pv_kw"].max()
                      / 1000.0).items():
        out[order[osmid]] = mw
    return out


def corner(case, kappa, idx, mw, *, discharging, pv_mw=None):
    p = np.asarray(case.node_pd, float) * kappa
    q = np.asarray(case.node_qd, float) * kappa
    p = p.copy()
    p[idx] += (-1.0 if discharging else 1.0) * mw
    if discharging and pv_mw is not None:
        p -= np.asarray(pv_mw, float)          # per node, not per battery bus
    res = bfs_power_flow(prepare_bfs(case), p_load_pu=p / case.base_mva,
                         q_load_pu=q / case.base_mva, v_slack=1.0)
    return np.asarray(res.v_mag, float)


@pytest.mark.parametrize("year", [2030, 2040, 2050])
def test_neither_voltage_limit_is_reachable_in_the_usable_window(case, year):
    """Both sides, each driven to its own worst corner."""
    idx, mw = battery_fleet(year)
    pv = pv_peak_by_node(year, case.n_nodes)
    slack_lo, slack_hi = [], []
    for kappa in (0.0, 0.5, 1.0, KAPPA_THERMAL, KAPPA_MAX):
        slack_lo.append(corner(case, kappa, idx, mw, discharging=False).min()
                        - EN50160_MV_V_MIN)
        slack_hi.append(EN50160_MV_V_MAX - corner(case, kappa, idx, mw,
                                                  discharging=True,
                                                  pv_mw=pv).max())
    assert min(slack_lo) > MIN_SLACK_VLO, f"(VLO) margin shrank: {min(slack_lo)}"
    assert min(slack_hi) > MIN_SLACK_VHI, f"(VHI) margin shrank: {min(slack_hi)}"


def test_the_upper_limit_corner_is_real_and_a_kappa_sweep_would_miss_it(case):
    """Discharge does lift the feeder above nominal, at *low* load.

    This is the half the load-scaling argument omits. If this ever stops holding
    -- if injection no longer raises voltage anywhere -- then the (VHI) reasoning
    above is about a corner that does not exist and should be re-derived.
    """
    idx, mw = battery_fleet(2050)
    pv = pv_peak_by_node(2050, case.n_nodes)
    high_at_light_load = corner(case, 0.0, idx, mw, discharging=True,
                                pv_mw=pv).max()
    high_at_heavy_load = corner(case, KAPPA_MAX, idx, mw, discharging=True,
                                pv_mw=pv).max()
    assert high_at_light_load > 1.0                      # injection does lift it
    assert high_at_light_load > high_at_heavy_load        # and (VHI) loosens with kappa


@pytest.mark.parametrize("year", [2030, 2050])
def test_the_margin_depends_on_the_declared_band(case, year):
    """Under the band this market previously assumed the margin is ~0.01 p.u.

    Asserted so that `gamma^v = 0` cannot be read as a property of the feeder
    alone: it follows from EN 50160's +-10%, and at the 0.95 floor the margin is
    a tenth of that -- the same order as the linearisation error gamma^v covers.
    """
    idx, mw = battery_fleet(year)
    worst = min(corner(case, k, idx, mw, discharging=False).min()
                for k in (1.0, KAPPA_THERMAL, KAPPA_MAX))
    assert worst - EN50160_MV_V_MIN > MIN_SLACK_VLO
    assert worst - LEGACY_V_MIN < 0.02


def test_the_clearing_itself_never_approaches_either_limit(x64):
    """The realised counterpart of the box bound, and not redundant with it.

    The box bound is a mathematical implication, and its premise is that the
    injection terms were listed completely -- which this round got wrong four
    times. This one does not depend on that premise: it runs the clearing and
    reads what the post-hoc sweep found, so a term missing from the list cannot
    make it pass. The two cover different failures. The box covers corners the
    clearing never visits; this covers terms the box never listed.

    **Vacuous on the adopted scenario, and recorded as vacuous rather than
    guarded**.  The sweep this reads is computed on every step
    whether or not the clearing dispatched anything, so on a market that never
    dispatches both depths are zero by absence and no defect in the voltage path
    could make them non-zero.  Measured 2026-08-24 at this module's own run
    point (`load_scale = KAPPA_THERMAL`, `episode_len` 8, the constant
    `baseline_action`, CPU, float64):

    - 0 of 128 periods over 16 drawn windows publish any requirement, and the
      cleared volume is 6.4e-15 MW, which is the solver's zero;
    - the state does advance -- cursor 4743 -> 4750 before auto-reset fires --
      so the eight hours are eight real hours and the market clears nothing in
      every one of them;
    - at `load_scale` 1.800, 21 of 128 periods (6 of 16 windows) do publish, and
      `worst_under == worst_over == 0.0` still holds there and at 2.0 and 2.4.

    So the voltage claim is **not known to be false; it is unexercised**, and
    nothing here is widened, weakened or added.  A non-triviality guard shaped
    like `test_swiss_env_l1.py`'s cannot be written either: `BASELINE_ACTION` is
    the saturation corner (-128, 128, -128), and a +-4 perturbation of any of its
    three columns moves the cleared volume in 0 of 16 windows at every scale
    from 1.1166 to 2.4, so "perturb an input, assert the output moves" has no
    input to perturb here.

    **The repair is a calibrated Swiss kappa, which does not exist yet.**
    The adopted kappa does not carry over to this
    configuration, and the Swiss value is unstandardised: the two are
    dimensionally incomparable because the published Swiss load curve peaks at
    0.852801 rather than 1, so a kappa calibrated on it carries a 1.1726 data
    factor.  That factor is also why `KAPPA_THERMAL` is the wrong quantity at
    this call site -- it is a registered-peak figure while `load_scale`
    multiplies the series, and 1.1166 x 1.1726 = 1.3093, which measures 0 of 128
    as well.  Calibrating it is new work and not this test's to invent.
    """
    pytest.skip(
        "vacuous on the adopted scenario: at this module's load_scale "
        "(KAPPA_THERMAL) the clearing publishes a requirement in 0 of 128 "
        "periods over 16 drawn windows and clears 6.4e-15 MW, so v_under and "
        "v_over are zero by absence rather than by slack -- the eight hours are "
        "real (cursor 4743->4750) and the market clears nothing in any of them. "
        "At load_scale 1.800, 21 of 128 periods (6 of 16 windows) do publish and "
        "worst_under == worst_over == 0.0 still holds, there and at 2.0 and 2.4, "
        "so the claim is unexercised rather than false. Nothing widened and no "
        "guard added; the repair is a calibrated Swiss kappa, which does not "
        "exist yet, and that is new work")
    import jax
    import pandas as pd
    from powermarketjax.envs.local_flexibility import (
        build_voltage_sensitivity, make_local_flex_env, make_local_flex_params)
    from powermarketjax.envs.local_flexibility.data import load_swiss_flex_series
    from powermarketjax.resources.battery import make_battery_bundle

    year = 2050
    c = create_case459_0(node_v_min=EN50160_MV_V_MIN, node_v_max=EN50160_MV_V_MAX)
    idx, mw = battery_fleet(year)
    series = load_swiss_flex_series(projection_year=year, tariff_category="C2",
                                    tariff_period=2026)
    reset, step, _, spec = make_local_flex_env(c, build_voltage_sensitivity(c),
                                               idx, period_hours=1.0)
    import powermarketjax
    from pathlib import Path
    pq = Path(powermarketjax.__file__).resolve().parents[1] / \
        "powermarketjax/data/parquet"
    bess = pd.read_parquet(pq / "SwissDN_459_0_MV_BESS.parquet")
    bess = bess[bess["projection_year"] == year]
    params = make_local_flex_params(
        load_series=np.asarray(series.load_mw, float),
        pv=np.asarray(series.pv, float),
        energy_price=np.asarray(series.energy_price, float),
        battery=make_battery_bundle(
            n_devices=len(idx), dt_hours=1.0,
            capacity_mwh=(bess["capacity_kwh"].to_numpy(float) / 1e3).tolist(),
            power_mw=mw.tolist()),
        cycle_cost=np.full(len(idx), 10.0), load_scale=KAPPA_THERMAL,
        learner_mask=np.ones(len(idx), bool), episode_len=8)

    key = jax.random.PRNGKey(0)
    _, state = reset(key, params)
    action = spec["baseline_action"](len(idx))
    worst_under = worst_over = 0.0
    for k in jax.random.split(key, 8):
        _, state, _, _, _, info = step(k, state, action, params)
        worst_under = max(worst_under, float(np.max(np.asarray(info["v_under"]))))
        worst_over = max(worst_over, float(np.max(np.asarray(info["v_over"]))))
    assert worst_under == 0.0 and worst_over == 0.0, (worst_under, worst_over)
