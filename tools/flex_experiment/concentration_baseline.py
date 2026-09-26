"""Local flexibility market: learning baselines across three battery fleets.

Reproduce with, from the repository root:

    PYTHONPATH=tools/flex_experiment python -m concentration_baseline --seeds 5

**What the treatment is.**  The SwissDN dataset publishes the battery fleet of
feeder `459_0` for three projection years, and a year fixes the population, the
placement and the rated power together, so those three cannot be told apart from
the published snapshots alone.  This experiment therefore runs a two-by-two:
each cell is a placement crossed with a fleet total, the diagonal is the data as
published, and each off-diagonal cell scales every battery of one placement to
the other year's total.  A comparison down a column moves only the megawatts,
with the buses, the count and the photovoltaic array held fixed.

**The 2030 fleet is excluded on measurement.**  At every scaling factor where a
requirement arises at all, between 62% and 76% of the periods carrying one end
in curtailment, so it never reaches the state the mechanism is about.
Curtailment is the feasibility backstop of §6 and cannot be removed, but a
configuration in which it is the usual outcome is a curtailment scheme with a
market attached, and reporting one would normalise it.

**Five arms, of which one learns.**  The truthful arm is not a strength-matched
baseline and must not be reported as one.  `never_clears` marks
up past the value of lost load so nothing it offers is accepted; it is the
discriminating check, and an ordering among the others means nothing unless they
beat it.  `incdec` offers at cost while planning the largest charge the envelope
allows, which is the behaviour §8 leaves in place; it was written as a poor
policy and beat the learner in every cell of the first run, so it is reported as
its own arm.

**Two learners, selected by `--algo`.**  `ippo` is the default and the path
every result on disk was produced on: PPO written out below, not
`powermarketjax/learning/ippo.py`.  `sac` runs the package's SAC
(`powermarketjax/learning/sac.py`) through **this** market's rollout, for
the benchmark's two SAC columns; the section headed SAC says what it imports, what it
had to decide for itself, and on which two points 05's SAC column may not be
read across a row against another market's.  Nothing outside that section and
the `--algo` branches in `main` is reached without the flag, so a run without it
writes the file it always wrote, byte for byte.

**What is declared and travels with any result**: the placement and
capacity years, the scaling factor, the two safety margins, the tariff category
and year, the photovoltaic month-expansion rule, and the degradation cost.  The
last has no source in any dataset of either jurisdiction.  The efficiencies are
**not** in that list -- they are published per device and are passed through.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

from powermarketjax.case.cases.distribution.case459_0 import (  # noqa: E402
    EN50160_MV_V_MAX, EN50160_MV_V_MIN, create_case459_0)
from powermarketjax.envs.local_flexibility import make_local_flex_env  # noqa: E402
from powermarketjax.envs.local_flexibility.data import (  # noqa: E402
    SWISS_BESS_FILE, _default_data_dir, load_swiss_flex_series)
from powermarketjax.envs.local_flexibility.env import (  # noqa: E402
    baseline_action, make_local_flex_params)
from powermarketjax.envs.local_flexibility.sensitivity import \
    build_voltage_sensitivity  # noqa: E402
from powermarketjax.resources.battery import make_battery_bundle  # noqa: E402

# ---------------------------------------------------------------- scenario ---
#: The two fleets that function as a market on this feeder.  The 2030 fleet is
#: excluded on measurement rather than on preference: at every scaling factor
#: where a requirement arises at all, between 62% and 76% of the periods that
#: carry one end in curtailment, so it never reaches the state the mechanism is
#: about.  Curtailment is the feasibility backstop of §6 and cannot be removed,
#: but a configuration in which it is the usual outcome is a curtailment scheme
#: with a market attached, and reporting one would normalise it.
YEARS = (2040, 2050)

#: The four cells that separate what the three published snapshots confound.
#: Each is (placement year, capacity year): the diagonal is the data as
#: published, and the two off-diagonal cells hold one factor while moving the
#: other by scaling every battery of that placement to the other year's fleet
#: total.  A comparison down a column identifies capacity with placement, count
#: and co-located photovoltaic capacity all held fixed, which is the contrast
#: three co-varying snapshots cannot provide.
CELLS = ((2040, 2040), (2040, 2050), (2050, 2040), (2050, 2050))
#: Calibrated against the reliability standard that applies to the quantity
#: `clearing.py` prices, by `kappa_scan.py`, 2026-08-17.  Shedding enters the
#: clearing objective at ``VOLL = 10 000 CHF/MWh``, i.e. as involuntary
#: interruption of firm demand, so the anchor is loss-of-load expectation --
#: **3 h/year** across most European systems, one hour in Sweden and fifteen in
#: the Czech Republic -- and **not** the curtailment allowance of a non-firm or
#: flexible connection, which runs to thousands of hours and applies to a party
#: that accepted non-firm access in exchange for earlier connection.  The feeder
#: is Swiss, where ElCom measured a 2022 SAIDI of 7 minutes per end user.
#:
#: **Two conditions, and both are read off `curtail_periods`.**  A scaling has
#: to leave a problem worth solving -- with no market the feeder must shed far
#: past the standard -- and it has to be solvable by a simple arm that is not
#: at a corner of the action space, because an operating point whose only
#: feasible policy is extreme is one no learner reaches, and then a learner that
#: sheds is reporting the operating point rather than its own failure.
#:
#: At 1.50, measured at 400 episodes (9600 periods, resolution 0.91 h/year):
#: no market sheds **380-405 h/year** (127-135x LOLE) across the four cells, and
#: the arm that bids at cost, offers its whole capability and charges at **half
#: the headroom** sheds in **0 of 9600 periods in all four**, which bounds it
#: below **2.7 h/year at 95%**.  That arm's charging component is
#: ``alpha_ch = 0``, the exact centre of the action space.
#:
#: **The resolution is part of the criterion, and getting that wrong is what
#: moved this constant twice in one day.**  `curtail_periods` is a count over
#: ``n_eval * EPISODE_LEN`` periods, so the smallest non-zero rate it can
#: express is ``8760 / (n * 24)``: 5.7 h/year at 64 episodes, 10.1 at 36.  Both
#: are coarser than the 3 h/year the standard names, so a "0.0 h/year" measured
#: at 64 episodes says only "no event in 1536 periods" and bounds the rate below
#: roughly 17 h/year -- not below 3.  1.55 was adopted on exactly that reading
#: and its true figure at 400 episodes is **17.3 h/year**, the bound rather than
#: the zero.  1.52 fails for a subtler version of the same error: its point
#: estimate is 3.7 h/year, but that is 4 events, whose 95% bound is about 9.
#:
#: `served` is deliberately not the criterion.  Its denominator is the periods a
#: requirement was published in, and `req_th > 0` tests the raw line limit while
#: the clearing buys against the limit less `THERMAL_MARGIN`; measured the same
#: day the two disagree by a factor of 7 where flows sit near the limit and by
#: 1.13 where they sit well past it.  `curtail_periods` needs no requirement
#: definition at all.
#:
#: The scan at 400 episodes, worst cell, `cost_halfcharge` against `never`:
#:
#:     kappa   no market      best simple arm       events
#:     1.45     31.9 h/yr      0.0 h/yr             0/9600
#:     1.50    405.1 h/yr      0.0 h/yr             0/9600   <- adopted
#:     1.52    743.7 h/yr      3.7 h/yr             4/9600
#:     1.55   1246.5 h/yr     17.3 h/yr            19/9600
#:     1.65      -- 616 h/yr at 64 episodes; fails by any reading --
#:
#: 1.45 clears the shedding bar too but leaves too small a problem: 32 h/year
#: without a market is a feeder that barely needs flexibility.  1.50 is the
#: largest scaling whose shedding is bounded below 5 h/year with margin.
#:
#: **Multiplies the series**, whose maximum is 0.852801 of the registered peak,
#: so 1.50 gives a peak of 17.550 MW, or 1.279 times the registered 13.7197 MW;
#: the other convention gives a number 1.1726 times this one.
#:
#: Measured with the photovoltaic arrays zeroed, which is what makes the two
#: placements face the same exogenous stress -- the no-market shedding above is
#: the same in all four cells because the demand series is bit-identical across
#: the two projection years and no battery acts on that arm.
KAPPA = 1.50
#: Episodes the calibration scan scores each arm on.  It is here rather
#: than in the scan script because it is part of what `KAPPA` means: the
#: shedding bound above is only as fine as `8760 / (n * 24)`, and at 400
#: that is 0.91 h/year.
KAPPA_EVAL_EPISODES = 400
THERMAL_MARGIN = 0.02
VOLTAGE_MARGIN = 0.0
TARIFF_CATEGORY, TARIFF_PERIOD = "C2", 2026
DELTA = 1.0
EPISODE_LEN = 24
#: No source in any dataset of either jurisdiction (§18).  The P2P market
#: derives its own from a pack replacement price over the lifetime bidirectional
#: throughput; that **method** transfers and that **value** does not, since this
#: is a 2.5 hour grid-scale fleet and that one is an 11 kWh household unit.
CYCLE_COST = 15.0

# --------------------------------------------------------------- algorithm ---
HIDDEN = 64
GAMMA, GAE_LAMBDA = 0.99, 0.95
CLIP_EPS, VALUE_COEF = 0.2, 0.5
#: Below this the third channel of §16 is the sweep's own residual rather than
#: an overload.  Measured 2026-08-19 on this machine, `459_0` at `KAPPA`, over
#: all 8760 periods of the arm that never clears in each of the four cells: the
#: channel is **exactly zero** in every period, so no residual band exists to
#: separate from and any positive cut gives the same count.  The constant is
#: kept rather than dropped because the quantity is a sweep output rather than
#: an exact one, and a variant that leaves an overload in place is expected to
#: report values whose scale is not known in advance.
OVERLOAD_EPS = 1e-9
EPOCHS = 4
LEARNING_RATE = 3e-4
LOG_STD_INIT, LOG_STD_FINAL = -0.5, -3.0
#: Rollout episodes per update.  **Raised from 16 on a measured diagnosis, not
#: on preference.**  At 16 the evaluation return rose to a peak and then fell
#: monotonically to 38% of it by iteration 200 -- a curve that does not settle,
#: so nothing it reports is the policy the run converged to.  Logging the three
#: action components located the drift precisely: the price component sat in
#: the saturated region of `softplus` and moved the price not at all, the
#: offered fraction moved by 2.6%, and the **planned charge** fell monotonically
#: from `sigmoid(+0.24) = 0.56` of the headroom to `sigmoid(-4.17) = 0.015`.
#: Planned charging is what the flexibility is made of -- `q_phys` is the plan
#: plus the sustainable discharge -- so the offer shrank and with it the return.
#:
#: That component has a narrow optimum: planning to charge costs money in every
#: period and is paid for only in the tenth of them where the market clears, so
#: the gradient on it is estimated from about 43 clearing events per update at
#: batch 16, and the training return swung between 1.4 and 18.4 across
#: consecutive updates.  At 64 the curve rises and then holds: the return ends
#: at 96% of its peak against 38%, the clearing fraction sits at 0.079 for the
#: last seventy iterations, and the planned charge settles instead of sliding.
#:
#: **It is also nearly free**, which is why 64 rather than 32.  Measured
#: 2026-08-17 on an otherwise idle RTX 4500 Ada, seconds per iteration are 14.4
#: at batch 16, 12.0 at 32 and 19.9 at 64: the interior-point solve is the
#: bottleneck and batching amortises it, so 16 was leaving the card idle.
BATCH = 64
EVAL_EPISODES = 64

#: The observation spans several orders of magnitude across its 23 channels, so
#: it is divided channel by channel before the network sees it.  The divisor is
#: the robust scale of each channel measured once from random rollouts of the
#: 2050 fleet and then frozen, so it is a fixed function of the scenario rather
#: than a running normalisation that would make two arms see different inputs.
OBS_CLIP = 10.0


#: Days of every month held out of training: the 5th, 15th and 25th select the
#: reported checkpoint, the 1st, 11th and 21st score it, and no hour of either
#: appears in a training episode.
#:
#: **Single days rather than contiguous blocks, and six of them rather than
#: three.**  An episode is 24 hours and a held-out window must lie entirely
#: inside held-out days, so an isolated day hosts exactly one episode --
#: midnight to midnight.  That is a virtue: the held-out episodes are then
#: non-overlapping and one per day, so a confidence interval over them is not
#: computed over correlated windows.  A two-day block would host 25 windows
#: instead of 2, but 25 windows sliding across 48 hours are heavily correlated
#: and would overstate precision.  The count therefore has to come from days,
#: and three days a month gives only 36 episodes a year to split two ways; six
#: gives 36 for each of validation and test.
#:
#: Spread through the month rather than consecutive because congestion on this
#: feeder is seasonal -- it binds in the cold months -- so a contiguous block
#: would put the held-out set in one season and measure the season.
VALIDATION_DAYS = (5, 15, 25)
TEST_DAYS = (1, 11, 21)


def series_day_of_month() -> np.ndarray:
    """Day of month for each period of the load series, from its own timestamps.

    Read rather than reconstructed from an assumed start date: the series is a
    projection whose first timestamp is not the obvious one, and a split built
    on a guessed epoch would land on the wrong dates while still looking
    perfectly regular.
    """
    table = pd.read_parquet(_default_data_dir() / "SwissDN_459_0_MV_Load_60min.parquet",
                            columns=["datetime"])
    stamps = np.sort(table["datetime"].unique())
    return stamps, pd.DatetimeIndex(stamps).day.to_numpy().astype(np.int32)


def day_split(dom: np.ndarray, episode_len: int):
    """Three disjoint pools of episode starts, by day of month.

    A start belongs to a split only if **the whole episode** lies inside that
    split's days.  That is the part it is easy to get wrong: a 24 hour window
    opening late on the 31st runs into the 1st, so a training pool built by
    "days that are not held out" would still show the learner a slice of every
    held-out day.  Windows that straddle a boundary are dropped from every
    pool, which costs `episode_len - 1` starts per boundary and buys the
    property that the three pools share no hour at all.

    Returns ``(train, validation, test)`` as int32 arrays of starts.
    """
    n_periods = dom.shape[0]
    label = np.zeros(n_periods, np.int8)              # 0 train
    label[np.isin(dom, VALIDATION_DAYS)] = 1
    label[np.isin(dom, TEST_DAYS)] = 2

    n_starts = n_periods - episode_len + 1
    # A window is in a pool only if every hour it covers carries that label.
    win = np.lib.stride_tricks.sliding_window_view(label, episode_len)
    homogeneous = (win == win[:, :1]).all(1)[:n_starts]
    first = label[:n_starts]
    pools = tuple(
        np.flatnonzero(homogeneous & (first == k)).astype(np.int32)
        for k in (0, 1, 2))
    return pools


#: The aggregator each cell's unilateral sweep deviates (the unilateral-deviation
#: curve read off figure 4's right panel: the one earning most at the 9 900 rung, 7 / 7 / 12 /
#: 12 over `CELLS`).  `floor_k` keeps it a learner for as long as any learner is
#: left, so that `floor_k = n - 1` reproduces exactly the landscape that sweep
#: measured -- one aggregator free, every other at the floor -- and `floor_k =
#: 0` is the published run.  Without the priority the single survivor at
#: `n - 1` would be aggregator 0, which is a different agent with a different
#: rating, and the two ends of the sweep would not be on the same object.
REFERENCE_DEVIATOR = {(2040, 2040): 7, (2040, 2050): 7,
                      (2050, 2040): 12, (2050, 2050): 12}


def floor_learner_mask(year, capacity_year, n, floor_k):
    """`floor_k` aggregators pinned to `BASELINE_ACTION`, the rest learning.

    The non-learner's baseline in this market is the truthful price -- which is
    the floor -- the full deliverable quantity and no planned charging
    (`envs/local_flexibility/env.py`, and appendix A.5's `(-128, +128, -128)`),
    so pinning is "bid at cost and do not withhold", not "bid randomly".

    Which aggregators get pinned is a choice and is made here rather than left
    to index order: the learners are the first `n - floor_k` of a priority that
    puts `REFERENCE_DEVIATOR` first and then ascending index.  `floor_k = 0`
    returns `np.ones(n, bool)` -- byte for byte the array every run before this
    flag passed -- so a run without the flag is the run it always was.
    """
    if floor_k == 0:
        return np.ones(n, bool)
    if not 0 <= floor_k < n:
        raise ValueError(f"floor_k must lie in [0, {n}), got {floor_k}")
    dev = REFERENCE_DEVIATOR[(year, year if capacity_year is None
                              else capacity_year)]
    priority = [dev] + [i for i in range(n) if i != dev]
    mask = np.zeros(n, bool)
    mask[priority[:n - floor_k]] = True
    return mask


def scenario(year: int, capacity_year: int = None, kappa: float = None,
             initial_soc: float = None, split: str = None,
             monitor: bool = False, floor_k: int = 0):
    """`year` fixes the placement and the photovoltaic array; `capacity_year`
    fixes the fleet total.  When they differ every battery of `year`'s placement
    is scaled by the ratio of the two fleet totals, which holds the count, the
    buses and the co-located generation fixed while moving only the megawatts.

    `kappa` overrides the load scaling for a calibration scan and defaults to
    `KAPPA`, so every existing caller is unaffected.  It is a keyword rather
    than a module-level rebind because the scan runs many values in one process
    and a shared default must not be edited to take a measurement.
    """
    case = create_case459_0(node_v_min=EN50160_MV_V_MIN,
                            node_v_max=EN50160_MV_V_MAX)
    sens = build_voltage_sensitivity(case)
    series = load_swiss_flex_series(projection_year=year,
                                    tariff_category=TARIFF_CATEGORY,
                                    tariff_period=TARIFF_PERIOD)
    table = pd.read_parquet(_default_data_dir() / SWISS_BESS_FILE)
    fleet = table.query("projection_year == @year").sort_values("osmid").copy()
    if capacity_year is not None and capacity_year != year:
        want = table.query("projection_year == @capacity_year")["nominal_power_kw"].sum()
        ratio = want / fleet["nominal_power_kw"].sum()
        fleet["nominal_power_kw"] *= ratio
        fleet["capacity_kwh"] *= ratio
    nodes = pd.read_parquet(_default_data_dir() / "SwissDN_459_0_MV_Nodes.parquet")
    bus_of = {str(o): i for i, o in enumerate(nodes["osmid"])}
    agent_bus = np.array([bus_of[str(o)] for o in fleet["osmid"]], np.int32)
    n = len(agent_bus)
    env = make_local_flex_env(case, sens, agent_bus,
                              voltage_margin=VOLTAGE_MARGIN,
                              thermal_margin=THERMAL_MARGIN,
                              period_hours=DELTA)
    # The efficiencies are published per device and must be passed: the bundle
    # constructor otherwise supplies 0.95/0.95, a round trip of 0.9025 against
    # the 0.85 the source registers for every device in every projection year.
    # That default moves the replacement cost of §9.5 from 149.88 to 142.04 and
    # also enters the headroom and the deliverable power, so a run that omits it
    # is running more efficient batteries than the data describes throughout.
    # `initial_soc` defaults to the constructor's 0.5 against a `soc_min` of
    # 0.1, so every episode opens holding stock that no period of it paid for.
    # The episode boundary does not price what is left either, which is the
    # shape the P2P market measured a boundary effect in; the override exists so
    # that effect can be measured here rather than assumed absent.
    battery = make_battery_bundle(
        n_devices=n, dt_hours=DELTA,
        capacity_mwh=(fleet["capacity_kwh"].to_numpy() / 1000.0).tolist(),
        power_mw=(fleet["nominal_power_kw"].to_numpy() / 1000.0).tolist(),
        eta_charge=fleet["eta_charge"].to_numpy().tolist(),
        eta_discharge=fleet["eta_discharge"].to_numpy().tolist(),
        **({} if initial_soc is None else {"initial_soc": float(initial_soc)}))
    # Photovoltaic output is set to zero in every cell, and this is what makes
    # the two-by-two identify what it claims to.  §3.1 defines the quantity per
    # aggregator, so the environment sees generation only at buses carrying a
    # battery; the 34-bus placement therefore comes with 63% more modelled
    # generation than the 24-bus one, and measured on the arm that never clears
    # -- no market at all -- that placement has 40% fewer periods carrying a
    # requirement (0.2604 against 0.1569).  A placement contrast run with the
    # published arrays would compare two different demands, which is the
    # supply-and-demand story this design exists to exclude.
    #
    # §14 names running with no photovoltaic generation, stated as load driven
    # only, as a declared scenario parameter, so this is a choice the
    # specification already admits rather than a quantity invented here.
    pv = np.zeros_like(series.pv)
    # `split` picks which days `reset` may open an episode on.  `None` is every
    # legal start, which is what every run before 2026-08-17 used and what the
    # calibration scan still uses; the three named splits share no hour.
    if split is None:
        pool = None
    else:
        _, dom = series_day_of_month()
        pools = dict(zip(("train", "validation", "test"),
                         day_split(dom, EPISODE_LEN)))
        if split not in pools:
            raise ValueError(f"split must be one of {sorted(pools)} or None, "
                             f"got {split!r}")
        pool = pools[split]
    params = make_local_flex_params(
        load_series=series.load_mw, pv=pv,
        energy_price=series.energy_price, battery=battery,
        cycle_cost=np.full(n, CYCLE_COST),
        cursor_pool=pool,
        # False is the mechanism as specified; True is the monitored variant of
        # §8, in which the operator refuses to publish a requirement against a
        # baseline the participant positioned itself.  The two differ in nothing
        # else, so a paired run measures what the positioning exposure is worth.
        monitor_baseline=bool(monitor),
        load_scale=KAPPA if kappa is None else float(kappa),
        learner_mask=floor_learner_mask(year, capacity_year, n, int(floor_k)),
        episode_len=EPISODE_LEN)
    return env, params, n


def obs_scale(env, params, n, key=0):
    """Frozen channel scales, measured once from random rollouts."""
    reset, _, step_auto, spec = env
    k = jax.random.PRNGKey(key)
    _, state = reset(k, params)
    rows = []
    for kk in jax.random.split(k, 32):
        act = jax.random.uniform(kk, (n, 3), minval=-2.0, maxval=2.0)
        obs, state, *_ = step_auto(kk, state, act, params)
        rows.append(np.asarray(obs))
    flat = np.concatenate(rows).reshape(-1, spec["obs_dim"])
    scale = np.maximum(np.abs(flat).mean(0), 1e-3)
    return jnp.asarray(scale)


# ------------------------------------------------------------------ policy ---
def init_policy(key, obs_dim):
    k1, k2, k3 = jax.random.split(key, 3)
    glorot = lambda k, i, o: jax.random.normal(k, (i, o)) * np.sqrt(2.0 / i)
    return dict(w1=glorot(k1, obs_dim, HIDDEN), b1=jnp.zeros(HIDDEN),
                w2=glorot(k2, HIDDEN, HIDDEN), b2=jnp.zeros(HIDDEN),
                mean=glorot(k3, HIDDEN, 3) * 0.01, mean_b=jnp.zeros(3),
                value=glorot(k3, HIDDEN, 1) * 0.01, value_b=jnp.zeros(1))


def init_policy_per_agent(key, obs_dim, n_agents):
    """`n_agents` independent copies of `init_policy`'s network, one per aggregator.

    Every leaf gains a leading `n_agents` axis and nothing else changes, which
    is the layout `powermarketjax/learning/ippo.py` writes under
    `per_agent_params=True`.  The two layouts therefore hold the same leaves and
    differ only in that axis, so a policy of one layout used as the other does
    not fail on shape -- it produces a different policy that looks like the
    right one.  That is why `--per-agent-params` is stamped into the result file
    (`main`) rather than left to be inferred from the leaves.

    The keys come from one `split`, so agent `j` is shaped by
    ``split(key, n_agents)[j]`` and no two agents start from the same draw.
    """
    return jax.vmap(lambda k: init_policy(k, obs_dim))(
        jax.random.split(key, n_agents))


def forward(policy, obs):
    h = jnp.tanh(obs @ policy["w1"] + policy["b1"])
    h = jnp.tanh(h @ policy["w2"] + policy["b2"])
    return (h @ policy["mean"] + policy["mean_b"],
            jnp.squeeze(h @ policy["value"] + policy["value_b"], -1))


def forward_per_agent(policy, obs):
    """`forward` with the agent axis of `obs` mapped against the parameter axis.

    `obs` is ``(..., n_agents, obs_dim)`` at both call sites -- one period's
    observation inside `make_rollout`, and a whole
    ``(BATCH, EPISODE_LEN, n_agents, obs_dim)`` block inside `loss_fn` -- so the
    agent axis is always the one before `obs_dim`.  It is moved to the front,
    `vmap`ped against the leading axis of `policy`, and moved back, so every
    caller keeps the shapes it had under a shared network: `mean` comes out
    ``(..., n_agents, 3)`` and `value` ``(..., n_agents)``.

    This mirrors `_apply_per_agent` in `powermarketjax/learning/ippo.py` axis for
    axis.  The one thing that module moves and this one does not is `log_std`,
    and the reason is that there is no `log_std` leaf here at all: it is annealed
    by the training loop and passed in, not learned.

    **A shape test cannot catch a crossed axis where `obs_dim == n_agents`**, so
    it is worth recording that on this scenario they never coincide: `obs_dim` is
    23 in all four cells while `n_agents` is 24 for the 2040 placement and 34 for
    the 2050 one, so `vmap` refuses a crossed layout in every cell.  That guard
    is a coincidence of this feeder, not a design, which is why the check that
    actually bites is behavioural and lives in
    `tests/tools/test_flex_per_agent_params_l0.py`: knock out agent `j`'s output
    head and require that agent `j` and no other agent moves.
    """
    zz = jnp.moveaxis(obs, -2, 0)
    mean, value = jax.vmap(forward)(policy, zz)
    return jnp.moveaxis(mean, 0, -2), jnp.moveaxis(value, 0, -1)


def log_prob(mean, log_std, raw):
    z = (raw - mean) / jnp.exp(log_std)
    return (-0.5 * z ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)


#: Mode 3 is the floor: it marks up so far that the offer exceeds the value of
#: lost load and therefore never clears, so its return is zero by construction.
#: It exists to answer a question the other arms cannot -- whether this
#: environment can tell policies apart at all.  If an arm does not beat it, no
#: ordering among the others means anything.
NEVER_CLEARS_ACTION = (+128.0, +128.0, -128.0)

#: Mode 4 is the baseline-positioning exploit §8 leaves in place: offer at the
#: replacement cost while planning the largest charge the envelope allows, so
#: the participant deepens the very constraint it is then paid to relieve.
#:
#: **It was written as a deliberately poor policy and the first run showed it
#: beating the learner in every cell** (157.0 against 128.6, 50.1 against 30.2),
#: which is the finding rather than a defect in it: forgoing the markup costs
#: less than the extra volume that inc-dec buys.  §16 records this behaviour as
#: a diagnostic and §18 leaves whether the operator should act on it undecided,
#: so it is reported as its own arm rather than folded into a baseline.
INCDEC_ACTION = (-128.0, +128.0, +128.0)

#: The action map of §9.3 is defined on all of R^3, so the policy's output is
#: used unsquashed and only widened by a fixed factor: the saturating baseline
#: sits at 128, far outside anything a tanh could reach, and clipping the policy
#: into a box would make the truthful reference unreachable by construction.
ACTION_GAIN = 4.0


def market_action(mode, learned, n, kk):
    """The five arms' action, given whatever action the learner proposed.

    Lifted out of `make_rollout` **character for character**, the way
    `make_loss_fn` was lifted out of `make_update`, so that the IPPO arm traces
    the expression it always traced.  It is shared rather than copied because
    the four reference arms belong to the market and not to the learner: modes 1
    to 4 discard `learned` entirely, which is what makes them bit-identical
    across `--algo`, and `tests/tools/test_flex_sac_l0.py` asserts exactly that.
    Two copies of this expression would be two definitions of the arm every
    ordering in §16 is measured against.
    """
    return jnp.where(
        mode == 0, learned,
        jnp.where(mode == 1, baseline_action(n),
                  jnp.where(
                      mode == 3,
                      jnp.broadcast_to(jnp.asarray(NEVER_CLEARS_ACTION),
                                       (n, 3)),
                      jnp.where(
                          mode == 4,
                          jnp.broadcast_to(jnp.asarray(INCDEC_ACTION),
                                           (n, 3)),
                          jax.random.uniform(
                              jax.random.fold_in(kk, 1), (n, 3),
                              minval=-2.0, maxval=2.0)))))


#: Index of ``state.soc`` in the observation of §9.4.  `_get_obs` stacks the
#: seven registered parameters first -- capacity, the two ratings, the two
#: efficiencies, and the two state-of-charge bounds -- and the carry follows
#: them.  `info["terminal_obs"]` is the unscaled observation, so this reads the
#: state of charge the episode actually ended on.
SOC_CHANNEL = 7


def replacement_cost(params):
    """§9.5's $c^{rep}$, per aggregator, in CHF/MWh.

    The same expression `action.py` builds the offer floor from.  It is
    recomputed here rather than read back because the environment returns it
    only inside `step`, and it is checked against the floor the runs realise:
    `plot_curves.check_floor` compares the truthful arm's cleared price with
    149.88, and this must agree with that or the settlement below is pricing
    the carry differently from the way the market prices delivery.

    **Computed in numpy on host copies, not in `jnp`.**  `make_rollout` runs
    inside the jitted `update`, and inside a trace every `jnp` operation is
    staged into the jaxpr and returns a tracer even when its inputs are
    closure constants -- so doing the arithmetic first and converting after
    raises `TracerArrayConversionError`.  Converting first keeps the whole
    expression on the host, and the result enters the rollout as a constant.
    """
    b = params.battery
    eta_rt = np.asarray(b.eta_charge) * np.asarray(b.eta_discharge)
    # `energy_price` is the whole year, `(n_periods,)`, and `env.step` takes
    # `[cursor]` from it.  The carry is settled once, at the end of the episode,
    # so a time-varying tariff would need that period's price carried out of the
    # scan.  The adopted tariff is annual and this asserts it rather than
    # assuming it: a wholesale-price arm would make it vary, and
    # the failure would otherwise be a silently wrong settlement price rather
    # than an error.
    price = np.asarray(params.energy_price)
    distinct = np.unique(price)
    if distinct.size != 1:
        raise ValueError(
            f"terminal settlement assumes a flat tariff; energy_price takes "
            f"{distinct.size} distinct values in [{price.min():.4f}, "
            f"{price.max():.4f}]. Carry the terminal period's price out of the "
            f"rollout instead of using a scalar here.")
    cyc = np.asarray(params.cycle_cost)
    return cyc * (1.0 + 1.0 / eta_rt) + float(distinct[0]) / eta_rt


def carry_value(params):
    """What a deliverable MWh sitting in the battery is worth, in CHF/MWh.

    **Not $c^{rep}$, and the difference is exactly one discharge cycle cost.**
    $c^{rep}$ is the marginal cost of *delivering* a megawatt hour and replacing
    it, which is why §9.3 makes it the offer floor.  Stock that is merely held
    has not been discharged yet, so pricing it at $c^{rep}$ credits a
    degradation cost that has not been incurred:

        c_rep  = c_cyc * (1 + 1/eta_rt) + pi_en / eta_rt
        carry  =        (c_cyc + pi_en) / eta_rt
        c_rep - carry = c_cyc                       (asserted below)

    At the Swiss configuration that is 149.88 against 134.88. Settling the carry
    at 149.88 pays 15 CHF for every megawatt hour left in the battery at the
    close, which is a risk-free return on hoarding and a boundary arbitrage in
    the opposite direction to the one this settlement exists to remove -- and it
    is not hypothetical: measured 2026-08-17 at kappa 1.55, it moved the
    half-charging arm from -8.41 to +26.91 and the inc-dec arm from 35.47 to
    70.87, most for the arms that charge hardest.

    At `carry` the two prices are each other's consistency check: selling a
    megawatt hour at the floor earns `c_rep`, costs one discharge cycle and
    consumes `carry` of stock, and `c_rep - c_cyc - carry` is zero.

    Numpy throughout, for the reason `replacement_cost` gives.
    """
    b = params.battery
    eta_rt = np.asarray(b.eta_charge) * np.asarray(b.eta_discharge)
    price = np.asarray(params.energy_price)
    distinct = np.unique(price)
    if distinct.size != 1:
        raise ValueError(
            f"terminal settlement assumes a flat tariff; energy_price takes "
            f"{distinct.size} distinct values in [{price.min():.4f}, "
            f"{price.max():.4f}]. Carry the terminal period's price out of the "
            f"rollout instead of using a scalar here.")
    cyc = np.asarray(params.cycle_cost)
    value = (float(distinct[0]) + cyc) / eta_rt
    gap = replacement_cost(params) - value
    if not np.allclose(gap, cyc, rtol=1e-5, atol=1e-5):
        raise ValueError(
            f"c_rep - carry must equal the cycle cost; got {gap} against "
            f"{cyc}. One of the two expressions no longer matches action.py.")
    return value


def terminal_settlement(params, terminal_obs_last):
    """Price the stock the episode ends holding, and charge it for the stock it
    opened holding.

    **The episode boundary is otherwise open at both ends.** `reset` puts every
    battery at ``initial_soc`` (0.5) against a ``soc_min`` of 0.1, so an episode
    opens with energy no period of it paid for, and it closes without the
    remainder being worth anything -- which makes selling the opening stock and
    never replacing it a strictly profitable policy that no part of the market
    is responsible for.  Measured 2026-08-17 at kappa 1.55, moving the opening
    state of charge to `soc_min` moved the truthful arm's return from +0.24 to
    -2.12 and the inc-dec arm's from 35.47 to 53.16, so the effect is real and
    is not even monotone across arms.

    Settling both ends at `carry_value` closes it: stock is worth what it cost
    to put there, so carrying it across the boundary is neither a gain nor a
    loss and the only way to earn is to move energy within the episode, which
    is what a 2.5 hour battery does.  The opening charge is a constant per
    episode, but **not** a constant across cells -- it scales with capacity,
    and capacity is one of the two treatments -- so it cannot be dropped as an
    offset.

    Only the difference of the two states enters, so ``soc_min`` cancels and
    the result does not depend on where the deliverable window is measured from.

    Returns a ``(n_agent,)`` adjustment to add to the last period's reward.
    """
    b = params.battery
    deliverable = b.eta_discharge * b.capacity
    soc_end = terminal_obs_last[:, SOC_CHANNEL]
    soc_start = jnp.asarray(b.initial_soc, jnp.float32)
    return carry_value(params) * deliverable * (soc_end - soc_start)


def make_rollout(env, params, scale, n, log_std, per_agent=False):
    reset, _, step_auto, spec = env
    # Bound HERE, in Python, at construction, and not chosen inside the scan.
    # `powermarketjax/learning/ippo.py` binds its own two paths the same way and
    # says why: bound this way the shared arm is `forward` itself rather than an
    # expression with an axis moved by zero, so the shared path is bit-identical
    # across the arrival of this option by construction and not by a belief
    # about how XLA folds a `moveaxis` of nothing.
    _forward = forward_per_agent if per_agent else forward

    def rollout(policy, key, mode, start=-1):
        k0, k1 = jax.random.split(key)
        _, state = reset(k0, params)
        # `start >= 0` pins the episode to that index of the series instead of
        # taking the one `reset` drew.  Evaluation uses it to run **each**
        # held-out day exactly once: the test pool holds 36 starts and drawing
        # 64 keys from it would weight those days randomly and report a
        # weighted mean rather than the mean over the held-out set.  Training
        # passes -1 and keeps the random draw.
        state = state.replace(
            cursor=jnp.where(start >= 0, jnp.int32(start), state.cursor))

        def body(carry, kk):
            state = carry
            obs = spec["get_obs"](state, params) / scale
            obs = jnp.clip(obs, -OBS_CLIP, OBS_CLIP)
            mean, value = _forward(policy, obs)
            noise = jax.random.normal(kk, mean.shape) * jnp.exp(log_std)
            learned = (mean + noise) * ACTION_GAIN
            act = market_action(mode, learned, n, kk)
            lp = log_prob(mean, log_std, mean + noise)
            _, s2, reward, costs, _, info = step_auto(kk, state, act, params)
            # `costs` carries all three channels of §16 and only the first
            # was returned until 2026-08-19.  The third is what the sweep of §7
            # measures on the **cleared** point, so it is the only quantity
            # that distinguishes a variant which relieves an overload from one
            # which leaves it in place; the monitored baseline is
            # exactly such a variant and no reported figure would have shown
            # the difference.  Appended rather than substituted so that the
            # positional consumers of this tuple keep their indices.
            return s2, (obs, mean + noise, lp, value, reward, costs[:, 0],
                        info["traded_volume"], info["price_avg"],
                        info["z"], info["req_th_count"], info["terminal_obs"],
                        info["req_v_count"], info["mu"], info["converged"],
                        costs[:, 2])

        _, out = jax.lax.scan(body, state, jax.random.split(k1, EPISODE_LEN))
        (obs, raw, lp, value, reward, shed, vol, price, z, req_th,
         term, req_v, mu, converged, overload) = out
        # The carry is settled on the last period rather than spread, because
        # it is one transaction at one price and spreading it would put a
        # payment in periods where nothing was carried.
        adj = terminal_settlement(params, term[-1])
        reward = reward.at[-1].add(adj)
        return (obs, raw, lp, value, reward, shed, vol, price, z, req_th,
                term, req_v, mu, converged, overload)

    return rollout


def advantages(reward, value, last_value):
    def step(carry, x):
        gae, nxt = carry
        r, v = x
        delta = r + GAMMA * nxt - v
        gae = delta + GAMMA * GAE_LAMBDA * gae
        return (gae, v), gae
    _, adv = jax.lax.scan(step, (jnp.zeros_like(value[-1]), last_value),
                          (reward, value), reverse=True)
    return adv, adv + value


def make_loss_fn(per_agent=False):
    """PPO's clipped surrogate plus the scaled value term, over one layout.

    Lifted out of `make_update` -- character for character, so the shared arm is
    the expression it always was -- so that the reductions can be gated without
    building a scenario.  `tests/tools/test_flex_per_agent_params_l0.py` asserts
    the identity a mean over the agent axis implies and a sum over it does not:
    with `n_agents` copies of one network, the gradient of the shared layout is
    the **sum** over agents of the per-agent gradients.  Under a sum over the
    agent axis that relation would be off by a factor of `n_agents` -- 24 on the
    2040 placement and 34 on the 2050 one -- and that factor is the number the
    gate bites with.
    """
    _forward = forward_per_agent if per_agent else forward

    def loss_fn(policy, data, log_std):
        mean, value = _forward(policy, data["obs"])
        lp = log_prob(mean, log_std, data["raw"])
        adv = data["gae"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ratio = jnp.exp(lp - data["logp"])
        pg = -jnp.minimum(ratio * adv,
                          jnp.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv).mean()
        # The value target is divided by a fixed scale before the squared error.
        # The trunk is shared with the policy head and the whole gradient goes
        # through one global-norm clip, so an unnormalised value loss lets the
        # cell with the larger returns take a larger share of the clipped
        # gradient -- which on its own produces an ordering across cells that
        # differ in reward magnitude, and is indistinguishable from the effect
        # under study.
        scale = data["ret_scale"]
        return pg + VALUE_COEF * (((value - data["target"]) / scale) ** 2).mean()

    return loss_fn


def make_update(env, params, scale, n, iterations, per_agent=False):
    """Build `(optimiser, update)` for one cell.

    **`per_agent` changes the parameter layout and nothing about the
    reductions**, and that is a decision taken by reading
    `powermarketjax/learning/ippo.py` rather than by preference: the three
    markets that already carry a `--per-agent-params` column reduce the way that
    module reduces, and a column that reduced differently here would not be the
    same treatment across markets.  What that module does, in its own words:

    * `_loss` standardises the advantages with **one scalar**, and its docstring
      says over which axes -- "Advantages are standardised within the minibatch,
      over the agent axis as well as the sample axis, so one shared scale serves
      every agent."  `loss_fn` below already computes
      ``(adv - adv.mean()) / (adv.std() + 1e-8)`` with no `axis=`, which is that
      same one scalar over every axis including the agent axis.
    * The policy and value terms are **means over every axis, the agent axis
      included, not sums over agents**: `ippo.py` writes
      ``pg = -jnp.minimum(unclipped, clipped).mean()`` and
      ``vf = 0.5 * ((value - batch["ret"]) ** 2).mean()``, and both expressions
      are the same under either layout -- the module binds the layout in
      `_apply` and leaves `_loss` alone.  `loss_fn` below is already written that
      way, so **neither line changes** when this flag is set.
    * The **one** quantity that module does compensate for the layout is the
      entropy: `_entropy` divides by `n_agents` under the per-agent path,
      because "`entropy` sums over every axis of `log_std` ... Left alone, the
      per-agent path would report `n_agents` times the number the shared path
      reports, and `ent_coef` would silently weigh that much more".  **There is
      nothing to compensate here**: `loss_fn` has no entropy term, and `log_std`
      is not a parameter at all -- it is annealed by the training loop and
      passed in.
    * The clip is left **joint** across agents, deliberately, and `ippo.py` is
      explicit that this is a property of the optimiser and not of the layout:
      "`optax.clip_by_global_norm` bounds the norm of the WHOLE parameter
      pytree, so with one network per agent it clips the `n_agents` gradients
      jointly rather than each on its own: an agent with a large gradient
      shrinks its neighbours' updates.  That is a property of the optimiser
      chain, not of per-agent parameters as such, and it is left in place rather
      than quietly replaced by a per-agent clip, because swapping it would make
      the two paths differ in two ways at once."  The chain below already clips
      the whole tree at 0.5, so it is left exactly as it is, for that reason.

    So the only thing this keyword does is pick which `forward` the loss and the
    rollout call, and it is bound in Python at construction for the reason
    `make_rollout` gives.
    """
    import optax
    # Annealed over a fixed horizon rather than over the stopping ceiling.  When
    # the two are the same argument, a cell that stops early stops with a larger
    # residual step size and a wider exploration width than one that runs long,
    # and those residuals then differ systematically across the cells being
    # compared.  SCHEDULE_LEN is the horizon the schedule is written for.
    schedule = optax.linear_schedule(LEARNING_RATE, 0.0, SCHEDULE_LEN * EPOCHS)
    optimiser = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(schedule))
    loss_fn = make_loss_fn(per_agent=per_agent)

    @jax.jit
    def update(policy, opt_state, key, log_std):
        key, ck = jax.random.split(key)
        roll = make_rollout(env, params, scale, n, log_std, per_agent=per_agent)
        obs, raw, lp, value, reward, shed, vol, price, z, req, term, *_ = jax.vmap(
            roll, in_axes=(None, 0, None))(policy, jax.random.split(ck, BATCH), 0)
        # **Zero, and this reverses the previous version deliberately.**  That
        # version bootstrapped from `terminal_obs` on the ground that `done` is
        # a time-limit truncation rather than a terminal state, which was right
        # while the boundary was open.  It is no longer: `reset` draws a fresh
        # episode window and returns every battery to `initial_soc`, so the only
        # quantity that crossed the boundary was the stored energy, and
        # `terminal_settlement` now prices exactly that.  Bootstrapping as well
        # would pay for the carry twice -- once explicitly and once through a
        # value head that has learned what the carry is worth.
        # Shaped and typed from `value`, not from `reward`: `advantages` carries
        # the bootstrap alongside the value estimate, and x64 makes the value
        # head float64 while the environment's reward is float32, so taking the
        # dtype from the wrong one fails the scan's carry-type check.
        last = jnp.zeros_like(value[:, -1])
        gae, target = jax.vmap(advantages)(reward, value, last)
        data = dict(obs=obs, raw=raw, logp=lp, gae=gae, target=target,
                    ret_scale=jnp.maximum(jnp.abs(target).mean(), 1.0))

        def epoch(carry, _):
            p, o = carry
            g = jax.grad(loss_fn)(p, data, log_std)
            u, o = optimiser.update(g, o)
            return (optax.apply_updates(p, u), o), None

        (policy, opt_state), _ = jax.lax.scan(epoch, (policy, opt_state),
                                              None, length=EPOCHS)
        return policy, opt_state, dict(ret=reward.sum(1).mean(),
                                       shed=shed.sum(1).mean(),
                                       volume=vol.mean(), price=price.mean())

    return optimiser, update


# --------------------------------------------------------------------- SAC ---
"""The benchmark puts two SAC columns in every market's row, accepted on the
condition "test it thoroughly first, wire it in only if it works".  This section is 05's half of
that: the package's SAC, driven through **this** market's rollout.

**Why the package's `make_sac` is not called.**  05's rollout is not the one
`learning/adapters.unpack_env` hands the package harness.  It pins an episode to
a named held-out day (`start`), it scales the observation by `obs_scale` and
clips it rather than standardising it, it widens the policy coordinate by
`ACTION_GAIN`, and -- the one that changes the quantity being learned -- it
settles the stored energy at the boundary with `terminal_settlement`.
Every one of those is part of what this market's IPPO column measures, so an SAC
column that skipped them would not be the same treatment.  What is reused is
therefore the learner, not the harness.

**Imported rather than restated.**  Both networks, both per-agent axis maps and
the critic's view of an action come from `powermarketjax/learning/sac.py`
itself; the log-density and the action map from `powermarketjax/learning/
policy.py`; the hyperparameters from `tools/benchmark/hyperparams.py:SAC_SHARED`,
the same object `run_rl_01/02/03.py` hand to `make_sac`.  Three of those names
are private to their modules.  They are imported anyway, and that is the lesser
evil: a copy here is a second definition of a quantity that has to be one
quantity across five markets, and it would drift silently.

**The two places 05 had no counterpart, decided here.**

1. *`log_std`.*  05's PPO has no `log_std` parameter -- the training loop anneals
   it from `LOG_STD_INIT` to `LOG_STD_FINAL` and passes it in.  SAC is given the
   package's `SACActor` instead, whose `log_std` is a state-dependent head
   squashed into ``[log_std_min, log_std_max]``, and the anneal is **not** fed to
   it.  The reason is not preference.  05's action space **was** unbounded when
   this was decided (`env.py`: ``action_low=-inf``; finite since 2026-09-10, so
   the premise below no longer holds and the decision has not been retaken --
   `tests/tools/test_flex_sac_l0.py` carries both measurements), so
   `policy.log_prob` added no `tanh` correction, and with a `log_std` that did
   not depend on the parameters the reparameterised
   ``log pi(mean + std * eps)`` reduced to
   ``sum(-0.5 eps^2 - log_std - 0.5 log 2pi)``, in which the actor's parameters
   did not appear at all.  SAC's actor loss would then have a `-log pi` term with
   an identically zero gradient, and the temperature would be tuning against a
   constant.  `test_annealed_log_std_kills_the_entropy_gradient` measures that
   zero rather than asserting it.
2. *The entropy term.*  05's PPO has none: `make_loss_fn` returns the clipped
   surrogate plus the value term and nothing else.  SAC keeps the package's
   autotuned temperature, target entropy ``-act_dim`` and all, because
   `sac.py` says in its own words that "the entropy terms of PPO and SAC are not
   the same quantity ... aligning one coefficient does not align the two
   learners (the lesson of market 04)".  Setting `alpha` to zero to match 05's
   PPO would not be SAC with 05's entropy convention; it would be a different
   algorithm with no name.

**Where 05's SAC column may not be read across a row.**  Two points, both
consequences of the above:

* *The target entropy is a different quantity here.*  On 01, 02, 03 and 04 the
  action is squashed (`action_low`/`_high` are finite: ``[1, markup_max]``,
  ancillary's reserve box, p2p's ``[-1, 1]``), so ``-act_dim`` is a target on the
  `tanh`-corrected density.  On 05 nothing is squashed, so it is a target on the
  plain Gaussian differential entropy, whose additive constant is different.
  The **number** ``-3`` therefore pins a different exploration width here than
  the same number pins there, and 05's `alpha` and `entropy` diagnostics must
  not be tabulated beside another market's.  Market outcomes -- return, cleared
  price, curtailment -- are unaffected and are what the row compares, which is
  the rule `sac.py` already states.
* *Evaluation is stochastic here and deterministic there.*
  `sac.make_sac_greedy_action` reports the mean; 05 reports the policy including
  its own exploration width, because that is what 05's IPPO column does
  (`evaluate(..., LOG_STD_FINAL, ...)`) and the within-row comparison has to hold
  the protocol fixed.  Under `--algo sac` the width is the actor's own head.  The
  deterministic reading is not lost: `final["ret_greedy"]` carries it, so the
  cross-market reading exists in the same file.

A third difference is inherited rather than decided: `SACActor` and `SoftQ` are
flax modules and flax's `param_dtype` defaults to float32, so SAC's parameters
are float32 while 05's PPO parameters are float64.  That is true of the package's
SAC on every market -- `run_rl_02.py` also turns x64 on -- and is recorded, not
repaired.  `log_alpha` is float64, because `jnp.log(cfg.init_alpha)` is taken
under x64; that is the package's arrangement too.

**THE COLUMN DOES NOT RUN AS DECLARED, AND THE REASON IS ALREADY WRITTEN DOWN
ELSEWHERE.**  Measured 2026-09-09, CPU, x64, cell `2040p_2040c`, seed 0, the
shared layout at `SAC_SHARED`: within the 1 536 gradient steps of the FIRST
iteration `q_mean` runs from 1.07e2 to 6.38e8, the actor's `|mean|` from 2.9e2
to 8.4e5, the `log_std` head pins at its ceiling `log_std_max = 2`, `q_loss`
reaches 9.0e18, and the training return collapses to 0.000 -- the policy
saturates §9.3's action map and the market clears nothing.  The end-to-end smoke
of that run reports `ret` equal to the `incdec` arm's **bit for bit**, which is
not the learner finding that strategy: a saturated action lands in the same
corner of the action map.

`policy.bounds_for` names this failure and its cause: "the critic reads the raw
coordinate and whose actor maximises the critic, so with no bound there is no
fixed point and the first iteration on the ancillary market diverged (measured
2026-09-03, `q_loss` 1e27 within 3 072 updates)".  The ancillary
market answered by publishing a box.  **05 is the only one of the five that
still declares `+-inf`** -- 01 and 02 bound the markup at `[1, markup_max]`, 03
publishes its reserve box, 04 declares `[-1, 1]`.  The control that separates
"the cause is the unbounded coordinate" from "the cause is something in this
wiring" is in `sac_bounds`: the same learner, the same keys, the same scenario,
with a finite box, holds `q_loss` between 8.0e2 and 3.2e3 and `q_mean` at 26.5
after the same 1 536 updates, four to fifteen orders of magnitude apart.

Publishing a box for 05 is a decision about the market and not about the
learner, so it is not taken here; `bounds_for` is explicit that "a learner does
not get to choose it".  Until it is taken, this column trains a critic with no
fixed point, and no number it produces should be tabulated.
"""
from functools import partial  # noqa: E402

from powermarketjax.learning.policy import (bounds_for,  # noqa: E402
                                            log_prob as squashed_log_prob,
                                            to_action)
from powermarketjax.learning.sac import (SACActor, SoftQ,  # noqa: E402
                                         _actor_per_agent, _q_action,
                                         _q_per_agent)

#: The learner diagnostics `--algo sac` writes into each curve point.  They are
#: SAC's own and have no IPPO counterpart, so they are added to the curve dict
#: only on the SAC path; an IPPO curve keeps exactly the eight keys it had.
SAC_DIAGNOSTICS = ("q_loss", "q_mean", "target_mean", "actor_loss",
                   "alpha_loss", "entropy", "alpha", "buffer_filled")


def shared_sac_config():
    """`(SAC_SHARED, SAC_PROVENANCE)` from `tools/benchmark/hyperparams.py`.

    Loaded by path rather than by putting `tools/benchmark` on `sys.path`, which
    would make fifty unrelated module names importable as a side effect of
    importing this driver.  Imported rather than restated because these
    numbers are one calibration across five markets: a copy here would be a
    second one, and the two would part company without either file changing.
    """
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "benchmark" / "hyperparams.py"
    spec = importlib.util.spec_from_file_location("_flex_hyperparams", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SAC_SHARED, mod.SAC_PROVENANCE


def sac_config(reward_scale):
    """`SAC_SHARED` with 05's batch and this cell's fitted reward scale.

    `n_envs` and `horizon` are what this driver already collects -- `BATCH`
    episodes of `EPISODE_LEN` periods -- so the two numbers are reported
    separately rather than as their product, which would hide the parallel width
    behind the sample count.  `reward_scale` is a per-cell measurement and
    `SAC_SHARED` ships it as `nan` on purpose; see `sac_reward_scale`.
    """
    import dataclasses
    base, provenance = shared_sac_config()
    if base.gamma != GAMMA:
        raise ValueError(
            f"SAC_SHARED.gamma={base.gamma} against this driver's GAMMA="
            f"{GAMMA}; the two arms of one cell have to discount the same "
            f"future or the column compares two objectives")
    return dataclasses.replace(base, n_envs=BATCH, horizon=EPISODE_LEN,
                               reward_scale=float(reward_scale)), provenance


def sac_reward_scale(env, params, scale, n, key):
    """Pooled standard deviation of the per-agent reward under the truthful arm.

    `sac.py`: the critic regresses on `reward / reward_scale`, and the scale is
    "the pooled standard deviation of the per-agent reward under the truthful
    action, fitted once and frozen, like the observation statistics -- a quantity
    derived by the apparatus, not a knob".  Floored at one for the reason
    `sac.reward_statistics` floors it.

    Fitted from **this** driver's rollout at `mode == 1` rather than from
    `sac.reward_statistics`, for one reason that matters: the reward this market
    pays includes `terminal_settlement`, and the package's rollout does not add
    it.  A scale fitted without it would be fitted on a different quantity from
    the one the critic then regresses on.  Mode 1 discards the network's output
    entirely, so no policy and no learner configuration enters this number --
    which is why the throwaway `dummy` below is allowed to be a PPO tree.
    """
    roll = make_rollout(env, params, scale, n, LOG_STD_FINAL)
    dummy = init_policy(key, env[3]["obs_dim"])
    reward = jax.vmap(roll, in_axes=(None, 0, None))(
        dummy, jax.random.split(key, BATCH), 1)[4]
    std = jnp.std(reward)
    return float(jnp.where(std > 1e-8, std, 1.0))


def sac_nets(cfg):
    """`(actor, critic)`: the package's own two modules, at `cfg`'s width."""
    return (SACActor(act_dim=3, hidden=tuple(cfg.hidden),
                     log_std_min=cfg.log_std_min, log_std_max=cfg.log_std_max),
            SoftQ(hidden=tuple(cfg.hidden)))


def sac_apply(cfg, per_agent=False):
    """`(actor_apply, critic_apply)` for one parameter layout.

    Bound HERE, in Python, at construction, for the reason `make_rollout` gives,
    and mapped by `sac._actor_per_agent` / `sac._q_per_agent` rather than by a
    second copy of the same `moveaxis`, so the agent axis of 05's SAC column is
    the agent axis of every other market's.
    """
    actor, qnet = sac_nets(cfg)
    return ((partial(_actor_per_agent, actor), partial(_q_per_agent, qnet))
            if per_agent else (actor.apply, qnet.apply))


def init_sac_policy(key, obs_dim, n, cfg, per_agent=False):
    """The full learner tree: actor, twin critics, their targets, `log_alpha`.

    `sac.make_sac` builds exactly this, and the two branches are its two
    branches: on the per-agent path every network is `vmap(init)`ed over
    `n` split keys so no two aggregators start from the same draw, and the
    temperature becomes one scalar per agent "because each independent learner
    has its own entropy target to meet" (`sac.py`).  The targets start equal to
    the critics, as they do there.

    The two layouts hold the same leaves and differ only in a leading agent
    axis, exactly as `init_policy` / `init_policy_per_agent` do, so a tree of one
    layout used as the other does not fail on shape -- it produces a different
    learner that looks like the right one.  That is why `--algo` and
    `--per-agent-params` are both stamped into the result file.
    """
    actor, qnet = sac_nets(cfg)
    ka, k1, k2 = jax.random.split(key, 3)
    z = jnp.zeros((n, obs_dim), jnp.float32)
    a0 = jnp.zeros((n, 3), jnp.float32)
    if per_agent:
        nets = dict(
            actor=jax.vmap(actor.init)(jax.random.split(ka, n), z),
            q1=jax.vmap(qnet.init)(jax.random.split(k1, n), z, a0),
            q2=jax.vmap(qnet.init)(jax.random.split(k2, n), z, a0),
            log_alpha=jnp.full((n,), jnp.log(cfg.init_alpha)))
    else:
        nets = dict(actor=actor.init(ka, z), q1=qnet.init(k1, z, a0),
                    q2=qnet.init(k2, z, a0),
                    log_alpha=jnp.asarray(jnp.log(cfg.init_alpha)))
    return dict(nets, q1_target=nets["q1"], q2_target=nets["q2"])


def sac_tree_shapes(policy):
    """``{path: [shape]}`` of a learner tree as it was actually built.

    Read off the tree rather than restated from `cfg`, so the stamp in the
    result file answers "what did this run build" and not "what did it intend to
    build" -- the distinction `run_rl_02.py` draws between the log and the
    product.
    """
    flat, _ = jax.tree_util.tree_flatten_with_path(policy)
    return {jax.tree_util.keystr(p): [list(v.shape), str(v.dtype)]
            for p, v in flat}


def sac_bounds(spec, bounds=None):
    """This market's action box, ``(low, high)``, shaped ``(n_agent, 3)``.

    Passed in rather than read from the spec because that is `make_sac`'s own
    signature -- ``make_sac(env, bounds, cfg, ...)`` -- and `bounds_for` says
    why the box belongs to the market and not to the learner.  `None` means
    "this market's", which is `bounds_for(spec)`: since 2026-09-10 that is
    ``-+ACTION_SATURATION`` on all three coordinates
    (`envs/local_flexibility/action.py`), where it was ``+-inf``
    before.

    **The default was never idle.** `bounds_for`'s docstring records that an
    unbounded coordinate is what diverged SAC on the ancillary market
    ("the critic reads the raw coordinate and whose actor maximises the critic,
    so with no bound there is no fixed point", `q_loss` 1e27 within 3 072
    updates), and that market answered by publishing a box.  05 was
    the last of the five without one.  This keyword exists so that the
    diagnostic asking whether that is the cause can be run without editing a
    shared module for the duration of a measurement, and
    `tests/tools/test_flex_sac_l0.py` uses it to hold one measurement at the
    unbounded box it was taken on.  A **run** must not use it: a box chosen by a
    learner is a second declaration of the action space, which is the thing
    `bounds_for` refuses.
    """
    return bounds_for(spec) if bounds is None else bounds


def make_sac_alpha_loss(target_entropy, per_agent=False):
    """SAC's temperature loss over one parameter layout.

    Lifted out of `make_sac_update` -- character for character, so the shared arm
    is the expression it always was -- for the reason `make_loss_fn` was lifted
    out of `make_update`: it is the one reduction the per-agent flag changes, so
    it has to be gateable without building a scenario.  `sac.py`'s own words for
    the per-agent branch: "each agent's temperature answers to its own entropy:
    the batch mean is taken per agent and the agents are summed, so the gradient
    on `log_alpha[j]` sees agent `j`'s samples only".

    The number the gate bites with is `n_agent`: a mean over the agent axis in
    place of the sum would put the per-agent gradient a factor of 24 (2040
    placement) or 34 (2050) below the shared one, and
    `tests/tools/test_flex_sac_l0.py` measures that factor rather than asserting
    it.
    """
    if per_agent:
        def alpha_loss(log_alpha, logp):
            per = jnp.mean(logp + target_entropy, axis=0)
            return jnp.sum(-jnp.exp(log_alpha) * per)
    else:
        def alpha_loss(log_alpha, logp):
            return -jnp.exp(log_alpha) * jnp.mean(logp + target_entropy)
    return alpha_loss


def sac_continuation(done, like):
    """The Q target's continuation mask: zero at the boundary.

    Lifted out of `make_sac_update` so that the stamp `learner_spec` carries is
    **computed by calling this**, not written down beside it.  `sac._cont`
    branches in Python on `ippo._BOOTSTRAP_AT_DONE[spec["termination"]]` and
    would return ones here, because 05 declares `"truncation"`; the reason this
    one does not is in `make_sac_update`'s docstring, and the reason it is a
    named function is that a stamp restating a constant reports the author's
    intention rather than the run's state.
    """
    return 1.0 - done[..., None].astype(like.dtype)


def make_sac_rollout(env, params, scale, n, cfg, per_agent=False, greedy=False,
                     bounds=None):
    """`make_rollout` with SAC's actor in place of PPO's.

    The signature of the returned function and the first fifteen of its outputs
    are `make_rollout`'s, so `evaluate` reads them by the same positions;
    `next_obs` and `done` follow, because a replayed transition has no neighbour
    in time to read its successor from -- the reason `sac._rollout` gives for the
    same two extra outputs.

    Three positions carry something different and all three are said here rather
    than left to be inferred.  Position 1 is `pre`, the pre-`ACTION_GAIN` policy
    coordinate SAC sampled, which is what position 1 already was for PPO
    (`mean + noise`).  Position 2 is `policy.log_prob` of that sample, which on
    this market equals the driver's own `log_prob` term for term because nothing
    is squashed.  Position 3 is the value estimate under PPO; SAC has no
    state-value head, so it is **zeros** and nothing reads it -- `evaluate`
    discards that position and `make_sac_update` does not collect it.

    `next_obs` is `info["terminal_obs"]`, scaled and clipped the way `obs` is.
    That field is the true successor observation at **every** step, terminal or
    not: `env.py` computes it as ``where(done, _get_obs(next_state), obs)`` where
    `obs` is already the post-`auto_reset` observation, so on an interior step
    the two branches agree and on the last step it is the successor the reset
    overwrote.  `test_terminal_obs_is_the_successor_observation` checks that
    reading against the environment rather than trusting it.

    `greedy` drops the exploration noise and returns the actor's mean, which is
    `sac.make_sac_greedy_action`'s protocol.  It is **not** the default: see the
    section note on why 05 evaluates with the policy's own width.
    """
    reset, _, step_auto, spec = env
    _actor, _ = sac_apply(cfg, per_agent=per_agent)
    low, high = sac_bounds(spec, bounds)

    def rollout(policy, key, mode, start=-1):
        k0, k1 = jax.random.split(key)
        _, state = reset(k0, params)
        state = state.replace(
            cursor=jnp.where(start >= 0, jnp.int32(start), state.cursor))

        def body(carry, kk):
            state = carry
            obs = spec["get_obs"](state, params) / scale
            obs = jnp.clip(obs, -OBS_CLIP, OBS_CLIP)
            mean, log_std = _actor(policy["actor"], obs)
            # Reparameterised, as `sac._act` does it: the sample is a
            # differentiable function of the actor's parameters, which is what
            # `_actor_loss` differentiates through.
            eps = jax.random.normal(kk, mean.shape, mean.dtype)
            pre = mean if greedy else mean + jnp.exp(log_std) * eps
            # `to_action` is the identity on this market's unbounded
            # coordinates; `ACTION_GAIN` is the market's own widening of the
            # policy coordinate (§9.3) and belongs to both learners alike.
            learned = to_action(pre, low, high) * ACTION_GAIN
            act = market_action(mode, learned, n, kk)
            lp = squashed_log_prob(pre, mean, log_std, low, high)
            _, s2, reward, costs, done, info = step_auto(kk, state, act, params)
            nxt = jnp.clip(info["terminal_obs"] / scale, -OBS_CLIP, OBS_CLIP)
            return s2, (obs, pre, lp, jnp.zeros_like(lp), reward, costs[:, 0],
                        info["traded_volume"], info["price_avg"],
                        info["z"], info["req_th_count"], info["terminal_obs"],
                        info["req_v_count"], info["mu"], info["converged"],
                        costs[:, 2], nxt, done)

        _, out = jax.lax.scan(body, state, jax.random.split(k1, EPISODE_LEN))
        (obs, pre, lp, value, reward, shed, vol, price, z, req_th,
         term, req_v, mu, converged, overload, nxt, done) = out
        # The same settlement the PPO arm gets, on the same period.  It is what
        # makes 05's episode boundary a settled one rather than a truncated one,
        # and `_cont` below is masked for exactly that reason.
        adj = terminal_settlement(params, term[-1])
        reward = reward.at[-1].add(adj)
        return (obs, pre, lp, value, reward, shed, vol, price, z, req_th,
                term, req_v, mu, converged, overload, nxt, done)

    return rollout


def make_sac_update(env, params, scale, n, cfg, per_agent=False, bounds=None,
                    target_entropy=None):
    """Build `(init_state, update, spec)` for one cell.

    `init_state` sits where `optimiser.init` sits on the IPPO path and `update`
    has that path's signature exactly -- ``(policy, opt_state, key, log_std) ->
    (policy, opt_state, stats)`` -- so `main`'s training loop is one loop and not
    two.  **`log_std` is accepted and ignored**, and the spec says so in
    `annealed_log_std_used`: 05's anneal is PPO's exploration schedule and SAC's
    width is a head of its own actor (see the section note).  A reader who wants
    to know whether the anneal reached this learner reads that key rather than
    inferring it from a signature that had to keep the position.

    **The boundary is masked, not bootstrapped, and this departs from the table
    `sac.make_sac` reads.**  That table is `ippo._BOOTSTRAP_AT_DONE`, it keys on
    `spec["termination"]`, and 05 declares `"truncation"`, which selects
    bootstrap -- its comment names 05 among the markets where "nothing is settled
    at the boundary".  That is true of the environment and false of this driver:
    `terminal_settlement` prices the stored energy into the last period's reward,
    which is exactly what `make_update` gives as its reason for a zero bootstrap
    on the PPO side -- "bootstrapping as well would pay for the carry twice, once
    explicitly and once through a value head that has learned what the carry is
    worth".  A Q function bootstrapping across that boundary would pay it twice
    the same way.  So the mask here is 04's branch of that table applied for 04's
    reason, and the two arms of this cell treat the boundary alike.

    Everything else is `sac.make_sac` line for line: twin critics on the soft
    Bellman target, the actor stepped against the freshly updated critics, the
    temperature stepped on the log-probabilities that actor step sampled, Polyak
    targets, and one `lax.scan` of ``utd_ratio * n_envs * horizon`` gradient
    steps over a fixed-size FIFO replay buffer sampled uniformly over its filled
    prefix.
    """
    import optax
    spec = env[3]
    _actor, _q = sac_apply(cfg, per_agent=per_agent)
    low, high = sac_bounds(spec, bounds)
    roll = make_sac_rollout(env, params, scale, n, cfg, per_agent=per_agent,
                            bounds=bounds)
    per_iter = cfg.n_envs * cfg.horizon
    if per_iter > cfg.buffer_size:
        raise ValueError(f"one iteration collects {per_iter} env-steps but the "
                         f"buffer holds {cfg.buffer_size}; the FIFO write would "
                         f"overwrite this iteration's own transitions")
    n_updates = int(round(cfg.utd_ratio * per_iter))
    if n_updates < 1:
        raise ValueError(f"utd_ratio {cfg.utd_ratio} gives {n_updates} updates "
                         f"per iteration")
    if not (cfg.reward_scale > 0.0) or not np.isfinite(cfg.reward_scale):
        raise ValueError(f"reward_scale={cfg.reward_scale!r} must be a finite "
                         f"positive number; fit it with `sac_reward_scale`")
    #: CleanRL's ``-dim(A)`` (`sac.py:254`), taken literally: minus the action
    #: dimension, **not rescaled by this market's box**.  Between 2026-09-10
    #: and 2026-09-11 this line computed ``-act_dim + sum_d log(0.5 w_d)``
    #: instead; that was reverted, and the two reasons are recorded here so the
    #: derivation is not repeated from scratch.
    #:
    #: **One, the rule has to be the same across the five markets.**
    #: `sac.py:254` is what 01, 02, 03 and 04 run under, and none of their
    #: boxes is ``[-1,1]`` either -- the per-coordinate correction
    #: ``log(0.5 w)`` is -0.6931 on 01 and 02, +2.5255 on 03's reserve columns
    #: and exactly 0 on 04.  Scaling here would make 05 the only market whose
    #: target moves with its box, and an asymmetry between the markets needs a
    #: principled reason rather than a local one.
    #:
    #: **Two, the literature gives the value, not the rescaling rule.**
    #: 1812.05905v2 table 1 gives the entropy target as ``-dim(A)``; nothing in
    #: either SAC paper says to rescale it by the box a market happens to
    #: publish.  Rescaling is an interpretation of why that value was chosen;
    #: ``-dim(A)`` is what the paper says.
    #:
    #: **What it costs on this box, measured, so that nobody re-derives it.**
    #: The scale of a squashed policy's entropy does move with the box:
    #: `policy.log_prob` subtracts ``log(0.5 (high - low))`` per squashed
    #: coordinate, measured on 05's box as exactly ``3 log 128 = 14.55609`` at
    #: every `log_std` from -5 to +2, a pure additive constant.  So on
    #: ``+-128`` the target ``-3`` sits below the entropy the actor can emit at
    #: its *initial* parameters (envelope ``[0.98, 16.46]`` over the whole
    #: `log_std` range, measured CPU, cell 0, 4 096 draws), and early in
    #: training the constraint is slack and `alpha` decays.  **That is a
    #: transient, not the fixed point**: `-3` becomes attainable once ``|mean|``
    #: reaches about 2, and reading C-AO, on a remote A10G over the full 300
    #: iterations, finds the run settling there -- entropy steady at -2.9, i.e.
    #: on its target, `alpha` in ``[1.2e-3, 1.2e-2]``, `q_loss` in
    #: ``[1.63e3, 9.71e3]``, no non-finite values, `eval` rising 1.84 -> 28.87.
    #: The scaled target reached its own value instead by letting the
    #: multiplier diverge -- `alpha` 0.23 -> 1.3 -> 5.4 -> 50 -> 306 and, since
    #: the Q target carries ``-alpha log pi``, `q_loss` 1.6e3 -> 4.8e8 with
    #: `q_mean` at 6.4e4, again with no non-finite values.  **So the cost of
    #: the literal value is a slack constraint early on, and the cost of the
    #: scaled one was a diverging Lagrange multiplier.**
    #:
    #: `target_entropy` is accepted as a keyword for the same reason
    #: `sac_bounds` accepts `bounds`: so a diagnostic can hold another value
    #: without editing a shared module for the duration of a measurement.  A
    #: run must not pass it.
    if target_entropy is None:
        target_entropy = -float(np.asarray(low).shape[-1])
    else:
        target_entropy = float(target_entropy)
    tx = (optax.adam(cfg.policy_lr), optax.adam(cfg.q_lr),
          optax.adam(cfg.alpha_lr))

    _alpha_loss = make_sac_alpha_loss(target_entropy, per_agent=per_agent)

    def init_state(policy):
        # Shapes and dtypes are read from the rollout by `eval_shape`, which
        # traces without executing, rather than written down here.  They are not
        # obvious: the observation is float32 (the environment's dtype, divided
        # by a float32 scale) while this driver runs under x64, so a buffer
        # typed by hand would store a float64 copy of a float32 number or
        # silently narrow one, and neither would announce itself.
        probe = jax.eval_shape(
            lambda p: roll(p, jax.random.PRNGKey(0), jnp.int32(0)), policy)
        slot = lambda s: jnp.zeros((cfg.buffer_size,) + s.shape[1:], s.dtype)
        buffer = dict(obs=slot(probe[0]), pre=slot(probe[1]),
                      reward=slot(probe[4]), next_obs=slot(probe[15]),
                      done=slot(probe[16]),
                      cursor=jnp.asarray(0, jnp.int32),
                      filled=jnp.asarray(0, jnp.int32))
        return dict(opt_actor=tx[0].init(policy["actor"]),
                    opt_q=tx[1].init((policy["q1"], policy["q2"])),
                    opt_alpha=tx[2].init(policy["log_alpha"]), buffer=buffer)

    def _push(buffer, traj):
        """FIFO write of one rollout's ``n_envs * horizon`` env-steps.

        The rollout stacks ``(n_envs, horizon, ...)`` where `sac._rollout` stacks
        ``(horizon, n_envs, ...)``, so the axes are swapped before flattening.
        Uniform sampling does not care about the order, but a partial wrap does
        -- 32 768 is not a multiple of 1 536 -- and matching the package costs
        one `moveaxis`.
        """
        idx = (buffer["cursor"] + jnp.arange(per_iter)) % cfg.buffer_size
        flat = lambda x: jnp.moveaxis(x, 0, 1).reshape((per_iter,) + x.shape[2:])
        out = dict(buffer)
        for k in ("obs", "pre", "reward", "next_obs", "done"):
            out[k] = buffer[k].at[idx].set(flat(traj[k]))
        out["cursor"] = (buffer["cursor"] + per_iter) % cfg.buffer_size
        out["filled"] = jnp.minimum(buffer["filled"] + per_iter, cfg.buffer_size)
        return out

    def _sample(buffer, key):
        idx = jax.random.randint(key, (cfg.batch_size,), 0, buffer["filled"])
        return {k: buffer[k][idx] for k in ("obs", "pre", "reward", "next_obs",
                                            "done")}

    def _q_loss(q_params, policy, batch, key):
        """Twin-Q regression on the soft Bellman target.

        No `_norm` here and none in `_actor_loss`: this driver scales the
        observation inside the rollout, so what the buffer holds is already what
        the network reads.  `sac.make_sac` normalises in the loss because its
        buffer holds the raw observation.
        """
        q1, q2 = q_params
        z, z_next = batch["obs"], batch["next_obs"]
        mean, log_std = _actor(policy["actor"], z_next)
        pre_next = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape,
                                                               mean.dtype)
        logp_next = squashed_log_prob(pre_next, mean, log_std, low, high)
        a_next = _q_action(pre_next, low, high)
        q_next = jnp.minimum(_q(policy["q1_target"], z_next, a_next),
                             _q(policy["q2_target"], z_next, a_next))
        alpha = jnp.exp(policy["log_alpha"])
        target = (batch["reward"] / cfg.reward_scale
                  + cfg.gamma * sac_continuation(batch["done"], q_next)
                  * (q_next - alpha * logp_next))
        target = jax.lax.stop_gradient(target)
        a = _q_action(batch["pre"], low, high)
        q1_pred, q2_pred = _q(q1, z, a), _q(q2, z, a)
        loss = jnp.mean((q1_pred - target) ** 2) + jnp.mean((q2_pred - target) ** 2)
        return loss, dict(q_loss=loss, q_mean=jnp.mean(q1_pred),
                          target_mean=jnp.mean(target))

    def _actor_loss(actor_params, policy, batch, key):
        z = batch["obs"]
        mean, log_std = _actor(actor_params, z)
        pre = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape,
                                                          mean.dtype)
        logp = squashed_log_prob(pre, mean, log_std, low, high)
        a = _q_action(pre, low, high)
        q = jnp.minimum(_q(policy["q1"], z, a), _q(policy["q2"], z, a))
        alpha = jax.lax.stop_gradient(jnp.exp(policy["log_alpha"]))
        loss = jnp.mean(alpha * logp - q)
        return loss, dict(actor_loss=loss, logp=jax.lax.stop_gradient(logp))

    @jax.jit
    def update(policy, opt_state, key, log_std):
        del log_std                     # see the docstring: PPO's schedule
        key, ck, k_upd = jax.random.split(key, 3)
        out = jax.vmap(roll, in_axes=(None, 0, None))(
            policy, jax.random.split(ck, cfg.n_envs), 0)
        obs, pre, reward = out[0], out[1], out[4]
        shed, vol, price = out[5], out[6], out[7]
        next_obs, done = out[15], out[16]
        buffer = _push(opt_state["buffer"],
                       dict(obs=obs, pre=pre, reward=reward,
                            next_obs=next_obs, done=done))

        def one(carry, k):
            policy, o_actor, o_q, o_alpha = carry
            k_batch, k_q, k_pi = jax.random.split(k, 3)
            batch = _sample(buffer, k_batch)
            (_, q_aux), g_q = jax.value_and_grad(_q_loss, has_aux=True)(
                (policy["q1"], policy["q2"]), policy, batch, k_q)
            upd, o_q = tx[1].update(g_q, o_q, (policy["q1"], policy["q2"]))
            q1, q2 = optax.apply_updates((policy["q1"], policy["q2"]), upd)
            policy = dict(policy, q1=q1, q2=q2)
            (_, pi_aux), g_pi = jax.value_and_grad(_actor_loss, has_aux=True)(
                policy["actor"], policy, batch, k_pi)
            upd, o_actor = tx[0].update(g_pi, o_actor, policy["actor"])
            policy = dict(policy,
                          actor=optax.apply_updates(policy["actor"], upd))
            a_loss, g_a = jax.value_and_grad(_alpha_loss)(policy["log_alpha"],
                                                          pi_aux["logp"])
            upd, o_alpha = tx[2].update(g_a, o_alpha, policy["log_alpha"])
            policy = dict(policy, log_alpha=optax.apply_updates(
                policy["log_alpha"], upd))
            polyak = lambda t, s: (1.0 - cfg.tau) * t + cfg.tau * s
            policy = dict(policy,
                          q1_target=jax.tree.map(polyak, policy["q1_target"],
                                                 policy["q1"]),
                          q2_target=jax.tree.map(polyak, policy["q2_target"],
                                                 policy["q2"]))
            aux = dict(q_loss=q_aux["q_loss"], q_mean=q_aux["q_mean"],
                       target_mean=q_aux["target_mean"],
                       actor_loss=pi_aux["actor_loss"], alpha_loss=a_loss,
                       # `-logp` is the sampled entropy, per agent
                       entropy=-jnp.mean(pi_aux["logp"]),
                       alpha=jnp.mean(jnp.exp(policy["log_alpha"])))
            return (policy, o_actor, o_q, o_alpha), aux

        (policy, o_actor, o_q, o_alpha), aux = jax.lax.scan(
            one, (policy, opt_state["opt_actor"], opt_state["opt_q"],
                  opt_state["opt_alpha"]), jax.random.split(k_upd, n_updates))
        opt_state = dict(opt_actor=o_actor, opt_q=o_q, opt_alpha=o_alpha,
                         buffer=buffer)
        stats = dict(jax.tree.map(jnp.mean, aux),
                     buffer_filled=buffer["filled"],
                     ret=reward.sum(1).mean(), shed=shed.sum(1).mean(),
                     volume=vol.mean(), price=price.mean())
        return policy, opt_state, stats

    #: `algo` is a literal of the builder that was actually called, not a copy of
    #: the caller's `--algo`, so a product cannot be stamped with an intention
    #: the run did not carry out.
    learner_spec = dict(
        algo="sac", per_agent_params=bool(per_agent), n_agent=int(n),
        act_dim=3, target_entropy=target_entropy,
        annealed_log_std_used=False,
        #: Asked of `sac_continuation` rather than restated: a stamp that copies
        #: a constant standing beside the code reports the intention, and this is
        #: the one field a reader would use to decide whether 05's SAC treats the
        #: boundary the way 05's PPO does.
        bootstrap_at_done=bool(
            jnp.all(sac_continuation(jnp.ones((1,), jnp.bool_),
                                     jnp.zeros((1, 1))) == 1.0)),
        boundary_note=("masked, not bootstrapped: terminal_settlement prices "
                       "the carry into the last period's reward, so a Q "
                       "bootstrap across the boundary would pay it twice"),
        env_steps_per_iteration=int(per_iter),
        n_envs=int(cfg.n_envs), horizon=int(cfg.horizon),
        updates_per_iteration=int(n_updates),
        eval_is_greedy=False,
        #: What the critic's action coordinate is bounded by.  On this market
        #: nothing bounds it; the value is stamped rather than assumed because
        #: it is what `bounds_for` names as the reason SAC diverged elsewhere.
        action_low=float(jnp.min(low)), action_high=float(jnp.max(high)),
        hyperparams={k: (list(v) if isinstance(v, tuple) else v)
                     for k, v in vars(cfg).items()})
    return init_state, update, learner_spec


#: The value of lost load, which prices the operator's backstop.  Imported
#: rather than restated so that the two cannot drift apart.
from powermarketjax.envs.local_flexibility.clearing import VOLL  # noqa: E402


def evaluate(env, params, scale, n, policy, key, mode, log_std,
             eval_starts=None, per_agent=False, algo="ippo", sac_cfg=None,
             greedy=False):
    """Every reported quantity, with the axis it is aggregated over named.

    Curtailment is returned beside the return and not below it.  It is the
    feasibility backstop of §6, so it can never be zero by construction, but a
    configuration in which it is the usual outcome is a curtailment scheme with
    a market attached; reporting it as a footnote would let that pass.

    `per_agent` must match the `policy` being passed.  Crossing the two is
    caught, but by `vmap` rather than by anything here: the leading axis of a
    shared `w1` is `obs_dim`, so mapping it against `n_agents` observations
    raises on the inconsistent sizes -- 23 against 24 or 34 on this scenario.
    That guard is why the flag is a keyword rather than something inferred from
    the leaf shapes: inference would have to guess which of those two numbers a
    leading axis is on, and on a scenario where they coincided it would guess
    silently.

    `algo` must match the `policy` too, and crossing THAT is caught by nothing
    here: an IPPO tree has keys ``w1 ... value_b`` and an SAC tree has
    ``actor, q1, q2, q1_target, q2_target, log_alpha``, so the wrong path raises
    a `KeyError` on the first lookup rather than producing numbers.  `log_std`
    and `sac_cfg` are each read by exactly one path -- `log_std` by IPPO, whose
    exploration width the training loop anneals, and `sac_cfg` by SAC, whose
    width is a head of its own actor -- and `greedy` by SAC alone; the SAC
    section says why 05 evaluates with the policy's own width by default.
    """
    roll = (make_rollout(env, params, scale, n, log_std, per_agent=per_agent)
            if algo == "ippo" else
            make_sac_rollout(env, params, scale, n, sac_cfg,
                             per_agent=per_agent, greedy=greedy))
    # One episode per start in the pool, enumerated rather than sampled, so the
    # reported number is the mean over the held-out days and not over a random
    # multiset of them.  `starts` is the whole `cursor_pool`; on an unrestricted
    # scenario that is the entire year, which would be far more episodes than
    # the runs need, so the caller passes the pool it wants scored.
    starts = jnp.asarray(params.cursor_pool, jnp.int32) if eval_starts is None \
        else jnp.asarray(eval_starts, jnp.int32)
    # The SAC rollout appends `next_obs` and `done` after the fifteen positions
    # this reads; nothing here consumes them.
    (_, _, _, _, reward, shed, vol, price, z, req_th, _,
     req_v, mu, converged, overload, *_) = jax.vmap(
        roll, in_axes=(None, 0, None, 0))(
            policy, jax.random.split(key, starts.shape[0]), mode, starts)
    (reward, shed, vol, price, z, req_th, req_v, mu, converged,
     overload) = map(np.asarray, (reward, shed, vol, price, z, req_th, req_v,
                                  mu, converged, overload))
    # The requirement has two families (§4) and `served` has to count both; the
    # previous version used the thermal count alone, and its own reference arm
    # showed 13% of curtailing periods carrying no thermal requirement at all.
    req = np.maximum(req_th, req_v)
    # `costs` is a system quantity replicated along the agent axis (env.py:519),
    # so one column is the whole of it; summing the axis would multiply the
    # curtailed energy by the population, which differs across the cells.
    shed = shed[..., 0]
    cleared = vol > 1e-9                       # periods where anything cleared
    binding = req > 0                          # periods carrying a requirement
    curtailing = shed > 1e-6
    return dict(
        # per participant, summed over the 24 periods, then averaged
        ret=float(reward.sum(1).mean()),
        ret_per_agent=reward.sum(1).mean(0).tolist(),
        # system totals per episode
        market_total=float(reward.sum(1).sum(1).mean()),
        shed_mwh=float(shed.sum(1).mean()),
        # The backstop as a share of what the operator spends, reported first.
        # Summed over periods and then divided, not averaged period by period:
        # a per-period ratio contributes zero on every period where the operator
        # spends nothing, so the mean of ratios is diluted by how often the
        # market is idle.  Measured on the arm that never clears, whose spend is
        # 100% curtailment by construction, the mean of ratios read 30%.
        # `shed` is already an energy in MWh, so the cost is VOLL times it and
        # multiplying by the period length again would double count.
        curtail_share=float(VOLL * shed.sum() / max(z.sum(), 1e-9)),
        curtail_periods=float(curtailing.mean()),
        served=float(1.0 - (curtailing & binding).sum() / max(binding.sum(), 1)),
        # price conditional on clearing, which is the price; the all-period
        # mean is zero on periods that cleared nothing and is not one
        price_given_cleared=float(price[cleared].mean()) if cleared.any() else 0.0,
        clearing_fraction=float(cleared.mean()),
        duty=float(binding.mean()),
        volume_given_cleared=float(vol[cleared].mean()) if cleared.any() else 0.0,
        volume_all=float(vol.mean()),
        # Solver gate.  `award` feeds `reward` continuously with no absorber, so
        # a diagnostic has to be in the gate; `mu` is the one
        # `clearing.py` names and the previous version collected none of them
        # across roughly 10^5 solves.
        mu_max=float(mu.max()), sweep_converged=bool(converged.all()),
        req_v_periods=float((req_v > 0).mean()),
        # §7 on the **cleared** point, per unit and never summed across
        # variants without saying so: `overload` is what the sweep still finds
        # after the clearing acted, so a variant that procures against a
        # counterfactual baseline shows its cost here and nowhere else.
        overload_periods=float((overload[..., 0] > OVERLOAD_EPS).mean()),
        overload_max=float(overload[..., 0].max()),
        overload_mean=float(overload[..., 0].mean()))


def iqm(x):
    x = np.sort(np.asarray(x).ravel())
    lo, hi = int(0.25 * len(x)), int(np.ceil(0.75 * len(x)))
    return float(x[lo:hi].mean())


def bootstrap_ci(x, draws=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x).ravel()
    stats = [iqm(rng.choice(x, len(x), replace=True)) for _ in range(draws)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


#: Declared before the runs rather than read off the curves.  A cell is called
#: converged when the interquartile mean of the evaluation return over the last
#: `PLATEAU_WINDOW` checkpoints moves by less than `PLATEAU_TOL` relative to the
#: window before it.  The previous version stopped at a fixed 300 iterations and
#: the ordering it reported existed only after iteration 120, so the stopping
#: point was doing part of the work.
PLATEAU_WINDOW = 5
PLATEAU_TOL = 0.05

#: The ceiling equals `SCHEDULE_LEN` because beyond it no iteration can move the
#: policy.  `optax.linear_schedule(3e-4, 0.0, SCHEDULE_LEN * EPOCHS)` returns
#: 1.8e-11 at gradient step `SCHEDULE_LEN * EPOCHS` and every step after it
#: (measured 2026-08-17), and the exploration width is clipped at
#: `LOG_STD_FINAL` by the `min(..., 1.0)` in `main`.  Evaluation is deterministic
#: given a policy and `select_key`, so the checkpoints past this point are not
#: merely uninformative, they are bit-identical to the one at it, and the
#: previous ceiling of 600 spent half the wall clock re-scoring a frozen policy.
#: This is safe only because the reported policy is the best checkpoint rather
#: than the endpoint; under endpoint reporting a fixed ceiling would put the
#: stopping point back into the result.
MAX_ITERATIONS = 300

#: The horizon the step size and the exploration width are annealed over, held
#: apart from the stopping ceiling so that where a cell stops does not change
#: how far its schedules have travelled.
SCHEDULE_LEN = 300


def plateaued(hist):
    if len(hist) < 2 * PLATEAU_WINDOW:
        return False
    ev = np.array([h["eval"] for h in hist], float)
    a = ev[-2 * PLATEAU_WINDOW:-PLATEAU_WINDOW].mean()
    b = ev[-PLATEAU_WINDOW:].mean()
    # A falling curve satisfies a two-sided tolerance exactly as a flat one
    # does.  The previous version tested only `abs(b - a)`, and every one of the
    # twenty runs it stopped was arrested on a decline, between 12% and 87%
    # below its own peak -- so the reported endpoint was a property of where the
    # rule fired rather than of where the policy settled.  Requiring the later
    # window not to be below the earlier one makes it a plateau test.
    return abs(b - a) <= PLATEAU_TOL * max(abs(a), 1.0) and b >= a


def main():
    import optax
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    # First seed index.  With `--seeds` it names a half-open range, so one
    # cell can be split across two cards and rejoined: the seed *is* the
    # PRNG key, so seed k is the same run wherever it is computed, and the
    # two part files hold disjoint seeds of one cell rather than two
    # samples of it.
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--out", default="docs/figures/flex-concentration")
    # The four cells are independent -- each builds its own case, its own
    # parameters and its own policies, and nothing is carried between them -- so
    # running them in one process is a scheduling choice and not a requirement.
    # Splitting them across cards is what makes the study fit in an evening: one
    # rollout is 24 periods times 80 interior point steps of strictly sequential
    # dense factorisation, which no amount of device parallelism shortens.
    # `--tag` keeps the part files apart; `merge_results` joins them.
    ap.add_argument("--cells", default="",
                    help="comma separated indices into CELLS; empty means all")
    #: The parameter layout.  Named `--per-agent-params` because that is the
    #: name `run_rl_01.py`, `run_rl_02.py` and `run_rl_03.py` already use for the
    #: same treatment; a different name here would make the same column of the
    #: five-market matrix read as a different thing.  The two layouts hold the
    #: same leaves and differ only in a leading agent axis, so a result file
    #: cannot be told apart from its numbers -- which is why the flag is stamped
    #: into `results[name]` below and not left to be inferred.
    ap.add_argument("--per-agent-params", action="store_true",
                    help="give every aggregator its own copy of the policy and "
                         "value network instead of one shared copy. Reductions "
                         "are unchanged: the loss means over the agent axis and "
                         "the advantages take one scale, exactly as "
                         "powermarketjax/learning/ippo.py does under the same "
                         "flag (see make_update)")
    #: Which learner.  `ippo` is the default and the path every
    #: result on disk was produced on, and a run without this flag writes the
    #: file it always wrote **byte for byte** -- including the absence of the
    #: two stamps below, because a column of this study is in flight on three
    #: cards as this is written and its part files are joined by a merge
    #: script, which would then be joining two formats.  So a
    #: results file WITHOUT an `algo` key was written by the IPPO path; the
    #: positive stamp on that path costs a re-run of the column and is not
    #: worth one today.
    ap.add_argument("--algo", choices=("ippo", "sac"), default="ippo",
                    help="learner: ippo (default, the existing path) or sac "
                         "(off-policy). Stamped as `algo` and `learner_spec` in "
                         "the result file. The name matches run_rl_01/02/03.py "
                         "so the same column of the five-market matrix reads "
                         "as the same thing")
    ap.add_argument("--tag", default="")
    #: How many aggregators are pinned to the truthful/floor baseline instead of
    #: learning (`floor_learner_mask`).  Zero is the published population and
    #: writes the file it always wrote.  The sweep this exists for asks whether
    #: the learners still collapse to the floor when a share of the population
    #: is a non-strategic price taker at cost; `n - 1` is figure 4's right-panel
    #: landscape, which no run before this flag ever trained in.
    ap.add_argument("--floor-k", type=int, default=0,
                    help="pin this many aggregators to the non-learner "
                         "baseline (truthful price = floor, full quantity, no "
                         "planned charge); 0 is the published population")
    #: The plateau rule stopped 5 of the 5 escaping seeds between iteration 230
    #: and 250 of 300, so "it was still above the floor at the end" and "it had
    #: not fallen back yet" are the same observation on those runs.  This flag
    #: runs the ceiling out so the two can be told apart.  Off by default: every
    #: existing result on disk was produced with the rule on.
    ap.add_argument("--no-stop", action="store_true",
                    help="ignore the plateau stopping rule and run every seed "
                         "to --max-iterations")
    #: The monitored variant of the specification's section 8, already a keyword
    #: of `scenario` and reachable from nowhere else: the operator refuses to
    #: publish a requirement against a baseline the participant positioned
    #: itself.  Off by default, so a run without it is the run it always was.
    #: It exists because the unmonitored arm measured here shows the learners
    #: raising traded volume to 52x the truthful arm while their price stays at
    #: the floor -- a paired run is what decides how much of that is positioning.
    ap.add_argument("--monitor", action="store_true",
                    help="monitored requirement (section 8): the operator will "
                         "not publish a requirement against a baseline the "
                         "participant positioned itself")
    args = ap.parse_args()
    # Zero iterations is the untrained control: the training loop below never
    # executes, so there is no curve, no checkpoint selection and no stopping
    # point, while the evaluation of the initial policy is unchanged.  Every
    # place that reads the curve branches on THIS and not on `hist` being empty,
    # so a history that comes out empty any other way still raises there.
    untrained = args.max_iterations == 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    cells = (CELLS if not args.cells
             else tuple(CELLS[int(i)] for i in args.cells.split(",")))

    for place, cap in cells:
        name = f"{place}p_{cap}c"
        # Training draws only from the training days; the two held-out sets are
        # passed to `evaluate` explicitly, which pins each episode to a start
        # rather than sampling, so `params.cursor_pool` never selects them.
        env, params, n = scenario(place, cap, split="train",
                                  floor_k=args.floor_k, monitor=args.monitor)
        scale = obs_scale(env, params, n)
        # Three disjoint sets of days, none of which the learner trains on.
        # The checkpoint that gets reported is chosen on the **validation** days
        # and then scored on the **test** days: choosing and scoring on the same
        # episodes makes the score the maximum of a noisy quantity over the
        # checkpoints rather than an estimate of the chosen policy, and doing
        # either on a training day makes it an in-sample number.
        _, _dom = series_day_of_month()
        _, val_starts, test_starts = day_split(_dom, EPISODE_LEN)
        select_key = jax.random.PRNGKey(12345)
        report_key = jax.random.PRNGKey(999)
        # Fitted once per cell, before any seed, and frozen: the critic's reward
        # scale is a property of this cell's reward distribution and not of the
        # run.  Its key is its own, so nothing on the IPPO path moves.
        sac_cfg = sac_prov = None
        if args.algo == "sac":
            r_scale = sac_reward_scale(env, params, scale, n,
                                       jax.random.PRNGKey(4242))
            sac_cfg, sac_prov = sac_config(r_scale)
            print(f"  {name} SAC reward_scale = {r_scale:.6e} (pooled std of "
                  f"the per-agent reward under the truthful arm over "
                  f"{BATCH} x {EPISODE_LEN} env-steps, terminal settlement "
                  f"included)", flush=True)
        # Passed to every `evaluate` call below, so the five call sites cannot
        # disagree about which learner they are scoring.
        ev_kw = dict(per_agent=args.per_agent_params, algo=args.algo,
                     sac_cfg=sac_cfg)
        curves, finals = [], []
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            key = jax.random.PRNGKey(seed)
            # Branched in Python, so the shared arm is the call it always was.
            if args.algo == "sac":
                policy = init_sac_policy(key, env[3]["obs_dim"], n, sac_cfg,
                                         per_agent=args.per_agent_params)
                init_state, update, lspec = make_sac_update(
                    env, params, scale, n, sac_cfg,
                    per_agent=args.per_agent_params)
                # Read off the tree that was built, not off `sac_cfg`.
                lspec = dict(lspec, param_shapes=sac_tree_shapes(policy),
                             hyperparams_provenance=sac_prov)
            else:
                policy = (init_policy_per_agent(key, env[3]["obs_dim"], n)
                          if args.per_agent_params
                          else init_policy(key, env[3]["obs_dim"]))
                optimiser, update = make_update(env, params, scale, n,
                                                args.max_iterations,
                                                per_agent=args.per_agent_params)
                init_state, lspec = optimiser.init, None
            opt_state = init_state(policy)
            hist = []
            best = (-np.inf, policy, 0)
            t0 = time.perf_counter()
            for it in range(args.max_iterations):
                frac = min(it / max(SCHEDULE_LEN - 1, 1), 1.0)
                log_std = LOG_STD_INIT + frac * (LOG_STD_FINAL - LOG_STD_INIT)
                key, sub = jax.random.split(key)
                policy, opt_state, stats = update(policy, opt_state, sub, log_std)
                if it % args.every == 0 or it == args.max_iterations - 1:
                    ev = evaluate(env, params, scale, n, policy, select_key,
                                  0, LOG_STD_FINAL,
                                  eval_starts=val_starts, **ev_kw)
                    if ev["ret"] > best[0]:
                        best = (ev["ret"], jax.tree.map(lambda a: a, policy), it)
                    hist.append(dict(it=it, train=float(stats["ret"]),
                                     eval=ev["ret"],
                                     shed=ev["shed_mwh"],
                                     curtail_share=ev["curtail_share"],
                                     served=ev["served"],
                                     price=ev["price_given_cleared"],
                                     clearing=ev["clearing_fraction"],
                                     volume=ev["volume_given_cleared"],
                                     # SAC's own diagnostics, which have no IPPO
                                     # counterpart (`sac.py`: the two entropy
                                     # terms are not the same quantity), so an
                                     # IPPO curve point keeps its eight keys.
                                     **({} if args.algo == "ippo" else
                                        {k: float(stats[k])
                                         for k in SAC_DIAGNOSTICS})))
                    if not args.no_stop and plateaued(hist):
                        break
            # The reported policy is the best checkpoint, not the last.  Every
            # run of the previous version ended below its own peak -- by 41% on
            # average and by 89% in the worst seed -- and no stopping rule
            # caught it, because the curve oscillates and a plateau test fires
            # on a local rise far below the global best.  Selecting the best
            # checkpoint on held-out episodes and scoring it on a disjoint set
            # answers "how well can this environment be played" rather than
            # "where did the optimiser happen to be arrested".
            final = evaluate(env, params, scale, n, best[1], report_key, 0,
                             LOG_STD_FINAL, eval_starts=test_starts, **ev_kw)
            final["selected_at"] = None if untrained else best[2]
            final["select_score"] = None if untrained else best[0]
            final["last_score"] = evaluate(
                env, params, scale, n, policy, report_key, 0, LOG_STD_FINAL,
                eval_starts=test_starts, **ev_kw)["ret"]
            if args.algo == "sac":
                # The cross-market reading, in the same file as the within-row
                # one: `ret` above scores the policy with its own exploration
                # width, which is 05's protocol for both arms, while
                # `sac.make_sac_greedy_action` reports the mean and that is the
                # protocol the other markets' SAC columns were scored under.
                final["ret_greedy"] = evaluate(
                    env, params, scale, n, best[1], report_key, 0,
                    LOG_STD_FINAL, eval_starts=test_starts,
                    **dict(ev_kw, greedy=True))["ret"]
            curves.append(hist)
            finals.append(final)
            stopped = None if untrained else hist[-1]["it"]
            print(f"  {name} seed {seed}: eval {final['ret']:.1f}  "
                  f"last {final['last_score']:.1f}  "
                  f"sel@{final['selected_at']}  "
                  f"price|cleared {final['price_given_cleared']:.1f}  "
                  f"curtail {final['curtail_share']:.1%}  "
                  f"served {final['served']:.1%}  "
                  f"mu {final['mu_max']:.1e}  "
                  f"stopped at {stopped}  "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)

        # The four reference arms ignore the policy -- `make_rollout` selects
        # their action by `mode` and the network output goes nowhere -- so they
        # are the same numbers under either layout.  `dummy` is still built in
        # the layout of the run, because the forward pass is evaluated whatever
        # the mode is and a crossed layout would fail there rather than be
        # ignored.
        # Built in the layout AND the algorithm of the run, for the same reason:
        # the forward pass is evaluated whatever the mode is, so a tree of the
        # other learner would raise there rather than be ignored.
        if args.algo == "sac":
            dummy = init_sac_policy(jax.random.PRNGKey(0), env[3]["obs_dim"], n,
                                    sac_cfg, per_agent=args.per_agent_params)
        else:
            dummy = (init_policy_per_agent(jax.random.PRNGKey(0),
                                           env[3]["obs_dim"], n)
                     if args.per_agent_params
                     else init_policy(jax.random.PRNGKey(0), env[3]["obs_dim"]))
        ref = {nm: evaluate(env, params, scale, n, dummy, report_key, m,
                            LOG_STD_FINAL, eval_starts=test_starts, **ev_kw)
               for nm, m in (("truthful", 1), ("random", 2),
                             ("never_clears", 3), ("incdec", 4))}
        rets = [f["ret"] for f in finals]
        peaks = [] if untrained else [max(h["eval"] for h in c) for c in curves]
        results[name] = dict(
            placement=place, capacity=cap, n_agent=n,
            curves=curves, final=finals, reference=ref,
            # Computed here rather than by hand afterwards: the previous version
            # defined `bootstrap_ci` and never called it, so the intervals it
            # printed came from outside the committed script.
            ret_iqm=iqm(rets), ret_ci=list(bootstrap_ci(rets)),
            # The peak beside the endpoint, so that a run arrested on a decline
            # is visible in the record rather than only in the curves.
            peak_iqm=None if untrained else iqm(peaks),
            peak_over_final=(None if untrained
                             else iqm(peaks) / max(iqm(rets), 1e-9)),
            stopped_at=[] if untrained else [c[-1]["it"] for c in curves],
            max_iterations=args.max_iterations, untrained_baseline=untrained,
            # Stamped, not inferred: the two parameter layouts produce files with
            # the same keys and the same shapes, so nothing else in here says
            # which one wrote it.
            per_agent_params=bool(args.per_agent_params),
            # Stamped only when they are not their defaults, for the reason the
            # `--algo` flag gives: a run without either writes the file it
            # always wrote byte for byte, so a results file WITHOUT these keys
            # was produced by the published population and the plateau rule.
            # `learner_indices` is read off the array that was built, not
            # recomputed from `floor_k` -- `final[].ret_per_agent` has to be
            # restricted to the learners before it means anything at large
            # `floor_k`, and a consumer that guessed the set from `floor_k`
            # alone would silently pick the wrong agents.
            **({} if (args.floor_k == 0 and not args.no_stop
                     and not args.monitor) else dict(
                floor_k=int(args.floor_k), no_stop=bool(args.no_stop),
                monitor_baseline=bool(args.monitor),
                learner_indices=[int(i) for i in
                                 np.flatnonzero(np.asarray(params.learner_mask))])),
            # Added only on the SAC path, so a run without `--algo` writes the
            # file it always wrote byte for byte (see the flag's own note).
            # `learner_spec["algo"]` is a literal of the builder that ran, so it
            # reports what was built and not what was asked for.
            **({} if args.algo == "ippo"
               else dict(algo="sac", learner_spec=lspec)))
        print(f"{name}: learner IQM {iqm([f['ret'] for f in finals]):.1f}  "
              f"truthful {ref['truthful']['ret']:.1f}  "
              f"random {ref['random']['ret']:.1f}  "
              f"never {ref['never_clears']['ret']:.1f}  "
              f"incdec {ref['incdec']['ret']:.1f}", flush=True)
        (out / f"results{args.tag}.json").write_text(
            json.dumps(results, indent=1, default=float))
        print(f"  checkpointed", flush=True)

    print(f"\nwrote {out / f'results{args.tag}.json'}")


if __name__ == "__main__":
    main()
