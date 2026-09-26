"""The best fixed markup for markets 01, 02 and 03.

No longer a column of the baseline matrix's table one (removed 2026-09-06;
the results that rest on this apparatus and its products cite it by footnote
instead).  `arms.markup_grid` does the sweep; this file is only the
three environment builds, the day sets and the product, in the shape the
other baselines already use.

**What it reports is a profit optimum over a grid, not a market outcome.**
`arms.rollout_action` returns the arm's reward and the CMDP cost vector, which
is what a fixed-markup policy is chosen on; it does not return the settlement
detail (`shed_mwh`, production cost) that `evaluation.system_cost` needs, so
this product carries no system cost.  The paired number is obtained by running
the winning level through that market's own evaluation driver, which is the same
code path the honest and constant arms already went through -- and when the
winner is the action-space endpoint, that run has already happened and is the
`constant` arm's product.

**Three things this file exists to keep straight.**

1. The days are the twelve of report §2.2, reached through `evaluation.open_day`
   exactly as the other arms reach them.  Market 02's evaluation environment
   holds only the held-out days, so its day indices are positions in that subset
   and the window days are recovered for the record.
2. Market 03's action is `(n_units, 3)` and only its first column is a markup;
   the other two are reserve offers whose truthful value is far outside the
   markup range, so it passes its own `action_of` and the default would sweep
   the wrong axis.
3. `at_upper_bound` is written into the product and into the printed summary.
   When it is true the number is a **lower bound** on what a fixed markup
   achieves.  `grid_spans_action_space` says which kind of upper bound it is:
   a grid too narrow, or a grid covering the whole action space whose optimum
   sits on the endpoint.

Every scenario constant is imported from the driver that already owns it, so
this arm cannot drift from the arms it will be tabulated beside.  CPU: the other
01 baselines were produced on CPU (`run_eval_01` says so), and a platform is one
of the items a "are these two comparable" judgement reads.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arms                                                      # noqa: E402
from curve_jsonl import repo_relative                            # noqa: E402
from evaluation import (_START_DIRTY, check_split_against_report,  # noqa: E402
                        commit_hash, open_day, open_day_start,
                        split_days, subset_position, runtime_stamp)

#: the sixty-day, four-season position both markets 02 and 03 evaluate on
POSITION = "tests/fixtures/day_ahead_position_29gb_T24_step1prime_seasons.npz"
#: the sixty-day commitment fixture market 01 evaluates on
FIXTURE = "day_ahead_commitment_29gb_T24_relax.npz"

#: Overridden by `--position` / `--fixture` before any builder runs, so that a
#: window other than the shipped sixty days can be evaluated without editing
#: this file.  Module-level rebinding is safe here **only because both names are
#: read inside the builders rather than imported by value**: a `from
#: run_bestfixed_grid import POSITION` elsewhere would bind the old string at
#: import time and this override would be silent.  Checked: no such import
#: exists in the tree.  `main` prints both after the override so the values in
#: force are visible rather than assumed.
def _override_paths(position, fixture):
    global POSITION, FIXTURE
    if position:
        POSITION = position
    if fixture:
        FIXTURE = fixture


def parse_levels(text):
    """`lo:hi:step` or a comma-separated list, ascending, in markup units."""
    if ":" in text:
        lo, hi, step = (float(x) for x in text.split(":"))
        n = int(round((hi - lo) / step)) + 1
        return [round(lo + step * i, 6) for i in range(n)]
    return [float(x) for x in text.split(",")]


def _checked_split(dates, case):
    ok, selected = check_split_against_report(dates, case)
    print(f"split matches report section 2.2: {ok}", flush=True)
    if not ok:
        raise SystemExit(f"the split rule no longer reproduces the report's "
                         f"evaluation days; rule gives {selected}")
    return split_days(len(dates), case)


def build_01():
    """Day-ahead: one day per episode, the day is the state's cursor."""
    from run_eval_01 import (CAP_SCALE, K, MARKUP_MAX, RAMP_SCALE, T,
                             VOLL_IN_EFFECT)
    from powermarketjax.case import load_case
    from powermarketjax.envs.day_ahead import (demand_from_meta, load_commitment,
                                               make_env)
    from powermarketjax.envs.day_ahead.commitment import FIXTURE_DIR
    from powermarketjax.learning.adapters import unpack_env

    path = Path(FIXTURE)
    if not path.is_absolute() and not path.exists():
        path = FIXTURE_DIR / FIXTURE
    fixture = load_commitment(path=path, n_periods=T)
    meta = fixture["meta"]
    for key, given in (("cap_scale", CAP_SCALE), ("ramp_scale", RAMP_SCALE),
                       ("voll", VOLL_IN_EFFECT)):
        got = meta.get(key)
        if got is None or abs(float(got) - given) > 1e-12:
            raise SystemExit(f"fixture meta {key}={got} disagrees with {given}")
    dates = [str(d) for d in meta["dates"]]
    ev, _tr = _checked_split(dates, meta["case"])

    env, spec = make_env(load_case(meta["case"]), fixture, demand_from_meta(meta),
                         n_segments=K, kind="markup", markup_max=MARKUP_MAX,
                         cap_scale=CAP_SCALE, ramp_scale=RAMP_SCALE)
    four = unpack_env((env, spec))
    return dict(
        four=four, params=env.make_params(episode_len=1), days=ev,
        window_days=ev, dates=[dates[d] for d in ev],
        day_of=lambda st: int(st.cursor), action_of=None, horizon=1,
        periods_per_day=None, period_of=None, to_window=lambda j: j,
        # `evaluation.system_cost`'s third term for day-ahead is verified 0.0
        # (that function's own docstring, `envs/day_ahead/clearing.py:251-253`)
        voll=VOLL_IN_EFFECT, other_shortfall_cost=0.0,
        run_point=dict(market="01 day-ahead wholesale", case=str(meta["case"]),
                       cap_scale=CAP_SCALE, ramp_scale=RAMP_SCALE,
                       voll=VOLL_IN_EFFECT, markup_max=MARKUP_MAX,
                       episode_len=1, window=meta.get("window"),
                       fixture=repo_relative(path)))


def build_02(days=None):
    """Real-time: forty-eight periods per episode on a day subset.

    `days` names the window days the environment is to hold.  Left `None` it is
    the held-out set, which is what every caller before the economic-withholding
    arm meant and what every product built by this file was built on: the
    default path is unchanged, and the control that says so is a rerun of the
    shipped sixty-day grid compared row by row against its own product.  The
    parameter exists because this market's environment *is* its day set: it is
    built from `subset_position`, so an arm that fits on training days cannot
    reach them by passing a day index -- there is no such day in the environment
    to pass.  Fitting on the held-out days and reading on them is the thing the
    split exists to prevent.
    """
    from run_rl_02 import CAP_SCALE, MARKUP_MAX, RAMP_SCALE, VOLL_IN_EFFECT
    from powermarketjax.case import load_case
    from powermarketjax.envs.day_ahead import demand_from_meta
    from powermarketjax.envs.real_time import load_da_position
    from powermarketjax.envs.real_time.demand import (T_RT,
                                                      half_hourly_from_meta)
    from powermarketjax.envs.real_time.env import make_env
    from powermarketjax.learning.adapters import unpack_env

    pos = load_da_position(path=POSITION)
    meta = pos["meta"]
    for key, given in (("cap_scale", CAP_SCALE), ("ramp_scale", RAMP_SCALE),
                       ("voll", VOLL_IN_EFFECT)):
        got = meta.get(key)
        if got is None or abs(float(got) - given) > 1e-12:
            raise SystemExit(f"fixture meta {key}={got} disagrees with {given}")
    dates = [str(d) for d in meta["dates"]]
    ev, _tr = _checked_split(dates, meta["case"])
    use = list(ev) if days is None else [int(d) for d in days]

    # both legs come from the position's own record: the realised series has a
    # loader per case since 2026-09-09, so naming the GB one here would serve
    # British demand to whatever network the fixture names
    hh, _ = half_hourly_from_meta(meta)
    fc, _a, _d = demand_from_meta(meta)
    env, spec = make_env(load_case(meta["case"]), subset_position(pos, use), hh,
                         fc, n_segments=1, markup_max=MARKUP_MAX,
                         cap_scale=CAP_SCALE, ramp_scale=RAMP_SCALE)
    four = unpack_env((env, spec))
    # the environment contains only the days it was built for, so day `use[i]`
    # of the window is day `i` of this environment -- the same mapping
    # `run_rl_02.py` uses when it evaluates
    return dict(
        four=four, params=env.make_params(episode_len=T_RT),
        days=list(range(len(use))), window_days=use,
        dates=[dates[d] for d in use],
        day_of=lambda st: int(st.cursor) // T_RT, action_of=None,
        horizon=T_RT, periods_per_day=T_RT,
        period_of=lambda st: int(st.cursor),
        to_window=lambda j: (use[j] if 0 <= j < len(use) else None),
        # real-time reuses day_ahead.make_clearing, so its third system_cost
        # term is verified 0.0 too (evaluation.system_cost's own docstring)
        voll=VOLL_IN_EFFECT, other_shortfall_cost=0.0,
        run_point=dict(market="02 real-time balancing", case=str(meta["case"]),
                       cap_scale=CAP_SCALE, ramp_scale=RAMP_SCALE,
                       voll=VOLL_IN_EFFECT, markup_max=MARKUP_MAX,
                       episode_len=T_RT, window=meta.get("window"),
                       n_lookahead=1, position=repo_relative(POSITION),
                       subset=("held-out days only" if days is None else
                               f"{len(use)} named days")
                              + "; day index is the subset "
                                "position, window_days gives the window index"))


def build_03():
    """Ancillary: forty-eight periods per episode, markup in column 0 only."""
    import jax.numpy as jnp
    import run_eval_03 as R
    from powermarketjax.case import load_case
    from powermarketjax.envs.ancillary.env import (AncillaryParams,
                                                   make_ancillary_env)
    from powermarketjax.envs.real_time.demand import load_gb_demand_half_hourly

    volr, pi_scale = R.volr_pi_scale(R.CASE)
    fx = np.load(POSITION, allow_pickle=True)
    meta = json.loads(str(fx["meta"]))
    for key, given in (("cap_scale", R.CAP_SCALE), ("ramp_scale", R.RAMP_SCALE),
                       ("voll", R.VOLL_IN_EFFECT)):
        got = meta.get(key)
        if got is None or abs(float(got) - given) > 1e-12:
            raise SystemExit(f"fixture meta {key}={got} disagrees with {given}")
    dates = [str(d) for d in meta["dates"]]
    ev, _tr = _checked_split(dates, meta["case"])

    day_index = np.asarray(fx["day_index"], np.int64)
    hh, _d = load_gb_demand_half_hourly()
    demand = np.asarray(hh[day_index], np.float64).reshape(-1)
    u = np.repeat(np.asarray(fx["u"], np.float64), 2, axis=2).transpose(0, 2, 1)
    u = u.reshape(-1, u.shape[-1])
    q_da = np.repeat(np.asarray(fx["q_da"], np.float64), 2, axis=2).transpose(0, 2, 1)
    q_da = q_da.reshape(-1, q_da.shape[-1])
    lmp_da = np.repeat(np.asarray(fx["lmp_da"], np.float64), 2, axis=1)
    lmp_da = lmp_da.reshape(-1, lmp_da.shape[-1])

    env = make_ancillary_env(load_case(R.CASE), R.THETA, volr, R.BETA,
                             pi_scale, n_segments=1, cap_scale=R.CAP_SCALE,
                             ramp_scale=R.RAMP_SCALE, period_hours=R.DELTA,
                             kind="markup", markup_max=R.MARKUP_MAX)
    params = AncillaryParams(
        demand=jnp.asarray(demand), forecast=jnp.asarray(demand * 1.02),
        commitment=jnp.asarray(u), q_da=jnp.asarray(q_da),
        lmp_da=jnp.asarray(lmp_da),
        learner_mask=jnp.ones(u.shape[1], bool), episode_len=R.T_DAY)

    baseline = np.asarray(env[3]["baseline_action"])

    def action_of(v):
        """Column 0 to the level; the two reserve columns stay truthful.

        Exactly how `run_eval_03.py` builds its constant arm.  Those columns are
        raw pre-softplus values held at -800, which softplus sends to zero, and
        `action_high` bounds the markup column alone -- filling them with the
        level would move them by 801 units and sweep a different axis.
        """
        a = np.array(baseline, copy=True)
        a[:, 0] = float(v)
        return jnp.asarray(a)

    return dict(
        four=env, params=params, days=ev, window_days=ev,
        dates=[dates[d] for d in ev],
        day_of=lambda st: int(st.cursor) // R.T_DAY, action_of=action_of,
        horizon=R.T_DAY, periods_per_day=R.T_DAY,
        period_of=lambda st: int(st.cursor), to_window=lambda j: j,
        # ancillary's third system_cost term is not a constant (§ see
        # `arms.rollout_action`'s `other_shortfall_cost_key` paragraph) --
        # `info["volr_cost"]` is already VOLR x reserve shortfall in dollars,
        # exactly what `run_eval_03.py`'s own day-eval loop accumulates before
        # calling `evaluation.system_cost`
        voll=R.VOLL_IN_EFFECT, other_shortfall_cost=0.0,
        other_shortfall_cost_key="volr_cost",
        run_point=dict(market="03 ancillary services", case=R.CASE,
                       cap_scale=R.CAP_SCALE, ramp_scale=R.RAMP_SCALE,
                       voll=R.VOLL_IN_EFFECT, volr=volr, beta=list(R.BETA),
                       theta=list(R.THETA), pi_scale=pi_scale,
                       period_hours=R.DELTA, markup_max=R.MARKUP_MAX,
                       episode_len=R.T_DAY, window=meta.get("window"),
                       position=repo_relative(POSITION),
                       forecast="actual demand x 1.02 (spec section 15)"))


BUILDERS = {"01": build_01, "02": build_02, "03": build_03}


def window_membership(cfg):
    """Which days each pinned episode actually scores, period by period.

    Not a check of `open_day` -- that verifies the day it opens.  This asks the
    question after it: an episode of `horizon` periods that opens *inside* a day
    runs past the end of it.  Market 03's `reset` draws a period rather than a
    day (`envs/ancillary/env.py:243`), so its windows do exactly that, and the
    periods that spill over are scored on whatever day follows -- which for the
    twelve held-out days of the sixty-day window is a **training** day.

    Reported per window with the denominator, and with each spilled day
    classified against the evaluation set, because "the evaluation ran partly on
    training data" is a limitation on every conclusion drawn from that market's
    arms, not a detail of this one.
    """
    if cfg["periods_per_day"] is None:
        return None, dict(spilled_periods=0,
                          total_periods=len(cfg["days"]) * cfg["horizon"],
                          spilled_onto_training_days=0,
                          note="one step per day; a window cannot leave its day")
    ppd, horizon = cfg["periods_per_day"], cfg["horizon"]
    ev_env_days = set(cfg["days"])
    windows, spill_total, spill_train = [], 0, 0
    # opened the same way the grid opens them, so this reports the spill of the
    # run that was actually swept and not of a second opening path.  Adding
    # `reset_on_day` made the two different for market 03 and this is where the difference has
    # to be visible: on `reset_on_day` the spill is zero by construction, which
    # is the claim, and reading it from the states is what checks it.
    spec = cfg["four"][3]
    on_day = spec.get("reset_on_day")
    for i, d in enumerate(cfg["days"]):
        if on_day is not None:
            _k, st = open_day_start(on_day, cfg["params"], d, cfg["day_of"],
                                    cfg["period_of"], spec["periods_per_day"])
        else:
            _k, st = open_day(cfg["four"][0], cfg["params"], d, cfg["day_of"])
        cursor = int(st.cursor)
        counts = {}
        for p in range(cursor, cursor + horizon):
            counts[p // ppd] = counts.get(p // ppd, 0) + 1
        spill = sorted((int(day), int(n)) for day, n in counts.items()
                       if int(day) != int(d))
        # the window day this environment's day index stands for, so a spilled
        # day can be named in the window's own numbering; identity for markets
        # 01 and 03, the subset position for 02
        to_window = cfg["to_window"]
        on_train = sum(n for day, n in spill if day not in ev_env_days)
        windows.append(dict(
            env_day=int(d), window_day=int(cfg["window_days"][i]),
            date=cfg["dates"][i], cursor=cursor, offset=cursor % ppd,
            periods_total=horizon, periods_on_named_day=int(counts.get(d, 0)),
            spill=[[day, n, ("eval" if day in ev_env_days else "train"),
                    to_window(day)] for day, n in spill],
            periods_on_training_days=int(on_train)))
        spill_total += horizon - int(counts.get(d, 0))
        spill_train += int(on_train)
    return windows, dict(spilled_periods=int(spill_total),
                         total_periods=int(horizon * len(cfg["days"])),
                         spilled_onto_training_days=int(spill_train))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True, choices=sorted(BUILDERS))
    ap.add_argument("--levels", default="1.0:2.0:0.05")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0,
                    help="the ONE key every level is rolled on; with the days "
                         "pinned it changes nothing in markets 01 to 03, whose "
                         "transitions consume no randomness")
    ap.add_argument("--position", default="",
                    help="day-ahead position for markets 02 and 03.  Default: "
                         "the shipped sixty-day product")
    ap.add_argument("--fixture", default="",
                    help="commitment fixture for market 01, a path or a name "
                         "under tests/fixtures/.  Default: the shipped "
                         "sixty-day product")
    args = ap.parse_args()
    _override_paths(args.position, args.fixture)
    print(f"position in force: {POSITION}\nfixture  in force: {FIXTURE}",
          flush=True)

    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")

    import powermarketjax
    print("package:", powermarketjax.__file__, flush=True)

    cfg = BUILDERS[args.market]()
    levels = parse_levels(args.levels)
    spec = cfg["four"][3]
    action_high = float(np.asarray(spec["action_high"]))
    print(f"market {args.market}  {len(levels)} levels "
          f"{levels[0]} .. {levels[-1]}  on {len(cfg['days'])} days "
          f"{cfg['days']} (window days {cfg['window_days']})", flush=True)
    print(f"dates: {cfg['dates']}", flush=True)
    print(f"horizon {cfg['horizon']}  episode_len "
          f"{getattr(cfg['params'], 'episode_len', None)}  action_high "
          f"{action_high}  action_of "
          f"{'default (pure markup)' if cfg['action_of'] is None else 'market-specific'}",
          flush=True)

    windows, spill = window_membership(cfg)
    if windows is not None:
        print(f"window membership: {spill['spilled_periods']} of "
              f"{spill['total_periods']} half-hours fall outside the day they "
              f"are named after, {spill['spilled_onto_training_days']} of them "
              f"on training days", flush=True)

    # opt-in, per-market: only `build_01` sets these two keys so far (§ see
    # `arms.rollout_action`'s docstring for why both-or-neither); markets 02
    # and 03 fall through to `cfg.get(...) is None` and get exactly the rows
    # they always have
    want_system_cost = cfg.get("voll") is not None
    t0 = time.time()
    rows, best, at_upper_bound = arms.markup_grid(
        cfg["four"], cfg["params"], levels, None, cfg["horizon"],
        jax.random.PRNGKey(args.seed), action_of=cfg["action_of"],
        days=cfg["days"], day_of_state=cfg["day_of"],
        period_of_state=cfg["period_of"], voll=cfg.get("voll"),
        other_shortfall_cost=cfg.get("other_shortfall_cost"),
        other_shortfall_cost_key=cfg.get("other_shortfall_cost_key"))
    wall = time.time() - t0

    for r in rows:
        per_day = np.asarray(r["reward_sum_per_env"])
        sc = (f"  system_cost_sum {np.asarray(r['system_cost_sum_per_env']).sum():14.6e}"
             if want_system_cost else "")
        print(f"  level {r['level']:.3f}  reward_mean {r['reward_mean']:14.6e}  "
              f"day total sum {per_day.sum():14.6e}  "
              f"min {per_day.min():12.4e}  max {per_day.max():12.4e}  "
              f"unconverged {r['unconverged_frac']:.4f}{sc}", flush=True)

    spans = levels[-1] >= action_high - 1e-12
    print(f"\nbest level {rows[best]['level']:.3f}   "
          f"at_upper_bound {at_upper_bound}   "
          f"grid_spans_action_space {spans}   "
          f"({wall:.1f} s wall, {wall / len(levels):.2f} s per level)",
          flush=True)
    if at_upper_bound:
        print("  the winner is the last grid point, so this is a LOWER BOUND on "
              "what a fixed markup achieves and not an optimum. "
              + ("The grid spans the whole action space, so the bound is the "
                 "action-space endpoint -- the same action as the `constant` "
                 "arm, which table one lists separately as the maximum bid."
                 if spans else
                 "The grid stops below action_high, so it was simply too narrow."),
              flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(
        arm="best_fixed_markup", market=args.market, levels=levels,
        best=best, best_level=rows[best]["level"],
        at_upper_bound=bool(at_upper_bound), grid_spans_action_space=bool(spans),
        action_high=action_high, days=cfg["days"],
        window_days=[int(d) for d in cfg["window_days"]], dates=cfg["dates"],
        horizon=cfg["horizon"], rows=rows, windows=windows,
        window_membership=spill, wall_seconds=wall,
        seconds_per_level=wall / len(levels),
        run_point=dict(cfg["run_point"], **runtime_stamp(),
                       commit=commit_hash(), commit_dirty=_START_DIRTY,
                       # the blob of the file that actually did the sweep: it
                       # survives a rebase, and it is unmoved by whatever else in
                       # the checkout was dirty when this ran
                       arms_blob=arms_blob(),
                       levels_grid=args.levels, key_seed=args.seed,
                       note=(
                           "reward is settlement profit; system_cost_sum_per_env "
                           "in each row is production cost plus shed energy at "
                           "voll, from evaluation.system_cost's own formula "
                           "(arms.rollout_action's opt-in voll/"
                           "other_shortfall_cost, both set for this run)"
                           if want_system_cost else
                           "reward is settlement profit; this product carries "
                           "no system cost -- run the winning level through "
                           "this market's evaluation driver for the paired "
                           "number")))
    path = out / f"grid_{args.market}.json"
    path.write_text(json.dumps(payload, indent=1))
    print(f"-> {repo_relative(path)}", flush=True)


def arms_blob():
    """`git rev-parse --verify HEAD:<path>`, or `"unknown"`.

    `--verify` matters: without it, a path absent from HEAD makes rev-parse echo
    the argument itself to stdout, and the product carries the literal string
    `HEAD:tools/...` where a blob should be (happened 2026-09-03 on market 01's
    first assemble, run before the file was committed).
    """
    import subprocess
    try:
        r = subprocess.run(["git", "rev-parse", "--verify", "-q",
                            "HEAD:tools/benchmark/arms.py"],
                           capture_output=True, text=True, timeout=10,
                           cwd=str(Path(__file__).resolve().parents[2]))
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:                                      # pragma: no cover
        return "unknown"


if __name__ == "__main__":
    main()
