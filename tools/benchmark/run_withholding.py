"""The economic-withholding arm for markets 01, 02 and 03: each unit's best
response to an honest fringe.

The single-firm problem of Hobbs, Metzler and Pang (2000, IEEE Trans. Power
Syst. 15(2)) in grid form.  Unit `u` sweeps its own markup multiplier `alpha`
over a grid while every other unit bids its true cost; the level with the
highest settlement profit *for u*, summed over the fitting days, is its best
response, and the arm is the vector of these best responses.  Its additive
markup is `m_u* = (alpha_u* - 1) * MC_u` (K = 1: one cost segment per unit).

The arm is fitted on training days and read on the held-out days, like a
learned policy is.  `--days train36` takes every ninth training day of the
365-day window (`tr[5::9]`, 36 days, disjoint from the 36 evaluation days).

**What the vector is and is not.**  It is a well-defined, order-independent
function of (fixture, fitting days, grid): rerunning `sweep` reproduces it.  It
is *not* self-consistent: when every unit plays its best response at once, most
would want to move again.  Measured on market 01, 2026-09-02/03, 365-day
fixture: Gauss-Seidel diagonalisation from this start (three unit orderings, 11
levels, ten passes each) did not converge and did not cycle -- 39/66 coordinates
were stable and identical across orderings, 27 mid-cost units kept flipping.
There is no pure-strategy grid equilibrium to report, so the arm is the
unilateral vector, documented as such.  It was adopted on 2026-09-03, and
diagonalising markets 02 and 03 was ruled out.

**What is market-specific and where it lives.**  Three things, all read from
`run_bestfixed_grid`'s builder rather than decided here, so this arm cannot
drift from the arms it is tabulated beside:

1. *how a day is opened.*  Market 03's `spec` offers `reset_on_day`
   and its episode must start at the day's first period; 01 and 02 reach a day
   through `evaluation.open_day`'s key search, which for them lands on period 0
   by construction.  Opening 03 by key search would score 265 of 576 pinned half
   hours on the following day.
2. *what an action is.*  Markets 01 and 02 take a markup vector; market 03 takes
   `(n_units, 3)` whose last two columns are raw pre-softplus reserve offers held
   at -800.  The profile is therefore carried as an alpha *vector* throughout and
   turned into a full action by the market's own `action_of` at the last moment.
3. *which days the environment holds.*  Market 02's environment is built from
   `subset_position`, so it holds only the days it was built for and its day
   indices are positions in that subset; markets 01 and 03 hold the whole window
   and index into it directly.  `build(market, which)` returns both numberings.

`reward_per_agent` is a mean over the horizon as well as over the environments,
so a per-unit total is `reward_per_agent * n_days * horizon`.  Market 01's
horizon is 1, which is why its own code could multiply by `n_days` alone.

Two modes:

    sweep      one shard per GPU; writes `<out-dir>/honest.json` (shard 0) and
               `<out-dir>/unit_NN.json` per unit with the whole profit curve.
               Units whose profit against the profile is exactly zero at
               alpha = 1 are skipped: never committed, nothing to respond to.
    assemble   reads a finished sweep, classifies every unit, rolls the joint
               vector next to honest / max on the evaluation and fitting days,
               and writes the product.

    JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \\
    conda run -n powermarketjax --no-capture-output python \\
        tools/benchmark/run_withholding.py sweep --market 02 --days train36 \\
        --shard 0/3 \\
        --fixture da_position_29gb_T24_y1_365d_c0.6_r1.00.npz \\
        --out-dir withholding_02/train36
    ... python tools/benchmark/run_withholding.py assemble --market 02 \\
        --dir withholding_02/train36 \\
        --out 02-y1-train36.json

**The reserve column (market 03 only).**  `--column reserve` sweeps the price a
unit offers for reserve instead of its energy markup: levels are in \$/MWh, the
same offer applies to both reserve products, the energy column stays at the
truthful 1.0, and the grid starts at 0, the truthful reserve offer.  A level L
reaches the action as the raw pre-softplus `log(exp(L / pi_scale) - 1)`, and
L = 0 as the environment's own `TRUTHFUL_ACTION` (-800), since the map's
infimum has no finite preimage.  `pi_scale` is read from the builder's
`run_point`.  `--uniform` sets every unit to the same level at once instead of
one unit at a time and writes `uniform.json`; with the reserve column the rows
also carry the mean reserve price and shortfall per product.  `assemble`
accepts only the default energy sweep.

Classification of unit u from its own curve, by the sign of its profit at
alpha*:  `honest` (alpha* = 1), `raise` (profit at alpha* > 0: price-raising
proper), `exit` (|profit at alpha*| <= EXIT_REL * |honest profit|: the unit
priced itself out of commitment and stopped paying its no-load; it earns
nothing), `partial` (alpha* > 1 but still losing: the markup only trimmed the
loss).  The split matters because most of market 01's gain over honest bidding
is loss avoidance by exit, not price-raising (train36, 2026-09-02: 54% exit,
30% raise -- almost all of it unit 35 -- 16% partial).
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arms                                                      # noqa: E402
import run_bestfixed_grid as G                                   # noqa: E402
from curve_jsonl import repo_relative                            # noqa: E402
from evaluation import (_START_DIRTY, commit_hash, open_day,     # noqa: E402
                        open_day_start, runtime_stamp, split_days)

#: Two levels give "the same" own profit within this relative tolerance, and
#: the best response is the *lowest* such level (a unit priced out is
#: indifferent along the whole plateau; the lowest level is the one that stays
#: nearest the market).  Within one process a plateau is exact to 12 digits;
#: across GPU processes the same rollout differs by 1.4e-7 relative (measured
#: 2026-09-02, GPU float64, 365-day fixture); distinct levels differed
#: by >= 1e-3 relative.
TIE_REL = 1e-6
#: `exit` band.  Any value in [0.011, 0.054] gives the same partition on the
#: 2026-09-02 train36 sweep of market 01 (nearest units at 1.07% and 5.4%).
#: Markets 02 and 03 report their own nearest units rather than inheriting it.
EXIT_REL = 0.02

MARKETS = ("01", "02", "03")


def pick_days(split, which):
    ev, tr = split
    if which == "eval":
        return [int(d) for d in ev]
    if which == "train36":
        return [int(d) for d in tr[5::9]]
    raise SystemExit(f"unknown --days {which}")


def best_index(own, tie_rel=TIE_REL):
    """Lowest level within `tie_rel` of the maximum own profit, and the plateau."""
    own = np.asarray(own, np.float64)
    top = float(own.max())
    tied = np.where(own >= top - tie_rel * max(1.0, abs(top)))[0]
    return int(tied.min()), [int(i) for i in tied]


def classify(own, best, honest_own, exit_rel=EXIT_REL):
    if best == 0:
        return "honest"
    if abs(own[best]) <= exit_rel * abs(honest_own):
        return "exit"
    return "raise" if own[best] > 0 else "partial"


def _jax_setup():
    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    import powermarketjax
    print("package:", powermarketjax.__file__, "devices:", jax.devices(), flush=True)


def _window_dates(market, path):
    """`(dates, case)` of the window, read the way that market's builder reads
    them.  The case comes back with the dates because the split needs both: a
    365-day window is GB's or NEM's, and only the fixture's `meta` says which."""
    if market == "01":
        from run_eval_01 import T
        from powermarketjax.envs.day_ahead import load_commitment
        from powermarketjax.envs.day_ahead.commitment import FIXTURE_DIR
        p = Path(path)
        if not p.is_absolute() and not p.exists():
            p = FIXTURE_DIR / path
        meta = load_commitment(path=p, n_periods=T)["meta"]
    else:
        fx = np.load(path, allow_pickle=True)
        meta = json.loads(str(fx["meta"]))
    return [str(d) for d in meta["dates"]], meta["case"]


def build(market, path, which):
    """`(cfg, env_days, window_days)` for one market and one day set.

    `env_days` index the environment `cfg` actually holds; `window_days` index
    the 365-day window.  They differ only for market 02, whose environment is
    built from `subset_position` and therefore contains exactly the days asked
    for, numbered from zero.  Everything else about the environment -- case,
    scenario factors, action layout, VOLL, the third `system_cost` term -- comes
    from `run_bestfixed_grid`'s builder unchanged.
    """
    if market == "01":
        G._override_paths("", path)
        print("fixture in force:", G.FIXTURE, flush=True)
    else:
        G._override_paths(path, "")
        print("position in force:", G.POSITION, flush=True)
    dates, case = _window_dates(market, path)
    split = split_days(len(dates), case)
    days = pick_days(split, which)
    if which == "train36":
        assert not set(days) & set(split[0]), "training subset overlaps eval days"
    if market == "02":
        cfg = G.build_02(days=days)
        env_days = [int(d) for d in cfg["days"]]
        assert env_days == list(range(len(days)))
        assert [int(d) for d in cfg["window_days"]] == days
    else:
        cfg = G.build_01() if market == "01" else G.build_03()
        env_days = days
        if which == "eval":
            assert list(cfg["days"]) == days, f"build_{market} disagrees with the split"
    return cfg, env_days, days


def alpha_layout(cfg):
    """`(n_units, truthful_alpha, action_of_alpha)` for this market's action.

    The profile is an alpha vector in every market; only the array handed to
    `step` differs.  Market 03's `build` already owns the mapping (`action_of`
    puts a level in column 0 and leaves the two reserve columns at their
    pre-softplus -800), and it is reused here rather than reimplemented, so a
    change to that layout cannot reach only one of the two arms.
    """
    base = np.asarray(cfg["four"][3]["baseline_action"], np.float64)
    if base.ndim == 1:
        n = base.shape[0]
        return n, base.copy(), lambda a: np.asarray(a, np.float64)
    if base.ndim == 2:
        n = base.shape[0]

        def action_of_alpha(a):
            out = np.array(base, copy=True)
            out[:, 0] = np.asarray(a, np.float64)
            return out
        return n, base[:, 0].copy(), action_of_alpha
    raise SystemExit(f"baseline_action has shape {base.shape}; this arm knows "
                     f"a markup vector and a (n_units, k) block whose column 0 "
                     f"is the markup, and nothing else")


def reserve_layout(cfg):
    """`(n_units, truthful_level, action_of_level, pi_scale)` for the reserve column.

    The level vector is a reserve offer in \$/MWh per unit, applied to both
    products; column 0 keeps the baseline's truthful energy level.
    """
    from powermarketjax.envs.ancillary.action import TRUTHFUL_ACTION
    base = np.asarray(cfg["four"][3]["baseline_action"], np.float64)
    if base.ndim != 2 or base.shape[1] < 2:
        raise SystemExit(f"--column reserve needs a (n_units, 1 + n_products) action, "
                         f"this market's baseline_action has shape {base.shape}")
    pi_scale = float(cfg["run_point"]["pi_scale"])
    n = base.shape[0]

    def raw(level):
        level = np.asarray(level, np.float64)
        if (level < 0).any():
            raise SystemExit("a reserve offer cannot be negative")
        safe = np.where(level > 0, level, 1.0)
        return np.where(level > 0, np.log(np.expm1(safe / pi_scale)), TRUTHFUL_ACTION)

    def action_of_level(v):
        out = np.array(base, copy=True)
        out[:, 1:] = raw(v)[:, None]
        return out
    return n, np.zeros(n), action_of_level, pi_scale


def _runner(cfg, env_days, action_of=None, info_mean_keys=()):
    """`row(alpha_vector) -> metrics dict`, days pinned, jit compiled once.

    The opening path is chosen by the market's `spec`, not by a market name:
    `reset_on_day` present means the day is asked for and its first period
    checked (`open_day_start`), absent means the key search (`open_day`).
    """
    import jax
    import jax.numpy as jnp
    four, params = cfg["four"], cfg["params"]
    spec = four[3]
    if action_of is None:
        _n, _truthful, action_of = alpha_layout(cfg)
    if "reset_on_day" in spec:
        if cfg["period_of"] is None:
            raise SystemExit("this market offers reset_on_day but its cfg has no "
                             "period_of; the offset inside the day could not be "
                             "read back and the episode could run past the day")
        opened = [open_day_start(spec["reset_on_day"], params, d, cfg["day_of"],
                                 cfg["period_of"], spec["periods_per_day"])[1]
                  for d in env_days]
        reset_states = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *opened)
        reset_keys = None
    else:
        reset_keys = jnp.stack([open_day(four[0], params, d, cfg["day_of"])[0]
                                for d in env_days])
        reset_states = None
    key = jax.random.PRNGKey(0)

    @jax.jit
    def run(action):
        return arms.rollout_action(four, params, action, len(env_days),
                                   cfg["horizon"], key, reset_keys=reset_keys,
                                   reset_states=reset_states, voll=cfg["voll"],
                                   other_shortfall_cost=cfg["other_shortfall_cost"],
                                   other_shortfall_cost_key=cfg.get("other_shortfall_cost_key"),
                                   info_mean_keys=tuple(info_mean_keys))

    def row(alpha):
        m = run(jnp.asarray(action_of(alpha)))
        jax.block_until_ready(m["reward_mean"])
        return {kk: np.asarray(vv).tolist() for kk, vv in m.items()}
    return row


def _totals(row, n_days, horizon):
    """Per-unit profit totals from a metrics row.

    `reward_per_agent` is `mean` over the horizon *and* the environment axis, so
    the total over the pinned days is the mean times both lengths.  Market 01's
    horizon is 1 and its own driver multiplied by `n_days` alone; that is the
    same number, not a different convention.
    """
    return np.asarray(row["reward_per_agent"], np.float64) * (n_days * horizon)


def sweep(args):
    k, n = (int(x) for x in args.shard.split("/"))
    _jax_setup()
    cfg, env_days, win_days = build(args.market, args.fixture, args.days)
    reserve = args.column == "reserve"
    if reserve:
        if args.profile is not None:
            raise SystemExit("--column reserve sweeps against the truthful profile only")
        n_units, truthful, action_of, pi_scale = reserve_layout(cfg)
        info_keys = ("reserve_price", "reserve_shortfall")
        print(f"reserve column: pi_scale {pi_scale} (from run_point), levels in $/MWh; "
              f"raw action at each level {action_of(np.full(n_units, 0.0))[0, 1]:.1f} (L=0)",
              flush=True)
    else:
        n_units, truthful, action_of = alpha_layout(cfg)
        info_keys = ()
    if args.profile is None:
        base = truthful.copy()
    else:
        pth, keyname = args.profile.split(":")
        base = np.asarray(json.load(open(pth))[keyname], np.float64)
        assert base.shape == truthful.shape
    levels = G.parse_levels(args.levels)
    if reserve:
        assert levels[0] == 0.0, "reserve levels must start at the truthful offer 0"
        for v in levels:
            print(f"   level {v:g} $/MWh -> raw {action_of(np.full(n_units, v))[0, 1]:.6f}",
                  flush=True)
    else:
        assert abs(levels[0] - 1.0) < 1e-12, "levels must start at the honest level 1.0"
    print(f"market {args.market}: {n_units} units, {len(levels)} levels "
          f"{levels[0]}..{levels[-1]}, {len(env_days)} {args.days} days "
          f"(window {win_days}), horizon {cfg['horizon']}; rivals "
          f"{'truthful' if args.profile is None else args.profile}", flush=True)
    row = _runner(cfg, env_days, action_of=action_of, info_mean_keys=info_keys)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    profile_row = row(base)                    # nobody deviates
    print(f"profile row: {time.time() - t0:.1f} s (includes compile); "
          f"unconverged {profile_row['unconverged_frac']:.6f}", flush=True)
    own_base = _totals(profile_row, len(env_days), cfg["horizon"])
    zero_units = [int(u) for u in np.where(own_base == 0.0)[0]]
    nonzero = [int(u) for u in np.where(own_base != 0.0)[0]]
    print(f"own profit against the profile: {len(nonzero)} non-zero, {len(zero_units)} "
          f"exactly zero (skipped): {zero_units}", flush=True)
    if k == 0:
        (out / "honest.json").write_text(json.dumps(dict(
            market=args.market, days_kind=args.days, days=env_days,
            window_days=win_days, horizon=cfg["horizon"],
            own_profit_total=own_base.tolist(),
            zero_units=zero_units, nonzero_units=nonzero, row=profile_row,
            profile=args.profile, profile_alpha=base.tolist(),
            fixture=(G.FIXTURE if args.market == "01" else G.POSITION),
            levels=levels, tie_rel=TIE_REL,
            **(dict(column="reserve", level_unit="$/MWh reserve offer, both products",
                    pi_scale=pi_scale) if reserve else {})), indent=1))

    if args.uniform:
        _uniform(args, out, row, levels, base, profile_row, env_days, win_days, cfg,
                 dict(column="reserve", level_unit="$/MWh reserve offer, both products",
                      pi_scale=pi_scale) if reserve else dict(column="energy"))
        return

    units = nonzero if args.units is None else [int(t) for t in args.units.split(",")]
    mine = [u for i, u in enumerate(units) if i % n == k]
    print(f"shard {k}/{n}: {len(mine)} units {mine}", flush=True)
    for u in mine:
        fn = out / f"unit_{u:02d}.json"
        if fn.exists():
            print(f"unit {u}: exists, skipped", flush=True)
            continue
        t0 = time.time()
        rows, cur = [], None
        for i, v in enumerate(levels):
            if abs(v - base[u]) < 1e-9:
                rows.append(profile_row)       # the unit's own level: no rollout
                cur = i
            else:
                a = base.copy()
                a[u] = v
                rows.append(row(a))
        if cur is None:
            raise SystemExit(f"unit {u}: its profile level {base[u]} is not on the grid")
        wall = time.time() - t0
        own = np.array([r["reward_per_agent"][u] for r in rows]) * (len(env_days) * cfg["horizon"])
        unconv = [float(r["unconverged_frac"]) for r in rows]
        best, tied = best_index(own)
        print(f"unit {u}: {wall:.1f} s  at1.0 {own[0]:.6e}  best alpha {levels[best]:.2f}  "
              f"at_bound {best == len(levels) - 1}  gain {own[best] - own[0]:+.6e}  "
              f"gain_vs_stay {own[best] - own[cur]:+.6e}  tied {[levels[i] for i in tied]}  "
              f"unconv_max {max(unconv):.6f}", flush=True)
        print("   " + " ".join(f"{a:.2f}:{p:.4e}" for a, p in zip(levels, own)), flush=True)
        fn.write_text(json.dumps(dict(
            unit=u, market=args.market, days_kind=args.days, days=env_days,
            window_days=win_days, horizon=cfg["horizon"], levels=levels,
            own_profit_total=own.tolist(), best_index=best, best_level=levels[best],
            at_bound=best == len(levels) - 1, tied_levels=[levels[i] for i in tied],
            gain=float(own[best] - own[0]), gain_vs_stay=float(own[best] - own[cur]),
            profile_index=cur, own_level_in_profile=float(base[u]), wall_seconds=wall,
            unconverged_frac=unconv, rows=rows,
            fixture=(G.FIXTURE if args.market == "01" else G.POSITION),
            tie_rel=TIE_REL,
            others="truthful" if args.profile is None else args.profile,
            **(dict(column="reserve", level_unit="$/MWh reserve offer, both products",
                    pi_scale=pi_scale) if reserve else {})), indent=1))


def _uniform(args, out, row, levels, base, profile_row, env_days, win_days, cfg, extra):
    """Every unit at the same level at once, one rollout per level."""
    if args.shard != "0/1":
        raise SystemExit("--uniform runs one rollout per level and takes no --shard")
    n_days, horizon = len(env_days), cfg["horizon"]
    rows = []
    for v in levels:
        t0 = time.time()
        a = np.full(base.shape, float(v))
        r = profile_row if np.allclose(a, base, rtol=0, atol=1e-9) else row(a)
        rows.append(r)
        per = _totals(r, n_days, horizon)
        print(f"uniform {v:g}: {time.time() - t0:.1f} s  total profit {per.sum():.6e}  "
              f"system cost {np.sum(r['system_cost_sum_per_env']):.6e}  "
              f"unconv {r['unconverged_frac']:.6f}"
              + (f"  reserve price {r['info_mean_reserve_price']}"
                 if "info_mean_reserve_price" in r else ""), flush=True)
    (out / "uniform.json").write_text(json.dumps(dict(
        market=args.market, days_kind=args.days, days=env_days, window_days=win_days,
        horizon=horizon, levels=levels,
        own_profit_total=[_totals(r, n_days, horizon).tolist() for r in rows],
        unconverged_frac=[float(r["unconverged_frac"]) for r in rows], rows=rows,
        fixture=(G.FIXTURE if args.market == "01" else G.POSITION), **extra), indent=1))


def assemble(args):
    d0 = Path(args.dir)
    honest = json.load(open(d0 / "honest.json"))
    if honest.get("profile") is not None:
        raise SystemExit("assemble is for a sweep against the truthful profile")
    if honest.get("column", "energy") != "energy":
        raise SystemExit("assemble classifies energy markups; this sweep is of another column")
    market = honest.get("market", args.market)
    if args.market is not None and market != args.market:
        raise SystemExit(f"sweep says market {market}, --market says {args.market}")
    honest_own = np.asarray(honest["own_profit_total"])
    units = {}
    for f in sorted(glob.glob(str(d0 / "unit_*.json"))):
        d = json.load(open(f))
        units[d["unit"]] = d
    missing = sorted(set(honest["nonzero_units"]) - set(units))
    if missing:
        raise SystemExit(f"sweep incomplete, missing units {missing}")
    fixture = args.fixture or honest["fixture"]
    _jax_setup()
    cfg, fit_env_days, fit_win_days = build(market, fixture, honest["days_kind"])
    assert fit_env_days == list(honest["days"]), "fitting days moved"
    n_units, truthful, _ = alpha_layout(cfg)
    hi = float(np.max(np.asarray(cfg["four"][3]["action_high"])))
    from powermarketjax.case import load_case
    from powermarketjax.envs.day_ahead.clearing import segment_costs
    _, seg_cost = segment_costs(load_case(cfg["run_point"]["case"]), 1)
    mc = np.asarray(seg_cost, np.float64).reshape(n_units, -1)[:, 0]

    alpha = truthful.copy()
    cls = {}
    print(f"{'unit':>4} {'MC':>7} {'honest':>13} {'alpha*':>6} {'m*':>7} {'gain':>13} {'bound':>5} class")
    for u in sorted(units):
        d = units[u]
        own = np.asarray(d["own_profit_total"])
        best, _ = best_index(own, d["tie_rel"])
        assert best == d["best_index"]
        cls[u] = classify(own, best, honest_own[u])
        alpha[u] = d["levels"][best]
        print(f"{u:>4} {mc[u]:7.2f} {honest_own[u]:13.4e} {alpha[u]:6.2f} {(alpha[u] - 1) * mc[u]:7.2f} "
              f"{d['gain']:+13.4e} {'*' if d['at_bound'] else ' ':>5} {cls[u]}")
    counts = {c: sum(1 for v in cls.values() if v == c) for c in ("honest", "raise", "partial", "exit")}
    counts["zero_skipped"] = len(honest["zero_units"])
    print("counts:", counts, flush=True)
    gains = {c: sum(units[u]["gain"] for u in units if cls[u] == c) for c in ("raise", "partial", "exit")}
    print("sum of unilateral gains on the fitting days by class:", {c: f"{g:.4e}" for c, g in gains.items()},
          flush=True)
    # the two units nearest the `exit` band, so the band's width is reported on
    # this market's own curves rather than inherited from market 01's
    ratios = sorted((abs(np.asarray(units[u]["own_profit_total"])[units[u]["best_index"]])
                     / abs(honest_own[u]), u) for u in units if honest_own[u] != 0
                    and units[u]["best_index"] != 0)
    below = [(u, r) for r, u in ratios if r <= EXIT_REL]
    above = [(u, r) for r, u in ratios if r > EXIT_REL]
    band = dict(exit_rel=EXIT_REL,
                nearest_below=([below[-1][0], below[-1][1]] if below else None),
                nearest_above=([above[0][0], above[0][1]] if above else None))
    print("exit band: nearest below", band["nearest_below"], " nearest above",
          band["nearest_above"], flush=True)

    results = {}
    for dname, which in (("eval", "eval"), ("fit", honest["days_kind"])):
        if which == honest["days_kind"]:
            c, edays, wdays = cfg, fit_env_days, fit_win_days
        else:
            c, edays, wdays = build(market, fixture, which)
        row = _runner(c, edays)
        res = {name: row(act) for name, act in
               (("honest", truthful), ("max", np.full(n_units, hi)), ("withholding", alpha))}
        results[dname] = dict(days=edays, window_days=wdays, arms=res)
        print(f"\n== {dname} days ({len(edays)}) ==")
        print(f"{'arm':>11} {'total profit':>14} {'system cost':>14} {'shed MWh':>10} {'unconv':>7} "
              f"{'#>0':>4} {'#<0':>4} {'#=0':>4}")
        for name, r in res.items():
            per = _totals(r, len(edays), c["horizon"])
            print(f"{name:>11} {per.sum():14.6e} {np.sum(r['system_cost_sum_per_env']):14.6e} "
                  f"{np.sum(r['shed_mwh_sum_per_env']):10.3e} {r['unconverged_frac']:7.4f} "
                  f"{int((per > 0).sum()):4d} {int((per < 0).sum()):4d} {int((per == 0).sum()):4d}",
                  flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        arm="economic_withholding", market=market, definition=(
            "per-unit best response on the alpha grid to an honest fringe, summed over "
            "the fitting days (Hobbs-Metzler-Pang 2000 single-firm problem, grid form); "
            "not self-consistent, see docs/notes/da-economic-withholding.zh.md"),
        alpha_star=alpha.tolist(), m_star=((alpha - 1) * mc).tolist(), mc=mc.tolist(),
        classes={str(u): c for u, c in cls.items()}, counts=counts,
        unilateral_gain={str(u): units[u]["gain"] for u in sorted(units)},
        gain_by_class=gains, zero_units=honest["zero_units"],
        levels=honest["levels"], tie_rel=TIE_REL, exit_rel=EXIT_REL,
        exit_band=band,
        fit_days_kind=honest["days_kind"], fit_days=fit_env_days,
        fit_window_days=fit_win_days,
        eval_days=results["eval"]["days"],
        eval_window_days=results["eval"]["window_days"],
        horizon=cfg["horizon"],
        n_at_bound=int(sum(units[u]["at_bound"] for u in units)),
        # honest / max / withholding on both day sets, per unit and per day
        arms={k: v["arms"] for k, v in results.items()},
        fixture=fixture,
        run_point=dict(cfg["run_point"], **runtime_stamp(), commit=commit_hash(),
                       commit_dirty=_START_DIRTY, tool_blob=_blob(),
                       note=("reward is settlement profit; per-unit totals are "
                             "reward_per_agent x n_days x horizon; "
                             "system_cost_sum_per_env is production cost plus shed "
                             "energy at voll plus this market's third term (market "
                             "03: info['volr_cost']), from evaluation.system_cost's "
                             "own formula"))),
        indent=1))
    print("->", repo_relative(out), flush=True)


def _blob(path="tools/benchmark/run_withholding.py"):
    """`git rev-parse --verify HEAD:<path>`, or `"unknown"`.

    `--verify` matters: without it, a path absent from HEAD makes rev-parse echo
    the argument itself to stdout, and the product carries the literal string
    `HEAD:tools/...` where a blob should be (happened 2026-09-03 on market 01's
    first assemble, run before the file was committed).
    """
    import subprocess
    try:
        r = subprocess.run(["git", "rev-parse", "--verify", "-q", f"HEAD:{path}"],
                           capture_output=True, text=True, timeout=10,
                           cwd=str(Path(__file__).resolve().parents[2]))
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:                                      # pragma: no cover
        return "unknown"


def build_parser(market=None):
    """The CLI.  `market` pins it, which is what `run_withholding_01.py` does."""
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    for name, fn in (("sweep", sweep), ("assemble", assemble)):
        p = sub.add_parser(name)
        if market is None:
            p.add_argument("--market", required=(name == "sweep"),
                           default=None, choices=list(MARKETS))
        else:
            p.set_defaults(market=market)
        p.set_defaults(fn=fn)
    s = sub.choices["sweep"]
    s.add_argument("--levels", default="1.0:2.0:0.05")
    s.add_argument("--days", default="train36", choices=["train36", "eval"])
    s.add_argument("--shard", default="0/1", help="k/n, round-robin over units")
    s.add_argument("--units", default=None, help="comma-separated override")
    s.add_argument("--column", default="energy", choices=["energy", "reserve"],
                   help="energy: markup multiplier, levels start at 1.0 (default); "
                        "reserve (market 03): reserve offer in $/MWh for both products, "
                        "levels start at 0, energy column held at 1.0")
    s.add_argument("--uniform", action="store_true",
                   help="every unit at the same level at once, one rollout per level; "
                        "writes uniform.json instead of unit_NN.json")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--fixture", required=True,
                   help="market 01: the commitment fixture; markets 02 and 03: "
                        "the day-ahead position")
    s.add_argument("--profile", default=None,
                   help="PATH:KEY of a JSON alpha vector the rivals play (default: truthful); "
                        "`assemble` accepts only truthful-profile sweeps")
    a = sub.choices["assemble"]
    a.add_argument("--dir", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--fixture", default=None, help="defaults to the sweep's")
    return ap


def main(market=None):
    args = build_parser(market).parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
