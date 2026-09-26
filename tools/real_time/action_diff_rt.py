"""Markets 02 and 03: the same quantity as `tools/day_ahead/action_diff.py`, on a per-period market.

Read-only.  Statistics come from `tools/day_ahead/action_diff.py` so the two markets are not
measured by two devices that merely agree in prose.

The grid difference from 01 is the whole reason this is a separate file.  01 has
`episode_len = 1`, so its twelve observations are twelve resets and no action
influences any of them.  Here the episode is a whole day and the observation at
period t follows from what was played before it, so the grid is taken from ONE
reference trajectory -- the truthful arm -- and every archive is scored at those
same observations.  Letting each archive walk its own trajectory would move the
grid with the policy, and two archives' numbers would then not be on one axis.

Because the grid is two-dimensional (day x period), the range decomposes and
both halves are reported:

  R_total   over all N = days * periods observations
  R_within  per-unit range inside one day, aggregated over days: "does this
            unit's action follow the time of day"
  R_across  per-unit range across days at a FIXED period, aggregated over
            periods: "does this unit's action follow which day it is"

`R_total >= max(R_within, R_across)` always; a policy that answers only to the
time of day and a policy that answers only to the day are two different failures
and the total alone does not separate them.

The run-point constants are imported inside each branch from that market's own
script -- `run_rl_02.py` for 02, `run_eval_03.py` for 03 -- because one block
serving two markets reads like one run point when it is two, and two devices
disagreeing silently about the same number is worse than both being stale.

**One thing about market 03 is flagged rather than fixed.**
The 03 branch loads `POSITION_02` -- the DAY-AHEAD position product -- and takes
its `dates` and day split from there, where the 01 device's abandoned 03 path
called `load_da_schedule` on an ancillary schedule instead.  Whether that is
correct has not been established: no result the paper cites rests on the 03
product, and inventing a schedule path here would be inventing a run point
nobody calibrated.  It is left exactly as it was and written down here.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "benchmark"))
sys.path.insert(0, str(REPO / "tools" / "day_ahead"))

import jax                                                        # noqa: E402
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp                                           # noqa: E402

from evaluation import open_day, split_days, subset_position      # noqa: E402
from hyperparams import SHARED                                    # noqa: E402
from action_diff import (cross_unit_dedup, named_weights,         # noqa: E402
                         numpy_forward_mean, numpy_to_action,
                         per_unit_stats, q, summarise)

#: The day-ahead position this device rolls the reference trajectory over.
#: It is the same bytes as the product the original runs read (equal sha256,
#: every array equal key by key, 2026-09-07).  The run point is in the file
#: name of that original, `da_position_29gb_T24_seasons60_c0.6_r1.00.npz`:
#: `29gb`, T = 24, the four-season 60-day window, `cap_scale` 0.6,
#: `ramp_scale` 1.00.
POSITION_02 = str(REPO / "tests" / "fixtures"
                  / "day_ahead_position_29gb_T24_step1prime_seasons.npz")


def rollout_grid(reset, step_fn, params, T, n_days, action, day_of):
    """(n_days, T, n_units, obs_dim) along one reference trajectory.

    The opening observation comes from `reset` on the key `open_day` returns,
    not from a `get_obs` call on the state: market 03's environment is the
    four-tuple form and has no `get_obs`, and taking the observation from the
    same call that built the state is the form both markets share.
    """
    step = jax.jit(step_fn)
    act = jnp.asarray(action)
    out = []
    for pos in range(n_days):
        key, _state = open_day(reset, params, pos, day_of)
        obs, state = reset(key, params)
        day = []
        for _t in range(T):
            day.append(np.asarray(obs, np.float64))
            key, k = jax.random.split(key)
            obs, state, *_rest = step(k, state, act, params)
        out.append(np.stack(day))
        print(f"    day {pos}: {T} observations", flush=True)
    return np.stack(out)


def decompose(A4, low2, high2):
    """`A4` is (n_days, T, n_units, act_dim).  Return the three ranges."""
    n_days, T = A4.shape[0], A4.shape[1]
    flat = A4.reshape(n_days * T, *A4.shape[2:])
    total = per_unit_stats(flat, low2, high2)
    within = np.median(A4.max(axis=1) - A4.min(axis=1), axis=0)     # (units, dim)
    across = np.median(A4.max(axis=0) - A4.min(axis=0), axis=0)     # (units, dim)
    return total, within, across


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="02", choices=("02", "03"))
    ap.add_argument("--archives", nargs="*", default=[],
                    help="label:path; markets 02/03 archives are msgpack (02) "
                         "or flax_flat_v1 npz (03). Produced by "
                         "`tools/benchmark/run_rl_02.py` (msgpack, market 02) "
                         "or `tools/benchmark/run_rl_03.py` (flax_flat_v1 npz, "
                         "market 03); neither is shipped in this repository.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with-untrained", action="store_true",
                    help="also score this seed's initialisation, which is the "
                         "iterations-0 control for the same run point")
    ap.add_argument("--stats-from", default=None,
                    help="npz whose obs_mean/obs_std the run used; "
                         "market 03 archives carry them")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Built from args alone, so this is checked before the market build and the
    # grid rollout below run -- both take minutes, and a run given neither flag
    # used to reach the end of that work, print its diagnostics, and exit 0
    # with nothing written (`--market 02`: 900 s, exit 0, zero products).
    todo = []
    if args.with_untrained:
        todo.append(("untrained", None))
    for a in args.archives:
        label, _, path = a.partition(":")
        if not path:
            label, path = Path(a).stem, a
        todo.append((label, Path(path)))
    if not todo:
        raise SystemExit(
            "nothing to measure: give --archives and/or --with-untrained")

    import powermarketjax
    print("package:", powermarketjax.__file__, flush=True)
    print("devices:", jax.devices(), flush=True)

    from powermarketjax.learning.ippo import (make_greedy_action, make_ippo,
                                              observation_statistics)
    from powermarketjax.learning.adapters import unpack_env
    from powermarketjax.learning.policy import bounds_for

    if args.market == "02":
        from powermarketjax.case import load_case
        from powermarketjax.envs.day_ahead import load_gb_demand
        from powermarketjax.envs.real_time import load_da_position
        from powermarketjax.envs.real_time.demand import (
            T_RT, load_gb_demand_half_hourly)
        from powermarketjax.envs.real_time.env import make_env
        # 02's run point, from 02's own trainer rather than copied here.
        from run_rl_02 import CAP_SCALE, MARKUP_MAX, RAMP_SCALE

        pos = load_da_position(path=POSITION_02)
        m = pos["meta"]
        dates = [str(d) for d in m["dates"]]
        ev_days, tr_days = split_days(len(dates), m["case"])
        case = load_case(m["case"])
        hh, _ = load_gb_demand_half_hourly()
        fc, _a, _d = load_gb_demand()
        build = lambda p: make_env(case, p, hh, fc, n_segments=1,
                                   markup_max=MARKUP_MAX, cap_scale=CAP_SCALE,
                                   ramp_scale=RAMP_SCALE, n_lookahead=1)
        train_obj, spec = build(subset_position(pos, tr_days))
        eval_obj, _espec = build(subset_position(pos, ev_days))
        train_params = train_obj.make_params(episode_len=T_RT)
        eval_params = eval_obj.make_params(episode_len=T_RT)
        four = unpack_env((train_obj, spec))
        bounds = bounds_for(four[3])
        T, n_days = int(T_RT), len(ev_days)
        day_of = lambda st: int(st.cursor) // T
        ref_action = np.asarray(eval_obj.truthful_action(), np.float64)
        # `run_rl_02.py:163` derives the three keys in this order, and the
        # statistics depend on that order (see `observation_statistics`'s own
        # docstring), so the split is reproduced rather than re-invented.
        key = jax.random.PRNGKey(args.seed)
        key, k_stat, k_init = jax.random.split(key, 3)
        print("recomputing obs_mean/obs_std (the msgpack carries neither) ...",
              flush=True)
        om_j, os_j = observation_statistics(four, train_params, k_stat,
                                            SHARED.n_envs, SHARED.horizon)
        om, os_ = np.asarray(om_j, np.float64), np.asarray(os_j, np.float64)
        cfg = SHARED
        reset_fn, step_fn = eval_obj.reset, eval_obj.step
        eval_params = eval_params
    else:
        from powermarketjax.case import load_case
        from powermarketjax.envs.ancillary.env import make_ancillary_env as make_anc_env
        # 03's run point, from 03's own evaluator.  The three scale constants
        # used to be module-level literals shared with market 02; they are equal
        # (0.60 / 1.00 / 2.0) but they are two run points, not one.
        from run_eval_03 import (BETA, CAP_SCALE, CASE, DELTA, MARKUP_MAX,
                                 PI_SCALE, RAMP_SCALE, T_DAY, THETA, VOLR)
        from run_rl_03 import build_params

        fx = np.load(POSITION_02, allow_pickle=True)
        meta_fx = json.loads(str(fx["meta"])) if isinstance(
            fx["meta"].tolist(), str) else fx["meta"].tolist()
        dates = [str(d) for d in meta_fx["dates"]]
        ev_days, tr_days = split_days(len(dates), meta_fx["case"])
        posd = {k: fx[k] for k in fx.files if k != "meta"}
        posd["meta"] = meta_fx
        case = load_case(CASE)
        four = make_anc_env(case, THETA, VOLR, BETA, PI_SCALE, n_segments=1,
                            cap_scale=CAP_SCALE, ramp_scale=RAMP_SCALE,
                            period_hours=DELTA, kind="markup",
                            markup_max=MARKUP_MAX)
        spec = four[3]
        train_params = build_params(subset_position(posd, tr_days), case, jnp)
        eval_params = build_params(subset_position(posd, ev_days), case, jnp)
        bounds = bounds_for(spec, reserve_columns=int(spec["n_prod"]))
        T, n_days = int(T_DAY), len(ev_days)
        day_of = lambda st: int(st.cursor) // T
        ref_action = np.asarray(spec["baseline_action"], np.float64)
        key = jax.random.PRNGKey(args.seed)
        key, k_stat, k_init = jax.random.split(key, 3)
        om_j, os_j = observation_statistics(four, train_params, k_stat,
                                            SHARED.n_envs, SHARED.horizon)
        om, os_ = np.asarray(om_j, np.float64), np.asarray(os_j, np.float64)
        # `observation_statistics` runs a rollout, so its output depends on the
        # platform it ran on; market 03's archives CARRY the statistics the run
        # actually used, and a policy has to be scored on the standardisation it
        # was trained under.  The archived pair is therefore what is used when
        # `--stats-from` names it, and the recomputed one is then kept only to
        # report how far a CPU recomputation lands from the GPU run's.
        # **Without `--stats-from`, market 03 is scored on the CPU
        # recomputation**, not on the statistics the run used.
        if args.stats_from:
            a = np.load(args.stats_from, allow_pickle=True)
            am, astd = np.asarray(a["obs_mean"], np.float64), np.asarray(a["obs_std"], np.float64)
            print(f"obs statistics taken from {args.stats_from}; a CPU "
                  f"recomputation differs by max|d| {np.abs(am - om).max():.3e} "
                  f"(mean) / {np.abs(astd - os_).max():.3e} (std), max relative "
                  f"{np.abs((am - om) / np.where(am != 0, am, 1)).max():.3e} / "
                  f"{np.abs((astd - os_) / astd).max():.3e}", flush=True)
            om, os_ = am, astd
            om_j, os_j = jnp.asarray(om), jnp.asarray(os_)
        cfg = SHARED
        reset_fn, step_fn = four[0], four[1]

    init, _it = make_ippo(four, bounds, cfg, om_j, os_j)
    template, *_ = init(k_init, train_params)

    print(f"grid: rolling the truthful arm over {n_days} evaluation days x {T} "
          f"periods", flush=True)
    Z4 = rollout_grid(reset_fn, step_fn, eval_params, T, n_days, ref_action, day_of)
    n_units, obs_dim = Z4.shape[2], Z4.shape[3]
    Zf = Z4.reshape(n_days * T, n_units, obs_dim)

    per_unit_distinct = [int(np.unique(Zf[:, i, :], axis=0).shape[0])
                         for i in range(n_units)]
    if min(per_unit_distinct) < 2:
        raise SystemExit("observation grid is degenerate: some unit sees one "
                         "observation on every grid point")
    print(f"grid: {Zf.shape[0]} observations x {n_units} units x {obs_dim} "
          f"coords; distinct observations per unit min {min(per_unit_distinct)} "
          f"max {max(per_unit_distinct)}", flush=True)

    low = np.asarray(bounds[0], np.float64).reshape(n_units, -1)
    high = np.asarray(bounds[1], np.float64).reshape(n_units, -1)
    act_dim = low.shape[1]

    # `todo` was built and checked non-empty right after argparse, above.
    from flax import serialization as fser
    rows = []
    for label, path in todo:
        if path is None:
            params = template
        elif str(path).endswith(".msgpack"):
            params = fser.from_bytes(template, Path(path).read_bytes())
        else:
            d = np.load(path, allow_pickle=True)
            tree = {}
            for k in d.files:
                if not k.startswith("params/"):
                    continue
                parts = k.split("/")[1:]
                node = tree
                for pp in parts[:-1]:
                    node = node.setdefault(pp, {})
                node[parts[-1]] = jnp.asarray(np.asarray(d[k], np.float64))
            params = {"params": tree}
        greedy = jax.jit(make_greedy_action(four[3], bounds, cfg, om_j, os_j))
        A = np.stack([np.asarray(greedy(params, jnp.asarray(z)), np.float64)
                      for z in Zf])
        if A.ndim == 2:
            A = A[:, :, None]
        w = named_weights(params)
        A_np = np.stack([numpy_to_action(numpy_forward_mean(w, (z - om) / os_),
                                         low, high) for z in Zf])
        dpath = float(np.abs(A - A_np).max())
        A4 = A.reshape(n_days, T, n_units, act_dim)
        st, within, across = decompose(A4, low, high)
        dedup = cross_unit_dedup(A)
        for d in range(act_dim):
            s = summarise(st, d, None, "all units")
            print(f"  {label:>10} coord {d}: R median {s['range_median']:.6f} "
                  f"({s['share_median']*100:.3f}% of span)  "
                  f"R q0/q25/q75/q1 {s['range_q'][0]:.4f}/{s['range_q'][1]:.4f}/"
                  f"{s['range_q'][3]:.4f}/{s['range_q'][4]:.4f}  "
                  f"L median {s['levels_median']:.1f}  flat {s['n_flat']}  "
                  f"within-day median {np.median(within[:, d]):.6f}  "
                  f"across-day median {np.median(across[:, d]):.6f}  "
                  f"cross-unit dedup {min(dedup)}-{max(dedup)}  "
                  f"2-path |d| {dpath:.1e}", flush=True)
            rows.append(dict(
                label=label, coord=int(d), n_obs=int(Zf.shape[0]),
                n_units=int(n_units), n_days=int(n_days), n_periods=int(T),
                two_path_max_abs_diff=dpath,
                cross_unit_dedup_min=int(min(dedup)),
                cross_unit_dedup_max=int(max(dedup)),
                all_units=s,
                within_day_median=float(np.median(within[:, d])),
                within_day_q=q(within[:, d]),
                across_day_median=float(np.median(across[:, d])),
                across_day_q=q(across[:, d]),
                range_per_unit=[float(x) for x in st["range"][:, d]],
                levels_per_unit=[int(x) for x in st["levels"][:, d]]))

    if args.out:
        np.savez(args.out, rows=json.dumps(rows),
                 meta=json.dumps(dict(market=args.market, seed=args.seed,
                                      n_days=int(n_days), n_periods=int(T),
                                      n_units=int(n_units))),
                 obs_grid_distinct_per_unit=np.asarray(per_unit_distinct))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
