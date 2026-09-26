"""Market 05's best fixed action, re-scored three ways the published cell did not
try: one aggregator deviating alone, a wider exploration width, and the
validation days.

The published best-fixed-action product scores the
winner `(+65.0518, 0, +128)` with **every** aggregator playing it.
An audit of the results (2026-09-09) listed three
measurements that cell does not contain, all zero-GPU, and this file is those
three on the same device -- `concentration_baseline`'s scenario, day split,
observation scale, rollout and reduction, `monitor_probe.constant_policy` for
the constant arm, the same `PRNGKey(999)` -- so that a number here and the
published one are the same kind of object.  Nothing in the scenario or the
evaluation is restated here.

**Z-2, unilateral deviation** (`--deviators`).  Three profiles per
configuration: (i) every aggregator at the floor `(-128, 0, +128)` -- the
truthful price, half the deliverable quantity offered, the planned charge at
its maximum, i.e. the winner with its markup removed; (ii) one aggregator at
the winner and the rest at the floor; (iii) every aggregator at the winner,
which is the published cell and is rescored here as the self-check.  Which
aggregator deviates is a parameter of this device that no published number
fixes, so `--deviators all` runs (ii) once per aggregator and the product
carries the whole distribution; a single index is accepted for a quick look.
`constant_policy` takes an `(n, 3)` array for this: `forward` adds `mean_b` to
`h @ mean`, which is `(n, 3)` already, so an `(n, 3)` bias broadcasts without
touching the network.

**Z-3, exploration width** (`--log-std`).  The published cell uses -12, the 28
learner seeds were scored at `LOG_STD_FINAL = -3.0`; the winner is rescored at
-3.0 with nothing else changed.  The noise enters as `(mean + noise) *
ACTION_GAIN` in `make_rollout`, so at -3.0 the action carries a standard
deviation of 0.199 in the units the environment sees.

**Z-4, validation days** (`--days validation`).  The published cell is scored on
the 36 test days (`TEST_DAYS = (1, 11, 21)`); the learners' checkpoints were
*selected* on the 36 validation days (`VALIDATION_DAYS = (5, 15, 25)`) and their
`select_score` is that selection maximum.  The winner and the four reference
arms are rescored on the validation pool; the key stays `PRNGKey(999)`, which is
the report key and not the selection key `PRNGKey(12345)`.

Every run rescores the three published constant strategies through both
`evaluate` and this file's per-episode reduction and refuses a mismatch, as
`bestfixed_05_grid --selfcheck` does; on the test days at -12 it also requires
the all-winner profile to reproduce `winner.ret` of the published product to
1e-9 relative, so a run that would report a new number on a device that no
longer reproduces the old one stops instead.

    JAX_PLATFORMS=cpu PYTHONPATH=.:tools/flex_experiment python -m deviation_05 \\
        --deviators all --out z2.json
    JAX_PLATFORMS=cpu PYTHONPATH=.:tools/flex_experiment python -m deviation_05 \\
        --log-std -3.0 --deviators none --out z3.json
    JAX_PLATFORMS=cpu PYTHONPATH=.:tools/flex_experiment python -m deviation_05 \\
        --days validation --deviators none --reference-arms --out z4.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from concentration_baseline import (ACTION_GAIN, CELLS, EPISODE_LEN,
                                    INCDEC_ACTION, NEVER_CLEARS_ACTION,
                                    day_split, evaluate, init_policy,
                                    make_rollout, obs_scale, scenario,
                                    series_day_of_month)
from monitor_probe import LOG_STD as LOG_STD_PUBLISHED

#: The published winner of the markup-axis extension (the best fixed action that
#: beats every learning seed): markup to 9 900
#: CHF/MWh, half the deliverable quantity, planned charge saturated.
WINNER = (65.05180443021462, 0.0, 128.0)
#: The winner with its markup removed: the truthful price, the other two
#: coordinates unchanged.  This is the "nobody deviates" profile of Z-2.
FLOOR = (-128.0, 0.0, 128.0)
PUBLISHED = (("truthful", (-128.0, 128.0, -128.0)),
             ("never_clears", tuple(NEVER_CLEARS_ACTION)),
             ("incdec", tuple(INCDEC_ACTION)))
REFERENCE_MODES = (("truthful", 1), ("random", 2), ("never_clears", 3),
                   ("incdec", 4))
REPORT_KEY = 999
PUBLISHED_PRODUCT = Path("docs/figures/fixed-action-04-05/05-bestfixed-markup-ext.json")


def constant_policy(obs_dim, action):
    """`monitor_probe.constant_policy`, accepting `(3,)` or `(n, 3)`.

    Weights zeroed, so `forward` returns `mean_b` for every aggregator; an
    `(n, 3)` bias gives each aggregator its own constant.
    """
    p = init_policy(jax.random.PRNGKey(0), obs_dim)
    z = jax.tree.map(jnp.zeros_like, p)
    return {**z, "mean_b": jnp.asarray(action, jnp.float32) / ACTION_GAIN}


def per_episode(env, params, scale, n, action, starts, log_std):
    """`(episode, participant)` returns on the published path.

    `bestfixed_05_grid.per_episode_returns` with the policy taking an `(n, 3)`
    action and `log_std` passed in rather than read from `monitor_probe`: the
    rollout is rebuilt and dispatched op by op on every call, exactly as the
    published cell was scored, so a number from here and `winner.ret` of the
    product are the same arithmetic.
    """
    roll = make_rollout(env, params, scale, n, log_std)
    starts = jnp.asarray(starts, jnp.int32)
    return _reduce(jax.vmap(roll, in_axes=(None, 0, None, 0))(
        constant_policy(env[3]["obs_dim"], action),
        jax.random.split(jax.random.PRNGKey(REPORT_KEY), starts.shape[0]),
        0, starts))


class Scorer:
    """`per_episode` compiled once, for the sweep over every deviator.

    Rebuilding the rollout per call costs minutes on a few pinned CPU cores,
    so the sweep over all `n` deviators uses one `jax.jit` of the same vmapped
    rollout with the policy as an argument.  **It is not the published
    arithmetic**: XLA fuses and reorders float32 work, and on the truthful arm
    the two paths differ at the 1e-8 level (measured 2026-09-15, -6.706e-08
    against -7.503e-08 on a quantity that is a zero).  So the three cells the
    audit asked for are scored on `per_episode`, and this path is used only
    for the sweep, with its difference to `per_episode` measured on two
    profiles it shares with the sweep and written into the product.
    """

    def __init__(self, env, params, scale, n, starts, log_std):
        self.obs_dim, self.n = env[3]["obs_dim"], n
        self.starts = jnp.asarray(starts, jnp.int32)
        self.keys = jax.random.split(jax.random.PRNGKey(REPORT_KEY),
                                     self.starts.shape[0])
        roll = make_rollout(env, params, scale, n, log_std)
        self._run = jax.jit(jax.vmap(roll, in_axes=(None, 0, None, 0)))

    def __call__(self, action):
        return _reduce(self._run(constant_policy(self.obs_dim, action),
                                 self.keys, 0, self.starts))


def _reduce(out):
    """`(episode, participant)` returns plus the clearing diagnostics."""
    reward, vol, price = np.asarray(out[4]), np.asarray(out[6]), np.asarray(out[7])
    mu, converged = np.asarray(out[12]), np.asarray(out[13])
    cleared = vol > 1e-9
    return dict(
        per_episode=reward.sum(1),                       # (episode, participant)
        # (episode, period, participant): which participant was paid when
        paid=np.abs(reward) > 1e-9,
        clearing_fraction=float(cleared.mean()),
        price_given_cleared=(float(price[cleared].mean()) if cleared.any()
                             else 0.0),
        volume_given_cleared=(float(vol[cleared].mean()) if cleared.any()
                              else 0.0),
        mu_max=float(mu.max()), sweep_converged=bool(converged.all()))


def f64mean(x, axis=None):
    """Mean in float64 of a float32 array."""
    return np.asarray(x, np.float64).mean(axis=axis)


def published_ret(pe):
    """`winner.ret` the way `bestfixed_05_grid` reduces it.

    That file takes each cell's per-episode mean over participants in float32
    (`r["per_episode"].mean(1)`), stores those 36 numbers into a float64 array
    and takes the winner's `ret` as their float64 mean, so the published
    485.0281344878453 is a float64 mean of float32 per-episode means; a plain
    float32 `.mean()` of the same array gives 485.02813720703125, 5.6e-9
    relative away, and the reproduction check has to compare like with like.
    """
    return float(np.asarray(pe.mean(1), np.float64).mean())


def summary(r):
    pe = r["per_episode"]
    return dict(ret=published_ret(pe), ret_f32=float(pe.mean()),
                ret_per_agent=f64mean(pe, 0).tolist(),
                ret_median_over_episodes=float(np.median(f64mean(pe, 1))),
                market_total=float(f64mean(pe.sum(1))),
                clearing_fraction=r["clearing_fraction"],
                price_given_cleared=r["price_given_cleared"],
                volume_given_cleared=r["volume_given_cleared"],
                mu_max=r["mu_max"], sweep_converged=r["sweep_converged"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="",
                    help="comma separated indices into CELLS; empty means all")
    ap.add_argument("--log-std", type=float, default=LOG_STD_PUBLISHED)
    ap.add_argument("--days", choices=("test", "validation"), default="test")
    ap.add_argument("--deviators", default="none",
                    help="'none', 'all', or a comma separated list of "
                         "aggregator indices that deviate alone to the winner")
    ap.add_argument("--reference-arms", action="store_true",
                    help="also score the four reference arms (modes 1-4) "
                         "through `evaluate` on the chosen days")
    ap.add_argument("--published", type=Path, default=PUBLISHED_PRODUCT,
                    help="product whose `winner.ret` the all-winner profile "
                         "must reproduce on the test days at the published "
                         "log_std; elsewhere it is only recorded")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _, dom = series_day_of_month()
    train, val, test = day_split(dom, EPISODE_LEN)
    starts = test if args.days == "test" else val
    assert not (set(np.asarray(val).tolist()) & set(np.asarray(test).tolist()))
    cells = (CELLS if not args.cells
             else tuple(CELLS[int(i)] for i in args.cells.split(",")))
    published = (json.loads(args.published.read_text())["results"]
                 if args.published.exists() else {})
    strict = args.days == "test" and args.log_std == LOG_STD_PUBLISHED

    results = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for place, cap in cells:
        name = f"{place}p_{cap}c"
        env, params, n = scenario(place, cap, split="train")
        scale = obs_scale(env, params, n)
        obs_dim = env[3]["obs_dim"]
        t_cell = time.time()
        entry = dict(placement=place, capacity=cap, n_agent=n,
                     days=args.days, eval_episodes=int(len(starts)),
                     starts=np.asarray(starts).tolist(),
                     log_std=args.log_std, report_key=f"PRNGKey({REPORT_KEY})",
                     winner=list(WINNER), floor=list(FLOOR))
        arrays = {}

        # -- self-check: both reductions agree on the three published arms ----
        checks = []
        for label, act in PUBLISHED:
            shared = evaluate(env, params, scale, n,
                              constant_policy(obs_dim, list(act)),
                              jax.random.PRNGKey(REPORT_KEY), 0, args.log_std,
                              eval_starts=starts)
            mine = per_episode(env, params, scale, n, act, starts, args.log_std)
            # float32 mean on both sides, as `bestfixed_05_grid --selfcheck` compares
            got = float(mine["per_episode"].mean())
            if abs(got - shared["ret"]) > 1e-9 * max(1.0, abs(shared["ret"])):
                raise SystemExit(
                    f"{name} {label}: this file's reduction gives {got!r} and "
                    f"`evaluate` gives {shared['ret']!r}; the two paths are not "
                    f"scoring the same thing")
            checks.append(dict(arm=label, action=list(act), ret=shared["ret"],
                               ret_this_file=got,
                               clearing_fraction=shared["clearing_fraction"],
                               price_given_cleared=shared["price_given_cleared"]))
            print(f"  selfcheck {name} {label:<13} ret {shared['ret']:+12.6f} "
                  f"(both paths)", flush=True)
        entry["selfcheck"] = checks

        # -- the four reference arms, which ignore the policy ---------------
        if args.reference_arms:
            dummy = init_policy(jax.random.PRNGKey(0), obs_dim)
            ref = {}
            for label, mode in REFERENCE_MODES:
                r = evaluate(env, params, scale, n, dummy,
                             jax.random.PRNGKey(REPORT_KEY), mode, args.log_std,
                             eval_starts=starts)
                ref[label] = {k: r[k] for k in ("ret", "ret_per_agent",
                                                "market_total", "clearing_fraction",
                                                "price_given_cleared",
                                                "volume_given_cleared", "shed_mwh",
                                                "mu_max", "sweep_converged")}
                print(f"  reference {name} {label:<13} ret {r['ret']:+12.6f}",
                      flush=True)
            entry["reference"] = ref

        # -- (iii) everybody at the winner: the published cell ----------------
        t = time.time()
        allw = per_episode(env, params, scale, n, WINNER, starts, args.log_std)
        entry["all_winner"] = dict(summary(allw), seconds=time.time() - t)
        arrays["all_winner"] = allw["per_episode"]
        pub = published.get(name, {}).get("winner", {}).get("ret")
        entry["all_winner"]["published_ret"] = pub
        if pub is not None:
            d = abs(entry["all_winner"]["ret"] - pub) / max(1.0, abs(pub))
            entry["all_winner"]["rel_diff_to_published"] = d
            if strict and d > 1e-9:
                raise SystemExit(
                    f"{name}: all-winner profile scores "
                    f"{entry['all_winner']['ret']!r} here and {pub!r} in "
                    f"{args.published}; the device no longer reproduces the "
                    f"published cell, so nothing new is reported")
        print(f"  {name} all_winner ret {entry['all_winner']['ret']:+12.6f}  "
              f"published {pub}  clr {allw['clearing_fraction']:.4f}  "
              f"price|clr {allw['price_given_cleared']:9.2f}", flush=True)

        # -- (i) everybody at the floor ---------------------------------------
        t = time.time()
        allf = per_episode(env, params, scale, n, FLOOR, starts, args.log_std)
        entry["all_floor"] = dict(summary(allf), seconds=time.time() - t)
        arrays["all_floor"] = allf["per_episode"]
        print(f"  {name} all_floor  ret {entry['all_floor']['ret']:+12.6f}  "
              f"clr {allf['clearing_fraction']:.4f}  "
              f"price|clr {allf['price_given_cleared']:9.2f}", flush=True)

        # -- (ii) one aggregator at the winner, the rest at the floor ---------
        if args.deviators != "none":
            devs = (list(range(n)) if args.deviators == "all"
                    else [int(i) for i in args.deviators.split(",")])
            rows = []
            own_dev = np.zeros((len(devs), len(starts)))
            # The sweep runs on the compiled path; the first deviator is also
            # scored on the published path and the two are compared, as is the
            # all-winner profile, so the product says how far the compiled
            # arithmetic sits from the published one on this configuration.
            score = Scorer(env, params, scale, n, starts, args.log_std)
            jw = score(WINNER)["per_episode"]
            path_diff = dict(all_winner_max_abs=float(np.abs(jw - allw["per_episode"]).max()),
                             all_winner_ret_jit=published_ret(jw))
            for row_i, j in enumerate(devs):
                act = np.tile(np.asarray(FLOOR), (n, 1))
                act[j] = WINNER
                t = time.time()
                r = score(act)
                if row_i == 0:
                    r_pub = per_episode(env, params, scale, n, act, starts, args.log_std)
                    path_diff["first_deviator"] = int(j)
                    path_diff["first_deviator_max_abs"] = float(
                        np.abs(r_pub["per_episode"] - r["per_episode"]).max())
                    path_diff["first_deviator_own_ret_published_path"] = float(
                        f64mean(r_pub["per_episode"][:, j]))
                    print(f"  {name} path check: deviator {j} jit vs published "
                          f"max|diff| {path_diff['first_deviator_max_abs']:.3e}; "
                          f"all_winner {path_diff['all_winner_max_abs']:.3e}", flush=True)
                pe = r["per_episode"]
                others = np.delete(np.arange(n), j)
                own_dev[row_i] = pe[:, j]
                rows.append(dict(
                    deviator=int(j),
                    # the deviator's own return, CHF per episode, and its gain
                    # over staying at the floor, paired on the same episodes
                    own_ret=float(f64mean(pe[:, j])),
                    own_gain_over_floor=float(f64mean(pe[:, j] - allf["per_episode"][:, j])),
                    own_episodes_positive=int((pe[:, j] > 1e-9).sum()),
                    own_periods_paid=int(r["paid"][:, :, j].sum()),
                    # everybody else, mean over the other n-1 aggregators
                    others_ret=float(f64mean(pe[:, others])),
                    others_change_from_floor=float(
                        f64mean(pe[:, others] - allf["per_episode"][:, others])),
                    others_max_abs_change=float(np.abs(
                        f64mean(pe[:, others], 0) - f64mean(allf["per_episode"][:, others], 0)).max()),
                    market_total=float(f64mean(pe.sum(1))),
                    clearing_fraction=r["clearing_fraction"],
                    price_given_cleared=r["price_given_cleared"],
                    volume_given_cleared=r["volume_given_cleared"],
                    mu_max=r["mu_max"], sweep_converged=r["sweep_converged"],
                    seconds=time.time() - t))
                print(f"  {name} deviator {j:2d}: own {rows[-1]['own_ret']:+10.4f} "
                      f"(floor {f64mean(allf['per_episode'][:, j]):+8.4f}, paid in "
                      f"{rows[-1]['own_periods_paid']} periods)  others "
                      f"{rows[-1]['others_ret']:+8.4f} "
                      f"(d {rows[-1]['others_change_from_floor']:+.2e})  "
                      f"price|clr {r['price_given_cleared']:9.2f}  "
                      f"{rows[-1]['seconds']:.1f}s", flush=True)
            own = np.array([r["own_ret"] for r in rows])
            entry["unilateral"] = dict(
                deviators=devs, rows=rows,
                own_ret_min=float(own.min()), own_ret_median=float(np.median(own)),
                own_ret_max=float(own.max()),
                own_ret_max_over_all_winner=float(own.max() / entry["all_winner"]["ret"]),
                own_ret_max_deviator=int(devs[int(np.argmax(own))]),
                others_max_abs_change=float(max(r["others_max_abs_change"] for r in rows)),
                path=("jit-compiled vmap of the same rollout; see path_diff"),
                path_diff=path_diff)
            arrays["unilateral_own"] = own_dev
            arrays["deviators"] = np.asarray(devs)

            # -- injection: the deviator sent back to the floor is profile (i)
            act = np.tile(np.asarray(FLOOR), (n, 1))
            act[devs[0]] = FLOOR
            inj = per_episode(env, params, scale, n, act, starts, args.log_std)
            diff = np.abs(inj["per_episode"] - allf["per_episode"]).max()
            entry["injection_no_deviation"] = dict(
                deviator=int(devs[0]), max_abs_diff_to_all_floor=float(diff),
                identical=bool(diff == 0.0))
            print(f"  {name} injection (deviator {devs[0]} back at the floor): "
                  f"max |diff| to all_floor = {diff:.3e}", flush=True)

        entry["seconds"] = time.time() - t_cell
        results[name] = entry
        args.out.write_text(json.dumps(dict(
            winner=list(WINNER), floor=list(FLOOR), log_std=args.log_std,
            days=args.days, report_key=f"PRNGKey({REPORT_KEY})",
            reduction=("mean over participants of the per-episode sum over the "
                       "24 periods, CHF per participant per episode; "
                       "`own_ret` is the deviator's own per-episode sum, CHF "
                       "per episode"),
            results=results), indent=1, default=float))
        np.savez(args.out.with_suffix(f".{name}.npz"),
                 starts=np.asarray(starts), log_std=args.log_std, **arrays)
        print(f"  {name} done in {entry['seconds']:.0f}s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
