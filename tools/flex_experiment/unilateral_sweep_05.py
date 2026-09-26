"""One aggregator raising its offer alone, the others held at the floor (-128, 0, +128): its own return as a function of its own offer level.

Data for the right-hand panel of figure 5; the products of the four configurations are one JSON file each.
The deviator is, in each configuration, the aggregator that earns the most at the 9 900 level (the recomputation
of the published best fixed action: 7 / 7 / 12 / 12).  The apparatus is borrowed whole from
`tools/flex_experiment/deviation_05.py`: the same scenario, the same 36 test days, the same `PRNGKey(999)`,
the same `log_std = -12`, and the same jit path `Scorer` (the one that recomputation's unilateral readings were swept with).
The only difference from deviation_05: the deviator's markup is not pinned at the winner's +65.05 but takes several levels along the markup axis.

    JAX_PLATFORMS=cpu PYTHONPATH=.:tools/flex_experiment taskset -c 56-63 \
        python tools/flex_experiment/unilateral_sweep_05.py --cell 0 --deviator 7 \
        --out unilateral_sweep_2040p_2040c.json
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, jax
from concentration_baseline import CELLS, EPISODE_LEN, day_split, obs_scale, scenario, series_day_of_month
from deviation_05 import FLOOR, WINNER, REPORT_KEY, Scorer, per_episode, f64mean, published_ret
from monitor_probe import LOG_STD as LOG_STD_PUBLISHED

C_REP = 149.882355
LEVELS = [-128.0, -6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 20.0,
          29.010360886042925, 38.02072177208585, 47.031082658128774, 56.0414435441717, 65.05180443021462]

def price_of(a):
    return C_REP * (1.0 + np.log1p(np.exp(a)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=int, default=0)
    ap.add_argument("--deviator", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    place, cap = CELLS[args.cell]; name = f"{place}p_{cap}c"
    _, dom = series_day_of_month(); _, val, test = day_split(dom, EPISODE_LEN)
    env, params, n = scenario(place, cap, split="train"); scale = obs_scale(env, params, n)
    t0 = time.time()
    score = Scorer(env, params, scale, n, test, LOG_STD_PUBLISHED)
    j = args.deviator
    # Known-correct states: everyone at the floor and everyone at the winner; the jit path against the published 5.390 / 485.028 etc.
    allf = score(FLOOR); allw = score(WINNER)
    ret_floor = published_ret(allf["per_episode"]); ret_win = published_ret(allw["per_episode"])
    print(f"{name}: all_floor {ret_floor:.6f}  all_winner {ret_win:.6f}  (compile+2 calls {time.time()-t0:.0f}s)", flush=True)
    rows = []
    for a in LEVELS:
        act = np.tile(np.asarray(FLOOR), (n, 1)); act[j] = (a, 0.0, 128.0)
        t = time.time(); r = score(act); pe = r["per_episode"]
        own = float(f64mean(pe[:, j])); others = float(f64mean(np.delete(pe, j, axis=1)))
        rows.append(dict(alpha_pi=a, price=float(price_of(a)), own_ret=own, own_gain_over_floor=own - float(f64mean(allf["per_episode"][:, j])),
                         own_episodes_positive=int((pe[:, j] > 1e-9).sum()), others_ret=others,
                         price_given_cleared=r["price_given_cleared"], clearing_fraction=r["clearing_fraction"],
                         sweep_converged=r["sweep_converged"], seconds=time.time() - t))
        print(f"  alpha {a:9.4f} price {rows[-1]['price']:9.2f} own {own:10.4f} gain {rows[-1]['own_gain_over_floor']:+10.4f} pos {rows[-1]['own_episodes_positive']:2d}/36 others {others:8.4f} conv {r['sweep_converged']} ({rows[-1]['seconds']:.0f}s)", flush=True)
    # One check against the published path: the deviator at the winner level, jit minus per_episode (the same path check as deviation_05)
    act = np.tile(np.asarray(FLOOR), (n, 1)); act[j] = WINNER
    r_pub = per_episode(env, params, scale, n, act, test, LOG_STD_PUBLISHED)
    own_pub = float(f64mean(r_pub["per_episode"][:, j])); own_jit = rows[-1]["own_ret"]
    print(f"  path check deviator {j} at winner: jit {own_jit:.6f} published-path {own_pub:.6f} diff {own_jit-own_pub:+.3e}", flush=True)
    out = dict(cell=name, n_agent=n, deviator=j, days="test", eval_episodes=int(len(test)), log_std=LOG_STD_PUBLISHED,
               report_key=f"PRNGKey({REPORT_KEY})", floor=list(FLOOR), winner=list(WINNER), c_rep=C_REP,
               all_floor_ret=ret_floor, all_winner_ret=ret_win, floor_own_ret=float(f64mean(allf["per_episode"][:, j])),
               rows=rows, path_check=dict(own_jit=own_jit, own_published_path=own_pub), seconds=time.time() - t0)
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}  total {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
