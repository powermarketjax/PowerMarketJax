"""Market 05's best fixed action: the joint grid over the three action coordinates.

Matrix table 1 note 1 leaves this cell undefined for 05 as well, because this
market's action is not a markup either: it is a price scale, a withholding
fraction and a planned charge.  The report page says in as many words that no
strategy it reports is a searched best open-loop constant; this file is that
search, on the sampling matrix section 5 fixes for this market -- the 36
held-out test days, enumerated rather than sampled, so every cell of the grid is
scored on exactly the same episodes and only paired differences inside one
configuration are ever taken.

**Nothing here restates the scenario or the evaluation.**  `concentration_baseline`
supplies the four configurations, the day split, the observation scale, the
rollout and the reduction; `monitor_probe.constant_policy` is the degenerate
network that emits one constant action, which is the same device the monitor
probe already uses, so a cell of this grid and an arm of that probe are the same
kind of object.  `LOG_STD` is -12 for the reason that probe records: at
`LOG_STD_FINAL` the sampled action is not constant.

**The grid is declared here and its rule is stated.**  Neither axis is placed by
looking at a return.  The action map (`envs/local_flexibility/action.py`) is
defined on all of R^3 and bounded by softplus and sigmoid, so an "endpoint of
the action space" means a saturation point, and +-128 is where the market's own
reference strategies already sit and where float32 makes the saturation exact.

    ALPHA_PI   the two saturation points -- -128 is exactly the offer floor,
               which is the truthful price, and +128 is the never-clearing
               strategy -- plus four interior points of the softplus ladder at
               0, 2, 5 and 20, whose price multipliers over the replacement cost
               are 1.693, 3.127, 6.007 and 21.000.
    ALPHA_Q    the two saturation points, 0, and +-2, giving offered fractions
               0, 0.119, 0.5, 0.881 and 1 of the deliverable quantity.
    ALPHA_CH   the same five, giving planned charge as the same five fractions
               of the charging headroom.

The three published constant strategies are grid points by construction --
truthful is (-128, +128, -128), never_clears is (+128, +128, -128) and inc-dec
is (-128, +128, +128) -- so this run reproduces three already published numbers
before it reports any new one, and `--selfcheck` requires it to.

    JAX_PLATFORMS=cpu PYTHONPATH=.:tools/flex_experiment python -m bestfixed_05_grid \
        --cells 0 --out bestfixed_grid_cell0.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from concentration_baseline import (CELLS, EPISODE_LEN, INCDEC_ACTION,
                                    NEVER_CLEARS_ACTION, day_split, evaluate,
                                    make_rollout, obs_scale, scenario,
                                    series_day_of_month)
from monitor_probe import LOG_STD, constant_policy

#: Column 0, the markup over the replacement cost.  See the module docstring.
ALPHA_PI = (-128.0, 0.0, 2.0, 5.0, 20.0, 128.0)
#: Column 1, the fraction of the deliverable quantity actually offered.
ALPHA_Q = (-128.0, -2.0, 0.0, 2.0, 128.0)
#: Column 2, the fraction of the charging headroom planned.
ALPHA_CH = (-128.0, -2.0, 0.0, 2.0, 128.0)

#: The replacement cost this feeder's action map produces, CHF/MWh.  It is a
#: constant here and not a per-agent or per-period array: `energy_price` is
#: annual and flat (`concentration_baseline` asserts that), `CYCLE_COST` is one
#: number for every aggregator, and every device in the four configurations
#: carries the same round-trip efficiency, so `c_rep = cyc (1 + 1/eta) +
#: price/eta` collapses to one value.  It is written here so the markup axis can
#: be placed in CHF/MWh rather than in raw action units, and `--selfcheck`
#: recomputes it from `params` and refuses a mismatch rather than trusting it.
C_REP = 149.882355
#: The value of lost load the clearing objective charges shedding at, CHF/MWh.
#: An offer above it never clears, so it is the ceiling of the markup axis.
VOLL = 10000.0
#: Column 0 continued: the segment `ALPHA_PI` leaves unswept.  Its top point
#: `+20` prices at 21 x C_REP = 3147.53 CHF/MWh and the next point `+128` never
#: clears by construction, so everything between 3147.53 and VOLL is unsampled
#: and the winner found on `ALPHA_PI` alone is a lower bound rather than a
#: searched optimum.  **The rule, stated before any return was looked at:**
#: five levels evenly spaced *in price* between that incumbent price and 99% of
#: VOLL, the incumbent excluded and the top level at 9900 CHF/MWh -- close to
#: the ceiling without crossing it.  The action value is recovered from the
#: price by inverting `price = C_REP (1 + softplus(alpha))`; softplus is the
#: identity to float32 at every value here (the smallest is 29, where the
#: correction is 2.5e-13 against a float32 spacing of 1.9e-06), so the
#: inversion is `price / C_REP - 1` exactly.
ALPHA_PI_PRICES = tuple(
    21.0 * C_REP + k * (0.99 * VOLL - 21.0 * C_REP) / 5.0 for k in range(1, 6))
ALPHA_PI_HIGH = tuple(p / C_REP - 1.0 for p in ALPHA_PI_PRICES)
#: Column 0 continued again: the last 1% that `ALPHA_PI_HIGH` still leaves.  Its
#: top point prices at 9900, which is 0.99 VOLL by that axis's own rule, so
#: (9900, VOLL) is unsampled and the winner found on it is again a lower bound
#: -- one whose gap an earlier note bounds at 0.96% to 1.02% by extrapolating the
#: slope of the last two levels.  **The rule, stated before any return was
#: looked at:** the four quarter points of (9900, 10000), plus one level 1
#: CHF/MWh below the ceiling.  If every one of them clears and the returns stay
#: monotone, that extrapolated bound is confirmed by measurement instead of by
#: a straight line; if one of them stops clearing, the ceiling is real and
#: lower than VOLL.
ALPHA_PI_TOP_PRICES = (9920.0, 9940.0, 9960.0, 9980.0, 9999.0)
ALPHA_PI_TOP = tuple(p / C_REP - 1.0 for p in ALPHA_PI_TOP_PRICES)

#: The published constant strategies, which are grid points.  `--selfcheck`
#: scores these through the shared `evaluate` and requires the two paths to
#: agree before any new number is reported.
PUBLISHED = (("truthful", (-128.0, 128.0, -128.0)),
             ("never_clears", tuple(NEVER_CLEARS_ACTION)),
             ("incdec", tuple(INCDEC_ACTION)))

REPORT_KEY = 999


def per_episode_returns(env, params, scale, n, action, starts):
    """One number per held-out episode per participant, plus the solver gate.

    The rollout is `concentration_baseline.make_rollout`, called the way
    `evaluate` calls it -- same vmap signature, same pinned starts, same
    terminal settlement inside it -- and only the reduction differs: `evaluate`
    averages the episode axis away and this keeps it, because a paired
    difference and a sign count both live on that axis.  `--selfcheck` requires
    the mean of what this returns to equal `evaluate`'s `ret`.
    """
    roll = make_rollout(env, params, scale, n, LOG_STD)
    starts = jnp.asarray(starts, jnp.int32)
    out = jax.vmap(roll, in_axes=(None, 0, None, 0))(
        constant_policy(env[3]["obs_dim"], list(action)),
        jax.random.split(jax.random.PRNGKey(REPORT_KEY), starts.shape[0]),
        0, starts)
    reward, vol, price = np.asarray(out[4]), np.asarray(out[6]), np.asarray(out[7])
    mu, converged = np.asarray(out[12]), np.asarray(out[13])
    cleared = vol > 1e-9
    return dict(
        # (episode, participant) in CHF per participant per episode
        per_episode=reward.sum(1),
        clearing_fraction=float(cleared.mean()),
        price_given_cleared=(float(price[cleared].mean()) if cleared.any()
                             else 0.0),
        volume_given_cleared=(float(vol[cleared].mean()) if cleared.any()
                              else 0.0),
        mu_max=float(mu.max()), sweep_converged=bool(converged.all()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="",
                    help="comma separated indices into CELLS; empty means all")
    ap.add_argument("--selfcheck", type=int, default=1)
    ap.add_argument("--resume-from", nargs="*", default=[],
                    help="products of earlier runs of this file whose finished "
                         "cells this one may reuse. Matched by the (pi, q, ch) "
                         "triple and not by position, so a sweep can be "
                         "re-split across more processes on a different slice "
                         "of the markup axis; the test days must be identical "
                         "and a mismatch is refused rather than ignored")
    ap.add_argument("--alpha-pi", type=float, nargs="+", default=list(ALPHA_PI),
                    help="a subset of the markup axis, so one configuration can "
                         "be split across processes; the product records the "
                         "axis actually swept and whether it is the default")
    ap.add_argument("--alpha-pi-top", action="store_true",
                    help="sweep ALPHA_PI_TOP instead: the last 1%% between "
                         "9900 CHF/MWh and VOLL that --alpha-pi-high leaves")
    ap.add_argument("--alpha-pi-high", action="store_true",
                    help="sweep ALPHA_PI_HIGH instead of --alpha-pi: the five "
                         "levels between the incumbent winner's price and 99%% "
                         "of VOLL that the default axis leaves unsampled. The "
                         "product records the axis actually swept")
    ap.add_argument("--alpha-q", type=float, nargs="+", default=list(ALPHA_Q),
                    help="a subset of the offered-fraction axis, so a run can "
                         "pin it at the incumbent winner's value instead of "
                         "re-sweeping it; recorded like --alpha-pi")
    ap.add_argument("--alpha-ch", type=float, nargs="+", default=list(ALPHA_CH),
                    help="a subset of the planned-charge axis; recorded like "
                         "--alpha-pi")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    alpha_pi = (ALPHA_PI_TOP if args.alpha_pi_top else
                ALPHA_PI_HIGH if args.alpha_pi_high else tuple(args.alpha_pi))
    alpha_q, alpha_ch = tuple(args.alpha_q), tuple(args.alpha_ch)
    _, dom = series_day_of_month()
    _, _val, test = day_split(dom, EPISODE_LEN)
    cells = (CELLS if not args.cells
             else tuple(CELLS[int(i)] for i in args.cells.split(",")))

    results = {}
    for place, cap in cells:
        name = f"{place}p_{cap}c"
        env, params, n = scenario(place, cap, split="train")
        scale = obs_scale(env, params, n)
        t_cell = time.time()

        # The markup axis is declared in CHF/MWh and stored in action units, so
        # `C_REP` is an input to where every level sits.  Recomputed from the
        # parameters actually built rather than trusted, and printed, so the
        # product's `markup_axis.price` is a read-back and not a restatement of
        # the constant: a scenario whose replacement cost moved would otherwise
        # relabel the whole axis silently.
        b = params.battery
        eta_rt = np.asarray(b.eta_charge) * np.asarray(b.eta_discharge)
        ep = np.unique(np.asarray(params.energy_price))
        got = np.asarray(params.cycle_cost) * (1.0 + 1.0 / eta_rt) \
            + float(ep[0]) / eta_rt
        if ep.size != 1 or abs(float(got.max()) - C_REP) > 1e-4 \
                or abs(float(got.min()) - C_REP) > 1e-4:
            raise SystemExit(
                f"{name}: this file places the markup axis at C_REP={C_REP} "
                f"CHF/MWh, and the scenario gives c_rep in "
                f"[{float(got.min())!r}, {float(got.max())!r}] over "
                f"{ep.size} distinct energy prices; every price on the axis "
                f"would be mislabelled")
        print(f"  c_rep {float(got.mean()):.6f} CHF/MWh (read back from "
              f"params); markup axis prices "
              f"{[round(C_REP * (1.0 + float(np.logaddexp(0.0, a))), 2) for a in alpha_pi]}",
              flush=True)

        checks = []
        if args.selfcheck:
            for label, act in PUBLISHED:
                shared = evaluate(env, params, scale, n,
                                  constant_policy(env[3]["obs_dim"], list(act)),
                                  jax.random.PRNGKey(REPORT_KEY), 0, LOG_STD,
                                  eval_starts=test)
                mine = per_episode_returns(env, params, scale, n, act, test)
                got = float(mine["per_episode"].mean())
                if abs(got - shared["ret"]) > 1e-9 * max(1.0, abs(shared["ret"])):
                    raise SystemExit(
                        f"{name} {label}: this file's reduction gives {got!r} "
                        f"and `evaluate` gives {shared['ret']!r}; the two paths "
                        f"are not scoring the same thing and no cell of this "
                        f"grid means what it says")
                checks.append(dict(cell=name, arm=label, action=list(act),
                                   ret=shared["ret"], ret_this_file=got,
                                   price_given_cleared=shared["price_given_cleared"],
                                   volume_given_cleared=shared["volume_given_cleared"],
                                   clearing_fraction=shared["clearing_fraction"]))
                print(f"  selfcheck {name} {label:<13} ret {shared['ret']:+10.4f} "
                      f"(both paths)", flush=True)

        shape = (len(alpha_pi), len(alpha_q), len(alpha_ch))
        per_ep = np.zeros(shape + (len(test),))
        rows = []
        done = np.zeros(shape, bool)
        for src in list(args.resume_from) + [str(args.out)]:
            npz = Path(src).with_suffix(f".{name}.npz")
            if not npz.exists():
                continue
            old = np.load(npz, allow_pickle=True)
            if old["test_starts"].tolist() != list(np.asarray(test)):
                raise SystemExit(
                    f"{npz} was scored on different test days; refusing to "
                    f"reuse its cells rather than mixing two evaluations")
            if old["alpha_q"].tolist() != list(alpha_q) \
                    or old["alpha_ch"].tolist() != list(alpha_ch):
                raise SystemExit(f"{npz} swept different q/ch axes")
            o_pi = old["alpha_pi"].tolist()
            by_cell = {}
            try:
                by_cell = {(c["alpha_pi"], c["alpha_q"], c["alpha_ch"]): c
                           for c in json.loads(Path(src).read_text())
                           ["results"][name]["cells"]}
            except (OSError, KeyError, ValueError):
                pass
            took = 0
            for a, x in enumerate(o_pi):
                if x not in alpha_pi:
                    continue
                i2 = alpha_pi.index(x)
                for b in range(len(alpha_q)):
                    for c in range(len(alpha_ch)):
                        if not old["done"][a, b, c] or done[i2, b, c]:
                            continue
                        per_ep[i2, b, c] = old["ret_per_episode"][a, b, c]
                        done[i2, b, c] = True
                        key = (x, alpha_q[b], alpha_ch[c])
                        if key in by_cell:
                            rows.append(by_cell[key])
                        took += 1
            if took:
                print(f"  reused {took} cells from {npz}", flush=True)
        if done.any():
            print(f"  resuming {name}: {int(done.sum())} of {done.size} cells "
                  f"already on disk", flush=True)

        def write_products(complete, _cell=name, _res=results, _rows=rows,
                           _per=per_ep, _done=done, _t=t_cell):
            """Dump after every cell; the winner only when the sweep is complete.

            A winner taken over a partly filled grid is not the winner, and a
            product that carried one would be read as if it were.
            """
            surface = np.where(_done, _per.mean(3), -np.inf)
            entry = dict(
                placement=place, capacity=cap, n_agent=n,
                eval_episodes=len(test), selfcheck=checks, cells=_rows,
                grid_levels=dict(alpha_pi=list(alpha_pi), alpha_q=list(alpha_q),
                                 alpha_ch=list(alpha_ch)),
                is_default_axes=dict(alpha_pi=bool(alpha_pi == ALPHA_PI),
                                     alpha_q=bool(alpha_q == ALPHA_Q),
                                     alpha_ch=bool(alpha_ch == ALPHA_CH)),
                markup_axis=dict(
                    c_rep=C_REP, voll=VOLL,
                    price=[C_REP * (1.0 + float(np.logaddexp(0.0, a)))
                           for a in alpha_pi],
                    is_alpha_pi_high=bool(alpha_pi == ALPHA_PI_HIGH),
                    is_alpha_pi_top=bool(alpha_pi == ALPHA_PI_TOP)),
                complete=bool(complete), cells_done=int(_done.sum()),
                cells_total=int(_done.size), seconds=time.time() - _t)
            if complete:
                wi, wj, wk = np.unravel_index(int(np.argmax(surface)),
                                              surface.shape)
                win = _per[wi, wj, wk]
                against = []
                for a, api2 in enumerate(alpha_pi):
                    for b, aq2 in enumerate(alpha_q):
                        for c, ach2 in enumerate(alpha_ch):
                            if (a, b, c) == (wi, wj, wk):
                                continue
                            diff = win - _per[a, b, c]
                            against.append(dict(
                                alpha_pi=api2, alpha_q=aq2, alpha_ch=ach2,
                                paired_difference=float(diff.mean()),
                                episodes_positive=int((diff > 0).sum()),
                                episodes_zero=int((diff == 0).sum()),
                                episodes=int(diff.size)))
                entry["winner"] = dict(
                    alpha_pi=float(alpha_pi[wi]), alpha_q=float(alpha_q[wj]),
                    alpha_ch=float(alpha_ch[wk]), ret=float(surface[wi, wj, wk]),
                    at_grid_edge=dict(
                        alpha_pi=bool(wi in (0, len(alpha_pi) - 1)),
                        alpha_q=bool(wj in (0, len(alpha_q) - 1)),
                        alpha_ch=bool(wk in (0, len(alpha_ch) - 1))))
                entry["against_other_cells"] = against
            _res[_cell] = entry
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(dict(
                grid=dict(alpha_pi=list(alpha_pi), alpha_q=list(alpha_q),
                          alpha_ch=list(alpha_ch), log_std=LOG_STD,
                          report_key=f"PRNGKey({REPORT_KEY})",
                          reduction=("mean over participants of the per-episode "
                                     "sum over the 24 periods, CHF per "
                                     "participant per episode")),
                results=_res), indent=1, default=float))
            np.savez(args.out.with_suffix(f".{_cell}.npz"),
                     alpha_pi=np.asarray(alpha_pi), alpha_q=np.asarray(alpha_q),
                     alpha_ch=np.asarray(alpha_ch), ret_per_episode=_per,
                     done=_done, test_starts=np.asarray(test),
                     complete=np.asarray(bool(complete)),
                     note=json.dumps({
                         "axes": "(alpha_pi, alpha_q, alpha_ch, episode)",
                         "units": "CHF per participant per episode, already "
                                  "averaged over participants"}))

        for i, api in enumerate(alpha_pi):
            for j, aq in enumerate(alpha_q):
                for k, ach in enumerate(alpha_ch):
                    if done[i, j, k]:
                        continue
                    t = time.time()
                    r = per_episode_returns(env, params, scale, n,
                                            (api, aq, ach), test)
                    # mean over participants, leaving one number per episode
                    per_ep[i, j, k] = r["per_episode"].mean(1)
                    rows.append(dict(alpha_pi=api, alpha_q=aq, alpha_ch=ach,
                                     ret=float(r["per_episode"].mean()),
                                     clearing_fraction=r["clearing_fraction"],
                                     price_given_cleared=r["price_given_cleared"],
                                     volume_given_cleared=r["volume_given_cleared"],
                                     mu_max=r["mu_max"],
                                     sweep_converged=r["sweep_converged"],
                                     seconds=time.time() - t))
                    print(f"  {name} pi {api:+7.1f} q {aq:+7.1f} ch {ach:+7.1f}  "
                          f"ret {rows[-1]['ret']:+10.4f}  "
                          f"clr {r['clearing_fraction']:.4f}  "
                          f"price|clr {r['price_given_cleared']:9.2f}  "
                          f"{rows[-1]['seconds']:.1f}s", flush=True)

                    done[i, j, k] = True
                    write_products(False)

        write_products(True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
