"""Where the gain from withholding half the deliverable quantity comes from.

A recorded measurement has no mechanism yet: on market 05's best fixed
action, offering only half the deliverable quantity returns 8.6% to 16.4% more
than offering all of it, in all four configurations, and the open question
asks for the three intermediate quantities -- the clearing price, the traded
volume and the published requirement -- to be taken back level by level so the
gain can be attributed to one of them.  This file takes them.

**Nothing here re-implements the rollout or the scenario.**  It calls
`concentration_baseline.make_rollout` exactly as `evaluate` and
`bestfixed_05_grid.per_episode_returns` call it -- same vmap signature, same
pinned test starts, same terminal settlement inside it -- and only keeps the
period axis that those two average away.  `--selfcheck` requires the mean of
what it keeps to equal `evaluate`'s `ret` at every level, so a level of this
sweep and a cell of that grid are the same object read at two granularities.

**The axis is the offered fraction, in the units the mechanism uses.**  The
action's column 1 is `sigmoid(alpha_q)`, so a fraction `f` is submitted as
`logit(f)`; `f = 1` is the saturation point +128, which is exact in float32 and
is the value the three published constant strategies already sit at.  The other
two coordinates are pinned at the best fixed action's own values, so a
difference between two levels is a difference in the offered fraction and in
nothing else.

**The decomposition, and what it can and cannot separate.**  Per episode the
community's revenue from the auction is `sum_t p_t q_t dt`, so between two
levels

    d(revenue) = Vbar d(P) + Pbar d(V)

with `V` the episode's traded energy, `P` its volume-weighted price, and each
bar the mean of the two levels -- the symmetric form, so the cross term is
split between the two channels rather than left as a third one that depends on
which level is called the base.  The part of `d(return)` that this does not
reach is reported as a residual and named: it is degradation, the planned
charge and the terminal settlement of the stock left in the battery, which are
in the reward and not in the auction's revenue.  A residual that carries the
gain is a result, not a failure of the decomposition.

    JAX_PLATFORMS=cpu PYTHONPATH=.:tools/flex_experiment python -m \\
        withholding_channels_05 --out withholding.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from concentration_baseline import (CELLS, DELTA, EPISODE_LEN, day_split,
                                    evaluate, make_rollout, obs_scale, scenario,
                                    series_day_of_month)
from monitor_probe import LOG_STD, constant_policy

#: The two coordinates the best fixed action pins, read off the fixed-action
#: baseline results for market 05: markup +20, which
#: prices at 21 x the replacement cost, and planned charge at the saturation
#: point.  They are held here so the only thing that moves is column 1.
ALPHA_PI_PINNED = 20.0
ALPHA_CH_PINNED = 128.0

#: The offered fractions this file sweeps, as fractions of the deliverable
#: quantity.  1.0 is the saturation point and is submitted as +128 rather than
#: as an infinite logit.
FRACTIONS = (0.25, 0.5, 0.75, 1.0)
SATURATION = 128.0

REPORT_KEY = 999


def alpha_q_for(fraction):
    """`alpha_q` submitting `fraction` of the deliverable quantity."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must lie in (0, 1], got {fraction}")
    if fraction == 1.0:
        return SATURATION
    return math.log(fraction / (1.0 - fraction))


def per_period(env, params, scale, n, action, starts):
    """Keep the period axis `evaluate` averages away.

    Returns arrays shaped ``(episode, period)`` for the three intermediates and
    ``(episode,)`` for the return, all read off the same rollout call the
    published evaluation makes.
    """
    roll = make_rollout(env, params, scale, n, LOG_STD)
    starts = jnp.asarray(starts, jnp.int32)
    out = jax.vmap(roll, in_axes=(None, 0, None, 0))(
        constant_policy(env[3]["obs_dim"], list(action)),
        jax.random.split(jax.random.PRNGKey(REPORT_KEY), starts.shape[0]),
        0, starts)
    reward, shed = np.asarray(out[4]), np.asarray(out[5])
    vol, price = np.asarray(out[6]), np.asarray(out[7])
    req_th, req_v = np.asarray(out[9]), np.asarray(out[11])
    mu, converged = np.asarray(out[12]), np.asarray(out[13])
    # sum over the period axis, then mean over the participant axis -- the
    # order `evaluate` uses and the order the unit "CHF per participant per
    # episode" names.  Doing it the other way round multiplies the result by
    # `n_agent / EPISODE_LEN`, which is exactly 1 on the two 24-aggregator
    # configurations and 34/24 on the other two; `--selfcheck` caught it on the
    # third configuration after the first two had passed.
    return dict(ret_per_episode=reward.sum(1).mean(1), vol=vol, price=price,
                req_th=req_th, req_v=req_v, shed=shed,
                mu_max=float(mu.max()), converged=bool(converged.all()))


def reduce_level(raw):
    """Episode-level `V`, `P` and revenue from the period arrays.

    `V` is energy, so the power the clearing reports is multiplied by the period
    length once and here rather than being carried as power and called energy
    later.  `P` is volume weighted, which is the only weighting under which
    `P V` is the revenue; a plain mean over periods would make the identity the
    decomposition rests on false in exactly the periods that clear nothing.
    """
    energy = raw["vol"] * DELTA                       # (episode, period)
    revenue = (raw["price"] * energy).sum(1)                 # (episode,)
    v = energy.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(v > 0.0, revenue / np.where(v > 0.0, v, 1.0), 0.0)
    cleared = raw["vol"] > 1e-9
    return dict(
        ret=raw["ret_per_episode"], revenue=revenue, energy=v, price_vw=p,
        cleared_fraction=float(cleared.mean()),
        price_given_cleared=(float(raw["price"][cleared].mean())
                             if cleared.any() else 0.0),
        volume_given_cleared=(float(raw["vol"][cleared].mean())
                              if cleared.any() else 0.0),
        req_th_mean=float(raw["req_th"].mean()),
        req_v_mean=float(raw["req_v"].mean()),
        req_th_given_cleared=(float(raw["req_th"][cleared].mean())
                              if cleared.any() else 0.0),
        shed_total=float(raw["shed"].sum(1).mean()),
        mu_max=raw["mu_max"], converged=raw["converged"])


def decompose(a, b, n_agent):
    """Split `b - a`'s revenue change into a price and a volume channel.

    Both levels are scored on the same episodes, so the difference is taken
    episode by episode and then averaged; taking it between two averages would
    be the same number here and a different one the moment the episode sets
    differ, and the paired form is what the matrix's section 5 requires for this
    market.  Returned per participant per episode, the unit every 05 return is
    reported in, so the channels add up to the reported gap and not to `n`
    times it.
    """
    d_p = b["price_vw"] - a["price_vw"]
    d_v = b["energy"] - a["energy"]
    p_bar = 0.5 * (a["price_vw"] + b["price_vw"])
    v_bar = 0.5 * (a["energy"] + b["energy"])
    price_channel = (v_bar * d_p) / n_agent
    volume_channel = (p_bar * d_v) / n_agent
    d_ret = b["ret"] - a["ret"]
    d_rev = (b["revenue"] - a["revenue"]) / n_agent
    return dict(
        d_return=float(d_ret.mean()),
        d_revenue=float(d_rev.mean()),
        price_channel=float(price_channel.mean()),
        volume_channel=float(volume_channel.mean()),
        # what the auction's revenue does not reach: degradation, the planned
        # charge and the terminal settlement of the stock left in the battery
        residual_channel=float((d_ret - d_rev).mean()),
        # a closure check on the identity, not a tolerance on the measurement
        revenue_closure=float(
            (d_rev - price_channel - volume_channel).mean()),
        episodes_return_positive=int((d_ret > 0).sum()),
        episodes=int(d_ret.size))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="",
                    help="comma separated indices into CELLS; empty means all")
    ap.add_argument("--fractions", type=float, nargs="+", default=list(FRACTIONS))
    ap.add_argument("--alpha-ch", type=float, default=ALPHA_CH_PINNED,
                    help="the planned charge to pin the sweep at. The default "
                         "is the winner's saturation point. Moving it is the "
                         "controlled contrast the withholding gain needs: with "
                         "`plan = sigmoid(alpha_ch) x headroom` at zero there "
                         "is no forgone-charging term in `q_phys`, so if "
                         "withholding stops paying there, the gain is the "
                         "forgone charge and not the discharge")
    ap.add_argument("--alpha-pi", type=float, default=ALPHA_PI_PINNED,
                    help="the markup to pin the sweep at. The default is the "
                         "150-cell grid's winner (+20); the extended markup "
                         "axis moved that winner to +65.0518, and the gain is "
                         "measured 'on this best fixed action', so which markup the "
                         "withholding gain was measured at has to be a "
                         "parameter rather than a constant")
    ap.add_argument("--selfcheck", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _, dom = series_day_of_month()
    _, _val, test = day_split(dom, EPISODE_LEN)
    cells = (CELLS if not args.cells
             else tuple(CELLS[int(i)] for i in args.cells.split(",")))
    fractions = tuple(args.fractions)
    alpha_pi = float(args.alpha_pi)
    alpha_ch = float(args.alpha_ch)
    print(f"offered fractions {list(fractions)} -> alpha_q "
          f"{[round(alpha_q_for(f), 6) for f in fractions]}; pinned "
          f"alpha_pi={alpha_pi} alpha_ch={alpha_ch}; "
          f"{len(test)} test days", flush=True)

    results = {}
    for place, cap in cells:
        name = f"{place}p_{cap}c"
        env, params, n = scenario(place, cap, split="train")
        scale = obs_scale(env, params, n)
        levels, t_cell = {}, time.time()
        for f in fractions:
            t = time.time()
            action = (alpha_pi, alpha_q_for(f), alpha_ch)
            raw = per_period(env, params, scale, n, action, test)
            lv = reduce_level(raw)
            if args.selfcheck:
                shared = evaluate(env, params, scale, n,
                                  constant_policy(env[3]["obs_dim"],
                                                  list(action)),
                                  jax.random.PRNGKey(REPORT_KEY), 0, LOG_STD,
                                  eval_starts=test)
                got = float(lv["ret"].mean())
                # The two paths sum the same float32 numbers in different
                # orders -- `evaluate` reduces the participant axis inside the
                # rollout and this file reduces it outside -- so the tolerance
                # is derived from the float32 spacing at the value being
                # compared and not pinned to any measured gap.  Eight
                # units in the last place is four orders of magnitude below the
                # smallest difference between two levels of this sweep, so a
                # real disagreement still fails.  The gap actually observed is
                # recorded beside the value rather than discarded: measured at
                # exactly one unit in the last place on the first level that
                # exercised this branch (96.0408935546875 against
                # 96.04088592529297).
                tol = 8.0 * float(np.spacing(
                    np.float32(max(abs(shared["ret"]), 1.0))))
                if abs(got - shared["ret"]) > tol:
                    raise SystemExit(
                        f"{name} f={f}: this file's reduction gives {got!r} and "
                        f"`evaluate` gives {shared['ret']!r}, a gap of "
                        f"{abs(got - shared['ret']):.3e} against a tolerance of "
                        f"{tol:.3e}; the period axis this file keeps is not the "
                        f"one that was averaged")
                lv["ret_evaluate"] = shared["ret"]
                lv["ret_gap_vs_evaluate"] = float(got - shared["ret"])
                lv["ret_gap_tolerance"] = tol
            levels[f] = lv
            print(f"  {name} f={f:<5} ret {float(lv['ret'].mean()):+10.4f}  "
                  f"E {float(lv['energy'].mean()):7.4f} MWh  "
                  f"P {float((lv['revenue'].sum() / max(lv['energy'].sum(), 1e-12))):9.2f} "
                  f"CHF/MWh  clr {lv['cleared_fraction']:.4f}  "
                  f"req_th {lv['req_th_mean']:.4f}  "
                  f"{time.time() - t:.1f}s", flush=True)

        base = max(fractions)
        entry = dict(
            placement=place, capacity=cap, n_agent=n, eval_episodes=len(test),
            pinned=dict(alpha_pi=alpha_pi, alpha_ch=alpha_ch,
                        is_default_alpha_pi=bool(alpha_pi == ALPHA_PI_PINNED),
                        is_default_alpha_ch=bool(alpha_ch == ALPHA_CH_PINNED)),
            fractions=list(fractions), alpha_q=[alpha_q_for(f) for f in fractions],
            levels={str(f): {k: (float(np.mean(v)) if isinstance(v, np.ndarray)
                                 else v)
                             for k, v in levels[f].items()}
                    for f in fractions},
            # every level against offering everything, which is the comparison
            # the withholding-gain measurement is about
            against_full={
                str(f): decompose(levels[base], levels[f], n)
                for f in fractions if f != base},
            seconds=time.time() - t_cell)
        results[name] = entry
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(dict(
            axis=dict(fractions=list(fractions), pinned_alpha_pi=alpha_pi,
                      pinned_alpha_ch=alpha_ch, log_std=LOG_STD,
                      report_key=f"PRNGKey({REPORT_KEY})",
                      period_hours=DELTA,
                      reduction=("mean over participants of the per-episode sum "
                                 "over the 24 periods, CHF per participant per "
                                 "episode; the channels are in the same unit")),
            results=results), indent=1, default=float))
        np.savez(args.out.with_suffix(f".{name}.npz"),
                 fractions=np.asarray(fractions),
                 ret_per_episode=np.stack([levels[f]["ret"] for f in fractions]),
                 energy=np.stack([levels[f]["energy"] for f in fractions]),
                 revenue=np.stack([levels[f]["revenue"] for f in fractions]),
                 price_vw=np.stack([levels[f]["price_vw"] for f in fractions]),
                 test_starts=np.asarray(test),
                 note=json.dumps({
                     "axes": "(offered fraction, episode)",
                     "units": "ret CHF per participant per episode; energy MWh "
                              "per episode community total; revenue CHF per "
                              "episode community total; price_vw CHF/MWh"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
