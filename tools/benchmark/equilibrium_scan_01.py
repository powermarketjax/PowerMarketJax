"""Unilateral deviation from the TRAINED profile, not from truthful bidding.

The trained policy settles near a markup of 1.59 while a single unit deviating
from a *truthful* field gains monotonically up to the action-space ceiling of
2.0.  Those two facts are only in tension if the relevant comparison is against
truthfulness -- and it is not.  What decides whether 1.59 is an equilibrium is
whether one unit can profit by moving while **the other 65 stay where the
policy put them**.  So the reference point here is "this unit also stays at the
trained action", and every number reported is a change from that.

Using the truthful field as the reference would answer a different question
(how much is there to gain over honest bidding) and would answer it with a
number that looks like it addresses this one.
"""
import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import (check_split_against_report, open_day, runtime_stamp,
                        split_days, subset_position)

CAP_SCALE = 0.60
RAMP_SCALE = 1.00
MARKUP_MAX = 2.0
VOLL_IN_EFFECT = 10_000.0
T = 24
K = 1
#: 1.60 is where the trained policy sits, so the grid is denser there: the
#: question is whether the optimum is AT that point, and a coarse grid could
#: straddle it and report a neighbour.
#: A unit counts as strongly deviating if alpha=2.0 beats the trained action
#: by more than this share of its reference profit.  Fixed before the sweep.
STRONG_DEVIATION = 0.10

#: The commitment fixture, named once and recorded in every product this
#: writes.  A module constant rather than a literal at the load site so
#: that the product's `fixture` field and the file actually opened cannot
#: drift apart -- which is the whole point of recording it.
FIXTURE_NAME = "day_ahead_commitment_29gb_T24_relax.npz"

GRID = (1.00, 1.15, 1.30, 1.45, 1.55, 1.60, 1.65, 1.75, 2.00)


def summarise(rows, ref_tot, units):
    """The three numbers the sweep exists for, plus the count that changes how
    number one was counted.

    That fourth line is not decoration.  Units whose reference profit is near
    zero are classified by absolute gain rather than by ratio, because a ratio
    against a vanishing denominator is not a magnitude -- a unit earning 3 that
    gains 1 would otherwise count as a 33% deviator and inflate the total that
    a downstream decision rests on.  The rule is sound and it is also invisible:
    somebody reading the three numbers has no way to know a different rule was
    applied to part of the population.  A correct treatment that leaves its
    trace only in the product, and not where the number is consumed, does not
    exist as far as the reader is concerned -- so it is printed here.
    """
    at_top = {r["unit"]: r for r in rows if abs(r["alpha"] - 2.00) < 1e-9}
    strong, weak, by_absolute = [], [], []
    for u_, r in at_top.items():
        ref_u = ref_tot[u_]
        tiny = abs(ref_u) <= 1.0
        if tiny:
            by_absolute.append(u_)
        big = r["delta"] > 0.0 if tiny else r["delta"] > STRONG_DEVIATION * abs(ref_u)
        (strong if big else weak).append((u_, r["delta"]))
    gain = sum(d for _u, d in strong)
    loss = sum(d for _u, d in weak)
    print(f"\n(1) units with a strong incentive to deviate: {len(strong)} of "
          f"{len(at_top)}  (threshold {STRONG_DEVIATION:.0%} of own reference)")
    print(f"    {sorted(u_ for u_, _d in strong)}")
    print(f"(1b) of those {len(at_top)}, {len(by_absolute)} were classified by "
          f"ABSOLUTE gain, not by ratio, because their reference profit is at or "
          f"below 1.0 in magnitude: {sorted(by_absolute)}")
    print(f"(2) at alpha=2.0, their total gain {gain:+.6e} vs everyone else's "
          f"total {loss:+.6e}   net {gain + loss:+.6e}")
    print(f"    gain exceeds the others' losses: {gain > -loss}")
    exact_zero = sum(1 for r in rows if r["delta"] == 0.0)
    print(f"    rows whose delta is EXACTLY zero: {exact_zero} of {len(rows)} "
          f"(a unit that cannot move its own profit by repricing)")
    # Units whose delta is zero at EVERY grid point are excluded from the
    # optimum distribution rather than counted at their argmax.  `max` returns
    # the first maximum, so an all-zero row reports the first grid point as
    # though the unit preferred it: on the earlier sweep that turned 42
    # indifferent units into "42 units want alpha=1.00" and inflated a real
    # count of 6 to a reported 48.  A degenerate input landing on a
    # meaningful-looking output is worse than one landing on NaN, because
    # nothing downstream can tell the difference.
    flat, best = [], {}
    for u_ in units:
        mine = [r for r in rows if r["unit"] == u_]
        if all(r["delta"] == 0.0 for r in mine):
            flat.append(u_)
        else:
            best[u_] = max(mine, key=lambda r: r["delta"])
    from collections import Counter
    dist = Counter(round(b["alpha"], 2) for b in best.values())
    print(f"(3) {len(flat)} of {len(units)} units cannot change their own profit "
          f"at ANY alpha -- EXCLUDED from the distribution below, not counted at "
          f"their argmax")
    print(f"    optimum alpha over the {len(best)} units that do respond: "
          + "  ".join(f"{a}:{n}" for a, n in sorted(dist.items())))
    near = sum(1 for b in best.values() if abs(b["alpha"] - 1.60) < 0.06)
    print(f"    of those {len(best)}, {near} have their optimum at 1.55-1.65")


def summarise_saved(path):
    """Regenerate the summary from a product, for sweeps that predate it."""
    z = np.load(path, allow_pickle=True)
    ref = dict(zip(z["reference_units"].tolist(), z["reference_profit"].tolist()))
    rows = [dict(unit=int(u), alpha=float(a), delta=float(d))
            for u, a, d in zip(z["unit"], z["alpha"], z["delta"])]
    summarise(rows, ref, sorted(ref))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="the trained archive to sweep (required unless "
                         "--summarise-saved)")
    ap.add_argument("--layout", choices=("auto", "shared", "per-agent"),
                    default="auto",
                    help="parameter layout of the checkpoint. `auto` reads it "
                         "from the file's `meta.per_agent_params` and refuses if "
                         "the key is absent; the two explicit values exist for "
                         "archives written before that stamp did (2026-08-25) "
                         "-- e.g. r1_01_wd052_curve.params.npz, which "
                         "the published deviation series was measured on. "
                         "Naming the layout is not the same as guessing it: the "
                         "leaf-shape check below still has to agree")
    ap.add_argument("--units", default="35,37,34")
    ap.add_argument("--out", default=None,
                    help="the product to write (required unless "
                         "--summarise-saved)")
    ap.add_argument("--collective-alpha", type=float, default=None,
                    help="instead of sweeping one unit at a time, move EVERY "
                         "unit to this alpha and report the per-agent profit "
                         "change from the trained profile. A unilateral "
                         "deviation and a collective move are different "
                         "questions on the same action space, and a net figure "
                         "from one does not explain a gap produced by the "
                         "other.")
    ap.add_argument("--summarise-saved", default=None,
                    help="skip the sweep; re-print the summary from an existing "
                         "product. For sweeps that finished before a summary "
                         "line existed -- the numbers come from the file, not "
                         "from a re-run, so they cannot drift from it.")
    args = ap.parse_args()
    if args.summarise_saved:
        summarise_saved(args.summarise_saved)
        return
    #: Required only for a sweep.  They were `required=True`, which made
    #: `--summarise-saved` -- the mode that reads an existing product and
    #: needs neither -- unusable on its own (argparse exited 2 before it ran).
    if not args.checkpoint or not args.out:
        ap.error("--checkpoint and --out are required unless --summarise-saved "
                 "is given")

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")

    from powermarketjax.case import load_case
    from powermarketjax.envs.day_ahead import (demand_from_meta, load_commitment,
                                               make_env)
    from powermarketjax.envs.day_ahead.commitment import FIXTURE_DIR
    from powermarketjax.learning.adapters import unpack_env
    from powermarketjax.learning.ippo import make_greedy_action, make_ippo
    from powermarketjax.learning.policy import bounds_for
    from hyperparams import SHARED
    import dataclasses

    ck = np.load(args.checkpoint, allow_pickle=True)
    cfg_saved = json.loads(str(ck["config"]))
    cfg = dataclasses.replace(
        SHARED, **{k: (tuple(v) if isinstance(v, list) else v)
                   for k, v in cfg_saved.items() if hasattr(SHARED, k)})
    #: **`per_agent_params` is NOT in `config`, and cannot be.**  `config`
    #: mirrors `IPPOConfig`, while the parameter layout is a keyword of
    #: `make_ippo` rather than a field of that dataclass -- deliberately, so the
    #: shape of `hyperparams` does not move under every archive already written.
    #: So it is read from `meta`, which is where `run_rl_01.write_params` stamps
    #: it (2026-08-28).
    #
    # Before this the rebuild always used the shared layout, and on an
    # independent-parameter archive the shape check below refused: measured,
    # `leaf 0: checkpoint (66, 64) vs rebuilt (64,)`, non-zero exit, named leaf,
    # no silent wrong number.  So this is a missing capability, not a bug --
    # which is why the refusal is kept exactly as it is below rather than
    # relaxed.
    meta_saved = json.loads(str(ck["meta"])) if "meta" in ck.files else {}
    declared = meta_saved.get("per_agent_params")
    if args.layout != "auto":
        told = args.layout == "per-agent"
        if declared is not None and bool(declared) != told:
            raise SystemExit(
                f"--layout {args.layout} contradicts the file's own "
                f"`meta.per_agent_params` = {bool(declared)}. The file wins; "
                f"drop the flag")
        declared = told
        print(f"--layout {args.layout} given explicitly"
              + ("" if "meta" not in ck.files or meta_saved.get("per_agent_params")
                 is not None else "; the file carries no stamp of its own"),
              flush=True)
    if declared is None:
        # Archives written before the stamp existed.  Refuse rather than assume
        # shared: the two layouts have the same leaf COUNT and differ only in a
        # leading axis, so guessing wrong reconstructs a different policy of the
        # right shape on some architectures.  The shape check below catches the
        # 66-agent case, and that is a property of this case rather than a
        # guarantee.
        raise SystemExit(
            f"{args.checkpoint} carries no `per_agent_params` in its `meta`, so "
            f"which parameter layout it holds cannot be read off the file. "
            f"Re-stamp it, or point this at an archive written after "
            f"2026-08-25 (`run_rl_01.write_params` stamps every file it writes)")
    per_agent_params = bool(declared)
    print(f"checkpoint cfg: n_envs={cfg.n_envs} horizon={cfg.horizon}  "
          f"per_agent_params={per_agent_params}", flush=True)

    fx = load_commitment(
        path=FIXTURE_DIR / FIXTURE_NAME,
        n_periods=T)
    meta = fx["meta"]
    dates_all = [str(d) for d in meta["dates"]]
    ok, _sel = check_split_against_report(dates_all, meta["case"])
    if not ok:
        raise SystemExit("the split rule no longer reproduces the report's days")
    ev_days, _tr = split_days(len(dates_all), meta["case"])

    # The environment is built on the fixture RESTRICTED to the evaluation days,
    # exactly as `run_rl_01.py` builds the one it evaluates the policy in.  This
    # is not a detail: inside a restricted fixture the cursor runs 0..11 over
    # those days, and `open_day` selects by cursor value.  The first version of
    # this script built the FULL fixture and passed the same 0..11, so it opened
    # calendar days 0-11 -- of which only 3 are evaluation days and 9 are days
    # the policy trained on.  Every number it produced was on the wrong days,
    # and nothing in the output looked wrong, because the labels came from the
    # evaluation-day list either way.
    env, spec = make_env(load_case(meta["case"]), subset_position(fx, ev_days),
                         demand_from_meta(meta),
                         n_segments=K, kind="markup", markup_max=MARKUP_MAX,
                         cap_scale=CAP_SCALE, ramp_scale=RAMP_SCALE)
    four = unpack_env((env, spec))
    ns = four[3]
    params_env = env.make_params(episode_len=1)
    bounds = bounds_for(ns)

    # The checkpoint stores `treedef` as a repr string, which cannot be parsed
    # back, so the structure is rebuilt by running `init` and the saved leaves
    # are substituted in flatten order.  That order is a property of the tree
    # structure, so it agrees as long as the structure does -- which the leaf
    # count and per-leaf shapes below check rather than assume.
    init, _iterate = make_ippo(four, bounds, cfg,
                               jnp.asarray(ck["obs_mean"]),
                               jnp.asarray(ck["obs_std"]),
                               per_agent_params=per_agent_params)
    fresh, _tx, _os, _es, _eo = init(jax.random.PRNGKey(0), params_env)
    leaves, treedef = jax.tree_util.tree_flatten(fresh)
    saved = [ck[f"p{i}"] for i in range(len(ck.files)) if f"p{i}" in ck.files]
    if len(saved) != len(leaves):
        raise SystemExit(f"checkpoint has {len(saved)} leaves, the rebuilt tree "
                         f"has {len(leaves)}; structures disagree")
    for i, (a, b) in enumerate(zip(saved, leaves)):
        if tuple(a.shape) != tuple(b.shape):
            raise SystemExit(f"leaf {i}: checkpoint {a.shape} vs rebuilt {b.shape}")
    params = jax.tree_util.tree_unflatten(
        treedef, [jnp.asarray(a) for a in saved])

    greedy = jax.jit(make_greedy_action(ns, bounds, cfg,
                                        jnp.asarray(ck["obs_mean"]),
                                        jnp.asarray(ck["obs_std"]),
                                        per_agent_params=per_agent_params))
    step = jax.jit(env.step)
    get_obs = env.get_obs
    day_of = lambda st: int(st.cursor)
    units = [int(u) for u in args.units.split(",")]

    # the trained profile, one 66-vector per evaluation day
    profile, keys, states = {}, {}, {}
    for pos, day in enumerate(ev_days):
        # `pos`, not `day`: inside the restricted fixture the cursor counts the
        # twelve selected days, and 0..11 is what those cursors are.
        k, st = open_day(env.reset, params_env, pos, day_of)
        profile[day] = np.asarray(greedy(params, get_obs(st, params_env)),
                                  np.float64)
        keys[day], states[day] = k, st
    prof = np.stack([profile[d] for d in ev_days])
    print(f"trained profile: mean {prof.mean():.4f}  min {prof.min():.4f}  "
          f"max {prof.max():.4f}  spread across units {prof.std(axis=1).mean():.4f}",
          flush=True)

    def run(day, action):
        _o, nxt, reward, _c, _d, info = step(keys[day], states[day],
                                             jnp.asarray(action), params_env)
        # `award` is not in `info`; it reaches the next state as `award_prev`,
        # the same route `lmp_prev` takes.  Reading it here rather than adding
        # an `info` key keeps this a read-only diagnostic.
        return (np.asarray(reward, np.float64), info,
                np.asarray(nxt.award_prev, np.float64))

    # reference: everybody at the trained action
    ref, ref_award, ref_commit = {}, {}, {}
    for day in ev_days:
        r, i, aw = run(day, profile[day])
        ref[day] = r
        ref_award[day] = aw.sum(axis=1)                 # MWh per unit over T
        ref_commit[day] = np.asarray(i["commitment"], np.float64).sum(axis=1)
    ref_tot = {u: float(sum(ref[d][u] for d in ev_days)) for u in units}
    print("reference (all 66 at the trained action), profit over 12 days:")
    for u in units:
        print(f"  unit {u}: {ref_tot[u]:+.6e}", flush=True)

    if args.collective_alpha is not None:
        A = args.collective_alpha
        tot = np.zeros(len(profile[ev_days[0]]))
        for day in ev_days:
            act = np.full_like(profile[day], A)
            r, _i, _aw = run(day, act)
            tot += r
        base = np.zeros_like(tot)
        for day in ev_days:
            base += ref[day]
        delta = tot - base
        up = int((delta > 0).sum()); dn = int((delta < 0).sum())
        print(f"\ncollective move: every unit to alpha={A:.2f}, "
              f"against the trained profile")
        print(f"  agents better off {up}, worse off {dn}, unchanged "
              f"{len(delta) - up - dn}")
        print(f"  total gain of the winners {delta[delta > 0].sum():+.6e}")
        print(f"  total loss of the losers  {delta[delta < 0].sum():+.6e}")
        print(f"  NET across all agents     {delta.sum():+.6e}")
        print(f"  trained-profile total     {base.sum():+.6e}")
        print(f"  all-at-{A:.2f} total        {tot.sum():+.6e}")
        order = np.argsort(-delta)
        print("  five biggest winners: " + ", ".join(
            f"u{int(i)} {delta[i]:+.3e}" for i in order[:5]))
        print("  five biggest losers:  " + ", ".join(
            f"u{int(i)} {delta[i]:+.3e}" for i in order[-5:]))
        np.savez(args.out, agent=np.arange(len(delta)), delta=delta,
                 profit_at_alpha=tot, profit_trained=base,
                 trained_profile=prof,
                 meta=json.dumps(dict(
                     collective_alpha=A,
                     question=("every unit moved together, which is what the "
                               "constant arm does; NOT a unilateral deviation"),
                     checkpoint=str(args.checkpoint), eval_days=len(ev_days),
                     cap_scale=CAP_SCALE, ramp_scale=RAMP_SCALE,
                     market="01 day-ahead wholesale", fixture=FIXTURE_NAME,
                     **runtime_stamp())))
        print(f"\nwrote {args.out}")
        return

    rows = []
    for u in units:
        for a in GRID:
            tot = aw_tot = cm_tot = rev_tot = nl_tot = 0.0
            for day in ev_days:
                act = profile[day].copy()
                act[u] = a
                r, i, aw = run(day, act)
                tot += float(r[u])
                # the cliff diagnostic: a unit that prices itself out keeps
                # paying no-load and start-up while its award goes to zero, so
                # award and committed periods are what separate "sold less at a
                # better price" from "was pushed out of the schedule"
                aw_tot += float(aw[u].sum())
                cm_tot += float(np.asarray(i["commitment"], np.float64)[u].sum())
                rev_tot += float(np.asarray(i["revenue"], np.float64)[u])
                nl_tot += float(np.asarray(i["no_load_cost"], np.float64)[u])
            d = tot - ref_tot[u]
            rows.append(dict(unit=u, alpha=a, profit=tot, delta=d,
                             rel=d / abs(ref_tot[u]) if ref_tot[u] else float("nan"),
                             award_mwh=aw_tot, committed_periods=cm_tot,
                             revenue=rev_tot, no_load=nl_tot))
            print(f"  unit {u:3d}  alpha {a:.2f}  profit {tot:+.6e}  "
                  f"delta {d:+.4e}  ({100 * rows[-1]['rel']:+.2f}%)  "
                  f"award {aw_tot:.4e} MWh  committed {cm_tot:.0f}/{12 * T}",
                  flush=True)

    # The three numbers the sweep exists to produce.  Number 2 is the one that
    # decides the reading: if the units with a strong incentive gain more at the
    # ceiling than everybody else loses there, then "the constant arm earns more
    # in aggregate" and "the shared policy stops at 1.58" have one cause -- a
    # single shared knob cannot raise those units without raising the rest.
    summarise(rows, ref_tot, units)

    print("\nbest deviation per unit:")
    for u in units:
        mine = [r for r in rows if r["unit"] == u]
        best = max(mine, key=lambda r: r["delta"])
        at_ref = [r for r in mine if abs(r["alpha"] - 1.60) < 1e-9]
        print(f"  unit {u}: best alpha {best['alpha']:.2f} "
              f"delta {best['delta']:+.4e} ({100 * best['rel']:+.2f}%)"
              + (f" | at 1.60 delta {at_ref[0]['delta']:+.4e}" if at_ref else ""))

    np.savez(args.out,
             unit=np.array([r["unit"] for r in rows]),
             alpha=np.array([r["alpha"] for r in rows]),
             profit=np.array([r["profit"] for r in rows]),
             delta=np.array([r["delta"] for r in rows]),
             award_mwh=np.array([r["award_mwh"] for r in rows]),
             committed_periods=np.array([r["committed_periods"] for r in rows]),
             revenue=np.array([r["revenue"] for r in rows]),
             no_load=np.array([r["no_load"] for r in rows]),
             reference_award_mwh=np.stack([ref_award[d] for d in ev_days]),
             reference_committed=np.stack([ref_commit[d] for d in ev_days]),
             trained_profile=prof,
             reference_profit=np.array([ref_tot[u] for u in units]),
             reference_units=np.array(units),
             meta=json.dumps(dict(
                 reference="all 66 units at the trained mean-regime action; the "
                           "deviation measured is from the TRAINED profile, not "
                           "from truthful bidding -- those answer different "
                           "questions and only this one bears on equilibrium",
                 checkpoint=str(args.checkpoint), grid=list(GRID),
                 strong_deviation_threshold=STRONG_DEVIATION,
                 threshold_note=(
                     "a unit counts as having a strong incentive to deviate if "
                     "its profit at alpha=2.0 exceeds its profit at the trained "
                     "action by more than 10% of the latter's magnitude. Fixed "
                     "before the sweep ran, from the three-unit probe: the two "
                     "near-equilibrium units gained 0.28% and 0.53%, the "
                     "deviating one 118%, so 10% is about twenty times the "
                     "former and a twelfth of the latter and does not sit near "
                     "either. Units whose reference profit is near zero are "
                     "counted by the absolute gain instead, since a ratio "
                     "against a vanishing denominator is not a magnitude."),
                 cap_scale=CAP_SCALE, ramp_scale=RAMP_SCALE, voll=VOLL_IN_EFFECT,
                 eval_days=len(ev_days), market="01 day-ahead wholesale", fixture=FIXTURE_NAME, **runtime_stamp())))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
