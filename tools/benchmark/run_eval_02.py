"""Market 02's baseline arms over the twelve evaluation days.

The path this walks is the one markets 01 and 03 plug into: it takes the days,
the paired metrics and the product format from `evaluation.py` and adds only
what is specific to this market -- how to build its environment, what its honest
arm is, and how to read a day back out of its state.

**The honest arm is truthful bidding**, `alpha = 1`, which the markup map sends
to the true-cost envelope.  It is a baseline rather than an instrument: it says
what the market does when nobody exercises the action, and every learned arm is
measured against it.

**The constant arm is `alpha = markup_max`, the upper endpoint of the action
space -- and an endpoint is not a searched optimum.**  It reports what one fixed,
maximally aggressive markup achieves; the best constant markup is a different
quantity and would need the grid sweep of `arms.markup_grid`, which reports
whether its winner sits on the grid's boundary for exactly this reason.  Reading
this arm as "the best a constant action can do" would present a naive baseline as
a strength-matched one, which is not allowed.  The caveat is repeated in
`run_point["note"]` on every product and printed at run time, so it reaches a
reader who sees only one of the three.

Evaluation is one episode per day, the full 48 periods, opened through the
environment's own `reset` (see `evaluation.open_day` for why not by hand).

CPU only.  Run point is stamped into every product.
"""
import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import (additive_markup_alpha, check_split_against_report,
                        effective_monitored_stamp, open_day,
                        parse_monitored_lines, split_days, system_cost,
                        write_day, runtime_stamp)

CAP_SCALE = 0.60
RAMP_SCALE = 1.00
#: What `p_min_scale` means on a position fixture that does not record it.  The
#: field did not exist before 2026-09-05 and every fixture written until then was
#: built on the registered case, so absent reads as 1.0.  `precommit.IMPLIED_PRIOR`
#: and `da_position.P_MIN_SCALE_IMPLIED_PRIOR` already record that reading; this
#: mirrors it rather than inventing a second one -- two devices reading an absent
#: field differently is worse than both reading it wrong.
P_MIN_SCALE_IMPLIED_PRIOR = 1.0
MARKUP_MAX = 2.0
VOLL_IN_EFFECT = 10_000.0
#: One segment per unit, which is what this market has always run at and what
#: the additive arm below requires.  Named rather than repeated at the
#: `make_env` call, so the arm and the environment cannot disagree about it.
N_SEGMENTS = 1
ARMS = ("honest", "constant", "uniform", "unilateral", "vector", "additive")


#: One note per arm, looked up rather than derived.  See the comment at the
#: `arm_note` assignment for why this is a table and not a conditional.
ARM_NOTES = {
    "honest": "honest arm is truthful bidding, alpha = 1",
    "constant": ("constant arm is alpha = markup_max, the upper endpoint of the "
                 "action space, NOT a searched optimum; the best constant markup "
                 "is a different quantity requiring arms.markup_grid"),
    "uniform": ("uniform arm: every unit bids `alpha`. This moves the price "
                "channel only where commitment is exogenous; it is not the "
                "constant arm unless alpha == markup_max"),
    "vector": ("vector arm: every unit bids the markup named for it in "
               "`alpha_file`, one fixed value per unit for the whole run. Used "
               "to ask how much the UNEVENNESS of a given markup profile moves "
               "the market, holding the profile constant. It is NOT a learned "
               "arm and must not be read as one: a policy's markup varies with "
               "state and period, and this arm freezes one summary of it"),
    "additive": ("additive arm: every unit bids its own true marginal cost "
                 "PLUS the same `add_m` $/MWh. It is the control group for the "
                 "uniform MULTIPLICATIVE arm and NOT that arm at another level: "
                 "a multiplicative markup raises an expensive unit's bid by more "
                 "dollars than a cheap one's, and this one removes exactly that "
                 "slope, so a difference between the two is the slope's doing "
                 "and not the level's. It reaches the multiplicative action "
                 "space as alpha_i = 1 + add_m / MC_i (ADR-0016 section 5), so "
                 "it needs no new action space and `alpha` in this stamp is the "
                 "profile MEAN, a summary -- `additive_m` is the treatment"),
    "unilateral": ("unilateral arm: the units in `units` bid `alpha`, every "
                   "other unit bids truthfully. This answers what one unit's own "
                   "action buys it, which is the quantity a gradient follows; it "
                   "is NOT the same question as moving everyone"),
}


#: The table must cover the choices, or an arm added to one and not the other
#: reaches `ARM_NOTES[args.arm]` only when someone runs it.  Checked at import so
#: it fails on the developer's machine rather than three hours into a sweep.
assert set(ARM_NOTES) == set(ARMS), (
    f"ARM_NOTES and ARMS disagree: {set(ARM_NOTES) ^ set(ARMS)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--position", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-lookahead", type=int, default=1,
                    help="1 is the market as specified; the N=4 row is a "
                         "separate lookahead experiment, not this baseline")
    ap.add_argument("--days", default="eval", choices=("eval", "train", "all"))
    #: **A day-range slice, so a run can be split across several processes.**
    #: This driver steps one environment per period; on 813nem it measured 24
    #: minutes per day, and **RSS grows by about 8.84 GB per hour** (measured
    #: 2026-09-18 00:0x: 103.09 G grew 0.221 G in 1.5 minutes; at 7 of 36 days
    #: it already held 103 G, and over 36 days it would exhaust this machine's
    #: 251 G first).  One process cannot finish 36 days, **running in segments
    #: is the only form that finishes**, and one new process per segment also
    #: cuts that growth off.  Days are independent of each other (each day
    #: opens itself once with `open_day`, and the product is one file per day),
    #: so slicing changes no day's result.
    #:
    #: Both are **position indices** (into the list `--days` selected),
    #: inclusive; with neither given not one day is dropped and the output is
    #: byte for byte what it was before the slice existed.  `day_slice` is
    #: stamped into the run point, otherwise a segment product cannot say which
    #: days it covers (a stamp can only be written by the writer; there is
    #: nowhere to add it later).
    ap.add_argument("--day-from", type=int, default=None,
                    help="position index into the selected day list, inclusive "
                         "(default: 0). Days are independent, so slicing is for "
                         "running one window per process")
    ap.add_argument("--day-to", type=int, default=None,
                    help="position index, inclusive (default: the last day)")
    #: The two scenario scales this market clears at, as flags rather than as the
    #: bare module constants.  `case73rts` adopts `cap_scale = 0.424`
    #: and the check below
    #: refuses a position fixture whose scenario disagrees, so with the constants
    #: alone this driver could only ever be pointed at `case29gb`'s scenario --
    #: measured 2026-09-12: `fixture meta cap_scale=0.424 disagrees with 0.6`.
    #: Each defaults to the constant it replaces, so a command line naming
    #: neither runs exactly what it always ran.
    #:
    #: **The third scale, `p_min_scale`, is deliberately NOT a flag.**  It is
    #: recorded by the position fixture itself and read from there below, the way
    #: `half_hourly_from_meta` reads the demand series: the commitment in the
    #: fixture was built at that minimum-output level and clearing it at another
    #: is not a choice anyone should be able to make on a command line.
    ap.add_argument("--cap-scale", type=float, default=CAP_SCALE)
    ap.add_argument("--ramp-scale", type=float, default=RAMP_SCALE)
    ap.add_argument("--arm", default="honest", choices=ARMS)
    #: **Clearing route and IPM settings: three flags, none of which changes
    #: the default.**  This driver could previously run only at the market's
    #: defaults, so the arms it produced and the trained arms `run_rl_02` runs
    #: with a recipe **were not the same apparatus**: on 813nem the three
    #: trained seeds take the low-rank arrowhead of `rated` + `reg_coef 1e-16`
    #: + `stop_tol (1e-8, 1e-7)`, while this driver could only take the dense
    #: all-lines route with the default reg/stop.  **Using the latter as the
    #: fixed reference for the former would count the route difference into
    #: the displacement** -- on the day-ahead position the two routes differ in
    #: price by only 4.93e-07 \$/MWh but in per-unit dispatch by up to 62.39 MW
    #: (the measurement in the comment in `run_rl_02`), and `q_da` is exactly
    #: what this market is handed.
    #:
    #: All three default to `None` / `"all"`, i.e. without the flags this is
    #: byte for byte the original apparatus.
    ap.add_argument("--monitored-lines", default="all",
                    help="line limits the clearing carries: all (default, "
                         "dense KKT, what every archive was produced on), "
                         "rated (the case's published ratings, low-rank), or "
                         "a comma-separated list of line indices")
    ap.add_argument("--reg-coef", type=float, default=None,
                    help="ipm.make_solver reg_coef (default: ipm.REG_COEF); 1e-16 on case813nem")
    ap.add_argument("--stop-tol", default=None,
                    help="mu_tol,dual_tol: stop the Newton loop at this tolerance, max_iter "
                         "as the cap (default: the fixed trip count); 1e-8,1e-7 on case813nem")
    #: `uniform`: every unit bids `--alpha`.  `unilateral`: the units named by
    #: `--units` bid `--alpha` and everyone else bids truthfully.  The two answer
    #: different questions and a percentage does not say which one it belongs to:
    #: an all-move comparison mixes "what my own action buys me" with "what
    #: everyone else's action buys me", and only the first is what a gradient can
    #: follow.  Measured 2026-08-18: agent 39's profit moves +136% between the
    #: honest and the all-at-cap arm, and how much of that it can reach alone was
    #: unmeasured until this option existed.
    ap.add_argument("--alpha", type=float, default=None,
                    help="markup for the uniform/unilateral arms")
    ap.add_argument("--units", default="",
                    help="comma-separated unit indices for the unilateral arm")
    #: The vector arm reads a per-unit markup profile from a file rather than
    #: taking it on the command line: the profiles this answers questions about
    #: come out of a probe over 66 units, and a 66-number command line is a
    #: transcription error waiting to happen.  The key is named so the file
    #: cannot be confused with a product that merely contains some alphas.
    ap.add_argument("--alpha-file", default="",
                    help="npz with key `alpha_per_unit` for the vector arm")
    #: The additive arm takes the markup in dollars rather than as a multiplier,
    #: because dollars is what it holds equal across units; the multiplier it
    #: needs is derived per unit and is not something anyone should type.
    ap.add_argument("--add-m", type=float, default=None,
                    help="$/MWh added to every unit's true marginal cost, for "
                         "the additive arm")
    #: **Off by default; when on, it only stores one more array and changes no
    #: stored key.**  This market's inter-period coupling passes only through
    #: `RealTimeState.p_prev` (`envs/real_time/env.py` line 135), and none of
    #: the stored `agent_profit` / `shed_mwh` / `production_cost` / `lmp` can
    #: answer "how far can the remaining units still ramp up at the moment of
    #: an outage" -- that needs the dispatch itself.  Stored as `dispatch`,
    #: shape `(T_RT, n_units)`, row t being the dispatch after period t clears
    #: (i.e. the `p_prev` read by period t+1's ramp rows).  Without this flag
    #: `arrays` is `None`, and `write_day`'s payload is key for key what it was.
    ap.add_argument("--save-dispatch", action="store_true",
                    help="also store the per-period dispatch (T_RT, n_units) "
                         "under key `dispatch`; default off, and off it writes "
                         "exactly the keys it always wrote")
    args = ap.parse_args()

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")

    import powermarketjax
    print("package:", powermarketjax.__file__, flush=True)

    from powermarketjax.case import load_case, scale_min_output
    from powermarketjax.envs.day_ahead import demand_from_meta
    from powermarketjax.envs.real_time import load_da_position
    from powermarketjax.envs.real_time.demand import (T_RT,
                                                      half_hourly_from_meta)
    from powermarketjax.envs.real_time.env import make_env

    pos = load_da_position(path=args.position)
    meta = pos["meta"]
    for key, given in (("cap_scale", args.cap_scale),
                       ("ramp_scale", args.ramp_scale),
                       ("voll", VOLL_IN_EFFECT)):
        got = meta.get(key)
        if got is None or abs(float(got) - given) > 1e-12:
            raise SystemExit(f"fixture meta {key}={got} disagrees with {given}")

    dates = [str(d) for d in meta["dates"]]
    ok, selected = check_split_against_report(dates, meta["case"])
    print(f"split matches report §2.2: {ok}")
    if not ok:
        raise SystemExit(
            f"the split rule no longer reproduces the report's evaluation days.\n"
            f"  rule gives: {selected}\n"
            f"Resolve this deliberately -- either the window moved and the "
            f"report's list needs updating, or the rule was edited. Evaluating "
            f"on days the report does not name is the one outcome to avoid.")

    ev, tr = split_days(len(dates), meta["case"])
    days = {"eval": ev, "train": tr, "all": sorted(ev + tr)}[args.days]
    #: The slice comes after `check_split_against_report` (above): that check
    #: asks whether **the whole evaluation day set the rule gives** agrees with
    #: the report, which has nothing to do with which segment this run covers,
    #: so the slice cannot go before it.
    _n_all = len(days)
    day_slice = None
    if args.day_from is not None or args.day_to is not None:
        _lo = 0 if args.day_from is None else int(args.day_from)
        _hi = _n_all - 1 if args.day_to is None else int(args.day_to)
        if not (0 <= _lo <= _hi < _n_all):
            raise SystemExit(f"--day-from/--day-to must lie in [0, {_n_all - 1}] with from <= to; "
                             f"got {_lo}..{_hi} (`--days {args.days}` has {_n_all} days)")
        days = days[_lo: _hi + 1]
        day_slice = [_lo, _hi, _n_all]
        print(f"day slice {_lo}..{_hi} of {_n_all}: {len(days)} days {days}", flush=True)

    # `p_min_scale` is the position's own record, read rather than declared.  The
    # commitment in the fixture was built at that minimum-output level, and
    # clearing it at another is SILENT: the interior-point method diverges rather
    # than raising, every array comes back well-formed and the product still
    # stamps the fixture's scale beside a dispatch run at another.  Measured on
    # `case73rts` 2026-09-12: unapplied, the committed minimum output exceeds
    # demand in 38 of the day's 48 periods and `worst mu` reaches 1.733e+293;
    # applied, 0 of 48 and 1.167e-11.  `scale_min_output(case, 1.0) is case`, so
    # every `case29gb` run is bit-for-bit what it was.
    p_min_scale = float(meta.get("p_min_scale", P_MIN_SCALE_IMPLIED_PRIOR))
    case = scale_min_output(load_case(meta["case"]), p_min_scale)
    # both legs come from the position's own record: the realised series has a
    # loader per case since 2026-09-09, so naming the GB one here would serve
    # British demand to whatever network the fixture names
    hh, _ = half_hourly_from_meta(meta)
    fc, _a, _d = demand_from_meta(meta)
    monitored = parse_monitored_lines(args.monitored_lines, case)
    stop_tol = (None if args.stop_tol is None
                else tuple(float(v) for v in args.stop_tol.split(",")))
    env, spec = make_env(case, pos, hh, fc, n_segments=N_SEGMENTS,
                         markup_max=MARKUP_MAX,
                         cap_scale=args.cap_scale,
                         ramp_scale=args.ramp_scale,
                         n_lookahead=args.n_lookahead,
                         monitored_lines=monitored,
                         reg_coef=args.reg_coef,
                         stop_tol=stop_tol)
    #: **Read back from the operator's own `spec`, not from the flag.**  This
    #: market builds two clearing operators (the stepping one and the one that
    #: opens the boundary), and a flag reaching one and not the other would be
    #: invisible in any product; `make_env` itself refuses the two disagreeing,
    #: and what the driver side has to check is "did the flag reach `make_env`
    #: at all".  A flag an operator did not take exits here, instead of being
    #: stamped as the value in effect.
    mon_stamp, kkt_route = effective_monitored_stamp(spec, requested=monitored)
    _b_stamp, _b_route = effective_monitored_stamp(
        {"clearing": spec["boundary_clearing"]}, requested=monitored)
    if _b_route != kkt_route:
        raise SystemExit(f"the step clearing took the {kkt_route} KKT route and "
                         f"the boundary {_b_route}; the episode would open at "
                         f"one route's vertex and be cleared on the other")
    ipm_reg_coef = float(spec["clearing"]["reg_coef"])
    ipm_stop_tol = spec["clearing"]["stop_tol"]
    if args.reg_coef is not None and ipm_reg_coef != float(args.reg_coef):
        raise SystemExit(f"--reg-coef {args.reg_coef} requested but the clearing "
                         f"was built on {ipm_reg_coef}")
    if ((None if stop_tol is None else tuple(stop_tol))
            != (None if ipm_stop_tol is None else tuple(ipm_stop_tol))):
        raise SystemExit(f"--stop-tol {stop_tol} requested but the clearing was "
                         f"built on {ipm_stop_tol}")
    print(f"monitored lines in effect: {'all' if mon_stamp is None else mon_stamp}; "
          f"kkt route: {kkt_route}; ipm reg_coef {ipm_reg_coef}; "
          f"ipm stop_tol {ipm_stop_tol}", flush=True)
    params = env.make_params(episode_len=T_RT)
    truthful = jnp.asarray(env.truthful_action())
    if args.arm == "honest":
        action = env.truthful_action()
    elif args.arm == "vector":
        if not args.alpha_file:
            raise SystemExit("--arm vector needs --alpha-file")
        blob = np.load(args.alpha_file, allow_pickle=True)
        if "alpha_per_unit" not in blob.files:
            raise SystemExit(
                f"{args.alpha_file} has no `alpha_per_unit` key (has "
                f"{sorted(blob.files)[:8]}); the vector arm will not guess which "
                "array is the markup profile")
        vec = jnp.asarray(np.asarray(blob["alpha_per_unit"], np.float64))
        if vec.shape != truthful.shape:
            raise SystemExit(
                f"`alpha_per_unit` has shape {vec.shape} but this market's "
                f"action is {truthful.shape}")
        # the same guard the unilateral arm carries: a profile that happens to
        # be all ones reproduces the honest arm exactly, and a silent honest run
        # under another arm's name is the failure this whole day has been about
        moved = int((np.asarray(vec) != np.asarray(truthful)).sum())
        if moved == 0:
            raise SystemExit(
                f"`alpha_per_unit` in {args.alpha_file} equals the truthful "
                "action on every unit, so this run would be the honest arm "
                "under the vector arm's name")
        action = vec
        print(f"arm=vector: {moved}/{vec.size} entries differ from truthful, "
              f"alpha in [{float(vec.min()):.4f}, {float(vec.max()):.4f}], "
              f"mean {float(vec.mean()):.4f}", flush=True)
    elif args.arm == "additive":
        if args.add_m is None:
            raise SystemExit("--arm additive needs --add-m (in $/MWh)")
        vec, add_info = additive_markup_alpha(case, N_SEGMENTS, args.add_m,
                                              MARKUP_MAX)
        if vec.shape != truthful.shape:
            raise SystemExit(
                f"the additive action has shape {vec.shape} but this market's "
                f"action is {truthful.shape}")
        action = vec
        print(f"arm=additive: every unit bids MC + {add_info['add_m']} $/MWh, "
              f"i.e. alpha in [{add_info['alpha_min']:.4f}, "
              f"{add_info['alpha_max']:.4f}], mean {add_info['alpha_mean']:.4f}; "
              f"MC spans [{add_info['mc_min']:.4f}, {add_info['mc_max']:.4f}] "
              f"$/MWh", flush=True)
    elif args.arm in ("uniform", "unilateral"):
        if args.alpha is None:
            raise SystemExit(f"--arm {args.arm} needs --alpha")
        if args.arm == "uniform":
            action = jnp.full_like(truthful, args.alpha)
            print(f"arm=uniform: every unit at alpha={args.alpha}", flush=True)
        else:
            idx = [int(t) for t in args.units.split(",") if t.strip()]
            if not idx:
                raise SystemExit("--arm unilateral needs --units")
            action = truthful.at[jnp.asarray(idx)].set(args.alpha)
            # refuse rather than report a deviation that did not happen: an empty
            # or out-of-range index list would silently reproduce the honest arm
            moved = int((action != truthful).sum())
            if moved != len(idx):
                raise SystemExit(f"--units named {len(idx)} units but {moved} "
                                 f"entries differ from truthful; check the indices")
            print(f"arm=unilateral: units {idx} at alpha={args.alpha}, "
                  f"the other {truthful.size - len(idx)} truthful", flush=True)
    else:
        # the upper endpoint of the action space, not a searched best constant
        action = jnp.full_like(jnp.asarray(env.truthful_action()), MARKUP_MAX)
        print(f"arm=constant: alpha = {MARKUP_MAX} is the UPPER ENDPOINT of the "
              f"action space, not a searched optimum. The best constant markup "
              f"is a different quantity and needs arms.markup_grid.", flush=True)
    step = jax.jit(env.step)
    day_of = lambda st: int(st.cursor) // T_RT

    # `alpha` is what each deviating unit bids; for the two arms that do not
    # take one it is the value they are pinned to, not None, so a reader never
    # has to know the convention to interpret the number.
    arm_alpha = {"honest": 1.0, "constant": float(MARKUP_MAX)}.get(
        args.arm, args.alpha)
    arm_units = (sorted(int(t) for t in args.units.split(",") if t.strip())
                 if args.arm == "unilateral" else None)
    # The vector arm's treatment IS the profile, so the profile travels in the
    # product.  Recording a single `alpha` for it would be a scalar standing in
    # for 66 numbers, and recording `null` would be a stamp field that reads as
    # recorded while naming nothing -- the same failure refused for `window`
    # earlier today.  The whole profile is 66 floats; it is cheaper to store it
    # than to make a reader reconstruct which run used which one.
    arm_profile = None
    if args.arm == "additive":
        # the same field the vector arm fills, so one reader handles both: the
        # treatment is a 66-number profile either way.  `additive_m` beside it is
        # the one number that generated it, and it is what a comparison across
        # m = 5 / 10 / 20 keys on.
        arm_profile = dict(
            file=None, n_units=add_info["n_units"],
            n_distinct=int(np.unique(np.asarray(action, np.float64)).size),
            min=add_info["alpha_min"], max=add_info["alpha_max"],
            mean=add_info["alpha_mean"],
            alpha_per_unit=add_info["alpha_per_unit"],
            mc_per_unit=add_info["mc_per_unit"], basis=add_info["basis"])
        arm_alpha = add_info["alpha_mean"]
    if args.arm == "vector":
        v = np.asarray(action, np.float64).ravel()
        arm_profile = dict(
            file=args.alpha_file, n_units=int(v.size),
            n_distinct=int(np.unique(v).size),
            min=float(v.min()), max=float(v.max()), mean=float(v.mean()),
            alpha_per_unit=[float(x) for x in v])
        arm_alpha = float(v.mean())   # labelled below as the profile's mean

    run_point = dict(cap_scale=args.cap_scale, ramp_scale=args.ramp_scale,
                     p_min_scale=p_min_scale,
                     #: All four stamps are taken from the operator's `spec`,
                     #: not from the flags; the names are the four `run_rl_02`
                     #: writes, so the two sides' products compare directly.
                     #: Which days this segment covers.  `null` means "the
                     #: whole window", not "not recorded" -- without the flags
                     #: it is null, read the same way as products from before
                     #: the slice existed.
                     day_slice=day_slice,
                     monitored_lines=mon_stamp, kkt_route=kkt_route,
                     ipm_reg_coef=ipm_reg_coef, ipm_stop_tol=ipm_stop_tol,
                     window=meta.get("window"), voll=VOLL_IN_EFFECT,
                     n_lookahead=args.n_lookahead, markup_max=MARKUP_MAX,
                     **runtime_stamp(),
                     market="02 real-time balancing", arm=args.arm,
                     # The markup every deviating unit bids, and which units they
                     # are.  Measured 2026-08-19: without these two the products
                     # of the `uniform` and `unilateral` arms cannot say what
                     # treatment produced them, and one such pair disagreed by a
                     # factor of 333 on shed with no way to tell whether the two
                     # runs used the same split.  `units` is null rather than
                     # absent for the arms it does not apply to -- an absent key
                     # reads as "not recorded", a null reads as "not applicable".
                     alpha=arm_alpha, units=arm_units,
                     # for the vector arm `alpha` above is the profile MEAN, a
                     # summary; `alpha_profile` is the treatment itself
                     alpha_profile=arm_profile,
                     # null for every other arm rather than absent: an absent
                     # key reads as "not recorded", a null as "not applicable"
                     additive_m=(None if args.arm != "additive"
                                 else float(args.add_m)),
                     # A dict rather than a conditional expression.  The previous
                     # form was a two-branch ternary whose fallback said "honest
                     # arm is truthful bidding, alpha = 1"; `uniform` and
                     # `unilateral` were added later and both landed in that
                     # fallback, so every non-uniform product on disk claimed to
                     # be the honest arm.  A missing key here raises instead.
                     arm_note=ARM_NOTES[args.arm],
                     note=("the N=4 look-ahead comparison reports a different "
                           "quantity from the retired '92%' figure: that one "
                           "compared N=1 against N=48, neither of which is an "
                           "arm anyone runs. Do not read the two against each "
                           "other."))

    rows = []
    for day in days:
        key, state = open_day(env.reset, params, day, day_of)
        prod, shed, prof, mus, disp = 0.0, [], None, [], []
        for _ in range(T_RT):
            _o, state, reward, _c, _dn, info = step(key, state, action, params)
            prod += float(np.sum(np.asarray(info["cost"], np.float64)))
            shed.append(float(info["shed_mwh"]))
            r = np.asarray(reward, np.float64)
            prof = r if prof is None else prof + r
            mus.append(float(info["mu"]))
            if args.save_dispatch:
                disp.append(np.asarray(state.p_prev, np.float64))
        sc = system_cost(prod, shed, VOLL_IN_EFFECT, 0.0)
        path = write_day(args.out_dir, args.arm, day, dates[day],
                         system_cost_value=sc, agent_profit=prof,
                         shed_mwh=np.asarray(shed), production_cost=prod,
                         run_point=run_point,
                         arrays=(dict(dispatch=np.stack(disp))
                                 if args.save_dispatch else None),
                         extra=dict(mu_max=max(mus),
                                    unconverged=int(sum(m > 1e-9 for m in mus)),
                                    shed_cells=int(sum(s > 0 for s in shed))))
        #: **Clear the compilation cache once a day.**  This driver steps one
        #: environment per period, one `open_day` plus 48 `step`s a day, and
        #: compiled artefacts accumulate day by day without being released.
        #: Measured (29gb, 12 evaluation days, 2026-09-18 01:3x, RSS as the
        #: current VmRSS of `/proc/self/status`, not the `ru_maxrss` peak):
        #: **without clearing 1.66 -> 5.22 G (+3.56 G over 12 days), clearing
        #: daily 1.63 -> 2.19 G (+0.56 G), a factor of 6.4.**  On 813nem, by the
        #: long baseline, 25 to 52 GB per hour; one process over 36 days would
        #: exhaust this machine's 251 G first -- which is how an honest arm,
        #: 7 days into 36, had to be killed at 2026-09-18 00:2x.
        #:
        #: **It changes no number**: what is cleared is the compiled
        #: executable, and the same HLO recompiles to the same thing; the
        #: byte-for-byte gate below checks exactly this.
        #:
        #: **One thing is still not fixed**: `jax.live_arrays()` grows by 25 a
        #: day on both curves and `clear_caches` removes none of them (140 ->
        #: 415 over 12 days), shapes scalar and `(66,)` -- small arrays held by
        #: a reference chain, not something this line can fix, and not the bulk
        #: of the RSS.  Recorded separately.
        jax.clear_caches()
        rows.append((day, dates[day], sc, float(prof.sum()),
                     int(sum(s > 0 for s in shed)), max(mus)))
        print(f"  day {day:3d} {dates[day]}  system_cost {sc:14.4e}  "
              f"profit {float(prof.sum()):14.4e}  shed_cells "
              f"{int(sum(s > 0 for s in shed)):2d}  mu_max {max(mus):.2e}",
              flush=True)

    print(f"\n{len(rows)} days written to {args.out_dir}")
    print(f"total system cost {sum(r[2] for r in rows):.6e}   "
          f"total profit {sum(r[3] for r in rows):.6e}   "
          f"shed cells {sum(r[4] for r in rows)}")


if __name__ == "__main__":
    main()
