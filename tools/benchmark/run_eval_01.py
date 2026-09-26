"""Market 01's honest arm and constant arm over the twelve evaluation days.

Days, paired metrics and product format all come from
`evaluation.py`; what is here is only what belongs to this market --
how to build its environment, what its two open-loop arms are, and how to read a
day back out of its state.

**The honest arm** is truthful bidding, the markup multiplier of one, which §9.3
sends to the true-cost envelope.  It is the baseline every learned arm is read
against, and it is not an instrument.

**The constant arm** is the markup of 2.0, which is the *upper endpoint* of the
action space fixed by §19.  It is the endpoint and **not a searched optimum**:
no grid was swept to find it, and the markup-grid arm that would have done the
searching is explicitly not part of this report.  It is chosen as the endpoint for
two reasons stated so that a reader does not supply a third: the endpoint is the
one constant point that requires no further choice, and it makes the instrument
strong, since a learned arm has to beat "everybody always bids the ceiling"
before it has shown anything.  Any reading of it as "the best fixed markup" is a
misreading, and the same sentence is written into every product this emits.

Day-ahead settles the whole quantity rather than a deviation from a position, so
unlike market 02 a negative profit here is a signal to investigate rather than a
property of the construction (report §3.2).

One episode per day at `episode_len = 1` (§19's `D = 1`), the full twenty-four
periods, opened through the environment's own `reset` -- see `evaluation.open_day`
for why not by hand.

CPU only.  The run point is stamped into every product.
"""
import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import (additive_markup_alpha, check_split_against_report,
                        effective_monitored_stamp, open_day, parse_monitored_lines,
                        split_days, system_cost, write_day, runtime_stamp)

CAP_SCALE = 0.60
RAMP_SCALE = 1.00
MARKUP_MAX = 2.0
VOLL_IN_EFFECT = 10_000.0
T = 24
K = 1

#: The two open-loop arms whose action is one scalar multiplier for every unit.
#: `None` means "the market's own truthful action".  The additive arm is not in
#: here and cannot be: its action is a 66-number profile, not a level, and
#: `--arms both` keeps meaning these two so that every invocation written before
#: the additive arm existed still runs the same two arms.
ARMS = {"honest": None, "constant": MARKUP_MAX}

ENDPOINT_NOTE = (
    "the constant arm bids the markup of 2.0, which is the upper endpoint of the "
    "action space of section 19 and NOT a searched optimum: no grid was swept, "
    "and the markup-grid arm that would have swept one is not part of this "
    "report. Reading it as 'the best fixed markup' is a misreading.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=None,
                    help="commitment fixture; defaults to the adopted T=24 one")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--days", default="eval", choices=("eval", "train", "all"))
    #: The two scenario scales, as flags rather than as the bare module
    #: constants -- the same change `run_eval_02.py` and `run_eval_03.py` took on
    #: 2026-09-12, for the same reason: `case73rts` adopts `cap_scale = 0.424`
    #: and the check below refuses a fixture whose scenario disagrees, so with
    #: the constants alone this driver could only be pointed at `case29gb`.
    #: Each defaults to the constant it replaces, so a command line naming
    #: neither runs exactly what it always ran.
    ap.add_argument("--cap-scale", type=float, default=CAP_SCALE)
    ap.add_argument("--ramp-scale", type=float, default=RAMP_SCALE)
    #: The third scenario scale, and it must be **named** rather than inferred:
    #: `load_commitment` (2026-09-13) refuses a fixture that declares a value
    #: other than 1.0 when the call does not name one, because `make_env` is
    #: blind to it and thirteen drivers were silently dropping it.  Defaults to
    #: `None`, which the loader reads as "the fixture must declare 1.0 or
    #: nothing" -- so every `case29gb` command line runs exactly what it always
    #: ran, and a `case73rts` one has to say `--p-min-scale 0.8` on purpose.
    ap.add_argument("--p-min-scale", type=float, default=None)
    #: Same flag and meaning as `run_rl_01.py`: 'all' is the dense route every
    #: archive was produced on, 'rated' the low-rank route on the lines with a
    #: published rating.  Stamped in `run_point`.
    ap.add_argument("--monitored-lines", default="all")
    #: `--unilateral U --alpha A` prices unit U at A and everyone else truthfully.
    #: It answers a different question from the constant arm and a percentage does
    #: not say which one it belongs to: moving everyone mixes "what my own action
    #: buys me" with "what everyone else's action buys me", and only the first is
    #: what a gradient can follow.  Measured in market 02 (2026-08-18): a unit
    #: whose profit moves +136% when everyone marks up loses 70% when it alone
    #: does.  Whether this market behaves the same way is the point of the option.
    ap.add_argument("--unilateral", default="",
                    help="comma-separated unit indices to deviate; others truthful")
    ap.add_argument("--alpha", type=float, default=None,
                    help="markup for the deviating units")
    ap.add_argument("--arms", default="both",
                    choices=("both", "honest", "constant", "additive"))
    #: The uniform ADDITIVE control group: every unit bids
    #: its own true marginal cost plus the same number of dollars.  Taken in
    #: $/MWh rather than as a multiplier because dollars is what it holds equal
    #: across units; the per-unit multiplier it needs is derived in
    #: `evaluation.additive_markup_alpha`, which is the one place that
    #: arithmetic is written for all three markets.
    ap.add_argument("--add-m", type=float, default=None,
                    help="$/MWh added to every unit's true marginal cost, for "
                         "--arms additive")
    args = ap.parse_args()

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")

    import powermarketjax
    print("package:", powermarketjax.__file__, flush=True)

    from powermarketjax.case import load_case, scale_min_output
    from powermarketjax.envs.day_ahead import (demand_from_meta, load_commitment,
                                               make_env, truthful_action)
    from powermarketjax.envs.day_ahead.commitment import FIXTURE_DIR

    path = (Path(args.fixture) if args.fixture else
            FIXTURE_DIR / "day_ahead_commitment_29gb_T24_relax.npz")
    fixture = load_commitment(path=path, n_periods=T,
                             p_min_scale=args.p_min_scale)
    meta = fixture["meta"]
    # the scenario is asserted against the fixture rather than assumed: a fixture
    # built at other factors describes another market and `make_env` would reject
    # it, but `voll` is not part of that check and has to be read here
    for key, given in (("cap_scale", args.cap_scale),
                       ("ramp_scale", args.ramp_scale),
                       ("voll", VOLL_IN_EFFECT)):
        got = meta.get(key)
        if got is None or abs(float(got) - given) > 1e-12:
            raise SystemExit(f"fixture meta {key}={got} disagrees with {given}")

    dates = [str(d) for d in meta["dates"]]
    ok, selected = check_split_against_report(dates, meta["case"])
    print(f"split matches report section 2.2: {ok}")
    if not ok:
        raise SystemExit(
            f"the split rule no longer reproduces the report's evaluation days.\n"
            f"  rule gives: {selected}\n"
            f"Resolve this deliberately -- either the window moved and the "
            f"report's list needs updating, or the rule was edited.")

    ev, tr = split_days(len(dates), meta["case"])
    days = {"eval": ev, "train": tr, "all": sorted(ev + tr)}[args.days]

    #: **The third scenario scale, applied to the case rather than passed to the
    #: operators.**  `make_env` checks `cap_scale`, `ramp_scale` and
    #: `n_segments` against the fixture and is *blind to `p_min_scale`* --
    #: `da_position.check_boundary_scenario` says so verbatim and guards its own
    #: driver against it, but this driver called `load_case` bare and so cleared
    #: a `case73rts` commitment sized for 0.80 x p_min at the registered 1.00.
    #: Measured 2026-09-13 on the 73rts 366-day fixture: the honest arm (markup
    #: 1.0) gave `shed -4.853e+02 MWh`, `mu 2.63e+290`, `profit -inf` on the
    #: net-load trough days and stayed clean (`mu 1.19e-11`) on the peak days --
    #: the overgeneration signature `scale_min_output`'s docstring describes,
    #: RTS carrying `sum p_min / sum p_max` 0.464 against `case29gb`'s 0.200.
    #: `scale_min_output(case, 1.0)` is the identity, so every `case29gb`
    #: product is bit-for-bit what it was.
    p_min_scale = float(meta.get("p_min_scale", 1.0))
    case = scale_min_output(load_case(meta["case"]), p_min_scale)
    print(f"p_min_scale in effect: {p_min_scale}", flush=True)
    monitored = parse_monitored_lines(args.monitored_lines, case)
    env, spec = make_env(case, fixture, demand_from_meta(meta), n_segments=K,
                         kind="markup", markup_max=MARKUP_MAX,
                         cap_scale=args.cap_scale, ramp_scale=args.ramp_scale,
                         monitored_lines=monitored)
    # stamped off the operators that were built, not off the flag; refuses if
    # the two disagree (`effective_monitored_stamp`)
    monitored_stamp, kkt_route = effective_monitored_stamp(spec, requested=monitored)
    print(f"monitored lines in effect: {'all' if monitored_stamp is None else monitored_stamp}; "
          f"kkt route: {kkt_route}", flush=True)
    params = env.make_params(episode_len=1)
    step = jax.jit(env.step)
    day_of = lambda st: int(st.cursor)

    truthful = truthful_action(case, K, T, kind="markup")
    # parsed before `run_point` because the stamp records both; see the comment
    # on `alpha` / `units` below
    uni = [int(t) for t in args.unilateral.split(",") if t.strip()]
    if uni and args.alpha is None:
        raise SystemExit("--unilateral needs --alpha")
    if args.arms == "additive" and args.add_m is None:
        raise SystemExit("--arms additive needs --add-m (in $/MWh)")
    # refuse rather than ignore: a run given --add-m under another arm would
    # produce the honest or constant arm's numbers under a command line that
    # reads as the additive treatment
    if args.add_m is not None and args.arms != "additive":
        raise SystemExit(f"--add-m is the additive arm's treatment but --arms "
                         f"is {args.arms!r}; it would be silently ignored")

    run_point = dict(cap_scale=args.cap_scale, ramp_scale=args.ramp_scale,
                     window=meta.get("window"), voll=VOLL_IN_EFFECT,
                     markup_max=MARKUP_MAX, episode_len=1,
                     monitored_lines=monitored_stamp,
                     kkt_route=kkt_route,
                     **runtime_stamp(),
                     market="01 day-ahead wholesale",
                     # Which units deviated and to what.  The arm name encodes
                     # both, but reading a treatment back out of a filename means
                     # parsing it, and a product that cannot state its own
                     # treatment in its metadata is not comparable against
                     # another one.  Measured 2026-08-19 on market 02: two runs
                     # of nominally the same construction disagreed by a factor
                     # of 333 on shed, and neither product recorded which units
                     # it moved, so the disagreement could not be diagnosed.
                     # `units` is null rather than absent for the arms it does
                     # not apply to: an absent key reads as "not recorded", a
                     # null reads as "not applicable".  Both fields describe the
                     # unilateral deviation only; the honest and constant arms
                     # are identified by `arm` and, for the constant, by
                     # `action_level` in the per-day extras.
                     alpha=(args.alpha if uni else None),
                     units=(sorted(uni) if uni else None),
                     fixture=path.name, note=ENDPOINT_NOTE)

    if uni:
        # `a105` not `a1.05`: a float rendered with `f"{float}"` gives a
        # variable-length token (`a1.5` vs `a1.05`), which sorts and greps
        # differently from what the writer had in mind.  Two decimals
        # scaled to an integer is fixed-length and matches the `--out-dir`
        # convention already in use (`eval01_u34_a105`).
        if abs(args.alpha * 100 - round(args.alpha * 100)) > 1e-9:
            raise SystemExit(f"--alpha {args.alpha} has more than two decimals; "
                             f"the arm name encodes it as an integer percent and "
                             f"would silently round it")
        wanted = [f"uni{'_'.join(map(str, uni))}_a{round(args.alpha * 100):03d}"]
    else:
        wanted = list(ARMS) if args.arms == "both" else [args.arms]
    for arm in wanted:
        if uni:
            action = jnp.asarray(truthful).at[jnp.asarray(uni)].set(args.alpha)
            # refuse rather than report a deviation that did not happen: an
            # out-of-range or duplicate index would silently reproduce the
            # honest arm, and that run looks exactly like a normal one
            moved = int((action != jnp.asarray(truthful)).sum())
            if moved != len(uni):
                raise SystemExit(f"--unilateral named {len(uni)} units but "
                                 f"{moved} entries differ from truthful")
            print(f"\narm {arm!r}: units {uni} at alpha={args.alpha}, "
                  f"the other {np.asarray(truthful).size - len(uni)} truthful",
                  flush=True)
            rows = []
            for day in days:
                key, state = open_day(env.reset, params, day, day_of)
                _o, _s, reward, _c, _dn, info = step(key, state, action, params)
                prod = float(np.sum(np.asarray(info["cost"], np.float64)))
                shed = float(info["shed_mwh"])
                prof = np.asarray(reward, np.float64)
                sc = system_cost(prod, [shed], VOLL_IN_EFFECT, 0.0)
                write_day(args.out_dir, arm, day, dates[day],
                          system_cost_value=sc, agent_profit=prof,
                          shed_mwh=np.asarray([shed]),
                          production_cost=prod, run_point=run_point)
                rows.append(sc)
            print(f"  {arm}: {len(rows)} days -> {args.out_dir}", flush=True)
            continue
        add_info = None
        if arm == "additive":
            level = None            # not a scalar level; the profile is below
            action, add_info = additive_markup_alpha(case, K, args.add_m,
                                                     MARKUP_MAX)
            print(f"\narm {arm!r}: every unit bids MC + {add_info['add_m']} "
                  f"$/MWh, i.e. alpha in [{add_info['alpha_min']:.4f}, "
                  f"{add_info['alpha_max']:.4f}], mean "
                  f"{add_info['alpha_mean']:.4f}; MC spans "
                  f"[{add_info['mc_min']:.4f}, {add_info['mc_max']:.4f}] $/MWh",
                  flush=True)
        else:
            level = ARMS[arm]
            action = truthful if level is None else jnp.full_like(truthful, level)
            print(f"\narm {arm!r}: action is "
                  + ("the market's truthful action (multiplier 1)" if level is None
                     else f"a constant multiplier of {level} (endpoint, not searched)"),
                  flush=True)
        # per arm rather than once for the invocation: `--arms` names one arm at
        # a time today, but a shared stamp would put `additive_m` on the honest
        # products the moment it names two, and a treatment recorded on a product
        # that did not get it is worse than one not recorded at all
        rp = dict(run_point, additive_m=(None if add_info is None
                                         else float(add_info["add_m"])),
                  alpha_profile=(None if add_info is None else dict(
                      file=None, n_units=add_info["n_units"],
                      min=add_info["alpha_min"], max=add_info["alpha_max"],
                      mean=add_info["alpha_mean"],
                      alpha_per_unit=add_info["alpha_per_unit"],
                      mc_per_unit=add_info["mc_per_unit"],
                      basis=add_info["basis"])),
                  alpha=(run_point["alpha"] if add_info is None
                         else add_info["alpha_mean"]))
        rows = []
        for day in days:
            key, state = open_day(env.reset, params, day, day_of)
            _o, _s, reward, _c, _dn, info = step(key, state, action, params)
            prod = float(np.sum(np.asarray(info["cost"], np.float64)))
            shed = float(info["shed_mwh"])
            prof = np.asarray(reward, np.float64)
            sc = system_cost(prod, [shed], VOLL_IN_EFFECT, 0.0)
            write_day(args.out_dir, arm, day, dates[day],
                      system_cost_value=sc, agent_profit=prof,
                      shed_mwh=np.asarray([shed]), production_cost=prod,
                      run_point=rp,
                      extra=dict(mu=float(info["mu"]),
                                 converged=bool(info["converged"]),
                                 # Both raw quantities are stored;
                                 # the share is not, so that a reader recomputes
                                 # it rather than inherits it.
                                 #
                                 # Basis, because two different things in this
                                 # repository are called a system cost.  The
                                 # CLEARING OBJECTIVE is offer-weighted and has
                                 # two terms (offer.p + VOLL.s); it contains no
                                 # no-load or start-up at all, those being fixed
                                 # once the commitment is fixed.  `system_cost`
                                 # here is the OTHER one: it is built from the
                                 # settlement's production cost, which is
                                 # energy + no_load + startup at TRUE cost, plus
                                 # VOLL times shed.  The share below is
                                 # no-load over that second quantity, and it is
                                 # the only one of the two the share is even
                                 # defined against.
                                 no_load_cost=float(np.sum(np.asarray(
                                     info["no_load_cost"], np.float64))),
                                 energy_cost=float(np.sum(np.asarray(
                                     info["energy_cost"], np.float64))),
                                 startup_cost=float(np.sum(np.asarray(
                                     info["startup_cost"], np.float64))),
                                 cost_basis=("system_cost = production_cost "
                                             "(energy + no_load + startup, at "
                                             "true cost) + VOLL * shed; NOT the "
                                             "clearing objective, which is "
                                             "offer-weighted and has two terms"),
                                 revenue=float(np.sum(np.asarray(info["revenue"],
                                                                 np.float64))),
                                 congested_line_periods=int(
                                     info["congested_line_periods"]),
                                 action_level=(None if level is None
                                               else float(level))))
            rows.append((day, dates[day], sc, float(prof.sum()), shed,
                         float(info["mu"]), bool(info["converged"]),
                         float(np.sum(np.asarray(info["no_load_cost"],
                                                 np.float64)))))
            print(f"  day {day:3d} {dates[day]}  system_cost {sc:14.4e}  "
                  f"profit {float(prof.sum()):14.4e}  shed {shed:.3e} MWh  "
                  f"mu {float(info['mu']):.2e}", flush=True)

        sc_all = np.array([r[2] for r in rows])
        nl_all = np.array([r[7] for r in rows])
        # printed from the two totals, never stored as a share
        print(f"  {arm}: no-load total {nl_all.sum():.6e} | system_cost total "
              f"{sc_all.sum():.6e} | share {100 * nl_all.sum() / sc_all.sum():.2f}% "
              f"over {len(rows)} days -- recomputed here from the two totals, "
              f"which are what the products store", flush=True)
        pr_all = np.array([r[3] for r in rows])
        sh_all = np.array([r[4] for r in rows])
        print(f"  {arm}: system_cost median {np.median(sc_all):.4e} "
              f"total {sc_all.sum():.4e} | profit total {pr_all.sum():.4e} "
              f"({int((pr_all < 0).sum())} of {len(rows)} days negative) | "
              f"shed total {sh_all.sum():.1f} MWh | "
              f"unconverged {int(sum(1 for r in rows if not r[6]))}")
        if (pr_all < 0).any():
            print("  NOTE: day-ahead settles the whole quantity, not a deviation, "
                  "so a negative day here is a signal to investigate rather than "
                  "a property of the construction (report section 3.2).")


if __name__ == "__main__":
    main()
