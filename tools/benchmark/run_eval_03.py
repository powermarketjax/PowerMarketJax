"""Market 03's two open-loop arms over the twelve evaluation days.

Days, paired metrics and product format come from `evaluation.py`;
this file adds only what belongs to this market -- how its environment is built,
what its two arms are, and how a day is read out of its state.

**The honest arm and the optimisation arm are the same number here, and that is
a property of the market rather than missing work.**  This market clears one
period at a time, its commitment is exogenous (§15), and under truthful offers
the clearing *is* the cost-minimising dispatch of that period -- so there is
nothing for an optimiser to improve on and no relax-round-resolve gap to
measure.  Market 01's two columns differ because its three-step clearing carries
a rounding heuristic; the same difference here would have to come from
somewhere, and there is nowhere for it to come from.  The consequence is worth
stating positively: **every gap this market's learned arms show is strategic
behaviour**, with no approximation error mixed in, which is a cleaner experiment
than 01 can run.

**The constant arm is an instrument, not a baseline.**  It runs
one fixed action -- markup at the cap, reserve offer at the truthful baseline --
and answers "did the optimisation fail": a learned arm no better than a single
constant action has not optimised.  Its markup of 2.0 is `MARKUP_MAX`, the
**upper endpoint of the action space, not a searched optimum**.  Nothing here
looked for the best constant markup, so this number is what one particular fixed
action achieves and a lower bound on what the best fixed action would; reading
it as "the best a constant policy can do" would be exactly the naive-baseline
substitution that is not allowed.

**The day-ahead position is the real one.**  Earlier work in this market built
`q_da` synthetically inside the driver because the position fixture was not in
the repository; it is now, and the synthetic construction is not used here.  A
synthetic day-ahead schedule would make the reserve market's opening state a
thing nobody committed to.

CPU only.  The run point is stamped into every product.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import (refuse_ineffective_lowrank_flags,
                        additive_markup_alpha, check_split_against_report,
                        converged_reward_split, effective_monitored_stamp,
                        open_day_start, parse_monitored_lines, split_days,
                        system_cost, write_day, runtime_stamp)

CASE = "29gb"
THETA = (1.0 / 6.0, 0.5)
#: The reserve price cap per case, and the action scale derived from it.  Two
#: conventions rather than two calibrations.  `pi_scale := volr / 5` fixes the
#: SHAPE of the action box -- section 306's ceiling depends only on the ratio --
#: so the three cases' learners act in boxes of the same shape, which is the
#: isomorphism a control needs to have; `volr` is then set proportional
#: to each fleet's highest marginal cost at minimum output, ratio 1.131 from the
#: British case.  `case29gb`'s 250 is the status quo and does not move, so every
#: unflagged 29gb run is bit-for-bit what it was.
#:
#: **Read these through `volr_pi_scale(case)`, never as two bare module
#: constants.**  `from run_eval_03 import VOLR` binds a value at import time and
#: nothing rebinds it afterwards, which is how `case73rts`'s whole 03
#: grid came to be cleared at the British cap while its products stamped 250 as
#: if it had been chosen -- 432 files.  The failure
#: left no trace in the products, so the accessor refuses an unknown case rather
#: than defaulting to one.
PI_SCALE_RATIO = 5.0
VOLR_BY_CASE = {"29gb": 250.0, "73rts": 136.0, "813nem": 147.0}
#: §18, the adopted requirement fractions.  This line states what the
#: adopted value **is**, so it moves whenever §18 moves; it is not a record
#: of what some product was measured at, and it is not the input a device
#: reads back.  Those two are the other reasons a fraction appears in this
#: repository, and they do not move with the adopted value.
BETA = (0.050, 0.050)


def volr_pi_scale(case_name):
    """The adopted ``(volr, pi_scale)`` for one case.

    Raises rather than defaulting: a case with no row has no adopted cap, and
    handing back another case's would be exactly the silent substitution
    described above.
    """
    key = str(case_name)
    if key not in VOLR_BY_CASE:
        raise SystemExit(
            f"no VOLR adopted for case {key!r}; VOLR_BY_CASE carries "
            f"{sorted(VOLR_BY_CASE)}. Add a row there and re-run what the old "
            f"value produced, rather than falling back to another case's cap")
    volr = VOLR_BY_CASE[key]
    return volr, volr / PI_SCALE_RATIO
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
DELTA = 0.5
T_DAY = 48
#: One segment per unit, which is what this market runs at and what the
#: additive arm requires.  Named rather than repeated at the
#: `make_ancillary_env` call, so the arm and the environment cannot disagree.
N_SEGMENTS = 1

#: Shed below this is the strict-interior phantom, not unserved energy.  The
#: clearing widens every zero-width box to `OFF_EPS` so the interior-point method
#: has somewhere to stand, and the shed variable of a bus with no load comes back
#: at ~2.3e-21 MWh rather than at zero.  Counting cells with `shed > 0` therefore
#: reports 48 of 48 periods shedding on every day, which is how this number first
#: read; it is an artefact of the floor and says nothing about the dispatch.
#: `tools/benchmark/run_eval_02.py` counts the same way and market 02 should
#: check whether its own products carry it.
SHED_EPS_MWH = 1e-6

#: The two arms.  `honest` is truthful bidding and doubles as this market's
#: optimisation column, for the reason in the module docstring.  `constant` is
#: the instrument at the action-space upper endpoint.
DEFAULT_ARMS = ("honest", "constant")

#: What `--arms` may name.  `additive` is the uniform additive control group
#: and is **not** in the default: it needs `--add-m`, and a
#: default that named it would break every invocation written before it existed.
ARMS = DEFAULT_ARMS + ("additive", "vector")


def gate_stamp(spec):
    """The two tolerances `converged` was computed with, read off the env's
    own ``spec`` -- the value in force, never the module constant:
    a caller that passed ``mu_tol=`` or ``dual_res_tol=`` to
    `make_ancillary_env` would otherwise be stamped with values it did not
    run.  Goes into the run point, the curve scenario and every day product,
    so that a product says which gate produced its `converged`-derived
    fields (`unconv_periods`, `unconverged_frac`): products before 2026-09-17
    carry ``mu_tol`` only, and their `converged` was the mu half alone."""
    return dict(mu_tol=float(spec["mu_tol"]), dual_res_tol=float(spec["dual_res_tol"]))


def dual_ok(drs, dual_res_tol):
    """Day-level dual check on the per-period residuals: ``max <= tol``, the
    same boundary `run_rl_02.py` writes for its `dual_ok` (that driver had
    ``<=`` first; this market was aligned to it on 2026-09-17, so the two
    fields of one name read one way).  `env.converged` itself tests ``<``;
    the two differ only on a residual exactly at the tolerance."""
    return bool(max(drs) <= float(dual_res_tol))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--position",
                    default="tests/fixtures/day_ahead_position_29gb_T24_step1prime_seasons.npz")
    ap.add_argument("--out-dir", required=True)
    #: **A flag, not a change to `CASE`.**  `CASE = "29gb"` has produced results
    #: that are in effect and cited, so moving it would be "change the value and
    #: re-run everything it produced"; and `run_rl_03` binds it with
    #: a from-import, which freezes the value at import time whatever anyone
    #: rebinds later.  So the case a run uses is stated on the command
    #: line, and defaults to the constant.
    ap.add_argument("--case", default=CASE,
                    help="the network case.  Must agree with the position "
                         "fixture's own meta['case']; naming it is an assertion")
    ap.add_argument("--days", default="eval", choices=("eval", "train", "all"))
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
    #: **Off by default.  When on, the clearing deliberately runs a scenario
    #: different from the one the position fixture describes, to ask one
    #: counterfactual: with the same day-ahead commitment and the same demand,
    #: what happens if the real-time clearing's ramp limits are wider.**
    #: The fixture guard above still checks `--ramp-scale` as before, so
    #: "running the wrong scenario by accident" is still stopped; this flag is
    #: a named exception that leaves a trace: the product's `run_point` writes
    #: both `fixture_ramp_scale` and `clearing_ramp_scale`, plus a
    #: `counterfactual` sentence, so a reader of the product sees at a glance
    #: that this is not a baseline arm.
    #: Without this flag `ramp_eff is args.ramp_scale` and `run_point` is key
    #: for key what it was.
    ap.add_argument("--clearing-ramp-scale", type=float, default=None,
                    help="counterfactual: clear at this ramp scale while the "
                         "fixture check still uses --ramp-scale; default off, "
                         "and off it runs exactly what it always ran")
    ap.add_argument("--monitored-lines", default="all",
                    help="line limits the clearing carries: all (default, what "
                         "every archive was produced on), rated (the case's "
                         "published ratings; on case813nem 7 of 1 278 -- the "
                         "other 1 271 are 1e6 MW placeholders whose rows hold "
                         "mu above MU_TOL on every period, see "
                         "envs/ancillary/clearing.py), or a comma-separated "
                         "list of line indices")
    #: The Newton system's route and the low-rank route's block
    #: (2026-09-17), the real-time driver's three flags plus the route itself:
    #: `--monitored-lines rated` takes the low-rank route by the operator's
    #: own rule (every column free under the local-border arrowhead,
    #: `envs/ancillary/clearing.py`), `--kkt dense` keeps the dense
    #: factorisation on the same rows -- the ruler the low-rank route is
    #: measured against -- and the sizing flags shrink the block for a
    #: measurement.  Each value in effect is read back off the operator's spec
    #: and stamped; a flag that does not reach the operator refuses the run.
    ap.add_argument("--kkt", default="auto", choices=["auto", "dense", "lowrank"],
                    help="linear-algebra route of the clearing's Newton system: auto "
                         "(dense on every line, low-rank on a proper subset such as "
                         "rated), dense, or lowrank; stamped as kkt_route")
    ap.add_argument("--lowrank-free-units", type=int, default=None,
                    help="units in the low-rank route's pivoted block (default: every "
                         "unit)")
    ap.add_argument("--lowrank-free-shed", type=int, default=None,
                    help="diagonal columns (shed plus reserve shortfall) in that block "
                         "(default: all of them, n_buses + n_prod)")
    ap.add_argument("--lu-batching", default="auto", choices=["auto", "sequential", "batched", "arrow"],
                    help="how that block is solved under vmap; auto is the local-border "
                         "arrowhead on this market's single-period shape")
    ap.add_argument("--ipm-freeze", type=float, default=None,
                    help="arm ipm's merit gate once mu is below this value "
                         "(solvers/ipm.py freeze_mu; default off = the loop "
                         "as it always was). case813nem needs it: after "
                         "convergence the fixed 60 steps drift the duals "
                         "(2026-09-16); the value in force is stamped")
    ap.add_argument("--stop-tol", default=None,
                    help="mu_tol,dual_tol: stop ipm's Newton loop at the first iterate "
                         "under both, max_iter as the cap (solvers/ipm.py stop_tol; "
                         "default off = the fixed trip count). case813nem: 1e-9,1e-4, "
                         "the market's own gate (2026-09-17); stamped as ipm_stop_tol")
    ap.add_argument("--max-iter", type=int, default=None,
                    help="ipm Newton steps: the fixed count, or the cap under --stop-tol "
                         "(default: the market's MAX_ITER, 60). case813nem with --stop-tol "
                         "1e-9,1e-4: 200 (36 days: double-gate unconverged 10.1%% at 60, "
                         "3.2%% at 200); stamped as ipm_max_iter")
    ap.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS),
                    choices=list(ARMS))
    ap.add_argument("--alpha-file", default="",
                    help="npz with key `alpha_per_unit` (energy markup per unit) for "
                         "the vector arm; the two reserve columns stay at the truthful "
                         "baseline. Default off")
    #: The uniform ADDITIVE control group: every unit adds the same number of
    #: dollars to its own true marginal cost on the ENERGY column, and its two
    #: reserve columns stay at the truthful baseline -- the same one-dimension
    #: difference the constant arm makes, so the three arms differ in one
    #: coordinate and not in two.  The per-unit multiplier it needs is derived in
    #: `evaluation.additive_markup_alpha`, the one place that arithmetic is
    #: written for all three markets.
    ap.add_argument("--add-m", type=float, default=None,
                    help="$/MWh added to every unit's true marginal cost on the "
                         "energy column, for the additive arm")
    args = ap.parse_args()
    #: The per-unit markup arm (not run by default): the energy column takes each
    #: unit's value from `--alpha-file`, and the two reserve columns stay at the
    #: true-cost baseline -- like the additive arm, it moves one coordinate only.
    #: It answers "what does the market do if only some units change their
    #: offers"; its use is the same as the vector arm of `run_eval_02.py` (the
    #: same file format, the same key).  When this arm is not named this block
    #: does not run, and the products are key for key what they were.
    if "vector" in args.arms and not args.alpha_file:
        raise SystemExit("--arms vector needs --alpha-file")
    if args.alpha_file and "vector" not in args.arms:
        raise SystemExit("--alpha-file is the vector arm's treatment but --arms does not name vector")
    if "additive" in args.arms and args.add_m is None:
        raise SystemExit("--arms additive needs --add-m (in $/MWh)")
    # refuse rather than ignore, so a command line that reads as the additive
    # treatment cannot produce the other arms' numbers
    if args.add_m is not None and "additive" not in args.arms:
        raise SystemExit(f"--add-m is the additive arm's treatment but --arms is "
                         f"{args.arms}; it would be silently ignored")

    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    import jax.numpy as jnp

    import powermarketjax
    print("package:", powermarketjax.__file__, flush=True)

    import json
    from powermarketjax.case import load_case, scale_min_output
    from powermarketjax.envs.ancillary.env import (AncillaryParams,
                                                   make_ancillary_env)
    from powermarketjax.envs.ancillary import clearing as anc_clearing
    from powermarketjax.envs.real_time.demand import half_hourly_from_meta

    fx = np.load(args.position, allow_pickle=True)
    meta = json.loads(str(fx["meta"]))
    for key, given in (("cap_scale", args.cap_scale),
                       ("ramp_scale", args.ramp_scale),
                       ("voll", VOLL_IN_EFFECT)):
        got = meta.get(key)
        if got is None or abs(float(got) - given) > 1e-12:
            raise SystemExit(f"fixture meta {key}={got} disagrees with {given}; "
                             f"the position and the clearing would describe "
                             f"different scenarios")

    # the network and the demand must name one case.  `half_hourly_from_meta`
    # takes the realised series from `meta["case"]`, so a `--case` that disagreed
    # would serve one country's network another country's half-hours -- the
    # failure fixed on the forecast side on 2026-09-05 and on the realised
    # side on 2026-09-09, which reached neither of this market's two drivers because both
    # declared their case rather than reading it
    if str(meta.get("case")) != args.case:
        raise SystemExit(
            f"--case {args.case} but the position fixture was built for "
            f"{meta.get('case')!r}; the network and the demand would come from "
            f"two different cases")

    dates = [str(d) for d in meta["dates"]]
    # the case goes to the split as well as to the network: `case813nem`'s
    # window is 365 days, exactly GB's, so a length alone cannot say which
    # thirty-six days are held out
    ok, selected = check_split_against_report(dates, args.case)
    print(f"split matches report section 2.2: {ok}")
    if not ok:
        raise SystemExit(
            f"the split rule no longer reproduces the report's evaluation "
            f"days.\n  rule gives: {selected}\nResolve this deliberately.")
    ev, tr = split_days(len(dates), args.case)
    days = {"eval": ev, "train": tr, "all": sorted(ev + tr)}[args.days]

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
    case = scale_min_output(load_case(args.case), p_min_scale)
    n_units = int(case.n_units) if hasattr(case, "n_units") else len(case.unit_p_min)

    # hourly day-ahead quantities expanded to the half-hourly market period;
    # the demand itself is the native half-hourly series rather than an hourly
    # one repeated, because this market's period *is* half an hour and the
    # requirement is a fraction of it
    day_index = np.asarray(fx["day_index"], np.int64)
    hh, _days = half_hourly_from_meta(meta)
    demand = np.asarray(hh[day_index], np.float64).reshape(-1)
    u = np.repeat(np.asarray(fx["u"], np.float64), 2, axis=2).transpose(0, 2, 1)
    u = u.reshape(-1, u.shape[-1])
    q_da = np.repeat(np.asarray(fx["q_da"], np.float64), 2, axis=2).transpose(0, 2, 1)
    q_da = q_da.reshape(-1, q_da.shape[-1])
    lmp_da = np.repeat(np.asarray(fx["lmp_da"], np.float64), 2, axis=1)
    lmp_da = lmp_da.reshape(-1, lmp_da.shape[-1])
    assert u.shape[0] == demand.shape[0] == q_da.shape[0] == lmp_da.shape[0], (
        u.shape, demand.shape, q_da.shape, lmp_da.shape)
    print(f"periods {u.shape[0]} = {len(dates)} days x {T_DAY}; units "
          f"{u.shape[1]}", flush=True)

    #: The (volr, pi_scale) pair for the case this run names, read once so the operator
    #: and every stamp below come from the same two numbers.
    volr, pi_scale = volr_pi_scale(args.case)
    print(f"volr/pi_scale: case {args.case} clears at volr={volr:g} "
          f"pi_scale={pi_scale:g}", flush=True)
    #: `monitored_lines` parsed after `case` exists (`rated` is a property of the
    #: case's ratings, not of the string), and the products stamped off the
    #: operator's own spec rather than off the flag (the shape of
    #: `run_rl_02.py`).  This market has one clearing operator, so there is one
    #: stamp to read and nothing to cross-check it against; the read-back still
    #: refuses a run whose operator disagrees with its command line.
    monitored = parse_monitored_lines(args.monitored_lines, case)
    stop_tol = None if args.stop_tol is None else tuple(float(v) for v in args.stop_tol.split(","))
    if args.lowrank_free_shed is None and args.lowrank_free_units is None:
        lowrank_free = None                      # the operator's own sizing
    else:
        auto_units, auto_shed = anc_clearing.default_lowrank_free(case, monitored, len(THETA))
        lowrank_free = (auto_units if args.lowrank_free_units is None else args.lowrank_free_units,
                        auto_shed if args.lowrank_free_shed is None else args.lowrank_free_shed)
    ramp_eff = (args.ramp_scale if args.clearing_ramp_scale is None
                else float(args.clearing_ramp_scale))
    if args.clearing_ramp_scale is not None:
        print(f"COUNTERFACTUAL: clearing at ramp_scale={ramp_eff:g} while the "
              f"position fixture was built at {args.ramp_scale:g}. This is NOT "
              f"a baseline arm; both scales are stamped on every product.",
              flush=True)
    env = make_ancillary_env(case, THETA, volr, BETA, pi_scale,
                             n_segments=N_SEGMENTS,
                             cap_scale=args.cap_scale,
                             ramp_scale=ramp_eff,
                             period_hours=DELTA, kind="markup",
                             markup_max=MARKUP_MAX,
                             monitored_lines=monitored,
                             freeze_mu=args.ipm_freeze,
                             stop_tol=stop_tol,
                             kkt=args.kkt, lowrank_free=lowrank_free,
                             lu_batching=args.lu_batching,
                             **({} if args.max_iter is None else {"max_iter": int(args.max_iter)}))
    reset, step, _step_ar, spec = env
    gate = gate_stamp(spec)
    mon_stamp, kkt_route = effective_monitored_stamp(
        {"clearing": spec["clearing_spec"]}, requested=monitored)
    if (args.kkt == "dense" and kkt_route != "dense") or (args.kkt == "lowrank" and not kkt_route.startswith("lowrank")):
        raise SystemExit(f"--kkt {args.kkt} but the operator was built on the {kkt_route} route; "
                         "the flag is not reaching make_clearing")
    refuse_ineffective_lowrank_flags(
        (("clearing", spec["clearing_spec"]),),
        lowrank_free if lowrank_free is not None else (
            (0, 0) if kkt_route == "dense" else anc_clearing.default_lowrank_free(case, monitored, len(THETA))),
        args.lu_batching)
    lowrank_free_stamp = [int(v) for v in spec["clearing_spec"]["lowrank_free"]]
    lu_batching_stamp = str(spec["clearing_spec"]["lu_batching"])
    freeze_stamp = spec["clearing_spec"]["freeze_mu"]
    stop_stamp = spec["clearing_spec"]["stop_tol"]
    max_iter_stamp = int(spec["clearing_spec"]["max_iter"])
    if args.max_iter is not None and max_iter_stamp != int(args.max_iter):
        raise SystemExit(f"--max-iter {args.max_iter} but the operator was built with "
                         f"max_iter={max_iter_stamp}; the flag is not reaching make_clearing")
    if (None if stop_tol is None else tuple(stop_tol)) != (None if stop_stamp is None else tuple(stop_stamp)):
        raise SystemExit(f"--stop-tol {stop_tol} but the operator was built with "
                         f"stop_tol={stop_stamp}; the flag is not reaching make_clearing")
    if freeze_stamp != args.ipm_freeze:
        raise SystemExit(f"--ipm-freeze {args.ipm_freeze} but the operator was built with "
                         f"freeze_mu={freeze_stamp}; the flag is not reaching make_clearing")
    print(f"monitored lines in effect: "
          f"{'all' if mon_stamp is None else mon_stamp}; kkt route: {kkt_route} "
          f"(lowrank_free {lowrank_free_stamp}, lu_batching {lu_batching_stamp}); "
          f"LP rows m={int(spec['clearing_spec']['m'])}; ipm freeze_mu in effect: "
          f"{freeze_stamp}; ipm stop_tol in effect: {stop_stamp}; "
          f"ipm max_iter in effect: {max_iter_stamp}", flush=True)
    params = AncillaryParams(
        demand=jnp.asarray(demand), forecast=jnp.asarray(demand * 1.02),
        commitment=jnp.asarray(u), q_da=jnp.asarray(q_da),
        lmp_da=jnp.asarray(lmp_da),
        learner_mask=jnp.ones(u.shape[1], bool), episode_len=T_DAY)

    baseline = np.asarray(spec["baseline_action"])
    actions = {"honest": jnp.asarray(baseline)}
    add_info = None
    if "additive" in args.arms:
        alpha, add_info = additive_markup_alpha(case, N_SEGMENTS, args.add_m,
                                                MARKUP_MAX)
        add = np.array(baseline, copy=True)
        # energy column only; the two reserve columns stay at the truthful
        # baseline, the same single-coordinate difference the constant arm makes
        add[:, 0] = np.asarray(alpha, np.float64)
        actions["additive"] = jnp.asarray(add)
        print(f"arm=additive: every unit bids MC + {add_info['add_m']} $/MWh on "
              f"the energy column, i.e. alpha in [{add_info['alpha_min']:.4f}, "
              f"{add_info['alpha_max']:.4f}], mean {add_info['alpha_mean']:.4f}; "
              f"MC spans [{add_info['mc_min']:.4f}, {add_info['mc_max']:.4f}] "
              f"$/MWh; reserve columns unchanged at the truthful baseline",
              flush=True)
    vec_info = None
    if "vector" in args.arms:
        blob = np.load(args.alpha_file, allow_pickle=True)
        if "alpha_per_unit" not in blob.files:
            raise SystemExit(f"{args.alpha_file} has no `alpha_per_unit` key")
        a = np.asarray(blob["alpha_per_unit"], np.float64).ravel()
        if a.shape != (baseline.shape[0],):
            raise SystemExit(f"`alpha_per_unit` has shape {a.shape}, expected ({baseline.shape[0]},)")
        hi = float(np.asarray(spec["action_high"]))
        if a.min() < 1.0 - 1e-12 or a.max() > hi + 1e-12:
            raise SystemExit(f"`alpha_per_unit` spans [{a.min()}, {a.max()}], outside [1, {hi}]")
        if np.array_equal(a, baseline[:, 0]):
            raise SystemExit("`alpha_per_unit` equals the truthful energy column")
        v = np.array(baseline, copy=True)
        v[:, 0] = a
        actions["vector"] = jnp.asarray(v)
        vec_info = dict(file=str(args.alpha_file), n_units=int(a.size), min=float(a.min()),
                        max=float(a.max()), mean=float(a.mean()),
                        alpha_per_unit=[float(x) for x in a])
        print(f"arm=vector: {int((a != baseline[:, 0]).sum())}/{a.size} units move on the "
              f"energy column, alpha in [{a.min():.4f}, {a.max():.4f}]; reserve columns "
              f"unchanged at the truthful baseline", flush=True)
    if "constant" in args.arms:
        const = np.array(baseline, copy=True)
        # markup column to the action-space upper endpoint; the reserve columns
        # stay at the truthful baseline so the two arms differ in one dimension
        # `action_high` is a scalar: it bounds the markup column only.  The two
        # reserve columns are raw pre-softplus actions with no bound, and the
        # baseline holds them at -800, which softplus sends to exactly zero --
        # the truthful reserve offer.  So "the upper endpoint" is a statement
        # about one of the three columns.
        const[:, 0] = float(np.asarray(spec["action_high"]))
        actions["constant"] = jnp.asarray(const)

    step_j = jax.jit(step)
    day_of = lambda st: int(st.cursor) // T_DAY
    period_of = lambda st: int(st.cursor)
    assert int(spec["periods_per_day"]) == T_DAY, (spec["periods_per_day"], T_DAY)

    run_point = dict(
        cap_scale=args.cap_scale, ramp_scale=ramp_eff,
        p_min_scale=p_min_scale, window=meta.get("window"),
        #: Two keys that appear only in counterfactual runs.  Without the flag
        #: they are not in the product, and `ramp_scale` is the fixture's value,
        #: so every existing product is key for key what it was.
        **({} if args.clearing_ramp_scale is None else dict(
            fixture_ramp_scale=args.ramp_scale,
            clearing_ramp_scale=ramp_eff,
            counterfactual=("clearing ramp limits scaled away from the "
                            "fixture's; the day-ahead commitment is the one "
                            "built at the fixture's scale and is unchanged. "
                            "NOT a baseline arm."))),
        voll=VOLL_IN_EFFECT, volr=volr, beta=list(BETA), pi_scale=pi_scale,
        markup_max=MARKUP_MAX, **runtime_stamp(), **gate,
        market="03 ancillary services", case=args.case,
        position_fixture=str(args.position),
        #: the row set and route the operator was BUILT with, read off its
        #: spec; and the ones the position was produced on, recorded beside
        #: them and not refused (the position is this market's input, not
        #: something it recomputes).  `None` on an old fixture is an absent
        #: field, not "the same".
        monitored_lines=mon_stamp, kkt_route=kkt_route,
        lowrank_free=lowrank_free_stamp, lu_batching=lu_batching_stamp,
        position_monitored_lines=meta.get("monitored_lines"),
        position_kkt_route=meta.get("kkt_route"),
        ipm_freeze_mu=freeze_stamp,
        ipm_stop_tol=stop_stamp,
        ipm_max_iter=max_iter_stamp,
        honest_is_also_optimisation=(
            "this market clears one period at a time with an exogenous "
            "commitment, so truthful offers already give the cost-minimising "
            "dispatch; the optimisation column and the honest column are the "
            "same number and the two are not independent evidence"),
        profit_sign_note=(
            "agent profit is large and negative on both arms, and that is the "
            "model rather than a defect: reward is settlement profit, which "
            "carries no-load and start-up costs, while a marginal-cost LMP "
            "recovers neither and this market pays no make-whole uplift. "
            "Measured over these twelve days the no-load cost alone is "
            "1.9824e+08 $, which is 136% of the honest arm's 1.4613e+08 $ "
            "profit shortfall -- so the shortfall is fully accounted for "
            "before start-up costs are counted at all. Do not read it as "
            "providers being underpaid for what the LMP is supposed to pay"),
        constant_arm_note=(
            "markup 2.0 is MARKUP_MAX, the upper endpoint of the action space "
            "and not a searched optimum; no search for the best constant "
            "markup was run, so this is a lower bound on what a fixed markup "
            "achieves"))

    for arm in args.arms:
        action = actions[arm]
        # per arm rather than once for the invocation: `--arms` takes several at
        # a time here, and a shared stamp would put `additive_m` on the honest
        # and constant products too
        rp = dict(run_point,
                  additive_m=(None if arm != "additive"
                              else float(add_info["add_m"])),
                  alpha_profile=(vec_info if arm == "vector" else None if arm != "additive" else dict(
                      file=None, n_units=add_info["n_units"],
                      min=add_info["alpha_min"], max=add_info["alpha_max"],
                      mean=add_info["alpha_mean"],
                      alpha_per_unit=add_info["alpha_per_unit"],
                      mc_per_unit=add_info["mc_per_unit"],
                      basis=add_info["basis"])))
        print(f"\n=== arm {arm} ===", flush=True)
        rows = []
        for day in days:
            # The day's *first* period, not a key that lands
            # somewhere inside the day.  Before this, 265 of these 576 half
            # hours belonged to the next day (all of them training days), so
            # the open-loop arms and the learning arm were not scored on the
            # same periods and no displacement between them could be reported.
            key, state = open_day_start(spec["reset_on_day"], params, day,
                                        day_of, period_of, T_DAY)
            prod, shed, prof, mus, seps = 0.0, [], None, [], []
            # Both prices, per period.  Same two arrays the learned arm's
            # driver writes, so the open-loop and learned products can be
            # read for price level and distribution with one reader rather
            # than one arm carrying a field the other has none of.
            lmps, res_prices = [], []
            volr_cost = 0.0
            short_mwh, noload, startup = 0.0, 0.0, 0.0
            # Same two lists the learned arm's driver keeps, so both arms'
            # products carry the same field rather than one of them requiring a
            # reader to notice that the other has none.  On these arms the split
            # is degenerate (measured 2026-08-28: 0 of 576 periods above
            # `mu_tol`), and a degenerate value recorded is not the same thing as
            # a value absent -- the learned arm leaves 209 to 217 of 576 above
            # it, and the displacement between the two is read across that gap.
            step_reward, step_converged = [], []
            drs = []
            for _ in range(T_DAY):
                _o, state, reward, _c, _dn, info = step_j(key, state, action,
                                                          params)
                prod += float(np.sum(np.asarray(info["cost"], np.float64)))
                shed.append(float(np.asarray(info["shed_mwh"])))
                r = np.asarray(reward, np.float64)
                prof = r if prof is None else prof + r
                step_reward.append(r.copy())
                step_converged.append(bool(np.asarray(info["converged"])))
                mus.append(float(info["mu"]))
                drs.append(float(info["dual_residual"]))
                # the fourth term of this market's objective, which
                # `system_cost` requires explicitly rather than defaulting to
                # zero: energy and reserve shortfall are priced separately here
                # and a default would have silently valued the second at nothing
                volr_cost += float(np.asarray(info["volr_cost"]))
                # the two cost components the negative-profit claim rests on,
                # plus the shortfall in MWh beside the money it was priced at:
                # `volr_cost` alone cannot say whether a day was expensive
                # because the requirement went unmet or because VOLR is large
                short_mwh += DELTA * float(np.sum(np.asarray(
                    info["reserve_shortfall"])))
                noload += float(np.sum(np.asarray(info["cost_noload"])))
                startup += float(np.sum(np.asarray(info["cost_startup"])))
                seps.append(float(np.min(np.asarray(
                    info["offer_separation_all"]))))
                lmps.append(np.asarray(info["lmp"], np.float64).copy())
                res_prices.append(np.asarray(info["reserve_price"],
                                             np.float64).copy())
            sc = system_cost(prod, shed, VOLL_IN_EFFECT, volr_cost)
            split = converged_reward_split(np.stack(step_reward),
                                           np.asarray(step_converged))
            write_day(args.out_dir, arm, day, dates[day], system_cost_value=sc,
                      agent_profit=prof, shed_mwh=np.asarray(shed),
                      production_cost=prod, run_point=rp,
                      arrays=dict(lmp=np.stack(lmps),
                                  reserve_price=np.stack(res_prices)),
                      extra=dict(mu_max=max(mus),
                                 #: the mu half ALONE (mu > 1e-9, the report
                                 #: floor this field has always used), kept so
                                 #: that products before and after the double
                                 #: gate (2026-09-17) read the same thing here;
                                 #: the double-gated count is `unconv_periods`
                                 unconverged=int(sum(m > 1e-9 for m in mus)),
                                 #: the dual half of `converged`, beside `mu`
                                 #: (same fields as `run_rl_02.py`); the
                                 #: products before this carried `mu_max` only
                                 dual_residual_max=max(drs),
                                 dual_ok=dual_ok(drs, gate["dual_res_tol"]),
                                 dual_res_tol=gate["dual_res_tol"],
                                 shed_cells=int(sum(s > SHED_EPS_MWH for s in shed)),
                                 shed_total_mwh=float(np.sum(shed)),
                                 reserve_shortfall_mwh=short_mwh,
                                 reserve_shortfall_cost=volr_cost,
                                 cost_noload=noload, cost_startup=startup,
                                 offer_separation_min=min(seps),
                                 unconv_periods=int(sum(not c for c in
                                                        step_converged)),
                                 unconv_reward_share=split["share_of_abs_reward"],
                                 unconv_max_abs_ratio=split["max_abs_ratio"],
                                 unconv_mean_reward=split["mean_unconverged"],
                                 conv_mean_reward=split["mean_converged"]))
            rows.append((day, sc, float(prof.sum()),
                         int(sum(s > SHED_EPS_MWH for s in shed)), max(mus), min(seps)))
            print(f"  day {day:3d} {dates[day]}  system_cost {sc:14.4e}  "
                  f"profit {float(prof.sum()):14.4e}  shed_cells "
                  f"{int(sum(s > SHED_EPS_MWH for s in shed)):2d}  mu_max {max(mus):.2e}  "
                  f"sep_min {min(seps):.2e}", flush=True)
        print(f"  {len(rows)} days -> {args.out_dir}")
        print(f"  total system cost {sum(r[1] for r in rows):.6e}   "
              f"total profit {sum(r[2] for r in rows):.6e}   "
              f"shed cells {sum(r[3] for r in rows)}   "
              f"unconverged periods "
              f"{sum(1 for r in rows if r[4] > 1e-9)}")


if __name__ == "__main__":
    main()
