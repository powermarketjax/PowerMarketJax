"""Market 03's learning arm.

Same product format as `run_eval_03.py`, so the learned arm's per-day numbers
land in one table beside the honest and constant arms with nothing to reconcile
afterwards.  What this adds is the training loop and the learning curve.

**Training draws only from the 48 training days**, by restricting the fixture
rather than by teaching `reset` about day sets (`evaluation.subset_position`).
Evaluation builds a second environment on the 12 held-out days.  Both day sets
come from the one shared split, so this driver cannot disagree with the other
arms about which days it never trained on.

**Nothing here derives the action layout or rebuilds the policy network.**
`action_layout` and `make_greedy_action` are imported.  This market would not
have caught a mistake in either: its action is `(n_units, 1 + n_prod)`, two
dimensional, so the `bounds[0].shape[-1]` ambiguity that produced a (66, 66)
action in the one-dimensional markets is invisible here.  An invisible mistake
is the reason to use the shared name, not a reason it does not matter.

**`reserve_columns=n_prod` is passed to `bounds_for`.**  The reserve columns are
pre-softplus actions with their own box, which this market publishes on the spec
and `bounds_for` reads; getting the count wrong is silent, because the energy
markup's bounds simply spread over the reserve columns and the policy trains
against a box that was never part of the market.

**Four quantities are reported, and the fourth is not a diagnostic here.**
Wall clock per iteration with the batch size beside it, the whole `reward_mean`
series, the paired metrics of the final policy on the twelve evaluation days,
and the whole `unconverged_frac` series.  `MU_TOL` marks the near-critical
region of this market's clearing, so how often training pushes the clearing into
that region is a result rather than a health check: a curve that improves while
`unconverged_frac` climbs is two findings, not one.  Since 2026-09-17 the flag
behind that series is the two-part gate (`mu < MU_TOL` and `dual_residual <
DUAL_RES_TOL`), so a curve from before it counts the mu half only; the run
point and the curve scenario stamp both tolerances (`gate_stamp`) so the two
kinds of archive can be told apart.  The day products' `unconverged` field
stays the mu-only count for the same reason.

**Offer separation is taken over the committed set**, which is what §17 asks
for.  Taking it over all 66 units is a denominator error this line has already
retracted once: uncommitted units all sit at the same offer, so including them
reports a separation that is systematically too small and belongs to no market.

CPU or GPU.  Run point is stamped into every product.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import (refuse_ineffective_lowrank_flags,
                        _START_DIRTY, check_split_against_report, commit_hash,
                        converged_reward_split, count_cells,
                        effective_monitored_stamp, open_day_start,
                        parse_monitored_lines, runtime_stamp, split_days,
                        subset_position, system_cost, write_day)
from curve_jsonl import CurveLog, repo_relative
from hyperparams import PROVENANCE, SAC_PROVENANCE, SAC_SHARED, SHARED

#: **The scenario constants are imported from `run_eval_03`, not copied.**  That
#: rule dates from 2026-08-26, the change that moved this market's beta_1 from
#: 0.020 to 0.050: "scenario constants are always imported from run_eval_03,
#: never copied by value".  This file was brought back from a feature branch a
#: day *before* that rule and carried a copy of the old pair, so
#: from then until this line existed the learned arm trained at
#: `(0.020, 0.050)` while the honest and constant arms it is compared against
#: evaluated at `(0.050, 0.050)` -- two scenarios in one table, and nothing in
#: either product said so.  Importing rather than restating is what makes the
#: pair unable to diverge again; restating the new value would only have moved
#: the next divergence to the next change.
#:
#: `run_eval_03` has no import-time side effect beyond its own `sys.path`
#: insert, and the insert above is what makes it importable when this file is
#: loaded by path (another ancillary tool does exactly that).
#: `volr_pi_scale` rather than `VOLR` / `PI_SCALE`: those two names no longer
#: exist, precisely because a from-import of them froze the British cap onto
#: every case.  The function is called with `args.case`.
from run_eval_03 import (CAP_SCALE, CASE, DELTA, MARKUP_MAX,
                         P_MIN_SCALE_IMPLIED_PRIOR, RAMP_SCALE, T_DAY, THETA,
                         VOLL_IN_EFFECT, dual_ok, gate_stamp, volr_pi_scale)
from run_eval_03 import BETA

ARM = "ippo"

#: The one pair of providers whose static columns are bitwise identical and
#: which the adopted commitment runs together.  It is accepted that a shared
#: policy therefore gives them identical offers; measured, they are the only
#: such pair.  Recorded per day is how often that tie is at the margin, because
#: a tie that never prices constrains nothing.
TIED_PAIR = (3, 8)

#: The clearing widens every zero-width box to `OFF_EPS`, so a bus with no load
#: returns ~2.3e-21 MWh rather than zero.  Passed explicitly because
#: `count_cells` refuses a default: this floor belongs to this market's clearing,
#: and market 02's counting is correct only because its environment floors the
#: quantity before the skeleton ever sees it.
SHED_FLOOR = 1e-6


def build_params(fx_dict, case, jnp):
    """Flatten a day-indexed position fixture into the half-hourly series.

    The realised series comes from the fixture's own record rather than from a
    named loader, so that pointing this driver at a `73rts` or `813nem` position
    does not serve British half-hours to an American or Australian network.  That
    is the 2026-09-09 change, which reached nine drivers and not this one: the check
    it left behind fires on a function that calls `demand_from_meta` *and* a
    single-case realised loader, and this function called only the latter.
    """
    from powermarketjax.envs.real_time.demand import half_hourly_from_meta
    from powermarketjax.envs.ancillary.env import AncillaryParams
    day_index = np.asarray(fx_dict["day_index"], np.int64)
    hh, _days = half_hourly_from_meta(fx_dict["meta"])
    demand = np.asarray(hh[day_index], np.float64).reshape(-1)
    u = np.repeat(np.asarray(fx_dict["u"], np.float64), 2, axis=2).transpose(0, 2, 1)
    u = u.reshape(-1, u.shape[-1])
    q_da = np.repeat(np.asarray(fx_dict["q_da"], np.float64), 2, axis=2).transpose(0, 2, 1)
    q_da = q_da.reshape(-1, q_da.shape[-1])
    lmp_da = np.repeat(np.asarray(fx_dict["lmp_da"], np.float64), 2, axis=1)
    lmp_da = lmp_da.reshape(-1, lmp_da.shape[-1])
    assert u.shape[0] == demand.shape[0] == q_da.shape[0] == lmp_da.shape[0]
    return AncillaryParams(
        demand=jnp.asarray(demand), forecast=jnp.asarray(demand * 1.02),
        commitment=jnp.asarray(u), q_da=jnp.asarray(q_da),
        lmp_da=jnp.asarray(lmp_da),
        learner_mask=jnp.ones(u.shape[1], bool), episode_len=T_DAY)


def _flatten_params(node):
    """`{"a": {"b": arr}}` -> `{"a/b": arr}`, the on-disk parameter layout."""
    flat = {}

    def walk(prefix, sub):
        if isinstance(sub, dict):
            for k, v in sub.items():
                walk(f"{prefix}/{k}" if prefix else str(k), v)
        else:
            flat[prefix] = np.asarray(sub)

    walk("", node)
    return flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--position",
                    default="tests/fixtures/day_ahead_position_29gb_T24_step1prime_seasons.npz")
    ap.add_argument("--out-dir", required=True)
    #: **A flag, not a change to `CASE`.**  Two separate reasons, and either
    #: alone would be enough.  `CASE = "29gb"` has produced results that are in
    #: effect and cited, so moving it is "change the value *and* re-run what it
    #: produced".  And the `from run_eval_03 import (... CASE ...)`
    #: above binds the value at import time, so rebinding it in this process --
    #: the obvious alternative -- does nothing at all.  The default is
    #: that constant, so a command line naming no case runs what it always ran.
    ap.add_argument("--case", default=CASE,
                    help="the network case.  Must agree with the position "
                         "fixture's own meta['case']; naming it is an assertion")
    #: Which learner; `ippo` is the default and the path
    #: every archive on disk was produced on.  Same wiring as `run_rl_01.py`.
    #: This driver's archive readers (`--init-params`, `--eval-only`) know the
    #: IPPO tree only and are refused for `sac` rather than misread.
    ap.add_argument("--algo", choices=("ippo", "sac"), default="ippo",
                    help="learner: ippo (default, the existing path) or sac "
                         "(off-policy). Stamped as `algo` and `arm` in every product.")
    ap.add_argument("--iterations", type=int, default=200)
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
    #: **Off by default.**  When on, samples this market judges `usable=False`
    #: (the mu and dual double gate) do not enter the gradient -- their values
    #: are not changed, since the reward is never altered -- and the number
    #: masked goes into the curve's `masked_samples`.
    #: Basis: measured 2026-09-17 on 73rts market 03 (six seeds), iterations
    #: with `unconv_max_abs_ratio` > 10 were 72-86 of 200 (36-43%), the largest
    #: single sample 638 to 3071 times the converged mean and 50 to 80% of that
    #: iteration's reward; on 813nem market 03 seed 0, up to iteration 84, it
    #: was 18 of 85, at most 126 times, 36.5%.
    ap.add_argument("--mask-unusable", action="store_true",
                    help="exclude samples this market marks unusable (the mu and "
                         "dual gate) from the loss (default: off, so existing "
                         "columns are unchanged)")
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--curve-out", default="")
    ap.add_argument("--weight-decay", type=float, default=None,
                    help="override SHARED.weight_decay for this run only.  The "
                         "shared value stays zero so that no other line's "
                         "unflagged run changes underneath it")
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="write the parameters every N iterations, in addition "
                         "to the final write.  Zero, the default, keeps the "
                         "old behaviour of writing only at the end.  This "
                         "exists because the policy collapse measured earlier "
                         "on this market is invisible "
                         "in the reward curve, so asking when it happens needs "
                         "the parameters themselves rather than any scalar "
                         "recorded per iteration")
    ap.add_argument("--per-agent-params", action="store_true",
                    help="give every provider its own copy of the policy and "
                         "value network. Archives written under this flag are "
                         "NOT interchangeable with shared ones: they have the "
                         "same number of leaves and differ only in a leading "
                         "axis")
    #: How many sequential pieces the environment step of the rollout is taken
    #: in.  A keyword of `make_ippo` and NOT a field of `IPPOConfig`, like
    #: `--per-agent-params` and for the same reason; unlike that flag it is not
    #: a hyperparameter at all -- the same `n_envs x horizon` batch is laid onto
    #: the device in pieces -- so a run under it is NOT off-shared.  It exists
    #: for `case813nem`: 2.842 GiB of device memory per environment, so the
    #: shared `n_envs=64` needs 182 GiB and one 24 GB card holds seven
    #: Defaults to 1, the single
    #: `vmap` every archive was produced on, and is stamped into the curve meta
    #: and `run_point` below so a product says how it was laid out.  Must
    #: divide `n_envs`; what it changes numerically has been measured
    #: separately.
    ap.add_argument("--env-chunks", type=int, default=1,
                    help="step the n_envs environments in this many sequential "
                         "pieces (lax.map over vmap) to cap peak device memory; "
                         "1 = the single vmap every archive was produced on. "
                         "Must divide n_envs. Not a hyperparameter: the batch "
                         "is unchanged, only how it sits on the device")
    ap.add_argument("--init-params", default="",
                    help="continue from a params npz this driver wrote instead "
                         "of from a fresh initialisation. The archive's OWN "
                         "obs_mean/obs_std are used, never a fresh fit: a refit "
                         "reconstructs a different policy from the same "
                         "weights. The archive holds no optimiser state, so "
                         "Adam restarts from zero moments -- a real "
                         "discontinuity. Same flag name and same meaning as "
                         "`run_rl_01.py`, `run_rl_02.py` and "
                         "`run_rl_01_boundary.py`, so one reader reads all four")
    ap.add_argument("--eval-only", default="",
                    help="path to a saved params npz: skip training and evaluate "
                         "that policy.  Exists because asking a new question of "
                         "a trained policy otherwise costs a full retrain, and "
                         "this market trains for hours")
    ap.add_argument("--eval-days", default="eval", choices=("eval", "train"),
                    help="which day set the final policy is evaluated on; the "
                         "training window is needed to say what proportion of "
                         "periods the agent actually saw priced at the cap, "
                         "which the held-out window cannot report")
    args = ap.parse_args()
    #: **The two archive flags cannot be given together**, as in `run_rl_02.py`.
    #: `_archive_path = args.init_params or args.eval_only` below loads the
    #: first, while the `--eval-only` branch prints the second, so a run given
    #: both evaluated A and reported B.  Refused here, before any environment
    #: is built.
    if args.init_params and args.eval_only:
        raise SystemExit(
            "--init-params and --eval-only both name an archive; give one. "
            "`--eval-only X` is `--init-params X --iterations 0` with the "
            "intent stamped, and two paths would silently pick one")

    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    import jax.numpy as jnp

    import powermarketjax
    print("package:", powermarketjax.__file__, flush=True)
    print("devices:", jax.devices(), flush=True)

    from powermarketjax.case import load_case, scale_min_output
    from powermarketjax.envs.ancillary.env import make_ancillary_env                # noqa: E402
    from powermarketjax.envs.ancillary import clearing as anc_clearing
    import dataclasses
    from powermarketjax.learning.ippo import (make_greedy_action, make_ippo,
                                              observation_statistics)
    from powermarketjax.learning.policy import bounds_for
    from powermarketjax.learning.sac import (make_sac, make_sac_greedy_action,
                                             reward_statistics)
    arm = args.algo
    if args.algo != "ippo" and (args.init_params or args.eval_only):
        raise SystemExit("--init-params / --eval-only read the IPPO parameter "
                         "tree (`params/...` keys); a SAC archive holds actor, "
                         "critics, targets and log_alpha and is not read here")
    if args.algo != "ippo" and args.weight_decay is not None:
        raise SystemExit("--weight-decay belongs to the IPPO optimiser chain; "
                         "SACConfig has no such field")
    if args.algo != "ippo" and args.mask_unusable:
        # Accepted-and-dropped again, and here it would go one worse
        # than the `--env-chunks` case below: `make_sac` does not take
        # `valid_key`, so nothing would be masked, **and the curve would still
        # stamp `mask_unusable: true`** next to `masked_samples: 0.0` -- the one
        # reading of that pair the comment at the stamp says means "no sample
        # was excluded".  A stamp that lies is worse than a flag that is ignored.
        raise SystemExit("--mask-unusable is wired through make_ippo's "
                         "`valid_key` only; the SAC learner does not take it, "
                         "so it is refused rather than silently ignored while "
                         "the curve stamps mask_unusable: true")

    # Rebind this module's `SHARED` rather than mutating the shared object:
    # `hyperparams.SHARED` is frozen, and other tools do `from hyperparams
    # import SHARED` at import time, so a mutation would reach them while a
    # rebind here reaches only this driver.  The effective value is printed
    # from the object actually handed to `make_ippo`, which is the only form
    # that shows it arrived rather than that it was sent.
    global SHARED
    if args.algo == "sac":
        # the same name carries the SAC configuration from here on, so every
        # `vars(SHARED)` stamp below records the configuration that ran
        SHARED = SAC_SHARED
    if args.weight_decay is not None:
        SHARED = dataclasses.replace(SHARED, weight_decay=args.weight_decay)
    if args.algo == "ippo":
        print(f"weight_decay in force: {SHARED.weight_decay:g} "
              f"({'optax.adamw' if SHARED.weight_decay else 'optax.adam'})",
              flush=True)

    fx = np.load(args.position, allow_pickle=True)
    meta = json.loads(str(fx["meta"]))
    for key, given in (("cap_scale", args.cap_scale),
                       ("ramp_scale", args.ramp_scale),
                       ("voll", VOLL_IN_EFFECT)):
        got = meta.get(key)
        if got is None or abs(float(got) - given) > 1e-12:
            raise SystemExit(f"fixture meta {key}={got} disagrees with {given}")
    # the network and the demand must name one case: `build_params` takes the
    # realised half-hours from `meta["case"]`, so a `--case` that disagreed would
    # serve one country's network another country's demand
    if str(meta.get("case")) != args.case:
        raise SystemExit(
            f"--case {args.case} but the position fixture was built for "
            f"{meta.get('case')!r}; the network and the demand would come from "
            f"two different cases")

    dates = [str(d) for d in meta["dates"]]
    # the case goes to the split as well as to the network; see run_eval_03
    ok, _sel = check_split_against_report(dates, args.case)
    print(f"split matches report section 2.2: {ok}", flush=True)
    if not ok:
        raise SystemExit("the split rule no longer reproduces the report's "
                         "evaluation days; resolve deliberately")
    ev_days, tr_days = split_days(len(dates), args.case)

    pos = {k: fx[k] for k in fx.files if k != "meta"}
    pos["meta"] = meta
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
    #: The (volr, pi_scale) pair for the case this run names, read once so the operator,
    #: the action box printed below and every stamp come from the same numbers.
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
    env = make_ancillary_env(case, THETA, volr, BETA, pi_scale, n_segments=1,
                             cap_scale=args.cap_scale,
                             ramp_scale=args.ramp_scale,
                             period_hours=DELTA, kind="markup",
                             markup_max=MARKUP_MAX,
                             monitored_lines=monitored,
                             freeze_mu=args.ipm_freeze,
                             stop_tol=stop_tol,
                             kkt=args.kkt, lowrank_free=lowrank_free,
                             lu_batching=args.lu_batching,
                             **({} if args.max_iter is None else {"max_iter": int(args.max_iter)}))
    spec = env[3]
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
    train_params = build_params(subset_position(pos, tr_days), case, jnp)
    eval_params = build_params(subset_position(pos, ev_days), case, jnp)
    print(f"train days {len(tr_days)}   eval days {len(ev_days)}   "
          f"disjoint {not set(tr_days) & set(ev_days)}", flush=True)

    # The reserve columns' box is this market's, not this driver's, and not the
    # algorithm's: `envs/ancillary/action.py` derives it from `pi_scale` and
    # `volr` and publishes it on the spec, and both algorithms read the same
    # one.  Until 2026-09-05 it was computed here and
    # passed only on the SAC branch, which made the same market two action
    # spaces depending on which learner was pointed at it.
    #
    # Printed from the spec rather than from anything computed here, so the
    # line is evidence that the box in force is the market's.
    print(f"reserve columns bounded to "
          f"({spec['reserve_low']!r}, {spec['reserve_high']!r}) in pre-softplus "
          f"units, from the market's pi_scale={spec['pi_scale']} and "
          f"volr={spec['volr']}; the same box for {args.algo}", flush=True)
    bounds = bounds_for(spec, reserve_columns=int(spec["n_prod"]))
    key = jax.random.PRNGKey(args.seed)
    key, k_stat, k_init = jax.random.split(key, 3)
    #: The archive, if either flag names one, probed BEFORE the statistics are
    #: fitted.  `observation_statistics` is 64 environments times a full horizon
    #: of clearings, and every archive this driver writes stores its own
    #: `obs_mean` / `obs_std`, so fitting them and then replacing them is pure
    #: waste: measured at 27 minutes on eight pinned cores before a single
    #: product was written.  Probing first also makes the
    #: provenance stronger -- the statistics in force are the ones the weights
    #: were fitted against, never ones re-derived from a seed.
    _archive_path = args.init_params or args.eval_only
    _archive = (np.load(_archive_path, allow_pickle=True) if _archive_path
                else None)
    _stored_stats = (_archive is not None and "obs_mean" in _archive.files)
    if _stored_stats:
        obs_mean = jnp.asarray(_archive["obs_mean"])
        obs_std = jnp.asarray(_archive["obs_std"])
        print(f"observation statistics taken from {repo_relative(_archive_path)}"
              f"; not refitted", flush=True)
    else:
        obs_mean, obs_std = observation_statistics(env, train_params, k_stat,
                                                   SHARED.n_envs, SHARED.horizon,
                                                   env_chunks=args.env_chunks)
        if _archive is not None:
            # Pre-statistics archive.  Refitting is the only option and it is
            # NOT bit-reproducible on GPU (about 2200 ulps, measured 2026-08-21),
            # so say so rather than let a silent refit look like the stored pair.
            print(f"{repo_relative(_archive_path)} stores no observation "
                  f"statistics, so they are refitted from seed {args.seed}; "
                  f"bit-identical across processes on CPU only", flush=True)
    # `reserve_price` is carried out of the rollout because the share of periods
    # whose price sits at the cap is a limit on what there was to learn, and it
    # only exists while it happens: running the final policy over the training
    # days answers what the FINAL policy sees there, and the policy changed
    # throughout.  The key is named here rather than in the shared module for
    # the reason its docstring gives.
    # `requirement` is carried out of the rollout for one reason: it is the only
    # form in which this driver can report the requirement fractions the
    # environment ENFORCED rather than the ones it sent.  `spec` publishes
    # `volr`, `pi_scale`, `mu_tol` and `period_hours` but not `beta`, and
    # `make_requirement` closes over the pair, so `info["requirement"]` --
    # `beta * sum(forecast)` for the period that was actually cleared -- is what
    # comes back.  Dividing by that period's own forecast recovers the pair.
    # This exists because the fractions in this file were a stale copy of
    # `run_eval_03`'s for a day and nothing in any log would have shown it: the
    # first explanation of "this parameter has no effect" is "I did not pass
    # it", and only a value read back from the callee can rule that out.
    if args.per_agent_params:
        print(f"PER-AGENT PARAMETERS: {int(spec['n_agents'])} independent copies "
              f"of the network. Products of this run are stamped "
              f"`per_agent_params: true` and must not be compared leaf-for-leaf "
              f"against a shared archive.", flush=True)
    if args.algo == "ippo":
        init, iterate = make_ippo(env, bounds, SHARED, obs_mean, obs_std,
                                  extra_info_keys=("reserve_price", "requirement"),
                                  per_agent_params=args.per_agent_params,
                                  env_chunks=args.env_chunks,
                                  #: With `valid_key=None` every expression of the
                                  #: learner is bitwise what it was before this flag
                                  #: existed, so off by default = existing columns
                                  #: unchanged.
                                  valid_key=("usable" if args.mask_unusable else None))
    else:
        if args.env_chunks != 1:
            # Accepted-and-dropped is the failure this refusal exists for:
            # `make_sac` has its own rollout and does not take the
            # keyword, so a value here would change nothing and read as if it had.
            raise SystemExit(
                f"--env-chunks={args.env_chunks} is wired through make_ippo "
                f"only; the SAC learner does not take it, so it is refused "
                f"rather than silently ignored")
        # the critic's reward scale, fitted once from the truthful rollout and
        # frozen like `obs_mean` / `obs_std`; `fold_in` so the IPPO path's key
        # stream is untouched by a branch it never takes
        scale = float(reward_statistics(env, train_params,
                                        jax.random.fold_in(k_stat, 1),
                                        SHARED.n_envs, SHARED.horizon))
        SHARED = dataclasses.replace(SHARED, reward_scale=scale)
        print(f"SAC reward_scale = {scale:.6e} (pooled std of the per-agent "
              f"reward under the truthful action over a {SHARED.n_envs} x "
              f"{SHARED.horizon} sample)", flush=True)
        init, iterate = make_sac(env, bounds, SHARED, obs_mean, obs_std,
                                 extra_info_keys=("reserve_price", "requirement"),
                                 per_agent_params=args.per_agent_params)
    params, tx, opt_state, env_state, env_obs = init(k_init, train_params)
    step_iter = jax.jit(iterate, static_argnums=(1,))

    def _params_from(blob, where):
        """The `params/` subtree of an archive this driver wrote.

        Whitelisting the prefix rather than skipping known-bad keys: a name list
        goes stale the moment the writer gains a field, and a dtype filter
        admits the numeric `iteration` that `--checkpoint-every` writes.  Both
        forms were tried and both broke; the prefix is the real criterion
        because `_flatten_params` is what puts it there.
        """
        node = {}
        for k in blob.files:
            if not k.startswith("params/"):
                continue
            cur, parts = node, k.split("/")
            for q in parts[:-1]:
                cur = cur.setdefault(q, {})
            cur[parts[-1]] = jnp.asarray(blob[k])
        if not node:
            raise SystemExit(f"{where} carries no `params/` keys; it holds "
                             f"{sorted(blob.files)}")
        return node

    if args.init_params:
        # Continue from a saved policy.  The optimiser state is NOT in the
        # archive, so Adam restarts from zero moments -- a real discontinuity,
        # stamped `optimizer_state_restored: false`, and one that showed up
        # in market 01 as a suppressed profit spread (RTM item 93).
        params = _params_from(_archive, args.init_params)
        print(f"init-params: {repo_relative(args.init_params)}; optimiser state "
              f"restarts from zero moments", flush=True)

    if args.eval_only:
        # The archive was already loaded above, to decide whether the
        # observation statistics needed fitting; reloading it here would be a
        # second read of the same file and a second chance for the two to
        # disagree about which archive is in force.
        params = _params_from(_archive, args.eval_only)
        print(f"eval-only: params from {repo_relative(args.eval_only)}"
              f"{'' if _stored_stats else ' (statistics refitted, see above)'}",
              flush=True)
        args.iterations = 0

    def _save_params(node, path, extra=None):
        """One implementation of the parameter write, used by both callers.

        The checkpoints and the final file have to be byte-compatible: the
        probe that reads them cannot tell which kind it was handed, and a
        checkpoint missing `obs_mean` would be silently unusable rather than
        an error, because the probe would fall back to recomputing the
        statistics and compare a policy against a reference it was not fitted
        on.
        """
        f = _flatten_params(node)
        f["obs_mean"] = np.asarray(obs_mean)
        f["obs_std"] = np.asarray(obs_std)
        # Which key convention this file uses, stored in the file itself.
        # The docstring that explains a layout does not travel with the .npz,
        # and the next reader has only the .npz: market 01 writes a pytree
        # `treedef` with `p0 p1 p2 ...`, this driver writes flat `params/...`
        # keys, and nothing in either file said which.
        f["layout"] = np.array("flax_flat_v1")
        f["hyperparams"] = np.array(json.dumps(
            {k: (list(v) if isinstance(v, (tuple, list)) else v)
             for k, v in vars(SHARED).items()
             if isinstance(v, (int, float, str, bool, tuple, list))}))
        # `hyperparams` is `vars(SHARED)`, i.e. the **learning** hyperparameters
        # only.  The scenario constants live as module-level names, so until
        # 2026-08-25 no checkpoint this project ever wrote could answer "which
        # scenario produced me": the 16 earlier market-03 checkpoints carry `layout` and
        # `hyperparams` and nothing else, and `hyperparams` has the 14 learning
        # keys with not one of `case`/`beta`/`volr`/`reg_coef`/`mu_tol` among them
        # (verified by reading `var200_s0/params_seed0.npz`).  File mtime is not a
        # substitute: it says when the file was written, not what the working
        # tree's constants were at that moment.
        #
        # **A stamp can only be written by the writer; there is nowhere to add it
        # afterwards**, so those 16 stay unattributable and this only helps from
        # here on.  That is also why this is fixed *before* the next retrain and
        # not as part of it -- fixing it during would leave the new products
        # unstamped too.
        # `window` is the one field this stamp was still missing
        # (2026-08-28): the ten constants below say what the market was,
        # and the window says which sixty days it ran on.  The window has moved
        # once already -- sixty consecutive days to four seasonal segments of
        # fifteen -- and every standing calibration measured before that move
        # was taken on the mildest third of the series, so an archive that
        # cannot name its window cannot be compared with one that can.  Read
        # from the position fixture's own `meta`, not from a constant here, for
        # the reason the fixture is checked against `--cap-scale` above: the
        # window is a property of the data file and not of this driver.
        f["scenario"] = np.array(json.dumps(dict(
            case=args.case, theta=list(THETA), volr=volr, beta=list(BETA),
            pi_scale=pi_scale, cap_scale=args.cap_scale,
            ramp_scale=args.ramp_scale, p_min_scale=p_min_scale,
            window=meta.get("window"),
            reg_coef=anc_clearing.REG_COEF, max_iter=anc_clearing.MAX_ITER,
            **gate)))
        # NOT in `hyperparams` and not a scenario factor either: the parameter
        # layout is a keyword of `make_ippo`, not a field of `IPPOConfig`, so
        # `vars(SHARED)` cannot carry it.  It goes in its own JSON blob rather
        # than as a bare top-level key, because this driver's reader treats every
        # non-whitelisted key as part of the parameter tree and two outages have
        # already come from exactly that.
        f["layout_meta"] = np.array(json.dumps(dict(
            per_agent_params=bool(args.per_agent_params),
            n_agents=int(spec["n_agents"]))))
        if extra:
            f.update({k: np.asarray(v) for k, v in extra.items()})
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **f)
        return f

    ckpt_dir = Path(args.out_dir) / "checkpoints"
    if args.checkpoint_every > 0:
        print(f"checkpointing every {args.checkpoint_every} iterations -> "
              f"{ckpt_dir}", flush=True)

    #: how often the curve is flushed to disk during training.  Same cadence
    #: as the checkpoints so that checking the two products against each other
    #: afterwards is one comparison of timestamps rather than two.
    curve_every = args.checkpoint_every if args.checkpoint_every > 0 else 10

    def _write_curve(rows, meta_obj):
        """Write the curve so far, atomically.

        The curve used to be written once, after the loop.  A run that died in
        the middle therefore lost every per-iteration reward while keeping all
        its checkpoints, and the two products look alike from outside: 2026-08-20
        `wd200c_s0` was taken by a machine crash at iteration 150 with sixteen
        checkpoints on disk and no curve at all, leaving only the every-20th
        line the log happens to print.  Incremental writes make the loss
        proportional to the interval instead of total.

        `meta` carries `partial`, because a file holding 100 rows because the
        run asked for 100 and a file holding 100 rows because the run died at
        100 are otherwise identical to whoever reads it next.
        """
        if not args.curve_out or not rows:
            return
        out = Path(args.curve_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        # the temporary name has to end in `.npz` as well: `np.savez` appends
        # that suffix itself when the name does not carry it, so a `.npz.tmp`
        # target silently becomes `.npz.tmp.npz` and the rename below fails
        # with FileNotFoundError.  Caught by the crash test, not by `ast.parse`.
        tmp = out.with_name(out.name + ".partial.npz")
        np.savez(tmp,
                 iteration=np.array([c["iteration"] for c in rows]),
                 reward_mean=np.array([c["reward_mean"] for c in rows]),
                 costs_mean=np.array([c["costs_mean"] for c in rows]),
                 seconds=np.array([c["seconds"] for c in rows]),
                 unconverged_frac=np.array([c["unconverged_frac"] for c in rows]),
                 price_at_cap_frac=np.array([c["price_at_cap_frac"] for c in rows]),
                 unconv_reward_share=np.array([c["unconv_reward_share"] for c in rows]),
                 unconv_max_abs_ratio=np.array([c["unconv_max_abs_ratio"] for c in rows]),
                 unconv_mean=np.array([c["unconv_mean"] for c in rows]),
                 conv_mean=np.array([c["conv_mean"] for c in rows]),
                 meta=json.dumps(meta_obj))
        # rename rather than write in place: a crash during the write would
        # otherwise destroy the rows already on disk as well as the new ones
        os.replace(tmp, out)

    if args.checkpoint_every > 0:
        # `params` is still the initialisation here.  The loop below rebinds it
        # on its first pass and only then writes `seed*_iter0000.npz`, so that
        # file is the policy after one update; nothing in the checkpoint set
        # held the policy before any.  Asking whether two runs of the same seed
        # agree needs both, otherwise the initialisation and the first rollout
        # cannot be told apart.
        _save_params(params, ckpt_dir / f"seed{args.seed}_init.npz",
                     extra=dict(iteration=-1))

    #: The JSONL curve's meta line, written before the first iteration.  It
    #: carries what the run was configured with; the `run_point` built after the
    #: loop is the other product and cannot exist yet.  `beta` here is the
    #: DECLARED pair -- what was sent.  What the environment enforced is read
    #: back per iteration into `beta_in_force`, and those are two different
    #: claims that must not share a field.
    curve_meta = dict(
        market="03 ancillary services", arm=arm, algo=args.algo, case=args.case,
        seed=args.seed, commit=commit_hash(), commit_dirty=_START_DIRTY,
        #: SAC only: the replay buffer's leaf dtypes as built (float64 here,
        #: because this market rolls out in float64; `pre`/`reward` follow the
        #: rollout's precision since 2026-09-18); None for IPPO
        replay_buffer_dtypes=({k: str(v.dtype) for k, v in opt_state["buffer"].items()}
                              if args.algo == "sac" else None),
        # scenario factors
        cap_scale=args.cap_scale, ramp_scale=args.ramp_scale,
        p_min_scale=p_min_scale, voll=VOLL_IN_EFFECT,
        markup_max=MARKUP_MAX, episode_len=T_DAY, window=meta.get("window"),
        beta=list(BETA), beta_source="run_eval_03.BETA (imported, not copied)",
        volr=volr, theta=list(THETA), pi_scale=pi_scale, period_hours=DELTA,
        reg_coef=anc_clearing.REG_COEF, max_iter=anc_clearing.MAX_ITER,
        **gate,
        position=repo_relative(args.position), shed_floor=SHED_FLOOR,
        monitored_lines=mon_stamp, kkt_route=kkt_route,
        lowrank_free=lowrank_free_stamp, lu_batching=lu_batching_stamp,
        position_monitored_lines=meta.get("monitored_lines"),
        position_kkt_route=meta.get("kkt_route"),
        ipm_freeze_mu=freeze_stamp,
        ipm_stop_tol=stop_stamp,
        ipm_max_iter=max_iter_stamp,
        train_days=len(tr_days), eval_days=len(ev_days),
        hyperparams={k: (list(v) if isinstance(v, tuple) else v)
                     for k, v in vars(SHARED).items()},
        hyperparams_provenance=(PROVENANCE if args.algo == "ippo"
                                else SAC_PROVENANCE),
        n_envs=SHARED.n_envs, horizon=SHARED.horizon,
        #: This driver has no batch-size flag, so it always runs the shared
        #: batch; `null` is "nothing overrode SHARED", not "nobody checked".
        off_shared=None,
        iterations_requested=args.iterations,
        checkpoint_every=int(args.checkpoint_every),
        #: NOT in `hyperparams` above: the parameter layout is a keyword of
        #: `make_ippo`, not a field of `IPPOConfig`, so `vars(SHARED)` cannot
        #: carry it.  A curve recorded without this key is one whose parameter
        #: layout nobody can read back off the product.
        per_agent_params=bool(args.per_agent_params),
        env_chunks=int(args.env_chunks),
        eval_days_set=args.eval_days, **runtime_stamp())
    jsonl = CurveLog(args.curve_out, curve_meta)
    if jsonl.path is not None:
        print(f"per-iteration JSONL -> {repo_relative(jsonl.path)} "
              f"(appended and fsynced every iteration)", flush=True)

    #: `params.forecast` is one total per period, so `requirement(forecast[i])`
    #: is `beta * forecast[i]` and the division below inverts it exactly.
    forecast_np = np.asarray(train_params.forecast, np.float64)

    curve, t0 = [], time.time()
    for it in range(args.iterations):
        t_it = time.time()
        params, opt_state, env_state, env_obs, key, m = step_iter(
            params, tx, opt_state, env_state, env_obs, key, train_params)
        # Block before stopping the clock: JAX dispatches asynchronously, so a
        # wall-clock read taken without this measures queueing, not compute.
        #
        # **Explicit since 2026-08-28.**  Until then this
        # line was `float(m["reward_mean"])`, which does block -- but only as a
        # side effect of a value the curve row happens to want.  That made the
        # correctness of every timing in this file depend on which fields the
        # row reads: deleting or reordering one would silently turn the whole
        # `seconds` column into dispatch time, and the symptom is smaller
        # numbers, not an error.  Markets 01, 02 and 04 all block explicitly;
        # this was the one driver that did not.
        params, opt_state, env_state, env_obs, key, m = jax.block_until_ready(
            (params, opt_state, env_state, env_obs, key, m))
        # the fraction of (step, env, product) cells the agent saw priced at the
        # cap during THIS iteration -- the quantity the substitute cannot give
        pr = np.asarray(m["step_reserve_price"], np.float64)
        at_cap = float(np.mean(pr >= volr - 1e-6))
        # how much of this iteration's reward came from steps that did not
        # converge: the frequency alone cannot say whether the curve describes
        # the policy or the solver
        split = converged_reward_split(m["step_reward"], m["step_converged"])
        # The requirement fractions IN FORCE, inverted from what the clearing
        # was handed rather than read off this file's constant.  `d_res` is
        # `beta * sum(forecast[cursor])` and the forecast here is one total per
        # period, so this recovers the pair exactly (float64, no tolerance
        # needed beyond the division itself).
        req = np.asarray(m["step_requirement"], np.float64)   # (H, E, n_prod)
        cur = np.asarray(m["step_cursor"], np.int64)          # (H, E)
        beta_force = req / forecast_np[cur][..., None]
        beta_in_force = [float(v) for v in beta_force.reshape(-1, req.shape[-1])[0]]
        beta_spread = float(np.max(np.abs(
            beta_force - np.asarray(beta_in_force))))
        # The optimiser's own five, out of `ippo._loss`'s aux, averaged over
        # this iteration's epochs and minibatches; and the per-agent reward as
        # a whole row, never reduced -- this market's collapse question is
        # exactly whether the providers differ from each other, and a mean over
        # the agent axis removes the information that question needs.
        rpa = np.asarray(m["reward_per_agent"], np.float64)
        curve.append(dict(seconds=time.time() - t_it, iteration=it,
                          price_at_cap_frac=at_cap,
                          unconv_reward_share=split["share_of_abs_reward"],
                          unconv_max_abs_ratio=split["max_abs_ratio"],
                          # signed, because the two above take absolute values
                          # and therefore cannot see a systematic offset: a step
                          # that reports reward too high keeps its magnitude
                          # share while dragging the curve
                          unconv_mean=split["mean_unconverged"],
                          conv_mean=split["mean_converged"],
                          reward_mean=float(m["reward_mean"]),
                          costs_mean=float(m["costs_mean"]),
                          unconverged_frac=float(m["unconverged_frac"]),
                          #: The same two keys as market 02.  **`masked_samples = 0`
                          #: has two meanings**, told apart by `mask_unusable` on the
                          #: same row: with the flag off, 0 means "nobody checked";
                          #: only with it on does 0 mean "no sample was excluded" --
                          #: the two `reward_mean`s mean different things, so the
                          #: flag must sit on the same row as the count.
                          #: **Do not divide it by `unconverged_frac` on the same
                          #: row.**  The mask is a **reverse cumulative AND** along
                          #: the horizon (the `jnp.flip(jnp.cumprod(jnp.flip(...)))`
                          #: in `ippo.py`; cited by expression, not by line number,
                          #: because line numbers drift): one unusable period masks
                          #: **every step before it** in that environment, because
                          #: those steps' returns contain that period's reward.
                          #: Measured 2026-09-18 on 73rts market 03, first
                          #: iteration: `unconverged_frac` 0.0042 (13 of 3 072
                          #: steps) against `masked_samples` **218**, a ratio of
                          #: 16.8 -- exactly the consequence of "the bad period
                          #: falls on average at step 16 of the horizon", not a
                          #: mismatch between the two quantities.
                          mask_unusable=bool(args.mask_unusable),
                          masked_samples=float(m.get("masked_samples", 0.0)),
                          # the learner's own diagnostics by name; SAC's
                          # `entropy` is `-log pi` of the sampled action, not
                          # IPPO's closed-form surrogate (`sac.py`)
                          **(dict(pg_loss=float(m["pg_loss"]),
                                  vf_loss=float(m["vf_loss"]),
                                  entropy=float(m["entropy"]),
                                  approx_kl=float(m["approx_kl"]),
                                  clip_frac=float(m["clip_frac"]))
                             if args.algo == "ippo" else
                             dict(q_loss=float(m["q_loss"]),
                                  q_mean=float(m["q_mean"]),
                                  target_mean=float(m["target_mean"]),
                                  actor_loss=float(m["actor_loss"]),
                                  alpha_loss=float(m["alpha_loss"]),
                                  alpha=float(m["alpha"]),
                                  entropy=float(m["entropy"]),
                                  buffer_filled=int(m["buffer_filled"]))),
                          reward_per_agent=rpa,
                          beta_in_force=beta_in_force))
        # append-and-fsync straight away, so a kill during the checkpoint below
        # still leaves this row on disk
        jsonl.iteration(curve[-1])
        if it == 0:
            # Printed once, and printed from the value the ENVIRONMENT returned
            # rather than from `BETA`.  If these two ever disagree, the constant
            # never reached the market and every number this run produces is
            # from a scenario nobody declared.
            print(f"beta in force: {tuple(beta_in_force)} "
                  f"(declared {tuple(BETA)}, source run_eval_03.BETA; read back "
                  f"from info['requirement'] / forecast over "
                  f"{req.shape[0]}x{req.shape[1]} cleared periods, spread "
                  f"{beta_spread:.3e})", flush=True)
            if max(abs(a - b) for a, b in zip(beta_in_force, BETA)) > 1e-12:
                print(f"  BETA MISMATCH: the environment enforced "
                      f"{tuple(beta_in_force)} while this run declared "
                      f"{tuple(BETA)}. The products of this run belong to the "
                      f"enforced scenario, not the declared one.", flush=True)
        if args.checkpoint_every > 0 and (it % args.checkpoint_every == 0
                                          or it == args.iterations - 1):
            _save_params(params, ckpt_dir / f"seed{args.seed}_iter{it:04d}.npz",
                         extra=dict(iteration=it))
        if it % curve_every == 0 or it == args.iterations - 1:
            _write_curve(curve, dict(
                partial=True, iterations_done=len(curve),
                iterations_requested=args.iterations, seed=args.seed,
                weight_decay=getattr(SHARED, "weight_decay", None),
                note="written during training; the write after the loop "
                     "replaces this with the full run_point and partial=False"))
        if it % 20 == 0 or it == args.iterations - 1:
            c = curve[-1]
            print(f"  iter {it:4d}  reward_mean {c['reward_mean']:+.6e}  "
                  f"costs_mean {c['costs_mean']:.4e}  unconverged "
                  f"{c['unconverged_frac']:.4f}  price_at_cap "
                  f"{c['price_at_cap_frac']:.4f}  unconv_reward_share "
                  f"{c['unconv_reward_share']:.4f}", flush=True)
            if args.algo == "ippo":
                line = (f"           pg {c['pg_loss']:+.4e}  vf {c['vf_loss']:.4e}  "
                        f"ent {c['entropy']:+.4f}  approx_kl {c['approx_kl']:+.3e}  "
                        f"clip_frac {c['clip_frac']:.4f}  ")
            else:
                line = (f"           q_loss {c['q_loss']:.4e}  q_mean "
                        f"{c['q_mean']:+.4e}  actor {c['actor_loss']:+.4e}  "
                        f"alpha {c['alpha']:.4e}  ent {c['entropy']:+.4f}  "
                        f"buffer {c['buffer_filled']}  ")
            print(line + f"reward_per_agent "
                  f"[min {c['reward_per_agent'].min():+.3e}, max "
                  f"{c['reward_per_agent'].max():+.3e}, n "
                  f"{c['reward_per_agent'].size}]", flush=True)
    jsonl.close()
    wall = time.time() - t0
    # the mean over all iterations carries the one-off compile of the first;
    # steady state is the median of the rest, which is what a per-iteration
    # figure is normally read as
    secs = [c["seconds"] for c in curve]
    if not secs:                       # eval-only: nothing was trained
        per_iter = compile_s = float("nan")
    else:
        per_iter = float(np.median(secs[1:])) if len(secs) > 1 else secs[0]
        compile_s = secs[0] - per_iter if len(secs) > 1 else float("nan")
    batch = SHARED.n_envs * SHARED.horizon
    if not secs:
        print("eval-only: no training performed", flush=True)
    else:
        print(f"training: {args.iterations} iterations in {wall:.1f} s; steady "
              f"state {per_iter:.3f} s/iteration (median of iterations 1..n), "
              f"first iteration {secs[0]:.1f} s of which about {compile_s:.1f} s "
              f"is compile, at batch = n_envs {SHARED.n_envs} x horizon "
              f"{SHARED.horizon} = {batch} env-steps", flush=True)

    # deterministic evaluation: the policy mean, through the shared entry point
    if args.algo == "ippo":
        greedy = make_greedy_action(spec, bounds, SHARED, obs_mean, obs_std,
                                    per_agent_params=args.per_agent_params)
    else:
        greedy = make_sac_greedy_action(spec, bounds, SHARED, obs_mean, obs_std,
                                        per_agent_params=args.per_agent_params)
    greedy_j = jax.jit(greedy)
    step_j = jax.jit(env[1])
    day_of = lambda st: int(st.cursor) // T_DAY
    period_of = lambda st: int(st.cursor)
    assert int(spec["periods_per_day"]) == T_DAY, (spec["periods_per_day"], T_DAY)

    run_point = dict(
        cap_scale=args.cap_scale, ramp_scale=args.ramp_scale,
        p_min_scale=p_min_scale, window=meta.get("window"),
        voll=VOLL_IN_EFFECT, volr=volr, beta=list(BETA), pi_scale=pi_scale,
        markup_max=MARKUP_MAX, market="03 ancillary services", **runtime_stamp(),
        case=args.case, arm=arm, algo=args.algo,
        seed=args.seed, iterations=args.iterations,
        per_agent_params=bool(args.per_agent_params),
        env_chunks=int(args.env_chunks),
        # The two stamps the other three drivers carry, with the same meanings,
        # so one filter reads all four markets.  Until this landed, market 03's
        # untrained control could not be selected by metadata at all: the note
        # that reported it had to name the run command and the log instead,
        # because nothing in the product said which of the two it was.
        #
        # `untrained_baseline` answers "are these weights at initialisation",
        # which is NOT "did this run train".  The two coincide everywhere except
        # at `--iterations 0` with an archive loaded, which trains nothing while
        # its weights are a trained policy's; filtering on the wrong one of the
        # two returns a trained policy's products (measured on market 01,
        # 2026-08-28).  BOTH archive flags count here, because this driver has
        # two ways to load one: `--init-params` continues from it and
        # `--eval-only` replays it.
        untrained_baseline=((args.iterations == 0
                             and not args.init_params
                             and not args.eval_only) or None),
        evaluated_without_training=((args.iterations == 0) or None),
        init_params=(repo_relative(args.init_params) if args.init_params
                     else None),
        eval_only=(repo_relative(args.eval_only) if args.eval_only else None),
        optimizer_state_restored=(False if args.init_params else None),
        batch_env_steps=batch, seconds_per_iteration=per_iter,
        hyperparams=str(PROVENANCE if args.algo == "ippo" else SAC_PROVENANCE),
        position_fixture=str(args.position),
        #: the row set and route the operator was BUILT with, read off its
        #: spec, and the ones the position was produced on beside them
        #: (recorded, not refused; `None` on an old fixture is an absent field)
        monitored_lines=mon_stamp, kkt_route=kkt_route,
        lowrank_free=lowrank_free_stamp, lu_batching=lu_batching_stamp,
        position_monitored_lines=meta.get("monitored_lines"),
        position_kkt_route=meta.get("kkt_route"),
        ipm_freeze_mu=freeze_stamp,
        ipm_stop_tol=stop_stamp,
        ipm_max_iter=max_iter_stamp,
        train_days=len(tr_days), eval_days=len(ev_days),
        note=("the honest arm is also this market's optimisation column; see "
              "run_eval_03.py for why, and do not read the two as independent"))

    scored_days = ev_days if args.eval_days == "eval" else tr_days
    scored_params = eval_params if args.eval_days == "eval" else train_params
    rows = []
    for day in scored_days:
        # `scored_params` is the 12-day subset, so the position of
        # this day inside the subset is its day index there; `reset_on_day`
        # opens that day's first period.  Before this the key search landed
        # anywhere inside the day and 353 of 576 half hours belonged to the
        # next held-out day, which is a different batch of periods from the
        # one the open-loop arms were scored on.
        k, state = open_day_start(spec["reset_on_day"], scored_params,
                                  scored_days.index(day), day_of, period_of,
                                  T_DAY)
        prod, shed, prof, mus, seps = 0.0, [], None, [], []
        drs = []
        volr_cost, short_mwh, noload, startup = 0.0, 0.0, 0.0, 0.0
        marginal, prices, lmps = [], [], []
        # Per-step reward and the per-step `converged` flag, kept so the day's
        # money can be split by whether its solve converged.  **The frequency
        # alone settles nothing** and this loop used to record only the
        # frequency: measured 2026-08-28, this market's learned arm leaves 209
        # to 217 of 576 evaluation periods above `mu_tol` while its open-loop
        # arms leave 0, so a displacement between the two arms was being read
        # off populations that differ in whether their prices can be believed.
        # What decides whether that displacement describes the policy or the
        # solver is how much of the reward those periods carry, which is what
        # `converged_reward_split` reports and what this now collects.
        step_reward, step_converged = [], []
        for _ in range(T_DAY):
            obs = spec["get_obs"](state, scored_params)
            action = greedy_j(params, obs)
            _o, state, reward, _c, _dn, info = step_j(k, state, action,
                                                      scored_params)
            prod += float(np.sum(np.asarray(info["cost"], np.float64)))
            shed.append(float(np.asarray(info["shed_mwh"])))
            r = np.asarray(reward, np.float64)
            prof = r if prof is None else prof + r
            step_reward.append(r.copy())
            step_converged.append(bool(np.asarray(info["converged"])))
            mus.append(float(info["mu"]))
            drs.append(float(info["dual_residual"]))
            # the fourth term of this market's objective, and the two
            # cost components the negative-profit claim rests on
            volr_cost += float(np.asarray(info["volr_cost"]))
            short_mwh += DELTA * float(np.sum(np.asarray(
                info["reserve_shortfall"])))
            noload += float(np.sum(np.asarray(info["cost_noload"])))
            startup += float(np.sum(np.asarray(info["cost_startup"])))
            # Whether the structurally tied pair actually prices.  `sep_min` is
            # a minimum over all provider pairs, so a tie makes it zero whether
            # or not the tied units are anywhere near the margin; the price is
            # set by the marginal provider alone.  A unit prices product j when
            # it holds reserve and its offer coincides with the product price.
            pr = np.asarray(info["reserve_price"], np.float64)
            ores = np.asarray(info["offer_res"], np.float64)
            res = np.asarray(info["reserve"], np.float64)
            for i in TIED_PAIR:
                at = np.abs(ores[i] - pr) <= 1e-6 * np.maximum(1.0, np.abs(pr))
                marginal.append(bool(np.any(at & (res[i] > 1e-6))))
            prices.append(pr.copy())
            # the energy price beside the reserve price.  Both are carried
            # out per period rather than reduced here: the matrix's price
            # level-and-distribution row asks for quantiles over bus-periods,
            # and a driver that recorded only `reserve_price_max` cannot
            # answer it without a full re-run of the training it came from.
            lmps.append(np.asarray(info["lmp"], np.float64).copy())
            # over the committed set: `offer_separation_all` spans all 66 units
            # and the de-committed ones all sit at the same offer, so its
            # minimum is dominated by units that provide nothing
            seps.append(float(np.min(np.asarray(
                info["offer_separation_committed"]))))
        sc = system_cost(prod, shed, VOLL_IN_EFFECT, volr_cost)
        split = converged_reward_split(np.stack(step_reward),
                                       np.asarray(step_converged))
        write_day(args.out_dir, arm, day, dates[day], system_cost_value=sc,
                  agent_profit=prof, shed_mwh=np.asarray(shed),
                  production_cost=prod, run_point=run_point,
                  arrays=dict(lmp=np.stack(lmps),
                              reserve_price=np.stack(prices)),
                  extra=dict(mu_max=max(mus),
                             dual_residual_max=max(drs),
                             dual_ok=dual_ok(drs, gate["dual_res_tol"]),
                             dual_res_tol=gate["dual_res_tol"],
                             #: the mu half ALONE (mu > 1e-9, the report floor
                             #: this field has always used), kept so that
                             #: products before and after the double gate
                             #: (2026-09-17) read the same thing here; the
                             #: double-gated count is `unconv_periods`
                             unconverged=int(count_cells(np.asarray(mus), 1e-9)),
                             shed_cells=int(count_cells(np.asarray(shed),
                                                        SHED_FLOOR)),
                             shed_total_mwh=float(np.sum(shed)),
                             reserve_shortfall_mwh=short_mwh,
                             reserve_shortfall_cost=volr_cost,
                             cost_noload=noload, cost_startup=startup,
                             offer_separation_min=min(seps),
                             tied_pair=list(TIED_PAIR),
                             tied_pair_marginal_periods=int(sum(marginal)),
                             tied_pair_period_denominator=len(marginal),
                             reserve_price_max=float(np.max(prices)),
                             reserve_price_at_cap_periods=int(np.sum(
                                 np.asarray(prices) >= volr - 1e-6)),
                             # the money, not just the count: an arm whose
                             # unconverged periods carry a large share of the
                             # reward cannot be compared with one whose do not
                             unconv_periods=int(sum(not c for c in
                                                    step_converged)),
                             unconv_reward_share=split["share_of_abs_reward"],
                             unconv_max_abs_ratio=split["max_abs_ratio"],
                             unconv_mean_reward=split["mean_unconverged"],
                             conv_mean_reward=split["mean_converged"]))
        #: **Clear the compilation cache once a day**, for the same reason as
        #: `run_eval_02` and `run_rl_02`: this loop does one `open_day` plus
        #: `episode_len` `step`s a day, and compiled artefacts accumulate day by
        #: day without being released.  **The mechanism was measured on 02**
        #: (29gb, 12 evaluation days, RSS taken as the current `VmRSS` in
        #: `/proc/self/status` -- `ru_maxrss` is the peak and cannot show a
        #: drop): without clearing +3.56 G over 12 days, clearing daily
        #: +0.56 G, **a factor of 6.4**; on 813nem, by the long baseline, 25 to
        #: 52 GB per hour.  **The growth on this 03 path has not been
        #: measured** -- what is carried over is the mechanism, not the number;
        #: to measure it, record `VmRSS` once before and once after this line
        #: over a dozen or so evaluation days.
        #: **It changes no number**: what is cleared is the compiled
        #: executable, and the same HLO recompiles to the same thing.
        #: **The cost is one extra recompilation a day**: this loop takes
        #: twenty-odd to forty seconds a day and a recompilation is on the
        #: order of seconds; the risk of one process running out of memory
        #: over 36 days is far more expensive than that time.
        jax.clear_caches()
        rows.append((day, sc, float(prof.sum())))
        print(f"  day {day:3d} {dates[day]}  system_cost {sc:14.4e}  "
              f"profit {float(prof.sum()):14.4e}  mu_max {max(mus):.2e}  "
              f"sep_min {min(seps):.2e}", flush=True)

    print(f"\n{len(rows)} days -> {args.out_dir}")
    print(f"total system cost {sum(r[1] for r in rows):.6e}   "
          f"total profit {sum(r[2] for r in rows):.6e}")
    # the trained parameters, because without them the only way to ask a new
    # question of this policy is to train it again
    params_path = Path(args.out_dir) / f"params_seed{args.seed}.npz"
    flat = _flatten_params(params)
    # The observation statistics travel with the parameters.  Observations are
    # standardised against a frozen reference computed once before training, so
    # parameters without that reference cannot be applied to anything: the
    # network would see inputs on a different scale from the ones it was fitted
    # on.  They are reproducible from the seed alone today, because `k_stat` is
    # split deterministically from it -- but that reproducibility depends on the
    # order of the splits above never changing, which is not a property anything
    # enforces, so the values are stored rather than relied upon.
    flat = _save_params(params, params_path)
    print(f"params -> {params_path} ({len(flat)} arrays, including obs_mean / "
          f"obs_std / hyperparams)")

    if args.curve_out:
        # one implementation for both writers, so the incremental file and the
        # final file cannot drift apart in column set or dtype
        _write_curve(curve, dict(run_point, partial=False,
                                 iterations_done=len(curve),
                                 iterations_requested=args.iterations))
        print(f"curve -> {args.curve_out} ({len(curve)} rows, partial=False)")


if __name__ == "__main__":
    main()
