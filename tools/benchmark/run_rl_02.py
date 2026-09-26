"""Market 02's learning arm.

Same shape as `run_eval_02.py`, on purpose: the learned arm's per-day numbers go
into the **same** product format as the honest and constant arms, so the three
sit in one table without anyone reconciling formats afterwards.  What this adds
is the training loop and the learning curve.

**Training draws only from the 48 training days.**  No market's `reset` takes a
day set, so the fixture is restricted instead (`evaluation.subset_position`) and
the environment can then only draw what it was given.  Evaluation builds a second
environment on the 12 held-out days and opens each of them by name.  The two day
sets come from the one shared split, so this driver cannot disagree with the
other arms about which days it never trained on.

**Evaluation is deterministic**: the policy's mean action, not a sample.  A
sampled evaluation would report the policy plus exploration noise, and the number
being compared against the other two arms is the policy.

**Hyperparameters come from `hyperparams.SHARED`**, one configuration for all
three markets, each field's provenance recorded there.  Nothing in this file
chooses a hyperparameter; if it did, "not tuned per market" would stop being true
of market 02 first.

CPU or GPU.  Run point is stamped into every product.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import (_START_DIRTY, check_split_against_report, commit_hash,
                        count_cells, effective_monitored_stamp, open_day,
                        parse_monitored_lines, refuse_ineffective_lowrank_flags,
                        split_days, subset_position, system_cost, write_day,
                        runtime_stamp)
from curve_jsonl import CurveLog, repo_relative
from hyperparams import PROVENANCE, SAC_PROVENANCE, SAC_SHARED, SHARED

# `_START_DIRTY` by name rather than re-derived: it is the checkout's state at
# process start, the same instant `commit_hash` reports, and a `git status` run
# later would answer a different question.  `write_day` writes the same pair.

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
#: **Two thresholds, two things, so two names.**  `envs.real_time.env.MU_TOL`
#: (1e-6) is the **gate**: it sets `info["converged"]`, the market's judgement
#: of "can this clearing be used".  The one below (1e-9) is only a **reporting
#: floor**: a healthy real-time clearing brings `mu` to the order of 1e-11
#: (measured 2026-09-16 on the evaluation days of 29gb and 73rts, per-period
#: `mu` median 1.2e-11 ... 2.0e-11), so "above 1e-9" counts the periods that
#: are "worse than healthy, but not necessarily failing the gate".
#: Neither name existed before, and the two concepts read as one thing in the
#: same driver.
MU_REPORT_FLOOR = 1e-9
#: `DUAL_RES_TOL` is taken from `envs.real_time.env`, not defined again here.
#: **One quantity with a default in each of two places is worse than both using
#: the old value**: this tolerance now decides both the day meta's `dual_ok`
#: and which samples the environment's `usable` masks, and the two must be the
#: same number.  The derivation (from the price consequence, and its dependence
#: on `REG_COEF`) is written beside that constant.
ARM = "ippo"
#: `real_time/env.py`'s floor, passed explicitly because `count_cells` refuses a
#: default -- the assumption belongs to this market, not to the skeleton.
SHED_FLOOR = 1e-6


def _ckpt(params_out, it, params, reward_mean, *, obs_mean, obs_std,
          hyperparams, scenario):
    """One checkpoint plus its line in the sibling index.

    Layout: `<params-out without suffix>_iter{n:04d}.npz`, and `_iters.json`
    recording each checkpoint's iteration and the `reward_mean` at that point.
    The index is what makes a checkpoint self-describing about the *curve*: the
    file name alone says which iteration but not where on the curve that sits,
    and a reader should not have to find the log.

    **The container changed on 2026-08-28 from a bare flax msgpack to the `.npz`
    of `params_npz`.**  `flax.serialization.to_bytes` writes
    the parameter tree and has nowhere to put anything else, so no checkpoint
    this driver had ever written could name its case, its scenario factors, or
    its own standardisation statistics -- and a checkpoint loaded without
    `obs_mean` / `obs_std` does not fail, it reconstructs a different policy
    that looks like the saved one.  The three `.msgpack` files already on disk
    keep their format; a stamp can only be written by the writer and there is
    nowhere to add one afterwards.  `params_npz.read` accepts what this writes
    and a policy-collapse probe on those files reads both forms.
    """
    import json
    from pathlib import Path
    import params_npz
    base = Path(params_out)
    stem = base.with_suffix("")
    ck = Path(f"{stem}_iter{it:04d}.npz")
    params_npz.write(ck, params, obs_mean, obs_std, hyperparams=hyperparams,
                     scenario=scenario,
                     meta=dict(market="02 real-time balancing", iteration=int(it),
                               reward_mean=float(reward_mean)))
    idx = Path(f"{stem}_iters.json")
    rows = json.loads(idx.read_text()) if idx.exists() else []
    rows.append(dict(iteration=int(it), reward_mean=float(reward_mean),
                     file=ck.name))
    idx.write_text(json.dumps(rows, indent=1))
    print(f"  checkpoint iter {it} -> {ck.name}  reward_mean {reward_mean:.6e}",
          flush=True)


def checkpoint_base(params_out, out_dir, seed):
    """The path `_ckpt` derives its checkpoint names from.

    Beside `--params-out` when it is given, which is the layout every existing
    market-02 checkpoint and its `_iters.json` index already have.  Without it,
    `<out-dir>/checkpoints/seed{S}`, so the files are
    `<out-dir>/checkpoints/seed{S}_iter{n:04d}.npz` -- the directory and names
    `run_rl_01.py` and `run_rl_03.py` write.  Before this, `--checkpoint-every`
    without `--params-out` was accepted and wrote nothing.
    """
    if params_out:
        return params_out
    return str(Path(out_dir) / "checkpoints" / f"seed{seed}.npz")


#: The requested-vs-effective guard on the low-rank flags,
#: `refuse_ineffective_lowrank_flags`, lives in `evaluation` since
#: 2026-09-17; imported above under its own name so `main` below and the
#: tests read as before.


def equally_spaced(pool, take, phase=0):
    """`take` entries of `pool`, equally spaced, the comb rotated by `phase`.

    Equally spaced rather than a contiguous slice because the held-out set this
    is compared against is scattered across the year (adjacent gaps median 10
    days on `case813nem`): a contiguous run of training days would put
    seasonality back into the comparison, which is the one thing the held-out
    split exists to keep out.

    `phase` rotates the whole comb so a second run can take a *different* set at
    the same spacing -- the negative control of the train-versus-held-out
    comparison.  **A phase of a
    whole tooth rotates the comb back onto itself**, and that reads like another
    set: measured 2026-09-20 before this was wired up, a pool of 48 taken 12
    (spacing 4) with `phase=4` gives index-for-index what `phase=0` gives, which
    would have made that negative control vacuous -- two "different" day sets
    that are the same days, so of course same sign and same magnitude.  Refused
    rather than normalised: the caller asked for another set and must learn it
    did not get one.  The property itself is checked, not the proxy
    `phase < spacing`, which is wrong when the spacing is not an integer.
    """
    take = int(take)
    phase = int(phase)
    if take < 1:
        raise ValueError(f"take must be >= 1; got {take}")
    if take > len(pool):
        raise ValueError(
            f"take {take} exceeds the {len(pool)} days in the pool; a comb "
            f"cannot take more teeth than the pool has days")
    step = len(pool) / take
    idx = sorted((phase + int(i * step)) % len(pool) for i in range(take))
    assert len(set(idx)) == take, (take, phase, idx)
    if phase:
        base = {int(i * step) % len(pool) for i in range(take)}
        if set(idx) == base:
            raise ValueError(
                f"phase {phase} picks the SAME days as phase 0 (spacing is "
                f"{len(pool)}/{take} = {step:g}; a phase of a whole tooth "
                f"rotates the comb back onto itself). Give a phase smaller "
                f"than the spacing")
    return [pool[i] for i in idx], idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--position", required=True)
    ap.add_argument("--out-dir", required=True)
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
    #: Which learner; `ippo` is the default and the path
    #: every archive on disk was produced on.  Same wiring as `run_rl_01.py`.
    ap.add_argument("--algo", choices=("ippo", "sac"), default="ippo",
                    help="learner: ippo (default, the existing path) or sac "
                         "(off-policy). Stamped as `algo` and `arm` in every product.")
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    #: Explicit override of the shared config's `weight_decay`.  The shared
    #: default is 0.0, which reproduces `optax.adam` exactly, so no line running
    #: without this flag has its device changed -- the positive case of "changing
    #: a shared module-level default changes every unflagged rig on every line".
    #: The value that took effect is written into the artefacts, and must be
    #: checked from there rather than from the log: the log records the
    #: intention, the product records the state.
    ap.add_argument("--weight-decay", type=float, default=None,
                    help="override hyperparams.SHARED.weight_decay for this run")
    #: Checkpoint every N iterations.  Layout: `<params-out without suffix>
    #: _iter{n:04d}.npz` when `--params-out` is given, otherwise
    #: `<out-dir>/checkpoints/seed{S}_iter{n:04d}.npz` (`checkpoint_base`)
    #: (the `params_npz` container since 2026-08-28; it was a
    #: bare flax msgpack before, which had nowhere to put the scenario or the
    #: standardisation statistics), plus a sibling `_iters.json` recording each
    #: checkpoint's iteration and the `reward_mean` at that point, so a
    #: checkpoint says where on the curve it sits without going back to the log.
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="write params every N iterations (0 = none): beside "
                         "--params-out when given, else <out-dir>/checkpoints/"
                         "seed{S}_iter{n:04d}.npz as run_rl_01/03 do")
    ap.add_argument("--per-agent-params", action="store_true",
                    help="give every unit its own copy of the policy and value "
                         "network. Archives written under this flag are NOT "
                         "interchangeable with shared ones: they have the same "
                         "number of leaves and differ only in a leading axis")
    #: How many sequential pieces the environment step of the rollout is taken
    #: in.  A keyword of `make_ippo` and NOT a field of `IPPOConfig`, like
    #: `--per-agent-params` and for the same reason; unlike that flag it is not
    #: a hyperparameter at all -- the same `n_envs x horizon` batch is laid onto
    #: the device in pieces -- so a run under it is NOT off-shared.  It exists
    #: for `case813nem`: 2.842 GiB of device memory per environment, so the
    #: shared `n_envs=64` needs 182 GiB and one 24 GB card holds seven.
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
    #: **The mask is off by default.**  When on, samples this market judges
    #: `usable=False` do not enter the gradient (their values are not changed,
    #: since the reward is never altered), and the number masked goes into the
    #: curve's `masked_samples`.  813nem's market 02 turns it on explicitly; the
    #: existing 29gb / 73rts columns on the default path are unaffected.
    ap.add_argument("--mask-unusable", action="store_true",
                    help="exclude samples this market marks unusable from the "
                         "loss (default: off, so existing columns are unchanged)")
    #: Which line limits this market's two clearings carry.  ``all`` is the
    #: default and is what every archive on disk was produced on: every line
    #: enforced, dense KKT route.  ``rated`` keeps only the lines whose rating
    #: the case publishes -- on `case813nem` 7 of 1 278, the rest sitting at
    #: 1e6 MW against 39 GW of capacity -- and takes the low-rank route, which
    #: is the difference between a tractable `case813nem` run and an
    #: intractable one.  A comma-separated list of indices is for controls.
    #:
    #: **The two routes are not interchangeable downstream.**  Measured on the
    #: day-ahead position over 18 days (2026-09-16): prices agree to 4.93e-07
    #: \$/MWh and each period's total to 1.00e-03 MW, but per-unit dispatch
    #: differs by up to 62.39 MW on 34 of 151 units -- a degenerate LP moving
    #: between vertices of the same optimal face.  A claim that consumes
    #: per-unit quantities must stay on one route from fixture to result.
    ap.add_argument("--monitored-lines", default="all",
                    help="line limits the clearing carries: all (default, "
                         "dense KKT, what every archive was produced on), "
                         "rated (the case's published ratings, low-rank), or "
                         "a comma-separated list of line indices")
    #: The low-rank route's free-column block (2026-09-16): shed
    #: columns and units kept in a pivoted block so a period that sheds load
    #: behind a binding line is solved to the dense route's residual.  The
    #: defaults are the market's own (`default_lowrank_free`: now
    #: every column, (151, 813) on 813nem, under the arrowhead solve where
    #: the width is a linear cost); smaller sizings are what the LU tiers
    #: were run at (523 = the region behind line 10, 128 = behind line 738),
    #: acceptable only because a period the block does not cover shows in
    #: `dual_ok` -- and a trial run (2026-09-17, (8,128)) showed one
    #: such sample turning a whole run NaN.  The value in effect is read back
    #: off the spec and stamped in `kkt_route`.
    ap.add_argument("--lowrank-free-shed", type=int, default=None,
                    help="shed columns in the low-rank route's pivoted block "
                         "(default: the case's sizing, 523 on 813nem)")
    ap.add_argument("--lowrank-free-units", type=int, default=None,
                    help="units in that block (default LOWRANK_FREE_UNITS)")
    ap.add_argument("--lu-batching", default="auto", choices=["auto", "sequential", "batched", "arrow"],
                    help="how that block is solved under vmap: auto "
                         "(sequential on CPU, batched on GPU), sequential, batched, "
                         "or arrow (no LU of the block: thin QR of its border plus a "
                         "small core; single-period shape only)")
    #: PILOT ONLY, the same flag `run_rl_01.py` carries and for the same
    #: reason: to price compilation separately from the steady state, and -- the
    #: reason it arrives here -- to make `case813nem` reachable at all on a CPU,
    #: where the shared 64 environments of a 151-unit, 813-bus, 1 278-line
    #: network do not fit in a smoke test's budget.  Any run that gives a value
    #: DIFFERENT from the shared one is stamped `off_shared` and must not be
    #: reported as the shared configuration.  `None` is the default, so a
    #: command line that does not name it runs exactly what it always ran.
    #: 2026-09-17: the IPM's regulariser coefficient and its stop.  Both
    #: default to the operator as it always was (`ipm.REG_COEF` 1e-14, fixed trip
    #: count); `case813nem` on the rated rows needs 1e-16 and a stop one notch
    #: below the gate.  Read back off the spec
    #: and stamped as `ipm_reg_coef` / `ipm_stop_tol`, next to `kkt_route`.
    ap.add_argument("--reg-coef", type=float, default=None,
                    help="ipm.make_solver reg_coef (default: ipm.REG_COEF); 1e-16 on case813nem")
    ap.add_argument("--stop-tol", default=None,
                    help="mu_tol,dual_tol: stop the Newton loop at this tolerance, max_iter "
                         "as the cap (default: the fixed trip count); 1e-8,1e-7 on case813nem")
    ap.add_argument("--n-envs", type=int, default=None,
                    help="PILOT ONLY. Overrides SHARED.n_envs. Any run whose "
                         "value differs from the shared one is stamped "
                         "off_shared and must not go in the report.")
    ap.add_argument("--sac-alpha", type=float, default=None,
                    help="pin SAC's temperature: SACConfig.init_alpha, and with "
                         "--init-params the archive's log_alpha leaf is replaced "
                         "by log of this value; stamped off_shared (an "
                         "ablation, 2026-09-20). Refused with --algo ippo")
    ap.add_argument("--sac-alpha-lr", type=float, default=None,
                    help="SACConfig.alpha_lr override; 0 freezes the temperature "
                         "(no autotuning); stamped off_shared. Refused with "
                         "--algo ippo")
    ap.add_argument("--curve-out", default="")
    ap.add_argument("--params-out", default="",
                    help="where to write the trained policy; without it the "
                         "only way to ask the policy a new question is to "
                         "retrain")
    ap.add_argument("--init-params", default="",
                    help="continue from a params npz this driver wrote instead "
                         "of from a fresh initialisation; with --iterations 0 "
                         "it is a pure evaluation of that archive (the way to "
                         "ask a checkpoint a new question without retraining). "
                         "The archive's OWN obs_mean/obs_std are used, never a "
                         "fresh fit: a refit reconstructs a different policy "
                         "from the same weights. The archive holds no optimiser "
                         "state, so Adam restarts from zero moments -- a real "
                         "discontinuity, stamped `optimizer_state_restored: "
                         "false`. Same flag name and same meaning as "
                         "`run_rl_01.py` and `run_rl_03.py`")
    ap.add_argument("--eval-only", default="",
                    help="path to a saved params npz: skip training and evaluate "
                         "that policy.  Exists because asking a new question of "
                         "a trained policy otherwise costs a full retrain, and "
                         "this market trains for hours.  Same flag name and same "
                         "meaning as `run_rl_01.py` and `run_rl_03.py`; unlike "
                         "`--init-params` it also reads a SAC archive, because "
                         "the graft below is keyed on the archive's own `algo` "
                         "stamp rather than on the learner it was written by")
    #: **Change the evaluation day set, not the training.**  The question is the
    #: gap between "the same archives, the same apparatus, with only the
    #: evaluation moved onto the training days", whereas `split_days` freezes the
    #: evaluation days per case (`CASE_EVAL_WINDOWS`) and no flag could change
    #: them before (`--eval-only` selects which archive to evaluate, not which
    #: days).
    #: **The vocabulary is copied from `run_eval_02.py` in this repository**
    #: (`--days {eval,train,all}` and where the `day_slice` stamp sits): two
    #: drivers with two words for the same thing is worse than both using the
    #: old word.
    #: **What cannot be copied is its `--day-from/--day-to`** -- that gives a
    #: **contiguous range**, whereas the held-out set is 36 days scattered across
    #: the year (median gap between neighbours 10), and a contiguous run of
    #: training days would put seasonality back in, the very thing the held-out
    #: set is there to avoid.  So what is given here is an **equally spaced**
    #: comb: `--day-take K` plus `--day-phase P`.
    #: **The training side is untouched**: `--days train` changes only the
    #: evaluation set, and the training environment is still built on the whole
    #: of `tr_days`.
    ap.add_argument("--days", default="eval", choices=("eval", "train", "all"),
                    help="which day pool to evaluate on: the frozen held-out "
                         "set (default, bit-for-bit what this driver always "
                         "did), the training days, or the whole window. Same "
                         "flag name and same meaning as `run_eval_02.py`")
    ap.add_argument("--day-take", type=int, default=None,
                    help="take K days equally spaced from the pool `--days` "
                         "selected, instead of all of them. Equally spaced "
                         "rather than a contiguous slice: the held-out set is "
                         "scattered across the year, and a contiguous run of "
                         "training days would put seasonality back in")
    ap.add_argument("--day-phase", type=int, default=0,
                    help="rotate the comb `--day-take` lays down, so a second "
                         "run can take a different K days at the same spacing "
                         "(the negative control of the train-versus-held-out "
                         "comparison)")
    args = ap.parse_args()
    #: **The two flags cannot be given together.**  Both name an archive, and
    #: given together "which one is used" could only be answered by reading the
    #: order of the code -- exactly the accepted-then-dropped shape: the command
    #: line looks as if both took effect.
    #: **Placed at argument validation rather than at the line that uses the
    #: archive** (a guard goes before the irreversible step) -- left later, it
    #: would only bite after the environment is built and the position file
    #: check has run, two minutes each time.
    if args.init_params and args.eval_only:
        raise SystemExit(
            "--init-params and --eval-only both name an archive; give one. "
            "`--eval-only X` is `--init-params X --iterations 0` with the "
            "intent stamped, and two paths would silently pick one")

    #: **`--day-phase` alone has no meaning, so it is refused rather than
    #: silently ignored** (the first explanation of "this parameter has no
    #: effect" is "I did not pass it in").  Placed here rather than at the line
    #: that uses the day set: that line comes after reading the position file
    #: and building the environment.
    if args.day_phase and args.day_take is None:
        raise SystemExit("--day-phase only rotates the comb --day-take lays "
                         "down; without --day-take it would be silently ignored")
    if args.day_take is not None and args.day_take < 1:
        raise SystemExit(f"--day-take must be >= 1; got {args.day_take}")

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")

    import powermarketjax
    print("package:", powermarketjax.__file__, flush=True)

    from powermarketjax.case import load_case, scale_min_output
    from powermarketjax.envs.day_ahead import demand_from_meta
    from powermarketjax.envs.real_time import load_da_position
    from powermarketjax.envs.real_time.demand import (T_RT,
                                                      half_hourly_from_meta)
    from powermarketjax.envs.real_time.clearing import default_lowrank_free
    from powermarketjax.envs.real_time.env import DUAL_RES_TOL, make_env
    from powermarketjax.learning.adapters import unpack_env
    from powermarketjax.learning.ippo import (make_greedy_action, make_ippo,
                                              observation_statistics)
    from powermarketjax.learning.policy import bounds_for
    from powermarketjax.learning.sac import (make_sac, make_sac_greedy_action,
                                             reward_statistics)
    arm = args.algo

    pos = load_da_position(path=args.position)
    meta = pos["meta"]
    for key, given in (("cap_scale", args.cap_scale),
                       ("ramp_scale", args.ramp_scale),
                       ("voll", VOLL_IN_EFFECT)):
        got = meta.get(key)
        if got is None or abs(float(got) - given) > 1e-12:
            raise SystemExit(f"fixture meta {key}={got} disagrees with {given}")

    dates = [str(d) for d in meta["dates"]]
    ok, _sel = check_split_against_report(dates, meta["case"])
    print(f"split matches report §2.2: {ok}", flush=True)
    if not ok:
        raise SystemExit("the split rule no longer reproduces the report's "
                         "evaluation days; resolve deliberately")
    ev_days, tr_days = split_days(len(dates), meta["case"])
    #: The day set is chosen **after** `check_split_against_report` (the order
    #: and reason of `run_eval_02.py`): that check asks whether the **whole**
    #: evaluation day set the rule gives agrees with the report, which has
    #: nothing to do with which days this run evaluates.
    #: `tr_days` is untouched -- it builds the training environment; this flag
    #: only changes the evaluation set.
    _pool = {"eval": ev_days, "train": tr_days,
             "all": sorted(ev_days + tr_days)}[args.days]
    #: **Both keys are always stamped, even when no flag is given** (decided
    #: 2026-09-20).  The branch that does not stamp would reproduce exactly the
    #: false pass found that day: a product without this key **cannot tell "run
    #: before the change" from "run after the change on the default path"**, and
    #: whether each needs re-reading is entirely different (when any input is
    #: missing, no key is produced).
    #:
    #: **`eval_day_set` stamps "which day numbers were actually used", not
    #: "which path was taken"** (same decision).  Until then no product was
    #: self-describing: in meta, `train_days=329`, `eval_days=36` and
    #: `day_index=9` are all counts and the day's ordinal, and **nothing records
    #: which 36 days they are** -- to know, one could only re-run `split_days`,
    #: which is exactly the thing that may change.  "A stamp stamped
    #: unconditionally carries no information" is realised exactly here:
    #: **stamping the path name is the kind that carries none, stamping the day
    #: numbers is the kind that does.**  Criteria always read `eval_day_set`, not
    #: `day_select` (stamp the quantity to be used directly; do not make the
    #: reader infer it from a label).
    day_select = "split_days" if args.days == "eval" else args.days
    if args.day_take is not None:
        #: The comb's arithmetic is lifted into a module-level function **so
        #: that criteria can call it directly** -- the two refusal branches
        #: (beyond the pool, a whole-tooth phase) and "K distinct, equally
        #: spaced" can all be checked without starting a full run.
        try:
            _pool, _idx = equally_spaced(_pool, args.day_take, args.day_phase)
        except ValueError as e:
            raise SystemExit(f"--day-take/--day-phase: {e}")
        day_select = (f"take={args.day_take},phase={args.day_phase},"
                      f"pool={args.days}")
    ev_days = [int(d) for d in _pool]

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
    #: parsed after `case` exists, because `rated` is a property of the case's
    #: line ratings and not of the string
    monitored = parse_monitored_lines(args.monitored_lines, case)
    if args.lowrank_free_shed is None and args.lowrank_free_units is None:
        lowrank_free = None                      # the case's own sizing
    else:
        auto_units, auto_shed = default_lowrank_free(case, monitored)
        lowrank_free = (auto_units if args.lowrank_free_units is None else args.lowrank_free_units,
                        auto_shed if args.lowrank_free_shed is None else args.lowrank_free_shed)
    stop_tol = None if args.stop_tol is None else tuple(float(v) for v in args.stop_tol.split(","))
    build = lambda p: make_env(case, p, hh, fc, n_segments=1,
                               markup_max=MARKUP_MAX,
                               cap_scale=args.cap_scale,
                               ramp_scale=args.ramp_scale,
                               monitored_lines=monitored,
                               lowrank_free=lowrank_free,
                               lu_batching=args.lu_batching,
                               reg_coef=args.reg_coef, stop_tol=stop_tol)

    train_env_obj, spec = build(subset_position(pos, tr_days))
    eval_env_obj, eval_spec = build(subset_position(pos, ev_days))
    #: **Read off each operator, never off the flag.**  `make_env` already
    #: refuses a train env whose two clearings disagree; what is left for the
    #: driver is that the flag reached `make_env` at all, and that the eval env
    #: -- a second `make_env` call with a different position slice -- landed on
    #: the same set.  A run whose operator disagrees with its command line is
    #: refused here rather than stamped with either value.
    mon_stamp, kkt_route = effective_monitored_stamp(spec, requested=monitored)
    #: the regulariser and the stop the step clearing was built with, off the
    #: spec (same discipline as the route); a flag the operator did not take is
    #: refused, not stamped
    ipm_reg_coef = float(spec["clearing"]["reg_coef"])
    ipm_stop_tol = spec["clearing"]["stop_tol"]
    if args.reg_coef is not None and ipm_reg_coef != float(args.reg_coef):
        raise SystemExit(f"--reg-coef {args.reg_coef} requested but the clearing was built on {ipm_reg_coef}")
    if (None if stop_tol is None else tuple(stop_tol)) != (None if ipm_stop_tol is None else tuple(ipm_stop_tol)):
        raise SystemExit(f"--stop-tol {stop_tol} requested but the clearing was built on {ipm_stop_tol}")
    _b_stamp, _b_route = effective_monitored_stamp(
        {"clearing": spec["boundary_clearing"]}, requested=monitored)
    _e_stamp, _e_route = effective_monitored_stamp(eval_spec, requested=monitored)
    if not (_b_route == _e_route == kkt_route):
        raise SystemExit(f"the step clearing took the {kkt_route} KKT route, "
                         f"the boundary {_b_route} and the eval env {_e_route}")
    refuse_ineffective_lowrank_flags(
        (("step", spec["clearing"]), ("boundary", spec["boundary_clearing"]),
         ("eval", eval_spec["clearing"])),
        lowrank_free if lowrank_free is not None else default_lowrank_free(case, monitored),
        args.lu_batching)
    print(f"monitored lines in effect: "
          f"{'all' if mon_stamp is None else mon_stamp}; "
          f"kkt route: {kkt_route}", flush=True)
    #: The route the POSITION was produced on, recorded beside the route this
    #: run clears on.  Recorded and **not** refused: the day-ahead position is
    #: an input to this market, not something it recomputes, so a mismatch is
    #: not automatically an error -- but a result built on a dense-route
    #: position and a low-rank real-time clearing is internally mixed, and
    #: nothing else in the product would say so.  Measured on the day-ahead
    #: position (2026-09-16, 18 days): the two routes agree on price to
    #: 4.93e-07 \$/MWh and disagree on per-unit dispatch by up to 62.39 MW, and
    #: `q_da` is exactly what this market is handed.
    #:
    #: `None` when the fixture predates the stamp (2026-09-16), and that is NOT
    #: read as "the same route": an absent field is an absent field.
    position_route = meta.get("kkt_route")
    position_monitored = meta.get("monitored_lines")
    if position_route is None:
        print("position fixture predates the kkt_route stamp; its route is "
              "unrecorded, not assumed equal to this run's", flush=True)
    elif str(position_route) != str(kkt_route):
        print(f"NOTE: the position was produced on the {position_route} route "
              f"and this run clears on {kkt_route}. Both are stamped; a claim "
              f"consuming per-unit quantities should not mix them", flush=True)
    train_params = train_env_obj.make_params(episode_len=T_RT)
    eval_params = eval_env_obj.make_params(episode_len=T_RT)
    #: Under `--days train`, `disjoint` **is supposed to be False** -- that is
    #: exactly the question being asked.  So the day set's origin and its first
    #: and last few day numbers are printed with it, so that `disjoint False` is
    #: not taken for an error.
    print(f"train days {len(tr_days)}   eval days {len(ev_days)}   "
          f"disjoint {not set(tr_days) & set(ev_days)}", flush=True)
    print(f"day select {day_select}", flush=True)
    print(f"evaluated day numbers: {ev_days}", flush=True)

    # `unpack_env` takes the (env, spec) pair `make_env` returns, not the
    # env alone; it normalises the action keys the harness reads
    train_env = unpack_env((train_env_obj, spec))
    bounds = bounds_for(train_env[3])
    key = jax.random.PRNGKey(args.seed)
    key, k_stat, k_init = jax.random.split(key, 3)
    import dataclasses
    if args.weight_decay is not None and args.algo != "ippo":
        raise SystemExit("--weight-decay belongs to the IPPO optimiser chain; "
                         "SACConfig has no such field")
    if args.mask_unusable and args.algo != "ippo":
        # Accepted-and-dropped, and one worse than the `--env-chunks`
        # refusal below: `valid_key` is handed to `make_ippo` only, `make_sac`
        # does not take it, so nothing would be masked -- **and the curve row
        # would still carry `mask_unusable: true` next to `masked_samples: 0.0`**,
        # which is the one reading the comment at that stamp says means "no
        # sample was excluded".  The `curve.append` is outside the
        # `algo == "ippo"` branch, so the stamp is written on the SAC path too.
        # A stamp that lies is worse than a flag that is ignored.
        raise SystemExit("--mask-unusable is wired through make_ippo's "
                         "`valid_key` only; the SAC learner does not take it, "
                         "so it is refused rather than silently ignored while "
                         "the curve stamps mask_unusable: true")
    base = SHARED if args.algo == "ippo" else SAC_SHARED
    cfg = (base if args.weight_decay is None
           else dataclasses.replace(base, weight_decay=args.weight_decay))
    #: The flag being present is not the test -- the value DIFFERING is.  A
    #: `--n-envs 64` repeats the shared value and changes nothing, and stamping
    #: that product off-shared would exclude a run that is on it.  Replaced from
    #: `cfg` and not from `base`, so an earlier `--weight-decay` override is not
    #: silently discarded (`run_rl_01.py` records the 2026-08-19 run where
    #: rebuilding from the base did exactly that).
    off_shared = {}
    if args.n_envs is not None:
        if args.n_envs != base.n_envs:
            off_shared["n_envs"] = [base.n_envs, args.n_envs]
        cfg = dataclasses.replace(cfg, n_envs=args.n_envs)
        if off_shared:
            print(f"OFF-SHARED PILOT: {off_shared} -- these products are "
                  f"stamped `off_shared` and must not be reported as the "
                  f"shared configuration", flush=True)
        else:
            print(f"--n-envs {args.n_envs} equals SHARED.n_envs; not "
                  f"off-shared", flush=True)
    if args.sac_alpha is not None or args.sac_alpha_lr is not None:
        if args.algo != "sac":
            raise SystemExit("--sac-alpha / --sac-alpha-lr belong to SACConfig; "
                             "this run is --algo ippo")
        #: Same rule as `--n-envs`: the value DIFFERING from SAC_SHARED is what
        #: stamps the product off-shared, not the flag being present.
        if args.sac_alpha is not None and args.sac_alpha != base.init_alpha:
            off_shared["init_alpha"] = [base.init_alpha, args.sac_alpha]
            cfg = dataclasses.replace(cfg, init_alpha=args.sac_alpha)
        if args.sac_alpha_lr is not None and args.sac_alpha_lr != base.alpha_lr:
            off_shared["alpha_lr"] = [base.alpha_lr, args.sac_alpha_lr]
            cfg = dataclasses.replace(cfg, alpha_lr=args.sac_alpha_lr)
        print(f"OFF-SHARED PILOT: {off_shared} -- temperature pinned by flag; "
              f"these products are stamped `off_shared` and must not be "
              f"reported as the shared configuration", flush=True)
    #: **After `cfg`, and from `cfg`.**  `run_rl_01.py:469` has always taken
    #: the statistics batch from `cfg`; this driver took it from `SHARED`,
    #: which was the same thing until `--n-envs` existed (2026-09-16).  Left
    #: where it was, the flag would still build `SHARED.n_envs` environments
    #: for the statistics pass -- on `case813nem` the 64 the flag exists to
    #: avoid -- while training ran on 4.  With no flag `cfg.n_envs is
    #: SHARED.n_envs` and `cfg.horizon is SHARED.horizon`, so every command
    #: line that does not name it draws the same statistics from the same key.
    #: `--init-params`: the archive decides whether the statistics are fitted
    #: at all.  Same shape as `run_rl_03.py`: with the pair stored, the fit is
    #: skipped -- a refit would reconstruct a different policy from the same
    #: weights, and `k_stat` is its own key so skipping the pass leaves
    #: `k_init` and the training stream untouched.  Without the flag every
    #: line below runs exactly as before.
    _archive_path = args.init_params or args.eval_only
    _init_tree = _init_mean = _init_std = None
    import params_npz
    if _archive_path:
        _init_tree, _init_mean, _init_std, _init_info = params_npz.read(
            _archive_path)
        #: **Undo `_archive_tree` by the archive's own `algo` stamp, not by the
        #: command line.**  On writing, IPPO writes the flax tree directly and
        #: SAC wraps `{actor, q1, q2, q1_target, q2_target, log_alpha}` in a
        #: `params` layer (see `_archive_tree` below and the comment beside it),
        #: so the inverse on reading is this one line.  The stamp is what the
        #: writer put and the command line is what the reader says: **when the
        #: two disagree the stamp wins and the run is refused on the spot**,
        #: otherwise a SAC archive under `--algo ippo` would be reported by the
        #: key-set comparison below as "wrong network", calling "the wrong
        #: learner was chosen" "the archive is broken".
        _stamped = (_init_info.get("scenario") or {}).get("algo")
        if _stamped is not None and _stamped != args.algo:
            raise SystemExit(
                f"{_archive_path} was written by --algo {_stamped}, this run is "
                f"--algo {args.algo}; the two learners' parameter trees differ "
                f"in shape and in what they mean")
        if (_stamped or args.algo) != "ippo":
            _init_tree = _init_tree["params"]
        _declared = (_init_info.get("scenario") or {}).get("per_agent_params")
        if _declared is not None and bool(_declared) != bool(args.per_agent_params):
            raise SystemExit(
                f"{_archive_path} declares per_agent_params={bool(_declared)} "
                f"but this run is per_agent_params={bool(args.per_agent_params)}; "
                f"the two layouts have the same leaf count and differ only in a "
                f"leading axis, so loading across them would not fail loudly")
    if _init_mean is not None:
        obs_mean, obs_std = jnp.asarray(_init_mean), jnp.asarray(_init_std)
        print(f"observation statistics taken from "
              f"{repo_relative(_archive_path)}; not refitted", flush=True)
    else:
        obs_mean, obs_std = observation_statistics(train_env, train_params, k_stat,
                                                   cfg.n_envs, cfg.horizon,
                                                   env_chunks=args.env_chunks)
    if args.weight_decay is not None:
        print(f"weight_decay override: SHARED {SHARED.weight_decay} -> "
              f"{cfg.weight_decay}", flush=True)
    if args.per_agent_params:
        print(f"PER-AGENT PARAMETERS: {int(train_env[3]['n_agents'])} independent "
              f"copies of the network. Products of this run are stamped "
              f"`per_agent_params: true` and must not be compared leaf-for-leaf "
              f"against a shared archive.", flush=True)
    if args.algo == "ippo":
        #: **Off by default** (decided 2026-09-17).  The environment has always
        #: emitted the `usable` key, and whether it is read is decided here: with
        #: `valid_key=None` every expression of the learner is bitwise what it
        #: was before this flag existed.  Why it cannot be on by default was
        #: measured -- in a one-day smoke test, of the 3 072 samples of one
        #: iteration, 29gb masks 56 (1.82%) and 73rts masks **529 (17.22%)**,
        #: with `mu` failing not once in either (`unconverged_frac` 5.55e-17):
        #: all were caught by `dual_residual`.  On by default would replace the
        #: premise "29gb / 73rts are unchanged byte for byte", which every change
        #: relies on, with a no-op assumption that has never been verified.
        init, iterate = make_ippo(train_env, bounds, cfg, obs_mean, obs_std,
                                  per_agent_params=args.per_agent_params,
                                  env_chunks=args.env_chunks,
                                  valid_key="usable" if args.mask_unusable else None)
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
        scale = float(reward_statistics(train_env, train_params,
                                        jax.random.fold_in(k_stat, 1),
                                        cfg.n_envs, cfg.horizon))
        cfg = dataclasses.replace(cfg, reward_scale=scale)
        print(f"SAC reward_scale = {scale:.6e} (pooled std of the per-agent "
              f"reward under the truthful action over a {cfg.n_envs} x "
              f"{cfg.horizon} sample)", flush=True)
        init, iterate = make_sac(train_env, bounds, cfg, obs_mean, obs_std,
                                 per_agent_params=args.per_agent_params)
    params, tx, opt_state, env_state, env_obs = init(k_init, train_params)
    if _init_tree is not None:
        # Key-for-key shape check against the freshly initialised tree, on the
        # on-disk key convention (`params_npz.flatten`, which is what wrote
        # the archive): the archive must be a policy of THIS layout and this
        # network, and a mismatch is reported by key rather than several
        # frames later.  The fresh tree keeps its container type; only the
        # leaves are replaced, so nothing downstream sees a different pytree.
        _fresh_flat = params_npz.flatten(params)
        _stored_flat = params_npz.flatten(_init_tree)
        if sorted(_fresh_flat) != sorted(_stored_flat):
            raise SystemExit(
                f"{_archive_path} holds keys {sorted(_stored_flat)}; this "
                f"run's network has {sorted(_fresh_flat)}")
        for k in _fresh_flat:
            if tuple(_stored_flat[k].shape) != tuple(_fresh_flat[k].shape):
                raise SystemExit(f"{_archive_path}: {k} has shape "
                                 f"{tuple(_stored_flat[k].shape)}, this run's "
                                 f"is {tuple(_fresh_flat[k].shape)}")
        _leaves, _treedef = jax.tree_util.tree_flatten(params)
        _paths = [
            "/".join(str(getattr(q, "key", q)) for q in path)
            for path, _ in jax.tree_util.tree_flatten_with_path(params)[0]]
        assert len(_paths) == len(_leaves) and set(_paths) == set(_fresh_flat)
        params = jax.tree_util.tree_unflatten(
            _treedef, [jnp.asarray(_stored_flat[k], leaf.dtype)
                       for k, leaf in zip(_paths, _leaves)])
        if args.eval_only:
            #: `--eval-only` does not train, so "Adam restarts from zero moments"
            #: is not a fact on this branch and should not be printed -- printing
            #: it would make a reader of the log think this run moved the weights.
            print(f"eval-only: params from {repo_relative(args.eval_only)}; "
                  f"training skipped", flush=True)
        else:
            print(f"init-params: {repo_relative(args.init_params)}; optimiser state "
                  f"restarts from zero moments", flush=True)
        if args.sac_alpha is not None:
            #: The archive carries its own autotuned `log_alpha`; a pinned
            #: temperature must replace that leaf, or the flag only renames
            #: the starting point the archive already overrode.
            params = dict(params, log_alpha=jnp.full_like(
                params["log_alpha"], jnp.log(args.sac_alpha)))
            print(f"--sac-alpha {args.sac_alpha}: the archive's log_alpha leaf "
                  f"is replaced (shape {tuple(params['log_alpha'].shape)})",
                  flush=True)
    if args.eval_only:
        #: Semantics identical to `run_rl_03.py`: skip training, evaluate this
        #: archive.  Placed **after** the graft, so that "the archive cannot be
        #: read" happens before "the iteration count is zeroed" -- otherwise a bad
        #: archive would quietly become a zero-iteration untrained control, and
        #: those two products carry opposite `untrained_baseline` stamps.
        args.iterations = 0
    # `tx` is an optax transform, i.e. a tuple of functions, so it cannot be
    # traced as an array; it is the second positional argument of `iterate`
    step_iter = jax.jit(iterate, static_argnums=(1,))

    # Per-iteration wall clock, first iteration kept separate: it carries the
    # compilation, which is paid once.  A mean over all of them would report a
    # number that is neither the compile cost nor the steady-state cost, and the
    # steady-state one is what a 3-seed run is scheduled from.
    #: The JSONL curve's meta line, written before the first iteration: it says
    #: what the run was configured with, which the `run_point` built after the
    #: loop cannot do yet.  A run killed in its first iteration still leaves a
    #: file that identifies itself.
    #: The scenario stamp carried by the curve's meta line **and** by every
    #: parameter file this run writes.  Assembled once so the two cannot
    #: disagree; before 2026-08-28 only the curve had it, because the parameter
    #: container had no room for it (see `_ckpt`).
    scenario_meta = dict(
        case=str(meta.get("case")), cap_scale=args.cap_scale,
        ramp_scale=args.ramp_scale, p_min_scale=p_min_scale,
        voll=VOLL_IN_EFFECT, markup_max=MARKUP_MAX, episode_len=T_RT,
        window=meta.get("window"), n_lookahead=1,
        position=repo_relative(args.position),
        #: NOT in `hyperparams`, and that is the point: it is a keyword of
        #: `make_ippo`, not a field of `IPPOConfig`, so `vars(cfg)` cannot carry
        #: it.  A product recorded without this key is one whose parameter
        #: layout nobody can read back off the file -- the two layouts have the
        #: same leaf COUNT and differ only in a leading axis.
        per_agent_params=bool(args.per_agent_params),
        env_chunks=int(args.env_chunks),
        #: the line set and KKT route the operators were actually built on, so a
        #: product says which of the two non-interchangeable dispatch routes it
        #: is on without going back to the command line
        monitored_lines=mon_stamp, kkt_route=kkt_route,
        ipm_reg_coef=ipm_reg_coef, ipm_stop_tol=ipm_stop_tol,
        position_monitored_lines=position_monitored,
        position_kkt_route=position_route,
        #: which learner wrote the archive, so a reader can undo `_archive_tree`
        algo=args.algo,
        #: SAC only: the replay buffer's leaf dtypes as built, read off the
        #: learner rather than declared.  Since 2026-09-18 `pre`/`reward` follow
        #: the rollout's precision (float32 here); a product that predates the
        #: stamp ran with float64 replay `pre`/`reward`
        replay_buffer_dtypes=({k: str(v.dtype) for k, v in opt_state["buffer"].items()}
                              if args.algo == "sac" else None))
    #: `params_npz` whitelists `params/...` keys, which is the flax tree of ONE
    #: network.  SAC's learner tree is `{actor, q1, q2, q1_target, q2_target,
    #: log_alpha}`, each a flax tree of its own, so it is written under a
    #: single `params` root and comes back as `read(...)[0]["params"]`; the
    #: `algo` stamp in `scenario` says which reading applies.
    _archive_tree = (lambda p: p) if args.algo == "ippo" else (
        lambda p: {"params": p})
    curve_meta = dict(
        market="02 real-time balancing", arm=arm,
        # `algo` arrives through `scenario_meta` below
        seed=args.seed, commit=commit_hash(), commit_dirty=_START_DIRTY,
        # scenario factors
        **scenario_meta,
        shed_floor=SHED_FLOOR, train_days=len(tr_days), eval_days=len(ev_days),
        #: The day set goes into the stamp, otherwise a product evaluated on
        #: training days cannot say which days it evaluated (a stamp can only
        #: be written by the writer; there is nowhere to add it later).
        eval_day_set=[int(d) for d in ev_days], day_select=day_select,
        hyperparams={k: (list(v) if isinstance(v, tuple) else v)
                     for k, v in vars(cfg).items()},
        hyperparams_provenance=(PROVENANCE if args.algo == "ippo"
                                else SAC_PROVENANCE),
        n_envs=cfg.n_envs, horizon=cfg.horizon,
        #: `null` is the statement "nothing overrode SHARED", not the statement
        #: "nobody checked": `--n-envs` exists since 2026-09-16 and an empty
        #: dict is normalised to `null` so the products written before it
        #: existed and the ones written without it read the same.
        off_shared=off_shared or None,
        iterations_requested=args.iterations,
        checkpoint_every=int(args.checkpoint_every),
        init_params=(repo_relative(args.init_params) if args.init_params
                     else None),
        #: A column with the same name and meaning as in run_point.  The curve's
        #: meta row is the first thing a consumer reads, and stamping it only in
        #: run_point would leave "how was this curve made" unanswerable.
        eval_only=(repo_relative(args.eval_only) if args.eval_only else None),
        optimizer_state_restored=(False if args.init_params else None),
        **runtime_stamp())
    jsonl = CurveLog(args.curve_out, curve_meta)
    if jsonl.path is not None:
        print(f"per-iteration JSONL -> {repo_relative(jsonl.path)} "
              f"(appended and fsynced every iteration)", flush=True)

    curve, per_iter, t0 = [], [], time.time()
    for it in range(args.iterations):
        t_it = time.time()
        params, opt_state, env_state, env_obs, key, m = step_iter(
            params, tx, opt_state, env_state, env_obs, key, train_params)
        jax.block_until_ready(params)          # or the timing measures dispatch
        per_iter.append(time.time() - t_it)
        # Spread of the markup ACROSS units, per iteration.  The terminal value
        # alone cannot say whether anything was learned: an untrained policy
        # already produces a spread of 0.0922 because the initialisation is not
        # constant, so "the markup differs between units" is true before any
        # training.  The judgement has to be movement against that baseline, and
        # only the series shows movement.
        #
        # `step_action` is **(horizon, n_envs, n_units)** -- three axes, not
        # four: `_act` reshapes to the market's own `action_shape`, and markup
        # is one number per unit, so there is no trailing `act_dim` axis here.
        # Axis 2 is the unit axis.
        #
        # The definition is "standard deviation ACROSS UNITS", which is not the
        # same sentence as "std over `axis=-1`".  They coincide only because
        # this market's action is 1-D; on a 2-D action market `axis=-1` is the
        # per-agent action axis, and the reduction would run over the wrong axis
        # **without raising**.  Anyone copying this line to another market must
        # copy the definition, not the index.
        spread = float(jnp.mean(jnp.std(m["step_action"], axis=2)))
        # The optimiser's own five, out of `ippo._loss`'s aux and averaged over
        # this iteration's epochs and minibatches.  They were returned all along
        # and dropped here: a reward curve alone cannot separate "the policy
        # stopped moving" from "the update stopped being applied", and
        # `approx_kl` / `clip_frac` are the two that do separate them.
        #
        # `reward_per_agent` is kept as a whole row and NOT reduced.  The
        # question this market is asked first is whether the units differ from
        # each other at all; a mean over the agent axis is precisely the
        # information that question needs.
        rpa = np.asarray(m["reward_per_agent"], np.float64)
        # the learner's own diagnostics by name; SAC's `entropy` is `-log pi`
        # of the sampled action, not IPPO's closed-form surrogate (`sac.py`)
        if args.algo == "ippo":
            diag = dict(pg_loss=float(m["pg_loss"]),
                        vf_loss=float(m["vf_loss"]),
                        entropy=float(m["entropy"]),
                        approx_kl=float(m["approx_kl"]),
                        clip_frac=float(m["clip_frac"]))
        else:
            diag = dict(q_loss=float(m["q_loss"]), q_mean=float(m["q_mean"]),
                        target_mean=float(m["target_mean"]),
                        actor_loss=float(m["actor_loss"]),
                        alpha_loss=float(m["alpha_loss"]),
                        alpha=float(m["alpha"]), entropy=float(m["entropy"]),
                        buffer_filled=int(m["buffer_filled"]))
        curve.append(dict(iteration=it,
                          seconds=per_iter[-1],
                          reward_mean=float(m["reward_mean"]),
                          costs_mean=float(m["costs_mean"]),
                          action_spread=spread,
                          unconverged_frac=float(m["unconverged_frac"]),
                          #: Number of samples masked (in environment-steps).
                          #: **0 has two meanings**, told apart by `mask_unusable`
                          #: on the same row: with the flag off, 0 means "nobody
                          #: checked"; only with it on does 0 mean "no sample was
                          #: excluded".  The two `reward_mean`s mean different
                          #: things, so the flag must sit on the same row as the
                          #: count.
                          mask_unusable=bool(args.mask_unusable),
                          masked_samples=float(m.get("masked_samples", 0.0)),
                          **diag,
                          reward_per_agent=rpa))
        # append-and-fsync immediately, so a kill during the checkpoint below
        # still leaves this row on disk
        jsonl.iteration(curve[-1])
        if args.checkpoint_every and (it + 1) % args.checkpoint_every == 0:
            _ckpt(checkpoint_base(args.params_out, args.out_dir, args.seed),
                  it + 1, _archive_tree(params),
                  float(m["reward_mean"]),
                  obs_mean=obs_mean, obs_std=obs_std,
                  hyperparams=curve_meta["hyperparams"],
                  scenario=scenario_meta)
        if it % 20 == 0 or it == args.iterations - 1:
            c = curve[-1]
            print(f"  iter {it:4d}  reward_mean {c['reward_mean']:+.6e}  "
                  f"costs_mean {c['costs_mean']:.4e}  spread "
                  f"{c['action_spread']:.5f}  unconverged "
                  f"{c['unconverged_frac']:.4f}", flush=True)
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
    # Save the trained policy before evaluating anything.  A pilot exists to buy
    # evidence, and a pilot whose policy cannot be re-questioned buys only the
    # curve: every later question ("what markup did it learn", "is it
    # heterogeneous across units") would otherwise cost a full retrain.
    jsonl.close()
    if args.params_out:
        import params_npz
        # same container as the checkpoints: the reader cannot tell which kind
        # of file it was handed, so the final write and the periodic ones have
        # to be the same format or one of them is silently unusable
        out = params_npz.write(
            Path(args.params_out).with_suffix(".npz"), _archive_tree(params),
            obs_mean,
            obs_std, hyperparams=curve_meta["hyperparams"],
            scenario=scenario_meta,
            meta=dict(market="02 real-time balancing", seed=args.seed,
                      iterations=args.iterations))
        print(f"wrote trained params -> {repo_relative(out)}", flush=True)

    steady = per_iter[1:] if len(per_iter) > 1 else per_iter
    #: From `cfg`, not from `SHARED`.  Until `--n-envs` existed (2026-09-16)
    #: the two were the same thing here and reading either was correct; the
    #: flag made this line able to report a batch that did not run, and the
    #: first `--n-envs 4` run on `case813nem` printed `n_envs 64` beside a
    #: 10.1 s iteration that only 4 explains.  The curve file was right
    #: throughout -- it takes `cfg.n_envs` -- which is the usual split: the log
    #: records the intention, the product records the state.
    batch = cfg.n_envs * cfg.horizon
    # `--iterations 0` is the untrained control, the same flag `run_rl_01.py:509`
    # uses for market 01: the loop above does not run, the weights stay at their
    # initialisation and everything below is the ordinary evaluation path.  The
    # branch is on the iteration count and not on `per_iter` being empty, so a
    # timing list that comes out empty any other way still raises here.
    if args.iterations == 0 and _archive_path:
        print(f"training: 0 iterations -- EVALUATION of the archive "
              f"{repo_relative(_archive_path)}, not an untrained control. "
              f"Every timing field below and in the curve file is null by "
              f"construction, not by failure.", flush=True)
    elif args.iterations == 0:
        print("training: 0 iterations -- UNTRAINED CONTROL, weights at "
              "initialisation. Every timing field below and in the curve file "
              "is null by construction, not by failure.", flush=True)
    else:
        print(f"training: {args.iterations} iterations in {time.time() - t0:.1f} s\n"
              f"  first iteration (with compile) {per_iter[0]:.2f} s\n"
              f"  steady state {np.mean(steady):.3f} s/iteration "
              f"(min {np.min(steady):.3f}, max {np.max(steady):.3f}, n={len(steady)})\n"
              f"  batch = n_envs {cfg.n_envs} x horizon {cfg.horizon} = "
              f"{batch} env-steps per iteration, {batch * int(spec['n_agents'])} "
              f"agent-transitions", flush=True)

    # --- deterministic evaluation on the held-out days ---
    # `make_greedy_action` is the deterministic entry point: the policy mean in
    # this market's own action shape.  It replaces a rebuild of the network plus
    # a second derivation of the action layout that used to live here -- the
    # duplication that let the `act_dim` defect reappear in this very file after
    # it had already been reported once.
    if args.algo == "ippo":
        greedy = make_greedy_action(train_env[3], bounds, cfg, obs_mean, obs_std,
                                    per_agent_params=args.per_agent_params)
    else:
        greedy = make_sac_greedy_action(train_env[3], bounds, cfg, obs_mean,
                                        obs_std,
                                        per_agent_params=args.per_agent_params)

    # this market exposes `get_obs` on the env object, not in `spec`; taken from
    # the eval env so it reads the held-out days' series
    eval_get_obs = eval_env_obj.get_obs
    step = jax.jit(eval_env_obj.step)
    greedy_j = jax.jit(greedy)
    day_of = lambda st: int(st.cursor) // T_RT

    run_point = dict(cap_scale=args.cap_scale, ramp_scale=args.ramp_scale,
                     p_min_scale=p_min_scale,
                     window=meta.get("window"), voll=VOLL_IN_EFFECT,
                     n_lookahead=1, markup_max=MARKUP_MAX, **runtime_stamp(),
                     market="02 real-time balancing", arm=arm, algo=args.algo,
                     per_agent_params=bool(args.per_agent_params),
                     env_chunks=int(args.env_chunks),
                     monitored_lines=mon_stamp, kkt_route=kkt_route,
                     ipm_reg_coef=ipm_reg_coef, ipm_stop_tol=ipm_stop_tol,
                     off_shared=off_shared or None,
                     position_monitored_lines=position_monitored,
                     position_kkt_route=position_route,
                     arm_note=(("IPPO" if args.algo == "ippo" else
                                "SAC (ADR-0016; critic on reward / "
                                f"reward_scale={cfg.reward_scale:.6e})")
                               + (", one policy PER AGENT"
                                  if args.per_agent_params
                                  else ", parameter-shared")
                               + "; evaluation is the mean "
                               "action, not a sample, so the number is the "
                               "policy and not the policy plus exploration"),
                     iterations=args.iterations, seed=args.seed,
                     # `untrained_baseline` answers "are these weights at
                     # initialisation", which is NOT "did this run train":
                     # the two part company at `--iterations 0 --init-params`,
                     # which trains nothing while its weights are a trained
                     # policy's (`run_rl_03.py` has the same pair)
                     #: **Both load flags are written out literally**, not
                     #: through the intermediate name `_archive_path`: a test
                     #: checks from the **source text** whether this line
                     #: mentions every load flag, and it cannot see through the
                     #: indirection `_archive_path = args.init_params or
                     #: args.eval_only`.  `not (a or b)` and `not a and not b`
                     #: are equivalent, and the literal form lets that criterion
                     #: bite -- the criterion is cheap and stable precisely
                     #: because it is literal, and a guard should not be changed
                     #: to suit a refactoring here.
                     untrained_baseline=((args.iterations == 0
                                          and not args.init_params
                                          and not args.eval_only) or None),
                     evaluated_without_training=(args.iterations == 0) or None,
                     init_params=(repo_relative(args.init_params)
                                  if args.init_params else None),
                     #: A column separate from `init_params` rather than one
                     #: merged "archive" column: "continue training from an
                     #: archive" and "only evaluate this archive" are two kinds
                     #: of product, and once merged they could only be inferred
                     #: back from `iterations`, which on `--init-params
                     #: --iterations 0` is exactly what cannot be inferred.
                     #: `run_rl_03.py` also has two columns.
                     eval_only=(repo_relative(args.eval_only)
                                if args.eval_only else None),
                     optimizer_state_restored=(False if args.init_params
                                               else None),
                     # the value that TOOK EFFECT, read off the config the
                     # optimiser was built from -- not `args.weight_decay`,
                     # which is null when the run uses the shared default and
                     # would leave the product unable to state its own setting
                     weight_decay=(float(cfg.weight_decay)
                                   if args.algo == "ippo" else None),
                     weight_decay_source=("--weight-decay"
                                          if args.weight_decay is not None
                                          else "hyperparams.SHARED"
                                          if args.algo == "ippo" else None),
                     checkpoint_every=int(args.checkpoint_every),
                     train_days=len(tr_days), eval_days=len(ev_days),
                     eval_day_set=[int(d) for d in ev_days],
                     day_select=day_select,
                     hyperparams=(PROVENANCE if args.algo == "ippo"
                                  else SAC_PROVENANCE))

    rows = []
    for pos_i, day in enumerate(ev_days):
        # the eval env only contains the SELECTED days (held-out by default,
        # training days under `--days train`), so day `d` of the full window is
        # row `pos_i` of that env
        k, state = open_day(eval_env_obj.reset, eval_params, pos_i, day_of)
        prod, shed, prof, mus, acts, lmps = 0.0, [], None, [], [], []
        drs, reps = [], []
        for _ in range(T_RT):
            # the eval environment's own `get_obs`, not the training one's:
            # they are separate builds and only this one was given the held-out
            # days' series
            obs = eval_get_obs(state, eval_params)
            act = greedy_j(params, obs)
            acts.append(np.asarray(act, np.float64))
            _o, state, reward, _c, _dn, info = step(k, state, act, eval_params)
            prod += float(np.sum(np.asarray(info["cost"], np.float64)))
            shed.append(float(info["shed_mwh"]))
            r = np.asarray(reward, np.float64)
            prof = r if prof is None else prof + r
            mus.append(float(info["mu"]))
            drs.append(float(info["dual_residual"]))
            reps.append(int(np.asarray(info["lmp_replaced"])))
            lmps.append(np.asarray(state.lmp_prev, np.float64))
        # market 02's clearing objective has no third shortfall term (no reserve
        # product), so the new fourth argument is 0.0 here -- passed explicitly
        # because `system_cost` refuses a default for it
        sc = system_cost(prod, shed, VOLL_IN_EFFECT, 0.0)
        write_day(args.out_dir, arm, day, dates[day], system_cost_value=sc,
                  agent_profit=prof, shed_mwh=np.asarray(shed),
                  production_cost=prod, run_point=run_point,
                  extra=dict(mu_max=max(mus),
                             unconverged=int(sum(m > MU_REPORT_FLOOR
                                                 for m in mus)),
                             #: The dual residual, beside `mu` rather than
                             #: instead of it: `converged` looks only at `mu`,
                             #: and a period whose `mu` happens to fall inside
                             #: the gate while its dual is already unusable was
                             #: reported by no product before this column.
                             dual_residual_max=max(drs),
                             #: Number of times a price in the observation was
                             #: replaced (NaN or beyond +-VOLL).
                             lmp_replaced=int(sum(reps)),
                             dual_ok=bool(max(drs) <= DUAL_RES_TOL),
                             dual_res_tol=DUAL_RES_TOL,
                             shed_cells=count_cells(shed, SHED_FLOOR),
                             # the question this market will be asked first: a
                             # UNIFORM markup cannot move the quantity channel
                             # at all (measured: constant arm shifts system cost
                             # by 7e-8%), so "did it learn anything that can"
                             # means "does the markup vary across units"
                             # std ACROSS UNITS then mean over periods; the
                             # stack is (48 periods, 66 units), so axis 1 is the
                             # unit axis.  Same definition as the training-time
                             # series, different regime (mean vs sampled).
                             action_spread_mean=float(np.mean(
                                 np.std(np.stack(acts), axis=1))),
                             action_mean=float(np.mean(np.stack(acts)))),
                  arrays=dict(action=np.stack(acts), lmp=np.stack(lmps)))
        #: **Clear the compilation cache once a day**, for the same reason and on
        #: the same measurement as `run_eval_02`: compiled artefacts accumulate
        #: day by day without being released.  Measured on 29gb's 12 evaluation
        #: days (RSS taken as the current `VmRSS` of `/proc/self/status` --
        #: `ru_maxrss` is the peak and cannot show a drop): without clearing
        #: +3.56 G over 12 days, clearing daily +0.56 G, **a factor of 6.4**.  On
        #: 813nem this path was measured at **+651 MB per minute (about +50 G
        #: over 36 days)**, steeper than the per-period path.
        #: **It changes no number**: what is cleared is the compiled executable,
        #: and the same HLO recompiles to the same thing.
        jax.clear_caches()
        rows.append((day, dates[day], sc, float(prof.sum()),
                     count_cells(shed, SHED_FLOOR)))
        print(f"  day {day:3d} {dates[day]}  system_cost {sc:14.4e}  "
              f"profit {float(prof.sum()):+14.4e}  shed_cells {rows[-1][4]:2d}",
              flush=True)

    print(f"\n{len(rows)} days written to {args.out_dir}")
    print(f"total system cost {sum(r[2] for r in rows):.6e}   "
          f"total profit {sum(r[3] for r in rows):+.6e}   "
          f"shed cells {sum(r[4] for r in rows)}")
    if args.curve_out:
        np.savez(args.curve_out,
                 iteration=np.array([c["iteration"] for c in curve]),
                 reward_mean=np.array([c["reward_mean"] for c in curve]),
                 costs_mean=np.array([c["costs_mean"] for c in curve]),
                 unconverged_frac=np.array([c["unconverged_frac"] for c in curve]),
                 action_spread=np.array([c["action_spread"] for c in curve]),
                 seconds_per_iteration=np.array(per_iter),
                 meta=json.dumps(dict(
                     run_point,
                     first_iteration_s=(None if args.iterations == 0
                                        else float(per_iter[0])),
                     steady_s_per_iteration=float(np.mean(per_iter[1:]))
                     if len(per_iter) > 1 else None,
                     batch_env_steps=int(cfg.n_envs * cfg.horizon))))
        print("wrote", args.curve_out)


if __name__ == "__main__":
    main()
