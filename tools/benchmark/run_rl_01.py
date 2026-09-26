"""Market 01's learning arm.

Same shape as `run_eval_01.py` on purpose: the learned arm's per-day numbers go
into the same product format as the honest and constant arms, so the three sit in
one table without anyone reconciling formats afterwards.  What this adds is the
training loop and the learning curve.

**Training draws only from the 48 training days.**  `reset` takes no day set, so
the fixture is restricted instead (`evaluation.subset_position`, which slices
every day-indexed array together with `meta["dates"]`) and the environment can
then only draw what it was given.  Evaluation builds a second environment on the
12 held-out days.  Both come from the one shared split.

**Evaluation is deterministic**: `make_greedy_action` gives the policy mean, so
the number compared against the other two arms is the policy and not the policy
plus exploration.

**Hyperparameters come from `hyperparams.shared_for("01")`** -- nothing here
chooses one.  That is `SHARED` with market 01's `horizon` (4, four one-clearing
episodes); `horizon` is declared per market because a rollout of one 24-hour
clearing is not the same quantity as one of a half-hour period.

**Day-ahead is a one-step episode** (`D = 1`, section 19), so this market is a
contextual bandit rather than a multi-step MDP: there is no credit assignment,
and `gamma` and GAE do no work.  That makes "reward rises then flattens" *easier*
to satisfy here than in markets 02 and 03, not harder, which is worth knowing
before reading the curve -- if this one fails to clear the constant arm, the
place to look is whether there is anything to learn, not the optimiser.

CPU or GPU.  Run point stamped into every product.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import (_START_DIRTY, check_split_against_report, commit_hash,
                        count_cells, effective_monitored_stamp, open_day,
                        parse_monitored_lines, split_days, subset_position,
                        system_cost, write_day, runtime_stamp)
from curve_jsonl import CurveLog, repo_relative
from hyperparams import (PROVENANCE, SAC_PROVENANCE, SAC_SHARED, SHARED,
                         sac_shared_for, shared_for)

#: Which row of `hyperparams.HORIZON` this driver is.
MARKET = "01"

# `_START_DIRTY` is imported by name rather than re-derived here on purpose: it
# is the state of the checkout at **process start**, which is the same instant
# `commit_hash` reports, and a second `git status` run later would answer a
# different question. It is private to `evaluation` because nothing outside it
# needed the pair separately until the JSONL curve did; `write_day` writes both.

CAP_SCALE = 0.60
RAMP_SCALE = 1.00
MARKUP_MAX = 2.0
VOLL_IN_EFFECT = 10_000.0
T = 24
K = 1
ARM = "ippo"

#: `count_cells` refuses a default floor, and day-ahead needs one that market 02
#: does not: `real_time/env.py:356` zeroes shed below `SHED_FLOOR` before it is
#: reported, and **day-ahead has no such step** -- `clearing.py:308` only zeroes
#: buses with no demand, which is not a magnitude floor.  Measured on this
#: market: `info["shed_mwh"]` on the honest arm's twelve evaluation days is
#: ~1.23e-19 rather than 0, and per cell the dust is ~1.4e-22, so `s > 0` counts
#: 696 of 696 cells as shed.  This floor sits thirteen orders above that dust and
#: five orders below the only real shed observed on this market (0.137 MWh,
#: in the exact-commitment arm).  Market 03 shipped `s > 0` for exactly this reason; the assumption
#: belongs to the market, not to the skeleton.
SHED_FLOOR = 1e-6


def collapse_criteria(params, obs_mean, obs_std, eval_obs):
    """The three collapse criteria, computed where the checkpoint is written.

    A stopping condition is only worth having if the quantity it reads exists
    at the moment the decision is due.  The wd=0.52 run of 2026-08-20 met both
    of its stopping conditions at iteration 40, and ran to 111 because the
    criteria were a batch job over checkpoints that nobody ran until the next
    day -- 70 iterations, about 2.8 hours.  This costs about 8 s of CPU against
    a 146 s iteration, so paying it every checkpoint returns on the first run
    that would otherwise overrun by one checkpoint interval.

    This PRINTS and never stops.  Both markets wrote "stop when saturation is
    nonzero" and both then continued and re-read the logs afterwards, which is
    the evidence that this judgement is not yet fit to be an automatic gate.
    It is an instrument until it has called several runs correctly.

    Same definitions as `checkpoint_criteria_01.py`, which computes them from
    the written file; that tool is the cross-check on this one, and the two
    agreeing on a checkpoint is a real check because the paths differ (in
    memory here, reloaded there).

    `full_sat` reads `|tanh| == 1.0` exactly, and that under-reports.  Measured
    in float64 on this machine 2026-08-21, reproducing 03's numbers: a whole
    interval of pre-activations maps to one float64 long before 19 --

        x = 17.00  the same float64 out to 17.0048  (width 0.005)
        x = 18.00                       18.0777            0.078
        x = 18.50                       18.9903            0.490
        x = 19.00  tanh is exactly 1.0 from here on

    so two generators whose pre-activations land in one of those intervals get
    a bit-identical hidden row, and a bit-identical action, while `full_sat`
    still says zero.  03 saw exactly that: saturation fraction 0.000 at
    iteration 130 with the action rows already merged.  `n_hidden` and
    `n_action` are the columns that see it, and `s2/s1` is the column that
    sees the direction collapse behind it -- a kernel whose second singular
    value has fallen away maps every input onto one line, and merging is then
    two generators landing on the same point of that line.  Weight decay
    penalises size, and this happens in direction, which is why it did not fix
    the curve.

    `s2/s1` on `Dense_2` has no value in this market and is reported `n/a`:
    day-ahead declares `action_shape == (n_units,)`, so one agent's action is a
    single number and the head is 64 -> 1 with exactly one singular value.  03's
    `Dense_2` ratio therefore has no counterpart here, and printing `nan` for it
    without saying why is how a structural absence gets read as a measurement.

    No threshold is set on `s2/s1`.  It is printed because it moves three and a
    half orders of magnitude while the Frobenius norm holds still; what value
    of it means "collapsed" has not been calibrated on this market, and putting
    a number here would be inventing one.
    """
    w = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        key = "/".join(getattr(q, "key", str(getattr(q, "idx", q))) for q in path)
        w[key] = np.asarray(leaf, np.float64)
    om = np.asarray(obs_mean, np.float64)
    os_ = np.asarray(obs_std, np.float64)

    full, nact, nhid = [], [], []
    for obs in eval_obs:
        z = (np.asarray(obs, np.float64) - om) / os_
        h = np.tanh(z @ w["params/Dense_0/kernel"] + w["params/Dense_0/bias"])
        o = np.tanh(h @ w["params/Dense_1/kernel"] + w["params/Dense_1/bias"])
        mean = o @ w["params/Dense_2/kernel"] + w["params/Dense_2/bias"]
        full.append(int((np.abs(o) == 1.0).all(axis=1).sum()))
        nhid.append(int(len(np.unique(o, axis=0))))
        nact.append(int(len(np.unique(mean, axis=0))))

    sv = np.linalg.svd(w["params/Dense_1/kernel"], compute_uv=False)
    sv2 = np.linalg.svd(w["params/Dense_2/kernel"], compute_uv=False)
    pos = sv[sv > 0]
    pr = pos / pos.sum()
    ratio = lambda v: float(v[1] / v[0]) if len(v) > 1 and v[0] > 0 else float("nan")
    return dict(full_sat=full, n_action=nact, n_hidden=nhid,
                spectral_entropy=float(np.exp(-(pr * np.log(pr)).sum())),
                participation=float(sv.sum() ** 2 / (sv ** 2).sum()),
                stable_rank=float((sv ** 2).sum() / sv.max() ** 2),
                s2s1_dense1=ratio(sv), s2s1_dense2=ratio(sv2),
                s1_dense1=float(sv[0]), s1_dense2=float(sv2[0]))


def write_params(path, params, obs_mean, obs_std, cfg, *, seed, iteration=None,
                 per_agent_params=False, extra_meta=None):
    """The one place weights are written, for both checkpoints and the final file.

    `obs_mean` / `obs_std` go into EVERY file, not only the final one.  A
    checkpoint without them does not fail when it is loaded -- it silently
    standardises with a reference the weights were never fitted against, and the
    policy it reconstructs is a different policy that looks like the saved one.

    Keys are named so a reader can whitelist what it wants.  Market 03 read its
    checkpoints with a blacklist -- skip these three keys, treat the rest as the
    parameter tree -- and adding `iteration` put a scalar into the tree, which
    surfaced several frames away from the cause.
    """
    flat, treedef = jax.tree_util.tree_flatten(params)
    #: Which parameter layout this file holds.  It is stamped here and not left
    #: to `vars(cfg)`, which does not carry it: `per_agent_params` is a keyword
    #: of `make_ippo` rather than a field of `IPPOConfig` precisely so that the
    #: shape of `hyperparams` does not move under every archive already written.
    #: Without this key the two layouts are indistinguishable from the file --
    #: they have the same leaf COUNT and differ only in a leading axis, so a
    #: reader that unflattens without checking gets a different policy rather
    #: than an error.
    meta = dict(market="01 day-ahead wholesale", seed=seed,
                per_agent_params=bool(per_agent_params))
    if iteration is not None:
        meta["iteration"] = iteration
    if extra_meta:
        meta.update(extra_meta)
    np.savez(path,
             **{f"p{i}": np.asarray(a) for i, a in enumerate(flat)},
             treedef=str(treedef), obs_mean=np.asarray(obs_mean),
             obs_std=np.asarray(obs_std),
             hyperparams=json.dumps({k: (list(v) if isinstance(v, tuple) else v)
                                     for k, v in vars(cfg).items()}),
             config=json.dumps({k: (list(v) if isinstance(v, tuple) else v)
                                for k, v in vars(cfg).items()}),
             meta=json.dumps(meta))
    return path


def final_params_path(curve_out, out_dir, seed):
    """Where the trained policy is written before evaluation.

    Beside `--curve-out` when it is given (`<curve-out stem>.params.npz`, the
    name every existing market-01 archive has); otherwise
    `<out-dir>/params_seed{S}.npz`, the name `run_rl_03.py` uses.  Before this,
    a run without `--curve-out` wrote no final weights at all.
    """
    if curve_out:
        return Path(curve_out).with_suffix(".params.npz")
    return Path(out_dir) / f"params_seed{seed}.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=None)
    #: The two scenario scales this market trains at, as flags rather than as the
    #: bare module constants.  `case73rts` adopts `cap_scale = 0.424`
    #: and the check below
    #: refuses a position fixture whose scenario disagrees, so with the constants
    #: alone this driver could only ever be pointed at `case29gb`'s scenario --
    #: measured 2026-09-12 on `run_eval_02.py`, which took these same two flags
    #: for the same reason: `fixture meta cap_scale=0.424 disagrees with 0.6`.
    #: Each defaults to the constant it replaces, so a command line naming
    #: neither runs exactly what it always ran.
    #:
    #: **The third scale, `p_min_scale`, is deliberately NOT a flag**, for the
    #: reason `run_eval_02.py` gives: the commitment in the fixture was built at
    #: that minimum-output level, and training against another one is not a
    #: choice that belongs on a command line.
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
    #: The action-space ceiling as a flag, defaulting to the constant it
    #: replaces, so a command line naming nothing runs exactly what it always
    #: ran.  Until 2026-09-14 the only way to train at another ceiling
    #: was a copy of this file with `MARKUP_MAX` edited (1.6 for
    #: `case73rts`); the value is stamped in the curve meta and `run_point`
    #: below, so a product says which ceiling it was trained under.  The
    #: module constant stays: `run_bestfixed_grid.py`, `action_diff.py` and
    #: eight other tools from-import it as the endpoint of their grids.
    ap.add_argument("--markup-max", type=float, default=MARKUP_MAX)
    ap.add_argument("--out-dir", required=True)
    #: Which learner.  `ippo` is the default and the path every archive in the
    #: repository was produced on; `sac` is the off-policy
    #: learner.  The two share the carry positions (`init` / `iterate`), the
    #: batch flags and the evaluation path; they differ in the configuration
    #: object, the greedy entry point and the diagnostics a row carries.
    ap.add_argument("--algo", choices=("ippo", "sac"), default="ippo",
                    help="learner: ippo (default, the existing path) or sac "
                         "(off-policy). The choice is stamped as `algo` and as "
                         "`arm` in every product.")
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--curve-out", default="")
    ap.add_argument("--weight-decay", type=float, default=None,
                    help="AdamW weight decay; overrides SHARED.weight_decay. "
                         "Passed to the optimiser, not to the network.")
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="write the weights every N iterations, to "
                         "`<out-dir>/checkpoints/seed{S}_iter%%04d.npz`. Needed to tell "
                         "a random walk from a drift: the endpoint alone fits "
                         "both, and only a trajectory separates sqrt(t) growth "
                         "from linear.")
    ap.add_argument("--pilot", action="store_true",
                    help="mark the products as a pilot; they do not go in the report")
    ap.add_argument("--n-envs", type=int, default=None,
                    help="PILOT ONLY. Overrides this market's n_envs to size the compile "
                         "and per-step cost separately. Any run that sets this is "
                         "stamped off-shared and must not go in the report.")
    ap.add_argument("--horizon", type=int, default=None,
                    help="PILOT ONLY. Overrides this market's horizon "
                         "(hyperparams.HORIZON['01'] = 4); same condition.")
    #: One network per agent instead of one shared by all of them.  It is a
    #: keyword of `make_ippo` and NOT a field of `IPPOConfig`, because the three
    #: drivers write `vars(cfg)` into every checkpoint's `hyperparams` and a new
    #: field there would alter every product already on disk.  The consequence
    #: is that `vars(cfg)` does not carry it either, so this flag has to be
    #: stamped into the curve meta and the archive meta EXPLICITLY -- which is
    #: done below, and is the only thing that lets a reader tell the two
    #: parameter layouts apart from the file.
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
    #: Which line limits the two clearing operators carry, and by which
    #: linear-algebra route (`parse_monitored_lines`).  "all" = every line,
    #: dense route, what every archive was produced on; "rated" = the lines
    #: with a published rating, low-rank route -- on `case813nem` that is 7 of
    #: 1 278 and the only way its learning cell fits a card.  Not a
    #: hyperparameter: the LP's optimum is the same whenever the dropped lines
    #: cannot bind; what it changes numerically has been measured separately.
    #: Stamped in `run_point` and the curve.
    ap.add_argument("--monitored-lines", default="all",
                    help="'all' (dense route, the default), 'rated' (the lines "
                         "with a published rating, low-rank route), or a "
                         "comma-separated list of line indices")
    ap.add_argument("--init-params", default="",
                    help="continue from a `.params.npz` this driver wrote "
                         "instead of from a fresh initialisation. The archive's "
                         "OWN obs_mean/obs_std are used, never a fresh fit: a "
                         "refit reconstructs a different policy from the same "
                         "weights (`write_params`'s docstring). The archive "
                         "holds no optimiser state, so Adam restarts from zero "
                         "moments -- a real discontinuity, recorded as "
                         "`optimizer_state_restored: false`.")
    ap.add_argument("--start-iteration", type=int, default=0,
                    help="the number the first iteration of THIS run is "
                         "recorded under. With --init-params it makes the "
                         "continuation count on from where the archive stopped, "
                         "so the curve rows and the checkpoint file names do "
                         "not collide with the earlier run's.")
    args = ap.parse_args()
    if args.start_iteration and not args.init_params:
        # Not fatal, but a run that renumbers its iterations without continuing
        # anything produces a curve whose x-axis claims a history it does not
        # have, and nothing downstream can tell that from a real continuation.
        print(f"WARNING: --start-iteration {args.start_iteration} without "
              f"--init-params: this run starts from a fresh initialisation and "
              f"its curve will nonetheless be numbered from "
              f"{args.start_iteration}", flush=True)

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")

    import powermarketjax
    print("package:", powermarketjax.__file__, flush=True)
    print("devices:", jax.devices(), flush=True)

    from powermarketjax.case import load_case, scale_min_output
    from powermarketjax.envs.day_ahead import (demand_from_meta, load_commitment,
                                               make_env)
    from powermarketjax.envs.day_ahead.commitment import FIXTURE_DIR
    from powermarketjax.learning.adapters import unpack_env
    from powermarketjax.learning.ippo import (make_greedy_action, make_ippo,
                                              observation_statistics)
    from powermarketjax.learning.policy import bounds_for
    from powermarketjax.learning.sac import (make_sac, make_sac_greedy_action,
                                             reward_statistics)
    arm = args.algo

    path = (Path(args.fixture) if args.fixture else
            FIXTURE_DIR / f"day_ahead_commitment_29gb_T{T}_relax.npz")
    fixture = load_commitment(path=path, n_periods=T,
                             p_min_scale=args.p_min_scale)
    meta = fixture["meta"]
    for key, given in (("cap_scale", args.cap_scale),
                       ("ramp_scale", args.ramp_scale),
                       ("voll", VOLL_IN_EFFECT)):
        got = meta.get(key)
        if got is None or abs(float(got) - given) > 1e-12:
            raise SystemExit(f"fixture meta {key}={got} disagrees with {given}")

    dates = [str(d) for d in meta["dates"]]
    ok, _sel = check_split_against_report(dates, meta["case"])
    print(f"split matches report section 2.2: {ok}", flush=True)
    if not ok:
        raise SystemExit("the split rule no longer reproduces the report's "
                         "evaluation days; resolve deliberately")
    ev_days, tr_days = split_days(len(dates), meta["case"])

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
    demand = demand_from_meta(meta)
    monitored = parse_monitored_lines(args.monitored_lines, case)
    build = lambda f: make_env(case, f, demand, n_segments=K, kind="markup",
                               markup_max=args.markup_max,
                               cap_scale=args.cap_scale,
                               ramp_scale=args.ramp_scale,
                               monitored_lines=monitored)
    train_env_obj, spec = build(subset_position(fixture, tr_days))
    eval_env_obj, _eval_spec = build(subset_position(fixture, ev_days))
    # stamped off the operators that were built, not off the flag; refuses if
    # the two disagree (`effective_monitored_stamp`)
    monitored_stamp, kkt_route = effective_monitored_stamp(spec, requested=monitored)
    assert effective_monitored_stamp(_eval_spec, requested=monitored) == (monitored_stamp, kkt_route)
    print(f"monitored lines in effect: {'all' if monitored_stamp is None else monitored_stamp}; "
          f"kkt route: {kkt_route}", flush=True)
    train_params = train_env_obj.make_params(episode_len=1)
    eval_params = eval_env_obj.make_params(episode_len=1)
    print(f"train days {len(tr_days)}   eval days {len(ev_days)}   "
          f"disjoint {not set(tr_days) & set(ev_days)}", flush=True)

    # `SHARED` is the carrier of the "one configuration for all three markets,
    # not tuned per market" claim, so it is never edited here. A pilot may
    # override the batch to price compilation separately, and any run that does
    # is stamped `off_shared` and excluded from the report by that stamp.
    #
    # **The base is this market's, not the module-level one.** `horizon` is
    # declared per market (`hyperparams.HORIZON`): 01's rollout is four
    # one-clearing episodes where 02/03's is one 48-half-hour episode. Reading
    # `SHARED.horizon` here would compare 01 against 02's value and stamp every
    # 01 product `off_shared` for running its own configuration -- which is what
    # it did until 2026-09-07, and what those products still carry (a stamp can
    # only be written by the writer, so the existing ones are not restamped).
    base = shared_for(MARKET) if args.algo == "ippo" else sac_shared_for(MARKET)
    cfg = base
    if args.weight_decay is not None and args.algo != "ippo":
        raise SystemExit("--weight-decay belongs to the IPPO optimiser chain; "
                         "SACConfig has no such field")
    if args.weight_decay is not None:
        import dataclasses as _dc
        cfg = _dc.replace(cfg, weight_decay=args.weight_decay)
        print(f"weight_decay = {args.weight_decay} (SHARED has "
              f"{SHARED.weight_decay})", flush=True)
    off_shared = {}
    if args.n_envs is not None or args.horizon is not None:
        import dataclasses
        changes = {}
        # the flag being present is not the test -- the value differing is.
        # A flag that repeats the shared value changes nothing, and stamping
        # that product off-shared would exclude a run that is on it.
        for name, given in (("n_envs", args.n_envs), ("horizon", args.horizon)):
            if given is None:
                continue
            changes[name] = given
            if given != getattr(base, name):
                off_shared[name] = [getattr(base, name), given]
        # replace from `cfg`, NOT from `SHARED`: rebuilding from SHARED here
        # discards any earlier override.  It did exactly that on 2026-08-19 --
        # `--weight-decay 0.2935` printed correctly and was then silently reset
        # to 0.0 by this line, so the log claimed decay while the optimiser had
        # none and the checkpoint recorded 0.0.  Caught because the checkpoint's
        # `hyperparams` was read back rather than trusted.
        cfg = dataclasses.replace(cfg, **changes)
        if off_shared:
            print(f"OFF-SHARED PILOT: {off_shared} -- these products are stamped "
                  f"`off_shared` and must not be reported as the shared "
                  f"configuration", flush=True)
        else:
            print(f"batch flags given but equal to market {MARKET}'s "
                  f"configuration; products stay on-shared", flush=True)

    train_env = unpack_env((train_env_obj, spec))
    bounds = bounds_for(train_env[3])
    key = jax.random.PRNGKey(args.seed)
    key, k_stat, k_init = jax.random.split(key, 3)
    obs_mean, obs_std = observation_statistics(train_env, train_params, k_stat,
                                               cfg.n_envs, cfg.horizon,
                                               env_chunks=args.env_chunks)

    # `--init-params`: continue an earlier run rather than start one.  Every
    # line below PRINTS, because each of them is a way the continuation can
    # differ from the run it claims to continue, and a difference that only
    # exists in the code is one nobody reading the log can see.
    init_archive, saved_hp = None, None
    if args.init_params:
        init_archive = np.load(args.init_params, allow_pickle=True)
        # The archive's OWN statistics, never a fresh fit.  `write_params`
        # records why: a refit standardises with a reference the weights were
        # not fitted against, and the policy that comes back is a different
        # policy that looks like the saved one.  The fit above has already run
        # by this line and its result is discarded here rather than skipped,
        # so the two are directly comparable and the difference is printed.
        fitted_mean, fitted_std = obs_mean, obs_std
        obs_mean = jnp.asarray(init_archive["obs_mean"])
        obs_std = jnp.asarray(init_archive["obs_std"])
        drift = float(jnp.max(jnp.abs(jnp.asarray(fitted_mean) - obs_mean)))
        print(f"init-params: {repo_relative(args.init_params)}", flush=True)
        print(f"  obs_mean/obs_std TAKEN FROM THE ARCHIVE, not refitted; the "
              f"fresh fit this run would otherwise have used differs from it by "
              f"up to {drift:.6e} on the mean and is discarded", flush=True)
        print("  optimizer_state_restored: false -- the archive carries weights "
              "and statistics only, so Adam's first and second moments restart "
              "at zero. That is a real discontinuity in the trajectory, not a "
              "formality, and it is stamped into the curve's meta line.",
              flush=True)
        # The archive's hyperparameters against this run's, field by field.
        # Printed and not enforced: a continuation at a different weight decay
        # is a legitimate experiment, and a continuation that changed one by
        # accident is not, and only the operator can tell which this is.
        # `hyperparams` and `config` carry the same JSON, but only archives
        # written after that key was added have the first.  The 200-iteration
        # pilot of 2026-08-18 has `config` alone, and reading `hyperparams`
        # unconditionally killed the first continuation launched against it.
        # Which key was read is printed, because "no fields differ" means
        # something different when the comparison never happened.
        hp_key = ("hyperparams" if "hyperparams" in init_archive.files
                  else "config" if "config" in init_archive.files else "")
        if not hp_key:
            print("  the archive carries NEITHER `hyperparams` NOR `config`, "
                  "so this run's configuration was NOT compared against the "
                  "one that produced those weights -- treat any agreement "
                  "downstream as unchecked", flush=True)
            saved_hp = None
        else:
            saved_hp = json.loads(str(init_archive[hp_key]))
            print(f"  configuration read from the archive's `{hp_key}` key",
                  flush=True)
    if init_archive is not None and saved_hp is not None:
        now_hp = {k: (list(v) if isinstance(v, tuple) else v)
                  for k, v in vars(cfg).items()}
        diff = {k: (saved_hp.get(k, "<absent>"), now_hp.get(k, "<absent>"))
                for k in sorted(set(saved_hp) | set(now_hp))
                if saved_hp.get(k, "<absent>") != now_hp.get(k, "<absent>")}
        if diff:
            print(f"  HYPERPARAMETERS DIFFER from the archive in {len(diff)} "
                  f"field(s) (not stopping):", flush=True)
            for k, (was, now) in diff.items():
                print(f"    {k}: archive {was!r} -> this run {now!r}",
                      flush=True)
        else:
            print(f"  hyperparameters match the archive field for field "
                  f"({len(now_hp)} fields compared)", flush=True)
        print(f"  archive meta: {json.loads(str(init_archive['meta']))}",
              flush=True)

    # The parameter LAYOUT, checked before any leaf is read.  A shared archive
    # and a per-agent one have the SAME number of leaves and differ only in a
    # leading axis, so the leaf-count check further down cannot separate them
    # and an unflatten would succeed and hand back a different policy.  The
    # per-leaf shape check does separate them, but only as a side effect of the
    # shapes happening to differ; this is the check that asks the question.
    if init_archive is not None:
        am = (json.loads(str(init_archive["meta"]))
              if "meta" in init_archive.files else {})
        declared = am.get("per_agent_params")
        if declared is None:
            print(f"  the archive does not declare `per_agent_params`, so it "
                  f"was written before that stamp existed; no driver could "
                  f"produce a per-agent archive then, but this run has NOT "
                  f"verified the layout from the file -- the per-leaf shape "
                  f"check below is the only thing separating the two layouts "
                  f"here", flush=True)
        elif bool(declared) != bool(args.per_agent_params):
            raise SystemExit(
                f"{args.init_params} declares per_agent_params={bool(declared)} "
                f"and this run was asked for {bool(args.per_agent_params)}. The "
                f"two layouts hold the same number of leaves and differ only in "
                f"a leading agent axis, so loading one as the other does not "
                f"fail -- it produces a different policy that looks like the "
                f"saved one. Pass "
                f"{'--per-agent-params' if declared else 'no --per-agent-params'} "
                f"or point at the other archive.")
        else:
            print(f"  parameter layout matches: per_agent_params="
                  f"{bool(declared)} in the archive and in this run",
                  flush=True)

    # one concrete observation to check the normalisation against, drawn the
    # same way the evaluation path draws it
    train_obs_sample = train_env[0](jax.random.PRNGKey(args.seed + 7071),
                                    train_params)[0]
    # `shed_mwh` is carried out of the rollout because the *sampled* shed
    # count has no other source.  The evaluation path reports the count under
    # the policy mean; a sampled policy is more dispersed and can reach shed
    # cells the mean never does, so "the untrained baseline is zero cells" is
    # a statement about one regime only and cannot stand in for the other.
    # Normalisation health, printed before anything is trained.
    #
    # `SharedActorCritic`'s `init_scale` is section 17's prescription and its
    # docstring states the precondition: it was measured "with the observation
    # standardised".  Standardised means the normalised observation is of order
    # one.  `observation_statistics` floors the standard deviation only at
    # `std > 1e-8`, so a coordinate whose sample standard deviation is small but
    # above that floor passes the guard and still divides a real deviation by a
    # nearly-zero number.  Measured 2026-08-18 at `n_envs=1, horizon=1`: the
    # normalised observation reached 3.1e7, the first Dense layer's
    # pre-activation reached 1.7e7, and `tanh` saturated so completely that the
    # cross-agent difference went from 3.99 before it to EXACTLY 0.0 after it --
    # every one of the 66 units then received an identical markup.
    #
    # That measurement is at a one-environment, one-step sample and may well be
    # an artefact of it, which is the whole reason this prints the number rather
    # than asserting on it: the quantity that matters is what the *pilot's* own
    # sample produces, and only the pilot's own run can report that.
    nrm = (train_obs_sample - obs_mean) / obs_std
    nrm_max = float(jnp.max(jnp.abs(nrm)))
    print(f"normalisation: max |(obs - mean) / std| = {nrm_max:.3e} over a "
          f"{cfg.n_envs} x {cfg.horizon} sample; order 1 is what `init_scale` "
          f"assumes, and tanh saturates hard above about 1e1", flush=True)
    if nrm_max > 1e2:
        print("  WARNING: the observation is NOT standardised at this sample "
              "size. The first tanh layer will saturate and the shared policy "
              "cannot distinguish agents; `action_spread` will be 0 by "
              "construction rather than by the policy's choice.", flush=True)

    if args.per_agent_params:
        print(f"PER-AGENT PARAMETERS: {int(train_env[3]['n_agents'])} independent "
              f"copies of the network. Products of this run are stamped "
              f"`per_agent_params: true` and must not be compared leaf-for-leaf "
              f"against a shared archive.", flush=True)
    if args.algo == "ippo":
        init, iterate = make_ippo(train_env, bounds, cfg, obs_mean, obs_std,
                                  extra_info_keys=("shed_mwh",),
                                  per_agent_params=args.per_agent_params,
                                  env_chunks=args.env_chunks)
    else:
        if args.env_chunks != 1:
            # Accepted-and-dropped is the failure this refusal exists for:
            # `make_sac` has its own rollout and does not take the
            # keyword, so a value here would change nothing and read as if it had.
            raise SystemExit(
                f"--env-chunks={args.env_chunks} is wired through make_ippo "
                f"only; the SAC learner does not take it, so it is refused "
                f"rather than silently ignored")
        # The critic's reward scale, fitted once from the truthful rollout and
        # frozen, like `obs_mean` / `obs_std` above.  `fold_in` rather than a
        # further `split` of `k_stat`, so the IPPO path's key stream is not
        # touched by a branch it never takes.
        import dataclasses as _dc
        scale = float(reward_statistics(train_env, train_params,
                                        jax.random.fold_in(k_stat, 1),
                                        cfg.n_envs, cfg.horizon))
        cfg = _dc.replace(cfg, reward_scale=scale)
        print(f"SAC reward_scale = {scale:.6e} (pooled std of the per-agent "
              f"reward under the truthful action over a {cfg.n_envs} x "
              f"{cfg.horizon} sample); the critic regresses on reward / "
              f"reward_scale", flush=True)
        init, iterate = make_sac(train_env, bounds, cfg, obs_mean, obs_std,
                                 extra_info_keys=("shed_mwh",),
                                 per_agent_params=args.per_agent_params)
    params, tx, opt_state, env_state, env_obs = init(k_init, train_params)
    if init_archive is not None:
        # The tree structure comes from the fresh `init`, not from the file:
        # `write_params` stores `treedef` as a *string*, which no reader can
        # turn back into a treedef.  The leaf ORDER is `tree_flatten`'s and is
        # what the writer enumerated as `p0 p1 ...`, so the two agree only when
        # the network is the one that wrote the archive -- which the shape check
        # below is what verifies.  `opt_state` is left exactly as `init`
        # returned it: zero moments, the discontinuity printed above.
        leaves, treedef = jax.tree_util.tree_flatten(params)
        stored = sorted(k for k in init_archive.files
                        if k.startswith("p") and k[1:].isdigit())
        if len(stored) != len(leaves):
            raise SystemExit(f"{args.init_params} holds {len(stored)} parameter "
                             f"leaves and this network has {len(leaves)}; it was "
                             f"not written by this configuration")
        new = []
        for i, leaf in enumerate(leaves):
            a = np.asarray(init_archive[f"p{i}"])
            if a.shape != tuple(np.shape(leaf)):
                want = tuple(np.shape(leaf))
                # The likeliest cause is named, because from here on it is the
                # likeliest cause: the two parameter layouts differ by exactly
                # one leading axis of length `n_agents`, and every other way of
                # getting here needs the network itself to have changed.
                hint = ""
                if len(a.shape) + 1 == len(want) or len(want) + 1 == len(a.shape):
                    hint = (f"  The shapes differ by one leading axis, which is "
                            f"what separates a shared archive from a per-agent "
                            f"one; check --per-agent-params against the "
                            f"archive's own `per_agent_params` stamp.")
                raise SystemExit(f"{args.init_params}: leaf p{i} has shape "
                                 f"{a.shape}, this network wants {want}.{hint}")
            new.append(jnp.asarray(a))
        params = jax.tree_util.tree_unflatten(treedef, new)
        print(f"  loaded {len(new)} parameter leaves "
              f"({sum(int(np.size(a)) for a in new)} scalars); optimiser state "
              f"is the fresh one from `init`", flush=True)
    step_iter = jax.jit(iterate, static_argnums=(1,))

    # Every iteration is timed and printed, not every twentieth.  Iteration 0
    # carries the whole XLA compilation of a vmapped, scanned day-ahead step and
    # is therefore not the steady state; reporting a mean over all iterations
    # would fold the two together, and on this market compilation is not small.
    # Printing every iteration also makes a long run observable while it runs,
    # which the first attempt at this pilot was not.
    # Evaluation-day observations, drawn once: `collapse_criteria` needs them at
    # every checkpoint and they do not depend on the parameters.
    eval_obs = []
    if args.checkpoint_every:
        for pos in range(len(ev_days)):
            _ck, _cst = open_day(eval_env_obj.reset, eval_params, pos,
                                 lambda s_: int(s_.cursor))
            eval_obs.append(np.asarray(eval_env_obj.get_obs(_cst, eval_params),
                                       np.float64))
        print(f"collapse criteria will be printed at every checkpoint over "
              f"{len(eval_obs)} evaluation days; they PRINT ONLY and never "
              f"stop the run", flush=True)

    #: The curve is flushed at the checkpoint cadence, so that checking the two
    #: products against each other afterwards is one comparison of timestamps.
    curve_every = args.checkpoint_every or 10

    def write_curve(rows, meta_obj):
        """Write the curve so far, atomically.

        It used to be written once, after the loop.  A run that died in the
        middle therefore lost every per-iteration reward while keeping all its
        checkpoints, and from outside the two products look alike: the
        `rl_01_wd052` run of 2026-08-20 was taken by a machine event at
        iteration 111 with eleven checkpoints on disk and no curve at all,
        leaving only the lines the log happens to print.  Incremental writes
        make the loss proportional to the interval instead of total.

        `meta` carries `partial`, because a file holding N rows because the run
        asked for N and a file holding N rows because the run died at N are
        otherwise identical to whoever reads it next.

        The columns come from the rows, never from a list written here: a
        writer that has to be edited in step with the producer will eventually
        not be, and the symptom is a series that exists only in stdout.
        """
        if not args.curve_out or not rows:
            return
        out = Path(args.curve_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        # the temporary name has to end in `.npz` too: `np.savez` appends that
        # suffix itself when the name does not carry it, so a `.npz.tmp` target
        # silently becomes `.npz.tmp.npz` and the rename fails.  03 hit exactly
        # this, and `ast.parse` passed throughout.
        tmp = out.with_name(out.name + ".partial.npz")
        cols = sorted({k for c in rows for k in c})
        np.savez(tmp,
                 **{k: np.array([c.get(k, float("nan")) for c in rows])
                    for k in cols},
                 meta=json.dumps(meta_obj))
        # rename rather than write in place: a crash during the write would
        # otherwise destroy the rows already on disk as well as the new ones
        os.replace(tmp, out)

    #: The JSONL curve's meta line.  It is written BEFORE the first iteration,
    #: so it carries what the run was configured with rather than what it
    #: achieved -- the run point that `write_curve` stamps into the `.npz` at
    #: the end is the other product and cannot exist yet.  A run killed in its
    #: first iteration still leaves a file that says what it was.
    #: The scenario stamp **every** product of this run carries: which case, and
    #: which scenario factors.  Assembled once and consumed by `curve_meta`
    #: below and by both `write_params` call sites, so a checkpoint cannot
    #: disagree with the curve written beside it.  Until 2026-08-28 the
    #: checkpoints carried `market` / `seed` / `iteration` and nothing else, so a
    #: `.npz` on its own could not say which case or which `cap_scale` produced
    #: it -- and the scenario has moved twice (`cap_scale` 0.4 -> 0.60,
    #: `ramp_scale` 0.25 -> 0.50 -> 1.00), which is exactly when an unstamped
    #: archive becomes unreadable rather than merely terse.
    scenario_meta = dict(case=str(meta.get("case")), cap_scale=args.cap_scale,
                         ramp_scale=args.ramp_scale, window=meta.get("window"),
                         #: SAC only: the replay buffer's leaf dtypes as built
                         #: (`pre`/`reward` follow the rollout's precision since
                         #: 2026-09-18); None for IPPO
                         replay_buffer_dtypes=(
                             {k: str(v.dtype) for k, v in opt_state["buffer"].items()}
                             if args.algo == "sac" else None))
    curve_meta = dict(
        market="01 day-ahead wholesale", arm=arm, algo=args.algo,
        seed=args.seed, commit=commit_hash(), commit_dirty=_START_DIRTY,
        # scenario factors
        **scenario_meta, voll=VOLL_IN_EFFECT,
        markup_max=args.markup_max, episode_len=1,
        fixture=path.name, shed_floor=SHED_FLOOR,
        train_days=len(tr_days), eval_days=len(ev_days),
        # every hyperparameter that took effect, plus where the shared set came
        # from; `hyperparams_effective` is the one to read, `PROVENANCE` is the
        # declaration and they differ whenever a flag overrode a field
        hyperparams={k: (list(v) if isinstance(v, tuple) else v)
                     for k, v in vars(cfg).items()},
        hyperparams_provenance=(PROVENANCE if args.algo == "ippo"
                                else SAC_PROVENANCE),
        n_envs=cfg.n_envs, horizon=cfg.horizon, off_shared=off_shared or None,
        #: NOT in `hyperparams` above, and that is the point: it is a keyword of
        #: `make_ippo`, not a field of `IPPOConfig`, so `vars(cfg)` cannot carry
        #: it.  A run recorded without this key is a run whose parameter layout
        #: nobody can read back off the product.
        per_agent_params=bool(args.per_agent_params),
        env_chunks=int(args.env_chunks),
        monitored_lines=monitored_stamp,
        kkt_route=kkt_route,
        iterations_requested=args.iterations,
        start_iteration=args.start_iteration,
        init_params=(repo_relative(args.init_params) if args.init_params
                     else None),
        #: `false` whenever this run continued from an archive, `null` when it
        #: started fresh.  The two are different statements: the second run has
        #: no optimiser state to restore, the first had one and did not.
        optimizer_state_restored=(False if args.init_params else None),
        pilot=bool(args.pilot), **runtime_stamp())
    jsonl = CurveLog(args.curve_out, curve_meta)
    if jsonl.path is not None:
        print(f"per-iteration JSONL -> {repo_relative(jsonl.path)} "
              f"(appended and fsynced every iteration)", flush=True)

    curve, t0 = [], time.time()
    for it in range(args.iterations):
        #: The number this iteration is RECORDED under.  `it` stays the local
        #: index because the flush cadence and the "last iteration" test are
        #: about this process; `gi` is what the curve, the JSONL and the
        #: checkpoint file names carry, so a continuation counts on from the
        #: archive instead of overwriting the run it continues.
        gi = args.start_iteration + it
        t_it = time.time()
        params, opt_state, env_state, env_obs, key, m = step_iter(
            params, tx, opt_state, env_state, env_obs, key, train_params)
        jax.block_until_ready(m["reward_mean"])   # else the timing is dispatch
        dt = time.time() - t_it
        # Sampled regime.  These three are deliberately NOT named `action_mean`
        # / `action_spread` / `shed_cells`: those names belong to the evaluation
        # path, where the action is the policy *mean* and carries no exploration
        # noise.  The two regimes differ by construction, and a table that put
        # them side by side under one name would show that difference as if it
        # were a result -- "dispersion fell" is the shape the false conclusion
        # takes, and it reads exactly like a finding.  Each is comparable only
        # with the same-regime baseline: iteration 0 here, `--iterations 0` there.
        # `spread` is the standard deviation across the 66 agents, averaged over
        # the horizon and environment axes.  `axis=-1` IS the agent axis here
        # and only here: day-ahead declares `action_shape == (n_units,)`, so one
        # agent's action is a single number (`ippo.action_layout`'s docstring
        # names this market as the one case where the last axis is ambiguous).
        # The two-dimensional markets declare `(n_agents, act_dim)`, where
        # `axis=-1` is `act_dim` and this same line would silently reduce over
        # the wrong axis.  **Do not copy this expression to 02 or 03** -- there
        # the agent axis is `-2`, and a spread taken over `act_dim` would still
        # produce a plausible number.
        a = np.asarray(m["step_action"], np.float64)      # (horizon, n_envs, 66)
        sh = np.asarray(m["step_shed_mwh"], np.float64)
        # The optimiser's own five, straight out of `ippo._loss`'s aux, averaged
        # over every epoch and minibatch of this iteration.  They were being
        # returned all along and dropped here; a reward curve alone cannot tell
        # "the policy stopped moving" from "the update stopped being applied",
        # and `approx_kl` / `clip_frac` are the two that separate them.
        #
        # `reward_per_agent` is stored as a whole row, NOT as a mean: the
        # judgement this market's collapse question turns on is whether the 66
        # generators differ from each other, and any reduction over the agent
        # axis is exactly the information that question needs.
        rpa = np.asarray(m["reward_per_agent"], np.float64)
        # The learner's own diagnostics, by name.  IPPO's five are the ones
        # every curve on disk carries; SAC's are its own and share only
        # `entropy`, which is NOT the same quantity (the module docstring of
        # `sac.py`): here it is `-log pi` of the sampled action, there the
        # closed-form surrogate.
        if args.algo == "ippo":
            diag = dict(pg_loss=float(m["pg_loss"]),
                        vf_loss=float(m["vf_loss"]),
                        entropy=float(m["entropy"]),
                        approx_kl=float(m["approx_kl"]),
                        clip_frac=float(m["clip_frac"]))
        else:
            diag = dict(q_loss=float(m["q_loss"]),
                        q_mean=float(m["q_mean"]),
                        target_mean=float(m["target_mean"]),
                        actor_loss=float(m["actor_loss"]),
                        alpha_loss=float(m["alpha_loss"]),
                        alpha=float(m["alpha"]),
                        entropy=float(m["entropy"]),
                        buffer_filled=int(m["buffer_filled"]))
        curve.append(dict(iteration=gi, seconds=dt,
                          reward_mean=float(m["reward_mean"]),
                          costs_mean=float(m["costs_mean"]),
                          unconverged_frac=float(m["unconverged_frac"]),
                          **diag,
                          reward_per_agent=rpa,
                          sampled_action_mean=float(a.mean()),
                          sampled_action_spread=float(a.std(axis=-1).mean()),
                          sampled_shed_cells=int((sh > SHED_FLOOR).sum()),
                          sampled_shed_total_cells=int(sh.size)))
        # append-and-fsync before anything else this iteration does, so a kill
        # during the checkpoint or the criteria below still leaves this row
        jsonl.iteration(curve[-1])
        if args.checkpoint_every and (it + 1) % args.checkpoint_every == 0:
            cdir = Path(args.out_dir) / "checkpoints"
            cdir.mkdir(parents=True, exist_ok=True)
            snap = write_params(cdir / f"seed{args.seed}_iter{gi + 1:04d}.npz",
                                params, obs_mean, obs_std, cfg,
                                seed=args.seed, iteration=gi + 1,
                                per_agent_params=args.per_agent_params,
                                extra_meta=dict(scenario_meta, algo=args.algo))
            print(f"    checkpoint -> {snap.name}", flush=True)
            # A failure here must not take down the run: this is a diagnostic
            # bolted onto hours of training.  It is printed loudly rather than
            # swallowed, because a diagnostic that fails quietly reads exactly
            # like one that passed.
            try:
                if args.algo != "ippo":
                    # the criteria read `SharedActorCritic`'s tanh layers by
                    # name; SAC's actor is ReLU with two heads and the
                    # saturation question is IPPO's.  Skipped, and said so.
                    raise RuntimeError("collapse criteria are defined on "
                                       "the IPPO actor only; skipped for "
                                       f"algo={args.algo}")
                t_cr = time.time()
                cr = collapse_criteria(params, obs_mean, obs_std, eval_obs)
                cr_s = time.time() - t_cr
                print(f"    criteria  {cr_s:.2f} s  "
                      f"full-sat rows/day {cr['full_sat']}  "
                      f"distinct hidden/day {cr['n_hidden']}  "
                      f"distinct actions/day {cr['n_action']}  "
                      f"expH {cr['spectral_entropy']:.2f}  "
                      f"PR {cr['participation']:.2f}  "
                      f"sr {cr['stable_rank']:.2f}  "
                      f"s2/s1 D1 {cr['s2s1_dense1']:.3e} (s1 "
                      f"{cr['s1_dense1']:.2f})  D2 "
                      + ("n/a, 64->1 head"
                         if cr["s2s1_dense2"] != cr["s2s1_dense2"]
                         else f"{cr['s2s1_dense2']:.3e}"), flush=True)
                if any(cr["full_sat"]):
                    print(f"    CRITERION MET (not stopping): fully saturated "
                          f"generators on "
                          f"{sum(1 for v in cr['full_sat'] if v)}"
                          f"/{len(cr['full_sat'])} evaluation days, worst "
                          f"{max(cr['full_sat'])}", flush=True)
            except Exception as exc:                      # noqa: BLE001
                print(f"    criteria FAILED: {type(exc).__name__}: {exc}",
                      flush=True)
        if (it + 1) % curve_every == 0 or it == args.iterations - 1:
            write_curve(curve, dict(
                partial=True, iterations_done=len(curve),
                iterations_requested=args.iterations, seed=args.seed,
                weight_decay=getattr(cfg, "weight_decay", None),
                off_shared=off_shared or None,
                per_agent_params=bool(args.per_agent_params),
                env_chunks=int(args.env_chunks),
                monitored_lines=monitored_stamp,
                kkt_route=kkt_route,
                note="written during training; the write after the loop "
                     "replaces this with the full run_point and partial=False"))
        c = curve[-1]
        print(f"  iter {gi:4d}  {dt:8.2f} s  reward_mean {c['reward_mean']:+.6e}  "
              f"costs_mean {c['costs_mean']:.4e}  unconverged "
              f"{c['unconverged_frac']:.4f}  sampled_action_mean "
              f"{c['sampled_action_mean']:.4f}  sampled_action_spread "
              f"{c['sampled_action_spread']:.4f}  sampled_shed_cells "
              f"{c['sampled_shed_cells']}/{c['sampled_shed_total_cells']}",
              flush=True)
        if args.algo == "ippo":
            line = (f"           pg {c['pg_loss']:+.4e}  vf {c['vf_loss']:.4e}  "
                    f"ent {c['entropy']:+.4f}  approx_kl {c['approx_kl']:+.3e}  "
                    f"clip_frac {c['clip_frac']:.4f}  ")
        else:
            line = (f"           q_loss {c['q_loss']:.4e}  q_mean "
                    f"{c['q_mean']:+.4e}  actor {c['actor_loss']:+.4e}  alpha "
                    f"{c['alpha']:.4e}  ent {c['entropy']:+.4f}  buffer "
                    f"{c['buffer_filled']}  ")
        print(line + f"reward_per_agent "
              f"[min {c['reward_per_agent'].min():+.3e}, max "
              f"{c['reward_per_agent'].max():+.3e}, n "
              f"{c['reward_per_agent'].size}]", flush=True)
    jsonl.close()
    wall = time.time() - t0
    per_iter = wall / max(args.iterations, 1)
    steady = [c["seconds"] for c in curve[1:]]
    steady_s = float(np.median(steady)) if steady else float("nan")
    # the batch each of those seconds bought, because a per-iteration time
    # without it is not comparable with any other market's
    batch = cfg.n_envs * cfg.horizon
    # Peak device memory, read from the device rather than from `nvidia-smi`.
    # Under `XLA_PYTHON_CLIENT_PREALLOCATE=false` the two agree; under the
    # default preallocation `nvidia-smi` reports the *pool*, which is the
    # fraction this process was told to reserve and therefore says nothing
    # about what the run needed -- the number it prints is the one I set.
    # Measured 2026-08-18: memory scales with `n_envs` alone (the vmap axis),
    # not with `n_envs * horizon`; the dominant buffers are the per-period
    # Newton systems, f64[n_envs, 24, 161, 161] and f64[n_envs, 162, 24, 162].
    try:
        st = jax.local_devices()[0].memory_stats() or {}
        peak = st.get("peak_bytes_in_use")
        limit = st.get("bytes_limit")
    except Exception:
        peak, limit = None, None
    # Sizes of the two things whose bytes are known exactly, computed inside
    # the guard for the reason the checkpoint criteria give: a diagnostic
    # bolted onto hours of training must not be able to end the run.
    pb = ob = None
    try:
        pb = sum(int(np.asarray(x).size) * int(np.asarray(x).dtype.itemsize)
                 for x in jax.tree_util.tree_leaves(params))
        ob = sum(int(np.asarray(x).size) * int(np.asarray(x).dtype.itemsize)
                 for x in jax.tree_util.tree_leaves(opt_state)
                 if hasattr(x, "shape"))
        if args.algo == "sac":
            # the replay buffer rides in the learner state; it is the one
            # thing here that scales with `buffer_size` and it is reported on
            # its own so the "optimiser state" share is not read as weights
            bb = sum(int(np.asarray(x).size) * int(np.asarray(x).dtype.itemsize)
                     for x in jax.tree_util.tree_leaves(opt_state["buffer"])
                     if hasattr(x, "shape"))
            print(f"replay buffer {bb / 2**20:.1f} MiB "
                  f"({cfg.buffer_size} env-steps); it is included in the "
                  f"optimiser-state figure below", flush=True)
    except Exception as exc:                              # noqa: BLE001
        print(f"  parameter/optimiser byte count FAILED: "
              f"{type(exc).__name__}: {exc}", flush=True)
    if peak is not None:
        # `bytes_limit` is the POOL, i.e. whatever `XLA_PYTHON_CLIENT_MEM_FRACTION`
        # reserved; `peak_bytes_in_use` is the demand.  Both are printed because
        # a peak without the limit it sat inside cannot be compared against a
        # run that reserved a different fraction.
        # The split matters because the parameters are the ONLY thing
        # `--per-agent-params` multiplies by `n_agents`.  If the peak barely
        # moves while `pb` moves 66-fold, the footprint is in the update's
        # activations and not in the weights, and those two have different
        # consequences for what can be run.
        share = ("" if pb is None or ob is None else
                 f"; parameters {pb / 2**20:.2f} MiB + optimiser state "
                 f"{ob / 2**20:.2f} MiB = {(pb + ob) / 2**20:.2f} MiB, i.e. "
                 f"{100.0 * (pb + ob) / peak:.2f}% of the peak; the remaining "
                 f"{(peak - pb - ob) / 2**20:.1f} MiB is everything else")
        print(f"peak device memory {peak / 2**30:.4f} GiB in use"
              + (f" out of a {limit / 2**30:.2f} GiB pool "
                 f"({100.0 * peak / limit:.1f}% of it)" if limit else "")
              + share, flush=True)
        print(f"  = {peak / 2**20 / max(cfg.n_envs, 1):.1f} MiB per environment "
              f"(measured 2026-08-18 under SHARED parameters to scale with "
              f"n_envs, not with n_envs x horizon; per-agent parameters "
              f"multiply the parameter axis by n_agents and that claim has to "
              f"be re-taken, not inherited)", flush=True)
    # `--iterations 0` is the untrained baseline: the loop above does not run,
    # the weights stay at their initialisation, and everything below is the
    # ordinary evaluation path.  It is deliberately this flag and not a separate
    # script -- a second script would drift from the one the pilot actually
    # uses, and then the baseline would not be a denominator for it.  The
    # normalisation is fitted at line 159 from a frozen reference before any
    # training, so it is the same `obs_mean` / `obs_std` a trained run gets.
    iter0 = curve[0]["seconds"] if curve else float("nan")
    if not curve and not args.init_params:
        print("training: 0 iterations -- UNTRAINED BASELINE, weights at "
              "initialisation. Timing fields below are nan by construction, "
              "not by failure.", flush=True)
    elif not curve:
        # `--iterations 0 --init-params X` is a REPLAY, not an untrained
        # baseline: this run trained nothing and its weights are a trained
        # policy's.  The old text said "weights at initialisation" three lines
        # below its own `loaded N parameter leaves`, which is the log saying the
        # opposite of what it had just done.
        print(f"training: 0 iterations, but weights were LOADED from "
              f"{repo_relative(args.init_params)} -- this is a replay of a "
              f"saved policy, NOT an untrained baseline. Timing fields below "
              f"are nan by construction, not by failure.", flush=True)
    else:
        print(f"training: {args.iterations} iterations in {wall:.1f} s | "
              f"iteration 0 (with compilation) {iter0:.2f} s | "
              f"steady-state median {steady_s:.2f} s/iter | batch "
              f"n_envs={cfg.n_envs} x horizon={cfg.horizon} = {batch} env-steps "
              f"per iteration, each step one T={T} three-step clearing",
              flush=True)

    # The trained policy is saved before evaluation, with the three things
    # `make_greedy_action` needs besides the weights: `obs_mean`, `obs_std` and
    # the config.  Weights alone cannot be turned back into a policy -- the
    # normalisation was fitted against a frozen reference and a different one
    # gives a different policy from the same weights.  Without this, asking the
    # policy any question the products do not already answer costs a full retrain.
    ckpt = final_params_path(args.curve_out, args.out_dir, args.seed)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt = write_params(ckpt, params, obs_mean, obs_std, cfg, seed=args.seed,
                        per_agent_params=args.per_agent_params,
                        extra_meta=dict(scenario_meta, algo=args.algo,
                                        iterations=args.iterations,
                                        off_shared=off_shared or None))
    print(f"wrote {ckpt}  (weights + obs_mean/obs_std + config)", flush=True)

    if args.algo == "ippo":
        greedy = make_greedy_action(train_env[3], bounds, cfg, obs_mean, obs_std,
                                    per_agent_params=args.per_agent_params)
    else:
        greedy = make_sac_greedy_action(train_env[3], bounds, cfg, obs_mean,
                                        obs_std,
                                        per_agent_params=args.per_agent_params)
    greedy_j = jax.jit(greedy)
    get_obs = eval_env_obj.get_obs
    step = jax.jit(eval_env_obj.step)
    day_of = lambda st: int(st.cursor)

    run_point = dict(cap_scale=args.cap_scale, ramp_scale=args.ramp_scale,
                     window=meta.get("window"), voll=VOLL_IN_EFFECT,
                     markup_max=args.markup_max, episode_len=1,

                     market="01 day-ahead wholesale", arm=arm, algo=args.algo,
                     #: The layout is part of the arm's identity, so it is in
                     #: the note as well as in its own field: a reader who sees
                     #: only `arm_note` would otherwise be told this run shared
                     #: parameters whatever it actually did.
                     per_agent_params=bool(args.per_agent_params),
                     env_chunks=int(args.env_chunks),
                     monitored_lines=monitored_stamp,
                     kkt_route=kkt_route,
                     arm_note=(("IPPO, " if args.algo == "ippo" else
                                "SAC (ADR-0016; critic on reward / "
                                f"reward_scale={cfg.reward_scale:.6e}), ")
                               + ("one policy PER AGENT"
                                  if args.per_agent_params
                                  else "parameter-shared")
                               + "; evaluation is the mean "
                               "action, not a sample, so the number is the policy "
                               "and not the policy plus exploration. D = 1 makes "
                               "this market a contextual bandit, so gamma and GAE "
                               "do no work here."),
                     shed_floor=SHED_FLOOR, iterations=args.iterations,
                     start_iteration=args.start_iteration,
                     init_params=(repo_relative(args.init_params)
                                  if args.init_params else None),
                     # `false` when this run continued from an archive, `null`
                     # when it started fresh: "had one and did not restore it"
                     # and "had none" are different statements about the run
                     optimizer_state_restored=(False if args.init_params
                                               else None),
                     seed=args.seed, train_days=len(tr_days),
                     eval_days=len(ev_days),
                     seconds_per_iteration_mean=per_iter,
                     seconds_iteration_0_with_compile=iter0,
                     seconds_per_iteration_steady_median=steady_s,
                     env_steps_per_iteration=batch,
                     n_envs=cfg.n_envs, horizon=cfg.horizon,
                     # the stamp goes on the products, not only on the log line
                     # and the checkpoint: a warning nobody re-reads and a
                     # checkpoint nobody opens do not stop a product being
                     # quoted as the shared configuration. This run's first
                     # version printed the warning and left the products
                     # unstamped, which is the failure the stamp exists for.
                     off_shared=off_shared or None,
                     pilot=bool(args.pilot),
                     hyperparams=(PROVENANCE if args.algo == "ippo"
                                  else SAC_PROVENANCE),
                     # PROVENANCE is the shared DECLARATION; this is what
                     # actually ran.  They differ whenever a flag overrides
                     # a field, and a product carrying only the former says
                     # what the configuration was supposed to be.
                     hyperparams_effective={
                         k: (list(v) if isinstance(v, tuple) else v)
                         for k, v in vars(cfg).items()},
                     # the file, not just the scenario numbers: those are
                     # module constants, so they cannot witness which file
                     # was opened. After the batch-3 rename the canonical
                     # and `_seasons` names carry swapped contents, and a
                     # product without this field reads identically either
                     # side of that swap.
                     fixture=path.name,
                     **runtime_stamp(),
                     # `untrained_baseline` answers "are these weights at
                     # initialisation", which is NOT the same question as "did
                     # this run train".  The two coincide everywhere except at
                     # `--iterations 0 --init-params X`, a replay of a saved
                     # policy: that run trains nothing and its weights are a
                     # trained policy's.  Measured 2026-08-28: such a product
                     # carried `untrained_baseline:
                     # true` beside `init_params` naming a trained archive, so
                     # FILTERING ON THIS FIELD RETURNED A TRAINED POLICY'S
                     # PRODUCTS.  Two batches on disk carry the wrong stamp,
                     # both replays and neither a quoted baseline: a
                     # deliberately perturbed smoke artefact (also stamped
                     # `pilot`) and the CPU replay made while measuring this.  No
                     # reported number reads either, which is why closing it
                     # was cheap.  That scope is a NEGATIVE result and so it
                     # carries its window: every `*.npz` carrying a dict `meta`
                     # anywhere in the repository outside `.git`, not a
                     # directory glob.  The first attempt scanned two globs and
                     # concluded "nothing on disk", which is the same statement
                     # over a window too narrow to support it.
                     # The property this field now has, and the one the check
                     # asserts: filtering on it returns only
                     # products whose policy never trained.
                     untrained_baseline=((args.iterations == 0
                                          and not args.init_params) or None),
                     # The OLD predicate, kept under a name that is true of it.
                     # Renaming rather than deleting matters: "this run did no
                     # training" is a real question about a product (it is what
                     # says the timing fields are nan by construction), and
                     # dropping it would make a statement that used to be
                     # readable off a product unreadable.  A replay is then
                     # `evaluated_without_training` with `init_params` set, and
                     # an untrained baseline is the same field with it null.
                     evaluated_without_training=((args.iterations == 0)
                                                 or None),
                     action_regime="policy_mean (deterministic evaluation; the training curve's `sampled_*` fields are the other regime and are not comparable with these)")
    if args.pilot:
        run_point["pilot_note"] = (
            "pilot run: one seed, iteration count chosen to size the real run. "
            "Not for the report.")

    rows = []
    for pos_i, day in enumerate(ev_days):
        k, state = open_day(eval_env_obj.reset, eval_params, pos_i, day_of)
        obs = get_obs(state, eval_params)
        act = greedy_j(params, obs)
        _o, nxt, reward, _c, _dn, info = step(k, state, act, eval_params)
        lmp = np.asarray(nxt.lmp_prev, np.float64)
        prod = float(np.sum(np.asarray(info["cost"], np.float64)))
        shed = [float(info["shed_mwh"])]
        prof = np.asarray(reward, np.float64)
        # 0.0 explicitly: day-ahead's clearing objective is `offer.p +
        # VOLL.s` (`envs/day_ahead/clearing.py:251-253`), two terms and no
        # third. Passing it is the statement that the objective was read,
        # which is why the parameter has no default.
        sc = system_cost(prod, shed, VOLL_IN_EFFECT, 0.0)
        cells = count_cells(shed, SHED_FLOOR)
        write_day(args.out_dir, arm, day, dates[day], system_cost_value=sc,
                  agent_profit=prof, shed_mwh=np.asarray(shed),
                  production_cost=prod, run_point=run_point,
                  extra=dict(mu=float(info["mu"]),
                             converged=bool(info["converged"]),
                             revenue=float(np.sum(np.asarray(info["revenue"],
                                                             np.float64))),
                             shed_cells=cells,
                             integrality_gap=float(info["integrality_gap"]),
                             congested_line_periods=int(
                                 info["congested_line_periods"]),
                             lmp_max=float(np.max(np.asarray(lmp))),
                             lmp_at_voll_cells=int(np.sum(
                                 np.abs(lmp - VOLL_IN_EFFECT) < 1e-6)),
                             action_mean=float(np.mean(np.asarray(act))),
                             # same definition as `sampled_action_spread`
                             # (std across the 66 agents) so the two are
                             # comparable *within* their own regime
                             action_spread=float(
                                 np.asarray(act, np.float64).std()),
                             action_regime="policy_mean",
                             action_min=float(np.min(np.asarray(act))),
                             action_max=float(np.max(np.asarray(act)))),
                  arrays=dict(action=np.asarray(act, np.float64),
                              commitment=np.asarray(info["commitment"], np.int8),
                              lmp=np.asarray(lmp, np.float64)))
        rows.append((day, dates[day], sc, float(prof.sum()), cells,
                     float(np.mean(np.asarray(act)))))
        print(f"  day {day:3d} {dates[day]}  system_cost {sc:14.4e}  "
              f"profit {float(prof.sum()):+14.4e}  shed_cells {cells:2d}  "
              f"mean_action {rows[-1][5]:.4f}", flush=True)

    print(f"\n{len(rows)} days written to {args.out_dir}")
    print(f"total system cost {sum(r[2] for r in rows):.6e}   "
          f"total profit {sum(r[3] for r in rows):+.6e}   "
          f"shed cells {sum(r[4] for r in rows)}")
    if args.curve_out:
        # Every key the curve rows carry, enumerated from the rows themselves
        # rather than listed here.  The previous version named four keys
        # explicitly, so when `sampled_action_mean` / `sampled_action_spread` /
        # `sampled_shed_cells` were added to the rows they printed to the log
        # and never reached the product -- the series existed only in stdout.
        # Adding a column and not checking that the column is in the output is
        # the failure this shape invites, and a writer that has to be edited in
        # step with the producer will eventually not be.
        keys = sorted({k for c in curve for k in c}) if curve else []
        # same writer as the in-loop flush, so the final file cannot differ in
        # shape from the partial ones it replaces
        write_curve(curve, dict(run_point, partial=False,
                                iterations_done=len(curve)))
        print(f"wrote {args.curve_out}  series: {', '.join(keys)}", flush=True)


if __name__ == "__main__":
    main()
