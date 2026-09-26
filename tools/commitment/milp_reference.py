"""The exact MILP reference for the three-step clearing, day by day, in parallel.

The measurement is "how far is the three-step clearing the environment actually
runs from the exact optimum", and the load-bearing word is *actually*.  There are
two chained sweeps in this repository and they are not the same chain:

    the fixture chain   `precommit.py --mode relax`, §6.3 written out in full,
                        (MU)/(MD) rows present.  Frozen as
                        `day_ahead_commitment_29gb_T24_relax.npz`; RTM 46 builds
                        the day-ahead position on top of it
    the step 1' chain   what the environment runs: the same relaxation with the
                        (MU)/(MD) *rows* dropped, their cross-day history kept as
                        variable bounds

They diverge on day 1 and never re-converge: a different relaxation rounds to a
different commitment, which ends the day at a different output, which is the next
day's boundary.  So a per-day gap is only meaningful against the chain whose
claim is being made, and this script measures the second one.  The first is
measured too, but only by LP and step 3, since that is enough to price what
dropping the rows costs as it accumulates.

Three things follow from the construction and all three are the point of it.

**The boundary is shared within a day.**  Day *d*'s exact MILP is solved from the
boundary the step 1' chain reached at the end of day *d-1*, not from a boundary
the MILP chained for itself.  `precommit.py --mode milp --days N` does the latter
and is therefore the wrong instrument here: from its second day onward the
difference it reports carries the drift between two chains as well as the cost of
rounding.

**The days are independent given that chain**, so the reference is a
`multiprocessing` map over days rather than a sequential chain of MILP solves.
The chain itself is LP-cheap and is built up front, in one process.

**The gap is reported split, never as one number.**  Shed load is priced at
VOLL = 10 000 $/MWh, three orders of magnitude above the energy offers, so a few
hundred MWh of shed dominates the objective and a total gap of a few percent can
be almost entirely a scarcity term.  Each day carries the physical cost
(energy + no-load + start-up) and the VOLL term separately.

**Which program is called exact is a flag, not a constant** (`--exact-mumd`), and
the two settings answer different questions.  Step 1' drops the (MU)/(MD) rows.
With `keep`, the exact side carries them (§6.1 as written) and the difference
prices the dropped rows together with the rounding -- what the environment pays
against the model §6.1 describes.  With `drop`, both sides run the same program
and the difference is the optimality gap of the three-step clearing itself.  A
claim of the form "the rounding heuristic is X% from the optimum" needs `drop`:
under `keep` the two sides are different models, and the exact side is the more
constrained one, so the reported figure is not a gap of the heuristic.  The
direction is unfavourable and that is the point of measuring it -- dropping rows
enlarges the feasible set, so the exact optimum can only get cheaper and the gap
can only grow.  Since 2026-08-15 the model this environment describes has no
(MU)/(MD) at all, which is what makes `drop` the setting a gap claim rests on; `keep` is left
as the default so that every product already frozen still names the command that
produced it.

The other asymmetry is deliberate too: `u > 0` is taken literally here because
HiGHS returns a vertex, where the zeros are exact; `relax.ROUND_EPS = 1e-9`
belongs to the interior-point path inside the environment and has no role offline.

The offer basis is true cost, markup 1, which is what M3 is anchored on.  The gap
at other markups has never been measured and is not measured here.

Usage:

    # reproduce three days of the chain against known values before spending
    # hours.  GATE was measured at cap 0.4 / ramp 0.25, so pass that scenario:
    # the gate checks the apparatus, not the adopted run point
    python tools/commitment/milp_reference.py --gate \
        --cap-scale 0.4 --ramp-scale 0.25

    # the adopted run point: four seasonal segments of 15 days,
    # chained inside a segment only, on the program the environment relaxes
    python tools/commitment/milp_reference.py \
        --segment-starts 5 103 187 285 --days 15 \
        --cap-scale 0.6 --ramp-scale 1.0 --exact-mumd drop --workers 60

The derived output path carries the case, the horizon, both scale factors, the
window and `--exact-mumd`, so no two of these land on the same file.  It used to
carry the case and the horizon alone, which is how the second command above came
to overwrite a committed fixture built at another scenario.
"""
import argparse
import datetime as dt
import json
import concurrent.futures as cf
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp

# Every number here comes from HiGHS on the CPU; `powermarketjax` is imported only
# for the case, the demand and the solver constants that go into the metadata.
# Two reasons to keep JAX off the accelerator anyway.  The workers would otherwise
# each claim device memory for nothing, on a machine where another run may hold
# it, and a JAX process that has initialised CUDA cannot be forked -- the child
# dies on `CUDA_ERROR_NOT_INITIALIZED` the first time it touches a jitted
# function, which is what `load_case` does.  `spawn` below covers the second on
# its own; this covers the first and makes the run insensitive to the GPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from precommit import (FIXTURE_DIR, MIP_REL_GAP, build, first_boundary,
                       run_lengths, solve, unpack)

from powermarketjax.case import load_case, scale_min_output
from powermarketjax.envs.day_ahead import clearing as clearing_mod
from powermarketjax.envs.day_ahead import relax as relax_mod
from powermarketjax.envs.day_ahead.clearing import VOLL
from powermarketjax.envs.day_ahead.demand import (demand_for_case, demand_meta,
                                                  demand_pairing)

#: Per-day wall-clock ceiling for the exact solve.  A day that hits it returns its
#: incumbent together with HiGHS's dual bound, which brackets the true optimum
#: rather than pretending to be it; `status` and `mip_gap` are stored per day so
#: such a day is visible in the fixture instead of averaged into it.
TIME_LIMIT = 5400.0

#: Scale on every unit's minimum output, applied to the case once rather than
#: threaded into each operator (`scale_min_output`'s own contract, and the same
#: call `precommit.py` and `da_position.py` make).  The default is the registered
#: value, so every product built before this flag existed was built at 1.0 and
#: keeps its name; `case73rts`'s adopted run point is 0.80, which this module
#: could not express at all until 2026-09-13 -- and, because the scale was not in
#: `derived_name` either, two runs at two scales would have landed on one path,
#: which is exactly what that function's docstring says cannot happen.
P_MIN_SCALE = 1.0

#: The offer basis.  `precommit.build` takes it as `offer = markup * cost`, and it
#: scales **the energy segments only** -- no-load and start-up costs do not move
#: with it -- so changing it changes the weight of commitment cost against energy
#: cost, and the optimal commitment with it.  Default 1.0 is the adopted basis;
#: the docstring above says the gap at other markups has never been measured.
MARKUP = 1.0

#: The fixture chain, for the cost-of-the-dropped-rows comparison only.
FIXTURE_CHAIN = FIXTURE_DIR / "day_ahead_commitment_29gb_T24_relax.npz"

#: `--gate`: what the step 1' chain must reproduce on these days, as a percentage
#: against the exact optimum, measured independently before this script existed.
#: The shed on day 14 is the fingerprint that separates the two chains -- the
#: fixture chain sheds nothing on any of the 60 days.
GATE = {3: dict(lp=-3.80, three=+1.66, milp=3.077541e7, shed=0.0),
        14: dict(lp=-3.40, three=+6.12, milp=3.394159e7, shed=77.4),
        25: dict(lp=-4.67, three=+1.31, milp=2.813650e7, shed=0.0)}

#: The adopted window: four 15-day seasonal segments, chained inside a
#: segment only.  Named here so `--segment-starts` has a value to be compared
#: against rather than a convention to be remembered; the script that merges the
#: commitment fixture's segments carries the same four numbers.
SEGMENT_STARTS = (5, 103, 187, 285)


#: `git rev-parse HEAD`, the dirty flag and the date, resolved at import rather
#: than at write time for the reason `tools/benchmark/evaluation.py:400` gives: a
#: hash written when the file lands answers "what was checked out when the product
#: was saved", which is not a question anyone asks of a product.  A local copy
#: rather than an import because that module pulls in JAX, and every worker here
#: is spawned for a HiGHS solve that must not touch a device.
def _resolve(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=Path(__file__).resolve().parents[2], timeout=10)
        return r.stdout.strip()
    except Exception:                                      # pragma: no cover
        return ""


_START_COMMIT = _resolve(["git", "rev-parse", "HEAD"]) or "unknown"
#: Tracked files only: an untracked scratch file is not a change to the code
#: that ran, and this tree always has some (see `evaluation._resolve_dirty`).
_START_DIRTY = bool(_resolve(["git", "status", "--porcelain",
                              "--untracked-files=no"]))
_START_DATE = dt.date.today().isoformat()

_CASE = None
_ACTUAL = None
_DAYS = None


#: The committed product of the retired run point, named here for one reason: the
#: derived name must never be it.
COMMITTED = "day_ahead_milp_reference_29gb_T24.npz"


def derived_name(case, periods, cap_scale, ramp_scale, starts, days, exact_mumd,
                 p_min_scale=P_MIN_SCALE, markup=MARKUP):
    """The output file name when `--out` is not given.

    Every input that changes what the numbers mean is in the name, because until
    2026-08-29 the name carried the case and the period count and nothing else --
    so the documented `--days 60` command line resolved to `COMMITTED`, a `cap 0.4 /
    ramp 0.25` product over 60 consecutive days, and overwrote it with whatever
    scenario the run was at.  It is silent at write time; `make_env` is where it
    would eventually be noticed, and only for the fixtures `make_env` reads.

    Written as a function rather than an f-string inside `main` so that "two runs
    that mean different things cannot land on one path" is a property a test can
    hold it to.
    """
    window = (f"d{'-'.join(str(int(x)) for x in starts)}x{int(days)}"
              if len(starts) > 1 else f"d{int(starts[0])}n{int(days)}")
    #: Omitted at the default so the products built before the scale existed
    #: keep the names they were written under; any other value is in the name.
    pmin = "" if float(p_min_scale) == P_MIN_SCALE else f"_pmin{float(p_min_scale):g}"
    mk = "" if float(markup) == MARKUP else f"_mk{float(markup):g}"
    return (f"day_ahead_milp_reference_{case}_T{int(periods)}"
            f"_c{float(cap_scale):g}_r{float(ramp_scale):.2f}_{window}{pmin}{mk}"
            f"_mumd-{exact_mumd}.npz")


def drop_mumd(lp):
    """The (MU)/(MD) rows of §6.3 removed, which is what step 1' is.

    They are exactly the inequality rows that touch neither a dispatch nor a shed
    column: `build` lays the variables out as ``[g ; s ; u ; v ; w]``, and every
    other inequality it writes -- line limits, segment width, (RMP) -- carries at
    least one `g` or `s` entry.  Selecting them structurally rather than by index
    keeps this working if `build` grows a row.

    The cross-day history stays.  It lives in `lo`/`hi` as forced-on and
    forced-off periods, not in these rows, so dropping the rows drops the
    within-day minimum up and down times and nothing else.
    """
    A = lp["A_ub"].tocsr()
    n_gs = lp["off_u"]                       # g and s occupy the first columns
    keep = np.diff(A[:, :n_gs].tocsr().indptr) > 0
    out = dict(lp)
    out["A_ub"], out["b_ub"] = A[keep], lp["b_ub"][keep]
    return out


def solve_milp(lp, mip_rel_gap=MIP_REL_GAP, time_limit=TIME_LIMIT):
    """§6.1 exactly: `precommit.solve(integral=True)` plus the bound bookkeeping.

    `solve` asserts `status == 0`, which is right for a sweep that must not
    silently ship a truncated solve and wrong here: a day that runs out of time
    still brackets its optimum, and that bracket is worth keeping.
    """
    t0 = time.perf_counter()
    integrality = np.zeros(len(lp["c"]))
    integrality[lp["integral_slice"]] = 1          # only u; (LOG) pins v and w
    cons = [LinearConstraint(lp["A_eq"], lp["b_eq"], lp["b_eq"]),
            LinearConstraint(lp["A_ub"], -np.inf, lp["b_ub"])]
    r = milp(lp["c"], constraints=cons, integrality=integrality,
             bounds=Bounds(lp["lo"], lp["hi"]),
             options=dict(mip_rel_gap=mip_rel_gap, time_limit=float(time_limit)))
    assert r.x is not None, f"no incumbent: status {r.status}: {r.message}"
    out = unpack(lp, np.asarray(r.x))
    out.update(obj=float(r.fun), seconds=time.perf_counter() - t0,
               status=int(r.status),
               dual_bound=float(r.mip_dual_bound), mip_gap=float(r.mip_gap))
    return out


def _physical(r):
    """The part of the objective that is not shed priced at VOLL."""
    return r["energy"] + r["no_load"] + r["start"]


def three_step(case, demand_t, boundary, kw, drop=True):
    """One day of the three-step clearing from a given boundary.

    ``drop`` selects which relaxation step 1 is: the environment's step 1' when
    true, §6.3 in full -- the fixture chain -- when false.
    """
    lp = build(case, demand_t, **boundary, **kw)
    first = solve(drop_mumd(lp) if drop else lp)
    u_int = (first["u"] > 0.0).astype(np.float64)          # step 2, literally
    third = solve(build(case, demand_t, u_fixed=u_int, **boundary, **kw))
    return lp, first, u_int, third


def chain(case, actual, day_index, kw, drop=True, log=None):
    """Sweep the days, carrying each day's own boundary into the next.

    Returns one record per day holding the boundary that day was solved against,
    which is what the exact solves are handed, and the two objectives that day's
    three-step path produced.
    """
    boundary = first_boundary(case, float(actual[day_index[0], 0]), **kw)
    out = []
    for day in day_index:
        demand_t = actual[day]
        _, first, u_int, third = three_step(case, demand_t, boundary, kw, drop)
        out.append(dict(day=int(day), boundary=boundary, first=first, u_int=u_int,
                        third=third))
        up, down = run_lengths(u_int, boundary["up_time"], boundary["down_time"])
        boundary = dict(p_init=third["p"][:, -1], u_prev=u_int[:, -1],
                        up_time=up, down_time=down)
        if log:
            log(out[-1])
    return out


def _init(case_name, periods, pairing=None, p_min_scale=P_MIN_SCALE):
    """Pool initialiser.  `pairing` is the demand kwargs, built once in `main`.

    It is passed rather than rebuilt here because a worker process gets no
    argparse namespace, and rebuilding it from a default would silently pair a
    non-GB case with whatever that default happened to be.
    """
    global _CASE, _ACTUAL, _DAYS
    _CASE = scale_min_output(load_case(case_name), p_min_scale)
    _, actual, days = demand_for_case(case_name, **(pairing or {}))
    _ACTUAL, _DAYS = actual[:, :periods].astype(np.float64), days


def one_day(job):
    """The exact optimum of one day, from the boundary the step 1' chain reached.

    ``exact_mumd`` selects *which program* is being called exact, and that choice
    decides what the gap is a gap against; see `--exact-mumd`.
    """
    day, boundary, kw, mip_rel_gap, time_limit, exact_mumd = job
    case, demand_t = _CASE, _ACTUAL[day]
    lp = build(case, demand_t, **boundary, **kw)
    exact = solve_milp(drop_mumd(lp) if exact_mumd == "drop" else lp,
                       mip_rel_gap, time_limit)
    # Step 3 under the exact commitment.  The dispatch problem given u is the one
    # either path solves, so this must reproduce `exact["obj"]`; it is the check
    # that the exact commitment is reachable through the environment's own step 3
    # rather than only through the branch-and-bound tree.
    u_milp = np.rint(exact["u"]).astype(np.float64)
    exact3 = solve(build(case, demand_t, u_fixed=u_milp, **boundary, **kw))
    return dict(day=int(day), obj_milp=exact["obj"], obj_milp_step3=exact3["obj"],
                milp_dual_bound=exact["dual_bound"], milp_gap=exact["mip_gap"],
                milp_status=exact["status"], seconds_milp=exact["seconds"],
                phys_milp=_physical(exact), voll_milp=exact["voll"],
                energy_milp=exact["energy"], no_load_milp=exact["no_load"],
                start_milp=exact["start"], shed_milp=exact["shed_mwh"],
                n_on_milp=float(u_milp.sum()),
                commitment_milp=u_milp.astype(np.int8))


def gate(case, actual, kw):
    """Rebuild the step 1' chain and check it against known values on three days.

    The chain is what everything downstream sits on, and it is cheap while the
    exact solves are not, so it is checked first.  The reference values were
    measured independently; reproducing them is what says this script rebuilt the
    same chain rather than a fourth one.
    """
    last = max(GATE)
    print(f"gate: step 1' chain over days 0..{last}")
    rec = chain(case, actual, np.arange(last + 1), kw, drop=True)
    worst = 0.0
    for day, want in sorted(GATE.items()):
        r = rec[day]
        got_lp = 100 * (r["first"]["obj"] / want["milp"] - 1)
        got_three = 100 * (r["third"]["obj"] / want["milp"] - 1)
        worst = max(worst, abs(got_lp - want["lp"]), abs(got_three - want["three"]))
        print(f"  day {day:>3}  lp {got_lp:+.2f}% (want {want['lp']:+.2f}%)   "
              f"three {got_three:+.2f}% (want {want['three']:+.2f}%)   "
              f"shed {r['third']['shed_mwh']:.1f} MWh (want {want['shed']:.1f})")
    print(f"gate worst disagreement {worst:.3f} percentage points "
          f"(the reference values are quoted to 0.01, so <= 0.01 is agreement)")
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", default="29gb")
    ap.add_argument("--demand-floor-mw", type=float, default=None,
                    help="the floor the case's demand series is clipped at, in "
                         "MW.  No default, because `load_rts_demand` and "
                         "`load_nem_demand` have none: it is part of the run "
                         "point, and 29gb takes no floor at all")
    ap.add_argument("--demand-netted", nargs="+", default=None,
                    help="73rts only: which renewable classes to net off "
                         "demand.  Defaults to all four, as `load_rts_demand` does")
    ap.add_argument("--days", type=int, default=60,
                    help="days per segment when --segment-starts is given, "
                         "otherwise the length of the single block")
    ap.add_argument("--start-day", type=int, default=0)
    ap.add_argument("--segment-starts", type=int, nargs="*", default=None,
                    metavar="DAY",
                    help="sweep one block of --days days from each of these day "
                         "indices, each chained from its own cold start and from "
                         "nothing before it; the adopted window is "
                         + " ".join(map(str, SEGMENT_STARTS)))
    ap.add_argument("--periods", type=int, default=24)
    ap.add_argument("--segments", type=int, default=1)
    ap.add_argument("--cap-scale", type=float, default=None,
                    help="if not given, **falls back to the value in the meta of "
                         "the 29gb fixture `FIXTURE_CHAIN`** (0.6); this file has "
                         "no module default of its own; `precommit` defaults to "
                         "0.4 and `da_position` to 0.60.  On fallback the "
                         "start-up line prints `(fallback)`.")
    ap.add_argument("--ramp-scale", type=float, default=None,
                    help="as above, falling back to the fixture meta's 1.0 "
                         "(`precommit` defaults to 0.25).")
    ap.add_argument("--p-min-scale", type=float, default=P_MIN_SCALE,
                    help="scale on every unit's minimum output, applied to the "
                         "case once rather than threaded into each operator.  "
                         "`case73rts`'s adopted run point is 0.80; the default "
                         "is the registered value and leaves the derived name "
                         "unchanged.")
    ap.add_argument("--markup", type=float, default=MARKUP,
                    help=f"offer basis: `offer = markup * cost`, **scaling only the "
                         f"energy segment prices**; no-load and start-up costs do "
                         f"not change with it.  Default {MARKUP} (the adopted "
                         f"basis, and the one M3 is anchored at); a non-default "
                         f"value goes into the product name as `_mk<v>`, and at "
                         f"the default the name is unchanged byte for byte.")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    ap.add_argument("--mip-rel-gap", type=float, default=MIP_REL_GAP)
    ap.add_argument("--exact-mumd", choices=("keep", "drop"), default="keep",
                    help="whether the exact side carries the (MU)/(MD) rows. "
                         "`keep` is §6.1 as written and prices the dropped rows "
                         "together with the rounding; `drop` puts both sides on "
                         "the same program, which is the only form in which the "
                         "difference is an optimality gap of the three-step "
                         "clearing rather than of a different model")
    ap.add_argument("--gate", action="store_true",
                    help="rebuild the step 1' chain and check three days against "
                         "known values, without solving anything exactly")
    ap.add_argument("--exact-days", type=int, nargs="*", default=None,
                    metavar="DAY",
                    help="solve exactly only these day indices; the step 1' "
                         "chain still runs over every day of the window.  The "
                         "two cannot be collapsed: day d's exact program is "
                         "solved from the boundary the chain reached at the end "
                         "of day d-1, so restricting the chain to the wanted "
                         "days instead would hand each of them a cold start, "
                         "which the module docstring measures as a different "
                         "and wrong instrument.  Used to price a scattered "
                         "evaluation set inside a long window")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # the scenario is the fixture chain's, so the two chains differ in the
    # relaxation and in nothing else
    fmeta = json.loads(str(np.load(FIXTURE_CHAIN, allow_pickle=True)["meta"]))
    kw = dict(K=args.segments,
              cap_scale=fmeta["cap_scale"] if args.cap_scale is None else args.cap_scale,
              ramp_scale=fmeta["ramp_scale"] if args.ramp_scale is None else args.ramp_scale,
              markup=args.markup)
    #: **This assertion asks about the fixture, not about this run.**  A first
    #: version of this change made it "agrees with the `--markup` passed", which
    #: would abort every run with `--markup != 1.0` -- **and that is exactly
    #: what the comment above `scenario_source` argues against**: that file is
    #: opened only to read `meta`, its arrays are never read, and both chains
    #: are recomputed from `kw`, so its scenario disagreeing with the flags
    #: **changes no number here**, and refusing it would only abort entirely
    #: correct runs.  M3 being anchored at true cost is a fact about the
    #: fixture and stays as it is; the run's own basis is recorded in meta as
    #: `markup`, and a NOTE is printed below when the two disagree.
    assert fmeta["markup"] == 1.0, "M3 is anchored at true cost"

    # The fallback above is deliberate -- the intent is "follow the fixture
    # chain", and it survives the chain being renamed.  What does not survive is
    # it being silent: the chain is reached through a fixed filename, so the same
    # path can come to hold a different scenario, and a run that inherited the new
    # one would look exactly like a run that inherited the old one.  Two changes,
    # neither of which removes the fallback.
    #
    # First, say out loud which scenario was resolved and where it came from.  It
    # is read back out of `fmeta` rather than echoed from `args`, because that is
    # the only form that shows what the run received rather than what the caller
    # believes it sent.
    src = ("the flags" if args.cap_scale is not None and args.ramp_scale is not None
           else f"{FIXTURE_CHAIN.name} (fallback)")
    print(f"scenario: cap_scale={kw['cap_scale']} ramp_scale={kw['ramp_scale']} "
          f"K={kw['K']} <- {src}; chain fixture {FIXTURE_CHAIN.name} carries "
          f"cap_scale={fmeta['cap_scale']} ramp_scale={fmeta['ramp_scale']}",
          flush=True)

    # Second, do NOT refuse a disagreement here, and the reason is worth stating
    # because refusing looks like the safer choice.  `FIXTURE_CHAIN` is opened for
    # its `meta` and nothing else -- its arrays are never read -- and both chains
    # this script compares are recomputed from `kw` (`prime` with the (MU)/(MD)
    # rows dropped, `full` with them kept).  So a scenario in that file which
    # disagrees with the flags changes no number here; rejecting it would abort
    # runs that are entirely correct, which is what four earlier artifacts
    # were, every one of them produced with flags that disagree with this
    # fixture's meta.
    #
    # What the disagreement does corrupt is the provenance: `meta["fixture_chain"]`
    # names a file, and a reader takes that to mean the chain was built at that
    # file's scenario.  So the product records the resolved scenario, where it came
    # from, and what the named file actually carries, and the three are allowed to
    # differ visibly instead of one of them quietly standing for the others.
    scenario_source = src
    chain_fixture_scenario = dict(cap_scale=float(fmeta["cap_scale"]),
                                  ramp_scale=float(fmeta["ramp_scale"]),
                                  markup=float(fmeta["markup"]))
    if float(kw["markup"]) != float(fmeta["markup"]):
        print(f"NOTE: the run is at markup={kw['markup']} while {FIXTURE_CHAIN.name} "
              f"carries {fmeta['markup']}. Both chains are recomputed at the run's "
              f"markup, so no number below comes from that file; the disagreement is "
              f"recorded in meta['chain_fixture_scenario'].", flush=True)
    if (abs(chain_fixture_scenario["cap_scale"] - float(kw["cap_scale"])) > 1e-12
            or abs(chain_fixture_scenario["ramp_scale"] - float(kw["ramp_scale"])) > 1e-12):
        print(f"NOTE: the run is at cap_scale={kw['cap_scale']} "
              f"ramp_scale={kw['ramp_scale']} while {FIXTURE_CHAIN.name} carries "
              f"{chain_fixture_scenario['cap_scale']} / "
              f"{chain_fixture_scenario['ramp_scale']}. Both chains are recomputed "
              f"at the run's scenario, so no number below comes from that file -- "
              f"only its name appears, as `fixture_chain` in the product meta, and "
              f"`chain_fixture_scenario` records what it actually holds.",
              flush=True)

    _pairing = demand_pairing(args.case,
                              floor_mw=args.demand_floor_mw,
                              netted=args.demand_netted)
    _init(args.case, args.periods, _pairing, args.p_min_scale)
    case, actual, days = _CASE, _ACTUAL, _DAYS

    if args.gate:
        gate(case, actual, kw)
        return

    # One block per segment start.  Each block gets its own cold start and carries
    # nothing across a segment boundary, because the segments are seasons months
    # apart -- day 103 does not continue day 19.  With no `--segment-starts` this
    # is the single block it has always been, from `--start-day`.
    starts = [args.start_day] if args.segment_starts is None else list(args.segment_starts)
    blocks = [np.arange(st, st + args.days) for st in starts]
    idx = np.concatenate(blocks)
    assert idx[-1] < len(days), f"only {len(days)} days available"
    assert len(set(idx.tolist())) == len(idx), "the blocks overlap"

    wall = time.perf_counter()
    print(f"step 1' chain over {len(idx)} days in {len(blocks)} "
          f"{'block' if len(blocks) == 1 else 'blocks'} "
          f"(starts {starts}, chained within a block only)", flush=True)
    log = lambda r: print(f"  day {r['day']:>3} {days[r['day']].date()}  "
                          f"lp {r['first']['obj']:.6e}  three "
                          f"{r['third']['obj']:.6e}  shed "
                          f"{r['third']['shed_mwh']:.1f} MWh  "
                          f"on {int(r['u_int'].sum())}", flush=True)
    prime = [r for b in blocks for r in chain(case, actual, b, kw, drop=True, log=log)]
    print(f"fixture chain over the same days (LP and step 3 only)", flush=True)
    full = [r for b in blocks for r in chain(case, actual, b, kw, drop=False)]
    seconds_chain = time.perf_counter() - wall

    if args.exact_days is None:
        exact_days = set(int(d) for d in idx)
    else:
        exact_days = set(int(d) for d in args.exact_days)
        missing = sorted(exact_days - set(int(d) for d in idx))
        assert not missing, (
            f"--exact-days lists {len(missing)} day(s) outside the chain's "
            f"window: {missing[:8]}. A day the chain never visited has no "
            f"boundary to be solved against")
        print(f"exact solves restricted to {len(exact_days)} of {len(idx)} "
              f"chained days: {sorted(exact_days)}", flush=True)
    jobs = [(r["day"], r["boundary"], kw, args.mip_rel_gap, args.time_limit,
             args.exact_mumd) for r in prime if r["day"] in exact_days]
    print(f"\n{len(jobs)} exact solves, {args.workers} workers, "
          f"time limit {args.time_limit:.0f} s, (MU)/(MD) on the exact side: "
          f"{args.exact_mumd}", flush=True)
    ctx = mp.get_context("spawn")     # not fork: JAX is multithreaded (see the top)
    #: **`ProcessPoolExecutor`, not `mp.Pool`, and the difference is a worker
    #: that dies.**  `Pool.imap_unordered` never re-dispatches the task a
    #: SIGKILLed worker was holding and the parent waits for that result
    #: forever, printing nothing: `_maintain_pool` quietly starts a
    #: replacement, so the worker count still reads right while every process
    #: sits in `futex_do_wait` at 0% CPU.  Measured 2026-09-13 on 813nem: three
    #: blocks hung that way for 105 minutes on a 96%-idle host, losing about
    #: three hours of chain work each, and the only outward sign was a log that
    #: had stopped growing -- which is what a long exact solve looks like too.
    #: `ProcessPoolExecutor` raises `BrokenProcessPool` from `.result()` the
    #: moment a worker exits abnormally, so the run dies where it broke.
    #:
    #: What killed those workers is the second half: **one 813nem exact solve
    #: peaks at `VmHWM` 11.2-11.7 GB**, the parent at 12.3 GB after the chain,
    #: so `--workers 5` on three concurrent blocks asks for about 208 GB and
    #: this 251 GB host OOM-killed into it.  Pick `--workers` from the peak per
    #: solve times the number of blocks you are running at once, not from the
    #: core count.
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx,
                                initializer=_init,
                                initargs=(args.case, args.periods, _pairing,
                                          args.p_min_scale)) as pool:
        futures = {pool.submit(one_day, j): j[0] for j in jobs}
        exact = []
        for fut in cf.as_completed(futures):
            try:
                r = fut.result()
            except cf.process.BrokenProcessPool:
                done = {f for f in futures if f.done() and not f.exception()}
                left = sorted(futures[f] for f in futures if f not in done)
                print(f"\nexact solve pool broke: a worker exited abnormally "
                      f"(OOM is the one seen in practice -- see the comment "
                      f"above for the per-solve peak). {len(done)} of "
                      f"{len(jobs)} days had returned; days still outstanding: "
                      f"{left}. Nothing is written: a product holding a subset "
                      f"of --exact-days would land on the same derived path as "
                      f"the full one and read as complete.", flush=True)
                raise
            exact.append(r)
            p = prime[list(idx).index(r["day"])]
            print(f"[{len(exact)}/{len(jobs)}] day {r['day']:>3} "
                  f"{days[r['day']].date()}  milp {r['obj_milp']:.6e} "
                  f"({r['seconds_milp']:.0f} s, status {r['milp_status']}, "
                  f"gap {r['milp_gap']:.2e})  total "
                  f"{100 * (p['third']['obj'] / r['obj_milp'] - 1):+.2f}%  phys "
                  f"{100 * (_physical(p['third']) / r['phys_milp'] - 1):+.2f}%  "
                  f"shed {r['shed_milp']:.0f}->{p['third']['shed_mwh']:.0f} MWh",
                  flush=True)
    exact.sort(key=lambda r: r["day"])
    seconds_total = time.perf_counter() - wall

    rows = {r["day"]: r for r in exact}
    out = []
    for p, f in zip(prime, full):
        if p["day"] not in exact_days:
            continue          # chained for the boundary, never solved exactly
        r = dict(rows[p["day"]])
        r.update(
            obj_lp=p["first"]["obj"], obj_three=p["third"]["obj"],
            phys_three=_physical(p["third"]), voll_three=p["third"]["voll"],
            energy_three=p["third"]["energy"], no_load_three=p["third"]["no_load"],
            start_three=p["third"]["start"], shed_three=p["third"]["shed_mwh"],
            n_on_three=float(p["u_int"].sum()),
            hamming=float(np.abs(p["u_int"] - r["commitment_milp"]).sum()),
            seconds_three=p["first"]["seconds"] + p["third"]["seconds"],
            commitment_three=p["u_int"].astype(np.int8),
            # the fixture chain on the same day, for the cost of the dropped rows
            obj_lp_fixture_chain=f["first"]["obj"],
            obj_three_fixture_chain=f["third"]["obj"],
            shed_fixture_chain=f["third"]["shed_mwh"],
            n_on_fixture_chain=float(f["u_int"].sum()),
            p_init=p["boundary"]["p_init"], commitment_prev=p["boundary"]["u_prev"],
            up_time=p["boundary"]["up_time"], down_time=p["boundary"]["down_time"])
        out.append(r)

    meta = dict(
        purpose="the exact optimum of each day, solved from the boundary the "
                "step 1' chain reached, so the difference against that chain's "
                "own three-step objective carries no boundary drift",
        chain="step 1' chain: the §6.3 relaxation with the (MU)/(MD) rows dropped "
              "and their cross-day history kept as variable bounds, rounded at "
              "u > 0, closed by step 3 with u fixed, and chained day to day on its "
              "own end-of-day output and run lengths.  Day 0's boundary is §15's "
              "cold start (`precommit.first_boundary`, one period, all-on u_prev). "
              "This is what `envs/day_ahead` runs and it is NOT the chain frozen "
              "in " + FIXTURE_CHAIN.name + ", which keeps the (MU)/(MD) rows; the "
              "two diverge from day 1 onward and the `*_fixture_chain` columns "
              "price that divergence",
        exact=("§6.1 as written, (MU)/(MD) rows present, u binary"
               if args.exact_mumd == "keep" else
               "§6.1 with the (MU)/(MD) rows dropped, u binary -- the exact "
               "optimum of the same program step 1' relaxes, so the difference "
               "against the three-step objective is the cost of the relaxation "
               "and the rounding and of nothing else"),
        exact_mumd=args.exact_mumd, p_min_scale=args.p_min_scale,
        rounding="u > 0, taken literally: HiGHS returns a vertex, where the zeros "
                 "are exact.  relax.ROUND_EPS belongs to the interior-point path "
                 "inside the environment and has no role here",
        mode="milp", case=args.case, n_periods=args.periods,
        segment_starts=[int(x) for x in starts], days_per_segment=args.days,
        # Which days were chained and which were solved exactly are two different
        # sets whenever --exact-days is given, and a reader who saw only one of
        # them would draw the wrong conclusion in either direction: the chain
        # length is what makes each boundary right, the exact set is what the
        # rows below cover.
        chained_days=[int(d) for d in idx],
        exact_days=sorted(int(d) for d in exact_days),
        exact_days_restricted=args.exact_days is not None,
        chained=("within a segment only" if len(blocks) > 1
                 else "straight through the block"),
        commit=_START_COMMIT, commit_dirty=_START_DIRTY, date=_START_DATE,
        n_segments=args.segments, cap_scale=kw["cap_scale"],
        ramp_scale=kw["ramp_scale"], markup=kw["markup"], voll=VOLL,
        offer_basis=fmeta["offer_basis"],
        #: **This sentence was hardcoded to GB until 2026-09-14**, so it read
        #: "realised GB transmission system demand ... `demand.load_gb_demand()`"
        #: on every product this script ever wrote, whatever the case.  Seven
        #: shipped files carry it wrongly: the 73rts reference and the six
        #: 813nem blocks.
        #:
        #: **The arrays in those files are their own case's** -- the check that
        #: settles it is the calendar, not the code: their `dates` are 2020
        #: (RTS) and 2025 (NEM) while the GB window is 2023-07-10..2024-07-08.
        #: The code says which loader it should have named; the dates say which
        #: series actually ran.
        #:
        #: **They are not rebuilt for a label**: a stamp can only be written by
        #: the writer and there is no later place to add it, so the handling of the
        #: seven is to say so here and beside their readings, not to spend hours
        #: re-solving 36 MILPs to change a sentence.
        #:
        #: `fixture_chain` and `chain_fixture_scenario` naming the 29gb file are
        #: **not** the same defect and are left alone -- see the comment above
        #: `scenario_source`: that file is opened for its `meta` and nothing
        #: else, and those two fields exist precisely to record what the named
        #: file carries so it can differ visibly from the resolved scenario.
        **demand_meta(args.case, **_pairing),
        demand=(f"realised system demand from `demand.{demand_meta(args.case, **_pairing)['demand_source']}()`, "
                "aggregated to the run's period length, not the day-ahead "
                "forecast (ADR-0011 as revised 2026-08-15)"),
        mip_rel_gap=args.mip_rel_gap, time_limit=args.time_limit,
        fixture_chain=FIXTURE_CHAIN.name,
        scenario_source=scenario_source,
        chain_fixture_scenario=chain_fixture_scenario,
        solver="scipy.optimize.milp / linprog, HiGHS.  Every number in this "
               "fixture was produced by HiGHS; the accelerator's `ipm` was not "
               "used.  The constants below are recorded so that a re-calibration "
               "there is detectable against what was frozen here, not because "
               "they entered any solve",
        clearing_max_iter=clearing_mod.MAX_ITER,
        relax_max_iter=relax_mod.MAX_ITER,
        relax_round_eps=relax_mod.ROUND_EPS,
        dual_start="cost_norm",
        dates=[str(days[r["day"]].date()) for r in out],
        day_index=[r["day"] for r in out],
        workers=args.workers, scipy=scipy.__version__,
        seconds_chain=seconds_chain, seconds_wall=seconds_total,
        seconds_cpu=float(sum(r["seconds_milp"] + r["seconds_three"] for r in out)))

    stacked = ("commitment_milp", "commitment_three", "p_init", "commitment_prev",
               "up_time", "down_time")
    arrays = {k: np.array([r[k] for r in out]) for k in out[0] if k not in stacked}
    arrays.update({k: np.stack([np.asarray(r[k]) for r in out]) for k in stacked})
    arrays["meta"] = np.array(json.dumps(meta, indent=1))

    path = Path(args.out) if args.out else FIXTURE_DIR / derived_name(
        args.case, args.periods, kw["cap_scale"], kw["ramp_scale"], starts,
        args.days, args.exact_mumd, args.p_min_scale, args.markup)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)

    gap = arrays["obj_three"] / arrays["obj_milp"] - 1
    phys = arrays["phys_three"] / arrays["phys_milp"] - 1
    voll_pp = (arrays["voll_three"] - arrays["voll_milp"]) / arrays["obj_milp"]
    print(f"\nwrote {path}  ({path.stat().st_size / 1e3:.0f} kB)")
    for name, v in (("total   ", gap), ("physical", phys), ("voll pp ", voll_pp)):
        q1, med, q3 = np.percentile(100 * v, [25, 50, 75])
        print(f"{name}  median {med:+.2f}%  q1 {q1:+.2f}%  q3 {q3:+.2f}%  "
              f"mean {100 * v.mean():+.2f}%  max {100 * v.max():+.2f}%")
    print(f"days shedding: three-step {int((arrays['shed_three'] > 1e-6).sum())}, "
          f"exact {int((arrays['shed_milp'] > 1e-6).sum())}, "
          f"fixture chain {int((arrays['shed_fixture_chain'] > 1e-6).sum())}")
    print(f"exact solves not proved optimal: "
          f"{int((arrays['milp_status'] != 0).sum())}")
    print(f"chain {seconds_chain / 60:.1f} min, wall {seconds_total / 60:.1f} min, "
          f"cpu {meta['seconds_cpu'] / 3600:.2f} h")


if __name__ == "__main__":
    main()
