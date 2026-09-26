"""The three things every baseline in the report shares: days, metrics, format.

This is deliberately **not** inside market 02's driver even though
market 02's honest arm is what first walks the path: markets 01 and 03 plug
their arms into the same three definitions, and a definition that lives in one
market's script is a definition three lines will each re-implement slightly
differently.

Nothing here solves anything.  It decides which days are evaluated, what the
pair of reported numbers means, and what a product looks like on disk.
"""
import json
import os
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from powermarketjax.envs.day_ahead.commitment import DEFAULT_CASE as _DEFAULT_CASE

#: Days per seasonal segment and segments per window (the adopted window:
#: four seasonal segments of fifteen days).  Read from the fixture's `meta` where
#: possible; these are the shape the split rule assumes and it refuses to run
#: against a window of another shape rather than silently selecting other days.
SEGMENT_LEN = 15
N_SEGMENTS = 4

#: The evaluation days the report lists (§2.2).  Recorded so that the rule below
#: can be checked against the document rather than trusted: the rule is the
#: definition, this is the cross-check.  **If the window changes these change,
#: and the mismatch should be seen** -- that is the point of keeping both.
REPORT_EVAL_DATES = (
    "2023-07-10", "2023-07-15", "2023-07-20",
    "2023-10-17", "2023-10-22", "2023-10-27",
    "2024-01-10", "2024-01-15", "2024-01-20",
    "2024-04-18", "2024-04-23", "2024-04-28",
)

#: The one-year window (Y1), the second run point: 365 consecutive days from
#: 2023-07-10 (Mon) to 2024-07-08 (Mon), day indices 5..369 of the 648-day GB
#: inventory.  A **strict superset** of the 60-day window above, so the two run
#: points differ in window length and in nothing else about the data.
YEAR_LEN = 365

#: Y1's 36 evaluation days as offsets into that window, and their dates.
#:
#: **Frozen rather than recomputed at import.**  The 60-day rule is three lines
#: because four equal segments make every stratum the same size; a 365-day
#: window has no such symmetry, so Y1's rule is a stratified systematic sample
#: over season x weekday/weekend and needs the calendar, which this module does
#: not read.  `tools/benchmark/y1_eval_split.py` is the rule; it derives these
#: two constants and `--verify` recomputes and compares them, so the pair is
#: auditable rather than trusted.  The criterion is exogenous to everything
#: reported: it reads the calendar and nothing else.
YEAR_EVAL_OFFSETS = (
    7, 19, 22, 37, 48, 51, 60, 68, 77, 92, 97, 106,
    121, 125, 136, 151, 159, 168, 183, 188, 197, 212, 216, 227,
    242, 250, 259, 274, 279, 289, 304, 307, 319, 340, 349, 357,
)
YEAR_EVAL_DATES = (
    "2023-07-17", "2023-07-29", "2023-08-01", "2023-08-16", "2023-08-27",
    "2023-08-30", "2023-09-08", "2023-09-16", "2023-09-25", "2023-10-10",
    "2023-10-15", "2023-10-24", "2023-11-08", "2023-11-12", "2023-11-23",
    "2023-12-08", "2023-12-16", "2023-12-25", "2024-01-09", "2024-01-14",
    "2024-01-23", "2024-02-07", "2024-02-11", "2024-02-22", "2024-03-08",
    "2024-03-16", "2024-03-25", "2024-04-09", "2024-04-14", "2024-04-24",
    "2024-05-09", "2024-05-12", "2024-05-24", "2024-06-14", "2024-06-23",
    "2024-07-01",
)


#: `case73rts`'s window: the whole of calendar 2020, 366 days, no start offset.
RTS_LEN = 366

#: `case813nem`'s window: 365 days from day 5 of the 370-day inventory, which is
#: 2025-02-01 to 2026-01-31 -- whole months, which is what its split was derived
#: on.  **Its length is GB's**, which is the one collision in this table and the
#: reason `split_days` takes a case at all: keyed on length alone, a NEM window
#: would silently receive GB's thirty-six offsets.
NEM_LEN = 365

#: The two new cases' evaluation days, offsets into each case's own **window**
#: and the dates at those offsets.
#:
#: **Derived by `tools/benchmark/rts_eval_split.py` and
#: `tools/benchmark/nem_eval_split.py`, which run the same rule `y1_eval_split.py`
#: runs for GB** -- they import `strata` / `allocate` / `systematic` from it
#: rather than transcribing them, so "the same criterion was used on all three
#: cases" is a fact about the code.  Both splits were accepted by the author
#: (RTS 2026-09-04, NEM 2026-09-06) as part of those cases' run points.
#:
#: **The offsets are stated here and also in the device that derives them, and
#: that copy is deliberate**: `evaluation` is imported both as a top-level module
#: and as `benchmark.evaluation`, and under the second the devices are not
#: importable by bare name -- so importing them here would break a third of the
#: tree.  `tests/tools/test_case_eval_windows_l0.py` compares the two copies on
#: every run, which is what makes the duplication unable to drift; each device's
#: `--verify` recomputes both from the calendar.
RTS_EVAL_OFFSETS = (
    7, 22, 24, 37, 54, 68, 74, 83, 98, 108, 112, 127,
    137, 142, 156, 170, 171, 183, 197, 200, 210, 223, 228, 237,
    251, 262, 266, 281, 291, 295, 310, 319, 327, 340, 343, 358,
)
RTS_EVAL_DATES = (
    "2020-01-08", "2020-01-23", "2020-01-25", "2020-02-07", "2020-02-24",
    "2020-03-09", "2020-03-15", "2020-03-24", "2020-04-08", "2020-04-18",
    "2020-04-22", "2020-05-07", "2020-05-17", "2020-05-22", "2020-06-05",
    "2020-06-19", "2020-06-20", "2020-07-02", "2020-07-16", "2020-07-19",
    "2020-07-29", "2020-08-11", "2020-08-16", "2020-08-25", "2020-09-08",
    "2020-09-19", "2020-09-23", "2020-10-08", "2020-10-18", "2020-10-22",
    "2020-11-06", "2020-11-15", "2020-11-23", "2020-12-06", "2020-12-09",
    "2020-12-24",
)
NEM_EVAL_OFFSETS = (
    9, 14, 24, 37, 42, 52, 67, 71, 81, 96, 105, 111,
    128, 134, 143, 158, 168, 172, 187, 197, 202, 219, 231, 234,
    249, 260, 263, 278, 288, 293, 312, 322, 326, 341, 350, 356,
)
NEM_EVAL_DATES = (
    "2025-02-10", "2025-02-15", "2025-02-25", "2025-03-10", "2025-03-15",
    "2025-03-25", "2025-04-09", "2025-04-13", "2025-04-23", "2025-05-08",
    "2025-05-17", "2025-05-23", "2025-06-09", "2025-06-15", "2025-06-24",
    "2025-07-09", "2025-07-19", "2025-07-23", "2025-08-07", "2025-08-17",
    "2025-08-22", "2025-09-08", "2025-09-20", "2025-09-23", "2025-10-08",
    "2025-10-19", "2025-10-22", "2025-11-06", "2025-11-16", "2025-11-21",
    "2025-12-10", "2025-12-20", "2025-12-24", "2026-01-08", "2026-01-17",
    "2026-01-23",
)

#: The case the 60-day rule below belongs to, from the package rather than
#: retyped.  **Not** a default for `split_days`'s `case` argument -- that
#: argument has none since 2026-09-12; the name is kept because the 60-day
#: window is a rule written on this case's calendar and on no other.
DEFAULT_CASE = _DEFAULT_CASE

#: `case -> (window length, evaluation offsets, evaluation dates)`.  A case that
#: is not here has no held-out split of its own and raises rather than being
#: given GB's -- the failure this table exists to make impossible, since
#: `case813nem`'s window is exactly as long as GB's.
#:
#: `29gb` has two windows and only the longer one is here: the 60-day one is a
#: rule (`split_days` below) rather than a frozen list, because four equal
#: seasonal segments make every stratum the same size.  **The two new cases have
#: no 60-day window at all** -- both adopted the full calendar year, and the GB 60-day window's
#: four segment starts are declared dates rather than the output of any rule, so
#: there is nothing to run on another calendar.
CASE_EVAL_WINDOWS = {
    "29gb": (YEAR_LEN, YEAR_EVAL_OFFSETS, YEAR_EVAL_DATES),
    "73rts": (RTS_LEN, RTS_EVAL_OFFSETS, RTS_EVAL_DATES),
    "813nem": (NEM_LEN, NEM_EVAL_OFFSETS, NEM_EVAL_DATES),
}


def split_days(n_days, case):
    """`(eval_days, train_days)` as day indices into the fixture's window.

    The rule of report §2.2: segment `s` (1-based) contributes its `s`-th,
    `s+5`-th and `s+10`-th day.  The offsets differ per segment on purpose --
    every segment starts on a Monday, so one shared offset triple would lock the
    weekday and the evaluation set would contain only three of the seven.

    `case` names whose window this is, and **it has no default**: `case813nem`'s
    window is 365 days long, exactly GB's, so a length alone cannot say which
    thirty-six days are held out, and a default of `29gb` (which is what this
    argument carried until 2026-09-12) turned every caller that did not name its
    case into one that silently took GB's days on any 365-day window.  Thirty-two
    callers in `tools/` did not name it.  A caller that has read a fixture has
    the answer at hand in `meta["case"]`; a caller that has not should say which
    case it is running rather than let this function guess.  `None` is refused
    for the same reason, so that `meta.get("case")` on a fixture written without
    one fails here rather than selecting days.

    Returned as plain sorted lists of `int`, disjoint, covering the window.
    """
    n_days = int(n_days)
    if case is None:
        raise ValueError(
            "split_days needs the case whose window this is; None was passed. "
            f"The cases with a held-out split are {sorted(CASE_EVAL_WINDOWS)}, "
            "and a window's length alone cannot name one (case813nem's is "
            "exactly as long as case29gb's)")
    case = str(case)
    if case not in CASE_EVAL_WINDOWS:
        raise ValueError(
            f"case {case!r} has no evaluation split; the three that do are "
            f"{sorted(CASE_EVAL_WINDOWS)}. Falling back to another case's "
            f"held-out days would evaluate on days the policy trained on")
    win_len, offsets, _dates = CASE_EVAL_WINDOWS[case]
    if n_days == win_len:
        # the frozen stratified sample.  A separate branch rather than a
        # generalised rule, because the windows have different symmetries and one
        # rule covering all of them would be a rule matching none.
        ev = sorted(offsets)
        tr = sorted(set(range(n_days)) - set(ev))
        assert not set(ev) & set(tr) and len(ev) + len(tr) == n_days
        return ev, tr
    if case != DEFAULT_CASE or n_days != SEGMENT_LEN * N_SEGMENTS:
        raise ValueError(
            f"the split rule of report §2.2 is written for "
            f"{N_SEGMENTS} x {SEGMENT_LEN} = {SEGMENT_LEN * N_SEGMENTS} days "
            f"on {DEFAULT_CASE}, or for case {case!r}'s {win_len}-day window; "
            f"this window has {n_days}. Selecting days by a rule written for a "
            f"different window would silently evaluate on the wrong ones")
    ev = sorted((s - 1) * SEGMENT_LEN + off
                for s in range(1, N_SEGMENTS + 1)
                for off in (s - 1, s - 1 + 5, s - 1 + 10))
    tr = sorted(set(range(n_days)) - set(ev))
    assert not set(ev) & set(tr) and len(ev) + len(tr) == n_days
    return ev, tr


def check_split_against_report(dates, case):
    """`(ok, selected_dates)` -- does the rule reproduce the recorded list?

    Separate from `split_days` so that changing the window breaks this check
    rather than the selector: the rule stays the definition, and a window change
    shows up here as a disagreement someone has to resolve on purpose.

    `case` selects which recorded list is the answer, and has no default for the
    same reason `split_days`'s has none.  Handing this function a `case813nem`
    window while naming `29gb` is **not** silent: its dates are compared against
    GB's and disagree on every one, so the caller's own refusal fires.  That is
    the loud half; the quiet half was `split_days(365)` called on its own, which
    cannot tell the two apart -- so a driver reading a fixture passes
    `meta["case"]` to both, and neither will run without it.
    """
    ev, _ = split_days(len(dates), case)
    case = str(case)
    got = tuple(str(dates[i]) for i in ev)
    win_len, _offsets, win_dates = CASE_EVAL_WINDOWS[case]
    want = win_dates if len(dates) == win_len else REPORT_EVAL_DATES
    return got == want, got


def open_day(reset, params, day, day_of_state, *, limit=20_000, key_fn=None):
    """Open a specific day through the environment's own `reset`.

    Every market's `reset` draws the day from its key, and none exposes a way to
    ask for one, so the day is selected by searching for a key that lands on it
    (`reset_on_day`, where a market has it, does this properly; this is the
    route for the markets that do not).  The alternative -- building the state by hand -- is rejected
    outright: the opening carry is constructed inside `reset`, and a hand-built
    state would carry one belonging to a different day, which looks exactly like
    a correct evaluation.

    **The search is not trusted; the state is.**  Two hardenings, both required,
    because the property this relies on is real today and guaranteed by nothing:
    each `reset` currently uses its key for the day draw and for nothing else,
    and whoever adds a second use of that key would silently couple it to day
    selection.

    1. the chosen key is fed to `reset` a second time and the day read back from
       *that* state.  Re-reading the state the loop just tested would restate the
       loop condition and could never fail; calling again asks the independent
       question the caller depends on -- is this key's day stable?
    2. the search is bounded and raises; there is no fallback to a nearby day.

    Both make that future change audible: it would either fail to find a key or
    find one that opens elsewhere, and each raises.

    Args:
        reset: the environment's `reset(key, params)`.
        day: the day index wanted.
        day_of_state: `state -> int`, how this market reads the day back out.
            Supplied by the caller because the three markets carry it
            differently; it is the read-back that makes this safe, so it is
            required rather than guessed.
    """
    key_fn = key_fn or (lambda i: jax.random.PRNGKey(i))
    for i in range(limit):
        key = key_fn(i)
        _obs, state = reset(key, params)
        if int(day_of_state(state)) == int(day):
            # Verify by calling `reset` again with the key being returned, not by
            # re-reading the state the loop just tested -- that would restate the
            # loop condition and could never fail.  This asks the independent
            # question the caller actually depends on: does *this key* open
            # *this day*, every time it is used?
            _o2, again = reset(key, params)
            got = int(day_of_state(again))
            if got != int(day):                            # pragma: no cover
                raise AssertionError(
                    f"key opened day {int(day_of_state(state))} once and {got} "
                    f"the next time: `reset` is not a function of its key alone, "
                    f"so key search no longer selects a day")
            return key, state
    raise RuntimeError(
        f"no key in {limit} draws opened day {day}. Either the window does not "
        f"contain it, or `reset` no longer chooses the day from the key alone -- "
        f"in which case key search has stopped meaning 'open day {day}' and this "
        f"must not fall back to a nearby day")


def open_day_start(reset_on_day, params, day, day_of_state, period_of_state,
                   periods_per_day, *, key=None):
    """Open day `day` at *its own first period*, through `env`'s `reset_on_day`.

    The replacement for `open_day`'s key search on markets that expose
    `reset_on_day`.  Both of `open_day`'s hardenings are kept and one
    is added, and all three are asked of the **returned state** rather than of
    the argument that was passed in:

    1. the day is read back out of the state and must equal the day requested;
    2. the offset inside that day is read back and must be zero.  **This is the
       question the key search could not ask.**  A key landing anywhere inside
       day `d` satisfies (1) and still runs the episode across the day boundary,
       which is how market 03's arms came to be scored on different periods;
    3. failure raises.  There is no fallback to whatever period was opened.

    `period_of_state` and `periods_per_day` are required rather than derived,
    for the reason `open_day` requires `day_of_state`: the read-back is the whole
    safety property, and a period length guessed here would make the check agree
    with itself instead of with the environment.  Take `periods_per_day` from
    `spec["periods_per_day"]`, which is the number `reset_on_day` itself used.

    Returns `(key, state)` like `open_day`, so a caller substituting this for it
    changes one line.  The key is returned because the drivers pass it on to
    `step`; `reset_on_day` ignores it.
    """
    key = jax.random.PRNGKey(0) if key is None else key
    _obs, state = reset_on_day(key, params, int(day))
    got = int(day_of_state(state))
    if got != int(day):
        raise AssertionError(
            f"`reset_on_day` was asked for day {int(day)} and opened day {got}: "
            f"either the day arithmetic inside the environment or "
            f"`day_of_state` here is wrong, and both make an evaluation that "
            f"looks correct")
    offset = int(period_of_state(state)) - int(day) * int(periods_per_day)
    if offset != 0:
        raise AssertionError(
            f"day {int(day)} opened {offset} periods into itself rather than at "
            f"its first period. The episode would run {offset} periods past the "
            f"day's end, and the arms of one market would then be scored on "
            f"different periods -- the defect this function exists to make loud")
    return key, state


def system_cost(production_cost, shed_mwh, voll, other_shortfall_cost):
    """The first of the paired numbers: what the day cost the system.

    Production cost plus unserved energy priced at `voll`, plus whatever else
    this market's clearing objective prices.  The first two are added rather
    than reported apart because a dispatch can always look cheap by shedding,
    and a baseline that reported production cost alone would rank a market that
    sheds above one that does not.

    `other_shortfall_cost` has **no default on purpose**, for the same reason
    `count_cells` demands its floor: a default would ship one market's objective
    to the other two.  This function was written with two terms because the two
    markets using it then had two, and the name says "system cost", which reads
    as complete.  Measured 2026-08-18: the ancillary market's objective is
    `offer.p + offer_res.r + VOLL.s + VOLR.u` (`envs/ancillary/clearing.py:355`),
    and its reserve shortfall was silently outside this sum, so an arm that
    bought lower production cost with more reserve shortfall scored *below* the
    truthful arm -- which is impossible for the objective the clearing actually
    minimises, and the impossibility was the only reason anyone looked.

    Verified per market rather than assumed:

      day-ahead   `envs/day_ahead/clearing.py:251-253`  offer.p + VOLL.s   -> 0.0
      real-time   reuses day_ahead.make_clearing                          -> 0.0
      ancillary   VOLR * reserve shortfall, in true-cost terms (the true
                  reserve cost is zero for every provider, so the as-bid
                  `offer_res.r` term is not part of the true system cost)

    Pass `0.0` explicitly where a market has no third term; passing it is the
    statement that its objective was read.
    """
    return (float(np.sum(production_cost))
            + float(voll) * float(np.sum(shed_mwh))
            + float(np.sum(other_shortfall_cost)))


def count_cells(x, floor):
    """How many entries of `x` exceed `floor` -- with `floor` required.

    `floor` has **no default on purpose**.  A default here would ship one
    market's assumption to the other two, which is the defect this function
    exists because of: market 02's `s > 0` counting is correct only because
    `real_time/env.py` has already floored the quantity to exact zero at
    `SHED_FLOOR`, and that step lives in the environment, not in this skeleton.
    Market 03 copied the skeleton, did not copy the step -- it does not appear in
    the code it was reading -- and counted the clearing's `OFF_EPS` dust
    (~1e-21 MWh) as shed, turning 26 cells into 48.

    The same rule the solver parameters follow (`max_iter`, `dual_start`,
    `reg_coef`: no defaults, passed explicitly) for the same reason: a default
    carried across markets is a calibration nobody performed.

    Report the smallest counted value alongside the count where it matters.  On
    market 02's products the smallest non-zero shed is 5.5 MWh against a floor of
    1e-6, five orders clear, so the count is insensitive to the threshold -- that
    is a stronger statement than "none fell in the gap".
    """
    import numpy as _np
    return int((_np.asarray(x) > float(floor)).sum())


def subset_position(position, days):
    """A position fixture restricted to `days`, so `reset` can only draw those.

    Training must draw from the 48 training days and never from the 12 held-out
    ones, but no market's `reset` takes a day set -- it draws uniformly over
    whatever the fixture contains.  Restricting the fixture is therefore the way
    to restrict the draw, and it needs no change to any environment.

    Every day-indexed array is sliced together with `meta["dates"]`, so the
    fixture stays self-describing: a subset whose `dates` still listed all sixty
    would defeat the window check in `make_env` for the wrong reason.
    """
    days = [int(d) for d in days]
    out = {}
    for k, v in position.items():
        if k == "meta":
            continue
        arr = np.asarray(v)
        out[k] = arr[days] if arr.ndim >= 1 and arr.shape[0] == len(
            position["meta"]["dates"]) else arr
    meta = dict(position["meta"])
    meta["dates"] = [str(position["meta"]["dates"][d]) for d in days]
    meta["n_days"] = len(days)
    meta["subset_of"] = "held-out split; see evaluation.split_days"
    out["meta"] = meta
    return out


def converged_reward_split(step_reward, step_converged):
    """How much of the reward came from steps whose solve did not converge.

    Takes two arrays and no market's names, because all three markets take their
    price from a dual and can therefore all be contaminated the same way.

    **The frequency alone does not settle anything.**  A run where 0.5% of steps
    are unconverged reads as reassuring, and it says nothing: what decides
    whether a learning curve describes the policy or the solver is how much of
    the reward those steps carry.  0.5% of steps carrying 0.5% of the reward is
    a limitation to disclose; 0.5% of steps carrying most of it means the curve
    cannot be read at all.  So the frequency and the money are reported
    separately, which is the whole point of the function.

    `share_of_abs_reward` is the main criterion.  Absolute value rather than
    signed sum, because a contaminated step is as damaging when it invents a
    large negative reward as a large positive one, and a signed sum lets the two
    cancel into a comfortable-looking number.

    `max_abs` per group is the second, and it exists to catch the shape the
    share misses: a handful of steps whose total is small against a large
    population but whose individual magnitudes are orders above anything the
    converged steps produce. That is exactly the day this market found, where
    two units on one day returned two orders more profit than any other day.

    Args:
        step_reward: per-step rewards, trailing axes beyond `step_converged`
            are summed (they are the agent axis in every market here).
        step_converged: boolean per step, broadcastable against the leading
            axes of `step_reward`.
    """
    r = np.asarray(step_reward, np.float64)
    c = np.asarray(step_converged)
    while r.ndim > c.ndim:
        r = r.sum(axis=-1)
    if r.shape != c.shape:
        raise ValueError(
            f"step_reward reduces to {r.shape} which does not match "
            f"step_converged {c.shape}; the two must index the same steps, and "
            f"a silent broadcast here would attribute rewards to the wrong ones")
    ok = np.asarray(c, bool)
    bad = ~ok
    total_abs = float(np.abs(r).sum())
    out = dict(
        n_steps=int(r.size),
        n_unconverged=int(bad.sum()),
        frac_unconverged=float(bad.mean()),
        share_of_abs_reward=(float(np.abs(r[bad]).sum() / total_abs)
                             if total_abs > 0 else 0.0),
        max_abs_unconverged=float(np.abs(r[bad]).max()) if bad.any() else 0.0,
        max_abs_converged=float(np.abs(r[ok]).max()) if ok.any() else 0.0,
        mean_unconverged=float(r[bad].mean()) if bad.any() else float("nan"),
        mean_converged=float(r[ok].mean()) if ok.any() else float("nan"),
    )
    # the ratio the second criterion exists for, guarded so an all-converged
    # iteration reports nothing rather than a division
    out["max_abs_ratio"] = (out["max_abs_unconverged"] / out["max_abs_converged"]
                            if out["max_abs_converged"] > 0 and bad.any()
                            else float("nan"))
    return out
def runtime_stamp(requires_x64: bool = True):
    """`platform`, `devices` and `dtype` read off the running process.

    These were eight separate literals (`platform="cpu", dtype="float64"`) until
    2026-08-19, when market 02 audited its own `run_point` key by key and found
    that its last three batches ran on GPU while every product said `cpu`.  The
    first sweep for them found five and reported that as the count; it had
    scanned `tools/benchmark/` because that is where the driver being fixed
    lived, and the remaining three sat in `tools/rt_scenario/`.  Scanning by
    directory finds the instances that share a directory with the one you
    started from; scanning by failure mode finds the rest.  The
    honest arm's stamp happened to be right because that run happened to be on
    CPU, which is the worst kind of correct: it makes the literal look tested.

    `platform` is one of the items every "are these two numbers comparable"
    judgement reads, so a literal there is not a cosmetic problem.  A constant
    cannot be wrong at the moment it is written and cannot be right after that.

    `dtype` is derived the same way and for the same reason: x64 is enabled by
    the caller, and a driver that forgot to enable it would still have written
    "float64".

    Raising rather than recording is deliberate.  Market 02 measured the failure
    mode on 2026-08-19: `jnp.zeros(1).dtype` is `float32` before x64 is enabled
    and `float64` after, so whether this stamp is true depends on which line
    calls it.  Its own driver happens to enable x64 at line 71 and build the
    stamp at line 144, which makes the stamp correct by position rather than by
    design -- move the stamp earlier for any reason and it silently starts
    recording the harness default while the run is still float64.  That is worse
    than the literal it replaces, because a derived value looks measured.

    **`requires_x64=False` is for the one market where x64-off is the design and
    not a forgotten line.**  Every driver that clears an LP or a QP here runs in
    float64, so for those x64 being off is a bug in the caller.  Market 04 is
    not one of them: its clearing is two sorted orders, two cumulative sums and
    a comparison reduction with no solver, `envs/p2p/clearing.py` writes float32
    explicitly at every entry point, and the out-of-package results this
    repository already reports for that market were produced with x64 off.
    Turning x64 on for it would change the numbers rather than record them.  A
    caller passing `False` is asserting that about its own market, so it should
    say so in its own product; the default is unchanged, which is why no
    existing driver's stamp moves.
    """
    if requires_x64 and not jax.config.jax_enable_x64:
        raise RuntimeError(
            "runtime_stamp() was called before jax_enable_x64 was set. The dtype "
            "it would record is the harness default, not this run's, and every "
            "clearing operator here needs float64. Move the call after "
            "`jax.config.update('jax_enable_x64', True)`, or pass "
            "requires_x64=False if this market's arithmetic is float32 by "
            "design (see the docstring).")
    #: `devices` says `cuda:0` on every card, because `CUDA_VISIBLE_DEVICES`
    #: renumbers what the process can see: three jobs pinned to physical cards
    #: 0, 1 and 2 all report `['cuda:0']`, so the products of a three-card batch
    #: cannot say which card produced which -- measured 2026-08-28 on this
    #: round's three market 02 seeds.  The mask itself is recorded beside it, as
    #: a string and not parsed, so `unset` and `""` stay distinguishable (the
    #: second makes every card invisible).  This is provenance, not a claim
    #: about which card the work ran on: with two cards visible the mask does
    #: not say which one a device index landed on.
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    return dict(platform=jax.default_backend(),
                devices=[str(d) for d in jax.devices()],
                cuda_visible_devices=("unset" if mask is None else mask),
                dtype=str(jnp.zeros(1).dtype))


def additive_markup_alpha(case, n_segments, add_m, markup_max):
    """The `kind="markup"` action whose offer is `MC_i + add_m` on every unit.

    The uniform **additive** control group: every unit adds the
    same `add_m` \\$/MWh to its own true marginal cost.  It exists to strip out
    the one thing a uniform **multiplicative** markup carries by construction --
    that an expensive unit's bid rises by more dollars than a cheap one's -- so
    that a displacement can be attributed to the level of the markup rather than
    to its slope across the merit order.

    The action space of all three markets is multiplicative, `offer[i,k,t] =
    alpha[i] * m_env[i,k]` (`envs/day_ahead/action.py`), so the additive
    treatment is reached at `alpha_i = 1 + add_m / MC_i` and **needs no new
    action space** -- which is the claim the additive control group rests on.  This function is
    the single place that arithmetic is written: three drivers call it, and the
    definition that lives in one driver is the definition the other two
    re-implement slightly differently.

    **One segment only.**  At `K > 1` no single `alpha_i` adds the same `add_m`
    to every segment of unit `i`: it would have to be `1 + add_m / m_env[i,k]`,
    which depends on `k`.  This refuses rather than returning an action whose
    offer is `MC + add_m` on one segment and something else on the rest.  All
    three eval drivers run at `K = 1`.

    **The bound is checked, because the offer map does not check it.**  The map
    documents that an action outside `[1, markup_max]` is *mapped rather than
    clipped*, so an `add_m` larger than `(markup_max - 1) * min(MC)` would
    quietly bid outside the action space every learned arm is confined to, and
    the product would look like any other.  Measured on `29gb`: the cheapest
    unit is at 21.9018 \\$/MWh, so `markup_max = 2.0` admits `add_m` up to
    21.9018 \\$/MWh and the requested 5 / 10 / 20 all fit, the last one with
    `alpha_max = 1.9132`.

    Returns `(alpha, info)`.  `alpha` is `(n_units,)` float64; `info` is the
    treatment as it goes into the product stamp -- the `add_m` that produced it,
    the marginal costs it was divided by, and the profile's extent -- because a
    product that cannot state its own treatment is not comparable against
    another one.
    """
    from powermarketjax.envs.day_ahead.clearing import segment_costs

    add_m = float(add_m)
    if not np.isfinite(add_m) or add_m <= 0.0:
        raise ValueError(f"add_m must be a positive finite $/MWh markup, got "
                         f"{add_m!r}; the honest arm is the add_m = 0 case and "
                         f"is run under its own name")
    if int(n_segments) != 1:
        raise ValueError(
            f"the additive arm is defined at one segment per unit and this run "
            f"has n_segments={n_segments}. A scalar markup multiplies the whole "
            f"cost envelope, so adding the same {add_m} $/MWh to every segment "
            f"would need a per-segment multiplier; refusing rather than adding "
            f"it to one segment and something else to the others")
    # the envelope the offer map itself multiplies, not a second reading of the
    # cost coefficients: `make_offer_map` takes `segment_costs(..., monotone=
    # False)` and then accumulates, which at K = 1 is `monotone=True` exactly
    _w, envelope = segment_costs(case, 1)
    mc = np.asarray(envelope, np.float64)[:, 0]
    if mc.min() <= 0.0:
        raise ValueError(
            f"{int((mc <= 0.0).sum())} of {mc.size} units have a non-positive "
            f"marginal cost (min {mc.min():.6g} $/MWh), so `1 + add_m / MC` is "
            f"not the additive treatment for them")
    alpha = 1.0 + add_m / mc
    if float(alpha.max()) > float(markup_max) + 1e-12:
        over = np.flatnonzero(alpha > float(markup_max) + 1e-12)
        raise ValueError(
            f"add_m = {add_m} $/MWh puts {over.size} of {mc.size} units outside "
            f"the action space [1, {markup_max}] (max alpha {alpha.max():.4f} on "
            f"unit {int(np.argmax(alpha))}, whose MC is {mc.min():.4f} $/MWh). "
            f"The offer map maps an out-of-range action rather than clipping it, "
            f"so this would run and the product would look normal. The largest "
            f"add_m this case admits is {(float(markup_max) - 1.0) * mc.min():.4f} "
            f"$/MWh.")
    info = dict(
        add_m=add_m, n_units=int(mc.size),
        mc_min=float(mc.min()), mc_median=float(np.median(mc)),
        mc_max=float(mc.max()),
        alpha_min=float(alpha.min()), alpha_mean=float(alpha.mean()),
        alpha_max=float(alpha.max()),
        alpha_per_unit=[float(x) for x in alpha],
        mc_per_unit=[float(x) for x in mc],
        basis=("uniform ADDITIVE control group, ADR-0016 section 5: every unit "
               "bids MC_i + add_m $/MWh, reached in the multiplicative action "
               "space as alpha_i = 1 + add_m / MC_i. It is NOT the uniform "
               "multiplicative arm and the two must not be read as the same "
               "treatment at a different level: this one adds the same number "
               "of dollars to every unit, that one adds a number proportional "
               "to the unit's own cost"))
    return jnp.asarray(alpha), info


def commit_hash(repo=None):
    """The commit a product was produced at, or `"unknown"` outside a checkout.

    Resolved ONCE, at import, and cached.  Measured 2026-08-19: market 03's
    seed 0 ran 09:26 to 15:07 and its products carry the hash of a commit made at
    10:25 -- i.e. after it started, adding the very channels that run is missing.
    Three products from three revisions all carried the same hash, so the field
    read as evidence of a shared revision while being the opposite.  Stamping at
    write time answers "what was checked out when the file landed", which is not
    a question anyone asks of a product.
    """
    return _START_COMMIT if repo is None else _resolve_commit(repo)


#: The checkout these functions ask about when the caller names none: the tree
#: this module was imported from, NOT the process's working directory.  A job
#: started from a worktree or from any other directory used to stamp that
#: directory's `git` answers onto products built from this tree's code, and the
#: two can be different commits -- the stamp then names a revision the run never
#: ran.  `parents[2]` because this
#: file is `<repo>/tools/benchmark/evaluation.py`.
_IMPORT_TREE = str(Path(__file__).resolve().parents[2])


def _resolve_commit(repo=None):
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, cwd=repo or _IMPORT_TREE, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:                                      # pragma: no cover
        return "unknown"


def _resolve_dirty(repo=None):
    """Whether the checkout had uncommitted changes to TRACKED files at start.

    A hash alone does not identify the code that ran: an edited working tree
    reports the hash of the commit it was edited from.  Without this flag a
    product from a dirty tree is indistinguishable from one built at that commit.

    **The tree asked about is the one this module was imported from**, not the
    process's working directory; see `_IMPORT_TREE`.  A caller may still name
    another checkout explicitly, which is what the tests do.

    **Untracked files do not count.**  This repository is shared by several
    lines at once and its working tree almost always holds untracked scratch
    files, so counting them made every product self-report `commit_dirty=True`
    regardless of the code.  Measured 2026-09-06: all eight products of the 03
    boxed rerun carried `commit_dirty=True` while `git status
    --untracked-files=no` was empty -- the only untracked path was a stray
    notes file at the repository root.  A flag that is always true carries no
    information and, worse, tells a reader to distrust products that were built
    from a clean tree.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, cwd=repo or _IMPORT_TREE, timeout=10)
        return bool(out.stdout.strip())
    except Exception:                                      # pragma: no cover
        return None


_START_COMMIT = _resolve_commit()
_START_DIRTY = _resolve_dirty()


def write_day(out_dir, arm, day_index, date, *, system_cost_value,
              agent_profit, shed_mwh, production_cost, run_point, extra=None,
              arrays=None):
    """One day, one arm, one file.

    Per day because days do not couple in any of the three markets' baselines --
    each is solved or rolled on its own -- so a change to which days are
    evaluated costs a re-run of those days and nothing else.

    `run_point` must carry the four stamp items (`cap_scale`, `ramp_scale`,
    window, `voll`); the arm's name, the commit and whether the tree was dirty
    are added here.  A product without them cannot be compared against another
    product, which is the whole reason these are baselines.

    The commit is the one resolved at import, not at write time; see
    `commit_hash`.
    """
    for item in ("cap_scale", "ramp_scale", "window", "voll"):
        if item not in run_point:
            raise ValueError(f"run_point is missing {item!r}; a baseline product "
                             f"without the four stamp items cannot be compared "
                             f"against any other")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(run_point, arm=arm, day_index=int(day_index), date=str(date),
                commit=commit_hash(), commit_dirty=_START_DIRTY,
                system_cost=float(system_cost_value),
                **(extra or {}))
    path = out_dir / f"{arm}_day{int(day_index):03d}.npz"
    payload = dict(agent_profit=np.asarray(agent_profit),
                   shed_mwh=np.asarray(shed_mwh),
                   production_cost=np.asarray(production_cost))
    # `arrays` is for whatever this market will be asked about next.  The paired
    # metrics answer "how did it do"; they cannot answer "what did it do", and a
    # product that cannot be re-interrogated costs a full retrain per question.
    for k, v in (arrays or {}).items():
        if k in payload:
            raise ValueError(f"{k!r} would overwrite a standard array")
        payload[k] = np.asarray(v)
    np.savez(path, meta=json.dumps(meta), **payload)
    return path


def parse_monitored_lines(text, case):
    """The ``--monitored-lines`` flag of the day-ahead drivers -> what
    `make_env` takes.

    ``"all"`` (the default of every driver) is ``None``: every line limit is
    enforced and both Newton systems go the dense route, which is what every
    archive was produced on.  ``"rated"`` is `clearing.rated_lines`: the lines
    whose rating the case publishes -- on ``case813nem`` 7 of 1 278, the rest
    being 1e6 MW and unreachable -- solved by the low-rank route.  Anything
    else is a comma-separated list of line indices, for controls.
    """
    from powermarketjax.envs.day_ahead.clearing import rated_lines
    text = (text or "all").strip().lower()
    if text == "all":
        return None
    if text == "rated":
        return rated_lines(case)
    return np.unique(np.asarray([int(t) for t in text.split(",") if t.strip()], np.int64))


def refuse_ineffective_lowrank_flags(specs, want_free, lu_batching):
    """Refuse a run whose operators were not built on the low-rank sizing and
    block solver its command line asked for (found in review, 2026-09-17): with
    ``lowrank_free=`` dropped from the `make_env` call,
    ``--lowrank-free-shed 128`` built every operator at (8, 523), stamped 523
    and refused nothing -- 128 was recorded nowhere.  Same shape as
    `effective_monitored_stamp`: the value in effect is read off each
    clearing's ``spec``, never off the flag.

    ``specs`` are ``(name, clearing spec)`` pairs -- the step, boundary and
    eval clearings of market 02, the one clearing of market 03; ``want_free``
    is the sizing the flags asked for, or the case's own when none was given;
    ``lu_batching`` is the flag as given, and ``"auto"`` is not compared (the
    operator resolves it to a platform value that has no counterpart on the
    command line).  Lived in `run_rl_02.py` until 2026-09-17, when it moved
    here, unchanged, for the two ancillary drivers to share; `run_rl_02`
    imports it back under the same name.
    """
    want_free = (int(want_free[0]), int(want_free[1]))
    for name, c in specs:
        eff_free = (int(c["lowrank_free"][0]), int(c["lowrank_free"][1]))
        if eff_free != want_free:
            raise SystemExit(f"--lowrank-free-* asked for {want_free} (requested) but the "
                             f"{name} clearing was built on {eff_free} (effective); "
                             "the sizing is not reaching make_env")
        if lu_batching != "auto" and str(c["lu_batching"]) != lu_batching:
            raise SystemExit(f"--lu-batching asked for {lu_batching} (requested) but the "
                             f"{name} clearing was built on {c['lu_batching']} (effective); "
                             "the flag is not reaching make_env")


def effective_monitored_stamp(spec, requested):
    """What a day-ahead driver stamps for ``monitored_lines`` and ``kkt_route``:
    the set and route the operators were actually built with, read off the
    ``spec`` `make_env` returned, never the parsed flag.

    Measured 2026-09-16 by an injection: a driver whose
    `make_env` call had lost the ``monitored_lines=`` argument still stamped the
    flag's line list into every product, and nothing in the products told the
    two runs apart.  So the stamp is taken from the operator, and a run whose
    operator disagrees with its command line is refused before it starts
    rather than labelled with either value.

    Returns ``(line list or None, route)``.
    """
    c = spec["clearing"]
    eff = c["monitored_lines"]
    eff_list = None if eff is None else [int(i) for i in np.asarray(eff)]
    req_list = None if requested is None else [int(i) for i in np.asarray(requested)]
    if eff_list != req_list:
        raise SystemExit(
            f"--monitored-lines asked for {req_list if req_list is not None else 'all'} "
            f"(requested) but the environment was built on "
            f"{eff_list if eff_list is not None else 'all'} (effective); the flag is "
            "not reaching make_env")
    return eff_list, str(c["kkt_route"])
