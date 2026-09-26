"""The day-ahead position the real-time market settles against — spec §15.

Produces the day-ahead position, a five-tuple, for each market day:

    u      (n_units, 24)      the commitment
    q_da   (n_units, 24)      the day-ahead schedule
    s_da   (24, n_buses)      the day-ahead shed, per bus
    lmp_da (24, n_buses)      the day-ahead price, per bus
    d_da   (24, n_buses)      the day-ahead net bus load

**The position is harvested from the day-ahead environment, not recomputed.**
The commitment is endogenous -- steps 1' and 2 run inside `env.step`
and `u` is a function of the offers -- so the position is the output of that
environment driven over the window, and `q_da`/`lmp_da` are what it actually
cleared.  §4 defines the deviation as `p - q_da` and §8 settles `q_da` at
`lmp_da`, and both symbols mean *what the day-ahead auction cleared*, not a
reproducible reference schedule.  Rebuilding the chain here instead would make
the two markets agree only as long as two implementations stayed aligned; this
repository has lost that bet before (`tests/solvers/reference_ipm.py` records a
reference certifying an implementation at a different trip count).

**Which day-ahead policy** is now a declared scenario choice, because `u` depends
on the offers.  This fixture freezes it at **truthful bidding, markup 1.0**,
matching the offer basis of the pre-commitment fixture.  The consequence belongs
in the paper: the real-time market measures deviation from a *true-cost*
day-ahead position, not from a *strategic* one, so it cannot see how day-ahead
strategic behaviour would move real-time deviations.

**Per-bus shed has to be re-solved.**  `env.step` reports `shed_mwh` as a scalar
and does not expose the per-bus vector, which is the same gap the pre-commitment
fixture had.  So each day is cleared once more with the commitment the
environment just produced, on the same inputs, and `check_position` asserts that
this re-solve reproduces the environment's `award` and `lmp` exactly.  That
assertion is the point: it converts "the position is what the environment
cleared" from a claim into a check.

**Why `s_da` is carried at all.**  §8 forms the day-ahead net injection as

    P_da[t, n] = sum_{i at n} q_da[i, t] + s_da[t, n] - d_da[t, n]

and the money-balance identity of the real-time leg needs `sum_n P_da[t, n] = 0`
so the `lambda_t` term drops out of the price expansion.  Without `s_da` that sum
equals the negated total day-ahead shed, and the identity fails by
`Delta * lambda_t * sum_n s_da[t, n]` on every period whose day-ahead solve shed
anything -- which §17 requires the discriminating scenario of item 47 to do.

Usage:

    python tools/commitment/da_position.py --days 5      # smoke
    python tools/commitment/da_position.py               # the 60-day window

**The three scenario scales are flags, not constants** (2026-09-12).  `case29gb`
runs at the defaults and its command lines are unchanged; the other two cases
have their own adopted run points and cannot be built without naming them:

    python tools/commitment/da_position.py --case 73rts --demand-floor-mw 2500 \
        --cap-scale 0.424 --ramp-scale 1.00 --p-min-scale 0.80 --fixture F
    python tools/commitment/da_position.py --case 813nem --demand-floor-mw 11500 \
        --cap-scale 1.00 --ramp-scale 1.00 --fixture F

`--fixture` is named in both because the derived path carries no scenario, and
the pre-commitment fixture has to have been built at the same one -- `make_env`
refuses a `cap_scale` / `ramp_scale` disagreement and `check_boundary_scenario`
below refuses a `p_min_scale` one.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

import jax.numpy as jnp  # noqa: E402  (must follow the x64 flag)

from powermarketjax.case import load_case, scale_min_output  # noqa: E402
from powermarketjax.envs.day_ahead import (demand_for_case, demand_meta,
                                           demand_pairing, load_commitment,  # noqa: E402
                                           make_clearing, make_env, make_offer_map,
                                           truthful_action)
from powermarketjax.envs.day_ahead.clearing import MAX_ITER, VOLL  # noqa: E402
from powermarketjax.envs.day_ahead.demand import T  # noqa: E402

#: `parse_monitored_lines` / `effective_monitored_stamp` live with the day-ahead
#: drivers, and this file is one directory over; the sibling `gap_report.py`
#: reaches them the same way.  **Both are imported rather than reimplemented**:
#: the stamp carries a refusal added on 2026-09-16 after a driver whose
#: `make_env` call had lost the `monitored_lines=` argument still stamped the
#: flag into every product, with nothing in the products telling the two runs
#: apart.  A second copy of that comparison would be a second chance to lose it.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "benchmark"))
from evaluation import (effective_monitored_stamp,  # noqa: E402
                        parse_monitored_lines)

#: Where this script writes.  Resolved from **this file**, not from the imported
#: package: the editable install points `powermarketjax` at whichever checkout was
#: installed, so a run launched from a worktree otherwise writes into that other
#: checkout.  That happened once (2026-08-14) and dropped a fixture into a
#: checkout shared with other sessions, untracked but not ignored.
FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"

#: Scenario parameters of the day-ahead scenario; declared, never defaulted.
#:
#: These were 0.4 / 0.25 until 2026-08-19, which were the *superseded*
#: scenario values -- so the comment above cited the right decision beside the
#: wrong numbers, and a `run_point` built from these constants would have stamped a
#: scenario the run did not use.  The fixture this tool writes was in fact built
#: at 0.60 / 1.00; the constants had simply not followed it.
CAP_SCALE, RAMP_SCALE = 0.60, 1.00

#: The default of the third scale, `p_min_scale`.  1.0 is the registered case,
#: which is what `case29gb` runs at and therefore what every product already in
#: the repository was built on; `case73rts` adopts 0.80 (2026-09-05).  It is a
#: **default**, not the only value, for the reason the two above are: changing
#: a module-level default changes every line that runs without the flag.
P_MIN_SCALE = 1.0

#: §9.3's action space.  `markup_max` only scales the action map; the position is
#: taken at the truthful action, which is markup 1.0 whatever the ceiling is.
MARKUP_MAX = 2.0

#: Periods of the real-time market per day-ahead period (§15): the day-ahead
#: market consumes the forecast hourly and the realisation is native half-hourly,
#: so one day-ahead hour covers two real-time periods.
PERIODS_PER_HOUR = 2

#: Relative bound on `sum_n P_da[t, n]`, against the period's own demand.
#:
#: **Relative, and derived from measurement.**  An absolute bound cannot be stated
#: here: the quantity is the equality row's own solve residual, so it scales with
#: the demand, and this repository has a recorded failure of writing such a bound
#: by feel (the first tolerances of one such gate sat 60x and 300x above the
#: errors they bounded).  Measured 2026-08-15 over the shipped 60-day
#: window on `case29gb` at K=1, RTX 4500 Ada, float64, `jax` 0.10.2, with the
#: day-ahead calibration (60 Newton steps, `dual_start="cost_norm"`): worst
#: period **2.017e-6** relative, stable across three runs.  The bound sits
#: 2.5x above that.  `main` prints the measured worst on every run, so a window
#: change that moves it is visible rather than silently absorbed.
#:
#: What the gate is separating, measured on the same run: omitting `s_da` moves
#: the sum to **8.447e-3** relative on the shedding days, **4.2e3 times** the
#: residual.  That separation comes from the data -- days 10, 14, 16 and 48 shed
#: on this chain -- and not from a knob, which is the difference between this
#: version and the first one.
BALANCE_RTOL = 5e-6

#: How many times the residual the omission of `s_da` must move the sum, on a
#: shedding day, for the balance gate to be a check rather than a formality.
#:
#: A **ratio** against the measured residual rather than an absolute floor, and
#: the first version got this wrong: comparing against `1e3 * BALANCE_RTOL` left
#: only a 1.7x margin at the measured 8.447e-3, which would have made the gate
#: fail on a quieter window for a reason that has nothing to do with `s_da`.  The
#: ratio moves with the demand scale the way both of its terms do.  Measured on
#: the shipped window: **4.2e3**.  The floor below is two orders under that.
DISCRIMINATION_FACTOR = 1e2

#: Bound on the disagreement between the re-solve and what the environment
#: cleared.  `award` reproduces **exactly** and is held to that; `lmp` does not,
#: and the reason is structural rather than a chain difference: it carries the
#: reduction `(mu+ - mu-) @ PTDF`, whose association order the compiler chooses,
#: and the two solves sit in different `jit` boundaries.  §15 records parallel
#: reductions differing by about one unit in the last place.
#:
#: Measured 2026-08-15, `case29gb`, 60 days, RTX 4500 Ada, float64 solve with a
#: float32 state, `jax` 0.10.2: **7.629e-06 on one run and 1.526e-05 on the
#: next** -- one ULP and then two, on entries spanning about 1e2, where one
#: float32 ULP is 7.6e-6.  **It varies from run to run**, which is the whole
#: reason this bound is derived from the ULP size rather than pinned to a
#: measurement: pinning it to the first observation would have made this gate go
#: red on the second run for a reason having nothing to do with the position.
#: The bound admits eight ULP, far below anything structural -- a price taken
#: from the wrong solve or the wrong day is relative order one.
LMP_ATOL = 6e-5


#: What `p_min_scale` means on a fixture that does not record it.  The field did
#: not exist before 2026-09-05 and every fixture written until then ran the
#: registered case, so absent reads as 1.0.  `precommit.IMPLIED_PRIOR` already
#: records that reading and this mirrors it rather than inventing a second one.
P_MIN_SCALE_IMPLIED_PRIOR = 1.0


def check_boundary_scenario(meta, p_min_scale):
    """Refuse a boundary fixture whose `p_min_scale` differs from this run's.

    **Only `p_min_scale`, and that is the point.**  `make_env` already refuses a
    fixture whose `cap_scale`, `ramp_scale` or `n_segments` disagree with the
    environment it is asked to build (`envs/day_ahead/env.py`, the
    ``fixture was built at cap_scale=...`` raise), so checking those here would
    be a second copy of a check that already fires -- measured: asking `harvest`
    for ``cap_scale=0.50`` against the shipped 0.60 fixture raises there, before
    this function existed.  `p_min_scale` is the one scale that check is blind
    to, because it is applied to the *case* rather than passed to the operators,
    and `make_env` never sees it.

    The failure this catches is silent in the product: the boundary `u` would
    come from a commitment sized for one minimum-output level and every array
    downstream would be well-formed, every gate in `check_position` would pass
    (they compare the re-solve against the environment, and both run this run's
    case), and `meta` would record this run's `p_min_scale` beside a `u` built
    at another.
    """
    got = meta.get("p_min_scale", P_MIN_SCALE_IMPLIED_PRIOR)
    if got is None or abs(float(got) - p_min_scale) > 1e-12:
        raise SystemExit(
            f"the boundary fixture was built at p_min_scale={got} and this run "
            f"asks for {p_min_scale}; the position would chain a commitment "
            f"sized for one minimum-output level into a clearing at another")


def harvest(case, fixture, demand, n_days, n_segments=1, verbose=True,
            monitored_lines=None, progress_every=0,
            cap_scale=CAP_SCALE, ramp_scale=RAMP_SCALE):
    """Drive the day-ahead environment over `n_days` and take the position.

    One continuous chain: `episode_len` is the whole window and `env.step` is
    used rather than `step_auto_reset`, because auto-reset would replace the
    state at the episode end and split the chain in two.

    `cap_scale` and `ramp_scale` are passed to **both** the environment and the
    per-bus re-solve, which is the whole reason they are one argument each here
    rather than read from the module: the re-solve's agreement with what the
    environment cleared is this function's own gate, and two operators built at
    two scales would fail it for a reason that has nothing to do with the chain.
    The third scale, `p_min_scale`, is not here -- it is applied to `case` by the
    caller, because `scale_min_output`'s contract is that it scales the case and
    not each operator.

    Returns the position arrays stacked over days, plus the per-day agreement
    between the re-solve and what the environment cleared.
    """
    env, spec = make_env(case, fixture, demand, n_segments=n_segments,
                         kind="markup", markup_max=MARKUP_MAX,
                         cap_scale=cap_scale, ramp_scale=ramp_scale,
                         monitored_lines=monitored_lines)
    params = env.make_params(episode_len=n_days)
    action = truthful_action(case, n_segments, T, kind="markup")

    # the same clearing operator the environment builds, for the per-bus re-solve
    clear, cspec = make_clearing(case, T, n_segments=n_segments,
                                 cap_scale=cap_scale, ramp_scale=ramp_scale,
                                 period_hours=1.0, max_iter=MAX_ITER,
                                 monitored_lines=monitored_lines)
    #: **The re-solve gate compares two operators, so both have to be on the same
    #: line set.**  `effective_monitored_stamp` reads the set off the operator and
    #: refuses when it disagrees with what was asked; it is called twice here, once
    #: for the environment's own clearing and once for the one built above, because
    #: a flag that reached one and not the other would leave the gate comparing a
    #: 7-line re-solve against a 1 278-line environment and reporting the
    #: difference as a numerical residual.
    mon_stamp, kkt_route = effective_monitored_stamp(spec, requested=monitored_lines)
    _c_stamp, _c_route = effective_monitored_stamp({"clearing": cspec},
                                                   requested=monitored_lines)
    if _c_route != kkt_route:
        raise SystemExit(f"env clearing took the {kkt_route} route but the re-solve "
                         f"operator took {_c_route}")
    if verbose:
        print(f"monitored lines in effect: "
              f"{'all' if mon_stamp is None else mon_stamp}; "
              f"kkt route: {kkt_route}", flush=True)
    offer_map, _ = make_offer_map(case, n_segments, T, kind="markup",
                                  markup_max=MARKUP_MAX)
    share = np.asarray(cspec["demand_share"], np.float64)
    offer64 = jnp.asarray(offer_map(action), jnp.float64)

    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key, params)
    # `reset` draws the start day from `[0, n_days - episode_len]`, so only a
    # full-window episode pins it at zero.  A short run lands somewhere else and
    # that is fine -- the chain is still continuous -- but the day it starts at
    # has to be recorded rather than assumed, since `day_index` is sliced by it.
    start = int(state.cursor)

    step = jax.jit(env.step)
    clear_j = jax.jit(clear)
    out = {k: [] for k in ("u", "q_da", "s_da", "lmp_da", "d_da", "mu",
                           "shed_mwh", "d_award", "d_lmp",
                           "line_dual_up", "line_dual_dn", "shed_dual")}
    wall = time.perf_counter()
    for d in range(n_days):
        # exactly what `step` will feed the operators (env.py: both are cast
        # from the float32 leaves, so taking the float64 series here instead
        # would break the agreement check for a reason unrelated to the chain)
        p_init64 = jnp.asarray(state.p_final, jnp.float64)
        demand64 = jnp.asarray(params.actual[state.cursor], jnp.float64)

        _obs, state, _r, _c, done, info = step(key, state, action, params)
        u = jnp.asarray(info["commitment"], jnp.float64)

        res = clear_j(offer64, u, demand64, p_init64)
        # the agreement check: the re-solve must reproduce what the environment
        # cleared, in the float32 the state stores
        out["d_award"].append(float(np.abs(
            np.asarray(res["award"], np.float32) - np.asarray(state.award_prev)).max()))
        out["d_lmp"].append(float(np.abs(
            np.asarray(res["lmp"], np.float32) - np.asarray(state.lmp_prev)).max()))

        out["u"].append(np.asarray(u, np.float64))
        out["q_da"].append(np.asarray(res["award"], np.float64))
        out["s_da"].append(np.asarray(res["shed"], np.float64))
        out["lmp_da"].append(np.asarray(res["lmp"], np.float64))
        out["d_da"].append(share[None, :] * np.asarray(demand64)[:, None])
        out["mu"].append(float(res["mu"]))
        out["shed_mwh"].append(float(np.asarray(res["shed"]).sum()))
        # the three duals of §7, in the specification's sign convention, from the
        # same solve that produced the price.  Stored rather than recomputed:
        # §8's money-balance identity is assembled from them and §17 requires
        # them to come from the solve that produced the price, not from a second
        # one -- and a quantity already computed is not thrown away.
        for name in ("line_dual_up", "line_dual_dn", "shed_dual"):
            out[name].append(np.asarray(res[name], np.float64))
        #: **With `progress_every=0` the fourth branch is always false and the
        #: first three are untouched, so the default behaviour is unchanged
        #: character for character.**  It is added because the three branches
        #: together can stay silent for a whole window: on 2026-09-17 two
        #: `da_position` jobs ran 67 h and 27 h, and none of the three signals
        #: could answer "which day is it on" -- the log (which is this very
        #: condition), the out-dir age (the npz is written only at the end), and
        #: RSS (measured, not monotone).  The consequence was one near-kill of a
        #: healthy job for "no movement in three days", and one reading of
        #: "day 2/365" as progress.
        #: A keyword rather than the global `args`: tests call this function
        #: directly.
        if verbose and (d < 3 or out["shed_mwh"][-1] > 1e-6 or d == n_days - 1
                        or (progress_every and d % progress_every == 0)):
            print(f"  day {d:3d}  mu {out['mu'][-1]:.2e}  "
                  f"shed {out['shed_mwh'][-1]:8.1f} MWh  "
                  f"lmp [{out['lmp_da'][-1].min():9.2f}, {out['lmp_da'][-1].max():9.2f}]"
                  f"  |dq| {out['d_award'][-1]:.2e}")
        if bool(done) and d != n_days - 1:
            raise SystemExit(f"episode ended early at day {d}; the chain is split")

    seconds = time.perf_counter() - wall
    if verbose:
        print(f"  {n_days} days in {seconds:.1f} s ({seconds / n_days:.2f} s/day)")
    stacked = {k: np.asarray(v) for k, v in out.items()}
    #: **A total, not a per-day rate.**  The timed region starts above the loop
    #: but `jax.jit` compiles lazily, so the *first* iteration pays the XLA
    #: compilation of both `step` and `clear`; `seconds / n_days` is therefore an
    #: intercept plus a slope, and on a one-day run it is almost all intercept.
    #: Measured 2026-09-13, `case73rts`, this host, **both on eight pinned CPU
    #: cores** (`JAX_PLATFORMS=cpu OMP_NUM_THREADS=8`): **8.32 s at `--days 1`**
    #: and **2560.1 s at `--days 366` = 6.99 s/day**.  Two points on the same
    #: machine give a slope of 6.99 s/day and an intercept of **1.33 s**, and that
    #: intercept is the compilation.
    #:
    #: **What stood here until 2026-09-14 put the one-day 8.32 s against the
    #: 3.91 s/day below it and called the difference "about 4.4 s of
    #: compilation".**  Those are different machines -- 8.32 s is CPU, 3.91 s/day
    #: is card 1 -- and two numbers off different machines subtract no better than
    #: two timestamps off different clocks.  The 4.4 s was three times
    #: the real figure.
    #:
    #: The reading that subtraction used was the same window on **card 1 at
    #: `MEM_FRACTION 0.35`: 1432.5 s = 3.91 s/day**.  What is wrong with quoting
    #: it is more than its platform:
    #:
    #: **That 366-day run was on a GPU and it wrote no product** -- the `agrees`
    #: gate failed at 2.00 ULP (device 1, no `npz` written; a repeat on an empty
    #: card took 1442.5 s and failed the same way).  The run that did write the fixture now in the tree took
    #: **2560.1 s = 6.99 s/day on eight pinned CPU cores**.  So the two numbers
    #: do not differ only by platform: one times *this code path*, the other
    #: times *building the product*, and on a GPU the second does not currently
    #: exist.  Quote 6.99 for "how long to build this fixture".
    #: A one-day `seconds` read as a per-day cost also overestimated a 366-day
    #: window in a pricing table built from it.  **That factor depends on which
    #: denominator it is taken against**: nine times as it was reported, against
    #: the GPU rate, and **4.9x** against the CPU slope that actually produced the
    #: product (34.5 s for a one-day product against 6.99).  So the
    #: field name is not enough on its own, and neither is the number without its
    #: `backend` -- which is why the meta above now carries one.
    stacked["seconds"] = seconds
    stacked["start_day"] = start
    stacked["monitored_lines"] = mon_stamp
    stacked["kkt_route"] = kkt_route
    return stacked, spec


def line_flow_overshoot(case, q_da, s_da, d_da, cap_scale=CAP_SCALE):
    """How far the position's recomputed line flows exceed their ratings, MW.

    **This is not the solver's primal residual, and it was first recorded here as
    if it were.**  It is the gap between two things that are not the same
    quantity: the clearing's line rows saw the solver's `x`, while these flows are
    recomputed from `award`, and `award` has had the phantom output of
    de-committed units masked out (`clearing.py` gives them a box of width
    `OFF_EPS` rather than zero, to keep the interior point method's interior, then
    multiplies by `u` on the way out).  So this is the **`OFF_EPS` phantom of
    de-committed units, not carried into the recomputed flow**.

    Measured 2026-08-15 on `case29gb`, 60 days, RTX 4500 Ada, `jax` 0.10.2:

    * it is **exactly proportional to `OFF_EPS`** -- sweeping 1e-3 to 1e-8 keeps
      the mantissa at 5.9896 across four decades;
    * it is **completely insensitive to `max_iter`** -- 60, 80, 120, 200 and 300
      Newton steps give bit-identical values, which is what a constant box width
      predicts and a convergence residual does not;
    * committing every unit, so that no phantom exists, drops it from 5.9896e-03
      to 7.6294e-07 MW, a factor of 7 850.

    **Do not try to shrink it by lowering `OFF_EPS`.**  That is the obvious next
    step and it is the wrong direction: at 1e-6 the dispatch moves by 16.7 MW and
    at 1e-8 by 119.7 MW, because the box stops preserving the strict interior
    (`clearing.py` records the same cliff).  Trading a 6e-3 MW recomputation gap
    for a 120 MW dispatch error is a net loss; the shipped 1e-3 sits three decades
    clear of that cliff.

    It does **not** bias the price: over `OFF_EPS` from 1e-3 to 1e-5, across three
    days chosen for maximum exposure, `max|dlmp|` is 1.4e-7 \\$/MWh.  Folded into
    the day-ahead leg at `q_da ~ 1e3` MW that is 1.4e-4 \\$ per unit-period
    against revenues of order 1e6, so `lmp_da` is clean in the only sense that
    matters downstream.

    A future `||r3||_inf` from the solver (ticket C7) is a *different* quantity --
    smaller, and measuring the solver rather than this masking -- so it will sit
    beside this one rather than replace it.  Measured independently at 1.8e-12
    against a recomputation excess of 4.87e-03 on the same day, a ratio of
    6.1e-10: the clearing satisfies its own rows, and this is not infeasibility.
    The 7.6e-07 that survives when every unit is committed is **not** the
    solver's scale either -- it is what this recomputation leaves once the
    phantom is gone, most likely float64 accumulation in the PTDF product.  It
    was called "the solver's own scale" here before `||r3||` was readable, which
    is what happens to a quantity nobody returns: it gets the nearest name going.

    **Caveat on any shed figure measured beside this one.**  The probes behind
    these numbers ran with `p_init = zeros`, which is the *non-chained* day
    boundary -- item 45 measured that construction shedding far more than the
    chained one (19 881 MWh against this fixture's 101.1 MWh on the same day and
    the same `u`).  The differential results above are unaffected, since both
    sides of every comparison carried the same `p_init` and it cancels; but a
    shed level taken from one of those probes must not be compared against this
    fixture's.
    """
    unit_bus = np.asarray(case.unit_node_idx, np.int64)
    PTDF = np.asarray(case.PTDF, np.float64)
    F = np.asarray(case.line_cap, np.float64) * cap_scale
    gen = np.zeros((q_da.shape[1], int(case.n_nodes)), np.float64)
    np.add.at(gen, (slice(None), unit_bus), q_da.T)
    flow = (gen + s_da - d_da) @ PTDF.T
    return np.abs(flow) - F[None, :]                       # (T, n_lines), signed


def net_injection(case, q_da, s_da, d_da):
    """`P_da[t, n]` of §8, from the three parts of the position."""
    unit_bus = np.asarray(case.unit_node_idx, np.int64)
    gen = np.zeros((q_da.shape[1], int(case.n_nodes)), np.float64)
    np.add.at(gen, (slice(None), unit_bus), q_da.T)
    return gen + s_da - d_da


def check_position(case, pos, mu_tol=1e-6, balance_rtol=BALANCE_RTOL):
    """The gates of item 46.

    Four things are asserted, and the third exists because the second can be
    satisfied vacuously.  A position that never sheds has `s_da` identically
    zero, and then omitting `s_da` changes the balance by nothing -- the gate
    compares a number against itself and cannot fail.  That is the shape of
    defect this repository has now recorded three times, so the gate asserts it
    is *able* to fail before asserting that it passes.
    """
    worst_rel, worst_drop_rel, shed_days = 0.0, 0.0, []
    for d in range(len(pos["q_da"])):
        P = net_injection(case, pos["q_da"][d], pos["s_da"][d], pos["d_da"][d])
        scale = np.maximum(pos["d_da"][d].sum(1), 1.0)              # (T,) MW
        worst_rel = max(worst_rel, float((np.abs(P.sum(1)) / scale).max()))
        # dropping s_da moves the sum by exactly sum_n s_da of that period
        drop = np.abs(pos["s_da"][d].sum(1)) / scale
        if pos["shed_mwh"][d] > 1e-6:
            shed_days.append(d)
            worst_drop_rel = max(worst_drop_rel, float(drop.max()))

    shed_total = float(np.sum(pos["shed_mwh"]))
    # **Three states, not two, and the third is why this window can be built at
    # all.**  `can_fail` and `discriminates` ask "would dropping `s_da` from the
    # money-balance identity of section 8 be noticed", and on a window that
    # never sheds `s_da` is identically zero, so the question has no content.
    # That is `vacuous`, and recording it as `False` is worse than recording it
    # as `pass` -- both read as a verdict on the position when neither is one.
    #
    # The wording below is the one the two already-written 60-day products
    # carry in their own `gates_note`, taken verbatim rather than paraphrased:
    # they were produced by a version of this script that had this handling and
    # that never reached the repository, so `tests/fixtures/
    # day_ahead_position_29gb_T24_step1prime_seasons.npz` -- which markets 02
    # and 03 both consume -- could be restored from git but not rebuilt.  This
    # closes that gap, the same one `merge_segments.py` closed for the
    # commitment fixture.  Found 2026-08-29 by running this script over a
    # 365-day window, where both gates came back `False` and nothing was
    # written.
    # No second threshold: `vacuous` is exactly "no day cleared the 1e-6 MWh bar
    # the loop above already applies".  Introducing a constant of its own would
    # put two numbers in this function answering the same question, which is the
    # arrangement that is worse than either number alone.
    vacuous = not shed_days
    reason = (
        "this scenario has no day-ahead shedding at all (s_da is identically "
        "zero), so the question this gate asks -- would dropping s_da from the "
        "money-balance identity of section 8 be noticed -- has no content "
        "here. NOT a failure: there is nothing to detect. The matching test "
        "must build its own operating point.")

    def state(ok):
        return "vacuous" if vacuous else ("pass" if ok else "fail")

    return dict(
        worst_balance_rel=worst_rel,
        balance_ok=dict(status="pass" if worst_rel < balance_rtol else "fail",
                        worst_balance_rel=worst_rel),
        shed_days=shed_days,
        can_fail=dict(status=state(len(shed_days) > 0),
                      **(dict(reason=reason, s_da_total_mwh=shed_total)
                         if vacuous else dict(shed_days=len(shed_days)))),
        worst_drop_rel=worst_drop_rel,
        discriminates=dict(
            status=state(worst_drop_rel > DISCRIMINATION_FACTOR * worst_rel),
            **(dict(reason=reason, s_da_total_mwh=shed_total)
               if vacuous else dict(worst_drop_rel=worst_drop_rel))),
        separation=(worst_drop_rel / worst_rel) if worst_rel > 0 else float("inf"),
        worst_mu=float(np.max(pos["mu"])),
        mu_ok=dict(status="pass" if float(np.max(pos["mu"])) < mu_tol else "fail",
                   worst_mu=float(np.max(pos["mu"]))),
        worst_d_award=float(np.max(pos["d_award"])),
        worst_d_lmp=float(np.max(pos["d_lmp"])),
        agrees=dict(status=("pass" if (np.max(pos["d_award"]) == 0.0
                                       and np.max(pos["d_lmp"]) < LMP_ATOL)
                            else "fail"),
                    worst_d_award=float(np.max(pos["d_award"])),
                    worst_d_lmp=float(np.max(pos["d_lmp"]))),
        total_shed_mwh=shed_total)


#: Bound on how far regenerating may move a continuous quantity of the position.
#:
#: **Regenerating is not bit-reproducible across machine conditions, and the
#: measurement says why it is still sound.**  Measured 2026-08-15 on `case29gb`,
#: 60 days, three runs of this script (RTX 4500 Ada, float64, `jax` 0.10.2):
#:
#:     two runs back to back on one card, same occupancy
#:         u, q_da, s_da, lmp_da, d_da   all bit-identical, max|delta| = 0.0
#:     against the fixture built when the card was near-empty (724 MiB vs ~18 GB)
#:         u        bit-identical        <- the discrete decision does not move
#:         d_da     bit-identical        <- pure numpy, no accelerator
#:         q_da     8.160e-07 MW
#:         lmp_da   7.362e-07 $/MWh
#:         s_da     2.274e-13 MW
#:     totals: agree to summation noise, **not** bit-identical
#:         sum q_da   0.0 between two of the runs, 1 ULP against a third
#:         sum s_da   1.705e-13 absolute, 4.523e-16 relative (~2 ULP)
#:
#: The totals line was **first reported here as bit-identical and that was
#: wrong** -- it was read off a `:.10f` print, which renders 34902125.6357123554
#: for two float64s that differ in their last bit, because eighteen significant
#: digits is past what the type carries.  `repr` separates them.  The corrected
#: statement is weaker but still carries the argument: the totals agree to a few
#: ULP of the summation itself (`sqrt(N) * eps` is 6.8e-14 for the 95 040 entries
#: of `q_da`), so they are consistent with the same objective reached by a
#: different summation order.
#:
#: The aggregate agreeing to summation noise while the split moves at 1e-7 is the
#: signature of the solution moving to **another point of a degenerate optimal
#: face**, not of error accumulating: the objective and the totals are unchanged
#: and only the division among tied units differs, and ties are the norm on these
#: cases.  The mechanism: a different process picks a different XLA kernel
#: path and the whole solution lands elsewhere on the face.
#:
#: So the gate is three-tier, because a bitwise bar can only be asked of
#: quantities that are actually bitwise: `u` and `d_da` are held to equality, the
#: three continuous quantities to this bound, and the totals to `TOTALS_RTOL` --
#: a *relative* bound, not equality, for the reason given above.  The totals are
#: what separates "moved along the face" from "moved off it"; an elementwise
#: tolerance alone cannot tell those apart, and it is that separation, not
#: bitwise equality, that the third tier buys.
POSITION_ATOL = 1e-5

#: Relative bound on the totals.  Not bitwise: summing 95 040 float64 entries in
#: a different order moves the result by a few ULP on its own, and `sqrt(N)*eps`
#: is 6.8e-14 there.  Measured worst 4.523e-16 relative (on `sum s_da`).
TOTALS_RTOL = 1e-13


def check_reproducible(old, pos, atol=POSITION_ATOL):
    """Compare a regenerated position against the one already on disk.

    Returns ``{"ok": bool, "lines": [str]}``.  See `POSITION_ATOL` for why the
    bar differs by quantity rather than being bitwise throughout.
    """
    exact = ("u", "d_da")
    approx = ("q_da", "s_da", "lmp_da")
    stored = dict(pos)
    stored["u"] = pos["u"].astype(np.int8)

    ok, lines = True, []
    for k in exact:
        if k not in old.files:
            continue
        same = np.array_equal(old[k], stored[k])
        ok &= same
        lines.append(f"{k:7s} bitwise {'yes' if same else 'NO'}"
                     + ("" if same else "   <- a discrete decision moved"))
    for k in approx:
        if k not in old.files:
            continue
        d = float(np.abs(old[k].astype(np.float64)
                         - np.asarray(stored[k], np.float64)).max())
        good = d <= atol
        ok &= good
        lines.append(f"{k:7s} max|delta| {d:.3e}  {'ok' if good else 'OVER'}")
    # the totals separate "moved along the degenerate face" from "moved off it",
    # to summation noise rather than bitwise -- see `TOTALS_RTOL`
    for k in ("q_da", "s_da"):
        if k not in old.files:
            continue
        a, b = float(np.asarray(old[k]).sum()), float(np.asarray(stored[k]).sum())
        rel = abs(a - b) / max(abs(a), 1e-30)
        good = rel <= TOTALS_RTOL
        ok &= good
        lines.append(f"sum {k:6s} rel {rel:.3e}  {'ok' if good else 'OFF THE FACE'}"
                     + ("" if good else f"   {a!r} vs {b!r}"))
    return dict(ok=ok, lines=lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", default="29gb")
    ap.add_argument("--demand-floor-mw", type=float, default=None,
                    help="the floor the case's demand series is clipped at, in "
                         "MW.  No default, because `load_rts_demand` and "
                         "`load_nem_demand` have none: it is part of the run "
                         "point, and 29gb takes no floor at all")
    ap.add_argument("--demand-netted", nargs="+", default=None,
                    help="73rts only: which renewable classes to net off "
                         "demand.  Defaults to all four, as `load_rts_demand` does")
    ap.add_argument("--mode", default="relax", choices=("relax", "milp"),
                    help="which pre-commitment fixture supplies the day boundary")
    ap.add_argument("--segments", type=int, default=1)
    #: The three scenario scales.  Defaults are the module constants,
    #: so a command line that names none of them runs exactly what this script
    #: ran before they existed; the form is `precommit.py`'s, which already takes
    #: all three, and the *form* is what is copied here rather than the values.
    ap.add_argument("--cap-scale", type=float, default=CAP_SCALE,
                    help=f"if not given, this file's {CAP_SCALE}; `precommit` "
                         f"defaults to 0.4 and `milp_reference` falls back to "
                         f"the 29gb fixture meta's 0.6 -- the three tools each "
                         f"have their own default for this quantity, so give it "
                         f"explicitly when comparing across tools.")
    ap.add_argument("--ramp-scale", type=float, default=RAMP_SCALE,
                    help=f"if not given, this file's {RAMP_SCALE} (`precommit`: 0.25).")
    ap.add_argument("--p-min-scale", type=float, default=P_MIN_SCALE,
                    help="scale on every unit's minimum output, applied to the "
                         "case once rather than threaded into each operator.  "
                         "1.0 is the registered case and the default; 73rts "
                         "adopts 0.80")
    ap.add_argument("--days", type=int, default=None, help="default: the whole window")
    ap.add_argument("--monitored-lines", default="all",
                    help="all | rated | comma-separated line indices. `all` is the\n"
                         "default and leaves this driver on the dense route every\n"
                         "existing product was made on; `rated` enforces only the\n"
                         "lines whose rating the case publishes (7 of 1 278 on\n"
                         "case813nem) and takes the low-rank route.")
    ap.add_argument("--fixture", default=None,
                    help="path to the pre-commitment fixture.  Default: the "
                         "path derived from --mode/--case, which is the shipped "
                         "60-day product.  A window other than that one has to "
                         "be named here, because the derived path carries no "
                         "window and two products would be indistinguishable "
                         "by name (the failure recorded for 2026-08-24)")
    #: Default 0 = off, existing behaviour unchanged; give 10 when launching a
    #: long job to get progress lines.
    #: The job launcher sets `PYTHONUNBUFFERED`, so no extra flush is needed.
    ap.add_argument("--progress-every", type=int, default=0,
                    help="print one progress line every N days (0 = off, the "
                         "existing condition only)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # once, to the case, so that every operator reading `unit_p_min` sees the
    # same one -- `scale_min_output`'s own contract, and the same call
    # `precommit.py` makes
    case = scale_min_output(load_case(args.case), args.p_min_scale)
    #: `p_min_scale` is passed here as well as checked below: the loader refuses a
    #: fixture declaring a value the call did not name (2026-09-13), and that refusal
    #: fires before this module's own `check_boundary_scenario` would.  Both stay --
    #: the loader's covers every one of the sixteen drivers, this one carries the
    #: message that names the two places the value has to agree.
    fixture = load_commitment(mode=args.mode, case=args.case, n_periods=T,
                              path=args.fixture, p_min_scale=args.p_min_scale)
    check_boundary_scenario(fixture["meta"], args.p_min_scale)
    _pairing = demand_pairing(args.case, floor_mw=args.demand_floor_mw,
                              netted=args.demand_netted)
    demand = demand_for_case(args.case, **_pairing)
    n_days = args.days or len(fixture["day_index"])

    print(f"case {args.case}  boundary from {args.mode}  {n_days} days  "
          f"K={args.segments}  truthful (markup 1.0)  step 1' chain")
    print(f"  scenario: cap_scale {args.cap_scale}  ramp_scale {args.ramp_scale}"
          f"  p_min_scale {args.p_min_scale}")
    monitored = parse_monitored_lines(args.monitored_lines, case)
    pos, spec = harvest(case, fixture, demand, n_days, n_segments=args.segments,
                        progress_every=args.progress_every,
                        cap_scale=args.cap_scale, ramp_scale=args.ramp_scale,
                        monitored_lines=monitored)
    g = check_position(case, pos)
    full = n_days == len(fixture["day_index"])
    if full and pos["start_day"] != 0:
        raise SystemExit(f"full window must start at day 0, got {pos['start_day']}")
    if not full:
        print(f"\n  short run: chain starts at day {pos['start_day']}, "
              f"not the shipped window")

    w = lambda k: g[k]["status"].upper()
    print("\ngates:")
    print(f"  re-solve reproduces the env   |dq| {g['worst_d_award']:.3e}  "
          f"|dlmp| {g['worst_d_lmp']:.3e}  {w('agrees')}")
    print(f"  sum_n P_da per period         {g['worst_balance_rel']:.3e} rel  "
          f"{w('balance_ok')}")
    print(f"  the gate can fail             {len(g['shed_days'])} shedding days "
          f"{g['shed_days'][:8]}  {w('can_fail')}")
    print(f"  dropping s_da is detected     {g['worst_drop_rel']:.3e} rel, "
          f"{g['separation']:.1e}x the residual  {w('discriminates')}")
    print(f"  worst mu                      {g['worst_mu']:.3e}  {w('mu_ok')}")
    print(f"  shed {g['total_shed_mwh']:.3e} MWh over {len(g['shed_days'])} days")

    # `can_fail` and `discriminates` are properties of the window, not of the
    # code: a stretch of days that never sheds cannot exercise them, and that is
    # `vacuous` rather than a failure.  `vacuous` does not block the write; only
    # `fail` does.  The distinction is carried into the product's `meta` as
    # `gates_schema` / `gates_note` so a reader sees which gates had content.
    bad = [k for k in ("agrees", "balance_ok", "mu_ok", "can_fail",
                       "discriminates") if g[k]["status"] == "fail"]
    if any(g[k]["status"] == "vacuous" for k in ("can_fail", "discriminates")):
        print("  (no shedding day in this window, so the two s_da gates have no "
              "content here -- recorded as vacuous, not as passed)")
    if bad:
        raise SystemExit(f"gate(s) failed: {bad}; the position was not written")

    _sd = int(pos["start_day"])
    position_dates = list(fixture["meta"]["dates"])[_sd: _sd + n_days]
    assert len(position_dates) == n_days, (
        f"the fixture lists {len(fixture['meta']['dates'])} dates and this run "
        f"takes {n_days} from offset {_sd}; that slice is short")
    # the demand pairing, for the reason given for `precommit.py` (2026-09-05): this
    # is a write side, not a diagnostic.  `day_ahead_position_*.npz` is consumed
    # by markets 02 and 03, its derived path carries no demand, and a reader
    # cannot know a floor before it opens the file -- the floor is what the file
    # records.  Two positions built on different series would otherwise be
    # indistinguishable by name and by meta alike.
    meta = dict(
        **demand_meta(args.case, **_pairing),
        stage="day-ahead position (rtm item 46)", case=args.case, n_periods=T,
        n_segments=args.segments, chain="step 1' (endogenous u, as env.step runs it)",
        day_ahead_policy="truthful, markup 1.0",
        boundary_fixture=f"day_ahead_commitment_{args.case}_T{T}_{args.mode}.npz",
        boundary="chained; day 0 from the fixture, thereafter advanced by env.step",
        cap_scale=args.cap_scale, ramp_scale=args.ramp_scale,
        p_min_scale=args.p_min_scale, markup_max=MARKUP_MAX, voll=VOLL,
        solver="powermarketjax.solvers.ipm via envs.day_ahead.clearing",
        max_iter=MAX_ITER, dual_start="cost_norm",
        periods_per_hour=PERIODS_PER_HOUR, n_days=n_days,
        start_day=int(pos["start_day"]), seconds=pos["seconds"],
        #: The line set and the KKT route are both taken from **the operator
        #: that was built**, not from the flag (`effective_monitored_stamp`).
        #: A reader of the fixture must be able to tell which line set built it:
        #: `all` and `rated` products can share a file name, and they are two
        #: different objects under test.
        monitored_lines=(None if pos["monitored_lines"] is None
                         else [int(i) for i in pos["monitored_lines"]]),
        kkt_route=str(pos["kkt_route"]),
        #: **`seconds` is meaningless without the four fields under it.**  Until
        #: 2026-09-14 the meta recorded the time and not the machine it was taken
        #: on, and two readings of "73rts, 366 days, this script" then sat in the
        #: repo disagreeing by 1.79x with nothing to tell them apart: **1432.5 s**
        #: in the comment below `stacked["seconds"]` and **2560.1 s** in
        #: `da_position_73rts_T24_366d_c0.424_r1.00_pmin0.80.npz`.  The second is
        #: `JAX_PLATFORMS=cpu OMP_NUM_THREADS=8` on eight pinned cores -- its
        #: launch log says so, the product could not.  The first was on a GPU and
        #: **there is no product to ask**: that run failed the `agrees` gate and
        #: wrote nothing, so the missing stamp cannot be filled in even in
        #: principle.  A stamp like this can only go in the writer; there is no
        #: later place to add it.
        backend=jax.default_backend(),
        devices=[str(d) for d in jax.devices()],
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
        cpu_affinity=len(os.sched_getaffinity(0)),
        # **`dates` and `window` are what every consumer reads**, and until
        # 2026-08-29 this writer emitted neither: `run_rl_02`, `run_rl_03`,
        # `run_eval_02`, `run_eval_03` and `run_bestfixed_grid` all do
        # `meta["dates"]` unguarded, so a position produced by this script could
        # not be consumed by any of them -- found by feeding a freshly built
        # 365-day position to `run_rl_03`, which died on `KeyError: 'dates'`
        # before its first iteration.  The already-shipped 60-day products carry
        # both keys, which is the same evidence as `gates_schema`/`gates_note`
        # that the version which wrote them never reached the repository.
        #
        # Sliced from the boundary fixture's own `dates` rather than recomputed
        # from the calendar: that list is the record of which days the fixture
        # covers, and `envs/day_ahead/env.py` already cross-checks it against
        # `day_index`, so taking it from anywhere else would introduce a second
        # answer to the same question.
        dates=position_dates,
        window=(fixture["meta"].get("window")
                or f"{n_days} consecutive days: {position_dates[0]}"
                   f" .. {position_dates[-1]}"),
        gates=g,
        gates_schema="each gate is pass / fail / vacuous; vacuous means the "
                     "phenomenon the gate tests for does not occur in this "
                     "scenario, which is not the same as the gate failing",
        gates_note=(
            "can_fail/discriminates are empty here because this scenario has no "
            "day-ahead shedding at all; they test whether dropping s_da from "
            "the money-balance identity would be noticed, and that question has "
            "no content when s_da is identically zero. The matching tests must "
            "build their own operating point."
            if g["can_fail"]["status"] == "vacuous" else
            f"can_fail/discriminates were exercised: {len(g['shed_days'])} of "
            f"{n_days} days shed."))
    path = Path(args.out) if args.out else FIXTURE_DIR / (
        f"day_ahead_position_{args.case}_T{T}_step1prime.npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        rep = check_reproducible(np.load(path), pos)
        for line in rep["lines"]:
            print("  " + line)
        if not rep["ok"]:
            raise SystemExit("regeneration moved the position; nothing written")

    np.savez_compressed(
        path, u=pos["u"].astype(np.int8), q_da=pos["q_da"], s_da=pos["s_da"],
        lmp_da=pos["lmp_da"], d_da=pos["d_da"], mu=pos["mu"],
        shed_mwh=pos["shed_mwh"],
        line_dual_up=pos["line_dual_up"], line_dual_dn=pos["line_dual_dn"],
        shed_dual=pos["shed_dual"],
        # per-day worst, see `line_flow_overshoot` for what this is and is not
        line_flow_overshoot=np.array([
            line_flow_overshoot(case, pos["q_da"][d], pos["s_da"][d],
                                pos["d_da"][d], cap_scale=args.cap_scale).max()
            for d in range(len(pos["q_da"]))]),
        day_index=np.asarray(
            fixture["day_index"][pos["start_day"]:pos["start_day"] + n_days], np.int32),
        meta=np.array(json.dumps(meta, indent=1)))
    print(f"\nwrote {path}  ({path.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
