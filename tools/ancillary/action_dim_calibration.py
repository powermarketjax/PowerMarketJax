"""Is `MAX_ITER = 60` enough once the offers come from the action space?

`REG_COEF` was calibrated over 240 real periods, but at **truthful offers** -- one
point of the action dimension.  `MAX_ITER` was never calibrated for this market at
all; its own docstring says so, and 60 is the day-ahead value.  Market 02 had the
same shape of gap and moved 40 -> 70 once offers were drawn from the action space.

**Why the answer here might go the other way.**  The truthful reserve offer is
`softplus(-800) * pi_scale`, which underflows to *exactly* zero in float32, so at
the calibration point every provider's reserve offer is bit-identical: the reserve
block is an exact n-way tie, the worst degeneracy there is.  Strategic offers
separate them.  So moving off the calibration point makes that block *less*
degenerate, and "lambda_res goes from zero to non-zero, therefore harder" does not
follow -- what changes is the dual's value, not the active set.  If iteration count
is driven by conditioning and degeneracy, 60 may be fine.

**What would refute that, and why both dimensions are recorded.**  Strategic
offers also change the energy markup, which reorders dispatch entirely.  If
failures track markup magnitude rather than reserve separation, the degeneracy
story is wrong and this is market 02 again.  Marginal counts cannot separate the
two when the dimensions are correlated, so the raw output is the **joint** grid:
every sample carries its separation and its markup, and failures are binned on
both together.

**Near-ties are a mechanism property, not a calibration defect.**  This market's
duals are unreliable when adjacent reserve offers are too close: relative
separation of 1e-5 and wider is clean, 1e-7 and narrower is not, and float32 can
express 3.8e-8, so sampling reaches into the bad region on its own.  Samples below
the threshold are reported in their own table and kept out of the calibration
sample; folding them in would read as "60 is not enough" and drive `MAX_ITER` up
against a problem more iterations cannot fix.

**Cells are gated at the truthful point before any offer is perturbed.**  Not
every (day, hour) of the commitment fixture admits a feasible clearing under this
operating-point construction -- measured, 11 of 16 do, and the reserve demand
fraction makes no difference to which, so it is the cell and not the parameter.
An infeasible cell returns garbage with `mu` at 1e19 to 1e182, which is
indistinguishable at a glance from "too few iterations" and would drive
`MAX_ITER` up against something more iterations cannot fix.  So each cell is
first solved at truthful offers with the reference iteration count, and only
cells that converge there enter the sample.  The number rejected is reported
rather than hidden: a gate that silently drops most of the population is a
different measurement from the one it claims to be.

**The sieve and the ranking are different quantities.**  `mu` is cheap and is used
only to flag; the ranking is by the consumed quantities -- per-unit reserve leg
revenue and per-unit energy revenue against a high-iteration reference -- because
`mu` failed to locate the cliff twice on market 02.

**The objective is recorded beside them, and that is not optional.**  A quantity
difference between two converged solves is only harmless if the objective agrees:
that is the licence condition for "quantities may differ, money may not",
and it has to be asserted by the same measurement rather than
assumed.  Without it, "probably an alternative optimum" is unfalsifiable -- and if
the objective in fact disagrees, the difference is a real error and the iteration
count that produced it is not sufficient after all.

**Energy revenue uses each unit's own nodal price.**  An earlier version of this
tool multiplied the award by the network-wide maximum LMP, which is not any unit's
revenue and inflates the energy leg against the reserve leg by an unknown factor.
`lmp` is per bus and `case.unit_node_idx` maps units onto it, so the right
quantity is available and there is no reason to report a proxy.

**The grid is joint in `reg_coef` and `max_iter`, and it was not always.**  The
first version of this tool held `reg_coef` at the module constant and swept the
iteration count alone, so its table reported one slice of a two-dimensional
surface as though it were a curve.  `max_iter`, `dual_start` and `reg_coef`
are one calibration rather than three knobs, which is
exactly the statement that the slice is not enough: a failure that more
iterations do not fix and a failure that a different regularisation does not fix
are different failures, and a sweep along one axis cannot tell them apart.

**Two references per sample, and they answer different questions.**  The
same-rung reference is `REFERENCE_ITER` steps at the row's own `reg_coef`, and a
difference against it means "this many steps is not enough *at this
regularisation*".  The anchor is `REFERENCE_ITER` steps at the adopted
`REG_COEF`, and a difference against it means "this row does not agree with the
adopted calibration at all".  A rung can be clean against itself and wrong
against the anchor -- converged to a different answer -- and only the second
column shows it.  At the adopted `reg_coef` the two references coincide by
construction, so that rung's same-rung column is comparable with the earlier
one-dimensional table.

**Both residuals are recorded, because `mu` alone cannot name the failure.**
This market's own operator docstring says a converging `mu` beside a stalled
stationarity residual is the recorded symptom of the `reg_coef` scale problem
rather than of too few steps -- so a table that reports only `mu` cannot support
either diagnosis.  `clear()` already returns `dual_residual` and
`primal_residual`; they were simply not being read.  **`dual_residual` is the raw
`|r1|` of the final iterate, not normalised**, while the 2026-08-14 `reg_coef`
ladder in `envs/ancillary/clearing.py` flagged on the *normalised* stationarity
residual.  `c_inf` is recorded on every row so the two can be put on the same
footing rather than compared across statistics.

    PYTHONPATH=<repo> python tools/ancillary/action_dim_calibration.py --seeds 8
    PYTHONPATH=<repo> python tools/ancillary/action_dim_calibration.py \
        --reg-coef 1e-18 --seeds 8      # the earlier one-dimensional sweep
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax                                                       # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                          # noqa: E402

from powermarketjax.case import load_case                        # noqa: E402
from powermarketjax.envs.ancillary.action import (                # noqa: E402
    offer_separation)
from powermarketjax.envs.ancillary.clearing import (MAX_ITER,     # noqa: E402
                                                    REG_COEF, make_clearing)
from powermarketjax.envs.ancillary.env import MU_TOL              # noqa: E402
from powermarketjax.envs.day_ahead.clearing import segment_costs  # noqa: E402

import tests.envs.ancillary.test_clearing_l2 as L2               # noqa: E402

#: Iteration counts to compare, plus the reference. The reference must be far
#: enough above the candidates that it is not itself on the cliff.
ITERS = (40, 60, 80, 100, 120)
REFERENCE_ITER = 240

#: Regularisation coefficients swept beside the iteration count; the adopted
#: `REG_COEF` must be one of them, since it is the anchor.  1e-14 is included
#: because it is what `solvers/ipm.py` hands any market that does not pass its
#: own value, and market 02 runs there today -- so the rung is a real operating
#: point of the repository and not a hypothetical.  1e-12 is left out: the
#: 2026-08-14 ladder flagged 93.3% of periods there, which is a rung with no
#: resolution left in it.
REG_COEFS = (1e-14, 1e-16, 1e-18, 1e-20)

#: Separation below which this market's duals are known to be unreliable
#: (measured in a provider-count scan): 1e-5 clean, 1e-7 not.
SEPARATION_FLOOR = 1e-5

#: Cheap sieve only; never the ranking.  **Imported, not restated.**  This market
#: uses 1e-9 (`envs/ancillary/env.py`); market 02 uses 1e-6, and writing the
#: number here once let market 02's value be applied to this market's data, which
#: inverted a conclusion: at 1e-6 the near-tie cells read as "converged" and the
#: environment looked blind to them, when in fact 1e-9 flags every one of them.
#: A tolerance belongs to a market, so it is taken from that market.

#: The declared reserve price scale (§12.3 has no cost basis for it, so it is a
#: parameter). Recorded with every result.
PI_SCALE = 50.0


def sample_actions(rng, n_units, n_prod, *, markup_hi):
    """One draw from both dimensions of the action space.

    Energy is a per-unit markup on the true segment cost; reserve is the raw
    pre-softplus action. Both are drawn per unit, so the reserve block separates
    by however much the draw separates it -- which is the quantity the sieve
    below measures rather than controls.
    """
    markup = rng.uniform(1.0, markup_hi, size=n_units)
    alpha = rng.uniform(-6.0, 3.0, size=(n_units, n_prod))
    return markup, alpha


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--periods", type=int, default=24)
    ap.add_argument("--markup-hi", type=float, default=2.0)
    # This default path holds a committed product, and a bare run overwrites it
    # with a different scenario rather than reproducing it: that file was written
    # at `pi_scale 10.0` and `volr 1000.0` under the previous `meta` schema (no
    # `cap_scale`, no `ramp_scale`, scalar `reg_coef`), while the defaults below
    # now resolve to 50.0 and `L2.VOLR` 250.0.  Nothing warns at write time -- the
    # scenario is in `meta` and the file name carries none of it -- so pass `--out`
    # when the committed product is still wanted.  Measured 2026-08-24 by loading
    # both; `..._pi50.npz` beside it is the adopted-scale run.
    ap.add_argument("--out", default="tests/fixtures/ancillary_action_dim_calibration.npz")
    # Scenario overrides.  They exist so that a recalibration does not require
    # editing `test_clearing_l2`: that module's tolerances are measured at its
    # own scenario, so moving its constants would move every L2 and L0 test at
    # the same time and cost the green baseline.  Defaults are L2's values, so
    # an invocation without flags reproduces what this file did before.
    ap.add_argument("--fixture", default=str(L2.FIXTURE),
                    help="commitment fixture; defaults to the one L2 uses")
    ap.add_argument("--cap-scale", type=float, default=L2.CAP_SCALE)
    ap.add_argument("--ramp-scale", type=float, default=L2.RAMP_SCALE)
    # A calibration has to be run at the value the market will be run at.  The
    # specification adopts VOLR at 250 (section 22) while `test_clearing_l2`
    # carried 1000 when this flag was added, and every device reading `L2.VOLR`
    # inherited that -- so this is a flag rather than a wait for the migration
    # that closes the gap.
    #
    # The default tracks `L2.VOLR` rather than naming a number, which means it
    # MOVES when that migration lands: after it, a bare invocation runs at the
    # adopted value and no longer reproduces what this file did before.  That is
    # the intended behaviour, not a regression, and the help text below prints
    # the live default rather than restating a number that would go stale on the
    # same commit that makes it wrong.
    ap.add_argument("--volr", type=float, default=L2.VOLR,
                    help=f"penalty on unmet reserve requirement; defaults to "
                         f"L2.VOLR, currently {L2.VOLR:g}, and follows it")
    # The sweep axis that used to be a constant.  Defaulting to the full grid
    # rather than to `REG_COEF` is deliberate: the one-dimensional run is the
    # special case now, and it is still available as `--reg-coef 1e-18`.
    # `PI_SCALE` is 50, section 22's adopted value: the only scale at which an
    # offer reaches the VOLR cap at a raw action of about 5.  This comment read
    # "the scale this file has always used is 10.0, which is this file's own
    # choice and not the adopted value", which was true until the constant was
    # moved and is now the wrong way round -- there is no longer a gap between
    # this file's scale and the market's, and the help text below prints one
    # number twice for that reason.  10.0 is what the committed
    # `ancillary_action_dim_calibration.npz` ran at, and running a calibration
    # at a scale the market does not use is how the 2026-08-17 arbitration came
    # to report a 26 980 $ failure that does not exist at 50, so this stays a
    # flag and every product records what it ran at.
    ap.add_argument("--pi-scale", type=float, default=PI_SCALE,
                    help=f"reserve offer price scale; this file's own default "
                         f"is {PI_SCALE:g}, the specification adopts 50")
    ap.add_argument("--reg-coef", type=float, nargs="+", default=list(REG_COEFS),
                    help="regularisation coefficients to sweep; the adopted "
                         "REG_COEF must be among them, it is the anchor")
    args = ap.parse_args()

    reg_coefs = tuple(sorted(set(args.reg_coef), reverse=True))
    if REG_COEF not in reg_coefs:
        raise ValueError(
            f"the adopted REG_COEF {REG_COEF:g} is the anchor every row is "
            f"differenced against and must be in the swept grid, got {reg_coefs}")

    case = load_case(L2.CASE)
    _, cost = segment_costs(case, 1)
    cap_scale, ramp_scale = args.cap_scale, args.ramp_scale
    # print what is in force, never what the caller believes: a fixture built at
    # one scenario and a clearing built at another disagree silently
    print(f"in force: cap_scale={cap_scale} ramp_scale={ramp_scale} "
          f"fixture={args.fixture}")
    fx = np.load(args.fixture, allow_pickle=True)
    fmeta = json.loads(str(fx["meta"])) if "meta" in fx else {}
    got = (fmeta.get("cap_scale"), fmeta.get("ramp_scale"))
    if None not in got and got != (cap_scale, ramp_scale):
        raise ValueError(
            f"fixture was built at cap_scale/ramp_scale {got}, this run was "
            f"asked for {(cap_scale, ramp_scale)}; the commitment and the "
            f"clearing would describe different scenarios")
    commitment = fx["commitment"]
    pmin = np.asarray(case.unit_p_min, np.float64)
    pmax = np.asarray(case.unit_p_max, np.float64)
    n_units = len(pmin)
    unit_node = np.asarray(case.unit_node_idx, np.int64)

    # (reg_coef, max_iter) -> operator.  The anchor is the adopted regularisation
    # at the reference iteration count; every row is differenced against both it
    # and its own rung's reference.
    anchor = (REG_COEF, REFERENCE_ITER)
    builders, specs = {}, {}
    for reg in reg_coefs:
        for it in (*ITERS, REFERENCE_ITER):
            builders[(reg, it)], specs[(reg, it)] = make_clearing(
                case, L2.THETA, args.volr, n_segments=1, cap_scale=cap_scale,
                ramp_scale=ramp_scale, period_hours=L2.DELTA, max_iter=it,
                reg_coef=reg)
    # Prove the operator received what this run asked for, rather than that this
    # run asked for it: the values below come back out of the callee's `spec`.
    for (reg, it), sp in specs.items():
        if (float(sp["reg_coef"]), int(sp["max_iter"])) != (float(reg), int(it)):
            raise RuntimeError(
                f"asked for reg_coef/max_iter {(reg, it)} and the operator "
                f"reports {(sp['reg_coef'], sp['max_iter'])}")
    spec = specs[anchor]
    n_prod = int(spec["n_prod"])
    clear = {k: jax.jit(v) for k, v in builders.items()}

    print(f"case {L2.CASE}, n_units {n_units}, n_prod {n_prod}, "
          f"adopted MAX_ITER {MAX_ITER}, adopted REG_COEF {REG_COEF:g}")
    # in force, read back from `spec`, not from the constants this file imported
    print(f"in force: reg_coef grid {tuple(f'{r:g}' for r in reg_coefs)}, "
          f"dual_start {spec['dual_start']!r}, voll {spec['voll']:g}, "
          f"volr {spec['volr']:g}, period_hours {spec['period_hours']}")
    # `dual_start` is "cost_norm", whose starting multipliers are
    # `max(1, |c|_inf) / s`, and VOLL sets `|c|_inf` here -- so VOLL is inside
    # this calibration rather than beside it.  It is printed for that reason.
    print(f"note: dual_start {spec['dual_start']!r} scales the starting duals "
          f"by max(1, |c|_inf), and |c|_inf is period_hours * voll = "
          f"{float(spec['period_hours']) * float(spec['voll']):g} here")
    print(f"iterations compared {ITERS} against reference {REFERENCE_ITER}, "
          f"anchor {anchor}")
    print(f"separation floor {SEPARATION_FLOOR:g}; samples below it are reported "
          f"separately, not calibrated on\n")

    # operating points: distinct (day, hour) cells of the fixture
    # Gate: the cell must be feasible at truthful offers before its offers are
    # perturbed, or a divergent reference makes every difference meaningless.
    truthful_res = jnp.zeros((n_units, n_prod))
    candidate = [(d, h) for d in range(min(args.periods, commitment.shape[0]))
                 for h in (0, 6, 12, 18)]
    cells, rejected = [], []
    for (d, h) in candidate:
        u = commitment[d, :, h].astype(np.float64)
        if u.sum() < 2:
            rejected.append((d, h, "fewer than two units committed"))
            continue
        supply = (np.asarray(spec["res_cap"]) * u[:, None]).sum(0)
        demand = float((pmin * u).sum()) + 0.85 * float(((pmax - pmin) * u).sum())
        p_prev = (pmin + 0.85 * (pmax - pmin)) * u
        got = clear[anchor](jnp.asarray(cost), truthful_res, jnp.asarray(u),
                            jnp.asarray(demand), jnp.asarray(0.70 * supply),
                            jnp.asarray(p_prev))
        mu0 = float(got["mu"])
        (cells if mu0 < MU_TOL else rejected).append(
            (d, h) if mu0 < MU_TOL else (d, h, f"truthful mu {mu0:.2e}"))
    print(f"operating points: {len(cells)} usable, {len(rejected)} rejected at the "
          f"truthful gate (of {len(candidate)} candidates)")
    if not cells:
        raise RuntimeError("no usable operating point; the gate rejected all")
    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        for (d, h) in cells:
            u = commitment[d, :, h].astype(np.float64)
            if u.sum() < 2:
                continue
            markup, alpha = sample_actions(rng, n_units, n_prod,
                                           markup_hi=args.markup_hi)
            offer = (markup[:, None] * cost)
            offer_res = np.asarray(jax.nn.softplus(jnp.asarray(alpha)) * args.pi_scale)
            supply = (np.asarray(spec["res_cap"]) * u[:, None]).sum(0)
            demand = float((pmin * u).sum()) + 0.85 * float(((pmax - pmin) * u).sum())
            p_prev = (pmin + 0.85 * (pmax - pmin)) * u
            d_res = 0.70 * supply

            sep = np.asarray(offer_separation(jnp.asarray(offer_res),
                                              jnp.asarray(u > 0)))
            args_in = (jnp.asarray(offer), jnp.asarray(offer_res), jnp.asarray(u),
                       jnp.asarray(demand), jnp.asarray(d_res), jnp.asarray(p_prev))
            # `dual_residual` comes out of the solver un-normalised, and the
            # ladder it has to be compared against normalised by the cost norm.
            # Reconstructed from the same pieces the operator concatenates, so
            # the row carries the divisor rather than leaving it to be guessed.
            c_inf = float(spec["period_hours"]) * max(
                float(offer.max()), float(offer_res.max()),
                float(spec["voll"]), float(spec["volr"]))
            anchor_out = {k: np.asarray(v)
                          for k, v in clear[anchor](*args_in).items()}
            anchor_res_rev = anchor_out["reserve"] * anchor_out["reserve_price"][None, :]
            anchor_en_rev = anchor_out["award"] * anchor_out["lmp"][unit_node]

            for reg in reg_coefs:
                ref = {k: np.asarray(v)
                       for k, v in clear[(reg, REFERENCE_ITER)](*args_in).items()}
                ref_res_rev = ref["reserve"] * ref["reserve_price"][None, :]
                ref_en_rev = ref["award"] * ref["lmp"][unit_node]

                for it in ITERS:
                    got = {k: np.asarray(v)
                           for k, v in clear[(reg, it)](*args_in).items()}
                    res_rev = got["reserve"] * got["reserve_price"][None, :]
                    en_rev = got["award"] * got["lmp"][unit_node]
                    rows.append(dict(
                        seed=seed, day=d, hour=h, iters=it, reg_coef=reg,
                        separation=float(np.min(sep)),
                        markup_max=float(markup.max()), markup_mean=float(markup.mean()),
                        mu=float(got["mu"]),
                        # `mu` cannot separate "too few steps" from "wrong
                        # regularisation scale"; the operator's own docstring
                        # names the stationarity residual as the discriminator
                        dual_residual=float(got["dual_residual"]),
                        dual_residual_rel=float(got["dual_residual"]) / c_inf,
                        primal_residual=float(got["primal_residual"]),
                        c_inf=c_inf,
                        # against this rung's own converged solve: "is this many
                        # steps enough at this regularisation"
                        d_reserve_revenue=float(np.max(np.abs(res_rev - ref_res_rev))),
                        d_energy_revenue=float(np.max(np.abs(en_rev - ref_en_rev))),
                        d_reserve_price=float(np.max(np.abs(
                            got["reserve_price"] - ref["reserve_price"]))),
                        # against the adopted calibration: "does this rung agree
                        # with the answer the market is being run on"
                        d_reserve_revenue_anchor=float(np.max(np.abs(
                            res_rev - anchor_res_rev))),
                        d_energy_revenue_anchor=float(np.max(np.abs(
                            en_rev - anchor_en_rev))),
                        d_reserve_price_anchor=float(np.max(np.abs(
                            got["reserve_price"] - anchor_out["reserve_price"]))),
                        # the licence condition for reading a quantity difference as
                        # harmless: objectives must agree, or the difference is money
                        objective=float(got["z"]),
                        d_objective=float(got["z"] - ref["z"]),
                        d_objective_rel=float(abs(got["z"] - ref["z"])
                                              / max(abs(ref["z"]), 1e-30)),
                        d_objective_anchor_rel=float(
                            abs(got["z"] - anchor_out["z"])
                            / max(abs(anchor_out["z"]), 1e-30)),
                    ))
        print(f"  seed {seed} done ({len(rows)} rows)", flush=True)

    keys = list(rows[0])
    arrays = {k: np.array([r[k] for r in rows]) for k in keys}
    # Scenario written at write time: an npz has nowhere to hang a banner
    # afterwards, and a product about a calibration is worth little if it cannot
    # say which scenario that calibration sat in.
    #
    # Every scenario field below is read back out of `spec`, which is what the
    # operator reports it received, rather than out of the constants this file
    # imported, which is only what this file asked for.  The field that used to
    # be `reg_coef` is gone on purpose: it held the imported constant, and this
    # product now sweeps the value, so a same-named field with a new meaning
    # would be worse than a rename.
    meta = dict(case=L2.CASE,
                cap_scale=float(spec["cap_scale"]),
                ramp_scale=float(spec["ramp_scale"]),
                commitment_fixture=str(args.fixture),
                n_units=n_units, n_prod=n_prod,
                theta=list(L2.THETA),
                volr_in_force=float(spec["volr"]),
                voll_in_force=float(spec["voll"]),
                voll_note="this calibration is not VOLL-independent: "
                          "dual_start is 'cost_norm', whose starting "
                          "multipliers are max(1, |c|_inf) / s, and |c|_inf is "
                          "period_hours * voll here, so moving VOLL moves the "
                          "starting point of every solve in this product",
                delta=float(spec["period_hours"]),
                dual_start_in_force=str(spec["dual_start"]),
                reg_coefs_in_force=sorted(
                    {float(sp["reg_coef"]) for sp in specs.values()}),
                adopted_reg_coef=REG_COEF, adopted_max_iter=MAX_ITER,
                anchor=dict(reg_coef=anchor[0], max_iter=anchor[1]),
                pi_scale_in_force=float(args.pi_scale), iters=list(ITERS),
                reference_iter=REFERENCE_ITER, mu_tol=MU_TOL,
                separation_floor=SEPARATION_FLOOR, seeds=args.seeds,
                markup_range=[1.0, args.markup_hi],
                alpha_range=[-6.0, 3.0],
                n_cells=len(cells), n_rejected_at_gate=len(rejected),
                n_samples=len(cells) * args.seeds,
                # Three denominators are in circulation for this measurement and
                # they differ by four, which is the range where a reader takes
                # them for a typo of one another rather than for three
                # constructions.  Spelled out here rather than only in a report,
                # because a reader who quotes a number from this file reads this
                # field and not the report.
                denominators=[
                    "704 = 88 cells x 8 seeds -- tests/fixtures/"
                    "ancillary_action_dim_calibration.npz, committed 2026-08-16, "
                    "old scenario, carries no scenario stamp because the writer "
                    "had none yet; 699 above the separation floor, 5 below",
                    "764 -- the table in docs/notes/handoffs/HANDOFF-03.md section 2.2, "
                    "run 2026-08-17 "
                    "at cap 0.6 / ramp 1.00, no product committed and no raw "
                    "material to recompute it; its own three figures (96 cells, "
                    "8 seeds, 764, 20 below the floor) cannot all hold at once, "
                    "and the only self-consistent reading is 768 minus 4 lost at "
                    "sample level, mechanism unidentified",
                    "this file = n_cells x seeds, both recorded above; the "
                    "separation floor is applied at analysis time and not here, "
                    "so no row is missing from this product on its account",
                ],
                stores="per-sample differences on a joint (reg_coef, max_iter) "
                       "grid, not verdicts; both action dimensions and both "
                       "residuals are kept on every row so failures can be "
                       "binned jointly and so that 'too few steps' and 'wrong "
                       "regularisation scale' can be told apart")
    arrays["meta"] = np.array(json.dumps(meta, indent=1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    # read the product back and name the columns: a column that was added but
    # not written looks exactly like a column that was never asked for
    back = np.load(args.out, allow_pickle=True)
    added = ("reg_coef", "dual_residual", "dual_residual_rel", "primal_residual",
             "c_inf", "d_reserve_revenue_anchor", "d_reserve_price_anchor")
    missing = [k for k in added if k not in back.files]
    if missing:
        raise RuntimeError(f"columns asked for but not in the product: {missing}")
    print(f"\nwrote {args.out} ({Path(args.out).stat().st_size / 1e3:.0f} kB), "
          f"{len(rows)} rows, {len(back.files) - 1} columns")
    print(f"  columns read back: {', '.join(added)} present; "
          f"reg_coef values in product "
          f"{sorted(set(np.asarray(back['reg_coef']).tolist()))}")


if __name__ == "__main__":
    main()
