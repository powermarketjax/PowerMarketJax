r"""`reg_coef` 1e-18 or 1e-20: which one agrees with an independent solver?

The joint grid established that 1e-20 converges where the adopted 1e-18 leaves a
residue no iteration count clears, and **that is not enough to move the
constant**: at 60 steps 1e-20 differs from its own converged solve by 7.35 \$ and
from the adopted calibration's answer by 1.20e+05 \$.  Both rungs are the same
interior-point method, so the grid cannot say which of the two answers is right
-- only that they differ.  HiGHS is a different algorithm and can.

**The subset is drawn before any result is looked at.**  Uniformly over the
(cell, seed) pairs of the grid, with its own generator and a fixed seed, so the
selection cannot follow the quantity being reported.  Picking the two failing
samples, or the samples where the rungs disagree most, would bias exactly the
direction the answer lies in.

**The judgement is on the consumed quantities, not the objective.**  An earlier
measurement found that the objective is blind to this failure: a
relative objective difference of 5.86e-06 sat beside a reserve-leg revenue
difference of 159 600 \$ at the same operating point.  So the comparison is on
per-unit revenue in **both** legs and on both prices, and the objective is
recorded beside them rather than used.

**The energy leg was missing until 2026-09-05.**  The criterion has always named
both legs, but this file only ever pulled the reserve price and reserve
quantities out of HiGHS, so the 768-cell census of that date answered for one
leg while reading as though it answered for two.  The energy leg needs a nodal
price, which HiGHS does not return; see `highs` for how it is reconstructed and
for the two checks that stop the run if the duals were read wrongly.

**The offers are not re-derived here.**  `sample_actions` is imported from the
calibration tool and replayed through the same generator protocol, so the
operating points are the same ones by construction rather than by transcription.
A re-implementation that drew "the same" offers would be a second thing to get
wrong, and its disagreement would be indistinguishable from the one being
measured.

**Two iteration counts, because one cannot separate the two axes.**  At the
adopted 60 the comparison answers "which rung is right where the market runs";
at the reference count it answers "which rung is right once step count is
excluded by construction".  A rung that agrees with HiGHS only at 240 is a
different finding from one that agrees at both.

    PYTHONPATH=<repo>:<repo>/tools/ancillary python tools/ancillary/highs_arbitration.py
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax                                                        # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                           # noqa: E402
from scipy.optimize import linprog                                # noqa: E402

from powermarketjax.case import load_case                         # noqa: E402
from powermarketjax.envs.ancillary.clearing import make_clearing   # noqa: E402
from powermarketjax.envs.ancillary.env import MU_TOL               # noqa: E402
from powermarketjax.envs.day_ahead.clearing import segment_costs   # noqa: E402
from tests.envs.ancillary.reference import build_lp               # noqa: E402

import tests.envs.ancillary.test_clearing_l2 as L2                # noqa: E402
from action_dim_calibration import (PI_SCALE, REFERENCE_ITER,     # noqa: E402
                                    SEPARATION_FLOOR, sample_actions)

#: The two rungs under arbitration.  The adopted value first; the grid found the
#: second strictly better on both flag statistics at every iteration count.
RUNGS = (1e-18, 1e-20)

#: The seed of the *subset* draw, which is a different generator from the one
#: that draws offers.  Fixed and declared so the subset is reproducible and so
#: it is visibly not a function of any result.
SUBSET_SEED = 20260817


#: Largest tolerated relative stationarity residual when reading HiGHS duals.
#: Measured 1.42e-17 on (day 12, hour 0, seed 4) 2026-09-05 on this CPU box,
#: while the three wrong sign conventions sat at 5.0e-02, 5.8e-02 and 7.6e-02 --
#: nine orders of separation, so this gate is a convention check and not a
#: precision one.  It exists because the energy leg below is *reconstructed*
#: from duals rather than read off, and a sign slip would produce a nodal price
#: that looks entirely ordinary.
DUAL_RESIDUAL_TOL = 1e-9


def highs(lp):
    """Both legs from HiGHS, or None if it fails.

    The reserve leg is read off.  The energy leg is not: HiGHS does not return a
    nodal price, so it has to come out of the duals, and that is a derivation
    with its own way of being wrong.  Two things keep it honest, and neither
    looks at the clearing operator, because the question here is "were the duals
    read correctly", not "is the operator right":

    * the stationarity identity ``c = A_ub^T mu + A_eq^T lam + z_lo + z_up``
      has to close, which pins the sign convention of all four multipliers;
    * the nodal price is taken from the **load-shed column**, which *is* an
      injection at its bus (unit coefficient on the balance row, +-PTDF on the
      line rows), so no sign is hand-derived here.  Its two constructions --
      through ``A^T y`` and through the shed column's own reduced cost -- are
      the same number by that identity, and are both computed and compared.
    """
    r = linprog(lp["c"], A_ub=lp["A_ub"], b_ub=lp["b_ub"], A_eq=lp["A_eq"],
                b_eq=lp["b_eq"],
                bounds=np.stack([lp["lo"], lp["hi"]], 1), method="highs")
    if r.status != 0:
        return None
    lam = -np.asarray(r.ineqlin.marginals)
    price = lam[lp["row0"]["rd"]: lp["row0"]["rd"] + lp["P"]] / lp["period_hours"]
    x = np.asarray(r.x)
    reserve = x[lp["r0"]: lp["s0"]].reshape(lp["n_u"], lp["P"])

    mu_i = np.asarray(r.ineqlin.marginals)
    lam_e = np.asarray(r.eqlin.marginals)
    z_lo = np.asarray(r.lower.marginals)
    z_up = np.asarray(r.upper.marginals)
    aty = lp["A_ub"].T @ mu_i + lp["A_eq"].T @ lam_e
    c_inf = max(float(np.abs(lp["c"]).max()), 1.0)
    stat = float(np.abs(lp["c"] - (aty + z_lo + z_up)).max()) / c_inf

    b0, n_b, ph = lp["s0"], lp["n_b"], lp["period_hours"]
    shed_cols = slice(b0, b0 + n_b)
    lmp_a = (aty[shed_cols] + z_up[shed_cols]) / ph            # through A^T y
    lmp_b = (lp["c"][shed_cols] - z_lo[shed_cols]) / ph        # through reduced cost
    d_lmp_ab = float(np.abs(lmp_a - lmp_b).max())

    n_u, K = lp["n_u"], lp["K"]
    g = x[lp["g0"]: lp["g0"] + n_u * K].reshape(n_u, K)
    award = lp["must"] + g.sum(1)
    return dict(reserve_price=price, reserve=reserve, z=float(lp["c"] @ x),
                lmp=lmp_a, award=award,
                stationarity_rel=stat, d_lmp_two_ways=d_lmp_ab)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixture",
                    default="tests/fixtures/day_ahead_commitment_29gb_T24_relax_seasons.npz")
    ap.add_argument("--cap-scale", type=float, default=0.6)
    ap.add_argument("--ramp-scale", type=float, default=1.0)
    ap.add_argument("--volr", type=float, default=250.0)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--periods", type=int, default=24)
    ap.add_argument("--subset", type=int, default=96,
                    help="how many (cell, seed) pairs to arbitrate; the "
                         "denominator is reported beside every figure")
    # The imported `PI_SCALE` is 50, section 22's adopted value (an offer reaches
    # the cap at a raw action of about 5 -- true at 50 and not at 10, where the cap
    # needs an action of 25 and is unreachable).  This comment read "the
    # calibration tool carries 10.0, which is that tool's own choice", which was
    # true until that constant was moved; the help text below prints one number
    # twice for the same reason.  10.0 is what the committed
    # `highs_arbitration_seasons.npz` ran at, and it records it as `pi_scale`,
    # where `ancillary_highs_targeted_pi50.npz` records `pi_scale_in_force` -- two
    # products of this file, two key names.  The verdict does not depend on it, since
    # both rungs see the same inputs, but the money does: the reserve offers are
    # five times larger at the adopted scale.  A flag rather than an edit to the
    # imported constant, which is another line's device.
    ap.add_argument("--pi-scale", type=float, default=PI_SCALE,
                    help=f"reserve offer price scale; defaults to the "
                         f"calibration tool's {PI_SCALE:g}, the specification "
                         f"adopts 50")
    # Targeted mode.  The uniform subset above answers "how often"; it cannot
    # answer "was this particular sample wrong", and a 1-in-764 event is not
    # something a 96-draw subset can be expected to contain (expected count
    # 0.13).  A named cell is therefore a follow-up on a stated hypothesis, not
    # a sample, and any figure it produces must be reported as such.
    ap.add_argument("--targets", nargs="+", default=None,
                    help="day:hour:seed triples to arbitrate instead of the "
                         "uniform subset; TARGETED, not a sample")
    ap.add_argument("--iters", type=int, nargs="+", default=[60, REFERENCE_ITER])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    case = load_case(L2.CASE)
    _, cost = segment_costs(case, 1)
    fx = np.load(args.fixture, allow_pickle=True)
    fmeta = json.loads(str(fx["meta"])) if "meta" in fx.files else {}
    got = (fmeta.get("cap_scale"), fmeta.get("ramp_scale"))
    if None not in got and got != (args.cap_scale, args.ramp_scale):
        raise ValueError(
            f"fixture was built at cap_scale/ramp_scale {got}, this run was "
            f"asked for {(args.cap_scale, args.ramp_scale)}")
    commitment = fx["commitment"]
    pmin = np.asarray(case.unit_p_min, np.float64)
    pmax = np.asarray(case.unit_p_max, np.float64)
    n_units = len(pmin)

    builders, specs = {}, {}
    for rung in RUNGS:
        for it in args.iters:
            builders[(rung, it)], specs[(rung, it)] = make_clearing(
                case, L2.THETA, args.volr, n_segments=1,
                cap_scale=args.cap_scale, ramp_scale=args.ramp_scale,
                period_hours=L2.DELTA, max_iter=it, reg_coef=rung)
    for (rung, it), sp in specs.items():
        if (float(sp["reg_coef"]), int(sp["max_iter"])) != (float(rung), int(it)):
            raise RuntimeError(f"operator reports {(sp['reg_coef'], sp['max_iter'])}"
                               f" for {(rung, it)}")
    spec = specs[(RUNGS[0], args.iters[0])]
    n_prod = int(spec["n_prod"])
    clear = {k: jax.jit(v) for k, v in builders.items()}
    print(f"in force: cap_scale {spec['cap_scale']} ramp_scale "
          f"{spec['ramp_scale']} voll {spec['voll']:g} volr {spec['volr']:g} "
          f"dual_start {spec['dual_start']!r}")
    print(f"rungs {RUNGS} x iters {tuple(args.iters)}, judged against HiGHS")

    cells = [(d, h) for d in range(min(args.periods, commitment.shape[0]))
             for h in (0, 6, 12, 18)]
    population = [(i, s) for s in range(args.seeds) for i in range(len(cells))]
    # drawn before any result exists, by its own generator
    if args.targets:
        wanted = set()
        for t in args.targets:
            d, h, sd = (int(v) for v in t.split(":"))
            wanted.add((cells.index((d, h)), sd))
        print(f"TARGETED at {sorted(args.targets)} -- {len(wanted)} named "
              f"(cell, seed) pairs.  This is a follow-up on a stated "
              f"hypothesis, not a uniform sample; do not quote a rate from it")
    else:
        pick = np.random.default_rng(SUBSET_SEED).choice(
            len(population), size=min(args.subset, len(population)), replace=False)
        wanted = {population[int(p)] for p in pick}
        print(f"subset {len(wanted)} of {len(population)} (cell, seed) pairs, "
              f"drawn uniformly with generator seed {SUBSET_SEED}")

    rows, n_highs_failed = [], 0
    for s in range(args.seeds):
        rng = np.random.default_rng(s)
        for i, (d, h) in enumerate(cells):
            # the generator must be advanced for every cell, drawn or not, or
            # the offers of the drawn ones would not be the grid's offers
            markup, alpha = sample_actions(rng, n_units, n_prod, markup_hi=2.0)
            if (i, s) not in wanted:
                continue
            u = commitment[d, :, h].astype(np.float64)
            offer = markup[:, None] * cost
            offer_res = np.asarray(jax.nn.softplus(jnp.asarray(alpha)) * args.pi_scale)
            supply = (np.asarray(spec["res_cap"]) * u[:, None]).sum(0)
            demand = float((pmin * u).sum()) + 0.85 * float(((pmax - pmin) * u).sum())
            p_prev = (pmin + 0.85 * (pmax - pmin)) * u
            d_res = 0.70 * supply

            lp = build_lp(case, offer, offer_res, u, demand, d_res, p_prev,
                          L2.THETA, args.volr, args.cap_scale, args.ramp_scale,
                          L2.DELTA, 1)
            ref = highs(lp)
            if ref is None:
                n_highs_failed += 1
                continue
            if ref["stationarity_rel"] > DUAL_RESIDUAL_TOL:
                raise RuntimeError(
                    f"HiGHS duals did not close at cell {(d, h)} seed {s}: "
                    f"relative stationarity residual {ref['stationarity_rel']:.3e} "
                    f"> {DUAL_RESIDUAL_TOL:g}; the energy leg below would be "
                    f"reconstructed from multipliers whose convention is not "
                    f"established, so this stops rather than reports")
            if ref["d_lmp_two_ways"] > 1e-6:
                raise RuntimeError(
                    f"the two nodal-price constructions disagree by "
                    f"{ref['d_lmp_two_ways']:.3e} $/MWh at {(d, h)} seed {s}")
            ref_rev = ref["reserve"] * u[:, None] * ref["reserve_price"][None, :]
            ref_en_rev = ref["award"] * u * ref["lmp"][lp["unit_bus"]]

            args_in = (jnp.asarray(offer), jnp.asarray(offer_res), jnp.asarray(u),
                       jnp.asarray(demand), jnp.asarray(d_res), jnp.asarray(p_prev))
            for rung in RUNGS:
                for it in args.iters:
                    out = {k: np.asarray(v)
                           for k, v in clear[(rung, it)](*args_in).items()}
                    rev = out["reserve"] * out["reserve_price"][None, :]
                    en_rev = out["award"] * out["lmp"][lp["unit_bus"]]
                    rows.append(dict(
                        day=d, hour=h, seed=s, cell=i, reg_coef=rung, iters=it,
                        mu=float(out["mu"]),
                        # the two consumed quantities, against a different algorithm
                        d_reserve_revenue_highs=float(np.max(np.abs(rev - ref_rev))),
                        d_reserve_price_highs=float(np.max(np.abs(
                            out["reserve_price"] - ref["reserve_price"]))),
                        # the other consumed leg, added 2026-09-05: the criterion
                        # names both, and until now only the reserve one was met
                        d_energy_revenue_highs=float(np.max(np.abs(
                            en_rev - ref_en_rev))),
                        d_lmp_highs=float(np.max(np.abs(
                            out["lmp"] - ref["lmp"]))),
                        stationarity_rel_highs=ref["stationarity_rel"],
                        highs_reserve_price_max=float(ref["reserve_price"].max()),
                        # recorded beside, never used as the judgement
                        d_objective_rel_highs=float(
                            abs(float(out["z"]) - ref["z"]) / max(abs(ref["z"]), 1e-30)),
                    ))
        print(f"  seed {s} done ({len(rows)} rows)", flush=True)

    if not rows:
        raise RuntimeError("no row produced; the subset or the gate is wrong")
    keys = list(rows[0])
    arrays = {k: np.array([r[k] for r in rows]) for k in keys}
    meta = dict(
        case=L2.CASE,
        cap_scale=float(spec["cap_scale"]), ramp_scale=float(spec["ramp_scale"]),
        voll_in_force=float(spec["voll"]), volr_in_force=float(spec["volr"]),
        dual_start_in_force=str(spec["dual_start"]),
        commitment_fixture=str(args.fixture),
        rungs=list(RUNGS), iters=list(args.iters), pi_scale_in_force=float(args.pi_scale),
        subset_seed=(None if args.targets else SUBSET_SEED),
        targeted=list(args.targets) if args.targets else None,
        subset_size=len(wanted),
        population_size=len(population), seeds=args.seeds,
        separation_floor=SEPARATION_FLOOR,
        separation_note="no separation filter is applied here; the floor is "
                        "recorded so the subset can be split at analysis time. "
                        "The joint grid measured 4 of 768 below it, and that "
                        "count is a property of the offer draw alone, "
                        "independent of volr, reg_coef and max_iter",
        n_highs_failed=n_highs_failed,
        mu_tol=MU_TOL,
        dual_residual_tol=DUAL_RESIDUAL_TOL,
        judged_on="per-unit reserve leg revenue and reserve price against "
                  "HiGHS, a different algorithm; the objective is recorded and "
                  "not used, because section 2.2 measured it blind to this "
                  "failure (5.86e-06 relative beside 159 600 $)",
        subset_note="drawn uniformly over (cell, seed) pairs by a generator "
                    "seeded independently of any result, before any result "
                    "existed; not the failing samples and not the "
                    "largest-disagreement samples",
    )
    arrays["meta"] = np.array(json.dumps(meta, indent=1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    back = np.load(args.out, allow_pickle=True)
    for k in ("d_reserve_revenue_highs", "d_reserve_price_highs", "reg_coef",
              "d_energy_revenue_highs", "d_lmp_highs"):
        if k not in back.files:
            raise RuntimeError(f"column {k} asked for but not in the product")
    print(f"\nwrote {args.out}, {len(rows)} rows, HiGHS failures "
          f"{n_highs_failed}/{len(wanted)}")

    print(f"\n=== judgement, on {len(wanted)} drawn (cell, seed) pairs ===")
    reg = arrays["reg_coef"]
    it_a = arrays["iters"]
    print(f"{'reg_coef':>9s} {'iters':>6s} | "
          f"{'worst d reserve revenue':>24s} {'worst d reserve price':>22s} "
          f"{'worst d energy revenue':>23s} {'worst d lmp':>12s} "
          f"{'unconverged':>12s}")
    for rung in RUNGS:
        for it in args.iters:
            m = (reg == rung) & (it_a == it)
            print(f"{rung:>9.0e} {it:>6d} | "
                  f"{arrays['d_reserve_revenue_highs'][m].max():>24.3e} "
                  f"{arrays['d_reserve_price_highs'][m].max():>22.3e} "
                  f"{arrays['d_energy_revenue_highs'][m].max():>23.3e} "
                  f"{arrays['d_lmp_highs'][m].max():>12.3e} "
                  f"{int((arrays['mu'][m] > MU_TOL).sum()):>5d}/{int(m.sum()):<6d}")


if __name__ == "__main__":
    main()
