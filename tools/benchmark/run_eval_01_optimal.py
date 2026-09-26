"""Market 01's optimisation baseline over the twelve evaluation days.

The arm is the exact unit commitment: the schedule a mixed-integer
solve of the full problem of section 6.1 chooses, rather than the one the
three-step approximation lands on.  `tools/commitment/milp_reference.py`
produced those commitments; this settles them.

**Where the prices come from, because a mixed-integer program has none.**  The
optimal commitment is an integer solution and carries no duals at all, so there
is no such thing as "the LMP of the MILP".  The price is produced the only way
this market defines a price: fix the commitment and re-solve the security
constrained economic dispatch, which is step 3 of the three-step clearing and the same solve
every other arm is priced by.  Per-agent profit then comes from the settlement of
section 8 on that solve's awards and prices.  This is not one defensible choice
among several -- it is what "the price under the optimal commitment" means here.

**Same convention as the two open-loop arms of `run_eval_01.py`**, deliberately, so the
three may be read against each other: the same clearing operator at the same
scenario, the same settlement, the same realised demand for the day, and the same
day boundary out of the commitment fixture.

**One thing checked rather than assumed.**  The mixed-integer reference built its own step-1' chain
and therefore its own day boundary, while the arms take theirs from the fixture.
The two agree on all twelve evaluation days bit for bit; they disagree on three
training days (8, 9 and 11), where a degenerate linear program let the two chains
pick different vertices and `u > ROUND_EPS` turned that into a commitment
difference -- the propagation the optimality-gap measurement recorded.  Because no
evaluation day is affected, the commitment settled here was optimised against the
boundary it is settled against.  The check is repeated at run time and refuses to
proceed rather than trusting this note.

CPU only.  Run point stamped into every product.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import (check_split_against_report, split_days, system_cost,
                        write_day, runtime_stamp)

CAP_SCALE = 0.60
RAMP_SCALE = 1.00
MARKUP_MAX = 2.0
VOLL_IN_EFFECT = 10_000.0
T = 24
K = 1
ARM = "optimal"

PRICE_NOTE = (
    "a mixed-integer program has no duals, so there is no LMP of the MILP. The "
    "commitment is fixed and step 3 of ADR-0003 re-solved; the prices and awards "
    "are that solve's, and per-agent profit is the section 8 settlement on them. "
    "Same clearing operator, settlement, demand and day boundary as the honest "
    "and constant arms of ticket 61, so the three are comparable.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--milp-glob", default=str(
        Path(__file__).resolve().parents[2]
        / "scratch" / "milp_reference_29gb_T24_c0.6_r1.00_d*.npz"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--days", default="eval", choices=("eval", "train", "all"))
    args = ap.parse_args()

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")

    import powermarketjax
    print("package:", powermarketjax.__file__, flush=True)

    from powermarketjax.case import load_case
    from powermarketjax.envs.day_ahead import (demand_from_meta, load_commitment,
                                               make_env, segment_costs)
    from powermarketjax.envs.day_ahead.clearing import make_clearing
    from powermarketjax.envs.day_ahead.commitment import FIXTURE_DIR
    from powermarketjax.envs.day_ahead.settlement import make_settlement

    fx_path = FIXTURE_DIR / "day_ahead_commitment_29gb_T24_relax.npz"
    fixture = load_commitment(path=fx_path, n_periods=T)
    meta = fixture["meta"]
    for key, given in (("cap_scale", CAP_SCALE), ("ramp_scale", RAMP_SCALE),
                       ("voll", VOLL_IN_EFFECT)):
        got = meta.get(key)
        if got is None or abs(float(got) - given) > 1e-12:
            raise SystemExit(f"fixture meta {key}={got} disagrees with {given}")

    dates = [str(d) for d in meta["dates"]]
    ok, selected = check_split_against_report(dates, meta["case"])
    print(f"split matches report section 2.2: {ok}")
    if not ok:
        raise SystemExit(f"split rule no longer reproduces the report: {selected}")
    ev, tr = split_days(len(dates), meta["case"])
    positions = {"eval": ev, "train": tr, "all": sorted(ev + tr)}[args.days]

    # the MILP commitments, keyed by the day_index they were solved for
    fdi = np.asarray(fixture["day_index"])
    milp = {}
    for f in sorted(glob.glob(args.milp_glob)):
        z = np.load(f)
        zm = json.loads(str(z["meta"]))
        for key, given in (("cap_scale", CAP_SCALE), ("ramp_scale", RAMP_SCALE)):
            if abs(float(zm[key]) - given) > 1e-12:
                raise SystemExit(f"{f}: {key}={zm[key]} disagrees with {given}")
        for j, d in enumerate(np.asarray(z["day"])):
            milp[int(d)] = dict(u=np.asarray(z["commitment_milp"][j], np.float64),
                                p_init=np.asarray(z["p_init"][j], np.float64),
                                cprev=np.asarray(z["commitment_prev"][j], np.float64),
                                status=int(z["milp_status"][j]),
                                gap=float(z["milp_gap"][j]),
                                obj=float(z["obj_milp"][j]))
    print(f"MILP commitments loaded for {len(milp)} days")

    case = load_case(meta["case"])
    env, spec = make_env(case, fixture, demand_from_meta(meta), n_segments=K,
                         kind="markup", markup_max=MARKUP_MAX,
                         cap_scale=CAP_SCALE, ramp_scale=RAMP_SCALE)
    params = env.make_params(episode_len=1)
    clear, _cspec = make_clearing(case, T, n_segments=K, cap_scale=CAP_SCALE,
                                  ramp_scale=RAMP_SCALE)
    settle = make_settlement(case)
    clear_j = jax.jit(clear)
    settle_j = jax.jit(settle)

    _w, cost_env = segment_costs(case, K)          # truthful offers, markup 1
    n_u = spec["n_units"]
    offer = jnp.broadcast_to(jnp.asarray(cost_env, jnp.float64)[:, :, None],
                             (n_u, K, T))

    run_point = dict(cap_scale=CAP_SCALE, ramp_scale=RAMP_SCALE,
                     window=meta.get("window"), voll=VOLL_IN_EFFECT,
                     markup_max=MARKUP_MAX, episode_len=1,
                     **runtime_stamp(),
                     market="01 day-ahead wholesale", fixture=fx_path.name,
                     note=PRICE_NOTE)

    rows = []
    for pos in positions:
        day = int(fdi[pos])
        if day not in milp:
            raise SystemExit(f"no MILP commitment for day_index {day}")
        m = milp[day]
        # The boundary this commitment was optimised against must be the boundary
        # it is settled against; verified per day rather than trusted.
        #
        # The comparison is against the fixture's own float64 arrays and not
        # against `params`, which carries a float32 copy (see `EnvParams`): on a
        # p_init of order 1e3 MW that rounding is ~1e-4 MW, which is float32
        # precision and not a difference in boundary. Comparing the float32 copy
        # would report a disagreement that does not exist -- it did, before this
        # was fixed. The solve below still uses the `params` copy, because that is
        # the boundary the two arms of `run_eval_01.py` ran on and strict comparability
        # with them is the point.
        fp = np.asarray(fixture["p_init"][pos], np.float64)
        fc = np.asarray(fixture["commitment_prev"][pos], np.float64)
        dp = float(np.abs(fp - m["p_init"]).max())
        dc = float(np.abs(fc - m["cprev"]).max())
        bp = np.asarray(params.boundary_p_init[pos], np.float64)
        bc = np.asarray(params.boundary_commitment[pos], np.float64)
        if dp > 1e-9 or dc > 1e-9:
            raise SystemExit(
                f"day {day}: the MILP chain's boundary differs from the fixture's "
                f"(max |dp_init| {dp:.3e} MW, max |dcommit| {dc:.3e}). The "
                f"commitment was optimised against one boundary and would be "
                f"settled against another, which is not the optimum of the day "
                f"being reported.")

        u = jnp.asarray(m["u"], jnp.float64)
        demand = jnp.asarray(params.actual[pos], jnp.float64)
        out = clear_j(offer, u, demand, jnp.asarray(bp))
        award, lmp, shed = out["award"], out["lmp"], out["shed"]
        money = settle_j(award, lmp, u, jnp.asarray(bc))

        prod = float(np.sum(np.asarray(money["cost"], np.float64)))
        shed_mwh = float(np.sum(np.asarray(shed, np.float64)))
        prof = np.asarray(money["profit"], np.float64)
        sc = system_cost(prod, [shed_mwh], VOLL_IN_EFFECT, 0.0)
        write_day(args.out_dir, ARM, pos, dates[pos],
                  system_cost_value=sc, agent_profit=prof,
                  shed_mwh=np.asarray([shed_mwh]), production_cost=prod,
                  run_point=run_point,
                  extra=dict(fixture_day_index=day, mu=float(out["mu"]),
                             milp_status=m["status"], milp_gap=m["gap"],
                             milp_proved_optimal=bool(m["status"] == 0),
                             obj_milp=m["obj"],
                             revenue=float(np.sum(np.asarray(money["revenue"],
                                                             np.float64))),
                             n_on=int((np.asarray(m["u"]) > 0.5).sum())))
        rows.append((pos, day, dates[pos], sc, float(prof.sum()), shed_mwh,
                     float(out["mu"]), m["status"], m["gap"]))
        flag = "" if m["status"] == 0 else f"  [time limit, gap {m['gap']:.2e}]"
        print(f"  pos {pos:3d} day {day:3d} {dates[pos]}  system_cost {sc:14.4e}  "
              f"profit {float(prof.sum()):14.4e}  shed {shed_mwh:.3e}  "
              f"mu {float(out['mu']):.2e}{flag}", flush=True)

    sc_all = np.array([r[3] for r in rows])
    pr_all = np.array([r[4] for r in rows])
    capped = [r for r in rows if r[7] != 0]
    print(f"\n  optimal: system_cost median {np.median(sc_all):.4e} "
          f"total {sc_all.sum():.4e} | profit total {pr_all.sum():.4e} "
          f"({int((pr_all < 0).sum())} of {len(rows)} days negative)")
    print(f"  days whose commitment was proved optimal: "
          f"{len(rows) - len(capped)} of {len(rows)}")
    if capped:
        print("  NOT proved optimal (the commitment is the best found within the "
              "solver time limit, so this arm is an upper bound on cost for these "
              "days, not the optimum):")
        for r in capped:
            print(f"    day {r[1]} {r[2]}  milp_gap {r[8]:.2e}")


if __name__ == "__main__":
    main()
