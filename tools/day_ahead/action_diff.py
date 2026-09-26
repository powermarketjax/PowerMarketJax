"""The quantity "one unit's action difference across
observations", and its baseline on the existing SHARED-parameter archives.

Read-only.  Nothing under `powermarketjax/` is touched; the package is imported
and called, never edited.

WHAT THIS MEASURES, AND WHY IT IS NOT THE EXISTING CRITERION
------------------------------------------------------------
Let `A` be the observation-by-unit matrix of greedy (mean, no exploration
noise) actions on one fixed observation grid: `A[n, i, d]` is what unit `i`
plays for action coordinate `d` at observation `n`.

  * The collapse note's criterion -- "action dedup rows" -- counts, WITHIN ONE
    ROW of `A`, how many distinct values there are ACROSS UNITS.
  * This device reports, WITHIN ONE COLUMN of `A`, how wide and how many-valued
    one unit's action is ACROSS OBSERVATIONS.

Per-unit parameters make the first go up mechanically (66 independent networks
almost surely emit 66 different numbers) without any state response having come
back.  The second is not inflated by that change, which is the whole reason the
matrix asks for it before the per-unit implementation lands.

Three statistics per unit and action coordinate:

  R_i  range   = max_n A[n,i,d] - min_n A[n,i,d]   (the action's own unit)
  r_i          = R_i / (high - low)                (dimensionless, span share)
  S_i  stdev   = population std over the N observations
  L_i  levels  = number of distinct values across the N observations, with the
                 values snapped to a tolerance first.  Bit-exact dedup is an
                 UPPER bound on "functionally distinct" (the collapse note
                 sec 5.3.4 shows two units 0.1--0.3 apart in pre-activation
                 landing on the same float64), so a tolerance is applied and
                 the bit-exact count is reported alongside.

Aggregation over units is by quantile, never by a single mean: the units split
into responders and non-responders and a mean over the two populations reports
neither.

OBSERVATION GRID (exogenous to the policy under test, on purpose)
-----------------------------------------------------------------
01: `episode_len = 1`, so each evaluation day contributes exactly one
    observation and the observation at reset does not depend on any action.
    N = 12 observations per unit.
02, 03: the episode is the whole day, so the observation at period t depends on
    what was played before it.  The grid is therefore taken from ONE reference
    trajectory -- the market's own baseline/truthful action -- and every archive
    is then evaluated at those same observations.  If each archive walked its
    own trajectory, the grid would move with the policy and two archives'
    numbers would not be on the same axis.

INJECTIONS (`--inject`)
-----------------------
`synthetic`: replaces the loaded parameters with a hand-built network whose
    action is a closed form of one standardised observation coordinate, so the
    device's R_i can be checked against an independently computed prediction.
    A device that reports zero because it feeds one observation twelve times
    fails here.
`obs`: perturbs one raw observation coordinate by `scale * obs_std[dim]` in a
    day-dependent pattern, holding the parameters fixed.  On a collapsed archive
    this answers whether a measured R_i = 0 is a property of the policy at the
    real observations or of a dead pipeline.

Markets 02 and 03 are measured by `tools/real_time/action_diff_rt.py`, not by
this file: `main` refuses `--market 02` and `03`, and the 02/03 products carry
`n_days` / `n_periods` / `within_day_median` keys this device never emits.
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "benchmark"))

import jax                                                        # noqa: E402
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp                                           # noqa: E402

from evaluation import open_day, split_days, subset_position      # noqa: E402
from hyperparams import SHARED                                    # noqa: E402

# --- market 01 run point, the one every 01 archive here was produced on ------
# Imported rather than copied.  `run_rl_01.py` is the script that trained every
# archive this device reads, so its constants ARE the run point; a second copy
# here would be a second place for it to live and a place for the two to drift.
# The fixture name is built the way `run_rl_01.py:288` builds it.
from run_rl_01 import (CAP_SCALE, K, MARKUP_MAX,                  # noqa: E402
                       RAMP_SCALE, T as T01)

FIXTURE_01 = f"day_ahead_commitment_29gb_T{T01}_relax.npz"

#: Where the 30-unit `ever_on` denominator comes from.  This fixture's
#: `ever_on` array was compared element for element on 2026-09-06 with the
#: copy the original runs read, and the two are equal (66 units, 30 on).
CROSS_SECTION_FIXTURE = (REPO / "tests" / "fixtures"
                         / "da_alpha_cross_section_wd052_iter110_seed0.npz")

#: Level-count tolerance, as a share of the action span.  Two actions closer
#: than this count as one level.  1e-9 is far below anything a market can act
#: on and far above float64 round-off in this forward pass, so the count it
#: gives is neither the bit-exact count nor an economically motivated one; the
#: bit-exact count is reported next to it so the gap is visible.
LEVEL_TOL_SHARE = 1e-9


def named_weights(params):
    """Flax pytree -> {"params/Dense_0/kernel": array}, float64 numpy."""
    out = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        key = "/".join(getattr(p, "key", str(getattr(p, "idx", p))) for p in path)
        out[key] = np.asarray(leaf, np.float64)
    return out


def numpy_forward_mean(w, z):
    """Second, independent path to the policy mean: plain numpy, no jax.

    Mirrors `SharedActorCritic.__call__`: two `tanh` hidden layers then a linear
    head.  It exists so the reported actions come out of two paths and not one.
    """
    h = z
    for layer in ("Dense_0", "Dense_1"):
        h = np.tanh(h @ w[f"params/{layer}/kernel"] + w[f"params/{layer}/bias"])
    return h @ w["params/Dense_2/kernel"] + w["params/Dense_2/bias"]


def numpy_to_action(mean, low, high):
    """`policy.to_action` in numpy: squash only the coordinates with both bounds finite."""
    sq = np.isfinite(low) & np.isfinite(high)
    mapped = low + 0.5 * (high - low) * (np.tanh(mean) + 1.0)
    return np.where(sq, mapped, mean)


def per_unit_stats(A, low, high, tol_share=LEVEL_TOL_SHARE):
    """Per-unit, per-coordinate statistics of `A` with shape (N, n_units, act_dim).

    Returns a dict of arrays shaped (n_units, act_dim).  `span` is `high - low`
    where both are finite and `nan` where a coordinate is unbounded, so the
    dimensionless share is simply not defined on the ancillary market's reserve
    columns rather than being divided by an invented number.
    """
    A = np.asarray(A, np.float64)
    n_obs = A.shape[0]
    span = np.where(np.isfinite(low) & np.isfinite(high), high - low, np.nan)
    rng = A.max(axis=0) - A.min(axis=0)
    std = A.std(axis=0)
    with np.errstate(invalid="ignore"):
        share = rng / span
    tol = np.where(np.isfinite(span), tol_share * span, tol_share)
    levels = np.empty(rng.shape, np.int64)
    levels_exact = np.empty(rng.shape, np.int64)
    for i in range(A.shape[1]):
        for d in range(A.shape[2]):
            col = np.sort(A[:, i, d])
            levels_exact[i, d] = int(np.unique(col).size)
            # snap: walk the sorted values and open a new level whenever the
            # gap to the last level's representative exceeds the tolerance
            n_lv, rep = 1, col[0]
            for v in col[1:]:
                if v - rep > tol[i, d]:
                    n_lv += 1
                    rep = v
            levels[i, d] = n_lv
    return dict(n_obs=n_obs, range=rng, share=share, std=std,
                levels=levels, levels_exact=levels_exact, span=span)


def q(v, qs=(0.0, 0.25, 0.5, 0.75, 1.0)):
    v = np.asarray(v, np.float64)
    return [float(np.quantile(v, x)) for x in qs]


def summarise(st, d, mask=None, label=""):
    """One coordinate's per-unit distribution, as quantiles and counts."""
    m = np.ones(st["range"].shape[0], bool) if mask is None else np.asarray(mask, bool)
    rng, share, std = st["range"][m, d], st["share"][m, d], st["std"][m, d]
    lv, lvx = st["levels"][m, d], st["levels_exact"][m, d]
    return dict(
        label=label, coord=int(d), n_units=int(m.sum()), n_obs=int(st["n_obs"]),
        range_q=q(rng), share_q=q(share), std_q=q(std),
        levels_q=q(lv), levels_exact_q=q(lvx),
        range_median=float(np.median(rng)), share_median=float(np.median(share)),
        levels_median=float(np.median(lv)),
        n_flat=int((rng == 0.0).sum()),
        n_below_1e3=int((rng < 1e-3).sum()),
        n_levels_1=int((lv == 1).sum()),
        span=float(st["span"][0, d]) if np.isfinite(st["span"][0, d]) else None,
    )


def cross_unit_dedup(A):
    """The EXISTING criterion, computed on the same matrix, for the comparison.

    Number of distinct action rows across units, one count per observation.
    """
    return [int(np.unique(A[n], axis=0).shape[0]) for n in range(A.shape[0])]


def spearman(x, y):
    def rank(v):
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v), np.float64)
        r[order] = np.arange(len(v), dtype=np.float64)
        for val in np.unique(v):
            mm = v == val
            if mm.sum() > 1:
                r[mm] = r[mm].mean()
        return r
    rx, ry = rank(np.asarray(x, np.float64)), rank(np.asarray(y, np.float64))
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


# --------------------------------------------------------------------------
# market 01
# --------------------------------------------------------------------------
def build_01(cfg_overrides=None):
    """Return `(four, env, params, bounds, obs_grid, meta)` for market 01.

    `obs_grid` has shape (12, n_units, obs_dim) and is RAW (not standardised):
    each archive carries its own `obs_mean`/`obs_std` and standardising here
    would freeze one archive's statistics onto all of them.
    """
    from powermarketjax.case import load_case
    from powermarketjax.envs.day_ahead import (load_commitment, load_gb_demand,
                                               make_env)
    from powermarketjax.envs.day_ahead.commitment import FIXTURE_DIR
    from powermarketjax.learning.adapters import unpack_env
    from powermarketjax.learning.policy import bounds_for

    fx = load_commitment(path=FIXTURE_DIR / FIXTURE_01, n_periods=T01)
    ev_days, tr_days = split_days(len(fx["meta"]["dates"]), fx["meta"]["case"])
    case_obj = load_case(fx["meta"]["case"])
    env, spec = make_env(case_obj, subset_position(fx, ev_days), load_gb_demand(),
                         n_segments=K, kind="markup", markup_max=MARKUP_MAX,
                         cap_scale=CAP_SCALE, ramp_scale=RAMP_SCALE)
    four = unpack_env((env, spec))
    p = env.make_params(episode_len=1)
    bounds = bounds_for(four[3])
    day_of = lambda st: int(st.cursor)
    Z = []
    for pos in range(len(ev_days)):
        _k, state = open_day(env.reset, p, pos, day_of)
        Z.append(np.asarray(env.get_obs(state, p), np.float64))
    Z = np.stack(Z)
    meta = dict(market="01", case=str(fx["meta"]["case"]), n_eval_days=len(ev_days),
                episode_len=1, n_obs_per_unit=len(ev_days),
                grid="one observation per evaluation day, taken at reset; "
                     "independent of the policy because episode_len = 1")
    return four, env, p, bounds, Z, meta


# --------------------------------------------------------------------------
def declared_layout(path):
    """`meta["per_agent_params"]` for an 01 archive, or None if it predates it.

    The stamp is read rather than inferred from the leaf shapes: the two layouts
    have the SAME leaf count and differ only in a leading axis, so a reader that
    guesses gets a different policy rather than an error on a market where
    `obs_dim` happens to equal `n_agents`.  `run_rl_01.py` writes this key for
    exactly this reason; `None` means the file was written before it existed,
    and every such file in this repository is shared.
    """
    d = np.load(path, allow_pickle=True)
    if "meta" not in d.files:
        return None
    return json.loads(str(d["meta"])).get("per_agent_params")


def load_archive_01(path, treedef, n_leaves):
    d = np.load(path, allow_pickle=True)
    saved = [d[f"p{i}"] for i in range(n_leaves) if f"p{i}" in d.files]
    if len(saved) != n_leaves:
        raise SystemExit(f"{path}: {len(saved)} leaves, tree has {n_leaves}")
    params = jax.tree_util.tree_unflatten(treedef, [jnp.asarray(a) for a in saved])
    return params, np.asarray(d["obs_mean"], np.float64), np.asarray(d["obs_std"], np.float64)


def numpy_forward_mean_per_agent(w, z):
    """The second path under per-agent parameters: unit i through network i.

    A plain Python loop over the agent axis, deliberately: it shares no axis
    bookkeeping with `ippo._apply_per_agent`, so a `moveaxis` that lined the
    parameter axis up against the wrong axis would show here as a mismatch
    rather than being reproduced by both paths.
    """
    out = []
    for i in range(z.shape[0]):
        h = z[i]
        for layer in ("Dense_0", "Dense_1"):
            h = np.tanh(h @ w[f"params/{layer}/kernel"][i]
                        + w[f"params/{layer}/bias"][i])
        out.append(h @ w["params/Dense_2/kernel"][i] + w["params/Dense_2/bias"][i])
    return np.stack(out)


def synthetic_params(template, obs_dim, act_dim, dim, c0=0.5, c1=1.0, c2=1.0):
    """A network whose mean is `c2 * tanh(c1 * tanh(c0 * z[dim]))`, exactly.

    Every other weight is zero, so the closed form can be recomputed outside
    this device and compared with what it reports.
    """
    w = {k: np.zeros_like(v) for k, v in named_weights(template).items()}
    w["params/Dense_0/kernel"][dim, 0] = c0
    w["params/Dense_1/kernel"][0, 0] = c1
    w["params/Dense_2/kernel"][0, :] = c2
    leaves, treedef = jax.tree_util.tree_flatten(template)
    named = named_weights(template)
    order = list(named)
    rebuilt = jax.tree_util.tree_unflatten(
        treedef, [jnp.asarray(w[k]) for k in order])
    return rebuilt, (c0, c1, c2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="01", choices=("01", "02", "03"))
    ap.add_argument("--archives", nargs="*", default=None,
                    help="archive paths; label:path also accepted")
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cross", default=str(CROSS_SECTION_FIXTURE),
                    help="product carrying the `ever_on` mask that defines the "
                         "30-unit denominator; absent, only the 66-unit "
                         "quantiles are reported")
    ap.add_argument("--inject", default="none",
                    choices=("none", "synthetic", "obs"))
    ap.add_argument("--inject-scale", default="1.0",
                    help="comma-separated list; the obs injection is run once per value so the smallest one that bites can be read off")
    ap.add_argument("--inject-dim", type=int, default=-1,
                    help="-1 picks the observation coordinate with the largest "
                         "Dense_0 row norm in the archive under test")
    args = ap.parse_args()

    import powermarketjax
    print("package:", powermarketjax.__file__, flush=True)
    print("devices:", jax.devices(), flush=True)

    from powermarketjax.learning.ippo import make_greedy_action, make_ippo

    if args.market != "01":
        raise SystemExit(
            "markets 02/03 go through tools/real_time/action_diff_rt.py "
            "instead of this file; see this module's docstring")

    four, env, p, bounds, Z, meta = build_01()
    spec = four[3]
    n_units, obs_dim = Z.shape[1], Z.shape[2]
    low_j, high_j = bounds
    low = np.asarray(low_j, np.float64)
    high = np.asarray(high_j, np.float64)
    act_dim = low.shape[-1] if low.ndim > 1 else 1
    low2 = low.reshape(n_units, act_dim)
    high2 = high.reshape(n_units, act_dim)

    # --- the grid's non-triviality, asserted before anything is reported -----
    # A device that hands the same observation to the policy N times reports
    # R_i = 0 for every policy, which reads exactly like a collapsed policy.
    per_unit_distinct = [int(np.unique(Z[:, i, :], axis=0).shape[0])
                         for i in range(n_units)]
    if min(per_unit_distinct) < 2:
        raise SystemExit(
            f"observation grid is degenerate: unit "
            f"{int(np.argmin(per_unit_distinct))} sees the same observation on "
            f"all {Z.shape[0]} grid points, so no action difference could exist")
    obs_range = (Z.max(axis=0) - Z.min(axis=0))
    print(f"grid: {Z.shape[0]} observations x {n_units} units x {obs_dim} coords; "
          f"distinct observations per unit min {min(per_unit_distinct)} max "
          f"{max(per_unit_distinct)}", flush=True)
    print(f"      raw observation range per unit: median over units of the "
          f"max-coordinate range = {np.median(obs_range.max(axis=1)):.4g}",
          flush=True)

    ever_on = None
    cs_path = Path(args.cross)
    if cs_path.exists():
        cs = np.load(cs_path, allow_pickle=True)
        ever_on = np.asarray(cs["ever_on"], bool)
        print(f"      truthful arm: {int(ever_on.sum())} of {len(ever_on)} units "
              f"on line at least once", flush=True)

    cfg = SHARED
    init, _it = make_ippo(four, bounds, cfg, jnp.zeros(obs_dim), jnp.ones(obs_dim))
    template, *_ = init(jax.random.PRNGKey(0), p)
    leaves, treedef = jax.tree_util.tree_flatten(template)
    # the per-agent tree, built the same way so an archive of either layout has
    # a template with the right leaf shapes to unflatten into
    init_pa, _it2 = make_ippo(four, bounds, cfg, jnp.zeros(obs_dim),
                              jnp.ones(obs_dim), per_agent_params=True)
    template_pa, *_ = init_pa(jax.random.PRNGKey(0), p)
    leaves_pa, treedef_pa = jax.tree_util.tree_flatten(template_pa)

    todo = []
    if args.ckpt_dir:
        for c in sorted(Path(args.ckpt_dir).glob(f"seed{args.seed}_iter*.npz")):
            todo.append((c.stem.split("iter")[1], c))
    for a in (args.archives or []):
        label, _, path = a.partition(":")
        if not path:
            label, path = Path(a).stem, a
        todo.append((label, Path(path)))
    if not todo:
        raise SystemExit("nothing to measure: give --ckpt-dir and/or --archives")

    scales = [float(x) for x in str(args.inject_scale).split(",") if x.strip()]

    rows = []
    for label, path in todo:
      for scale in (scales if args.inject == "obs" else [0.0]):
          pa = bool(declared_layout(path))
          params, om, os_ = load_archive_01(
              path, treedef_pa if pa else treedef,
              len(leaves_pa) if pa else len(leaves))
          cfg_a = dataclasses.replace(SHARED, **{
              k: (tuple(v) if isinstance(v, list) else v)
              for k, v in json.loads(str(np.load(path, allow_pickle=True)["config"])).items()
              if hasattr(SHARED, k)})
          Zi = Z.copy()
          note = ""
          if args.inject == "synthetic":
              dim = args.inject_dim if args.inject_dim >= 0 else 0
              params, (c0, c1, c2) = synthetic_params(template, obs_dim, act_dim, dim)
              note = f"synthetic policy on standardised coord {dim}, c=({c0},{c1},{c2})"
          elif args.inject == "obs":
              w = named_weights(params)
              dim = (args.inject_dim if args.inject_dim >= 0
                     else int(np.argmax(np.abs(w["params/Dense_0/kernel"]).sum(axis=1))))
              n = Zi.shape[0]
              pattern = (np.arange(n) - (n - 1) / 2.0) / ((n - 1) / 2.0)
              Zi[:, :, dim] += scale * os_[dim] * pattern[:, None]
              note = (f"obs coord {dim} perturbed by "
                      f"{scale:+.4g} * obs_std over a linear pattern")

          greedy = jax.jit(make_greedy_action(spec, bounds, cfg_a,
                                              jnp.asarray(om), jnp.asarray(os_),
                                              per_agent_params=pa))
          A_pkg = np.stack([np.asarray(greedy(params, jnp.asarray(z)), np.float64)
                            for z in Zi])
          if A_pkg.ndim == 2:                       # 01: (N, n_units)
              A_pkg = A_pkg[:, :, None]
          # second path: plain numpy, same standardisation, no jax
          w = named_weights(params)
          fwd = numpy_forward_mean_per_agent if pa else numpy_forward_mean
          A_np = np.stack([numpy_to_action(fwd(w, (z - om) / os_),
                                           low2, high2) for z in Zi])
          dpath = float(np.abs(A_pkg - A_np).max())

          if args.inject == "synthetic":
              # third, fully independent computation of the same number: the
              # closed form the synthetic network was built to realise.  It shares
              # no line of code with either path above.
              zz = (Zi - om) / os_
              mean_cf = c2 * np.tanh(c1 * np.tanh(c0 * zz[:, :, dim]))
              a_cf = low2[:, 0] + 0.5 * (high2[:, 0] - low2[:, 0]) * (np.tanh(mean_cf) + 1.0)
              r_cf = a_cf.max(axis=0) - a_cf.min(axis=0)
              r_dev = A_pkg[:, :, 0].max(axis=0) - A_pkg[:, :, 0].min(axis=0)
              print(f"    closed form vs device: max|action diff| "
                    f"{np.abs(a_cf - A_pkg[:, :, 0]).max():.3e}, max|R diff| "
                    f"{np.abs(r_cf - r_dev).max():.3e}, "
                    f"closed-form R median {np.median(r_cf):.6f}", flush=True)

          st = per_unit_stats(A_pkg, low2, high2)
          dedup = cross_unit_dedup(A_pkg)
          s_all = summarise(st, 0, None, "all units")
          s_on = summarise(st, 0, ever_on, "ever-on units") if ever_on is not None else None
          if s_on is not None:
              print(f"    ever-on 30 units: R median {s_on['range_median']:.6f} "
                    f"({s_on['share_median']*100:.3f}% of span), L median "
                    f"{s_on['levels_median']:.1f}, flat {s_on['n_flat']}/30",
                    flush=True)
          row = dict(label=label + (f"+obs{scale:g}" if args.inject == "obs" else ""),
                     per_agent_params=pa, path=str(path), note=note,
                     n_obs=int(Z.shape[0]), n_units=int(n_units),
                     two_path_max_abs_diff=dpath,
                     cross_unit_dedup=dedup,
                     cross_unit_dedup_min=int(min(dedup)),
                     cross_unit_dedup_max=int(max(dedup)),
                     all_units=s_all, ever_on=s_on,
                     range_per_unit=[float(x) for x in st["range"][:, 0]],
                     levels_per_unit=[int(x) for x in st["levels"][:, 0]])
          rows.append(row)
          print(f"  {label:>10}  R median {s_all['range_median']:.6f} "
                f"({s_all['share_median']*100:.3f}% of span)  "
                f"R q0/q25/q75/q1 "
                f"{s_all['range_q'][0]:.4f}/{s_all['range_q'][1]:.4f}/"
                f"{s_all['range_q'][3]:.4f}/{s_all['range_q'][4]:.4f}  "
                f"L median {s_all['levels_median']:.1f}  flat {s_all['n_flat']}  "
                f"cross-unit dedup {min(dedup)}-{max(dedup)}  "
                f"2-path |d| {dpath:.1e}" + (f"  [{note}]" if note else ""),
                flush=True)

    if len(rows) > 2:
        lab = [r["label"] for r in rows]
        rmed = [r["all_units"]["range_median"] for r in rows]
        dmin = [r["cross_unit_dedup_min"] for r in rows]
        print(f"\nspearman(R median, cross-unit dedup min) over {len(rows)} "
              f"archives: {spearman(rmed, dmin):+.3f}")

    if args.out:
        np.savez(args.out, rows=json.dumps(rows), meta=json.dumps(meta),
                 obs_grid_distinct_per_unit=np.asarray(per_unit_distinct),
                 inject=args.inject, inject_scale=str(args.inject_scale))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
