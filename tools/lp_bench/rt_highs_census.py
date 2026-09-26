"""Nodal prices of the real-time clearing operator against HiGHS, market 02.

The operator is `make_rt_clearing` (one half-hour period, `MAX_ITER` iterations).
States are taken from the day-ahead commitment fixture over the given evaluation
days: the commitment is the fixture's hour h, the demand is that hour's realised
demand, and `p_init` is HiGHS's output at hour h-1 of the same day's 24-period
day-ahead LP (the fixture's `p_init` at h = 0).  The judgement is on prices and
the objective only; output differences are checked separately for whether they
fall on tied units.

The fixture is built by `tools/commitment/precommit.py`; it must be the
`case29gb` T=24 relaxed commitment at cap_scale 0.6 and ramp_scale 1.0.

    JAX_PLATFORMS=cpu python tools/lp_bench/rt_highs_census.py \\
        --fixture <commitment.npz> --out runs/rt_highs/census.npy [DAY ...]
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np                                                # noqa: E402
import jax                                                        # noqa: E402
import jax.numpy as jnp                                           # noqa: E402

jax.config.update("jax_enable_x64", True)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "benchmark"))

from powermarketjax.case import load_case                         # noqa: E402
from powermarketjax.envs.day_ahead import segment_costs           # noqa: E402
from powermarketjax.envs.real_time.clearing import make_rt_clearing, MAX_ITER  # noqa: E402
from powermarketjax.envs.day_ahead.commitment import load_commitment  # noqa: E402
from powermarketjax.envs.day_ahead.demand import demand_from_meta  # noqa: E402
from tests.envs.day_ahead import reference                        # noqa: E402
from evaluation import YEAR_EVAL_OFFSETS                          # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "highs_degeneracy_census", REPO / "tools" / "lp_bench" / "highs_degeneracy_census.py")
cen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cen)

CAP, PH = 0.6, 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixture", required=True,
                    help="case29gb T=24 relaxed commitment fixture (cap 0.6, ramp 1.0)")
    ap.add_argument("--out", required=True, help="where the per-instance rows go (.npy)")
    ap.add_argument("days", type=int, nargs="*",
                    help="evaluation-day offsets; default all of YEAR_EVAL_OFFSETS")
    args = ap.parse_args()

    days = args.days or list(YEAR_EVAL_OFFSETS)
    fx = load_commitment(path=Path(args.fixture), n_periods=24)
    meta = fx['meta']
    case = load_case(meta['case'])
    width, cost = segment_costs(case, 1)
    _, actual, _ = demand_from_meta(meta)
    di = np.asarray(fx['day_index'])
    nu = int(case.n_units)
    ub = np.asarray(case.unit_node_idx)
    clear, cspec = make_rt_clearing(case, n_segments=1, cap_scale=CAP, ramp_scale=1.0,
                                    period_hours=PH)
    clear = jax.jit(clear)
    print('rt spec', {k: cspec.get(k) for k in ('cap_scale', 'ramp_scale', 'max_iter',
                                                'period_hours')}, 'MAX_ITER', MAX_ITER)
    off24 = np.broadcast_to(cost[:, :, None], (nu, 1, 24)).copy()
    off1 = off24[:, :, :1].copy()
    rows = []
    for d in days:
        u24 = np.asarray(fx['commitment'][d], float)
        dem24 = np.asarray(actual[di[d]], float)[:24]
        p0 = np.asarray(fx['p_init'][d], float)
        da = cen.highs(reference.build_lp(case, off24, u24, dem24, p0, CAP, 1.0))
        for h in range(24):
            pin = p0 if h == 0 else da['award'][:, h - 1]
            u = u24[:, h:h + 1]
            dem = dem24[h:h + 1]
            lp = reference.build_lp(case, off1, u, dem, pin, CAP, 1.0, period_hours=PH)
            ref = cen.highs(lp)
            if ref is None or ref['stationarity_rel'] > cen.DUAL_RESIDUAL_TOL:
                rows.append((d, h, np.nan, np.nan, np.nan, np.nan, 0))
                continue
            out = {k: np.asarray(v) for k, v in clear(jnp.asarray(off1), jnp.asarray(u),
                                                      jnp.asarray(dem), jnp.asarray(pin)).items()}
            dl = float(np.abs(out['lmp'][0] - ref['lmp'][0]).max())
            dq = np.abs(out['award'][:, 0] - ref['award'][:, 0])
            Zi = float((off1[:, 0, 0] * out['award'][:, 0]).sum() + cen.VOLL * out['shed'].sum())
            Zh = float((off1[:, 0, 0] * ref['award'][:, 0]).sum() + cen.VOLL * ref['shed'].sum())
            tie = np.abs(off1[:, 0, 0] - ref['lmp'][0, ub]) <= 1e-6 * np.maximum(
                np.abs(ref['lmp'][0, ub]), 1)
            untied = int(((dq > 1e-3) & ~tie).sum())
            rows.append((d, h, dl, float(dq.max()), abs(Zi - Zh) / abs(Zh),
                         float(out['mu']), untied))
        print(f'day {d} done', flush=True)
    R = np.array(rows, float)
    print(f'instances {len(R)}  failed {int(np.isnan(R[:, 2]).sum())}')
    print(f'max |dLMP| {np.nanmax(R[:, 2]):.3e} $/MWh  max |d_award| {np.nanmax(R[:, 3]):.3e} MW  '
          f'max relZ {np.nanmax(R[:, 4]):.3e}  max mu {np.nanmax(R[:, 5]):.3e}  '
          f'untied award diffs {int(np.nansum(R[:, 6]))}')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, R)


if __name__ == "__main__":
    main()
