"""Day-ahead settlement (market 01) against a NumPy loop reference.

The reference loops over units and periods and rewrites the formulas stated in
`powermarketjax/envs/day_ahead/settlement.py`.  Over the evaluation days the
commitment comes from the fixture, awards and LMPs come from the JAX clearing
under truthful offers, and both `make_settlement` (JAX) and the loop below settle
them; revenue, cost, profit and the three cost components are compared.  A
second pass assigns the 66 units to 7 agents at random, to check the
`unit_to_agent` sums.  The run ends with an injection (no-load cost charged as
if every unit were always on), which the comparison has to catch.

The fixture is built by `tools/commitment/precommit.py`; it must be the
`case29gb` T=24 relaxed commitment at cap_scale 0.6 and ramp_scale 1.0.

    JAX_PLATFORMS=cpu python tools/lp_bench/da_settlement_reference.py --fixture <commitment.npz>
"""
import argparse
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
from powermarketjax.envs.day_ahead import make_clearing, segment_costs  # noqa: E402
from powermarketjax.envs.day_ahead.settlement import make_settlement  # noqa: E402
from powermarketjax.envs.day_ahead.commitment import load_commitment  # noqa: E402
from powermarketjax.envs.day_ahead.demand import demand_from_meta  # noqa: E402
from evaluation import YEAR_EVAL_OFFSETS                          # noqa: E402


def settle_ref(case, award, lmp, u, u_prev, agent_of, n_agents, dt=1.0):
    bus = np.asarray(case.unit_node_idx)
    a = np.asarray(case.unit_cost_a, np.float64)
    b = np.asarray(case.unit_cost_b, np.float64)
    c = np.asarray(case.unit_cost_c, np.float64)
    nl = np.asarray(case.unit_no_load_cost, np.float64)
    su = np.asarray(case.unit_startup_cost, np.float64)
    n, T = award.shape
    out = {k: np.zeros(n_agents) for k in ('revenue', 'energy_cost', 'no_load_cost', 'startup_cost')}
    for i in range(n):
        g = agent_of[i]
        for t in range(T):
            p = award[i, t]
            out['revenue'][g] += dt * lmp[t, bus[i]] * p
            out['energy_cost'][g] += dt * (a[i] / 3 * p ** 3 + b[i] / 2 * p ** 2 + c[i] * p)
            out['no_load_cost'][g] += dt * nl[i] * u[i, t]
            before = u_prev[i] if t == 0 else u[i, t - 1]
            if u[i, t] > before:
                out['startup_cost'][g] += su[i] * (u[i, t] - before)
    out['cost'] = out['energy_cost'] + out['no_load_cost'] + out['startup_cost']
    out['profit'] = out['revenue'] - out['cost']
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixture", required=True,
                    help="case29gb T=24 relaxed commitment fixture (cap 0.6, ramp 1.0)")
    args = ap.parse_args()

    fx = load_commitment(path=Path(args.fixture), n_periods=24)
    meta = fx['meta']
    case = load_case(meta['case'])
    width, cost = segment_costs(case, 1)
    _, actual, _ = demand_from_meta(meta)
    di = np.asarray(fx['day_index'])
    nu = int(case.n_units)
    clear, _ = make_clearing(case, 24, n_segments=1, cap_scale=0.6, ramp_scale=1.0)
    clear = jax.jit(clear)
    offer = np.broadcast_to(cost[:, :, None], (nu, 1, 24)).copy()
    rng = np.random.default_rng(0)
    parts = {'one agent per unit': np.arange(nu), '7 agents': rng.integers(0, 7, nu)}
    parts['7 agents'][:7] = np.arange(7)
    worst = {}
    for name, part in parts.items():
        settle = jax.jit(make_settlement(case, unit_to_agent=part))
        na = int(part.max()) + 1
        for d in YEAR_EVAL_OFFSETS:
            u = np.asarray(fx['commitment'][d], float)
            p0 = np.asarray(fx['p_init'][d], float)
            uprev = np.asarray(fx['commitment_prev'][d], float)
            dem = np.asarray(actual[di[d]], float)[:24]
            out = clear(jnp.asarray(offer), jnp.asarray(u), jnp.asarray(dem), jnp.asarray(p0))
            aw, lmp = np.asarray(out['award']), np.asarray(out['lmp'])
            got = {k: np.asarray(v) for k, v in settle(jnp.asarray(aw), jnp.asarray(lmp),
                                                        jnp.asarray(u), jnp.asarray(uprev)).items()}
            ref = settle_ref(case, aw, lmp, u, uprev, part, na)
            for k in ref:
                ab = np.abs(got[k] - ref[k])
                rel = ab / np.maximum(np.abs(ref[k]), 1.0)
                w = worst.setdefault((name, k), [0.0, 0.0, 0.0])
                w[0] = max(w[0], ab.max())
                w[1] = max(w[1], rel.max())
                w[2] = max(w[2], np.abs(ref[k]).max())
        print(f'{name}: {len(YEAR_EVAL_OFFSETS)} days done', flush=True)
    for (name, k), (ab, rel, sc) in worst.items():
        print(f'{name:20s} {k:14s} max abs {ab:.3e} $   max rel {rel:.3e}   (scale {sc:.3e} $)')
    # Injection: the reference charges no-load as if every unit were always on;
    # the comparison has to catch it.
    u = np.asarray(fx['commitment'][YEAR_EVAL_OFFSETS[0]], float)
    u2 = np.ones_like(u)
    ref_bad = settle_ref(case, aw, lmp, u2, np.ones(nu), np.arange(nu), nu)
    good = settle_ref(case, aw, lmp, u, np.ones(nu), np.arange(nu), nu)
    print('injection (no-load charged as always-on) max abs diff vs the correct ref:',
          np.abs(ref_bad['cost'] - good['cost']).max())


if __name__ == "__main__":
    main()
