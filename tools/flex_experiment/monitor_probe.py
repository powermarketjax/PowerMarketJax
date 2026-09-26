"""Measure what the positioning exposure of §8 is worth, by closing it.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:tools/flex_experiment python -m monitor_probe

§8 of the specification measures the baseline physically, which removes
misreporting and leaves positioning: a participant may plan a charge, deepen
the constraint that charge creates, and be paid to forgo it.  §8 names that as
the distribution-level increase-decrease game, states that it is left in place
deliberately, and names a market monitor as what would address it.

`monitor_baseline` is that monitor in its strongest form: the operator solves
§6 against a baseline with the planned charging removed, so congestion a
participant created earns nothing.

**It is not one change, and the first version of this probe said it was.**
`env.py` obtains the award and the directed curtailment from a single call to
`clear`, so passing the counterfactual baseline also curtails against it, while
`cleared_injection` and the sweep of §7 keep the real injection.  A monitored
run therefore leaves real overloads unrelieved, and the third channel of §16 is
the only quantity that shows it.  Both sides are reported here for that reason;
a return comparison alone credits the monitor with savings it did not make.

The arms are constant.  The point is not what a learner does but what the
mechanism offers, and a constant arm makes the comparison a property of the
mechanism.  `LOG_STD` is -12 rather than `LOG_STD_FINAL`: at -3 the sampled
action carries a standard deviation of 0.199 in the units the environment sees,
which is not a constant arm even though the pairing survives it.

`incdec` is `INCDEC_ACTION`, the same array §5.4 of the report scores, so the
monitored comparison and the arm table refer to one strategy.  The saturating
arms at plus or minus five are kept beside it because they are what the
calibration scan of `kappa_scan.py` uses, and at `alpha_price = -5` the offer is
150.89 rather than the floor of 149.88.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from concentration_baseline import (ACTION_GAIN, CELLS, EPISODE_LEN,
                                    INCDEC_ACTION, day_split, evaluate,
                                    init_policy, obs_scale, scenario,
                                    series_day_of_month)

LOG_STD = -12.0

ARMS = (("incdec", list(INCDEC_ACTION)),
        ("cost_maxcharge", [-5.0, 5.0, 5.0]),
        ("cost_halfcharge", [-5.0, 5.0, 0.0]),
        ("cost_nocharge", [-5.0, 5.0, -5.0]),
        ("passive", [-5.0, 5.0, -128.0]))

KEYS = ("ret", "volume_given_cleared", "clearing_fraction", "shed_mwh",
        "price_given_cleared", "overload_periods", "overload_max",
        "overload_mean", "volume_all")


def constant_policy(obs_dim, action):
    p = init_policy(jax.random.PRNGKey(0), obs_dim)
    z = jax.tree.map(jnp.zeros_like, p)
    return {**z, "mean_b": jnp.asarray(action, jnp.float32) / ACTION_GAIN}


def main() -> int:
    #: No options: the probe is fixed by the module constants.  The parser
    #: exists so that `--help` prints this and exits instead of running the
    #: whole probe and overwriting the JSON below; run without arguments it
    #: does exactly what it did before.
    argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Takes no options.  Writes monitor_probe.json into the "
               "flex-concentration figures directory, relative to the "
               "working directory.").parse_args()
    _, dom = series_day_of_month()
    _, _, test = day_split(dom, EPISODE_LEN)
    rows = []
    print("%-14s %-8s %-16s %8s %8s %7s %8s %9s %9s"
          % ("cell", "monitor", "arm", "ret", "vol|clr", "clear", "shed/ep",
             "ovl_frac", "ovl_max"))
    for place, cap in CELLS:
        name = f"{place}p_{cap}c"
        for mon in (False, True):
            env, params, n = scenario(place, cap, split="train", monitor=mon)
            scale = obs_scale(env, params, n)
            obs_dim = env[3]["obs_dim"]
            for label, act in ARMS:
                r = evaluate(env, params, scale, n,
                             constant_policy(obs_dim, act),
                             jax.random.PRNGKey(999), 0, LOG_STD,
                             eval_starts=test)
                rows.append(dict(cell=name, monitor=bool(mon), arm=label,
                                 n_agent=n, **r))
                print("%-14s %-8s %-16s %8.3f %8.4f %7.4f %8.4f %9.4f %9.2e"
                      % (name, mon, label, r["ret"],
                         r["volume_given_cleared"], r["clearing_fraction"],
                         r["shed_mwh"], r["overload_periods"],
                         r["overload_max"]), flush=True)
    out = Path("docs/figures/flex-concentration/monitor_probe.json")
    out.write_text(json.dumps(rows, indent=1, default=float))
    print(f"wrote {out}")

    print("\nwhat closing the exposure changes, per cell and arm:")
    print("%-14s %-16s %9s %9s %11s %11s"
          % ("cell", "arm", "return", "volume", "ovl as spec", "ovl monitored"))
    by = {(r["cell"], r["arm"]): r for r in rows if not r["monitor"]}
    for r in rows:
        if not r["monitor"]:
            continue
        o = by[(r["cell"], r["arm"])]
        def pct(a, b):
            return 100.0 * (a / b - 1.0) if abs(b) > 1e-9 else float("nan")
        print("%-14s %-16s %+8.1f%% %+8.1f%% %11.4f %11.4f"
              % (r["cell"], r["arm"], pct(r["ret"], o["ret"]),
                 pct(r["volume_given_cleared"], o["volume_given_cleared"]),
                 o["overload_periods"], r["overload_periods"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
