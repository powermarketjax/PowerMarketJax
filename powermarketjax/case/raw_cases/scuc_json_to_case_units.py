# Vendored from PowerZooJax. Do not edit to track upstream;
# this file is now this repository's code.
# Source        : powerzoojax/case/raw_cases/scuc_json_to_case_units.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
# Upstream licence: MIT, Copyright (c) 2026 PowerZooJax Contributors.
"""Extract SCUC-style generator JSON into ``case118``-style unit table rows.

Keep this file aligned with ``PowerZoo/powerzoo/case/raw_cases/scuc_json_to_case_units.py``.

Input JSON shape matches ``raw_cases/case14.json`` (``Generators`` dict with
piecewise MW/USD curves, MW/h ramps, hours min up/down, multi-tier startup $).

Conversions (PowerZooJax / ``make_uc_params`` conventions):

- **mc_a, mc_b, mc_c**: marginal cost ``MC(p)=mc_a·p²+mc_b·p+mc_c``; fit the integrated
  curve ``TC=(mc_a/3)p³+(mc_b/2)p²+mc_c·p`` to JSON breakpoints plus ``(0,0)``.
  If ``mc_a<0``, refit with ``mc_a=0`` (no hot/cold / tiered startup in cost curve).
- **ramp_up / ramp_down**: JSON ``Ramp * limit (MW)`` is **MW/h** → store
  ``ramp_frac = ramp_mw_per_h / p_max`` (fraction of ``p_max`` per hour), same as case118.
- **min_up_time / min_down_time**: JSON hours → **steps**:
  ``round(hours / delta_t_hours)`` (default 0.5 h step).
- **init_start_up_cost**: single scalar = **max** of JSON startup tiers (use ``--startup-agg``).
- **keep_time**: ``abs(Initial status (h)) / delta_t_hours``.
- **init_state**: ``1`` if initial hours ``> 0`` else ``0``.
- **init_power**: JSON MW clipped to ``[p_min, p_max]``.
- **init_no_load_cost**: ``0`` (omitted in JSON).

Example (IEEE-14 + bundled JSON)::

    python -m powermarketjax.case.raw_cases.scuc_json_to_case_units \\
        powermarketjax/case/raw_cases/case14.json \\
        --use-default-case14-prefix \\
        --p-max 332.4 140 100 100 100

Paste printed lines into ``case14.py`` ``_units`` data (after checking bus order matches
JSON ``Bus`` → ``b1``..``bN``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def fit_mc_jax_nonneg(
    mw_pts: Sequence[float],
    cost_pts: Sequence[float],
) -> Tuple[float, float, float]:
    """Fit (mc_a, mc_b, mc_c) for TC = (a/3)p³+(b/2)p²+c·p with nonnegative a."""
    p = np.asarray([0.0] + list(mw_pts), dtype=np.float64)
    c = np.asarray([0.0] + list(cost_pts), dtype=np.float64)
    a_mat = np.column_stack([p ** 3 / 3.0, p ** 2 / 2.0, p])
    coef, *_ = np.linalg.lstsq(a_mat, c, rcond=None)
    a, b, cc = float(coef[0]), float(coef[1]), float(coef[2])
    if a < 0.0:
        a2 = np.column_stack([p ** 2 / 2.0, p])
        bc, *_ = np.linalg.lstsq(a2, c, rcond=None)
        b, cc = float(bc[0]), float(bc[1])
        a = 0.0
    return a, b, cc


def extract_row(
    gen: Dict[str, Any],
    *,
    p_max: float,
    p_min: float,
    delta_t_hours: float,
    startup_agg: str,
    fuel: str,
) -> Tuple[float, float, float, float, float, float, float, float, float, float, float, str]:
    """Returns mc_a..c, ramp_up, ramp_down, su, keep, ip, st, mu, md, fuel."""
    mw = gen["Production cost curve (MW)"]
    cc = gen["Production cost curve ($)"]
    a, b, c = fit_mc_jax_nonneg(mw, cc)

    ramp_h = float(gen["Ramp up limit (MW)"])
    rfrac = ramp_h / max(float(p_max), 1e-6)

    su_list = [float(x) for x in gen["Startup costs ($)"]]
    if startup_agg == "max":
        su = max(su_list)
    elif startup_agg == "min":
        su = min(su_list)
    else:
        su = float(np.mean(su_list))

    muh = float(gen["Minimum uptime (h)"])
    mdh = float(gen["Minimum downtime (h)"])
    mu = float(round(muh / delta_t_hours))
    md = float(round(mdh / delta_t_hours))

    init_h = float(gen["Initial status (h)"])
    st = 1.0 if init_h > 0 else 0.0
    keep = abs(init_h) / delta_t_hours

    ip = float(np.clip(float(gen["Initial power (MW)"]), p_min, p_max))

    return a, b, c, rfrac, rfrac, su, keep, ip, st, mu, md, fuel


def _case14_matpower_prefixes() -> List[List[Any]]:
    return [
        [1, 1, 232.4, -16.9, 10, 0, 1.06, 100, 1, 332.4, 0],
        [2, 2, 40, 42.4, 50, -40, 1.045, 100, 1, 140, 0],
        [3, 3, 0, 23.4, 40, 0, 1.01, 100, 1, 100, 0],
        [4, 6, 0, 12.2, 24, -6, 1.07, 100, 1, 100, 0],
        [5, 8, 0, 17.4, 24, -6, 1.09, 100, 1, 100, 0],
    ]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Print Python _units data rows from SCUC JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("json_path", type=Path)
    ap.add_argument(
        "--generator-order",
        nargs="+",
        default=["g1", "g2", "g3", "g4", "g5"],
    )
    ap.add_argument("--p-max", type=float, nargs="+", required=True)
    ap.add_argument("--p-min", type=float, nargs="+", default=None)
    ap.add_argument("--delta-t-hours", type=float, default=0.5)
    ap.add_argument("--fuel", default="gas")
    ap.add_argument("--startup-agg", choices=("max", "min", "mean"), default="max")
    ap.add_argument(
        "--use-default-case14-prefix",
        action="store_true",
        help="Prefix rows with fixed MATPOWER case14 Pg/Qg/... columns",
    )
    args = ap.parse_args()

    p_min = args.p_min if args.p_min is not None else [0.0] * len(args.p_max)
    if len(p_min) != len(args.p_max):
        raise SystemExit("--p-min length must match --p-max")
    if len(args.generator_order) != len(args.p_max):
        raise SystemExit("--generator-order length must match --p-max")

    data = json.loads(args.json_path.read_text(encoding="utf-8"))
    gens: Dict[str, Any] = data["Generators"]

    prefixes = _case14_matpower_prefixes() if args.use_default_case14_prefix else None
    if prefixes is not None and len(prefixes) != len(args.p_max):
        raise SystemExit("default case14 prefix has 5 rows; match --p-max count")

    print("        # unit rows: same header as case118 (see case14.py).")
    for i, gkey in enumerate(args.generator_order):
        g = gens[gkey]
        pmx = float(args.p_max[i])
        pmi = float(p_min[i])
        a, b, c, ru, rd, su, keep, ip, st, mu, md, fuel = extract_row(
            g,
            p_max=pmx,
            p_min=pmi,
            delta_t_hours=args.delta_t_hours,
            startup_agg=args.startup_agg,
            fuel=args.fuel,
        )
        if prefixes is not None:
            pre = prefixes[i]
            row = (
                f"        [{pre[0]}, {pre[1]}, {pre[2]}, {pre[3]}, {pre[4]}, {pre[5]}, "
                f"{pre[6]}, {pre[7]}, {pre[8]}, {pre[9]}, {pre[10]}, "
                f"{a:.8g}, {b:.8g}, {c:.8g}, {pmx}, {pmi}, "
                f"\"{fuel}\", {ru:.10g}, {rd:.10g}, {su}, {keep}, {ip}, {st}, {mu}, {md}, 0.0],"
            )
        else:
            row = (
                f"        # {gkey}: mc=({a:.8g},{b:.8g},{c:.8g}) "
                f"ramp={ru:.8g} su={su} keep={keep} ip={ip} st={st} mu={mu} md={md}"
            )
        print(row)


if __name__ == "__main__":
    main()
