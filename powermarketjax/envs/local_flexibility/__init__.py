"""Local flexibility market, the fifth of the five markets.

The whole market is here: the network constants and the published requirement,
the clearing linear program, the verification against the nonlinear power flow,
the pay-as-bid settlement, the action map and the environment layer.

    sens = build_voltage_sensitivity(case)          # setup, numpy, float64
    agent_bus = draw_agent_buses(case, sens, 40, seed=0)
    reset, step, step_auto_reset, spec = make_local_flex_env(case, sens, agent_bus)
    params = make_local_flex_params(load_mw, pv, price, battery, cycle_cost,
                                    load_scale, learner_mask, episode_len)

Everything downstream of `build_voltage_sensitivity` is per unit, the line
ratings included: `line_cap` is in MVA and is converted once, inside
`sensitivity`.  The environment layer also speaks megawatts, since the battery
ratings and the demand series are registered in MW.

`load_scale` (written $\\kappa$), the placement seed and the substation the
demand series comes from have no defaults.  `make_local_flex_env` rejects a
case whose registered load sums to a negative number, since scaling such a case
raises voltage magnitudes instead of depressing them.

Two of the three exogenous series are absent from this repository: `data.py`
resolves each through the manifest registry and raises `MissingSeries` for
those two.
"""
from .action import make_action_map
from .data import (AEDT_WINDOW, AEST_WINDOW, LocalFlexSeries, MissingSeries,
                   load_energy_price, load_local_flex_series, load_pv_series,
                   load_substation_demand)
from .env import (LocalFlexParams, LocalFlexState, agent_bus_candidates,
                  baseline_action, draw_agent_buses, make_local_flex_env,
                  make_local_flex_params)
from .requirement import make_requirement
from .settlement import make_settlement
from .verification import cleared_injection, make_verification
from .sensitivity import VoltageSensitivity, build_voltage_sensitivity

__all__ = ["VoltageSensitivity", "build_voltage_sensitivity", "make_requirement",
           "make_verification", "cleared_injection", "make_settlement",
           "make_action_map", "LocalFlexState", "LocalFlexParams",
           "make_local_flex_env", "make_local_flex_params", "baseline_action",
           "agent_bus_candidates", "draw_agent_buses",
           "LocalFlexSeries", "MissingSeries", "load_substation_demand",
           "AEDT_WINDOW", "AEST_WINDOW",
           "load_pv_series", "load_energy_price", "load_local_flex_series"]
