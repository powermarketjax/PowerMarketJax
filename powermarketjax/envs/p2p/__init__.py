"""Peer-to-peer local energy market: households trade among themselves.

Every household meters an injection and an offtake, may hold a battery, and in
each period is either in surplus or in deficit.  Surpluses are offered and
deficits are bid for in a double auction that clears at one price; whatever the
auction does not award is traded with the grid at the outside option.

    series = load_fluvius_households(n_households=n_agents)
    reset, step, step_auto_reset, spec = make_p2p_env(n_agents, PI_EXP, PI_RET,
                                                      period_hours=0.25)
    params = make_p2p_params(series.injection, series.offtake, battery, kappa,
                             learner_mask, H)
    obs, state = reset(key, params)
    obs, state, reward, costs, done, info = step(key, state, action, params)

The four operators ``step`` routes are importable on their own and chain:

    act, aspec = make_action_map(n_agents, PI_EXP, PI_RET, period_hours=0.25)
    sub = act(action, soc, p_pv, load, battery)     # pure, jittable, vmappable

    clear, cspec = make_clearing(n_agents, PI_EXP, PI_RET)
    out = clear(sub["price"], sub["q_sell"], sub["q_buy"])

    settle = make_settlement(PI_EXP, PI_RET)
    money = settle(sub["q_sell"], sub["q_buy"], out["award_sell"],
                   out["award_buy"], out["clearing_price"], kappa,
                   sub["throughput"])

Clearing is two sorts, two cumulative sums, a comparison reduction and a
gather, all of fixed shape in ``n_agents``: there is no solver, no callback and
no data-dependent shape anywhere in this market.  Every array is float32,
written on each constructor rather than obtained by disabling
``jax_enable_x64``, which stays on process-wide for the markets that solve LPs.

``pi_exp`` and ``pi_ret`` are closed over at construction, which is where
``0 <= pi_exp < pi_ret`` is checked.  The degradation price and the battery
sizing are runtime arguments, so a scenario sweep is a call rather than a
rebuild.

Modules:

* ``action``           -- action in, submission out
* ``clearing``         -- double auction: submissions in, awards and price out
* ``settlement``       -- awards and price in, profit per agent out
* ``baseline_pricing`` -- two reference pricing rules, **not** the mechanism
* ``env``              -- the operators behind the 6-tuple interface
* ``households``       -- the exogenous metered series, ``(n_periods, N)``
"""
from .action import make_action_map, make_soc_advance
from .baseline_pricing import make_baseline_pricing
from .clearing import make_clearing
from .env import (P2PParams, P2PState, baseline_action, make_p2p_env,
                  make_p2p_params)
from .households import load_fluvius_households, single_offset_windows
from .settlement import make_settlement, make_terminal_settlement

__all__ = ["make_action_map", "make_soc_advance", "make_clearing",
           "make_settlement", "make_terminal_settlement",
           "make_baseline_pricing",
           "make_p2p_env", "make_p2p_params", "baseline_action",
           "P2PState", "P2PParams",
           "load_fluvius_households", "single_offset_windows"]
