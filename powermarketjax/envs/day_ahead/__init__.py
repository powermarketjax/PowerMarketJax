"""Day-ahead wholesale market, the first of the five markets.

    env, spec = make_env(case, load_commitment(), load_gb_demand(),
                         kind="markup", markup_max=2.0,
                         cap_scale=0.60, ramp_scale=1.00)
    params = env.make_params(episode_len=7)
    obs, state = env.reset(key, params)
    obs, state, reward, costs, done, info = env.step(key, state, action, params)

**The commitment is solved from the offers, not read from the fixture.**
``step`` runs all three clearing stages: the relaxed unit commitment, the
rounding to integers, and the fixed-commitment dispatch whose duals are the
prices.  So who runs is a function of what the agents bid.  The commitment
fixture supplies only the day window and the initial day boundary.

``cap_scale`` and ``ramp_scale`` have no default.  At the registered values
``case29gb`` neither congests nor binds on ramp, so an environment built without
them studies a network without locational prices and periods that do not couple.

Layout:

* ``clearing``   -- offers and a fixed commitment in, awards and LMPs out
* ``kkt``        -- block-tridiagonal Newton system and matrix-free G for it
* ``kkt_lowrank`` -- the same Newton systems, low-rank in a monitored line set
* ``relax``      -- the relaxed commitment LP and the rounding
* ``relax_kkt``  -- the same Newton system for the relaxed commitment
* ``action``     -- raw actions in, monotone offer curves out
* ``demand``     -- the GB forecast/realisation pair, hourly, per market day
* ``settlement`` -- awards and LMPs in, profit per agent out
* ``commitment`` -- the fixture: day window and initial boundary per market day
* ``env``        -- `DayAheadState`, `step`, auto-reset, observation, `costs`

The interior point method is shared with the other four markets and lives in
``powermarketjax.solvers.ipm``; its Newton-step count is a per-market
calibration and stays here, as ``clearing.MAX_ITER``.
"""
from .action import make_offer_map, truthful_action
from .clearing import OFF_EPS, VOLL, make_clearing, rated_lines, segment_costs
from .commitment import load_commitment
from .demand import (demand_for_case, demand_from_meta, demand_meta,
                     demand_pairing, load_gb_demand)
from .env import COST_NAMES, MU_TOL, DayAheadState, EnvParams, make_env
from .settlement import make_settlement

__all__ = ["make_clearing", "make_offer_map", "make_settlement", "load_gb_demand",
           "demand_from_meta", "demand_for_case", "demand_meta", "demand_pairing",
           "load_commitment", "make_env", "truthful_action", "DayAheadState",
           "EnvParams", "segment_costs", "rated_lines", "VOLL", "OFF_EPS", "MU_TOL",
           "COST_NAMES"]
