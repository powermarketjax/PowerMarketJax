"""Real-time balancing market: one step is one half-hour period.

Settlement has two parts.  The day-ahead schedule is a financially binding
contract paid at the day-ahead price, and only the deviation from it is settled
at the real-time price.

The clearing operator is **not** here.  It is the day-ahead operator at a single
period -- `envs.day_ahead.make_clearing(case, n_periods=1)` -- so this market
adds no clearing code.  Two consequences of that reuse are load bearing:

* `envs.day_ahead.clearing`'s objective carries **no** factor of `period_hours`,
  so its duals are already \\$/MWh and `settlement.py` applies `Delta` exactly
  once.  Dividing by `Delta` again would halve every price at this market's
  `Delta = 0.5 h`.  `envs.day_ahead.relax` uses the opposite convention, so name
  the operator: this market consumes `clearing.py`'s duals and only those;
* `period_hours` does enter the ramp limits, so the per-period ramp allowance
  here is half the day-ahead market's against a demand step that is not halved.

Layout:

* ``position``  -- the frozen day-ahead position and the hour-to-period map
* ``clearing``  -- the day-ahead operator at T=1, plus this market's Newton budget
* ``demand``    -- the realised series at 48 periods per day, one loader per case
* ``boundary``  -- the episode's opening carry `p_prev`
* ``settlement``-- the two-settlement rule and its money-balance identity
* ``env``       -- `RealTimeState`, `step`, auto-reset, observation, `costs`
"""
from .boundary import make_boundary
from .clearing import MAX_ITER, make_rt_clearing
from .demand import (half_hourly_from_meta, load_gb_demand_half_hourly,
                     load_nem_demand_half_hourly, load_rts_demand_half_hourly)
from .position import (PERIODS_PER_HOUR, T_RT, hour_of_period,
                       load_da_position, to_real_time)
from .env import COST_NAMES, RealTimeState, make_env
from .settlement import make_settlement

__all__ = ["load_da_position", "hour_of_period", "to_real_time",
           "load_gb_demand_half_hourly", "load_rts_demand_half_hourly",
           "load_nem_demand_half_hourly", "half_hourly_from_meta",
           "make_boundary", "make_rt_clearing",
           "make_settlement", "make_env", "RealTimeState", "COST_NAMES",
           "MAX_ITER", "PERIODS_PER_HOUR", "T_RT"]
