"""Ancillary services market: energy and operating reserve cleared together.

    from powermarketjax.envs.ancillary import make_clearing, make_requirement

    clear, spec = make_clearing(case, theta=(1/6, 0.5), volr=..., cap_scale=0.60,
                                ramp_scale=1.00, period_hours=0.5)

`theta` and `volr` have no defaults, and neither do `cap_scale` and
`ramp_scale` in practice: every scenario parameter of this market is declared
with its result.
"""
from .clearing import MAX_ITER, OFF_EPS, VOLL, make_clearing
from .requirement import make_requirement
from .settlement import make_settlement
from .env import make_ancillary_env
__all__ = ["MAX_ITER", "OFF_EPS", "VOLL", "make_clearing",
           "make_requirement", "make_settlement"]
