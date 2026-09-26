"""Numerical solvers shared by the market environments.

Model-agnostic by rule: nothing here may import from ``powermarketjax.envs`` or
know what an offer, a bus or a price is.  A market that needs structure exploited
supplies it through the pluggable interfaces (``ops``, ``kkt``) rather than by
specialising the algorithm here -- ``envs.day_ahead.kkt`` is the worked example.

**Calibration constants stay with the market, not here.**  ``max_iter`` has no
default for that reason: a shared default would make one market's calibration
look like a property of the algorithm, when different markets need different
trip counts.
"""
from . import ipm

__all__ = ["ipm"]
