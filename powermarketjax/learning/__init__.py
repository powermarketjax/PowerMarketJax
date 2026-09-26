"""The learning side of the benchmark.

Nothing under `powermarketjax/envs/` imports this package: the markets do not
know that a learner exists, which is what keeps `make_*_env` usable on its own.
The dependency runs the other way, and only through the `(reset, step,
step_auto_reset, spec)` tuple every market's `make_*_env` returns.

`optax` is listed in the `rl` extra, but it also arrives with `flax`, a core
dependency, so this package imports on a `dev` install as well.  What the extra
adds on top is `rlax` and `distrax`, which only `tools/p2p_experiment/` uses.
"""
from .adapters import unpack_env
from .ippo import IPPOConfig, make_ippo, observation_statistics
from .policy import SharedActorCritic, bounds_for
from .sac import SACConfig, make_sac, make_sac_greedy_action, reward_statistics

__all__ = ["IPPOConfig", "SACConfig", "SharedActorCritic", "bounds_for",
           "make_ippo", "make_sac", "make_sac_greedy_action",
           "observation_statistics", "reward_statistics", "unpack_env"]
