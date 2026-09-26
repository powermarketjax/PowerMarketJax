# Vendored from PowerZooJax. Do not edit to track upstream;
# this file is now this repository's code.
# Source        : powerzoojax/utils/typing.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# NOT verbatim. Beyond the package rename (powerzoojax -> powermarketjax, in
# both import statements and path strings): docstrings and comments were
# added on 2026-08-20 for API documentation. No executable line was changed.
# Upstream licence: MIT, Copyright (c) 2026 PowerZooJax Contributors.
"""
Type Definitions and Hints

Provides consistent type annotations across PowerMarketJax.

Uses:
- chex.Array for JAX arrays
- TypeVar for generic typing
- Protocol for structural typing

This improves code readability and enables better IDE support.
"""

from typing import TypeVar, Union, Tuple, Dict, Any, Callable, Protocol
import chex
import jax.numpy as jnp

# Basic JAX types
PRNGKey = chex.PRNGKey
Array = chex.Array
Scalar = Union[float, chex.Array]

# Shape types
Shape = Tuple[int, ...]
DType = jnp.dtype

# Generic state/params types
StateT = TypeVar('StateT')
ParamsT = TypeVar('ParamsT')
ObsT = TypeVar('ObsT')
ActionT = TypeVar('ActionT')

# Environment return types
StepReturn = Tuple[Array, StateT, Scalar, bool, Dict[str, Any]]
ResetReturn = Tuple[Array, StateT]


class EnvironmentProtocol(Protocol[StateT, ParamsT]):
    """Protocol defining the environment interface.
    
    Environments implementing this protocol are guaranteed to have
    the standard reset/step methods with correct signatures.
    """
    
    def reset(self, key: PRNGKey, params: ParamsT) -> Tuple[Array, StateT]:
        """Start a new episode, returning the first observation and state."""
        ...
    
    def step(
        self, 
        key: PRNGKey, 
        state: StateT, 
        action: Array, 
        params: ParamsT
    ) -> StepReturn:
        """Advance one step, returning the `StepReturn` tuple."""
        ...


class PolicyProtocol(Protocol):
    """Protocol for policy functions.
    
    Policies take observations and return actions.
    """
    
    def __call__(self, obs: Array, key: PRNGKey) -> Array:
        """Map an observation to an action, drawing any randomness from `key`."""
        ...


# Reward function type
RewardFn = Callable[[StateT, Array, ParamsT], Scalar]

# Observation function type  
ObsFn = Callable[[StateT, ParamsT], Array]
