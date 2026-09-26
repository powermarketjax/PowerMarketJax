# Vendored from PowerZooJax. Do not edit to track upstream;
# this file is now this repository's code.
# Source        : powerzoojax/utils/__init__.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# NOT verbatim. Beyond the package rename (powerzoojax -> powermarketjax, in
# both import statements and path strings): the module docstring's one
# remaining mention of the upstream project name was changed to PowerMarketJax
# on 2026-08-21. The clause above covers imports and path strings, not prose,
# which is why this one line makes the file non-verbatim. No executable line
# was changed.
# Upstream licence: MIT, Copyright (c) 2026 PowerZooJax Contributors.
"""Shared lightweight utilities for PowerMarketJax.

The public surface here is intentionally small:
- PRNG splitting for batched env execution
- `batch_reset` / `batch_step` helpers built on `jax.vmap`
- `scan_rollout` for fixed-length trajectory collection without Python loops
- common typing aliases used across env, task, and training code
"""

from powermarketjax.utils.jax_utils import (
    split_key_for_envs,
    batch_reset,
    batch_step,
    scan_rollout,
)
from powermarketjax.utils.typing import (
    PRNGKey,
    Array,
    Scalar,
)

__all__ = [
    "split_key_for_envs",
    "batch_reset",
    "batch_step",
    "scan_rollout",
    "PRNGKey",
    "Array",
    "Scalar",
]
