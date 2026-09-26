# Vendored from PowerZooJax. Do not edit to track upstream;
# this file is now this repository's code.
# Source        : powerzoojax/data/nonstationary.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# NOT verbatim. Beyond the package name change (powerzoojax -> powermarketjax,
# in both import statements and path strings), on 2026-08-24 the concrete
# ``episode_start`` in ``NonstationarySampler``'s usage example was replaced by
# its bound. The example claimed 2340; the code it shows returns 3174 on
# jax 0.10.2 / CPU. The draw moves with the JAX PRNG, so no literal is correct
# for long and none is given. Upstream still reads 2340 at a7641de.
# Upstream licence: MIT, Copyright (c) 2026 PowerZooJax Contributors.
"""Non-stationary episode sampler for DSO and similar tasks.

Episode-start rolling, demand drift, and zone-holdout OOD for non-stationary training:

1. **Month rolling** — episode start drifts through the training window:
   ``phase_k = episode_idx / total_episodes``
   ``base_step = phase_k * total_steps``
   ``start ~ Normal(base_step, sigma_steps)``

2. **Linear demand drift** — load multiplied by
   ``drift_factor = 1.0 + drift_rate * episode_idx``

3. **Zone holdout** — swap feeder substations to unseen zones (OOD).

All sampling is JAX-PRNG-based (reproducible, deterministic given key).
The sampler itself runs at setup / evaluation time (CPU), producing
``(episode_start, drift_factor)`` pairs that feed into
``make_dso_load_profiles`` or ``make_dso_params``.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


class EpisodeConfig(NamedTuple):
    """Episode-level non-stationarity parameters."""
    episode_start: int    # offset into the full load profile (in steps)
    drift_factor: float   # multiplicative load scaling (>= 1.0 during training)


class NonstationarySampler:
    """Generates non-stationary episode parameters for DSO task.

    Usage::

        sampler = NonstationarySampler(
            total_steps=10080,     # 7 months × 48 steps/day
            total_episodes=500,
        )
        cfg = sampler.sample(episode_idx=100, key=jax.random.PRNGKey(0))
        # → EpisodeConfig(episode_start=<draw in [0, max_start]>,
        #                 drift_factor=1.03)

    ``drift_factor`` is exact: ``1.0 + drift_rate * episode_idx``.
    ``episode_start`` is a PRNG draw and moves with the JAX version, so no
    literal is given here; the call above returned 3174 on jax 0.10.2, CPU,
    measured 2026-08-24.

    The sampler supports four modes:

    - ``"train"`` — rolling center + drift (default)
    - ``"iid"`` — uniform random start, no drift (same window)
    - ``"drift_shock"`` — fixed high drift factor (OOD)
    - ``"fixed"`` — fixed start, no drift (for reproducible eval)

    Args:
        total_steps: Length of the full feeder shape array (in time steps).
            For Ausgrid training window (7 months @ 48 steps/day) ≈ 10,080.
        total_episodes: Expected total training episodes (for phase_k calc).
        max_steps: Episode length (default 48 = 24h @ 30min).
        drift_rate: Per-episode drift rate (default 0.0003 = +0.03%/episode).
        rolling_sigma_steps: Gaussian σ for month-rolling start sampling,
            in time steps (default 720 = 15 days × 48 steps/day).
    """

    def __init__(
        self,
        total_steps: int,
        total_episodes: int = 500,
        *,
        max_steps: int = 48,
        drift_rate: float = 0.0003,
        rolling_sigma_steps: float = 720.0,
    ):
        self.total_steps = total_steps
        self.total_episodes = max(total_episodes, 1)
        self.max_steps = max_steps
        self.drift_rate = drift_rate
        self.rolling_sigma_steps = rolling_sigma_steps
        # Maximum valid start so that a full episode fits
        self.max_start = max(total_steps - max_steps, 0)

    def sample(
        self,
        episode_idx: int,
        key: jax.Array,
        *,
        mode: str = "train",
    ) -> EpisodeConfig:
        """Sample episode parameters.

        Args:
            episode_idx: Current episode index (0-based).
            key: JAX PRNG key.
            mode: Sampling mode — ``"train"``, ``"iid"``,
                ``"drift_shock"``, or ``"fixed"``.

        Returns:
            ``EpisodeConfig(episode_start, drift_factor)``.
        """
        if mode == "train":
            return self._sample_train(episode_idx, key)
        elif mode == "iid":
            return self._sample_iid(key)
        elif mode == "drift_shock":
            return self._sample_drift_shock(key)
        elif mode == "fixed":
            return EpisodeConfig(episode_start=0, drift_factor=1.0)
        else:
            raise ValueError(f"Unknown mode '{mode}'. "
                             f"Available: train, iid, drift_shock, fixed")

    def _sample_train(self, episode_idx: int, key: jax.Array) -> EpisodeConfig:
        """Train mode: rolling center + linear drift."""
        # Phase: linearly sweep through training window
        phase_k = episode_idx / self.total_episodes  # ∈ [0, 1]
        base_step = phase_k * self.max_start

        # Gaussian jitter around the rolling center
        noise = float(jax.random.normal(key)) * self.rolling_sigma_steps
        start = int(np.clip(base_step + noise, 0, self.max_start))

        # Linear demand drift
        drift = 1.0 + self.drift_rate * episode_idx

        return EpisodeConfig(episode_start=start, drift_factor=drift)

    def _sample_iid(self, key: jax.Array) -> EpisodeConfig:
        """IID mode: uniform random start, no drift."""
        start = int(jax.random.randint(key, (), 0, self.max_start + 1))
        return EpisodeConfig(episode_start=start, drift_factor=1.0)

    def _sample_drift_shock(self, key: jax.Array) -> EpisodeConfig:
        """Drift-shock OOD: uniform start + large fixed drift (1.25)."""
        start = int(jax.random.randint(key, (), 0, self.max_start + 1))
        return EpisodeConfig(episode_start=start, drift_factor=1.25)

    def sample_batch(
        self,
        start_idx: int,
        batch_size: int,
        key: jax.Array,
        *,
        mode: str = "train",
    ) -> list:
        """Sample a batch of episode configs.

        Args:
            start_idx: First episode index in the batch.
            batch_size: Number of episodes.
            key: JAX PRNG key (split internally).
            mode: Sampling mode.

        Returns:
            List of ``EpisodeConfig`` of length ``batch_size``.
        """
        keys = jax.random.split(key, batch_size)
        return [
            self.sample(start_idx + i, keys[i], mode=mode)
            for i in range(batch_size)
        ]


def apply_drift(
    load_profiles_p: jnp.ndarray,
    load_profiles_q: jnp.ndarray,
    drift_factor: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Scale load profiles by a drift factor.

    Args:
        load_profiles_p: (T, n_bus) active load in p.u.
        load_profiles_q: (T, n_bus) reactive load in p.u.
        drift_factor: Multiplicative scaling (e.g. 1.03 for +3% load growth).

    Returns:
        Scaled ``(load_profiles_p, load_profiles_q)``.
    """
    return (
        load_profiles_p * drift_factor,
        load_profiles_q * drift_factor,
    )
