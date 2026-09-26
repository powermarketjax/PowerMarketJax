"""The shared-parameter actor-critic, and the squashing it needs.

One network serves every agent.  The agent axis is a leading batch axis, not a
`vmap` over parameters, so a rollout of `E` environments and `N` agents is one
forward pass over `(E, N, obs_dim)`.

**Bounds are per coordinate and some of them are infinite.**  The ancillary
action concatenates an energy markup, which is bounded to `[1, markup_max]`,
with a raw reserve action, whose own box `envs/ancillary/action.py` derives
from that market's `pi_scale` and `volr`.  A learner that squashed every
coordinate into the energy bounds would silently cap the reserve action at
`markup_max`, so squashing is decided **per coordinate** from whether both
bounds are finite.  The per-coordinate machinery stays even though the
ancillary box is now finite: the day-ahead and real-time markets still publish
one pair for the whole action, and `_squashable` is what keeps a market that
leaves a coordinate unbounded from having one invented for it.

**`spec["action_low"]` does not describe the whole action of the ancillary
market.**  It carries the energy map's bounds while the action also has `n_prod`
reserve columns, so a caller that reads it alone bounds the reserve columns to
the markup range.  `bounds_for` builds the full arrays and is what the harness
uses; the incomplete key is left as it is because changing a published `spec`
belongs to that market's line, not here.
"""
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

#: Squashed coordinates are pushed to `atanh` of this at the extremes rather
#: than to +-1, so that a saturated policy still has a finite log-density.  At
#: float64 `atanh(1 - 1e-6)` is 7.25, well inside the range a `tanh` unit
#: reaches, so the clip binds only on genuinely saturated actions.
TANH_CLIP = 1.0 - 1e-6


def bounds_for(spec: dict, reserve_columns: int = 0
               ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Full per-coordinate bounds for one market, shaped like the action.

    `reserve_columns` is the number of trailing reserve columns; it is zero for
    the day-ahead and real-time markets, whose action is the markup alone, and
    `n_prod` for the ancillary market.  The value is passed rather than
    inferred because `spec` does not record which columns the energy bounds
    cover, which is the gap the module docstring describes.

    **Their box comes from the market, not from here and not from the caller.**
    It is read off `spec["reserve_low"]` and `spec["reserve_high"]`, which
    `envs/ancillary/action.py` derives from that market's `pi_scale` and
    `volr`; see its module docstring for why the ends are those two numbers and
    why the box is a change of coordinates rather than a restriction.  A
    learner does not get to choose it: an action space that each algorithm
    declares for itself is one quantity with two defaults, which is worse than
    either default alone.  The box was introduced for SAC
    (`learning/sac.py`), whose critic reads the raw coordinate and whose actor
    maximises the critic, so with no bound there is no fixed point and the
    first iteration on the ancillary market diverged (measured 2026-09-03,
    `q_loss` 1e27 within 3 072 updates).  It now holds for
    every algorithm on that market, IPPO included.

    A spec that declares reserve columns without declaring their box raises
    rather than falling back to +-inf: the fallback would be a second, silent
    declaration of the action space, and the archives produced under it would
    look exactly like the ones produced under the market's own.
    """
    shape = tuple(spec["action_shape"])
    low = np.full(shape, float(spec["action_low"]), np.float64)
    high = np.full(shape, float(spec["action_high"]), np.float64)
    if reserve_columns:
        missing = [k for k in ("reserve_low", "reserve_high") if k not in spec]
        if missing:
            raise KeyError(
                f"reserve_columns={reserve_columns} but the spec does not "
                f"carry {missing}: the market that publishes reserve columns "
                f"has to publish their box too, because nothing else knows "
                f"this market's `pi_scale` and `volr`")
        lo, hi = float(spec["reserve_low"]), float(spec["reserve_high"])
        if not lo < hi:
            raise ValueError(f"the spec's reserve box must be an increasing "
                             f"pair, got {(lo, hi)!r}")
        low[..., -reserve_columns:] = lo
        high[..., -reserve_columns:] = hi
    return jnp.asarray(low), jnp.asarray(high)


class SharedActorCritic(nn.Module):
    """One policy and one value head, shared by every agent.

    `init_scale` is the output layer's initialisation scale.  The right value is
    a per-market calibration: too small or too large a scale changes how much of
    the action space the initial policy can reach.
    """

    act_dim: int
    hidden: Sequence[int] = (64, 64)
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, obs):
        """Return ``(mean, log_std, value)`` for a batch of observations.

        Every leading axis of `obs` is a batch axis -- in a rollout those are
        the environment axis and the agent axis -- and only the last one is
        `obs_dim`, so one call serves every agent of every environment.  `mean`
        is the **pre-squash** location of the Gaussian, not an action: it is
        `to_action` that maps it into the market's box.  `log_std` is a free
        parameter of the module rather than a head, so the exploration scale is
        state-independent and shared across agents.  `value` has its trailing
        unit axis dropped, so it broadcasts against the per-agent reward.
        """
        x = obs
        for h in self.hidden:
            x = nn.tanh(nn.Dense(
                h, kernel_init=nn.initializers.orthogonal(np.sqrt(2.0)),
                bias_init=nn.initializers.zeros)(x))
        mean = nn.Dense(
            self.act_dim,
            kernel_init=nn.initializers.orthogonal(self.init_scale),
            bias_init=nn.initializers.zeros)(x)
        log_std = self.param("log_std", nn.initializers.zeros, (self.act_dim,))
        value = nn.Dense(
            1, kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.zeros)(x)[..., 0]
        return mean, log_std, value


def _squashable(low, high):
    """Per-coordinate mask: a coordinate is squashed only if both bounds are finite."""
    return jnp.isfinite(low) & jnp.isfinite(high)


def to_action(pre, low, high):
    """Map a real-valued sample to the action box, per coordinate.

    Bounded coordinates go through `tanh` and an affine map, so the bound holds
    by construction and the environment never has to repair an out-of-range
    action -- which `envs/day_ahead/action.py` refuses to do on the ground that
    repairing an action is the learner's business, not the market's.  Unbounded
    coordinates pass through.
    """
    sq = _squashable(low, high)
    t = jnp.tanh(pre)
    mapped = low + 0.5 * (high - low) * (t + 1.0)
    return jnp.where(sq, mapped, pre)


def log_prob(pre, mean, log_std, low, high):
    """Log-density of the squashed sample, summed over the action coordinates.

    The `tanh` Jacobian enters only on the coordinates that were squashed, which
    is why the correction is masked rather than applied wholesale: applying it
    on an unbounded coordinate would subtract a term for a transform that never
    happened, and nothing downstream would report it.
    """
    std = jnp.exp(log_std)
    gauss = -0.5 * (((pre - mean) / std) ** 2 + 2.0 * log_std
                    + jnp.log(2.0 * jnp.pi))
    t = jnp.clip(jnp.tanh(pre), -TANH_CLIP, TANH_CLIP)
    sq = _squashable(low, high)
    # The width enters the log only where the coordinate is squashed.  On an
    # unbounded coordinate `high - low` is `inf`, and although `where` below
    # discards that branch's VALUE, its GRADIENT with respect to `pre` is
    # `inf * 0 = nan` and `where` does not discard gradients.  IPPO never
    # differentiated through `pre` here (it is stored data), so the defect was
    # invisible until SAC reparameterised the sample (measured 2026-09-03 on
    # the ancillary market: `q_loss = nan` on the first update).  Substituting
    # a finite width on the masked coordinates leaves every selected value
    # bit-identical and makes the masked branch's gradient finite.
    width = jnp.where(sq, high - low, 1.0)
    correction = jnp.log(0.5 * width * (1.0 - t ** 2))
    return jnp.sum(gauss - jnp.where(sq, correction, 0.0), axis=-1)


def entropy(log_std, low, high):
    """Entropy of the pre-squash Gaussian, summed over coordinates.

    The squashed variable's entropy has no closed form; this is the usual
    surrogate, and it does not go to zero when the squashed policy becomes
    deterministic at a bound.
    """
    del low, high
    return jnp.sum(log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e))


def sample_pre(key, mean, log_std):
    """Draw a pre-squash Gaussian sample, reparameterised through `key`.

    The sample lives in the unbounded space `SharedActorCritic` emits, before
    `to_action` maps it into the market's box.  It is this pre-squash value,
    not the action, that the rollout stores and `log_prob` scores.
    """
    return mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape,
                                                       mean.dtype)
