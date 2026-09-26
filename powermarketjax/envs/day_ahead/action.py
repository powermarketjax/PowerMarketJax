"""Day-ahead action map: raw action in, offer curve out.

Array in, array out, pure.  The offer curve is non-decreasing in the segment
index by construction rather than by a repair step:

    full offer      alpha (n_units, K, T) real
                    offer[i,k,t] = m_env[i,1] + sum_{j<=k} softplus(alpha[i,j,t]) * m_env[i,j]

    scalar markup   alpha (n_units,) in [1, markup_max]
                    offer[i,k,t] = alpha[i] * m_env[i,k]

where ``m_env`` is the running maximum of the segment costs.  Both forms hold for
**every** real action.  Two properties of that arithmetic are load bearing.

**The increment is scaled by the envelope, not by the raw segment cost**, which
is negative wherever the true marginal cost falls over a unit's operating range;
a negative increment would make the curve descend.

**The map is smooth.**  `softplus` is strictly increasing with a strictly
positive derivative, so an agent bidding far below cost still gets gradient.  The
one limit is float64: below about ``alpha = -740`` both softplus and its
derivative underflow to exactly zero, so the offer sits on its cost floor and
produces no gradient.

The action lives on the **unit** axis, which with one generating company per unit
is also the agent axis.
"""
from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .clearing import segment_costs

#: The action that makes `softplus` return exactly zero.  Both float32 and
#: float64 underflow `exp` well before this, so `logaddexp(x, 0)` is `0.0` bit
#: for bit rather than merely small.  It is the only way to reach a zero
#: increment, which `truthful_action` needs and the map otherwise never gives:
#: `softplus` is strictly positive on every representable action.
ZERO_INCREMENT = -800.0


def truthful_action(
    case,
    n_segments: int,
    n_periods: int,
    kind: str = "full",
) -> jnp.ndarray:
    """The action whose offer is the true cost, i.e. the envelope itself.

    Non-learning agents bid this action, which the offer map turns into a
    truthful offer.  It is not the obvious one:

        kind="markup"   alpha = 1, which is the lower bound of its own space
        kind="full"     alpha[i, 0] = ZERO_INCREMENT, and for k >= 1
                        softplus(alpha[i, k]) = (m_env[k] - m_env[k-1]) / m_env[k]

    The first segment is the awkward one.  The offer map prices it at
    ``m_env[1] * (1 + softplus(alpha))``, which is *strictly above* cost for every
    finite action, so a truthful first segment needs a zero increment and that is
    reachable only through the underflow above.  The later segments invert
    softplus exactly, and a flat stretch of the envelope gives a zero increment
    there too.

    Returns an array of the shape `make_offer_map` expects for that ``kind``.
    """
    _, raw = segment_costs(case, n_segments, monotone=False)
    envelope = np.maximum.accumulate(raw, axis=1)                 # (n_units, K)
    n_units = envelope.shape[0]

    if kind == "markup":
        return jnp.ones((n_units,))
    if kind != "full":
        raise ValueError(f"kind must be 'full' or 'markup', got {kind!r}")

    action = np.full((n_units, n_segments), ZERO_INCREMENT)
    with np.errstate(divide="ignore", invalid="ignore"):
        step = np.diff(envelope, axis=1) / envelope[:, 1:]        # in [0, 1)
        # inverse softplus; log(expm1(0)) is -inf, which the sentinel replaces.
        # A zero envelope entry leaves `step` non-finite and takes the same
        # branch, which is right: a zero cost needs a zero increment.
        action[:, 1:] = np.where(step > 0.0, np.log(np.expm1(step)), ZERO_INCREMENT)
    return jnp.broadcast_to(jnp.asarray(action)[:, :, None],
                            (n_units, n_segments, n_periods))


def make_offer_map(
    case,
    n_segments: int,
    n_periods: int,
    kind: str = "full",
    markup_max: Optional[float] = None,
) -> Tuple[Callable, Dict]:
    """Build the action map for one case, segment count and horizon.

    Returns ``(offer_map, spec)`` where ``offer_map(action)`` is pure and
    jittable and returns the ``(n_units, n_segments, n_periods)`` offer that
    `clearing` consumes, and ``spec`` carries the action shape and its bounds:

        kind="full"    shape (n_units, K, T), unbounded
        kind="markup"  shape (n_units,), bounded to [1, markup_max]

    The bounds belong to the action space, not to this function.  Clipping an
    out-of-range action here would be a repair of the action, and `markup_max`
    is a scenario parameter with no defensible default, so it is required rather
    than assumed.
    """
    _, raw = segment_costs(case, n_segments, monotone=False)
    # `segment_costs(..., monotone=True)` returns the same array; it is written
    # out here because the envelope is the whole reason the map below is
    # monotone.
    envelope = np.maximum.accumulate(raw, axis=1)
    if envelope.min() < 0.0:
        # The envelope is nonnegative exactly when the first segment is.  If it
        # ever is not, the increments below change sign and produce a
        # non-monotone offer that nothing downstream can detect, so this fails
        # at construction instead.
        raise ValueError(
            f"segment cost envelope has negative entries (min "
            f"{envelope.min():.4g}); the offer map cannot keep offers "
            f"non-decreasing under it")

    n_units = envelope.shape[0]
    m_env = jnp.asarray(envelope)[:, :, None]           # (n_units, K, 1)
    T = n_periods

    if kind == "full":
        spec = dict(shape=(n_units, n_segments, T), low=-np.inf, high=np.inf)

        def offer_map(action):
            """Full offer curve: ``(n_units, K, T)`` action in, the same shape out.

            The action is unbounded and every entry of the offer returned is an
            as-bid price in \\$/MWh for that unit, segment and period.  The result
            is non-decreasing in the segment index for **every** finite action,
            since each increment is a strictly positive multiple of a nonnegative
            envelope entry, so no repair step follows and none may be added.
            """
            # pi_k = pi_{k-1} + softplus(alpha_k) * m_env_k, unrolled into the
            # cumulative sum it is, so the map is one pass and vmaps cleanly
            increment = jax.nn.softplus(action) * m_env
            return m_env[:, :1] + jnp.cumsum(increment, axis=1)

    elif kind == "markup":
        if markup_max is None:
            raise ValueError("kind='markup' needs markup_max: it bounds the "
                             "action space and has no defensible default")
        spec = dict(shape=(n_units,), low=1.0, high=float(markup_max))

        def offer_map(action):
            """Scalar markup: ``(n_units,)`` action in, ``(n_units, K, T)`` offer out.

            Each entry scales that unit's whole cost envelope, so the offer stays
            non-decreasing in the segment index because the envelope is and the
            multiplier is positive.  The action is expected in
            ``[1, markup_max]``; that bound belongs to the action space and is not
            imposed here, so an action outside it is mapped rather than clipped.
            """
            # one multiplier for the whole day, so the period axis is a
            # broadcast rather than a decision
            return jnp.broadcast_to(action[:, None, None] * m_env,
                                    (n_units, n_segments, T))

    else:
        raise ValueError(f"kind must be 'full' or 'markup', got {kind!r}")

    return offer_map, spec
