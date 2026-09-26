"""The reserve half of the action, and the offer-separation diagnostic.

The energy half is the offer map of the real-time market, which is the day-ahead
map at one period, and it is reused unchanged.  What is new here is the reserve
map and one derived quantity.

    pi_res[i, j] = softplus(alpha_res[i, j]) * pi_scale

The map is nonnegative and strictly increasing in the raw action, so every real
action gives an admissible offer and the order of actions is the order of
offers.  Its infimum is zero, the truthful reserve offer, so the truthful
baseline is a limit of the action space rather than a point of it: the action
standing in for that limit has to be declared, and `TRUTHFUL_ACTION` is that
declaration.  Under float32 softplus underflows to exactly zero well before
that action, so the baseline is exactly representable even though it is a
limit.

**The learner's coordinate is bounded, and the bounds are this market's own
constants.**  In price the reserve action ranges over `[0, volr]`: zero because
reserve has no fuel cost, `volr` because the clearing caps the reserve dual
there, so an offer above `volr` is never accepted.  `spec["low"]` and
`spec["high"]` are that price interval written in the pre-softplus coordinate
the learner actually moves -- `RESERVE_ACTION_LOW` and `log(exp(volr /
pi_scale) - 1)` -- so a learner that squashes into them can reach every offer
the market can accept and none it cannot.  The box is a change of coordinates,
not a restriction of the strategy space, and it is declared here rather than by
each learner: an action space that two algorithms each declare for themselves
is one quantity with two defaults.

The box and `TRUTHFUL_ACTION` do not compete.  The box is where the learner
moves; `TRUTHFUL_ACTION` is the declared stand-in for the map's infimum, which
no finite coordinate attains, and it reaches the rollout as
`spec["baseline_action"]` with no squashing in between.  What the box costs is
that the learner's floor is `softplus(RESERVE_ACTION_LOW) * pi_scale` rather
than exactly zero; measured on three evaluation days of the year window, that
floor and an exactly-zero offer clear to the same system cost, the same
per-agent profit and the same reserve price to within the offer difference
itself.

`pi_scale` has no cost basis to anchor it, because the truthful reserve offer
is zero, so it is a required parameter with no default.

**Offer separation** is computed here rather than in the clearing, since it is
a property of the offers the action produced and the clearing is handed those
offers already built.  It is the smallest gap between adjacent offers of a
product, normalised by the median offer of that product; the minimum rather
than a mean is what matters, because the duals become unreliable through the
closest pair rather than through the average spread, and the reserve price is
not trustworthy below a measured separation.
"""
from typing import Callable, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np

#: The raw action that stands in for the truthful reserve offer.  Softplus
#: underflows to exactly zero much sooner in float32 than in float64, and this
#: market clears in float64, so the value must clear the float64 point rather
#: than the float32 one, or the truthful baseline would be exactly tied in one
#: precision but minutely separated in the other.
TRUTHFUL_ACTION = -800.0

#: The learner's floor in pre-softplus units.  It is not a tuned value and not
#: a second truthful action: it is the smallest coordinate the learner needs,
#: because `softplus(-20) * pi_scale` is 1.03e-7 \$/MWh at the adopted scale and
#: nothing in this market distinguishes that from the zero `TRUTHFUL_ACTION`
#: stands for.  Pushing it down to `TRUTHFUL_ACTION` instead would make the
#: box 32 times wider at the adopted scale and put 97% of it below 1e-7
#: \$/MWh, so a `tanh` squashed into it would spend almost all of its range
#: on offers this market cannot tell apart.
RESERVE_ACTION_LOW = -20.0


def make_reserve_offer_map(
    n_units: int,
    n_prod: int,
    pi_scale: float,
    volr: float,
) -> Tuple[Callable, Dict]:
    """Build the reserve offer map for one population and product ladder.

    Args:
        n_units: number of providers.
        n_prod: number of products.
        pi_scale: the declared price scale; required, since it has no
            defensible default.
        volr: the value of lost reserve, \\$/MWh, which is where the clearing
            caps the reserve dual.  Required for the same reason `pi_scale` is,
            and taken here rather than in the learner because the upper end of
            the action box is a fact about this market, not about the
            algorithm pointed at it.

    Returns:
        ``(reserve_offer_map, spec)`` where the map takes ``(n_units, n_prod)``
        raw actions and returns offers in \\$/MWh, and ``spec`` carries the
        action shape and its bounds.  The bounds are finite and are the image
        of the price interval ``[0, volr]`` under the inverse map, up to the
        one point the coordinate cannot represent: see the module docstring.
    """
    if not np.isfinite(pi_scale) or pi_scale <= 0.0:
        raise ValueError(f"pi_scale must be a positive finite price scale, "
                         f"got {pi_scale!r}")
    if not np.isfinite(volr) or volr <= 0.0:
        raise ValueError(f"volr must be a positive finite price, got {volr!r}")
    scale = float(pi_scale)
    #: The coordinate whose offer reaches `volr` exactly.  `expm1` rather
    #: than `exp(x) - 1` because `volr / pi_scale` is small whenever the
    #: scale is chosen near the cap, and there the subtraction loses the
    #: digits that decide the end.
    high = float(np.log(np.expm1(float(volr) / scale)))
    if not np.isfinite(high) or high <= RESERVE_ACTION_LOW:
        raise ValueError(
            f"volr={volr!r} at pi_scale={scale!r} puts the top of the reserve "
            f"action box at {high!r}, which is not above its floor "
            f"{RESERVE_ACTION_LOW!r}: the box would be empty or inverted, and "
            f"a learner squashing into it would have no admissible offer")
    spec = dict(shape=(n_units, n_prod), low=RESERVE_ACTION_LOW, high=high,
                pi_scale=scale, volr=float(volr))

    def reserve_offer_map(action: chex.Array) -> chex.Array:
        """`(n_units, n_prod)` raw actions to reserve offers in \\$/MWh.

        Nonnegative and strictly increasing in the action, so every real action
        is admissible and no offer has to be clipped.
        """
        return jax.nn.softplus(action) * scale

    return reserve_offer_map, spec


def truthful_reserve_action(n_units: int, n_prod: int,
                            dtype=jnp.float32) -> chex.Array:
    """The declared stand-in for the truthful reserve offer, which is zero."""
    return jnp.full((n_units, n_prod), TRUTHFUL_ACTION, dtype)


def offer_separation(offer_res: chex.Array,
                     subset: chex.Array = None) -> chex.Array:
    """Smallest adjacent gap in each product's offers, relative to its median.

    Returns ``(n_prod,)``.  With ``subset`` given, only the providers it selects
    take part; the mask is applied by pushing the others to ``+inf`` before the
    sort, so the shape stays static and the function stays jittable.

    Two different questions need this quantity over two different populations,
    and one number cannot answer both.  Over **everyone** it answers whether the
    program contains a degenerate block, and under a learner mask it is zero by
    construction, because every non-learner submits the same baseline action and
    two of them are already an exact tie.  That zero is correct: the population
    really is partly tied and the split inside that block really is arbitrary.
    Over the **learners** it answers whether the learning signal is polluted: if
    the learners are separated from each other and from the tied block, their
    own cleared quantities are pinned and the arbitrariness inside the block
    does not reach their reward.

    The tie among non-learners is not to be broken.  Offsetting their offers
    would make this number look better while manufacturing a cost difference
    that does not exist: the truthful reserve cost of every provider is zero,
    and inventing otherwise is forbidden.  An indicator that reads badly is not
    a reason to change the mechanism.
    """
    if subset is not None:
        keep = jnp.asarray(subset, bool)[:, None]
        offer_res = jnp.where(keep, offer_res, jnp.inf)
    sorted_offers = jnp.sort(offer_res, axis=0)
    diffs = jnp.diff(sorted_offers, axis=0)
    # The excluded providers sort to the end as +inf, so the last differences
    # are inf and inf-inf is nan; taking the minimum over a column holding a
    # nan returns nan and the guard below would then read every subset as
    # having no separation at all.  Dropping the non-finite differences first
    # is what makes the subset path measure the subset.
    diffs = jnp.where(jnp.isfinite(diffs), diffs, jnp.inf)
    gap = jnp.min(diffs, axis=0)
    finite = jnp.where(jnp.isfinite(offer_res), offer_res, jnp.nan)
    median = jnp.nanmedian(finite, axis=0)
    ok = jnp.isfinite(gap) & (median > 0.0)
    return jnp.where(ok, gap / jnp.where(median > 0.0, median, 1.0), 0.0)
