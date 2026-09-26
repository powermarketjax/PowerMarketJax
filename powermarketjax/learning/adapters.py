"""One shape for five markets, so the harness stops indexing tuples positionally.

The five markets return one of two shapes.  Ancillary, P2P and local flexibility
each return a four-tuple ``(reset, step, step_auto_reset, spec)``.  Day-ahead
returns ``(DayAheadEnv(reset, step, step_auto_reset, get_obs, make_params),
spec)``, and real-time returns ``(env, spec)`` with the same five attributes.
Indexing the second pair positionally is what makes the mismatch dangerous
rather than merely untidy: ``env[3]`` is ``get_obs`` there, a callable, so a
harness that reads ``spec = env[3]`` receives a function and fails much later,
somewhere that does not name the cause.

The ``spec`` keys diverge the same way.  Three markets declare ``action_shape``,
``action_low``, ``action_high`` and ``baseline_action`` at the top level; the
day-ahead market nests the first three under ``action`` and declares no baseline
at all, and the real-time market declares ``action_shape`` alone.  This module
fills the gaps and leaves the market's own dictionary untouched.

One divergence is deliberately left alone, because normalising it would hide a
question rather than answer it: ``baseline_action`` is an **array** in the
ancillary market and a **function of the state** in the P2P market, where the
truthful side of the offer depends on the current net position.  The name is
shared and the type is not, so a caller that treats the key as an action array
works for one market and fails for the other.  This module passes whichever it
finds through unchanged.
"""
from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp
import numpy as np

__all__ = ["unpack_env"]

#: the fields a market's environment object carries when it is not the four-tuple
_ENV_ATTRS = ("reset", "step", "step_auto_reset")


def _callables(built: Any) -> Tuple[Callable, Callable, Callable, Dict]:
    """The three pure functions and the spec, whichever shape they arrived in."""
    if len(built) == 4 and all(callable(x) for x in built[:3]):
        return built[0], built[1], built[2], built[3]
    if len(built) == 2:
        env, spec = built
        missing = [a for a in _ENV_ATTRS if not hasattr(env, a)]
        if missing:
            raise TypeError(f"a two-element return must carry {_ENV_ATTRS} on its "
                            f"first element; this one is missing {missing}")
        return env.reset, env.step, env.step_auto_reset, spec
    raise TypeError(f"expected the four-tuple env interface or an (env, spec) pair, "
                    f"got {len(built)} elements")


def unpack_env(built: Any) -> Tuple[Callable, Callable, Callable, Dict]:
    """Return ``(reset, step, step_auto_reset, spec)`` for any of the five markets.

    ``spec`` is a copy carrying ``action_shape``, ``action_low``, ``action_high``
    and ``baseline_action`` at the top level whatever the market declared, so the
    harness reads one set of names.  The market's own dictionary is not modified,
    because two callers unpacking the same built environment must not see each
    other's additions.

    ``baseline_action`` is filled only where it follows from what the market
    declared.  For ``kind="markup"`` the truthful offer is a multiplier of one,
    which is exactly ``action_low``, so the baseline is the lower corner of the
    action space by construction rather than by a rule invented here.  For any
    other action space this function does not guess: it leaves the key absent
    and the caller fails on a missing key instead of on a plausible wrong array.
    """
    reset, step, step_auto_reset, spec = _callables(built)
    out = dict(spec)

    nested = spec.get("action")
    if "action_shape" not in out and isinstance(nested, dict):
        out["action_shape"] = nested["shape"]
    if "action_low" not in out and isinstance(nested, dict):
        out["action_low"] = nested["low"]
    if "action_high" not in out and isinstance(nested, dict):
        out["action_high"] = nested["high"]

    env_obj = built[0] if len(built) == 2 else None
    own = getattr(env_obj, "truthful_action", None)
    derived = None
    if out.get("kind") == "markup" and "action_shape" in out and "action_low" in out:
        derived = jnp.full(out["action_shape"], float(out["action_low"]))

    if "baseline_action" not in out:
        if own is not None:
            out["baseline_action"] = own()
        elif derived is not None:
            out["baseline_action"] = derived

    # where both routes exist the derivation is checked against the market's own
    # rather than trusted.  This is the one place both are in scope, and a silent
    # disagreement would put every arm on the wrong baseline while every number
    # still looked ordinary.
    if own is not None and derived is not None:
        theirs, mine = np.asarray(own()), np.asarray(derived)
        if theirs.shape != mine.shape or not np.allclose(theirs, mine):
            raise ValueError(
                "the baseline derived from action_low disagrees with the "
                f"market's own truthful_action: {mine} against {theirs}")

    return reset, step, step_auto_reset, out
