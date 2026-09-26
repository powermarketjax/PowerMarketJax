"""Which matmuls a compiled module runs, and in what element type.

**One implementation, two consumers.**  A GPU-side audit script reads this on
the card and `tests/learning/test_learner_dtype_l0.py` reads it in CI; a second
copy of the classification would be one criterion with two definitions, which is
worse than either alone.  Nothing here imports jax, so a test can use
it without a module-level `jax_enable_x64` arriving with it.

**What it is for.**  With `jax_enable_x64` on, a float32 parameter against a
float64 input promotes the matmul back to float64, so the dtype that was passed
in says nothing about the precision that ran.  The element type of the matmul in
the COMPILED module is what says it.

**Both `dot` and `custom-call` are counted.**  Which one a matmul becomes is the
backend's choice: on CPU it stays a `dot`, on GPU it is usually a `custom-call`
into cuBLAS.  A pattern matching only `dot(` therefore returns an empty result
on GPU, and an empty result reads exactly like a clean one -- so the
callers are given the count and are expected to refuse a verdict at zero.
"""
import re

_ELEM = re.compile(r"^\s*%?[\w.\-]+ = ([a-z0-9]+)\[")
_OPNAME = re.compile(r'op_name="([^"]*)"')

#: The scopes, in the order they are tested.  `jvp(` covers both the forward
#: pass under differentiation and `transpose(jvp(...))`, the backward one, which
#: is the whole of what a gradient step does with the networks.
GRADIENT_STEP = "gradient-step"
ROLLOUT_POLICY = "rollout-policy"
CLEARING = "clearing/other"
UNATTRIBUTED = "unattributed"

#: The rollout's own policy application, by the name flax gives it under `vmap`.
#: Named per learner rather than matched loosely: `SACActor` and
#: `SharedActorCritic` also appear under `jvp(...)` in the gradient step, and the
#: test above for `jvp` is what separates the two.
_ROLLOUT_NAMES = ("vmap(SACActor)", "vmap(SharedActorCritic)",
                  "vmap(_actor_per_agent)", "vmap(_apply_per_agent)")


def classify(op_name):
    """Which of the four scopes an instruction's `op_name` metadata places it in."""
    if not op_name:
        return UNATTRIBUTED
    if "jvp(" in op_name:
        return GRADIENT_STEP
    if any(n in op_name for n in _ROLLOUT_NAMES):
        return ROLLOUT_POLICY
    return CLEARING


def matmul_audit(text):
    """``{(scope, element_type): count}`` over every matmul in a compiled module."""
    rows = {}
    for ln in text.splitlines():
        if "= " not in ln:
            continue
        if not (" dot(" in ln
                or ("custom-call(" in ln and "gemm" in ln.lower())):
            continue
        m = _ELEM.match(ln)
        et = m.group(1) if m else ln.split("= ")[1].split("[")[0]
        nm = _OPNAME.search(ln)
        key = (classify(nm.group(1) if nm else ""), et)
        rows[key] = rows.get(key, 0) + 1
    return rows


def gradient_step_types(rows):
    """``{element_type: count}`` for the gradient step alone; may be empty."""
    return {et: c for (scope, et), c in rows.items() if scope == GRADIENT_STEP}


def format_audit(rows):
    """The table, one line per (scope, element type), for a log."""
    return "\n".join(f"      {scope:16s} {et:6s} x{c:4d}"
                     for (scope, et), c in sorted(rows.items()))
