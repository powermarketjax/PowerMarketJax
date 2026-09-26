"""L0: the x64 guard on `evaluation.runtime_stamp` still bites, and its one exit.

`runtime_stamp` refuses to stamp a product while x64 is off,
because the `dtype` it would record is the harness default rather than the run's
-- market 02 measured that failure mode and the refusal is what replaced it.
Market 04 is the first market in this repository whose arithmetic is float32 by
design: its clearing is two sorted orders, two cumulative sums and a comparison
reduction, `envs/p2p/clearing.py` writes float32 at every entry point, and the
out-of-package results its numbers are compared against were produced with x64
off.  So `requires_x64=False` exists, and this file is what keeps it from
becoming a way to skip the check everywhere.

**Both branches are exercised for free, because this test session runs with x64
off.**  That is asserted first rather than assumed: if a conftest ever turns x64
on, the guard test below would pass while testing nothing, and the assertion is
what turns that into a failure with a reason instead of a green tick.
"""
import inspect
import pathlib
import sys

import jax
import jax.numpy as jnp
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "benchmark"))

import evaluation                                                 # noqa: E402


def test_this_session_runs_without_x64():
    """The precondition both tests below depend on, stated out loud."""
    assert not jax.config.jax_enable_x64, (
        "this session enables x64, so `test_the_guard_still_bites` would pass "
        "without the guard doing anything; either restore an x64-off session or "
        "re-point that test at whatever now distinguishes the two branches")
    assert jnp.zeros(1).dtype == jnp.float32


def test_the_guard_still_bites():
    """Default behaviour is unchanged: no keyword, x64 off, still refused."""
    with pytest.raises(RuntimeError, match="jax_enable_x64"):
        evaluation.runtime_stamp()


def test_the_default_is_the_strict_branch():
    """The four drivers that clear an LP keep the strict branch by default.

    Read off the live signature rather than restated, so a change of the default
    fails here instead of silently relaxing every existing caller.
    """
    default = inspect.signature(evaluation.runtime_stamp).parameters[
        "requires_x64"].default
    assert default is True, (
        f"requires_x64 defaults to {default!r}; every driver that stamps a "
        f"product without passing the keyword would stop being checked")


def test_the_opt_out_stamps_what_the_process_actually_is():
    """`requires_x64=False` records, and records the float32 it really is.

    The dtype is asserted against `jnp.zeros(1).dtype` read here rather than
    against the literal "float32": the whole point of this helper is that the
    value is derived from the process, and comparing it to a literal would
    reintroduce the constant it replaced.
    """
    stamp = evaluation.runtime_stamp(requires_x64=False)
    assert set(stamp) == {"platform", "devices", "cuda_visible_devices",
                          "dtype"}
    assert stamp["dtype"] == str(jnp.zeros(1).dtype)
    assert stamp["platform"] == jax.default_backend()
    assert stamp["devices"] and all(isinstance(d, str) for d in stamp["devices"])


def test_the_card_mask_is_recorded_and_its_two_empty_states_stay_apart(monkeypatch):
    """`cuda_visible_devices`, added 2026-08-28 for a gap this round measured.

    `devices` reports `cuda:0` on every card, because `CUDA_VISIBLE_DEVICES`
    renumbers what the process can see: three market 02 seeds pinned to physical
    cards 0, 1 and 2 all wrote `['cuda:0']`, so the products of that batch could
    not say which card produced which.  The mask is recorded as a **string** and
    not parsed, because `unset` and `""` are different states -- the empty string
    makes every card invisible -- and a `None`/`""` pair in JSON would read as
    the same thing.
    """
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert evaluation.runtime_stamp(requires_x64=False)[
        "cuda_visible_devices"] == "unset"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert evaluation.runtime_stamp(requires_x64=False)[
        "cuda_visible_devices"] == ""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    assert evaluation.runtime_stamp(requires_x64=False)[
        "cuda_visible_devices"] == "2"
