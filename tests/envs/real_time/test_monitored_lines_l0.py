"""L0: `monitored_lines` reaches **both** of this market's clearing operators.

This market builds two.  `env.make_env` builds the one each step calls, and
`boundary.make_boundary` builds a second one that produces the dispatch the
first period starts from.  A `monitored_lines` that reached one and not the
other opens every episode at the dense route's vertex and then clears it on the
low-rank one, and **nothing in any product says so**: every array still comes
back well-formed and the stamp a driver writes would name whichever operator it
happened to ask.

So the checks here read the line set **off each operator's own `spec`**, never
off the argument that was passed, and the last one reverts the wiring in a copy
of the source to confirm the runtime refusal would actually fire.

`case29gb` publishes a rating for every line, so `rated_lines` there is the
whole set and the two routes coincide; the route split only has content on
`case813nem`, where 1 271 of 1 278 lines sit at 1e6 MW.  The tests below
therefore use an explicit index list rather than `"rated"` -- an explicit list
exercises the same wiring on the small case and does not need the big one.
"""
import ast
import pathlib

import jax
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.day_ahead import load_gb_demand
from powermarketjax.envs.real_time import load_da_position
from powermarketjax.envs.real_time.boundary import make_boundary
from powermarketjax.envs.real_time.clearing import make_rt_clearing
from powermarketjax.envs.real_time.demand import load_gb_demand_half_hourly
from powermarketjax.envs.real_time.env import make_env

REPO = pathlib.Path(__file__).resolve().parents[3]
ENV_SRC = REPO / "powermarketjax" / "envs" / "real_time" / "env.py"

CAP_SCALE, RAMP_SCALE, MARKUP_MAX = 0.60, 1.00, 2.0
CHAIN = "step1prime_seasons"
#: A few line indices, enough to make the low-rank route the one taken.  Kept
#: small and explicit so this file states its own input rather than deriving it
#: from the case it is checking against.
SOME_LINES = np.array([0, 3, 7], np.int64)


@pytest.fixture(scope="module", autouse=True)
def x64():
    prev = jax.config.jax_enable_x64, jax.config.jax_default_matmul_precision
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    yield
    jax.config.update("jax_enable_x64", prev[0])
    jax.config.update("jax_default_matmul_precision", prev[1])


@pytest.fixture(scope="module")
def inputs(x64):
    pos = load_da_position(chain=CHAIN)
    case = load_case(pos["meta"]["case"])
    hh, _d = load_gb_demand_half_hourly()
    forecast, _a, _days = load_gb_demand()
    return case, pos, hh, forecast


def _build(inputs, monitored):
    case, pos, hh, forecast = inputs
    return make_env(case, pos, hh, forecast, n_segments=1,
                    markup_max=MARKUP_MAX, cap_scale=CAP_SCALE,
                    ramp_scale=RAMP_SCALE, monitored_lines=monitored)


def _eff(spec_piece):
    """The line set an operator was actually built on, as a plain list."""
    m = spec_piece["monitored_lines"]
    return None if m is None else [int(i) for i in np.asarray(m)]


def test_the_default_is_every_line_on_the_dense_route(inputs):
    """No argument must be what the market was before the argument existed."""
    _env, spec = _build(inputs, None)
    for name in ("clearing", "boundary_clearing"):
        assert _eff(spec[name]) is None, f"{name} dropped line limits by default"
        assert str(spec[name]["kkt_route"]) == "dense", (
            f"{name} left the dense route with no argument asking it to")


def test_both_operators_get_the_set_and_the_route(inputs):
    """The load-bearing one: read off each operator, not off the argument."""
    _env, spec = _build(inputs, SOME_LINES)
    want = [int(i) for i in SOME_LINES]
    assert _eff(spec["clearing"]) == want, "the step clearing kept every line"
    assert _eff(spec["boundary_clearing"]) == want, (
        "the boundary kept every line while the step clearing did not -- the "
        "episode would open at one route's vertex and be cleared on the other")
    # the stamp carries the free-column sizing since 2026-09-16, e.g.
    # "lowrank+free(8,0)" here; what this test asks is that both operators
    # took the low-rank route and the same variant of it
    assert str(spec["clearing"]["kkt_route"]).startswith("lowrank")
    assert str(spec["boundary_clearing"]["kkt_route"]) == str(spec["clearing"]["kkt_route"])


def test_the_wrapper_passes_it_through(inputs):
    """`make_rt_clearing` and `make_boundary` each on their own, so a failure
    says which of the two layers lost it."""
    case, _pos, _hh, _fc = inputs
    want = [int(i) for i in SOME_LINES]
    _clear, cspec = make_rt_clearing(case, n_segments=1, cap_scale=CAP_SCALE,
                                     ramp_scale=RAMP_SCALE, period_hours=0.5,
                                     monitored_lines=SOME_LINES)
    assert _eff(cspec) == want, "make_rt_clearing did not pass it to make_clearing"
    bspec = {}
    _b, _offer = make_boundary(case, n_segments=1, cap_scale=CAP_SCALE,
                               period_hours=0.5, monitored_lines=SOME_LINES,
                               spec_out=bspec)
    assert _eff(bspec) == want, "make_boundary did not pass it to its clearing"


def test_make_boundary_keeps_its_two_value_return(inputs):
    """Eight call sites unpack exactly two values; `spec_out` exists so that
    reading the effective set back does not change that."""
    case, _pos, _hh, _fc = inputs
    out = make_boundary(case, n_segments=1, cap_scale=CAP_SCALE,
                        period_hours=0.5)
    assert isinstance(out, tuple) and len(out) == 2, (
        f"make_boundary returned {len(out)} values; the call sites in "
        f"tools/rt_scenario/ and tests/envs/real_time/ unpack two")


def test_the_guard_fires_when_the_boundary_misses_it(inputs, monkeypatch):
    """The injection half, run through the public API.

    `make_env` is called exactly as the passing tests above call it, with the
    one difference that the boundary is handed a factory which drops
    `monitored_lines` -- the defect this whole file exists for.  The refusal
    has to come from `make_env` itself; the monkeypatch touches no file on
    disk, so it does not change the setup any other concurrent run is
    using.
    """
    import powermarketjax.envs.real_time.env as env_mod
    real = env_mod.make_boundary

    def deaf(*a, **kw):
        kw.pop("monitored_lines", None)          # the wire that goes missing
        return real(*a, **kw)

    monkeypatch.setattr(env_mod, "make_boundary", deaf)
    with pytest.raises(ValueError, match="the episode would open at one"):
        _build(inputs, SOME_LINES)


def test_the_mismatch_check_bites():
    """The source still reads the way the guard above depends on.

    A companion to the monkeypatch test, not a substitute: that one proves the
    guard fires, this one proves the call site it guards has not been renamed
    out from under it.  Checked as a string count first, so a rename turns this
    red instead of silently matching nothing.
    """
    src = ENV_SRC.read_text()
    # re-anchored 2026-09-17: the call now carries `lowrank_free` and
    # `lu_batching` after `spec_out`, so the tail is these two lines
    anchor = ("                                              monitored_lines=monitored_lines,\n"
              "                                              spec_out=bspec,\n")
    assert src.count(anchor) == 1, (
        "the boundary call in real_time/env.py no longer reads as this test "
        "expects; re-anchor it rather than deleting the check")
    broken = src.replace(anchor, "                                              spec_out=bspec,\n")
    tree = ast.parse(broken)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "make_env")
    call = next(c for c in ast.walk(fn) if isinstance(c, ast.Call)
                and isinstance(c.func, ast.Name) and c.func.id == "make_boundary")
    assert not [k for k in call.keywords if k.arg == "monitored_lines"], (
        "the reversion did not remove the argument, so this test proves nothing")
    # and the guard that would catch it at run time is still in the real source
    assert "the episode would open at one route's" in src, (
        "real_time/env.py no longer refuses a boundary on a different line set")
