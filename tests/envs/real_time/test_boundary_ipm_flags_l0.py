"""L0: `reg_coef` and `stop_tol` reach **both** of this market's clearing operators.

This is `test_monitored_lines_l0.py`'s defect a second time, on two different
keywords.  `env.make_env` builds the operator each step calls and
`boundary.make_boundary` builds the one that produces the dispatch the first
period opens from.  `reg_coef` / `stop_tol` reached the first and not the
second, while `ipm_reg_coef` / `ipm_stop_tol` in every product were read off the
first alone -- **the stamp said the recipe was in effect and the opening solve
ran on the defaults** (found 2026-09-17).

**The numbers were right and the stamp was wrong.**  Measured on
`case813nem` rated, 13 days sampled every 30: the default boundary's
`dual_residual` stays under 3.2e-9 and `|dp_prev|` under 5.3e-8 MW, because the
opening solve is a truthful zero-ramp clearing that does not reach the fixed
point the recipe exists to escape.  That is the harder half to notice -- a
product whose numbers are fine and whose provenance is false.

The first test below is the original draft, which is red before the fix.  The second
is the half it did not have: the refusal has to fire when only one operator gets
the flags, or a later edit can drop the pass-through and go green again.
"""
import jax
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.day_ahead import load_gb_demand
from powermarketjax.envs.real_time import load_da_position
from powermarketjax.envs.real_time.demand import load_gb_demand_half_hourly
from powermarketjax.envs.real_time.env import make_env

CHAIN = "step1prime_seasons"
CAP_SCALE, RAMP_SCALE, MARKUP_MAX = 0.60, 1.00, 2.0
REG, STOP = 1e-16, (1e-8, 1e-7)


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


def _build(inputs, **kw):
    case, pos, hh, forecast = inputs
    return make_env(case, pos, hh, forecast, n_segments=1,
                    markup_max=MARKUP_MAX, cap_scale=CAP_SCALE,
                    ramp_scale=RAMP_SCALE, **kw)


def test_both_operators_take_the_same_ipm_flags(inputs):
    """The original draft: read the effective value off each operator's own spec."""
    _env, spec = _build(inputs, reg_coef=REG, stop_tol=STOP)
    c, b = spec["clearing"], spec["boundary_clearing"]
    assert c["reg_coef"] == REG and tuple(c["stop_tol"]) == STOP, (
        f"the step clearing did not take the flags: {c['reg_coef']}, {c['stop_tol']}")
    assert b["reg_coef"] == c["reg_coef"], (
        f"the boundary was built with reg_coef={b['reg_coef']} while the step "
        f"clearing got {c['reg_coef']}; the stamp would name one and the episode "
        f"would open on the other")
    assert tuple(b["stop_tol"]) == tuple(c["stop_tol"]), (b["stop_tol"], c["stop_tol"])


def test_the_default_leaves_both_operators_on_the_market_defaults(inputs):
    """No flags must be what the market was before the flags existed.

    Without this, a pass-through that hard-coded the recipe would satisfy the
    test above and silently change every run that passes nothing.
    """
    _env, spec = _build(inputs)
    for name in ("clearing", "boundary_clearing"):
        assert spec[name]["stop_tol"] is None, (
            f"{name} took a stop_tol with no argument asking for one")
    assert spec["clearing"]["reg_coef"] == spec["boundary_clearing"]["reg_coef"]


def test_the_guard_fires_when_only_one_operator_gets_them(inputs, monkeypatch):
    """The injection half, through the public API.

    `make_env` is called exactly as the passing tests call it, with the one
    difference that the boundary is handed a factory which drops the two
    keywords -- the defect this file exists for.  The refusal has to come from
    `make_env`; the monkeypatch touches no file on disk, so it does not change
    the setup any other concurrent run is using.
    """
    import powermarketjax.envs.real_time.env as env_mod
    real = env_mod.make_boundary

    def deaf(*a, **kw):
        kw.pop("reg_coef", None)
        kw.pop("stop_tol", None)
        return real(*a, **kw)

    monkeypatch.setattr(env_mod, "make_boundary", deaf)
    with pytest.raises(ValueError, match="the stamp would name one operator"):
        _build(inputs, reg_coef=REG, stop_tol=STOP)
