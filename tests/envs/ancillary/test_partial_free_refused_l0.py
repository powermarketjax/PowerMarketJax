"""L0: the local-border arrowhead refuses a **partial** free set.

`n_free[0]` on the local-border route must hold either no unit or every unit.
The `local` branch of `_make_kkt` zeroes the free units' rows of `H0^-1 U` by
**selection** -- `inf * 0` is NaN, so they cannot be removed by weight.  A unit
left *outside* the free set whose capacity-sharing row binds is not in that
selection: its block is singular to round-off, its rows of `X_unit` are `inf`,
and the capacitance sum turns the whole Newton direction non-finite.

**`ipm`'s `ok` swallows it**, which is why this needed deriving rather than
observing: the run does not stop and the products do not say so (derived
2026-09-17; injected, 29gb gives 230 non-finite components and a `case813nem`
cell at `(32, 815)` gives 7 of 60 steps non-finite after the stop point).

Both ends are safe and stay allowed: `n_fu == n_units` zeroes every unit row,
`n_fu == 0` does not take this branch.  Only the middle was unguarded, and it is
reachable only by passing `--lowrank-free-*` explicitly -- no default wiring
gets there, which is why it survived this long.
"""
import jax
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.ancillary.clearing import make_clearing
from powermarketjax.envs.day_ahead import kkt_lowrank
from tests.envs.ancillary.test_lowrank_route_l0 import THETA, VOLR, _rows


@pytest.fixture(scope="module", autouse=True)
def x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


@pytest.fixture(scope="module")
def system(x64):
    """The original draft system: a third of the capacity-sharing rows bind."""
    gb = load_case("29gb")
    _clear, spec = make_clearing(gb, THETA, VOLR, n_segments=1, cap_scale=0.6,
                                 ramp_scale=1.0, period_hours=0.5,
                                 monitored_lines=[0, 1, 2])
    M, S, CS, RD, a = _rows(spec)
    m, row0, n_u = spec["m"], spec["row0"], spec["n_units"]
    D = np.full(m, 1e-14)
    D[row0["line_up"]] = 3e14
    D[row0["rd"]] = 5e12
    D[row0["cs"]: row0["cs"] + n_u] = 1e-12
    D[row0["cs"] + np.arange(0, n_u, 3)] = 1e14          # a third of them bind
    rhs = np.random.default_rng(7).standard_normal(spec["n"] + 1)
    return spec, (M, S, CS, RD, a), D, rhs


def test_a_partial_free_set_is_refused(system):
    """The load-bearing one: `0 < n_fu < n_units` must not build at all."""
    spec, rows, _D, _rhs = system
    n_u, n_b, P = spec["n_units"], spec["n_buses"], spec["n_prod"]
    with pytest.raises(ValueError, match="either no unit or every unit"):
        kkt_lowrank.make_kkt_ancillary(spec, *rows, n_free=(n_u // 2, n_b + P))


@pytest.mark.parametrize("n_fs_of", ["zero", "all"])
def test_the_allowed_configurations_still_build(system, n_fs_of):
    """The refusal must not take the configurations that are allowed.

    Without this, a guard that refused every non-zero `n_free[0]` would pass the
    test above and quietly disable the route the market actually uses
    (`(n_units, n_buses + n_prod)` is the default on `case813nem`).

    **This asserts that they build, not that they are accurate.**  `(0, *)` is
    allowed in the sense of *not refused*; on this deliberately pathological `D`
    it is not good -- measured here, `(0, 0)` and `(0, n_b + P)` both come back
    230/230 non-finite, which is the injected figure and is the behaviour of
    the plain route on a system built to break it.  Asserting finiteness there
    would be asserting something false about a configuration this guard is not
    about.
    """
    spec, rows, D, rhs = system
    n_u, n_b, P = spec["n_units"], spec["n_buses"], spec["n_prod"]
    n_free = (n_u, 0 if n_fs_of == "zero" else n_b + P)
    factor, apply = kkt_lowrank.make_kkt_ancillary(spec, *rows, n_free=n_free)
    import jax.numpy as jnp
    z = np.asarray(jax.jit(apply)(jax.jit(factor)(jnp.asarray(D), jnp.asarray(1e-8)),
                                  jnp.asarray(rhs)))
    #: The fully free set is the only tier this route zeroes cleanly, so
    #: finiteness **can** be asserted here -- measured, both `n_fs` give 0/230
    #: non-finite.
    assert np.isfinite(z).all(), (
        f"n_free={n_free} built but produced {int((~np.isfinite(z)).sum())} "
        f"non-finite components")


def test_an_empty_free_set_is_not_refused(system):
    """`(0, 0)` stays buildable: the guard is about the middle, not about zero.

    No finiteness claim -- see the note above; on this system it is 230/230
    non-finite and that is the plain route's own behaviour, not this guard's
    business.
    """
    spec, rows, _D, _rhs = system
    kkt_lowrank.make_kkt_ancillary(spec, *rows, n_free=(0, 0))
