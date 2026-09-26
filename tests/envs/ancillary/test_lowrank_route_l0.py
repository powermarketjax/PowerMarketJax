"""L0: the low-rank KKT route of the joint energy-reserve clearing
(2026-09-17) -- the switch, the stamps, the matrix-free row products, and the
JAX constraints on the route the environment runs.

The route is `kkt_lowrank.make_kkt_ancillary`: the same Newton system as the
dense factorisation, solved through the per-unit direct sum plus a low-rank
border (lines, requirement rows, balance) with every column free under the
local-border arrowhead.  What is checked here needs no case of any size:

1. ``kkt="auto"`` is dense on the default rows and on an index array naming
   every line, low-rank on a proper subset; ``"dense"`` / ``"lowrank"``
   force one; the stamps name the sizing and the block solver in the
   real-time market's form.
2. `make_ops_ancillary`'s ``(Gx, GTy)`` are the dense products to round-off
   on random vectors, against ``G`` rebuilt from the operator's own layout.
3. The environment on the low-rank route runs under ``jit(vmap(scan))``,
   compiles once, and returns finite rewards; the route's source calls no
   host callback.
"""
import inspect
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.ancillary.clearing import default_lowrank_free, make_clearing
from powermarketjax.envs.ancillary.env import AncillaryParams, make_ancillary_env
from powermarketjax.envs.day_ahead import kkt_lowrank
from powermarketjax.envs.day_ahead.clearing import rated_lines
from powermarketjax.solvers import ipm

THETA = (1.0 / 6.0, 0.5)
VOLR, PI_SCALE, BETA = 250.0, 50.0, (0.05, 0.05)
DELTA = 0.5


@pytest.fixture(scope="module", autouse=True)
def x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


@pytest.fixture(scope="module")
def gb():
    return load_case("29gb")


def _build(case, **kw):
    return make_clearing(case, THETA, VOLR, n_segments=1, cap_scale=0.6, ramp_scale=1.0,
                         period_hours=DELTA, **kw)


def test_auto_is_dense_on_every_line_and_lowrank_on_a_proper_subset(gb):
    n_lines = int(np.asarray(gb.PTDF).shape[0])
    n_u, n_b = len(np.asarray(gb.unit_p_min)), int(gb.n_nodes)
    _, dflt = _build(gb)
    assert dflt["kkt_route"] == "dense" and dflt["lowrank_free"] == (0, 0)
    _, every = _build(gb, monitored_lines=np.arange(n_lines))
    assert every["kkt_route"] == "dense"
    assert np.array_equal(rated_lines(gb), np.arange(n_lines)), "29gb rates every line"
    _, sub = _build(gb, monitored_lines=[0, 1, 2])
    assert sub["kkt_route"] == f"lowrank+free({n_u},{n_b + 2})+lu:arrow"
    assert sub["lowrank_free"] == (n_u, n_b + 2) == default_lowrank_free(gb, [0, 1, 2], 2)
    assert sub["lu_batching"] == "arrow"
    _, forced_dense = _build(gb, monitored_lines=[0, 1, 2], kkt="dense")
    assert forced_dense["kkt_route"] == "dense" and forced_dense["lowrank_free"] == (0, 0)
    _, plain = _build(gb, monitored_lines=[0, 1, 2], kkt="lowrank", lowrank_free=(0, 0))
    assert plain["kkt_route"] == "lowrank"
    #: An explicit sizing is still honoured, **but its unit half can only be 0
    #: or n_units**.  The local-border branch zeroes, by selection, the
    #: `H0^-1 U` rows of the units inside the free set; a unit outside the set
    #: whose capacity-sharing row binds is not in that zeroing, its block is
    #: singular to round-off, the whole Newton direction turns non-finite, and
    #: `ipm`'s `ok` swallows it (the `local` guard in `kkt_lowrank.py`; four
    #: cases in `tests/envs/ancillary/test_partial_free_refused_l0.py`).  This
    #: line used to read `(3, 5)`, i.e. 3 of the 66 units in the set -- exactly
    #: the refused tier.  Changed to the equally explicit `(n_u, 5)` with the
    #: unit half full, so "an explicit sizing is honoured" is still checked, and
    #: the next line adds the refusal itself.
    _, sized = _build(gb, monitored_lines=[0, 1, 2], lowrank_free=(n_u, 5))
    assert sized["kkt_route"] == f"lowrank+free({n_u},5)+lu:arrow"
    with pytest.raises(ValueError, match="either no unit or every unit"):
        _build(gb, monitored_lines=[0, 1, 2], lowrank_free=(3, 5))
    assert default_lowrank_free(gb, None, 2) == (0, 0) == default_lowrank_free(gb, np.arange(n_lines), 2)
    with pytest.raises(ValueError, match="kkt must be one of"):
        _build(gb, kkt="sparse")


def _rows(spec):
    """``(M, S, CS, RD, a)`` rebuilt from the operator's own layout, the way
    `make_clearing` builds them."""
    K, P, n_u, n_b = spec["K"], spec["n_prod"], spec["n_units"], spec["n_buses"]
    col0, PTDF, unit_bus = spec["col0"], spec["PTDF"], spec["unit_bus"]
    n_g, n_r, n_l = n_u * K, n_u * P, PTDF.shape[0]
    seg_bus = np.zeros((n_b, n_g)); seg_bus[np.repeat(unit_bus, K), np.arange(n_g)] = 1.0
    seg_of_unit = np.zeros((n_u, n_g)); seg_of_unit[np.repeat(np.arange(n_u), K), np.arange(n_g)] = 1.0
    res_of_unit = np.zeros((n_u, n_r)); res_of_unit[np.repeat(np.arange(n_u), P), np.arange(n_r)] = 1.0
    prod_of_res = np.zeros((P, n_r)); prod_of_res[np.tile(np.arange(P), n_u), np.arange(n_r)] = 1.0
    z = lambda r, c: np.zeros((r, c))
    M = np.hstack([PTDF @ seg_bus, z(n_l, n_r), PTDF, z(n_l, P)])
    S = np.hstack([seg_of_unit, z(n_u, n_r + n_b + P)])
    CS = np.hstack([seg_of_unit, res_of_unit, z(n_u, n_b + P)])
    RD = np.hstack([z(P, n_g), -prod_of_res, z(P, n_b), -np.eye(P)])
    a = np.zeros(spec["n"]); a[col0["g"]: col0["g"] + n_g] = 1.0; a[col0["s"]: col0["s"] + n_b] = 1.0
    return M, S, CS, RD, a


def test_row_products_match_the_dense_matrix(gb):
    _, spec = _build(gb, monitored_lines=[0, 1, 2])
    M, S, CS, RD, _ = _rows(spec)
    n = spec["n"]
    G = np.vstack([M, -M, S, -S, CS, RD, np.eye(n), -np.eye(n)])
    assert G.shape == (spec["m"], n)
    Gx, GTy = kkt_lowrank.make_ops_ancillary(spec, M)
    rng = np.random.default_rng(0)
    for _ in range(3):
        v, y = rng.standard_normal(n), rng.standard_normal(spec["m"])
        np.testing.assert_allclose(np.asarray(Gx(jnp.asarray(v))), G @ v, rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(GTy(jnp.asarray(y))), G.T @ y, rtol=0, atol=1e-12)
    # a wrong requirement row is refused by the constructor's own check
    bad = RD.copy(); bad[0, 0] = 1.0
    with pytest.raises(AssertionError, match="RD must touch"):
        kkt_lowrank.make_kkt_ancillary(spec, M, S, CS, bad, _rows(spec)[4])


def test_lowrank_solves_the_same_system_as_the_dense_factorisation(gb):
    """One Newton system with random positive weights: the low-rank
    ``(factor, apply)`` against the dense ``[[G'DG + reg I, A'], [A, 0]]``."""
    _, spec = _build(gb, monitored_lines=[0, 1, 2])
    M, S, CS, RD, a = _rows(spec)
    n, m = spec["n"], spec["m"]
    G = np.vstack([M, -M, S, -S, CS, RD, np.eye(n), -np.eye(n)])
    rng = np.random.default_rng(1)
    D = 10.0 ** rng.uniform(-6, 6, m)
    reg = 1e-8
    K = np.block([[(G * D[:, None]).T @ G + reg * np.eye(n), a[:, None]], [a[None, :], np.zeros((1, 1))]])
    rhs = rng.standard_normal(n + 1)
    f_dn, a_dn = ipm._dense_kkt(n, jnp.asarray(G), jnp.asarray(a[None, :]))
    z = np.asarray(jax.jit(a_dn)(jax.jit(f_dn)(jnp.asarray(D), jnp.asarray(reg)), jnp.asarray(rhs)))
    rel_dense = np.linalg.norm(K @ z - rhs) / np.linalg.norm(rhs)
    for n_free in (spec["lowrank_free"], (0, 0)):
        factor, apply = kkt_lowrank.make_kkt_ancillary(spec, M, S, CS, RD, a, n_free=n_free)
        z = np.asarray(jax.jit(apply)(jax.jit(factor)(jnp.asarray(D), jnp.asarray(reg)), jnp.asarray(rhs)))
        rel = np.linalg.norm(K @ z - rhs) / np.linalg.norm(rhs)
        # the reading adopted for this route: at most ten times the dense route's
        # residual (measured 2026-09-17, CPU, 4 cores, this seed: 1.1e-9
        # against the dense route's own; the absolute bound is a backstop)
        assert rel < 1e-7 and rel <= 10.0 * rel_dense + 1e-12, (n_free, rel, rel_dense)


def test_no_host_callback_in_the_route():
    src = inspect.getsource(kkt_lowrank)
    assert "pure_callback" not in src and "host_callback" not in src


def test_environment_runs_under_jit_vmap_scan_and_compiles_once(gb):
    reset, step, _ar, spec = make_ancillary_env(gb, THETA, VOLR, BETA, PI_SCALE, n_segments=1,
                                                cap_scale=0.6, ramp_scale=1.0, period_hours=DELTA,
                                                kind="markup", markup_max=2.0,
                                                monitored_lines=[0, 1, 2])
    cs = spec["clearing_spec"]
    assert cs["kkt_route"].startswith("lowrank+free(") and cs["kkt_route"].endswith("+lu:arrow")
    n_units, n_buses = int(cs["n_units"]), int(cs["n_buses"])
    pmin = jnp.asarray(np.asarray(gb.unit_p_min, np.float64))
    pmax = jnp.asarray(np.asarray(gb.unit_p_max, np.float64))
    n_periods, n_lanes, horizon = 4, 2, 3
    u = jnp.ones((n_periods, n_units))
    demand = jnp.full((n_periods,), 0.6 * float(pmax.sum()))
    params = AncillaryParams(demand=demand, forecast=demand, commitment=u,
                             q_da=(pmin + 0.5 * (pmax - pmin)) * u,
                             lmp_da=jnp.full((n_periods, n_buses), 40.0),
                             learner_mask=jnp.ones((n_units,), bool), episode_len=n_periods)
    baseline = spec["baseline_action"]

    def rollout(key, p):
        _obs, state = reset(key, p)

        def body(carry, _):
            key, state = carry
            key, sub = jax.random.split(key)
            _o, state, r, _c, _d, info = step(sub, state, baseline, p)
            return (key, state), (r, info["mu"])

        _, (r, mu) = jax.lax.scan(body, (key, state), None, length=horizon)
        return r, mu

    f = jax.jit(jax.vmap(rollout, in_axes=(0, None)))
    r, mu = f(jax.random.split(jax.random.PRNGKey(0), n_lanes), params)
    assert f._cache_size() == 1
    r2, _ = f(jax.random.split(jax.random.PRNGKey(1), n_lanes), params)
    assert f._cache_size() == 1
    assert np.isfinite(np.asarray(r)).all() and np.isfinite(np.asarray(r2)).all()
    assert np.asarray(mu).shape == (n_lanes, horizon)
