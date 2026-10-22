# Written for this repository on 2026-10-22 -- no upstream counterpart.
"""The sweep's fixed point must satisfy branch power balance, not just settle."""

import numpy as np
import jax.numpy as jnp
import pytest

from powermarketjax.case import create_case33bw
from powermarketjax.physics.bfs_power_flow import (
    bfs_power_flow, build_radial_topology, prepare_bfs)

# pandapower 3.5.6 Newton-Raphson on the impedances `prepare_bfs` stores for
# case33bw, nominal load, 100 MVA base; CPU, 2026-10-22.
_PANDAPOWER_V_MIN = 0.913090453
_PANDAPOWER_LOSS_PU = 0.002026772


def test_two_bus_matches_analytic_high_voltage_solution():
    # Two-bus case reported by Massimiliano Brighindi in
    # https://github.com/powermarketjax/PowerMarketJax/issues/1
    topo = build_radial_topology(2, np.array([0]), np.array([1]),
                                 np.array([0.1]), np.array([0.0]))
    r = bfs_power_flow(topo, jnp.array([0.0, 2.0]), jnp.zeros(2))
    rr = float(topo.r_pu[0])
    p_send = float(r.p_branch[0])
    assert bool(r.converged)
    np.testing.assert_allclose(float(r.v_mag[1]), (1 + np.sqrt(1 - 8 * rr)) / 2,
                               atol=1e-5)
    assert abs(p_send - 2.0 - rr * p_send ** 2) < 1e-5


def test_case33_matches_independent_solver():
    case = create_case33bw()
    r = bfs_power_flow(prepare_bfs(case), case.node_pd / case.base_mva,
                       case.node_qd / case.base_mva)
    assert bool(r.converged)
    np.testing.assert_allclose(float(r.v_mag.min()), _PANDAPOWER_V_MIN, atol=1e-5)
    np.testing.assert_allclose(float(r.p_loss.sum()), _PANDAPOWER_LOSS_PU, atol=1e-6)


@pytest.mark.parametrize("scale", [1.0, 2.0, 3.0])
def test_case33_loss_accounting_closes(scale):
    case = create_case33bw()
    topo = prepare_bfs(case)
    p = scale * np.asarray(case.node_pd) / case.base_mva
    r = bfs_power_flow(topo, jnp.asarray(p),
                       jnp.asarray(scale * np.asarray(case.node_qd) / case.base_mva))
    slack = int(case.slack_bus_idx)
    root = np.asarray(topo.sending_node_idx) == slack
    assert bool(r.converged)
    np.testing.assert_allclose(
        float(np.asarray(r.p_branch)[root].sum()) - (p.sum() - p[slack]),
        float(r.p_loss.sum()), atol=1e-5)


@pytest.mark.parametrize("load", [2.6, 5.6, 20.0])
def test_two_bus_beyond_collapse_never_converges(load):
    topo = build_radial_topology(2, np.array([0]), np.array([1]),
                                 np.array([0.1]), np.array([0.0]))
    r = bfs_power_flow(topo, jnp.array([0.0, load]), jnp.zeros(2))
    assert not bool(r.converged)
    assert np.isfinite(np.asarray(r.v_mag)).all()
