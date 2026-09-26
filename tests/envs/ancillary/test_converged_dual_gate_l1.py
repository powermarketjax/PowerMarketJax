"""`converged` is a two-part gate: ``mu < mu_tol`` AND
``dual_residual < dual_res_tol``.  Nothing in `tests/envs/ancillary` asserted
the second half -- an injection reverting the gate to ``mu`` alone left the
package the colour it was.  This L1 is the negative control that bites: it flips the dual
half alone and reads the flag, two-sided, on a period whose ``mu`` half passes
(so a gate that ignored the dual residual would fail the strict side and a
gate that ignored ``mu`` would still pass the loose side vacuously -- the
``mu`` half is asserted separately from ``info``).  The reward must not move
(`test_env_l0.test_the_reward_does_not_depend_on_convergence`, same rule)."""
import jax
import jax.numpy as jnp
import numpy as np

from powermarketjax.envs.ancillary.env import DUAL_RES_TOL, MU_TOL, make_ancillary_env
from tests.envs.ancillary.test_env_l0 import (BETA, CASE, DELTA, EPISODE, PI_SCALE,
                                              THETA, VOLR, AncillaryParams, DAYS,
                                              FIXTURE, _action, load_case,
                                              load_gb_demand, x64)  # noqa: F401


def _build(**gate):
    """The operating point of `test_env_l0._build`, with the gate's tolerances
    as keywords (that helper takes `mu_tol` only)."""
    case = load_case(CASE)
    fx = np.load(FIXTURE, allow_pickle=True)
    idx = fx["day_index"]
    _, actual, _ = load_gb_demand()
    u = np.repeat(fx["commitment"][:DAYS], 2, axis=2).transpose(0, 2, 1).reshape(-1, 66)
    demand = np.repeat(actual[idx[:DAYS]], 2, axis=1).reshape(-1)
    pmin = np.asarray(case.unit_p_min, np.float64)
    pmax = np.asarray(case.unit_p_max, np.float64)
    room = np.maximum(((pmax - pmin) * u).sum(1, keepdims=True), 1.0)
    frac = np.clip((demand[:, None] - (pmin * u).sum(1, keepdims=True)) / room, 0.0, 1.0)
    q_da = (pmin + frac * (pmax - pmin)) * u
    env = make_ancillary_env(case, THETA, VOLR, BETA, PI_SCALE, n_segments=1,
                             cap_scale=0.6, ramp_scale=1.0, period_hours=DELTA,
                             kind="markup", markup_max=2.0, **gate)
    params = AncillaryParams(
        demand=jnp.asarray(demand), forecast=jnp.asarray(demand * 1.02),
        commitment=jnp.asarray(u), q_da=jnp.asarray(q_da),
        lmp_da=jnp.full((len(demand), int(case.n_nodes)), 40.0),
        learner_mask=jnp.ones(66, bool), episode_len=EPISODE)
    return env, params


def _step(env, params):
    reset, step, _, spec = env
    _, state = jax.jit(reset)(jax.random.PRNGKey(0), params)
    out = jax.jit(step)(jax.random.PRNGKey(1), state, _action(spread=0.2), params)
    return spec, out


def test_converged_reads_the_dual_residual_as_well_as_mu(x64):
    """Same solve, the dual tolerance alone flipped: the flag flips, the reward
    does not; and the default gate is the conjunction of the two stamped
    tolerances, recomputed here from the same `info`."""
    strict_spec, strict = _step(*_build(dual_res_tol=0.0))
    loose_spec, loose = _step(*_build(dual_res_tol=float("inf")))
    assert strict_spec["dual_res_tol"] == 0.0 and loose_spec["dual_res_tol"] == float("inf")
    info_s, info_l = strict[5], loose[5]
    # the mu half passes on this period, so the strict side is not vacuous
    assert float(info_s["mu"]) < MU_TOL and float(info_l["mu"]) < MU_TOL
    assert float(info_s["dual_residual"]) > 0.0
    assert not bool(info_s["converged"]), "a dual tolerance of 0 must fail the gate"
    assert bool(info_l["converged"]), "an infinite dual tolerance must pass it"
    np.testing.assert_array_equal(np.asarray(strict[2]), np.asarray(loose[2]))   # reward
    np.testing.assert_array_equal(np.asarray(strict[3]), np.asarray(loose[3]))   # costs

    spec, out = _step(*_build())
    info = out[5]
    assert spec["dual_res_tol"] == DUAL_RES_TOL and spec["mu_tol"] == MU_TOL
    want = (float(info["mu"]) < MU_TOL) and (float(info["dual_residual"]) < DUAL_RES_TOL)
    assert bool(info["converged"]) == want
