"""L0 for `ipm.make_solver(freeze_mu=...)` under `vmap`: the gate is armed and
refuses per lane, so a lane's result cannot depend on which other lanes share
its batch.  The docstring claimed this ("the `vmap` behaviour is
unchanged") with no test under `tests/solvers/`; it was measured
(2026-09-17, CPU 4 cores) and this is that measurement as a test.

Exercised through the market-03 operator on two `case813nem` rated-rows cells
of `tests/fixtures/ge05_03_cells_813nem.npz`, because those are the solves
where the gate actually refuses steps (the plain loop drifts to a dual
residual of 1e0 on cell 0_0); a toy LP that never arms would pass any
batching rule vacuously.  Two-sided: the gate is shown to act (frozen and
plain single-lane results differ), and the lane is shown to be the same to
the bit across two batch compositions.  Single-lane and batched kernels are
NOT asserted equal: they differ in rounding and the gate is discontinuous
(measured |dlmp| 2.4e-4 $/MWh on cell 0_0), which is a property of batched
LU, not of the gate."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.ancillary.clearing import make_clearing
from powermarketjax.envs.day_ahead.clearing import rated_lines

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "ge05_03_cells_813nem.npz"
THETA, VOLR, DELTA, FREEZE_MU = (1.0 / 6.0, 0.5), 147.0, 0.5, 1e-9


@pytest.fixture(scope="module")
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _cell(name):
    fx = np.load(FIXTURE, allow_pickle=True)
    return tuple(jnp.asarray(np.asarray(fx[f"{name}_{k}"], np.float64))
                 for k in ("offer", "offer_res", "u", "demand", "d_res", "p_prev"))


def _stack(*cells):
    return tuple(jnp.stack([c[i] for c in cells]) for i in range(6))


def _lane(out, i):
    return {k: np.asarray(v)[i] for k, v in out.items()}


#: The dual-residual bound that separates the plain loop from the frozen one,
#: asserted only on the calibration setup: `CALIBRATED_CORES` visible cores.
#: The separation moves with the number of XLA CPU threads, and over the nine
#: core counts measured (2026-09-22/23, CPU, x64, `case813nem` rated rows,
#: dense route, cells 0_0 and 0_24; dual residual of cell 0_0) the ranges
#: overlap, so no single bound holds on every count:
#:
#:   cores   batched frozen (lane 0)   single-lane frozen   single-lane plain
#:     1         3.0e-08                  2.5e-05              1.8e+00
#:     2         6.8e-08                  3.6e-04              4.9e-03
#:     4         2.7e-07                  1.3e-04              1.0e+00
#:     6         1.7e-09                  7.8e-05              1.0e-02
#:     8         4.1e-04                  8.3e-05              3.4e-02
#:    12         9.2e-07                  8.9e-07              2.4e-05
#:    16         2.4e-03                  7.4e-05              2.8e+00
#:    24         3.7e-05                  1.7e-08              1.4e-01
#:    32         2.9e-07                  2.8e-04              8.1e-01
#:
#: Each count reproduces to the bit from run to run; lane invariance across
#: batch compositions held on all nine.  The thread pool is fixed when JAX
#: starts, so the test cannot pin it; it asks how many cores the process sees.
MAGNITUDE_BOUND = 1e-3
CALIBRATED_CORES = 4


@pytest.fixture(scope="module")
def solved(x64):
    case = load_case("813nem")
    # the dense route, what the gate was measured on: the plain loop's
    # walk-off this test rests on is the dense factorisation's; the
    # low-rank route a proper subset takes by default since 2026-09-17 ends the
    # same 60 steps at 2e-10 (`tests/envs/ancillary/test_lowrank_cells_l2.py`)
    kw = dict(n_segments=1, cap_scale=1.0, ramp_scale=1.0, period_hours=DELTA,
              monitored_lines=rated_lines(case), kkt="dense")
    frozen, spec = make_clearing(case, THETA, VOLR, freeze_mu=FREEZE_MU, **kw)
    plain, _ = make_clearing(case, THETA, VOLR, **kw)
    a, b = _cell("0_0"), _cell("0_24")
    f_a = {k: np.asarray(v) for k, v in jax.jit(frozen)(*a).items()}
    p_a = {k: np.asarray(v) for k, v in jax.jit(plain)(*a).items()}
    vf = jax.jit(jax.vmap(frozen))
    ab, ba = vf(*_stack(a, b)), vf(*_stack(b, a))
    return dict(spec=spec, f_a=f_a, p_a=p_a, ab=ab, ba=ba)


def test_the_gate_refuses_per_lane_not_per_batch(solved):
    assert solved["spec"]["freeze_mu"] == FREEZE_MU
    # the gate acts on cell 0_0: frozen and plain single-lane results differ
    assert not np.array_equal(solved["f_a"]["lmp"], solved["p_a"]["lmp"])
    # the same lane in two batch compositions is the same to the bit
    ab, ba = solved["ab"], solved["ba"]
    for key in ab:
        np.testing.assert_array_equal(_lane(ab, 0)[key], _lane(ba, 1)[key], err_msg=f"lane a, {key}")
        np.testing.assert_array_equal(_lane(ab, 1)[key], _lane(ba, 0)[key], err_msg=f"lane b, {key}")


def _magnitudes_separate(solved, bound):
    """The plain loop's final residual above `bound`, the frozen one below it,
    single-lane and inside the batch."""
    f = float(solved["f_a"]["dual_residual"])
    p = float(solved["p_a"]["dual_residual"])
    b = float(np.asarray(solved["ab"]["dual_residual"])[0])
    return f < bound < p and b < bound


def test_the_gate_holds_the_residual_three_decades_below_the_plain_loop(solved):
    """Two magnitude checks, on the calibration setup only (see `MAGNITUDE_BOUND`)."""
    import os
    cores = len(os.sched_getaffinity(0))
    if cores != CALIBRATED_CORES:
        pytest.skip(
            f"the magnitude separation between the two solves moves with the number "
            f"of XLA CPU threads and the ranges measured on 9 core counts overlap, so "
            f"no single bound holds on all of them; asserted only on the calibration "
            f"setup of {CALIBRATED_CORES} cores (this process sees {cores})")
    assert _magnitudes_separate(solved, MAGNITUDE_BOUND), (
        float(solved["f_a"]["dual_residual"]), float(solved["p_a"]["dual_residual"]),
        float(np.asarray(solved["ab"]["dual_residual"])[0]))
    # and the check bites here: a bound below the frozen residual must fail
    assert not _magnitudes_separate(solved, 1e-9)
