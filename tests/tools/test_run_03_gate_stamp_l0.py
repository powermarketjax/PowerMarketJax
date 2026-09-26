"""L0: the market-03 drivers stamp the gate they actually ran and read the
day-level dual check one way.

Found on review (2026-09-17): `run_rl_03.py` stamped
``mu_tol`` from the module constant and nothing for ``dual_res_tol``, while
`converged` (and so `unconv_periods`, `unconverged_frac`) had become the
two-part gate -- an archive could not say which gate produced its counts;
and the day products' `dual_ok` was ``<`` here against ``<=`` in
`run_rl_02.py` under the same name.  `gate_stamp` reads both tolerances off
the env's ``spec``, `dual_ok` is the ``<=`` of market 02, and the
call sites are anchored by source count so a driver that drops one goes red
rather than silently stamping a constant.  On the tree before the
fix the import itself fails."""
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
BENCH = REPO / "tools" / "benchmark"
sys.path.insert(0, str(BENCH))
from run_eval_03 import dual_ok, gate_stamp  # noqa: E402

SRC_RL = (BENCH / "run_rl_03.py").read_text()
SRC_EVAL = (BENCH / "run_eval_03.py").read_text()


def test_gate_stamp_reads_the_spec_not_the_module_constants():
    from powermarketjax.envs.ancillary.env import DUAL_RES_TOL, MU_TOL
    spec = {"mu_tol": 3.0e-9, "dual_res_tol": 5.0e-5, "m": 856}
    assert (3.0e-9, 5.0e-5) != (MU_TOL, DUAL_RES_TOL)   # the injection is distinguishable
    assert gate_stamp(spec) == {"mu_tol": 3.0e-9, "dual_res_tol": 5.0e-5}
    assert gate_stamp({"mu_tol": MU_TOL, "dual_res_tol": DUAL_RES_TOL}) == {
        "mu_tol": MU_TOL, "dual_res_tol": DUAL_RES_TOL}


def test_dual_ok_is_the_closed_boundary_of_market_02():
    assert dual_ok([1e-6, 1e-4], 1e-4) is True          # `<=`, as run_rl_02.py
    assert dual_ok([1e-6, 1e-4 * (1 + 1e-12)], 1e-4) is False
    assert dual_ok([0.0], 0.0) is True


def test_call_sites_are_anchored():
    """Both drivers stamp the gate off the spec everywhere the gate is
    written, and neither reads the module constant for it any more."""
    assert SRC_RL.count("gate = gate_stamp(spec)") == 1
    assert SRC_RL.count("**gate") == 2                    # run point + curve scenario
    assert SRC_RL.count('dual_ok=dual_ok(drs, gate["dual_res_tol"])') == 1
    assert SRC_RL.count('dual_res_tol=gate["dual_res_tol"]') == 1
    assert SRC_RL.count("anc_env.MU_TOL") == 0 and SRC_RL.count("anc_env.DUAL_RES_TOL") == 0
    assert SRC_EVAL.count("gate = gate_stamp(spec)") == 1
    assert SRC_EVAL.count("**gate,") == 1                  # run point
    assert SRC_EVAL.count('dual_ok=dual_ok(drs, gate["dual_res_tol"])') == 1
    assert SRC_EVAL.count('dual_res_tol=gate["dual_res_tol"]') == 1
    assert "import (DUAL_RES_TOL," not in SRC_EVAL
    # `unconverged` stays the mu-only report count in both, and says so
    assert SRC_RL.count("unconverged=int(count_cells(np.asarray(mus), 1e-9))") == 1
    assert SRC_EVAL.count("unconverged=int(sum(m > 1e-9 for m in mus))") == 1
    assert SRC_RL.count("the mu half ALONE") == 1 and SRC_EVAL.count("the mu half ALONE") == 1
