"""L0: `equilibrium_scan_01.py --summarise-saved P` works on its own.

**The failure this exists for.**  `--checkpoint` and `--out` were declared
`required=True`, so argparse exited 2 before `--summarise-saved` -- the mode that
only re-reads an existing product and needs neither -- could run.  A sweep still
needs both, and is still refused without them.

A subprocess on a small product with the keys `summarise_saved` reads; the
summary is the same function either way, so only reachability is tested.
"""
import os
import pathlib
import subprocess
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
DRIVER = REPO / "tools" / "benchmark" / "equilibrium_scan_01.py"


def _run(*argv):
    return subprocess.run([sys.executable, str(DRIVER), *argv], capture_output=True,
                          text=True, timeout=300,
                          env={**os.environ, "JAX_PLATFORMS": "cpu"})


def test_summarise_saved_runs_without_checkpoint_and_out(tmp_path):
    grid = (1.0, 1.6, 2.0)
    unit = np.repeat([34, 35], len(grid))
    alpha = np.tile(grid, 2)
    delta = np.array([0.0, 5.0, 20.0, 0.0, 0.0, 0.0])
    p = tmp_path / "sweep.npz"
    np.savez(p, unit=unit, alpha=alpha, delta=delta,
             reference_units=np.array([34, 35]),
             reference_profit=np.array([100.0, 50.0]))
    out = _run("--summarise-saved", str(p))
    text = out.stdout + out.stderr
    assert out.returncode == 0, f"rc {out.returncode}; tail {text[-400:]!r}"
    assert "(1) units with a strong incentive to deviate: 1 of 2" in text, text[-400:]


def test_a_sweep_without_checkpoint_is_still_refused():
    out = _run("--units", "34")
    assert out.returncode == 2
    assert "--checkpoint and --out are required" in out.stderr, out.stderr[-300:]
