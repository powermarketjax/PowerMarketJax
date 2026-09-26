"""L0: `constrained_baseline.py` refuses a seed range that names no seed.

**The failure this exists for.**  `--seeds` is the exclusive end of the range
(`range(--seed-start, --seeds)`), not a count, so `--seed-start 3 --seeds 2`
looped over nothing and exited 0 with an empty result, which reads as a run that
worked.  Every non-empty range behaves as before.
"""
import os
import pathlib
import subprocess
import sys

import pytest

# The driver reaches distrax and rlax through preliminary_reference at import time;
# both come with the rl extra, which the base install does not include.
pytest.importorskip("distrax", reason="preliminary_reference builds its policy on distrax")
pytest.importorskip("rlax", reason="preliminary_reference imports rlax")

REPO = pathlib.Path(__file__).resolve().parents[2]
P2P = REPO / "tools" / "p2p_experiment"


def _run(*argv):
    return subprocess.run(
        [sys.executable, str(P2P / "constrained_baseline.py"), *argv],
        capture_output=True, text=True, timeout=300,
        env={**os.environ, "JAX_PLATFORMS": "cpu",
             "PYTHONPATH": f"{REPO}{os.pathsep}{P2P}"})


def test_an_empty_seed_range_is_refused():
    out = _run("--seed-start", "3", "--seeds", "2", "--iterations", "1")
    text = out.stdout + out.stderr
    assert out.returncode != 0, f"an empty seed range exited 0; tail {text[-300:]!r}"
    assert "names no seed" in text, text[-300:]
