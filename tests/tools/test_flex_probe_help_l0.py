"""L0: `monitor_probe.py --help` and `network_census.py --help` print and exit.

**The failure this exists for.**  Neither script parsed its arguments, so
`--help` was ignored and the whole probe or census ran and then overwrote its
committed JSON under `docs/figures/flex-concentration/`.  Each now builds an
argument parser with no options; run without arguments it does what it did.

Run in an empty working directory with a short timeout: the scripts write
relative to the working directory, so a regression here cannot touch the
committed files, and it shows up as a timeout, a non-zero exit, or a `docs/`
directory appearing.
"""
import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
FLEX = REPO / "tools" / "flex_experiment"


@pytest.mark.parametrize("script", ["monitor_probe.py", "network_census.py"])
def test_help_prints_and_runs_nothing(tmp_path, script):
    out = subprocess.run([sys.executable, str(FLEX / script), "--help"],
                         cwd=tmp_path, capture_output=True, text=True, timeout=120,
                         env={**os.environ, "JAX_PLATFORMS": "cpu",
                              "PYTHONPATH": f"{REPO}{os.pathsep}{FLEX}"})
    assert out.returncode == 0, (out.returncode, (out.stdout + out.stderr)[-400:])
    assert out.stdout.startswith("usage: "), out.stdout[:200]
    assert not (tmp_path / "docs").exists(), "--help wrote a product"
