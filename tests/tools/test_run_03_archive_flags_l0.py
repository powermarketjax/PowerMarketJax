"""L0: `run_rl_03.py` refuses `--init-params A --eval-only B`, as `run_rl_02.py` does.

**The failure this exists for.**  The driver loads `args.init_params or
args.eval_only` -- A -- while its `--eval-only` branch prints B, so a run given
both evaluated one archive and reported another.  Refusing the pair changes no
legal run: each flag alone behaves exactly as before.

A subprocess with a deliberately unusable `--position`: the refusal has to bite
at argument validation, before the position is read and before JAX is imported.
"""
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
DRIVER = REPO / "tools" / "benchmark" / "run_rl_03.py"


def _run(*extra):
    return subprocess.run(
        [sys.executable, str(DRIVER), "--position", "/dev/null",
         "--out-dir", "/nonexistent/never-written", *extra],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "JAX_PLATFORMS": "cpu"})


def test_both_archive_flags_are_refused_before_any_work():
    out = _run("--init-params", "a.npz", "--eval-only", "b.npz")
    text = out.stdout + out.stderr
    assert out.returncode != 0, "giving both archive flags was accepted"
    assert "both name an archive" in text, (
        f"exited {out.returncode} but not for the reason this test is about; "
        f"tail: {text[-400:]!r}")
    assert "package:" not in text, "the refusal came after JAX was imported"


def test_one_archive_flag_alone_is_not_refused_by_this_guard():
    """The negative control: `--eval-only` alone gets past argument validation
    (it then fails on the unusable position, which is not this guard)."""
    text = (lambda o: o.stdout + o.stderr)(_run("--eval-only", "b.npz"))
    assert "both name an archive" not in text
