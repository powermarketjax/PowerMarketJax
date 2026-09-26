"""L0: `run_rl_02.py --eval-only` exists, works for a SAC archive, and cannot
be given together with `--init-params`.

**The failure this exists for** (2026-09-19): the command
lines prepared for the SAC checkpoint comparison were copied from the
IPPO ones without checking that the flags they used hold for SAC.  Market 02
had only `--init-params`, and that flag *explicitly refused* `--algo sac`, so
all four steps exited inside 40 seconds and wrote `days=0`.  Nothing in the
four command lines was wrong about IPPO; what was missing was the question
"does this flag hold for the learner I am pointing it at".

**Why the unwrap is keyed on the archive's own `algo` stamp** rather than on
`--algo`.  The driver wraps on write: IPPO's flax tree goes in as it is, and
SAC's `{actor, q1, q2, q1_target, q2_target, log_alpha}` goes in under one
extra `params` root (`_archive_tree`), so the inverse on read has to know which
learner wrote the file.  `scenario["algo"]` is stamped for exactly that -- its
comment in the driver says "so a reader can undo `_archive_tree`" -- and the
stamp is the writer's word while `--algo` is the reader's.  Keying on the stamp
turns "you selected the wrong learner" into that sentence, instead of into a
key-set mismatch that reads as "this archive is broken".

The checks are by parse and by round-trip rather than by running the driver:
one `--iterations 1` on this market is hours, and the two things that can
actually go wrong -- the unwrap and the guard -- are reachable without it.
The one subprocess case is the guard, which is why it was moved to argument
validation: it must bite before the two minutes of environment construction.
"""
import ast
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
BENCH = REPO / "tools" / "benchmark"
SRC = (BENCH / "run_rl_02.py").read_text()
SRC_03 = (BENCH / "run_rl_03.py").read_text()

sys.path.insert(0, str(BENCH))
import params_npz  # noqa: E402  (the module under test's own format helper)


def _add_argument(src, flag):
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == flag):
            return node
    return None


# ── the flag itself ─────────────────────────────────────────────────────────

def test_the_flag_is_declared_with_the_same_default_as_market_03():
    """Same name and same empty default, so one reader reads both drivers."""
    for src, who in ((SRC, "run_rl_02.py"), (SRC_03, "run_rl_03.py")):
        node = _add_argument(src, "--eval-only")
        assert node is not None, f"{who} does not declare --eval-only"
        default = [k.value for k in node.keywords if k.arg == "default"]
        assert default and default[0].value == "", (
            f"{who}'s --eval-only default is {default!r}, not the empty string; "
            f"a truthy default would make every run an evaluation")


def test_eval_only_sets_iterations_to_zero():
    """The flag's whole content: skip training.  Checked where it is written,
    because a flag that loads the archive and still trains is a silent
    difference from market 03's meaning of the same name."""
    tree = ast.parse(SRC)
    main = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    found = []
    for node in ast.walk(main):
        if not (isinstance(node, ast.If) and "eval_only" in ast.unparse(node.test)):
            continue
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Assign) and len(sub.targets) == 1
                    and ast.unparse(sub.targets[0]) == "args.iterations"
                    and isinstance(sub.value, ast.Constant)
                    and sub.value.value == 0):
                found.append(ast.unparse(node.test))
    assert found, ("no `args.iterations = 0` under an `if ... eval_only` in "
                   "main(); --eval-only would load the archive and then train")


# ── the unwrap ─────────────────────────────────────────────────────────────

def test_the_unwrap_is_keyed_on_the_archive_stamp_not_on_the_flag():
    """`--algo` is the reader's word; the stamp is the writer's.

    A companion to the round-trip below, not a substitute: that one proves the
    unwrap is the right shape, this one proves the driver decides *whether* to
    unwrap from the file rather than from the command line.
    """
    anchor = '_stamped = (_init_info.get("scenario") or {}).get("algo")'
    assert SRC.count(anchor) == 1, (
        "run_rl_02.py no longer reads the archive's own `algo` stamp; "
        "re-anchor this check rather than deleting it")
    unwrap = [ln for ln in SRC.splitlines() if '_init_tree["params"]' in ln]
    assert len(unwrap) == 1, f"expected one unwrap line, found {unwrap}"
    guard = SRC[:SRC.index(unwrap[0])].splitlines()[-1]
    assert "_stamped or args.algo" in guard, (
        f"the unwrap is guarded by {guard.strip()!r}; it has to consult the "
        f"stamp, falling back to --algo only when the archive carries none")


def test_a_sac_shaped_archive_round_trips_through_the_stamp(tmp_path):
    """The load-bearing one: write as the driver writes, read as it reads.

    The fresh SAC parameter tree is `{actor, q1, ..., log_alpha}`, each a flax
    tree of its own (`{"params": ...}`).  `_archive_tree` puts one more
    `params` root on top, so the file's keys are `params/actor/params/...` and
    the reader's `read(...)[0]` carries that extra root.  What the driver's
    graft compares is `params_npz.flatten` of the fresh tree against
    `params_npz.flatten` of the stored one, key set for key set -- so the
    unwrap is correct exactly when those two sets are equal.
    """
    fresh = {"actor": {"params": {"Dense_0": {"kernel": np.zeros((3, 4), np.float32)}}},
             "q1": {"params": {"Dense_0": {"kernel": np.ones((4, 1), np.float32)}}},
             "log_alpha": np.array(-1.0, np.float32)}
    # exactly what the driver's `_archive_tree` does for a non-IPPO run
    params_npz.write(tmp_path / "sac.npz", {"params": fresh},
                     np.zeros(3, np.float32), np.ones(3, np.float32),
                     hyperparams={}, scenario={"algo": "sac"},
                     meta={"market": "02 real-time balancing"})
    tree, mean, std, info = params_npz.read(tmp_path / "sac.npz")
    assert info["scenario"]["algo"] == "sac", "the stamp did not survive the write"
    unwrapped = tree["params"]                      # the driver's inverse
    assert sorted(params_npz.flatten(unwrapped)) == sorted(params_npz.flatten(fresh)), (
        "the unwrapped tree's key set differs from the fresh learner tree's; "
        "the graft in run_rl_02.py compares exactly these two sets")
    #: the negative control that gives the assertion above its content: WITHOUT
    #: the unwrap the sets differ, which is the `days=0` failure this file is for
    assert sorted(params_npz.flatten(tree)) != sorted(params_npz.flatten(fresh)), (
        "the stored tree matches the fresh one even without unwrapping, so this "
        "test would pass on a driver that never unwraps")
    #: and the statistics travel with the weights, which is why market 02 reads
    #: through `params_npz` rather than through market 03's `_params_from`
    assert mean.shape == (3,) and std.shape == (3,)


def test_an_ippo_archive_still_needs_no_unwrap(tmp_path):
    """The other side of the same key: IPPO must be unchanged by all of this."""
    fresh = {"params": {"Dense_0": {"kernel": np.zeros((2, 2), np.float32)}}}
    params_npz.write(tmp_path / "ippo.npz", fresh,
                     np.zeros(2, np.float32), np.ones(2, np.float32),
                     hyperparams={}, scenario={"algo": "ippo"},
                     meta={"market": "02 real-time balancing"})
    tree, _m, _s, info = params_npz.read(tmp_path / "ippo.npz")
    assert info["scenario"]["algo"] == "ippo"
    assert sorted(params_npz.flatten(tree)) == sorted(params_npz.flatten(fresh)), (
        "an IPPO archive no longer reads back as the fresh tree; the unwrap "
        "must not apply to it")


# ── the guard ──────────────────────────────────────────────────────────────

def test_the_two_archive_flags_are_refused_together_before_any_heavy_work():
    """Both flags name an archive, so accepting both would silently pick one.

    Run as a subprocess with a deliberately unusable `--position`: the guard
    has to bite before the position is read, which is what "moved to argument
    validation" means.  A timeout here is a failure of that placement, not of
    the machine -- environment construction on this market is minutes.
    """
    out = subprocess.run(
        [sys.executable, str(BENCH / "run_rl_02.py"),
         "--position", "/dev/null", "--init-params", "a.npz",
         "--eval-only", "b.npz", "--out-dir", "/tmp/never-written"],
        capture_output=True, text=True, timeout=120,
        env={**__import__("os").environ, "JAX_PLATFORMS": "cpu"})
    assert out.returncode != 0, "giving both archive flags was accepted"
    assert "both name an archive" in out.stdout + out.stderr, (
        f"exited {out.returncode} but not for the reason this test is about; "
        f"tail: {(out.stdout + out.stderr)[-400:]!r}")
