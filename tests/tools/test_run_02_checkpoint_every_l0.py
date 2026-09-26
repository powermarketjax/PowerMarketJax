"""L0: `run_rl_02.py --checkpoint-every N` writes checkpoints without `--params-out`.

**The failure this exists for.**  The periodic write sat under
`if args.checkpoint_every and args.params_out and ...`, so a run given
`--checkpoint-every` and no `--params-out` was accepted, trained, and wrote no
checkpoint at all -- while `run_rl_01.py` and `run_rl_03.py` write theirs under
`<out-dir>/checkpoints/` with no other flag needed.  Someone following the
command lines of markets 01 and 03 would believe a market-02 run had saved its
checkpoints.

The fix keeps the layout next to `--params-out` when that flag is given (every
existing market-02 checkpoint and its `_iters.json` index are laid out that
way) and otherwise writes `<out-dir>/checkpoints/seed{S}_iter{n:04d}.npz`, the
names markets 01 and 03 use.  Checked by parse and by one direct call of the
writer, not by running the driver: one iteration on this market is minutes of
environment construction and the bug is entirely in which path is chosen.
"""
import ast
import json
import pathlib
import re
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
BENCH = REPO / "tools" / "benchmark"
SRC = (BENCH / "run_rl_02.py").read_text()

sys.path.insert(0, str(BENCH))
import run_rl_02  # noqa: E402


def _ckpt_guards():
    """The test expression of every `if` whose body calls `_ckpt` in `main`."""
    main = next(n for n in ast.walk(ast.parse(SRC))
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    out = []
    for node in ast.walk(main):
        if isinstance(node, ast.If) and any(
                isinstance(s, ast.Call) and getattr(s.func, "id", None) == "_ckpt"
                for b in node.body for s in ast.walk(b)):
            out.append(ast.unparse(node.test))
    return out


def test_the_periodic_write_is_not_gated_on_params_out():
    guards = _ckpt_guards()
    assert len(guards) == 1, f"expected one guarded `_ckpt` call in main, found {guards}"
    assert "checkpoint_every" in guards[0], guards[0]
    assert "params_out" not in guards[0], (
        f"the periodic write is still conditioned on --params-out: {guards[0]!r}; "
        f"--checkpoint-every alone would write nothing")


def test_without_params_out_the_base_is_the_01_03_checkpoint_directory():
    assert run_rl_02.checkpoint_base("", "/r/out", 3) == "/r/out/checkpoints/seed3.npz"


def test_with_params_out_the_existing_layout_is_unchanged():
    assert run_rl_02.checkpoint_base("x/run.npz", "/r/out", 3) == "x/run.npz"


def test_the_writer_puts_the_file_where_markets_01_and_03_put_theirs(tmp_path):
    tree = {"params": {"Dense_0": {"kernel": np.zeros((2, 3), np.float32)}}}
    run_rl_02._ckpt(run_rl_02.checkpoint_base("", str(tmp_path), 0), 1, tree, 0.5,
                    obs_mean=np.zeros(2, np.float32), obs_std=np.ones(2, np.float32),
                    hyperparams={}, scenario={"algo": "ippo"})
    written = sorted(p.name for p in (tmp_path / "checkpoints").iterdir())
    assert written == ["seed0_iter0001.npz", "seed0_iters.json"], written
    assert re.fullmatch(r"seed\d+_iter\d{4}\.npz", written[0])
    rows = json.loads((tmp_path / "checkpoints" / "seed0_iters.json").read_text())
    assert rows == [dict(iteration=1, reward_mean=0.5, file="seed0_iter0001.npz")]
