"""L0: `run_rl_01.py` writes the trained policy whether or not `--curve-out` is given.

**The failure this exists for.**  The final write sat under `if args.curve_out:`,
so a run given only `--out-dir` -- the second training command in the README --
trained, evaluated and wrote no weights at all, while `run_rl_03.py` always writes
`<out-dir>/params_seed{S}.npz`.  The fix keeps `<curve-out stem>.params.npz` when
`--curve-out` is given (every existing market-01 archive is named that way) and
otherwise writes `<out-dir>/params_seed{S}.npz`.

Checked by parse and by calling the path helper, not by running the driver: the
bug is entirely in whether the write is reached and where it goes.
"""
import ast
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
BENCH = REPO / "tools" / "benchmark"
SRC = (BENCH / "run_rl_01.py").read_text()

sys.path.insert(0, str(BENCH))
import run_rl_01  # noqa: E402


def _main():
    return next(n for n in ast.walk(ast.parse(SRC))
                if isinstance(n, ast.FunctionDef) and n.name == "main")


def _enclosing_ifs(tree, target):
    """The test expressions of every `if` that contains `target`."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and any(sub is target for b in node.body
                                             for sub in ast.walk(b)):
            out.append(ast.unparse(node.test))
    return out


def test_the_final_write_is_not_conditioned_on_curve_out():
    main = _main()
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "final_params_path"]
    assert len(calls) == 1, (
        "main() does not derive the final weights' path through "
        "final_params_path; the final write may still depend on --curve-out")
    guards = _enclosing_ifs(main, calls[0])
    assert not any("curve_out" in g for g in guards), (
        f"the final write is still inside {guards}; without --curve-out no "
        f"weights would be written")


def test_without_curve_out_the_weights_go_to_out_dir():
    assert run_rl_01.final_params_path("", "/r/out", 2) == pathlib.Path(
        "/r/out/params_seed2.npz")


def test_with_curve_out_the_existing_name_is_unchanged():
    assert run_rl_01.final_params_path("runs/a/curve.jsonl", "/r/out", 2) == \
        pathlib.Path("runs/a/curve.params.npz")
