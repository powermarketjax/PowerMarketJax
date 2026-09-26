"""L0: `run_rl_01.py --checkpoint-every`'s help names the path the code writes.

**The failure this exists for.**  The help said checkpoints go to
`<curve-out stem>_iter%04d.params.npz`, while the code writes
`<out-dir>/checkpoints/seed{S}_iter{n:04d}.npz`; a reader following the help
looked for files that were never written.  Checked by parse: the help string
and the write site are read out of the same source.
"""
import ast
import pathlib

SRC = (pathlib.Path(__file__).resolve().parents[2]
       / "tools" / "benchmark" / "run_rl_01.py").read_text()


def _help(flag):
    for node in ast.walk(ast.parse(SRC)):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == flag):
            kw = [k.value for k in node.keywords if k.arg == "help"]
            return ast.literal_eval(kw[0]) if kw else ""
    raise AssertionError(f"{flag} not declared")


def test_checkpoint_help_matches_the_write_site():
    assert 'Path(args.out_dir) / "checkpoints"' in SRC, "write site moved; re-anchor"
    assert 'f"seed{args.seed}_iter' in SRC, "write site moved; re-anchor"
    h = _help("--checkpoint-every")
    assert "<out-dir>/checkpoints/seed{S}_iter" in h, h
    assert "curve-out" not in h, h
