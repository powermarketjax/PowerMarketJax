"""The per-iteration JSONL curve: the writer, the checker, and what the drivers ask for.

Three things are checked here and they fail for different reasons.

**The writer** (`benchmark.curve_jsonl`) has to produce a file that is complete
before the process ends, because the failure it was built for is the process not
ending -- `rl_01_wd052` on 2026-08-20 died at iteration 111 and left eleven
checkpoints and no curve.  So the test reads the file back *while it is still
open*.

**The checker** (`benchmark.check_curve_jsonl`) is the thing that makes "the
column arrived" a statement rather than a hope, and a checker that cannot be
shown to bite is worth nothing.  Each of its checks gets a case that must fail:
a misspelled key, a NaN, a wrong-length per-agent row.  The misspelling case is
the one the ticket asks to be demonstrated by hand as well, and this is its
permanent form -- the manual run happens once, this runs every time.

**The drivers** are checked statically, by AST: each of the three must name all
six of the ticket's series as keyword arguments of its `curve.append(dict(...))`
call.  No driver is imported and no `main()` runs, so this costs milliseconds
and cannot be defeated by a driver that is expensive to start.  It does not
check that the values are right -- only that the columns are asked for at all,
which is exactly the regression that has happened before (three `sampled_*`
series reached stdout and never reached the product).
"""
import ast
import json
import math
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from benchmark import check_curve_jsonl                             # noqa: E402
from benchmark.curve_jsonl import CurveLog, jsonl_path, repo_relative  # noqa: E402

#: The six the ticket adds, spelled once.  A driver that drops one of them is
#: the regression this file exists for, so the list is not derived from the
#: drivers -- deriving it from the thing under test would make it vacuous.
REQUIRED_SERIES = ("pg_loss", "vf_loss", "entropy", "approx_kl", "clip_frac",
                   "reward_per_agent")

DRIVERS = ("run_rl_01.py", "run_rl_02.py", "run_rl_03.py")


def _meta(**kw):
    base = dict(market="test", case="29gb", seed=0, commit="0" * 40,
                cap_scale=0.6, ramp_scale=1.0, voll=1e4, markup_max=2.0,
                episode_len=1, window=None, hyperparams={"lr": 3e-4},
                n_envs=8, horizon=4, off_shared=None)
    base.update(kw)
    return base


def _row(it, n_agents=4, **kw):
    row = dict(iteration=it, reward_mean=1.0, costs_mean=0.0,
               unconverged_frac=0.0, pg_loss=-0.1, vf_loss=0.2, entropy=1.3,
               approx_kl=1e-4, clip_frac=0.05,
               reward_per_agent=np.arange(n_agents, dtype=np.float64))
    row.update(kw)
    return row


def test_jsonl_path_sits_beside_the_npz():
    assert jsonl_path("runs/x/curve.npz").name == "curve.jsonl"
    assert jsonl_path("runs/x/curve").name == "curve.jsonl"


def test_repo_relative_never_leaks_an_absolute_path():
    """The path discipline, checked on both sides of the repository boundary."""
    inside = repo_relative(REPO / "tools" / "benchmark" / "curve_jsonl.py")
    assert inside == "tools/benchmark/curve_jsonl.py"
    outside = repo_relative("/etc/hostname")
    assert not pathlib.PurePosixPath(outside).is_absolute()
    assert outside.startswith("..")


def test_rows_are_on_disk_before_the_file_is_closed(tmp_path):
    """The whole point: a killed process must leave every completed iteration."""
    log = CurveLog(tmp_path / "c.npz", _meta())
    log.iteration(_row(0))
    log.iteration(_row(1))
    # deliberately NOT closed -- this is the state a `kill -9` leaves
    lines = (tmp_path / "c.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["kind"] == "meta"
    assert [json.loads(x)["kind"] for x in lines[1:]] == ["iter", "iter"]
    log.close()


def test_numpy_values_survive_the_round_trip(tmp_path):
    with CurveLog(tmp_path / "c.npz", _meta()) as log:
        log.iteration(_row(0, n_agents=3, sampled_shed_cells=np.int64(7),
                           seconds=np.float64(1.5)))
    row = json.loads((tmp_path / "c.jsonl").read_text().splitlines()[1])
    assert row["reward_per_agent"] == [0.0, 1.0, 2.0]
    assert row["sampled_shed_cells"] == 7
    assert row["seconds"] == 1.5


def test_nan_stays_nan_rather_than_becoming_null(tmp_path):
    """`null` would conflate "not finite" with "not recorded"."""
    with CurveLog(tmp_path / "c.npz", _meta()) as log:
        log.iteration(_row(0, approx_kl=float("nan")))
    row = json.loads((tmp_path / "c.jsonl").read_text().splitlines()[1])
    assert row["approx_kl"] is not None and math.isnan(row["approx_kl"])


def test_a_value_with_no_json_form_raises_rather_than_being_stringified(tmp_path):
    with CurveLog(tmp_path / "c.npz", _meta()) as log:
        with pytest.raises(TypeError):
            log.iteration(_row(0, whatever=object()))


def test_no_curve_out_writes_nothing(tmp_path):
    log = CurveLog("", _meta())
    log.iteration(_row(0))
    log.close()
    assert log.path is None
    assert not list(tmp_path.iterdir())


def _write(tmp_path, rows, meta=None):
    with CurveLog(tmp_path / "c.npz", meta or _meta()) as log:
        for r in rows:
            log.iteration(r)
    return tmp_path / "c.jsonl"


def test_checker_passes_a_good_file(tmp_path):
    path = _write(tmp_path, [_row(0), _row(1)])
    assert check_curve_jsonl.check(path, check_curve_jsonl.DEFAULT_REQUIRED,
                                   4, 2) == []


def test_checker_bites_on_a_misspelled_key(tmp_path):
    """The ticket's "one verification that must fail", in its permanent form.

    A checker that passes whatever it is given cannot witness that the column
    arrived, so asking it for a key nobody writes has to be an error.
    """
    path = _write(tmp_path, [_row(0)])
    bad = check_curve_jsonl.check(path, ("reward_per_agnet",), None, 1)
    assert bad and "reward_per_agnet" in bad[0]


def test_checker_bites_on_a_missing_series(tmp_path):
    for key in REQUIRED_SERIES:
        row = _row(0)
        del row[key]
        path = _write(tmp_path, [row])
        bad = check_curve_jsonl.check(path, check_curve_jsonl.DEFAULT_REQUIRED,
                                      4, 1)
        assert bad and key in bad[0], f"dropping {key} was not caught"


def test_checker_bites_on_nan_and_on_a_wrong_agent_count(tmp_path):
    path = _write(tmp_path, [_row(0, pg_loss=float("nan"))])
    assert any("pg_loss" in b for b in
               check_curve_jsonl.check(path, check_curve_jsonl.DEFAULT_REQUIRED,
                                       4, 1))
    path = _write(tmp_path, [_row(0, n_agents=3)])
    assert any("length 3" in b for b in
               check_curve_jsonl.check(path, check_curve_jsonl.DEFAULT_REQUIRED,
                                       4, 1))


def test_checker_bites_on_a_meta_line_that_cannot_identify_the_run(tmp_path):
    meta = _meta()
    del meta["commit"]
    del meta["seed"]
    path = _write(tmp_path, [_row(0)], meta=meta)
    bad = check_curve_jsonl.check(path, check_curve_jsonl.DEFAULT_REQUIRED, 4, 1)
    assert any("commit" in b for b in bad) and any("seed" in b for b in bad)


def _dict_keywords(expr, bound):
    """Keyword names a `dict(...)` expression contributes, following `**x`.

    Since 2026-09-03 the three drivers assemble the learner's own diagnostics
    as a separate `dict(...)` per algorithm (`--algo ippo` / `--algo sac`) and
    splat it into the row, either as `**diag` with `diag = dict(...) if ...
    else dict(...)` assigned just above, or inline as `**(dict(...) if ...
    else dict(...))`.  Both branches are read, so the names of EVERY
    algorithm's series count as recorded, and the six IPPO series are still
    required of every driver.
    """
    names = set()
    if isinstance(expr, ast.IfExp):
        return _dict_keywords(expr.body, bound) | _dict_keywords(expr.orelse, bound)
    if isinstance(expr, ast.Name) and expr.id in bound:
        return _dict_keywords(bound[expr.id], bound)
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) \
            and expr.func.id == "dict":
        for k in expr.keywords:
            if k.arg:
                names.add(k.arg)
            else:
                names |= _dict_keywords(k.value, bound)
    return names


def _curve_append_keywords(source):
    """Every keyword name of a `curve.append(dict(...))` call in `source`."""
    tree = ast.parse(source)
    # `name = <dict expression>` assignments, so a `**name` splat can be
    # followed to the dict it carries
    bound = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            bound.setdefault(node.targets[0].id, node.value)
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "append"
                and isinstance(f.value, ast.Name) and f.value.id == "curve"):
            continue
        for arg in node.args:
            names |= _dict_keywords(arg, bound)
    return names


@pytest.mark.parametrize("driver", DRIVERS)
def test_every_driver_records_the_six_series(driver):
    src = (REPO / "tools" / "benchmark" / driver).read_text(encoding="utf-8")
    names = _curve_append_keywords(src)
    assert names, f"{driver} has no `curve.append(dict(...))` to read"
    missing = [k for k in REQUIRED_SERIES if k not in names]
    assert not missing, (f"{driver} does not record {missing}; they are "
                         f"returned by `ippo.iterate` and dropping them is the "
                         f"regression this test exists for")


@pytest.mark.parametrize("driver", DRIVERS)
def test_every_driver_writes_the_jsonl(driver):
    """The rows existing is not the same as the rows being written out."""
    src = (REPO / "tools" / "benchmark" / driver).read_text(encoding="utf-8")
    assert "CurveLog(" in src, f"{driver} builds no CurveLog"
    assert "jsonl.iteration(" in src, f"{driver} never appends an iteration row"
