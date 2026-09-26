"""L0: the parameter container markets 02 and 03 share round-trips and is stamped.

Market 02 wrote a bare flax msgpack, which has no
container for `obs_mean` / `obs_std`, for the scenario, or for the iteration --
so no checkpoint it had ever written could name the case or the scenario factors
that produced it.  What is under test here is the container, not any market's
arithmetic: the three refusals below are the reason a swap of container is safe
to make, and each one is a mistake this repository has actually made.
"""
import json
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "benchmark"))

import params_npz                                                 # noqa: E402


def _tree():
    return {"params": {"Dense_0": {"kernel": np.arange(12.0).reshape(3, 4),
                                   "bias": np.array([1.0, 2.0, 3.0, 4.0])},
                       "Dense_1": {"kernel": np.eye(4)}}}


def _write(tmp_path, **kw):
    kw.setdefault("hyperparams", {"lr": 3e-4, "weight_decay": 0.0})
    kw.setdefault("scenario", {"case": "29gb", "cap_scale": 0.6,
                               "ramp_scale": 1.0, "window": "four-season"})
    kw.setdefault("meta", {"market": "02 real-time balancing", "iteration": 40})
    return params_npz.write(tmp_path / "p.npz", _tree(), np.zeros(3),
                            np.ones(3), **kw)


def test_round_trip_is_bitwise(tmp_path):
    path = _write(tmp_path)
    tree, om, os_, info = params_npz.read(path)
    flat_in, flat_out = params_npz.flatten(_tree()), params_npz.flatten(tree)
    assert sorted(flat_in) == sorted(flat_out)
    for k in flat_in:
        assert np.array_equal(flat_in[k], flat_out[k]), k
    assert np.array_equal(np.asarray(om), np.zeros(3))
    assert np.array_equal(np.asarray(os_), np.ones(3))


def test_the_scenario_and_the_statistics_come_back_out(tmp_path):
    """The whole reason for the swap: a `.npz` on its own says what made it."""
    path = _write(tmp_path)
    _t, om, _os, info = params_npz.read(path)
    assert info["scenario"]["case"] == "29gb"
    assert info["scenario"]["cap_scale"] == 0.6
    assert info["scenario"]["window"] == "four-season"
    assert info["meta"]["iteration"] == 40
    assert om is not None


def test_extra_numeric_keys_do_not_enter_the_parameter_tree(tmp_path):
    """Market 03's measured failure, twice over.

    A name-list filter went stale when the writer gained a fourth field and a
    `<U12` scenario stamp reached `jnp.asarray`; a dtype filter then admitted the
    int64 `iteration` and `.apply` raised several frames from the cause.  The
    prefix is the criterion, so an extra key of any dtype must stay out.
    """
    path = params_npz.write(
        tmp_path / "p.npz", _tree(), np.zeros(3), np.ones(3),
        hyperparams={}, scenario={"case": "29gb"}, meta={"market": "x"},
        extra=dict(iteration=np.int64(40), note=np.array("a string")))
    tree, _om, _os, info = params_npz.read(path)
    assert sorted(tree) == ["params"]
    assert sorted(tree["params"]) == ["Dense_0", "Dense_1"]
    assert sorted(info["extra"]) == ["iteration", "note"]


def test_a_tree_with_no_params_prefix_is_refused(tmp_path):
    """An empty tree loads without error and applies to nothing; refuse the write."""
    with pytest.raises(ValueError, match="no `params/` key"):
        params_npz.write(tmp_path / "p.npz", {"Dense_0": {"kernel": np.eye(2)}},
                         np.zeros(1), np.ones(1), hyperparams={},
                         scenario={}, meta={})


def test_a_file_of_another_layout_is_refused(tmp_path):
    """Unflattening market 01's convention yields the right shape and wrong contents."""
    path = tmp_path / "other.npz"
    np.savez(path, **{"params/Dense_0/kernel": np.eye(2),
                      "obs_mean": np.zeros(1), "obs_std": np.ones(1),
                      "layout": np.array("pytree_treedef_string_v0")})
    with pytest.raises(ValueError, match="carries layout"):
        params_npz.read(path)


def test_market_03s_own_files_are_readable_through_this(tmp_path):
    """One convention, two writers.  `LAYOUT` is what makes that checkable.

    Two assertions, and the second is the one with content: the first reads the
    other writer's source, the second reads a file that writer produced.  A
    source match would still pass if market 03 changed what it puts *around* the
    layout key.
    """
    assert params_npz.LAYOUT == "flax_flat_v1"
    src = (REPO / "tools" / "benchmark" / "run_rl_03.py").read_text()
    assert 'f["layout"] = np.array("flax_flat_v1")' in src, (
        "market 03 no longer stamps this layout string, so the two writers have "
        "drifted and `read` will refuse one of them")
    # a real market 03 product, if this working tree has one.  Skipped rather
    # than faked when it does not: a hand-built file would only restate this
    # module's own writer and could never catch a drift in the other one.
    found = sorted((REPO / "scratch").glob("r1_03_s*/params_seed*.npz"))
    if not found:
        pytest.skip("no market 03 product in this tree to read")
    tree, om, _os, info = params_npz.read(found[0])
    assert "params" in tree and tree["params"], found[0]
    assert info["scenario"]["case"], info["scenario"]
    assert np.asarray(om).size
