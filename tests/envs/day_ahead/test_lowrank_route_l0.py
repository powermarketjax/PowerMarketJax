"""L0 for the two switches of the low-rank route (review, 2026-09-16):
what the operator's ``spec`` says about the LP it solves and the route it
solves it by, and how a driver is meant to stamp that.

Written red first.  Measured 2026-09-16: a copy of `run_eval_01.py` with the ``monitored_lines=`` argument
to `make_env` deleted, run with ``--monitored-lines rated``, produced twelve
day products whose meta said ``monitored_lines = [0..98]`` -- the same as the
unmodified driver -- because the stamp was taken from the parsed flag, not
from the operator that was built.
Three facts are asserted here so that cannot recur:

1. `make_clearing` / `make_relax` write into ``spec`` the monitored set they
   actually carry (``None`` when every line, as before) and the route they
   actually built (``kkt_route``), whatever the flags were.
2. ``kkt="auto"`` picks the dense sweep whenever the monitored set is every
   line, index array or not: the low-rank route's capacitance matrix is
   ``T (k + 1)`` wide and on `case29gb` / `case73rts` ``rated_lines`` is every
   line, so ``auto`` must not send those cases down it.
3. `evaluation.effective_monitored_stamp` reads the stamp off ``spec`` and
   refuses when it disagrees with what the command line asked for, before a
   run starts.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

from powermarketjax.case import load_case
from powermarketjax.envs.day_ahead import make_clearing, rated_lines
from powermarketjax.envs.day_ahead.relax import make_relax

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "benchmark"))
from evaluation import effective_monitored_stamp  # noqa: E402

from .test_lowrank_l2 import x64  # noqa: F401


def _spec(builder, case, **kw):
    _, spec = builder(case, 2, n_segments=1, **kw)
    return spec


@pytest.mark.parametrize("builder", [make_clearing, make_relax], ids=["clearing", "relax"])
def test_spec_carries_the_effective_set_and_route(builder):
    gb = load_case("29gb")
    n_gb = np.asarray(gb.line_cap).shape[0]
    s = _spec(builder, gb)
    assert s["monitored_lines"] is None and s["n_lines"] == n_gb and s["kkt_route"] == "dense"
    # the full set given as an index array is the same LP as None: dense under auto
    s = _spec(builder, gb, monitored_lines=np.arange(n_gb))
    assert s["kkt_route"] == "dense", "auto sent an all-lines LP down the low-rank route"
    assert np.array_equal(s["monitored_lines"], np.arange(n_gb)) and s["n_l"] == n_gb
    # forcing is still honoured on the same LP, both ways
    assert _spec(builder, gb, monitored_lines=np.arange(n_gb), kkt="lowrank")["kkt_route"] == "lowrank"
    nem = load_case("813nem")
    mon = rated_lines(nem)
    assert mon.size == 7
    s = _spec(builder, nem, monitored_lines=mon)
    assert s["kkt_route"] == "lowrank" and np.array_equal(s["monitored_lines"], mon)
    assert s["n_l"] == 7 and s["n_lines"] == np.asarray(nem.line_cap).shape[0]
    assert _spec(builder, nem, monitored_lines=mon, kkt="dense")["kkt_route"] == "dense"


def test_effective_stamp_reads_spec_and_refuses_a_mismatch():
    mon = np.array([10, 134, 188])
    env_spec = dict(clearing=dict(monitored_lines=mon, n_lines=1278, kkt_route="lowrank"))
    stamp, route = effective_monitored_stamp(env_spec, requested=mon)
    assert stamp == [10, 134, 188] and route == "lowrank"
    env_spec = dict(clearing=dict(monitored_lines=None, n_lines=1278, kkt_route="dense"))
    assert effective_monitored_stamp(env_spec, requested=None) == (None, "dense")
    # the injection: the flag asked for three lines, the operator carries all
    with pytest.raises(SystemExit) as e:
        effective_monitored_stamp(env_spec, requested=mon)
    assert "requested" in str(e.value) and "effective" in str(e.value)
    # and the other direction
    env_spec = dict(clearing=dict(monitored_lines=mon, n_lines=1278, kkt_route="lowrank"))
    with pytest.raises(SystemExit):
        effective_monitored_stamp(env_spec, requested=None)
