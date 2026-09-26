"""The per-iteration training record: one JSON object per line, flushed as it runs.

**Why this exists beside the `.npz` curve.**  All three learning drivers already
write a `.npz` curve, and all three write it by rebuilding the whole file every
`curve_every` iterations.  That makes the loss from a killed process
proportional to the interval rather than total, which was the fix it was
introduced for -- but the interval is 10 iterations by default and an iteration
of market 01 costs about 146 s, so the tail that a kill still destroys is up to
about 25 minutes of training.  Appending one line per iteration makes the tail
one iteration.

**It is a second product, not a replacement.**  The `.npz` files are what the
plotting and reading tools open, and their column layout is depended on
elsewhere; nothing here changes them.  What this adds is the record that is
complete: the drivers' `.npz` columns are a hand-picked subset of what
`ippo.iterate` returns, and the losses, the entropy, the KL and the per-agent
reward were reaching stdout at best.

**`kind` is on every line and is the first thing a reader should branch on.**
Line 1 is `{"kind": "meta", ...}` and carries the run's identity -- market,
case, seed, commit, scenario factors, every hyperparameter, batch shape.  Every
line after it is `{"kind": "iter", ...}`.  A reader that assumes line 1 is an
iteration gets a row with no `iteration` key rather than a plausible one.

**NaN is written as the JSON extension `NaN`, not as `null`.**  `json.loads`
reads it back to `float("nan")`, which is what the value was; mapping it to
`null` would make "this metric was not finite" and "this metric was not
recorded" the same token.  This costs strict-JSON compliance: a parser without
the extension will refuse the line.  The consumers are Python.

**Paths in the meta line are repository-relative**, via `repo_relative`.  A
product carrying an absolute path names the machine and the account it was
produced on, and this repository's rule is that anything that can be committed
or shipped as supplementary material carries neither.
"""
import json
import os
from pathlib import Path

import numpy as np

#: `tools/benchmark/curve_jsonl.py` -> the checkout root.
REPO = Path(__file__).resolve().parents[2]


def repo_relative(path):
    """`path` as a relative path from the repository root.

    Deliberately unconditional: a path outside the checkout comes back as
    `../../something` rather than as an absolute path.  That is still readable,
    still resolvable against a known root, and carries no home directory and no
    account name.  Returning the absolute path for the outside case -- the
    obvious alternative -- would put the leak back exactly where products from
    an external data directory land, which is the case the rule is for.
    """
    return os.path.relpath(Path(path).resolve(), REPO)


def _encode(obj):
    """`json.dumps(default=...)` for the values these drivers actually hold.

    numpy scalars, numpy arrays and jax arrays all reach here; `tolist` covers
    the last of those without importing jax into a module that a test may want
    to load without a device.  Anything else raises, rather than being
    stringified: a metric silently written as `"Array(1.0, dtype=float64)"`
    reads as a value in the file and is not one.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return repo_relative(obj)
    if hasattr(obj, "tolist"):            # jax.Array
        return np.asarray(obj).tolist()
    raise TypeError(f"{type(obj).__name__} has no JSON form here; convert it at "
                    f"the call site so the conversion is visible")


def jsonl_path(curve_out):
    """`<curve-out stem>.jsonl`, beside the `.npz` and named after it."""
    return Path(curve_out).with_suffix(".jsonl")


class CurveLog:
    """The open file, or a no-op when the driver was given no `--curve-out`.

    A driver run without `--curve-out` writes no curve at all today, and making
    this the one place that branches on it keeps the `if` out of the training
    loop, where an accidental `None` would surface as an `AttributeError` after
    the first iteration rather than at start-up.
    """

    def __init__(self, curve_out, meta):
        self.path = None
        self._fh = None
        if not curve_out:
            return
        self.path = jsonl_path(curve_out)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # "w", not "a": a re-run with the same `--curve-out` replaces its own
        # curve `.npz`, and a `.jsonl` that instead grew would hold two runs
        # under one meta line -- the second run's rows attributed to the first
        # run's commit and hyperparameters.
        self._fh = self.path.open("w", encoding="utf-8")
        self._write(dict(kind="meta", **meta))

    def iteration(self, row):
        """One iteration's record.  `row` is the driver's own curve row."""
        self._write(dict(kind="iter", **row))

    def _write(self, obj):
        if self._fh is None:
            return
        self._fh.write(json.dumps(obj, default=_encode, allow_nan=True,
                                  ensure_ascii=False) + "\n")
        # flush AND fsync.  `flush` alone survives the process being killed,
        # which is the failure the `.npz` rewrite interval already covers
        # badly; `fsync` also survives the machine going away, which is what
        # actually took the `rl_01_wd052` run of 2026-08-20 at iteration 111.
        # It costs one disk sync per iteration against iterations that cost
        # tens of seconds.
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False
