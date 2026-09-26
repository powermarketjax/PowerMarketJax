"""The on-disk parameter format markets 02 and 03 share: one `.npz`, self-describing.

Market 01 writes a pytree with `p0 p1 p2 ...` plus a *stringified* `treedef`,
which no reader can reconstruct the tree from; market 03 writes flat
`params/<layer>/<name>` keys and stamps `layout="flax_flat_v1"` so a reader can
tell which convention it was handed.  This module is that second convention,
extracted when market 02 needed it (2026-08-28).

**Why market 02 could not keep `flax.serialization.to_bytes`.**  A bare flax
msgpack is the parameter tree and nothing else -- there is no container for
`obs_mean` / `obs_std`, for the scenario, or for the iteration.  A checkpoint
without the standardisation statistics does not fail when it is loaded: it
standardises with a reference the weights were never fitted against and
reconstructs a *different* policy that looks like the saved one, which is why a
policy-collapse probe on those files has to refit them per seed.  And a stamp can only be written by the writer --
there is nowhere to add one afterwards -- so the three `.msgpack` files already
on disk stay as they are and this only helps from here on.

Both markets' files are readable through `read`, since the layout stamp is what
distinguishes them and both write the same one.
"""
import json
from pathlib import Path

import numpy as np

#: The value of the `layout` key.  Same string market 03 writes, so a reader
#: that accepts one accepts the other; changing it makes every existing file
#: unreadable, which is the point of stamping it at all.
LAYOUT = "flax_flat_v1"


def flatten(node):
    """`{"a": {"b": arr}}` -> `{"a/b": arr}`, the on-disk parameter layout."""
    flat = {}

    def walk(prefix, sub):
        if isinstance(sub, dict):
            for k, v in sub.items():
                walk(f"{prefix}/{k}" if prefix else str(k), v)
        else:
            flat[prefix] = np.asarray(sub)

    walk("", node)
    return flat


def unflatten(flat):
    """`{"a/b": arr}` -> `{"a": {"b": arr}}`, the inverse of `flatten`."""
    node = {}
    for key, value in flat.items():
        cur = node
        parts = key.split("/")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value
    return node


def write(path, params, obs_mean, obs_std, *, hyperparams, scenario, meta,
          extra=None):
    """Write one parameter file.  Every argument is required except `extra`.

    `obs_mean` / `obs_std` go into **every** file including checkpoints, for the
    reason in the module docstring.  `scenario` and `meta` are required rather
    than defaulted because a default would ship one market's scenario to the
    other: the whole reason this file exists is that a `.npz` on its own has to
    be able to say which case and which scenario factors produced it.
    """
    flat = flatten(params)
    if not any(k.startswith("params/") for k in flat):
        raise ValueError(
            f"no `params/` key in the flattened tree, only {sorted(flat)[:6]}. "
            f"Readers whitelist that prefix -- `_flatten_params` is what puts "
            f"it there -- so a file without it loads as an empty tree rather "
            f"than as an error")
    out = dict(flat)
    out["obs_mean"] = np.asarray(obs_mean)
    out["obs_std"] = np.asarray(obs_std)
    out["layout"] = np.array(LAYOUT)
    out["hyperparams"] = np.array(json.dumps(hyperparams))
    out["scenario"] = np.array(json.dumps(scenario))
    out["meta"] = np.array(json.dumps(meta))
    if extra:
        out.update({k: np.asarray(v) for k, v in extra.items()})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return Path(path)


def read(path):
    """`(params, obs_mean, obs_std, info)` from a file `write` produced.

    Only keys under `params/` become the tree.  A name list or a dtype filter
    were both tried in market 03 and both admitted something they should not
    have -- a `<U12` scenario stamp reached `jnp.asarray` and took down
    `--eval-only` on all three seeds; a dtype filter admitted the int64
    `iteration` and `.apply` raised several frames from the cause.  The prefix
    is the criterion because `flatten` is what puts it there.
    """
    z = np.load(path, allow_pickle=True)
    layout = str(z["layout"]) if "layout" in z.files else None
    if layout != LAYOUT:
        raise ValueError(
            f"{path} carries layout {layout!r}, not {LAYOUT!r}. Reading it "
            f"anyway would unflatten keys written under another convention, "
            f"which yields a tree of the right shape and the wrong contents")
    tree = unflatten({k: z[k] for k in z.files if k.startswith("params/")})
    info = {}
    for key in ("hyperparams", "scenario", "meta"):
        if key in z.files:
            info[key] = json.loads(str(z[key]))
    info["extra"] = {k: z[k] for k in z.files
                     if not k.startswith("params/")
                     and k not in ("obs_mean", "obs_std", "layout",
                                   "hyperparams", "scenario", "meta")}
    return tree, z["obs_mean"], z["obs_std"], info
