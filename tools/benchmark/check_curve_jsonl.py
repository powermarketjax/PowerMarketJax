"""Assert that a training run's `.jsonl` carries the columns it was meant to.

**The failure this exists for is "the column was added and never arrived."**
`run_rl_01.py` already records that failure in its own comments: three
`sampled_*` series were appended to the curve rows, printed to stdout every
iteration, and never reached the product, because the writer named its columns
explicitly and nobody re-read the output.  A column that is added without
anybody looking at the output is a column that may or may not exist, and the
two states are indistinguishable from the log.

So this reads the file back through a consumer's path and asserts **per key**,
loudly, naming the first key and the first line that fails.  It is not a
smoke test of the driver: a driver that ran to completion and wrote nothing
useful passes every check a driver can run on itself.

**`--require` takes the key names, and getting one wrong must be an error.**
That is the point of the option existing at all rather than the list being
hardcoded: a check whose expectations cannot be stated wrongly cannot be shown
to bite.  Ask for `reward_per_agnet` and this exits non-zero; that run is the
evidence that asking for `reward_per_agent` and passing means something.

Exit code 0 on success, 1 on any failure.  Usage:

    python tools/benchmark/check_curve_jsonl.py <path.jsonl> --n-agents 66
"""
import argparse
import json
import math
import sys
from pathlib import Path

#: The six series the per-iteration log was added for, plus the two every
#: driver already had.  Overridable
#: so that a market with extra series can ask for those too.
DEFAULT_REQUIRED = ("iteration", "reward_mean", "costs_mean", "unconverged_frac",
                    "pg_loss", "vf_loss", "entropy", "approx_kl", "clip_frac",
                    "reward_per_agent")

#: What the meta line has to identify the run by.  A curve that cannot say which
#: commit, seed and scenario produced it is not comparable with any other curve,
#: which is the same rule `evaluation.write_day` enforces on the day products.
META_REQUIRED = ("market", "case", "seed", "commit", "cap_scale", "ramp_scale",
                 "voll", "markup_max", "episode_len", "window", "hyperparams",
                 "n_envs", "horizon", "off_shared")


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) \
        and math.isfinite(v)


def check(path, required, n_agents, min_iters):
    """Return a list of failure strings; empty means the file is good."""
    bad = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        return [f"{path} is empty"]

    try:
        meta = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        return [f"line 1 is not JSON: {exc}"]
    if meta.get("kind") != "meta":
        bad.append(f"line 1 has kind={meta.get('kind')!r}, expected 'meta'")
    for k in META_REQUIRED:
        if k not in meta:
            bad.append(f"meta line is missing {k!r}")
    # `off_shared` and `window` are allowed to be null -- absent and null are
    # different statements and only the first is a defect.

    rows = 0
    prev_iter = None
    for n, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            bad.append(f"line {n} is not JSON: {exc}")
            continue
        if row.get("kind") != "iter":
            bad.append(f"line {n} has kind={row.get('kind')!r}, expected 'iter'")
            continue
        rows += 1
        for k in required:
            if k not in row:
                bad.append(f"line {n}: missing key {k!r}")
                continue
            v = row[k]
            if isinstance(v, list):
                if not v:
                    bad.append(f"line {n}: {k!r} is an empty list")
                elif not all(_finite(x) for x in v):
                    n_bad = sum(1 for x in v if not _finite(x))
                    bad.append(f"line {n}: {k!r} has {n_bad}/{len(v)} "
                               f"non-finite entries")
                elif n_agents is not None and k == "reward_per_agent" \
                        and len(v) != n_agents:
                    bad.append(f"line {n}: reward_per_agent has length "
                               f"{len(v)}, expected {n_agents}")
            elif not _finite(v):
                bad.append(f"line {n}: {k!r} is {v!r}, not a finite number")
        it = row.get("iteration")
        if isinstance(it, int) and prev_iter is not None and it <= prev_iter:
            bad.append(f"line {n}: iteration {it} does not follow {prev_iter}")
        if isinstance(it, int):
            prev_iter = it

    if rows < min_iters:
        bad.append(f"{rows} iteration rows, expected at least {min_iters}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--n-agents", type=int, default=None,
                    help="expected length of `reward_per_agent`; omitted means "
                         "the length is not checked, which is weaker and should "
                         "be said out loud rather than defaulted to")
    ap.add_argument("--require", default=",".join(DEFAULT_REQUIRED),
                    help="comma-separated key names every iteration row must "
                         "carry as a finite number or a list of them")
    ap.add_argument("--min-iters", type=int, default=1)
    args = ap.parse_args()

    required = tuple(k.strip() for k in args.require.split(",") if k.strip())
    bad = check(args.path, required, args.n_agents, args.min_iters)
    if bad:
        print(f"FAIL {args.path}", file=sys.stderr)
        for b in bad[:40]:
            print(f"  {b}", file=sys.stderr)
        if len(bad) > 40:
            print(f"  ... and {len(bad) - 40} more", file=sys.stderr)
        raise SystemExit(1)
    n = sum(1 for line in Path(args.path).read_text(encoding="utf-8").splitlines()
            if line.strip()) - 1
    print(f"OK {args.path}: {n} iteration rows, each carrying "
          f"{', '.join(required)}"
          + (f"; reward_per_agent length {args.n_agents}"
             if args.n_agents is not None else "; length unchecked"))


if __name__ == "__main__":
    main()
