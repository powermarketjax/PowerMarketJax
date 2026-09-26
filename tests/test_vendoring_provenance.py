# Written for this repository on 2026-08-06 -- no upstream counterpart.
"""Every file claiming to be a verbatim copy must actually be one.

The verbatim claim is mechanically checkable and should be re-run
after any vendored file changes. Nothing enforced that until this file existed, so the
claim was aspirational.

Skips when the upstream checkout is absent -- which is the case in CI, and is
required anyway: this repository does not depend on the sibling repo at runtime.
Point ``POWERMARKETJAX_UPSTREAM`` at a PowerZooJax checkout to run it.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = Path(os.environ.get("POWERMARKETJAX_UPSTREAM", REPO.parent / "PowerZooJax"))

# This repository is organised by responsibility instead of mirroring upstream's tree,
# so the package rename alone does not make the two sides comparable.
REMAP = [
    ("powerzoojax", "powermarketjax"),
    ("powermarketjax.envs.grid", "powermarketjax.physics"),
    ("powermarketjax.envs.resource", "powermarketjax.resources"),
    ("powermarketjax.envs.base", "powermarketjax.resources.env_base"),
    ("powermarketjax.envs.spaces", "powermarketjax.spaces"),
    ("powermarketjax/envs/grid", "powermarketjax/physics"),
    ("powermarketjax/envs/resource", "powermarketjax/resources"),
    ("powermarketjax/envs/base", "powermarketjax/resources/env_base"),
    ("powermarketjax/envs/spaces", "powermarketjax/spaces"),
    ("tests.grid", "tests.physics"),
    ("tests.resource.", "tests.resources."),
    ("tests/grid", "tests/physics"),
    ("tests/resource/", "tests/resources/"),
]


def _strip_leading_comments(text: str) -> str:
    lines = text.split("\n")
    i = 0
    while i < len(lines) and (lines[i].startswith("#") or not lines[i].strip()):
        i += 1
    return "\n".join(lines[i:])


def _normalise(text: str) -> str:
    for old, new in REMAP:
        text = text.replace(old, new)
    return text


def _upstream_blob(sha: str, path: str):
    r = subprocess.run(["git", "-C", str(UPSTREAM), "show", f"{sha}:{path}"],
                       capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def _vendored_files():
    """(path, upstream_path, sha) for every file declaring a verbatim copy."""
    out = []
    for rel in subprocess.run(["git", "-C", str(REPO), "ls-files", "*.py"],
                              capture_output=True, text=True).stdout.split():
        head = (REPO / rel).read_text()[:2500]
        src = re.search(r"^#\s*Source\s*:\s*(\S+)", head, re.M)
        sha = re.search(r"^#\s*Upstream commit:\s*(\S+)", head, re.M)
        if src and sha and "Verbatim copy" in head:
            out.append((rel, src.group(1), sha.group(1)))
    return out


pytestmark = pytest.mark.skipif(
    not (UPSTREAM / ".git").exists(),
    reason=f"upstream checkout not found at {UPSTREAM} (expected in CI)",
)


def test_verbatim_files_match_upstream():
    files = _vendored_files()
    assert files, "no file declares a verbatim copy -- has the header format changed?"

    mismatched, unreachable = [], []
    for rel, src, sha in files:
        blob = _upstream_blob(sha, src)
        if blob is None:
            unreachable.append(f"{rel} -> {sha}:{src}")
            continue
        if _normalise(_strip_leading_comments(blob)) != _strip_leading_comments(
                (REPO / rel).read_text()):
            mismatched.append(f"{rel}  (upstream {sha}:{src})")

    assert not unreachable, (
        "provenance header points at something upstream does not have:\n  "
        + "\n  ".join(unreachable))
    assert not mismatched, (
        f"{len(mismatched)} of {len(files)} files claim 'Verbatim copy' but differ "
        f"from upstream. Either restore them, or change the header to 'NOT "
        f"verbatim' and record the deviation:\n  " + "\n  ".join(mismatched))
