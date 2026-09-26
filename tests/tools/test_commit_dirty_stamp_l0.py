"""L0: the `commit_dirty` stamp counts tracked changes, and only those.

**Why this needs a test rather than a careful reading.** The stamp is written
into every product's `meta` and is what a reader consults to decide whether the
hash beside it identifies the code that ran.  It failed silently in the
direction that destroys its value: `git status --short` counts untracked files,
this repository is shared and always holds untracked scratch files, so the flag
read `True` on every product regardless of the code.  Measured 2026-09-06, the
03 boxed rerun's eight products all carried `commit_dirty=True` while
`git status --untracked-files=no` was empty.

A flag that is always `True` is not a conservative flag -- it is a flag with no
information, and it tells a reader to distrust products built from a clean
tree.  Both assertions below are needed: the first is the one that was broken,
the second is what keeps the fix from being "always report clean".
"""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "benchmark"))
from evaluation import _resolve_dirty                        # noqa: E402


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True,
                   capture_output=True, text=True)


@pytest.fixture()
def repo(tmp_path):
    """A real checkout with one committed file, so `git status` has meaning."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.invalid")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "tracked.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.py")
    _git(tmp_path, "commit", "-q", "-m", "one tracked file")
    return tmp_path


def test_clean_tree_is_clean(repo):
    assert _resolve_dirty(str(repo)) is False


def test_untracked_file_does_not_make_it_dirty(repo):
    """The defect this test exists for: an untracked file is not a code change.

    Two of them, one of which sits at the repository root exactly like the
    stray `RESUME.md` that made the 2026-09-06 products self-report dirty.
    """
    (repo / "RESUME.md").write_text("scratch\n", encoding="utf-8")
    (repo / "notes.txt").write_text("scratch\n", encoding="utf-8")
    out = subprocess.run(["git", "status", "--short"], cwd=repo,
                         capture_output=True, text=True).stdout
    assert out.strip(), "fixture is vacuous: git reports nothing to ignore"
    assert _resolve_dirty(str(repo)) is False


def test_modified_tracked_file_is_dirty(repo):
    """The other side: the flag must still fire on a real edit."""
    (repo / "tracked.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert _resolve_dirty(str(repo)) is True


def test_staged_but_uncommitted_tracked_file_is_dirty(repo):
    """Staging is not committing; the code that ran is still not the commit."""
    (repo / "tracked.py").write_text("VALUE = 3\n", encoding="utf-8")
    _git(repo, "add", "tracked.py")
    assert _resolve_dirty(str(repo)) is True


def test_the_tree_asked_about_is_the_import_tree_not_the_cwd(repo, monkeypatch):
    """A run started from somewhere else must still stamp THIS checkout.

    The defect: `cwd=None` hands `git` the process's working directory, so a job
    launched from a worktree -- or from any directory that happens to be another
    checkout -- stamped that directory's commit onto products built from this
    tree's code.  The two can be different revisions, and the product then names
    one the run never ran.

    `repo` here is the fixture's throwaway checkout, used as the **negative
    control**: standing in it must not change what the default answers, and its
    own hash must differ from this tree's, or the assertion below would pass
    for the wrong reason.
    """
    from evaluation import _IMPORT_TREE, _resolve_commit

    here = _resolve_commit(_IMPORT_TREE)
    other = _resolve_commit(str(repo))
    assert here not in ("", "unknown"), "this tree has no resolvable HEAD"
    assert other not in ("", "unknown"), "the fixture checkout has no HEAD"
    # the control: the two checkouts really are different, so the next
    # assertion has content
    assert here != other, (
        "the fixture repo resolved to this tree's HEAD; the check below would "
        "pass whatever cwd did")

    monkeypatch.chdir(repo)
    assert _resolve_commit() == here, (
        "standing in another checkout changed what the default resolves to -- "
        "`cwd=None` is reaching the process directory again")
    assert _resolve_dirty() == _resolve_dirty(_IMPORT_TREE), (
        "the dirty flag followed the cwd rather than the import tree")
