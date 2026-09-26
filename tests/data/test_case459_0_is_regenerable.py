# Written for this repository on 2026-08-15 -- no upstream counterpart.
"""The link between feeder 459_0's embedded tables and the parquets they came from.

`powermarketjax/case/` is self-contained: no case file reads `data/`, and opening
that coupling for one case was not worth it.  So 459_0's node and branch tables
are embedded in the case module, which means the same numbers live in this
repository twice.

**A generator does not pay that cost, because nobody runs a generator.**  What
pays it is this test: it regenerates the module from the parquets and asserts the
result is byte-identical to the committed file.  Whether a number is trustworthy
does not depend on how it was produced -- it depends on whether a link exists
that breaks when the two sides diverge.  The module docstring records provenance;
this is the link.

It fails in both directions on purpose: hand-editing the case module fails here,
and so does regenerating the parquets without regenerating the module.
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools/case_audit"
TARGET = REPO / "powermarketjax/case/cases/distribution/case459_0.py"


@pytest.fixture(scope="module")
def rendered():
    sys.path.insert(0, str(TOOLS))
    try:
        import generate_case459_0
        return generate_case459_0.render()
    finally:
        sys.path.remove(str(TOOLS))


def test_the_committed_case_module_is_what_the_parquets_generate(rendered):
    committed = TARGET.read_text()
    if committed == rendered:
        return
    # a diff worth reading, rather than 31 kB of "not equal"
    c, r = committed.splitlines(), rendered.splitlines()
    first = next((i for i, (a, b) in enumerate(zip(c, r)) if a != b), min(len(c), len(r)))
    raise AssertionError(
        f"{TARGET.name} is not what the parquets generate.\n"
        f"  committed {len(c)} lines, regenerated {len(r)} lines; "
        f"first difference at line {first + 1}\n"
        f"  committed:   {c[first] if first < len(c) else '<eof>'}\n"
        f"  regenerated: {r[first] if first < len(r) else '<eof>'}\n"
        f"  regenerate with: PYTHONPATH=. python "
        f"tools/case_audit/generate_case459_0.py --write")


def test_the_generator_does_not_read_the_file_it_is_checked_against():
    """Guard against the link becoming circular.

    If `render` ever consulted the committed module -- to carry a value it could
    not derive, say -- the test above would be comparing the file with itself and
    would pass for any content at all.  `render` must build only from the
    parquets; `TARGET` belongs to the write/check plumbing.
    """
    import inspect
    sys.path.insert(0, str(TOOLS))
    try:
        import generate_case459_0
        body = inspect.getsource(generate_case459_0.render)
    finally:
        sys.path.remove(str(TOOLS))
    assert "TARGET" not in body
    assert "SwissDN_459_0_MV_Nodes.parquet" in body
    assert "SwissDN_459_0_MV_Edges.parquet" in body
