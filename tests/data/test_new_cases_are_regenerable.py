# Written for this repository on 2026-08-24 -- no upstream counterpart.
"""The link between `case73rts` / `case813nem` and the CSVs they came from.

`powermarketjax/case/` is self-contained: no case file reads `data/`.  So both of
these cases carry their node, unit, line and load tables inline, which means the
same numbers live in this repository twice -- once in the vendored CSVs under
`case/raw_cases/`, once in the generated module under `case/cases/transmission/`.

**A converter does not pay that cost, because nobody runs a converter.**  What
pays it is this test.  `case459_0` has had that link since 2026-08-15
(`test_case459_0_is_regenerable.py`); these two cases were added without one, so
until now they were the two largest generated files in the tree that nothing held
to their sources.

The header of each generated module is produced by the `HEADER` template inside
its converter, not written by hand.  That is the specific failure this catches:
editing the generated file -- including tidying its comments -- forks it from the
template silently, because the numbers still parse and every other test still
passes.  It fails in the other direction too: regenerating the CSVs without
regenerating the module fails here.

Both converters are invoked through `main`, not through their internals, so what
is compared is what the documented command produces:

    python -m powermarketjax.case.raw_cases.rts_gmlc_to_case
    python -m powermarketjax.case.raw_cases.egrimod_nem_to_case

Scope, stated rather than implied: this test compares text, so it catches a
divergence between converter and committed module.  It says nothing about whether
either matches the upstream dataset -- that is what the `LICENCE.md` and the
provenance prose in each `HEADER` record -- and nothing about whether the case is
electrically sensible, which is `tests/case/test_cases.py`.
"""
import importlib
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[2]

#: (converter module, committed module) for each case generated from vendored CSVs.
CASES = [
    ("powermarketjax.case.raw_cases.rts_gmlc_to_case",
     "powermarketjax/case/cases/transmission/case73rts.py"),
    ("powermarketjax.case.raw_cases.egrimod_nem_to_case",
     "powermarketjax/case/cases/transmission/case813nem.py"),
]

#: The CSV basenames each converter must be seen to read.  Asserting on these is
#: what keeps the anti-circularity check below honest: a `build_tables` that had
#: stopped reading the sources would still pass a "does not mention OUT_PATH"
#: check on its own.
RAW_INPUTS = {
    "powermarketjax.case.raw_cases.rts_gmlc_to_case":
        ("gen.csv", "branch.csv", "bus.csv"),
    "powermarketjax.case.raw_cases.egrimod_nem_to_case":
        ("network_nodes.csv", "network_edges.csv", "generators.csv"),
}


#: Explicit, because `ids=` is called once per argument rather than once per
#: tuple: a callable receives the module string and the path string separately,
#: and indexing either of them yields a single character.  The ids read `o-o0`
#: that way, and a failure that cannot name which case failed is the whole point
#: of having two.
CASE_IDS = ["case73rts", "case813nem"]


@pytest.mark.parametrize("converter,committed", CASES, ids=CASE_IDS)
def test_the_committed_case_module_is_what_the_csvs_generate(
        converter, committed, tmp_path):
    mod = importlib.import_module(converter)

    # The vendored CSVs are committed, so their absence is a broken checkout and
    # not a reason to skip.  A skip here would read like ordinary CI behaviour.
    missing = [n for n in RAW_INPUTS[converter] if not (mod.RAW_DIR / n).exists()]
    assert not missing, (
        f"{converter} cannot run: {mod.RAW_DIR} is missing {missing}. These CSVs "
        f"are committed; a checkout without them is broken, not unsupported.")

    out = tmp_path / Path(committed).name
    with redirect_stdout(io.StringIO()):
        with mock.patch.object(sys, "argv", [converter, "--out", str(out)]):
            mod.main()

    regenerated = out.read_text()
    target = REPO / committed
    if target.read_text() == regenerated:
        return

    # a diff worth reading, rather than 100 kB of "not equal"
    c, r = target.read_text().splitlines(), regenerated.splitlines()
    first = next((i for i, (a, b) in enumerate(zip(c, r)) if a != b),
                 min(len(c), len(r)))
    raise AssertionError(
        f"{Path(committed).name} is not what the CSVs generate.\n"
        f"  committed {len(c)} lines, regenerated {len(r)} lines; "
        f"first difference at line {first + 1}\n"
        f"  committed:   {c[first] if first < len(c) else '<eof>'}\n"
        f"  regenerated: {r[first] if first < len(r) else '<eof>'}\n"
        f"  regenerate with: python -m {converter}")


@pytest.mark.parametrize("converter,committed", CASES, ids=CASE_IDS)
def test_the_converter_does_not_read_the_file_it_is_checked_against(
        converter, committed):
    """Guard against the link becoming circular.

    If `build_tables` ever consulted the committed module -- to carry a value it
    could not derive from the CSVs, say -- the test above would be comparing the
    file with itself and would pass for any content at all.  `build_tables` must
    build only from the CSVs; `OUT_PATH` belongs to the write plumbing in `main`,
    which is why the check is on `build_tables` and not on the module.
    """
    import inspect
    mod = importlib.import_module(converter)
    body = inspect.getsource(mod.build_tables)

    assert "OUT_PATH" not in body, (
        f"{converter}.build_tables mentions OUT_PATH, so the comparison above "
        f"may be reading the committed file back into its own expected value.")
    assert Path(committed).name not in body, (
        f"{converter}.build_tables names {Path(committed).name}.")
    for csv in RAW_INPUTS[converter]:
        assert csv in body, (
            f"{converter}.build_tables no longer reads {csv}; either the input "
            f"set changed, in which case RAW_INPUTS here is stale, or the "
            f"converter stopped deriving the table from its source.")
