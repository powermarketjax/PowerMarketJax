# Vendored test fixture from PowerZooJax.
# Source        : tests/conftest.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Trimmed to this repository's scope -- NOT verbatim.
#   Upstream's version also seeded sys.path for its `benchmarks/` namespace
#   package and for the sibling PowerZoo repo. Neither exists here, and
#   this repository does not depend on the sibling repo, so only the shared
#   `prng_key` fixture is kept.
"""Shared fixtures for PowerMarketJax tests."""

import json
from pathlib import Path

import jax
import pytest


@pytest.fixture
def prng_key():
    """Deterministic PRNG key for reproducible tests."""
    return jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# Datasets that are registered but not shipped.
#
# Some manifests under `powermarketjax/data/manifests/` name a parquet file
# (and its `.json` sidecar) whose licence does not allow redistribution, so a
# fresh checkout does not carry it; the manifest's preparation script under
# `tools/data_prep/` rebuilds it from the public source.  A test that needs
# such a file fails with a missing-file error deep inside a loader.  The hook
# below turns exactly that failure into a skip whose reason names the script,
# and nothing else: the file must be registered in a manifest AND absent from
# the data directory at the moment the report is made, so a present file that
# a loader fails to read still fails.

_DATA = Path(__file__).resolve().parents[1] / "powermarketjax" / "data"

#: Manifests whose preparation script does not follow the ``<name>.py`` rule.
_PREP_SCRIPT_OVERRIDES = {
    "aemo_nsw1_dispatch_price": "aemo_nsw1_price_and_rooftop_pv",
    "aemo_nsw1_rooftop_pv": "aemo_nsw1_price_and_rooftop_pv",
}

#: Manifests with no script that rebuilds the shipped file.  The Ausgrid
#: zone-substation series is the imputed release; the source publishes the raw
#: one, with about 2.5% of its values missing, so fetching it again does not
#: reproduce what the tests read.
_NOT_REPRODUCIBLE = {
    "ausgrid_zone_substation_fy25_imputed": (
        "Ausgrid zone-substation series is not redistributable (Ausgrid retains "
        "copyright) and its imputed release cannot be reproduced by a script; "
        "test skipped"),
}


def _registered_data_files():
    """``{file name: manifest name}`` for every parquet and sidecar a manifest names."""
    out = {}
    for m in sorted((_DATA / "manifests").glob("*.json")):
        raw = json.loads(m.read_text())
        pq = raw.get("parquet_file")
        if pq:
            name = raw.get("name", m.stem)
            out[Path(pq).name] = name
            out[Path(pq).with_suffix(".json").name] = name
    return out


_REGISTERED = _registered_data_files()


def missing_data_skip_reason(text):
    """The skip reason if ``text`` names a registered data file that is absent, else None."""
    for fname, name in _REGISTERED.items():
        if fname in text and not (_DATA / "parquet" / fname).exists():
            if name in _NOT_REPRODUCIBLE:
                return _NOT_REPRODUCIBLE[name]
            script = _PREP_SCRIPT_OVERRIDES.get(name, name)
            return (f"dataset '{name}' is not shipped ({fname} missing): "
                    f"run `python tools/data_prep/{script}.py` first")
    return None


def _as_skip(report, reason):
    report.outcome = "skipped"
    report.longrepr = (str(report.fspath), 0, f"Skipped: {reason}")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.failed and call.excinfo is not None:
        exc, text = call.excinfo.value, []
        while exc is not None:
            text.append(str(exc))
            text.append(str(getattr(exc, "filename", "") or ""))
            exc = exc.__cause__ or exc.__context__
        reason = missing_data_skip_reason("\n".join(text))
        if reason is not None:
            _as_skip(report, reason)


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    outcome = yield
    report = outcome.get_result()
    if report.failed:
        reason = missing_data_skip_reason(str(report.longrepr))
        if reason is not None:
            report.outcome = "skipped"
            report.longrepr = (str(collector.path), 0, f"Skipped: {reason}")
