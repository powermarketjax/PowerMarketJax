# Vendored test from PowerZooJax.
# Source        : tests/data/test_registry.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# NOT verbatim. Beyond the package rename (powerzoojax -> powermarketjax, in
# both import statements and path strings), six assertions were retargeted from
# the four datacenter datasets (google_dc_2019, alibaba_dc_2018,
# alibaba_gpu_2020, azure_dc_v2) to vendored ones. Those datasets are out of
# scope here (they belong to the datacenter line, not to any of the
# five markets) and are therefore not shipped. Tests touched:
# test_discovers_all_manifests, test_list_sources, test_list_signals,
# test_find_by_source, test_get_manifest, test_resolve_signals.
"""Tests for powermarketjax.data.registry — DatasetRegistry discovery."""

from pathlib import Path

import pytest

from powermarketjax.data.registry import DatasetRegistry
from powermarketjax.data import signals as S

MANIFEST_DIR = (
    Path(__file__).resolve().parents[2] / "powermarketjax/data/manifests"
)


@pytest.fixture(scope="module")
def registry():
    """Registry that reads the bundled manifests/ directory."""
    return DatasetRegistry(MANIFEST_DIR)


class TestDatasetRegistry:

    def test_discovers_all_manifests(self, registry):
        datasets = registry.list_datasets()
        manifest_files = sorted(MANIFEST_DIR.glob("*.json"))
        assert len(datasets) == len(manifest_files)
        assert {
            "gb_forecast_actual_demand",
            "gb_gen_by_type",
            "gb_market_mid",
            "aemo_5min_demand",
            "aemo_forecast",
            "ausgrid_zone_substation_fy25_imputed",
        }.issubset(set(datasets))

    def test_list_sources(self, registry):
        sources = registry.list_sources()
        assert "gb" in sources
        assert "aemo" in sources
        assert "ausgrid" in sources

    def test_list_signals(self, registry):
        sigs = registry.list_signals()
        assert S.LOAD_ACTUAL_MW in sigs
        assert S.SOLAR_AVAILABLE_MW in sigs
        assert S.WIND_AVAILABLE_MW in sigs
        assert S.MARKET_MID_PRICE_APX in sigs

    def test_find_by_signal(self, registry):
        results = registry.find_by_signal(S.LOAD_ACTUAL_MW)
        assert len(results) >= 1
        names = [m.name for m in results]
        assert any("demand" in n or "forecast" in n for n in names)

    def test_find_by_signal_with_source(self, registry):
        results = registry.find_by_signal(S.LOAD_ACTUAL_MW, source="gb")
        for m in results:
            assert m.source == "gb"

    def test_find_by_signal_with_data_type(self, registry):
        results = registry.find_by_signal(
            S.LOAD_ACTUAL_MW, data_type=S.FORECAST_PANEL,
        )
        for m in results:
            assert m.data_type == S.FORECAST_PANEL

    def test_find_by_source(self, registry):
        """What this asserts is the filter, not how many AEMO datasets exist.

        It read `len(results) == 2` until 2026-08-15, when registering the two
        NSW1 series turned it red at four.  The count was never the property
        under test: any dataset added under this source breaks it, and the
        breakage says nothing about `find_by_source`.  Restated as the filter's
        own invariants -- everything returned carries the source, nothing
        carrying the source is left out -- the test now moves with the registry
        instead of against it.
        """
        results = registry.find_by_source("aemo")
        names = {m.name for m in results}
        assert {"aemo_5min_demand", "aemo_forecast"} <= names
        assert all(m.source == "aemo" for m in results)
        assert names == {n for n in registry.list_datasets()
                         if registry.get_manifest(n).source == "aemo"}

    def test_get_manifest(self, registry):
        m = registry.get_manifest("aemo_5min_demand")
        assert m.source == "aemo"
        assert m.time_mode == "calendar"
        assert m.cyclical is False

    def test_get_manifest_unknown(self, registry):
        with pytest.raises(KeyError, match="Unknown dataset"):
            registry.get_manifest("nonexistent")

    def test_resolve_signals(self, registry):
        result = registry.resolve_signals([S.LOAD_ACTUAL_MW, S.WIND_AVAILABLE_MW])
        assert S.LOAD_ACTUAL_MW in result
        assert S.WIND_AVAILABLE_MW in result

    def test_resolve_signals_missing(self, registry):
        with pytest.raises(ValueError, match="Cannot resolve"):
            registry.resolve_signals(["nonexistent.signal"])

    def test_empty_registry(self, tmp_path):
        """Registry with empty dir has no datasets."""
        empty_dir = tmp_path / "empty_manifests"
        empty_dir.mkdir()
        r = DatasetRegistry(empty_dir)
        assert r.list_datasets() == []
        assert r.list_signals() == []
