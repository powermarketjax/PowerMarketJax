# Vendored test from PowerZooJax.
# Source        : tests/data/test_manifest.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
"""Tests for powermarketjax.data.manifest — DatasetManifest dataclass."""

import json
import tempfile
from pathlib import Path

import pytest

from powermarketjax.data.manifest import DatasetManifest


@pytest.fixture
def sample_manifest_dict():
    return {
        "name": "test_dataset",
        "source": "test",
        "data_type": "actual_series",
        "time_mode": "calendar",
        "resolution": "30min",
        "parquet_file": "test.parquet",
        "column_map": {"RawCol": "signal.name"},
        "index_map": {"startTime": "datetime"},
        "derived": {"wind.available_mw": "Wind Offshore + Wind Onshore"},
        "normalize": {"signal.name": 100.0},
        "data_epoch": "2024-01-01T00:00:00",
        "cyclical": False,
        "region_values": ["A", "B"],
        "date_range": ["2024-01-01", "2024-12-31"],
        "metadata_json": "test.json",
    }


@pytest.fixture
def sample_manifest(sample_manifest_dict):
    return DatasetManifest(
        name=sample_manifest_dict["name"],
        source=sample_manifest_dict["source"],
        data_type=sample_manifest_dict["data_type"],
        time_mode=sample_manifest_dict["time_mode"],
        resolution=sample_manifest_dict["resolution"],
        parquet_file=sample_manifest_dict["parquet_file"],
        column_map=sample_manifest_dict["column_map"],
        index_map=sample_manifest_dict["index_map"],
        derived=sample_manifest_dict["derived"],
        normalize=sample_manifest_dict["normalize"],
        data_epoch=sample_manifest_dict["data_epoch"],
        cyclical=sample_manifest_dict["cyclical"],
        region_values=sample_manifest_dict["region_values"],
        date_range=tuple(sample_manifest_dict["date_range"]),
        metadata_json=sample_manifest_dict["metadata_json"],
    )


class TestDatasetManifest:

    def test_signals_property(self, sample_manifest):
        sigs = sample_manifest.signals
        assert "signal.name" in sigs
        assert "wind.available_mw" in sigs
        assert len(sigs) == 2

    def test_raw_columns_needed(self, sample_manifest):
        cols = sample_manifest.raw_columns_needed
        assert "RawCol" in cols
        assert "startTime" in cols
        # Derived expression "Wind Offshore + Wind Onshore" is tokenized
        # by splitting on "+" then whitespace, yielding individual words.
        assert "Wind" in cols
        assert "Offshore" in cols
        assert "Onshore" in cols

    def test_from_json(self, sample_manifest_dict, tmp_path):
        json_path = tmp_path / "test.json"
        json_path.write_text(json.dumps(sample_manifest_dict))

        m = DatasetManifest.from_json(json_path)
        assert m.name == "test_dataset"
        assert m.source == "test"
        assert m.data_type == "actual_series"
        assert m.time_mode == "calendar"
        assert m.resolution == "30min"
        assert m.parquet_file == "test.parquet"
        assert m.column_map == {"RawCol": "signal.name"}
        assert m.cyclical is False
        assert m.date_range == ("2024-01-01", "2024-12-31")

    def test_to_dict_roundtrip(self, sample_manifest):
        d = sample_manifest.to_dict()
        assert d["name"] == "test_dataset"
        assert d["date_range"] == ["2024-01-01", "2024-12-31"]
        assert isinstance(d["column_map"], dict)

    def test_from_json_missing_date_range(self, tmp_path):
        data = {
            "name": "minimal",
            "source": "x",
            "data_type": "actual_series",
            "time_mode": "profile",
            "resolution": "300s",
            "parquet_file": "x.parquet",
        }
        path = tmp_path / "minimal.json"
        path.write_text(json.dumps(data))

        m = DatasetManifest.from_json(path)
        assert m.date_range is None
        assert m.column_map == {}
        assert m.cyclical is False

    def test_defaults(self):
        m = DatasetManifest(
            name="d", source="s", data_type="actual_series",
            time_mode="calendar", resolution="30min",
            parquet_file="d.parquet",
        )
        assert m.column_map == {}
        assert m.index_map == {}
        assert m.derived == {}
        assert m.normalize == {}
        assert m.region_values == []
        assert m.date_range is None
        assert m.metadata_json is None
        assert m.source_url is None
        assert m.source_urls is None
        assert m.source_organization is None
