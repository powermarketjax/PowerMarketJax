# Vendored from PowerZooJax. Do not edit to track upstream;
# this file is now this repository's code.
# Source        : powerzoojax/data/registry.py
# Upstream commit: a7641de (2026-05-07)
# Copied on     : 2026-08-05
# Verbatim copy; only the package name changed (powerzoojax -> powermarketjax,
# in both import statements and path strings).
# Upstream licence: MIT, Copyright (c) 2026 PowerZooJax Contributors.
"""Dataset registry: discovers and indexes all manifest files."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .manifest import DatasetManifest


class DatasetRegistry:
    """Load all manifest JSONs and build a reverse index from signal -> datasets."""

    def __init__(self, manifest_dir: Optional[Path] = None):
        if manifest_dir is None:
            manifest_dir = Path(__file__).resolve().parent / "manifests"
        self._manifest_dir = manifest_dir
        self._manifests: Dict[str, DatasetManifest] = {}
        self._signal_index: Dict[str, List[str]] = {}
        if self._manifest_dir.is_dir():
            self._load_manifests()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _load_manifests(self) -> None:
        for json_path in sorted(self._manifest_dir.glob("*.json")):
            try:
                m = DatasetManifest.from_json(json_path)
                self._manifests[m.name] = m
                for sig in m.signals:
                    self._signal_index.setdefault(sig, []).append(m.name)
            except Exception as exc:  # noqa: BLE001
                import warnings
                warnings.warn(
                    f"Failed to load manifest {json_path.name}: {exc}",
                    stacklevel=2,
                )

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def find_by_signal(
        self,
        signal: str,
        source: Optional[str] = None,
        data_type: Optional[str] = None,
    ) -> List[DatasetManifest]:
        """Return manifests that provide *signal*, optionally filtered."""
        names = self._signal_index.get(signal, [])
        results = [self._manifests[n] for n in names]
        if source is not None:
            results = [m for m in results if m.source == source]
        if data_type is not None:
            results = [m for m in results if m.data_type == data_type]
        return results

    def find_by_source(self, source: str) -> List[DatasetManifest]:
        return [m for m in self._manifests.values() if m.source == source]

    def list_signals(self) -> List[str]:
        return sorted(self._signal_index.keys())

    def list_sources(self) -> List[str]:
        return sorted({m.source for m in self._manifests.values()})

    def list_datasets(self) -> List[str]:
        return sorted(self._manifests.keys())

    def get_manifest(self, name: str) -> DatasetManifest:
        if name not in self._manifests:
            raise KeyError(
                f"Unknown dataset '{name}'. "
                f"Available: {self.list_datasets()}"
            )
        return self._manifests[name]

    def resolve_signals(
        self,
        signals: List[str],
        source: Optional[str] = None,
    ) -> Dict[str, DatasetManifest]:
        """Map each requested signal to the best matching manifest.

        Returns ``{signal: manifest}``.  Raises if any signal cannot be
        resolved.
        """
        result: Dict[str, DatasetManifest] = {}
        missing: list[str] = []
        for sig in signals:
            candidates = self.find_by_signal(sig, source=source)
            if not candidates:
                missing.append(sig)
            else:
                result[sig] = candidates[0]
        if missing:
            raise ValueError(
                f"Cannot resolve signals: {missing}. "
                f"Available signals: {self.list_signals()}"
            )
        return result
