"""``MANIFEST.json``: provenance for every artifact under the data root.

Distinct from :mod:`smartann.bench.manifest`, which records *runs*. This one
records *files*: where each came from, its SHA-256, its size, and when it was
prepared. Acceptance check 7 of Phase 1 is that no artifact exists without an
entry here.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

__all__ = ["ArtifactRecord", "DataManifest"]


@dataclass(frozen=True)
class ArtifactRecord:
    path: str  # relative to the data root, posix form
    dataset: str
    stage: str  # "raw" | "processed" | "groundtruth"
    source_url: str | None
    sha256: str
    n_bytes: int
    prepared_at: str
    extra: dict[str, Any]


class DataManifest:
    """Read/modify/write ``<data_root>/MANIFEST.json``."""

    FILENAME = "MANIFEST.json"

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root)
        self.path = self.data_root / self.FILENAME
        self._entries: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            self._entries = loaded.get("artifacts", {})

    def record(
        self,
        path: Path,
        dataset: str,
        stage: str,
        sha256: str,
        source_url: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        rel = path.resolve().relative_to(self.data_root.resolve()).as_posix()
        rec = ArtifactRecord(
            path=rel,
            dataset=dataset,
            stage=stage,
            source_url=source_url,
            sha256=sha256,
            n_bytes=path.stat().st_size,
            prepared_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            extra=extra or {},
        )
        self._entries[rel] = asdict(rec)
        return rec

    def get(self, path: Path) -> dict[str, Any] | None:
        rel = path.resolve().relative_to(self.data_root.resolve()).as_posix()
        return self._entries.get(rel)

    def __len__(self) -> int:
        return len(self._entries)

    def for_dataset(self, dataset: str) -> list[dict[str, Any]]:
        return [e for e in self._entries.values() if e["dataset"] == dataset]

    def total_bytes(self, dataset: str | None = None) -> int:
        entries = self.for_dataset(dataset) if dataset else list(self._entries.values())
        return sum(int(e["n_bytes"]) for e in entries)

    def missing_files(self) -> list[str]:
        """Recorded artifacts whose file is gone -- a stale manifest is a lie."""
        return [rel for rel in self._entries if not (self.data_root / rel).exists()]

    def unrecorded_files(self, patterns: tuple[str, ...] = ("**/*",)) -> list[str]:
        """Files under the data root with no manifest entry."""
        skip = {self.FILENAME, ".gitkeep"}
        out: list[str] = []
        for pattern in patterns:
            for p in self.data_root.glob(pattern):
                if not p.is_file() or p.name in skip:
                    continue
                rel = p.relative_to(self.data_root).as_posix()
                # checkpoints and caches are working files, not artifacts
                if rel.endswith((".part", ".ckpt.npz")) or "/cache/" in f"/{rel}":
                    continue
                if rel not in self._entries:
                    out.append(rel)
        return sorted(out)

    def write(self) -> Path:
        self.data_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": 1,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "artifacts": self._entries,
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return self.path
