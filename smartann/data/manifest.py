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
            self._entries = self._read_artifacts()

    def _read_artifacts(self) -> dict[str, dict[str, Any]]:
        """Artifacts currently on disk, tolerating a corrupt file.

        A manifest damaged by an interrupted write must not make the whole
        pipeline unrunnable: recording what we know beats recording nothing.
        The atomic write in :meth:`write` makes this rare, not impossible.
        """
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        artifacts = loaded.get("artifacts", {})
        return artifacts if isinstance(artifacts, dict) else {}

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

    def annotate(self, path: Path, extra: dict[str, Any]) -> dict[str, Any]:
        """Merge extra facts into an existing entry, keeping its checksum.

        Some properties of an artifact are only measured later -- duplicate
        counts need the whole array in memory, which the preparation pass has no
        reason to pay for. Re-recording would mean re-hashing a 488 MB file to
        add one integer.
        """
        rel = path.resolve().relative_to(self.data_root.resolve()).as_posix()
        entry = self._entries.get(rel)
        if entry is None:
            raise KeyError(f"{rel} has no manifest entry to annotate")
        entry["extra"] = {**entry.get("extra", {}), **extra}
        return entry

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

    def _self_describing_index_roots(self) -> set[str]:
        """Index roots that carry their own provenance record.

        An index is a build output, not a source artifact: it is reproducible
        from ``(config.yaml, seed)``, it is rebuilt routinely, and each root
        already stores a ``meta.json`` naming the config hash it was built
        from. Hashing its 13,154 constituent files into the data manifest would
        record the same provenance a second time and churn on every rebuild.

        The exclusion is conditional, not blanket. A root WITHOUT ``meta.json``
        is a half-written or orphaned build, and its files stay in the
        unrecorded list -- which is the one case where a stray index file
        actually means something is wrong.
        """
        roots: set[str] = set()
        index_dir = self.data_root / "index"
        if not index_dir.is_dir():
            return roots
        for dataset_dir in index_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            for root in dataset_dir.iterdir():
                if root.is_dir() and (root / "meta.json").is_file():
                    roots.add(root.relative_to(self.data_root).as_posix())
        return roots

    def unrecorded_files(self, patterns: tuple[str, ...] = ("**/*",)) -> list[str]:
        """Files under the data root with no manifest entry."""
        skip = {self.FILENAME, ".gitkeep", self.FILENAME + ".tmp"}
        described = self._self_describing_index_roots()
        out: list[str] = []
        for pattern in patterns:
            for p in self.data_root.glob(pattern):
                if not p.is_file() or p.name in skip:
                    continue
                rel = p.relative_to(self.data_root).as_posix()
                # checkpoints and caches are working files, not artifacts
                if rel.endswith((".part", ".ckpt.npz")) or "/cache/" in f"/{rel}":
                    continue
                if any(rel.startswith(f"{root}/") for root in described):
                    continue
                if rel not in self._entries:
                    out.append(rel)
        return sorted(out)

    def write(self) -> Path:
        """Merge our entries onto whatever is on disk, then write.

        Re-reading first is not belt-and-braces, it is required. Two dataset
        preparations running concurrently each construct a manifest, add their
        own artifacts, and write at the end; a plain overwrite means the later
        writer silently erases the earlier one's entries. That is exactly what
        happened when NYTimes and SQuAD were prepared in parallel -- NYTimes'
        four artifacts vanished from the manifest, and the unrecorded-files
        check caught it. Ours win for keys we touched; everything else on disk
        survives.
        """
        self.data_root.mkdir(parents=True, exist_ok=True)

        merged: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            merged.update(self._read_artifacts())
        merged.update(self._entries)
        self._entries = merged

        payload = {
            "schema": 1,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "artifacts": merged,
        }
        # Write-then-rename so a crash mid-write cannot leave a partial manifest.
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(self.path)
        return self.path
