"""Writing a run: queries.parquet plus manifest.json, and nothing else.

Every figure and every summary statistic in this project is derived from the
parquet and from nowhere else (Section 3, rule 1). That only holds if the writer
is faithful, so this module does the minimum: flatten ``QueryRecord`` to columns,
write once at the end of the run, and refuse to write a manifest that claims a
measurement the platform cannot make.

Deliberately not general. There is no schema migration, no partitioning, no
append mode. A run writes one file.
"""

from __future__ import annotations

import dataclasses
import json
import platform
import time
from pathlib import Path
from typing import Any

from smartann.telemetry.monitor import CapabilityTable, assert_manifest_consistent
from smartann.types import QueryRecord

__all__ = ["RunWriter", "flatten_record"]

# Nested dataclasses are flattened with these prefixes so the parquet stays a
# flat table: a column per scalar, no struct columns to unpack at plot time.
_NESTED = {"config": "cfg", "device": "dev"}


def flatten_record(rec: QueryRecord) -> dict[str, Any]:
    """One QueryRecord -> one flat row.

    Lists (retrieved ids and distances) are kept as lists; pyarrow stores them
    natively and score-equality recall has to be recomputable from the row
    without reloading the index.
    """
    row: dict[str, Any] = {}
    for f in dataclasses.fields(rec):
        value = getattr(rec, f.name)
        if f.name == "config":
            for sub in ("ann", "scr"):
                inner = getattr(value, sub)
                for g in dataclasses.fields(inner):
                    row[f"cfg_{sub}_{g.name}"] = getattr(inner, g.name)
        elif f.name == "device":
            for g in dataclasses.fields(value):
                row[f"dev_{g.name}"] = getattr(value, g.name)
        else:
            row[f.name] = value
    return row


class RunWriter:
    """Collects records in memory, writes once at the end.

    In-memory is a deliberate limit, not an oversight: these runs are thousands
    of queries, not millions, and a streaming writer would add a flush policy
    that could silently truncate a run on a crash. If a run ever outgrows
    memory, that is the point to add streaming and say so.
    """

    def __init__(
        self,
        run_dir: str | Path,
        run_id: str,
        experiment_id: str,
        capabilities: CapabilityTable,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.capabilities = capabilities
        self.records: list[QueryRecord] = []
        self.events: list[dict[str, Any]] = []
        self._started = time.time()

    def append(self, record: QueryRecord) -> None:
        self.records.append(record)

    def event(self, name: str, **detail: Any) -> None:
        """Mark something that happened during the run, with its wall-clock time.

        Used for the contention window so a figure can shade exactly the region
        the burners were running, rather than a region inferred from the latency
        trace it is supposed to be evidence for.
        """
        self.events.append({"name": name, "t_wall": time.time(), **detail})

    def write(self, extra: dict[str, Any] | None = None) -> tuple[Path, Path]:
        import pandas as pd

        from smartann.bench.manifest import git_state, library_versions

        if not self.records:
            raise ValueError(f"run {self.run_id} has no records; refusing to write an empty run")

        self.run_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([flatten_record(r) for r in self.records])
        parquet = self.run_dir / "queries.parquet"
        df.to_parquet(parquet, index=False)

        manifest: dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "n_queries": len(self.records),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._started)),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "wall_seconds": time.time() - self._started,
            "hostname": platform.node(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git": git_state(),
            "libraries": library_versions(),
            "capabilities": self.capabilities.as_dict(),
            "events": self.events,
            # Fields the platform cannot measure are absent here, never zero.
            "measured": dict.fromkeys(self.capabilities.unavailable_fields()),
            **(extra or {}),
        }
        # The gate (ADR-0020): a manifest cannot claim a measurement the
        # capability table says is unavailable.
        assert_manifest_consistent(manifest, self.capabilities)

        manifest_path = self.run_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8")
        return parquet, manifest_path
