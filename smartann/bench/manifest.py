"""Run manifests: the reproducibility record required by Section 3, rule 4.

Every run writes a ``manifest.json`` capturing git SHA, config hash, interpreter
and library versions, hostname, seed, and wall-clock start/end. A result whose
manifest is missing or dirty is not a result.
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

from smartann.config import canonical_dict, config_hash

__all__ = ["Manifest", "git_state", "library_versions"]

# Libraries whose exact version can change a numeric result.
_TRACKED_LIBRARIES = (
    "numpy",
    "scipy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "hnswlib",
    "sentence-transformers",
    "llama-cpp-python",
    "pydantic",
    "psutil",
)


def library_versions(names: tuple[str, ...] = _TRACKED_LIBRARIES) -> dict[str, str]:
    """Installed version of each tracked library, or "not-installed"."""
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not-installed"
    return out


def git_state(repo: Path | None = None) -> dict[str, Any]:
    """Current git SHA, branch and dirty flag.

    ``dirty=True`` means the working tree had uncommitted changes when the run
    started, so the SHA alone does not identify the code that produced it. This
    is recorded rather than raised, but it should be treated as disqualifying
    for any number that goes into the paper.
    """
    cwd = repo or Path(__file__).resolve().parents[2]

    def _git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "sha": sha,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "available": sha is not None,
    }


@dataclass
class Manifest:
    """Reproducibility record for one run.

    Usage::

        m = Manifest.start(experiment_id="E2", run_id=..., seed=0, config=cfg)
        ...
        m.finish().write(out_dir)
    """

    experiment_id: str
    run_id: str
    seed: int
    config: Any
    config_hash: str
    git: dict[str, Any]
    hostname: str
    python_version: str
    platform_: str
    libraries: dict[str, str]
    wall_clock_start: float
    wall_clock_start_iso: str
    wall_clock_end: float | None = None
    wall_clock_end_iso: str | None = None
    duration_s: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def start(
        cls,
        experiment_id: str,
        run_id: str,
        seed: int,
        config: Any,
        extra: dict[str, Any] | None = None,
    ) -> Manifest:
        now = time.time()
        return cls(
            experiment_id=experiment_id,
            run_id=run_id,
            seed=seed,
            config=canonical_dict(config),
            config_hash=config_hash(config),
            git=git_state(),
            hostname=socket.gethostname(),
            python_version=sys.version,
            platform_=platform.platform(),
            libraries=library_versions(),
            wall_clock_start=now,
            wall_clock_start_iso=_iso(now),
            extra=extra or {},
        )

    def finish(self) -> Manifest:
        """Stamp the end time. Returns self so it can be chained with ``write``."""
        now = time.time()
        self.wall_clock_end = now
        self.wall_clock_end_iso = _iso(now)
        self.duration_s = now - self.wall_clock_start
        return self

    def to_dict(self) -> dict[str, Any]:
        d = canonical_dict(self)
        # ``platform_`` is only underscored to dodge the stdlib module name
        d["platform"] = d.pop("platform_")
        return dict(d)

    def write(self, out_dir: str | Path) -> Path:
        """Write ``manifest.json`` into *out_dir* and return its path."""
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        target = path / "manifest.json"
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts)) + "Z"
