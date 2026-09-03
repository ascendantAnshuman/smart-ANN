"""Config loading, validation and deterministic hashing.

Design note (ADR-0002): the Section 6 contracts live in :mod:`smartann.types` as
plain frozen dataclasses with no pydantic dependency. This module validates
*those* dataclasses through ``pydantic.TypeAdapter`` instead of redeclaring
their fields in parallel ``BaseModel`` classes. That keeps a single source of
truth for field names and types; the value-range rules that a dataclass cannot
express live here as explicit check functions.
"""

from __future__ import annotations

import dataclasses
import difflib
import hashlib
import json
import os
from pathlib import Path
from typing import Any, TypeVar, get_args, get_type_hints

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from smartann.types import AnnConfig, Constraints, PipelineConfig, ScrConfig

__all__ = [
    "CONFIG_HASH_LENGTH",
    "ConfigError",
    "ConstraintsModel",
    "PathsConfig",
    "canonical_dict",
    "config_hash",
    "dump_yaml",
    "load_yaml",
    "resolve_paths",
    "validate_ann_config",
    "validate_pipeline_config",
    "validate_scr_config",
]

CONFIG_HASH_LENGTH = 12

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when a config is structurally valid but semantically impossible."""


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
class PathsConfig(BaseModel):
    """Where bulk artefacts live.

    Defaults point *outside* the source tree because this repository sits in a
    OneDrive-synced folder: background sync contends for I/O and CPU during
    timed benchmark loops, which would corrupt exactly the latency numbers the
    project exists to measure (ADR-0003). Override with the SMARTANN_DATA_ROOT
    and SMARTANN_RESULTS_ROOT environment variables, or in YAML.
    """

    model_config = ConfigDict(extra="forbid")

    data_root: Path = Field(default=Path("C:/smartann-data"))
    results_root: Path = Field(default=Path("C:/smartann-results"))

    @property
    def raw(self) -> Path:
        return self.data_root / "raw"

    @property
    def processed(self) -> Path:
        return self.data_root / "processed"

    @property
    def index(self) -> Path:
        return self.data_root / "index"

    @property
    def groundtruth(self) -> Path:
        return self.data_root / "groundtruth"

    @property
    def pareto(self) -> Path:
        return self.data_root / "pareto"


def resolve_paths(overrides: dict[str, Any] | None = None) -> PathsConfig:
    """Build a PathsConfig from (defaults <- environment <- explicit overrides)."""
    values: dict[str, Any] = {}
    env_data = os.environ.get("SMARTANN_DATA_ROOT")
    if env_data:
        values["data_root"] = Path(env_data)
    env_results = os.environ.get("SMARTANN_RESULTS_ROOT")
    if env_results:
        values["results_root"] = Path(env_results)
    if overrides:
        values.update(overrides)
    return PathsConfig(**values)


# ---------------------------------------------------------------------------
# Canonicalisation and hashing
# ---------------------------------------------------------------------------
def canonical_dict(obj: Any) -> Any:
    """Recursively convert *obj* into JSON-safe primitives with a stable shape.

    Dataclasses, pydantic models, mappings, sequences, Paths and sets are all
    reduced to dicts/lists/scalars. Dict keys are coerced to ``str`` so that
    ``json.dumps(sort_keys=True)`` can order them deterministically.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: canonical_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, BaseModel):
        return {k: canonical_dict(v) for k, v in obj.model_dump().items()}
    if isinstance(obj, dict):
        return {str(k): canonical_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [canonical_dict(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(canonical_dict(v) for v in obj)
    if isinstance(obj, Path):
        # forward slashes, so a hash does not change between Windows and Linux
        return obj.as_posix()
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    return str(obj)


def config_hash(obj: Any, length: int = CONFIG_HASH_LENGTH) -> str:
    """Deterministic short hash of any config object.

    Sorted-key JSON -> SHA256 -> first ``length`` hex chars (Section 7, Phase 0).
    Stable across processes and platforms: it never touches Python's salted
    ``hash()``, and Paths are normalised to posix form.
    """
    payload = json.dumps(
        canonical_dict(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------
def _is_structured(target: Any) -> bool:
    """True if *target* is a dataclass or pydantic model whose fields we can enumerate."""
    if isinstance(target, type) and issubclass(target, BaseModel):
        return True
    return dataclasses.is_dataclass(target) and isinstance(target, type)


def _field_types(target: type) -> dict[str, Any]:
    """Resolved ``{field_name: annotation}`` for a dataclass or pydantic model.

    ``from __future__ import annotations`` makes dataclass annotations strings,
    so they are resolved through ``get_type_hints`` against the defining module.
    """
    if isinstance(target, type) and issubclass(target, BaseModel):
        return {name: f.annotation for name, f in target.model_fields.items()}
    if not dataclasses.is_dataclass(target):
        return {}
    try:
        hints = get_type_hints(target)
    except Exception:  # pragma: no cover - unresolvable forward refs
        hints = {}
    return {f.name: hints.get(f.name, Any) for f in dataclasses.fields(target)}


def _structured_member(annotation: Any) -> type | None:
    """The single dataclass/model type in *annotation*, unwrapping ``X | None``."""
    if _is_structured(annotation):
        return annotation  # type: ignore[no-any-return]
    candidates = [arg for arg in get_args(annotation) if _is_structured(arg)]
    return candidates[0] if len(candidates) == 1 else None


def _reject_unknown_keys(
    data: dict[str, Any], target: type, path: str | Path, prefix: str = ""
) -> None:
    """Fail loudly on keys *target* does not declare, recursing into nested models.

    ``pydantic.TypeAdapter`` over a stdlib dataclass silently *drops* unknown
    keys, and it refuses an ``extra="forbid"`` config for such types. That
    default is a reproducibility hazard here: a typo like ``ef_lst: 128`` would
    be discarded, the run would proceed at the default effort, and the recorded
    config hash would not match what the author believed they ran. So the check
    is done explicitly, at every nesting level, with a suggestion for near-misses.
    """
    if not _is_structured(target):
        return
    types = _field_types(target)
    known = set(types)

    unknown = sorted(set(data) - known)
    if unknown:
        hints = []
        for key in unknown:
            close = difflib.get_close_matches(key, sorted(known), n=1, cutoff=0.7)
            hints.append(f"{prefix}{key!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        raise ConfigError(
            f"{path}: unknown field(s) for {target.__name__}: {', '.join(hints)}. "
            f"Known fields: {', '.join(sorted(known))}"
        )

    for key, value in data.items():
        nested = _structured_member(types[key])
        if nested is not None and isinstance(value, dict):
            _reject_unknown_keys(value, nested, path, prefix=f"{prefix}{key}.")


def load_yaml(path: str | Path, target: type[T]) -> T:
    """Load *path* and validate it into *target* (a dataclass or pydantic model).

    Unknown keys are an error, not a warning -- see :func:`_reject_unknown_keys`.
    """
    raw = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if data is None:
        raise ConfigError(f"{path} is empty")
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping, got {type(data).__name__}")
    _reject_unknown_keys(data, target, path)
    validated: T = TypeAdapter(target).validate_python(data)
    return validated


def dump_yaml(obj: Any, path: str | Path | None = None) -> str:
    """Serialise *obj* to YAML. Writes to *path* when given; always returns the text."""
    text: str = yaml.safe_dump(
        canonical_dict(obj),
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=True,
    )
    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------------
# Semantic validation (ranges the dataclasses cannot express)
# ---------------------------------------------------------------------------
def validate_ann_config(cfg: AnnConfig) -> AnnConfig:
    """Check an AnnConfig is physically meaningful. Returns it unchanged."""
    if cfg.nprobe < 1:
        raise ConfigError(f"nprobe must be >= 1, got {cfg.nprobe}")
    if cfg.ef_centroid < 1:
        raise ConfigError(f"ef_centroid must be >= 1, got {cfg.ef_centroid}")
    if cfg.ef_list < 1:
        raise ConfigError(f"ef_list must be >= 1, got {cfg.ef_list}")
    if cfg.k < 1:
        raise ConfigError(f"k must be >= 1, got {cfg.k}")
    if cfg.rerank_candidates < cfg.k:
        raise ConfigError(
            f"rerank_candidates ({cfg.rerank_candidates}) must be >= k ({cfg.k}): "
            "the rerank pool cannot be smaller than the result set"
        )
    if cfg.ef_centroid < cfg.nprobe:
        raise ConfigError(
            f"ef_centroid ({cfg.ef_centroid}) must be >= nprobe ({cfg.nprobe}): "
            "hnswlib cannot return more neighbours than ef"
        )
    return cfg


def validate_scr_config(cfg: ScrConfig) -> ScrConfig:
    """Check a ScrConfig is physically meaningful. Returns it unchanged."""
    if cfg.sliding_window_size < 1:
        raise ConfigError(f"sliding_window_size must be >= 1, got {cfg.sliding_window_size}")
    if cfg.overlap_size < 0:
        raise ConfigError(f"overlap_size must be >= 0, got {cfg.overlap_size}")
    if cfg.overlap_size >= cfg.sliding_window_size:
        raise ConfigError(
            f"overlap_size ({cfg.overlap_size}) must be < sliding_window_size "
            f"({cfg.sliding_window_size}): the window would never advance"
        )
    if cfg.context_extension_size < 0:
        raise ConfigError(f"context_extension_size must be >= 0, got {cfg.context_extension_size}")
    if cfg.top_windows < 1:
        raise ConfigError(f"top_windows must be >= 1, got {cfg.top_windows}")
    if not 0.0 <= cfg.similarity_threshold <= 1.0:
        raise ConfigError(f"similarity_threshold must be in [0, 1], got {cfg.similarity_threshold}")
    return cfg


def validate_pipeline_config(cfg: PipelineConfig) -> PipelineConfig:
    """Validate both halves of a pipeline decision."""
    validate_ann_config(cfg.ann)
    validate_scr_config(cfg.scr)
    return cfg


class ConstraintsModel(BaseModel):
    """Validated view of :class:`smartann.types.Constraints`."""

    model_config = ConfigDict(extra="forbid")

    latency_target_ms: float = Field(gt=0)
    latency_hard_ms: float = Field(gt=0)
    energy_budget_mj: float = Field(gt=0)
    min_recall: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("latency_hard_ms")
    @classmethod
    def _hard_above_target(cls, v: float, info: Any) -> float:
        target = info.data.get("latency_target_ms")
        if target is not None and v < target:
            raise ValueError(
                f"latency_hard_ms ({v}) must be >= latency_target_ms ({target}): "
                "a hard p99 bound below the soft SLO is unsatisfiable by construction"
            )
        return v

    def to_dataclass(self) -> Constraints:
        return Constraints(
            latency_target_ms=self.latency_target_ms,
            latency_hard_ms=self.latency_hard_ms,
            energy_budget_mj=self.energy_budget_mj,
            min_recall=self.min_recall,
        )
