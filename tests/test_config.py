"""Phase 0 acceptance tests: config round-trip, hash determinism, validation."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from smartann.config import (
    CONFIG_HASH_LENGTH,
    ConfigError,
    ConstraintsModel,
    canonical_dict,
    config_hash,
    dump_yaml,
    load_yaml,
    resolve_paths,
    validate_ann_config,
    validate_pipeline_config,
    validate_scr_config,
)
from smartann.types import AnnConfig, PipelineConfig, ScrConfig


def make_ann(**kw: object) -> AnnConfig:
    base = {"nprobe": 4, "ef_centroid": 64, "ef_list": 64, "k": 5, "per_cluster_k": 20}
    base.update(kw)
    return AnnConfig(**base)  # type: ignore[arg-type]


def make_scr(**kw: object) -> ScrConfig:
    base = {
        "sliding_window_size": 3,
        "overlap_size": 2,
        "context_extension_size": 1,
        "top_windows": 1,
        "similarity_threshold": 0.0,
        "enabled": True,
    }
    base.update(kw)
    return ScrConfig(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("obj", "target"),
    [
        (make_ann(), AnnConfig),
        (make_scr(), ScrConfig),
        (PipelineConfig(ann=make_ann(), scr=make_scr()), PipelineConfig),
    ],
)
def test_yaml_round_trip_is_identity(obj: object, target: type, tmp_path: Path) -> None:
    """config -> YAML -> config must be the identity (Phase 0 acceptance)."""
    path = tmp_path / "cfg.yaml"
    dump_yaml(obj, path)
    restored = load_yaml(path, target)
    assert restored == obj
    assert config_hash(restored) == config_hash(obj)


def test_round_trip_survives_two_generations(tmp_path: Path) -> None:
    cfg = PipelineConfig(ann=make_ann(), scr=make_scr(similarity_threshold=0.37))
    first = tmp_path / "a.yaml"
    second = tmp_path / "b.yaml"
    dump_yaml(cfg, first)
    once = load_yaml(first, PipelineConfig)
    dump_yaml(once, second)
    twice = load_yaml(second, PipelineConfig)
    assert twice == cfg
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def test_hash_shape() -> None:
    h = config_hash(make_ann())
    assert len(h) == CONFIG_HASH_LENGTH
    assert all(c in "0123456789abcdef" for c in h)


def test_hash_is_key_order_independent() -> None:
    a = {"alpha": 1, "beta": {"x": 1, "y": 2}}
    b = {"beta": {"y": 2, "x": 1}, "alpha": 1}
    assert config_hash(a) == config_hash(b)


def test_hash_changes_with_any_field() -> None:
    base = make_ann()
    for fieldname, newvalue in [
        ("nprobe", 8),
        ("ef_centroid", 128),
        ("ef_list", 32),
        ("k", 10),
        ("per_cluster_k", 50),
    ]:
        assert config_hash(make_ann(**{fieldname: newvalue})) != config_hash(base), fieldname


def test_hash_distinguishes_bool_from_int() -> None:
    """True == 1 in Python; the canonical form must not collapse them."""
    assert config_hash({"v": True}) != config_hash({"v": 1})


def test_hash_rejects_nan() -> None:
    """NaN has no canonical JSON form, so hashing it must fail loudly."""
    with pytest.raises(ValueError):
        config_hash({"v": float("nan")})


def test_hash_is_stable_across_processes(tmp_path: Path) -> None:
    """The same config must hash identically in a fresh interpreter.

    Run with an explicit non-zero PYTHONHASHSEED: this is what catches an
    accidental dependency on Python's salted ``hash()`` or on set iteration
    order, which would silently break reproducibility across runs.
    """
    script = tmp_path / "hash_child.py"
    script.write_text(
        textwrap.dedent(
            """
            from smartann.config import config_hash
            from smartann.types import AnnConfig, PipelineConfig, ScrConfig

            cfg = PipelineConfig(
                ann=AnnConfig(nprobe=4, ef_centroid=64, ef_list=64, k=5, per_cluster_k=20),
                scr=ScrConfig(
                    sliding_window_size=3,
                    overlap_size=2,
                    context_extension_size=1,
                    top_windows=1,
                    similarity_threshold=0.0,
                ),
            )
            print(config_hash(cfg))
            """
        ),
        encoding="utf-8",
    )
    expected = config_hash(PipelineConfig(ann=make_ann(), scr=make_scr()))

    seen = set()
    for seed in ("0", "1", "12345"):
        proc = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            check=True,
            env={**_child_env(), "PYTHONHASHSEED": seed},
        )
        seen.add(proc.stdout.strip())

    assert seen == {expected}, f"hash drifted across processes: {seen} vs {expected}"


def _child_env() -> dict[str, str]:
    import os

    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_canonical_dict_is_json_serialisable() -> None:
    cfg = PipelineConfig(ann=make_ann(), scr=make_scr())
    json.dumps(canonical_dict(cfg))  # must not raise


def test_paths_hash_is_platform_independent() -> None:
    """Windows and posix spellings of the same root must hash alike."""
    assert config_hash({"p": Path("C:/a/b")}) == config_hash({"p": Path("C:\\a\\b")})


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_valid_configs_pass() -> None:
    cfg = PipelineConfig(ann=make_ann(), scr=make_scr())
    assert validate_pipeline_config(cfg) is cfg


@pytest.mark.parametrize(
    ("kwargs", "needle"),
    [
        ({"nprobe": 0}, "nprobe"),
        ({"ef_list": 0}, "ef_list"),
        ({"k": 0}, "k"),
        ({"k": 10, "per_cluster_k": 5}, "per_cluster_k"),
        ({"nprobe": 32, "ef_centroid": 16}, "ef_centroid"),
    ],
)
def test_invalid_ann_config_raises(kwargs: dict[str, object], needle: str) -> None:
    with pytest.raises(ConfigError, match=needle):
        validate_ann_config(make_ann(**kwargs))


@pytest.mark.parametrize(
    ("kwargs", "needle"),
    [
        ({"sliding_window_size": 0}, "sliding_window_size"),
        ({"overlap_size": -1}, "overlap_size"),
        ({"sliding_window_size": 2, "overlap_size": 2}, "never advance"),
        ({"top_windows": 0}, "top_windows"),
        ({"similarity_threshold": 1.5}, "similarity_threshold"),
    ],
)
def test_invalid_scr_config_raises(kwargs: dict[str, object], needle: str) -> None:
    with pytest.raises(ConfigError, match=needle):
        validate_scr_config(make_scr(**kwargs))


def test_constraints_reject_hard_below_target() -> None:
    with pytest.raises(ValidationError):
        ConstraintsModel(latency_target_ms=100.0, latency_hard_ms=50.0, energy_budget_mj=1.0)


def test_constraints_to_dataclass() -> None:
    m = ConstraintsModel(
        latency_target_ms=100.0, latency_hard_ms=250.0, energy_budget_mj=2.0, min_recall=0.9
    )
    dc = m.to_dataclass()
    assert dc.latency_target_ms == 100.0
    assert dc.min_recall == 0.9


def test_load_yaml_rejects_empty(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        load_yaml(p, AnnConfig)


def test_load_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_yaml(p, AnnConfig)


def test_load_yaml_rejects_unknown_field(tmp_path: Path) -> None:
    """An unknown key must fail, not be silently dropped.

    pydantic's TypeAdapter ignores extra keys on stdlib dataclasses by default,
    which would let a typo change the effective config while the recorded hash
    still looked correct.
    """
    p = tmp_path / "extra.yaml"
    p.write_text(
        "nprobe: 4\nef_centroid: 64\nef_list: 64\nk: 5\nper_cluster_k: 20\nbogus: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="bogus"):
        load_yaml(p, AnnConfig)


def test_load_yaml_suggests_near_miss(tmp_path: Path) -> None:
    p = tmp_path / "typo.yaml"
    p.write_text(
        "nprobe: 4\nef_centroid: 64\nef_lst: 64\nk: 5\nper_cluster_k: 20\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="did you mean 'ef_list'"):
        load_yaml(p, AnnConfig)


def test_load_yaml_rejects_unknown_nested_field(tmp_path: Path) -> None:
    """The check must recurse: a typo inside ann: is just as damaging."""
    p = tmp_path / "nested.yaml"
    p.write_text(
        textwrap.dedent(
            """
            ann:
              nprobe: 4
              ef_centroid: 64
              ef_list: 64
              k: 5
              per_cluster_k: 20
              ef_lst: 128
            scr:
              sliding_window_size: 3
              overlap_size: 2
              context_extension_size: 1
              top_windows: 1
              similarity_threshold: 0.0
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=r"ann\.'ef_lst'"):
        load_yaml(p, PipelineConfig)


def test_load_yaml_accepts_valid_nested(tmp_path: Path) -> None:
    p = tmp_path / "ok.yaml"
    expected = PipelineConfig(ann=make_ann(), scr=make_scr())
    dump_yaml(expected, p)
    assert load_yaml(p, PipelineConfig) == expected


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def test_paths_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMARTANN_DATA_ROOT", "D:/elsewhere")
    p = resolve_paths()
    assert p.data_root == Path("D:/elsewhere")
    assert p.processed == Path("D:/elsewhere/processed")


def test_paths_explicit_override_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMARTANN_DATA_ROOT", "D:/from-env")
    p = resolve_paths({"data_root": Path("E:/explicit")})
    assert p.data_root == Path("E:/explicit")
