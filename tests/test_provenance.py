"""What ran on the handset: the counts, and the claims that cannot be counted.

Two different things are tested here, for two different reasons.

The **counts** come from the recorded files, so they are tested the ordinary
way: build a small fake `results/device` and check the summary matches it.

The **stage table** is a constant -- a query record cannot tell you that no
embedder was imported -- so it is checked against the device scripts themselves.
This is the test that matters. If somebody later runs generation on the phone,
or switches the device trace to embed its own queries, the table in
`provenance.py` becomes a lie that nothing else would catch, because it is
printed into the viva notes and the dashboard as fact.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from smartann.telemetry.provenance import PIPELINE_STAGES, device_evidence

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
DEVICE_SCRIPTS = ("device_trace.py", "device_c5.py")


def _write_run(
    directory: Path,
    name: str,
    rows: list[dict[str, object]],
    minutes: float | None = None,
    summary: bool = False,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}_queries.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )
    if minutes is not None:
        key, suffix = ("minutes", "_summary.json") if summary else ("minutes_actual", "_meta.json")
        (directory / f"{name}{suffix}").write_text(json.dumps({key: minutes}), encoding="utf-8")


# ---------------------------------------------------------------------------
# The counts
# ---------------------------------------------------------------------------
def test_counts_and_durations_come_from_the_files(tmp_path: Path) -> None:
    _write_run(tmp_path, "trace", [{"nprobe": n, "t_total_ms": 1.0} for n in (1, 4, 8)], 25.0)
    _write_run(
        tmp_path,
        "c5",
        [{"nprobe": 4, "arm": "adaptive", "reasons": ["hot"], "recall_at_k": 0.8}] * 7,
        22.0,
        summary=True,
    )
    e = device_evidence(tmp_path)
    assert e.total_queries == 10
    assert e.total_minutes == pytest.approx(47.0)
    assert {r.name for r in e.runs} == {"trace", "c5"}


def test_a_run_that_recorded_policy_reasons_counts_as_controller_on_device(
    tmp_path: Path,
) -> None:
    """Read from the records, not guessed from the run's name: a record carrying
    `reasons` or `arm` is one where a policy chose that query's effort there."""
    _write_run(tmp_path, "sweep", [{"nprobe": 1}] * 3)
    _write_run(tmp_path, "control", [{"nprobe": 4, "reasons": ["temperature 55C"]}] * 5)
    e = device_evidence(tmp_path)
    assert e.total_queries == 8
    assert e.controller_queries == 5  # only the run that actually decided
    by_name = {r.name: r for r in e.runs}
    assert by_name["control"].ran_controller
    assert not by_name["sweep"].ran_controller


def test_recall_scored_on_device_is_reported_separately(tmp_path: Path) -> None:
    _write_run(tmp_path, "a", [{"nprobe": 4, "recall_at_k": 0.9}])
    _write_run(tmp_path, "b", [{"nprobe": 4}])
    by_name = {r.name: r for r in device_evidence(tmp_path).runs}
    assert by_name["a"].scored_recall
    assert not by_name["b"].scored_recall


def test_the_count_is_exact_even_beyond_the_parse_sample(tmp_path: Path) -> None:
    """Fields are sampled for speed; the query count is a claim about how much
    ran on the phone and is never sampled."""
    from smartann.telemetry.provenance import SAMPLE_LINES

    n = SAMPLE_LINES + 250
    _write_run(tmp_path, "big", [{"nprobe": 8}] * n)
    assert device_evidence(tmp_path).total_queries == n


def test_a_partial_last_line_does_not_break_the_scan(tmp_path: Path) -> None:
    """A run killed mid-write leaves one truncated record."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "x_queries.jsonl").write_text(
        '{"nprobe": 1}\n{"nprobe": 4}\n{"nprobe": 8, "t_tot', encoding="utf-8"
    )
    e = device_evidence(tmp_path)
    assert e.total_queries == 3


def test_an_empty_directory_reports_no_evidence_rather_than_raising(tmp_path: Path) -> None:
    e = device_evidence(tmp_path)
    assert not e.any
    assert e.total_queries == 0
    assert "No handset runs" in e.headline()


def test_a_missing_duration_stays_absent(tmp_path: Path) -> None:
    _write_run(tmp_path, "nometa", [{"nprobe": 1}])
    assert math.isnan(device_evidence(tmp_path).runs[0].minutes)


def test_the_device_name_is_taken_only_from_a_handset_table(tmp_path: Path) -> None:
    """A host table's `measured_under` must never supply a device name, or the
    notes would attribute laptop measurements to a phone."""
    _write_run(tmp_path, "trace", [{"nprobe": 1}])
    host = tmp_path / "host_table.json"
    host.write_text(json.dumps({"measured_under": "THIS LAPTOP, not the handset: ..."}))
    assert device_evidence(tmp_path, host).device == ""

    phone = tmp_path / "device_table.json"
    phone.write_text(json.dumps({"measured_under": "vivo I2304, on-device, 25-min trace"}))
    assert device_evidence(tmp_path, phone).device == "vivo I2304"


# ---------------------------------------------------------------------------
# The claims that cannot be counted
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("script", DEVICE_SCRIPTS)
def test_the_device_scripts_import_no_embedder(script: str) -> None:
    """Backs the 'embed the query -- host only' row. The device runs load
    pre-computed query vectors; if that ever changes, the table must change."""
    source = (SCRIPTS / script).read_text(encoding="utf-8")
    assert "queries.npy" in source, f"{script} no longer loads pre-computed query vectors"
    assert "Embedder" not in source, f"{script} now imports an embedder; update PIPELINE_STAGES"


@pytest.mark.parametrize("script", DEVICE_SCRIPTS)
def test_the_device_scripts_import_no_generator(script: str) -> None:
    """Backs the 'generate the answer -- host only' row."""
    source = (SCRIPTS / script).read_text(encoding="utf-8")
    for name in ("HFGenerator", "flan-t5", "llama_cpp"):
        assert name not in source, f"{script} now generates on-device; update PIPELINE_STAGES"


def test_the_device_scripts_do_run_the_real_index() -> None:
    """Backs the two 'handset' retrieval rows. A device run against a stub would
    make every on-device latency meaningless."""
    for script in DEVICE_SCRIPTS:
        source = (SCRIPTS / script).read_text(encoding="utf-8")
        assert "EcoVectorIndex" in source
        assert "index.probe" in source


def test_the_c5_script_does_construct_the_device_policy() -> None:
    """Backs the 'controller decides the effort level -- handset' row."""
    source = (SCRIPTS / "device_c5.py").read_text(encoding="utf-8")
    assert "DeviceStatePolicy" in source


def test_every_stage_carries_evidence_someone_could_check() -> None:
    """A row asserting 'handset' with no way to verify it is the thing this
    module exists to avoid."""
    assert len(PIPELINE_STAGES) >= 5
    for stage in PIPELINE_STAGES:
        assert stage.evidence.strip()
        assert stage.where in ("handset", "host only")


def test_both_retrieval_and_generation_positions_are_represented() -> None:
    """The table must say something about both halves. A table listing only the
    stages that flatter the project would be worse than no table."""
    on_device = {s.name for s in PIPELINE_STAGES if s.on_device}
    host_only = {s.name for s in PIPELINE_STAGES if not s.on_device}
    assert on_device and host_only
    assert any("generate" in s for s in host_only)
    assert any("search" in s or "probe" in s for s in on_device)
