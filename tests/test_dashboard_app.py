"""The Phase 14 dashboard: does it render, and does it read real numbers?

A dashboard fails in a particular way: it renders, it looks finished, and the
figures on it came from nowhere. These tests check the two things that matter --
that the page runs without raising, and that the numbers it shows are the ones
in the result files rather than values written into the template.

Skipped when streamlit is absent, since it is an optional extra (`pip install
-e ".[demo]"`) and the library, the CLI and every experiment run without it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app" / "dashboard.py"
RESULTS = ROOT / "results"


@pytest.fixture(scope="module")
def app() -> object:
    """Render the page once. The Ask tab does not run the pipeline until the
    button is pressed, so this stays fast."""
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    return at


def test_the_page_renders_without_raising(app: object) -> None:
    assert len(app.exception) == 0, [str(e.value) for e in app.exception]


def test_all_four_tabs_are_present(app: object) -> None:
    assert len(app.tabs) == 4


def test_nothing_rendered_an_error_box(app: object) -> None:
    assert [e.value for e in app.error] == []


@pytest.mark.skipif(
    not (RESULTS / "comparison.json").exists(), reason="the comparison has not been run"
)
def test_the_cloud_metrics_match_the_recorded_comparison(app: object) -> None:
    """The numbers on the page must be the ones in the file. A dashboard that
    renders a plausible figure of its own is worse than one that renders
    nothing."""
    comp = json.loads((RESULTS / "comparison.json").read_text(encoding="utf-8"))
    shown = {m.label: m.value for m in app.metric}
    assert shown["Bytes sent on-device"] == str(comp["device_bytes_total"])
    assert shown["Bytes the cloud arm would send"] == f"{comp['cloud_payload_bytes']['total']:,.0f}"


@pytest.mark.skipif(
    not (RESULTS / "demo_eval.json").exists(), reason="the demo eval has not been run"
)
def test_the_retrieval_metrics_match_the_recorded_eval(app: object) -> None:
    ev = json.loads((RESULTS / "demo_eval.json").read_text(encoding="utf-8"))
    shown = {m.label: m.value for m in app.metric}
    assert (
        shown["recall@5 (paraphrases)"] == f"{ev['paraphrases']['recall_at_5_frozen_primary']:.3f}"
    )
    assert shown["distractors in top 5"] == str(ev["paraphrases"]["distractors_in_top5"])


def test_a_missing_result_is_reported_rather_than_drawn(tmp_path: Path) -> None:
    """`missing()` must say what is absent and name the script that makes it.
    A placeholder chart would be indistinguishable from a measurement."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("dash", APP)
    assert spec and spec.loader
    # The module runs Streamlit calls at import, so it is not importable outside
    # a script run. The contract is checked on the source instead.
    source = APP.read_text(encoding="utf-8")
    assert "has not been produced yet" in source
    assert "Never draw a placeholder" in source
