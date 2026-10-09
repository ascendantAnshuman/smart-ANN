"""The dashboard must show absence, and must not execute what it is given.

Two properties matter more than layout.

**A section with no input says "not run".** A panel that silently disappears
when its file is missing makes "we measured this and it was small" look exactly
like "we never measured this", which is the confusion Section 3 rule 1 exists to
prevent. Every collector returns None on a missing input and the renderer prints
the absence.

**Questions and answers are escaped.** Both are free text: one typed by a user,
one produced by a language model from retrieved documents. They are interpolated
into HTML, so markup in either must arrive as visible characters rather than as
tags. The page is opened in a browser and handed to other people.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from smartann.dashboard import DashboardState, collect
from smartann.dashboard_html import render_html


def empty_state() -> DashboardState:
    return DashboardState(generated_at="now", git_sha="abc1234", results_dir=Path("results"))


def test_a_bare_directory_collects_nothing_and_says_so(tmp_path: Path) -> None:
    state = collect(results=tmp_path, repo=tmp_path)
    assert state.profile is None
    assert state.thermal is None
    assert state.qa is None
    assert state.joint is None
    assert set(state.missing) >= {"profile table", "thermal attribution", "verification"}


def test_an_empty_state_still_renders_a_page(tmp_path: Path) -> None:
    """The most common reason to open this is to see how far a run has got."""
    html = render_html(collect(results=tmp_path, repo=tmp_path))
    assert html.startswith("<!doctype html>")
    assert html.rstrip().endswith("</html>")
    assert "Not run" in html
    assert "SMART-ANN" in html


def test_a_missing_section_never_silently_vanishes(tmp_path: Path) -> None:
    """Absence is printed, not implied by a gap in the layout."""
    html = render_html(collect(results=tmp_path, repo=tmp_path))
    for title in ("Effort levels", "Why the device is hot", "Answer quality", "Joint control"):
        assert title in html


def test_malformed_json_is_treated_as_absent(tmp_path: Path) -> None:
    """A truncated file must not crash the page, and must not look like data."""
    (tmp_path / "phase7").mkdir()
    (tmp_path / "phase7" / "qa_summary.json").write_text("{not json", encoding="utf-8")
    state = collect(results=tmp_path, repo=tmp_path)
    assert state.qa is None
    assert "answer quality (phase 7)" in state.missing


def test_answers_are_escaped_not_executed() -> None:
    """A question or answer carrying markup must render as text."""
    state = empty_state()
    state.answers = [
        {
            "question": "<script>alert('x')</script>",
            "answer": "<img src=x onerror=alert(1)>",
            "policy": {"chosen_nprobe": 8, "budget_ms": 120},
            "timing_ms": {"total": 900.0},
            "work": {"n_cluster_loads": 8},
        }
    ]
    html = render_html(state)
    # What neutralises an injection is the delimiters, not the payload text.
    # `onerror=alert(1)` inside an escaped tag is a string a reader sees, so
    # asserting its absence would be asserting the wrong thing.
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html
    assert "<img src=x" not in html
    assert "&lt;img src=x" in html


def test_a_single_answer_object_is_accepted_as_well_as_a_list(tmp_path: Path) -> None:
    """`ask --json-out` writes an object for one question and a list for many."""
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"question": "q", "answer": "a"}), encoding="utf-8")
    state = collect(results=tmp_path, repo=tmp_path, answers_path=path)
    assert len(state.answers) == 1

    path.write_text(json.dumps([{"question": "q1"}, {"question": "q2"}]), encoding="utf-8")
    assert len(collect(results=tmp_path, repo=tmp_path, answers_path=path).answers) == 2


def test_figures_are_embedded_so_the_page_travels(tmp_path: Path) -> None:
    """A page that only renders beside its image directory breaks when emailed."""
    figs = tmp_path / "figures"
    figs.mkdir()
    # A one-pixel PNG is enough to prove the bytes make the round trip.
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    (figs / "fig_thermal_attribution.png").write_bytes(png)

    state = collect(results=tmp_path, repo=tmp_path)
    assert len(state.figures) == 1
    name, src = state.figures[0]
    assert name == "fig_thermal_attribution"
    assert src.startswith("data:image/png;base64,")
    assert 'src="data:image/png;base64,' in render_html(state)


def test_a_figure_that_is_not_on_disk_is_skipped_not_linked(tmp_path: Path) -> None:
    """A broken image icon would read as a failed measurement."""
    (tmp_path / "figures").mkdir()
    state = collect(results=tmp_path, repo=tmp_path)
    assert state.figures == []
    assert "<img" not in render_html(state)


def test_profile_values_reach_the_page(tmp_path: Path) -> None:
    state = empty_state()
    state.profile = {
        "dataset": "squad",
        "measured_under": "a stub bench",
        "entries": [
            {
                "nprobe": 1,
                "n_samples": 100,
                "p50_ms": 5.0,
                "p95_ms": 10.0,
                "recall": 0.60,
                "approx_tokens": 10.0,
            },
            {
                "nprobe": 8,
                "n_samples": 100,
                "p50_ms": 50.0,
                "p95_ms": 100.0,
                "recall": 0.91,
                "approx_tokens": 10.0,
            },
        ],
    }
    html = render_html(state)
    assert "0.9100" in html
    assert "100.0 ms" in html
    assert "a stub bench" in html, "the conditions a table was measured under must be shown"


def test_both_themes_define_every_colour_role() -> None:
    """A role defined only in one block borrows the host theme in the other."""
    html = render_html(empty_state())
    roles = ("--surface-0", "--surface-1", "--text-primary", "--series-1", "--series-3")
    for role in roles:
        # Once on bare :root, once in the media query, once under data-theme.
        assert html.count(f"{role}:") >= 3, f"{role} is not defined for every theme"


def test_the_page_loads_nothing_from_the_network() -> None:
    """Self-contained is the point: no CDN, no external stylesheet, no remote image."""
    html = render_html(empty_state())
    for bad in ("http://", "https://", "<script"):
        assert bad not in html


# ---------------------------------------------------------------------------
# The device-response section
# ---------------------------------------------------------------------------


def _profile(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "dataset": "squad",
                "measured_under": "a stub",
                "min_samples": 100,
                "entries": [
                    {
                        "nprobe": 1,
                        "n_samples": 1000,
                        "p50_ms": 5.0,
                        "p95_ms": 10.0,
                        "recall": 0.60,
                        "approx_tokens": 100.0,
                    },
                    {
                        "nprobe": 8,
                        "n_samples": 1000,
                        "p50_ms": 50.0,
                        "p95_ms": 100.0,
                        "recall": 0.90,
                        "approx_tokens": 100.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def _telemetry(path: Path, hottest: list[float], skin: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "t": float(i * 10),
            "skin_c": s,
            "hottest_c": h,
            "pct": 80,
            "status": "DISCHARGING",
            "current_ua": -300000,
            "slope": 0.0,
        }
        for i, (h, s) in enumerate(zip(hottest, skin, strict=True))
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_the_device_section_needs_both_a_table_and_a_trace(tmp_path: Path) -> None:
    _profile(tmp_path / "device_profile_table.json")
    state = collect(results=tmp_path, repo=tmp_path)
    assert state.device_response is None, "a table alone cannot replay anything"
    assert "device response" in state.missing
    assert "Not run" in render_html(state)


def test_the_two_thermometers_are_replayed_separately(tmp_path: Path) -> None:
    """The C5 finding, in the collector: same run, same policy, different answer."""
    _profile(tmp_path / "device_profile_table.json")
    _telemetry(tmp_path / "device" / "c5_telemetry.jsonl", [50.0] * 8, [30.0] * 8)

    d = collect(results=tmp_path, repo=tmp_path).device_response
    assert d is not None
    by = {s["sensor"]: s for s in d["series"]}
    assert by["skin"]["peak_c"] == 30.0
    assert by["hottest"]["peak_c"] == 50.0
    assert by["skin"]["n_derated"] == 0, "30 C crosses no threshold"
    assert by["hottest"]["n_derated"] == by["hottest"]["n_points"], "50 C is above hot"
    assert max(by["hottest"]["levels"]) < max(by["skin"]["levels"])


def test_the_section_reports_the_recall_the_derate_costs(tmp_path: Path) -> None:
    _profile(tmp_path / "device_profile_table.json")
    _telemetry(tmp_path / "device" / "c5_telemetry.jsonl", [50.0] * 8, [30.0] * 8)

    state = collect(results=tmp_path, repo=tmp_path)
    page = render_html(state)
    assert "giving up" in page
    assert "0.9000" in page, "the budget-only recall must be printed, not described"


def test_a_trace_with_no_profile_table_collects_nothing(tmp_path: Path) -> None:
    """Without measured levels there is no menu to choose from, so no claim to make."""
    _telemetry(tmp_path / "device" / "c5_telemetry.jsonl", [50.0] * 4, [30.0] * 4)
    assert collect(results=tmp_path, repo=tmp_path).device_response is None


# ---------------------------------------------------------------------------
# The contributions board
# ---------------------------------------------------------------------------


def test_a_contribution_with_no_evidence_is_not_demonstrated(tmp_path: Path) -> None:
    rows = {c["id"]: c for c in collect(results=tmp_path, repo=tmp_path).contributions}
    assert rows, "the board must list the claims even when nothing has been run"
    assert all(c["state"] != "yes" for c in rows.values()), (
        "an empty results directory must not produce a demonstrated contribution"
    )


def test_c5_is_reported_as_a_negative_result(tmp_path: Path) -> None:
    """The board must be able to say no. Five greens would be the least believable page."""
    rows = {c["id"]: c for c in collect(results=tmp_path, repo=tmp_path).contributions}
    assert rows["C5"]["state"] != "yes"


def test_a_recorded_verdict_is_quoted_rather_than_paraphrased(tmp_path: Path) -> None:
    (tmp_path / "phase8").mkdir()
    (tmp_path / "phase8" / "joint_summary.json").write_text(
        json.dumps({"verdict": "C3 SUPPORTED on this surface", "joint_cells_on_pareto_front": 3}),
        encoding="utf-8",
    )
    state = collect(results=tmp_path, repo=tmp_path)
    rows = {c["id"]: c for c in state.contributions}
    assert rows["C3"]["state"] == "yes"
    assert rows["C3"]["verdict"] == "C3 SUPPORTED on this surface"
    assert "C3 SUPPORTED on this surface" in render_html(state)


def test_a_fit_with_no_recorded_sample_count_says_so(tmp_path: Path) -> None:
    """ "?" formatted as a figure would read as a measurement nobody took."""
    (tmp_path / "energy_model.json").write_text(json.dumps({"a": 1.0}), encoding="utf-8")
    rows = {c["id"]: c for c in collect(results=tmp_path, repo=tmp_path).contributions}
    assert "not recorded" in rows["C4"]["detail"]


def test_a_list_valued_count_is_counted_not_printed(tmp_path: Path) -> None:
    """The defect this pins: a page of JSON printed where a figure belonged."""
    (tmp_path / "phase8").mkdir()
    (tmp_path / "phase8" / "joint_summary.json").write_text(
        json.dumps(
            {
                "verdict": "C3 SUPPORTED",
                "joint_cells_on_pareto_front": [{"nprobe": 1}, {"nprobe": 4}, {"nprobe": 8}],
            }
        ),
        encoding="utf-8",
    )
    rows = {c["id"]: c for c in collect(results=tmp_path, repo=tmp_path).contributions}
    assert rows["C3"]["detail"] == "3 joint cells on the Pareto front"
    assert "nprobe" not in rows["C3"]["detail"]


def test_evidence_paths_are_portable(tmp_path: Path) -> None:
    """The page gets emailed; a backslash path reads as a local accident."""
    (tmp_path / "phase8").mkdir()
    (tmp_path / "phase8" / "joint_summary.json").write_text(json.dumps({"cells": []}), "utf-8")
    rows = {c["id"]: c for c in collect(results=tmp_path, repo=tmp_path).contributions}
    assert "\\" not in rows["C3"]["evidence"]
    assert rows["C3"]["evidence"] == "phase8/joint_summary.json"


def test_every_recorded_run_is_summarised_not_just_the_replayed_one(tmp_path: Path) -> None:
    """One run is an anecdote; the repetition across runs is the finding."""
    _profile(tmp_path / "device_profile_table.json")
    _telemetry(tmp_path / "device" / "c5_telemetry.jsonl", [50.0] * 4, [30.0] * 4)
    _telemetry(tmp_path / "device" / "e2_telemetry.jsonl", [60.0] * 4, [31.0] * 4)

    d = collect(results=tmp_path, repo=tmp_path).device_response
    assert d is not None
    names = {a["name"] for a in d["across_traces"]}
    assert names == {"c5", "e2"}
    assert all(not a["skin_crossed"] for a in d["across_traces"])
    assert all(a["hottest_crossed"] for a in d["across_traces"])


# ---------------------------------------------------------------------------
# The sections that were on disk but unshown
# ---------------------------------------------------------------------------


def test_every_result_file_on_disk_reaches_the_state(tmp_path: Path) -> None:
    """A result nobody renders is a result nobody sees. These were collected late."""
    (tmp_path / "phase6").mkdir()
    (tmp_path / "phase6" / "policy_comparison.json").write_text(
        json.dumps(
            {
                "budget_ms": 14.0,
                "arms": [
                    {
                        "policy": "static",
                        "n": 10,
                        "p50_ms": 11.0,
                        "p95_ms": 14.2,
                        "violation_pct": 5.98,
                        "recall": 0.85,
                        "nprobe_p50": 12.0,
                        "overhead_p95_pct": 0.006,
                    },
                    {
                        "policy": "budget",
                        "n": 10,
                        "p50_ms": 8.0,
                        "p95_ms": 9.5,
                        "violation_pct": 0.17,
                        "recall": 0.78,
                        "nprobe_p50": 8.0,
                        "overhead_p95_pct": 0.17,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "energy_model.json").write_text(
        json.dumps(
            {
                "measured": {
                    "total_energy_j": 1266.0,
                    "mean_power_w": 1.006,
                    "power_median_w": 0.868,
                    "mj_per_query_allin": 131.0,
                    "voltage_range_v": [4.058, 4.118],
                    "skin_range_c": [30.4, 34.0],
                },
                "attribution": {"mj_per_cluster": 3.2, "r2": 0.16},
                "effort_vs_power": {
                    "quintiles": [
                        {"work_lo": 5.0, "work_hi": 114.0, "power_median_w": 0.729, "n": 32},
                        {"work_lo": 162.0, "work_hi": 202.0, "power_median_w": 1.441, "n": 32},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    state = collect(results=tmp_path, repo=tmp_path)
    assert state.policy6 is not None
    assert state.energy is not None

    page = render_html(state)
    assert "5.98" in page, "the static policy's violation rate must be on the page"
    assert "35x" in page, "the improvement must be computed, not described"
    assert "131" in page and "mJ" in page


def test_a_validation_row_of_another_shape_does_not_crash_the_page(tmp_path: Path) -> None:
    """The defect this pins: two rows carry `mape`, not `mape_contended`.

    A KeyError here took the whole page down, which breaks the rule that
    sections fail independently.
    """
    (tmp_path / "cost_model.json").write_text(
        json.dumps(
            {
                "latency_mape_target": 15.0,
                "latency": {"fitted_on": "3 runs", "n_fit": 21388},
                "validation": [
                    {
                        "term": "t_centroid",
                        "mape_contended": 33.7,
                        "mean_actual_ms": 0.418,
                        "mean_pred_ms": 0.087,
                    },
                    {"term": "t_total::quiet", "mape": 14.8},
                ],
            }
        ),
        encoding="utf-8",
    )
    page = render_html(collect(results=tmp_path, repo=tmp_path))
    assert "t_total::quiet" in page
    assert "quiet machine only" in page, "the condition of measurement must be named"
    assert "under CPU contention" in page


def test_the_page_offers_navigation_to_every_section() -> None:
    """A long page is read by someone hunting one number while being questioned."""
    page = render_html(empty_state())
    for anchor in ("#index", "#effort", "#policies", "#device", "#energy", "#heat", "#figures"):
        assert f'href="{anchor}"' in page
        assert f'id="{anchor[1:]}"' in page, f"{anchor} is linked but no section carries it"


def test_the_header_names_the_device_the_numbers_came_from(tmp_path: Path) -> None:
    """A reader's first question about an on-device result is: which device?"""
    _profile(tmp_path / "device_profile_table.json")
    page = render_html(collect(results=tmp_path, repo=tmp_path))
    assert "a stub" in page, "the measured_under string must be shown, not retyped"
    assert "&amp;middot;" not in page, "the separator must stay an entity, not print literally"


def test_provenance_is_absent_rather_than_invented(tmp_path: Path) -> None:
    assert 'class="meta prov"' not in render_html(collect(results=tmp_path, repo=tmp_path))


def test_a_phase9_result_reaches_the_page(tmp_path: Path) -> None:
    (tmp_path / "phase9").mkdir()
    (tmp_path / "phase9" / "query_adaptive.json").write_text(
        json.dumps(
            {
                "signal": "centroid_dist_gap",
                "signal_meaning": "how much the nearest wins by",
                "levels": [1, 8],
                "n_fit": 500,
                "n_test": 500,
                "seed": 0,
                "recall_resolution": 0.0002,
                "method": "held out",
                "bucket_edges": [1.2],
                "separation": [
                    {
                        "bucket": 0,
                        "gap_lo": 1.0,
                        "gap_hi": 1.2,
                        "share": 0.5,
                        "recall_by_level": {"1": 0.50, "8": 0.88},
                    },
                    {
                        "bucket": 1,
                        "gap_lo": 1.2,
                        "gap_hi": 3.0,
                        "share": 0.5,
                        "recall_by_level": {"1": 0.72, "8": 0.93},
                    },
                ],
                "comparisons": [
                    {
                        "uniform_nprobe": 8,
                        "budget_ms": 100.0,
                        "assignment": [8, 1],
                        "held_out": {
                            "n": 500,
                            "adaptive_recall": 0.91,
                            "uniform_recall": 0.90,
                            "delta_recall": 0.01,
                            "adaptive_mean_cost_ms": 80.0,
                            "uniform_mean_cost_ms": 100.0,
                        },
                    }
                ],
                "verdict": "C6 SUPPORTED: a made-up verdict for this test",
            }
        ),
        encoding="utf-8",
    )
    state = collect(results=tmp_path, repo=tmp_path)
    assert state.query_adaptive is not None
    page = render_html(state)
    assert "centroid_dist_gap" in page
    assert "C6 SUPPORTED: a made-up verdict for this test" in page, "the verdict is quoted"
    assert "+20%" in page, "20% less device time, computed from the two measured costs"
    assert "time saved" in page, "the sign of the cost column must be unambiguous"


def test_a_missing_phase9_result_says_not_run(tmp_path: Path) -> None:
    page = render_html(collect(results=tmp_path, repo=tmp_path))
    assert 'id="queryfit"' in page
    assert "phase9_query_adaptive" in page
