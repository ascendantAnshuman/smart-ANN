"""Cost models. The energy half is the one that must refuse to lie."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smartann.cost import (
    MOBILERAG_REFERENCE,
    EnergyCoefficients,
    EnergyModel,
    assert_energy_measurable,
    fit_latency_model,
)
from smartann.cost.latency import TERM_FEATURES, UNIDENTIFIABLE, mape
from smartann.telemetry import Capability, CapabilityTable


def _frame(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    loads = rng.integers(1, 64, size=n).astype(float)
    util = rng.uniform(0.1, 0.9, size=n)
    exact = loads * rng.uniform(400, 600, size=n)
    df = pd.DataFrame(
        {
            "n_cluster_loads": loads,
            # Deliberately NOT proportional to loads: perfectly collinear
            # features make the split arbitrary, which is a real property of
            # the recorded data too (r=0.966, see LatencyModel notes).
            "bytes_read": loads * rng.uniform(150_000, 600_000, size=n),
            "n_dist_exact": exact,
            "n_dist_modeled": np.zeros(n),
            "cfg_ann_ef_centroid": np.full(n, 128.0),
            "dev_cpu_util": util,
        }
    )
    # A known generating process, so the fit can be checked against truth.
    df["t_centroid"] = 0.08 + 0.01 * util
    df["t_cluster_load"] = 0.03 + 0.69 * loads + 0.6 * util
    df["t_cluster_search"] = 0.0003 * exact + 0.03 * util
    df["t_merge"] = 0.006 + 0.0029 * loads
    df["t_total"] = df[list(TERM_FEATURES)].sum(axis=1)
    return df


# ---------------------------------------------------------------------------
# latency
# ---------------------------------------------------------------------------
def test_fit_recovers_a_known_generating_process() -> None:
    m = fit_latency_model(_frame(), fitted_on="synthetic")
    load = m.terms["t_cluster_load"]
    assert load.coefficients[0] == pytest.approx(0.69, rel=1e-3)
    assert load.r2 > 0.999


def test_prediction_is_the_sum_of_its_terms() -> None:
    """A single black-box regression would hide which stage is mispredicted."""
    df = _frame()
    m = fit_latency_model(df, fitted_on="synthetic")
    per_term = m.predict_terms(df)
    np.testing.assert_allclose(m.predict(df), sum(per_term.values()), rtol=1e-9)


def test_fit_refuses_a_frame_missing_a_term() -> None:
    df = _frame().drop(columns=["t_merge"])
    with pytest.raises(KeyError, match="t_merge"):
        fit_latency_model(df, fitted_on="synthetic")


def test_fit_refuses_too_few_rows() -> None:
    with pytest.raises(ValueError, match="too few"):
        fit_latency_model(_frame(n=10), fitted_on="synthetic")


def test_mape_excludes_near_zero_denominators() -> None:
    """t_merge is genuinely sub-microsecond at nprobe=1; % error there is noise."""
    actual = np.array([0.0, 0.001, 10.0])
    pred = np.array([5.0, 5.0, 11.0])
    assert mape(actual, pred, floor_ms=0.05) == pytest.approx(10.0)


def test_unidentifiable_features_are_declared_not_fitted() -> None:
    """ef_centroid is constant in every recorded run; R2 was 0.0000 (ADR-0033)."""
    assert "cfg_ann_ef_centroid" in UNIDENTIFIABLE
    assert "cfg_ann_ef_centroid" not in TERM_FEATURES["t_centroid"]


def test_every_cpu_competing_stage_carries_the_state_feature() -> None:
    """A work-only model cannot express work-divided-by-available-machine."""
    for term, feats in TERM_FEATURES.items():
        assert "dev_cpu_util" in feats, f"{term} has no machine-state feature"


# ---------------------------------------------------------------------------
# energy: the model must refuse to be fitted where it cannot be measured
# ---------------------------------------------------------------------------
def _caps(energy_available: bool) -> CapabilityTable:
    return CapabilityTable(
        [
            Capability("cpu_util", True, "psutil"),
            Capability(
                "energy_mj_measured", energy_available, "x", "" if energy_available else "no RAPL"
            ),
        ]
    )


def test_energy_fit_is_refused_where_energy_cannot_be_measured() -> None:
    """Phase 3 must not close with an unevaluable criterion marked PASS."""
    with pytest.raises(RuntimeError, match="unavailable"):
        assert_energy_measurable(_caps(False))


def test_energy_fit_is_allowed_where_it_can_be_measured() -> None:
    assert_energy_measurable(_caps(True))  # must not raise


def test_reference_coefficients_are_marked_unfitted() -> None:
    """Nothing downstream may mistake the paper's numbers for our measurement."""
    assert MOBILERAG_REFERENCE.fitted is False
    assert "NOT measured here" in MOBILERAG_REFERENCE.source


def test_energy_is_linear_in_both_times() -> None:
    m = EnergyModel(MOBILERAG_REFERENCE)
    single = m.predict_mj(np.array([10.0]), np.array([0.0]))
    double = m.predict_mj(np.array([20.0]), np.array([0.0]))
    assert double[0] == pytest.approx(2 * single[0])


def test_cpu_current_costs_more_than_io_current() -> None:
    """The whole reason t_cluster_load's assignment matters (ADR-0031)."""
    m = EnergyModel(MOBILERAG_REFERENCE)
    as_cpu = m.predict_mj(np.array([10.0]), np.array([0.0]))[0]
    as_io = m.predict_mj(np.array([0.0]), np.array([10.0]))[0]
    assert as_cpu > as_io
    assert as_cpu / as_io == pytest.approx(2300.0 / 800.0, rel=1e-6)


def test_cluster_load_is_billed_as_cpu_not_io() -> None:
    """MobileRAG puts cluster access in the I/O term; we measured it as CPU."""
    m = EnergyModel(MOBILERAG_REFERENCE)
    df = _frame(n=100)
    cpu_ms, io_ms = m.split_from_record(df)
    np.testing.assert_allclose(cpu_ms, df[list(TERM_FEATURES)].sum(axis=1).to_numpy())
    assert np.all(io_ms == 0.0), "warm cache: I/O time is not separable here"


def test_non_physical_coefficients_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-physical"):
        EnergyCoefficients(voltage_v=0.0, i_cpu_ua=2300.0, i_io_ua=800.0, fitted=False, source="x")
