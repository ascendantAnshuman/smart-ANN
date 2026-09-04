"""Per-query latency, modelled term by term rather than as one black box.

A single regression onto total latency would hide which stage is mispredicted,
and Phase 6 needs to know *where* a prediction fails: a policy choosing `nprobe`
is choosing how many cluster loads to pay for, so a model that is right overall
but wrong about the load term would mislead exactly the decision it exists to
inform.

So each stage is fitted separately against the features that physically drive
it, and the residuals are reported per term. A bad term is visible instead of
averaged away.

**`t_cluster_load` is CPU, not I/O.** Measured directly (ADR-0026 amendment):
reading the bytes is 1-5% of load time and 95-99% is `hnswlib.load_index`
rebuilding the graph, or for flat clusters the page-table work plus the scan
itself. Phase 3.5 re-derives this split on the phone, where flash is slower
relative to the CPU, rather than assuming it transfers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["TERM_FEATURES", "UNIDENTIFIABLE", "LatencyModel", "TermFit", "fit_latency_model"]

# Which measured quantities drive each stage. Chosen from the mechanism, not by
# searching for correlations: a term whose features were picked because they fit
# would predict well on this machine and mislead on the phone.
# A work-only model CANNOT predict latency, and the first fit demonstrated it:
# 14.46% MAPE on quiet data and 36.44% on held-out contended data, underpredicting
# every stage. Between those phases cpu_util went 0.418 -> 1.000 while
# t_cluster_load went 4.111 -> 10.597 ms for IDENTICAL work. Latency is a function
# of work AND of how much machine is available to do it; a model with only work
# counters has no way to express the second.
#
# `dev_cpu_util` is therefore a feature of every stage that competes for CPU. This
# is a specification correction, not a search: the variable was named from the
# mechanism before the refit, it is the same quantity the controller already reads,
# and the fix is validated on held-out contended data rather than on the data that
# motivated it.
TERM_FEATURES: dict[str, tuple[str, ...]] = {
    # Centroid graph walk. NOTE: ef_centroid is constant at 128 across every run
    # recorded so far, so its coefficient is UNIDENTIFIABLE and the first fit
    # returned R2 = 0.0000 -- an intercept wearing a regression's clothes. Kept as
    # a state-only term until a run varies ef_centroid; see notes below.
    "t_centroid": ("dev_cpu_util",),
    # Opening probed clusters. Work: files opened and bytes touched. State: the
    # dominant stage, so the most contention-sensitive.
    "t_cluster_load": ("n_cluster_loads", "bytes_read", "dev_cpu_util"),
    # The in-cluster scan. n_dist_exact is COUNTED for flat clusters, so this term
    # is regressed on a measurement rather than on a model (ADR-0023).
    "t_cluster_search": ("n_dist_exact", "n_dist_modeled", "dev_cpu_util"),
    # Merging per-cluster candidate lists into the final top-k.
    "t_merge": ("n_cluster_loads", "dev_cpu_util"),
}

# Features whose coefficient cannot be identified from the runs recorded so far,
# because the feature does not vary. Reported rather than silently fitted.
UNIDENTIFIABLE: dict[str, str] = {
    "cfg_ann_ef_centroid": (
        "constant at 128 in every recorded run, so t_centroid's dependence on it "
        "is unidentifiable. A Phase 6 profile sweep that varies ef_centroid would "
        "identify it -- though ADR-0025 found ef_centroid to be free, so it may "
        "never need to vary."
    ),
}


@dataclass(frozen=True)
class TermFit:
    """One stage's fitted coefficients and how well they hold."""

    term: str
    features: tuple[str, ...]
    intercept_ms: float
    coefficients: tuple[float, ...]
    n_fit: int
    r2: float
    mape_fit: float
    mean_ms: float

    def predict(self, frame: Any) -> np.ndarray:
        out = np.full(len(frame), self.intercept_ms, dtype=np.float64)
        for name, c in zip(self.features, self.coefficients, strict=True):
            out += c * frame[name].to_numpy(dtype=np.float64)
        return out

    def describe(self) -> str:
        parts = [f"{self.intercept_ms:.4g}"]
        parts += [f"{c:+.4g}*{n}" for n, c in zip(self.features, self.coefficients, strict=True)]
        return f"{self.term} = {' '.join(parts)}  (R2={self.r2:.4f}, n={self.n_fit:,})"


@dataclass(frozen=True)
class LatencyModel:
    """Sum of per-stage fits. Predicts total per-query latency in ms."""

    terms: dict[str, TermFit]
    fitted_on: str
    n_fit: int
    notes: list[str] = field(default_factory=list)

    def predict(self, frame: Any) -> np.ndarray:
        total = np.zeros(len(frame), dtype=np.float64)
        for fit in self.terms.values():
            total += fit.predict(frame)
        return total

    def predict_terms(self, frame: Any) -> dict[str, np.ndarray]:
        return {name: fit.predict(frame) for name, fit in self.terms.items()}

    def describe(self) -> str:
        lines = [f"latency model, fitted on {self.fitted_on} ({self.n_fit:,} queries)"]
        lines += ["  " + f.describe() for f in self.terms.values()]
        lines += ["  NOTE: " + n for n in self.notes]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "fitted_on": self.fitted_on,
            "n_fit": self.n_fit,
            "notes": self.notes,
            "terms": {
                name: {
                    "features": list(f.features),
                    "intercept_ms": f.intercept_ms,
                    "coefficients": list(f.coefficients),
                    "r2": f.r2,
                    "mape_fit": f.mape_fit,
                    "mean_ms": f.mean_ms,
                    "expression": f.describe(),
                }
                for name, f in self.terms.items()
            },
        }


def mape(actual: np.ndarray, predicted: np.ndarray, floor_ms: float = 0.05) -> float:
    """Mean absolute percentage error, guarded against a near-zero denominator.

    Some stages are genuinely sub-microsecond (t_merge at nprobe=1), and a
    percentage error against ~0 is unbounded noise rather than a large error.
    Samples below `floor_ms` are excluded and the count is reported by the
    caller, so the exclusion is visible rather than silent.
    """
    mask = np.abs(actual) >= floor_ms
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / np.abs(actual[mask])) * 100.0)


def _fit_one(term: str, features: tuple[str, ...], frame: Any) -> TermFit:
    y = frame[term].to_numpy(dtype=np.float64)
    x = np.column_stack(
        [np.ones(len(frame))] + [frame[f].to_numpy(dtype=np.float64) for f in features]
    )
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return TermFit(
        term=term,
        features=features,
        intercept_ms=float(beta[0]),
        coefficients=tuple(float(b) for b in beta[1:]),
        n_fit=len(frame),
        r2=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        mape_fit=mape(y, pred),
        mean_ms=float(y.mean()),
    )


def fit_latency_model(frame: Any, fitted_on: str) -> LatencyModel:
    """Fit every stage on *frame*.

    The caller decides what `frame` contains, and that choice matters more than
    the regression: fitting on quiet data and validating on contended data tests
    the model on conditions it was not fitted to, whereas fitting on both would
    let it interpolate between them and report a flattering error.
    """
    missing = [t for t in TERM_FEATURES if t not in frame.columns]
    if missing:
        raise KeyError(f"frame is missing latency terms {missing}")
    if len(frame) < 50:
        raise ValueError(f"only {len(frame)} rows; too few to fit a per-term model")

    terms = {t: _fit_one(t, feats, frame) for t, feats in TERM_FEATURES.items()}
    return LatencyModel(
        terms=terms,
        fitted_on=fitted_on,
        n_fit=len(frame),
        notes=[
            "t_cluster_load is a CPU term, not I/O: 95-99% of it is reconstruction "
            "or mapping, measured directly (ADR-0026 amendment). Phase 3.5 re-derives "
            "the split on the phone.",
            "t_cluster_search is regressed on n_dist_exact, which is COUNTED for flat "
            "clusters, and on n_dist_modeled separately so a residual can be attributed "
            "to the measured or the modelled part (ADR-0023).",
            "dev_cpu_util is a feature of every CPU-competing stage. A work-only model "
            "scored 36.44% MAPE on held-out contended data because identical work takes "
            "2.6x longer on a saturated machine, and no work counter can express that.",
            "WEAKLY IDENTIFIED: n_cluster_loads and bytes_read are correlated at "
            "r=0.966 in the recorded data (design-matrix condition number 4.9e6), "
            "because clusters are of similar size. Their COMBINED effect is well "
            "determined -- t_cluster_load has R2 0.99 -- but the individual "
            "coefficients are unstable and must not be interpreted separately.",
            *(f"UNIDENTIFIABLE: {k} -- {v}" for k, v in UNIDENTIFIABLE.items()),
        ],
    )
