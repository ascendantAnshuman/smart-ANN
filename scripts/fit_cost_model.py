#!/usr/bin/env python
"""Phase 3: fit the latency model, specify the energy model, report honestly.

Fits on QUIET-phase queries and validates on CONTENDED ones, so the model is
tested on conditions it was not fitted to. Fitting on both would let it
interpolate between them and report a flattering error.

The energy model is implemented and unit-tested but NOT fitted: this platform
can measure no energy (ADR-0007, ADR-0008), and Phase 3 will not close with an
unevaluable criterion marked PASS.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from smartann.config import resolve_paths
from smartann.cost import (
    MOBILERAG_REFERENCE,
    EnergyModel,
    assert_energy_measurable,
    fit_latency_model,
)
from smartann.cost.latency import TERM_FEATURES, mape
from smartann.telemetry.windows import WindowsMonitor

LATENCY_MAPE_TARGET = 15.0


def load_phases(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one run into (quiet, contended) using its recorded events."""
    df = pd.read_parquet(run_dir / "queries.parquet")
    mf = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    ev = {e["name"]: e["t_wall"] for e in mf["events"]}
    t = df["t_wall"].to_numpy()
    quiet = df[t < ev["burners_start"]]
    contended = df[(t >= ev["burners_start"]) & (t < ev["burners_stop"])]
    return quiet, contended


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--out", default="results/cost_model.json")
    args = ap.parse_args()

    paths = resolve_paths()
    roots = (
        [Path(r) for r in args.runs] if args.runs else sorted(paths.results_root.glob("E_*/aimd"))
    )
    roots = [r for r in roots if (r / "queries.parquet").exists()]
    if not roots:
        print("no runs with queries.parquet found", file=sys.stderr)
        return 1

    quiet_parts, contended_parts = [], []
    for r in roots:
        q, c = load_phases(r)
        quiet_parts.append(q)
        contended_parts.append(c)
        print(f"  {r.parent.name}/{r.name}: {len(q):,} quiet, {len(c):,} contended", flush=True)
    quiet = pd.concat(quiet_parts, ignore_index=True)
    contended = pd.concat(contended_parts, ignore_index=True)

    print(
        f"\nfitting on {len(quiet):,} quiet queries, validating on {len(contended):,} contended\n",
        flush=True,
    )

    model = fit_latency_model(quiet, fitted_on=f"quiet phases of {len(roots)} run(s)")
    print(model.describe(), flush=True)

    # -- held-out validation ------------------------------------------------
    print("\n--- per-term error, CONTENDED (never fitted on) ---", flush=True)
    rows = []
    for term in TERM_FEATURES:
        actual = contended[term].to_numpy(dtype=np.float64)
        pred = model.terms[term].predict(contended)
        m = mape(actual, pred)
        rows.append(
            {
                "term": term,
                "mape_contended": m,
                "mean_actual_ms": float(actual.mean()),
                "mean_pred_ms": float(pred.mean()),
            }
        )
        print(
            f"  {term:<18} MAPE {m:8.2f}%   mean actual {actual.mean():8.3f} ms  "
            f"predicted {pred.mean():8.3f} ms",
            flush=True,
        )

    for label, frame in (("quiet (fitted)", quiet), ("contended (held out)", contended)):
        actual = frame["t_total"].to_numpy(dtype=np.float64)
        pred = model.predict(frame)
        total_mape = mape(actual, pred)
        verdict = "PASS" if total_mape < LATENCY_MAPE_TARGET else "FAIL"
        print(
            f"\n[{verdict}] total latency MAPE on {label}: {total_mape:.2f}% "
            f"(target < {LATENCY_MAPE_TARGET:.0f}%)",
            flush=True,
        )
        rows.append({"term": f"t_total::{label}", "mape": total_mape})

    # -- energy: specified, implemented, deliberately not fitted ------------
    print("\n--- energy model ---", flush=True)
    energy = EnergyModel(MOBILERAG_REFERENCE)
    print(energy.describe(), flush=True)
    caps = WindowsMonitor().capabilities()
    try:
        assert_energy_measurable(caps)
        energy_status = "FITTED (platform can measure energy)"
    except RuntimeError as exc:
        energy_status = f"NOT FITTED, correctly refused: {exc}"
        print(f"\n[BLOCKED] {exc}", flush=True)

    cpu_ms, io_ms = energy.split_from_record(quiet)
    predicted_mj = energy.predict_mj(cpu_ms, io_ms)
    print(
        f"\nmodel output on quiet queries: mean {predicted_mj.mean():.4f} mJ/query "
        f"(a PREDICTION, not a measurement -- recorded as energy_mj_model only)",
        flush=True,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "latency": model.as_dict(),
                "validation": rows,
                "latency_mape_target": LATENCY_MAPE_TARGET,
                "energy": {
                    "form": "E = V * [I_cpu * t_cpu + I_io * t_io]",
                    "coefficients": MOBILERAG_REFERENCE.__dict__,
                    "status": energy_status,
                    "t_cluster_load_assigned_to": "CPU (ADR-0026 amendment: 95-99% reconstruction)",
                    "mean_predicted_mj_per_query": float(predicted_mj.mean()),
                    "criterion": "energy MAPE < 25% is NOT evaluated in Phase 3 and is not "
                    "marked PASS. Phase 3.5 fits and validates it on the phone.",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
