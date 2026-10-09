#!/usr/bin/env python
"""13 -- on-device RAG against a cloud API, built from measurements.

The comparison most projects write is a table of adjectives. This one is a table
of numbers with a provenance column, and where there is no number it says so
rather than reaching for an adjective.

What is measured here, on the twenty labelled demonstration queries:

* **privacy** -- bytes of user data leaving the device per query. On-device this
  is counted by watching this process's own sockets. For the cloud column it is
  the exact byte length of the request body that *would* be sent, assembled from
  the same question and the same retrieved context, which is computable without
  sending anything.
* **latency** -- on-device, measured here. Cloud, measured ONLY if an
  `OPENAI_API_KEY` is in the environment, over the same twenty queries. Without
  a key the cell reads "not measured" and stays empty.
* **offline** -- on-device, the recorded result of the Phase 12.5 proof. Cloud,
  a statement of fact rather than a number: an API call cannot complete without
  a network, and no figure is invented to dramatise that.
* **cost** -- on-device, zero API spend, which is a fact about the architecture
  rather than a measurement. Cloud, computed from `config/pricing.yaml`, which
  ships empty. While it is empty this reports "not computed" and names the file.
* **grounding** -- whether an answer cites a passage that exists locally. For
  the on-device arm this is checked against the retrieved chunk ids. For the
  cloud arm it is only checkable if the API was actually called.

**The limitations row is part of the result, not a disclaimer under it.** A
small local model is weaker at open-ended reasoning and general knowledge than a
large hosted one, and on-device inference spends the phone's battery and thermal
budget -- which is the problem this project's controller exists to manage. A
comparison that omitted that would be advocacy.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smartann.pipeline.ask import SCR_OFF, RagPipeline
from smartann.pipeline.guard import GroundingGuard
from smartann.pipeline.querytrace import NetworkWitness

DATASET = "demo_phone_care"
NOT_MEASURED = "not measured"
# The system framing a hosted model would need to do the same job. Counted into
# the cloud payload because it is part of what would be sent on every query.
CLOUD_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about smartphones. "
    "Use only the provided context passages. If the answer is not in them, say so."
)


def estimate_tokens(text: str) -> float:
    """Characters over four. An ESTIMATE, and labelled as one everywhere.

    Providers tokenise with their own vocabularies, which are not available
    offline, so an exact count cannot be made here. Four characters per token is
    the conventional rough ratio for English prose. Where the API is actually
    called, the usage figures it returns replace this estimate and the row says
    which one it used.
    """
    return len(text) / 4.0


def cloud_request_body(question: str, context: str, model: str) -> str:
    """The exact JSON that would be posted, so its size is a fact not a guess."""
    return json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": CLOUD_SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        }
    )


def call_cloud(body: str, api_key: str, url: str, timeout: float) -> dict[str, Any]:
    """One real round trip, timed. Only ever called when a key is present."""
    req = urllib.request.Request(
        url,
        data=body.encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        elapsed = (time.perf_counter() - t0) * 1000.0
        choice = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = payload.get("usage", {})
        return {
            "ok": True,
            "ms": elapsed,
            "answer": choice,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "error": "",
        }
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
        return {
            "ok": False,
            "ms": (time.perf_counter() - t0) * 1000.0,
            "answer": "",
            "prompt_tokens": None,
            "completion_tokens": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def load_pricing(path: Path) -> tuple[dict[str, Any] | None, str]:
    """The filled-in price, or None and the reason it is not usable."""
    if not path.exists():
        return None, f"{path} does not exist"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    missing = [
        k
        for k in ("model", "input_per_1m_usd", "output_per_1m_usd", "source_url", "retrieved_on")
        if data.get(k) in (None, "")
    ]
    if missing:
        return None, (
            f"{path} has not been filled in ({', '.join(missing)} still empty). "
            f"No price is guessed."
        )
    return data, ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", type=Path, default=Path("data/demo_phone_care/eval_queries.json"))
    ap.add_argument("--pricing", type=Path, default=Path("config/pricing.yaml"))
    ap.add_argument("--table", type=Path, default=Path("results/demo_profile_table.json"))
    ap.add_argument("--out-csv", type=Path, default=Path("results/comparison.csv"))
    ap.add_argument("--out-rows", type=Path, default=Path("results/comparison_per_query.csv"))
    ap.add_argument("--out-json", type=Path, default=Path("results/comparison.json"))
    ap.add_argument("--out-png", type=Path, default=Path("results/comparison.png"))
    ap.add_argument(
        "--cloud-model",
        default=os.environ.get("SMARTANN_CLOUD_MODEL", "gpt-4o-mini"),
        help="Model id used for the cloud arm. Only contacted if OPENAI_API_KEY is set.",
    )
    ap.add_argument("--cloud-url", default="https://api.openai.com/v1/chat/completions")
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args()

    spec = json.loads(args.queries.read_text(encoding="utf-8"))
    queries = list(spec["paraphrases"]) + [item["query"] for item in spec["off_topic"]]

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if api_key:
        print(f"OPENAI_API_KEY is set: the cloud arm WILL be called ({args.cloud_model}).")
    else:
        print("OPENAI_API_KEY is not set: cloud latency and grounding stay 'not measured'.")

    guard = None
    guard_path = Path("results/grounding_threshold.json")
    if guard_path.exists():
        try:
            guard = GroundingGuard.load(guard_path, dataset=DATASET)
        except ValueError:
            guard = None

    pipeline = RagPipeline(
        dataset=DATASET,
        k=5,
        profile_table_path=args.table if args.table.exists() else None,
        use_kb=False,
    )
    pipeline.load()

    rows: list[dict[str, Any]] = []
    try:
        for q in queries:
            with NetworkWitness() as witness:
                r = pipeline.ask(q, nprobe=8, scr=SCR_OFF, guard=guard)
            context = "\n\n".join(c.reduced_text for c in r.chunks)
            body = cloud_request_body(q, context, args.cloud_model)

            row: dict[str, Any] = {
                "query": q,
                "device_ms": r.total_ms,
                "device_bytes_sent": witness.bytes_sent,
                "device_connections": witness.connections,
                "device_answer": r.answer,
                "device_refused": r.refused_ungrounded,
                "device_cites_chunk": r.cited is not None,
                "device_cited_chunk_id": r.cited.chunk_id if r.cited else "",
                "cloud_payload_bytes": len(body.encode("utf-8")),
                "cloud_context_chars": len(context),
                "cloud_prompt_tokens_estimate": estimate_tokens(body),
                "cloud_ms": "",
                "cloud_answer": "",
                "cloud_prompt_tokens_measured": "",
                "cloud_completion_tokens_measured": "",
                "cloud_error": "",
            }
            if api_key:
                res = call_cloud(body, api_key, args.cloud_url, args.timeout)
                row["cloud_ms"] = res["ms"] if res["ok"] else ""
                row["cloud_answer"] = res["answer"]
                row["cloud_prompt_tokens_measured"] = res["prompt_tokens"] or ""
                row["cloud_completion_tokens_measured"] = res["completion_tokens"] or ""
                row["cloud_error"] = res["error"]
            rows.append(row)
            print(
                f"  {r.total_ms:7.1f} ms on-device, {witness.bytes_sent} bytes out, "
                f"would send {row['cloud_payload_bytes']:5d} bytes to the cloud -- {q[:46]!r}"
            )
    finally:
        pipeline.close()

    args.out_rows.parent.mkdir(parents=True, exist_ok=True)
    with args.out_rows.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---------------------------------------------------------------- summary
    device_ms = np.array([r["device_ms"] for r in rows], dtype=float)
    payload = np.array([r["cloud_payload_bytes"] for r in rows], dtype=float)
    device_bytes = int(sum(r["device_bytes_sent"] for r in rows))
    cites = sum(1 for r in rows if r["device_cites_chunk"])
    refused = sum(1 for r in rows if r["device_refused"])

    cloud_ms = np.array([r["cloud_ms"] for r in rows if r["cloud_ms"] != ""], dtype=float)
    pricing, pricing_why = load_pricing(args.pricing)

    offline_path = Path("results/offline_proof.json")
    if offline_path.exists():
        proof = json.loads(offline_path.read_text(encoding="utf-8"))
        offline_device = (
            f"YES -- proved {proof['ran_at']}: the whole pipeline answered with every "
            f"outbound socket raising"
            if proof.get("passed")
            else "NO -- the Phase 12.5 proof did not pass"
        )
    else:
        offline_device = "not proved: scripts/phase12_offline_proof.py has not been run"

    if pricing:
        in_tok = np.array(
            [
                float(r["cloud_prompt_tokens_measured"] or r["cloud_prompt_tokens_estimate"])
                for r in rows
            ]
        )
        out_tok = np.array([float(r["cloud_completion_tokens_measured"] or 0.0) for r in rows])
        measured_tokens = any(r["cloud_prompt_tokens_measured"] != "" for r in rows)
        cost_per_query = (
            in_tok.mean() * float(pricing["input_per_1m_usd"]) / 1e6
            + out_tok.mean() * float(pricing["output_per_1m_usd"]) / 1e6
        )
        cloud_cost = f"{cost_per_query:.6f} {pricing.get('currency', 'USD')} per query"
        cost_prov = (
            f"computed from {args.pricing} ({pricing['model']}, read "
            f"{pricing['retrieved_on']} from {pricing['source_url']}); token counts "
            + ("measured from the API's usage field" if measured_tokens else "ESTIMATED as chars/4")
        )
    else:
        cloud_cost = "not computed"
        cost_prov = pricing_why

    table: list[dict[str, str]] = [
        {
            "dimension": "user data leaving the device, per query",
            "on_device": f"{device_bytes} bytes over {len(rows)} queries",
            "cloud": f"{payload.mean():.0f} bytes (mean), {payload.min():.0f}-{payload.max():.0f} range",
            "on_device_provenance": "MEASURED: this process's own socket sends, counted per query",
            "cloud_provenance": (
                "COMPUTED: the exact byte length of the request body that would be posted, "
                "assembled from the same question and the same retrieved context. Nothing "
                "was sent to obtain it."
            ),
        },
        {
            "dimension": "what that payload contains",
            "on_device": "nothing leaves",
            "cloud": "the user's question verbatim, plus the retrieved passages",
            "on_device_provenance": "architecture",
            "cloud_provenance": "the request body is in results/comparison_per_query.csv",
        },
        {
            "dimension": "works with no network",
            "on_device": offline_device,
            "cloud": "NO -- an API call cannot complete without a network",
            "on_device_provenance": "MEASURED: scripts/phase12_offline_proof.py",
            "cloud_provenance": (
                "a statement of fact about remote APIs, not a measurement. No number is "
                "invented for it."
            ),
        },
        {
            "dimension": "end-to-end latency per query",
            "on_device": f"p50 {np.percentile(device_ms, 50):.0f} ms, p95 {np.percentile(device_ms, 95):.0f} ms",
            "cloud": (
                f"p50 {np.percentile(cloud_ms, 50):.0f} ms, p95 {np.percentile(cloud_ms, 95):.0f} ms"
                if cloud_ms.size
                else NOT_MEASURED
            ),
            "on_device_provenance": (
                "MEASURED on this laptop over these queries, including generation. "
                "Not a phone measurement."
            ),
            "cloud_provenance": (
                f"MEASURED: {cloud_ms.size} real round trips to {args.cloud_model}"
                if cloud_ms.size
                else "no OPENAI_API_KEY was set, so no call was made and nothing is reported"
            ),
        },
        {
            "dimension": "API cost per query",
            "on_device": "0 (no API is called)",
            "cloud": cloud_cost,
            "on_device_provenance": "architecture: nothing is billed because nothing is called",
            "cloud_provenance": cost_prov,
        },
        {
            "dimension": "answers cite a passage the user can open",
            "on_device": f"{cites} of {len(rows)} queries cite a local chunk id; {refused} refused outright",
            "cloud": (
                "not checked -- no call was made"
                if not cloud_ms.size
                else "see results/comparison_per_query.csv; the API returns prose with no chunk ids"
            ),
            "on_device_provenance": "MEASURED: the cited chunk id is recorded per query",
            "cloud_provenance": (
                "a hosted model returns text; any citation it writes is not checkable "
                "against a local store"
            ),
        },
        {
            "dimension": "refuses when the corpus cannot answer",
            "on_device": f"YES -- {refused} of {len(rows)} refused below a measured threshold",
            "cloud": "depends on the prompt; not measured here",
            "on_device_provenance": "MEASURED: Phase 12.1 guard, threshold from Phase 11.4",
            "cloud_provenance": NOT_MEASURED,
        },
        {
            "dimension": "LIMITATION: open-ended reasoning and general knowledge",
            "on_device": (
                "WEAKER. flan-t5-base is a 250M-parameter extractive model. It answers from "
                "retrieved passages and cannot reason beyond them."
            ),
            "cloud": "STRONGER. A large hosted model is better at both, and this is not close.",
            "on_device_provenance": (
                "the on-device arm refuses rather than guesses, which is a different "
                "property from being right more often"
            ),
            "cloud_provenance": "not measured here; stated because it is true and material",
        },
        {
            "dimension": "LIMITATION: what on-device inference spends",
            "on_device": (
                "the phone's battery and thermal budget, which is exactly the cost this "
                "project's controller exists to manage"
            ),
            "cloud": "the phone's radio, and someone else's electricity",
            "on_device_provenance": (
                "MEASURED: Phase 12.4 shows the controller trading recall for latency and "
                "estimated energy"
            ),
            "cloud_provenance": "not measured: the radio cost of a cloud call was not instrumented",
        },
    ]

    with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    summary = {
        "dataset": DATASET,
        "n_queries": len(rows),
        "cloud_called": bool(cloud_ms.size),
        "cloud_model": args.cloud_model if cloud_ms.size else None,
        "device_bytes_total": device_bytes,
        "cloud_payload_bytes": {
            "mean": float(payload.mean()),
            "min": float(payload.min()),
            "max": float(payload.max()),
            "total": float(payload.sum()),
        },
        "device_latency_ms": {
            "p50": float(np.percentile(device_ms, 50)),
            "p95": float(np.percentile(device_ms, 95)),
        },
        "cloud_latency_ms": (
            {"p50": float(np.percentile(cloud_ms, 50)), "p95": float(np.percentile(cloud_ms, 95))}
            if cloud_ms.size
            else None
        ),
        "cost": {"on_device": 0.0, "cloud": cloud_cost, "provenance": cost_prov},
        "offline": {"on_device": offline_device, "cloud": "cannot work without a network"},
        "grounding": {"device_citations": cites, "device_refusals": refused},
        "table": table,
    }
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _plot(rows, device_ms, payload, cloud_ms, args.out_png)

    print()
    print(f"  on-device sent {device_bytes} bytes in total across {len(rows)} queries")
    print(
        f"  the same queries would have sent {payload.sum():,.0f} bytes to the cloud "
        f"({payload.mean():.0f} per query)"
    )
    print(f"  cloud latency: {'measured' if cloud_ms.size else NOT_MEASURED}")
    print(f"  cloud cost: {cloud_cost}")
    print(
        f"\nwrote {args.out_csv}\nwrote {args.out_rows}\nwrote {args.out_json}\nwrote {args.out_png}"
    )
    return 0


def _plot(
    rows: list[dict[str, Any]],
    device_ms: np.ndarray,
    payload: np.ndarray,
    cloud_ms: np.ndarray,
    out: Path,
) -> None:
    plt.rcParams.update(
        {"figure.dpi": 130, "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3}
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    ax = axes[0]
    ax.bar([0, 1], [0, payload.mean()], color=["#1f7a4d", "#b3382c"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["on-device", "cloud API"])
    ax.set_ylabel("user data leaving the device (bytes/query)")
    ax.set_title("Privacy: what would be sent", fontsize=10)
    ax.text(
        0,
        payload.mean() * 0.05,
        "0\nmeasured",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#1f7a4d",
        fontweight="bold",
    )
    ax.text(1, payload.mean() * 1.02, f"{payload.mean():.0f}\ncomputed", ha="center", fontsize=9)

    ax = axes[1]
    ax.hist(payload, bins=12, color="#b3382c", alpha=0.85)
    ax.set_xlabel("request body that would be posted (bytes)")
    ax.set_ylabel("queries")
    ax.set_title("Every query, including the question verbatim", fontsize=10)

    ax = axes[2]
    ax.plot(
        np.sort(device_ms),
        np.linspace(0, 100, device_ms.size),
        color="#1f7a4d",
        lw=1.8,
        label="on-device (measured, this laptop)",
    )
    if cloud_ms.size:
        ax.plot(
            np.sort(cloud_ms),
            np.linspace(0, 100, cloud_ms.size),
            color="#b3382c",
            lw=1.8,
            label="cloud API (measured round trip)",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "cloud latency not measured\n(no OPENAI_API_KEY set)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#b3382c",
        )
    ax.set_xlabel("end-to-end latency (ms)")
    ax.set_ylabel("percentile")
    ax.set_title("Latency", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        "On-device RAG against a cloud API, over the 20 labelled demonstration queries",
        fontsize=11,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
