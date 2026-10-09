#!/usr/bin/env python
"""12.5 -- answer the demonstration question with the network taken away.

The privacy claim in this project is not "we do not send anything on purpose".
It is "this cannot send anything", and the difference is testable. Every
outbound connect, datagram send and name resolution is patched to raise, and
then the whole pipeline runs: index open, embedding, controller, retrieval,
grounding guard, generation.

If any stage reaches for the network the run fails with the attempt named, which
is the point. A model that quietly checks for an update on first use, a
tokenizer that resolves a hub URL, a telemetry call in a dependency -- all of
them are invisible to a byte counter on a machine that happens to be offline,
and all of them fail loudly here.

**Loading happens inside the block too.** Loading the generator outside it and
only answering inside would prove the easy half. The expensive, surprising part
is bringing a 250 MB model up from a local cache with no hub reachable, and that
is exactly where a hidden fetch would live.

**And it found one.** With the libraries at their defaults, loading a model that
is already in the local cache still opens a connection to `huggingface.co:443`
to check it. That is the ML libraries' behaviour at load time, not the query
path's: the question itself sends nothing either way, which the query traces
measure separately. Setting `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`
tells them to trust the cache, and the whole pipeline then runs with every
outbound path raising.

**Both runs are recorded.** Reporting only the run with the switch set would
claim an offline system while quietly depending on a network call; reporting
only the run without it would hide that a one-line fix exists. A real deployment
ships the model files on the device, where neither applies.

**Each run gets its own process, and that is load-bearing.** Run both in one
interpreter and the second inherits the first's resolved model paths, cached in
module state. The default-configuration run then "passes" for the sole reason
that the offline run before it had already done the work -- a green result
produced by the test's own structure. Subprocesses make the two independent.

Writes `results/offline_proof.json`, which is what the dashboard's "works
offline" badge reads. The badge reports the result of this run, with its
timestamp, rather than a claim typed into a template.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smartann.pipeline.offline import NetworkBlockedError, block_network

QUESTION = "Why does my phone get hot even when it is not charging?"
OFFLINE_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


def attempt(question: str, dataset: str, nprobe: int, offline_env: bool) -> dict[str, Any]:
    """One run of the whole pipeline with the network blocked.

    `offline_env` sets the two variables that tell the ML libraries to trust
    their local cache. The run without them is kept because its failure is the
    finding, not an inconvenience to be configured away.
    """
    started = time.perf_counter()
    passed = False
    answer = ""
    error = ""
    attempts: list[str] = []
    n_chunks = 0

    saved = {k: os.environ.get(k) for k in OFFLINE_VARS}
    for key in OFFLINE_VARS:
        if offline_env:
            os.environ[key] = "1"
        else:
            os.environ.pop(key, None)

    try:
        with block_network() as report:
            # Imported inside the block as well, so that an import-time fetch in
            # any dependency is caught rather than paid for in advance.
            from smartann.pipeline.ask import SCR_OFF, RagPipeline
            from smartann.pipeline.guard import GroundingGuard

            guard = None
            guard_path = Path("results/grounding_threshold.json")
            if guard_path.exists():
                try:
                    guard = GroundingGuard.load(guard_path, dataset=dataset)
                except ValueError:
                    guard = None

            pipeline = RagPipeline(dataset=dataset, k=5, use_kb=False)
            try:
                pipeline.load()
                result = pipeline.ask(question, nprobe=nprobe, scr=SCR_OFF, guard=guard)
            finally:
                pipeline.close()

            answer = result.answer
            n_chunks = len(result.chunks)
            attempts = list(report.attempts)
            passed = bool(answer.strip()) and n_chunks > 0
    except NetworkBlockedError as exc:
        error = f"the pipeline tried to use the network: {exc}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    return {
        "offline_env_set": offline_env,
        "passed": passed,
        "answer": answer,
        "n_chunks_retrieved": n_chunks,
        "seconds": time.perf_counter() - started,
        "network_attempts": attempts,
        "error": error,
    }


def run_in_subprocess(script: Path, mode: str, args: argparse.Namespace) -> dict[str, Any]:
    """One attempt, in a fresh interpreter, so it inherits no cached state."""
    import subprocess

    out = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            mode,
            "--question",
            args.question,
            "--dataset",
            args.dataset,
            "--nprobe",
            str(args.nprobe),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in reversed(out.stdout.splitlines()):
        if line.startswith("{"):
            parsed: dict[str, Any] = json.loads(line)
            return parsed
    return {
        "offline_env_set": mode == "with",
        "passed": False,
        "answer": "",
        "n_chunks_retrieved": 0,
        "seconds": float("nan"),
        "network_attempts": [],
        "error": f"the subprocess produced no result (exit {out.returncode}): "
        + (out.stderr.strip().splitlines() or ["no stderr"])[-1],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--question", default=QUESTION)
    ap.add_argument("--dataset", default="demo_phone_care")
    ap.add_argument("--nprobe", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("results/offline_proof.json"))
    ap.add_argument(
        "--mode",
        choices=("both", "with", "without"),
        default="both",
        help="Internal: 'with'/'without' run ONE attempt and print its JSON, so that "
        "'both' can run each in its own interpreter.",
    )
    args = ap.parse_args()

    if args.mode in ("with", "without"):
        result = attempt(args.question, args.dataset, args.nprobe, offline_env=args.mode == "with")
        print(json.dumps(result))
        return 0 if result["passed"] else 1

    script = Path(__file__).resolve()
    print(f"blocking all outbound network, then answering: {args.question!r}")

    print()
    print("  run 1: libraries told to trust their local cache (own process)")
    with_env = run_in_subprocess(script, "with", args)
    verdict = "PASSED" if with_env["passed"] else "FAILED"
    detail = f" -- {with_env['error']}" if with_env["error"] else ""
    print(f"    {verdict} in {with_env['seconds']:.1f}s{detail}")

    print()
    print("  run 2: libraries at their defaults (own process)")
    without_env = run_in_subprocess(script, "without", args)
    verdict = "PASSED" if without_env["passed"] else "FAILED"
    detail = f" -- {without_env['error']}" if without_env["error"] else ""
    print(f"    {verdict} in {without_env['seconds']:.1f}s{detail}")

    passed = bool(with_env["passed"])
    payload = {
        "test": "offline_proof",
        "passed": passed,
        "ran_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "question": args.question,
        "dataset": args.dataset,
        "answer": with_env["answer"],
        "n_chunks_retrieved": with_env["n_chunks_retrieved"],
        "seconds": with_env["seconds"],
        "runs": {"with_offline_env": with_env, "without_offline_env": without_env},
        "finding": (
            "With HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 the whole pipeline runs "
            "with every outbound path raising. Without them, loading a model that is "
            "ALREADY in the local cache still opens a connection to huggingface.co to "
            "check it. That is the ML libraries' behaviour at load time, not the query "
            "path: the question itself sends nothing either way, which the query traces "
            "measure separately."
        ),
        "what_was_blocked": (
            "socket.connect, socket.connect_ex, socket.sendto, socket.create_connection "
            "and socket.getaddrinfo, including loopback. Name resolution is included "
            "because a DNS lookup is itself traffic leaving the device."
        ),
        "what_this_proves": (
            "the index, embedder, generator and controller all ran from local files "
            "with every outbound path raising. It does not prove the models were "
            "obtained offline -- they were downloaded once into the local cache."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print()
    print(f"wrote {args.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
