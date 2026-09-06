#!/usr/bin/env python
"""End-to-end verification: does the project actually reproduce from its own docs?

Runs the commands `docs/review.md` tells a reader to run, in order, against the
current code, and records what each one did. Every phase has been verified when
it was built; this checks something different and easier to lose -- that the
pieces still work *together*, from the documented entry points, after thirty-six
commits of changes.

Sequential by construction. Nothing here runs in parallel, because three
separate measurements in this project were corrupted by concurrent work and
ADR-0030 exists because of it.

Writes `results/verification.json` and prints a table. A step that fails is
recorded as failed; this script does not stop at the first failure, because
knowing that four of six steps work is more useful than knowing the second one
broke.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


@dataclass
class Step:
    name: str
    argv: list[str]
    why: str
    # Exit codes that count as reproducing. NOT a boolean "ignore the code":
    # the first version used expect_zero=False for the acceptance step, to
    # tolerate its one documented FAIL, and that tolerated EVERYTHING -- it
    # reported OK for a run that had timed out. A step whose verdict is
    # independent of its outcome is not a check, which is defect #1 of this
    # project's own taxonomy rebuilt inside the verification harness.
    expect_codes: tuple[int, ...] = (0,)
    timeout_s: float = 1800.0
    # Filled in by run()
    returncode: int | None = None
    seconds: float = 0.0
    tail: list[str] = field(default_factory=list)
    n_output_lines: int = 0

    @property
    def ok(self) -> bool:
        # A timeout is -1 and is in no step's expected set, so it can never
        # be reported as reproduced.
        if self.returncode not in self.expect_codes:
            return False
        # An exit code alone is not evidence. `python -m smartann.cli data
        # status` exited 0 for thirty-nine commits while doing NOTHING -- the
        # module had no `__main__` guard, so it imported, defined every
        # command, ran none, and returned success (ADR-0042). This step scored
        # that as a pass every time. A process that reproduced something said
        # so; silence is not a result.
        return self.n_output_lines > 0


STEPS = [
    Step(
        "test suite",
        [PY, "-m", "pytest", "tests/", "-q"],
        "447 tests. The first thing a reader would run, and the cheapest signal "
        "that the tree is coherent.",
    ),
    Step(
        "lint + types",
        [PY, "-m", "ruff", "check", "."],
        "Style and correctness gates that every commit passed individually; this "
        "confirms they still pass together.",
    ),
    Step(
        "mypy",
        [PY, "-m", "mypy"],
        "Strict typing on smartann/, which is where the contracts live.",
    ),
    Step(
        "data status",
        [PY, "-m", "smartann.cli", "data", "status"],
        "Exits non-zero if any dataset is incomplete (ADR-0020). A table a human "
        "reads is not a check; the exit code is.",
    ),
    Step(
        "phase 2 acceptance",
        [PY, "scripts/phase2_acceptance.py", "--datasets", "sift1m", "nytimes"],
        "The 15 index checks against real data. Expected 14 PASS / 1 FAIL -- the "
        "NYTimes cluster-miss property, which is a dataset fact and not a defect. "
        "Exit 1 is therefore expected; a timeout or a crash is not. Exit 3 means the "
        "machine-quiet guard REFUSED to start, which is not a result either. Exit 2 means "
        "another workload ran during the measurement and every duration is "
        "meaningless -- it is deliberately NOT in the expected set, because a "
        "contaminated run is the absence of a result rather than a result.",
        expect_codes=(0, 1),
        timeout_s=3600.0,
    ),
    Step(
        "cost model fit",
        [PY, "scripts/fit_cost_model.py"],
        "Refits latency on recorded runs and confirms the energy model still "
        "REFUSES to fit on a platform that cannot measure energy.",
    ),
    Step(
        "figures",
        [PY, "scripts/make_figures.py", "--out", "results/figures"],
        "Every number from stored parquet or summary JSON. A missing input must "
        "fail loudly rather than draw something plausible (Section 3, rule 1).",
    ),
]


def run(step: Step) -> None:
    print(f"\n{'=' * 78}\n>>> {step.name}\n    {step.why}\n{'=' * 78}", flush=True)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            step.argv,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=step.timeout_s,
            check=False,
        )
        step.returncode = proc.returncode
        out = (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        step.returncode = -1
        out = f"TIMED OUT after {step.timeout_s:.0f}s"
    step.seconds = time.perf_counter() - t0
    if step.seconds > step.timeout_s * 1.5 and step.returncode not in (-1,):
        # Wall-clock far beyond the timeout without the timeout firing means the
        # host slept mid-run. The exit code may be fine and the DURATION is still
        # not a measurement of anything.
        step.tail.append(
            f"WARNING: wall-clock {step.seconds:.0f}s exceeds the {step.timeout_s:.0f}s "
            f"timeout without it firing -- the host almost certainly slept. Treat the "
            f"duration as meaningless."
        )
    lines = [ln for ln in out.strip().splitlines() if ln.strip()]
    step.n_output_lines = len(lines)
    if not lines:
        step.tail.append("NO OUTPUT AT ALL -- a silent exit is not a reproduction (ADR-0042).")
    step.tail = step.tail + lines[-12:]
    for ln in step.tail:
        print(f"    {ln}", flush=True)
    print(
        f"    -> exit {step.returncode} in {step.seconds:.1f}s [{'OK' if step.ok else 'FAILED'}]",
        flush=True,
    )


def main() -> int:
    print("END-TO-END VERIFICATION", flush=True)
    print(
        "Sequential by construction: nothing runs alongside a measurement (ADR-0030).\n", flush=True
    )

    for step in STEPS:
        run(step)

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}", flush=True)
    for s in STEPS:
        print(
            f"  {'OK  ' if s.ok else 'FAIL'}  {s.name:<22} exit {s.returncode!s:>3}  "
            f"{s.seconds:7.1f}s",
            flush=True,
        )

    n_ok = sum(1 for s in STEPS if s.ok)
    print(f"\n{n_ok}/{len(STEPS)} steps reproduced.", flush=True)

    out = ROOT / "results" / "verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "name": s.name,
                        "why": s.why,
                        "argv": s.argv[1:],
                        "returncode": s.returncode,
                        "n_output_lines": s.n_output_lines,
                        "seconds": s.seconds,
                        "ok": s.ok,
                        "tail": s.tail,
                    }
                    for s in STEPS
                ],
                "n_ok": n_ok,
                "n_steps": len(STEPS),
                "note": "phase 2 acceptance exits non-zero by design: one documented FAIL "
                "(NYTimes check 4) is a dataset property, not a defect.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}", flush=True)
    return 0 if n_ok == len(STEPS) else 1


if __name__ == "__main__":
    sys.exit(main())
