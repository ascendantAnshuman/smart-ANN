#!/usr/bin/env python
"""Bundle a paper into one zip that compiles on Overleaf without edits.

There are two manuscripts drawn from the same recorded results:

    paper/main.tex      the 8-page IEEE conference version
    paper/journal.tex   the ~18-page journal version

Both are packaged by this script, separately, into their own zip:

    python scripts/package_paper.py              # conference (the default)
    python scripts/package_paper.py --journal    # journal
    python scripts/package_paper.py --both       # both

**It refuses to package a paper whose numbers have drifted.** The check that
asserts every figure in the manuscript against the file it came from runs
first, against that manuscript, and a failure stops the build. A zip is what
gets emailed to a co-author and uploaded to a submission site, so it is the
last place a stale number can hide.

**The graphics path is rewritten, not maintained twice.** In the repository
both manuscripts read their figures from `../results/paper/`, which is where
the figure script writes them. Overleaf has no such parent, so the packaged
copy gets `\\graphicspath{{./}}` and the figures sit beside it. Rewriting on
the way out means the two copies cannot disagree about where a figure lives.

The bundle also carries the result files the paper's numbers came from and the
scripts that read them, so a reader who wants to check a figure can, without
cloning anything.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path()
OUT = Path("dist")

# The result files every number in the paper is drawn from. Listed rather than
# globbed: a reader should be able to see what the paper rests on, and a file
# that quietly appears in results/ should not quietly appear in a submission.
RESULTS = [
    "device_profile_table.json",
    "phase9/query_adaptive.json",
    "phase6/policy_comparison.json",
    "phase7/qa_summary.json",
    "phase8/joint_summary.json",
    "energy_model.json",
    "thermal_model.json",
    "thermal_attribution.json",
    "device/c5_summary.json",
    "phase2_summary.json",
    "cost_model.json",
    "regime_summary.json",
    "verification.json",
]

FIGURES = [
    "fig1_architecture",
    "fig2_effort_return",
    "fig3_difficulty",
    "fig4_c6",
    "fig5_sensors",
    "fig6_policies",
    "fig7_energy",
    "fig8_trace",
]

CONFERENCE_NOTES = """\
SMART-ANN -- conference paper (IEEE, 2 columns, ~9 pages)
=========================================================

OVERLEAF (easiest)
  1. New Project -> Upload Project -> select this zip.
  2. Set the main document to main.tex if it is not detected.
  3. Compile. The graphics path already points at this folder.

LOCALLY (TeX Live or MiKTeX)
  pdflatex main && pdflatex main      (twice, for cross-references)

BEFORE SUBMITTING
  - Verify every bibliography entry against the publisher's record.
    Entries [1]-[10] were written from memory and [11]-[25] were supplied by
    the authors. NEITHER SET HAS BEEN CHECKED.
  - Check the page count against the venue's limit. README.md lists what to
    cut first if you are over.

WHAT IS IN HERE
  main.tex            the paper
  fig*.pdf            the eight figures, vector, built from results/
  figures-png/        the same figures as images, for previewing
  results/            every file the paper's numbers are drawn from
  scripts/            the code that turns results/ into the figures, and the
                      checker that asserts the paper's numbers against their
                      sources
  README.md           fuller notes, including what to cut if over length
"""

JOURNAL_NOTES = """\
SMART-ANN -- journal paper (IEEEtran, ~18-20 pages)
===================================================

This is the expanded journal version of the same work as the conference
paper. Same measurements, same figures, more depth: a formal problem
statement, two algorithm listings, six additional tables of data that were
only summarised before, a Discussion, Threats to Validity, and 15 newer
references.

OVERLEAF (easiest)
  1. New Project -> Upload Project -> select this zip.
  2. Set the main document to journal.tex if it is not detected.
  3. Compile three times (cross-references, then float placement settles).

LOCALLY (TeX Live or MiKTeX)
  pdflatex journal && pdflatex journal && pdflatex journal

  Needs: IEEEtran, algorithm, algpseudocode, booktabs, multirow, cite,
  hyperref. All are in a standard TeX Live install.

SUBMISSION vs CAMERA-READY
  The file is two-column camera-ready by default. Many IEEE Transactions want
  a double-spaced single-column file at first submission; the comment block
  under \\documentclass gives the one-line swap. Nothing else changes.

THIS FILE HAS NOT BEEN COMPILED
  There is no LaTeX toolchain in the repository it was written in. It has been
  checked structurally -- balanced environments, no undefined \\ref or \\cite,
  no uncited \\bibitem, matched tabular column counts -- and every number in it
  passes scripts/check_paper_numbers.py against the recorded results. But
  compile it once yourself before you trust the layout.

BEFORE SUBMITTING
  - Verify every bibliography entry. The 24 carried over from the conference
    version were never checked against publishers' records. The 15 newer ones
    were read off arXiv listing pages during the revision; confirm them,
    especially park2025mobilerag (the baseline) and anything cited as
    "SIGMOD 2026".
  - If the conference version has been submitted, published or posted
    anywhere, disclose that in the cover letter and cite it. Journals expect a
    substantially extended version of a conference paper to say so.
  - Re-run the checker after any experiment re-run:
        python scripts/check_paper_numbers.py paper/journal.tex

WHAT IS IN HERE
  journal.tex         the paper
  fig*.pdf            the eight figures, vector, built from results/
  figures-png/        the same figures as images, for previewing
  results/            every file the paper's numbers are drawn from
  scripts/            the code that turns results/ into the figures, and the
                      checker that asserts the paper's numbers against their
                      sources
  README.md           fuller notes: what changed, what to cut if over length,
                      and what to verify before submitting

NOTE ON FIGURE NUMBERS
  The journal version places the energy figure before the thermal ones, so the
  rendered figure numbers no longer match the filenames:
      fig5_sensors.pdf  renders as Fig. 7
      fig6_policies.pdf renders as Fig. 5
      fig7_energy.pdf   renders as Fig. 6
      fig8_trace.pdf    renders as Fig. 8
"""

PAPERS = {
    "conference": {
        "zip": "smartann-paper",
        "tex": "paper/main.tex",
        "as": "main.tex",
        "readme": "paper/README.md",
        "notes": CONFERENCE_NOTES,
    },
    "journal": {
        "zip": "smartann-journal-paper",
        "tex": "paper/journal.tex",
        "as": "journal.tex",
        "readme": "paper/README-journal.md",
        "notes": JOURNAL_NOTES,
    },
}


def run(label: str, argv: list[str]) -> bool:
    """Run a checking step. The repository root goes on PYTHONPATH because the
    checker imports `smartann` to replay the telemetry traces, and a plain
    checkout has not been pip-installed."""
    print(f"  {label} ...", end=" ", flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(Path.cwd()), env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    r = subprocess.run(argv, capture_output=True, text=True, check=False, env=env)
    if r.returncode != 0:
        print("FAILED")
        print(r.stdout[-2000:] or r.stderr[-2000:])
        return False
    print("ok")
    return True


def package(kind: str) -> int:
    spec = PAPERS[kind]
    py = sys.executable
    src_tex = ROOT / spec["tex"]

    if not src_tex.exists():
        print(f"refusing to package: {src_tex} does not exist", file=sys.stderr)
        return 1

    print(f"\n=== {kind} paper ({spec['tex']}) ===")
    print("checking the paper against the recorded results")
    if not run("figures are current", [py, "scripts/make_paper_figures.py"]):
        return 1
    if not run(
        "numbers match their sources",
        [py, "scripts/check_paper_numbers.py", spec["tex"]],
    ):
        print("\nrefusing to package: a number in the paper does not match its source")
        return 1

    stage = OUT / spec["zip"]
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    # The manuscript, with the graphics path pointed at the bundle's own folder.
    tex = src_tex.read_text(encoding="utf-8")
    packaged = re.sub(r"\\graphicspath\{(?:\{[^}]*\})+\}", r"\\graphicspath{{./}}", tex)
    if packaged == tex:
        print(
            f"warning: the graphics path was not rewritten; check {spec['tex']}",
            file=sys.stderr,
        )
    (stage / spec["as"]).write_text(packaged, encoding="utf-8")

    readme = ROOT / spec["readme"]
    if readme.exists():
        shutil.copy2(readme, stage / "README.md")
    else:
        print(f"  note: {spec['readme']} is not on disk and is omitted")

    # Figures: PDF beside the manuscript for the build, PNG for preview.
    png = stage / "figures-png"
    png.mkdir()
    missing = []
    for f in FIGURES:
        src = ROOT / "results" / "paper" / f"{f}.pdf"
        if not src.exists():
            missing.append(f"{f}.pdf")
            continue
        shutil.copy2(src, stage / f"{f}.pdf")
        p = ROOT / "results" / "paper" / f"{f}.png"
        if p.exists():
            shutil.copy2(p, png / f"{f}.png")
    if missing:
        print(f"refusing to package: missing figures {missing}", file=sys.stderr)
        return 1

    # The evidence, and the code that reads it.
    for rel in RESULTS:
        src = ROOT / "results" / rel
        if not src.exists():
            print(f"  note: {rel} is not on disk and is omitted")
            continue
        dst = stage / "results" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    (stage / "scripts").mkdir()
    for s in ("make_paper_figures.py", "check_paper_numbers.py", "phase9_query_adaptive.py"):
        shutil.copy2(ROOT / "scripts" / s, stage / "scripts" / s)

    (stage / "HOW-TO-BUILD.txt").write_text(spec["notes"], encoding="utf-8")

    zip_path = OUT / f"{spec['zip']}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(stage))

    n = sum(1 for p in stage.rglob("*") if p.is_file())
    print(f"wrote {zip_path}  ({zip_path.stat().st_size / 1024:,.0f} KB, {n} files)")
    print(f"staged at {stage}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--journal", action="store_true", help="package paper/journal.tex")
    g.add_argument("--both", action="store_true", help="package both manuscripts")
    args = ap.parse_args()

    if args.both:
        kinds = ["conference", "journal"]
    elif args.journal:
        kinds = ["journal"]
    else:
        kinds = ["conference"]

    for kind in kinds:
        rc = package(kind)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
