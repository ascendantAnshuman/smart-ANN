# The conference paper

IEEEtran, conference mode, targeting 8 pages.

## Build it

There is no LaTeX toolchain in this repo. Two routes:

**Overleaf** (easiest). Upload `main.tex` and the eight PDFs from
`results/paper/`, putting the figures in the same folder as `main.tex`, then
change the graphics path at the top of `main.tex`:

```latex
\graphicspath{{./}}
```

**Locally**, with TeX Live or MiKTeX installed:

```bash
python scripts/make_paper_figures.py   # writes results/paper/*.pdf
cd paper
pdflatex main && pdflatex main         # twice, for the cross-references
```

## Before you submit

**Regenerate and re-check after any experiment re-run.** Every number in the
paper is traceable to a file under `results/`, and a script asserts that:

```bash
python scripts/make_paper_figures.py
python scripts/check_paper_numbers.py
```

`check_paper_numbers.py` runs 85 assertions against the recorded results and
fails if a figure in the text has drifted from the file it came from. It caught
two errors while the paper was being written. Run it before every submission.

**Verify the references yourself.** Entries [1]-[10] were written from memory
of the literature; [11]-[25] were supplied by the authors. Neither set has been
checked against the publishers' records. Confirm every author list, venue, year and page range — in particular
`park2025mobilerag`, whose details came from this repository's own README
rather than from the paper itself. An incorrect citation is the reviewer's
first impression.

**Author block.** Three authors, all at School of Computer Science and
Engineering, Vellore Institute of Technology, Vellore, India. No placeholders
remain.

**Check the page count.** With 25 references the text is roughly 5.9 pages;
the eight figures, six tables and three-author title block bring it to about
**9**. If your venue counts references against an 8-page limit you must cut.
In order of least damage to the argument:

1. Section VI-C (joint control) — three sentences and no figure; the weakest
   standalone claim. Saves ~0.2 page.
2. Section VI-B (context reduction) and Table IV — a negative result, but not
   the one the paper is built on. Saves ~0.5 page.
3. Section VI-D (storage regime) and Table V — interesting, but it argues
   methodology rather than the contribution. Saves ~0.5 page.
4. Figure 2 — its point is restated in the Table I caption. Saves ~0.3 page.

Do **not** cut Figures 3, 4, 5 or 8: they carry the two novelty claims.

## What each figure shows, and where its numbers come from

| Figure | Claim | Source |
|---|---|---|
| 1 | The two-phase split is what makes per-query allocation possible | drawn |
| 2 | The effort/return curve is steeply asymmetric | `device_profile_table.json` |
| 3 | Difficulty separates recall 3.5x; marginal value differs 45x | `phase9/query_adaptive.json` |
| 4 | Allocation beats uniform effort on held-out queries | `phase9/query_adaptive.json` |
| 5 | The skin sensor never reaches the derate threshold; the hot zone always does | `device/*_telemetry.jsonl` |
| 6 | Budgeted control cuts deadline misses 35x for 0.068 recall | `phase6/policy_comparison.json` |
| 7 | Power rises 98% with retrieval work | `energy_model.json` |
| 8 | The sensor choice determines the entire behaviour of the shield | `device/c5_telemetry.jsonl` |

## The novelty claims, in order of strength

1. **Query-difficulty-conditioned effort allocation from a probe-time signal**
   (Section V). The signal is computed for free by the index and was previously
   discarded. Held-out validated against handset-measured costs.
2. **The sensor-divergence negative result** (Section VII). Reproducible across
   six runs, and directly actionable for anyone building a thermally-aware
   mobile system.
3. **Per-query energy measured rather than extrapolated** (Section VI-D).
4. **Distance computations are the wrong cost model on a phone** (Section VI-E):
   per-cluster graphs cut distance work 4x and still lose 2.6x on wall-clock.
