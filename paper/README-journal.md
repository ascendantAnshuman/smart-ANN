# The journal paper (`journal.tex`)

`main.tex` is the 8-page IEEE **conference** version. `journal.tex` is the
expanded **journal** version of the same work. They share every number and
every figure; they differ in scope, structure and length.

## Build it

There is no LaTeX toolchain in this repo, so `journal.tex` has **not been
compiled**. It has been checked structurally — balanced environments, zero
undefined `\ref`, zero undefined `\cite`, zero uncited `\bibitem`, matched
`tabular` column counts — but you must compile it once before you trust the
layout.

**Overleaf** (easiest):

1. New project → upload `journal.tex`.
2. Upload the eight PDFs from `results/paper/` into the same folder.
   `\graphicspath` already includes `{./}`, so nothing needs editing.
3. Compiler: pdfLaTeX. Compile three times (cross-references, then float
   placement settles).

**Locally**, with TeX Live or MiKTeX:

```bash
python scripts/make_paper_figures.py    # writes results/paper/*.pdf
cd paper
pdflatex journal && pdflatex journal && pdflatex journal
```

`journal.tex` needs `IEEEtran`, `algorithm`, `algpseudocode`, `booktabs`,
`multirow`, `cite`, `hyperref` — all in a standard TeX Live install.

### Submission vs camera-ready

The file is set to two-column camera-ready by default:

```latex
\documentclass[journal]{IEEEtran}
```

Many IEEE Transactions want a double-spaced single-column file at first
submission. The comment block directly under that line gives the one-line
swap; nothing else changes.

## Expected length

~13,200 words of body text (the conference version is ~5,800), 8 figures,
12 tables, 39 references. In two-column IEEEtran that lands around **18–20
pages**. Check your target journal's limit — IEEE Access has none, most
Transactions want 12–14 pages plus overlength charges.

If you must cut, cut in this order (least damage to the argument first):

1. Section IX-E (index footprint) — three sentences, no figure.
2. Table III (cost model terms) — the point survives in the prose.
3. Table IV (thermal window sweep) — ditto.
4. Section IX-C (joint control) and Table X.

Do **not** cut Tables V (separation matrix), VI (assignments) or VII (held-out
result), or Figures 3, 4, 7 and 8. Those carry the two novelty claims.

Note that the journal version places the energy figure before the thermal
ones, so the **rendered** figure numbers no longer match the **filenames**:
`fig5_sensors.pdf` renders as Fig. 7, `fig6_policies.pdf` as Fig. 5,
`fig7_energy.pdf` as Fig. 6, `fig8_trace.pdf` as Fig. 8.

## What is new relative to `main.tex`

Nothing in the **results** is new. Every number came out of the same files
under `results/`. What is new is structure, depth, and four tables that
report data the conference version had room only to summarise.

New sections:

| Section | What it adds |
|---|---|
| Nomenclature | Standard for a Transactions submission. |
| II. Background | Separated from Related Work; states the three mobile constraints explicitly. |
| III. Related Work | Four subsections plus a positioning table (Table I). |
| IV. Problem Formulation | Formal statement, recall definition, the objective, and the reduction to a bucketed assignment. |
| V-C | The policy algebra, plus Algorithm 1 (runtime path). |
| VI-B | Thermal model **window sweep** — new table, from `thermal_model.json`. |
| VII-C | "Is the signal monotone?" — the 25-cell ordering argument. |
| VIII-A | Algorithm 2 (offline fit) and the multiple-choice-knapsack framing. |
| VIII-E | Why the gains are bounded in absolute terms. |
| X-C | The three-part breakdown of the thermal failure. |
| XI. Discussion | Four transfer conditions, composition with other adaptive mechanisms, implications for index designers. |
| XII. Threats to Validity | Construct / internal / external, replacing the flat Limitations list. |
| XIII. Reproducibility | What the artifact enforces mechanically. |

New tables, all from existing recorded results:

| Table | Source file | What it shows |
|---|---|---|
| I (positioning) | — | Prior work on three axes. |
| III (cost model) | `cost_model.json` | Per-term R² and MAPE; why the model was not deployed. |
| IV (thermal windows) | `thermal_model.json` | R² swings 6×, τ does not. |
| V (separation) | `phase9/query_adaptive.json` | The full 5×5 recall matrix. |
| VI (assignments) | `phase9/query_adaptive.json` | The fitted `nprobe` per bucket per budget. |
| X (joint grid) | `phase8/joint_summary.json` | All nine cells, not just the verdict. |
| XI (storage regime) | `regime_summary.json` | All four sweep points, not one. |

Corrections made while expanding:

- The storage-regime wall-clock penalty is **2.1×–2.7×** across the sweep, not
  a flat 2.6×. The conference version quoted the single matched-recall point.
- The three Pareto-front cells are `(1, keep1)`, `(4, keep2)`, `(4, keep1)` —
  the 16/keep2 cell is classed `reduction_only`, not joint.
- DARTH is cited as SIGMOD 2026 (arXiv:2505.19001), and its reported speedup is
  given as the paper's own 6.8×–13.6× average rather than a peak figure.

## Before you submit — verify these yourself

**1. The references.** 24 entries carried over from `main.tex` were never
checked against publishers' records (the conference README says so). The 15 new
entries ([26]–[39]) were pulled from arXiv listing pages during this revision
and the titles, author lists and IDs were read off those pages — but **confirm
every one**, in particular:

- `park2025mobilerag` — its details came from this repo's README, not from the
  paper. This is the baseline you build on; getting it wrong is the reviewer's
  first impression.
- `liu2026percache` (arXiv:2601.11553) — the listing showed a year
  inconsistent with the arXiv ID. Check which is right.
- Anything cited as "SIGMOD 2026" — confirm it has actually appeared.

**2. The test-suite count.** The Reproducibility section no longer quotes a
number, because `results/verification.json` says 447 and `main.tex` says 599.
Re-run `pytest tests/ -q`, then put the real number back if you want it.

**3. Re-run the number check.** Every quantitative claim should still trace to
a file under `results/`:

```bash
python scripts/make_paper_figures.py
python scripts/check_paper_numbers.py
```

Note that `check_paper_numbers.py` was written against `main.tex`. The journal
version adds tables it does not yet assert — Tables V, VI, VII, VIII, X and XI.
Extending it to cover them is worth an hour.

## On originality and similarity checking

Two things are worth separating.

**Self-overlap is the real risk here.** This manuscript shares its results, and
in places its phrasing, with `main.tex`, `docs/SMART-ANN_Review2_Final.html`,
`docs/viva_handbook.html` and the review PDFs in `docs/`. If any of those has
been uploaded, posted, submitted or indexed anywhere, a similarity checker will
flag this file against your own earlier work. That is a real problem for a
journal submission and it is not solved by rewording — it is solved by
disclosing it. If `main.tex` was submitted or published anywhere, say so in the
cover letter and cite it; most journals permit a substantially extended version
of a conference paper and expect exactly that disclosure.

**Everything else is just citation hygiene.** Quote nothing without quotation
marks, cite every claim you did not measure yourself, and make sure each of the
39 references says what you say it says. A paper whose numbers all trace to
recorded runs and whose sources are all real has nothing to hide from a
similarity report; a high percentage on such a report is almost always
bibliography and standard terminology, which you can usually exclude in the
tool's settings.

I did not write this text to score a particular way against an AI-detection
tool, and you should not ask anyone to. Those tools are unreliable in both
directions, and text engineered to defeat them reads worse, not better. If your
institution requires a disclosure of AI assistance, make it — the honest
disclosure is cheaper than the alternative, and the substance of this paper is
your measurements, which no tool disputes.
