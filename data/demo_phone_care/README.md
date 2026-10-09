# The phone-care demonstration corpus

71 short documents about why a smartphone gets hot, written for this project.

## What this is, and what it is not

**Original text, authored for the demonstration.** Nothing here was scraped,
copied or generated from a source document. Every record carries
`"source": "authored for demo"` for that reason, and the label is in the data
rather than in this file so it travels with the document into the index, the
trace and the interface.

**It is explanatory writing, not a measurement.** The documents describe how
phones behave in general terms. They contain no numbers produced by this
project, and nothing in them should be quoted as a result of it. Everything this
project *measured* lives in `results/` and is labelled there.

**It is committed, unlike everything else under `data/`.** The rest of `data/`
is downloaded or derived and is reproducible from a script, so `.gitignore`
excludes it. This corpus has no upstream to re-fetch it from: if it were
ignored, a fresh clone could not run the demonstration. `.gitignore` carries an
explicit exception for this directory.

## Files

| File | What it holds |
|---|---|
| `documents.json` | the 71 documents: `id`, `title`, `text`, `source` |
| `eval_queries.json` | the Phase 11.4 evaluation set and its hand-written relevance labels |

## The documents

**47 on the demonstration question** — why a phone heats up when it is *not*
charging.

*Causes*: sustained processor and graphics load, video recording and playback,
the camera, on-device AI, augmented reality and face effects, video calls,
background apps and synchronisation, downloads and updates, large file
operations and encryption, weak mobile signal, 5G, wifi against mobile data,
mobile hotspot and tethering, navigation, voice assistants, social feeds with
autoplay, screen brightness, hot environments, humid and still air, cars,
pockets and bedding, cases, an ageing battery, older phones running newer
software, runaway apps, malware, and heat after a software update.

*Mechanism*: which components produce the heat, how a phone sheds it without a
fan, why the case can feel mild while the inside is very hot, why some processor
work costs far more than other work, what charging adds on top, why lithium-ion
cells dislike heat, and the feedback loop between heat and battery wear.

*Judgement and response*: thermal throttling, telling normal warmth from
abnormal, what to do immediately, settings and habits that help, how to measure
your own phone, common cooling myths that do not work, and when heat is
dangerous (swelling, burns, shutdowns).

**24 distractors** on unrelated phone topics: storage, photo composition and
editing, Bluetooth pairing, screen protectors, transferring photos, email setup,
ringtones, backing up contacts, memory cards, megapixels, lock screen security,
do not disturb, sharing a wifi password, SIMs and eSIMs, app permissions,
widgets, text size, voicemail, duplicate contacts, keyboard settings, finding a
lost phone, two-factor authentication, and cable standards.

The distractors are not padding. Several are deliberate near misses — the
screen-protector document is about something stuck to the outside of a phone,
like the heat-trapping case document; the megapixel document is about the
camera, like the camera-heat document; the cable document is about connectors,
which is where charging starts. Without them, retrieval would only ever have to
choose between documents that all answer the question, and a retrieval score
would prove nothing.

A few topical documents are deliberately *negative* answers — Bluetooth is
rarely why a phone is hot, and most popular cooling remedies do not work. A
corpus in which every document says "yes, this causes heat" cannot show a
retriever distinguishing a cause from a non-cause.

Each document is 169–199 words, which is one to two paragraphs. At the
3-sentence windows this project chunks with, that is 2–5 chunks per document and
255 chunks in total.

## How it is indexed

Through the same path as the SQuAD benchmark, deliberately — the same chunker,
the same `gte-small` embedder, the same `build_ecovector` builder, the same
on-disk layout:

```powershell
& $PY scripts\phase11_build_demo_index.py
```

That writes `chunks.sqlite` and `base.npy` into the data root and builds a
16-cluster index, every cluster flat and mmap-read. `resolve_index_root` finds
it through `configs/index/demo_phone_care_ecovector.yaml`.

## The evaluation labels

`eval_queries.json` holds 10 paraphrases of the heating question and 10
off-topic queries, with relevance assigned **by reading the documents, before
any retrieval was run against them**.

Two gold sets, because 47 of the 71 documents touch heat in some way and scoring
against all of them would make almost any retrieval look perfect:

- `gold_primary` — the two documents that answer the general question on their
  own. `recall@5` is measured against this set, and the set is **frozen** at
  the two it held when the corpus was 36 documents, so numbers stay comparable
  across revisions. `gold_primary_extended` adds the one document the
  expansion contributed that also answers the question alone, and both scores
  are reported rather than the yardstick being moved.
- `gold_relevant` — those two plus every specific-cause document. These count
  towards `relevant@5`, which asks how much of the retrieved context was useful.

The off-topic queries are split in two, and the split matters: five are
answerable from the distractor documents, where the right behaviour is to answer
rather than refuse, and five have nothing relevant in the corpus at all, where
the right behaviour is to decline. Phase 12.1's refusal threshold is calibrated
from the gap between those two groups, and that gap cannot be seen if they are
measured together.
