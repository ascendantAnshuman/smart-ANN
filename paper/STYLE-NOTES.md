# Style analysis: Radzik, and how `journal.tex` was rewritten against it

Source texts analysed:

1. Cooper, Kang & Radzik, *A simple model of influence*, WAW 2023.
2. Elsässer & Radzik, *Recent Results in Population Protocols for Exact
   Majority and Leader Election*, SIGACT News Distributed Computing Column.
3. Cooper, Radzik & Shiraga, *Discrete Incremental Voting on Expanders*,
   Discrete Mathematics 349(1), 2026.

The three share one hand closely enough that the features below appear in all
of them. Where a feature appears in only one or two, it is noted.

---

## 1. The single most distinctive habit: bold run-in paragraph headings

He subdivides with a bold or italic phrase ending in a full stop, after which
the paragraph simply continues. He uses this *instead of* numbered
subsections for stretches of several pages.

> **Joining Protocol.** The process models how networks can partition into
> disjoint subgraphs which we call *fragments*…

> **Background on distributed pull voting.** Distributed voting has
> applications in various fields…

> **Features of discrete incremental voting.** The general model of pull
> voting on connected graphs regards the opinions as incommensurate…

Others seen: *Summary of results.*, *Notation.*, *Proof outline.*, *Graphs
with small second eigenvalue.*, *Discrete incremental voting: Previous work.*,
*Looking backwards: A Polya urn process.*, *Limiting process: Stick breaking.*,
*Maximum fragment size. Finite case.*, *Case k tends to infinity.*

**Applied:** every `\subsection` in the old draft that carried one idea became
a bold run-in lead. Sections V, VI, VII, VIII, XI and XII of the new file are
built this way; only Section IX, which really does have four independent
experiments, keeps numbered subsections.

## 2. Sentence case in headings, not Title Case

*"The effect of stubborn vertices"*, *"Analysis for random graphs G(n,m)"*,
*"Coupling DIV with pull voting"*, *"Proof of the main theorem, Theorem 1"*,
*"Further results and open problems"*. Never *"The Effect of Stubborn
Vertices"*.

**Applied:** every heading and the title were lowercased past the first word.

## 3. "We" is the subject; passive is reserved for the mechanism

Active first-person-plural for every authorial move: *We propose*, *We show*,
*We prove*, *We assume*, *We work with*, *We next show*, *We briefly note*,
*We remark that*, *we obtain the claim*, *We use the notation*.

Passive only where the agent genuinely is the process: *"The edges of σ are
inspected in the permutation order"*, *"a vertex v ∈ V is chosen uniformly at
random"*, *"The contacted neighbour v becomes passive"*, *"Interactions are
only allowed between adjacent nodes"*.

**Applied:** the pipeline description keeps its passives ("It is embedded; the
probe phase searches…"); everything else was moved to "we".

## 4. "provided" instead of "if" for conditions

> *provided* λk = o(1) … *provided* t ⩽ N/ω … *provided* ω is constant …
> *provided* that the required synchronization is implemented

**Applied:** eight occurrences, e.g. "Uniform allocation would be defensible
*provided* the cost–recall curve were close to linear", "graphs win *provided*
ef_L · M′ < N/Nc".

## 5. "which" where American style would use "that"

> *"the opinion of the vertex they point to"* … *"a vertex which holds fixed
> views"* … *"the value the process converges to"* … *"the first step when
> B(T) is either empty"* … *"the number of interactions which a node has had"*

Restrictive *which* is used freely; *that* is comparatively rare.

**Applied:** throughout — "the sensor **which** Android exposes", "a
decomposition whose residual term is known to be contaminated", "the
measurement **which** forced the controller".

## 6. Parenthetical glosses, heavily

> vertices (people) … an active (or opinionated) vertex … stubborn
> (intransigent, autocratic, dictatorial) … (with probability 1/n) … (i.e.,
> interactions along 'active edges' of the network) … with high probability
> (w.h.p.) … uniformly at random (u.a.r.)

Every abbreviation is expanded parenthetically on first use.

**Applied:** RAG, HNSW, IVF, ANN, MAPE, AIMD are all now expanded on first
use in exactly this form. The old draft used AIMD and MAPE unexpanded.

## 7. Connectives: Thus / Hence / Moreover / Note that / In which case

*Thus* is his workhorse. Also characteristic is sentence-initial **"In which
case"** — *"In which case using (8),"*, *"In which case the probability DIV
does not return ⌊c⌋ or ⌈c⌉ is O(n^−√log n)"*. And **"It follows that"**,
**"and the result follows"**, **"We note that"**, **"We remark that"**.

**Applied:** "and so" / "so that" / "Thus" replace most of the em-dashes.

## 8. "so that" for consequence, where the draft used an em-dash

> *"Groups can merge … so that the number of opinions is reduced."*
> *"each sensor becomes activated … so that two sensors can only communicate
> if they are close enough"*

**Applied:** the earlier drafts leaned hard on em-dashes for consequence. The
rewrite converts them to "so that", ", and so", or a full stop. `main.tex`
(the conference version, same prose lineage) contains 32 em-dashes;
`journal.tex` now contains 5, and all 5 are the "no value" placeholder in a
table cell, so the running prose has none.

## 9. Spelling: Oxford — `-our` with `-ize`

`neighbour`, `behaviour`, `colour` ("k-coloured Polya urn") alongside
`randomized`, `synchronization`, `utilize`, `stabilizing`, `initializes`.

**Applied:** the old draft was `-our` + `-ise` (normalised, quantisation,
synchronisation, optimiser). All `-ise`/`-isa` forms switched to `-ize`/`-iza`;
`neighbour`, `behaviour`, `labelled`, `modelled`, `favourable` kept.

## 10. Equations cited bare, by number

*"the recurrence (1)"*, *"from (5)"*, *"using (7)"*, *"Combining (19) and (20)
with T = …"*. Never "Eq. (1)" or "Equation (1)".

**Applied:** "the resolution floor of~(4)", "the constraint of~(6)",
"Problem~(6) is a multiple-choice knapsack".

## 11. Figure captions are *descriptive*, not argumentative

This is the biggest single change. His captions say what is plotted and
nothing else:

> **Fig. 1.** Simulation of the number of active vertices. The simulation is
> based on G(n,p). For large m = Np, G(n,p) ∼ G(n,m) so the results are
> equivalent. The blue plot is the simulation, with values of p interpolated
> at 0.02, the first entry being p = 0.02. The red curve is (3) giving b_m
> evaluated at p = m/N.

The interpretation lives in the body text.

**Applied:** every caption rewritten. For example —

*Before:* "The curve the controller chooses from, measured on the handset. The
first 38 ms of additional latency buys 0.20 recall; the last 164 ms buys 0.03.
**A curve this asymmetric is why allocation matters.**"

*After:* "The curve the controller chooses from, measured on the handset. Each
marked point is one row of Table II, plotted as p95 latency against recall@10.
The annotation compares the two ends of the curve: thirty-two clusters cost 26
times the latency of one cluster and return 1.53 times the recall."

The argument ("which is precisely the condition under which who receives the
effort matters more…") moved into Section I, where he would have put it.

## 12. Results listed as "Summary of results.", not as bold contribution bullets

> **Summary of results.** As an illustration of the process, in this paper we
> make an analysis of the edge model for random graphs G(n,m), providing the
> following results.
> – Theorem 1 gives the w.h.p. number of fragments in G(n,m) for m ≪ (n 2)…
> – Theorem 2 gives the equivalent number of fragments in the presence of
>   stubborn vertices…

Each item names the *object* (a theorem, a lemma, a table) and says what it
gives. No bold claim-labels, no "To our knowledge we are the first".

**Applied:** the six bold-labelled `\item`s became a "Summary of results."
paragraph whose items begin "Table V gives…", "Fig. 7 and Fig. 8 give…". The
novelty claim was demoted to a single sentence after the list, which is where
he puts that kind of remark.

## 13. Understatement, and a dry authorial aside about once per section

> *"It turns out to be quite a lot of work."*
> *"Lemma 3 although elegant is a limiting result."*
> *"This is similar to the previous case."*
> *"Interestingly none of the above arguments sheds any light on the final
> consensus opinion."*
> *"However, people being what they are, it seems possible that…"*

He is dry, never epigrammatic. The old draft had several closing aphorisms
("A measurement discipline you only trust when it agrees with you is not a
measurement discipline"). Those are not his register.

**Applied:** aphoristic closers were cut or flattened. Two dry asides were
kept because they are his register rather than mine: *"eleven points against
three free parameters is a number which flatters itself"* and *"a shield which
is always engaged is not a shield, it is a lower performance ceiling"*.

## 14. Closing section: open questions, phrased as questions

> *"A natural open question is how close can we get the convergence or
> stabilization time … towards the O(log n) bound"*
> *"An interesting open question is whether the Ω(log n) lower bound … can be
> shown if the monotonicity or output dominance assumptions are dropped."*
> *"Finally, it is still open whether…"*

**Applied:** the Conclusion is now "Conclusion and open questions", and the
four future-work items were rephrased from assertions ("Composition with early
termination would combine…") into the open-question form ("A natural second
question is whether composition with early termination works as cleanly as it
appears it should…").

---

## What was deliberately *not* changed

Every number, table, figure, and the content of the conclusion. The checker
confirms it:

```
$ python scripts/check_paper_numbers.py paper/journal.tex
85 checks against the recorded results
every checked number in the paper matches the file it came from
```

The title keeps its first clause ("Spend effort where it buys something"),
lowercased. A plainer Radzik-style title would be
*"Query-difficulty-conditioned retrieval control for on-device RAG"* — his
titles are flat and descriptive (*A simple model of influence*, *Discrete
incremental voting on expanders*). Changing it is a one-line edit if you want
it; it was left alone because it is content, not style.
