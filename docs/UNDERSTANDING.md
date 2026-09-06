# Understanding the output

Every number this project prints, where it came from, and what it means.

This is written to be read alongside a terminal session. If you run the
commands in `docs/RUNBOOK.md` and something looks strange, the explanation is
here.

---

## 1. `data status`

```
 dataset   kind   dl   ver   proc   gt   gtd      n_base   n_queries   dim   space    search     Nc        disk
 nytimes   ann    OK   OK     OK    OK   OK      289,761       9,991   256   cosine   ip        538    604.8 MB
 sift1m    ann    OK   OK     OK    OK   OK    1,000,000      10,000   128   l2       l2       2000   1005.4 MB
 squad     qa     OK   OK     OK    OK   OK       46,369       1,000   384   cosine   ip        215    217.4 MB

MANIFEST.json  17 artifacts, 1.8 GB recorded
  every artifact on disk has a manifest entry
```

### The columns

| Column | Meaning |
|---|---|
| `kind` | `ann` = a vector benchmark with published ground truth. `qa` = question answering, has **text**, so it is the only set that can measure answer quality. |
| `dl` | Downloaded from the source URL in `configs/datasets.yaml`. |
| `ver` | **Verified** — SHA-256 re-checked against the registry. This is what makes "the data is intact" a check rather than a hope. |
| `proc` | Processed into the project's own format: `base.npy`, `queries.npy`, and for SQuAD a `chunks.sqlite`. |
| `gt` | Ground truth **IDs** present — which vectors are truly the nearest neighbours. |
| `gtd` | Ground truth **distances** present. Stored separately and in float64, because recall here is tie-aware and needs the distances, not just the IDs. |
| `n_base` | Vectors in the corpus. |
| `n_queries` | Held-out query vectors. |
| `dim` | Embedding dimension. |
| `space` | The dataset's **semantic** metric — what "similar" means for this data. |
| `search` | The metric the **index** actually uses. Differs from `space` for cosine sets: vectors are unit-normalised at preparation, after which cosine similarity and inner product are the same ranking, so the index uses `ip`. |
| `Nc` | Number of k-means clusters the index partitions into. |
| `disk` | Bytes on disk for that dataset. |

### Where each dataset came from

| Dataset | Source | Why it is here |
|---|---|---|
| **SIFT1M** | `ann-benchmarks.com/sift-128-euclidean.hdf5` | The standard million-scale ANN benchmark. Ships its own exact ground truth, so recall numbers are comparable with published work. |
| **NYTimes** | `ann-benchmarks.com/nytimes-256-angular.hdf5` | A second benchmark with a different metric and a different distribution. It is where the cluster-miss property lives that makes one acceptance check legitimately fail. |
| **SQuAD** | Stanford SQuAD v1.1 | Real questions with real gold answers. The only dataset with text, so the only one that can measure EM/F1 and content reduction. |

Both `ann` sets come from **ann-benchmarks**, a public suite whose HDF5 files
contain a `neighbors` array — brute-force ground truth computed by someone
else. That is an *external reference*, and it is how the project caught a bug
where its own distance computation had dropped a term.

### The last two lines

```
MANIFEST.json  17 artifacts, 1.8 GB recorded
  every artifact on disk has a manifest entry
```

`MANIFEST.json` records, for every file: its SHA-256, its size, its source URL,
and when it was prepared. The second line is a **two-way** check —
no recorded artifact is missing from disk, and no file on disk lacks a record.

The datasets live at `C:\smartann-data`, **outside the repository**. Gigabytes
of downloadable data do not belong in version control. What is committed is the
pipeline that fetches and verifies them.

> **The exit code is the check, not the table.** `data status` exits non-zero if
> anything is incomplete. A human reading a table is not a verification.

---

## 2. `pytest tests/ -q`

```
.............................................................. [100%]
```

447 tests. Each dot is one test. **The thing to look for is the absence of
`F`** — any `F` is a failure and the letter is what matters, not the count.

The tests are not only about correctness of results. A large share encode
*rules*: that a manifest cannot claim a measurement the platform cannot make,
that an entry point must produce output, that an energy fit must refuse an
intercept it did not measure.

---

## 3. `verify_all.py`

This runs the commands `docs/review.md` tells a reader to run, in order, and
records what each did. It exists because every phase was verified when it was
built, and the thing that gets lost is whether the pieces still work
*together*, from the documented entry points.

### Reading the summary

```
  OK    test suite             exit   0     26.0s
  OK    lint + types           exit   0      0.2s
  OK    mypy                   exit   0      3.4s
  OK    data status            exit   0      2.8s
  OK    phase 2 acceptance     exit   1    800.0s
  OK    cost model fit         exit   0      1.5s
  OK    figures                exit   0      4.8s

7/7 steps reproduced.
```

### Why `phase 2 acceptance` shows exit 1 and is still OK

One of the fifteen index checks genuinely fails: NYTimes check 4, which is a
property of that dataset rather than a defect in the code. Exit 1 is the
expected outcome and the verification lists `(0, 1)` as acceptable **for that
step only**.

An earlier version used a boolean "ignore the exit code" flag instead, and that
tolerated *everything* — it once reported OK for a step that had timed out.
Naming the acceptable codes explicitly is the difference between an escape
hatch that says *what is expected* and one that says *ignore the result*.

### The exit codes, and why there are four

| Code | Meaning | Counts as reproduced? |
|---|---|---|
| `0` | Everything passed | yes |
| `1` | A check failed — a **result** | yes, for the acceptance step only |
| `2` | Another workload ran during the measurement; every duration is meaningless | **no** |
| `3` | The machine-quiet guard **refused to start** | **no** |
| `-1` | Timed out | **no** |

Codes 2 and 3 exist because both were once reported as successes.

> **A step that produces no output at all is failed regardless of its exit
> code.** For thirty-nine commits `python -m smartann.cli data status` exited 0
> while doing nothing — the module had no `__main__` guard, so it imported,
> defined every command, ran none, and returned success. The verification
> checked data status *through that broken entry point* and scored the silent
> success as a pass. An exit code is a claim; output is evidence.

### If you see this

```
RuntimeError: machine is 26% busy (median of 4 samples, threshold 25%).
    -> exit 3 in 1.5s [FAILED]
```

The acceptance step **did not run**. It refuses to take timing measurements on
a busy machine, because background load has corrupted measurements in this
project four separate times — once by another experiment, once by this
project's own figure generation perturbing the latencies it was plotting.

**What to do:** close whatever is running (games and their launchers are the
usual culprit; a syncing OneDrive folder is another) and run it again. The step
normally takes about 800 seconds. If it took 1.5, it declined.

---

## 4. `fit_cost_model.py`

```
  t_cluster_load     MAPE    28.73%   mean actual    4.891 ms  predicted    2.080 ms
  t_cluster_search   MAPE    31.17%   mean actual    0.968 ms  predicted    0.475 ms
  t_merge            MAPE    71.86%   mean actual    0.044 ms  predicted    0.015 ms
[PASS] total latency MAPE on quiet (fitted): 14.75% (target < 15%)
[FAIL] total latency MAPE on contended (held out): 35.87% (target < 15%)
```

**MAPE** is Mean Absolute Percentage Error — on average, how far the prediction
sits from the measurement.

The two lines that matter are the last two, and **both are reported**. The
model was fitted on a quiet machine and hits its 15% target there. On contended
data it was never fitted on, it fails at 35.87%.

That failure has a cause, and the cause is the interesting part: `cpu_util` is
a **censored feature**. It saturates at 1.0, so it cannot express how *much*
contention there is — 12,469 samples all read exactly 1.000 while the latency
they were meant to explain spanned 41×. The model has no variable left to move.

This is why the budget-aware policy reads a **table of measurements** rather
than predicting from this model: a policy predicting from it would be most
confident exactly where it is most wrong.

### The energy block

```
E = 3.85 V * [2300.0 uA * t_cpu + 800.0 uA * t_io]  [NOT FITTED (reference values)]
  source: MobileRAG (arXiv:2507.01079) §3.4.3, reported for their device.
[BLOCKED] cannot fit an energy model: this platform reports energy_mj_measured as unavailable.
```

Those coefficients are **the baseline paper's numbers for the baseline paper's
device**. They are not measured here and not fitted here.

The fitter *refuses* to fit, because the development laptop cannot measure
energy — no RAPL on Windows, no populated `powercap` under WSL2. Anything the
model outputs is recorded as `energy_mj_model`, never as a measurement.

The real energy numbers come from the phone, in section 6.

---

## 5. `fit_thermal_model.py`

```
trace: 19,144 queries, 25.0 min
skin temperature: 32.90 -> 37.51 C (rise 4.61 C)

 window s     n       R2     a (work)    tau s
        3   446   0.1088    6.082e-04    107.3
       15    89   0.2295    9.427e-04     67.1
       30    44   0.3035    9.143e-04     67.9  <-- chosen
       60    22   0.1868    5.935e-04     97.1  (too few intervals -- not quoted)
      120    11   0.6908    8.694e-04     76.3  (too few intervals -- not quoted)
```

### Where this data came from

A 25-minute run **on the phone** (vivo I2304), executing the real index against
the real query stream while sampling thermal sensors every 3 seconds.

### The model

```
dT/dt = a·work_rate + b·(T − T_ref) + c
```

A first-order lumped thermal mass: temperature rises with work done and falls
toward ambient. `τ = −1/b` is the **time constant** — how long a change in
retrieval effort takes to show up in temperature.

### Why a whole sweep instead of one number

The differencing window is an analysis choice, and it moves R² **sixfold**.
Reporting one row would hide that the answer depends on a decision no reader
would think to question.

**The 120 s row scores highest (R² 0.69) and is excluded.** Eleven data points
against three free parameters is a number that flatters itself.

**What actually supports the result is stability, not R².** τ stays between 67
and 107 seconds and `a` stays between 5.9 and 9.4e-04 across *every* window,
while R² swings from 0.11 to 0.69. A point estimate that barely moves across
analysis choices is stronger evidence than any single goodness-of-fit.

Short windows fail for a measurable reason: skin rose about 0.003 °C/s, while
differencing adjacent 3-second samples gives a dT/dt whose standard deviation
is 0.051 °C/s — **sixteen times the signal**. The sensor quantises at ~0.005 °C,
so short differencing measures quantisation, not physics.

### Why τ = 68 s is the most useful number in the project

It is the thermal **observation delay**. An earlier control law collapsed
because its control period was shorter than its observation delay — it acted
again before its previous action could appear in what it was reading, so every
action looked ineffective and it decreased to the floor.

τ = 68 s says any thermally-aware controller sampling faster than about a
minute will reproduce that collapse. There is now a measured number saying so
instead of an argument.

### The latency table

```
 nprobe       n    p50 ms    p95 ms  clusters
      1   3,829      5.59     13.82      1.00
      4   3,829     20.50     52.26      4.00
      8   3,829     40.46    101.46      8.00
     16   3,829     82.61    189.82     16.00
     32   3,828    169.21    354.10     32.00
```

`nprobe` is how many clusters the search opens. Clean linear scaling, and
roughly **13× slower than the laptop** at matched settings — which is the gap
that makes adaptive effort worth having on a phone at all.

---

## 6. `fit_energy_model.py`

```
trace: 9,670 queries, 218 samples, 21.0 min
  status: {'DISCHARGING': 218}, every current negative

MEASURED (no model needed):
  total energy      1266.6 J
  mean power        1.006 W
  all-in per query  130.99 mJ

EFFORT vs POWER (binned, no intercept required):
  work    4.7- 113.8 clusters/s -> 0.729 W  (n=32)
  work  113.8- 124.7 clusters/s -> 0.742 W  (n=31)
  work  124.7- 142.2 clusters/s -> 0.942 W  (n=31)
  work  142.2- 161.8 clusters/s -> 1.334 W  (n=31)
  work  161.8- 202.2 clusters/s -> 1.441 W  (n=32)
  correlation +0.4010
```

### Where this came from

A 21-minute run on the phone, **unplugged**, alternating 60 seconds of querying
with 30 seconds of idle. Current and voltage sampled from the battery every 5
seconds via Termux:API.

### Why "DISCHARGING, every current negative" is printed first

It is a precondition, checked before anything is integrated.

While the phone is **charging**, the current's sign alternates between samples
under constant load — the charger modulating. An earlier 20-minute run
completed entirely on charge and its current had a **median of 0 mA** with a
103/102 split between positive and negative. Integrating over a sign that is a
coin flip yields approximately zero. That run was not a weak measurement; it
was not a measurement.

Both the trace and the fit now refuse to proceed unless the device is
discharging.

### Why the effort/power relationship is **binned** rather than fitted

Binning needs no intercept. It says only what was observed, at the work rates
that were actually visited.

### The attribution, and why it once could not be reported

The natural next step is a regression giving marginal energy per cluster. On an
earlier trace it returned an idle power of **−0.209 W** — a phone consuming
negative power.

The cause was the experiment, not the estimator: work rate never approached
zero, so the intercept extrapolated 95 units past the nearest observation.
The number was computed, marked `reportable: false`, and printed with its
reasoning rather than published.

**The fix was to change the experiment**, adding idle periods so the baseline
is measured rather than extrapolated. With the duty-cycled trace:

| | value | 95% CI (10,000 bootstrap resamples) |
|---|---|---|
| marginal cost | **3.206 mJ per cluster** | [2.145, 4.201] — excludes zero |
| fitted baseline | 0.7091 W | [0.5957, 0.8385] — positive |
| **directly measured idle** | **0.6571 W** | median of 60 observed idle intervals |

The fitted intercept and the directly measured idle power agree to within
**7.9%** — from the same data, by different routes. That agreement is the check
the earlier trace could not perform at all.

> **If you see `PER-QUERY ATTRIBUTION: NOT REPORTABLE` with `intercept -0.209 W`,
> you are looking at the superseded trace.** The default is `--prefix c4_`. The
> old continuous trace is `--prefix e2_` and is kept only for comparison.

### The finding this bounds

| | energy | share of a query |
|---|---|---|
| all-in per query | 130.99 mJ | 100% |
| retrieval @ nprobe=1 | 3.21 mJ | **2.4%** |
| retrieval @ nprobe=8 | 25.65 mJ | **19.5%** |
| retrieval @ nprobe=32 | 102.60 mJ | 78.2% |

**The screen and radios are about 65% of mean power.** Retrieval is a
*minority* of what a query costs at any sane operating point.

That is the honest reading: an energy-budget controller adjusting retrieval
effort moves roughly a fifth of the total, not all of it. The dial is real and
its range is now known — which is worth more than a larger number that ignored
the baseline.

**Still weak, and stated as such:** R² = 0.165. The slope's interval spans a
factor of two. What supports the estimate is not its R² but that it excludes
zero, that its intercept is physical, and that the intercept is corroborated by
direct measurement.

---

## 7. The C5 result

```
static   1.83  0.9054
adaptive 1.77  0.8762
```

Violation percentage and recall, for a device-state-aware policy against a
tuned static baseline, both selecting against the **same** 110 ms budget,
interleaved query by query on a phone heating itself from 31.5 to 36.5 °C.

**This is a negative result.**

| | violation difference | recall difference |
|---|---|---|
| adaptive − static | −0.066 pp, 95% CI **[−0.379, +0.257]** | −0.0292, 95% CI **[−0.0334, −0.0249]** |
| verdict | **includes zero** — no detectable benefit | **excludes zero** — the cost is real |

The policy paid recall and bought nothing.

### It is not a broken policy

On the 901 queries (6.4%) where it actually derated:

| on derated moments | p95 | violations |
|---|---|---|
| adaptive | **43.15 ms** | **0.0%** |
| static | 77.56 ms | 0.8% |

It halved p95 and eliminated violations *at those moments*. **The mechanism
works. It chose the wrong moments** — static's violation rate at those same
moments was 0.8%, *below* its 1.8% overall.

### Why

Every reason the policy ever cited was `temperature rising`, at +0.05 to
+0.13 °C/s. The absolute tiers — 42 °C warm, 46 °C hot — **never fired once**,
because skin peaked at 36.5 °C.

The phone warms measurably but never approaches throttling. So temperature
rises and latency does not respond to it. The thermal signal is real,
measurable, and **not predictive of the thing the controller exists to
prevent**.

### Was the test just too small?

No. The confidence interval's half-width is 0.318 percentage points, so any
benefit larger than about 0.3 pp *would* have been detected. The experiment had
the power to find a real effect and did not.

### What can be claimed

Not "device-state control does not work". Something narrower and more useful:
*on a workload that never drives the device into thermal throttling,
device-state control has nothing to fix.* A fair test needs a workload that
reaches throttling — sustained generation, a hotter ambient, or an external
thermal load. That is a different experiment, not a longer one.

---

## 8. The figures

| File | What it shows |
|---|---|
| `fig_device.png` | Power against retrieval work (left) and device latency with the fitted τ (right). **Start here** — it is the only figure measured on the target hardware. |
| `fig1_headline.png` | The controller under contention, and the quiet-phase collapse that diagnosed the control law. |
| `fig_regime.png` | The baseline paper's per-cluster graph condition against what was measured. |
| `fig_scr_quality.png` | Answer quality against context reduction, with paired bootstrap intervals. |
| `fig_joint.png` | The joint quality/cost surface; ringed points are on the Pareto front. |
| `fig_footprint.png` | Resident memory against the baseline paper's reported figures. |
| `fig_recall_effort.png` | Recall against retrieval effort. |

---

## 9. Two things that look wrong and are not

**`phase 2 acceptance` exits 1.** One NYTimes check genuinely fails as a
dataset property. A verification demanding exit 0 there could only be satisfied
by hiding a real result.

**The cost model reports a FAIL and is kept.** 35.87% MAPE on contended data is
a real limitation with an identified cause, and it is the reason the budget
policy reads measurements instead of predictions. Deleting the failing number
would remove the evidence for a design decision.

## 10. One thing that looks fine and is not

**A command that prints nothing.** For thirty-nine commits
`python -m smartann.cli <anything>` imported the module, ran nothing, and
exited 0. Silence is the failure mode to suspect, and the verification now
fails any step that produces no output at all.
