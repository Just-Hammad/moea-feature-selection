# Multi-objective evolutionary feature selection on high-dimensional biological data

A budget-matched, nested-cross-validated comparison of NSGA-II feature selection against
random subsets, standard filter and embedded baselines, a model-X knockoff filter, and an
oracle with access to the true informative features.

![tests](https://img.shields.io/badge/tests-47_passing-brightgreen)
![python](https://img.shields.io/badge/python-3.12-blue)
![license](https://img.shields.io/badge/license-MIT-lightgrey)
<!-- Zenodo DOI badge goes here once raw results are archived. -->

> **The question.** Multi-objective evolutionary algorithms are widely applied to feature
> selection on omics data. Two things are rarely reported alongside those results: whether the
> evolutionary search beats *random subsets given the same evaluation budget*, and whether the
> reported accuracy survives a protocol in which the test fold never participates in selection.
> This study reports both.

**Findings are in [`report/RESULTS.md`](report/RESULTS.md)**, regenerated from raw results by
`make all`. A summary is in [Findings](#findings) below.

---

## Reproduce

```bash
make install && make test
```

```bash
make reproduce
```

`make reproduce` runs the study, runs the permuted-label control, and rebuilds every table and
figure. `make all` rebuilds the report from existing raw results **without re-running anything**,
so the report cannot silently disagree with the data it describes. `make quick` exercises the
entire pipeline in under a minute.

---

## Findings

Two datasets, 50 paired blocks each (10 seeds x 5 outer folds), 900 matched evaluations.
Full tables, effect sizes and figures in [`report/RESULTS.md`](report/RESULTS.md).

### 1 · The evolutionary search wins the objective it optimises and not the one that matters

NSGA-II beats matched-budget random subsets **decisively on the inner-CV Pareto front**, and
**not significantly on held-out error** — on both datasets.

| Dataset | Hypervolume (inner) | Outer-fold balanced error |
|---|---|---|
| synthetic | δ = **+0.999** (large), Holm *p* < 0.0001 | δ = +0.065 (**negligible**), Holm *p* = 1.00 |
| colon | δ = **+0.936** (large), Holm *p* < 0.0001 | δ = −0.192 (small), Holm *p* = 0.23 |

*Cliff's δ, NSGA-II vs random subsets, on raw per-block values, 50 paired blocks. For hypervolume
a positive δ favours NSGA-II; for error a negative δ favours NSGA-II. Friedman was significant for
every metric on both datasets (p < 1e-36), so the post-hoc is licensed.*

888 unique evaluations of multi-objective search buy a Pareto front that dominates random
sampling almost everywhere, and buy no statistically detectable generalisation advantage over
drawing the same number of subsets at random. **This gap is only visible because selection and
evaluation are separated** — reporting the inner-CV error, as is common, would have shown a large
and entirely illusory win.

![inner vs outer](report/fig2_distributions.png)

*Top row: the inner objective the search optimises — NSGA-II and NSGA-III clearly beat random
subsets. Bottom row: held-out error, same selector ordering — the advantage does not survive.
The vertical movement between rows is the gap between optimising and generalising.*

### 2 · Search cost bought nothing that a univariate filter did not buy for 1% of it

On the synthetic problem, mean Friedman ranks on held-out error (1 = best, 9 selectors):

```
Oracle 1.00  <  mRMR 3.84  <  LASSO 3.95  <  t-test+BH 4.49  <  Random 5.71  <  NSGA-III 5.99  <  NSGA-II 6.17
```

`t-test + BH` used **8 evaluations**; NSGA-II used **888** and ranked below both it and random
sampling. On colon the ordering is the same at the top (`t-test+BH` 2.78, `LASSO` 2.89, `mRMR` 2.98,
`NSGA-II` 4.13).

### 3 · NSGA-II recovered none of the planted features; a t-test recovered 7 of 10

With the ground-truth support known, on the synthetic dataset:

| Selector | Precision | Recall | Features | Evaluations |
|---|---|---|---|---|
| Oracle | 1.000 | 1.00 | 10 | — |
| LASSO path | 0.136 | **0.70** | 54.5 | 7 |
| t-test + BH | 0.060 | **0.70** | 100 | 8 |
| mRMR | 0.140 | 0.55 | 50 | 7 |
| NSGA-III | 0.061 | 0.10 | 9 | 891 |
| **NSGA-II** | **0.000** | **0.00** | 14.5 | 888 |
| Random subsets | 0.000 | 0.00 | 22 | 900 |

NSGA-II found feature sets that fit the inner folds well while containing **none** of the ten
features that actually generate the labels. In a 2^2000 space, 900 evaluations is far too sparse to
locate a 10-feature signal — so the search optimises the only thing it can reach, which is inner-CV
noise. Methods using marginal statistics rather than search recover most of the signal for a
thousandth of the compute.

### 4 · The oracle gap: every method leaves most of the recoverable signal on the table

An oracle given the true support reaches **0.044** balanced error. The best real selector reaches
**0.374** — an 8.5× gap. The signal is fully recoverable in principle; no method here gets near it.
Without the oracle this table would read as "the dataset is too hard", which would be wrong.

### 5 · Knockoff+ cannot make a discovery at q = 0.1, by construction

The knockoff+ threshold requires `(1 + #{W ≤ −t}) / #{W ≥ t} ≤ q`, so at least **⌈1/q⌉ = 10**
features must clear the threshold before *any* selection is possible. At n = 120 it selected
nothing on every fold. Plain knockoff (offset = 0) selected 2 features on the synthetic data,
**both true positives** — realised FDR 0, recall 0.20.

The guarantee is real and it is being honoured. It is simply unreachable at these sample sizes,
which is worth reporting rather than hiding behind a q chosen after the fact.

### 6 · The evolved feature sets are mutually disjoint across folds

Median Jaccard overlap between the supports chosen on different outer folds of the same seed
(100 pairs per selector per dataset):

| Selector | synthetic | colon |
|---|---|---|
| Oracle | 1.000 | — |
| t-test + BH | 0.255 | 0.277 |
| mRMR | 0.222 | 0.250 |
| LASSO path | 0.181 | 0.187 |
| **NSGA-II** | **0.000** | **0.000** |
| **NSGA-III** | **0.000** | **0.000** |
| Random subsets | 0.000 | 0.000 |

Re-run on a 4/5 resample of the same cohort, NSGA-II returns a feature set sharing *no members*
with the previous one — indistinguishable from random sampling on this measure. A selector this
unstable may still have found a predictive subspace, but it has not identified biomarkers, and
those are materially different claims. This number is rarely published; it should be.

### 7 · The control that licenses all of the above

With labels shuffled before splitting, every selector returns to chance:

| | permuted labels | real labels |
|---|---|---|
| t-test + BH | 0.458 | 0.255 |
| mRMR | 0.462 | 0.255 |
| LASSO path | 0.480 | 0.250 |
| NSGA-III | 0.489 | 0.361 |
| Random subsets | 0.498 | 0.375 |
| NSGA-II | 0.500 | 0.362 |

![permutation control](report/fig3_permutation.png)

**Verdict: PASS.** Lowest permuted median 0.458 against a chance level of 0.5. Nothing in this
pipeline is reading the held-out fold. Paired with the signal-detection test in
`tests/test_nested_cv.py`, this is what licenses reading the numbers above as generalisation rather
than as memorisation.

---

## Method

### Problem

Two objectives, both minimised, both normalised to [0, 1]:

```
f1 = min(1, |S| / cap)      size of the selected feature set
f2 = inner-CV error         misclassification rate over the inner folds
```

### Protocol

| | |
|---|---|
| **Outer** | 5-fold stratified. **The only honest error estimate.** Never touched by selection |
| **Inner** | 3-fold stratified, used for the `f2` objective and for choosing one solution from a front |
| **Replication** | 10 seeds, giving 50 paired blocks per dataset |
| **Budget** | **900 unique evaluations** (30 × 30), identical for NSGA-II, NSGA-III and random subsets |
| **Scaling** | `StandardScaler` fitted on the outer training rows only |

Everything label-dependent — scaling, selection, model fitting — happens strictly inside the
outer training fold. The single solution carried to the outer fold is chosen from the inner
front by inner-CV error alone; letting the held-out fold pick the front point is precisely the
bias the design exists to exclude.

### Methods compared

| | Method | Kind |
|---|---|---|
| | **NSGA-II**, **NSGA-III** | multi-objective evolutionary, sparse log-uniform initialisation |
| | **Random subsets** | the ablation — same budget, same cardinality distribution, no search |
| | **t-test + BH** | univariate filter with Benjamini–Hochberg FDR |
| | **mRMR** | greedy max-relevance / min-redundancy |
| | **LASSO path** | L1 logistic regression over a regularisation grid |
| | **Knockoff / Knockoff+** | model-X Gaussian knockoffs, both offsets |
| | **Oracle** | uses the true support — a ceiling, not a competitor (synthetic only) |

### Statistical analysis

Friedman across selectors with `(dataset, seed, outer fold)` as blocks — within a block every
selector saw identical training rows and an identical held-out fold, so the comparison is
properly paired. Post-hoc pairwise Wilcoxon signed-rank with Holm correction, computed **only**
where Friedman was significant. Cliff's δ on raw per-block values, with Romano et al. magnitude
labels. Distributions reported as median [IQR] throughout — never mean ± sd, which assumes a
normality these bounded, skewed quantities do not have.

---

## Design decisions that change the answer

Four choices here are easy to get wrong in ways that quietly invalidate the result. Each is
implemented deliberately and tested.

**1 · The permuted-label control.** `make permute` shuffles `y` before splitting, destroying all
signal while preserving class balance and the feature correlation structure. Every selector must
then land at chance. Paired with a signal-detection test, this is the only cheap way to prove the
absence of a leak: a pipeline that is merely broken passes the permutation test trivially, and a
pipeline that leaks passes the signal test spectacularly. Both tests ship in `tests/test_nested_cv.py`.

**2 · The oracle ceiling.** Without a selector that knows the true support, a table in which every
method sits near chance is ambiguous between *these methods are weak* and *this dataset carries no
learnable signal*. The oracle separates those readings.

**3 · Hypervolume normalisation.** The size axis is scaled by the cardinality cap, **not by p**.
Dividing by p looks more natural but at p = 2000 with a 200-feature cap every attainable solution
has `f1 ≤ 0.1`; the size axis collapses and hypervolume silently degenerates into an error-only
indicator that cannot distinguish 2 features from 200. The reference point is the fixed,
problem-derived `(1.0, 1.0)` — a module constant, never an argument tuned per experiment.

**4 · Budget parity is measured, not asserted.** NSGA-II revisits solutions and the objective cache
absorbs the duplicates, so its *unique* evaluation count can fall below `pop × gens`. Table 3 of the
report records what each method actually spent, so a reader can check the ablation rather than trust it.

---

## Limitations

Stated because they bound what the results support.

- **Two datasets.** One synthetic, one real (colon, n = 62). Conclusions are about these problems, not about omics data in general.
- **One evaluation budget.** 900 evaluations, stated and matched. This study cannot say whether the evolutionary methods would separate from random given substantially more budget — a budget sweep is the obvious next experiment.
- **One inner classifier.** L2 logistic regression, fixed and untuned by design, to keep the study about selection rather than classification. A different inner model could reorder the methods.
- **Knockoff screening breaks the strict FDR guarantee.** Features are screened to 500 on the same data used for selection, so the honest name for the procedure is *screened knockoffs*. This is recorded in the per-run metadata and reported.
- **Second-order Gaussian knockoffs are an approximation.** The construction assumes a known feature distribution. On p ≫ n omics data it is estimated with shrinkage, and the two diagnostics recorded per run (`s`, `shrinkage`) exist precisely because the approximation can fail in two opposite directions — see `selectors/knockoff.py`.
- **Hypervolumes are not comparable across studies** using a different size normalisation or reference point. Both are stated in every report.

---

## Layout

```
src/mofs/
├── objective.py          shared bi-objective evaluator -- every selector scored identically
├── problem.py            pymoo wrapper + cardinality repair
├── nested_cv.py          the outer/inner split; the correctness gate
├── data.py               datasets, each carrying a provenance record and a content hash
├── selectors/
│   ├── evolutionary.py   NSGA-II, NSGA-III, seeded sparse initialisation
│   ├── baselines.py      random subsets, t-test+BH, mRMR, LASSO path, oracle
│   └── knockoff.py       model-X Gaussian knockoffs, both offsets, with diagnostics
├── stats/
│   ├── compare.py        Friedman -> Holm-corrected Wilcoxon -> Cliff's delta
│   └── multiobj.py       hypervolume, IGD, attainment surfaces (moocore)
├── runner.py             experiment orchestration; writes raw results + a manifest
└── report/build.py       regenerates every table and figure from raw results
```

Each run writes a **manifest** recording the full config, the resolved seeds, dataset content
hashes, the indicator backend, library versions and wall-clock — so a results file fully
determines the conditions that produced it.

`stats/compare.py` is API-compatible with the shared statistics contract used across this group of
projects, so it can be swapped for the companion `metaheuristics-bench` implementation with a
one-line import change.

---

## Citation

See [`CITATION.cff`](CITATION.cff). Raw results are archived to Zenodo; the DOI badge above links
to the archived record.
