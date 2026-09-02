# Reward sparsity in NEAT — experiment code

A complete, dependency-light implementation of the roadmap: NEAT from scratch,
a deterministic navigation task, an η-parameterised reward, per-generation
logging, ablation, distribution-aware statistics, and figures.

Everything runs on CPU with `numpy`, `scipy`, `pandas`, `matplotlib`.

---

## Run order

Each script maps to a roadmap stage and refuses to skip ahead.

```bash
pip install -r requirements.txt

python scripts/02_sparsity_check.py                    # Stage 3 — verify η is sound
python scripts/01_baseline.py --check-reproducibility  # Stage 1 — first milestone
python scripts/00_calibrate.py --seeds 6               # Stage 5 — freeze the setup
python scripts/03_pilot.py --workers 8                 # runtime + seed count
#    -> now fill in PREREGISTRATION.md and commit it
python scripts/04_run_sweep.py --workers 8 --seeds 30  # Stage 4 — main experiment
python scripts/05_ablation.py --out results/core_sweep # Stage 6 — functional topology
python scripts/06_analyze.py --out results/core_sweep  # Stages 5/7/9 — stats + figures
python scripts/09_methods.py --out results/core_sweep  # Stage 11 — methods text
```

`06_analyze.py` writes `report.md` keyed to H1–H5 plus `descriptives.csv`,
`tests.csv`, `posthoc.csv` and ~25 figures. `09_methods.py` writes a Methodology
section generated from the configuration that was actually run, so the numbers
in the paper cannot drift from the numbers in the sweep.

Robustness (run before writing, not after):

```bash
python scripts/04_run_sweep.py --mode gated      --out results/robust_gated --workers 8
python scripts/04_run_sweep.py --mode checkpoint --out results/robust_checkpoint --workers 8
python scripts/08_compare.py --a results/core_sweep --b results/robust_gated \
    --label-a quantized --label-b gated
```

Extensions (Stage 10 — only if the baseline showed something):

```bash
python scripts/07_extensions.py --extension novelty   --lam 0.5 --etas 0.75 1.0
python scripts/07_extensions.py --extension parsimony --etas 0.0 0.5 1.0
python scripts/08_compare.py --a results/core_sweep --b results/ext_novelty
```

Tests: `python tests/test_smoke.py`

---

## The one design decision the roadmap left open

The roadmap defers the exact definition of η to implementation time. That
choice turns out to matter more than any other, so here is the reasoning.

The obvious reading of "sparse reward" is temporal: pay out less often. For an
evolutionary algorithm that reading is **degenerate**. Fitness is the sum of an
episode's rewards, so redistributing reward inside an episode leaves the sum —
and therefore the entire selection signal — untouched. Potential-based shaping
telescopes exactly; you would sweep η from 0 to 1 and measure nothing.

Sparsity has to reduce the *information content* of fitness. That is what H1
already says. So:

```
fitness(η) = B·1[goal reached] + w·C_η(trajectory)
L(η) = 2^round(K(1−η)),  K = log₂(L_max)     # 256, 128, 64, …, 2, 1
```

`C_η` is intermediate credit quantised to `L(η)` levels. Three properties make
the sweep interpretable:

| invariant | why it matters |
|---|---|
| max fitness = B + w for every η | a successful agent scores identically in every condition |
| the environment, dynamics and optimal policy never change | η cannot smuggle in task difficulty |
| the `L(η)` ladder is dyadic, so successive partitions are **nested** | increasing η can only destroy information, never rearrange it — monotone for every individual trajectory, not just on average |

`scripts/02_sparsity_check.py` verifies all of these numerically and exits
non-zero if any fails.

Two alternative operationalisations are implemented (`--mode checkpoint`,
`--mode gated`) so the headline result can be checked against the definition.
`gated` destroys information through sampling noise rather than quantisation,
which is a genuinely different mechanism — if the result holds under both, it
is about sparsity; if it flips, it is about your reward function, and the paper
has to say so.

### Reporting units

Performance is always reported with `true_objective()`, the η=0 score, never
the training fitness. Otherwise "sparse runs score lower" is true by definition
and says nothing. This single choice is what separates a finding from a
tautology, and it is enforced throughout: `final_true_best`, `success_rate` and
every ablation score use it.

---

## What gets recorded

Per generation, per run — not just the final network:

- **performance** — best/mean/median true score, success rate
- **H1 selection** — distinct fitness values, differentiation index, fitness CV,
  and the **selection differential** (mean fitness of individuals that actually
  reproduced, minus the population mean). The last is the classical realised
  measure of selection pressure and is the quantity that plugs into the causal
  chain.
- **H2 innovation** — structural mutation events, novel innovation ids, and TIR
- **H3 topology** — nodes, hidden nodes, connections, enabled connections,
  density, depth, for both population mean and champion
- **dynamics** — species count, species age, compatibility threshold, mean
  pairwise genomic distance

### TIR, concretely

The roadmap's formulation needs an operational definition tied to the genome.
`InnovationTracker` records the generation in which each innovation id first
appeared, so:

```
TIR_g = |{ids first seen in generation g−1} ∩ {ids present in generation g}|
        ────────────────────────────────────────────────────────────────────
                     |{ids first seen in generation g−1}|
```

That is the retention rate of newly created structure. It is `NaN` in
generation 0 by construction, and the analysis code is NaN-aware.

---

## Ablation (Stage 6)

Champions are ablated **under the η=0 objective**, never under their training
reward. At η=1 nearly every ablation would produce zero measured change simply
because the training signal is degenerate — that would manufacture the exact
result the study is testing for.

Two complementary measures, because single-component ablation systematically
under-counts redundancy when two components are mutually redundant:

- `functional_conn_fraction` — one-at-a-time ablation, Δ > τ (default 0.01)
- `seq_removable_fraction` — greedy sequential pruning to a minimal network
  that stays within τ of the original

Report both. They bracket the truth from opposite sides. The word "bloat"
belongs in the manuscript only if total structure grows with η **and** the
functional *count* does not.

---

## Pre-registration

The code logs ~40 quantities per generation, which is right for a longitudinal
study but also means an unconstrained analysis has enough freedom to find
something in almost any dataset. `PREREGISTRATION.md` is a short template for
declaring one primary metric per hypothesis, the multiplicity correction, and
the stopping rule before the sweep runs. Fill it in after the pilot, commit it,
and treat everything else in `tests.csv` as exploratory.

The template's suggested H1 primary is `mean_selection_differential` rather than
`mean_distinct_fitness`. Distinct-value counts are partly a property of the
quantiser — at η=1 there is exactly one value by construction, which is the
manipulation working rather than a finding. The selection differential measures
how much better the individuals that actually reproduced were, which is the
quantity that enters the causal chain.

---

## Statistics

`06_analyze.py` writes `descriptives.csv` before anything else — inspect it
first, as the roadmap asks. Then per metric: Kruskal–Wallis (ε² effect size),
Dunn post-hoc with tie correction and Holm + BH adjustment, Cliff's δ for every
pair, bootstrap CIs on condition medians, Spearman ρ and Jonckheere–Terpstra
for monotone trend.

Threshold analysis compares a continuous two-segment fit against a linear fit
by AIC, with a bootstrap CI on the breakpoint. The report says "breakpoint at
η = x, ΔAIC = y" and explicitly declines to say "phase transition". Raise that
language only if the CI is narrow and the break replicates under the other
sparsity modes.

---

## Calibration is not optional

`00_calibrate.py` checks four preconditions and exits non-zero if any fails:

1. the dense endpoint solves the task in 30–95% of runs (not at the floor)
2. it is not solved by generation 2 (not at the ceiling)
3. topology actually changes over a dense run (H2/H3 have range)
4. sparsity measurably reduces fitness differentiation (η does something)

All four are about *range*, never about the sparsity–topology relationship, so
running it cannot bias the result. Check 3 is the one people skip: if the
dense baseline never complexifies, a null result on H3 is an artefact of the
setup rather than a finding about neuroevolution, and the honest move is to
lengthen the runs or pick a task that rewards nonlinearity — before collecting
data, not after.

---

## Cost

Roughly 1 s per 1000 episodes per core. One run at the shipped defaults
(120 generations × population 120 × 300 steps) is ~2 minutes; the full sweep
(9 conditions × 30 seeds) is ~9 core-hours, so about 70 minutes on 8 cores.
`03_pilot.py` measures this on your machine and converts observed variance into
a required seed count instead of assuming 30.

---

## Layout

```
neat_sparsity/
  config.py       configuration + hash (runs with different hashes never pool)
  genome.py       genome, innovation tracking, mutation, crossover
  network.py      phenotype compilation (topological sort, cached)
  population.py   speciation, reproduction, instrumented generation loop
  env.py          deterministic 2D navigation task
  reward.py       η-parameterised reward + the η-independent true objective
  runner.py       run/sweep orchestration, logging, serialisation
  metrics.py      every metric definition, fixed before results are seen
  ablation.py     functional vs non-functional topology
  stats.py        Kruskal–Wallis, Dunn, Cliff's δ, bootstrap, threshold fit
  plots.py        figures, all with run-level uncertainty
  analysis.py     loading, reshaping, and the metric registry
  extensions.py   novelty pressure, adaptive parsimony
scripts/          00–09, one per stage
tests/            smoke tests including a determinism check
PREREGISTRATION.md  primary metrics and decision rules, filled in before the sweep
```

Reward sparsity is read in exactly one place, `reward.py`. It cannot reach the
mutation operators, the speciation code, or the environment, so it can only act
through selection. That is the experimental control, enforced structurally
rather than by convention.

## Reproducibility

A run is fully determined by `(config, seed)`. `01_baseline.py
--check-reproducibility` runs the same seed twice and asserts the two
generation-by-generation logs are identical; if that ever fails, some
unordered iteration has crept in and the results are not trustworthy.

Every output file carries the config hash, and `06_analyze.py` refuses to pool
runs whose hashes differ.
