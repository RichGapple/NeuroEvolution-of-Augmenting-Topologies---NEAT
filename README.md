# When does reward resolution matter?

**How much reward information does a learner actually need — and what decides
that?** This repository contains a controlled study comparing NEAT
(neuroevolution) against DQN (deep reinforcement learning) as the reward signal
is progressively stripped of information, on identical tasks under identical
interaction budgets.

2,760 runs. Pre-registered. Every configuration hashed so incomparable runs can
never be pooled.

---

## The finding

> **Reward resolution matters only on tasks that undirected search cannot solve
> on its own.**

The mediating property is measured, not assumed. Sample a few thousand randomly
drifting policies and count how many reach the goal:

| | policies reaching the goal | does reward resolution matter? |
|---|---|---|
| **Maze A** | 0 / 4,000 | **yes** — resolution becomes decisive |
| **Maze B** | 3 / 4,000 | **barely** — both methods solve it at every resolution |

A NEAT run evaluates 10,000 genomes, so on maze B roughly 8 solve the task *by
chance*, and elitism preserves the first one permanently. The reward never has
to do anything. On maze A luck never delivers a solution, so the reward has to
actually guide the search — and that is when how finely it is graded starts to
bind.

### Where each method breaks

On maze A, normalising each method against its own dense-reward ceiling:

| | holds performance down to | levels | bits | breaks between |
|---|---|---|---|---|
| **NEAT** | η = 0.625 | **8** | 3 | 0.625 → 0.75 |
| **DQN** | η = 0.25 | **64** | 6 | 0.25 → 0.375 |

**DQN requires 8× finer reward resolution than NEAT — three additional bits.**
NEAT tolerates losing 97% of the distinguishable feedback levels (256 → 8)
before its performance moves at all.

### Why

Two mechanisms, both measured rather than argued.

**Rank invariance.** NEAT's parent selection is truncation — sort by fitness,
keep the top fraction. That reads only the *ordering*, never the magnitudes, so
quantising the reward is harmless until it produces enough ties that the cut
falls inside a tied group. Instrumentation confirms it: the share of selection
decisions resolved by genome key rather than fitness rises monotonically with η
in every one of seven configurations, from 0.03 at η=0 to 0.67 at η=0.875.

And the threshold is **not a property of the algorithm**. Changing the fraction
of each species allowed to reproduce from 0.3 to 0.5 moves it from 8 levels to
64 — the same factor that separates NEAT from DQN. Halving the population moves
it from 8 to 256.

**Retention asymmetry.** At η=1, where the reward carries zero information about
anything except reaching the goal:

- DQN found the good region during exploration (`best_ever` = 0.698) and then
  **lost it** (`final` = 0.000)
- NEAT, with the same zero information, solved maze B in **30/30 runs**

Evolution keeps a lucky discovery for free, via elitism. Value-based RL cannot,
because retention runs through the same value function that needs the reward in
order to train.

---

## The two mazes

Identical except for one wall.

![The two mazes](figures/fig_mazes.png)

Green is the true shortest route; coral dashed is the route taken by a policy
that always moves closer to the goal — which is what the reward gradient
encodes.

**Maze A is deceptive.** One wall forces a detour north, another forces a detour
south, and those requirements contradict. The real route is north → east →
**south** → east → north, and that southward leg moves *away* from the goal for
about 40 steps. Because credit is based on the closest approach ever achieved,
those 40 steps pay exactly zero. There is no gradient anywhere pointing at the
escape, so the coral route dead-ends at (58, 90).

**Maze B shortens the second wall** from y<100 to y<78. Now greedy following
works and the coral route reaches the goal. Same start, same goal, same reward,
same everything else.

That single change is the paper's controlled manipulation: it separates reward
*resolution* from reward *deceptiveness*, which are otherwise confounded.

---

## How reward sparsity is defined

The obvious reading of "sparse reward" is temporal — pay out less often. For an
evolutionary algorithm that reading is **degenerate**: fitness is the sum over
an episode, so moving reward around inside the episode leaves the sum, and
therefore the entire selection signal, untouched. You would sweep the parameter
and measure nothing.

So sparsity here reduces the *information content* of the fitness:

```
fitness(η) = B·1[goal reached] + w·C_η(trajectory)
L(η)       = 2^(8(1−η))        →  256, 128, 64, 32, 16, 8, 4, 2, 1
```

`C_η` is intermediate credit quantised to `L(η)` levels. Three invariants make
the sweep interpretable, all verified numerically by
`scripts/02_sparsity_check.py`:

| invariant | why it matters |
|---|---|
| max fitness = 2.0 at every η | a successful agent scores identically in every condition |
| environment, dynamics and optimal policy never change | η cannot smuggle in task difficulty |
| the level ladder is dyadic, so partitions are **nested** | raising η can only destroy information, never rearrange it — for every trajectory, not on average |

**Reporting is always in η-independent units** (the η=0 score), never the
training reward of each condition. Otherwise "sparse runs score lower" is true
by definition and says nothing. That single choice is what separates a finding
from a tautology.

Reward sparsity is read in exactly one file, `reward.py`. It cannot reach the
mutation operators, the speciation code, or the environment, so it can only act
through selection. The experimental control is enforced structurally rather
than by convention.

---

## Comparing an evolutionary learner against a gradient learner

NEAT scores an **episode**: one number at the end. DQN needs a reward at **every
step**. Hand DQN a lump sum at the end and you have not run "DQN at η=0" — you
have run DQN with terminal-only reward at every η, and rigged the comparison.

The fix is a telescoping per-step reward. With `p_t` the progress after step *t*
and `Q(·)` the quantiser:

```
r_t  = w · ( Q(p_t) − Q(p_{t−1}) )       for t = 1..T
r_T += B                                  if the goal was reached

Σ r_t = w·C_η + B·1[goal] = the NEAT fitness,  exactly
```

This is potential-based shaping (Ng, Harada & Russell 1999) applied to the
quantised potential, so the optimal policy is provably unchanged.

**Verified numerically, not asserted:** `max |return − fitness| = 0.000e+00`
across 784 trajectories × 9 η values × 3 sparsity modes × 2 credit bases,
including exact quantiser bin edges. `scripts/10_dqn_check.py` runs this and
exits non-zero on failure, so no sweep can start without it.

This construction is reusable by anyone comparing episode-level and step-level
learners.

---

## Repository layout

```
neat_sparsity/        NEAT from scratch — genome, speciation, instrumented loop
  reward.py             the η-parameterised reward (the only place η is read)
  env.py                deterministic 2D navigation task
  stats.py              Kruskal–Wallis, Dunn, Cliff's δ, bootstrap, thresholds

dqn_sparsity/         the DQN arm — imports neat_sparsity, never reimplements it
  shaping.py            telescoping reward with exact return equivalence
  step_env.py           reset/step wrapper over the same NavEnv (conformance-tested)
  nets.py               numpy MLP + Adam + Huber, no framework dependency
  agent.py              replay, target network, Double DQN, ε-greedy
  mazes.py              maze A (deceptive) and maze B (not)
  maze_analysis.py      computes deceptiveness and drift-reachability

scripts/              00–09 NEAT arm · 10–22 DQN arm, figures, mechanism
tests/                smoke tests + 23 DQN tests including a determinism check
```

Dependency-light: `numpy`, `scipy`, `pandas`, `matplotlib`. CPU only, no GPU,
no deep-learning framework.

---

## Running it

```bash
pip install -r requirements.txt

# NEAT arm
python scripts/02_sparsity_check.py                     # verify η is sound
python scripts/00_calibrate.py --seeds 6                # freeze the setup
python scripts/04_run_sweep.py --workers 8 --seeds 30   # main sweep
python scripts/05_ablation.py --out results/core_sweep
python scripts/06_analyze.py  --out results/core_sweep

# DQN arm
python scripts/10_dqn_check.py                          # conformance gate
python scripts/11_dqn_calibrate.py --seeds 8 --workers 8
python scripts/13_dqn_sweep.py --outdir results/dqn_A --maze A --workers 8 --seeds 30
python scripts/14_dqn_analyze.py --outdir results/dqn_A

# comparison and figures
python scripts/15_crossover.py --neat results/core_sweep --dqn results/dqn_A --outdir results/cmp_A
python scripts/17_threshold_figure.py --neat results/core_sweep --dqn results/dqn_A --outdir results/cmp_A
python scripts/18_maze_figures.py --outdir results/figures

# mechanism
python scripts/patch_population_ties.py --verify && python scripts/patch_population_ties.py --apply
python scripts/21_mechanism_sweep.py --workers 8
python scripts/22_aggregate_ties.py --sweep results/mechanism
```

Tests: `python tests/test_smoke.py` and `python tests/test_dqn.py`

Cost on 8 cores: NEAT sweep ~1.2 h, DQN sweep ~2.2 h, mechanism sweep ~1.4 h.

On Windows, set `OMP_NUM_THREADS=1` (and the MKL/OpenBLAS equivalents) before
parallel sweeps — otherwise each worker spawns its own BLAS thread pool and they
fight over cores.

---

## Methodological choices worth stealing

- **Config hashing.** Every output carries a hash of everything that could
  change a result. Analysis scripts refuse to pool runs whose hashes differ.
- **Calibration gates that test range, never the hypothesis.** The dense
  endpoint must be solvable but not trivial, and topology must actually change,
  *before* any sweep runs. Failing a gate is a configuration problem, not a
  finding.
- **Reporting in condition-independent units**, so degradation is a measurement
  rather than a definition.
- **Reproducibility assertions.** The same seed run twice must produce
  byte-identical generation logs; if it doesn't, some unordered iteration has
  crept in and nothing downstream is trustworthy.
- **Measurement-only instrumentation, verified as such.** The selection-tie
  patch is checked by running the same seed before and after and asserting every
  pre-existing field is identical.
- **A deviations log.** 17 entries, including the ones that don't flatter the
  work.

---

## Honest limitations

- **The DQN baseline is not fully validated.** On CartPole-v1 under published
  tuned hyperparameters it reaches a best 100-episode mean of 347 against the
  canonical 475 threshold. The resolution ratio is therefore reported as an
  **upper bound** — a stronger implementation could only need *fewer* levels,
  narrowing the ratio rather than reversing it. On both experimental tasks the
  implementation is provably at ceiling in the dense condition, so it does not
  limit the anchor point.
- **Two tasks, both navigation, and one saturates.** Drift-reachability is
  proposed as a general moderator but demonstrated within one task family.
- **The mechanism direction was not predicted correctly.** The pre-registration
  predicted a looser selection cut would tolerate coarser reward; the opposite
  was observed. The threshold moves — which is the claim under test — but the
  explanation for *which way* is post-hoc and labelled as such.
- **One of three sparsity operationalisations has not been run.** `gated` has;
  `checkpoint` has not. Until it does, the causal-chain claim is not supported
  by the study's own stated criteria.
- **No crossover between the two methods exists** on this grid. Reported as a
  null result rather than extrapolated.

---

## Related work

- Stanley & Miikkulainen (2002) — NEAT
- Lehman & Stanley (2011), *Abandoning Objectives* — the canonical deceptive-maze
  result, using a closely related domain. Their answer to deception is to
  abandon the objective; this work asks how much *resolution* the objective needs
- Davarynejad et al., fitness granulation — coarsens fitness to save computation
  on the grounds that EAs care about rankings rather than exact values. That
  assumption is what this work measures
- Salimans et al. (2017) — evolution strategies as an alternative to RL
- Ng, Harada & Russell (1999) — potential-based shaping

---

## Status

All four experimental cells complete. Paper in preparation.

Author: R Vishal Prasad (2607110017)