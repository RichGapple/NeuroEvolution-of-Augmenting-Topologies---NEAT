# `dqn_sparsity` — the DQN arm

A drop-in companion to `neat_sparsity`. Runs DQN across the same reward-sparsity
grid, on the same maze, with the same reward, under a matched interaction
budget, and produces the NEAT-vs-DQN crossover analysis.

Copy the `dqn_sparsity/`, `scripts/` and `tests/` contents into your existing
repo. Nothing in `neat_sparsity/` is modified — the DQN arm imports it.

```
your-repo/
├── neat_sparsity/          unchanged
├── dqn_sparsity/           <- new
├── scripts/
│   ├── 00_calibrate.py     ... 09_methods.py    existing
│   └── 10_dqn_check.py     ... 16_cost_report.py  <- new
└── tests/
    ├── test_smoke.py       existing
    └── test_dqn.py         <- new
```

No new dependencies. numpy, scipy, pandas and matplotlib only, same as before.

---

## The one thing to understand before running anything

NEAT scores an **episode**: one number at the end, from the whole trajectory.
DQN needs a reward at **every step**. If you hand DQN a single number at the end,
you have not run "DQN at η = 0" — you have run "DQN with terminal-only reward",
which is the sparsest possible setting. The comparison would be rigged in NEAT's
favour and a reviewer will catch it in the first paragraph of your methods.

`shaping.py` fixes this by emitting per-step rewards that **telescope**:

```
r_t = w · ( Q(p_t) − Q(p_{t−1}) )        p_t = 1 − min(d_0..d_t)/d_0
r_T += B                                  if the goal was reached
```

Summing collapses the telescope:

```
Σ r_t = w · ( Q(p_T) − Q(p_0) ) + B·1[goal]  =  w·C_η + B·1[goal]
      = RewardModel.fitness(traj)            exactly, since p_0 = 0
```

With γ = 1 the undiscounted DQN return **is** the NEAT fitness, for every
trajectory and every η. That identity is what makes the comparison legitimate.
It is also potential-based shaping (Ng, Harada & Russell 1999) on the quantised
potential, so the optimal policy is provably unchanged — cite that when asked.

`scripts/10_dqn_check.py` verifies the identity numerically over hundreds of
trajectories × 9 η values × 3 sparsity modes, including exact quantiser bin
edges. It reports `max |return − fitness| = 0.000e+00`. Do not run a sweep until
it does.

---

## Run order

```bash
# 0. gates -- nothing else runs until these pass
python scripts/10_dqn_check.py            # conformance + return equivalence
python tests/test_dqn.py                  # 18 tests

# 1. calibration: does DQN work on this task at all?
python scripts/11_dqn_calibrate.py --seeds 8 --workers 8
#    -> freeze the config hash, record it in the pre-registration

# 2. pilot: how long, how many seeds?
python scripts/12_dqn_pilot.py --seeds 6 --budget 200000 --mde 0.15

# 3. the sweep
python scripts/13_dqn_sweep.py \
    --outdir results/dqn_sweep --workers 8 --seeds 30 \
    --match-neat results/sweep

# 4. single-arm analysis: stats, 22 figures, report.md
python scripts/14_dqn_analyze.py --outdir results/dqn_sweep

# 5. the headline: NEAT vs DQN
python scripts/15_crossover.py \
    --neat results/sweep --dqn results/dqn_sweep \
    --outdir results/comparison

# 6. honest compute accounting
python scripts/16_cost_report.py \
    --neat results/sweep --dqn results/dqn_sweep \
    --outdir results/comparison
```

Or `make -f Makefile.dqn all`.

---

## Budget matching

`--match-neat <dir>` derives DQN's budget from the NEAT sweep:

```
generations × pop_size × max_steps = 100 × 100 × 150 = 1,500,000 env steps
```

This is NEAT's **upper bound**. NEAT episodes terminate early on success, so NEAT
actually consumes fewer steps. Giving the baseline the larger number is
deliberate: if your conclusion is that evolution wins in sparse conditions, that
conclusion is far more persuasive when the baseline was handed extra budget.

Measured throughput: ~6,500 env steps/s single core, so a 1.5M-step run is
~4 minutes and the 270-run sweep is **~17 core-hours, about 2.2 h on 8 workers**.
Comparable to the NEAT sweep's 9 core-hours.

---

## Outputs

`results/dqn_sweep/`

| file | contents |
|---|---|
| `summary.csv` | one row per run, 30+ columns |
| `descriptives.csv` | medians, IQR, CIs per condition — **read this first** |
| `tests.csv` | Kruskal–Wallis, ε², Spearman ρ, Cliff's δ per metric |
| `posthoc.csv` | Dunn pairwise, Holm and BH adjusted |
| `report.md` | generated write-up with the null results stated as nulls |
| `figures/` | 22 PNG + PDF |
| `eta=*/seed=*/blocks.csv` | 100 evaluation blocks per run |
| `eta=*/seed=*/policy.npz` | the trained network |

`results/comparison/`

| file | contents |
|---|---|
| `crossover.json` | η\* and its bootstrap CI |
| `per_condition.csv` | Mann–Whitney + Cliff's δ at every η, Holm-adjusted |
| `sample_efficiency.csv` | budget fraction DQN needed to match NEAT |
| `cost_table.csv` | env steps / wall-clock / memory, kept separate |
| `breakpoints.json` | each arm's own segmented-regression breakpoint |
| `report.md` | including a "what this does and does not license" section |
| `figures/fig_crossover.*` | **the paper's headline figure** |

---

## Metrics specific to this arm

The DQN arm needs its own manipulation check — proof that η actually reaches
the learner. These are the counterparts of NEAT's "distinct fitness values" and
"selection entropy":

| metric | meaning |
|---|---|
| `zero_reward_fraction` | share of transitions carrying no signal at all. Should rise with η. |
| `distinct_rewards` | distinct reward values in the replay buffer. Should fall with η. |
| `reward_entropy` | normalised Shannon entropy of the reward histogram. Should fall with η. |
| `q_spread` | mean max(Q) − min(Q) over visited states. The DQN analogue of NEAT's selection differential: when it collapses, the agent can no longer tell actions apart. |
| `terminal_correction` | how much of the return arrives as the end-of-episode correction rather than as shaped per-step reward. If it dominates at high η, the agent is effectively on terminal-only reward there — report that. |

**If these do not move with η, stop.** The manipulation is not reaching the
learner and no performance result below it is interpretable.

---

## The shaping ablation

```bash
python scripts/13_dqn_sweep.py --shaping terminal --outdir results/dqn_terminal
```

Pays the entire episode fitness in one lump at the final step. This is **not**
the fair comparison — do not report it as "DQN". It is a second, cheap
experiment measuring how much of DQN's performance depends on having temporal
structure in the reward at all. The gap between `telescoping` and `terminal` is
a result in its own right and costs you one extra sweep.

---

## Sparsity modes

`--mode {quantized,checkpoint,gated}` must match the NEAT sweep you compare
against. `15_crossover.py` refuses to compare arms whose sparsity modes differ.

Your pre-registration commits to running all three on the NEAT side. Run them on
this side too, or restrict every claim to `quantized` explicitly.

---

## Things this code will not let you do

`compare.assert_comparable()` raises rather than warns when:

- either arm spans more than one config hash
- the two arms used different sparsity modes
- the two arms used different η grids
- the required columns are missing

A crossover computed across mismatched configurations is worse than no crossover
at all, so these are hard failures, not warnings.

---

## Claims you can and cannot make

**Can:**

- "At η = X, method A scores higher than method B, Cliff's δ = ..., p = ..."
- "The two curves cross at η = X, 95% CI [...]" — *if the CI is narrow*
- "DQN reached NEAT's final score using X% of the interaction budget at η = 0"
- "NEAT required more environment interaction but strictly simpler
  infrastructure: no automatic differentiation, no replay buffer, no GPU, and
  near-linear parallel scaling."

**Cannot:**

- "NEAT is better than DQN." The result is condition-dependent by construction.
  That is the entire point of the study.
- "NEAT is computationally cheaper." It is not, on this task. It consumes *more*
  environment interaction. Report the three cost columns separately and never
  collapse them into one number — the two arms move in opposite directions on
  sample efficiency versus hardware footprint, and a single figure hides exactly
  the trade the paper is about.
- Anything beyond this maze. One task is one task.

---

## Known limitations to disclose in the paper

1. **The environment is deterministic with a fixed start.** For NEAT that is a
   feature (zero environment variance). For DQN it is harsher than usual: all
   exploration must come from ε-greedy, with no environmental stochasticity to
   help. State this. If a reviewer objects, add a randomised-start variant as a
   robustness check.
2. **The learner is a hand-written numpy MLP.** Guarded by a finite-difference
   gradient check (`max relative error ~7e-7`) and by calibration gates, but say
   so and point at both.
3. **Hyperparameters are tuned once at η = 0 and frozen.** Re-tuning per
   condition would turn the sweep into a comparison of tuning effort.
4. **DQN only.** A reviewer will ask why not PPO. The honest answer is time;
   adding PPO is the highest-value next step and reuses everything except
   `agent.py`.
5. **`final_success_rate` is not the same quantity across arms.** For DQN it is
   the greedy policy's success on an evaluation episode; for NEAT it is the
   fraction of the population that solved. Use `final_true_best` for the
   cross-arm result — it is directly comparable.
