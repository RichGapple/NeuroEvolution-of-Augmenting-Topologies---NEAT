# Pre-registration addendum — DQN comparison arm

**Status: COMPLETED.** Sections 1–6 and 8 were fixed before any DQN data
existed. Section 7 is a factual record filled in after each gate ran. Section 10
scores the predictions against the results. Section 9 logs every deviation.

---

## 0. Honesty note on what was and was not recorded in advance

A pre-registration is worthless if it is written after the fact, so the
provenance of each part is stated explicitly:

| Item | When fixed | Evidence |
|---|---|---|
| §1 question, §2 predictions P1–P5, §3 primary metric, §4 primary test, §5 falsifiers, §6 fixed parameters, §8 exploratory list | **Before any DQN run** | Present in the version of this file shipped with the DQN arm, prior to `11_dqn_calibrate.py` being executed |
| §2 predicted crossover location | **Before the sweep, but recorded in project correspondence rather than in this file** | Stated as "η ≈ 0.5–0.7" in a project message dated before the sweep. This field was left blank in the file itself. Recorded here as a deviation, not as a filled prediction. |
| §7 record | **After each gate ran** | Factual record of outputs, not a prediction |
| §9 deviations | **Ongoing** | Each entry dated |
| §10 outcomes | **After the sweep** | Scoring, not prediction |

Nothing in §2 has been altered since it was written.

---

## 1. Question

Does the reward-resolution threshold at which learning fails differ between a
gradient-based reinforcement learner (DQN) and a neuroevolutionary one (NEAT),
on an identical task under an identical reward manipulation and a matched
interaction budget?

## 2. Directional predictions

Recorded before running. Being wrong here is fine and publishable; being vague
is not.

| Prediction | Direction |
|---|---|
| P1 | At η = 0, DQN scores **higher** than NEAT (dense reward favours gradient methods) |
| P2 | At η = 1, DQN scores **lower** than NEAT (no temporal credit assignment possible) |
| P3 | A crossover exists somewhere on the grid |
| P4 | DQN's breakpoint is at a **lower** η than NEAT's |
| P5 | DQN reaches NEAT's final score using **< 30%** of the budget at η = 0 |

Predicted crossover location, before seeing any data: **η ≈ 0.5–0.7**
*(recorded in project correspondence, not in this file — see §0)*

## 3. Primary metric

**`final_true_best`** — the η = 0 objective score of the final greedy policy.
One metric, declared in advance. Everything else in `report.md` is exploratory.

Rationale: it is the only quantity that means exactly the same thing in both
arms. `final_success_rate` does not — for DQN it is the greedy policy's success
on an evaluation episode, for NEAT it is the fraction of the population that
solved.

## 4. Primary test

Mann–Whitney U at each η, Holm-adjusted across the grid, α = 0.05, reported with
Cliff's δ and bootstrap CIs. Non-parametric because the outcome has a hard
ceiling at 2.0 and a floor at 0.

Crossover located by linear interpolation between the bracketing grid points,
with a 4,000-resample bootstrap CI.

**Decision rule agreed in advance:** if the crossover CI is wider than 0.30 on
the η axis, the crossover is reported as *located between two grid points*, not
as a point estimate. No exceptions negotiated after seeing the width.

## 5. What would falsify the hypothesis

- No crossover: one arm dominates the whole grid. → Report the null crossover.
  Do not extrapolate past the grid.
- DQN wins at η = 1 too. → The "evolution is robust to sparse reward" folklore
  is overstated at this scale. Publishable as a negative result.
- Neither arm degrades. → The manipulation is not reaching the learners. Check
  `zero_reward_fraction` and `reward_entropy` before touching anything else.

**Outcome: the first falsifier fired.** No crossover exists; NEAT dominates the
whole grid in absolute score. Reported as a null crossover in
`results/cmp_A/report.md`, with no extrapolation past the grid.

## 6. Fixed in advance

- **Budget:** 1,500,000 env steps per DQN run = NEAT's upper bound
  (100 gen × 100 pop × 150 steps). Deliberately generous to the baseline.
- **Hyperparameters:** tuned once at η = 0, then frozen for the whole grid.
  Per-condition tuning would make the sweep a comparison of tuning effort.
- **Seeds:** 30 per condition, the same seed values as the NEAT arm (1000–1029).
- **Shaping:** `telescoping` is the comparison. `terminal` is an ablation and
  will never be reported as "DQN".
- **Stopping rule:** no adding seeds after seeing results. If more power is
  needed, that is a new experiment with a new hash.

All five held. No seeds were added after results were seen; the mechanism sweep
(§9) uses a disjoint seed range (5000–5019) and a distinct config hash.

## 7. Record before the sweep

```
DQN config hash:        b6a1821ae237   (full 9-point eta grid, maze A)
DQN calibration hash:   ef0f9bd07368   (etas = [0, 1]; differs because
                                        etas is part of the hash)
NEAT config hash:       5b727c4ff37c

Conformance check:      PASS           (10_dqn_check.py)
  max |return - fitness|:  0.000e+00   across 784 trajectories
                                       x 9 eta x 3 sparsity modes
                                       x 2 credit bases, incl. bin edges
  C1 StepNavEnv == NavEnv.rollout:      PASS, 200 policies x 2 passes,
                                        byte-identical
  C3 max fitness eta-invariant:         PASS, 2.0 at every eta
  C4 finite-difference gradient check:  PASS, max rel. error 6.84e-07
  Test suite:                           23/23 passing

Calibration:            PASS           (11_dqn_calibrate.py, revised D1)
  D1 greedy-optimum attainment:  0.697 / 0.717 = 97%    PASS
     (dense solve rate 0/8 — expected on a deceptive maze; see §9)
  D2 not solved immediately:     earliest solution block = None   PASS
  D3 untrained -> trained:       0.000 -> 0.697                   PASS
  D4 eta=0 vs eta=1:             0.697 vs 0.053 (13.2x)           PASS
  D5 headroom:                   0.000 < 0.697 < 2.000            PASS
  zero-reward transition fraction: 0.679 (eta=0) -> 1.000 (eta=1)

Maze B sweep hashes:    NEAT 437d19b53606, DQN 9e5e577808f6

Date:                   5 September 2026
```

## 8. Analyses that are exploratory, not confirmatory

Everything below is fine to report, and must be labelled exploratory:

- all metrics other than `final_true_best`
- sample-efficiency curves
- the `terminal` shaping ablation *(not run)*
- per-arm segmented-regression breakpoints
- `q_spread` as an analogue of NEAT's selection differential
- cost accounting

**Added to the exploratory list after the fact**, and labelled as such in the
manuscript:

- the maze B comparison (§9)
- the drift-reachability measurement (§9)
- the ceiling-normalised threshold analysis (§9)
- the mechanism sweep and everything derived from it (§9)
- the CartPole-v1 baseline validation (§9)

## 9. Deviations log

Any change after this document is signed goes here, with a date and a reason.
An empty log is a claim; a populated one is honesty.

| # | Date | Change | Reason |
|---|---|---|---|
| 1 | 2026-09-04 | γ changed 1.0 → 0.99; added `reward_scale = 10.0` applied learner-side only | γ = 1.0 removes the contraction property of the Bellman operator, so targets have no fixed point. Separately, a forward step at η = 0 earns ~0.012 of reward, which with Huber δ = 1.0 collapsed the Q-function to a constant (`td_loss` 3.1e-5, `q_spread` 0.0038) while performance sat at random-policy level. Measured effect at η = 0: 0.139 → 0.697. Reporting remains in unscaled true-objective units; a regression test asserts `reward_scale` cannot leak into any reported value. |
| 2 | 2026-09-04 | Timeout-as-terminal tested and **rejected** | The observation vector has no time channel, so bootstrapping to zero at step 150 teaches contradictory targets for states that look identical early and late. Measured: 0.362 → 0.118. Kept non-terminal. |
| 3 | 2026-09-05 | Calibration gate D1 revised from "solve the maze in 30–95% of runs" to "reach ≥ 95% of the maze's greedy-descent optimum". Gate D5 (headroom) added. | Grid analysis (`maze_analysis.py`) shows maze A is deceptive: greedy distance-descent traps at (58, 90), true objective 0.717. Since the intermediate credit term is monotone in distance-to-goal, that point is the ceiling for any gradient follower. DQN reached 0.697 (97%) across 8 seeds with 7/8 inside a 0.05 band — a converged learner at its signal's ceiling, not a broken one. The original gate was measuring maze topology rather than learner health. On a non-deceptive maze the greedy optimum *is* the goal, so the revised gate reduces to the original. |
| 4 | 2026-09-05 | Added `credit_basis` option ("min" / "current") to the shaping module | Exposes an alternative telescoping decomposition with denser per-step signal at identical episode total. Default unchanged ("min"). Both bases verified return-equivalent to 0.000e+00. |
| 5 | 2026-09-05 | **Added maze B**, identical to maze A except the second obstacle is shortened from y < 100 to y < 78 | On maze A, reward *resolution* and reward *deceptiveness* are confounded: DQN fails at every η and the cause is unidentifiable. Two mazes differing in one property separate them. |
| 6 | 2026-09-05 | Maze B calibration skipped | The maze B sweep's η = 0 cell answered the same question on 30 seeds rather than 8 (DQN 2.000 in all 30). Recorded rather than quietly omitted. |
| 7 | 2026-09-05 | **Maze B retained as a deception control only; not reported as an η sweep for NEAT** | NEAT scored 2.000 in all 270 maze B runs including η = 1, where the reward carries zero information. Drift-reachable solve rate is 0.075% on B versus 0% on A, so ~8 of the 10,000 genomes a run evaluates solve B by undirected search alone. Maze B therefore fails calibration gate C1's upper bound (100% > 95%) and cannot detect NEAT degradation. Tightening `max_steps` does not fix it — the drift rate stays ~0.1% down to 8% slack, below which the path no longer fits. |
| 8 | 2026-09-05 | **Added the drift-reachability measurement** (4,000 undirected policies per maze, weight scales σ ∈ {0.5, 1, 2, 4}) | Not anticipated in the original design. Emerged from diagnosing deviation 7 and became the study's primary explanatory variable. Fully exploratory; labelled as such. |
| 9 | 2026-09-05 | **Added ceiling-normalised threshold analysis** as the primary cross-arm comparison, superseding the pre-registered crossover | The pre-registered comparison of absolute scores conflates deception (which sets each arm's ceiling) with sparsity (which sets where each arm leaves its ceiling). The segmented-regression breakpoints produced by `15_crossover.py` are also unusable: a continuous two-segment fit cannot represent a step function, and the η = 0.875 breakpoints are boundary artefacts of the degenerate endpoint. The normalised analysis compares thresholds on the identical maze and budget. Exploratory. |
| 10 | 2026-09-05 | **Added the mechanism sweep** (`21_mechanism_sweep.py`), 840 runs, seeds 5000–5019, disjoint from the main sweep | Tests whether the reward-resolution threshold is a property of the selection cut, as claimed. Predictions recorded in `results/mechanism/predictions.json` **before** the sweep executed. |
| 11 | 2026-09-05 | **Mechanism predictions were wrong in direction.** Predicted a looser truncation cut would tolerate coarser reward; the opposite was observed. | Recorded rather than reframed. Among conditions reaching ceiling: s = 0.2 → 8 levels, s = 0.3 → 8 levels, s = 0.5 → 64 levels, pop50 → 256 levels, pop200 → 8 levels. The threshold *does* move, which is the claim under test, but not in the predicted direction. The post-hoc account — that a cut near the median lands where individuals are densest in fitness and therefore most likely to tie — **must be labelled post-hoc** in the manuscript. |
| 12 | 2026-09-05 | Verdict logic in `21_mechanism_sweep.py` does not exclude conditions that fail to reach ceiling | Conditions s = 0.1 (ceiling 0.721) and s = 0.7 (ceiling 1.463) never solve the maze at any η, so their "thresholds" are meaningless. The script's automated 0/3 verdict is therefore invalid and is superseded by the manual reading in deviation 11. Logic to be fixed; the underlying data is unaffected. |
| 13 | 2026-09-05 | Added `arbitrary_selection_fraction` instrumentation to `neat_sparsity/population.py` | Direct measurement of the claimed mechanism: the share of the population whose survival was decided by genome key rather than fitness, because the truncation cut fell inside a tied group. Measurement-only; verified by running the same seed before and after and asserting every pre-existing generation-record field is identical. No config field changed, so hashes are unaffected and runs remain poolable. |
| 14 | 2026-09-05 | **Added CartPole-v1 baseline validation** (`20_dqn_benchmark.py`) | Not in the original design. Added to address the objection that a hand-written numpy DQN might be weak rather than merely correct. **Outcome: the objection is not fully closed.** Under published tuned hyperparameters (RL Baselines3 Zoo CartPole settings, MSE loss, `train_every = 2`), the implementation reaches a best 100-episode mean of 347 (range 314–386, n = 10) against the canonical 475 threshold. Consequence for the manuscript: the resolution ratio is reported as an **upper bound** on DQN's requirement, since a stronger implementation could only require fewer levels, narrowing the ratio rather than reversing it. On both experimental tasks the implementation is provably at ceiling in the dense condition (maze B 2.000 in 30/30; maze A 0.707 against an analytic maximum of 0.717), so it does not limit the η = 0 anchor. |
| 15 | 2026-09-05 | `import resource` (POSIX-only) removed from `dqn_sparsity/runner.py`; replaced with a cross-platform `sysinfo.py` | Broke the package at import time on Windows. Also fixed a latent unit bug: `ru_maxrss` is kilobytes on Linux but bytes on macOS. An AST-walking regression test now blocks module-scope POSIX-only imports. No effect on any result. |
| 16 | 2026-09-05 | `max_steps` corrected in `CALIBRATION.md` and `PREREGISTRATION.md` from 200 to **150** | The written record was wrong; the code ran 150. Confirmed by recomputing hashes: calibration hash `939192a95e0e` reproduces as `core_config()` with `max_steps = 150` and `etas = [0, 1]`. The runs are internally consistent; only the documentation was in error. |
| 17 | 2026-09-05 | `--mode gated` sweep run (`results/robust_gated`); `--mode checkpoint` **still outstanding** | The NEAT arm's `PREREGISTRATION.md` §4 commits to both, and its interpretation checklist states the causal chain is supported only if the ordering is consistent across sparsity modes. Until `checkpoint` runs, that claim is unsupported by the study's own criteria and must not be made. |

## 10. Prediction outcomes

Scored after the sweep. **2 of 5 supported.**

| | Prediction | Observed | Verdict |
|---|---|---|---|
| P1 | DQN > NEAT at η = 0 | NEAT 2.000, DQN 0.707 | **NOT SUPPORTED** |
| P2 | DQN < NEAT at η = 1 | NEAT 0.445, DQN 0.000 | **SUPPORTED** |
| P3 | A crossover exists | None on the grid; NEAT dominates throughout | **NOT SUPPORTED** |
| P4 | DQN's breakpoint at lower η than NEAT's | DQN leaves its ceiling between η = 0.25 and 0.375 (64 levels); NEAT between 0.625 and 0.75 (8 levels) | **SUPPORTED** |
| P5 | DQN matches NEAT using < 30% of budget at η = 0 | Never matched it at any budget fraction | **NOT SUPPORTED** |

**Interpretation.** P1, P3 and P5 all failed for the same reason, and it is not
the one anticipated: maze A is deceptive, so DQN is capped at the analytically
computed greedy-descent optimum (0.717) and cannot reach NEAT's score at any η
or any budget. The pre-registered framing assumed both arms could reach the
goal at η = 0; they cannot.

P4 — the prediction that survives — is the one that became the paper's
headline, once restated in ceiling-normalised terms: **NEAT holds its ceiling
down to 8 reward levels, DQN only to 64.**

---

## Signature

Recorded by: R Vishal Prasad (2607110017)
Date completed: 5 September 2026