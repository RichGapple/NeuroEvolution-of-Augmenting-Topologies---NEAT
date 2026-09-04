#!/usr/bin/env python3
"""Generate the Methodology section from the config that was actually run.

Papers drift from code. The usual failure is a methods section describing a
population size, mutation rate or seed count that stopped being true three
revisions ago. Generating the prose from `config.json` and `summary.csv` makes
that impossible: if the numbers in the paper are wrong, the sweep is wrong.

Writes <out>/analysis/methods.md, ready to paste into Section 3.

Usage
-----
    python scripts/09_methods.py --out results/core_sweep
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from neat_sparsity.analysis import load_summary, load_ablation
from neat_sparsity.config import ExperimentConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/core_sweep")
    args = ap.parse_args()

    cfg = ExperimentConfig.load(os.path.join(args.out, "config.json"))
    df = load_summary(args.out)
    abl = load_ablation(args.out)

    n = cfg.neat
    e = cfg.env
    s = cfg.sparsity
    etas = sorted(df["eta"].unique())
    seeds = df["seed"].nunique()
    L = ", ".join(f"{x:g}\u2192{s.levels(x)}" for x in etas)

    obstacles = "; ".join(f"({a:g},{b:g})-({c:g},{d:g})" for a, b, c, d in e.obstacles)

    md = f"""# 3. Methodology

*(generated from config hash `{cfg.config_hash()}` — the configuration these
results were produced under. Regenerate with `scripts/09_methods.py` after any
change to the sweep.)*

## 3.1 Environment

Agents are evaluated in a deterministic continuous two-dimensional navigation
task in a {e.width:g}\u00d7{e.height:g} arena containing {len(e.obstacles)} axis-aligned
rectangular obstacles at {obstacles}. The agent starts at
({e.start[0]:g}, {e.start[1]:g}) with heading {e.start_heading:g} rad and must reach a goal at
({e.goal[0]:g}, {e.goal[1]:g}) within a radius of {e.goal_radius:g} units. Episodes last at most
{e.max_steps} steps.

Observations comprise {e.n_rangefinders} rangefinder readings spanning a
{np.degrees(e.rangefinder_fov):.0f}\u00b0 field of view and capped at {e.rangefinder_max:g} units,
the normalised Euclidean distance to the goal, and the sine and cosine of the
goal bearing relative to the agent's heading, giving {e.n_observations()} inputs.
The three outputs are interpreted as a discrete action by argmax: turn left,
turn right, or move forward ({e.turn_rate:g} rad per turn, {e.speed:g} units per
forward step). The agent has radius {e.agent_radius:g}; forward motion into an
obstacle or wall is rejected and the agent remains in place.

The environment contains no stochasticity. Given a genome, the trajectory is a
deterministic function of the policy, so all between-condition variance
originates in the evolutionary algorithm and the reward manipulation rather
than in the task.

## 3.2 Neuroevolution

We use a from-scratch NEAT implementation with global innovation numbering,
per-generation innovation memory, explicit fitness sharing, and speciation by
compatibility distance. Genomes are initialised
{"fully connected" if n.initial_connection == "full" else "sparsely connected"}
between the {n.num_inputs} inputs and {n.num_outputs} outputs with no hidden
nodes. Networks are constrained to be feedforward: a candidate connection is
rejected if it would create a cycle.

Population size is {n.pop_size}. Per genome per generation, an add-node mutation
occurs with probability {n.add_node_prob:g} and an add-connection mutation with
probability {n.add_conn_prob:g}; connection enable/disable toggles with
probability {n.toggle_enable_prob:g}. Connection weights mutate with probability
{n.weight_mutate_prob:g}, of which {n.weight_perturb_prob:.0%} are Gaussian
perturbations (\u03c3 = {n.weight_perturb_std:g}) and the remainder are replacements
drawn from N(0, {n.weight_replace_std:g}\u00b2); weights are clamped to
\u00b1{n.weight_clamp:g}. Node biases mutate analogously with probability
{n.bias_mutate_prob:g}.

Offspring are produced by crossover with probability {n.crossover_prob:g}
(interspecies with probability {n.interspecies_mating_prob:g}); a gene disabled
in either parent remains disabled with probability {n.inherit_disabled_prob:g}.
Compatibility distance weights disjoint/excess genes by {n.c_disjoint:g} and mean
matched-weight difference by {n.c_weight:g}. The compatibility threshold starts at
{n.compat_threshold:g} and{" is adapted by " + format(n.compat_adjust, "g") +
" per generation toward " + str(n.target_species) + " species"
if n.dynamic_compat else " is held fixed"}. Species that fail to improve for
{n.stagnation_generations} generations are removed, except the
{n.species_elitism} best. The top {n.survival_threshold:.0%} of each species may
reproduce and the best {n.elitism} genomes of each species are copied unchanged.
All activations are {n.activation}.

## 3.3 Reward sparsity

Reward sparsity \u03b7 \u2208 [0, 1] is the sole independent variable. Fitness is

    f(\u03b7) = B\u00b71[goal reached] + w\u00b7C_\u03b7(\u03c4)

with B = {s.success_bonus:g} and w = {s.progress_weight:g}. The intermediate
credit term C_\u03b7 quantises the trajectory's fractional progress toward the goal,
p(\u03c4) = 1 \u2212 d_min/d_0, to L(\u03b7) levels:

    C_\u03b7(\u03c4) = \u230a p(\u03c4)\u00b7L(\u03b7) \u230b / L(\u03b7),    L(\u03b7) = 2^round(K(1\u2212\u03b7)),  K = log\u2082 {s.levels_max}

At \u03b7 = 0 this gives {s.levels(0.0)} levels, effectively continuous shaping; at
\u03b7 = 1 it gives a single level, so unsuccessful trajectories receive no credit
and reward is purely terminal. Over the \u03b7 grid used here the ladder is
{L}.

Three properties hold by construction and are verified numerically before every
sweep. First, maximum attainable fitness is B + w = {s.success_bonus + s.progress_weight:g}
at every \u03b7, so a successful agent is scored identically in all conditions.
Second, the environment, its dynamics and the optimal policy are unchanged by
\u03b7, so task difficulty is not confounded with reward sparsity. Third, because
successive values of L(\u03b7) divide one another, the induced partitions of the
progress interval are nested and C_\u03b7 is monotone non-increasing in \u03b7 for every
individual trajectory: raising \u03b7 can only destroy reward information, never
rearrange it.

All performance results are reported using an \u03b7-independent objective, defined
as the \u03b7 = 0 fitness. Reporting performance in training-fitness units would make
the observation that sparse conditions score lower true by definition.

## 3.4 Experimental design and control

We ran {len(etas)} sparsity conditions (\u03b7 = {", ".join(f"{x:g}" for x in etas)})
\u00d7 {seeds} independent random seeds = {len(df)} evolutionary runs of
{cfg.generations} generations each. Population size, mutation probabilities,
compatibility and speciation parameters, activation functions, generation count,
selection mechanism, initial topology, genome representation, evaluation
procedure, computational budget and the environment are identical across
conditions; only \u03b7 differs. Every run is fully determined by its configuration
and seed, and every output file records a hash of the configuration so that runs
produced under different settings cannot be pooled.

The seed count was set from a pilot that measured between-seed variance rather
than assumed a priori. Before the main sweep we verified that the dense
condition solves the task in a non-trivial fraction of runs but not
immediately, that network topology has range to vary under dense reward, and
that \u03b7 measurably reduces fitness differentiation.

## 3.5 Recorded metrics

Rather than only final networks, we record population and topology statistics
every generation of every run. Performance: best, mean and median score on the
\u03b7-independent objective, and success rate. Fitness differentiation: number of
distinct fitness values, a differentiation index derived from the normalised
entropy of the selection distribution, the fitness coefficient of variation,
and the selection differential (mean fitness of reproducing individuals minus
population mean). Topology: node count, hidden-node count, total and enabled
connection counts, density and depth, for both the population mean and the
generation champion. Innovation: structural mutation events, count of novel
innovation identifiers, and the Topological Innovation Rate

    TIR_g = |I_{{g-1}} \u2229 P_g| / |I_{{g-1}}|

where I_{{g-1}} is the set of innovation identifiers first created in generation
g\u22121 and P_g the set present in generation g — i.e. the retention rate of newly
introduced structure. Evolutionary dynamics: species count, species age,
compatibility threshold, and mean pairwise compatibility distance as a measure
of genomic diversity.

## 3.6 Functional topology analysis

Champions are analysed by ablation under the \u03b7-independent objective. Scoring
ablations under the training reward would be circular: at high \u03b7 the training
signal is degenerate, so nearly every ablation would register no change
regardless of whether the removed structure mattered.

Two procedures are applied. Single-component ablation disables one connection,
or removes one hidden node together with all its incident connections, and
records \u0394 = score(intact) \u2212 score(ablated); a component is classified as
functional when \u0394 > \u03c4 = {abl['tau'].iloc[0] if not abl.empty and 'tau' in abl else 0.01:g}
on a scale whose maximum is {s.success_bonus + s.progress_weight:g}. Because
single-component ablation under-counts redundancy when two components are
mutually redundant, we additionally perform greedy sequential pruning, removing
the least damaging component repeatedly while the score stays within \u03c4 of the
original, and report the fraction of structure removable this way. The two
measures bracket the amount of non-functional structure from opposite
directions.

## 3.7 Statistical analysis

Distributions were inspected before tests were selected. The unit of analysis is
the run. Across conditions we use the Kruskal\u2013Wallis test with \u03b5\u00b2 as effect
size, followed where warranted by Dunn's test with tie correction and Holm
adjustment. Pairwise effects are reported as Cliff's \u03b4 with the usual
negligible/small/medium/large thresholds. Condition medians carry 95 percent
bootstrap confidence intervals (10\u2074 resamples). Monotone trend across \u03b7 is
assessed by Spearman's \u03c1 and the Jonckheere\u2013Terpstra test.

Nonlinearity is assessed by comparing a continuous two-segment piecewise-linear
fit over \u03b7, with the breakpoint selected by grid search over the \u03b7 values,
against a single linear fit, using AIC; a bootstrap over runs gives a confidence
interval on the breakpoint. We describe a preferred piecewise fit as a change in
slope at an estimated breakpoint, and do not use phase-transition language.

Effect sizes and confidence intervals accompany every reported test. A
non-significant result with a wide interval is reported as underpowered rather
than as evidence of no effect.
"""

    andir = os.path.join(args.out, "analysis")
    os.makedirs(andir, exist_ok=True)
    path = os.path.join(andir, "methods.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"wrote {path}  ({len(md.split())} words)")
    print("\nCheck the numbers against your intent, then paste into Section 3.")


if __name__ == "__main__":
    main()
