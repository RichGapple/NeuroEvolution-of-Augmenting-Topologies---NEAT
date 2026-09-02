"""Per-generation measurements.

All metric definitions live here and are fixed *before* any result is inspected
(roadmap Appendix C).  Three families:

  performance      -- reported in sparsity-independent units (true objective)
  topology         -- size, density, depth, innovation
  evolutionary     -- species, diversity, fitness differentiation

Fitness differentiation deserves a note.  H1 says sparsity "reduces the amount
of useful fitness differentiation between individuals".  Three complementary
operationalisations are recorded:

  n_distinct_fitness  : how many distinguishable fitness values exist
  fitness_entropy     : Shannon entropy of the selection distribution, in nats
  selection_differential : mean(parents) - mean(population), in true-objective
                           units -- i.e. how much better the individuals that
                           actually get to reproduce are.  This is the classical
                           quantitative-genetics measure of realised selection
                           pressure and is the one that plugs directly into the
                           causal chain of Section 5.
"""

from __future__ import annotations

import math
from statistics import mean, median, pstdev
from typing import Dict, Iterable, List, Sequence, Set

from .genome import Genome, HIDDEN, INPUT, OUTPUT


# --------------------------------------------------------------------------- #
# Topology
# --------------------------------------------------------------------------- #
def network_depth(g: Genome) -> int:
    """Longest path from any input to any output over enabled edges (0 if none)."""
    adj: Dict[int, List[int]] = {}
    indeg: Dict[int, int] = {k: 0 for k in g.nodes}
    for c in g.conns.values():
        if c.enabled and c.key[0] in g.nodes and c.key[1] in g.nodes:
            adj.setdefault(c.key[0], []).append(c.key[1])
            indeg[c.key[1]] += 1
    order: List[int] = []
    ready = [k for k in sorted(g.nodes) if indeg[k] == 0]
    while ready:
        k = ready.pop(0)
        order.append(k)
        for nxt in adj.get(k, ()):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
    if len(order) != len(g.nodes):          # cyclic
        return -1
    dist = {k: 0 for k in g.nodes}
    for k in order:
        for nxt in adj.get(k, ()):
            dist[nxt] = max(dist[nxt], dist[k] + 1)
    outs = [dist[k] for k, n in g.nodes.items() if n.type == OUTPUT]
    return max(outs) if outs else 0


def genome_topology(g: Genome) -> Dict[str, float]:
    n_nodes = len(g.nodes)
    n_hidden = sum(1 for n in g.nodes.values() if n.type == HIDDEN)
    n_enabled = sum(1 for c in g.conns.values() if c.enabled)
    n_total = len(g.conns)
    # density relative to the maximum a feedforward graph of this size could hold
    n_in = sum(1 for n in g.nodes.values() if n.type == INPUT)
    n_out = sum(1 for n in g.nodes.values() if n.type == OUTPUT)
    max_edges = n_in * (n_hidden + n_out) + n_hidden * (n_hidden - 1) / 2 + n_hidden * n_out
    density = n_enabled / max_edges if max_edges > 0 else 0.0
    return {
        "nodes": n_nodes,
        "hidden_nodes": n_hidden,
        "connections": n_total,
        "enabled_connections": n_enabled,
        "disabled_connections": n_total - n_enabled,
        "density": density,
        "depth": network_depth(g),
    }


def innovation_set(g: Genome) -> Set[int]:
    return {c.innovation for c in g.conns.values()}


# --------------------------------------------------------------------------- #
# Fitness differentiation
# --------------------------------------------------------------------------- #
def fitness_entropy(fitnesses: Sequence[float]) -> float:
    """Entropy (nats) of the fitness-proportional selection distribution.

    NOTE THE DIRECTION.  Maximum entropy means *uniform* selection probability,
    i.e. no differentiation at all.  So high entropy = weak selection.  The
    report uses `fitness_differentiation` = 1 - normalised entropy instead, so
    that every H1 metric points the same way (higher = more differentiation).
    """
    lo = min(fitnesses)
    shifted = [f - lo + 1e-12 for f in fitnesses]
    tot = sum(shifted)
    if tot <= 0:
        return 0.0
    ps = [s / tot for s in shifted]
    return -sum(p * math.log(p) for p in ps if p > 0)


def normalized_entropy(fitnesses: Sequence[float]) -> float:
    n = len(fitnesses)
    if n <= 1:
        return 0.0
    return fitness_entropy(fitnesses) / math.log(n)


def n_distinct(values: Sequence[float], tol: float = 1e-9) -> int:
    out: List[float] = []
    for v in sorted(values):
        if not out or abs(v - out[-1]) > tol:
            out.append(v)
    return len(out)


def gini(values: Sequence[float]) -> float:
    xs = sorted(max(0.0, v) for v in values)
    n = len(xs)
    s = sum(xs)
    if n == 0 or s == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * cum) / (n * s) - (n + 1) / n


# --------------------------------------------------------------------------- #
# Diversity
# --------------------------------------------------------------------------- #
def mean_pairwise_distance(genomes: Sequence[Genome], cfg, rng, n_pairs: int = 40) -> float:
    n = len(genomes)
    if n < 2:
        return 0.0
    tot, cnt = 0.0, 0
    for _ in range(n_pairs):
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        tot += Genome.distance(genomes[i], genomes[j], cfg)
        cnt += 1
    return tot / cnt if cnt else 0.0


# --------------------------------------------------------------------------- #
# Per-generation record
# --------------------------------------------------------------------------- #
def generation_record(
    generation: int,
    genomes: Sequence[Genome],
    true_scores: Sequence[float],
    species_sizes: Sequence[int],
    species_ages: Sequence[int],
    innovations_now: Set[int],
    innovations_prev: Set[int],
    innovations_prev_new: Set[int],
    structural_events: int,
    compat_threshold: float,
    diversity: float,
    selection_differential: float,
    reached_goal_count: int,
) -> Dict[str, float]:
    """Assemble one row of the longitudinal log."""
    fits = [g.fitness for g in genomes]
    tops = [genome_topology(g) for g in genomes]

    def m(key: str) -> float:
        return mean(t[key] for t in tops)

    # ---- Topological Innovation Rate ------------------------------------- #
    # innovations_prev_new : ids that appeared for the first time in gen-1
    # TIR = fraction of those still present in the current population.
    if innovations_prev_new:
        retained = len(innovations_prev_new & innovations_now)
        tir = retained / len(innovations_prev_new)
    else:
        tir = float("nan")
        retained = 0

    new_now = innovations_now - innovations_prev

    rec = {
        "generation": generation,
        # --- performance (sparsity-independent units) --------------------- #
        "true_best": max(true_scores),
        "true_mean": mean(true_scores),
        "true_median": median(true_scores),
        "success_rate": reached_goal_count / len(genomes),
        # --- training fitness (eta-dependent, for diagnostics only) ------- #
        "fit_best": max(fits),
        "fit_mean": mean(fits),
        "fit_std": pstdev(fits) if len(fits) > 1 else 0.0,
        "fit_variance": (pstdev(fits) ** 2) if len(fits) > 1 else 0.0,
        # --- fitness differentiation (H1) --------------------------------- #
        "n_distinct_fitness": n_distinct(fits),
        "selection_entropy_norm": normalized_entropy(fits),
        "fitness_differentiation": 1.0 - normalized_entropy(fits),
        "fitness_cv": (pstdev(fits) / mean(fits)) if mean(fits) > 1e-12 else 0.0,
        "fitness_gini": gini(fits),
        "selection_differential": selection_differential,
        # --- topology (H3) ------------------------------------------------- #
        "mean_nodes": m("nodes"),
        "mean_hidden": m("hidden_nodes"),
        "mean_connections": m("connections"),
        "mean_enabled_connections": m("enabled_connections"),
        "mean_density": m("density"),
        "mean_depth": m("depth"),
        "max_nodes": max(t["nodes"] for t in tops),
        "max_connections": max(t["connections"] for t in tops),
        "champion_nodes": tops[max(range(len(fits)), key=lambda i: fits[i])]["nodes"],
        "champion_connections":
            tops[max(range(len(fits)), key=lambda i: fits[i])]["connections"],
        # --- innovation (H2) ------------------------------------------------ #
        "structural_events": structural_events,
        "new_innovations": len(new_now),
        "retained_innovations": retained,
        "tir": tir,
        "cumulative_innovations": len(innovations_now | innovations_prev),
        # --- evolutionary dynamics ------------------------------------------ #
        "n_species": len(species_sizes),
        "species_size_mean": mean(species_sizes) if species_sizes else 0.0,
        "species_size_max": max(species_sizes) if species_sizes else 0,
        "species_age_mean": mean(species_ages) if species_ages else 0.0,
        "compat_threshold": compat_threshold,
        "genomic_diversity": diversity,
    }
    return rec
