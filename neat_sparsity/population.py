"""Population, speciation and the instrumented generation loop.

The loop is deliberately verbose about bookkeeping: parents chosen, structural
mutation events, innovation ids per generation.  Those are the quantities the
study needs, and they cannot be reconstructed from a saved final genome.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from .config import NEATConfig
from .genome import Genome, InnovationTracker
from .metrics import generation_record, innovation_set, mean_pairwise_distance


@dataclass
class Species:
    key: int
    representative: Genome
    members: List[Genome] = field(default_factory=list)
    created: int = 0
    last_improved: int = 0
    best_fitness: float = float("-inf")
    adjusted_fitness_sum: float = 0.0
    offspring: int = 0

    def age(self, generation: int) -> int:
        return generation - self.created


class Population:
    def __init__(self, cfg: NEATConfig, seed: int) -> None:
        self.cfg = cfg
        self.seed = seed
        self.rng = random.Random(seed)
        self.tracker = InnovationTracker(start_node_key=cfg.num_inputs + cfg.num_outputs)
        self.generation = 0
        self._next_genome_key = 0
        self._next_species_key = 0
        self.compat_threshold = cfg.compat_threshold

        self.genomes: List[Genome] = [self._new_genome() for _ in range(cfg.pop_size)]
        self.species: List[Species] = []

        # innovation history for TIR
        self._prev_innovations: Set[int] = set()
        self._prev_new_innovations: Set[int] = set()
        self._structural_events = 0

        # selection-tie instrumentation (measurement only)
        self.arbitrary_selection_fraction = 0.0
        self.cut_ambiguous_species_fraction = 0.0

    # -- helpers -------------------------------------------------------------- #
    def _key(self) -> int:
        k = self._next_genome_key
        self._next_genome_key += 1
        return k

    def _new_genome(self) -> Genome:
        g = Genome.new_minimal(self._key(), self.cfg, self.tracker, self.rng)
        g.birth_generation = 0
        return g

    # -- speciation ------------------------------------------------------------ #
    def speciate(self) -> None:
        cfg = self.cfg
        # keep representatives, empty membership
        for s in self.species:
            s.members = []
        unspeciated = list(self.genomes)

        for g in unspeciated:
            best_s, best_d = None, float("inf")
            for s in self.species:
                d = Genome.distance(g, s.representative, cfg)
                if d < best_d:
                    best_d, best_s = d, s
            if best_s is not None and best_d < self.compat_threshold:
                best_s.members.append(g)
                g.species_id = best_s.key
            else:
                s = Species(key=self._next_species_key, representative=g.copy(),
                            members=[g], created=self.generation,
                            last_improved=self.generation)
                self._next_species_key += 1
                self.species.append(s)
                g.species_id = s.key

        self.species = [s for s in self.species if s.members]
        for s in self.species:
            # new representative = member closest to the old one (standard)
            s.representative = min(
                s.members, key=lambda m: Genome.distance(m, s.representative, cfg)
            ).copy()

        if cfg.dynamic_compat:
            if len(self.species) < cfg.target_species:
                self.compat_threshold = max(cfg.compat_min,
                                            self.compat_threshold - cfg.compat_adjust)
            elif len(self.species) > cfg.target_species:
                self.compat_threshold = min(cfg.compat_max,
                                            self.compat_threshold + cfg.compat_adjust)

    # -- reproduction ----------------------------------------------------------- #
    def _cull_stagnant(self) -> None:
        cfg = self.cfg
        for s in self.species:
            best = max(m.fitness for m in s.members)
            if best > s.best_fitness:
                s.best_fitness = best
                s.last_improved = self.generation
        ranked = sorted(self.species, key=lambda s: s.best_fitness, reverse=True)
        keep: List[Species] = []
        for i, s in enumerate(ranked):
            stagnant = (self.generation - s.last_improved) >= cfg.stagnation_generations
            if stagnant and i >= cfg.species_elitism and len(ranked) - len(keep) > 1:
                continue
            keep.append(s)
        if keep:
            self.species = [s for s in self.species if s in keep]

    def _allocate_offspring(self) -> None:
        """Explicit fitness sharing -> proportional offspring allocation."""
        cfg = self.cfg
        all_fit = [g.fitness for g in self.genomes]
        fmin, fmax = min(all_fit), max(all_fit)
        span = (fmax - fmin) if fmax > fmin else 1.0

        for s in self.species:
            n = len(s.members)
            s.adjusted_fitness_sum = sum(
                ((g.fitness - fmin) / span + 1e-3) / n for g in s.members
            )
            for g in s.members:
                g.adjusted_fitness = ((g.fitness - fmin) / span + 1e-3) / n

        total = sum(s.adjusted_fitness_sum for s in self.species)
        if total <= 0:
            share = cfg.pop_size / max(1, len(self.species))
            for s in self.species:
                s.offspring = int(share)
        else:
            for s in self.species:
                s.offspring = int(round(cfg.pop_size * s.adjusted_fitness_sum / total))
        for s in self.species:
            s.offspring = max(s.offspring, cfg.min_species_size)

        # fix rounding drift deterministically
        diff = cfg.pop_size - sum(s.offspring for s in self.species)
        order = sorted(self.species, key=lambda s: (-s.adjusted_fitness_sum, s.key))
        i = 0
        while diff != 0 and order:
            s = order[i % len(order)]
            if diff > 0:
                s.offspring += 1
                diff -= 1
            elif s.offspring > cfg.min_species_size:
                s.offspring -= 1
                diff += 1
            i += 1
            if i > 10000:
                break

    def reproduce(self) -> float:
        """Create the next generation.  Returns the selection differential."""
        cfg = self.cfg
        self._cull_stagnant()
        if not self.species:                      # total collapse -> restart safely
            self.species = [Species(self._next_species_key, self.genomes[0].copy(),
                                    list(self.genomes), self.generation, self.generation)]
            self._next_species_key += 1
        self._allocate_offspring()

        pop_mean = sum(g.fitness for g in self.genomes) / len(self.genomes)
        parent_fitnesses: List[float] = []

        # selection-tie instrumentation (measurement only)
        n_arbitrary = 0
        n_ambiguous_species = 0
        n_species_considered = 0

        self.tracker.advance_generation()
        self._structural_events = 0
        new_genomes: List[Genome] = []

        for s in self.species:
            members = sorted(s.members, key=lambda g: (-g.fitness, g.key))
            n_survivors = max(1, int(round(len(members) * cfg.survival_threshold)))
            pool = members[:n_survivors]

            # -- measurement only; does not affect reproduction --------------- #
            # Truncation selection reads ranks, not magnitudes. It only becomes
            # arbitrary when the cut falls strictly inside a tied group: then
            # who survives is decided by genome key, not by fitness. Count how
            # many individuals had their fate decided that way.
            n_species_considered += 1
            if 0 < n_survivors < len(members):
                boundary = members[n_survivors - 1].fitness
                if members[n_survivors].fitness == boundary:
                    tied = [m for m in members if m.fitness == boundary]
                    n_arbitrary += len(tied)
                    n_ambiguous_species += 1
            # ------------------------------------------------------------------ #

            n = s.offspring
            # elitism
            for e in range(min(cfg.elitism, len(members), n)):
                elite = members[e].copy(self._key())
                elite.birth_generation = self.generation + 1
                new_genomes.append(elite)
                n -= 1

            for _ in range(max(0, n)):
                p1 = pool[self.rng.randrange(len(pool))]
                parent_fitnesses.append(p1.fitness)
                if len(pool) > 1 and self.rng.random() < cfg.crossover_prob:
                    if (len(self.species) > 1
                            and self.rng.random() < cfg.interspecies_mating_prob):
                        other = self.species[self.rng.randrange(len(self.species))]
                        p2 = other.members[self.rng.randrange(len(other.members))]
                    else:
                        p2 = pool[self.rng.randrange(len(pool))]
                    parent_fitnesses.append(p2.fitness)
                    child = Genome.crossover(p1, p2, self._key(), cfg, self.rng)
                else:
                    child = p1.copy(self._key())
                child.birth_generation = self.generation + 1
                child.novel_innovations = []
                self._structural_events += child.mutate(cfg, self.tracker, self.rng)
                new_genomes.append(child)

        # size correction
        while len(new_genomes) > cfg.pop_size:
            new_genomes.pop()
        while len(new_genomes) < cfg.pop_size:
            src = self.genomes[self.rng.randrange(len(self.genomes))]
            child = src.copy(self._key())
            child.birth_generation = self.generation + 1
            child.novel_innovations = []
            self._structural_events += child.mutate(cfg, self.tracker, self.rng)
            new_genomes.append(child)

        self.genomes = new_genomes
        self.generation += 1

        # Exposed for the generation record. See scripts/patch_population_ties.py
        # for what these mean and why they are the paper's mechanism test.
        self.arbitrary_selection_fraction = n_arbitrary / max(1, cfg.pop_size)
        self.cut_ambiguous_species_fraction = (
            n_ambiguous_species / max(1, n_species_considered))

        if parent_fitnesses:
            return sum(parent_fitnesses) / len(parent_fitnesses) - pop_mean
        return 0.0


# --------------------------------------------------------------------------- #
def run_evolution(
    cfg: NEATConfig,
    seed: int,
    evaluate: Callable[[Sequence[Genome]], Tuple[List[float], List[float], int]],
    generations: int,
    diversity_sample: int = 40,
    on_generation: Optional[Callable[[int, dict], None]] = None,
) -> Tuple[List[dict], Genome, Population]:
    """Run one complete evolutionary trial.

    `evaluate` receives the population and must return
    (training_fitnesses, true_objective_scores, n_reached_goal).
    Assigning the fitness is left to the caller so that reward sparsity lives
    entirely outside the evolutionary algorithm.
    """
    pop = Population(cfg, seed)
    metric_rng = random.Random(seed ^ 0x5EED)
    records: List[dict] = []
    best_genome: Optional[Genome] = None
    best_true = float("-inf")
    selection_differential = 0.0

    for gen in range(generations):
        fits, trues, n_reached = evaluate(pop.genomes)
        for g, f in zip(pop.genomes, fits):
            g.fitness = f

        pop.speciate()

        innovations_now: Set[int] = set()
        for g in pop.genomes:
            innovations_now |= innovation_set(g)
        prev_new = pop._prev_new_innovations
        new_now = innovations_now - pop._prev_innovations

        rec = generation_record(
            generation=gen,
            genomes=pop.genomes,
            true_scores=trues,
            species_sizes=[len(s.members) for s in pop.species],
            species_ages=[s.age(gen) for s in pop.species],
            innovations_now=innovations_now,
            innovations_prev=pop._prev_innovations,
            innovations_prev_new=prev_new,
            structural_events=pop._structural_events,
            compat_threshold=pop.compat_threshold,
            diversity=mean_pairwise_distance(pop.genomes, cfg, metric_rng, diversity_sample),
            selection_differential=selection_differential,
            reached_goal_count=n_reached,
        )
        # mechanism metrics from the previous generation's reproduction step
        rec["arbitrary_selection_fraction"] = pop.arbitrary_selection_fraction
        rec["cut_ambiguous_species_fraction"] = pop.cut_ambiguous_species_fraction
        records.append(rec)
        if on_generation is not None:
            on_generation(gen, rec)

        i_best = max(range(len(trues)), key=lambda i: (trues[i], fits[i]))
        if trues[i_best] > best_true:
            best_true = trues[i_best]
            best_genome = pop.genomes[i_best].copy()

        pop._prev_innovations = innovations_now
        pop._prev_new_innovations = new_now

        if gen < generations - 1:
            selection_differential = pop.reproduce()

    return records, best_genome, pop
