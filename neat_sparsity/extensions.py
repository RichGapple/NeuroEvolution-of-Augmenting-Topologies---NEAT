"""Extensions A and B (roadmap Section 21).

These answer a *different* question from the core study -- "can we mitigate the
effect?" rather than "what is the effect?" -- and they must not be run until
the baseline phenomenon is established.  `scripts/07_extensions.py` refuses to
run without a completed core sweep for exactly that reason.

Extension A -- novelty pressure
    Behaviour descriptor: the agent's final (x, y).  Novelty = mean distance to
    the k nearest neighbours among the current population plus an archive.
    Selection fitness = (1 - lambda) * normalised objective
                      +      lambda  * normalised novelty
    lambda = 1 is pure novelty search.  Note that novelty is computed from
    behaviour, not reward, so it is *unaffected by eta* -- which is the whole
    point: it supplies selection information that reward sparsity has removed.

Extension B -- adaptive parsimony
    A size penalty whose coefficient adapts each generation to hold mean genome
    size near a moving target, so that structural growth must "pay for itself"
    in objective terms.  Implemented as a selection-time penalty only; it never
    touches the mutation operators, so topological innovation is still allowed
    to happen -- only its retention is taxed.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from .genome import Genome


# --------------------------------------------------------------------------- #
# Extension A -- novelty
# --------------------------------------------------------------------------- #
class NoveltyArchive:
    def __init__(self, k: int = 15, add_threshold: float = 6.0,
                 max_size: int = 2000, min_adds_per_gen: int = 1) -> None:
        self.k = k
        self.add_threshold = add_threshold
        self.max_size = max_size
        self.min_adds_per_gen = min_adds_per_gen
        self.points: List[Tuple[float, float]] = []

    @staticmethod
    def _d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def novelty(self, b: Tuple[float, float],
                population: Sequence[Tuple[float, float]]) -> float:
        pool = list(population) + self.points
        if len(pool) <= 1:
            return 0.0
        ds = sorted(self._d(b, o) for o in pool)
        ds = ds[1:] if ds and ds[0] == 0.0 else ds     # drop self-match
        k = min(self.k, len(ds))
        return sum(ds[:k]) / k if k else 0.0

    def update(self, behaviours: Sequence[Tuple[float, float]],
               novelties: Sequence[float]) -> int:
        added = 0
        order = sorted(range(len(behaviours)), key=lambda i: -novelties[i])
        for rank, i in enumerate(order):
            if novelties[i] > self.add_threshold or rank < self.min_adds_per_gen:
                self.points.append(behaviours[i])
                added += 1
        if len(self.points) > self.max_size:
            self.points = self.points[-self.max_size:]
        return added


def _normalise(xs: Sequence[float]) -> List[float]:
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-12:
        return [0.0] * len(xs)
    return [(x - lo) / (hi - lo) for x in xs]


def blend_novelty(objective: Sequence[float], novelty: Sequence[float],
                  lam: float) -> List[float]:
    o = _normalise(objective)
    n = _normalise(novelty)
    return [(1.0 - lam) * a + lam * b for a, b in zip(o, n)]


# --------------------------------------------------------------------------- #
# Extension B -- adaptive parsimony
# --------------------------------------------------------------------------- #
class AdaptiveParsimony:
    """Penalty coefficient adapted to hold mean complexity near a target.

    target_size drifts upward only when the objective is improving, so growth
    is permitted exactly when it is being paid for.
    """

    def __init__(self, initial_target: Optional[float] = None,
                 coefficient: float = 0.0, lr: float = 0.02,
                 max_coefficient: float = 0.2,
                 growth_allowance: float = 0.05) -> None:
        self.target = initial_target
        self.c = coefficient
        self.lr = lr
        self.max_c = max_coefficient
        self.growth_allowance = growth_allowance
        self._best_so_far = float("-inf")

    def penalise(self, fitnesses: Sequence[float],
                 sizes: Sequence[int]) -> List[float]:
        if self.target is None:
            self.target = sum(sizes) / len(sizes)
        return [f - self.c * max(0.0, s - self.target)
                for f, s in zip(fitnesses, sizes)]

    def update(self, fitnesses: Sequence[float], sizes: Sequence[int]) -> None:
        mean_size = sum(sizes) / len(sizes)
        best = max(fitnesses)
        improving = best > self._best_so_far + 1e-9
        self._best_so_far = max(self._best_so_far, best)

        if improving:
            # growth has earned itself: let the target follow
            self.target = max(self.target, mean_size * (1.0 + self.growth_allowance))
            self.c = max(0.0, self.c - self.lr)
        elif mean_size > self.target:
            self.c = min(self.max_c, self.c + self.lr)
        else:
            self.c = max(0.0, self.c - self.lr * 0.5)

    def state(self) -> dict:
        return {"parsimony_c": self.c, "parsimony_target": self.target}


# --------------------------------------------------------------------------- #
def make_extended_evaluator(cfg, eta: float, seed: int, novelty_lambda: float = 0.0,
                            parsimony: bool = False):
    """Evaluator that optionally adds novelty pressure and/or size penalty."""
    import random

    from .env import NavEnv
    from .network import build_network
    from .reward import RewardModel, true_objective

    env = NavEnv(cfg.env)
    model = RewardModel(cfg.sparsity, eta,
                        random.Random((seed * 7919) ^ int(eta * 1e6) ^ 0xBEEF))
    archive = NoveltyArchive() if novelty_lambda > 0 else None
    parsi = AdaptiveParsimony() if parsimony else None
    log: List[dict] = []

    def evaluate(genomes):
        raw, trues, behaviours, reached = [], [], [], 0
        for g in genomes:
            traj = env.rollout(build_network(g, cfg.neat))
            raw.append(model.fitness(traj))
            trues.append(true_objective(traj, cfg.sparsity))
            behaviours.append((traj.final_x, traj.final_y))
            reached += int(traj.reached_goal)

        fits = list(raw)
        entry = {}
        if archive is not None:
            nov = [archive.novelty(b, behaviours) for b in behaviours]
            fits = blend_novelty(fits, nov, novelty_lambda)
            entry["archive_added"] = archive.update(behaviours, nov)
            entry["mean_novelty"] = sum(nov) / len(nov)
            entry["archive_size"] = len(archive.points)
        if parsi is not None:
            sizes = [len(g.nodes) + len(g.conns) for g in genomes]
            fits = parsi.penalise(fits, sizes)
            parsi.update(raw, sizes)
            entry.update(parsi.state())
        log.append(entry)
        return fits, trues, reached

    return evaluate, log
