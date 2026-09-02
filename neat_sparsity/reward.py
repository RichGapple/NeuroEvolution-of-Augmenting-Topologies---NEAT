"""Reward sparsity: the independent variable.

Why this definition
-------------------
A naive reading of "sparse reward" is *temporal*: pay out less often.  For an
evolutionary algorithm that reading is degenerate, because fitness is the sum
of an episode's rewards -- redistributing reward within an episode leaves the
sum, and therefore the selection signal, unchanged.  Sparsity has to reduce the
*information content* of the fitness, which is exactly hypothesis H1.

We therefore define eta as the coarseness of intermediate feedback:

    fitness(eta) = B * 1[goal reached] + w * C_eta(trajectory)

where C_eta is the intermediate ("partial credit") term and

    L(eta) = round(L_max ** (1 - eta))       L(0) = L_max,  L(1) = 1

is the number of distinguishable credit levels.

Invariants that make the comparison across eta legitimate:
  * The environment, its dynamics, and the optimal policy are identical.
  * Max attainable fitness is B + w for every eta (a successful agent is
    scored identically under every condition).
  * Min attainable fitness is 0 for every eta.
  * C_eta is monotone non-increasing in eta for a fixed trajectory, and
    C_1 = 1[goal reached].
The only thing that changes is how finely non-identical trajectories can be
told apart -- i.e. fitness differentiation.

Three modes are provided so that the headline result can be checked for
robustness against the operationalisation (roadmap Section 9 leaves the exact
definition to implementation time).
"""

from __future__ import annotations

import math
import random
from typing import Optional

from .config import SparsityConfig
from .env import Trajectory


def _progress(traj: Trajectory) -> float:
    """Best fractional progress toward the goal, in [0, 1]."""
    if traj.reached_goal:
        return 1.0
    if traj.initial_distance <= 0:
        return 1.0
    p = 1.0 - (traj.min_distance / traj.initial_distance)
    return min(1.0, max(0.0, p))


def _quantize(p: float, levels: int) -> float:
    if levels <= 1:
        return 1.0 if p >= 1.0 else 0.0
    q = math.floor(p * levels) / levels
    return min(1.0, max(0.0, q))


class RewardModel:
    """Turns a Trajectory into a scalar fitness at a given sparsity level."""

    def __init__(self, cfg: SparsityConfig, eta: float,
                 rng: Optional[random.Random] = None) -> None:
        self.cfg = cfg
        self.eta = float(eta)
        self.levels = cfg.levels(eta)
        self.rng = rng or random.Random(0)

    # -- credit terms ---------------------------------------------------------- #
    def _credit_quantized(self, traj: Trajectory) -> float:
        return _quantize(_progress(traj), self.levels)

    def _credit_checkpoint(self, traj: Trajectory) -> float:
        """Concentric rings around the goal, crossed in order."""
        if traj.reached_goal:
            return 1.0
        L = self.levels
        if L <= 1:
            return 0.0
        d0 = traj.initial_distance
        radii = [d0 * (1.0 - (k + 1) / L) for k in range(L)]   # decreasing
        reached = 0
        idx = 0
        for d in traj.distances:
            while idx < L and d <= radii[idx]:
                idx += 1
                reached = idx
        return reached / L

    def _credit_gated(self, traj: Trajectory) -> float:
        """Potential-based shaping delivered per step with probability 1-eta.

        Expectation equals the dense credit; increasing eta destroys information
        through sampling noise rather than through quantisation.
        """
        if traj.reached_goal:
            return 1.0            # keeps max fitness identical across eta
        if self.eta >= 1.0:
            return 0.0
        d0 = traj.initial_distance if traj.initial_distance > 0 else 1.0
        keep = 1.0 - self.eta
        total = 0.0
        ds = traj.distances
        for i in range(1, len(ds)):
            delta = (ds[i - 1] - ds[i]) / d0
            if self.rng.random() < keep:
                total += delta / keep          # importance weight -> unbiased
        return min(1.0, max(0.0, total))

    # -- public ---------------------------------------------------------------- #
    def fitness(self, traj: Trajectory) -> float:
        mode = self.cfg.mode
        if mode == "quantized":
            credit = self._credit_quantized(traj)
        elif mode == "checkpoint":
            credit = self._credit_checkpoint(traj)
        elif mode == "gated":
            credit = self._credit_gated(traj)
        else:
            raise ValueError(f"unknown sparsity mode: {mode!r}")

        f = self.cfg.progress_weight * credit
        if traj.reached_goal:
            f += self.cfg.success_bonus
        return f

    @property
    def max_fitness(self) -> float:
        return self.cfg.success_bonus + self.cfg.progress_weight


def true_objective(traj: Trajectory, cfg: SparsityConfig) -> float:
    """Sparsity-independent score used for *reporting* and for ablation.

    Results must never be reported in units that themselves depend on eta --
    otherwise "sparse runs score lower" is a tautology.  This is the eta = 0
    fitness and is identical for every condition.
    """
    f = cfg.progress_weight * _progress(traj)
    if traj.reached_goal:
        f += cfg.success_bonus
    return f
