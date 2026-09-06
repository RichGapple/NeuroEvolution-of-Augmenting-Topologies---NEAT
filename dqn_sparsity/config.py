"""Configuration for the DQN arm of the reward-sparsity study.

Design rule
-----------
This module owns *only* the learner.  The environment and the reward come from
`neat_sparsity` unchanged -- they are imported, never reimplemented.  That is
what makes the NEAT/DQN comparison legitimate: both arms are driven by the same
`EnvConfig` and the same `SparsityConfig`, so a difference between arms cannot
be an artefact of two slightly different mazes or two slightly different
quantisers.

As in the NEAT arm, `config_hash()` fingerprints everything that could change a
result.  Runs with different hashes must never be pooled.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from neat_sparsity.config import EnvConfig, SparsityConfig


# --------------------------------------------------------------------------- #
@dataclass
class DQNConfig:
    """Hyperparameters of the learner.

    Tuning policy (state this in the paper): these are tuned ONCE at eta = 0
    and then frozen for the whole eta grid.  Re-tuning per condition would make
    the sweep a comparison of tuning effort rather than of reward sparsity.
    """

    # -- architecture --------------------------------------------------------- #
    hidden: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "relu"                 # relu | tanh

    # -- optimisation --------------------------------------------------------- #
    lr: float = 1e-3
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    grad_clip: float = 10.0
    huber_delta: float = 1.0                 # <=0 disables (plain MSE)

    # Learner-side preconditioning, NOT part of the reward definition.
    # At eta = 0 a single forward step earns ~0.012 of reward (2 units of a
    # 113-unit journey, quantised into 256 levels). With Huber delta = 1.0
    # every target sits deep in the quadratic regime, gradients are of order
    # 1e-2, and the network converges to predicting zero everywhere: td_loss
    # collapses to ~3e-5 and q_spread to ~0.004 while performance stays at
    # random-policy level. Multiplying the reward by a positive constant leaves
    # the optimal policy unchanged and fixes this. Measured effect at eta = 0:
    # final true objective 0.139 -> 0.697.
    # Reporting is always in unscaled true-objective units.
    reward_scale: float = 10.0

    # "min"     -- credit tracks best-ever progress (mirrors the NEAT fitness
    #              literally). Monotone ratchet: only a NEW closest approach
    #              pays, so only ~9% of transitions carry reward even at eta=0,
    #              and none are ever negative.
    # "current" -- credit tracks current distance. Still sums to exactly the
    #              same episode total (the terminal correction guarantees it),
    #              but pays on ~13% of transitions and can go negative when the
    #              agent moves away. Strictly more informative per step at
    #              identical total information.
    credit_basis: str = "min"

    # -- Q-learning ----------------------------------------------------------- #
    # 0.99, not 1.0. gamma = 1.0 makes the undiscounted return exactly equal
    # the NEAT fitness, which is elegant, but it also removes the contraction
    # property of the Bellman operator: the targets have no fixed point and the
    # value function drifts. Measured on this maze, gamma = 1.0 scores at
    # random-policy level. The undiscounted return still equals the fitness
    # exactly (10_dqn_check.py proves it); the agent simply optimises a
    # discounted proxy of it. Say so in the methods section.
    gamma: float = 0.99
    double_dqn: bool = True
    target_update_every: int = 1000          # in environment steps

    # -- replay --------------------------------------------------------------- #
    buffer_size: int = 100_000
    batch_size: int = 64
    warmup_steps: int = 2_000                # random actions, no learning
    train_every: int = 4                     # env steps between gradient steps

    # -- exploration ---------------------------------------------------------- #
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.20             # fraction of the budget spent decaying

    # -- budget --------------------------------------------------------------- #
    # NEAT upper bound = generations * pop_size * max_steps = 100*100*150 = 1.5e6.
    # Giving DQN the same number is deliberately generous, because NEAT episodes
    # terminate early on success and so NEAT actually consumes fewer.  Being
    # visibly generous to the baseline is what makes a crossover result credible.
    budget_steps: int = 1_500_000
    eval_points: int = 100                   # matches NEAT's 100 generations
    eval_episodes: int = 1                   # env is deterministic -> 1 suffices

    def eps_at(self, step: int) -> float:
        decay = max(1, int(self.eps_decay_frac * self.budget_steps))
        if step >= decay:
            return self.eps_end
        frac = step / decay
        return self.eps_start + frac * (self.eps_end - self.eps_start)


# --------------------------------------------------------------------------- #
@dataclass
class DQNExperimentConfig:
    name: str = "dqn_sweep"
    env: EnvConfig = field(default_factory=EnvConfig)
    sparsity: SparsityConfig = field(default_factory=SparsityConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)

    etas: List[float] = field(default_factory=lambda:
                              [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0])
    seeds: List[int] = field(default_factory=lambda: list(range(1000, 1030)))

    def config_hash(self) -> str:
        """Fingerprint everything that could change a result.

        `seeds` and `name` are excluded on purpose: they identify which runs you
        did, not what a run means.  Everything else is in.
        """
        payload = {
            "env": asdict(self.env),
            "sparsity": asdict(self.sparsity),
            "dqn": asdict(self.dqn),
            "etas": self.etas,
            "arm": "dqn",
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def budget_note(self) -> str:
        n = self.env.max_steps
        return (f"budget {self.dqn.budget_steps:,} env steps "
                f"(NEAT upper bound for 100 gen x 100 pop x {n} steps "
                f"= {100 * 100 * n:,})")


# --------------------------------------------------------------------------- #
def core_config() -> DQNExperimentConfig:
    """The frozen default.  Change this and the hash changes and nothing pools."""
    return DQNExperimentConfig()


def matched_to_neat(neat_cfg, budget_steps: Optional[int] = None
                    ) -> DQNExperimentConfig:
    """Build a DQN config that inherits env + sparsity + eta grid from a NEAT config.

    Use this rather than hand-copying values.  If the NEAT side ever changes the
    maze or the quantiser, this propagates the change instead of silently
    letting the two arms drift apart.
    """
    cfg = DQNExperimentConfig()
    cfg.env = neat_cfg.env
    cfg.sparsity = neat_cfg.sparsity
    cfg.etas = list(neat_cfg.etas)
    cfg.seeds = list(neat_cfg.seeds)
    if budget_steps is None:
        budget_steps = (neat_cfg.generations
                        * neat_cfg.neat.pop_size
                        * neat_cfg.env.max_steps)
    cfg.dqn.budget_steps = int(budget_steps)
    return cfg
