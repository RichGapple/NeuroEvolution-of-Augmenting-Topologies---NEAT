"""Configuration objects.

Everything that can influence a run lives in a dataclass here, so that a run is
fully described by (config, seed).  `config_hash()` gives a short digest that is
written into every output file: if two runs have different hashes they are not
comparable, and the analysis scripts will refuse to pool them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import List, Tuple


# --------------------------------------------------------------------------- #
# NEAT
# --------------------------------------------------------------------------- #
@dataclass
class NEATConfig:
    pop_size: int = 100

    # --- I/O (must match the environment) ---------------------------------- #
    num_inputs: int = 8
    num_outputs: int = 3
    initial_connection: str = "full"          # "full" | "sparse"
    initial_sparse_fraction: float = 0.5

    # --- weights / biases --------------------------------------------------- #
    weight_init_std: float = 1.0
    weight_mutate_prob: float = 0.80
    weight_perturb_prob: float = 0.90         # else: full replacement
    weight_perturb_std: float = 0.50
    weight_replace_std: float = 1.00
    weight_clamp: float = 8.0

    bias_mutate_prob: float = 0.70
    bias_perturb_prob: float = 0.90
    bias_perturb_std: float = 0.30
    bias_replace_std: float = 1.00
    bias_clamp: float = 8.0

    # --- structural mutations ---------------------------------------------- #
    add_node_prob: float = 0.03
    add_conn_prob: float = 0.08
    toggle_enable_prob: float = 0.01
    delete_conn_prob: float = 0.00
    feedforward: bool = True                  # reject cycle-creating connections
    add_conn_tries: int = 20

    # --- reproduction ------------------------------------------------------- #
    crossover_prob: float = 0.75
    interspecies_mating_prob: float = 0.001
    inherit_disabled_prob: float = 0.75       # P(gene stays disabled if disabled in either parent)

    # --- speciation --------------------------------------------------------- #
    c_excess: float = 1.0
    c_disjoint: float = 1.0
    c_weight: float = 0.5
    compat_threshold: float = 3.0
    dynamic_compat: bool = True
    target_species: int = 10
    compat_adjust: float = 0.10
    compat_min: float = 0.50
    compat_max: float = 12.0

    stagnation_generations: int = 20
    species_elitism: int = 2                  # protect N best species from stagnation culling
    elitism: int = 2                          # copy N best genomes of a species unchanged
    survival_threshold: float = 0.30          # top fraction of a species allowed to reproduce
    min_species_size: int = 2

    # --- activations -------------------------------------------------------- #
    activation: str = "tanh"
    output_activation: str = "tanh"


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
@dataclass
class EnvConfig:
    """Deterministic continuous 2D navigation task.

    The world is a rectangle [0,W] x [0,H] containing axis-aligned rectangular
    obstacles.  NOTHING here may change between reward-sparsity conditions.
    """
    width: float = 100.0
    height: float = 100.0
    start: Tuple[float, float] = (10.0, 10.0)
    start_heading: float = 0.0                # radians
    goal: Tuple[float, float] = (90.0, 90.0)
    goal_radius: float = 6.0

    # (x0, y0, x1, y1)
    obstacles: List[Tuple[float, float, float, float]] = field(
        default_factory=lambda: [
            (30.0, 0.0, 40.0, 62.0),
            (60.0, 38.0, 70.0, 100.0),
            (0.0, 74.0, 22.0, 84.0),
            (78.0, 12.0, 100.0, 22.0),
        ]
    )

    max_steps: int = 150
    speed: float = 2.0                        # units per forward step
    turn_rate: float = 0.25                   # radians per turn step
    agent_radius: float = 1.5
    n_rangefinders: int = 5
    rangefinder_fov: float = 2.0944           # 120 degrees, total
    rangefinder_max: float = 40.0
    collision_ends_episode: bool = False      # False => agent just cannot move into walls

    def n_observations(self) -> int:
        # rangefinders + goal distance + sin/cos of relative bearing
        return self.n_rangefinders + 3


# --------------------------------------------------------------------------- #
# Reward sparsity  (the independent variable)
# --------------------------------------------------------------------------- #
@dataclass
class SparsityConfig:
    """eta in [0,1]: 0 = dense, 1 = terminal-only.

    mode:
      "quantized"  (default) -- intermediate feedback is delivered at a
                    resolution of L(eta) levels.  L(0)=levels_max (near
                    continuous), L(1)=1 (no partial credit at all).
      "gated"      -- potential-based shaping is delivered per step with
                    probability (1-eta); expectation is preserved, information
                    is destroyed by sampling noise.
      "checkpoint" -- L(eta) concentric checkpoints must be reached in order;
                    credit = (highest consecutive checkpoint)/L.

    In every mode the *maximum attainable fitness* and the environment dynamics
    are identical across eta.  Only how much information the fitness carries
    about intermediate behaviour changes.
    """
    mode: str = "quantized"
    levels_mode: str = "dyadic"    # "dyadic" (recommended) | "geometric"
    levels_max: int = 256          # L(0); a power of two for the dyadic ladder
    success_bonus: float = 1.0     # B: paid iff the goal is reached
    progress_weight: float = 1.0   # w: scales the intermediate signal
    fitness_floor: float = 1e-6    # keeps adjusted fitness strictly positive

    def levels(self, eta: float) -> int:
        """Number of distinguishable credit levels.  L(0)=levels_max, L(1)=1.

        The dyadic ladder L = 2**round(K(1-eta)), K = log2(levels_max), is the
        default because successive levels *divide* one another.  The induced
        partitions of the progress interval are therefore nested, which makes
        the quantised credit monotone non-increasing in eta for every single
        trajectory -- i.e. increasing eta can only ever destroy information,
        never rearrange it.  With levels_max = 256 the ladder lines up exactly
        with the nine-point eta grid: 256, 128, 64, 32, 16, 8, 4, 2, 1.
        """
        eta = min(max(float(eta), 0.0), 1.0)
        if self.levels_mode == "geometric":
            return max(1, int(round(self.levels_max ** (1.0 - eta))))
        import math as _m
        k = _m.log2(max(2, self.levels_max))
        return max(1, 2 ** int(k * (1.0 - eta) + 0.5))


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentConfig:
    name: str = "core_sweep"
    generations: int = 100
    etas: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    seeds: List[int] = field(default_factory=lambda: list(range(1000, 1030)))

    neat: NEATConfig = field(default_factory=NEATConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    sparsity: SparsityConfig = field(default_factory=SparsityConfig)

    # analysis
    diversity_sample: int = 40      # pairs sampled for mean genomic distance
    save_champion_every: int = 0    # 0 = only final champion
    ablation_reference_eta: float = 0.0   # objective used to score ablations

    def __post_init__(self) -> None:
        # The genome I/O size is dictated by the environment; enforce it here so
        # a mismatched config can never be run.
        self.neat.num_inputs = self.env.n_observations()
        self.neat.num_outputs = 3   # turn-left / turn-right / forward

    # ---- serialisation ----------------------------------------------------- #
    def to_dict(self) -> dict:
        return asdict(self)

    def config_hash(self, exclude_seeds: bool = True) -> str:
        d = self.to_dict()
        if exclude_seeds:
            d.pop("seeds", None)
            d.pop("name", None)
        blob = json.dumps(d, sort_keys=True, default=str).encode()
        return hashlib.sha1(blob).hexdigest()[:12]

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)

    @staticmethod
    def load(path: str) -> "ExperimentConfig":
        with open(path) as fh:
            d = json.load(fh)
        neat = NEATConfig(**d.pop("neat"))
        envd = d.pop("env")
        envd["start"] = tuple(envd["start"])
        envd["goal"] = tuple(envd["goal"])
        envd["obstacles"] = [tuple(o) for o in envd["obstacles"]]
        env = EnvConfig(**envd)
        sp = SparsityConfig(**d.pop("sparsity"))
        cfg = ExperimentConfig(neat=neat, env=env, sparsity=sp, **d)
        return cfg


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #
def pilot_config() -> ExperimentConfig:
    """Small + fast: used to measure runtime and between-seed variance."""
    cfg = ExperimentConfig(name="pilot")
    cfg.generations = 100
    cfg.neat.pop_size = 100
    cfg.env.max_steps = 150
    cfg.etas = [0.0, 0.5, 1.0]
    cfg.seeds = list(range(1000, 1006))
    return cfg


def core_config() -> ExperimentConfig:
    """The main sweep described in the roadmap (Stage 4)."""
    cfg = ExperimentConfig(name="core_sweep")
    cfg.generations = 100
    cfg.neat.pop_size = 100
    cfg.etas = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    cfg.seeds = list(range(1000, 1030))     # 30 independent seeds per condition
    return cfg
