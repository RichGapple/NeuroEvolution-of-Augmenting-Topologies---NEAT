"""NEAT reward-sparsity study.

Package layout mirrors the roadmap stages:
    config      experiment configuration + reproducibility hashing
    genome      genome, innovation tracking, mutation, crossover   (Stage 1)
    network     phenotype compilation                              (Stage 1)
    population  speciation, reproduction, instrumented loop        (Stage 1)
    env         deterministic 2D navigation task                   (Stage 2)
    reward      eta-parameterised reward sparsity                  (Stage 3)
    runner      run/sweep orchestration + logging                  (Stage 4)
    metrics     per-generation measurements                        (Stage 4/5)
    ablation    functional vs non-functional topology              (Stage 6)
    stats       distribution-aware statistics                      (Stage 9)
    plots       figures
    analysis    loading/reshaping raw outputs
"""

__version__ = "1.0.0"

from .config import (ExperimentConfig, NEATConfig, EnvConfig, SparsityConfig,
                     pilot_config, core_config)

__all__ = ["ExperimentConfig", "NEATConfig", "EnvConfig", "SparsityConfig",
           "pilot_config", "core_config", "__version__"]
