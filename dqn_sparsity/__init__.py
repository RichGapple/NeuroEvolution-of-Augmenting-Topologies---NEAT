"""DQN arm of the reward-sparsity study.

Companion to `neat_sparsity`.  The environment (`NavEnv`) and the reward model
(`RewardModel`, `true_objective`) are imported from that package unchanged --
never reimplemented -- so that a difference between the two arms cannot be an
artefact of two slightly different tasks.

Layout
------
    config.py     learner hyperparameters + config hashing + budget matching
    step_env.py   reset/step wrapper over NavEnv, with a conformance guarantee
    shaping.py    per-step reward that provably sums to the NEAT fitness
    nets.py       numpy MLP, Adam, Huber (no framework dependency)
    agent.py      DQN: replay, target net, Double DQN, epsilon-greedy
    runner.py     training loop, evaluation blocks, logging, parallel sweep
    sysinfo.py    cross-platform process metrics (Windows-safe)
    mazes.py      maze A (deceptive) and B (non-deceptive)
    maze_analysis.py  computes deceptiveness, path length, feasibility
    plots.py      DQN and cross-arm figures
    compare.py    per-condition tests, crossover point, cost accounting

Run order
---------
    scripts/10_dqn_check.py       conformance + return equivalence  (must pass)
    scripts/11_dqn_calibrate.py   gates: does DQN work at all here?
    scripts/12_dqn_pilot.py       runtime + variance -> seed count
    scripts/13_dqn_sweep.py       the sweep
    scripts/14_dqn_analyze.py     stats, figures, report.md
    scripts/15_crossover.py       NEAT vs DQN  <- the paper's headline
    scripts/16_cost_report.py     honest compute accounting
"""

from .config import DQNConfig, DQNExperimentConfig, core_config, matched_to_neat
from .step_env import StepNavEnv, rollout_stepwise, rollout_monolithic
from .shaping import StepwiseReward, verify_equivalence, report_true_objective
from .agent import DQNAgent, ReplayBuffer
from .runner import run_single, run_sweep, load_blocks, evaluate_greedy

__all__ = [
    "DQNConfig", "DQNExperimentConfig", "core_config", "matched_to_neat",
    "StepNavEnv", "rollout_stepwise", "rollout_monolithic",
    "StepwiseReward", "verify_equivalence", "report_true_objective",
    "DQNAgent", "ReplayBuffer",
    "run_single", "run_sweep", "load_blocks", "evaluate_greedy",
]

__version__ = "1.0.0"
