"""Training loop, evaluation schedule, logging and the parallel sweep.

Logging schema
--------------
Deliberately mirrors the NEAT arm so the two can be loaded by the same code:

    outdir/eta=<x>/seed=<n>/blocks.csv     one row per evaluation block
    outdir/eta=<x>/seed=<n>/run.json       one summary dict per run
    outdir/summary.csv                     all runs concatenated

A *block* is the DQN counterpart of a NEAT generation.  The budget is split into
`eval_points` (default 100) equal blocks, so a DQN run produces 100 log rows just
as a NEAT run produces 100 generations, and the trajectory plots line up on a
common x-axis of "fraction of budget consumed".

Everything reported is in true-objective (eta = 0) units, exactly as on the NEAT
side.  Training reward is logged too, but only as a diagnostic -- never as a
performance measure.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from typing import Callable, List, Optional, Sequence

import numpy as np

from neat_sparsity.env import Trajectory
from .agent import DQNAgent
from .config import DQNExperimentConfig
from .shaping import StepwiseReward, report_true_objective
from .step_env import StepNavEnv
from . import mazes
from .sysinfo import peak_rss_mb


# --------------------------------------------------------------------------- #
def evaluate_greedy(env: StepNavEnv, agent: DQNAgent, cfg,
                    n_episodes: int = 1) -> dict:
    """Run the greedy policy and score it on the sparsity-independent objective."""
    trues, reached, steps, states = [], [], [], []
    for _ in range(n_episodes):
        obs = env.reset()
        while not env.done:
            states.append(obs)
            a = agent.act(obs, greedy=True)
            obs, _, _ = env.step(a)
        traj = env.trajectory()
        trues.append(report_true_objective(traj, cfg.sparsity))
        reached.append(float(traj.reached_goal))
        steps.append(traj.steps_taken)
    return {
        "true_best": float(np.max(trues)),
        "true_mean": float(np.mean(trues)),
        "success_rate": float(np.mean(reached)),
        "eval_steps": float(np.mean(steps)),
        "states": np.asarray(states, dtype=np.float32),
    }


# --------------------------------------------------------------------------- #
def run_single(cfg: DQNExperimentConfig, eta: float, seed: int, outdir: str,
               shaping_mode: str = "telescoping",
               verbose: bool = False,
               on_block: Optional[Callable[[int, dict], None]] = None) -> dict:
    t0 = time.time()
    rng = random.Random(seed)
    np_seed = seed

    env = StepNavEnv(cfg.env)
    eval_env = StepNavEnv(cfg.env)
    agent = DQNAgent(env.obs_dim, env.n_actions, cfg.dqn, seed=np_seed)
    shaper = StepwiseReward(cfg.sparsity, eta, mode=shaping_mode,
                            credit_basis=cfg.dqn.credit_basis, rng=rng)

    budget = cfg.dqn.budget_steps
    block_len = max(1, budget // cfg.dqn.eval_points)

    records: List[dict] = []
    obs = env.reset()
    shaper.reset(env.d0)

    episodes = 0
    ep_return = 0.0
    ep_returns_block: List[float] = []
    zero_frac_block: List[float] = []
    corrections_block: List[float] = []
    first_solution_step: Optional[int] = None
    best_true_ever = 0.0

    for step in range(1, budget + 1):
        a = agent.act(obs)
        nobs, done, info = env.step(a)
        traj = env.trajectory() if done else None
        r = shaper.step_reward(info, done, traj_if_done=traj)

        # `done` from timeout is not a true terminal state for bootstrapping;
        # only goal-reaching truncates the value backup.
        # Timeout is NOT treated as terminal. The observation vector has no
        # time channel, so bootstrapping to zero at step 150 would teach
        # contradictory targets for states that look identical early and late.
        # Measured: doing so drops the final score from 0.36 to 0.11.
        terminal = bool(info["reached"])
        # reward_scale is optimiser preconditioning; see DQNConfig. Everything
        # logged and reported below is unscaled.
        agent.observe(obs, a, r * cfg.dqn.reward_scale, nobs, terminal)
        ep_return += r
        obs = nobs

        if done:
            episodes += 1
            ep_returns_block.append(ep_return)
            zero_frac_block.append(shaper.zero_reward_fraction)
            corrections_block.append(shaper.terminal_correction)
            if traj.reached_goal and first_solution_step is None:
                first_solution_step = step
            ep_return = 0.0
            obs = env.reset()
            shaper.reset(env.d0)

        if step % block_len == 0 or step == budget:
            block = len(records)
            ev = evaluate_greedy(eval_env, agent, cfg, cfg.dqn.eval_episodes)
            tstats = agent.drain_train_stats()
            best_true_ever = max(best_true_ever, ev["true_best"])
            rec = {
                "block": block,
                "env_steps": step,
                "budget_frac": round(step / budget, 6),
                "episodes": episodes,
                "epsilon": round(cfg.dqn.eps_at(step), 6),
                # performance, sparsity-independent units
                "true_best": ev["true_best"],
                "true_mean": ev["true_mean"],
                "success_rate": ev["success_rate"],
                "eval_steps": ev["eval_steps"],
                # learning diagnostics
                "td_loss": tstats["td_loss"],
                "grad_norm": tstats["grad_norm"],
                "n_updates": agent.updates,
                "q_spread": agent.q_spread_on(ev["states"]),
                # sparsity diagnostics (the manipulation check for this arm)
                "train_return_mean": float(np.mean(ep_returns_block))
                    if ep_returns_block else float("nan"),
                "zero_reward_fraction": agent.buffer.zero_reward_fraction(),
                "distinct_rewards": agent.buffer.distinct_rewards(),
                "reward_entropy": agent.buffer.reward_entropy(),
                "terminal_correction_mean": float(np.mean(corrections_block))
                    if corrections_block else float("nan"),
            }
            records.append(rec)
            ep_returns_block.clear()
            zero_frac_block.clear()
            corrections_block.clear()
            if on_block is not None:
                on_block(block, rec)
            if verbose and (block % 10 == 0 or step == budget):
                print(f"    block {block:3d}  steps={step:>8,}  "
                      f"true_best={rec['true_best']:.3f}  "
                      f"loss={rec['td_loss']:.4f}  eps={rec['epsilon']:.3f}  "
                      f"zero_r={rec['zero_reward_fraction']:.2f}", flush=True)

    run_dir = os.path.join(outdir, f"eta={eta:g}", f"seed={seed}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "blocks.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        w.writeheader()
        for r in records:
            w.writerow(r)
    np.savez_compressed(os.path.join(run_dir, "policy.npz"),
                        *agent.get_weights())

    last10 = records[-10:]

    summary = {
        "arm": "dqn",
        "eta": eta,
        "seed": seed,
        "config_hash": cfg.config_hash(),
        "sparsity_mode": cfg.sparsity.mode,
        "shaping_mode": shaping_mode,
        "maze": mazes.identify(cfg.env),
        "credit_basis": cfg.dqn.credit_basis,
        "gamma": cfg.dqn.gamma,
        "reward_scale": cfg.dqn.reward_scale,
        "levels": cfg.sparsity.levels(eta),
        "wallclock_s": round(time.time() - t0, 2),
        # -- performance, directly comparable with the NEAT arm --------------- #
        "final_true_best": records[-1]["true_best"],
        "final_true_mean": records[-1]["true_mean"],
        "best_true_ever": best_true_ever,
        "final_success_rate": records[-1]["success_rate"],
        "success_rate_last10": float(np.mean([r["success_rate"] for r in last10])),
        "ever_solved": bool(first_solution_step is not None),
        "first_solution_step": first_solution_step,
        "first_solution_block": next(
            (r["block"] for r in records if r["success_rate"] > 0), None),
        # -- cost accounting --------------------------------------------------- #
        "env_steps": records[-1]["env_steps"],
        "episodes": episodes,
        "gradient_updates": agent.updates,
        "n_params": agent.q.n_params(),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        # -- learning diagnostics ---------------------------------------------- #
        "final_td_loss": records[-1]["td_loss"],
        "mean_td_loss": float(np.nanmean([r["td_loss"] for r in records])),
        "final_q_spread": records[-1]["q_spread"],
        "mean_q_spread": float(np.nanmean([r["q_spread"] for r in records])),
        # -- sparsity manipulation checks -------------------------------------- #
        "mean_zero_reward_fraction":
            float(np.nanmean([r["zero_reward_fraction"] for r in records])),
        "final_distinct_rewards": records[-1]["distinct_rewards"],
        "mean_reward_entropy":
            float(np.nanmean([r["reward_entropy"] for r in records])),
        "mean_terminal_correction":
            float(np.nanmean([r["terminal_correction_mean"] for r in records])),
    }
    with open(os.path.join(run_dir, "run.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


# --------------------------------------------------------------------------- #
def _worker(args):
    cfg, eta, seed, outdir, shaping_mode = args
    return run_single(cfg, eta, seed, outdir, shaping_mode=shaping_mode)


def run_sweep(cfg: DQNExperimentConfig, outdir: str, workers: int = 1,
              shaping_mode: str = "telescoping",
              etas: Optional[Sequence[float]] = None,
              seeds: Optional[Sequence[int]] = None) -> List[dict]:
    os.makedirs(outdir, exist_ok=True)
    etas = list(etas if etas is not None else cfg.etas)
    seeds = list(seeds if seeds is not None else cfg.seeds)
    jobs = [(cfg, e, s, outdir, shaping_mode) for e in etas for s in seeds]

    print(f"{len(jobs)} runs  |  config hash {cfg.config_hash()}  |  "
          f"{cfg.budget_note()}", flush=True)

    rows: List[dict] = []
    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            for i, row in enumerate(pool.imap_unordered(_worker, jobs), 1):
                rows.append(row)
                print(f"  [{i}/{len(jobs)}] eta={row['eta']:g} seed={row['seed']} "
                      f"true={row['final_true_best']:.3f} "
                      f"({row['wallclock_s']:.0f}s)", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            row = _worker(j)
            rows.append(row)
            print(f"  [{i}/{len(jobs)}] eta={row['eta']:g} seed={row['seed']} "
                  f"true={row['final_true_best']:.3f} "
                  f"({row['wallclock_s']:.0f}s)", flush=True)

    rows.sort(key=lambda r: (r["eta"], r["seed"]))
    with open(os.path.join(outdir, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(outdir, "config.json"), "w") as fh:
        json.dump({"config_hash": cfg.config_hash(),
                   "budget_steps": cfg.dqn.budget_steps,
                   "shaping_mode": shaping_mode,
        "maze": mazes.identify(cfg.env),
        "credit_basis": cfg.dqn.credit_basis,
        "gamma": cfg.dqn.gamma,
        "reward_scale": cfg.dqn.reward_scale,
                   "etas": etas, "seeds": seeds}, fh, indent=2)
    return rows


# --------------------------------------------------------------------------- #
def load_blocks(outdir: str):
    """Long-format table: one row per (eta, seed, block).  Mirrors
    neat_sparsity.analysis.load_generations."""
    import pandas as pd
    frames = []
    for eta_dir in sorted(os.listdir(outdir)):
        if not eta_dir.startswith("eta="):
            continue
        eta = float(eta_dir.split("=")[1])
        for seed_dir in sorted(os.listdir(os.path.join(outdir, eta_dir))):
            p = os.path.join(outdir, eta_dir, seed_dir, "blocks.csv")
            if not os.path.exists(p):
                continue
            df = pd.read_csv(p)
            df["eta"] = eta
            df["seed"] = int(seed_dir.split("=")[1])
            frames.append(df)
    if not frames:
        import pandas as pd
        return pd.DataFrame()
    import pandas as pd
    return pd.concat(frames, ignore_index=True)
