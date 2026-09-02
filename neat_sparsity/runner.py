"""Run orchestration.

One *run* = one (eta, seed) pair.  Each run writes:
    runs/eta=<eta>/seed=<seed>/generations.csv   longitudinal record
    runs/eta=<eta>/seed=<seed>/run.json          manifest + summary
    runs/eta=<eta>/seed=<seed>/champion.json     best genome (for ablation)

Nothing is aggregated at write time -- the analysis stage reads the raw files.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

from .config import ExperimentConfig
from .env import NavEnv
from .genome import Genome, ConnGene, NodeGene
from .metrics import genome_topology
from .network import build_network
from .population import run_evolution
from .reward import RewardModel, true_objective


# --------------------------------------------------------------------------- #
# genome (de)serialisation
# --------------------------------------------------------------------------- #
def genome_to_dict(g: Genome) -> dict:
    return {
        "key": g.key,
        "fitness": g.fitness,
        "birth_generation": g.birth_generation,
        "nodes": [{"key": n.key, "type": n.type, "bias": n.bias}
                  for n in sorted(g.nodes.values(), key=lambda n: n.key)],
        "conns": [{"in": c.key[0], "out": c.key[1], "weight": c.weight,
                   "enabled": c.enabled, "innovation": c.innovation}
                  for c in sorted(g.conns.values(), key=lambda c: c.innovation)],
    }


def genome_from_dict(d: dict) -> Genome:
    g = Genome(d["key"])
    g.fitness = d.get("fitness", 0.0)
    g.birth_generation = d.get("birth_generation", 0)
    for n in d["nodes"]:
        g.nodes[n["key"]] = NodeGene(n["key"], n["type"], n["bias"])
    for c in d["conns"]:
        key = (c["in"], c["out"])
        g.conns[key] = ConnGene(key, c["weight"], c["enabled"], c["innovation"])
    return g


# --------------------------------------------------------------------------- #
def make_evaluator(cfg: ExperimentConfig, eta: float, seed: int):
    """Build the evaluation closure for one run."""
    env = NavEnv(cfg.env)
    reward_rng = random.Random((seed * 7919) ^ int(eta * 1e6) ^ 0xBEEF)
    model = RewardModel(cfg.sparsity, eta, reward_rng)

    def evaluate(genomes) -> Tuple[List[float], List[float], int]:
        fits, trues, reached = [], [], 0
        for g in genomes:
            net = build_network(g, cfg.neat)
            traj = env.rollout(net)
            fits.append(model.fitness(traj))
            trues.append(true_objective(traj, cfg.sparsity))
            reached += int(traj.reached_goal)
        return fits, trues, reached

    return evaluate, env, model


def evaluate_genome(g: Genome, cfg: ExperimentConfig) -> Tuple[float, bool, int]:
    """Score a single genome on the sparsity-independent objective."""
    env = NavEnv(cfg.env)
    traj = env.rollout(build_network(g, cfg.neat))
    return true_objective(traj, cfg.sparsity), traj.reached_goal, traj.steps_taken


# --------------------------------------------------------------------------- #
def run_single(cfg: ExperimentConfig, eta: float, seed: int, outdir: str,
               verbose: bool = False) -> dict:
    t0 = time.time()
    evaluate, env, model = make_evaluator(cfg, eta, seed)

    def cb(gen: int, rec: dict) -> None:
        if verbose and (gen % 10 == 0 or gen == cfg.generations - 1):
            print(f"    gen {gen:4d}  true_best={rec['true_best']:.3f} "
                  f"succ={rec['success_rate']:.2f} nodes={rec['mean_nodes']:.1f} "
                  f"conns={rec['mean_connections']:.1f} species={rec['n_species']}",
                  flush=True)

    records, champion, pop = run_evolution(
        cfg.neat, seed, evaluate, cfg.generations,
        diversity_sample=cfg.diversity_sample, on_generation=cb,
    )

    run_dir = os.path.join(outdir, f"eta={eta:g}", f"seed={seed}")
    os.makedirs(run_dir, exist_ok=True)

    fieldnames = list(records[0].keys())
    with open(os.path.join(run_dir, "generations.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow(r)

    with open(os.path.join(run_dir, "champion.json"), "w") as fh:
        json.dump(genome_to_dict(champion), fh)

    champ_true, champ_reached, champ_steps = evaluate_genome(champion, cfg)
    topo = genome_topology(champion)

    # generations-to-solution: first generation with a goal-reaching individual
    gen_solved = next((r["generation"] for r in records if r["success_rate"] > 0), None)
    gen_solved_50 = next((r["generation"] for r in records if r["success_rate"] >= 0.5), None)

    summary = {
        "eta": eta,
        "seed": seed,
        "config_hash": cfg.config_hash(),
        "sparsity_mode": cfg.sparsity.mode,
        "levels": cfg.sparsity.levels(eta),
        "generations": cfg.generations,
        "wallclock_s": round(time.time() - t0, 2),
        # performance (sparsity-independent)
        "final_true_best": records[-1]["true_best"],
        "final_true_mean": records[-1]["true_mean"],
        "best_true_ever": max(r["true_best"] for r in records),
        "final_success_rate": records[-1]["success_rate"],
        "ever_solved": bool(gen_solved is not None),
        "gen_first_solution": gen_solved,
        "gen_success_50pct": gen_solved_50,
        "champion_true": champ_true,
        "champion_reached": champ_reached,
        "champion_steps": champ_steps,
        # topology
        "final_mean_nodes": records[-1]["mean_nodes"],
        "final_mean_connections": records[-1]["mean_connections"],
        "final_mean_density": records[-1]["mean_density"],
        "final_mean_depth": records[-1]["mean_depth"],
        "champion_nodes": topo["nodes"],
        "champion_hidden": topo["hidden_nodes"],
        "champion_connections": topo["connections"],
        "champion_enabled_connections": topo["enabled_connections"],
        # innovation
        "total_structural_events": sum(r["structural_events"] for r in records),
        "total_new_innovations": sum(r["new_innovations"] for r in records),
        "mean_tir": _nanmean([r["tir"] for r in records]),
        # dynamics
        "final_n_species": records[-1]["n_species"],
        "mean_n_species": sum(r["n_species"] for r in records) / len(records),
        "mean_diversity": sum(r["genomic_diversity"] for r in records) / len(records),
        "final_diversity": records[-1]["genomic_diversity"],
        "mean_distinct_fitness": sum(r["n_distinct_fitness"] for r in records) / len(records),
        "mean_fitness_differentiation":
            sum(r["fitness_differentiation"] for r in records) / len(records),
        "mean_selection_entropy":
            sum(r["selection_entropy_norm"] for r in records) / len(records),
        "mean_fitness_cv": sum(r["fitness_cv"] for r in records) / len(records),
        "mean_selection_differential":
            sum(r["selection_differential"] for r in records) / len(records),
    }
    with open(os.path.join(run_dir, "run.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def _nanmean(xs: Sequence[float]) -> float:
    vals = [x for x in xs if x == x]      # drop NaN
    return sum(vals) / len(vals) if vals else float("nan")


# --------------------------------------------------------------------------- #
def _worker(args):
    cfg_dict, eta, seed, outdir = args
    cfg = ExperimentConfig.load(cfg_dict) if isinstance(cfg_dict, str) else cfg_dict
    return run_single(cfg, eta, seed, outdir, verbose=False)


def run_sweep(cfg: ExperimentConfig, outdir: str, workers: int = 1,
              verbose: bool = True) -> List[dict]:
    os.makedirs(outdir, exist_ok=True)
    cfg_path = os.path.join(outdir, "config.json")
    cfg.save(cfg_path)

    jobs = [(cfg_path, eta, seed, outdir) for eta in cfg.etas for seed in cfg.seeds]
    results: List[dict] = []

    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_worker, jobs), 1):
                results.append(res)
                if verbose:
                    print(f"[{i}/{len(jobs)}] eta={res['eta']:g} seed={res['seed']} "
                          f"true_best={res['final_true_best']:.3f} "
                          f"nodes={res['final_mean_nodes']:.1f} "
                          f"({res['wallclock_s']}s)", flush=True)
    else:
        for i, job in enumerate(jobs, 1):
            res = _worker(job)
            results.append(res)
            if verbose:
                print(f"[{i}/{len(jobs)}] eta={res['eta']:g} seed={res['seed']} "
                      f"true_best={res['final_true_best']:.3f} "
                      f"nodes={res['final_mean_nodes']:.1f} "
                      f"({res['wallclock_s']}s)", flush=True)

    results.sort(key=lambda r: (r["eta"], r["seed"]))
    with open(os.path.join(outdir, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    return results
