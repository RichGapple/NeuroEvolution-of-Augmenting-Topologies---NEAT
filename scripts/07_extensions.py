#!/usr/bin/env python3
"""Stage 10 -- extensions.  Run ONLY after the core sweep has been analysed.

Extension A: does behavioural novelty pressure compensate for reduced
             reward-based selection pressure under sparse reward?
Extension B: can adaptive parsimony reduce non-functional structural growth
             without suppressing useful innovation?

Both are second research questions.  This script refuses to run unless a core
sweep exists in --baseline, because "can we mitigate the effect?" is only
meaningful once the effect has been characterised.

Usage
-----
    python scripts/07_extensions.py --baseline results/core_sweep \
        --extension novelty --lam 0.5 --etas 0.75 1.0 --seeds 15
    python scripts/07_extensions.py --baseline results/core_sweep \
        --extension parsimony --etas 0.0 0.5 1.0 --seeds 15
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neat_sparsity.config import ExperimentConfig, core_config
from neat_sparsity.extensions import make_extended_evaluator
from neat_sparsity.metrics import genome_topology
from neat_sparsity.population import run_evolution
from neat_sparsity.runner import genome_to_dict, evaluate_genome, _nanmean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="results/core_sweep")
    ap.add_argument("--out", default=None)
    ap.add_argument("--extension", choices=["novelty", "parsimony", "both"],
                    required=True)
    ap.add_argument("--lam", type=float, default=0.5,
                    help="novelty weight; 1.0 = pure novelty search")
    ap.add_argument("--etas", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--seed-start", type=int, default=3000)
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.baseline, "summary.csv")):
        raise SystemExit(
            f"no completed core sweep at {args.baseline}. Extensions are a second\n"
            "research question; establish the baseline phenomenon first "
            "(scripts 00 -> 06)."
        )
    if not os.path.exists(os.path.join(args.baseline, "analysis", "report.md")):
        raise SystemExit(
            f"{args.baseline} has runs but no analysis. Run scripts/06_analyze.py "
            "and read the report before extending."
        )

    cfg = ExperimentConfig.load(os.path.join(args.baseline, "config.json"))
    cfg.etas = args.etas
    cfg.seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    out = args.out or f"results/ext_{args.extension}"
    os.makedirs(out, exist_ok=True)
    cfg.save(os.path.join(out, "config.json"))

    use_nov = args.extension in ("novelty", "both")
    use_par = args.extension in ("parsimony", "both")
    lam = args.lam if use_nov else 0.0

    print(f"extension={args.extension}  lambda={lam}  "
          f"etas={cfg.etas}  seeds={len(cfg.seeds)}  "
          f"generations={cfg.generations}  pop={cfg.neat.pop_size}")
    print(f"baseline config hash {cfg.config_hash()} (identical to the core sweep)\n")

    rows = []
    for eta in cfg.etas:
        for seed in cfg.seeds:
            ev, log = make_extended_evaluator(cfg, eta, seed,
                                              novelty_lambda=lam, parsimony=use_par)
            recs, champ, _ = run_evolution(cfg.neat, seed, ev, cfg.generations,
                                           diversity_sample=cfg.diversity_sample)
            run_dir = os.path.join(out, f"eta={eta:g}", f"seed={seed}")
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "generations.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
                w.writeheader(); w.writerows(recs)
            with open(os.path.join(run_dir, "champion.json"), "w") as fh:
                json.dump(genome_to_dict(champ), fh)
            with open(os.path.join(run_dir, "extension_log.json"), "w") as fh:
                json.dump(log, fh)

            topo = genome_topology(champ)
            gen_solved = next((r["generation"] for r in recs if r["success_rate"] > 0), None)
            row = {
                "eta": eta, "seed": seed, "extension": args.extension,
                "novelty_lambda": lam,
                "config_hash": cfg.config_hash(),
                "sparsity_mode": cfg.sparsity.mode,
                "generations": cfg.generations,
                "wallclock_s": 0,
                "final_true_best": recs[-1]["true_best"],
                "final_true_mean": recs[-1]["true_mean"],
                "best_true_ever": max(r["true_best"] for r in recs),
                "final_success_rate": recs[-1]["success_rate"],
                "ever_solved": gen_solved is not None,
                "gen_first_solution": gen_solved,
                "gen_success_50pct": next(
                    (r["generation"] for r in recs if r["success_rate"] >= 0.5), None),
                "champion_true": evaluate_genome(champ, cfg)[0],
                "champion_reached": evaluate_genome(champ, cfg)[1],
                "champion_steps": evaluate_genome(champ, cfg)[2],
                "final_mean_nodes": recs[-1]["mean_nodes"],
                "final_mean_connections": recs[-1]["mean_connections"],
                "final_mean_density": recs[-1]["mean_density"],
                "final_mean_depth": recs[-1]["mean_depth"],
                "champion_nodes": topo["nodes"],
                "champion_hidden": topo["hidden_nodes"],
                "champion_connections": topo["connections"],
                "champion_enabled_connections": topo["enabled_connections"],
                "total_structural_events": sum(r["structural_events"] for r in recs),
                "total_new_innovations": sum(r["new_innovations"] for r in recs),
                "mean_tir": _nanmean([r["tir"] for r in recs]),
                "final_n_species": recs[-1]["n_species"],
                "mean_n_species": sum(r["n_species"] for r in recs) / len(recs),
                "mean_diversity": sum(r["genomic_diversity"] for r in recs) / len(recs),
                "final_diversity": recs[-1]["genomic_diversity"],
                "mean_distinct_fitness":
                    sum(r["n_distinct_fitness"] for r in recs) / len(recs),
                "mean_fitness_differentiation":
                    sum(r["fitness_differentiation"] for r in recs) / len(recs),
                "mean_fitness_cv": sum(r["fitness_cv"] for r in recs) / len(recs),
                "mean_selection_differential":
                    sum(r["selection_differential"] for r in recs) / len(recs),
            }
            with open(os.path.join(run_dir, "run.json"), "w") as fh:
                json.dump(row, fh, indent=2)
            rows.append(row)
            print(f"eta={eta:g} seed={seed} true={row['final_true_best']:.3f} "
                  f"succ={row['final_success_rate']:.2f} "
                  f"nodes={row['final_mean_nodes']:.1f}", flush=True)

    with open(os.path.join(out, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"\ndone -> {out}/summary.csv")
    print("compare against the baseline at the same eta values:")
    print(f"    python scripts/08_compare.py --a {args.baseline} --b {out}")


if __name__ == "__main__":
    main()
