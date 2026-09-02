"""Stage 6 -- functional vs. non-functional topology.

Procedure (fixed in advance, roadmap Section 16):

  1. Take the champion genome of a run.
  2. Score it on the *sparsity-independent* objective (eta = 0).  Ablations must
     never be scored under the training eta: at eta = 1 almost every ablation
     produces zero change simply because the training signal is degenerate,
     which would manufacture the very result we are testing for.
  3. Single-component ablation:
       - connection ablation: set enabled = False for one connection.
       - node ablation: remove one hidden node *and every incident connection*.
         (Removing a node without its edges is undefined; this rule is applied
         uniformly.)
  4. delta = score_intact - score_ablated.
  5. A component is FUNCTIONAL if delta > tau, where tau is a fixed absolute
     threshold on the true objective (default 0.01, i.e. 1% of the max score
     of 2.0 -- 0.5% of full range).  The environment is deterministic, so the
     evaluation noise floor is exactly zero and tau is a pure effect-size
     threshold, not a noise threshold.

Reported quantities per run:
    functional_conn_fraction, functional_node_fraction
    mean_|delta| for each component type
    'harmful' components (delta < -tau): removing them *improves* the score
"""

from __future__ import annotations

import json
import os
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

from .config import ExperimentConfig
from .env import NavEnv
from .genome import Genome, HIDDEN
from .network import build_network
from .reward import true_objective


DEFAULT_TAU = 0.01


def score(g: Genome, cfg: ExperimentConfig, env: Optional[NavEnv] = None) -> float:
    env = env or NavEnv(cfg.env)
    traj = env.rollout(build_network(g, cfg.neat))
    return true_objective(traj, cfg.sparsity)


def ablate_connection(g: Genome, key: Tuple[int, int]) -> Genome:
    h = g.copy()
    if key in h.conns:
        h.conns[key].enabled = False
    return h


def ablate_node(g: Genome, node_key: int) -> Genome:
    h = g.copy()
    if node_key in h.nodes and h.nodes[node_key].type == HIDDEN:
        del h.nodes[node_key]
        for ck in [k for k in h.conns if node_key in k]:
            del h.conns[ck]
    return h


def ablation_report(g: Genome, cfg: ExperimentConfig,
                    tau: float = DEFAULT_TAU) -> Dict[str, object]:
    env = NavEnv(cfg.env)
    base = score(g, cfg, env)

    conn_rows: List[dict] = []
    for key in sorted(k for k, c in g.conns.items() if c.enabled):
        s = score(ablate_connection(g, key), cfg, env)
        conn_rows.append({"component": "connection", "id": f"{key[0]}->{key[1]}",
                          "innovation": g.conns[key].innovation,
                          "score": s, "delta": base - s})

    node_rows: List[dict] = []
    for nk in g.hidden_keys():
        s = score(ablate_node(g, nk), cfg, env)
        node_rows.append({"component": "node", "id": str(nk),
                          "innovation": -1, "score": s, "delta": base - s})

    def frac_functional(rows):
        return (sum(1 for r in rows if r["delta"] > tau) / len(rows)) if rows else float("nan")

    def frac_harmful(rows):
        return (sum(1 for r in rows if r["delta"] < -tau) / len(rows)) if rows else float("nan")

    return {
        "baseline_score": base,
        "tau": tau,
        "n_enabled_connections": len(conn_rows),
        "n_hidden_nodes": len(node_rows),
        "functional_conn_fraction": frac_functional(conn_rows),
        "functional_node_fraction": frac_functional(node_rows),
        "harmful_conn_fraction": frac_harmful(conn_rows),
        "harmful_node_fraction": frac_harmful(node_rows),
        "mean_abs_conn_delta": mean(abs(r["delta"]) for r in conn_rows) if conn_rows else 0.0,
        "mean_abs_node_delta": mean(abs(r["delta"]) for r in node_rows) if node_rows else 0.0,
        "max_conn_delta": max((r["delta"] for r in conn_rows), default=0.0),
        # absolute counts, needed for the redundancy analysis
        "n_functional_conns": sum(1 for r in conn_rows if r["delta"] > tau),
        "n_functional_nodes": sum(1 for r in node_rows if r["delta"] > tau),
        "components": conn_rows + node_rows,
    }


def minimal_functional_network(g: Genome, cfg: ExperimentConfig,
                               tau: float = DEFAULT_TAU) -> Dict[str, object]:
    """Greedy sequential pruning: repeatedly remove the least-damaging component
    while the score stays within tau of the original.

    Single-component ablation under-counts redundancy when two components are
    mutually redundant (each alone is removable, both are not).  Sequential
    pruning gives the complementary upper bound on non-functional structure, so
    the paper can report both and avoid over-claiming either way.
    """
    env = NavEnv(cfg.env)
    base = score(g, cfg, env)
    cur = g.copy()
    removed = 0

    while True:
        best_key, best_score = None, float("-inf")
        for key in sorted(k for k, c in cur.conns.items() if c.enabled):
            s = score(ablate_connection(cur, key), cfg, env)
            if s > best_score:
                best_score, best_key = s, key
        if best_key is None or best_score < base - tau:
            break
        cur = ablate_connection(cur, best_key)
        removed += 1

    # drop hidden nodes that no longer participate
    for nk in cur.hidden_keys():
        if not any(c.enabled and nk in c.key for c in cur.conns.values()):
            cur = ablate_node(cur, nk)

    n0 = sum(1 for c in g.conns.values() if c.enabled)
    n1 = sum(1 for c in cur.conns.values() if c.enabled)
    return {
        "baseline_score": base,
        "pruned_score": score(cur, cfg, env),
        "enabled_conns_before": n0,
        "enabled_conns_after": n1,
        "hidden_before": len(g.hidden_keys()),
        "hidden_after": len(cur.hidden_keys()),
        "removable_fraction": (n0 - n1) / n0 if n0 else float("nan"),
        "removed_sequentially": removed,
    }


# --------------------------------------------------------------------------- #
def run_ablation_sweep(outdir: str, tau: float = DEFAULT_TAU,
                       sequential: bool = True, verbose: bool = True) -> List[dict]:
    """Ablate every champion under `outdir` (produced by run_sweep)."""
    from .runner import genome_from_dict

    cfg = ExperimentConfig.load(os.path.join(outdir, "config.json"))
    rows: List[dict] = []

    for eta_dir in sorted(os.listdir(outdir)):
        if not eta_dir.startswith("eta="):
            continue
        eta = float(eta_dir.split("=")[1])
        for seed_dir in sorted(os.listdir(os.path.join(outdir, eta_dir))):
            run_dir = os.path.join(outdir, eta_dir, seed_dir)
            champ_path = os.path.join(run_dir, "champion.json")
            if not os.path.exists(champ_path):
                continue
            seed = int(seed_dir.split("=")[1])
            with open(champ_path) as fh:
                g = genome_from_dict(json.load(fh))

            rep = ablation_report(g, cfg, tau)
            row = {"eta": eta, "seed": seed,
                   **{k: v for k, v in rep.items() if k != "components"}}
            if sequential:
                row.update({f"seq_{k}": v
                            for k, v in minimal_functional_network(g, cfg, tau).items()})
            rows.append(row)

            with open(os.path.join(run_dir, "ablation.json"), "w") as fh:
                json.dump(rep, fh, indent=2)
            if verbose:
                print(f"eta={eta:g} seed={seed}  base={rep['baseline_score']:.3f} "
                      f"functional_conns={rep['functional_conn_fraction']:.2f} "
                      f"({rep['n_enabled_connections']} enabled)", flush=True)

    if rows:
        import csv
        with open(os.path.join(outdir, "ablation_summary.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows
