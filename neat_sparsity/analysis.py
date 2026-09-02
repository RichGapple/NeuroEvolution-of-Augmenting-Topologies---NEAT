"""Loading and reshaping the raw run outputs for analysis."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def load_summary(outdir: str) -> pd.DataFrame:
    path = os.path.join(outdir, "summary.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    rows = []
    for eta_dir in sorted(os.listdir(outdir)):
        if not eta_dir.startswith("eta="):
            continue
        for seed_dir in sorted(os.listdir(os.path.join(outdir, eta_dir))):
            p = os.path.join(outdir, eta_dir, seed_dir, "run.json")
            if os.path.exists(p):
                rows.append(json.load(open(p)))
    return pd.DataFrame(rows)


def load_generations(outdir: str) -> pd.DataFrame:
    """Long-format longitudinal table: one row per (eta, seed, generation)."""
    frames = []
    for eta_dir in sorted(os.listdir(outdir)):
        if not eta_dir.startswith("eta="):
            continue
        eta = float(eta_dir.split("=")[1])
        for seed_dir in sorted(os.listdir(os.path.join(outdir, eta_dir))):
            p = os.path.join(outdir, eta_dir, seed_dir, "generations.csv")
            if not os.path.exists(p):
                continue
            df = pd.read_csv(p)
            df["eta"] = eta
            df["seed"] = int(seed_dir.split("=")[1])
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_ablation(outdir: str) -> pd.DataFrame:
    p = os.path.join(outdir, "ablation_summary.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def groups_from(df: pd.DataFrame, metric: str,
                by: str = "eta") -> Dict[float, List[float]]:
    """{eta: [one value per seed]} -- the unit of analysis is the RUN."""
    out: Dict[float, List[float]] = {}
    if metric not in df.columns:
        return out
    for k, sub in df.groupby(by):
        vals = pd.to_numeric(sub[metric], errors="coerce").tolist()
        out[float(k)] = [v for v in vals]
    return out


def curves_from(gen_df: pd.DataFrame, metric: str) -> Dict[float, np.ndarray]:
    """{eta: array (n_seeds, n_generations)} for trajectory plots."""
    out: Dict[float, np.ndarray] = {}
    if metric not in gen_df.columns:
        return out
    for eta, sub in gen_df.groupby("eta"):
        piv = sub.pivot_table(index="seed", columns="generation", values=metric)
        out[float(eta)] = piv.to_numpy(dtype=float)
    return out


def check_comparability(outdir: str) -> None:
    """Refuse to pool runs produced by different configurations."""
    df = load_summary(outdir)
    if df.empty:
        raise SystemExit(f"no runs found under {outdir}")
    hashes = set(df["config_hash"].astype(str))
    if len(hashes) > 1:
        raise SystemExit(
            "runs under this directory were produced with different configs "
            f"({sorted(hashes)}); they are not comparable."
        )
    counts = df.groupby("eta")["seed"].nunique()
    if counts.nunique() > 1:
        print(f"WARNING: unequal seed counts per condition:\n{counts}\n")


# --------------------------------------------------------------------------- #
# Metric registry: what gets tested, and in which direction it is interpreted.
# Fixed before looking at results (roadmap Appendix C).
# --------------------------------------------------------------------------- #
RUN_METRICS = [
    # (column, label, hypothesis)
    ("final_true_best",            "final best score (true objective)",     "performance"),
    ("final_success_rate",         "final success rate",                    "performance"),
    ("gen_first_solution",         "generations to first solution",         "performance"),
    ("mean_distinct_fitness",      "distinct fitness values / generation",  "H1"),
    ("mean_fitness_differentiation","fitness differentiation index",        "H1"),
    ("mean_fitness_cv",            "fitness coefficient of variation",      "H1"),
    ("mean_selection_differential","mean selection differential",           "H1"),
    ("total_structural_events",    "total structural mutation events",      "H2"),
    ("total_new_innovations",      "total novel innovations",               "H2"),
    ("mean_tir",                   "mean topological innovation rate",      "H2"),
    ("final_mean_nodes",           "final mean nodes",                      "H3"),
    ("final_mean_connections",     "final mean connections",                "H3"),
    ("final_mean_density",         "final mean network density",            "H3"),
    ("final_mean_depth",           "final mean network depth",              "H3"),
    ("champion_nodes",             "champion nodes",                        "H3"),
    ("champion_connections",       "champion connections",                  "H3"),
    ("mean_n_species",             "mean species count",                    "dynamics"),
    ("mean_diversity",             "mean genomic diversity",                "dynamics"),
    ("final_diversity",            "final genomic diversity",               "dynamics"),
]

ABLATION_METRICS = [
    ("functional_conn_fraction",   "functional connection fraction",        "H4"),
    ("functional_node_fraction",   "functional hidden-node fraction",       "H4"),
    ("n_functional_conns",         "functional connection count",           "H4"),
    ("seq_removable_fraction",     "sequentially removable fraction",       "H4"),
]

TRAJECTORY_METRICS = [
    ("mean_nodes",             "mean nodes"),
    ("mean_connections",       "mean connections"),
    ("mean_density",           "mean density"),
    ("true_best",              "best score (true objective)"),
    ("success_rate",           "success rate"),
    ("n_species",              "species count"),
    ("genomic_diversity",      "genomic diversity"),
    ("n_distinct_fitness",     "distinct fitness values"),
    ("fitness_differentiation","fitness differentiation index"),
    ("tir",                    "topological innovation rate"),
    ("selection_differential", "selection differential"),
]
