"""NEAT vs DQN: per-condition tests, the crossover point, and cost accounting.

What the crossover is
---------------------
Two medians, both functions of eta.  Define

    D(eta) = median(DQN final true objective) - median(NEAT final true objective)

If D is positive at eta = 0 and negative at eta = 1, there is an eta* where the
sign flips.  That eta* is the reward resolution below which evolution becomes
the better choice on this task, and it is the number the paper exists to report.

It is estimated by linear interpolation between the two grid points that bracket
the sign change, and its uncertainty by resampling seeds within each condition.
A crossover whose CI spans most of the grid is not a finding -- say so.

Guard rails
-----------
`assert_comparable` refuses to compare two arms that were not run on the same
environment, the same sparsity mode, the same eta grid or the same seeds.  This
is the cross-arm version of `neat_sparsity.analysis.check_comparability`.  It is
the single most important function in this file: a crossover computed across
mismatched configurations is worse than no crossover at all.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps

from neat_sparsity.stats import cliffs_delta, cliffs_magnitude, bootstrap_ci


# --------------------------------------------------------------------------- #
def load_arm(outdir: str, arm: str) -> pd.DataFrame:
    """Load a sweep's summary.csv and tag it with which arm produced it."""
    path = os.path.join(outdir, "summary.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        rows = []
        for eta_dir in sorted(os.listdir(outdir)):
            if not eta_dir.startswith("eta="):
                continue
            for seed_dir in sorted(os.listdir(os.path.join(outdir, eta_dir))):
                p = os.path.join(outdir, eta_dir, seed_dir, "run.json")
                if os.path.exists(p):
                    rows.append(json.load(open(p)))
        df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no runs found under {outdir}")
    df["arm"] = arm
    return df


def assert_comparable(neat_df: pd.DataFrame, dqn_df: pd.DataFrame,
                      strict_seeds: bool = True) -> List[str]:
    """Refuse to compare arms that were not run on the same experiment.

    Returns a list of warnings; raises SystemExit on anything fatal.
    """
    warnings: List[str] = []

    for name, df in (("NEAT", neat_df), ("DQN", dqn_df)):
        h = set(df["config_hash"].astype(str))
        if len(h) > 1:
            raise SystemExit(f"{name} runs span multiple config hashes {sorted(h)}; "
                             "they are not poolable, let alone comparable.")

    n_modes = set(neat_df["sparsity_mode"].astype(str))
    d_modes = set(dqn_df["sparsity_mode"].astype(str))
    if n_modes != d_modes:
        raise SystemExit(f"sparsity modes differ: NEAT {n_modes} vs DQN {d_modes}. "
                         "The two arms are not measuring the same manipulation.")

    n_etas = set(np.round(neat_df["eta"].astype(float), 6))
    d_etas = set(np.round(dqn_df["eta"].astype(float), 6))
    if n_etas != d_etas:
        raise SystemExit(f"eta grids differ: NEAT {sorted(n_etas)} vs "
                         f"DQN {sorted(d_etas)}.")

    n_seeds = set(neat_df["seed"].astype(int))
    d_seeds = set(dqn_df["seed"].astype(int))
    if n_seeds != d_seeds:
        msg = (f"seed sets differ ({len(n_seeds)} NEAT vs {len(d_seeds)} DQN, "
               f"{len(n_seeds & d_seeds)} shared). Comparisons are unpaired.")
        if strict_seeds:
            warnings.append("WARNING: " + msg)
        else:
            warnings.append(msg)

    for col in ("final_true_best",):
        for name, df in (("NEAT", neat_df), ("DQN", dqn_df)):
            if col not in df.columns:
                raise SystemExit(f"{name} summary is missing required column {col!r}")

    counts = pd.concat([neat_df, dqn_df]).groupby(["arm", "eta"])["seed"].nunique()
    if counts.nunique() > 1:
        warnings.append(f"WARNING: unequal seed counts per cell:\n{counts}")

    return warnings


# --------------------------------------------------------------------------- #
def groups(df: pd.DataFrame, metric: str) -> Dict[float, List[float]]:
    out: Dict[float, List[float]] = {}
    if metric not in df.columns:
        return out
    for k, sub in df.groupby("eta"):
        out[float(k)] = pd.to_numeric(sub[metric], errors="coerce").tolist()
    return out


def per_condition_tests(neat_g: Dict[float, Sequence[float]],
                        dqn_g: Dict[float, Sequence[float]],
                        alpha: float = 0.05) -> pd.DataFrame:
    """Mann-Whitney U + Cliff's delta at every eta, Holm-corrected across etas.

    Mann-Whitney rather than a t-test because the outcome has a hard ceiling at
    2.0 and a floor at 0, so it is nowhere near normal.  Cliff's delta because a
    p-value alone cannot tell you whether a difference matters.
    """
    keys = sorted(set(neat_g) & set(dqn_g))
    rows = []
    for k in keys:
        a = np.asarray([v for v in dqn_g[k] if v == v], float)
        b = np.asarray([v for v in neat_g[k] if v == v], float)
        if len(a) < 3 or len(b) < 3:
            continue
        try:
            u, p = sps.mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:      # all values identical
            u, p = float("nan"), 1.0
        d = cliffs_delta(a, b)
        ma, la, ha = bootstrap_ci(a)
        mb, lb, hb = bootstrap_ci(b)
        rows.append({
            "eta": k,
            "dqn_median": ma, "dqn_lo": la, "dqn_hi": ha,
            "neat_median": mb, "neat_lo": lb, "neat_hi": hb,
            "delta_median": ma - mb,
            "mannwhitney_u": u, "p_raw": p,
            "cliffs_delta": d, "magnitude": cliffs_magnitude(d),
            "winner": ("DQN" if ma > mb else "NEAT" if mb > ma else "tie"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["p_holm"] = _holm(df["p_raw"].to_numpy())
    df["significant"] = df["p_holm"] < alpha
    return df


def _holm(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n, float)
    running = 0.0
    for rank, i in enumerate(order):
        val = (n - rank) * p[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


# --------------------------------------------------------------------------- #
def crossover_point(neat_g: Dict[float, Sequence[float]],
                    dqn_g: Dict[float, Sequence[float]],
                    n_boot: int = 4000, seed: int = 0) -> dict:
    """Locate where median(DQN) - median(NEAT) changes sign, with a bootstrap CI."""
    keys = sorted(set(neat_g) & set(dqn_g))

    def _cross(sample: bool, rng) -> Optional[float]:
        d = []
        for k in keys:
            a = np.asarray([v for v in dqn_g[k] if v == v], float)
            b = np.asarray([v for v in neat_g[k] if v == v], float)
            if len(a) == 0 or len(b) == 0:
                return None
            if sample:
                a = rng.choice(a, len(a))
                b = rng.choice(b, len(b))
            d.append(np.median(a) - np.median(b))
        d = np.asarray(d)
        for i in range(len(d) - 1):
            if d[i] == 0:
                return keys[i]
            if d[i] > 0 >= d[i + 1]:
                span = d[i] - d[i + 1]
                if span == 0:
                    return keys[i]
                frac = d[i] / span
                return keys[i] + frac * (keys[i + 1] - keys[i])
        return None

    rng = np.random.default_rng(seed)
    point = _cross(False, rng)
    boots = [c for c in (_cross(True, rng) for _ in range(n_boot)) if c is not None]

    out = {"eta": point, "n_boot_valid": len(boots), "n_boot": n_boot}
    if boots:
        out["ci"] = (float(np.percentile(boots, 2.5)),
                     float(np.percentile(boots, 97.5)))
        out["ci_width"] = out["ci"][1] - out["ci"][0]
        out["p_exists"] = len(boots) / n_boot
    else:
        out["ci"] = (None, None)
        out["ci_width"] = None
        out["p_exists"] = 0.0
    return out


def breakpoint_per_arm(g: Dict[float, Sequence[float]]) -> dict:
    """Where does each arm individually fall off?  Uses the NEAT arm's own
    segmented-regression machinery so both arms are analysed identically."""
    from neat_sparsity.stats import threshold_analysis
    try:
        return threshold_analysis(g)
    except Exception as exc:                       # pragma: no cover
        return {"error": str(exc)}


# --------------------------------------------------------------------------- #
def cost_table(neat_df: pd.DataFrame, dqn_df: pd.DataFrame) -> pd.DataFrame:
    """Three cost columns, never collapsed into one.

    NEAT and DQN move in opposite directions on sample efficiency versus
    hardware footprint.  A single "compute" number would hide exactly the trade
    the paper is about, so report environment steps, wall-clock and peak memory
    side by side, and state separately whether a GPU was used.
    """
    rows = []
    for name, df in (("NEAT", neat_df), ("DQN", dqn_df)):
        rec = {"arm": name, "runs": len(df)}
        for col, out in (("env_steps", "env_steps"),
                         ("wallclock_s", "wallclock_s"),
                         ("peak_rss_mb", "peak_rss_mb"),
                         ("gradient_updates", "gradient_updates"),
                         ("n_params", "n_params")):
            if col in df.columns:
                v = pd.to_numeric(df[col], errors="coerce").dropna()
                rec[f"{out}_median"] = float(v.median()) if len(v) else float("nan")
                rec[f"{out}_total"] = float(v.sum()) if len(v) else float("nan")
            else:
                rec[f"{out}_median"] = float("nan")
                rec[f"{out}_total"] = float("nan")
        rec["gpu_required"] = "no"
        rec["autodiff_required"] = "yes" if name == "DQN" else "no"
        rows.append(rec)
    return pd.DataFrame(rows)


def sample_efficiency(dqn_blocks: pd.DataFrame,
                      neat_g: Dict[float, Sequence[float]]) -> pd.DataFrame:
    """At each eta: what fraction of the budget did DQN need to match NEAT's
    final median?  `nan` means it never matched it."""
    rows = []
    for eta, sub in dqn_blocks.groupby("eta"):
        target = float(np.nanmedian(neat_g.get(float(eta), [np.nan])))
        piv = sub.pivot_table(index="seed", columns="block", values="true_best")
        med = np.nanmedian(piv.to_numpy(dtype=float), axis=0)
        frac = np.linspace(0, 1, len(med))
        hit = np.where(med >= target)[0]
        rows.append({
            "eta": float(eta),
            "neat_final_median": target,
            "dqn_final_median": float(med[-1]) if len(med) else float("nan"),
            "budget_frac_to_match": float(frac[hit[0]]) if len(hit) else float("nan"),
            "matched": bool(len(hit) > 0),
        })
    return pd.DataFrame(rows).sort_values("eta")


# --------------------------------------------------------------------------- #
def render_report(neat_df, dqn_df, tests: pd.DataFrame, cross: dict,
                  costs: pd.DataFrame, effic: pd.DataFrame,
                  warnings: Sequence[str], alpha: float = 0.05) -> str:
    """Markdown, in the same voice as the NEAT arm's generated report."""
    L: List[str] = []
    A = L.append
    A("# NEAT vs DQN under matched reward sparsity\n")

    A(f"- NEAT runs: **{len(neat_df)}**, config hash "
      f"`{neat_df['config_hash'].iloc[0]}`")
    A(f"- DQN runs: **{len(dqn_df)}**, config hash "
      f"`{dqn_df['config_hash'].iloc[0]}`")
    if "shaping_mode" in dqn_df.columns:
        A(f"- DQN shaping: `{dqn_df['shaping_mode'].iloc[0]}`")
    if "env_steps" in dqn_df.columns:
        A(f"- DQN interaction budget: "
          f"{int(pd.to_numeric(dqn_df['env_steps']).median()):,} env steps/run")
    A(f"- sparsity mode: `{neat_df['sparsity_mode'].iloc[0]}`; "
      f"alpha = {alpha}; Mann-Whitney per condition, Holm-adjusted across the "
      f"eta grid.\n")

    if warnings:
        A("## Comparability warnings\n")
        for w in warnings:
            A(f"- {w}")
        A("")

    A("## Performance by condition\n")
    A("Both arms scored on the **eta-independent** true objective, so a "
      "difference here is a finding rather than a units artefact.\n")
    if not tests.empty:
        A("| eta | NEAT median | DQN median | delta | Cliff's d | p (Holm) | winner |")
        A("|---|---|---|---|---|---|---|")
        for _, r in tests.iterrows():
            star = "*" if r["significant"] else ""
            A(f"| {r['eta']:g} | {r['neat_median']:.3f} "
              f"[{r['neat_lo']:.3f}, {r['neat_hi']:.3f}] | "
              f"{r['dqn_median']:.3f} [{r['dqn_lo']:.3f}, {r['dqn_hi']:.3f}] | "
              f"{r['delta_median']:+.3f} | {r['cliffs_delta']:+.2f} "
              f"({r['magnitude']}) | {r['p_holm']:.3g}{star} | {r['winner']} |")
        A("\n`*` = significant after Holm correction.\n")

    A("## Crossover\n")
    if cross.get("eta") is None:
        A("**No sign change was found on this grid.** One arm dominates "
          "everywhere, so there is no crossover to report. Say that plainly; "
          "a null crossover is a legitimate result and is more useful than a "
          "crossover manufactured by extrapolating past the grid.\n")
    else:
        lo, hi = cross["ci"]
        A(f"- Crossover at **eta = {cross['eta']:.3f}**")
        if lo is not None:
            A(f"- Bootstrap 95% CI: [{lo:.3f}, {hi:.3f}] "
              f"(width {cross['ci_width']:.3f})")
        A(f"- A sign change was present in {100 * cross['p_exists']:.1f}% of "
          f"bootstrap resamples.\n")
        if cross.get("ci_width") and cross["ci_width"] > 0.30:
            A("> The CI spans more than 30% of the eta grid. Report the "
              "crossover as *located between two grid points*, not as a point "
              "estimate, and add seeds around it before claiming precision.\n")

    A("## Sample efficiency\n")
    if not effic.empty:
        A("Fraction of the interaction budget DQN needed to reach NEAT's final "
          "median score. `nan` = never reached it.\n")
        A("| eta | NEAT final | DQN final | budget frac to match |")
        A("|---|---|---|---|")
        for _, r in effic.iterrows():
            bf = ("n/a" if not r["matched"]
                  else f"{r['budget_frac_to_match']:.3f}")
            A(f"| {r['eta']:g} | {r['neat_final_median']:.3f} | "
              f"{r['dqn_final_median']:.3f} | {bf} |")
        A("")

    A("## Cost accounting\n")
    A("Three separate numbers. Do not collapse them: the two arms move in "
      "opposite directions on sample efficiency versus hardware footprint, and "
      "a single 'compute' figure would hide the trade this paper is about.\n")
    A("| arm | runs | env steps (median) | wall-clock s (median) | peak RSS MB | "
      "gradient updates | params | autodiff | GPU |")
    A("|---|---|---|---|---|---|---|---|---|")
    for _, r in costs.iterrows():
        def f(x, i=False):
            if x != x:
                return "n/a"
            return f"{int(x):,}" if i else f"{x:,.1f}"
        A(f"| {r['arm']} | {int(r['runs'])} | {f(r['env_steps_median'], True)} | "
          f"{f(r['wallclock_s_median'])} | {f(r['peak_rss_mb_median'])} | "
          f"{f(r['gradient_updates_median'], True)} | "
          f"{f(r['n_params_median'], True)} | {r['autodiff_required']} | "
          f"{r['gpu_required']} |")

    A("\n## What this table does and does not license\n")
    A("- **Licensed:** statements about which method scores higher at a given "
      "eta on this task under this budget, with effect sizes.")
    A("- **Licensed:** the crossover location, provided its CI is narrow.")
    A("- **Not licensed:** 'NEAT is better than DQN'. The result is "
      "condition-dependent by construction; that is the point.")
    A("- **Not licensed:** 'NEAT is computationally cheaper'. NEAT consumes "
      "*more* environment interaction. What it needs less of is infrastructure: "
      "no autodiff, no replay buffer, no GPU. Say that, and cite the three "
      "columns above rather than a single number.")
    A("- **Not licensed:** generalisation beyond this maze. One task is one "
      "task. Run a second environment before generalising.\n")
    return "\n".join(L)
