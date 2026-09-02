"""Statistics (roadmap Sections 19-20).

Deliberate choices:
  * Non-parametric omnibus (Kruskal-Wallis) because run-level distributions of
    evolutionary outcomes are routinely skewed, bimodal (solved / not solved)
    and heteroscedastic across conditions -- but the tests are only run *after*
    `describe_distributions` has been inspected.
  * Dunn's post-hoc with tie correction, Holm and Benjamini-Hochberg adjusted.
  * Cliff's delta as the effect size, because it is the non-parametric partner
    of the rank test and is meaningful for bimodal data.
  * Bootstrap CIs on condition medians, so every figure can carry uncertainty.
  * Trend: Spearman rho + Jonckheere-Terpstra (ordered alternative), which is
    the right test for "does the outcome change monotonically with eta".
  * Threshold: two-segment piecewise-linear fit vs. a linear fit, compared by
    AIC, with a bootstrap CI on the breakpoint.  Nothing is called a phase
    transition; the output is "breakpoint estimate + CI + AIC difference".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sps


# --------------------------------------------------------------------------- #
# Descriptives -- ALWAYS look at these before choosing a test
# --------------------------------------------------------------------------- #
def describe_distributions(groups: Dict[float, Sequence[float]]) -> List[dict]:
    rows = []
    for k in sorted(groups):
        x = np.asarray(groups[k], dtype=float)
        x = x[~np.isnan(x)]
        if len(x) == 0:
            continue
        row = {
            "condition": k,
            "n": int(len(x)),
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "sd": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
            "iqr": float(np.subtract(*np.percentile(x, [75, 25]))),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "n_unique": int(len(np.unique(x))),
        }
        degenerate = len(np.unique(x)) < 2
        row["skew"] = float("nan") if (degenerate or len(x) <= 2) else float(sps.skew(x))
        row["shapiro_p"] = (float("nan") if (degenerate or not (3 <= len(x) <= 5000))
                            else float(sps.shapiro(x).pvalue))
        rows.append(row)
    # Levene across all groups (variance homogeneity)
    arrays = [np.asarray(groups[k], float)[~np.isnan(np.asarray(groups[k], float))]
              for k in sorted(groups)]
    arrays = [a for a in arrays if len(a) > 1]
    if len(arrays) > 1:
        try:
            lev = sps.levene(*arrays, center="median").pvalue
        except Exception:
            lev = float("nan")
        for r in rows:
            r["levene_p_all"] = float(lev)
    return rows


# --------------------------------------------------------------------------- #
# Effect sizes
# --------------------------------------------------------------------------- #
def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """delta in [-1,1].  >0 means a tends to exceed b."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = 0
    lt = 0
    b_sorted = np.sort(b)
    for x in a:
        gt += int(np.searchsorted(b_sorted, x, side="left"))
        lt += int(len(b) - np.searchsorted(b_sorted, x, side="right"))
    return (gt - lt) / (len(a) * len(b))


def cliffs_magnitude(d: float) -> str:
    ad = abs(d)
    if ad != ad:
        return "undefined"
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def rank_biserial_from_u(u: float, n1: int, n2: int) -> float:
    return 1.0 - 2.0 * u / (n1 * n2)


def epsilon_squared(h: float, n: int, k: int) -> float:
    """Kruskal-Wallis effect size."""
    if n <= k:
        return float("nan")
    return max(0.0, (h - k + 1) / (n - k))   # clamped: negative values are noise


# --------------------------------------------------------------------------- #
# Omnibus + post hoc
# --------------------------------------------------------------------------- #
@dataclass
class OmnibusResult:
    statistic: float
    pvalue: float
    epsilon_squared: float
    k: int
    n: int


def kruskal(groups: Dict[float, Sequence[float]]) -> OmnibusResult:
    keys = sorted(groups)
    arrays = []
    for k in keys:
        a = np.asarray(groups[k], float)
        arrays.append(a[~np.isnan(a)])
    arrays = [a for a in arrays if len(a) > 0]
    n = sum(len(a) for a in arrays)
    if len(arrays) < 2 or all(len(np.unique(np.concatenate(arrays))) == 1 for _ in [0]):
        return OmnibusResult(float("nan"), float("nan"), float("nan"), len(arrays), n)
    h, p = sps.kruskal(*arrays)
    return OmnibusResult(float(h), float(p), epsilon_squared(h, n, len(arrays)),
                         len(arrays), n)


def dunn(groups: Dict[float, Sequence[float]],
         method: str = "holm") -> List[dict]:
    """Dunn's test with tie correction; p adjusted by Holm and BH."""
    keys = sorted(groups)
    data, labels = [], []
    for k in keys:
        a = np.asarray(groups[k], float)
        a = a[~np.isnan(a)]
        data.append(a)
        labels.append(k)
    all_x = np.concatenate(data)
    n = len(all_x)
    ranks = sps.rankdata(all_x)

    idx, out_ranks = 0, []
    for a in data:
        out_ranks.append(ranks[idx:idx + len(a)])
        idx += len(a)

    _, counts = np.unique(all_x, return_counts=True)
    tie_sum = float(np.sum(counts ** 3 - counts))
    sigma_common = (n * (n + 1) / 12.0) - tie_sum / (12.0 * (n - 1)) if n > 1 else 0.0

    rows = []
    for i, j in combinations(range(len(labels)), 2):
        ni, nj = len(data[i]), len(data[j])
        if ni == 0 or nj == 0:
            continue
        mi, mj = out_ranks[i].mean(), out_ranks[j].mean()
        se = math.sqrt(max(sigma_common, 0.0) * (1.0 / ni + 1.0 / nj))
        z = (mi - mj) / se if se > 0 else 0.0
        p = 2 * (1 - sps.norm.cdf(abs(z)))
        d = cliffs_delta(data[i], data[j])
        rows.append({"a": labels[i], "b": labels[j], "z": float(z), "p_raw": float(p),
                     "cliffs_delta": float(d), "magnitude": cliffs_magnitude(d),
                     "median_a": float(np.median(data[i])),
                     "median_b": float(np.median(data[j]))})

    if rows:
        praw = np.array([r["p_raw"] for r in rows])
        for r, ph, pb in zip(rows, _holm(praw), _bh(praw)):
            r["p_holm"] = float(ph)
            r["p_bh"] = float(pb)
    return rows


def _holm(p: np.ndarray) -> np.ndarray:
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def _bh(p: np.ndarray) -> np.ndarray:
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = min(prev, p[i] * m / (rank + 1))
        adj[i] = min(1.0, val)
        prev = val
    return adj


# --------------------------------------------------------------------------- #
# Trend
# --------------------------------------------------------------------------- #
def trend_tests(groups: Dict[float, Sequence[float]]) -> dict:
    keys = sorted(groups)
    xs, ys = [], []
    for k in keys:
        for v in groups[k]:
            if v == v:
                xs.append(k); ys.append(v)
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    out: dict = {}
    if len(xs) > 2 and len(np.unique(ys)) > 1:
        rho, p = sps.spearmanr(xs, ys)
        out["spearman_rho"] = float(rho)
        out["spearman_p"] = float(p)
    else:
        out["spearman_rho"] = float("nan")
        out["spearman_p"] = float("nan")

    arrays = [np.asarray(groups[k], float) for k in keys]
    arrays = [a[~np.isnan(a)] for a in arrays]
    arrays = [a for a in arrays if len(a) > 0]
    if len(arrays) >= 3:
        try:
            jt = sps.jonckheereterpstra  # not in all scipy versions
        except AttributeError:
            jt = None
        if jt is not None:
            try:
                res = jt(*arrays)
                out["jonckheere_stat"] = float(res.statistic)
                out["jonckheere_p"] = float(res.pvalue)
            except Exception:
                pass
        else:
            out.update(_jonckheere(arrays))
    return out


def _jonckheere(arrays: List[np.ndarray]) -> dict:
    """Jonckheere-Terpstra test for ordered alternatives (normal approximation)."""
    k = len(arrays)
    ns = [len(a) for a in arrays]
    n = sum(ns)
    jt = 0.0
    for i in range(k - 1):
        for j in range(i + 1, k):
            a, b = arrays[i], arrays[j]
            bs = np.sort(b)
            for x in a:
                lo = np.searchsorted(bs, x, side="left")
                hi = np.searchsorted(bs, x, side="right")
                jt += lo + 0.5 * (hi - lo)
    mu = (n ** 2 - sum(x ** 2 for x in ns)) / 4.0
    var = (n ** 2 * (2 * n + 3) - sum(x ** 2 * (2 * x + 3) for x in ns)) / 72.0
    z = (jt - mu) / math.sqrt(var) if var > 0 else 0.0
    return {"jonckheere_stat": float(jt), "jonckheere_z": float(z),
            "jonckheere_p": float(2 * (1 - sps.norm.cdf(abs(z))))}


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def bootstrap_ci(x: Sequence[float], stat=np.median, n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float]:
    a = np.asarray(x, float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    point = float(stat(a))
    if len(a) == 1:
        return point, point, point
    boots = stat(rng.choice(a, size=(n_boot, len(a)), replace=True), axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Nonlinearity / threshold
# --------------------------------------------------------------------------- #
def _fit_linear(x, y):
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    return float(np.sum(resid ** 2)), coef


def _fit_piecewise(x, y, bp):
    """Continuous two-segment fit with breakpoint bp."""
    A = np.vstack([np.ones_like(x), x, np.maximum(0.0, x - bp)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    return float(np.sum(resid ** 2)), coef


def _aic(rss: float, n: int, k: int) -> float:
    if rss <= 0:
        rss = 1e-12
    return n * math.log(rss / n) + 2 * k


def threshold_analysis(groups: Dict[float, Sequence[float]],
                       n_boot: int = 1000, seed: int = 0) -> dict:
    """Compare linear vs. two-segment fit over eta on run-level observations."""
    keys = sorted(groups)
    xs, ys = [], []
    for k in keys:
        for v in groups[k]:
            if v == v:
                xs.append(float(k)); ys.append(float(v))
    x = np.asarray(xs); y = np.asarray(ys)
    if len(np.unique(x)) < 4 or len(np.unique(y)) < 2:
        return {"note": "insufficient distinct conditions/values for threshold analysis"}

    cand = np.unique(x)[1:-1]
    rss_lin, _ = _fit_linear(x, y)
    best_bp, best_rss, best_coef = None, float("inf"), None
    for bp in cand:
        rss, coef = _fit_piecewise(x, y, bp)
        if rss < best_rss:
            best_bp, best_rss, best_coef = float(bp), rss, coef

    n = len(x)
    aic_lin = _aic(rss_lin, n, 2)
    aic_pw = _aic(best_rss, n, 4)   # +1 for the breakpoint itself

    rng = np.random.default_rng(seed)
    bps = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        if len(np.unique(xb)) < 3:
            continue
        c = np.unique(xb)[1:-1]
        if len(c) == 0:
            continue
        r_best, bp_best = float("inf"), None
        for bp in c:
            r, _ = _fit_piecewise(xb, yb, bp)
            if r < r_best:
                r_best, bp_best = r, float(bp)
        if bp_best is not None:
            bps.append(bp_best)

    out = {
        "breakpoint": best_bp,
        "slope_before": float(best_coef[1]),
        "slope_after": float(best_coef[1] + best_coef[2]),
        "rss_linear": rss_lin,
        "rss_piecewise": best_rss,
        "aic_linear": aic_lin,
        "aic_piecewise": aic_pw,
        "delta_aic": float(aic_lin - aic_pw),   # >2 favours the piecewise model
        "piecewise_preferred": bool(aic_lin - aic_pw > 2.0),
    }
    if bps:
        out["breakpoint_ci95"] = [float(np.percentile(bps, 2.5)),
                                  float(np.percentile(bps, 97.5))]
        out["breakpoint_bootstrap_sd"] = float(np.std(bps))
    return out


# --------------------------------------------------------------------------- #
def full_report(groups: Dict[float, Sequence[float]], label: str,
                seed: int = 0) -> dict:
    """Descriptives + omnibus + post hoc + trend + threshold for one metric."""
    return {
        "metric": label,
        "descriptives": describe_distributions(groups),
        "bootstrap_medians": {
            str(k): dict(zip(("median", "lo95", "hi95"),
                             bootstrap_ci(groups[k], seed=seed)))
            for k in sorted(groups)
        },
        "omnibus_kruskal": kruskal(groups).__dict__,
        "posthoc_dunn": dunn(groups),
        "trend": trend_tests(groups),
        "threshold": threshold_analysis(groups, seed=seed),
    }
