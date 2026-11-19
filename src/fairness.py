"""Descriptive subgroup metrics and bootstrap uncertainty for fairness gaps."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

RATE_METRICS = ("predicted_positive_rate", "TPR", "FPR", "FNR")


def _as_aligned_arrays(y_true, y_pred, groups):
    y = np.asarray(y_true)
    pred = np.asarray(y_pred)
    group = pd.Series(groups).reset_index(drop=True).fillna("Missing").astype(str)
    if y.ndim != 1 or pred.ndim != 1 or len(y) != len(pred) or len(y) != len(group):
        raise ValueError("y_true, y_pred, and groups must be one-dimensional and equally sized.")
    if not set(np.unique(y)).issubset({0, 1}) or not set(np.unique(pred)).issubset({0, 1}):
        raise ValueError("y_true and y_pred must contain only binary values 0/1.")
    return y.astype(int), pred.astype(int), group


def _rates(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    positives, negatives = y == 1, y == 0
    tpr = float(pred[positives].mean()) if positives.any() else np.nan
    fpr = float(pred[negatives].mean()) if negatives.any() else np.nan
    return {
        "predicted_positive_rate": float(pred.mean()) if len(pred) else np.nan,
        "TPR": tpr, "FPR": fpr,
        "FNR": float(1.0 - tpr) if np.isfinite(tpr) else np.nan,
    }


def group_fairness_report(y_true, y_pred, groups, min_group_size: int = 100) -> pd.DataFrame:
    """Compute subgroup metrics; small and missing-value groups are flagged."""
    if min_group_size < 1:
        raise ValueError("min_group_size must be at least 1.")
    y, pred, group = _as_aligned_arrays(y_true, y_pred, groups)
    rows = []
    for value in sorted(group.unique()):
        mask = (group == value).to_numpy()
        yt, yp = y[mask], pred[mask]
        n = int(mask.sum())
        positives, negatives = int((yt == 1).sum()), int((yt == 0).sum())
        fp, fn = int(((yt == 0) & (yp == 1)).sum()), int(((yt == 1) & (yp == 0)).sum())
        rows.append({
            "group": value, "n": n, "actual_positive": positives, "actual_negative": negatives,
            **_rates(yt, yp),
            "precision": precision_score(yt, yp, zero_division=0),
            "recall": recall_score(yt, yp, zero_division=0),
            "F1": f1_score(yt, yp, zero_division=0),
            "error_rate": (fp + fn) / n if n else np.nan,
            "disparate_impact_vs_highest_rate": np.nan,
            "small_group_warning": n < min_group_size,
        })
    out = pd.DataFrame(rows)
    eligible = (~out["small_group_warning"]) & (out["group"] != "Missing")
    max_rate = out.loc[eligible, "predicted_positive_rate"].max()
    if pd.notna(max_rate) and max_rate > 0:
        out.loc[eligible, "disparate_impact_vs_highest_rate"] = (
            out.loc[eligible, "predicted_positive_rate"] / max_rate
        )
    return out


def fairness_differences(report: pd.DataFrame) -> dict[str, float | None]:
    """Return max-minus-min gaps over adequately sized, non-missing groups."""
    usable = report.loc[(~report["small_group_warning"]) & (report["group"] != "Missing")]
    result: dict[str, float | None] = {}
    for metric in RATE_METRICS:
        values = usable[metric].dropna()
        result[f"{metric}_max_min_gap"] = (
            float(values.max() - values.min()) if len(values) >= 2 else None
        )
    return result


def fairness_gap_bootstrap_ci(
    y_true, y_pred, groups, *, min_group_size: int = 100, n_bootstrap: int = 500,
    confidence_level: float = 0.95, random_state: int = 42,
) -> dict:
    """Estimate percentile CIs for max-min subgroup gaps by resampling within groups."""
    if n_bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replicates for interval estimation.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")
    y, pred, group = _as_aligned_arrays(y_true, y_pred, groups)
    group_values = []
    for value in sorted(group.unique()):
        mask = (group == value).to_numpy()
        if int(mask.sum()) >= min_group_size and value != "Missing":
            group_values.append((value, y[mask], pred[mask]))
    if len(group_values) < 2:
        return {
            "n_bootstrap": n_bootstrap, "confidence_level": confidence_level,
            "method": "within-group nonparametric percentile bootstrap",
            "groups_used": [name for name, _, _ in group_values],
            "metrics": {name: {"estimate": None, "ci_lower": None, "ci_upper": None}
                        for name in RATE_METRICS},
        }
    point_gaps = fairness_differences(group_fairness_report(y, pred, group, min_group_size))
    samples = {metric: [] for metric in RATE_METRICS}
    rng = np.random.default_rng(random_state)
    for _ in range(n_bootstrap):
        per_group = {metric: [] for metric in RATE_METRICS}
        for _, group_y, group_pred in group_values:
            indices = rng.integers(0, len(group_y), size=len(group_y))
            rates = _rates(group_y[indices], group_pred[indices])
            for metric in RATE_METRICS:
                per_group[metric].append(rates[metric])
        for metric, values in per_group.items():
            finite = np.asarray(values, dtype=float)
            finite = finite[np.isfinite(finite)]
            if len(finite) >= 2:
                samples[metric].append(float(finite.max() - finite.min()))
    alpha = 1.0 - confidence_level
    intervals = {}
    for metric in RATE_METRICS:
        values = np.asarray(samples[metric], dtype=float)
        if len(values):
            low, high = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
            intervals[metric] = {
                "estimate": point_gaps[f"{metric}_max_min_gap"],
                "ci_lower": float(low), "ci_upper": float(high),
            }
        else:
            intervals[metric] = {"estimate": None, "ci_lower": None, "ci_upper": None}
    return {
        "n_bootstrap": n_bootstrap, "confidence_level": confidence_level,
        "method": "within-group nonparametric percentile bootstrap",
        "groups_used": [name for name, _, _ in group_values], "metrics": intervals,
    }
