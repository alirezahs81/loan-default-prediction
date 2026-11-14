"""Train, select, evaluate, and persist leakage-conscious loan-default pipelines."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DATA_PATH, EXCLUDED_COLUMNS, MODEL_PATH, RANDOM_STATE, REPORT_DIR, TEST_SIZE
from src.data import load_data
from src.fairness import fairness_differences, fairness_gap_bootstrap_ci, group_fairness_report

VALIDATION_FRACTION_OF_DEVELOPMENT = 0.25
BOOTSTRAP_REPLICATES = 500


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Fit imputation, scaling and encoding only inside each training partition."""
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ], remainder="drop", sparse_threshold=0)


def build_pipeline(X: pd.DataFrame, estimator) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor(X)), ("model", estimator)])


def candidate_models() -> dict:
    return {
        "LogisticRegression_balanced": LogisticRegression(
            max_iter=1500, class_weight="balanced", solver="lbfgs", random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=160, learning_rate=0.08, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=RANDOM_STATE,
        ),
    }


def evaluate(name: str, estimator: Pipeline, X_eval: pd.DataFrame, y_eval: pd.Series):
    probability = estimator.predict_proba(X_eval)[:, list(estimator.classes_).index(1)]
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_eval, prediction, labels=[0, 1]).ravel()
    metrics = {
        "model": name, "threshold": 0.5,
        "accuracy": float(accuracy_score(y_eval, prediction)),
        "precision_default": float(precision_score(y_eval, prediction, zero_division=0)),
        "recall_default": float(recall_score(y_eval, prediction, zero_division=0)),
        "f1_default": float(f1_score(y_eval, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_eval, probability)),
        "pr_auc": float(average_precision_score(y_eval, probability)),
        "brier_score": float(brier_score_loss(y_eval, probability)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    return metrics, probability, prediction


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if pd.isna(value):
        return None
    return value


def _write_calibration_and_threshold_reports(y_test, probability) -> None:
    observed, predicted = calibration_curve(y_test, probability, n_bins=10, strategy="quantile")
    pd.DataFrame({
        "bin": np.arange(1, len(observed) + 1),
        "mean_predicted_probability": predicted, "observed_default_rate": observed,
    }).to_csv(REPORT_DIR / "calibration_curve.csv", index=False)
    rows = []
    for threshold in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
        pred = (probability >= threshold).astype(int)
        rows.append({
            "threshold": threshold, "accuracy": accuracy_score(y_test, pred),
            "precision_default": precision_score(y_test, pred, zero_division=0),
            "recall_default": recall_score(y_test, pred, zero_division=0),
            "f1_default": f1_score(y_test, pred, zero_division=0),
            "predicted_default_rate": float(pred.mean()),
        })
    pd.DataFrame(rows).to_csv(REPORT_DIR / "threshold_analysis.csv", index=False)


def main(data_path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Use validation PR-AUC for selection and reserve test data for final evaluation."""
    data_path = Path(data_path).expanduser().resolve()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = load_data(data_path)
    y = frame["Status"].astype(int)
    excluded = [column for column in EXCLUDED_COLUMNS if column in frame.columns]
    X = frame.drop(columns=excluded)
    if X.shape[1] == 0:
        raise ValueError("No eligible model features remain after exclusions.")

    X_dev, X_test, y_dev, y_test, dev_idx, test_idx = train_test_split(
        X, y, frame.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    X_fit, X_val, y_fit, y_val, fit_idx, val_idx = train_test_split(
        X_dev, y_dev, dev_idx, test_size=VALIDATION_FRACTION_OF_DEVELOPMENT,
        random_state=RANDOM_STATE, stratify=y_dev,
    )

    results = []
    for name, estimator in candidate_models().items():
        pipe = build_pipeline(X_fit, estimator)
        pipe.fit(X_fit, y_fit)
        metrics, _, _ = evaluate(name, pipe, X_val, y_val)
        results.append({"model": name, **{
            f"validation_{k}": v for k, v in metrics.items() if k != "model"
        }})
    comparison = pd.DataFrame(results).sort_values(
        ["validation_pr_auc", "validation_roc_auc"], ascending=False
    ).reset_index(drop=True)
    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    # Demographic-feature ablation is diagnostic only and cannot select the primary model.
    ablation_excluded = [c for c in [
        "ID", "Status", "rate_of_interest", "Interest_rate_spread", "Upfront_charges",
    ] if c in frame.columns]
    X_full = frame.drop(columns=ablation_excluded)
    diagnostic = build_pipeline(
        X_full.loc[fit_idx],
        HistGradientBoostingClassifier(
            max_iter=160, learning_rate=0.08, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=RANDOM_STATE,
        ),
    )
    diagnostic.fit(X_full.loc[fit_idx], y_fit)
    diag_metrics, _, _ = evaluate(
        "HistGradientBoosting_with_demographics_diagnostic",
        diagnostic, X_full.loc[val_idx], y_val,
    )
    primary = comparison.loc[comparison["model"] == "HistGradientBoosting"].iloc[0].to_dict()
    diag = {"model": diag_metrics["model"], **{
        f"validation_{k}": v for k, v in diag_metrics.items() if k != "model"
    }}
    primary["model"] = "HistGradientBoosting_without_demographics"
    pd.DataFrame([primary, diag]).to_csv(
        REPORT_DIR / "sensitive_feature_ablation.csv", index=False,
    )

    best_name = str(comparison.iloc[0]["model"])
    final_pipeline = build_pipeline(X_dev, candidate_models()[best_name])
    final_pipeline.fit(X_dev, y_dev)
    test_metrics, probability, prediction = evaluate(best_name, final_pipeline, X_test, y_test)
    selected = comparison.loc[comparison["model"] == best_name].iloc[0]
    best_metrics = {
        **test_metrics, "validation_pr_auc": float(selected["validation_pr_auc"]),
        "validation_roc_auc": float(selected["validation_roc_auc"]),
        "model_selection_partition": "validation (20% of full dataset)",
        "final_evaluation_partition": "untouched test (20% of full dataset)",
    }

    numeric = X_dev.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X_dev.columns if c not in numeric]
    artifact = {
        "artifact_version": 2, "pipeline": final_pipeline,
        "feature_columns": X.columns.tolist(), "numeric_features": numeric,
        "categorical_features": categorical,
        "category_values": {
            c: sorted(X_dev[c].dropna().astype(str).unique().tolist()) for c in categorical
        },
        "excluded_columns": excluded, "target": "Status", "threshold": 0.5,
        "best_model_name": best_name,
        "status_semantics": "0=non-default, 1=default (corroborated, not source-owner verified)",
        "training_rows": int(len(X_dev)), "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)), "validation_selection_metric": "average precision (PR-AUC)",
        "random_state": RANDOM_STATE,
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    joblib.dump(artifact, MODEL_PATH)
    (REPORT_DIR / "best_model_metrics.json").write_text(
        json.dumps(_json_safe(best_metrics), indent=2), encoding="utf-8",
    )

    holdout = frame.loc[test_idx]
    gaps, intervals = {}, {}
    for column in ("Gender", "Region", "age"):
        if column in holdout:
            groups = holdout[column].reset_index(drop=True)
            report = group_fairness_report(y_test.reset_index(drop=True), prediction, groups)
            report.to_csv(REPORT_DIR / f"fairness_{column.lower()}.csv", index=False)
            gaps[column] = fairness_differences(report)
            intervals[column] = fairness_gap_bootstrap_ci(
                y_test.reset_index(drop=True), prediction, groups,
                min_group_size=100, n_bootstrap=BOOTSTRAP_REPLICATES, random_state=RANDOM_STATE,
            )
    (REPORT_DIR / "fairness_gaps.json").write_text(
        json.dumps(_json_safe(gaps), indent=2), encoding="utf-8",
    )
    (REPORT_DIR / "fairness_gap_confidence_intervals.json").write_text(
        json.dumps(_json_safe(intervals), indent=2), encoding="utf-8",
    )
    _write_calibration_and_threshold_reports(y_test, probability)

    metadata = {
        "dataset_file": data_path.name,
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "rows": int(len(frame)), "columns": int(frame.shape[1]),
        "duplicate_rows": int(frame.duplicated().sum()),
        "target_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "excluded_columns": excluded, "selected_model": best_name,
        "selection_metric": "validation average precision (PR-AUC)",
        "split_proportions": {
            "fit": round(len(X_fit) / len(frame), 4),
            "validation": round(len(X_val) / len(frame), 4),
            "test": round(len(X_test) / len(frame), 4),
        },
        "random_state": RANDOM_STATE, "versions": artifact["versions"],
    }
    (REPORT_DIR / "training_metadata.json").write_text(
        json.dumps(_json_safe(metadata), indent=2), encoding="utf-8",
    )
    print(f"Selected model (validation PR-AUC): {best_name}")
    print("Final untouched-test metrics:", json.dumps(_json_safe(best_metrics), indent=2))
    print(f"Saved model: {MODEL_PATH}")
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DATA_PATH), help="Path to Loan_Default.csv")
    main(parser.parse_args().data)
