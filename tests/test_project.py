import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.data import load_data
from src.fairness import fairness_differences, fairness_gap_bootstrap_ci, group_fairness_report
from src.predict import predict_loan_default
from src.train import build_preprocessor, _threshold_analysis_frame


def test_loader_rejects_missing_file():
    with pytest.raises(FileNotFoundError):
        load_data("not-a-real-file.csv")


def test_loader_validates_binary_target(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"Status": [0, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="binary"):
        load_data(path)


def test_loader_rejects_duplicate_columns(tmp_path):
    path = tmp_path / "duplicate.csv"
    path.write_text("Status,Status\n0,0\n1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate column"):
        load_data(path)


def test_preprocessor_handles_missing_and_unknown_categories():
    frame = pd.DataFrame({"amount": [1.0, np.nan, 3.0], "kind": ["a", "b", None]})
    preprocessor = build_preprocessor(frame)
    transformed = preprocessor.fit_transform(frame)
    unknown = preprocessor.transform(pd.DataFrame({"amount": [2.0], "kind": ["new"]}))
    assert transformed.shape[0] == 3
    assert unknown.shape[1] == transformed.shape[1]


def _dummy_artifact(path):
    model = DummyClassifier(strategy="prior")
    X = pd.DataFrame({"x": [0, 1, 2, 3], "kind": ["a", "a", "b", "b"]})
    model.fit(X, [0, 0, 1, 1])
    joblib.dump({
        "pipeline": model, "feature_columns": ["x", "kind"],
        "numeric_features": ["x"], "categorical_features": ["kind"],
        "category_values": {"kind": ["a", "b"]},
        "excluded_columns": ["ID", "Status"], "threshold": 0.5,
        "best_model_name": "dummy",
    }, path)


def test_prediction_validates_schema_and_returns_probability(tmp_path):
    path = tmp_path / "model.joblib"
    _dummy_artifact(path)
    result = predict_loan_default({"x": 5, "kind": "a", "ID": 7}, path)
    assert 0 <= result["default_probability"] <= 1
    with pytest.raises(ValueError, match="Missing required"):
        predict_loan_default({"x": 5}, path)
    with pytest.raises(ValueError, match="Unexpected input fields"):
        predict_loan_default({"x": 5, "kind": "a", "oops": 1}, path)


def test_prediction_rejects_invalid_numeric_and_warns_on_unknown(tmp_path):
    path = tmp_path / "model.joblib"
    _dummy_artifact(path)
    with pytest.raises(ValueError, match="numeric"):
        predict_loan_default({"x": "bad", "kind": "a"}, path)
    with pytest.raises(ValueError, match="infinity"):
        predict_loan_default({"x": float("inf"), "kind": "a"}, path)
    result = predict_loan_default({"x": 2, "kind": "new"}, path)
    assert len(result["warnings"]) == 1


def test_fairness_report_and_gap_exclude_small_or_missing_groups():
    report = group_fairness_report(
        [0, 1, 0, 1, 1], [0, 1, 1, 0, 1], ["A", "A", "B", "B", "Missing"],
        min_group_size=2,
    )
    assert "TPR" in report.columns
    assert report.loc[report.group == "Missing", "small_group_warning"].iloc[0]
    assert fairness_differences(report)["TPR_max_min_gap"] is not None


def test_bootstrap_intervals_reproducible():
    y = np.array([0, 1] * 100)
    pred = np.array([0, 1, 1, 1] * 50)
    groups = np.array(["A"] * 100 + ["B"] * 100)
    first = fairness_gap_bootstrap_ci(
        y, pred, groups, min_group_size=50, n_bootstrap=100, random_state=7
    )
    second = fairness_gap_bootstrap_ci(
        y, pred, groups, min_group_size=50, n_bootstrap=100, random_state=7
    )
    assert first == second
    for metric in first["metrics"].values():
        assert 0 <= metric["ci_lower"] <= metric["ci_upper"] <= 1



def test_threshold_analysis_uses_the_supplied_validation_labels_and_probabilities():
    validation_labels = np.array([0, 1, 1, 0])
    validation_probabilities = np.array([0.1, 0.8, 0.6, 0.4])

    report = _threshold_analysis_frame(
        validation_labels, validation_probabilities,
    )

    baseline = report.loc[report.threshold == 0.5].iloc[0]
    assert baseline["accuracy"] == 1.0
    assert baseline["precision_default"] == 1.0
    assert baseline["recall_default"] == 1.0
    assert baseline["f1_default"] == 1.0
    assert len(report) == 7
