"""Validated inference using the exact fitted preprocessing/model pipeline."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import DATA_PATH, MODEL_PATH


def load_artifact(model_path: str | Path = MODEL_PATH) -> dict:
    """Load a trusted project artifact and validate its required structure."""
    path = Path(model_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Trained model not found at {path}. Run: python -m src.train")
    artifact = joblib.load(path)  # Only load artifacts from a trusted source.
    required = {"pipeline", "feature_columns"}
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise ValueError("Invalid model artifact format.")
    features = artifact["feature_columns"]
    if (not isinstance(features, list) or not features
            or not all(isinstance(feature, str) for feature in features)
            or len(features) != len(set(features))):
        raise ValueError("Model artifact contains an invalid feature schema.")
    numeric = set(artifact.get("numeric_features", []))
    categorical = set(artifact.get("categorical_features", []))
    if numeric & categorical or numeric | categorical != set(features):
        raise ValueError("Model artifact feature types do not match its feature schema.")
    if not callable(getattr(artifact["pipeline"], "predict_proba", None)):
        raise ValueError("Saved pipeline must implement predict_proba().")
    return artifact


def _validate_input(input_data: dict, artifact: dict) -> tuple[pd.DataFrame, list[str]]:
    if not isinstance(input_data, dict) or not input_data:
        raise ValueError("input_data must be a non-empty dictionary.")
    if not all(isinstance(key, str) for key in input_data):
        raise ValueError("All input feature names must be strings.")
    expected = artifact["feature_columns"]
    explicitly_ignored = set(artifact.get("excluded_columns", []))
    unknown = sorted(set(input_data) - set(expected) - explicitly_ignored)
    if unknown:
        raise ValueError(f"Unexpected input fields: {unknown}")
    missing = [column for column in expected if column not in input_data]
    if missing:
        raise ValueError(f"Missing required model inputs: {missing}")
    frame = pd.DataFrame([{column: input_data[column] for column in expected}], columns=expected)
    numeric_features = set(artifact.get("numeric_features", []))
    categorical_features = set(artifact.get("categorical_features", []))
    for column in numeric_features:
        raw_value = frame.at[0, column]
        if isinstance(raw_value, (dict, list, tuple, set)):
            raise ValueError(f"Feature {column!r} must be a scalar numeric value.")
        original = frame[column]
        numeric = pd.to_numeric(original, errors="coerce")
        if (original.notna() & numeric.isna()).any():
            raise ValueError(f"Feature {column!r} must contain numeric values or missing values.")
        if np.isinf(numeric.dropna().to_numpy(dtype=float)).any():
            raise ValueError(f"Feature {column!r} cannot contain positive or negative infinity.")
        frame[column] = numeric
    warnings = []
    known_categories = artifact.get("category_values", {})
    for column in categorical_features:
        value = frame.at[0, column]
        if isinstance(value, (dict, list, tuple, set)):
            raise ValueError(f"Feature {column!r} must be a scalar category value.")
        if value is None or pd.isna(value):
            continue
        value = str(value)
        frame.at[0, column] = value
        known = known_categories.get(column)
        if known is not None and value not in known:
            warnings.append(f"Unseen category {value!r} for {column!r}; the encoder will ignore it.")
    return frame, warnings


def predict_loan_default(input_data: dict, model_path: str | Path = MODEL_PATH) -> dict:
    """Return a binary prediction and class probabilities for one record."""
    artifact = load_artifact(model_path)
    frame, warnings = _validate_input(input_data, artifact)
    pipe = artifact["pipeline"]
    classes = list(getattr(pipe, "classes_", []))
    if 1 not in classes:
        raise ValueError("Saved pipeline was not fitted with the default class (1).")
    probability = float(pipe.predict_proba(frame)[0, classes.index(1)])
    threshold = float(artifact.get("threshold", 0.5))
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Model artifact contains an invalid classification threshold.")
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("Model pipeline returned an invalid probability.")
    prediction = int(probability >= threshold)
    return {
        "prediction": prediction,
        "label": "default" if prediction else "non-default",
        "default_probability": round(probability, 6),
        "non_default_probability": round(1.0 - probability, 6),
        "threshold": threshold,
        "risk_band": "higher estimated risk" if prediction else "lower estimated risk",
        "model": artifact.get("best_model_name"),
        "warnings": warnings,
    }


if __name__ == "__main__":
    frame = pd.read_csv(DATA_PATH)
    example = frame.drop(columns=["Status"]).iloc[0].to_dict()
    print(predict_loan_default(example))
