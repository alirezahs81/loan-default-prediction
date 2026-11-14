"""Dataset loading and target validation."""
import csv
from pathlib import Path

import pandas as pd

TARGET = "Status"


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the loan dataset and validate its binary target and basic structure."""
    data_path = Path(path).expanduser()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    with data_path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    duplicate_names = sorted({name for name in header if header.count(name) > 1})
    if duplicate_names:
        raise ValueError(f"Dataset contains duplicate column names: {duplicate_names}")

    frame = pd.read_csv(data_path)
    if frame.empty:
        raise ValueError("Dataset is empty.")
    if TARGET not in frame.columns:
        raise ValueError(f"Required target column {TARGET!r} is missing.")
    if frame[TARGET].isna().any():
        raise ValueError(f"Target column {TARGET!r} contains missing values.")

    target = pd.to_numeric(frame[TARGET], errors="coerce")
    if target.isna().any() or not target.isin([0, 1]).all():
        observed = frame[TARGET].dropna().unique().tolist()
        raise ValueError(f"{TARGET} must contain only binary values 0/1; found {observed[:10]}")
    frame[TARGET] = target.astype("int8")
    if frame[TARGET].nunique() != 2:
        raise ValueError("Both target classes (0 and 1) must be present.")
    return frame
