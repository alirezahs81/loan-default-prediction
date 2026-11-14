# Loan Default Prediction

An educational end-to-end classification project for exploring historical loan-default patterns. It includes exploratory analysis, a leakage-conscious train/validation/test workflow, subgroup error analysis with bootstrap uncertainty estimates, validated inference, a Streamlit demonstration, automated tests, and GitHub Actions CI.

> **Responsible-use notice:** This is an educational portfolio prototype, not a credit decision system. Do not use its output alone to approve, deny, price, collect, or otherwise make decisions about an individual.

## Dataset

The project archive's CSV appears to match the [Loan Default Dataset by yasserh on Kaggle](https://www.kaggle.com/datasets/yasserh/loan-default-dataset), with 148,670 rows and 34 columns. The archive did not include a clear redistribution license or authoritative source-owner data dictionary. The raw CSV is therefore **not included** in this public repository.

Obtain the dataset from its source, review the current license and terms, then place it at `data/Loan_Default.csv`. See [dataset provenance](docs/dataset_provenance.md). The target is interpreted as `Status=0` non-default and `Status=1` default based on public descriptions; this remains corroborated rather than source-owner verified.

## Setup

Python 3.11–3.13 is supported by the CI matrix.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place the licensed dataset at `data/Loan_Default.csv`, then train:

```bash
python -m src.train
```

This generates the fitted pipeline under `models/` and aggregate evaluation reports under `reports/`. These generated artifacts are ignored by Git by default. To run the demo after training:

```bash
streamlit run src/app.py
```

Run tests and lint:

```bash
python -m pytest -q
ruff check src tests
```

## Methodology

- Stratified 60% fit / 20% validation / 20% final test split, with fixed random seed 42.
- Imputation, scaling, and one-hot encoding are fitted within the training pipeline.
- Balanced Logistic Regression and HistGradientBoosting are compared on validation data.
- Model selection uses validation Average Precision (PR-AUC); the test set is reserved for final evaluation.
- The selected model is refitted on the full 80% development partition before one final test evaluation.
- The primary model excludes `ID`, direct demographic fields (`Gender`, `Region`, `age`), and pricing fields with undocumented timing (`rate_of_interest`, `Interest_rate_spread`, `Upfront_charges`).

This is a conservative exclusion policy, not proof that all remaining features are available before the target event. No authoritative feature timestamps or full data dictionary were supplied, so complete prevention of temporal leakage cannot be certified.

## Prior local evaluation snapshot

The repaired project archive previously reported the following held-out test results for HistGradientBoosting at the baseline threshold 0.5. Re-run training locally against the exact source data and installed dependency versions to reproduce and verify these values; this public repository does not ship the raw data or fitted binary model.

| Metric | Reported value |
|---|---:|
| Accuracy | 90.05% |
| Precision (default) | 93.71% |
| Recall (default) | 63.89% |
| F1 (default) | 75.98% |
| ROC-AUC | 89.77% |
| PR-AUC / Average Precision | 84.75% |
| Brier score | 0.0809 |

At this threshold, recall indicates that a meaningful share of actual defaults are missed. The 0.5 threshold is a descriptive baseline, not a validated lending threshold.

## Fairness and limitations

Subgroup diagnostics examine predicted-default rates and error rates for Gender, Region, and age bands on the held-out test set. These are descriptive measures; predicted default is not a loan rejection or adverse-action decision. Bootstrap intervals quantify sampling uncertainty only and do not establish fairness. See [fairness and limitations](docs/fairness_and_limitations.md).

## Repository contents

- `src/train.py`: split, train, select, evaluate, and report.
- `src/predict.py`: schema-validated single-record inference.
- `src/fairness.py`: subgroup metrics and bootstrap intervals.
- `src/app.py`: Streamlit demonstration.
- `tests/`: automated tests.
- `.github/workflows/ci.yml`: tests across Python versions and Ruff lint.

