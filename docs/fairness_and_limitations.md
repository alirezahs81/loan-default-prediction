# Fairness, evaluation, and limitations

## Intended use

This repository is an educational machine-learning portfolio project. It is not a validated credit-risk product, lending policy, underwriting tool, or substitute for legal, compliance, model-risk, or domain review. Do not use its output alone to approve, deny, price, collect, or otherwise make decisions about a person.

## Data and target uncertainty

The source data appears to match the Kaggle Loan Default Dataset by yasserh. The supplied project archive did not include an authoritative data dictionary, timestamp definitions, or a clear redistribution license. The target interpretation (`Status=0` non-default and `Status=1` default) is corroborated by public descriptions but is not verified by source-owner metadata.

## Leakage controls and remaining uncertainty

The training workflow uses stratified fit, validation, and final test partitions. Preprocessing is fitted inside each training pipeline. Candidate models are selected using validation Average Precision (PR-AUC), then the selected estimator is refitted on the development partition and evaluated on the untouched test partition.

The primary model excludes the record identifier, direct demographic fields (`Gender`, `Region`, `age`), and pricing fields (`rate_of_interest`, `Interest_rate_spread`, `Upfront_charges`) whose timing is undocumented. This reduces known leakage risks but does not prove that all remaining fields are available before the target event. The data contains no reliable event timestamps or feature-lineage documentation, so temporal leakage cannot be fully ruled out.

## Fairness analysis

Subgroup reports are descriptive diagnostics on the held-out test set. The reported predicted-positive rate refers to predicted default, not a loan rejection or adverse-action rate. The reported ratio is therefore not the conventional adverse-impact ratio for credit approval.

Bootstrap confidence intervals quantify sampling variability within the observed test split. They do not correct for multiple comparisons, label bias, selection effects, distribution shift, measurement error, or uncertainty in group definitions. Removing direct demographic fields does not remove correlated proxy information.

Small groups and missing-value groups are flagged and excluded from max-minus-min gap summaries. Group metrics should be interpreted with sample sizes and uncertainty intervals, not as proof of fairness or unfairness.

## Model and threshold limitations

The 0.5 threshold is a transparent baseline, not an economically, legally, or operationally validated lending threshold. Precision, recall, calibration, and subgroup errors depend on the dataset and may change under distribution shift. High accuracy or precision does not establish suitability for individual decisions.

## Reproducibility

Run the training command locally to regenerate the model and reports. The dataset must be obtained separately and checked against its current source terms. Model artifacts created by joblib should only be loaded from trusted sources.
