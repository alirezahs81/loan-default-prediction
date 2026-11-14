# Dataset provenance and redistribution

The project archive's `Loan_Default.csv` appears to match the [Loan Default Dataset by yasserh on Kaggle](https://www.kaggle.com/datasets/yasserh/loan-default-dataset). The archive did not include a license file, a source-owner data dictionary, or explicit redistribution permission.

For that reason, the raw CSV is intentionally excluded from this public repository. Obtain the dataset from its source, review the current license and platform terms, and place the file locally at `data/Loan_Default.csv`. Do not redistribute it unless the source terms explicitly permit that use.

The project interprets `Status=0` as non-default and `Status=1` as default based on public descriptions of the dataset. This interpretation is corroborated, not independently verified against an authoritative source-owner data dictionary.

The source archive does not document event timestamps, feature availability times, sampling methodology, or a complete feature dictionary. Therefore, the project cannot certify that all remaining predictors were known at a particular prediction moment. See `fairness_and_limitations.md` for additional limitations.
