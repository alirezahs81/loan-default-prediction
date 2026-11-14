from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Loan_Default.csv"
MODEL_PATH = ROOT / "models" / "loan_default_pipeline.joblib"
REPORT_DIR = ROOT / "reports"
TARGET = "Status"
ID_COLUMNS = ["ID"]
SENSITIVE_COLUMNS = ["Gender", "Region", "age"]
# Exclude direct demographics and pricing fields with undocumented timing.
EXCLUDED_COLUMNS = ["ID", "Status", "Gender", "Region", "age",
                    "rate_of_interest", "Interest_rate_spread", "Upfront_charges"]
RANDOM_STATE = 42
TEST_SIZE = 0.20
