# config.py — Single source of truth for all hyperparameters and paths.
# Every other file imports from here. Never hardcode values elsewhere.

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
TRAIN_PATH     = "data/train.csv"
TEST_PATH      = "data/test.csv"
FULL_DATA_PATH = "data/fulldataset.csv"  # EDA ONLY — never import in src/ or app/
OUTPUT_PATH    = "data/predictions.csv"

# ---------------------------------------------------------------------------
# Model artifacts
# ---------------------------------------------------------------------------
MODEL_PATH = "experiments/model.joblib"
LOG_DIR    = "experiments/logs"

# ---------------------------------------------------------------------------
# Column roles
# ---------------------------------------------------------------------------
TARGET_COL = "label"
ID_COL     = "ID"

# Numeric features expected after preprocessing
NUMERIC_COLS = [
    "loan_amount", "applicant_income", "population",
    "minority_population", "hud_median_family_income",
    "tract_to_msamd_income", "number_of_owner_occupied_units",
    "A", "B", "C"
]

# Columns to drop entirely (not useful for modeling)
DROP_COLS = ["county_code", "loan_amount_bin"]

# Right-skewed numeric columns that benefit from log1p transform
LOG_TRANSFORM_COLS = [
    "loan_amount", "applicant_income", "population",
    "minority_population", "hud_median_family_income",
    "number_of_owner_occupied_units"
]

# ---------------------------------------------------------------------------
# MLP hyperparameters
# ---------------------------------------------------------------------------
MLP_HIDDEN_LAYERS = (100, 50)
MLP_ALPHA         = 0.0001
MLP_LEARNING_RATE = 0.01
MLP_MAX_ITER      = 1000
RANDOM_STATE      = 42

# ---------------------------------------------------------------------------
# Training / validation split
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2
CV_FOLDS  = 5

# ---------------------------------------------------------------------------
# Confidence band thresholds for prediction output
# ---------------------------------------------------------------------------
HIGH_CONFIDENCE_APPROVED = 0.75
HIGH_CONFIDENCE_REJECTED = 0.30

