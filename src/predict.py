"""
src/predict.py — Generate final predictions on test.csv.

Produces: data/predictions.csv  (ID + Prediction columns)

Usage:
    python src/predict.py

CRITICAL contract:
  - train.csv is always loaded alongside test.csv
  - All encoding maps are learned from train — never from test
  - We never look at test labels (there are none; this is the submission file)
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    TRAIN_PATH,
    TEST_PATH,
    OUTPUT_PATH,
    TARGET_COL,
    ID_COL,
    MODEL_PATH,
)
from src.preprocessing import run_all_preprocessing
from src.features import run_all_feature_engineering, get_final_feature_columns
from src.model import load_model


# ---------------------------------------------------------------------------
# Single-row inference (used by app/main.py to serve live /predict requests)
# ---------------------------------------------------------------------------

def build_inference_frame(features: dict, train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a one-row DataFrame for a live prediction request that lines up
    with train_df's raw columns AND dtypes.

    Why this matters: constructing `pd.DataFrame([a_dict])` directly from a
    JSON request produces `object`-dtype columns (because most values are
    None for fields the caller didn't supply). Passing that straight into
    run_all_preprocessing / run_all_feature_engineering breaks numeric ops
    like np.log1p, which require a real float64 column. Casting every column
    to train_df's dtype up front makes the single request row behave exactly
    like a row read out of test.csv.
    """
    exclude = {TARGET_COL, ID_COL}
    raw_cols = [c for c in train_df.columns if c not in exclude]

    row = {c: features.get(c) for c in raw_cols}
    df = pd.DataFrame([row])

    for c in raw_cols:
        try:
            df[c] = df[c].astype(train_df[c].dtype)
        except (ValueError, TypeError):
            # Fall back to plain object dtype rather than fail the request
            df[c] = df[c].astype(object)

    return df


def predict_single(features: dict, train_df: pd.DataFrame, pipeline) -> float:
    """
    Run one applicant's raw feature dict through the exact same preprocessing
    + feature engineering pipeline used at training time, and return the
    model's approval probability.

    `train_df` is passed in (rather than reloaded from disk) so the caller —
    app/main.py — can load train.csv once at process startup and reuse it
    across requests instead of re-reading a ~300k row CSV every call.
    """
    test_df = build_inference_frame(features, train_df)

    train_clean, test_clean = run_all_preprocessing(train_df.copy(), test_df)
    train_feat, test_feat = run_all_feature_engineering(train_clean, test_clean)

    feature_cols = get_final_feature_columns(train_feat)
    available_cols = [c for c in feature_cols if c in test_feat.columns]

    X = test_feat[available_cols].values
    prob = pipeline.predict_proba(X)[0][1]
    return float(prob)


def predict() -> None:
    """
    End-to-end prediction on unseen test data.

    Steps:
      1. Load train.csv AND test.csv — train is required so that all
         preprocessing and encoding maps are learned only from train
      2. Apply identical preprocessing to both
      3. Apply identical feature engineering to both
      4. Extract feature columns (same set as training)
      5. Load the saved model pipeline
      6. Predict approval probability on test set
      7. Write ID + binary Prediction to OUTPUT_PATH

    Why load train alongside test?
      Target mean encodings in features.py require the train labels to compute
      category-level approval rates.  If we passed test alone, those encodings
      would have no signal to learn from.
    """
    print("[predict] Loading data...")
    for path in [TRAIN_PATH, TEST_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required data file not found: '{path}'")

    train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
    test_df  = pd.read_csv(TEST_PATH,  low_memory=False)
    print(f"[predict] Train: {len(train_df):,} rows  |  Test: {len(test_df):,} rows")

    # --- Preprocessing ---
    print("[predict] Running preprocessing...")
    train_clean, test_clean = run_all_preprocessing(train_df.copy(), test_df.copy())

    # --- Feature engineering ---
    print("[predict] Running feature engineering...")
    train_feat, test_feat = run_all_feature_engineering(train_clean, test_clean)

    # --- Feature selection: must match the training feature set exactly ---
    # get_final_feature_columns uses train to determine the column list.
    feature_cols = get_final_feature_columns(train_feat)

    # Keep only columns that actually exist in test_feat (some may be absent)
    available_cols = [c for c in feature_cols if c in test_feat.columns]
    if len(available_cols) < len(feature_cols):
        missing = set(feature_cols) - set(available_cols)
        print(f"[predict] Warning: {len(missing)} feature(s) missing from test: {missing}")

    X_test = test_feat[available_cols].values

    # --- Load model ---
    pipeline = load_model(MODEL_PATH)

    # --- Predict ---
    print("[predict] Generating predictions...")
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions   = (probabilities >= 0.5).astype(int)

    # --- Save results ---
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    results = pd.DataFrame({
        ID_COL:       test_df[ID_COL].values,
        "Prediction": predictions,
    })
    results.to_csv(OUTPUT_PATH, index=False)
    print(f"[predict] Saved {len(results):,} predictions to {OUTPUT_PATH}")
    print(f"[predict] Approval rate: {predictions.mean():.2%}")
    print("[predict] Done.")


if __name__ == "__main__":
    predict()
