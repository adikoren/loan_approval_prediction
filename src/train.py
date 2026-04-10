"""
src/train.py — Training script.  Run once to produce experiments/model.joblib.

Usage:
    python src/train.py

What it does:
  1. Loads train.csv
  2. Runs the full preprocessing + feature engineering pipeline
  3. Splits into train / validation sets (stratified on target)
  4. Fits the MLP pipeline
  5. Evaluates on the validation set and prints metrics
  6. Saves the fitted pipeline and a JSON log to experiments/
"""

import os
import sys
import json
import datetime

import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure project root is on the path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    TRAIN_PATH,
    TARGET_COL,
    ID_COL,
    MODEL_PATH,
    LOG_DIR,
    TEST_SIZE,
    RANDOM_STATE,
)
from src.preprocessing import run_all_preprocessing
from src.features import run_all_feature_engineering, get_final_feature_columns
from src.model import build_pipeline, save_model
from src.evaluate import evaluate, plot_confusion_matrix, plot_roc_curve


def train() -> None:
    """
    Full end-to-end training routine.

    Steps:
      1. Load train.csv
      2. Preprocess (pass train as both args — no test set during training)
      3. Feature engineering
      4. Stratified train/val split — stratify=y preserves class distribution
         in both splits, which is critical for imbalanced datasets
      5. Build and fit MLP pipeline
      6. Evaluate on val split as a sanity check before saving
      7. Save model artefact + experiment log
    """
    print("[train] Loading data...")
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Training data not found at '{TRAIN_PATH}'.")

    train_raw = pd.read_csv(TRAIN_PATH, low_memory=False)
    print(f"[train] Loaded {len(train_raw):,} rows, {train_raw.shape[1]} columns.")

    # --- Preprocessing ---
    # Pass train_raw as both arguments: no separate test set exists at train time.
    # All imputation maps are learned from train_raw only.
    print("[train] Running preprocessing...")
    train_clean, _ = run_all_preprocessing(train_raw.copy(), train_raw.copy())

    # --- Feature engineering ---
    print("[train] Running feature engineering...")
    train_feat, _ = run_all_feature_engineering(train_clean, train_clean.copy())

    # --- Feature / target split ---
    feature_cols = get_final_feature_columns(train_feat)
    X = train_feat[feature_cols].values
    y = train_feat[TARGET_COL].values
    print(f"[train] {len(feature_cols)} feature columns selected.")

    # --- Stratified train / validation split ---
    # stratify=y ensures class proportions are identical in both splits.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"[train] Train size: {len(X_train):,}  |  Val size: {len(X_val):,}")

    # --- Build and fit pipeline ---
    print("[train] Fitting MLP pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # --- Evaluate before saving — never save without a sanity check ---
    print("[train] Evaluating on validation set...")
    metrics = evaluate(pipeline, X_val, y_val)

    # --- Plot diagnostics ---
    y_pred_val = pipeline.predict(X_val)
    plot_confusion_matrix(y_val, y_pred_val)
    plot_roc_curve(pipeline, X_val, y_val)

    # --- Save model ---
    save_model(pipeline, MODEL_PATH)

    # --- Log experiment ---
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(LOG_DIR, f"train_{timestamp}.json")
    log = {
        "timestamp":     timestamp,
        "train_rows":    len(X_train),
        "val_rows":      len(X_val),
        "n_features":    len(feature_cols),
        "feature_cols":  feature_cols,
        "metrics":       metrics,
        "model_path":    MODEL_PATH,
    }
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[train] Experiment log saved to {log_path}")
    print("[train] Done.")


if __name__ == "__main__":
    train()
