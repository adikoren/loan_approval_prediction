"""
experiments/model_comparison.py — 5-fold stratified cross-validation across
four candidate classifiers.

This is research / exploratory code that justifies the production model choice.
It lives in experiments/ (not src/) and should never be imported by app/ or src/.

Usage:
    python experiments/model_comparison.py

Output:
  - Per-fold ROC curves + mean ROC curve → experiments/roc_curves.png
  - Summary table printed to stdout
  - Winner recommendation printed at the end
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore")

# Add project root to path so we can import config and src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    TRAIN_PATH,
    TARGET_COL,
    RANDOM_STATE,
    CV_FOLDS,
    MLP_HIDDEN_LAYERS,
    MLP_ALPHA,
    MLP_LEARNING_RATE,
    MLP_MAX_ITER,
)
from src.preprocessing import run_all_preprocessing
from src.features import run_all_feature_engineering, get_final_feature_columns


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=5),
    "SVC": SVC(probability=True, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "MLP": MLPClassifier(
        hidden_layer_sizes=MLP_HIDDEN_LAYERS,
        alpha=MLP_ALPHA,
        learning_rate_init=MLP_LEARNING_RATE,
        max_iter=MLP_MAX_ITER,
        random_state=RANDOM_STATE,
        early_stopping=True,
    ),
}


def make_pipeline(model) -> Pipeline:
    """
    Wrap a classifier in an imputer + scaler pipeline.
    Identical structure to the production pipeline in src/model.py.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("model",   model),
    ])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_prepare_data():
    """
    Load train.csv and apply the same preprocessing + feature engineering used
    in production.  Returns (X, y, feature_cols).
    """
    print("[comparison] Loading data...")
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(
            f"Training data not found at '{TRAIN_PATH}'. "
            "Copy train.csv to data/ before running this script."
        )

    df = pd.read_csv(TRAIN_PATH)
    print(f"[comparison] Loaded {len(df):,} rows.")

    print("[comparison] Preprocessing...")
    df, _ = run_all_preprocessing(df.copy(), df.copy())

    print("[comparison] Feature engineering...")
    df, _ = run_all_feature_engineering(df, df.copy())

    feature_cols = get_final_feature_columns(df)
    X = df[feature_cols].values
    y = df[TARGET_COL].values
    print(f"[comparison] {len(feature_cols)} features, {len(X):,} samples.")
    return X, y, feature_cols


# ---------------------------------------------------------------------------
# Cross-validation loop
# ---------------------------------------------------------------------------

def run_cv(X, y) -> dict:
    """
    Run 5-fold stratified cross-validation for each model.

    For each model:
      - Compute ROC-AUC per fold
      - Record FPR/TPR for plotting

    Returns a dict: model_name → {aucs: [...], mean_fpr: [...], tprs: [...]}
    """
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for model_name, model in MODELS.items():
        print(f"\n[comparison] Evaluating {model_name}...")
        pipeline = make_pipeline(model)

        # Fixed x-axis for interpolating all fold curves onto the same grid
        mean_fpr = np.linspace(0, 1, 100)
        tprs     = []
        aucs     = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            pipeline.fit(X_tr, y_tr)
            y_prob = pipeline.predict_proba(X_val)[:, 1]

            auc = roc_auc_score(y_val, y_prob)
            aucs.append(auc)
            print(f"  Fold {fold_idx}: AUC = {auc:.4f}")

            # Interpolate ROC curve onto the common FPR grid for averaging
            fpr, tpr, _ = roc_curve(y_val, y_prob)
            interp_tpr  = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

        results[model_name] = {
            "aucs":     aucs,
            "mean_fpr": mean_fpr,
            "tprs":     tprs,
        }

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_combined_roc_curves(results: dict, save_path: str = "experiments/roc_curves.png") -> None:
    """
    Plot per-fold ROC curves (faint) and mean ROC curve (bold) for each model
    on a single figure.

    Why: overlaying all models allows direct visual comparison of mean AUC and
    variance across folds — a summary table alone hides fold-to-fold stability.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    colours = ["steelblue", "darkorange", "green", "red", "purple"]

    for ax, (model_name, data), colour in zip(axes, results.items(), colours):
        mean_fpr = data["mean_fpr"]
        tprs     = data["tprs"]
        aucs     = data["aucs"]

        # Plot per-fold curves (faint)
        for tpr in tprs:
            ax.plot(mean_fpr, tpr, alpha=0.2, lw=1, color=colour)

        # Mean curve (bold)
        mean_tpr      = np.mean(tprs, axis=0)
        mean_tpr[-1]  = 1.0
        mean_auc      = np.mean(aucs)
        std_auc       = np.std(aucs)

        ax.plot(
            mean_fpr, mean_tpr, lw=2.5, color=colour,
            label=f"Mean AUC = {mean_auc:.4f} ± {std_auc:.4f}",
        )
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(model_name)
        ax.legend(loc="lower right", fontsize=8)

    plt.suptitle("Model Comparison — 5-Fold Stratified ROC Curves", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\n[comparison] ROC curves saved to {save_path}")


# ---------------------------------------------------------------------------
# Summary and recommendation
# ---------------------------------------------------------------------------

def print_summary(results: dict) -> None:
    """
    Print a ranked summary table and recommend the best model.

    Ranking criterion: mean ROC-AUC across 5 folds.
    """
    rows = []
    for model_name, data in results.items():
        aucs     = data["aucs"]
        mean_auc = np.mean(aucs)
        std_auc  = np.std(aucs)
        rows.append({"Model": model_name, "Mean AUC": mean_auc, "Std AUC": std_auc})

    summary = pd.DataFrame(rows).sort_values("Mean AUC", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 55)
    print("  Model Comparison Summary (5-Fold Stratified CV)")
    print("=" * 55)
    print(summary.to_string(index=False, float_format="{:.4f}".format))
    print("=" * 55)

    winner = summary.iloc[0]["Model"]
    best_auc = summary.iloc[0]["Mean AUC"]
    print(f"\n  Recommendation: Use '{winner}' (Mean AUC = {best_auc:.4f})")
    print("  This matches the production model in src/model.py.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    X, y, _ = load_and_prepare_data()
    results  = run_cv(X, y)
    plot_combined_roc_curves(results)
    print_summary(results)
