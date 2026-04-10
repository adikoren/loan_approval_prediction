"""
src/evaluate.py — Post-training evaluation utilities.

Run after src/train.py to generate performance reports and diagnostic plots.
Results are saved to experiments/ so they are version-controllable alongside
the model artefact.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.pipeline import Pipeline


def evaluate(
    pipeline: Pipeline,
    X_val,
    y_val,
    class_names: list = None,
) -> dict:
    """
    Print a full sklearn classification report and return key metrics as a dict.

    Returned keys: accuracy, roc_auc, precision, recall, f1.

    Why return a dict?  train.py logs these values to the experiment log file
    so each run is reproducible without re-running evaluation manually.
    """
    if class_names is None:
        class_names = ["Rejected", "Approved"]

    y_pred      = pipeline.predict(X_val)
    y_prob      = pipeline.predict_proba(X_val)[:, 1]

    acc     = accuracy_score(y_val, y_pred)
    auc     = roc_auc_score(y_val, y_prob)
    prec    = precision_score(y_val, y_pred, zero_division=0)
    rec     = recall_score(y_val, y_pred, zero_division=0)
    f1      = f1_score(y_val, y_pred, zero_division=0)

    print("\n" + "=" * 55)
    print("  Evaluation Report")
    print("=" * 55)
    print(classification_report(y_val, y_pred, target_names=class_names))
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print("=" * 55 + "\n")

    return {
        "accuracy":  acc,
        "roc_auc":   auc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
    }


def plot_confusion_matrix(
    y_true,
    y_pred,
    save_path: str = "experiments/confusion_matrix.png",
    class_names: list = None,
) -> None:
    """
    Plot and save a seaborn heatmap of the confusion matrix.

    Why: accuracy alone is misleading for imbalanced classes.  The confusion
    matrix shows *where* the model errs — false approvals vs false rejections
    have very different real-world costs.
    """
    if class_names is None:
        class_names = ["Rejected", "Approved"]

    cm = confusion_matrix(y_true, y_pred)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Confusion matrix saved to {save_path}")


def plot_roc_curve(
    pipeline: Pipeline,
    X_val,
    y_val,
    save_path: str = "experiments/roc_curve.png",
) -> None:
    """
    Plot and save the ROC curve with AUC score annotated.

    Why: ROC-AUC is the primary metric for this task because the dataset is
    class-imbalanced.  The curve shows the trade-off between sensitivity and
    specificity across all decision thresholds — far more informative than
    a single accuracy number.
    """
    y_prob = pipeline.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    auc = roc_auc_score(y_val, y_prob)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] ROC curve saved to {save_path}")
