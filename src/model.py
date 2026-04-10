"""
src/model.py — Production MLP pipeline definition.

Defines the single model that ships to production.  Research comparisons
live in experiments/model_comparison.py — never here.
"""

import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MLP_HIDDEN_LAYERS,
    MLP_ALPHA,
    MLP_LEARNING_RATE,
    MLP_MAX_ITER,
    RANDOM_STATE,
)


def build_pipeline() -> Pipeline:
    """
    Build and return the untrained sklearn Pipeline for the MLP classifier.

    Pipeline stages:
      1. SimpleImputer(strategy='mean')  — catches any residual NaN that
         survived feature engineering (e.g. test rows with unseen categories)
      2. StandardScaler                  — zero-mean / unit-variance; critical
         for MLP because gradient updates are scale-sensitive
      3. MLPClassifier                   — two hidden layers, L2 regularisation

    Why Pipeline?  Fitting the scaler and imputer inside the pipeline guarantees
    they are always fitted only on training data and consistently applied to
    validation / test data — no leakage.

    Why MLP?  It achieved the best ROC-AUC in 5-fold stratified cross-validation
    across LogisticRegression, KNN, SVC, RandomForest, and MLP.
    See experiments/model_comparison.py for the full comparison.
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN_LAYERS,
            alpha=MLP_ALPHA,
            learning_rate_init=MLP_LEARNING_RATE,
            max_iter=MLP_MAX_ITER,
            random_state=RANDOM_STATE,
            early_stopping=True,       # halt training when val loss stops improving
            validation_fraction=0.1,   # 10% of training data held out for early stopping
        )),
    ])
    return pipeline


def save_model(pipeline: Pipeline, path: str) -> None:
    """
    Persist a fitted Pipeline to disk using joblib.

    Creates the parent directory if it does not already exist.

    Why joblib?  It is the standard sklearn serialisation format — handles
    numpy arrays and sparse matrices more efficiently than pickle.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"[model] Saved pipeline to {path}")


def load_model(path: str) -> Pipeline:
    """
    Load a fitted Pipeline from disk.

    Raises FileNotFoundError with a clear message if the file is missing so that
    app/main.py gives a useful startup error rather than a cryptic AttributeError.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at '{path}'. "
            "Run `python src/train.py` first to train and save the model."
        )
    pipeline = joblib.load(path)
    print(f"[model] Loaded pipeline from {path}")
    return pipeline
