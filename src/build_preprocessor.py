"""
src/build_preprocessor.py — Fit and save experiments/preprocessor.joblib.

Run this script once after training to bake all preprocessing statistics
from train.csv into a compact artifact.  The API server loads this artifact
at startup instead of loading train.csv, keeping memory well under 512 MB.

Usage:
    python src/build_preprocessor.py
"""

import os
import sys

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessor import PreprocessorFitter

TRAIN_PATH = "data/train.csv"
OUTPUT_PATH = "experiments/preprocessor.joblib"


def build():
    print(f"[build_preprocessor] Loading {TRAIN_PATH}...")
    train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
    print(f"[build_preprocessor] Fitting PreprocessorFitter on {len(train_df):,} rows...")
    fitter = PreprocessorFitter(train_df)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    joblib.dump(fitter, OUTPUT_PATH)
    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"[build_preprocessor] Saved to {OUTPUT_PATH}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    build()
