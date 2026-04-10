"""
src/features.py — Feature engineering and target-mean encoding.

Separated from preprocessing.py for clarity:
  - preprocessing.py handles cleaning / imputation
  - features.py handles numeric transforms and categorical encoding

CRITICAL invariant: all encoding maps are *always* learned from train_df only.
They are then applied to test_df, with a global-mean fallback for unseen categories.
"""

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TARGET_COL, ID_COL, DROP_COLS


# ---------------------------------------------------------------------------
# Numeric log-transforms
# ---------------------------------------------------------------------------

def transform_loan_amount_log(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Clip loan_amount at the 99th-percentile (train) then apply np.log1p.

    Why: loan_amount is right-skewed; large outliers can destabilise gradient
    descent inside the MLP.  Clipping before the log removes extreme outliers
    while the log compresses the remaining range.
    """
    col = "loan_amount"
    if col not in train_df.columns:
        return train_df, test_df

    cap = float(train_df[col].quantile(0.99))

    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df[col] = np.log1p(train_df[col].clip(upper=cap))
    test_df[col]  = np.log1p(test_df[col].clip(upper=cap))
    return train_df, test_df


def transform_applicant_income_log(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Clip applicant_income at the 99th-percentile (train) then apply np.log1p.

    Why: income distributions are universally right-skewed; the log transform
    brings them closer to normal, which the StandardScaler can then centre
    properly before MLP training.
    """
    col = "applicant_income"
    if col not in train_df.columns:
        return train_df, test_df

    cap = float(train_df[col].quantile(0.99))

    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df[col] = np.log1p(train_df[col].clip(upper=cap))
    test_df[col]  = np.log1p(test_df[col].clip(upper=cap))
    return train_df, test_df


def impute_and_transform_population(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Impute missing population with the train median, then apply np.log1p.

    Why: population is skewed and sparse; median imputation is robust to
    outliers, and the log transform compresses the long right tail.
    """
    col = "population"
    if col not in train_df.columns:
        return train_df, test_df

    median_val = float(train_df[col].median())
    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df[col] = np.log1p(train_df[col].fillna(median_val))
    test_df[col]  = np.log1p(test_df[col].fillna(median_val))
    return train_df, test_df


def impute_and_transform_minority_population(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Impute missing minority_population with train median, then log1p transform.

    Why: same rationale as population — skewed distribution with sparse values.
    """
    col = "minority_population"
    if col not in train_df.columns:
        return train_df, test_df

    median_val = float(train_df[col].median())
    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df[col] = np.log1p(train_df[col].fillna(median_val))
    test_df[col]  = np.log1p(test_df[col].fillna(median_val))
    return train_df, test_df


def impute_and_transform_hud_income(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Impute missing hud_median_family_income with train median, then log1p.

    Why: HUD income is a census-level income figure — right-skewed across
    tracts, and some tracts are missing entirely.
    """
    col = "hud_median_family_income"
    if col not in train_df.columns:
        return train_df, test_df

    median_val = float(train_df[col].median())
    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df[col] = np.log1p(train_df[col].fillna(median_val))
    test_df[col]  = np.log1p(test_df[col].fillna(median_val))
    return train_df, test_df


def impute_and_transform_owner_occupied_units(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Impute missing number_of_owner_occupied_units with train median, then log1p.

    Why: highly skewed count variable; median is preferred over mean for
    sparse count data.
    """
    col = "number_of_owner_occupied_units"
    if col not in train_df.columns:
        return train_df, test_df

    median_val = float(train_df[col].median())
    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df[col] = np.log1p(train_df[col].fillna(median_val))
    test_df[col]  = np.log1p(test_df[col].fillna(median_val))
    return train_df, test_df


# ---------------------------------------------------------------------------
# Target mean encodings
# ---------------------------------------------------------------------------

def _target_mean_encode(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    col: str,
    target_col: str = TARGET_COL,
) -> tuple:
    """
    Replace a categorical column with the mean target value per category,
    learned exclusively from train_df.

    Unseen categories in test_df receive the global mean target rate so that
    the model never encounters NaN in a feature it was trained on.

    Returns: (train_df, test_df) with `col` replaced by float mean-encoded values.
    """
    global_mean = float(train_df[target_col].mean())
    encoding_map = train_df.groupby(col)[target_col].mean().to_dict()

    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df[col] = train_df[col].map(encoding_map).fillna(global_mean)
    # Unseen categories in test fall back to global_mean
    test_df[col]  = test_df[col].map(encoding_map).fillna(global_mean)
    return train_df, test_df


def encode_race_with_target_mean(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Target mean encode applicant_race_name_1.

    Why: race is a high-cardinality categorical; target encoding collapses it
    to a single float representing approval probability per category, which is
    directly informative for the MLP without creating many dummy columns.

    CRITICAL: encoding map always learned from train — never recalculated on test.
    """
    col = "applicant_race_name_1"
    if col not in train_df.columns:
        return train_df, test_df
    return _target_mean_encode(train_df, test_df, col)


def encode_categorical_features_with_target_mean(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Apply target mean encoding to all remaining object-dtype columns
    (excluding TARGET_COL, ID_COL, and already-encoded columns).

    Why: a single function handles the long tail of categorical columns so
    that none accidentally survive as object dtype, which would crash the MLP.
    Unseen categories in test receive the global mean approval rate.
    """
    object_cols = [
        c for c in train_df.select_dtypes(include="object").columns
        if c not in {TARGET_COL, ID_COL}
    ]
    for col in object_cols:
        train_df, test_df = _target_mean_encode(train_df, test_df, col)
    return train_df, test_df


def encode_agency_with_target_mean(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Target mean encode the agency column.
    Missing agency values are filled with the modal agency before encoding.

    Why: agency has modest cardinality but high predictive power — different
    agencies have systematically different approval rates.
    """
    col = "agency"
    if col not in train_df.columns:
        return train_df, test_df

    # Fill missing agency before encoding to avoid NaN keys in the map
    modal_agency = str(train_df[col].mode().iloc[0])
    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df[col] = train_df[col].fillna(modal_agency)
    test_df[col]  = test_df[col].fillna(modal_agency)
    return _target_mean_encode(train_df, test_df, col)


def encode_census_tract_with_target_mean(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Target mean encode census_tract_number.

    Why: census tract is a fine-grained geographic identifier with hundreds of
    unique values; target encoding captures the local approval rate without
    creating hundreds of dummy columns.
    """
    col = "census_tract_number"
    if col not in train_df.columns:
        return train_df, test_df
    return _target_mean_encode(train_df, test_df, col)


def encode_county_with_target_mean(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Target mean encode county column.

    Why: county-level approval rates vary systematically; target encoding
    represents this as a single informative float.
    """
    col = "county"
    if col not in train_df.columns:
        return train_df, test_df
    return _target_mean_encode(train_df, test_df, col)


def encode_msamd_with_target_mean(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Target mean encode msamd (metropolitan statistical area / metropolitan division).

    Why: metro-area level lending patterns are a strong predictor; target
    encoding gives the model a compact representation of area-level approval rate.
    """
    col = "msamd"
    if col not in train_df.columns:
        return train_df, test_df
    return _target_mean_encode(train_df, test_df, col)


def encode_D_with_target_mean(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Target mean encode mystery column D.

    Why: D is a categorical column of unknown provenance.  Target encoding
    is the safest strategy — it avoids assumptions about cardinality and
    handles unseen values gracefully via the global mean fallback.
    """
    col = "D"
    if col not in train_df.columns:
        return train_df, test_df
    return _target_mean_encode(train_df, test_df, col)


# ---------------------------------------------------------------------------
# Feature column selection
# ---------------------------------------------------------------------------

def get_final_feature_columns(train_df: pd.DataFrame) -> list:
    """
    Return the definitive list of feature columns after all preprocessing and
    encoding steps have been applied.

    Drops: TARGET_COL, ID_COL, DROP_COLS, and any remaining object columns
    (object columns indicate an encoding step was missed — they would crash
    the MLP).

    Why: single source of truth for feature columns — both train.py and
    predict.py call this so the feature sets are always identical.
    """
    exclude = set([TARGET_COL, ID_COL] + DROP_COLS)
    feature_cols = [
        c for c in train_df.columns
        if c not in exclude and train_df[c].dtype != object
    ]
    return feature_cols


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def run_all_feature_engineering(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Apply all feature transforms and encodings in the correct order.

    Order matters:
      1. Log transforms on skewed numerics (before StandardScaler inside pipeline)
      2. Target mean encodings for all categoricals (must happen after cleaning)
      3. Bulk fallback encoding for any remaining object columns

    Returns feature-engineered (train_df, test_df).
    """
    # --- Numeric transforms ---
    train_df, test_df = transform_loan_amount_log(train_df, test_df)
    train_df, test_df = transform_applicant_income_log(train_df, test_df)
    train_df, test_df = impute_and_transform_population(train_df, test_df)
    train_df, test_df = impute_and_transform_minority_population(train_df, test_df)
    train_df, test_df = impute_and_transform_hud_income(train_df, test_df)
    train_df, test_df = impute_and_transform_owner_occupied_units(train_df, test_df)

    # --- Targeted categorical encodings ---
    train_df, test_df = encode_agency_with_target_mean(train_df, test_df)
    train_df, test_df = encode_race_with_target_mean(train_df, test_df)
    train_df, test_df = encode_census_tract_with_target_mean(train_df, test_df)
    train_df, test_df = encode_county_with_target_mean(train_df, test_df)
    train_df, test_df = encode_msamd_with_target_mean(train_df, test_df)
    train_df, test_df = encode_D_with_target_mean(train_df, test_df)

    # --- Bulk encoding of any remaining object columns ---
    train_df, test_df = encode_categorical_features_with_target_mean(train_df, test_df)

    return train_df, test_df
