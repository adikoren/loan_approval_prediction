"""
src/preprocessing.py — Data cleaning and imputation pipeline.

All functions follow the (train_df, test_df) -> (train_df, test_df) contract so that
train and test are always transformed identically and no information leaks from test
into the encoding/imputation logic.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def calculate_missing_percentage(df: pd.DataFrame) -> pd.Series:
    """
    Return the percentage of missing values per column.

    Why: first step before deciding imputation strategy — columns above the
    threshold will be dropped entirely, the rest need imputation.
    """
    return (df.isnull().sum() / len(df)) * 100


# ---------------------------------------------------------------------------
# Column-level cleaning
# ---------------------------------------------------------------------------

def drop_high_missing_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold: float = 0.5,
) -> tuple:
    """
    Drop columns where the fraction of missing values in *train* exceeds threshold.
    The same columns are dropped from test regardless of their missingness there.

    Why: columns with >50% missing add noise, not signal, and reliable imputation
    becomes impossible.
    """
    missing_frac = train_df.isnull().mean()
    cols_to_drop = missing_frac[missing_frac > threshold].index.tolist()

    train_df = train_df.drop(columns=cols_to_drop, errors="ignore")
    test_df  = test_df.drop(columns=cols_to_drop, errors="ignore")
    return train_df, test_df


def impute_and_filter_owner_occupancy(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Impute missing owner_occupancy values using property_type as context,
    then filter rows to recognised occupancy categories.

    Why: owner_occupancy has structured missingness tied to property type —
    using the modal value within each property_type group is far better than
    a global mode fill.
    """
    col = "owner_occupancy"

    if col not in train_df.columns:
        return train_df, test_df

    # Learn mode per property_type from train only
    mode_map = (
        train_df.dropna(subset=[col])
        .groupby("property_type")[col]
        .agg(lambda x: x.mode().iloc[0])
    )
    global_mode = str(train_df[col].mode().iloc[0])

    def _fill(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        mask = df[col].isnull()
        df.loc[mask, col] = df.loc[mask, "property_type"].map(mode_map).fillna(global_mode)
        return df

    train_df = _fill(train_df)
    test_df  = _fill(test_df)

    # Keep only the three recognised occupancy categories
    valid_values = {
        "Owner-occupied as a principal dwelling",
        "Not owner-occupied as a principal dwelling",
        "Not applicable",
    }
    train_df = train_df[train_df[col].isin(valid_values)].copy()
    # Do not filter test — we cannot drop rows we need to predict on
    return train_df, test_df


def encode_preapproval_feature(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Clean the preapproval column and binary-encode it:
      1 if preapproval was explicitly requested, 0 otherwise.

    Why: the raw column has multiple label variants for the same concept;
    collapsing to binary reduces noise and makes the feature usable as-is.
    """
    col = "preapproval"
    if col not in train_df.columns:
        return train_df, test_df

    def _encode(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[col] = df[col].fillna("unknown").astype(str).str.strip().str.lower()
        # Any label containing "requested" maps to 1; everything else to 0
        df[col] = df[col].apply(lambda v: 1 if "requested" in v else 0)
        return df

    train_df = _encode(train_df)
    test_df  = _encode(test_df)
    return train_df, test_df


def normalize_applicant_ethnicity(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Standardize inconsistent ethnicity label strings by lowercasing and stripping.

    Why: the raw data encodes the same concept in multiple string variants;
    normalisation prevents duplicate categories later.
    """
    col = "applicant_ethnicity"
    if col not in train_df.columns:
        return train_df, test_df

    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[col] = df[col].astype(str).str.strip().str.lower()
        return df

    train_df = _normalize(train_df)
    test_df  = _normalize(test_df)
    return train_df, test_df


def impute_applicant_ethnicity_by_race(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Fill missing applicant_ethnicity using applicant_race_name_1 as a proxy.

    Why: race and ethnicity are correlated in this dataset — using the modal
    ethnicity per race group is far better than global mode imputation.
    The mapping is always learned from train only.
    """
    eth_col  = "applicant_ethnicity"
    race_col = "applicant_race_name_1"

    if eth_col not in train_df.columns or race_col not in train_df.columns:
        return train_df, test_df

    # Learn mode mapping from train
    mode_map = (
        train_df.dropna(subset=[eth_col])
        .groupby(race_col)[eth_col]
        .agg(lambda x: x.mode().iloc[0])
    )
    global_mode = str(train_df[eth_col].mode().iloc[0])

    def _fill(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        mask = df[eth_col].isnull() | (df[eth_col].astype(str).str.strip() == "nan")
        df.loc[mask, eth_col] = (
            df.loc[mask, race_col].map(mode_map).fillna(global_mode)
        )
        return df

    train_df = _fill(train_df)
    test_df  = _fill(test_df)
    return train_df, test_df


def map_applicant_ethnicity_to_numeric(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Map normalised ethnicity strings to integer codes.

    Why: required before target-mean encoding in features.py — integer codes
    allow the encoding map to be stored as a simple dict.
    """
    col = "applicant_ethnicity"
    if col not in train_df.columns:
        return train_df, test_df

    mapping = {
        "hispanic or latino": 1,
        "not hispanic or latino": 2,
        "information not provided by applicant in mail, internet, or telephone application": 3,
        "not applicable": 4,
        "no co-applicant": 5,
    }

    def _map(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].map(mapping).fillna(3)  # default: not provided
        return df

    train_df = _map(train_df)
    test_df  = _map(test_df)
    return train_df, test_df


def clean_and_summarize_race_distribution(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Clean applicant_race_name_1, consolidating rare / verbose categories into 'Other'.

    Why: reduces cardinality so that target encoding in features.py produces
    stable estimates, and prevents the model from overfitting to categories that
    appear only a handful of times in training data.
    """
    col = "applicant_race_name_1"
    if col not in train_df.columns:
        return train_df, test_df

    # Categories to keep as-is (by lowercase substring match)
    keep = {
        "white", "black or african american", "asian",
        "american indian or alaska native",
        "information not provided", "not applicable", "no co-applicant"
    }

    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[col] = df[col].astype(str).str.strip()

        def _consolidate(val: str) -> str:
            val_str = str(val)
            v = val_str.lower()
            for k in keep:
                if k in v:
                    return val_str  # keep original casing for readability
            return "Other"

        df[col] = df[col].apply(_consolidate)
        return df

    train_df = _clean(train_df)
    test_df  = _clean(test_df)
    return train_df, test_df


def encode_co_applicant_ethnicity(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Normalize, impute, and numeric-encode co_applicant_ethnicity using the same
    logic applied to applicant_ethnicity.

    Why: co-applicant information adds signal for joint applications and should
    be treated consistently with the primary applicant fields.
    """
    col = "co_applicant_ethnicity"
    if col not in train_df.columns:
        return train_df, test_df

    mapping = {
        "hispanic or latino": 1,
        "not hispanic or latino": 2,
        "information not provided by applicant in mail, internet, or telephone application": 3,
        "not applicable": 4,
        "no co-applicant": 5,
    }

    def _encode(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].map(mapping).fillna(5)  # default: no co-applicant
        return df

    train_df = _encode(train_df)
    test_df  = _encode(test_df)
    return train_df, test_df


def encode_co_applicant_race(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Clean co_applicant_race_name_1, consolidating rare categories into 'Other'.

    Why: mirrors applicant race treatment for consistency; co-applicant race
    will receive target mean encoding in features.py.
    """
    col = "co_applicant_race_name_1"
    if col not in train_df.columns:
        return train_df, test_df

    keep = {
        "white", "black or african american", "asian",
        "american indian or alaska native",
        "information not provided", "not applicable", "no co-applicant"
    }

    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[col] = df[col].fillna("No co-applicant").astype(str).str.strip()

        def _consolidate(val: str) -> str:
            val_str = str(val)
            v = val_str.lower()
            for k in keep:
                if k in v:
                    return val_str
            return "Other"

        df[col] = df[col].apply(_consolidate)
        return df

    train_df = _clean(train_df)
    test_df  = _clean(test_df)
    return train_df, test_df


def impute_tract_to_msamd_income(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Impute missing tract_to_msamd_income using a geographic hierarchy:
      census_tract_number → county → global mean fallback.

    Why: a single global mean ignores geographic structure.  Census tract is
    the finest grain; falling back to county and then global mean handles the
    progressively coarser cases gracefully.  All maps are learned from train.
    """
    col = "tract_to_msamd_income"
    if col not in train_df.columns:
        return train_df, test_df

    # Build imputation maps from train
    tract_mean  = train_df.groupby("census_tract_number")[col].mean()
    county_mean = train_df.groupby("county")[col].mean()
    global_mean = float(train_df[col].mean())

    def _fill(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        mask = df[col].isnull()
        # Try census tract first
        df.loc[mask, col] = df.loc[mask, "census_tract_number"].map(tract_mean)
        mask = df[col].isnull()
        # Fall back to county
        df.loc[mask, col] = df.loc[mask, "county"].map(county_mean)
        # Final fallback: global mean
        df[col] = df[col].fillna(global_mean)
        return df

    train_df = _fill(train_df)
    test_df  = _fill(test_df)
    return train_df, test_df


def impute_lien_status(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Fill missing lien_status with the mode within each loan_type group.

    Why: lien status is structurally determined by loan type; within-group
    mode is more informative than the global mode.
    """
    col = "lien_status"
    if col not in train_df.columns:
        return train_df, test_df

    mode_map    = train_df.groupby("loan_type")[col].agg(lambda x: x.mode().iloc[0])
    global_mode = str(train_df[col].mode().iloc[0])

    def _fill(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        mask = df[col].isnull()
        df.loc[mask, col] = df.loc[mask, "loan_type"].map(mode_map).fillna(global_mode)
        return df

    train_df = _fill(train_df)
    test_df  = _fill(test_df)
    return train_df, test_df


def clean_and_categorize_applicant_sex(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Standardise applicant_sex labels to a small controlled vocabulary:
      Male | Female | Not provided | Not applicable | No co-applicant

    Why: raw data has inconsistent label strings for the same concept;
    standardisation prevents duplicate categories.
    """
    col = "applicant_sex"
    if col not in train_df.columns:
        return train_df, test_df

    def _clean(val: str) -> str:
        v = str(val).strip().lower()
        if "female" in v:
            return "Female"
        if "male" in v:
            return "Male"
        if "not applicable" in v:
            return "Not applicable"
        if "no co-applicant" in v:
            return "No co-applicant"
        return "Not provided"

    def _apply(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[col] = df[col].astype(str).apply(_clean)
        return df

    train_df = _apply(train_df)
    test_df  = _apply(test_df)
    return train_df, test_df


def clean_and_categorize_co_applicant_sex(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Standardise co_applicant_sex labels using the same logic as applicant_sex.

    Why: consistency between applicant and co-applicant fields is important
    for target encoding to produce comparable numeric values.
    """
    col = "co_applicant_sex"
    if col not in train_df.columns:
        return train_df, test_df

    def _clean(val: str) -> str:
        v = str(val).strip().lower()
        if "female" in v:
            return "Female"
        if "male" in v:
            return "Male"
        if "not applicable" in v:
            return "Not applicable"
        if "no co-applicant" in v:
            return "No co-applicant"
        return "Not provided"

    def _apply(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[col] = df[col].astype(str).apply(_clean)
        return df

    train_df = _apply(train_df)
    test_df  = _apply(test_df)
    return train_df, test_df


def impute_census_tract_using_train_mappings(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Fill missing census_tract_number in test using county → most common tract
    mappings learned from train.

    Why: census_tract has structured missingness that correlates with county;
    using test data to impute test would be leakage.
    """
    col = "census_tract_number"
    if col not in train_df.columns:
        return train_df, test_df

    county_to_tract = (
        train_df.dropna(subset=[col])
        .groupby("county")[col]
        .agg(lambda x: x.mode().iloc[0])
    )

    def _fill(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        mask = df[col].isnull()
        df.loc[mask, col] = df.loc[mask, "county"].map(county_to_tract)
        # Any still-missing tracts get a placeholder sentinel
        df[col] = df[col].fillna(-1)
        return df

    # Apply to both — train may also have rare missing values
    train_df = _fill(train_df)
    test_df  = _fill(test_df)
    return train_df, test_df


def impute_county_using_census_tract_mapping(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Fill missing county in test using census_tract_number → county mappings
    learned from train.

    Why: census tract is a finer geographic unit that uniquely identifies a
    county — using this avoids arbitrary global-mode imputation.
    """
    col = "county"
    if col not in train_df.columns:
        return train_df, test_df

    tract_to_county = (
        train_df.dropna(subset=[col])
        .groupby("census_tract_number")[col]
        .agg(lambda x: x.mode().iloc[0])
    )
    global_mode = str(train_df[col].mode().iloc[0])

    def _fill(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        mask = df[col].isnull()
        df.loc[mask, col] = (
            df.loc[mask, "census_tract_number"].map(tract_to_county).fillna(global_mode)
        )
        return df

    train_df = _fill(train_df)
    test_df  = _fill(test_df)
    return train_df, test_df


def impute_msamd_using_mappings(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Fill missing msamd using county → msamd mappings learned from train.

    Why: msamd (metropolitan statistical area) is determined by county;
    using train mappings prevents leakage.
    """
    col = "msamd"
    if col not in train_df.columns:
        return train_df, test_df

    county_to_msamd = (
        train_df.dropna(subset=[col])
        .groupby("county")[col]
        .agg(lambda x: x.mode().iloc[0])
    )
    global_mode = str(train_df[col].mode().iloc[0])

    def _fill(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        mask = df[col].isnull()
        df.loc[mask, col] = (
            df.loc[mask, "county"].map(county_to_msamd).fillna(global_mode)
        )
        return df

    train_df = _fill(train_df)
    test_df  = _fill(test_df)
    return train_df, test_df


def impute_abc_with_random_samples(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Fill missing values in columns A, B, C by sampling from their training
    distributions at random (with replacement).

    Why: A, B, C appear to be synthetic noise columns added to the dataset.
    Random sampling from the training distribution is the least-biased
    approach — it preserves distributional shape without introducing
    correlation with the target.
    """
    rng = np.random.default_rng(seed=42)

    for col in ["A", "B", "C"]:
        if col not in train_df.columns:
            continue

        # Collect non-null training values
        train_vals = train_df[col].dropna().values

        def _fill(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            mask = df[col].isnull()
            n_missing = mask.sum()
            if n_missing > 0:
                df.loc[mask, col] = rng.choice(train_vals, size=n_missing, replace=True)
            return df

        train_df = _fill(train_df)
        test_df  = _fill(test_df)

    return train_df, test_df


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def run_all_preprocessing(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Run every preprocessing step in the correct order and return clean DataFrames.

    Call this single function from train.py and predict.py to guarantee that
    the ordering is never accidentally changed.

    Steps applied:
      1. Drop high-missing columns
      2. Impute / filter owner_occupancy
      3. Encode preapproval
      4. Normalise applicant ethnicity
      5. Impute ethnicity by race
      6. Map ethnicity to numeric
      7. Clean race distribution
      8. Encode co-applicant ethnicity
      9. Encode co-applicant race
      10. Impute census tract (train mappings)
      11. Impute county from census tract
      12. Impute msamd from county
      13. Impute tract_to_msamd_income (geographic hierarchy)
      14. Impute lien_status by loan type
      15. Standardise applicant sex
      16. Standardise co-applicant sex
      17. Impute A, B, C with random samples
    """
    train_df, test_df = drop_high_missing_columns(train_df, test_df)
    train_df, test_df = impute_and_filter_owner_occupancy(train_df, test_df)
    train_df, test_df = encode_preapproval_feature(train_df, test_df)
    train_df, test_df = normalize_applicant_ethnicity(train_df, test_df)
    train_df, test_df = impute_applicant_ethnicity_by_race(train_df, test_df)
    train_df, test_df = map_applicant_ethnicity_to_numeric(train_df, test_df)
    train_df, test_df = clean_and_summarize_race_distribution(train_df, test_df)
    train_df, test_df = encode_co_applicant_ethnicity(train_df, test_df)
    train_df, test_df = encode_co_applicant_race(train_df, test_df)
    train_df, test_df = impute_census_tract_using_train_mappings(train_df, test_df)
    train_df, test_df = impute_county_using_census_tract_mapping(train_df, test_df)
    train_df, test_df = impute_msamd_using_mappings(train_df, test_df)
    train_df, test_df = impute_tract_to_msamd_income(train_df, test_df)
    train_df, test_df = impute_lien_status(train_df, test_df)
    train_df, test_df = clean_and_categorize_applicant_sex(train_df, test_df)
    train_df, test_df = clean_and_categorize_co_applicant_sex(train_df, test_df)
    train_df, test_df = impute_abc_with_random_samples(train_df, test_df)

    return train_df, test_df
