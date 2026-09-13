"""
src/preprocessor.py — Pre-fitted preprocessor for low-memory inference.

At startup the server loads experiments/preprocessor.joblib (produced by
src/build_preprocessor.py).  The PreprocessorFitter object bakes in every
encoding map / statistic computed from train.csv so that single-row inference
no longer requires loading the ~500 MB raw CSV.
"""

import numpy as np
import pandas as pd

from config import TARGET_COL, ID_COL


class PreprocessorFitter:
    """
    Fits all preprocessing and feature-engineering transforms from train_df,
    stores only the lightweight statistics needed for single-row transform,
    and exposes transform_single() for inference.
    """

    def __init__(self, train_df: pd.DataFrame):
        self.raw_cols = [c for c in train_df.columns if c not in {TARGET_COL, ID_COL}]
        self.raw_dtypes = {c: str(train_df[c].dtype) for c in self.raw_cols}

        # Step 1: drop_high_missing_columns
        missing_frac = train_df.isnull().mean()
        self.dropped_cols = missing_frac[missing_frac > 0.5].index.tolist()
        df = train_df.drop(columns=self.dropped_cols, errors="ignore")

        # Step 2: impute_and_filter_owner_occupancy
        col = "owner_occupancy"
        self.owner_occ_mode_map = (
            df.dropna(subset=[col])
            .groupby("property_type")[col]
            .agg(lambda x: x.mode().iloc[0])
            .to_dict()
        )
        self.owner_occ_global_mode = str(df[col].mode().iloc[0])
        mask = df[col].isnull()
        df.loc[mask, col] = df.loc[mask, "property_type"].map(self.owner_occ_mode_map).fillna(self.owner_occ_global_mode)
        valid_values = {
            "Owner-occupied as a principal dwelling",
            "Not owner-occupied as a principal dwelling",
            "Not applicable",
        }
        df = df[df[col].isin(valid_values)].copy()

        # Step 3: encode_preapproval_feature
        col = "preapproval"
        df[col] = df[col].fillna("unknown").astype(str).str.strip().str.lower().apply(lambda v: 1 if "requested" in v else 0)

        # Step 4: normalize_applicant_ethnicity
        col = "applicant_ethnicity"
        df[col] = df[col].astype(str).str.strip().str.lower()

        # Step 5: impute_applicant_ethnicity_by_race
        eth_col = "applicant_ethnicity"
        race_col = "applicant_race_name_1"
        self.eth_by_race_map = (
            df.dropna(subset=[eth_col])
            .groupby(race_col)[eth_col]
            .agg(lambda x: x.mode().iloc[0])
            .to_dict()
        )
        self.eth_global_mode = str(df[eth_col].mode().iloc[0])
        mask = df[eth_col].isnull() | (df[eth_col].astype(str).str.strip() == "nan")
        df.loc[mask, eth_col] = df.loc[mask, race_col].map(self.eth_by_race_map).fillna(self.eth_global_mode)

        # Step 6: map_applicant_ethnicity_to_numeric
        self.eth_numeric_map = {
            "hispanic or latino": 1,
            "not hispanic or latino": 2,
            "information not provided by applicant in mail, internet, or telephone application": 3,
            "not applicable": 4,
            "no co-applicant": 5,
        }
        df[eth_col] = df[eth_col].astype(str).str.strip().str.lower().map(self.eth_numeric_map).fillna(3)

        # Step 7: clean_and_summarize_race_distribution
        self.keep_races = {
            "white", "black or african american", "asian",
            "american indian or alaska native",
            "information not provided", "not applicable", "no co-applicant",
        }

        def _clean_race(val):
            v = str(val).lower()
            for k in self.keep_races:
                if k in v:
                    return str(val)
            return "Other"

        df[race_col] = df[race_col].astype(str).str.strip().apply(_clean_race)

        # Step 8: encode_co_applicant_ethnicity
        col = "co_applicant_ethnicity"
        df[col] = df[col].astype(str).str.strip().str.lower().map(self.eth_numeric_map).fillna(5)

        # Step 9: encode_co_applicant_race
        col = "co_applicant_race_name_1"
        df[col] = df[col].fillna("No co-applicant").astype(str).str.strip().apply(_clean_race)

        # Step 10: impute_census_tract_using_train_mappings
        col = "census_tract_number"
        self.county_to_tract_map = (
            df.dropna(subset=[col]).groupby("county")[col].agg(lambda x: x.mode().iloc[0]).to_dict()
        )
        mask = df[col].isnull()
        df.loc[mask, col] = df.loc[mask, "county"].map(self.county_to_tract_map).fillna(-1)

        # Step 11: impute_county_using_census_tract_mapping
        col = "county"
        self.tract_to_county_map = (
            df.dropna(subset=[col]).groupby("census_tract_number")[col].agg(lambda x: x.mode().iloc[0]).to_dict()
        )
        self.county_global_mode = str(df[col].mode().iloc[0])
        mask = df[col].isnull()
        df.loc[mask, col] = df.loc[mask, "census_tract_number"].map(self.tract_to_county_map).fillna(self.county_global_mode)

        # Step 12: impute_msamd_using_mappings
        col = "msamd"
        self.county_to_msamd_map = (
            df.dropna(subset=[col]).groupby("county")[col].agg(lambda x: x.mode().iloc[0]).to_dict()
        )
        self.msamd_global_mode = str(df[col].mode().iloc[0])
        mask = df[col].isnull()
        df.loc[mask, col] = df.loc[mask, "county"].map(self.county_to_msamd_map).fillna(self.msamd_global_mode)

        # Step 13: impute_tract_to_msamd_income
        col = "tract_to_msamd_income"
        self.tmi_tract_mean = df.groupby("census_tract_number")[col].mean().to_dict()
        self.tmi_county_mean = df.groupby("county")[col].mean().to_dict()
        self.tmi_global_mean = float(df[col].mean())
        mask = df[col].isnull()
        df.loc[mask, col] = df.loc[mask, "census_tract_number"].map(self.tmi_tract_mean)
        mask = df[col].isnull()
        df.loc[mask, col] = df.loc[mask, "county"].map(self.tmi_county_mean)
        df[col] = df[col].fillna(self.tmi_global_mean)

        # Step 14: impute_lien_status
        col = "lien_status"
        self.lien_mode_map = df.groupby("loan_type")[col].agg(lambda x: x.mode().iloc[0]).to_dict()
        self.lien_global_mode = str(df[col].mode().iloc[0])
        mask = df[col].isnull()
        df.loc[mask, col] = df.loc[mask, "loan_type"].map(self.lien_mode_map).fillna(self.lien_global_mode)

        # Steps 15 & 16: sex columns
        def _clean_sex(val):
            v = str(val).strip().lower()
            if "female" in v: return "Female"
            if "male" in v: return "Male"
            if "not applicable" in v: return "Not applicable"
            if "no co-applicant" in v: return "No co-applicant"
            return "Not provided"

        for col in ["applicant_sex", "co_applicant_sex"]:
            df[col] = df[col].astype(str).apply(_clean_sex)

        # Step 17: A / B / C — use training medians as stable defaults.
        # (Original training used random sampling which can't be reproduced
        # exactly at inference time; medians keep features in-distribution.)
        self.abc_medians = {
            col: float(df[col].dropna().median())
            for col in ["A", "B", "C"] if col in df.columns
        }
        for col, med in self.abc_medians.items():
            df[col] = df[col].fillna(med)

        # Feature engineering statistics (computed on cleaned train)
        self.loan_amount_cap = float(df["loan_amount"].quantile(0.99))
        self.applicant_income_cap = float(df["applicant_income"].quantile(0.99))
        self.pop_median = float(df["population"].median())
        self.min_pop_median = float(df["minority_population"].median())
        self.hud_income_median = float(df["hud_median_family_income"].median())
        self.owner_units_median = float(df["number_of_owner_occupied_units"].median())
        self.global_target_mean = float(df[TARGET_COL].mean())

        # Target mean encodings
        self.target_encodings: dict = {}
        self.agency_mode = str(df["agency"].mode().iloc[0])
        df_agency = df["agency"].fillna(self.agency_mode)
        self.target_encodings["agency"] = df.groupby(df_agency)[TARGET_COL].mean().to_dict()

        for cat in ["applicant_race_name_1", "census_tract_number", "county", "msamd", "D"]:
            if cat in df.columns:
                self.target_encodings[cat] = df.groupby(cat)[TARGET_COL].mean().to_dict()

        remaining_obj = [
            c for c in df.select_dtypes(include="object").columns
            if c not in {TARGET_COL, ID_COL} and c not in self.target_encodings
        ]
        for cat in remaining_obj:
            self.target_encodings[cat] = df.groupby(cat)[TARGET_COL].mean().to_dict()

        from src.features import run_all_feature_engineering, get_final_feature_columns
        df_feat, _ = run_all_feature_engineering(df.copy(), df.copy())
        self.final_feature_cols = get_final_feature_columns(df_feat)

    def transform_single(self, features: dict) -> np.ndarray:
        """
        Transform a raw feature dict into a NumPy array for model.predict_proba().
        Mirrors run_all_preprocessing + run_all_feature_engineering using
        stored statistics — no train.csv needed at inference time.
        """
        row = {c: features.get(c) for c in self.raw_cols}
        df = pd.DataFrame([row])
        for c in self.raw_cols:
            try:
                df[c] = df[c].astype(self.raw_dtypes[c])
            except Exception:
                df[c] = df[c].astype(object)

        df = df.drop(columns=self.dropped_cols, errors="ignore")

        col = "owner_occupancy"
        if col in df.columns:
            mask = df[col].isnull()
            df.loc[mask, col] = df.loc[mask, "property_type"].map(self.owner_occ_mode_map).fillna(self.owner_occ_global_mode)

        col = "preapproval"
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str).str.strip().str.lower().apply(lambda v: 1 if "requested" in v else 0)

        col = "applicant_ethnicity"
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

        eth_col = "applicant_ethnicity"
        race_col = "applicant_race_name_1"
        if eth_col in df.columns and race_col in df.columns:
            mask = df[eth_col].isnull() | (df[eth_col].astype(str).str.strip() == "nan")
            df.loc[mask, eth_col] = df.loc[mask, race_col].map(self.eth_by_race_map).fillna(self.eth_global_mode)

        if eth_col in df.columns:
            df[eth_col] = df[eth_col].astype(str).str.strip().str.lower().map(self.eth_numeric_map).fillna(3)

        if race_col in df.columns:
            def _clean_race(val):
                v = str(val).lower()
                for k in self.keep_races:
                    if k in v: return str(val)
                return "Other"
            df[race_col] = df[race_col].astype(str).str.strip().apply(_clean_race)

        col = "co_applicant_ethnicity"
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().map(self.eth_numeric_map).fillna(5)

        col = "co_applicant_race_name_1"
        if col in df.columns:
            def _clean_co_race(val):
                v = str(val).lower()
                for k in self.keep_races:
                    if k in v: return str(val)
                return "Other"
            df[col] = df[col].fillna("No co-applicant").astype(str).str.strip().apply(_clean_co_race)

        col = "census_tract_number"
        if col in df.columns:
            mask = df[col].isnull()
            df.loc[mask, col] = df.loc[mask, "county"].map(self.county_to_tract_map).fillna(-1)

        col = "county"
        if col in df.columns:
            mask = df[col].isnull()
            df.loc[mask, col] = df.loc[mask, "census_tract_number"].map(self.tract_to_county_map).fillna(self.county_global_mode)

        col = "msamd"
        if col in df.columns:
            mask = df[col].isnull()
            df.loc[mask, col] = df.loc[mask, "county"].map(self.county_to_msamd_map).fillna(self.msamd_global_mode)

        col = "tract_to_msamd_income"
        if col in df.columns:
            mask = df[col].isnull()
            df.loc[mask, col] = df.loc[mask, "census_tract_number"].map(self.tmi_tract_mean)
            mask = df[col].isnull()
            df.loc[mask, col] = df.loc[mask, "county"].map(self.tmi_county_mean)
            df[col] = df[col].fillna(self.tmi_global_mean)

        col = "lien_status"
        if col in df.columns:
            mask = df[col].isnull()
            df.loc[mask, col] = df.loc[mask, "loan_type"].map(self.lien_mode_map).fillna(self.lien_global_mode)

        def _clean_sex(val):
            v = str(val).strip().lower()
            if "female" in v: return "Female"
            if "male" in v: return "Male"
            if "not applicable" in v: return "Not applicable"
            if "no co-applicant" in v: return "No co-applicant"
            return "Not provided"

        for col in ["applicant_sex", "co_applicant_sex"]:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(_clean_sex)

        for col, med in self.abc_medians.items():
            if col in df.columns:
                df[col] = df[col].fillna(med)

        # Feature engineering — log transforms
        if "loan_amount" in df.columns:
            df["loan_amount"] = np.log1p(df["loan_amount"].clip(upper=self.loan_amount_cap))
        if "applicant_income" in df.columns:
            df["applicant_income"] = np.log1p(df["applicant_income"].clip(upper=self.applicant_income_cap))
        if "population" in df.columns:
            df["population"] = np.log1p(df["population"].fillna(self.pop_median))
        if "minority_population" in df.columns:
            df["minority_population"] = np.log1p(df["minority_population"].fillna(self.min_pop_median))
        if "hud_median_family_income" in df.columns:
            df["hud_median_family_income"] = np.log1p(df["hud_median_family_income"].fillna(self.hud_income_median))
        if "number_of_owner_occupied_units" in df.columns:
            df["number_of_owner_occupied_units"] = np.log1p(df["number_of_owner_occupied_units"].fillna(self.owner_units_median))

        if "agency" in df.columns:
            df["agency"] = df["agency"].fillna(self.agency_mode)

        for cat_col, enc_map in self.target_encodings.items():
            if cat_col in df.columns:
                df[cat_col] = df[cat_col].map(enc_map).fillna(self.global_target_mean)

        available = [c for c in self.final_feature_cols if c in df.columns]
        return df[available].values
