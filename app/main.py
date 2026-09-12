import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import pandas as pd
from typing import Optional

from config import TRAIN_PATH, MODEL_PATH
from src.model import load_model
from src.predict import predict_single
from rag.pipeline import explain

app = FastAPI(title="LoanSight with RAG Documentation")

# --- Load the trained ML pipeline once at startup ---
try:
    model_pipeline = load_model(MODEL_PATH)
except Exception as e:
    print(f"Warning: Could not load ML model: {e}")
    model_pipeline = None

# --- Load train.csv once at startup ---
# The preprocessing/feature-engineering pipeline (src/preprocessing.py,
# src/features.py) always learns its imputation and target-mean-encoding
# maps from train.csv, exactly as it does in src/predict.py. Loading it
# once here means every request reuses the same in-memory DataFrame
# instead of re-reading a ~300k row CSV per call.
try:
    train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
    print(f"Loaded {len(train_df):,} training rows for inference alignment.")
except Exception as e:
    print(f"Warning: Could not load training data: {e}")
    train_df = None


class ApplicantFeatures(BaseModel):
    loan_amount: Optional[float] = None
    applicant_income: Optional[float] = None
    population: Optional[float] = None
    minority_population: Optional[float] = None
    hud_median_family_income: Optional[float] = None
    tract_to_msamd_income: Optional[float] = None
    number_of_owner_occupied_units: Optional[float] = None
    A: Optional[float] = None
    B: Optional[float] = None
    C: Optional[float] = None
    property_type: Optional[int] = None
    loan_purpose: Optional[str] = None
    owner_occupancy: Optional[str] = None
    preapproval: Optional[str] = None
    applicant_ethnicity: Optional[str] = None
    applicant_race_name_1: Optional[str] = None
    co_applicant_ethnicity: Optional[str] = None
    co_applicant_race_name_1: Optional[str] = None
    census_tract_number: Optional[float] = None
    county: Optional[str] = None
    msamd: Optional[str] = None
    lien_status: Optional[str] = None
    applicant_sex: Optional[str] = None
    co_applicant_sex: Optional[str] = None
    agency: Optional[int] = None
    D: Optional[int] = None
    loan_type: Optional[int] = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_pipeline is not None,
        "train_data_loaded": train_df is not None,
    }


@app.post("/predict")
def predict(features: ApplicantFeatures):
    features_dict = features.model_dump()

    # ML model prediction
    if model_pipeline is not None and train_df is not None:
        try:
            prob = predict_single(features_dict, train_df, model_pipeline)
        except Exception as e:
            print(f"Model prediction failed: {e}. Falling back to dummy prediction.")
            prob = 0.82 if (features.applicant_income or 0) > (features.loan_amount or 0) * 0.2 else 0.45
    else:
        # Dummy prediction if model/training data wasn't loaded
        prob = 0.82 if (features.applicant_income or 0) > (features.loan_amount or 0) * 0.2 else 0.45

    decision = "approved" if prob >= 0.5 else "denied"

    # RAG Explanation Layer
    explanation = explain(decision, features_dict)

    return {
        "decision": decision,
        "confidence": round(prob, 3),
        "explanation": explanation,
    }


app.mount("/static", StaticFiles(directory="frontend"), name="frontend")


@app.get("/")
def read_root():
    return RedirectResponse(url="/static/index.html")
