import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from typing import Optional

import joblib

from config import MODEL_PATH
from src.model import load_model
from src.preprocessor import PreprocessorFitter
from rag.pipeline import explain

PREPROCESSOR_PATH = "experiments/preprocessor.joblib"

app = FastAPI(title="LoanSight with RAG Documentation")

# --- Load the trained ML pipeline once at startup ---
try:
    model_pipeline = load_model(MODEL_PATH)
except Exception as e:
    print(f"Warning: Could not load ML model: {e}")
    model_pipeline = None

# --- Load pre-fitted preprocessor (~145 KB) instead of train.csv (~500 MB) ---
# PreprocessorFitter bakes all encoding maps from train at build time.
# This keeps startup memory well within the 512 MB DigitalOcean Basic tier.
try:
    preprocessor: PreprocessorFitter = joblib.load(PREPROCESSOR_PATH)
    print(f"Loaded preprocessor from {PREPROCESSOR_PATH}")
except Exception as e:
    print(f"Warning: Could not load preprocessor: {e}")
    preprocessor = None


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
    property_type: Optional[str] = None
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
    agency: Optional[str] = None
    D: Optional[int] = None
    loan_type: Optional[str] = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_pipeline is not None,
        "preprocessor_loaded": preprocessor is not None,
    }


@app.post("/predict")
def predict(features: ApplicantFeatures):
    features_dict = features.model_dump()

    # ML model prediction via lightweight preprocessor (no train.csv needed)
    if model_pipeline is not None and preprocessor is not None:
        try:
            X = preprocessor.transform_single(features_dict)
            prob = float(model_pipeline.predict_proba(X)[0, 1])
        except Exception as e:
            print(f"Model prediction failed: {e}. Falling back to dummy prediction.")
            prob = 0.82 if (features.applicant_income or 0) > (features.loan_amount or 0) * 0.2 else 0.45
    else:
        # Dummy prediction if model/preprocessor wasn't loaded
        prob = 0.82 if (features.applicant_income or 0) > (features.loan_amount or 0) * 0.2 else 0.45

    decision = "approved" if prob >= 0.5 else "denied"

    # approval_probability is the raw model score for the positive (approved)
    # class; confidence is how sure the model is in whichever decision was
    # actually made. These are the same number only when the decision is
    # "approved" — for a denial (prob < 0.5), confidence is 1 - prob, not
    # prob itself. Previously "confidence" was always just prob, so e.g. a
    # denial with prob=0.12 displayed as "12% confidence" — actually 88%
    # confidence in that denial.
    approval_probability = prob
    confidence = approval_probability if decision == "approved" else 1.0 - approval_probability

    # RAG Explanation Layer
    rag_result = explain(
        decision, features_dict, confidence=confidence, approval_probability=approval_probability
    )

    return {
        "decision": decision,
        "confidence": round(confidence, 3),
        "approval_probability": round(approval_probability, 3),
        "explanation": rag_result["explanation"],
        "retrieved_sources": rag_result["sources"],
    }


app.mount("/static", StaticFiles(directory="frontend"), name="frontend")


@app.get("/")
def read_root():
    return RedirectResponse(url="/static/index.html")
