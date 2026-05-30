from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import joblib
import pandas as pd
from rag.pipeline import explain
from typing import Optional

app = FastAPI(title="LoanSight with RAG Documentation")

# Load existing model (assuming experiments/model.joblib exists)
# If it doesn't exist, this will throw an error when starting the app.
try:
    model_pipeline = joblib.load("experiments/model.joblib")
    print("Loaded ML model successfully.")
except Exception as e:
    print(f"Warning: Could not load ML model: {e}")
    model_pipeline = None

# We use the features standard for our model (adjust if they differ)
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

@app.post("/predict")
def predict(features: ApplicantFeatures):
    features_dict = features.model_dump()
    
    # ML model prediction
    if model_pipeline is not None:
        try:
            # The model might expect a specific dataframe format
            df = pd.DataFrame([features_dict])
            prob = model_pipeline.predict_proba(df)[0][1]
        except Exception as e:
            # Fallback if the model breaks due to missing features
            print(f"Model prediction failed: {e}. Falling back to dummy prediction.")
            prob = 0.82 if features.applicant_income > features.loan_amount * 0.2 else 0.45
    else:
        # Dummy prediction if model wasn't loaded
        prob = 0.82 if features.applicant_income > features.loan_amount * 0.2 else 0.45

    decision = "approved" if prob >= 0.5 else "denied"

    # RAG Explanation Layer
    explanation = explain(decision, features_dict)

    return {
        "decision": decision,
        "confidence": round(prob, 3),
        "explanation": explanation
    }

app.mount("/static", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
def read_root():
    return RedirectResponse(url="/static/index.html")
