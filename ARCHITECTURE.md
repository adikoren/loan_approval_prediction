# LoanSight — Architecture & Design Plan

## Overview

LoanSight is structured as three independent layers that compose together at serving time:

1. **ML Layer** (`src/`) — Trains a binary classifier on HMDA mortgage application data
2. **RAG Layer** (`rag/`) — Retrieves relevant federal regulation text and generates a natural-language explanation for the model's decision
3. **API + UI Layer** (`app/` + `frontend/`) — Exposes a FastAPI `/predict` endpoint and a static web form

---

## System Architecture

```
                          ┌─────────────────────────┐
                          │   Browser (frontend/)    │
                          │  21-field HTML form      │
                          └──────────┬──────────────┘
                                     │ POST /predict (JSON)
                          ┌──────────▼──────────────┐
                          │   FastAPI  (app/main.py) │
                          └──────┬──────────┬────────┘
                                 │          │
               ┌─────────────────▼──┐  ┌───▼─────────────────────┐
               │   ML Pipeline      │  │   RAG Pipeline           │
               │   (src/model.py)   │  │   (rag/pipeline.py)      │
               │                    │  │                          │
               │  SimpleImputer     │  │  1. build_query()        │
               │  → StandardScaler  │  │  2. retrieve() ChromaDB  │
               │  → MLPClassifier   │  │  3. generate() Flan-T5   │
               └─────────────────────┘  └──────────────────────────┘
                        │                          │
                        └──────────┬───────────────┘
                                   │
                          ┌────────▼───────┐
                          │ JSON Response  │
                          │ { decision,    │
                          │   confidence,  │
                          │   explanation }│
                          └────────────────┘
```

---

## ML Pipeline (`src/`)

### Data Flow

```
data/train.csv
    │
    ▼
preprocessing.py          (17 cleaning steps)
    │  - Drop high-missing columns (>50%)
    │  - Impute owner_occupancy via property_type
    │  - Encode preapproval → binary
    │  - Normalize + impute applicant/co-applicant ethnicity, race, sex
    │  - Impute geographic fields: census_tract, county, msamd
    │  - Impute tract_to_msamd_income via geographic hierarchy
    │  - Impute lien_status via loan_type mode
    │  - Random-sample impute A, B, C
    ▼
features.py               (Feature engineering)
    │  - log1p transforms: loan_amount, applicant_income, population,
    │    minority_population, hud_median_family_income, owner_occupied_units
    │  - Target-mean encode: agency, race, census_tract, county, msamd, D
    │  - Bulk target-mean encode all remaining object columns
    ▼
model.py / train.py       (Training)
    │  Pipeline: SimpleImputer → StandardScaler → MLPClassifier(100,50)
    │  Stratified 80/20 split
    │  Early stopping enabled (10% validation fraction)
    ▼
experiments/model.joblib  (Serialized pipeline)
```

### Why MLP?

MLP was selected after 5-fold stratified cross-validation comparison against:
- Logistic Regression
- K-Nearest Neighbors
- Support Vector Classifier
- Random Forest

MLP produced the best ROC-AUC on this dataset. The two-layer architecture `(100, 50)` provides sufficient capacity to capture non-linear interactions between income, loan amount, geographic features, and demographic encodings without overfitting.

### Anti-Leakage Invariant

All preprocessing and encoding maps (mode values, mean targets, quantile caps) are **always learned from `train_df` only** and applied to `test_df`. This is enforced by the `(train_df, test_df) → (train_df, test_df)` function contract throughout `preprocessing.py` and `features.py`.

---

## RAG Pipeline (`rag/`)

### Why RAG?

The ML model produces a binary decision (approved/denied) and a probability score. But for a real lending application, regulators and applicants need to understand *why*. RAG bridges this gap by:

1. **Retrieving** the most semantically relevant passages from federal regulations (FHA Guidelines, Fannie Mae Selling Guide, HMDA rules)
2. **Grounding** a locally-running LLM's explanation in those passages

This ensures explanations are not hallucinated — they cite actual regulatory context.

### Data Flow

```
docs/*.pdf
    │
    ▼  (run once: python rag/ingest.py)
ingest.py
    │  - pdfplumber extracts raw text
    │  - chunk_text() splits into 300-word windows (50-word overlap)
    │  - SentenceTransformer encodes chunks → 384-dim vectors
    │  - ChromaDB stores (text, vector, id) in rag_db/
    ▼
rag_db/  (persistent ChromaDB)

At inference time (per /predict request):
    │
    ▼
pipeline.py → generator.build_query(decision, features)
    │  Builds a natural-language query, e.g.:
    │  "Loan approved. Loan type code: 1. Income: $85000. Amount: $250000."
    ▼
retriever.retrieve(query, k=3)
    │  - Encodes query with all-MiniLM-L6-v2
    │  - ChromaDB cosine similarity search → top 3 chunks
    ▼
generator.generate_explanation(decision, features, chunks)
    │  - Constructs prompt with retrieved context + key applicant fields
    │  - Flan-T5-base generates 20–150 token explanation
    ▼
return explanation string
```

### Model Choices

| Component | Choice | Reason |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | Fast 384-dim encoder, excellent semantic similarity, free |
| Vector DB | ChromaDB local | Zero-config, persistent, no server needed |
| LLM | `google/flan-t5-base` | Instruction-following seq2seq, fully free and local, no API key |
| Chunk size | 300 words / 50 overlap | Captures full regulation clauses without exceeding token limits |

---

## API Layer (`app/main.py`)

### Endpoint

```
POST /predict
Content-Type: application/json

{
  "loan_amount": 250000,
  "applicant_income": 85000,
  "population": 5000,
  ... (21 fields total)
  "A": null,   // imputed automatically
  "B": null,
  "C": null,
  "D": null
}

→ {
  "decision": "approved",
  "confidence": 0.823,
  "explanation": "..."
}
```

### Static File Serving

FastAPI mounts the `frontend/` directory at `/static` and redirects the root `/` to `index.html`. This keeps the frontend and backend in a single deployable process with clean separation — no CORS issues, no separate server.

---

## Frontend (`frontend/`)

- **21 editable fields** matching the actual HMDA training features
- **A, B, C, D** fields are excluded from the form and sent as `null` automatically — the backend's `Optional` Pydantic fields accept them gracefully
- The JS payload is sent to `/predict`, waits for the JSON response, then animates the confidence bar and displays the decision + explanation

---

## Configuration (`config.py`)

Single source of truth. Every file imports constants from here. Never hardcode paths or hyperparameters in individual modules.

| Constant | Value | Purpose |
|---|---|---|
| `TRAIN_PATH` | `data/train.csv` | Training data location |
| `MODEL_PATH` | `experiments/model.joblib` | Serialized model location |
| `MLP_HIDDEN_LAYERS` | `(100, 50)` | MLP architecture |
| `MLP_MAX_ITER` | `1000` | Training iteration cap |
| `TEST_SIZE` | `0.2` | Validation split fraction |
| `TARGET_COL` | `label` | Binary approval column |
| `DROP_COLS` | `county_code, loan_amount_bin` | Columns dropped before training |
