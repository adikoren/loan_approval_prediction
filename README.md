# LoanSight — ML Loan Approval Prediction with RAG

LoanSight is an end-to-end machine learning system that predicts whether a home loan application will be approved or denied, and then generates a natural-language explanation for that decision grounded in federal lending regulations — using a Retrieval-Augmented Generation (RAG) pipeline.

Built as a portfolio project demonstrating the full MLOps stack: data engineering, model training, LLM integration, and a production-ready API with a web frontend.

---

## Live Demo

Start the server and open your browser:

```bash
uvicorn app.main:app --reload
```

Navigate to **http://localhost:8000** — you will land directly on the interactive prediction interface.

---

## Project Structure

```
loan_approval_prediction_update/
├── data/
│   ├── train.csv               # HMDA training data (293k rows)
│   ├── test.csv                # HMDA test data
│   └── predictions.csv         # Output from src/predict.py
│
├── src/                        # Core ML pipeline
│   ├── preprocessing.py        # Data cleaning and imputation (17 steps)
│   ├── features.py             # Feature engineering and target-mean encoding
│   ├── model.py                # MLP sklearn pipeline definition
│   ├── train.py                # Full training script
│   ├── predict.py              # Batch prediction on test.csv
│   └── evaluate.py             # Metrics, confusion matrix, ROC curve
│
├── rag/                        # Retrieval-Augmented Generation layer
│   ├── ingest.py               # PDF chunking + embedding into ChromaDB
│   ├── retriever.py            # Semantic search over regulation chunks
│   ├── generator.py            # Flan-T5 explanation generation
│   └── pipeline.py             # Orchestrates the full RAG flow
│
├── app/
│   └── main.py                 # FastAPI server: /predict endpoint + static files
│
├── frontend/
│   ├── index.html              # Form UI with all 21 HMDA input fields
│   ├── style.css               # Clean white design system
│   └── script.js               # API calls and UI state management
│
├── docs/                       # Federal regulation PDFs (FHA, Fannie Mae, HMDA)
├── rag_db/                     # Persistent ChromaDB vector store
├── experiments/
│   ├── model.joblib            # Trained MLP pipeline (serialized)
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── logs/                   # Per-run JSON training logs
│
├── config.py                   # Single source of truth for all paths/hyperparams
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Ingest regulation PDFs into the vector store
```bash
python rag/ingest.py
```

### 3. Train the ML model
```bash
python src/train.py
```

### 4. Launch the API + web UI
```bash
uvicorn app.main:app --reload
```

---

## Model Performance

Trained on **293,664 HMDA rows**, evaluated on a stratified 20% hold-out:

| Metric     | Score  |
|------------|--------|
| Accuracy   | 78.8%  |
| ROC-AUC    | **87.3%** |
| Precision  | 79.3%  |
| Recall     | 77.9%  |
| F1         | 78.6%  |

---

## Tech Stack

| Layer         | Technology                          |
|---------------|--------------------------------------|
| ML Model      | scikit-learn MLP Classifier          |
| Data Pipeline | pandas, numpy                        |
| Embeddings    | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store  | ChromaDB (local persistent)          |
| LLM           | `google/flan-t5-base` (local, free)  |
| API           | FastAPI + Uvicorn                    |
| Frontend      | HTML / CSS / Vanilla JS              |

---

## Key Design Decisions

- **Local-First**: Both the embedding model and LLM run entirely on your machine — no API keys, no usage costs.
- **No Leakage**: All preprocessing and encoding maps are learned exclusively from `train.csv` and applied to test data.
- **Separation of Concerns**: `src/` (ML) and `rag/` (LLM/RAG) are fully independent modules. The API in `app/` wires them together at serving time.
- **A, B, C, D Fields**: The HMDA dataset includes four anonymous columns (A, B, C, D). Their values are imputed from training distribution during ingest and sent as `null` from the frontend — the preprocessing pipeline handles them transparently.
