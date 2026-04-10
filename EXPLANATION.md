# LoanSight — File-by-File Explanation

Focused walkthrough of every file in `src/` and `rag/`, with design rationale for each decision.

---

## `src/` — ML Pipeline

### `src/preprocessing.py`

**Purpose:** Clean and impute the raw HMDA CSV before any feature engineering.

**17 steps applied in `run_all_preprocessing()`:**

| Step | Function | What it does | Why |
|------|----------|--------------|-----|
| 1 | `drop_high_missing_columns` | Drops columns where >50% of train values are missing | Columns that sparse cannot be imputed reliably and add noise |
| 2 | `impute_and_filter_owner_occupancy` | Fills missing `owner_occupancy` using modal value per `property_type` group | Occupancy is structurally tied to property type — within-group mode beats global mode |
| 3 | `encode_preapproval_feature` | Maps `preapproval` string variants → binary 0/1 | Raw column has multiple label forms for the same concept |
| 4 | `normalize_applicant_ethnicity` | Lowercases + strips ethnicity strings | Prevents duplicate categories from casing inconsistencies |
| 5 | `impute_applicant_ethnicity_by_race` | Fills missing ethnicity using modal ethnicity per race group | Race and ethnicity are correlated; within-group mode is more informative |
| 6 | `map_applicant_ethnicity_to_numeric` | Maps ethnicity labels → integer codes 1–5 | Required before target-mean encoding in `features.py` |
| 7 | `clean_and_summarize_race_distribution` | Consolidates rare/verbose race categories into "Other" | Reduces cardinality for stable target encoding |
| 8 | `encode_co_applicant_ethnicity` | Same logic as step 4–6 for co-applicant | Consistency between applicant and co-applicant fields |
| 9 | `encode_co_applicant_race` | Same consolidation as step 7 for co-applicant | Mirrors applicant race treatment |
| 10 | `impute_census_tract_using_train_mappings` | Fills missing `census_tract_number` using county→tract map from train | Uses geographic hierarchy; avoids leakage from test |
| 11 | `impute_county_using_census_tract_mapping` | Fills missing `county` using tract→county map from train | Census tract uniquely identifies county |
| 12 | `impute_msamd_using_mappings` | Fills missing `msamd` using county→msamd map from train | Metro-area is determined by county; train mapping prevents leakage |
| 13 | `impute_tract_to_msamd_income` | Three-level geographic fallback: tract → county → global mean | Single global mean ignores geographic structure |
| 14 | `impute_lien_status` | Fills missing `lien_status` with modal value per `loan_type` | Lien status is determined by loan type |
| 15 | `clean_and_categorize_applicant_sex` | Normalizes sex labels to 5 controlled values | Raw data has inconsistent string variants |
| 16 | `clean_and_categorize_co_applicant_sex` | Same as step 15 for co-applicant | Consistency |
| 17 | `impute_abc_with_random_samples` | Fills missing A, B, C by random sampling from train distribution | A, B, C are anonymous synthetic columns; distribution-preserving imputation is least-biased |

**Key invariant:** Every imputation map (mode, mean, quantile) is learned from `train_df` only. Test rows receive the same transformation without recalculating. This is enforced by the `(train_df, test_df) → (train_df, test_df)` function signature on every step.

---

### `src/features.py`

**Purpose:** Apply numeric transforms and encode all categorical columns as floating-point values suitable for the MLP.

#### Numeric Transforms (log1p)

Applied to right-skewed columns to bring them closer to a normal distribution before `StandardScaler`:

| Column | Why skewed |
|---|---|
| `loan_amount` | Mortgage amounts have a very long right tail |
| `applicant_income` | Income distributions are universally right-skewed |
| `population` | Census tract populations span orders of magnitude |
| `minority_population` | Sparse, heavily right-tailed count |
| `hud_median_family_income` | Cross-tract income varies by orders of magnitude |
| `number_of_owner_occupied_units` | Skewed count variable |

All columns are clipped at the 99th percentile (train) *before* log transform to neutralize extreme outliers.

#### Target-Mean Encoding

All categorical columns are replaced with the **mean approval rate** for each category, learned only from train. Unseen categories in test fall back to the global mean.

**Why not one-hot?** The HMDA dataset has many categorical columns with high cardinality (census_tract has thousands of unique values). One-hot encoding would create thousands of sparse columns that destabilize MLP training. Target encoding compresses each column to a single informative float.

Specifically encoded:
- `agency` — Different agencies have systematically different approval rates
- `applicant_race_name_1` — After consolidation in preprocessing
- `census_tract_number` — Local geographic approval rate
- `county` — County-level approval rate
- `msamd` — Metro-area approval rate
- `D` — Anonymous categorical column

#### `get_final_feature_columns()`

Single source of truth for the exact feature list used both in `train.py` and `predict.py`. Excludes: `TARGET_COL`, `ID_COL`, `DROP_COLS`, and any remaining `object` dtype columns (an encoding step was missed if any remain).

**Result: 27 feature columns after full pipeline.**

---

### `src/model.py`

**Purpose:** Define and serialize the production sklearn Pipeline.

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),  # catches any residual NaN
    ("scaler",  StandardScaler()),                 # zero-mean / unit-variance
    ("mlp",     MLPClassifier(
        hidden_layer_sizes=(100, 50),              # two hidden layers
        alpha=0.0001,                              # L2 regularization
        learning_rate_init=0.01,
        max_iter=1000,
        early_stopping=True,                       # halts when val loss plateaus
        validation_fraction=0.1,
    )),
])
```

**Why `SimpleImputer` inside the pipeline?** Even after all preprocessing steps, a test row could have unseen geographic categories that result in `NaN` after target encoding. The in-pipeline imputer is a safety net.

**Why `StandardScaler`?** MLP gradient updates are scale-sensitive. Without normalization, high-magnitude features (e.g. `hud_median_family_income`) dominate the gradient signal, making training unstable.

**Why `early_stopping=True`?** Prevents overfitting on the large training set. Training halts when validation loss stops improving, not when `max_iter` is reached.

---

### `src/train.py`

**Purpose:** Full end-to-end training run. Call once to produce `experiments/model.joblib`.

**Steps:**
1. Load `data/train.csv`
2. Run `run_all_preprocessing(train, train)` — passes train as both args since no separate test set at train time
3. Run `run_all_feature_engineering()`
4. Stratified 80/20 train/validation split (`stratify=y` preserves class proportions)
5. Fit the sklearn Pipeline
6. Evaluate on validation set → print metrics
7. Save model to `experiments/model.joblib`
8. Write JSON log with timestamp, metrics, feature list to `experiments/logs/`

**Result:** 293k rows → 234k train / 58k val. Training completes in ~2 minutes on CPU.

---

### `src/predict.py`

**Purpose:** Batch predictions on `data/test.csv`. Outputs `data/predictions.csv`.

**Critical:** Always loads `train.csv` alongside `test.csv` because target-mean encodings require train labels to compute category-level approval rates. Passing test alone would break encoding.

---

### `src/evaluate.py`

**Purpose:** Post-training diagnostics saved to `experiments/`.

- `evaluate()` — returns dict of accuracy, ROC-AUC, precision, recall, F1
- `plot_confusion_matrix()` — seaborn heatmap showing where the model errs (false approvals vs false rejections have very different real-world costs)
- `plot_roc_curve()` — primary metric plot for imbalanced datasets; shows sensitivity/specificity tradeoff across all thresholds

---

## `rag/` — Retrieval-Augmented Generation

### `rag/ingest.py`

**Purpose:** One-time setup script. Reads all PDFs in `docs/`, chunks them, embeds the chunks, and stores them in ChromaDB.

**Run with:**
```bash
python rag/ingest.py
```

**`chunk_text(text, chunk_size=300, overlap=50)`**
- Splits extracted text into overlapping word windows
- 300-word windows capture full regulation clauses (a typical clause is 100–200 words)
- 50-word overlap prevents relevant context from being split across chunk boundaries
- Each chunk gets a unique ID: `{pdf_filename}_{chunk_index}`

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- Produces 384-dimensional semantic vectors
- Fast encoding (thousands of chunks per minute on CPU)
- Free, local, no API key

**Storage:** ChromaDB `PersistentClient` writes to `./rag_db/` — survives server restarts.

---

### `rag/retriever.py`

**Purpose:** At inference time, find the 3 regulation chunks most semantically similar to the current loan decision.

```python
def retrieve(query: str, k: int = 3) -> list[str]:
    query_vec = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_vec, n_results=k)
    return results['documents'][0]
```

**The query** is built from the decision + key applicant fields (e.g. `"Loan approved. Loan type code: 1. Income: $85000. Amount: $250000."`).

ChromaDB performs cosine similarity search over the stored regulation embeddings and returns the top-k matching text chunks. These chunks become the grounding context for the LLM.

**Module-level initialization:** The `SentenceTransformer` model and ChromaDB client are instantiated once at import time — not per-request — so there's no latency penalty after the first request.

---

### `rag/generator.py`

**Purpose:** Use `google/flan-t5-base` to generate a plain-English explanation of the loan decision, grounded in the retrieved regulation chunks.

**`build_query(decision, features)`**
Constructs the ChromaDB search query string from the API features dict. Uses `loan_type`, `applicant_income`, and `loan_amount` as the key discriminating fields.

**`generate_explanation(decision, features, chunks)`**
```
Prompt structure:
  "Explain briefly why a loan application was {DECISION} based on the context.
   Context: {retrieved_regulation_chunks}
   Income: ${income}, Loan amount: ${amount}, Loan type code: {type}
   Explanation:"
```

**Model:** `google/flan-t5-base`
- Seq2Seq architecture (encoder-decoder), designed for instruction-following
- `max_length=512` for input (truncated), `max_length=150 / min_length=20` for output
- Runs entirely locally — no Hugging Face API token or billing required
- Weights are cached after first download to `~/.cache/huggingface/`

**Why Flan-T5 and not GPT-4?**
- Zero cost, zero rate limits
- Sufficient capacity for short regulatory summaries
- Keeps the project self-contained and reproducible without credentials

---

### `rag/pipeline.py`

**Purpose:** Single entry point that wires retriever and generator together.

```python
def explain(decision: str, features: dict) -> str:
    query  = build_query(decision, features)
    chunks = retrieve(query, k=3)
    return generate_explanation(decision, features, chunks)
```

Called once per `/predict` request from `app/main.py`. The three steps always execute in order: query construction → vector retrieval → LLM generation.

---

## Final Validation Results

Training run on 2026-04-10:

```
[train] Loaded 293,664 rows, 34 columns.
[train] 27 feature columns selected.
[train] Train size: 234,931  |  Val size: 58,733

  Accuracy  : 0.7878
  ROC-AUC   : 0.8729
  Precision : 0.7932
  Recall    : 0.7787
  F1        : 0.7859
```

Confusion matrix and ROC curve saved to `experiments/`.
