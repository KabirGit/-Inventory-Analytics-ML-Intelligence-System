# ML_INTELLIGENCE_SYSTEM — Comprehensive Project Context

## 1. Project Overview

This is a **Sales Analytics ML System** that provides two core machine learning capabilities:

1. **Invoice Flagging** — Binary classification to detect risky/anomalous invoices.
2. **Freight Cost Prediction** — Regression to predict freight cost based on dollar amount.

The system is served via a **Streamlit** web application and backed by a **SQLite** database (`data/inventory.db`). Models are trained offline and serialized as `.pkl` files for inference.

---

## 2. Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10.13 |
| Web Framework | Streamlit 1.32.0 |
| ML Libraries | scikit-learn 1.4.2, Optuna (hyperparameter tuning) |
| Data | SQLite (via sqlite3 + SQLAlchemy), Pandas 2.1.4, NumPy 1.26.4 |
| Visualization | Matplotlib, Seaborn, Altair 4.2.2 |
| Stats | SciPy, Statsmodels |
| Serialization | Joblib |
| Deployment target | Likely Streamlit Cloud (has `runtime.txt`) |

---

## 3. Directory Structure

```
ML_INTELLIGENCE_SYSTEM/
├── app.py                          # Streamlit web app (entry point)
├── config.py                       # Centralized path configuration
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version for deployment (3.10.13)
├── .gitignore
│
├── data/
│   └── inventory.db                # SQLite database (source of truth)
│
├── models/                         # Production model artifacts (used by app.py)
│   ├── predict_flag_invoice.pkl    # Trained invoice flagging model
│   ├── predict_freight_cost_model.pkl  # Trained freight cost model
│   └── scaler.pkl                  # StandardScaler for invoice features
│
├── FreightCostPrediction/          # Training pipeline for freight model
│   ├── __init__.py
│   ├── DataPreprocessing.py        # Data loading & feature prep
│   ├── model.py                    # Model definitions (LR, DT, RF)
│   ├── train.py                    # Training orchestrator
│   └── models/                     # (gitignored duplicate)
│
├── InvoiceFlagging/                # Training pipeline for invoice model
│   ├── __init__.py
│   ├── data_preprocessing.py       # Data loading, labeling, scaling
│   ├── model_eval.py              # Model training (RF w/ Optuna, LR) & evaluation
│   ├── train.py                    # Training orchestrator
│   └── models/                     # (gitignored duplicate)
│
├── Inference/                      # Standalone inference scripts
│   ├── __init__.py
│   ├── predict_flagged_invoice.py  # Invoice flag prediction utility
│   └── predict_freight.py          # Freight cost prediction utility
│
└── jupyter notebooks/              # EDA & prototyping (gitignored)
    ├── invoice flagging.ipynb
    ├── predicting_freight_cost.ipynb
    ├── VendorPerformanceAnalysis.ipynb
    └── seaborn.ipynb
```

---

## 4. Configuration (`config.py`)

Centralized path management using `pathlib.Path`. Supports environment variable overrides:

```python
BASE_DIR = Path(__file__).resolve().parent  # Project root

DATA_PATH = Path(os.getenv("DB_PATH", BASE_DIR / "data" / "inventory.db"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))
```

- `DB_PATH` env var overrides the database location.
- `MODEL_DIR` env var overrides the model directory.
- `MODEL_DIR` is auto-created if missing.

---

## 5. Database Schema (SQLite: `data/inventory.db`)

### Known Tables (inferred from queries):

#### `vendor_invoice`
| Column | Type | Notes |
|--------|------|-------|
| PONumber | TEXT/INT | Purchase Order identifier |
| Quantity | NUMERIC | Invoice line quantity |
| Dollars | NUMERIC | Invoice dollar amount |
| Freight | NUMERIC | Actual freight cost (target for regression) |
| InvoiceDate | DATE | Date invoice was created |
| PODate | DATE | Date PO was placed |
| PayDate | DATE | Date payment was made |

#### `purchases`
| Column | Type | Notes |
|--------|------|-------|
| PONumber | TEXT/INT | Purchase Order identifier (FK to vendor_invoice) |
| Brand | TEXT | Product brand |
| Quantity | NUMERIC | Purchase quantity |
| Dollars | NUMERIC | Purchase dollar amount |
| ReceivingDate | DATE | Date goods were received |
| PODate | DATE | Date PO was placed |

---

## 6. ML Pipeline Details

### 6.1 Freight Cost Prediction

**Task:** Regression — predict `Freight` from `Dollars`.

**Feature:** Single feature — `Dollars` (from `vendor_invoice` table).

**Target:** `Freight` column.

**Models Compared:**
1. Linear Regression
2. Decision Tree Regressor (max_depth=4)
3. Random Forest Regressor (max_depth=4)

**Selection Criteria:** Lowest MAE wins.

**Metrics:** MAE, MSE, R² Score.

**Training Script:** `FreightCostPrediction/train.py` → saves best model to `models/predict_freight_cost_model.pkl`.

**Data Split:** 80/20 train/test, random_state=42.

---

### 6.2 Invoice Flagging

**Task:** Binary classification — flag risky invoices (1 = risky, 0 = safe).

**Features (5):**
1. `invoice_quantity` — quantity on the invoice
2. `invoice_dollars` — dollar amount on the invoice
3. `total_quantity` — aggregated purchase quantity for that PO
4. `total_dollars` — aggregated purchase dollars for that PO
5. `average_receiving_delay` — avg days between PODate and ReceivingDate for that PO

**Target:** `flag_invoice` (engineered label, NOT from database).

**Label Engineering Logic:**
```python
def create_invoice_risk_label(row):
    if abs(row["invoice_dollars"] - row["total_dollars"]) > 5:
        return 1  # Dollar mismatch → risky
    if row["average_receiving_delay"] > 10:
        return 1  # Late receiving → risky
    return 0      # Safe
```

**Preprocessing:** StandardScaler applied to all 5 features. Scaler saved as `models/scaler.pkl`.

**Models Available:**
1. Logistic Regression (currently used in `train.py`)
2. Random Forest Classifier with Optuna hyperparameter tuning (available in `model_eval.py` but not used in current training script)

**Optuna Config (for RF, if enabled):**
- n_estimators: 50–300
- max_depth: 5–30
- min_samples_split: 2–10
- min_samples_leaf: 1–5
- max_features: sqrt or log2
- class_weight: balanced
- CV: 3-fold, scoring: F1
- Trials: 20

**Training Script:** `InvoiceFlagging/train.py` → saves model to `models/predict_flag_invoice.pkl`.

**Data Split:** 80/20 train/test (no fixed random_state in split).

---

## 7. Streamlit App (`app.py`)

**Entry point:** `streamlit run app.py`

**Pages (sidebar navigation):**

### 🏠 Home
- Description of the app and its capabilities.

### 🚩 Invoice Flagging
- **Inputs:** 5 numeric fields (invoice_quantity, invoice_dollars, total_quantity, total_dollars, average_receiving_delay)
- **Process:** Scale input with saved scaler → predict with saved model
- **Output:** "Invoice is RISKY" (prediction=1) or "Invoice is SAFE" (prediction=0)

### 🚚 Freight Prediction
- **Input:** Single numeric field (Dollars)
- **Process:** Predict with saved model (no scaling needed)
- **Output:** Predicted freight cost as a float

**Model Loading:** Uses `@st.cache_resource` for efficient caching.

---

## 8. Inference Module (`Inference/`)

Standalone prediction utilities (usable outside Streamlit):

### `predict_flagged_invoice.py`
- `predict_invoice_flag(input_data: dict) → pd.DataFrame`
- Accepts dict with lists for each feature, returns DataFrame with `Predicted_Flag` column.

### `predict_freight.py`
- `predict_freight(input_data: dict, model_path: str) → pd.DataFrame`
- Accepts dict with `Dollars` key, returns DataFrame with `Predicted_Freight` column.

---

## 9. Data Flow Summary

```
SQLite DB (inventory.db)
    │
    ├──[FreightCostPrediction/train.py]──→ models/predict_freight_cost_model.pkl
    │
    ├──[InvoiceFlagging/train.py]──→ models/predict_flag_invoice.pkl
    │                              └→ models/scaler.pkl
    │
    └──[app.py (Streamlit)]──→ Loads .pkl models → Serves predictions via UI
```

---

## 10. Key Design Decisions & Notes

1. **Label is synthetic** — `flag_invoice` is NOT stored in the database. It's computed at training time using rule-based logic (dollar mismatch > $5 OR receiving delay > 10 days).
2. **Freight model uses single feature** — Only `Dollars` is used to predict `Freight`. This is a simple univariate regression.
3. **Scaler only for Invoice Flagging** — Freight model does NOT use a scaler.
4. **Model selection is automated** for freight (best MAE), but hardcoded to Logistic Regression for invoice flagging (despite RF+Optuna being available).
5. **No API layer** — Predictions are served only through Streamlit UI or direct Python imports from `Inference/`.
6. **No authentication** — The Streamlit app has no auth/access control.
7. **Database is local SQLite** — No remote DB connection; the `.db` file must be present locally.
8. **Jupyter notebooks are gitignored** — They exist for EDA but are not part of the production pipeline.

---

## 11. How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train Models (if needed)
```bash
python -m FreightCostPrediction.train
python -m InvoiceFlagging.train
```

### Run the App
```bash
streamlit run app.py
```

### Run Inference Standalone
```bash
python -m Inference.predict_freight
python -m Inference.predict_flagged_invoice
```

---

## 12. Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DB_PATH` | `<project_root>/data/inventory.db` | Path to SQLite database |
| `MODEL_DIR` | `<project_root>/models/` | Directory containing .pkl model files |

---

## 13. Dependencies (Exact Versions)

```
streamlit==1.32.0
pandas==2.1.4
numpy==1.26.4
scikit-learn==1.4.2
joblib (unpinned)
sqlalchemy (unpinned)
matplotlib (unpinned)
seaborn (unpinned)
scipy (unpinned)
statsmodels (unpinned)
altair==4.2.2
optuna (used in code but NOT in requirements.txt — potential issue)
```

**Note:** `optuna` is imported in `InvoiceFlagging/model_eval.py` but is NOT listed in `requirements.txt`. This will cause an ImportError if the RF training path is used.

---

## 14. Known Issues / Gaps

1. **Missing `optuna` in requirements.txt** — Will fail if RF training is invoked.
2. **No random_state in invoice flagging split** — Results are not reproducible across runs.
3. **No input validation** in Streamlit app — negative or extreme values are accepted.
4. **No model versioning** — Models are overwritten on retrain with no history.
5. **No logging** — Only print statements for training output.
6. **No error handling** in model loading — App will crash if `.pkl` files are missing.
7. **Freight model is very simple** — Single feature regression may underfit.
8. **Label leakage risk** — `total_dollars` is used both in label creation AND as a feature for invoice flagging.
