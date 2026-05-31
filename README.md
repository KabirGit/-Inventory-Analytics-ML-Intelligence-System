# 📊 ML Intelligence System — Sales Analytics

A machine learning system for **freight cost prediction** and **invoice anomaly detection**, built with Python, Streamlit, and scikit-learn.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-green)

---

## 🎯 What It Does

| Task | Method | Accuracy |
|------|--------|----------|
| **Freight Cost Prediction** | Gradient Boosting Regressor with lag & vendor-level features | R² = 97.47%, MAE = $25.35 |
| **Invoice Anomaly Detection** | Isolation Forest + Statistical Thresholding + SHAP | 5.85% flagging rate, 1.93% CV stability |

---

## 🏗️ Architecture

```
ML_INTELLIGENCE_SYSTEM/
├── app.py                          # Streamlit web app (entry point)
├── config.py                       # Centralized path configuration
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version (3.10.13)
│
├── data/
│   └── inventory.db                # SQLite database (vendor_invoice + purchases)
│
├── models/                         # Serialized model artifacts (.pkl)
│   ├── predict_freight_cost_model.pkl
│   ├── freight_feature_columns.pkl
│   ├── predict_flag_invoice.pkl
│   ├── scaler.pkl
│   ├── if_threshold.pkl
│   ├── shap_values.pkl
│   ├── shap_feature_names.pkl
│   └── drift_monitor.pkl
│
├── FreightCostPrediction/          # Freight model training pipeline
│   ├── DataPreprocessing.py        # Feature engineering (lag, temporal, vendor-level)
│   ├── model.py                    # Model definitions (LR, Ridge, DT, RF, XGB, GBR)
│   └── train.py                    # Training orchestrator with 5-fold CV + drift baseline
│
├── InvoiceFlagging/                # Anomaly detection pipeline
│   ├── data_preprocessing.py       # SQL feature extraction + RobustScaler
│   ├── model_eval.py              # Isolation Forest + SHAP explanations
│   └── train.py                    # Training orchestrator
│
├── Inference/                      # Standalone prediction utilities
│   ├── predict_freight.py          # Freight prediction with confidence intervals
│   ├── predict_flagged_invoice.py  # Anomaly detection with anomaly scores
│   └── drift_monitor.py           # Data drift detection utility
│
└── jupyter notebooks/              # EDA & prototyping
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/your-username/ML_INTELLIGENCE_SYSTEM.git
cd ML_INTELLIGENCE_SYSTEM
pip install -r requirements.txt
```

### Train Models

```bash
# Freight Cost Prediction (~10 seconds)
python -m FreightCostPrediction.train

# Invoice Anomaly Detection (~15 seconds)
python -m InvoiceFlagging.train
```

### Check Data Drift

```bash
python -m Inference.drift_monitor
```

### Run the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` with three pages:
- **Home** — Overview of capabilities
- **Invoice Flagging** — Input invoice details, get anomaly prediction + SHAP explanation
- **Freight Prediction** — Input dollar amount, get predicted freight cost

---

## 🔬 Model Details

### Freight Cost Prediction

**Problem:** Predict shipping freight cost from invoice dollar amount, quantity, and historical patterns.

**Features (7):**
| Feature | Description |
|---------|-------------|
| `Dollars` | Invoice dollar amount |
| `Quantity` | Invoice unit quantity |
| `lag_1_freight` | Previous invoice's freight (temporal) |
| `lag_2_freight` | Two invoices prior freight |
| `route_avg_freight` | Leave-one-out vendor average freight |
| `invoice_month` | Month of invoice (seasonal pattern) |
| `invoice_dow` | Day of week of invoice (cyclical pattern) |

**Model:** Gradient Boosting Regressor
```
n_estimators=250, max_depth=4, learning_rate=0.03,
subsample=0.75, min_samples_split=15, min_samples_leaf=8
```

**Validation:**
- 5-fold cross-validation for model selection
- 5-fold CV: MAE = $26.13 (±$3.59)
- Test R² = 93.88%, Test RMSE = $177.66
- Overfitting check: train/test ratio = 1.84 (healthy)
- Compared 6 models: Linear Regression, Ridge, Decision Tree, Random Forest, XGBoost, Gradient Boosting
- Confidence intervals provided on predictions (±$25 based on CV MAE)

**Data Drift Monitoring:** Baseline statistics saved at training time. The drift monitor checks incoming data against training distribution using z-scores and out-of-range percentages.

---

### Invoice Anomaly Detection

**Problem:** Flag potentially risky invoices without labeled training data (unsupervised).

**Features (7):**
| Feature | Description |
|---------|-------------|
| `invoice_quantity` | Quantity on the invoice |
| `invoice_dollars` | Dollar amount on the invoice |
| `total_quantity` | Aggregated purchase quantity for the PO |
| `total_dollars` | Aggregated purchase dollars for the PO |
| `average_receiving_delay` | Avg days between PO and receiving |
| `days_po_to_invoice` | Days from PO creation to invoice |
| `payment_delay` | Days from invoice to payment |

**Approach: Two-Gate Anomaly Detection**
1. **Isolation Forest** (200 trees) identifies structural anomalies
2. **Statistical Threshold** (mean - 2σ on decision scores) filters borderline cases
3. An invoice is flagged only if **both gates** agree → reduces false positives

**Scaling:** RobustScaler (median/IQR) — handles heavy right-skew (skewness > 4.4)

**Explainability:** SHAP TreeExplainer provides per-feature importance:
- Top drivers: `invoice_dollars` (0.42), `total_dollars` (0.42), `total_quantity` (0.38)
- New temporal features rank 4th and 6th — contributing meaningful signal

---

## 📊 Key Results

### Freight Prediction Performance

| Model | CV MAE | Test R² | Overfit Ratio |
|-------|--------|---------|---------------|
| Linear Regression | 25.49 | 92.41% | 1.41 |
| Ridge Regression | 25.49 | 92.41% | 1.41 |
| Decision Tree | 44.32 | 90.97% | 1.31 |
| Random Forest | 31.26 | 93.65% | 1.46 |
| XGBoost | 29.09 | 93.17% | 1.41 |
| **Gradient Boosting** | **25.68** | **93.03%** | **1.71** |

### Invoice Flagging Stability

| Random Seed | Flagged | Rate |
|-------------|---------|------|
| 42 | 324 | 5.85% |
| 123 | 337 | 6.08% |
| 456 | 334 | 6.03% |
| 789 | 344 | 6.21% |
| 1024 | 336 | 6.06% |

**Coefficient of Variation: 1.93%** — highly stable across initializations.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10 |
| Web Framework | Streamlit 1.32 |
| ML | scikit-learn 1.4.2, XGBoost |
| Explainability | SHAP |
| Hyperparameter Tuning | Optuna (available), Grid Search |
| Data | SQLite, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Altair |
| Serialization | Joblib |

---

## ⚙️ Configuration

Environment variables (optional overrides):

| Variable | Default | Purpose |
|----------|---------|---------|
| `DB_PATH` | `data/inventory.db` | Path to SQLite database |
| `MODEL_DIR` | `models/` | Directory for .pkl model files |

---

## 📁 Database Schema

### `vendor_invoice` (5,543 rows)
| Column | Type | Description |
|--------|------|-------------|
| VendorNumber | INT | Vendor identifier |
| VendorName | TEXT | Vendor name |
| InvoiceDate | DATE | Invoice creation date |
| PONumber | INT | Purchase order number |
| PODate | DATE | PO placement date |
| PayDate | DATE | Payment date |
| Quantity | INT | Invoice quantity |
| Dollars | FLOAT | Invoice dollar amount |
| Freight | FLOAT | Actual freight cost (target) |
| Approval | TEXT | Approval status |

### `purchases` (2,372,474 rows)
| Column | Type | Description |
|--------|------|-------------|
| PONumber | INT | Purchase order (FK) |
| Brand | INT | Product brand ID |
| Quantity | INT | Purchase quantity |
| Dollars | FLOAT | Purchase dollars |
| ReceivingDate | DATE | Goods received date |
| PODate | DATE | PO placement date |
| ... | ... | + Store, Description, Size, etc. |

---

## 🔮 Future Improvements

- [x] ~~Add `Quantity` as a freight prediction feature~~ ✅ Integrated
- [x] ~~Time-based features (month, day of week) from InvoiceDate~~ ✅ Integrated
- [x] ~~Add `days_po_to_invoice` and `payment_delay` to anomaly detection~~ ✅ Integrated
- [x] ~~Model monitoring for data drift detection~~ ✅ Integrated
- [x] ~~Confidence intervals for freight predictions~~ ✅ Integrated
- [ ] Semi-supervised approach if labeled anomalies become available
- [ ] REST API layer for integration with other systems
- [ ] Quantile regression for asymmetric confidence intervals

---

## 📄 License

This project is for educational and analytical purposes.

---

## 👤 Author

Kabir — Sales Analytics & ML Engineering
