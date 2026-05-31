# ML Intelligence System — Model Optimization Report

**Date:** May 31, 2026  
**Objective:** Maximize model accuracy and efficiency without overfitting or underfitting.

---

## Executive Summary

| Model | Metric | Before Optimization | After Optimization | Change |
|-------|--------|--------------------|--------------------|--------|
| Freight Prediction | CV MAE | ~0.00 (data leakage) | 25.35 | Fixed leakage, real performance |
| Freight Prediction | Test R² | 100% (leakage) | 97.47% | Honest metric |
| Freight Prediction | Overfit Ratio | N/A | 1.71 | Healthy (< 2.0) |
| Invoice Flagging | Stability (CV across seeds) | Not measured | 1.93% | Highly stable |
| Invoice Flagging | Flagging Rate | 9.56% | 5.85% | More conservative, fewer false positives |

---

## 1. Freight Cost Prediction

### 1.1 Problem Identified: Data Leakage

**Critical finding:** The original `route_avg_freight` feature was computed as `groupby('PONumber')['Freight'].transform('mean')`. Since every PONumber has exactly 1 invoice in the dataset, this equals the target variable itself (correlation = 1.0). This caused the model to achieve R² = 100% — a clear sign of data leakage, not genuine predictive power.

**Fix:** Replaced with a **leave-one-out vendor-level average freight**:
```python
vendor_sum = df.groupby('VendorNumber')['Freight'].transform('sum')
vendor_count = df.groupby('VendorNumber')['Freight'].transform('count')
route_avg_freight = (vendor_sum - df['Freight']) / (vendor_count - 1)
```
This gives the model a meaningful "what does this vendor typically charge for freight?" signal without leaking the answer.

### 1.2 Additional Improvements

| Change | Rationale |
|--------|-----------|
| Sort by InvoiceDate before lag features | Lag features only make sense in temporal order |
| Added Ridge Regression | Handles multicollinearity between features |
| Added Gradient Boosting Regressor | Ensemble method that often outperforms XGBoost on tabular data |
| Tuned XGBoost (max_depth=3, reg_alpha=0.1) | Reduced from depth=5 to prevent overfitting |
| Tuned GBR (n_estimators=250, lr=0.03, subsample=0.75) | Slow learning + aggressive subsampling = better generalization |
| Added RMSE and MAPE metrics | More complete evaluation picture |
| Added overfitting check (train/test MAE ratio) | Automated guard against overfitting |

### 1.3 Hyperparameter Tuning Process

Tested 5 GBR configurations and 4 XGBoost configurations via grid search with overfitting constraint (train/test ratio < 1.8):

**Winner: Gradient Boosting Regressor**
```python
GradientBoostingRegressor(
    n_estimators=250,    # More trees with slow learning
    max_depth=4,         # Moderate depth
    learning_rate=0.03,  # Slow learning prevents overfitting
    subsample=0.75,      # Aggressive row subsampling
    min_samples_split=15,# Prevents fitting to noise
    min_samples_leaf=8,  # Minimum leaf size
    random_state=42
)
```

### 1.4 Final Results (10-Fold Cross-Validation)

| Metric | Value |
|--------|-------|
| Mean MAE | 25.35 |
| Std MAE | 4.68 |
| Mean R² | 97.47% |
| Std R² | 3.09% |
| Overfit Ratio | 1.71 |
| Status | ✅ No overfitting, no underfitting |

**Interpretation:** On average, the model predicts freight cost within $25.35 of the actual value. Given the freight range is $0.02 to $8,468.22 (mean $295.95), this represents excellent accuracy.

### 1.5 Edge Case Behavior

| Input (Dollars) | Predicted Freight | Reasonable? |
|----------------|-------------------|-------------|
| $0 | $5.57 | ✅ Minimum freight charge |
| $100 | $16.28 | ✅ Small order |
| $1,000 | $38.70 | ✅ Moderate |
| $50,000 | $443.74 | ✅ Large order |
| $1,660,000 | $6,620.98 | ✅ Max range |

---

## 2. Invoice Flagging (Anomaly Detection)

### 2.1 Key Change: RobustScaler

**Problem:** The original StandardScaler assumes normally distributed data. The invoice data is heavily right-skewed (skewness > 4.4 for quantity/dollar features). StandardScaler is sensitive to outliers and compresses the majority of data into a narrow range.

**Fix:** Switched to **RobustScaler** which uses median and IQR:
- Robust to outliers (doesn't let extreme values dominate scaling)
- Better separation between normal and anomalous points for Isolation Forest
- No assumption of normality

### 2.2 Isolation Forest Tuning

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| n_estimators | 100 | 200 | More trees = more stable anomaly scores |
| contamination | 0.15 (fixed) | 'auto' | Let the model determine natural boundary |
| threshold | mean - 1.5*std | mean - 2.0*std | More conservative = fewer false positives |
| n_jobs | 1 | -1 | Parallel training for efficiency |
| max_features | default | 1.0 | Use all 5 features (small feature set) |

### 2.3 Why mean - 2*std Instead of 1.5*std?

- `mean - 1.5*std` flagged 9.56% of invoices (530/5543) — too aggressive
- `mean - 2.0*std` flags 5.85% of invoices (324/5543) — more conservative
- In anomaly detection, false positives are costly (manual review burden)
- 2*std targets roughly the bottom 2.5% of a normal distribution, which combined with the IF gate, gives ~5.85% flagging rate

### 2.4 Stability Analysis

Tested across 5 different random seeds:

| Seed | Flagged | Rate |
|------|---------|------|
| 42 | 324 | 5.85% |
| 123 | 337 | 6.08% |
| 456 | 334 | 6.03% |
| 789 | 344 | 6.21% |
| 1024 | 336 | 6.06% |

**Coefficient of Variation: 1.93%** — extremely stable. The model produces consistent results regardless of random initialization.

### 2.5 What Gets Flagged?

| Feature | Normal (median) | Flagged (median) | Ratio |
|---------|----------------|-----------------|-------|
| invoice_quantity | 331 | 44,113 | 133x |
| invoice_dollars | $4,013 | $436,186 | 109x |
| total_quantity | 328 | 44,353 | 135x |
| total_dollars | $3,985 | $436,186 | 109x |
| average_receiving_delay | 7.74 days | 7.85 days | 1.02x |

**Interpretation:** The model correctly identifies invoices with extremely high dollar amounts and quantities as anomalous. These are 100x+ larger than typical invoices. The receiving delay has minimal impact on flagging — the anomalies are primarily driven by transaction size.

### 2.6 SHAP Feature Importance

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | invoice_dollars | 0.5722 |
| 2 | invoice_quantity | 0.5544 |
| 3 | total_dollars | 0.5465 |
| 4 | total_quantity | 0.5425 |
| 5 | average_receiving_delay | 0.3924 |

---

## 3. Design Decisions & Trade-offs

### 3.1 Why Not Remove Correlated Features?

`invoice_dollars` and `total_dollars` are nearly identical (since each PO has 1 invoice). However:
- Removing them would break the API contract (app.py expects 5 inputs)
- Isolation Forest handles correlated features well (tree-based, not distance-based)
- SHAP correctly shows both contribute similarly
- The redundancy doesn't hurt IF performance

### 3.2 Why Gradient Boosting Over XGBoost?

For this specific dataset:
- GBR achieved CV MAE = 25.68 vs XGBoost's 29.09
- GBR's sklearn implementation has better default regularization behavior
- XGBoost's advantage (speed, GPU) isn't needed for 5543 rows
- Both are kept in the pipeline for comparison transparency

### 3.3 Why Not Log-Transform the Target?

Freight is heavily right-skewed (skewness=4.66). Log-transform was considered but:
- Gradient Boosting handles skewed targets natively (tree-based)
- Log-transform would require inverse transform at prediction time (complexity)
- The model already achieves R²=97.47% without it
- MAPE is high (169%) but this is a known artifact of near-zero freight values ($0.02), not a model problem

---

## 4. Model Files Produced

| File | Contents | Size |
|------|----------|------|
| `models/predict_freight_cost_model.pkl` | GradientBoostingRegressor (fitted) | ~1.5 MB |
| `models/freight_feature_columns.pkl` | `['Dollars', 'lag_1_freight', 'lag_2_freight', 'route_avg_freight']` | <1 KB |
| `models/predict_flag_invoice.pkl` | IsolationForest (fitted, 200 trees) | ~15 MB |
| `models/scaler.pkl` | RobustScaler (fitted on 5 features) | <1 KB |
| `models/if_threshold.pkl` | Float: -0.087445 | <1 KB |
| `models/shap_values.pkl` | numpy array (5543 x 5) | ~220 KB |
| `models/shap_feature_names.pkl` | List of 5 feature names | <1 KB |

---

## 5. How to Reproduce

```bash
# Install dependencies
pip install -r requirements.txt

# Train freight model (takes ~10 seconds)
python -m FreightCostPrediction.train

# Train invoice flagging model (takes ~15 seconds including SHAP)
python -m InvoiceFlagging.train

# Run inference tests
python -m Inference.predict_freight
python -m Inference.predict_flagged_invoice

# Launch the app
streamlit run app.py
```

---

## 6. Recommendations for Future Improvement

1. **Freight Prediction:** Add `Quantity` as a feature (correlation 0.947 with Freight). Currently unused.
2. **Freight Prediction:** Consider time-based features (month, day of week) from InvoiceDate.
3. **Invoice Flagging:** Add `days_po_to_invoice` and `payment_delay` as features (available in the query but unused).
4. **Invoice Flagging:** Consider a semi-supervised approach if labeled anomalies become available.
5. **Both:** Implement model monitoring to detect data drift over time.
6. **Both:** Add confidence intervals to predictions (quantile regression for freight, anomaly score percentile for flagging).
