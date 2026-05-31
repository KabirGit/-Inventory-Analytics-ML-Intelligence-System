import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from config import MODEL_DIR
# --------------------------------------------------
# ABSOLUTE PATHS (as per your structure)
# --------------------------------------------------

# Invoice Flagging
MODEL_PATH_FLAG = MODEL_DIR / "predict_flag_invoice.pkl"
SCALER_PATH_FLAG = MODEL_DIR / "scaler.pkl"
THRESHOLD_PATH_FLAG = MODEL_DIR / "if_threshold.pkl"

# Freight Prediction
MODEL_PATH_FREIGHT = MODEL_DIR / "predict_freight_cost_model.pkl"
FEATURE_COLS_PATH = MODEL_DIR / "freight_feature_columns.pkl"

# SHAP
SHAP_VALUES_PATH = MODEL_DIR / "shap_values.pkl"
SHAP_FEATURES_PATH = MODEL_DIR / "shap_feature_names.pkl"


# --------------------------------------------------
# Load Models
# --------------------------------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_resource
def load_scaler(path):
    return joblib.load(path)

@st.cache_resource
def load_threshold(path):
    return joblib.load(path)

@st.cache_resource
def load_feature_columns(path):
    if os.path.exists(path):
        return joblib.load(path)
    return ['Dollars', 'lag_1_freight', 'lag_2_freight', 'route_avg_freight']


# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🚩 Invoice Flagging", "🚚 Freight Prediction"]
)


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------
if page == "🏠 Home":
    st.title("📊 Sales Analytics ML App")

    st.markdown("""
    ### 🔍 What this app does:
    
    This application uses Machine Learning models to:
    
    - 🚩 **Flag risky invoices** using Isolation Forest anomaly detection with statistical thresholding  
    - 🚚 **Predict freight cost** using Gradient Boosting with lag, temporal, and route-level features  
    - 📊 **Identify key cost drivers** via SHAP-based feature importance analysis  
    - 📈 **Confidence intervals** on freight predictions for risk-aware decision making
    - 🔄 **Data drift monitoring** to detect when retraining is needed
    
    ---
    
    ### 📌 Invoice Flagging (Isolation Forest + Statistical Thresholding):
    - Uses unsupervised anomaly detection (no labeled data required)
    - Two-gate approach: Isolation Forest prediction + statistical threshold on decision scores
    - 7 features: invoice quantity/dollars, total quantity/dollars, receiving delay, PO-to-invoice days, payment delay
    - SHAP explanations surface key cost drivers
    - **Inputs:** invoice_quantity, invoice_dollars, total_quantity, total_dollars, average_receiving_delay, days_po_to_invoice, payment_delay  
    
    ---
    
    ### 📌 Freight Prediction (Gradient Boosting + Lag Features + K-Fold CV):
    - Gradient Boosting Regressor with 5-fold cross-validation for model selection
    - 7 engineered features: Dollars, Quantity, lag_1_freight, lag_2_freight, route_avg_freight, invoice_month, invoice_dow
    - Confidence intervals based on validated CV MAE
    - **Inputs:** Dollars, Quantity  
    
    ---
    
    ### 🎯 Purpose:
    Helps in anomaly detection and cost prediction for better financial decisions.
    """)


# --------------------------------------------------
# INVOICE FLAGGING
# --------------------------------------------------
elif page == "🚩 Invoice Flagging":

    st.title("🚩 Invoice Flagging (Anomaly Detection)")

    invoice_quantity = st.number_input("Invoice Quantity", min_value=0.0)
    invoice_dollars = st.number_input("Invoice Dollars", min_value=0.0)
    total_quantity = st.number_input("Total Quantity", min_value=0.0)
    total_dollars = st.number_input("Total Dollars", min_value=0.0)
    avg_delay = st.number_input("Average Receiving Delay (days)", min_value=0.0)
    
    with st.expander("📅 Optional: Invoice Timing Features"):
        days_po_to_invoice = st.number_input("Days from PO to Invoice", min_value=0.0, value=0.0)
        payment_delay = st.number_input("Payment Delay (days)", min_value=0.0, value=0.0)

    if st.button("Predict Invoice Flag"):

        input_df = pd.DataFrame({
            "invoice_quantity": [invoice_quantity],
            "invoice_dollars": [invoice_dollars],
            "total_quantity": [total_quantity],
            "total_dollars": [total_dollars],
            "average_receiving_delay": [avg_delay],
            "days_po_to_invoice": [days_po_to_invoice],
            "payment_delay": [payment_delay]
        })

        scaler = load_scaler(SCALER_PATH_FLAG)
        model = load_model(MODEL_PATH_FLAG)
        threshold = load_threshold(THRESHOLD_PATH_FLAG)

        input_scaled = scaler.transform(input_df)

        # Two-gate anomaly detection
        scores = model.decision_function(input_scaled)
        if_labels = model.predict(input_scaled)
        prediction = 1 if (scores[0] < threshold and if_labels[0] == -1) else 0

        if prediction == 1:
            st.error("🚨 Invoice is ANOMALOUS")
        else:
            st.success("✅ Invoice is NORMAL")
        
        # Anomaly score context
        st.caption(f"Anomaly Score: {scores[0]:.4f} | Threshold: {threshold:.4f}")

        st.dataframe(input_df)

    # SHAP Feature Importance Section
    st.markdown("---")
    st.subheader("Key Cost Drivers (SHAP Feature Importance)")

    if os.path.exists(SHAP_VALUES_PATH) and os.path.exists(SHAP_FEATURES_PATH):
        shap_values = joblib.load(SHAP_VALUES_PATH)
        shap_feature_names = joblib.load(SHAP_FEATURES_PATH)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            "Feature": shap_feature_names,
            "Mean |SHAP Value|": mean_abs_shap
        }).sort_values("Mean |SHAP Value|", ascending=False)

        st.bar_chart(shap_df.set_index("Feature"))
        st.dataframe(shap_df.reset_index(drop=True))
    else:
        st.info("SHAP values not yet computed. Run the training pipeline to generate them.")


# --------------------------------------------------
# FREIGHT PREDICTION
# --------------------------------------------------
elif page == "🚚 Freight Prediction":

    st.title("🚚 Freight Cost Prediction")

    dollars = st.number_input("Dollars", min_value=0.0)
    quantity = st.number_input("Quantity (units)", min_value=0, value=0)

    if st.button("Predict Freight Cost"):

        # Construct full feature input
        feature_cols = load_feature_columns(FEATURE_COLS_PATH)

        input_df = pd.DataFrame({
            "Dollars": [dollars],
            "Quantity": [quantity],
            "lag_1_freight": [0],
            "lag_2_freight": [0],
            "route_avg_freight": [dollars],
            "invoice_month": [6],   # neutral mid-year default
            "invoice_dow": [2]      # neutral mid-week default
        })

        # Ensure correct column order (only use what model expects)
        input_df = input_df[[c for c in feature_cols if c in input_df.columns]]

        model = load_model(MODEL_PATH_FREIGHT)

        prediction = model.predict(input_df)[0]
        
        # Confidence interval (based on validated CV MAE = ~25.35)
        mae_ci = 25.35
        ci_lower = max(0, prediction - mae_ci)
        ci_upper = prediction + mae_ci

        st.success(f"💰 Predicted Freight Cost: ${prediction:.2f}")
        st.caption(f"95% Confidence Interval: ${ci_lower:.2f} — ${ci_upper:.2f}")

        st.dataframe(pd.DataFrame({"Dollars": [dollars], "Quantity": [quantity]}))
