import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# ABSOLUTE PATHS (as per your structure)
# --------------------------------------------------

# Invoice Flagging
MODEL_PATH_FLAG = r"C:\Users\Kabir\Desktop\my SALES ANALYSIS\InvoiceFlagging\models\predict_flag_invoice.pkl"
SCALER_PATH_FLAG = r"C:\Users\Kabir\Desktop\my SALES ANALYSIS\InvoiceFlagging\models\scaler.pkl"  # adjust if different

# Freight Prediction
MODEL_PATH_FREIGHT = r"C:\Users\Kabir\Desktop\my SALES ANALYSIS\FreightCostPrediction\models\predict_freight_cost_model.pkl"


# --------------------------------------------------
# Load Models
# --------------------------------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_resource
def load_scaler(path):
    return joblib.load(path)


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
    
    - 🚩 Flag risky invoices  
    - 🚚 Predict freight cost  
    
    ---
    
    ### 📌 Invoice Flagging Inputs:
    - invoice_quantity  
    - invoice_dollars  
    - total_quantity  
    - total_dollars  
    - average_receiving_delay  
    
    ---
    
    ### 📌 Freight Prediction Input:
    - Dollars  
    
    ---
    
    ### 🎯 Purpose:
    Helps in anomaly detection and cost prediction for better financial decisions.
    """)


# --------------------------------------------------
# INVOICE FLAGGING
# --------------------------------------------------
elif page == "🚩 Invoice Flagging":

    st.title("🚩 Invoice Flagging")

    invoice_quantity = st.number_input("Invoice Quantity", min_value=0.0)
    invoice_dollars = st.number_input("Invoice Dollars", min_value=0.0)
    total_quantity = st.number_input("Total Quantity", min_value=0.0)
    total_dollars = st.number_input("Total Dollars", min_value=0.0)
    avg_delay = st.number_input("Average Receiving Delay", min_value=0.0)

    if st.button("Predict Invoice Flag"):

        input_df = pd.DataFrame({
            "invoice_quantity": [invoice_quantity],
            "invoice_dollars": [invoice_dollars],
            "total_quantity": [total_quantity],
            "total_dollars": [total_dollars],
            "average_receiving_delay": [avg_delay]
        })

        scaler = load_scaler(SCALER_PATH_FLAG)
        model = load_model(MODEL_PATH_FLAG)

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
            st.error("⚠️ Invoice is RISKY")
        else:
            st.success("✅ Invoice is SAFE")

        st.dataframe(input_df)


# --------------------------------------------------
# FREIGHT PREDICTION
# --------------------------------------------------
elif page == "🚚 Freight Prediction":

    st.title("🚚 Freight Cost Prediction")

    dollars = st.number_input("Dollars", min_value=0.0)

    if st.button("Predict Freight Cost"):

        input_df = pd.DataFrame({
            "Dollars": [dollars]
        })

        model = load_model(MODEL_PATH_FREIGHT)

        prediction = model.predict(input_df)[0]

        st.success(f"💰 Predicted Freight Cost: {prediction:.2f}")

        st.dataframe(input_df)