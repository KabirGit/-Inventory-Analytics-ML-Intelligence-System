import joblib
import numpy as np
import pandas as pd
from config import MODEL_DIR


# --------------------------------------------------
# 1. Load Model
# --------------------------------------------------
def load_model(model_path):
    model = joblib.load(model_path)
    return model


# --------------------------------------------------
# 2. Load Scaler
# --------------------------------------------------
def load_scaler(scaler_path):
    scaler = joblib.load(scaler_path)
    return scaler


# --------------------------------------------------
# 3. Load Threshold
# --------------------------------------------------
def load_threshold(threshold_path):
    threshold = joblib.load(threshold_path)
    return threshold


# --------------------------------------------------
# 4. Prediction Function
# --------------------------------------------------
def predict_invoice_flag(input_data: dict):
    """
    input_data: dictionary with required features
    returns: DataFrame with predictions (Predicted_Flag: 0=normal, 1=anomalous)
    
    Required keys: invoice_quantity, invoice_dollars, total_quantity, total_dollars, average_receiving_delay
    Optional keys: days_po_to_invoice, payment_delay (default to 0 if not provided)
    """

    # Convert dict to DataFrame
    df = pd.DataFrame(input_data)

    # Fill optional features with defaults if not provided
    if 'days_po_to_invoice' not in df.columns:
        df['days_po_to_invoice'] = 0
    if 'payment_delay' not in df.columns:
        df['payment_delay'] = 0

    # Ensure correct feature order (must match training)
    expected_features = [
        'invoice_quantity',
        'invoice_dollars',
        'total_quantity',
        'total_dollars',
        'average_receiving_delay',
        'days_po_to_invoice',
        'payment_delay'
    ]

    df = df[expected_features]

    # Load scaler, model, and threshold
    scaler = load_scaler(MODEL_DIR / "scaler.pkl")
    model = load_model(MODEL_DIR / "predict_flag_invoice.pkl")
    threshold = load_threshold(MODEL_DIR / "if_threshold.pkl")

    # Scale data
    df_scaled = scaler.transform(df)

    # Two-gate anomaly detection
    scores = model.decision_function(df_scaled)
    if_labels = model.predict(df_scaled)  # -1 for anomaly, 1 for normal
    predictions = np.where((scores < threshold) & (if_labels == -1), 1, 0)

    # Anomaly score percentile (for confidence/monitoring)
    result_df = df.copy()
    result_df["Predicted_Flag"] = predictions
    result_df["Anomaly_Score"] = scores

    return result_df


# --------------------------------------------------
# 5. Main Function (Demo)
# --------------------------------------------------
def main():

    # Demo input (3 samples)
    input_data = {
        "invoice_quantity": [10, 5, 2],
        "invoice_dollars": [2000, 300, 50],
        "total_quantity": [12, 6, 3],
        "total_dollars": [2100, 320, 55],
        "average_receiving_delay": [5, 12, 3]
    }

    result = predict_invoice_flag(input_data)

    print("\nInvoice Flag Predictions:\n")
    print(result)


# --------------------------------------------------
if __name__ == "__main__":
    main()
