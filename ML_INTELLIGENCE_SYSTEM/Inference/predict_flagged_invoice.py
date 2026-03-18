import joblib
import pandas as pd
import os


# --------------------------------------------------
# 1. Load Model
# --------------------------------------------------
def load_model(model_path):
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


# --------------------------------------------------
# 2. Load Scaler
# --------------------------------------------------
def load_scaler(scaler_path):
    with open(scaler_path, "rb") as f:
        scaler = joblib.load(f)
    return scaler


# --------------------------------------------------
# 3. Prediction Function
# --------------------------------------------------
def predict_invoice_flag(input_data: dict, model_path: str, scaler_path: str):
    """
    input_data: dictionary with required features
    returns: DataFrame with predictions
    """

    # Convert dict to DataFrame
    df = pd.DataFrame(input_data)

    # Ensure correct feature order (VERY IMPORTANT)
    expected_features = [
        'invoice_quantity',
        'invoice_dollars',
        'total_quantity',
        'total_dollars',
        'average_receiving_delay'
    ]

    df = df[expected_features]

    # Load scaler & model
    scaler = load_scaler(scaler_path)
    model = load_model(model_path)

    # Scale data
    df_scaled = scaler.transform(df)

    # Predict
    predictions = model.predict(df_scaled)

    # Output DataFrame
    result_df = df.copy()
    result_df["Predicted_Flag"] = predictions

    return result_df


# --------------------------------------------------
# 4. Main Function (Demo)
# --------------------------------------------------
def main():


    model_path = r"C:\Users\Kabir\Desktop\my SALES ANALYSIS\ML INTELLIGENCE SYSTEM\InvoiceFlagging\models\predict_flag_invoice.pkl"
    scaler_path = r"C:\Users\Kabir\Desktop\my SALES ANALYSIS\ML INTELLIGENCE SYSTEM\InvoiceFlagging\models\scaler.pkl"

    # Demo input (3 samples)
    input_data = {
        "invoice_quantity": [10, 5, 2],
        "invoice_dollars": [2000, 300, 50],
        "total_quantity": [12, 6, 3],
        "total_dollars": [2100, 320, 55],
        "average_receiving_delay": [5, 12, 3]
    }

    result = predict_invoice_flag(input_data, model_path, scaler_path)

    print("\nInvoice Flag Predictions:\n")
    print(result)


# --------------------------------------------------
if __name__ == "__main__":
    main()
    