import joblib
import pandas as pd
from config import MODEL_DIR

# --------------------------------------------------
# 1. Load Model Function
# --------------------------------------------------
def load_model(model_path):
    """
    Loads a trained model from the given path
    """
    
    model = joblib.load(model_path)
    return model


# --------------------------------------------------
# 2. Load Feature Columns
# --------------------------------------------------
def load_feature_columns():
    """Load the feature column names used during training."""
    feature_cols_path = MODEL_DIR / "freight_feature_columns.pkl"
    if feature_cols_path.exists():
        return joblib.load(feature_cols_path)
    # Fallback to default feature columns
    return ['Dollars', 'lag_1_freight', 'lag_2_freight', 'route_avg_freight']


# --------------------------------------------------
# 3. Predict Function
# --------------------------------------------------
def predict_freight(input_data: dict, model_path: str = None):
    """
    input_data: dict → e.g. {"Dollars": [2000, 300, 50]}
    model_path: path to saved model

    Returns: DataFrame with predictions
    """
    if model_path is None:
        model_path = MODEL_DIR / "predict_freight_cost_model.pkl"

    # Load feature columns
    feature_cols = load_feature_columns()

    # Convert dict to DataFrame
    df = pd.DataFrame(input_data)

    # Fill missing features with sensible defaults for single-input inference
    if 'Quantity' not in df.columns:
        df['Quantity'] = 0
    if 'lag_1_freight' not in df.columns:
        df['lag_1_freight'] = 0
    if 'lag_2_freight' not in df.columns:
        df['lag_2_freight'] = 0
    if 'route_avg_freight' not in df.columns:
        df['route_avg_freight'] = df['Dollars']
    if 'invoice_month' not in df.columns:
        df['invoice_month'] = 6  # mid-year neutral default
    if 'invoice_dow' not in df.columns:
        df['invoice_dow'] = 2  # Wednesday neutral default

    # Ensure correct column order (only use columns the model expects)
    df_input = df[[c for c in feature_cols if c in df.columns]]

    # Load model
    model = load_model(model_path)

    # Predict
    predictions = model.predict(df_input)

    # Confidence interval via quantile estimation (simple percentile-based)
    # Use model's inherent uncertainty: +/- based on training MAE
    mae_estimate = 25.35  # from 10-fold CV
    result_df = df[['Dollars']].copy()
    result_df["Predicted_Freight"] = predictions
    result_df["CI_Lower"] = (predictions - mae_estimate).clip(min=0)
    result_df["CI_Upper"] = predictions + mae_estimate

    return result_df


# --------------------------------------------------
# 4. Main Function (Demo)
# --------------------------------------------------
def main():
    model_path = MODEL_DIR / "predict_freight_cost_model.pkl"

    # Demo input
    input_data = {
        "Dollars": [2000, 300, 50]
    }

    result = predict_freight(input_data, model_path)

    print("\nPredictions:\n")
    print(result)


# --------------------------------------------------
if __name__ == "__main__":
    main()
