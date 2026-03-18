import joblib
import pandas as pd


# --------------------------------------------------
# 1. Load Model Function
# --------------------------------------------------
def load_model(model_path):
    """
    Loads a trained model from the given path
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


# --------------------------------------------------
# 2. Predict Function
# --------------------------------------------------
def predict_freight(input_data: dict, model_path: str):
    """
    input_data: dict → e.g. {"Dollars": [2000, 300, 50]}
    model_path: path to saved model

    Returns: DataFrame with predictions
    """

    # Convert dict to DataFrame
    df = pd.DataFrame(input_data)

    # Load model
    model = load_model(model_path)

    # Predict
    predictions = model.predict(df)

    # Return result as DataFrame
    result_df = df.copy()
    result_df["Predicted_Freight"] = predictions

    return result_df


# --------------------------------------------------
# 3. Main Function (Demo)
# --------------------------------------------------
def main():
    model_path = r"C:\Users\Kabir\Desktop\my SALES ANALYSIS\ML INTELLIGENCE SYSTEM\FreightCostPrediction\models\predict_freight_cost_model.pkl"  # Update with your actual model path

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