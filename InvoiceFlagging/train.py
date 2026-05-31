import joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from InvoiceFlagging.data_preprocessing import load_data, prepare_and_scale
from InvoiceFlagging.model_eval import train_isolation_forest
from config import MODEL_DIR

def main():
    MODEL_DIR.mkdir(exist_ok=True)

    # Load and prepare data
    df = load_data()
    X_scaled, feature_names, scaler = prepare_and_scale(df)

    # Train Isolation Forest with statistical thresholding + SHAP
    model, threshold = train_isolation_forest(X_scaled, feature_names)

    # Save model
    joblib.dump(model, MODEL_DIR / "predict_flag_invoice.pkl")
    print(f"\nModel saved: {MODEL_DIR / 'predict_flag_invoice.pkl'}")

    # Save threshold
    joblib.dump(threshold, MODEL_DIR / "if_threshold.pkl")
    print(f"Threshold saved: {MODEL_DIR / 'if_threshold.pkl'}")

    # Scaler already saved in prepare_and_scale
    print(f"Scaler saved: {MODEL_DIR / 'scaler.pkl'}")

if __name__ == "__main__":
    main()
