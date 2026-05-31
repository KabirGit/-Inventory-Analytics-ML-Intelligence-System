import joblib
import numpy as np
from pathlib import Path
import sys
from sklearn.model_selection import KFold, cross_val_score

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from FreightCostPrediction.DataPreprocessing import load_vendor_invoice, split_data, prepare_features
from FreightCostPrediction.model import (
    train_lr, train_ridge, train_dtr, train_rfr,
    get_xgboost_model, get_gbr_model, evaluate_model
)
from config import DATA_PATH, MODEL_DIR


def main():

    MODEL_DIR.mkdir(exist_ok=True)

    # load data
    df = load_vendor_invoice(str(DATA_PATH))
    print(f"\nDataset size: {len(df)} rows")

    # prepare features (now with leakage-free route_avg_freight)
    x, y, feature_cols = prepare_features(df)
    print(f"Features: {feature_cols}")
    print(f"Target: Freight (mean={y.mean():.2f}, std={y.std():.2f})")

    # 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    model_instances = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree (depth=4)": DecisionTreeRegressor(max_depth=4, random_state=42),
        "Random Forest (tuned)": RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_split=10,
            min_samples_leaf=5, max_features='sqrt', random_state=42, n_jobs=-1
        ),
        "XGBoost (tuned)": get_xgboost_model(),
        "Gradient Boosting (tuned)": get_gbr_model()
    }

    cv_results = []
    print("\n" + "=" * 60)
    print("5-FOLD CROSS-VALIDATION (scoring: neg_mean_absolute_error)")
    print("=" * 60)

    for name, model in model_instances.items():
        scores = cross_val_score(model, x, y, cv=kf, scoring='neg_mean_absolute_error')
        mean_mae = -scores.mean()
        std_mae = scores.std()
        print(f"  {name:30s} | MAE: {mean_mae:10.4f} (+/- {std_mae:.4f})")
        cv_results.append({"model": name, "mae": mean_mae, "std": std_mae})

    # Choose best model by lowest CV MAE
    best_cv_info = min(cv_results, key=lambda x: x["mae"])
    best_cv_name = best_cv_info["model"]
    print(f"\n  >>> Best (CV): {best_cv_name} | MAE={best_cv_info['mae']:.4f}")

    # Final train/test split for evaluation and saving
    xtrain, xtest, ytrain, ytest = split_data(x, y)
    print(f"\nTrain size: {len(xtrain)}, Test size: {len(xtest)}")

    # Train all models on the split
    trained_models = {}
    trained_models["Linear Regression"] = train_lr(xtrain, ytrain)
    trained_models["Ridge Regression"] = train_ridge(xtrain, ytrain)
    trained_models["Decision Tree (depth=4)"] = train_dtr(xtrain, ytrain)
    trained_models["Random Forest (tuned)"] = train_rfr(xtrain, ytrain)
    
    xgb_model = get_xgboost_model()
    xgb_model.fit(xtrain, ytrain)
    trained_models["XGBoost (tuned)"] = xgb_model
    
    gbr_model = get_gbr_model()
    gbr_model.fit(xtrain, ytrain)
    trained_models["Gradient Boosting (tuned)"] = gbr_model

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION (80/20 split)")
    print("=" * 60)
    
    results = []
    for name, model in trained_models.items():
        results.append(evaluate_model(model, xtest, ytest, name))

    # Select best from test evaluation (use MAE as primary metric)
    best_test_info = min(results, key=lambda x: x["mae"])
    best_test_name = best_test_info["model"]
    best_model = trained_models[best_test_name]

    # Check for overfitting: compare train vs test performance
    print("\n" + "=" * 60)
    print("OVERFITTING CHECK (Train MAE vs Test MAE)")
    print("=" * 60)
    
    for name, model in trained_models.items():
        train_pred = model.predict(xtrain)
        train_mae = np.mean(np.abs(ytrain - train_pred))
        test_result = next(r for r in results if r["model"] == name)
        test_mae = test_result["mae"]
        ratio = test_mae / train_mae if train_mae > 0 else float('inf')
        status = "OK" if ratio < 2.0 else "OVERFIT"
        print(f"  {name:30s} | Train: {train_mae:8.4f} | Test: {test_mae:8.4f} | Ratio: {ratio:.2f} [{status}]")

    # Save the best model
    model_path = MODEL_DIR / "predict_freight_cost_model.pkl"
    joblib.dump(best_model, model_path)

    # Save feature column names
    feature_cols_path = MODEL_DIR / "freight_feature_columns.pkl"
    joblib.dump(feature_cols, feature_cols_path)

    print(f"\n{'=' * 60}")
    print(f"FINAL RESULT")
    print(f"{'=' * 60}")
    print(f"  Best model: {best_test_name}")
    print(f"  Test MAE:   {best_test_info['mae']:.4f}")
    print(f"  Test RMSE:  {best_test_info['rmse']:.4f}")
    print(f"  Test R2:    {best_test_info['r2 score']:.2f}%")
    print(f"  Test MAPE:  {best_test_info['mape']:.2f}%")
    print(f"  Model path: {model_path}")
    print(f"  Features:   {feature_cols_path}")

    # Save drift monitor baseline
    from Inference.drift_monitor import DriftMonitor
    monitor = DriftMonitor.from_training_data(x.values, feature_cols)
    monitor.save()
    print(f"  Drift monitor saved: {MODEL_DIR / 'drift_monitor.pkl'}")

if __name__ == "__main__":
    main()
