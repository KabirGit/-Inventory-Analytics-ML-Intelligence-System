import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from config import MODEL_DIR


def train_isolation_forest(X_scaled, feature_names):
    """
    Train Isolation Forest with statistical thresholding and SHAP explanations.
    
    Optimization rationale:
    - n_estimators=200: more trees = more stable anomaly scores (reduced variance)
    - max_samples='auto' (default 256): efficient subsampling for large datasets
    - max_features=1.0: use all features since we only have 5
    - contamination='auto': let the model determine threshold from scores rather than
      forcing a fixed contamination rate. We apply our own statistical threshold anyway.
    - bootstrap=False: standard IF behavior (sampling without replacement)
    
    The two-gate approach (IF label + statistical threshold) provides:
    1. IF catches structural anomalies in feature space
    2. Statistical threshold (mean - 2*std) filters out borderline cases,
       reducing false positives while keeping true anomalies
    
    Returns: (fitted model, threshold float)
    """
    # 1. Isolation Forest with optimized parameters
    model = IsolationForest(
        n_estimators=200,
        max_samples='auto',
        max_features=1.0,
        contamination='auto',  # Don't force contamination; use statistical threshold
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)

    # 2. Statistical thresholding
    scores = model.decision_function(X_scaled)
    
    # Use mean - 2*std for a more conservative threshold (fewer false positives)
    # This targets roughly the bottom 2.5% of a normal distribution
    threshold = scores.mean() - 2.0 * scores.std()

    # Two-gate labeling: must be flagged by BOTH IF and statistical threshold
    if_labels = model.predict(X_scaled)  # -1 for anomaly, 1 for normal
    flags = np.where((scores < threshold) & (if_labels == -1), 1, 0)

    # 3. Evaluation summary
    n_flagged = flags.sum()
    n_total = len(flags)
    contamination_rate = n_flagged / n_total

    print("\n" + "=" * 60)
    print("INVOICE FLAGGING - ISOLATION FOREST RESULTS")
    print("=" * 60)
    print(f"  Total invoices:        {n_total}")
    print(f"  Flagged as anomalous:  {n_flagged}")
    print(f"  Safe invoices:         {n_total - n_flagged}")
    print(f"  Contamination rate:    {contamination_rate:.4f} ({contamination_rate*100:.2f}%)")
    print(f"  Score mean:            {scores.mean():.6f}")
    print(f"  Score std:             {scores.std():.6f}")
    print(f"  Threshold (mean-2*std):{threshold:.6f}")
    print(f"  Score range:           [{scores.min():.6f}, {scores.max():.6f}]")
    
    # Distribution of anomaly scores
    percentiles = np.percentile(scores, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    print(f"\n  Score percentiles:")
    for p, v in zip([1, 5, 10, 25, 50, 75, 90, 95, 99], percentiles):
        marker = " <-- threshold" if abs(v - threshold) < 0.01 else ""
        print(f"    P{p:2d}: {v:.6f}{marker}")

    # 4. SHAP explanations
    try:
        import shap
        # TreeExplainer is exact and fast for tree-based models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)

        # Save SHAP values and feature names
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(shap_values, MODEL_DIR / "shap_values.pkl")
        joblib.dump(feature_names, MODEL_DIR / "shap_feature_names.pkl")

        # Print feature importance by mean absolute SHAP value
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = sorted(zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True)

        print(f"\n  {'=' * 50}")
        print(f"  KEY COST DRIVERS (SHAP Feature Importance)")
        print(f"  {'=' * 50}")
        for i, (feat, val) in enumerate(feature_importance):
            bar = "█" * int(val / max(mean_abs_shap) * 20)
            print(f"    {i+1}. {feat:30s} {val:.4f} {bar}")

    except Exception as e:
        print(f"\n  [WARNING] SHAP computation failed: {e}")
        print("  Continuing without SHAP explanations.")

    return model, threshold


def evaluate_anomalies(model, X_scaled, threshold):
    """Apply two-gate anomaly detection and return flags."""
    scores = model.decision_function(X_scaled)
    if_labels = model.predict(X_scaled)
    flags = np.where((scores < threshold) & (if_labels == -1), 1, 0)
    return flags
