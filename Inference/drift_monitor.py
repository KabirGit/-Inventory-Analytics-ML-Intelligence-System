"""
Data Drift Monitor — detects when incoming data distribution shifts
from the training distribution. Lightweight, no extra dependencies.

Usage:
    from Inference.drift_monitor import DriftMonitor
    monitor = DriftMonitor.load()
    report = monitor.check(new_data_df)
    if report['drift_detected']:
        print("WARNING: Data drift detected, consider retraining")
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from config import MODEL_DIR


class DriftMonitor:
    """Monitors feature distribution drift using simple statistical tests."""
    
    def __init__(self, feature_stats: dict, feature_names: list):
        """
        feature_stats: dict of {feature_name: {mean, std, min, max, q25, q50, q75}}
        """
        self.feature_stats = feature_stats
        self.feature_names = feature_names
    
    @classmethod
    def from_training_data(cls, X: np.ndarray, feature_names: list):
        """Build a monitor from training data statistics."""
        stats = {}
        for i, name in enumerate(feature_names):
            col = X[:, i] if isinstance(X, np.ndarray) else X.iloc[:, i].values
            stats[name] = {
                'mean': float(np.mean(col)),
                'std': float(np.std(col)),
                'min': float(np.min(col)),
                'max': float(np.max(col)),
                'q25': float(np.percentile(col, 25)),
                'q50': float(np.percentile(col, 50)),
                'q75': float(np.percentile(col, 75)),
            }
        return cls(feature_stats=stats, feature_names=feature_names)
    
    def save(self, path=None):
        """Save monitor to disk."""
        if path is None:
            path = MODEL_DIR / "drift_monitor.pkl"
        joblib.dump({'stats': self.feature_stats, 'features': self.feature_names}, path)
    
    @classmethod
    def load(cls, path=None):
        """Load monitor from disk."""
        if path is None:
            path = MODEL_DIR / "drift_monitor.pkl"
        if not Path(path).exists():
            return None
        data = joblib.load(path)
        return cls(feature_stats=data['stats'], feature_names=data['features'])
    
    def check(self, X_new: np.ndarray, threshold: float = 2.0) -> dict:
        """
        Check for drift in new data batch.
        
        Uses a simple z-score approach: if the mean of new data is more than
        `threshold` standard deviations away from training mean, flag drift.
        
        Returns dict with per-feature drift status and overall flag.
        """
        report = {'drift_detected': False, 'features': {}}
        
        for i, name in enumerate(self.feature_names):
            if name not in self.feature_stats:
                continue
            
            col = X_new[:, i] if isinstance(X_new, np.ndarray) else X_new.iloc[:, i].values
            train_stats = self.feature_stats[name]
            
            new_mean = float(np.mean(col))
            train_mean = train_stats['mean']
            train_std = train_stats['std']
            
            # Z-score of new mean relative to training distribution
            if train_std > 0:
                z_score = abs(new_mean - train_mean) / train_std
            else:
                z_score = 0.0 if new_mean == train_mean else float('inf')
            
            # Range check: % of new values outside training [min, max]
            out_of_range = float(np.mean((col < train_stats['min']) | (col > train_stats['max'])))
            
            drifted = z_score > threshold or out_of_range > 0.2
            
            report['features'][name] = {
                'z_score': z_score,
                'out_of_range_pct': out_of_range,
                'drifted': drifted,
                'new_mean': new_mean,
                'train_mean': train_mean
            }
            
            if drifted:
                report['drift_detected'] = True
        
        return report
    
    def print_report(self, report: dict):
        """Pretty-print a drift report."""
        status = "⚠️  DRIFT DETECTED" if report['drift_detected'] else "✅ No drift"
        print(f"\n{'=' * 50}")
        print(f"DATA DRIFT REPORT — {status}")
        print(f"{'=' * 50}")
        
        for name, info in report['features'].items():
            flag = "🔴" if info['drifted'] else "🟢"
            print(f"  {flag} {name:30s} z={info['z_score']:.2f}  OOR={info['out_of_range_pct']*100:.1f}%")
        
        if report['drift_detected']:
            print(f"\n  Recommendation: Consider retraining the model.")


def main():
    """Demo: check drift on the current training data (should show no drift)."""
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from FreightCostPrediction.DataPreprocessing import load_vendor_invoice, prepare_features
    from config import DATA_PATH
    
    # Load and check freight model drift
    monitor = DriftMonitor.load()
    if monitor is None:
        print("No drift monitor found. Run training first to generate one.")
        return
    
    df = load_vendor_invoice(str(DATA_PATH))
    x, y, _ = prepare_features(df)
    
    report = monitor.check(x.values)
    monitor.print_report(report)


if __name__ == "__main__":
    main()
