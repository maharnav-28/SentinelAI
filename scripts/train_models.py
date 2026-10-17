"""Train and evaluate supervised classifier and unsupervised anomaly detector."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    load_raw_data,
    download_uci_dataset,
    prepare_features_and_target,
    get_train_test_split,
)
from src.train_classifier import train_and_evaluate_all
from src.anomaly_detector import train_anomaly_detector
from src.utils import DATA_RAW_FILE


def main():
    print("=" * 60)
    print("SentinelAI: Training Classification & Anomaly Models")
    print("=" * 60)

    # 1. Ensure dataset exists
    if not DATA_RAW_FILE.exists():
        print("Dataset not found locally. Downloading official UCI dataset...")
        download_uci_dataset()

    # 2. Load dataset
    print(f"Loading dataset from: {DATA_RAW_FILE}...")
    df = load_raw_data(DATA_RAW_FILE)

    # 3. Extract features and target with leakage prevention
    print("Extracting features with strict leakage prevention...")
    X, y = prepare_features_and_target(df)
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    print(f"Predictor features: {list(X.columns)}")

    # 4. Stratified train/test split
    print("Generating stratified 80/20 train/test split...")
    X_train, X_test, y_train, y_test = get_train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train samples: {len(X_train)} (Failures: {y_train.sum()} / {len(y_train):,})")
    print(f"Test samples:  {len(X_test)} (Failures: {y_test.sum()} / {len(y_test):,})")

    # 5. Train and evaluate classifiers (Baseline vs Candidate)
    print("\n--- Training Supervised Classifiers ---")
    _, classifier_metrics = train_and_evaluate_all(X_train, y_train, X_test, y_test)

    # Print summary table
    print("\n=== Model Evaluation Comparison (Held-Out Test Set: 2,000 samples) ===")
    print(f"{'Metric':<22} | {'Logistic Regression':<20} | {'Random Forest':<15}")
    print("-" * 65)
    for metric_name in ["accuracy", "balanced_accuracy", "precision", "recall", "f1_score", "pr_auc", "roc_auc"]:
        lr_val = classifier_metrics["models"]["logistic_regression"][metric_name]
        rf_val = classifier_metrics["models"]["random_forest"][metric_name]
        print(f"{metric_name:<22} | {lr_val:<20.4f} | {rf_val:<15.4f}")

    # 6. Train and evaluate Isolation Forest
    print("\n--- Training Unsupervised Anomaly Detector ---")
    _, anomaly_metrics = train_anomaly_detector(X_train, X_test, y_test, contamination=0.035)

    print("\nModel training complete! All artifacts saved to 'artifacts/'.")


if __name__ == "__main__":
    main()
