"""Unsupervised anomaly detection using Isolation Forest on continuous sensor telemetry."""

import json
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import (
    NUMERICAL_FEATURES,
    ANOMALY_DETECTOR_PATH,
    ANOMALY_METRICS_PATH,
    ensure_directories_exist,
)


def build_anomaly_pipeline(contamination: float = 0.035) -> Pipeline:
    """Build Scikit-learn Pipeline with StandardScaler and Isolation Forest.

    Contamination is calibrated to ~3.5% reflecting the expected proportion of anomalous operations.
    Isolation Forest isolates observations by randomly selecting a feature and split value.
    Anomalies require significantly fewer splits to isolate in tree space.
    """
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "detector",
                IsolationForest(
                    contamination=contamination,
                    random_state=42,
                    n_estimators=100,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return pipeline


def train_anomaly_detector(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    contamination: float = 0.035,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Train Isolation Forest on training sensor data and evaluate on held-out test data.

    Evaluation against ground-truth failure labels is provided for informational comparison,
    with explicit caveats that unsupervised anomalies do not directly equate to failure events.
    """
    ensure_directories_exist()

    # Extract strictly numerical sensor features
    X_train_num = X_train[NUMERICAL_FEATURES].copy()
    X_test_num = X_test[NUMERICAL_FEATURES].copy()

    pipeline = build_anomaly_pipeline(contamination=contamination)
    print(f"Fitting Isolation Forest on {len(X_train_num)} numerical sensor readings...")
    pipeline.fit(X_train_num)

    # Save fitted detector pipeline
    joblib.dump(pipeline, ANOMALY_DETECTOR_PATH)
    print(f"Saved anomaly detector pipeline to {ANOMALY_DETECTOR_PATH}")

    # Compute baseline reference statistics (means & standard deviations for feature deviation attribution)
    scaler = pipeline.named_steps["scaler"]
    feature_stats = {
        feat: {
            "mean": float(round(scaler.mean_[i], 3)),
            "std": float(round(scaler.scale_[i], 3)),
        }
        for i, feat in enumerate(NUMERICAL_FEATURES)
    }

    # Evaluate against held-out test set
    # IsolationForest: 1 for normal, -1 for anomaly.
    test_preds_raw = pipeline.predict(X_test_num)
    test_scores = pipeline.decision_function(X_test_num)

    # Map -1 (anomaly) -> 1, 1 (normal) -> 0 for metric comparison against failure labels
    anomaly_flags = np.where(test_preds_raw == -1, 1, 0)
    cm = confusion_matrix(y_test, anomaly_flags)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model_type": "Isolation Forest",
        "contamination_parameter": contamination,
        "features_used": NUMERICAL_FEATURES,
        "feature_baseline_stats": feature_stats,
        "test_evaluation_vs_failure_labels": {
            "precision_vs_failure": float(round(precision_score(y_test, anomaly_flags, zero_division=0), 4)),
            "recall_vs_failure": float(round(recall_score(y_test, anomaly_flags, zero_division=0), 4)),
            "f1_vs_failure": float(round(f1_score(y_test, anomaly_flags, zero_division=0), 4)),
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
            "interpretation_note": (
                "Ground-truth failure labels serve as an imperfect reference for unsupervised anomaly detection. "
                "An unusual operating point (statistical outlier) is not necessarily a mechanical breakdown, "
                "and certain failures (such as gradual tool wear) can occur within standard operating envelopes."
            ),
        },
        "score_interpretation": {
            "decision_function_threshold": 0.0,
            "rule": "Score < 0 indicates atypical / anomalous operating condition; Score >= 0 indicates normal envelope.",
        },
    }

    with open(ANOMALY_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved anomaly detection metrics to {ANOMALY_METRICS_PATH}")
    return pipeline, metrics


def load_anomaly_detector() -> Tuple[Pipeline, Dict[str, Any]]:
    """Load the saved anomaly detector pipeline and metadata."""
    if not ANOMALY_DETECTOR_PATH.exists() or not ANOMALY_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Anomaly detector artifacts not found. Please run 'python scripts/train_models.py' first."
        )

    pipeline = joblib.load(ANOMALY_DETECTOR_PATH)
    with open(ANOMALY_METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return pipeline, metrics


def assess_anomaly(
    pipeline: Pipeline,
    metrics: Dict[str, Any],
    input_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Assess whether a machine reading is anomalous and pinpoint driving sensor deviations.

    Returns:
    - is_anomaly: bool (True if unusual reading)
    - decision_score: float (Scikit-learn decision function value, <0 is unusual)
    - normalized_anomaly_score: float in [0, 100] for visual gauge
    - feature_deviations: dict of z-scores indicating how each sensor deviates from training baseline
    - technical_explanation: objective description distinguishing statistical anomaly from root cause
    """
    df_input = pd.DataFrame([{feat: float(input_data[feat]) for feat in NUMERICAL_FEATURES}])

    raw_pred = pipeline.predict(df_input)[0]  # 1: normal, -1: outlier
    decision_score = float(pipeline.decision_function(df_input)[0])
    is_anomaly = bool(raw_pred == -1)

    # Compute sensor z-score deviations relative to training baseline
    stats = metrics.get("feature_baseline_stats", {})
    deviations = {}
    significant_deviations = []

    for feat in NUMERICAL_FEATURES:
        feat_val = float(input_data[feat])
        if feat in stats:
            mean = stats[feat]["mean"]
            std = stats[feat]["std"]
            z = (feat_val - mean) / std if std > 0 else 0.0
            deviations[feat] = {
                "reading": feat_val,
                "baseline_mean": mean,
                "z_score": round(z, 2),
                "is_extreme": abs(z) >= 2.0,
            }
            if abs(z) >= 2.0:
                direction = "high" if z > 0 else "low"
                significant_deviations.append(f"{feat} ({direction}, z={z:.1f})")
        else:
            deviations[feat] = {"reading": feat_val, "z_score": 0.0, "is_extreme": False}

    # Format technical explanation
    if is_anomaly:
        driver_text = (
            f"Significant sensor deviations detected: {', '.join(significant_deviations)}."
            if significant_deviations
            else "Sensor combination falls outside the multivariate training envelope."
        )
        explanation = (
            f"Atypical operating reading flagged (decision score: {decision_score:.3f} < 0.0). {driver_text} "
            "Note: This flags statistical distinctiveness from normal baseline cycles, "
            "not a confirmed physical defect or failure mode."
        )
    else:
        explanation = (
            f"Nominal operating reading (decision score: {decision_score:.3f} >= 0.0). "
            "All sensors operate within the multivariate distribution of normal training data."
        )

    return {
        "is_anomaly": is_anomaly,
        "status_label": "UNUSUAL / ATYPICAL" if is_anomaly else "NORMAL ENVELOPE",
        "decision_score": round(decision_score, 4),
        "feature_deviations": deviations,
        "significant_deviations": significant_deviations,
        "explanation": explanation,
    }
