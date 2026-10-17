"""Supervised classification pipelines for machine failure risk prediction."""

import json
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    CLASSIFIER_PIPELINE_PATH,
    CLASSIFIER_METRICS_PATH,
    ensure_directories_exist,
)


def create_preprocessor() -> ColumnTransformer:
    """Create Scikit-learn ColumnTransformer for preprocessing.

    - Numerical features: Standardized with StandardScaler (zero mean, unit variance)
    - Categorical features: One-hot encoded with explicit category ordering
    """
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        categories=[["L", "M", "H"]],
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def build_pipeline(model_type: str = "random_forest") -> Pipeline:
    """Construct an end-to-end Scikit-learn Pipeline with preprocessing and classifier.

    Supported model_types: 'logistic_regression' (baseline), 'random_forest' (candidate).
    Uses class_weight='balanced' to handle the ~3.4% severe class imbalance.
    """
    preprocessor = create_preprocessor()

    if model_type == "logistic_regression":
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
        )
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'. Choose 'logistic_regression' or 'random_forest'.")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Evaluate pipeline on held-out test data with comprehensive metrics for imbalanced classification."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(round(accuracy_score(y_test, y_pred), 4)),
        "balanced_accuracy": float(round(balanced_accuracy_score(y_test, y_pred), 4)),
        "precision": float(round(precision_score(y_test, y_pred, zero_division=0), 4)),
        "recall": float(round(recall_score(y_test, y_pred, zero_division=0), 4)),
        "f1_score": float(round(f1_score(y_test, y_pred, zero_division=0), 4)),
        "roc_auc": float(round(roc_auc_score(y_test, y_prob), 4)),
        "pr_auc": float(round(average_precision_score(y_test, y_prob), 4)),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "sample_counts": {
            "test_total": int(len(y_test)),
            "test_actual_failures": int(y_test.sum()),
            "test_actual_normal": int((y_test == 0).sum()),
        },
    }
    return metrics


def train_and_evaluate_all(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Train both baseline and candidate models, evaluate on held-out test set, and save results."""
    ensure_directories_exist()

    models = {
        "logistic_regression": build_pipeline("logistic_regression"),
        "random_forest": build_pipeline("random_forest"),
    }

    comparison = {}
    fitted_pipelines = {}

    for name, pipe in models.items():
        print(f"Training {name} on {len(X_train)} samples...")
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        metrics = evaluate_model(pipe, X_test, y_test)
        comparison[name] = metrics
        print(
            f"  {name} Results -> Recall: {metrics['recall']:.3f}, Precision: {metrics['precision']:.3f}, "
            f"F1: {metrics['f1_score']:.3f}, PR-AUC: {metrics['pr_auc']:.3f}, ROC-AUC: {metrics['roc_auc']:.3f}"
        )

    # We select Random Forest as the primary operational pipeline due to superior non-linear modeling
    best_pipeline = fitted_pipelines["random_forest"]

    # Save the selected pipeline
    joblib.dump(best_pipeline, CLASSIFIER_PIPELINE_PATH)
    print(f"Saved primary classifier pipeline to {CLASSIFIER_PIPELINE_PATH}")

    # Save comparison metadata
    metadata = {
        "primary_model": "random_forest",
        "baseline_model": "logistic_regression",
        "models": comparison,
        "features": {
            "numerical": NUMERICAL_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
        "class_imbalance": {
            "train_failures": int(y_train.sum()),
            "train_normal": int((y_train == 0).sum()),
            "test_failures": int(y_test.sum()),
            "test_normal": int((y_test == 0).sum()),
        },
    }

    with open(CLASSIFIER_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved evaluation comparison metrics to {CLASSIFIER_METRICS_PATH}")
    return best_pipeline, metadata


def load_classifier() -> Tuple[Pipeline, Dict[str, Any]]:
    """Load the saved classifier pipeline and its evaluation metrics."""
    if not CLASSIFIER_PIPELINE_PATH.exists() or not CLASSIFIER_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Classifier artifacts not found. Please train models first using 'python scripts/train_models.py'."
        )

    pipeline = joblib.load(CLASSIFIER_PIPELINE_PATH)
    with open(CLASSIFIER_METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return pipeline, metrics


def estimate_failure_risk(pipeline: Pipeline, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate machine failure risk from a dictionary of validated readings.

    Returns:
    - predicted_class: 0 (Normal) or 1 (Failure Risk)
    - failure_probability: estimated float in [0.0, 1.0]
    - risk_tier: 'Low', 'Moderate', 'High', or 'Critical'
    - explanation: context on trade-off and probability calibration note
    """
    df_input = pd.DataFrame([input_data])
    predicted_class = int(pipeline.predict(df_input)[0])
    prob = float(pipeline.predict_proba(df_input)[0, 1])

    if prob < 0.20:
        tier = "Low"
    elif prob < 0.50:
        tier = "Moderate"
    elif prob < 0.80:
        tier = "High"
    else:
        tier = "Critical"

    return {
        "predicted_class": predicted_class,
        "failure_probability": round(prob, 4),
        "risk_percentage": round(prob * 100.0, 2),
        "risk_tier": tier,
        "calibration_notice": (
            "Model estimate derived from Random Forest probability outputs. "
            "This reflects an empirical risk score rather than a strictly calibrated frequentist probability."
        ),
    }
