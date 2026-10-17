"""Tests for supervised classification pipelines, evaluation metrics, and risk estimator."""

import pytest
import pandas as pd
from sklearn.pipeline import Pipeline

from src.train_classifier import (
    build_pipeline,
    evaluate_model,
    load_classifier,
    estimate_failure_risk,
)
from src.data_loader import (
    load_raw_data,
    prepare_features_and_target,
    get_train_test_split,
)
from src.utils import DATA_RAW_FILE


@pytest.fixture(scope="module")
def sample_data():
    df = load_raw_data(DATA_RAW_FILE)
    X, y = prepare_features_and_target(df)
    return get_train_test_split(X, y, test_size=0.2, random_state=42)


def test_build_pipelines():
    """Verify pipelines assemble with proper ColumnTransformer steps."""
    pipe_lr = build_pipeline("logistic_regression")
    assert isinstance(pipe_lr, Pipeline)
    assert "preprocessor" in pipe_lr.named_steps
    assert "classifier" in pipe_lr.named_steps

    pipe_rf = build_pipeline("random_forest")
    assert isinstance(pipe_rf, Pipeline)

    with pytest.raises(ValueError):
        build_pipeline("unsupported_model")


def test_classifier_evaluation_metrics(sample_data):
    """Verify evaluation returns all required imbalanced classification metrics."""
    X_train, X_test, y_train, y_test = sample_data

    # Use small subset for rapid testing of metric calculation logic
    pipe = build_pipeline("logistic_regression")
    pipe.fit(X_train.iloc[:500], y_train.iloc[:500])

    metrics = evaluate_model(pipe, X_test.iloc[:200], y_test.iloc[:200])

    required_keys = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "confusion_matrix",
    ]
    for key in required_keys:
        assert key in metrics, f"Metric '{key}' missing from evaluation results."

    cm = metrics["confusion_matrix"]
    assert "true_negatives" in cm
    assert "false_positives" in cm
    assert "false_negatives" in cm
    assert "true_positives" in cm


def test_load_trained_classifier():
    """Verify saved classifier pipeline and metadata can be loaded."""
    pipeline, metrics = load_classifier()
    assert isinstance(pipeline, Pipeline)
    assert "models" in metrics
    assert "random_forest" in metrics["models"]
    assert "logistic_regression" in metrics["models"]


def test_estimate_failure_risk():
    """Verify failure risk estimation output structure and boundary handling."""
    pipeline, _ = load_classifier()

    # Nominal healthy reading
    nominal_input = {
        "Type": "L",
        "Air temperature [K]": 298.0,
        "Process temperature [K]": 308.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 35.0,
        "Tool wear [min]": 30.0,
    }
    result_nominal = estimate_failure_risk(pipeline, nominal_input)

    assert "predicted_class" in result_nominal
    assert result_nominal["predicted_class"] in [0, 1]
    assert 0.0 <= result_nominal["failure_probability"] <= 1.0
    assert result_nominal["risk_tier"] in ["Low", "Moderate", "High", "Critical"]
    assert "calibration_notice" in result_nominal

    # Extreme reading (high wear, low speed, high torque)
    stressed_input = {
        "Type": "L",
        "Air temperature [K]": 304.0,
        "Process temperature [K]": 311.0,
        "Rotational speed [rpm]": 1250.0,
        "Torque [Nm]": 70.0,
        "Tool wear [min]": 230.0,
    }
    result_stressed = estimate_failure_risk(pipeline, stressed_input)
    # Stressed reading should yield higher estimated risk than nominal reading
    assert result_stressed["failure_probability"] > result_nominal["failure_probability"]
