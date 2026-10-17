"""Tests for unsupervised anomaly detector (Isolation Forest) and deviation diagnostics."""

import pytest
import pandas as pd
from sklearn.pipeline import Pipeline

from src.anomaly_detector import (
    build_anomaly_pipeline,
    load_anomaly_detector,
    assess_anomaly,
)
from src.utils import NUMERICAL_FEATURES


def test_build_anomaly_pipeline():
    """Verify Isolation Forest pipeline includes scaler and detector steps."""
    pipe = build_anomaly_pipeline(contamination=0.035)
    assert isinstance(pipe, Pipeline)
    assert "scaler" in pipe.named_steps
    assert "detector" in pipe.named_steps


def test_load_anomaly_detector():
    """Verify saved anomaly detector artifacts can be loaded from disk."""
    pipeline, metrics = load_anomaly_detector()
    assert isinstance(pipeline, Pipeline)
    assert "feature_baseline_stats" in metrics
    for feat in NUMERICAL_FEATURES:
        assert feat in metrics["feature_baseline_stats"]


def test_assess_nominal_reading():
    """Verify nominal sensor readings are classified as normal envelope."""
    pipeline, metrics = load_anomaly_detector()

    nominal_reading = {
        "Type": "L",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1530.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 60.0,
    }

    result = assess_anomaly(pipeline, metrics, nominal_reading)
    assert "is_anomaly" in result
    assert isinstance(result["is_anomaly"], bool)
    assert "decision_score" in result
    assert isinstance(result["decision_score"], float)
    assert "feature_deviations" in result
    assert "explanation" in result
    assert result["is_anomaly"] is False
    assert result["decision_score"] >= 0.0


def test_assess_extreme_anomaly_reading():
    """Verify extreme outlier sensor readings are flagged as unusual."""
    pipeline, metrics = load_anomaly_detector()

    # Extreme outlier reading (near max physical speed & torque)
    extreme_reading = {
        "Type": "L",
        "Air temperature [K]": 308.0,
        "Process temperature [K]": 319.0,
        "Rotational speed [rpm]": 2850.0,
        "Torque [Nm]": 85.0,
        "Tool wear [min]": 260.0,
    }

    result = assess_anomaly(pipeline, metrics, extreme_reading)
    assert result["is_anomaly"] is True
    assert result["decision_score"] < 0.0
    assert result["status_label"] == "UNUSUAL / ATYPICAL"
    assert len(result["significant_deviations"]) > 0
