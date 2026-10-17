"""Tests for utilities, path handling, and feature input validation."""

import pytest

from src.utils import (
    validate_input_features,
    FEATURE_BOUNDS,
    ensure_directories_exist,
    PROJECT_ROOT,
)


def test_validate_input_features_valid():
    """Verify clean validation for nominal machine readings."""
    valid_input = {
        "Type": "L",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 60.0,
    }
    sanitized = validate_input_features(valid_input)
    assert sanitized["Type"] == "L"
    assert sanitized["Air temperature [K]"] == 300.0
    assert sanitized["Torque [Nm]"] == 40.0


def test_validate_input_features_case_insensitivity():
    """Verify lower-case product type is sanitized to upper-case."""
    input_data = {
        "Type": "m",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 60.0,
    }
    sanitized = validate_input_features(input_data)
    assert sanitized["Type"] == "M"


def test_validate_input_features_invalid_type():
    """Verify ValueError is raised for invalid product type."""
    bad_input = {
        "Type": "Z",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 60.0,
    }
    with pytest.raises(ValueError, match="Invalid machine Type"):
        validate_input_features(bad_input)


def test_validate_input_features_out_of_bounds():
    """Verify ValueError is raised if a sensor reading exceeds physical boundaries."""
    # Negative torque is physically invalid
    bad_torque = {
        "Type": "L",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": -10.0,
        "Tool wear [min]": 60.0,
    }
    with pytest.raises(ValueError, match="out of realistic physical operating range"):
        validate_input_features(bad_torque)

    # Temperature way above reasonable boundary
    bad_temp = {
        "Type": "L",
        "Air temperature [K]": 450.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 60.0,
    }
    with pytest.raises(ValueError, match="out of realistic physical operating range"):
        validate_input_features(bad_temp)


def test_validate_input_features_missing_reading():
    """Verify ValueError is raised when a required sensor reading is omitted."""
    incomplete = {
        "Type": "L",
        "Air temperature [K]": 300.0,
        # Process temperature omitted
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 60.0,
    }
    with pytest.raises(ValueError, match="Missing required sensor reading"):
        validate_input_features(incomplete)


def test_validate_input_features_non_numeric():
    """Verify ValueError is raised when numerical sensor has a string text value."""
    corrupt_val = {
        "Type": "L",
        "Air temperature [K]": "three hundred",
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 60.0,
    }
    with pytest.raises(ValueError, match="must be a numerical value"):
        validate_input_features(corrupt_val)


def test_ensure_directories_exist():
    """Verify utility ensures directories exist."""
    ensure_directories_exist()
    assert (PROJECT_ROOT / "data" / "raw").exists()
    assert (PROJECT_ROOT / "data" / "maintenance_docs").exists()
    assert (PROJECT_ROOT / "artifacts").exists()
