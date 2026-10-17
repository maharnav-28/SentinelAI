"""Tests for data loading, validation, leakage prevention, and stratified splitting."""

import pytest
import pandas as pd
from pathlib import Path

from src.data_loader import (
    load_raw_data,
    prepare_features_and_target,
    get_train_test_split,
    get_dataset_summary,
)
from src.utils import (
    DATA_RAW_FILE,
    RAW_EXPECTED_COLUMNS,
    TARGET_COLUMN,
    LEAKAGE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)


def test_load_raw_data_success():
    """Verify raw dataset loads correctly and contains all expected columns."""
    assert DATA_RAW_FILE.exists(), "Raw dataset must exist for tests."
    df = load_raw_data(DATA_RAW_FILE)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10000
    for col in RAW_EXPECTED_COLUMNS:
        assert col in df.columns, f"Expected column '{col}' missing from raw dataset."


def test_load_raw_data_missing_file(tmp_path: Path):
    """Verify FileNotFoundError is raised when file does not exist."""
    fake_path = tmp_path / "non_existent.csv"
    with pytest.raises(FileNotFoundError):
        load_raw_data(fake_path)


def test_load_raw_data_invalid_schema(tmp_path: Path):
    """Verify ValueError is raised if critical schema columns are missing."""
    corrupt_file = tmp_path / "corrupt.csv"
    corrupt_df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
    corrupt_df.to_csv(corrupt_file, index=False)

    with pytest.raises(ValueError, match="missing expected columns"):
        load_raw_data(corrupt_file)


def test_prepare_features_and_target_leakage_prevention():
    """Verify target and all leakage features are stripped from predictors."""
    df = load_raw_data(DATA_RAW_FILE)
    X, y = prepare_features_and_target(df)

    # 1. Target must be separated cleanly
    assert TARGET_COLUMN not in X.columns
    assert len(y) == len(df)
    assert y.name == TARGET_COLUMN

    # 2. Every single leakage column must be excluded
    for leak_col in LEAKAGE_COLUMNS:
        assert leak_col not in X.columns, f"Leakage column '{leak_col}' was NOT dropped from predictors!"

    # 3. Only legitimate predictive features must remain
    expected_predictors = set(CATEGORICAL_FEATURES + NUMERICAL_FEATURES)
    assert set(X.columns) == expected_predictors
    assert len(X.columns) == 6


def test_stratified_train_test_split():
    """Verify stratified splitting maintains minority class ratio."""
    df = load_raw_data(DATA_RAW_FILE)
    X, y = prepare_features_and_target(df)

    X_train, X_test, y_train, y_test = get_train_test_split(X, y, test_size=0.2, random_state=42)

    assert len(X_train) == 8000
    assert len(X_test) == 2000

    # Class failure rate should match within ~0.2%
    overall_rate = y.mean()
    train_rate = y_train.mean()
    test_rate = y_test.mean()

    assert abs(train_rate - overall_rate) < 0.005
    assert abs(test_rate - overall_rate) < 0.005


def test_get_dataset_summary():
    """Verify summary dictionary outputs correct statistics."""
    df = load_raw_data(DATA_RAW_FILE)
    summary = get_dataset_summary(df)

    assert summary["total_samples"] == 10000
    assert summary["failures"] == 339
    assert summary["normal"] == 9661
    assert summary["failure_rate_pct"] == 3.39
