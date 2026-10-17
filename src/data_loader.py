"""Data loading, dataset download, schema validation, and leakage-free feature separation."""

import io
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import (
    DATA_RAW_FILE,
    RAW_EXPECTED_COLUMNS,
    TARGET_COLUMN,
    LEAKAGE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ensure_directories_exist,
)

UCI_DATASET_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
)


def download_uci_dataset(destination_file: Path = DATA_RAW_FILE) -> Path:
    """Download the official UCI AI4I 2020 Predictive Maintenance dataset.

    Extracts 'ai4i2020.csv' and saves it to the specified destination.
    If the automated download fails, raises a RuntimeError with manual instructions.
    """
    ensure_directories_exist()

    if destination_file.exists() and destination_file.stat().st_size > 0:
        return destination_file

    print(f"Downloading dataset from UCI repository: {UCI_DATASET_ZIP_URL}...")
    headers = {"User-Agent": "Mozilla/5.0 (SentinelAI-HealthAssistant/1.0)"}
    request = urllib.request.Request(UCI_DATASET_ZIP_URL, headers=headers)

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read()
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                # Search for ai4i2020.csv inside zip
                target_csv_name = None
                for filename in z.namelist():
                    if filename.endswith(".csv"):
                        target_csv_name = filename
                        break

                if not target_csv_name:
                    raise FileNotFoundError("Could not find any .csv file inside UCI zip archive.")

                with z.open(target_csv_name) as csv_file:
                    with open(destination_file, "wb") as out_file:
                        out_file.write(csv_file.read())

        print(f"Successfully downloaded and saved dataset to: {destination_file}")
        return destination_file

    except Exception as exc:
        error_msg = (
            f"Failed to automatically download UCI AI4I 2020 dataset ({exc}).\n"
            f"Manual Setup Instructions:\n"
            f"1. Download the zip manually from:\n"
            f"   {UCI_DATASET_ZIP_URL}\n"
            f"   or visit: https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2B\n"
            f"2. Extract 'ai4i2020.csv' and place it at:\n"
            f"   {destination_file.resolve()}"
        )
        raise RuntimeError(error_msg) from exc


def load_raw_data(filepath: Path = DATA_RAW_FILE) -> pd.DataFrame:
    """Load raw dataset CSV and validate columns against expected schema."""
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset file not found at '{filepath}'. Please run 'python scripts/download_data.py' first."
        )

    df = pd.read_csv(filepath)

    # Validate column presence
    missing_cols = [c for c in RAW_EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset at '{filepath}' is missing expected columns: {missing_cols}. "
            f"Expected schema: {RAW_EXPECTED_COLUMNS}"
        )

    return df


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract features (X) and target (y) with strict target leakage prevention.

    Features excluded:
    - 'Machine failure': Supervised target.
    - 'UDI': Row index identifier.
    - 'Product ID': Redundant with 'Type' and contains arbitrary sequence numbers.
    - 'TWF', 'HDF', 'PWF', 'OSF', 'RNF': Specific failure mode indicators encoding failure.

    Retained predictive features:
    - Categorical: 'Type' (L, M, H)
    - Numerical: 'Air temperature [K]', 'Process temperature [K]',
                 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not present in dataframe.")

    y = df[TARGET_COLUMN].copy()

    # Drop target and all leakage / non-predictive columns
    cols_to_drop = [TARGET_COLUMN] + [c for c in LEAKAGE_COLUMNS if c in df.columns]
    X = df.drop(columns=cols_to_drop).copy()

    expected_features = set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES)
    actual_features = set(X.columns)

    if actual_features != expected_features:
        unexpected = actual_features - expected_features
        missing = expected_features - actual_features
        raise ValueError(
            f"Feature mismatch after leakage removal.\n"
            f"Unexpected columns: {unexpected}\n"
            f"Missing columns: {missing}"
        )

    # Ensure column ordering is deterministic
    ordered_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    X = X[ordered_cols]

    return X, y


def get_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a stratified train/test split.

    Stratification is essential because failures represent only ~3.4% of samples.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate high-level metadata summary of the raw dataset."""
    total_samples = len(df)
    failure_counts = df[TARGET_COLUMN].value_counts().to_dict() if TARGET_COLUMN in df else {}
    failures = failure_counts.get(1, 0)
    failure_rate = (failures / total_samples) * 100.0 if total_samples > 0 else 0.0

    return {
        "total_samples": total_samples,
        "features_count": df.shape[1],
        "failures": failures,
        "normal": failure_counts.get(0, 0),
        "failure_rate_pct": round(failure_rate, 2),
        "columns": list(df.columns),
    }
