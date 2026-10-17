"""Central utilities, path constants, schema definitions, and helpers for SentinelAI."""

from pathlib import Path
from typing import Dict, List, Any
import os
from dotenv import load_dotenv

# Automatically load environment variables from .env if present
load_dotenv()

# Project directory paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_RAW_FILE = DATA_RAW_DIR / "ai4i2020.csv"
MAINTENANCE_DOCS_DIR = DATA_DIR / "maintenance_docs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Artifact file paths
CLASSIFIER_PIPELINE_PATH = ARTIFACTS_DIR / "classifier_pipeline.joblib"
CLASSIFIER_METRICS_PATH = ARTIFACTS_DIR / "classifier_metrics.json"
ANOMALY_DETECTOR_PATH = ARTIFACTS_DIR / "anomaly_detector.joblib"
ANOMALY_METRICS_PATH = ARTIFACTS_DIR / "anomaly_metrics.json"
FAISS_INDEX_PATH = ARTIFACTS_DIR / "faiss_index.bin"
RAG_CHUNKS_PATH = ARTIFACTS_DIR / "rag_chunks.json"

# UCI AI4I 2020 Dataset Schema Constants
RAW_EXPECTED_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

TARGET_COLUMN = "Machine failure"

# Leakage columns that must NOT be seen by the model:
# - 'UDI': arbitrary identifier
# - 'Product ID': serial code whose letter is already in 'Type' and digits are arbitrary sequence numbers
# - 'TWF', 'HDF', 'PWF', 'OSF', 'RNF': specific failure mode flags directly revealing the target
LEAKAGE_COLUMNS = [
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

NUMERICAL_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

CATEGORICAL_FEATURES = [
    "Type",
]

# Physical sensor boundaries for validation and UI controls
FEATURE_BOUNDS: Dict[str, Dict[str, Any]] = {
    "Air temperature [K]": {"min": 290.0, "max": 310.0, "default": 300.0, "step": 0.1, "unit": "K"},
    "Process temperature [K]": {"min": 300.0, "max": 320.0, "default": 310.0, "step": 0.1, "unit": "K"},
    "Rotational speed [rpm]": {"min": 1000.0, "max": 3000.0, "default": 1500.0, "step": 10.0, "unit": "rpm"},
    "Torque [Nm]": {"min": 0.0, "max": 100.0, "default": 40.0, "step": 0.5, "unit": "Nm"},
    "Tool wear [min]": {"min": 0.0, "max": 300.0, "default": 60.0, "step": 1.0, "unit": "min"},
    "Type": {"options": ["L", "M", "H"], "default": "L", "descriptions": {"L": "Low (50%)", "M": "Medium (30%)", "H": "High (20%)"}},
}


def ensure_directories_exist() -> None:
    """Ensure that all necessary project directories exist."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    MAINTENANCE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def validate_input_features(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate user input reading against realistic bounds.

    Returns the sanitized dictionary of inputs or raises ValueError.
    """
    sanitized = {}

    # Validate Type
    product_type = str(input_data.get("Type", "L")).upper()
    if product_type not in FEATURE_BOUNDS["Type"]["options"]:
        raise ValueError(
            f"Invalid machine Type '{product_type}'. Must be one of {FEATURE_BOUNDS['Type']['options']}."
        )
    sanitized["Type"] = product_type

    # Validate numerical sensor inputs
    for feat in NUMERICAL_FEATURES:
        if feat not in input_data:
            raise ValueError(f"Missing required sensor reading: '{feat}'")
        try:
            val = float(input_data[feat])
        except (ValueError, TypeError):
            raise ValueError(f"Sensor reading '{feat}' must be a numerical value, got {input_data[feat]}")

        bounds = FEATURE_BOUNDS[feat]
        if val < bounds["min"] or val > bounds["max"]:
            raise ValueError(
                f"Sensor reading '{feat}'={val} is out of realistic physical operating range "
                f"[{bounds['min']}, {bounds['max']}] {bounds['unit']}."
            )
        sanitized[feat] = val

    return sanitized
