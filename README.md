# SentinelAI — Smart Machine Health Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-00599C.svg)](https://github.com/facebookresearch/faiss)
[![Tests: Pytest](https://img.shields.io/badge/Tests-27%20Passed-brightgreen.svg)](https://docs.pytest.org/)

An interpretable, industrial-grade predictive maintenance and machine health monitoring system built with **Scikit-learn**, **Sentence-Transformers**, **FAISS**, and **Streamlit**. 

Developed for rotating industrial machinery (milling and turning equipment), **SentinelAI** combines supervised failure risk estimation, unsupervised anomaly detection, and a grounded Retrieval-Augmented Generation (RAG) assistant for traceable maintenance operating procedures.

---

## Table of Contents
1. [Problem Statement & Objectives](#problem-statement--objectives)
2. [Core Engineering Distinction](#core-engineering-distinction)
3. [System Architecture](#system-architecture)
4. [Dataset & Strict Leakage Prevention](#dataset--strict-leakage-prevention)
5. [Supervised Failure Risk Classification](#supervised-failure-risk-classification)
6. [Unsupervised Anomaly Detection](#unsupervised-anomaly-detection)
7. [RAG Maintenance Assistant & Fallback](#rag-maintenance-assistant--fallback)
8. [Measured Benchmark Results](#measured-benchmark-results)
9. [Project Organization](#project-organization)
10. [Installation & Setup](#installation--setup)
11. [Running the Application](#running-the-application)
12. [Testing](#testing)
13. [Deployment Guide (Self-Hosted)](#deployment-guide-self-hosted)
14. [Troubleshooting & FAQ](#troubleshooting--faq)
15. [Limitations & Future Roadmap](#limitations--future-roadmap)

---

## Problem Statement & Objectives

Unplanned downtime in precision manufacturing costs industrial facilities billions of dollars annually. When a cutting tool catastrophically fractures or spindle bearings seize, the financial repercussions include ruined workpieces, severe equipment damage, operator safety hazards, and multi-day assembly line halts.

**SentinelAI** addresses this challenge with three interconnected engineering objectives:
1. **Estimate Machine Failure Risk:** Predict failure probability on incoming operational telemetry using a supervised model trained with strict leakage prevention.
2. **Flag Atypical Sensor Telemetry:** Identify statistically unusual multidimensional sensor readings using unsupervised Isolation Forest without requiring failure labels.
3. **Provide Grounded Maintenance Guidance:** Retrieve verified Standard Operating Procedures (SOPs) using FAISS dense vector search and synthesize grounded repair recommendations without hallucinations.
4. **Deliver a Transparent Dashboard:** Present live telemetry, risk tiers, z-score parameter diagnostics, and traceable citations through a responsive Streamlit UI.

---

## Core Engineering Distinction

Industrial maintenance systems must never conflate statistical signals. SentinelAI strictly separates:

| Signal Type | Mechanism | Meaning | Action Required |
| :--- | :--- | :--- | :--- |
| **Failure Risk Score** | Supervised Random Forest Classifier | Empirical likelihood of machine breakdown based on historical failure data. | Schedule targeted preventive maintenance before cycle completion. |
| **Unusual Reading (Anomaly)** | Unsupervised Isolation Forest Detector | Sensor combination is statistically atypical ($z$-score deviations from normal cluster). | Inspect sensors for calibration drift or benign heavy operating point. |
| **Maintenance Guidance** | Dense Vector Search (FAISS) + RAG | Verified procedure extracted directly from technical operating manuals. | Follow documented Lockout/Tagout (LOTO) steps and replacement criteria. |

> [!IMPORTANT]
> An unusual sensor reading does **not** automatically indicate a confirmed machine breakdown, nor does it establish physical root cause without mechanical inspection.

---

## System Architecture

```
                                [ Machine Sensor Telemetry ]
                   (T_air, T_process, Rotational speed, Torque, Tool wear, Type)
                                              |
                   +--------------------------+--------------------------+
                   |                                                     |
                   v                                                     v
      [ Supervised Classification ]                         [ Unsupervised Anomaly ]
    - Standard Scaler (Numerical)                         - Standard Scaler (Sensors only)
    - OneHotEncoder (Type: L, M, H)                       - Isolation Forest (Contamination=3.5%)
    - Random Forest (Balanced Weights)                                   |
                   |                                                     v
                   v                                          [ Atypical Flag & Z-Scores ]
        [ Failure Risk Score (%) ]                            - Decision Score (<0 indicates outlier)
        - Risk Tier (Low/Mod/High/Crit)                       - Parameter Deviation Breakdown
                   \                                                     /
                    +-------------------------+-------------------------+
                                              |
                                              v
                              [ RAG Maintenance Assistant ]
                        - Markdown SOPs: Thermal, Wear, Power, Bearings
                        - Embeddings: all-MiniLM-L6-v2 (384-dim)
                        - Vector Index: FAISS IndexFlatIP (Cosine Similarity)
                                              |
                        +---------------------+---------------------+
                        | (API Key Present)                         | (No API Key / Offline)
                        v                                           v
            [ Grounded LLM Response ]                   [ Retrieval-Only Fallback ]
            - Strict Context Prompting                  - Direct Ranked Passages
            - Traceable Document Citations              - Verified Excerpts & Scores
```

---

## Dataset & Strict Leakage Prevention

SentinelAI uses the official **UCI AI4I 2020 Predictive Maintenance Dataset**:
- **Source:** [UCI Machine Learning Repository (ID 601)](https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2B)
- **Scale:** 10,000 synthetic records reflecting real-world industrial milling dynamics.
- **Class Imbalance:** 9,661 normal operating records (96.61%) vs. 339 failure records (3.39%).

### Strict Leakage Prevention Protocol
Target leakage occurs when training data contains features that directly encode the target or would not be available in real-world deployment:
1. **Target:** `Machine failure` (binary target: 0 = nominal, 1 = failure).
2. **Excluded Failure Modes:** `TWF` (Tool Wear Failure), `HDF` (Heat Dissipation Failure), `PWF` (Power Failure), `OSF` (Overstrain Failure), and `RNF` (Random Failure). These flags encode exact breakdown events and would lead to artificial, unusable 100% test accuracy.
3. **Excluded Identifiers:** `UDI` is an arbitrary sequence number (1–10,000).
4. **Product ID Treatment:** `Product ID` (e.g. `L47181`) consists of a quality letter (`L`, `M`, `H`) and an arbitrary sequence number. The variant is already captured cleanly in `Type` (Low 50%, Medium 30%, High 20%). The numeric serial suffix is dropped to prevent arbitrary leaf splits.
5. **Legitimate Predictor Features:**
   - Categorical: `Type` (One-hot encoded)
   - Numerical: `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`
6. **Isolated Preprocessing:** Scaling and encoding parameters are fit **only** on the training set through `sklearn.pipeline.Pipeline` and `ColumnTransformer`.

---

## Supervised Failure Risk Classification

We evaluate two complementary classification paradigms on a held-out 20% stratified test set (2,000 samples: 1,932 normal, 68 failures):
1. **Baseline Model:** Logistic Regression with `class_weight='balanced'`.
2. **Primary Candidate:** Random Forest Classifier (100 trees, `max_depth=10`, `class_weight='balanced'`).

### Why Accuracy is Misleading
In imbalanced predictive maintenance (3.39% positive class), a naive "predict zero always" baseline achieves **96.61% accuracy** while missing **100% of machine breakdowns**!
- SentinelAI optimizes for **Recall** (minimizing False Negatives / missed breakdowns) and **PR-AUC** (Precision-Recall Area Under Curve).

### False Negatives vs. False Positives
- **False Negative (Missed Failure):** The model fails to alert; machine fractures during production. Cost: Tool destruction, spindle damage, scrapped parts, unplanned shutdown. Extremely high cost.
- **False Positive (False Alarm):** The model flags a healthy machine. Cost: 10-minute visual inspection by a technician. Low cost.
- *Trade-off:* While Logistic Regression captures 82.4% Recall, it generates **339 false alarms** (Precision = 14.2%), inducing severe operator alarm fatigue. Random Forest catches **75.0% of failures** with only **54 false alarms** (Precision = 48.6%), raising PR-AUC from **0.382** to **0.647**.

---

## Unsupervised Anomaly Detection

Built using **Isolation Forest** on the continuous numerical telemetry:
- **Theory:** Anomalous data points require fewer random axis-aligned partitions to isolate in feature space compared to normal points in dense clusters.
- **Parameters:** `contamination=0.035` (calibrated to the ~3.5% empirical rate of atypical operations), 100 isolation trees.
- **Score Interpretation:**
  - Decision function score $\ge 0.0$: Nominal operating envelope.
  - Decision function score $< 0.0$: Atypical sensor combination.
- **Z-Score Parameter Diagnostics:**
  The system computes standardized deviations for every sensor:
  $$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
  Any sensor with $|z| \ge 2.0$ is highlighted to show which physical parameter drives the atypical status.

---

## RAG Maintenance Assistant & Fallback

The maintenance assistant provides traceable technical advice grounded in four comprehensive Standard Operating Procedures located in `data/maintenance_docs/`:
1. `SOP_01_Thermal_Dissipation_HDF.md`: Heat dissipation threshold ($\Delta T < 8.6\text{ K}$, speed $< 1380\text{ rpm}$), convective airflow, and fan cleaning.
2. `SOP_02_Tool_Wear_TWF.md`: Tool wear limits (200 min for L, 220 min for M, 240 min for H), 180 min inspection, and insert replacement.
3. `SOP_03_Spindle_Power_and_Torque_PWF.md`: Mechanical power formula $P = \frac{2\pi N \tau}{60}$, power cutoff limits (3.5 kW – 9 kW), and overstrain product ($\tau \cdot t$).
4. `SOP_04_Rotational_Speed_and_Bearings.md`: Speed limits (1,200 – 2,800 rpm), ISO VG 32 grease re-lubrication (2.0 cm³), vibration (< 1.8 mm/s RMS), and runout (< 5 µm).

### Transparent Vector Search
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` produces 384-dimensional dense vectors.
- **Indexing:** Vectors are L2-normalized and indexed using `faiss.IndexFlatIP`, enabling exact cosine similarity retrieval without external vector database servers.

### Zero-API-Key Fallback Mode
If no `LLM_API_KEY` is provided in `.env`:
- The application **does not crash**.
- It activates **Retrieval-Only Mode**, displaying top ranked SOP passages, similarity scores, and verified section citations.
- If an API key is configured (Gemini or OpenAI), the assistant synthesizes a grounded answer under strict anti-hallucination prompt constraints.

---

## Measured Benchmark Results

Measured on the untouched 2,000-sample test set (20% stratified holdout):

| Metric | Logistic Regression (Baseline) | Random Forest (Candidate) | Winner / Rationale |
| :--- | :---: | :---: | :--- |
| **Accuracy** | 82.45% | **96.45%** | Random Forest (+14.0%) |
| **Balanced Accuracy** | 82.40% | **86.10%** | Random Forest (+3.7%) |
| **Recall (Failure Class 1)** | **82.35%** | 75.00% | Logistic Regression catches 56/68; RF catches 51/68 |
| **Precision (Failure Class 1)**| 14.18% | **48.57%** | Random Forest cuts false alarms from 339 to **54** |
| **F1-Score** | 0.2419 | **0.5896** | Random Forest (+143% improvement) |
| **PR-AUC (Avg Precision)** | 0.3818 | **0.6472** | **Random Forest demonstrates superior precision-recall trade-off** |
| **ROC-AUC** | 0.9070 | **0.9607** | Random Forest (+5.4%) |

### Confusion Matrices (Test Set: 2,000 Samples)
- **Logistic Regression:** TN = 1,593 | FP = 339 | FN = 12 | TP = 56
- **Random Forest:** TN = 1,878 | FP = 54 | FN = 17 | TP = 51

---

## Project Organization

```
SentinelAI/
├── .env.example               # Configuration template with placeholders only
├── .gitignore                  # Git ignore rules (secrets, venvs, caches)
├── README.md                   # Complete architectural and operational documentation
├── requirements.txt            # Reproducible dependency constraints
├── pytest.ini                  # Pytest configuration with pythonpath set
├── app.py                      # Interactive Streamlit multi-page dashboard
│
├── src/                        # Core application modules
│   ├── __init__.py
│   ├── utils.py                # Paths, bounds, schema constants, and validation
│   ├── data_loader.py          # Download, schema check, leakage drop, splitting
│   ├── train_classifier.py     # LR & RF pipelines, metrics, and risk estimator
│   ├── anomaly_detector.py     # Isolation Forest pipeline and z-score diagnostics
│   └── rag.py                  # Chunking, SentenceTransformers, FAISS, LLM/Fallback
│
├── scripts/                    # Automation and pipeline runners
│   ├── download_data.py        # Automated download of official UCI dataset
│   ├── train_models.py         # Model training & artifact generation runner
│   └── build_rag_index.py      # SOP chunking & FAISS vector index generator
│
├── data/                       # Data storage
│   ├── raw/                    # Stores ai4i2020.csv (auto-downloaded)
│   └── maintenance_docs/       # Markdown engineering SOPs (thermal, wear, power, speed)
│
├── artifacts/                  # Persisted models and vector stores
│   ├── classifier_pipeline.joblib
│   ├── classifier_metrics.json
│   ├── anomaly_detector.joblib
│   ├── anomaly_metrics.json
│   ├── faiss_index.bin
│   └── rag_chunks.json
│
├── tests/                      # Automated test suite (27 unit & integration tests)
│   ├── test_data_loader.py
│   ├── test_classifier.py
│   ├── test_anomaly_detector.py
│   ├── test_rag.py
│   └── test_utils.py
│
└── docs/                       # Educational & Interview resources
    └── learning_guide.md       # Comprehensive study guide for technical interviews
```

---

## Installation & Setup

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/<your-username>/SentinelAI.git
cd SentinelAI

# Create and activate virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Linux/macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables (Optional)
```bash
cp .env.example .env
```
Open `.env` in any text editor. If you wish to use LLM generation, add your API key:
```ini
LLM_PROVIDER=gemini
LLM_API_KEY=your_actual_key_here
LLM_MODEL=gemini-1.5-flash
```
*Note: If left unset, the RAG assistant automatically runs in **Retrieval-Only Mode**.*

### 3. Initialize Dataset and Train Models
Run the setup pipeline:
```bash
# 1. Download official UCI AI4I 2020 dataset
python scripts/download_data.py

# 2. Train supervised classifier and anomaly detector
python scripts/train_models.py

# 3. Build FAISS vector index from maintenance SOPs
python scripts/build_rag_index.py
```

---

## Running the Application

Launch the Streamlit interactive dashboard:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Dashboard Navigation:
1. **Overview:** Architectural data flow and engineering distinctions.
2. **Failure Risk Classifier:** Input sensor sliders, calculate failure probability %, and view risk tier.
3. **Anomaly Detection:** Check Isolation Forest decision score and inspect per-sensor Z-score deviations.
4. **Maintenance Assistant:** Ask questions grounded in SOPs with verified citations.
5. **Model Evaluation:** Side-by-side benchmark comparison table and confusion matrices.
6. **About & Methodology:** Dataset provenance and target leakage details.

---

## Testing

Execute the complete automated test suite using `pytest`:
```bash
python -m pytest -v
```

Expected output:
```
tests/test_anomaly_detector.py::test_build_anomaly_pipeline PASSED
tests/test_anomaly_detector.py::test_load_anomaly_detector PASSED
tests/test_anomaly_detector.py::test_assess_nominal_reading PASSED
tests/test_anomaly_detector.py::test_assess_extreme_anomaly_reading PASSED
tests/test_classifier.py::test_build_pipelines PASSED
tests/test_classifier.py::test_classifier_evaluation_metrics PASSED
tests/test_classifier.py::test_load_trained_classifier PASSED
tests/test_classifier.py::test_estimate_failure_risk PASSED
tests/test_data_loader.py::test_load_raw_data_success PASSED
tests/test_data_loader.py::test_load_raw_data_missing_file PASSED
tests/test_data_loader.py::test_load_raw_data_invalid_schema PASSED
tests/test_data_loader.py::test_prepare_features_and_target_leakage_prevention PASSED
tests/test_data_loader.py::test_stratified_train_test_split PASSED
tests/test_data_loader.py::test_get_dataset_summary PASSED
tests/test_rag.py::test_chunk_document PASSED
tests/test_rag.py::test_build_faiss_index_missing_dir PASSED
tests/test_rag.py::test_rag_artifacts_load PASSED
tests/test_rag.py::test_retrieve_relevant_chunks PASSED
tests/test_rag.py::test_retrieve_empty_query PASSED
tests/test_rag.py::test_answer_maintenance_question_zero_api_key_fallback PASSED
tests/test_utils.py::test_validate_input_features_valid PASSED
tests/test_utils.py::test_validate_input_features_case_insensitivity PASSED
tests/test_utils.py::test_validate_input_features_invalid_type PASSED
tests/test_utils.py::test_validate_input_features_out_of_bounds PASSED
tests/test_utils.py::test_validate_input_features_missing_reading PASSED
tests/test_utils.py::test_validate_input_features_non_numeric PASSED
tests/test_utils.py::test_ensure_directories_exist PASSED
============================= 27 passed in 48.82s =============================
```

---

## Deployment Guide (Self-Hosted)

> [!NOTE]
> Per project requirements, live deployment is not executed automatically. The instructions below allow you to deploy the application on your own infrastructure when ready.

### Option A: Streamlit Community Cloud (Recommended for Demos)
1. Push your repository to GitHub (ensure `.env` is ignored).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and connect your GitHub account.
3. Select repository `SentinelAI`, branch `main`, and main file path `app.py`.
4. In Advanced Settings, add environment variables under **Secrets**:
   ```toml
   LLM_PROVIDER = "gemini"
   LLM_API_KEY = "your_key"
   LLM_MODEL = "gemini-1.5-flash"
   ```
5. Click **Deploy**. Streamlit Cloud will run `requirements.txt` and launch the app.

### Option B: Docker Container
Create a simple `Dockerfile` in the project root:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
Build and run:
```bash
docker build -t sentinel-ai .
docker run -p 8501:8501 --env-file .env sentinel-ai
```

---

## Troubleshooting & FAQ

- **Q: Why does the app say "Model Artifacts Not Found"?**  
  *A:* You need to run the three setup scripts first: `python scripts/download_data.py`, `python scripts/train_models.py`, and `python scripts/build_rag_index.py`.
- **Q: How does the app work without an internet connection?**  
  *A:* The machine learning models (Random Forest, Isolation Forest) and FAISS vector retrieval run 100% locally on your CPU. Once the Sentence Transformer weights are cached, no internet connection is required.
- **Q: Can I use a local LLM instead of Gemini/OpenAI?**  
  *A:* Yes. Set `LLM_PROVIDER=openai` and `LLM_BASE_URL=http://localhost:11434/v1` in `.env` to connect to Ollama or any OpenAI-compatible local server.

---

## Limitations & Future Roadmap

1. **Synthetic Telemetry:** AI4I 2020 is based on realistic milling simulations. Real-world machine telemetry includes high-frequency vibrational noise, sensor drift, and intermittent dropped packets.
2. **Probability Calibration:** The Random Forest output probabilities are monotonic risk scores rather than strictly calibrated frequentist probabilities. Future iterations could incorporate isotonic regression or Platt scaling.
3. **Time-Series Dynamics:** The current dataset provides cycle-aggregated snapshots. Future enhancements could incorporate temporal sequence models (e.g. streaming windowing) to predict Remaining Useful Life (RUL).

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
Attribution to UCI Machine Learning Repository for the AI4I 2020 Predictive Maintenance Dataset.
