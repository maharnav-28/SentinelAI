# SentinelAI: Technical Interview & Conceptual Learning Guide

*Tailored for Electrical Engineering and Applied Machine Learning technical interviews.*

---

## Table of Contents
1. [Bridging Electrical Engineering & Machine Learning](#1-bridging-electrical-engineering--machine-learning)
2. [End-to-End File Architecture & Code Purpose](#2-end-to-end-file-architecture--code-purpose)
3. [The Complete Data Flow](#3-the-complete-data-flow)
4. [Deep Dive: Core Machine Learning Concepts](#4-deep-dive-core-machine-learning-concepts)
   - [Target Leakage Prevention](#target-leakage-prevention)
   - [Severe Class Imbalance & The Metric Trap](#severe-class-imbalance--the-metric-trap)
   - [False Negatives vs. False Positives Cost Asymmetry](#false-negatives-vs-false-positives-cost-asymmetry)
   - [Why Random Forest Outperformed Logistic Regression](#why-random-forest-outperformed-logistic-regression)
   - [Isolation Forest: Anomaly Isolation in Tree Space](#isolation-forest-anomaly-isolation-in-tree-space)
   - [Vector Search & RAG Mechanics](#vector-search--rag-mechanics)
5. [Common Technical Interview Questions & High-Scoring Answers](#5-common-technical-interview-questions--high-scoring-answers)
6. [Interview Preparation Study Sequence](#6-interview-preparation-study-sequence)

---

## 1. Bridging Electrical Engineering & Machine Learning

In electrical and mechanical systems, physical laws dictate machine behavior:
- **Ohm's & Joule's Laws:** $I^2R$ power loss creates heat in motor stator windings. When cooling airflow drops at low speeds, temperature escalates rapidly.
- **Mechanical Power:** Mechanical spindle power is $P = \frac{2 \pi N \tau}{60}$ Watts, linking rotational speed $N$ (rpm) and shaft torque $\tau$ (Nm).
- **Thermodynamics:** Heat transfer from cutting contact zone to ambient air requires a positive thermal gradient ($\Delta T = T_{\text{process}} - T_{\text{air}}$).
- **Tool Mechanics:** Tool wear ($t_{\text{wear}}$) degrades edge sharpness, increasing friction and requiring higher torque to shear the same chip volume.

Traditional rule-based monitoring relies on static upper/lower threshold alarms (e.g., "sound alarm if torque $> 60\text{ Nm}$"). However, industrial failures arise from **multivariate non-linear interactions**:
- A torque of 55 Nm may be perfectly safe at 1,500 rpm with a sharp tool, but causes catastrophic motor breakdown when combined with 230 minutes of tool wear and poor cooling.

**Machine learning replaces rigid single-variable thresholds with multivariate decision boundaries learned directly from empirical operational telemetry.**

---

## 2. End-to-End File Architecture & Code Purpose

Every file in SentinelAI has a single, well-defined responsibility:

### `src/utils.py`
- **Purpose:** Centralized project path resolution, schema constants, and physical boundary validation.
- **Key Functions:**
  - `validate_input_features(input_data)`: Sanitizes user inputs, checks numerical boundaries (e.g. non-negative torque, plausible Kelvin temperatures), and raises clear `ValueError`s.
  - `ensure_directories_exist()`: Creates directories dynamically to prevent runtime I/O exceptions.
- **Why it matters:** Guarantees that neither the app nor scripts rely on brittle relative paths or unverified sensor ranges.

### `src/data_loader.py`
- **Purpose:** Dataset acquisition, schema verification, target leakage elimination, and stratified splitting.
- **Key Functions:**
  - `download_uci_dataset()`: Downloads official UCI AI4I 2020 zip and extracts `ai4i2020.csv` with automated retry and manual fallback guidance.
  - `load_raw_data()`: Validates that all 14 expected columns exist.
  - `prepare_features_and_target()`: Drops target `Machine failure` and all leakage columns (`UDI`, `Product ID`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`).
  - `get_train_test_split()`: Implements stratified 80/20 train/test splitting to preserve the 3.39% failure class ratio.

### `src/train_classifier.py`
- **Purpose:** Supervised Scikit-learn pipelines, model training, evaluation, artifact persistence, and real-time failure risk scoring.
- **Key Functions:**
  - `create_preprocessor()`: Assembles `ColumnTransformer` with `StandardScaler` for continuous sensors and `OneHotEncoder` for machine `Type`.
  - `build_pipeline(model_type)`: Constructs an end-to-end `Pipeline` encapsulating preprocessing and classifier with `class_weight='balanced'`.
  - `evaluate_model()`: Computes Precision, Recall, F1, Balanced Accuracy, PR-AUC, ROC-AUC, and full confusion matrix.
  - `estimate_failure_risk()`: Computes risk probability and assigns risk tiers (Low, Moderate, High, Critical) with calibration notices.

### `src/anomaly_detector.py`
- **Purpose:** Unsupervised outlier detection using `IsolationForest` on continuous sensor telemetry.
- **Key Functions:**
  - `build_anomaly_pipeline(contamination)`: Preprocessing pipeline with `StandardScaler` and `IsolationForest`.
  - `train_anomaly_detector()`: Fits detector on training data, extracts baseline means/stds for feature attribution, and saves model.
  - `assess_anomaly()`: Evaluates single reading, outputs decision score ($<0$ indicates atypical), and calculates per-sensor $z$-score deviations.

### `src/rag.py`
- **Purpose:** Grounded RAG maintenance assistant using Sentence Transformers, FAISS, and secure LLM calling with zero-crash fallback.
- **Key Functions:**
  - `chunk_document()`: Splits markdown technical manuals by section heading (`##`), retaining title and section metadata.
  - `build_faiss_index()`: Encodes text chunks with `all-MiniLM-L6-v2` into 384-dimensional dense vectors, normalizes them, and builds a `faiss.IndexFlatIP` index.
  - `retrieve_relevant_chunks()`: Computes cosine similarity via FAISS inner product search.
  - `answer_maintenance_question()`: Dispatches prompt to LLM if configured; activates `retrieval_only_fallback` if no API key is set or if offline.

### `app.py`
- **Purpose:** Interactive Streamlit dashboard with 6 specialized views, telemetry preset injection, real-time derived physical parameter calculation ($\Delta T$, Power, Strain product), and responsive UI.

### `scripts/`
- `download_data.py`: CLI runner to download and verify the raw CSV.
- `train_models.py`: CLI runner to train both classifiers and the anomaly detector, printing evaluation comparison tables.
- `build_rag_index.py`: CLI runner to parse SOPs and build the FAISS vector index.

---

## 3. The Complete Data Flow

```
1. RAW DATA INGESTION
   UCI AI4I 2020 ZIP -> Extracted to data/raw/ai4i2020.csv (10,000 rows, 14 cols)
                                |
2. DATA CLEANING & LEAKAGE STRIPPING
   Drop Target: 'Machine failure'
   Drop Leakage: 'UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'
   Remaining Predictors: 'Type' (cat) + 5 numerical sensor readings
                                |
3. STRATIFIED TRAIN/TEST SPLIT
   8,000 Train Samples (271 failures) | 2,000 Test Samples (68 failures)
                                |
4. PIPELINE FITTING (TRAIN SET ONLY)
   [Numerical -> StandardScaler] + [Categorical -> OneHotEncoder]
                                |
        +-----------------------+-----------------------+
        |                                               |
        v                                               v
5A. SUPERVISED CLASSIFIER                       5B. UNSUPERVISED DETECTOR
    Random Forest (balanced)                        Isolation Forest (contam=3.5%)
    Saved to artifacts/                             Saved to artifacts/
                                |
6. RAG INDEXING PIPELINE
   Markdown SOPs -> Section Chunking -> SentenceTransformers Embeddings (384-dim)
   -> L2 Normalized -> FAISS IndexFlatIP -> Saved to artifacts/
                                |
7. STREAMLIT RUNTIME INFERENCE
   User Sliders -> Input Validation ->
   - Classifier predicts risk probability (%) & tier
   - Anomaly detector calculates decision score & sensor z-scores
   - RAG retrieves top-k SOP passages & generates grounded answer
```

---

## 4. Deep Dive: Core Machine Learning Concepts

### Target Leakage Prevention
**Definition:** Target leakage occurs when information from outside the training dataset (or features that would not be available at prediction time in production) is used to create the model.

In the UCI AI4I 2020 dataset:
- `TWF`, `HDF`, `PWF`, `OSF`, `RNF` are five binary indicator columns representing specific failure modes. For instance, if `HDF == 1`, a Heat Dissipation Failure has occurred. If these columns are included in $X$, any model will simply learn: `Machine failure = TWF or HDF or PWF or OSF or RNF`. The model would achieve 100% test accuracy in the lab, but would be **completely useless in production** because failure mode flags are only known *after* a machine has broken down!
- `Product ID` (e.g. `L47181`): The first letter is the product quality variant (Low, Medium, High). The digits are sequential production counters. The variant is already captured cleanly in `Type`. Including the sequence digits creates spurious correlation where a decision tree splits on arbitrary serial numbers.
- `UDI`: Row number (1 to 10000). Must be dropped.

### Severe Class Imbalance & The Metric Trap
In the dataset, only **339 out of 10,000 cycles (3.39%)** result in machine failure.
- **The Metric Trap:** If an engineer optimizes solely for **Accuracy**, a trivial model that always predicts `0` (Normal) achieves:
  $$\text{Accuracy} = \frac{9,661}{10,000} = 96.61\%$$
  Yet this model has **0.0% Recall**—it misses every single machine failure!
- **How SentinelAI Solves This:**
  1. We evaluate using **Recall**, **Precision**, **F1-Score**, **Balanced Accuracy**, and **PR-AUC**.
  2. We configure `class_weight='balanced'`. In Scikit-learn, this scales the loss function inversely proportional to class frequencies:
     $$w_j = \frac{n_{\text{samples}}}{n_{\text{classes}} \cdot n_j}$$
     For failure class 1, the weight is $\approx 14.75$, while for normal class 0, it is $\approx 0.52$. The model is penalized ~28 times more for missing a failure than for a false alarm.

### False Negatives vs. False Positives Cost Asymmetry
In predictive maintenance:
- **False Negative (FN):** Machine fails, but model predicted Normal.
  - *Consequence:* Spindle motor burnout, destroyed carbide tool holders, workpiece scrapped, unpredicted line downtime. Extremely high financial cost.
- **False Positive (FP):** Machine is normal, but model predicted Failure.
  - *Consequence:* Maintenance technician performs a 10-minute visual inspection or tool offset check. Moderate labor cost.
- **Engineering Conclusion:** High Recall is non-negotiable. However, if Precision drops too low (as with Logistic Regression at 14.2%), operators experience **alarm fatigue** and start ignoring alerts.

### Why Random Forest Outperformed Logistic Regression
Measured Results on 2,000 Test Samples:
- Logistic Regression: Precision = 14.2%, Recall = 82.4%, PR-AUC = 0.382
- Random Forest: Precision = **48.6%**, Recall = **75.0%**, PR-AUC = **0.647**

**Why?**
1. **Non-Linear Interactions:** Mechanical failure depends on compound conditions. For example, Heat Dissipation Failure requires *both* $\Delta T < 8.6\text{ K}$ *and* $\text{Speed} < 1,380\text{ rpm}$. Logistic Regression fits a single linear hyperplane ($\sum w_i x_i + b = 0$) and cannot capture this logical conjunction without manual interaction terms.
2. **Decision Trees:** Random Forest partitions feature space into orthogonal bounding boxes, naturally isolating localized hazard zones in multivariate parameter space.
3. **Ensemble Averaging:** By averaging 100 decorrelated trees trained on bootstrap samples (bagging), variance is drastically reduced, cutting false alarms from 339 down to 54.

### Isolation Forest: Anomaly Isolation in Tree Space
Unlike density-based methods (e.g. DBSCAN) or distance-based methods that suffer from the curse of dimensionality, **Isolation Forest** exploits two quantitative properties of anomalies:
1. They are few in number (minority).
2. They possess attribute values markedly different from nominal points.

**Algorithm:**
- An ensemble of isolation trees (iTrees) is constructed.
- At each node, a feature is selected at random, and a split point is chosen uniformly at random between the feature's minimum and maximum values.
- Nominal points in dense clusters require many recursive splits to isolate (deep in the tree).
- Anomalous points located in sparse regions require very few splits to isolate (shallow path length $h(x)$).

**Decision Function:**
- Decision score $s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$, where $c(n)$ is the average path length of unsuccessful searches in a Binary Search Tree.
- In Scikit-learn, `decision_function(X)` is shifted so that **negative values indicate outliers/anomalies** and positive values indicate inliers.

### Vector Search & RAG Mechanics
- **Why RAG?** Large Language Models (LLMs) hallucinate when asked specific technical maintenance questions. They do not know your plant's exact torque limits or spindle lubrication intervals.
- **Embeddings:** `all-MiniLM-L6-v2` maps arbitrary-length sentences into a continuous 384-dimensional vector space where semantically similar engineering phrases are located close together.
- **Cosine Similarity via FAISS Inner Product:**
  Cosine similarity between query vector $\mathbf{q}$ and document vector $\mathbf{d}$ is:
  $$\text{sim}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|}$$
  By L2-normalizing all vectors ($\|\mathbf{q}\| = 1, \|\mathbf{d}\| = 1$) upon indexing, the inner product equals cosine similarity:
  $$\mathbf{q} \cdot \mathbf{d} = \cos(\theta)$$
  `faiss.IndexFlatIP` computes this dot product in parallel on CPU across millions of vectors in microseconds.

---

## 5. Common Technical Interview Questions & High-Scoring Answers

### Q1: "Why did you exclude failure modes like TWF, HDF, PWF from your feature set?"
> **Candidate Answer:**  
> "Including TWF, HDF, PWF, OSF, and RNF would cause catastrophic target leakage. In the AI4I dataset, those five columns represent the physical failure modes that compose the target `Machine failure`. If left in the training data, the model trivially learns the mathematical identity `Machine failure = sum(failure_modes) > 0`, achieving deceptive 100% test accuracy. However, in a real factory, failure mode classifications are only diagnosed post-mortem by maintenance technicians after equipment breakdown. To build a genuinely predictive system, we must use strictly pre-failure telemetry available at inference time: temperatures, speed, torque, tool wear, and product type."

### Q2: "Your dataset has a 96.6% to 3.4% class imbalance. Why not rely on standard classification accuracy?"
> **Candidate Answer:**  
> "In an imbalanced setting, accuracy is misleading because a dummy classifier that predicts the majority class (Normal) 100% of the time achieves 96.6% accuracy while having a Recall of 0%—meaning it misses every catastrophic machine failure. In predictive maintenance, the business cost of a missed breakdown (False Negative) is orders of magnitude higher than a preventive inspection (False Positive). Therefore, we evaluated our models using Precision, Recall, F1-Score, Balanced Accuracy, and PR-AUC (Precision-Recall Area Under Curve), which directly reflects minority class discrimination regardless of majority class dominance."

### Q3: "How does your system distinguish between an anomaly and a failure prediction?"
> **Candidate Answer:**  
> "This is a fundamental architectural boundary in SentinelAI.  
> 1. A **failure risk score** is supervised: it reflects how strongly the current telemetry matches historical patterns of documented machine breakdowns.  
> 2. An **anomaly** is unsupervised: using Isolation Forest, it measures the statistical distance of a reading from the multivariate cluster of nominal training cycles.  
> An unusual reading does not mean a machine has failed—for example, a machine may run an unusual high-speed test cycle without failing. Conversely, a machine might experience a gradual tool wear failure under normal speeds. By reporting both signals independently, SentinelAI avoids false root-cause claims."

### Q4: "Explain the working principle of Isolation Forest. Why is it preferable over distance-based methods?"
> **Candidate Answer:**  
> "Traditional anomaly detectors like k-NN or Local Outlier Factor compute Euclidean distances or local densities between points, which scale poorly with sample size ($O(n^2)$) and suffer from distance concentration in higher dimensions.  
> Isolation Forest isolates anomalies instead of profiling normal points. It recursively partitions feature space using random orthogonal splits. Because anomalies are rare and situated in sparse feature space, they are isolated in significantly fewer splits, resulting in short tree path lengths. Isolation Forest runs in linear time $O(t \cdot \psi \log \psi)$ (where $t$ is tree count and $\psi$ is sub-sample size), making it memory-efficient and ideal for real-time sensor streaming."

### Q5: "How does your RAG assistant ensure reliable operation without external API keys or during network outages?"
> **Candidate Answer:**  
> "SentinelAI was engineered for air-gapped industrial reliability. The Sentence Transformers embedding model (`all-MiniLM-L6-v2`) and the FAISS vector index run entirely locally on CPU. If an LLM API key is absent or network connectivity drops, the system gracefully falls back to **Retrieval-Only Mode**. It presents the top retrieved maintenance SOP passages, similarity scores, and verified section titles directly to the user. This ensures the technician always gets verified operating limits without the application crashing."

### Q6: "Why did you use Scikit-learn Pipelines instead of scaling the dataset before splitting?"
> **Candidate Answer:**  
> "Preprocessing before splitting introduces data snooping (leakage). If we compute the mean and standard deviation over the full dataset with `StandardScaler`, the training data inherits statistical properties of the test set. By encapsulating `StandardScaler` and `OneHotEncoder` inside an `sklearn.pipeline.Pipeline`, all transformation parameters ($\mu, \sigma$) are fit strictly on the training fold during cross-validation or train/test splitting, guaranteeing untouched evaluation on the held-out test data."

---

## 6. Interview Preparation Study Sequence

To master this repository for an upcoming technical interview, review files in this exact order:

1. **`src/utils.py`:** Understand sensor boundaries, engineering units (Kelvin, rpm, Nm, min), and input sanitation.
2. **`src/data_loader.py`:** Understand the leakage prevention rationale and stratified splitting logic.
3. **`src/train_classifier.py`:** Study `ColumnTransformer`, `class_weight='balanced'`, metric computation, and the false negative/positive trade-off.
4. **`src/anomaly_detector.py`:** Study `IsolationForest` decision thresholding ($<0$ vs $\ge 0$) and $z$-score deviation attribution.
5. **`src/rag.py`:** Study section-based document chunking, FAISS vector normalization, cosine similarity, and zero-key fallback design.
6. **`tests/`:** Review the 27 unit tests to understand edge case handling and software reliability practices.
7. **`app.py`:** Run the dashboard, experiment with the Telemetry Presets, and observe how derived parameters ($\Delta T$, Power, Strain) connect to the ML outputs.
