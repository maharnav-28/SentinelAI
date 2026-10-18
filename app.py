"""SentinelAI: Smart Machine Health Assistant - Interactive Streamlit Dashboard."""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Configure project paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    FEATURE_BOUNDS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    CLASSIFIER_PIPELINE_PATH,
    CLASSIFIER_METRICS_PATH,
    ANOMALY_DETECTOR_PATH,
    ANOMALY_METRICS_PATH,
    FAISS_INDEX_PATH,
    RAG_CHUNKS_PATH,
    DATA_RAW_FILE,
    validate_input_features,
)
from src.train_classifier import load_classifier, estimate_failure_risk
from src.anomaly_detector import load_anomaly_detector, assess_anomaly
from src.rag import load_rag_artifacts, answer_maintenance_question

# Page Configuration
st.set_page_config(
    page_title="SentinelAI — Machine Health Assistant",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for clean, restrained engineering UI with accessible contrast
st.markdown(
    """
    <style>
    /* Metric Cards on Overview Page - dark elevated background with red accent */
    .metric-card {
        background-color: #1a1f2c;
        border: 1px solid #2e3648;
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }
    .metric-card h4 {
        color: #ffffff !important;
        margin-top: 0;
        margin-bottom: 8px;
        font-size: 1.05rem;
        font-weight: 600;
    }
    .metric-card p {
        color: #cbd5e1 !important;
        margin-bottom: 0;
        font-size: 0.92rem;
        line-height: 1.55;
    }
    .metric-card b {
        color: #ffffff !important;
    }

    /* Status indicators with high contrast in dark theme */
    .status-normal {
        color: #22c55e !important;
        font-weight: bold;
    }
    .status-warning {
        color: #f59e0b !important;
        font-weight: bold;
    }
    .status-danger {
        color: #ef4444 !important;
        font-weight: bold;
    }

    /* Disclaimer / Warning Box - preserves warning styling with readable dark text */
    .disclaimer-box {
        background-color: #fef3c7;
        border: 1px solid #fde68a;
        border-left: 4px solid #f59e0b;
        color: #1f2937 !important;
        padding: 12px 14px;
        border-radius: 6px;
        font-size: 0.9rem;
        line-height: 1.5;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .disclaimer-box b {
        color: #111827 !important;
        font-weight: 700;
    }
    .disclaimer-box i {
        color: #1f2937 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading machine learning models...")
def get_classifier():
    return load_classifier()


@st.cache_resource(show_spinner="Loading anomaly detector...")
def get_anomaly_detector():
    return load_anomaly_detector()


@st.cache_resource(show_spinner="Loading RAG vector index & embeddings...")
def get_rag_system():
    return load_rag_artifacts()


def check_artifacts_ready() -> bool:
    """Verify all trained artifacts exist before running dashboard."""
    required = [
        CLASSIFIER_PIPELINE_PATH,
        CLASSIFIER_METRICS_PATH,
        ANOMALY_DETECTOR_PATH,
        ANOMALY_METRICS_PATH,
        FAISS_INDEX_PATH,
        RAG_CHUNKS_PATH,
    ]
    return all(p.exists() for p in required)


def rebuild_missing_artifacts() -> tuple[bool, str]:
    """Rebuild missing models and vector store using the project's data and training pipelines."""
    try:
        from src.data_loader import (
            download_uci_dataset,
            load_raw_data,
            prepare_features_and_target,
            get_train_test_split,
        )
        from src.train_classifier import train_and_evaluate_all
        from src.anomaly_detector import train_anomaly_detector
        from src.rag import build_faiss_index
        from src.utils import MAINTENANCE_DOCS_DIR

        # 1. Dataset check / download
        if not DATA_RAW_FILE.exists():
            download_uci_dataset()

        # 2. Classifier & Anomaly models
        if (
            not CLASSIFIER_PIPELINE_PATH.exists()
            or not ANOMALY_DETECTOR_PATH.exists()
            or not CLASSIFIER_METRICS_PATH.exists()
            or not ANOMALY_METRICS_PATH.exists()
        ):
            df = load_raw_data(DATA_RAW_FILE)
            X, y = prepare_features_and_target(df)
            X_train, X_test, y_train, y_test = get_train_test_split(X, y, test_size=0.2, random_state=42)
            train_and_evaluate_all(X_train, y_train, X_test, y_test)
            train_anomaly_detector(X_train, X_test, y_test, contamination=0.035)

        # 3. FAISS RAG index
        if not FAISS_INDEX_PATH.exists() or not RAG_CHUNKS_PATH.exists():
            build_faiss_index(MAINTENANCE_DOCS_DIR)

        st.cache_resource.clear()
        return True, "Artifacts successfully initialized."
    except Exception as e:
        return False, str(e)


def render_sidebar():
    """Render navigation sidebar and quick telemetry presets."""
    st.sidebar.title("⚙️ SentinelAI")
    st.sidebar.caption("Smart Machine Health Assistant")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Failure Risk Classifier",
            "Anomaly Detection",
            "Maintenance Assistant",
            "Model Evaluation",
            "About & Methodology",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Telemetry Presets")
    st.sidebar.caption("Pre-configured operating points for demonstration:")

    preset_choice = st.sidebar.selectbox(
        "Load Preset Example",
        [
            "Manual Controls",
            "Nominal Steady Cutting (Healthy)",
            "Thermal Heat Dissipation Warning",
            "High Tool Wear & Overload",
            "Extreme Speed / Torque Spike",
        ],
    )

    # Preset configurations
    presets = {
        "Manual Controls": {
            "Type": "L",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500.0,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 60.0,
        },
        "Nominal Steady Cutting (Healthy)": {
            "Type": "L",
            "Air temperature [K]": 298.5,
            "Process temperature [K]": 308.8,
            "Rotational speed [rpm]": 1520.0,
            "Torque [Nm]": 38.0,
            "Tool wear [min]": 40.0,
        },
        "Thermal Heat Dissipation Warning": {
            "Type": "L",
            "Air temperature [K]": 303.2,
            "Process temperature [K]": 310.5,  # Delta T = 7.3 K (< 8.6 K)
            "Rotational speed [rpm]": 1320.0,  # Below 1380 rpm
            "Torque [Nm]": 44.0,
            "Tool wear [min]": 95.0,
        },
        "High Tool Wear & Overload": {
            "Type": "M",
            "Air temperature [K]": 299.0,
            "Process temperature [K]": 309.2,
            "Rotational speed [rpm]": 1390.0,
            "Torque [Nm]": 62.0,
            "Tool wear [min]": 215.0,  # > 200 min wear limit
        },
        "Extreme Speed / Torque Spike": {
            "Type": "H",
            "Air temperature [K]": 301.0,
            "Process temperature [K]": 310.5,
            "Rotational speed [rpm]": 2750.0,  # Near upper speed boundary
            "Torque [Nm]": 75.0,  # High torque
            "Tool wear [min]": 120.0,
        },
    }

    selected_readings = presets[preset_choice]
    return page, selected_readings, preset_choice


def render_sensor_inputs(current_values: dict, key_prefix: str = "input") -> dict:
    """Render interactive input sliders and selectors for sensor readings."""
    st.subheader("Machine Operating Telemetry")
    col1, col2, col3 = st.columns(3)

    with col1:
        product_type = st.selectbox(
            "Product Variant (Type)",
            options=FEATURE_BOUNDS["Type"]["options"],
            index=FEATURE_BOUNDS["Type"]["options"].index(current_values.get("Type", "L")),
            format_func=lambda x: f"{x} - {FEATURE_BOUNDS['Type']['descriptions'][x]}",
            key=f"{key_prefix}_type",
        )
        air_temp = st.slider(
            "Air Temperature [K]",
            min_value=FEATURE_BOUNDS["Air temperature [K]"]["min"],
            max_value=FEATURE_BOUNDS["Air temperature [K]"]["max"],
            value=float(current_values.get("Air temperature [K]", 300.0)),
            step=FEATURE_BOUNDS["Air temperature [K]"]["step"],
            key=f"{key_prefix}_air_temp",
            help="Ambient shop-floor temperature in Kelvin (~298 K = 25°C).",
        )

    with col2:
        proc_temp = st.slider(
            "Process Temperature [K]",
            min_value=FEATURE_BOUNDS["Process temperature [K]"]["min"],
            max_value=FEATURE_BOUNDS["Process temperature [K]"]["max"],
            value=float(current_values.get("Process temperature [K]", 310.0)),
            step=FEATURE_BOUNDS["Process temperature [K]"]["step"],
            key=f"{key_prefix}_proc_temp",
            help="Direct cutting contact zone temperature in Kelvin.",
        )
        rot_speed = st.slider(
            "Rotational Speed [rpm]",
            min_value=FEATURE_BOUNDS["Rotational speed [rpm]"]["min"],
            max_value=FEATURE_BOUNDS["Rotational speed [rpm]"]["max"],
            value=float(current_values.get("Rotational speed [rpm]", 1500.0)),
            step=FEATURE_BOUNDS["Rotational speed [rpm]"]["step"],
            key=f"{key_prefix}_speed",
            help="Spindle rotational speed in revolutions per minute.",
        )

    with col3:
        torque = st.slider(
            "Torque [Nm]",
            min_value=FEATURE_BOUNDS["Torque [Nm]"]["min"],
            max_value=FEATURE_BOUNDS["Torque [Nm]"]["max"],
            value=float(current_values.get("Torque [Nm]", 40.0)),
            step=FEATURE_BOUNDS["Torque [Nm]"]["step"],
            key=f"{key_prefix}_torque",
            help="Spindle cutting resistance torque in Newton-meters.",
        )
        tool_wear = st.slider(
            "Tool Wear [min]",
            min_value=FEATURE_BOUNDS["Tool wear [min]"]["min"],
            max_value=FEATURE_BOUNDS["Tool wear [min]"]["max"],
            value=float(current_values.get("Tool wear [min]", 60.0)),
            step=FEATURE_BOUNDS["Tool wear [min]"]["step"],
            key=f"{key_prefix}_wear",
            help="Accumulated cut duration with current tool insert in minutes.",
        )

    # Derived physical calculations for EE / mechanical engineering context
    delta_t = round(proc_temp - air_temp, 2)
    power_watts = round((2.0 * np.pi * rot_speed * torque) / 60.0, 1)
    overstrain_product = round(tool_wear * torque, 1)

    st.markdown(
        f"**Derived Physical Parameters:** &nbsp;&nbsp; "
        f"Temperature Differential ($\\Delta T$): **{delta_t} K** &nbsp;|&nbsp; "
        f"Mechanical Power ($P$): **{power_watts} W** &nbsp;|&nbsp; "
        f"Strain Product ($\\tau \\cdot t$): **{overstrain_product} min·Nm**"
    )

    return {
        "Type": product_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": proc_temp,
        "Rotational speed [rpm]": rot_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    }


def render_overview():
    """Render system overview, architecture, and engineering distinction."""
    st.title("SentinelAI: Smart Machine Health Assistant")
    st.markdown(
        "A practical, interpretable predictive maintenance platform designed for milling and turning machinery. "
        "Built on the **UCI AI4I 2020 Predictive Maintenance Dataset**."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <h4>1. Supervised Risk Classification</h4>
                <p>Estimates probability of machine failure using a <b>Random Forest</b> model evaluated against a Logistic Regression baseline. Trained with strict target-leakage prevention on 6 operational features.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <h4>2. Unsupervised Anomaly Detection</h4>
                <p>Flags statistically unusual sensor patterns using <b>Isolation Forest</b> across continuous telemetry. Diagnoses individual sensor z-score deviations without relying on failure labels.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <h4>3. RAG Maintenance Assistant</h4>
                <p>Retrieves verified Standard Operating Procedures (SOPs) using <b>Sentence Transformers</b> & <b>FAISS</b> vector search, producing grounded repair guidance with a zero-API-key fallback.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Crucial Engineering Distinction")
    st.info(
        "**Core Engineering Principle:** SentinelAI rigorously distinguishes between three independent operational signals:\n"
        "1. **Predicted Failure Risk:** A supervised statistical model estimate reflecting historical failure correlations. *It is not a certainty.* \n"
        "2. **Unusual Sensor Reading (Anomaly):** An unsupervised multivariate outlier flagging an atypical operating point. *It is not a confirmed breakdown.* \n"
        "3. **Documented Maintenance Guidance:** A verified engineering procedure retrieved from traceable SOPs. *It is not an automated guess.*"
    )

    st.subheader("System Architecture")
    st.markdown(
        r"""
        ```
        [ Sensor Telemetry (T_air, T_proc, Speed, Torque, Wear, Type) ]
                                    |
            +-----------------------+-----------------------+
            |                                               |
            v                                               v
    [ Supervised Classifier ]                     [ Isolation Forest ]
     StandardScaler + OneHot                       StandardScaler
     Random Forest (class_weight=balanced)         Contamination = 3.5%
            |                                               |
            v                                               v
    [ Failure Risk Score % ]                      [ Atypical Flag & Z-Scores ]
            \                                               /
             +----------------------+----------------------+
                                    |
                                    v
            [ RAG Assistant: FAISS + Sentence-Transformers ]
            [ Technical SOPs: Thermal, Wear, Power, Bearings ]
                                    |
                                    v
            [ Grounded Maintenance Guidance & Verified Citations ]
        ```
        """
    )


def render_failure_risk(readings: dict):
    """Render supervised failure risk classification page."""
    st.title("Supervised Machine Failure Risk")
    st.markdown(
        "Evaluate the likelihood of machine breakdown from current operating telemetry using "
        "the fitted **Random Forest Pipeline**."
    )

    input_data = render_sensor_inputs(readings, key_prefix="risk")

    if st.button("Evaluate Failure Risk", type="primary"):
        try:
            validated = validate_input_features(input_data)
            pipeline, metrics = get_classifier()
            result = estimate_failure_risk(pipeline, validated)

            prob = result["risk_percentage"]
            tier = result["risk_tier"]

            st.markdown("---")
            st.subheader("Model Assessment")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Estimated Failure Risk", f"{prob:.1f}%")
            with col2:
                tier_colors = {
                    "Low": "status-normal",
                    "Moderate": "status-warning",
                    "High": "status-danger",
                    "Critical": "status-danger",
                }
                st.markdown(f"**Risk Tier:** <span class='{tier_colors[tier]}'>{tier}</span>", unsafe_allow_html=True)
            with col3:
                pred_label = "FAILURE RISK FLAGGED (Class 1)" if result["predicted_class"] == 1 else "NOMINAL OPERATION (Class 0)"
                st.markdown(f"**Classifier Decision:** `{pred_label}`")

            # Progress bar visualization
            st.progress(min(max(float(prob / 100.0), 0.0), 1.0))

            # Operational guidance based on risk tier
            if tier in ["High", "Critical"]:
                st.error(
                    f"**Attention Required:** Model estimated high risk ({prob:.1f}%). "
                    "Recommend immediate inspection of cutting inserts, tool wear register, and thermal clearance. "
                    "Consult the **Maintenance Assistant** tab for relevant SOP guidance."
                )
            elif tier == "Moderate":
                st.warning(
                    f"**Elevated Risk ({prob:.1f}%):** Machine is operating near boundary limits. "
                    "Schedule inspection at the next planned maintenance interval."
                )
            else:
                st.success(
                    f"**Nominal Status ({prob:.1f}%):** Operating telemetry aligns with standard failure-free machining cycles."
                )

            st.markdown(
                f"<div class='disclaimer-box'><b>Disclaimer & Calibration Notice:</b> {result['calibration_notice']} "
                "Predictive maintenance models prioritize <b>Recall</b> to minimize unpredicted downtime, "
                "which may lead to cautious, conservative risk estimates.</div>",
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(f"Error during risk estimation: {e}")


def render_anomaly_detection(readings: dict):
    """Render unsupervised anomaly detection page with sensor deviation diagnostics."""
    st.title("Unsupervised Anomaly Detection")
    st.markdown(
        "Detect atypical operating conditions using **Isolation Forest**. "
        "Unlike supervised classification, anomaly detection operates without failure labels to identify unusual sensor combinations."
    )

    input_data = render_sensor_inputs(readings, key_prefix="anomaly")

    if st.button("Analyze Operating Normality", type="primary"):
        try:
            validated = validate_input_features(input_data)
            pipeline, metrics = get_anomaly_detector()
            assessment = assess_anomaly(pipeline, metrics, validated)

            st.markdown("---")
            st.subheader("Isolation Forest Assessment")

            col1, col2 = st.columns(2)
            with col1:
                status_class = "status-danger" if assessment["is_anomaly"] else "status-normal"
                st.markdown(
                    f"**Operating State:** <span class='{status_class}'>{assessment['status_label']}</span>",
                    unsafe_allow_html=True,
                )
                st.metric("Decision Function Score", f"{assessment['decision_score']:.3f}")
                st.caption("Threshold: Scores < 0.0 indicate isolated / atypical observations.")

            with col2:
                st.markdown(f"**Diagnostic Summary:**\n{assessment['explanation']}")

            # Feature deviation breakdown (Z-Scores)
            st.subheader("Sensor Deviation Analysis (vs. Training Baseline)")
            st.caption(
                "Standard score (Z-Score) indicates how many standard deviations the reading lies from nominal training mean. "
                "|Z| >= 2.0 indicates significant deviation."
            )

            dev_data = []
            for feat, dev in assessment["feature_deviations"].items():
                dev_data.append(
                    {
                        "Sensor Parameter": feat,
                        "Current Reading": f"{dev['reading']} {FEATURE_BOUNDS[feat]['unit']}",
                        "Baseline Mean": f"{dev.get('baseline_mean', 'N/A')}",
                        "Z-Score": dev["z_score"],
                        "Significant Deviation": "⚠️ YES (|z| >= 2)" if dev["is_extreme"] else "Normal",
                    }
                )

            st.table(pd.DataFrame(dev_data))

            st.markdown(
                "<div class='disclaimer-box'><b>Engineering Boundary:</b> "
                "An atypical sensor reading flags statistical distinctiveness from normal baseline cycles. "
                "It does <i>not</i> automatically confirm a hardware defect, nor does it identify root cause without mechanical inspection.</div>",
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(f"Error analyzing sensor readings: {e}")


def render_maintenance_assistant():
    """Render RAG maintenance assistant with search, grounded answers, and source citations."""
    st.title("RAG Maintenance Assistant")
    st.markdown(
        "Ask technical questions regarding CNC maintenance, failure criteria, and operating SOPs. "
        "Answers are grounded in verified documentation using **Sentence Transformers** and **FAISS** vector search."
    )

    # Sample query selector for user convenience
    preset_questions = [
        "Select an example question or enter custom inquiry below...",
        "What is the tool wear limit and when must inserts be replaced?",
        "What causes Heat Dissipation Failure (HDF) and what is the temperature threshold?",
        "How is spindle power calculated and what are the power failure limits?",
        "What are the spindle bearing lubrication intervals and vibration limits?",
        "What is the maximum torque and speed recommended during roughing?",
    ]
    selected_preset = st.selectbox("Quick Inquiries", preset_questions)

    default_text = "" if selected_preset.startswith("Select an example") else selected_preset
    user_query = st.text_input("Maintenance Inquiry:", value=default_text, placeholder="e.g. How do I inspect tool wear?")

    col_btn, col_k = st.columns([4, 1])
    with col_k:
        top_k = st.number_input("Passages to retrieve", min_value=1, max_value=5, value=3)

    if st.button("Search & Consult Assistant", type="primary"):
        if not user_query.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Searching technical documentation & synthesizing guidance..."):
            try:
                index, chunks, embedder = get_rag_system()
                response = answer_maintenance_question(
                    query=user_query,
                    index=index,
                    chunks=chunks,
                    embedder=embedder,
                    top_k=int(top_k),
                )

                st.markdown("---")
                st.subheader("Technical Maintenance Guidance")

                # Show answer
                st.markdown(response["answer"])

                # Source Citations
                st.markdown("---")
                st.subheader(f"Retrieved Source Evidence ({len(response['sources'])} passages)")
                for src in response["sources"]:
                    with st.expander(
                        f"📄 [{src['doc_name']}] {src['section']} (Similarity: {src['similarity_score']:.3f})"
                    ):
                        st.markdown(f"**Document Title:** {src['doc_title']}")
                        st.markdown(f"**Excerpt:**\n```\n{src['snippet']}\n```")

            except Exception as e:
                st.error(f"RAG Error: {e}")


def render_model_evaluation():
    """Render test set model comparison table, confusion matrices, and metric trade-offs."""
    st.title("Model Evaluation & Benchmarking")
    st.markdown(
        "Measured performance on the **unseen held-out test set** (2,000 samples, 20% stratified split). "
        "Strictly evaluated without data snooping or target leakage."
    )

    try:
        _, clf_metrics = get_classifier()
        _, anom_metrics = get_anomaly_detector()

        st.subheader("Supervised Classification: Baseline vs. Candidate")

        models = clf_metrics["models"]
        metrics_display = [
            ("Accuracy", "accuracy"),
            ("Balanced Accuracy", "balanced_accuracy"),
            ("Recall (Failure Class 1)", "recall"),
            ("Precision (Failure Class 1)", "precision"),
            ("F1-Score", "f1_score"),
            ("PR-AUC (Average Precision)", "pr_auc"),
            ("ROC-AUC", "roc_auc"),
        ]

        table_rows = []
        for label, key in metrics_display:
            table_rows.append(
                {
                    "Evaluation Metric": label,
                    "Logistic Regression (Baseline)": f"{models['logistic_regression'][key]:.4f}",
                    "Random Forest (Primary Candidate)": f"{models['random_forest'][key]:.4f}",
                }
            )

        st.table(pd.DataFrame(table_rows))

        # Confusion Matrices visualization
        st.subheader("Confusion Matrix Comparison")
        col1, col2 = st.columns(2)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
        fig.patch.set_facecolor("#1a1f2c")

        cm_lr = models["logistic_regression"]["confusion_matrix"]
        cm_rf = models["random_forest"]["confusion_matrix"]

        mat_lr = np.array([[cm_lr["true_negatives"], cm_lr["false_positives"]], [cm_lr["false_negatives"], cm_lr["true_positives"]]])
        mat_rf = np.array([[cm_rf["true_negatives"], cm_rf["false_positives"]], [cm_rf["false_negatives"], cm_rf["true_positives"]]])

        cax1 = ax1.matshow(mat_lr, cmap=plt.cm.Blues, alpha=0.85)
        thresh_lr = mat_lr.max() / 2.0
        for i in range(2):
            for j in range(2):
                val = mat_lr[i, j]
                cell_color = "white" if val > thresh_lr else "#0f172a"
                ax1.text(j, i, str(val), va="center", ha="center", fontsize=12, fontweight="bold", color=cell_color)
        ax1.set_title("Logistic Regression", pad=15, color="#ffffff", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Predicted Label", color="#e2e8f0", fontsize=10)
        ax1.set_ylabel("True Label", color="#e2e8f0", fontsize=10)
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.set_xticklabels(["Normal (0)", "Failure (1)"], color="#cbd5e1")
        ax1.set_yticklabels(["Normal (0)", "Failure (1)"], color="#cbd5e1")
        ax1.set_facecolor("#1a1f2c")

        cax2 = ax2.matshow(mat_rf, cmap=plt.cm.Greens, alpha=0.85)
        thresh_rf = mat_rf.max() / 2.0
        for i in range(2):
            for j in range(2):
                val = mat_rf[i, j]
                cell_color = "white" if val > thresh_rf else "#0f172a"
                ax2.text(j, i, str(val), va="center", ha="center", fontsize=12, fontweight="bold", color=cell_color)
        ax2.set_title("Random Forest", pad=15, color="#ffffff", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Predicted Label", color="#e2e8f0", fontsize=10)
        ax2.set_ylabel("True Label", color="#e2e8f0", fontsize=10)
        ax2.set_xticks([0, 1])
        ax2.set_yticks([0, 1])
        ax2.set_xticklabels(["Normal (0)", "Failure (1)"], color="#cbd5e1")
        ax2.set_yticklabels(["Normal (0)", "Failure (1)"], color="#cbd5e1")
        ax2.set_facecolor("#1a1f2c")

        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Engineering Analysis: False Negatives vs. False Positives")
        st.markdown(
            """
            In industrial predictive maintenance, the costs of classification errors are heavily asymmetric:
            - **False Negatives (Missed Failures):** A machine fails unexpectedly during a production run. 
              Costs include catastrophic tool breakage, scrapped workpieces, spindle motor destruction, and unplanned factory downtime.
            - **False Positives (False Alarms):** The model flags a healthy machine for maintenance. 
              Costs include a brief 10-minute visual inspection by a technician and minor scheduling disruption.

            **Why Random Forest is superior here:**
            - Logistic Regression achieves 82.4% Recall, but generates **339 false alarms** (Precision = 14.2%), causing severe alarm fatigue.
            - Random Forest maintains a strong **75.0% Recall** while cutting false alarms to only **54** (Precision = 48.6%), raising PR-AUC from 0.382 to **0.647**.
            """
        )

        st.markdown("---")
        st.subheader("Unsupervised Anomaly Detection Reference Evaluation")
        st.markdown(
            "Isolation Forest evaluated against failure labels as an imperfect reference point:"
        )
        anom_eval = anom_metrics["test_evaluation_vs_failure_labels"]
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Precision (vs Failure Labels)", f"{anom_eval['precision_vs_failure']:.4f}")
        with col_b:
            st.metric("Recall (vs Failure Labels)", f"{anom_eval['recall_vs_failure']:.4f}")
        with col_c:
            st.metric("F1-Score", f"{anom_eval['f1_vs_failure']:.4f}")

        st.caption(anom_eval["interpretation_note"])

    except Exception as e:
        st.error(f"Error displaying evaluation metrics: {e}")


def render_about():
    """Render About page with dataset provenance, schema, and technology stack."""
    st.title("About SentinelAI: Dataset & Methodology")

    st.subheader("Dataset Provenance & Schema")
    st.markdown(
        """
        - **Source:** [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2B)
        - **Total Records:** 10,000 synthetic CNC milling cycles reflecting real physical machine failure models.
        - **Class Balance:** 9,661 normal operating records (96.61%), 339 failure records (3.39%).
        """
    )

    st.subheader("Strict Leakage Prevention Protocol")
    st.markdown(
        """
        To guarantee genuine generalization and prevent data leakage:
        1. **Supervised Target:** `Machine failure` (binary 0 or 1). Excluded from predictors.
        2. **Failure Mode Codes Excluded:** `TWF` (Tool Wear Failure), `HDF` (Heat Dissipation Failure), 
           `PWF` (Power Failure), `OSF` (Overstrain Failure), and `RNF` (Random Failure) are omitted because 
           they explicitly encode the failure event.
        3. **Row Identifiers Excluded:** `UDI` is an arbitrary sequence number (1–10,000) that carries no physical meaning.
        4. **Product ID Treatment:** `Product ID` (e.g. `L47181`) consists of a quality letter (`L`, `M`, `H`) 
           and a sequential serial number. The serial number is non-physical; the quality variant is already 
           cleanly captured in the `Type` feature. Thus, `Product ID` is excluded.
        5. **Preprocessing Isolation:** All scaling (`StandardScaler`) and encoding (`OneHotEncoder`) 
           are fit strictly inside Scikit-learn pipelines on training splits only.
        """
    )

    st.subheader("Technology Stack & Design Rationale")
    st.markdown(
        """
        - **Data Handling:** `pandas`, `numpy`
        - **Modeling:** `scikit-learn` (Pipelines, ColumnTransformer, RandomForestClassifier, LogisticRegression, IsolationForest)
        - **Vector Search:** `faiss-cpu` (Inner Product / Cosine Similarity Index)
        - **Dense Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
        - **Dashboard:** `streamlit`
        - **Testing:** `pytest`
        
        *Architectural Constraint:* SentinelAI deliberately avoids heavy orchestration frameworks (LangChain, LlamaIndex), 
        deep neural networks, and black-box wrappers to maintain complete transparency, fast execution, 
        and clear interview explainability for an electrical engineering student.
        """
    )


def main():
    if not check_artifacts_ready():
        st.warning("⚠️ **Required Model Artifacts Missing on Clean Environment**")
        st.info(
            "Some pre-trained model binaries or vector indices were not found in `artifacts/`. "
            "You can initialize them automatically using the project's data and training pipelines below."
        )
        if st.button("⚙️ Initialize & Build Artifacts Now", type="primary"):
            with st.spinner("Downloading dataset, fitting models, and building vector index (~15s)..."):
                success, msg = rebuild_missing_artifacts()
                if success:
                    st.success("Initialization complete! Reloading dashboard...")
                    st.rerun()
                else:
                    st.error(f"Failed to auto-generate artifacts: {msg}")
                    st.code(
                        "Manual Setup Instructions:\n"
                        "python scripts/download_data.py\n"
                        "python scripts/train_models.py\n"
                        "python scripts/build_rag_index.py"
                    )
        return

    page, readings, preset_choice = render_sidebar()

    if page == "Overview":
        render_overview()
    elif page == "Failure Risk Classifier":
        render_failure_risk(readings)
    elif page == "Anomaly Detection":
        render_anomaly_detection(readings)
    elif page == "Maintenance Assistant":
        render_maintenance_assistant()
    elif page == "Model Evaluation":
        render_model_evaluation()
    elif page == "About & Methodology":
        render_about()


if __name__ == "__main__":
    main()
