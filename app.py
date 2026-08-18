"""Interactive Streamlit application for ML Assignment 2."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from model.train_models import (
    TARGET_COLUMN,
    calculate_metrics,
    evaluate_all_models,
    train_all_models,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_EVALUATION_ROWS = 5_000
STUDENT_NAME = "Ayush Khamrui"
STUDENT_ID = "2025AC05152"

st.set_page_config(
    page_title="Diagnostic Model Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #14213d;
        --muted: #5b677a;
        --line: #dce3ee;
        --surface: #ffffff;
        --canvas: #f4f7fb;
        --accent: #2457d6;
        --accent-soft: #e9efff;
        --success: #0d766e;
    }
    .stApp {
        background:
            radial-gradient(circle at 82% 3%, rgba(36, 87, 214, 0.08), transparent 23rem),
            var(--canvas);
        color: var(--ink);
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1240px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }
    [data-testid="stSidebar"] {
        background: #edf2f8;
        border-right: 1px solid #d7e0ec;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.85rem;
    }
    .hero-panel {
        position: relative;
        overflow: hidden;
        padding: 2.2rem 2.35rem;
        margin-bottom: 1.6rem;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 22px;
        background: linear-gradient(125deg, #13213d 0%, #1d3769 58%, #2457d6 120%);
        box-shadow: 0 18px 40px rgba(20, 33, 61, 0.14);
        color: white;
    }
    .hero-panel::after {
        content: "";
        position: absolute;
        width: 250px;
        height: 250px;
        right: -70px;
        top: -125px;
        border-radius: 50%;
        border: 42px solid rgba(255, 255, 255, 0.07);
    }
    .hero-kicker {
        margin-bottom: 0.55rem;
        color: #b9caff;
        font-size: 0.76rem;
        font-weight: 750;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }
    .hero-title {
        max-width: 760px;
        margin: 0;
        color: #ffffff;
        font-size: clamp(2rem, 4vw, 3.2rem);
        font-weight: 760;
        letter-spacing: -0.045em;
        line-height: 1.03;
    }
    .hero-copy {
        max-width: 720px;
        margin: 0.9rem 0 1.25rem;
        color: #dfe8ff;
        font-size: 1.02rem;
        line-height: 1.65;
    }
    .hero-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
    }
    .hero-meta span {
        padding: 0.42rem 0.72rem;
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.08);
        color: #f4f7ff;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .section-intro {
        margin: 1.8rem 0 0.75rem;
    }
    .section-kicker {
        color: var(--accent);
        font-size: 0.72rem;
        font-weight: 760;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .section-title {
        margin: 0.18rem 0 0.2rem;
        color: var(--ink);
        font-size: 1.42rem;
        font-weight: 730;
        letter-spacing: -0.02em;
    }
    .section-copy {
        margin: 0;
        color: var(--muted);
        font-size: 0.93rem;
        line-height: 1.55;
    }
    .winner-panel,
    .notice-panel {
        margin: 0.85rem 0 1rem;
        padding: 0.9rem 1rem;
        border-radius: 13px;
    }
    .winner-panel {
        border: 1px solid #b9ddd7;
        background: #eaf7f4;
        color: #165c57;
    }
    .notice-panel {
        border: 1px solid #cbd8f2;
        background: #edf3ff;
        color: #294a87;
    }
    .winner-panel strong,
    .notice-panel strong {
        margin-right: 0.35rem;
        color: inherit;
    }
    .sidebar-brand {
        padding: 1.2rem 0 0.65rem;
        border-bottom: 1px solid #d5deea;
        margin-bottom: 0.35rem;
    }
    .sidebar-brand strong {
        display: block;
        color: var(--ink);
        font-size: 1.12rem;
        letter-spacing: -0.02em;
    }
    .sidebar-brand span {
        color: var(--muted);
        font-size: 0.8rem;
    }
    .sidebar-label {
        margin: 0.2rem 0 -0.35rem;
        color: #66758c;
        font-size: 0.7rem;
        font-weight: 750;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    [data-testid="stMetric"] {
        min-height: 108px;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 0.9rem 1rem;
        background: rgba(255, 255, 255, 0.88);
        box-shadow: 0 8px 22px rgba(37, 55, 86, 0.055);
    }
    [data-testid="stMetricLabel"] {
        color: #69758a;
        font-size: 0.78rem;
        font-weight: 650;
    }
    [data-testid="stMetricValue"] {
        color: var(--ink);
        font-weight: 740;
        letter-spacing: -0.035em;
    }
    [data-testid="stDataFrame"] {
        overflow: hidden;
        border: 1px solid var(--line);
        border-radius: 15px;
        box-shadow: 0 8px 22px rgba(37, 55, 86, 0.045);
    }
    [data-testid="stFileUploader"] section {
        border: 1px dashed #9cadc3;
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.65);
    }
    .stButton > button,
    .stDownloadButton > button {
        border: 1px solid #c9d5e5;
        border-radius: 10px;
        font-weight: 650;
        transition: all 160ms ease;
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: var(--accent);
        color: var(--accent);
        transform: translateY(-1px);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
        padding: 0.25rem;
        border-radius: 12px;
        background: #e9eef5;
    }
    .stTabs [data-baseweb="tab"] {
        height: 2.55rem;
        border-radius: 9px;
        padding: 0 1rem;
        color: #526078;
        font-weight: 650;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff;
        color: var(--ink);
        box-shadow: 0 2px 8px rgba(37, 55, 86, 0.08);
    }
    [data-testid="stAlert"] {
        border-radius: 13px;
    }
    details {
        border: 1px solid var(--line) !important;
        border-radius: 14px !important;
        background: rgba(255, 255, 255, 0.72) !important;
    }
    @media (max-width: 760px) {
        [data-testid="stMainBlockContainer"] { padding-top: 1rem; }
        .hero-panel { padding: 1.55rem 1.35rem; border-radius: 17px; }
        .hero-copy { font-size: 0.94rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Training the six classifiers...")
def get_models():
    return train_all_models()


def read_uploaded_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.read_csv(PROJECT_ROOT / "test_data.csv")
    if uploaded_file.size > MAX_UPLOAD_BYTES:
        raise ValueError("The CSV is larger than the 5 MB upload limit.")
    return pd.read_csv(uploaded_file, low_memory=False)


def prepare_features(
    frame: pd.DataFrame, expected_features: list[str]
) -> pd.DataFrame:
    """Validate untrusted CSV input and return only numeric model features."""
    if frame.empty:
        raise ValueError("The CSV has no data rows.")
    if len(frame) > MAX_EVALUATION_ROWS:
        raise ValueError(
            f"The CSV has more than {MAX_EVALUATION_ROWS:,} rows. "
            "Upload test data only."
        )

    missing_columns = [
        column for column in expected_features if column not in frame.columns
    ]
    if missing_columns:
        preview = ", ".join(missing_columns[:5])
        suffix = " ..." if len(missing_columns) > 5 else ""
        raise ValueError(f"Missing required feature columns: {preview}{suffix}")

    numeric_features = frame[expected_features].apply(
        pd.to_numeric, errors="coerce"
    )
    invalid_columns = numeric_features.columns[
        numeric_features.isna().any(axis=0)
    ].tolist()
    if invalid_columns:
        preview = ", ".join(invalid_columns[:5])
        suffix = " ..." if len(invalid_columns) > 5 else ""
        raise ValueError(
            "Feature values must be numeric and non-empty. "
            f"Check: {preview}{suffix}"
        )
    if not np.isfinite(numeric_features.to_numpy(dtype=float)).all():
        raise ValueError("Feature values cannot contain infinity.")
    return numeric_features.astype(float)


def render_hero() -> None:
    st.markdown(
        f"""
        <section class="hero-panel">
            <div class="hero-kicker">Machine Learning · Assignment 2</div>
            <h1 class="hero-title">Diagnostic Model Studio</h1>
            <p class="hero-copy">
                Compare six classification methods on one reproducible test set,
                inspect model quality, and explore individual predictions.
            </p>
            <div class="hero-meta">
                <span>{STUDENT_NAME}</span>
                <span>Student ID {STUDENT_ID}</span>
                <span>569 records · 30 features · 6 models</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_section_intro(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-intro">
            <div class="section-kicker">{kicker}</div>
            <h2 class="section-title">{title}</h2>
            <p class="section-copy">{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(metric_row: dict[str, float]) -> None:
    columns = st.columns(6)
    metric_names = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    for column, metric_name in zip(columns, metric_names):
        value = metric_row[metric_name]
        display = "N/A" if pd.isna(value) else f"{value:.3f}"
        column.metric(metric_name, display)


def render_score_profile(comparison: pd.DataFrame) -> None:
    plot_data = comparison[["F1", "MCC"]].sort_values("F1")
    figure, axis = plt.subplots(figsize=(8.5, 4.6))
    figure.patch.set_alpha(0)
    axis.set_facecolor("none")
    plot_data.plot.barh(
        ax=axis,
        color=["#2457d6", "#71a6ff"],
        width=0.68,
    )
    axis.set_xlim(0, 1.02)
    axis.set_xlabel("Score", color="#5b677a")
    axis.set_ylabel("")
    axis.grid(axis="x", color="#dce3ee", linewidth=0.8, alpha=0.8)
    axis.grid(axis="y", visible=False)
    axis.tick_params(axis="both", colors="#5b677a", labelsize=9)
    axis.legend(frameon=False, loc="lower right", ncols=2)
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def render_diagnostics(model, features: pd.DataFrame, target: pd.Series) -> None:
    predictions = model.predict(features)
    left, right = st.columns([0.9, 1.1])
    with left:
        st.subheader("Confusion matrix")
        matrix = confusion_matrix(target, predictions, labels=[0, 1])
        figure, axis = plt.subplots(figsize=(4.8, 3.6))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Malignant", "Benign"],
            yticklabels=["Malignant", "Benign"],
            ax=axis,
        )
        axis.set_xlabel("Predicted diagnosis")
        axis.set_ylabel("Actual diagnosis")
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)
    with right:
        st.subheader("Classification report")
        report = classification_report(
            target,
            predictions,
            labels=[0, 1],
            target_names=["Malignant", "Benign"],
            output_dict=True,
            zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")


render_hero()

models, x_train, default_x_test, _, _ = get_models()
expected_features = list(x_train.columns)

with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-brand">
            <strong>Evaluation controls</strong>
            <span>{STUDENT_NAME} · {STUDENT_ID}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sidebar-label">Model view</div>',
        unsafe_allow_html=True,
    )
    selected_model = st.selectbox(
        "Choose a model",
        ["Compare all models", *models.keys()],
        help="Choose a single model for detailed diagnostics.",
    )
    st.markdown(
        '<div class="sidebar-label">Test dataset</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload test data (CSV)",
        type=["csv"],
        help="Use the supplied test_data.csv schema. Include target for evaluation metrics.",
    )
    st.download_button(
        "Download sample test data",
        data=(PROJECT_ROOT / "test_data.csv").read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
    )
    st.caption(
        "Use the bundled sample first. Uploads are limited to 5 MB and "
        f"{MAX_EVALUATION_ROWS:,} rows."
    )

try:
    uploaded_frame = read_uploaded_data(uploaded_file)
    evaluation_features = prepare_features(uploaded_frame, expected_features)
except Exception as error:
    st.error(f"The CSV could not be used: {error}")
    st.stop()

has_target = TARGET_COLUMN in uploaded_frame.columns
evaluation_target = None
if has_target:
    evaluation_target = pd.to_numeric(uploaded_frame[TARGET_COLUMN], errors="coerce")
    if (
        evaluation_target.isna().any()
        or not np.isfinite(evaluation_target.to_numpy(dtype=float)).all()
        or not set(evaluation_target.unique()).issubset({0, 1})
    ):
        st.error("The target column must contain only 0 (malignant) or 1 (benign).")
        st.stop()
    evaluation_target = evaluation_target.astype(int)

render_section_intro(
    "Dataset status",
    "Evaluation snapshot",
    "A quick view of the data currently loaded for this session.",
)
overview_a, overview_b, overview_c, overview_d = st.columns(4)
overview_a.metric("Rows evaluated", f"{len(evaluation_features):,}")
overview_b.metric("Features", len(expected_features))
overview_c.metric("Models ready", len(models))
overview_d.metric("Target labels", "Available" if has_target else "Missing")

if selected_model == "Compare all models":
    render_section_intro(
        "Cross-model analysis",
        "Model comparison",
        "Review every required metric on the same uploaded test data.",
    )
    if not has_target:
        st.markdown(
            """
            <div class="notice-panel">
                <strong>Metrics unavailable.</strong>
                Add a target column to calculate evaluation scores. Predictions
                are still available below.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### Prediction sample")
        prediction_frame = pd.DataFrame(index=uploaded_frame.index)
        for name, model in models.items():
            prediction_frame[f"{name} prediction"] = model.predict(
                evaluation_features
            )
        st.dataframe(prediction_frame.head(25), width="stretch")
        st.download_button(
            "Download all predictions",
            prediction_frame.to_csv(index=False).encode("utf-8"),
            file_name="all_model_predictions.csv",
            mime="text/csv",
        )
    else:
        comparison = evaluate_all_models(models, evaluation_features, evaluation_target)
        winner = comparison["F1"].idxmax()
        score_tab, prediction_tab = st.tabs(
            ["Performance summary", "Prediction sample"]
        )
        with score_tab:
            styled = comparison.round(4).style.highlight_max(
                axis=0,
                color="#dbe8ff",
            )
            st.dataframe(styled, width="stretch")
            st.markdown(
                f"""
                <div class="winner-panel">
                    <strong>Current F1 leader:</strong> {winner}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("#### Score profile")
            st.caption(
                "F1 summarizes precision and recall; MCC reflects overall "
                "binary-classification quality."
            )
            render_score_profile(comparison)
        with prediction_tab:
            prediction_frame = pd.DataFrame(index=uploaded_frame.index)
            for name, model in models.items():
                prediction_frame[f"{name} prediction"] = model.predict(
                    evaluation_features
                )
            st.dataframe(prediction_frame.head(25), width="stretch")
            st.download_button(
                "Download all predictions",
                prediction_frame.to_csv(index=False).encode("utf-8"),
                file_name="all_model_predictions.csv",
                mime="text/csv",
            )
else:
    model = models[selected_model]
    render_section_intro(
        "Selected model",
        selected_model,
        "Inspect model-level quality, error patterns, and row-level predictions.",
    )
    predictions = model.predict(evaluation_features)
    probabilities = model.predict_proba(evaluation_features)[:, 1]

    if has_target:
        metric_row = calculate_metrics(model, evaluation_features, evaluation_target)
        render_metric_cards(metric_row)

    # Export only the validated schema, not arbitrary extra upload columns.
    output = evaluation_features.copy()
    if has_target:
        output[TARGET_COLUMN] = evaluation_target
    output["predicted_target"] = predictions
    output["predicted_diagnosis"] = pd.Series(
        predictions, index=output.index
    ).map({0: "malignant", 1: "benign"})
    output["benign_probability"] = probabilities

    if has_target:
        diagnostics_tab, prediction_tab = st.tabs(
            ["Diagnostics", "Prediction sample"]
        )
        with diagnostics_tab:
            render_diagnostics(model, evaluation_features, evaluation_target)
        with prediction_tab:
            st.dataframe(output.head(25), width="stretch")
            st.download_button(
                "Download predictions",
                output.to_csv(index=False).encode("utf-8"),
                file_name="model_predictions.csv",
                mime="text/csv",
            )
    else:
        st.markdown(
            """
            <div class="notice-panel">
                <strong>Metrics unavailable.</strong>
                Add a target column for metrics and diagnostics. Predictions
                are available below.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(output.head(25), width="stretch")
        st.download_button(
            "Download predictions",
            output.to_csv(index=False).encode("utf-8"),
            file_name="model_predictions.csv",
            mime="text/csv",
        )

with st.expander("Dataset and reproducibility notes"):
    st.markdown(
        f"""
        - Dataset: Breast Cancer Wisconsin (Diagnostic), provided by scikit-learn from UCI.
        - Original size: 569 observations and 30 numeric features.
        - Fixed split: 80% training / 20% testing, stratified, random state 42.
        - Current training rows: {len(x_train)}; bundled test rows: {len(default_x_test)}.
        - Class labels: 0 = malignant, 1 = benign.
        - CSV safety limits: 5 MB and {MAX_EVALUATION_ROWS:,} rows; model features must be finite numbers.
        - Uploaded data is evaluated in memory and is not written to disk by this app.
        """
    )
