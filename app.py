"""Streamlit serving application for the Bank Term-Deposit MLOps Pipeline.

Features:
- Live model inference with class probability visualization and batch CSV scoring.
- Production payload and prediction logging for data drift tracking.
- Interactive model metadata, lineage, and benchmark inspector.
- Real-time data drift monitoring dashboard.
"""
from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.monitor_drift import (
    INFERENCE_LOG_PATH,
    RAW_FALLBACK_PATH,
    REFERENCE_DATA_PATH,
    analyze_drift,
    generate_sample_inferences,
    load_inference_logs,
    log_inference,
)

MODEL_PATH = BASE_DIR / "model.pkl"
DATA_PATH = BASE_DIR / "bank.csv"
SELECTION_PATH = BASE_DIR / "reports" / "model_selection.json"
METRICS_PATH = BASE_DIR / "reports" / "metrics.json"
COMPARISON_CSV_PATH = BASE_DIR / "reports" / "model_comparison.csv"

EDU_OPTIONS = ["primary", "secondary", "tertiary", "unknown"]
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
DEFAULT_POUTCOME = "unknown"
MODEL_ALIASES = {"KNeighbors": "KNN", "SVC": "SVM"}

st.set_page_config(
    page_title="Bank Term Deposit Predictor",
    page_icon="🏦",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Load model.pkl with file integrity checks."""
    if not MODEL_PATH.exists():
        return None
    size = MODEL_PATH.stat().st_size
    head = MODEL_PATH.read_bytes()[:200]
    if size == 0:
        raise ValueError("model.pkl is empty. Run `dvc pull` or `dvc repro`.")
    if b"git-lfs" in head or b"outs:" in head or b"md5:" in head:
        raise ValueError("model.pkl is a Git LFS / DVC pointer, not the real model. Run `dvc pull`.")
    if head[:1] != b"\x80":
        raise ValueError(
            "model.pkl is not a valid pickle file (it may have been corrupted by CRLF "
            "conversion or saved as text). Re-run `dvc repro` and mark *.pkl as binary "
            "in .gitattributes."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="Loading reference dataset...")
def load_reference_data() -> pd.DataFrame | None:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    raw_path = BASE_DIR / RAW_FALLBACK_PATH
    if raw_path.exists():
        return pd.read_csv(raw_path)
    return None


@st.cache_data
def load_reports():
    selection, metrics = None, None
    if SELECTION_PATH.exists():
        with open(SELECTION_PATH, "r", encoding="utf-8") as f:
            selection = json.load(f)
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    return selection, metrics


def final_estimator(model):
    return model.steps[-1][1] if hasattr(model, "steps") else model


def model_label(model) -> str:
    name = type(final_estimator(model)).__name__.replace("Classifier", "")
    return MODEL_ALIASES.get(name, name)


def get_input_cols(model) -> list[str]:
    cols = getattr(model, "feature_names_in_", None)
    if cols is None:
        raise ValueError(
            "This model does not know its input columns. It must be a sklearn Pipeline "
            "(preprocessing + estimator) fitted on a raw-column DataFrame, so that it "
            "exposes `feature_names_in_`."
        )
    return list(cols)


def predict_batch(raw: pd.DataFrame, mdl, input_cols: list[str]) -> pd.DataFrame:
    prepared = raw.copy()
    if "poutcome" in input_cols and "poutcome" not in prepared.columns:
        prepared["poutcome"] = DEFAULT_POUTCOME

    missing = [c for c in input_cols if c not in prepared.columns]
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(missing)}")

    X = prepared[input_cols]
    pred = mdl.predict(X)

    classes = list(getattr(mdl, "classes_", [0, 1]))
    positive = 1 if 1 in classes else ("yes" if "yes" in classes else classes[-1])

    if hasattr(mdl, "predict_proba"):
        proba = mdl.predict_proba(X)[:, classes.index(positive)].round(4)
    else:
        proba = np.full(len(prepared), np.nan)

    out = raw.copy()
    out["probability_yes"] = proba
    out["prediction"] = np.where(pred == positive, "yes", "no")
    return out


# -----------------------------------------------------------------------------
# Load Model, Reference Data & Reports
# -----------------------------------------------------------------------------
model = None
input_cols: list[str] = []
model_error: str | None = None

try:
    model = load_model()
    if model is not None:
        input_cols = get_input_cols(model)
except Exception as exc:  # noqa: BLE001
    model_error = str(exc)

df_ref = load_reference_data()
selection, metrics = load_reports()
ui_cols = [c for c in input_cols if c != "poutcome"]

# -----------------------------------------------------------------------------
# Sidebar: MLOps Model Lineage & System Info
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🏦 MLOps Platform")
    st.caption("Automated Champion Model Serving")

    if selection and "champion" in selection:
        champ = selection["champion"]
        crit = champ.get("criteria", {})
        st.subheader("🏆 Active Champion")
        st.info(f"**Model:** {champ.get('display_name', champ.get('model_name'))}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Val Acc", f"{crit.get('validation_accuracy_pct', 0.0):.1f}%")
            st.metric("Latency", f"{crit.get('prediction_time_ms', 0.0):.2f} ms")
        with col_b:
            if metrics:
                st.metric("Test Acc", f"{metrics.get('accuracy', 0.0) * 100:.1f}%")
            st.metric("Size", f"{crit.get('model_size_mb', 0.0):.3f} MB")

        with st.expander("Selection Rationale"):
            st.write(champ.get("explanation", "Selected via multi-criteria rules."))
    elif model is not None:
        st.subheader("🏆 Active Champion")
        st.info(f"**Model:** {model_label(model)}")
    else:
        st.warning("Model metadata not found. Run `dvc repro` to build.")

    st.divider()
    st.subheader("📡 Live Monitoring")
    log_df = load_inference_logs(BASE_DIR / INFERENCE_LOG_PATH)
    st.metric("Logged Inferences", len(log_df))

    st.caption("Powered by DVC, MLflow, Optuna & Streamlit")


def render_input(col: str, ref_df: pd.DataFrame | None):
    """Render one Streamlit input widget per model feature column."""
    if col == "age":
        return st.number_input("Age", 18, 100, 40)
    if col == "job":
        options = sorted(ref_df["job"].dropna().unique()) if ref_df is not None and "job" in ref_df.columns else ["admin.", "technician", "management", "blue-collar"]
        return st.selectbox("Job", options)
    if col == "marital":
        options = sorted(ref_df["marital"].dropna().unique()) if ref_df is not None and "marital" in ref_df.columns else ["married", "single", "divorced"]
        return st.selectbox("Marital status", options)
    if col == "education":
        return st.selectbox("Education", EDU_OPTIONS, index=1)
    if col == "balance":
        return st.number_input("Average yearly balance (€)", -10000, 120000, 1500, step=100)
    if col == "default":
        return st.selectbox("Credit in default?", ["no", "yes"])
    if col == "housing":
        return st.selectbox("Housing loan?", ["no", "yes"])
    if col == "loan":
        return st.selectbox("Personal loan?", ["no", "yes"])
    if col == "contact":
        options = sorted(ref_df["contact"].dropna().unique()) if ref_df is not None and "contact" in ref_df.columns else ["cellular", "telephone", "unknown"]
        return st.selectbox("Contact type", options)
    if col == "day":
        return st.number_input("Last contact day of month", 1, 31, 15)
    if col == "month":
        return st.selectbox("Last contact month", MONTHS, index=4)
    if col == "duration":
        return st.number_input("Last contact duration (seconds)", 0, 5000, 180)
    if col == "campaign":
        return st.number_input("Contacts during this campaign", 1, 60, 2)
    if col == "pdays":
        return st.number_input("Days since previous contact (-1 = never)", -1, 900, -1)
    if col == "previous":
        return st.number_input("Previous campaign contacts", 0, 300, 0)
    return None


# -----------------------------------------------------------------------------
# Main Application Tabs
# -----------------------------------------------------------------------------
tab_predict, tab_batch, tab_metrics, tab_monitoring = st.tabs([
    "🏦 Live Inference",
    "📂 Batch Scoring",
    "📊 Model Lineage & Metrics",
    "🛡️ Data Drift & Monitoring",
])

# ------------------ TAB 1: LIVE INFERENCE ------------------
with tab_predict:
    st.header("Term Deposit Subscription Prediction")
    st.write(
        "Provide customer and campaign details below to receive real-time predictions "
        "from the production champion model:"
    )

    if model_error:
        st.error(f"Could not load model: {model_error}")
        st.stop()

    if model is None:
        st.error(
            f"Champion model not found at `{MODEL_PATH}`. "
            "Please run `dvc repro` to train and select the champion model."
        )
        st.stop()

    values = {}
    cols = st.columns(3)
    for idx, col_name in enumerate(ui_cols):
        with cols[idx % 3]:
            values[col_name] = render_input(col_name, df_ref)

    input_data = pd.DataFrame([values])
    if "poutcome" in input_cols and "poutcome" not in input_data.columns:
        input_data["poutcome"] = DEFAULT_POUTCOME

    if st.button("Predict Subscription", type="primary"):
        res_df = predict_batch(input_data, model, input_cols)
        res = res_df.iloc[0]
        pred_label = str(res["prediction"])
        prob_yes = float(res["probability_yes"]) if pd.notna(res["probability_yes"]) else None

        proba_dict = None
        if prob_yes is not None:
            proba_dict = {
                "No": round(1.0 - prob_yes, 4),
                "Yes": round(prob_yes, 4),
            }

        log_inference(
            input_data,
            pred_label,
            {"no": proba_dict["No"], "yes": proba_dict["Yes"]} if proba_dict else None,
            log_path=BASE_DIR / INFERENCE_LOG_PATH,
        )

        if pred_label == "yes":
            st.success("### Prediction: **YES** — Likely to Subscribe")
        else:
            st.warning("### Prediction: **NO** — Unlikely to Subscribe")

        if proba_dict:
            st.write("##### Subscription Probabilities")
            proba_df = pd.DataFrame(
                list(proba_dict.items()), columns=["Deposit", "Probability"]
            ).set_index("Deposit")
            st.bar_chart(proba_df)
        else:
            st.info("This model does not provide class probabilities.")

# ------------------ TAB 2: BATCH SCORING ------------------
with tab_batch:
    st.header("Batch CSV Scoring")
    if model is None or model_error:
        st.info("Load a valid `model.pkl` first to run batch scoring.")
    else:
        st.write("Required input columns: " + ", ".join(f"`{c}`" for c in ui_cols))
        template = pd.DataFrame([{c: None for c in ui_cols}])
        st.download_button(
            "Download empty template",
            template.to_csv(index=False),
            "template.csv",
            "text/csv",
        )

        uploaded = st.file_uploader("Upload customer CSV", type="csv")
        if uploaded is not None:
            try:
                batch_df = pd.read_csv(uploaded)
                scored_df = predict_batch(batch_df, model, input_cols)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not score uploaded file: {exc}")
            else:
                st.success(
                    f"Scored {len(scored_df):,} rows. "
                    f"{(scored_df['prediction'] == 'yes').mean():.1%} predicted `yes`."
                )
                st.dataframe(scored_df, use_container_width=True)
                st.download_button(
                    "Download predictions",
                    scored_df.to_csv(index=False),
                    "predictions.csv",
                    "text/csv",
                )

# ------------------ TAB 3: MODEL LINEAGE & METRICS ------------------
with tab_metrics:
    st.header("Model Lineage & Multi-Criteria Benchmark")
    st.write(
        "Candidate models were evaluated across validation accuracy, generalization gap, "
        "model size, training time, and inference latency to pick the production champion:"
    )

    if COMPARISON_CSV_PATH.exists():
        st.dataframe(pd.read_csv(COMPARISON_CSV_PATH), use_container_width=True)
    elif selection:
        st.json(selection)
    else:
        st.info("Run `dvc repro` to generate comparison metrics.")

    if metrics:
        st.subheader("Final Test Metrics")
        st.json(metrics)

    if df_ref is not None:
        st.subheader("Reference Dataset Overview")
        m1, m2, m3 = st.columns(3)
        m1.metric("Rows", f"{len(df_ref):,}")
        m2.metric("Columns", df_ref.shape[1])
        if "deposit" in df_ref.columns:
            sub_rate = (
                df_ref["deposit"].astype(str).str.strip().str.lower().isin({"yes", "1", "true"})
            ).mean()
            m3.metric("Subscription Rate", f"{sub_rate:.1%}")
        st.dataframe(df_ref.head(20), use_container_width=True)

# ------------------ TAB 4: DATA DRIFT MONITORING ------------------
with tab_monitoring:
    st.header("Production Data Drift & Quality Monitoring")
    st.write(
        "Monitoring production inputs against the reference training dataset using "
        "**Kolmogorov-Smirnov (KS)** for numeric features and **Chi-Square** for categorical features."
    )

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 3])
    with col_btn1:
        run_check = st.button("Run Drift Check", type="primary")
    with col_btn2:
        sim_drift = st.button("Simulate Drift")

    log_path = BASE_DIR / INFERENCE_LOG_PATH
    if sim_drift:
        log_df = generate_sample_inferences(n_samples=50, drift=True, log_path=log_path)
        run_check = True
    else:
        log_df = load_inference_logs(log_path)

    if not log_df.empty:
        st.write(f"**Total production records logged:** `{len(log_df)}`")

        if run_check or len(log_df) >= 5:
            ref_file = BASE_DIR / REFERENCE_DATA_PATH
            fallback_file = BASE_DIR / RAW_FALLBACK_PATH
            if ref_file.exists():
                ref_df = pd.read_csv(ref_file)
            elif fallback_file.exists():
                ref_df = pd.read_csv(fallback_file).drop_duplicates()
            else:
                ref_df = None

            if ref_df is not None:
                report = analyze_drift(ref_df, log_df)
                status_color = "red" if report["dataset_drift_detected"] else "green"
                status_msg = (
                    "⚠️ DRIFT DETECTED"
                    if report["dataset_drift_detected"]
                    else "✅ HEALTHY DISTRIBUTION"
                )

                st.markdown(f"#### Status: :{status_color}[{status_msg}]")
                st.write(
                    f"Drifted features: **{report['drifted_features']} / {report['total_features']}** "
                    f"({report['drift_share_pct']}%)"
                )

                drift_rows = []
                for feat, d in report["features"].items():
                    drift_rows.append({
                        "Feature": feat,
                        "Test": d.get("test", "KS"),
                        "Status": "DRIFT ⚠️" if d["drift_detected"] else "HEALTHY ✅",
                        "p-value": round(float(d["p_value"]), 4),
                        "Statistic": round(float(d["ks_statistic"]), 4),
                        "Reference": d.get("reference_summary", ""),
                        "Production": d.get("current_summary", ""),
                        "Shift": round(float(d["mean_shift"]), 2),
                    })
                st.dataframe(pd.DataFrame(drift_rows), use_container_width=True)

        with st.expander("View Recent Production Logs (Last 10 Requests)"):
            st.dataframe(log_df.tail(10), use_container_width=True)
    else:
        st.info(
            "No production inferences logged yet. Submit a prediction in **Live Inference** "
            "or click **Simulate Drift** above."
        )