"""
Bank Term-Deposit Prediction - Streamlit app
--------------------------------------------
Predicts whether a customer will subscribe to a term deposit ("deposit" = yes/no).

Files expected next to this app.py:
    model.pkl                      - sklearn Pipeline (preprocessing -> model) that accepts RAW columns
    bank.csv                       - raw dataset (dropdown values + overview)
    reports/model_comparison.csv   - (optional) model comparison table from `dvc repro`
    reports/metrics.json           - (optional) final test metrics from `dvc repro`

The app reads the input columns from the model itself (`feature_names_in_`), so it adapts
to the pipeline: e.g. no "duration" input is shown when training used drop_duration: true.

Run locally:  streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
DATA_PATH = BASE_DIR / "bank.csv"
REPORTS_DIR = BASE_DIR / "reports"

EDU_OPTIONS = ["primary", "secondary", "tertiary", "unknown"]
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
DEFAULT_POUTCOME = "unknown"   # sent when the model needs poutcome but the UI doesn't ask for it
MODEL_ALIASES = {"KNeighbors": "KNN", "SVC": "SVM"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Load model.pkl. Raises a clear error instead of a cryptic unpickling error."""
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


@st.cache_data(show_spinner="Loading data...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


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
            "exposes `feature_names_in_`. See the note in README / train script."
        )
    return list(cols)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    if "poutcome" in INPUT_COLS and "poutcome" not in raw.columns:
        raw["poutcome"] = DEFAULT_POUTCOME

    missing = [c for c in INPUT_COLS if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(missing)}")

    X = raw[INPUT_COLS]
    pred = model.predict(X)

    classes = list(getattr(model, "classes_", [0, 1]))
    positive = 1 if 1 in classes else ("yes" if "yes" in classes else classes[-1])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, classes.index(positive)].round(4)
    else:  # e.g. SVM trained without probability=True
        proba = np.full(len(raw), np.nan)

    out = raw.copy()
    out["probability_yes"] = proba
    out["prediction"] = np.where(pred == positive, "yes", "no")
    return out


# ---------------------------------------------------------------------------
# Page setup + startup checks
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Bank Term Deposit Predictor", page_icon="🏦", layout="wide")
st.title("🏦 Bank Term Deposit Predictor")
st.caption("Predict whether a customer will subscribe to a term deposit after a marketing call.")

missing_files = [p.name for p in (MODEL_PATH, DATA_PATH) if not p.exists()]
if missing_files:
    st.error(f"Missing required file(s): {', '.join(missing_files)}. Place them next to app.py.")
    st.stop()

try:
    model = load_model()
    INPUT_COLS = get_input_cols(model)
except Exception as e:  # noqa: BLE001 - show any load problem in the UI
    st.error(f"Could not load the model: {e}")
    st.stop()

df_ref = load_data()
MODEL_LABEL = model_label(model)
UI_COLS = [c for c in INPUT_COLS if c != "poutcome"]   # poutcome is not asked

tab_single, tab_batch, tab_info = st.tabs(
    ["🔮 Single prediction", "📂 Batch prediction", "ℹ️ Model info"]
)


def render_input(col: str):
    """One input widget per model column."""
    if col == "age":
        return st.number_input("Age", 18, 100, 40)
    if col == "job":
        return st.selectbox("Job", sorted(df_ref["job"].unique()))
    if col == "marital":
        return st.selectbox("Marital status", sorted(df_ref["marital"].unique()))
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
        return st.selectbox("Contact type", sorted(df_ref["contact"].unique()))
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


# ---------------------------------------------------------------------------
# Tab 1: single prediction
# ---------------------------------------------------------------------------
with tab_single:
    st.subheader("Customer details")
    values = {}
    cols = st.columns(3)
    for i, c in enumerate(UI_COLS):
        with cols[i % 3]:
            values[c] = render_input(c)

    if st.button("Predict", type="primary"):
        try:
            res = predict(pd.DataFrame([values])).iloc[0]
        except Exception as e:  # noqa: BLE001
            st.error(f"Prediction failed: {e}")
        else:
            if res["prediction"] == "yes":
                st.success("✅ Likely to SUBSCRIBE to a term deposit")
            else:
                st.warning("❌ Unlikely to subscribe to a term deposit")
            if pd.notna(res["probability_yes"]):
                st.metric("Probability of 'yes'", f"{res['probability_yes']:.1%}")
                st.progress(float(res["probability_yes"]))
            else:
                st.info("This model does not provide probabilities.")


# ---------------------------------------------------------------------------
# Tab 2: batch prediction
# ---------------------------------------------------------------------------
with tab_batch:
    st.subheader("Upload a CSV")
    st.write("Required columns: " + ", ".join(f"`{c}`" for c in UI_COLS))

    template = pd.DataFrame([{c: None for c in UI_COLS}])
    st.download_button("Download empty template", template.to_csv(index=False),
                       "template.csv", "text/csv")

    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        try:
            batch = pd.read_csv(up)
            result = predict(batch)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not score this file: {e}")
        else:
            st.success(f"Scored {len(result):,} rows. "
                       f"{(result['prediction'] == 'yes').mean():.1%} predicted 'yes'.")
            st.dataframe(result, use_container_width=True)
            st.download_button("Download predictions", result.to_csv(index=False),
                               "predictions.csv", "text/csv")


# ---------------------------------------------------------------------------
# Tab 3: model info
# ---------------------------------------------------------------------------
with tab_info:
    st.subheader("Model")
    st.write(f"**Algorithm:** {MODEL_LABEL}")
    st.write("**Input columns:** " + ", ".join(f"`{c}`" for c in INPUT_COLS))
    if "duration" not in INPUT_COLS:
        st.caption("`duration` is excluded (only known after the call ends -> target leakage).")

    comp_path = REPORTS_DIR / "model_comparison.csv"
    if comp_path.exists():
        st.subheader("Model comparison")
        st.dataframe(pd.read_csv(comp_path), use_container_width=True)

    metrics_path = REPORTS_DIR / "metrics.json"
    if metrics_path.exists():
        st.subheader("Final test metrics")
        try:
            st.json(json.loads(metrics_path.read_text()))
        except json.JSONDecodeError:
            st.caption("metrics.json could not be read.")

    st.subheader("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df_ref):,}")
    c2.metric("Columns", df_ref.shape[1])
    if "deposit" in df_ref.columns:
        c3.metric("Subscription rate", f"{(df_ref['deposit'] == 'yes').mean():.1%}")
    st.dataframe(df_ref.head(20), use_container_width=True)