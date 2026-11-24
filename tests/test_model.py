from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

MODEL_PATH = Path("models/model.pkl")
X_TEST_PATH = Path("data/processed/X_test.csv")

FALLBACK_FEATURES = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
]


@pytest.fixture(scope="module")
def model():
    assert MODEL_PATH.exists(), f"Champion model not found at {MODEL_PATH}"
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def sample_features(model):
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
    elif X_TEST_PATH.exists():
        cols = list(pd.read_csv(X_TEST_PATH, nrows=1).columns)
    else:
        cols = FALLBACK_FEATURES

    if X_TEST_PATH.exists():
        df = pd.read_csv(X_TEST_PATH)
        return df[cols].copy()

    row = {}
    for col in cols:
        if col in {"age", "balance", "day", "duration", "campaign", "pdays", "previous"}:
            row[col] = 30 if col != "pdays" else -1
        elif col == "job":
            row[col] = "admin."
        elif col == "marital":
            row[col] = "married"
        elif col == "education":
            row[col] = "secondary"
        elif col in {"default", "housing", "loan"}:
            row[col] = "no"
        elif col == "contact":
            row[col] = "cellular"
        elif col == "month":
            row[col] = "may"
        elif col == "poutcome":
            row[col] = "unknown"
        else:
            row[col] = "unknown"
    return pd.DataFrame([row])


def test_model_pipeline_structure(model):
    """Ensure the champion is a Pipeline with Preprocessor and Model stages."""
    assert hasattr(model, "steps"), "Model should be a scikit-learn Pipeline"
    step_names = [name for name, _ in model.steps]
    assert "Preprocessor" in step_names, "Pipeline missing 'Preprocessor' stage"
    assert "Model" in step_names, "Pipeline missing 'Model' stage"


def test_single_row_inference(model, sample_features):
    """Test inference on a single customer record."""
    single = sample_features.iloc[[0]]
    pred = model.predict(single)
    assert len(pred) == 1
    assert pred[0] in {0, 1, "yes", "no"}


def test_batch_inference_and_probabilities(model, sample_features):
    """Test batch inference and verify probability distributions sum to ~1.0."""
    batch = sample_features.iloc[: min(3, len(sample_features))]
    preds = model.predict(batch)
    assert len(preds) == len(batch)
    for p in preds:
        assert p in {0, 1, "yes", "no"}

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(batch)
        assert probas.shape[0] == len(batch)
        for row in probas:
            assert np.isclose(np.sum(row), 1.0, atol=1e-5)


def test_prediction_determinism(model, sample_features):
    """Ensure identical inputs yield identical outputs (deterministic)."""
    single = sample_features.iloc[[0]]
    pred_1 = model.predict(single)[0]
    pred_2 = model.predict(single)[0]
    assert pred_1 == pred_2