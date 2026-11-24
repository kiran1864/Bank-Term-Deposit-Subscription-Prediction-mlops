from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.monitor_drift import (
    FEATURE_COLUMNS,
    analyze_drift,
    generate_alert_markdown,
    log_inference,
    send_drift_alert,
)


def test_log_inference_writes_record(tmp_path):
    """Verify that logging a deposit prediction appends a structured record."""
    test_log = tmp_path / "test_logs.csv"
    values = []
    for col in FEATURE_COLUMNS:
        if col in {"age", "balance", "day", "duration", "campaign", "pdays", "previous"}:
            values.append(35 if col != "pdays" else -1)
        elif col == "job":
            values.append("technician")
        elif col == "marital":
            values.append("single")
        elif col == "education":
            values.append("tertiary")
        elif col in {"default", "housing", "loan"}:
            values.append("no")
        elif col == "contact":
            values.append("cellular")
        elif col == "month":
            values.append("aug")
        elif col == "poutcome":
            values.append("unknown")
        else:
            values.append("unknown")

    input_sample = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    log_inference(input_sample, "yes", {"no": 0.15, "yes": 0.85}, log_path=test_log)

    assert test_log.exists()
    logged_df = pd.read_csv(test_log)
    assert len(logged_df) == 1
    assert logged_df.iloc[0]["predicted_class"] == "yes"
    assert "timestamp" in logged_df.columns
    assert logged_df.iloc[0]["prob_yes"] == 0.85


def test_analyze_drift_no_drift_when_identical():
    """Verify that identical numeric distributions report no drift."""
    rng = np.random.RandomState(42)
    numeric_cols = [c for c in FEATURE_COLUMNS if c in {"age", "balance", "day", "campaign", "pdays", "previous"}]
    data = {col: rng.normal(30.0, 5.0, 100) for col in numeric_cols}
    ref_df = pd.DataFrame(data)
    curr_df = pd.DataFrame(data)

    report = analyze_drift(ref_df, curr_df)
    assert report["dataset_drift_detected"] is False
    assert report["drifted_features"] == 0
    assert report["drift_share_pct"] == 0.0


def test_analyze_drift_detects_distribution_shift():
    """Verify that significant numeric shifts are detected by the KS test."""
    rng = np.random.RandomState(42)
    numeric_cols = [c for c in FEATURE_COLUMNS if c in {"age", "balance", "day", "campaign", "pdays", "previous"}]
    ref_data = {col: rng.normal(30.0, 2.0, 150) for col in numeric_cols}
    curr_data = {col: rng.normal(80.0, 2.0, 150) for col in numeric_cols}

    ref_df = pd.DataFrame(ref_data)
    curr_df = pd.DataFrame(curr_data)

    report = analyze_drift(ref_df, curr_df)
    assert report["dataset_drift_detected"] is True
    assert report["drifted_features"] == len(numeric_cols)


def test_generate_alert_markdown_creates_valid_report(tmp_path):
    """Verify that alert markdown is written with expected sections."""
    alert_path = tmp_path / "test_alert.md"
    rng = np.random.RandomState(42)
    numeric_cols = [c for c in FEATURE_COLUMNS if c in {"age", "balance", "day", "campaign", "pdays", "previous"}]
    ref_df = pd.DataFrame({col: rng.normal(30.0, 2.0, 100) for col in numeric_cols})
    curr_df = pd.DataFrame({col: rng.normal(75.0, 2.0, 100) for col in numeric_cols})

    report = analyze_drift(ref_df, curr_df)
    md_text = generate_alert_markdown(report, output_path=alert_path)

    assert alert_path.exists()
    assert "DATA DRIFT DETECTED" in md_text
    assert "Continuous Training Required" in md_text
    assert "Kolmogorov-Smirnov" in md_text


def test_send_drift_alert_graceful_without_webhook():
    """Verify send_drift_alert handles missing webhook without crashing."""
    report = {
        "dataset_drift_detected": True,
        "drifted_features": 4,
        "total_features": 4,
        "drift_share_pct": 100.0,
    }
    result = send_drift_alert(report, webhook_url=None)
    assert result is False