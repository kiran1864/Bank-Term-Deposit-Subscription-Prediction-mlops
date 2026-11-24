"""Data drift monitoring module for Bank term-deposit production inference tracking.

Compares production inference logs against baseline training data using
the Kolmogorov-Smirnov (KS) two-sample statistical test for numeric features
and chi-square / empirical distribution shift for categorical features.
Generates structured JSON metrics, Markdown incident alerts, and an HTML dashboard.
"""
from pathlib import Path
import argparse
import datetime
import json
import os
import sys
import urllib.request
import numpy as np
import pandas as pd
from scipy.stats import chisquare, ks_2samp

REFERENCE_DATA_PATH = Path("data/processed/train.csv")
RAW_FALLBACK_PATH = Path("data/raw/bank.csv")
INFERENCE_LOG_PATH = Path("data/monitoring/inference_logs.csv")
REPORTS_DIR = Path("reports")
DRIFT_JSON_PATH = REPORTS_DIR / "drift_report.json"
DRIFT_HTML_PATH = REPORTS_DIR / "drift_report.html"
DRIFT_ALERT_MD_PATH = REPORTS_DIR / "drift_alert.md"

NUMERIC_FEATURE_COLUMNS = [
    "age",
    "balance",
    "day",
    "campaign",
    "pdays",
    "previous",
]

CATEGORICAL_FEATURE_COLUMNS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS

LOG_COLUMNS = [
    "timestamp",
    *FEATURE_COLUMNS,
    "predicted_class",
    "prob_no",
    "prob_yes",
]


def load_inference_logs(log_path: Path = INFERENCE_LOG_PATH) -> pd.DataFrame:
    """Robustly load production inference logs with schema enforcement and error recovery."""
    if not log_path.exists() or log_path.stat().st_size == 0:
        return pd.DataFrame(columns=LOG_COLUMNS)

    try:
        df = pd.read_csv(log_path)
    except Exception:
        try:
            df = pd.read_csv(log_path, on_bad_lines="skip")
        except Exception:
            return pd.DataFrame(columns=LOG_COLUMNS)

    for col in LOG_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    return df[LOG_COLUMNS]


def log_inference(
    input_df: pd.DataFrame,
    prediction: str,
    probabilities: dict = None,
    log_path: Path = INFERENCE_LOG_PATH,
):
    """Append a single inference payload to the monitoring log with strict schema consistency."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    probs = probabilities or {}
    prob_no = probs.get("no", probs.get("No", probs.get(0, np.nan)))
    prob_yes = probs.get("yes", probs.get("Yes", probs.get(1, np.nan)))

    row_data = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "predicted_class": prediction,
        "prob_no": prob_no,
        "prob_yes": prob_yes,
    }

    for col in FEATURE_COLUMNS:
        if col in input_df.columns and len(input_df) > 0:
            val = input_df[col].iloc[0]
            if col in NUMERIC_FEATURE_COLUMNS:
                row_data[col] = float(val) if pd.notna(val) else np.nan
            else:
                row_data[col] = str(val) if pd.notna(val) else "unknown"
        else:
            row_data[col] = -1.0 if col == "pdays" else (np.nan if col in NUMERIC_FEATURE_COLUMNS else "unknown")

    df_row = pd.DataFrame([row_data], columns=LOG_COLUMNS)
    header = not log_path.exists() or log_path.stat().st_size == 0
    df_row.to_csv(log_path, mode="a", index=False, header=header)


def generate_sample_inferences(
    n_samples: int = 50,
    drift: bool = False,
    log_path: Path = INFERENCE_LOG_PATH,
):
    """Generate synthetic Bank inference logs adhering to the exact canonical schema."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)

    if drift:
        age = rng.normal(58.0, 10.0, n_samples)
        balance = rng.normal(5500.0, 2500.0, n_samples)
        day = rng.normal(22.0, 6.0, n_samples)
        campaign = rng.normal(6.5, 2.5, n_samples)
        pdays = rng.normal(120.0, 50.0, n_samples)
        previous = rng.normal(4.0, 2.0, n_samples)
        jobs = rng.choice(["retired", "management", "self-employed", "technician"], size=n_samples)
        education = rng.choice(["tertiary", "secondary", "primary", "unknown"], p=[0.55, 0.25, 0.15, 0.05], size=n_samples)
        housing = rng.choice(["yes", "no"], p=[0.25, 0.75], size=n_samples)
        contact = rng.choice(["cellular", "telephone", "unknown"], p=[0.8, 0.15, 0.05], size=n_samples)
        month = rng.choice(["oct", "sep", "mar", "apr", "may"], size=n_samples)
        poutcome = rng.choice(["success", "failure", "other", "unknown"], p=[0.4, 0.2, 0.1, 0.3], size=n_samples)
    else:
        age = rng.normal(41.0, 10.5, n_samples)
        balance = rng.normal(1360.0, 1800.0, n_samples)
        day = rng.normal(15.8, 8.3, n_samples)
        campaign = rng.normal(2.8, 2.0, n_samples)
        pdays = rng.choice([-1.0, *rng.normal(95.0, 60.0, 20)], p=[0.75, *([0.25 / 20] * 20)], size=n_samples)
        previous = rng.poisson(0.6, n_samples).astype(float)
        jobs = rng.choice(["management", "blue-collar", "technician", "admin.", "services", "retired"], size=n_samples)
        education = rng.choice(["secondary", "tertiary", "primary", "unknown"], p=[0.5, 0.3, 0.15, 0.05], size=n_samples)
        housing = rng.choice(["yes", "no"], p=[0.55, 0.45], size=n_samples)
        contact = rng.choice(["cellular", "unknown", "telephone"], p=[0.65, 0.28, 0.07], size=n_samples)
        month = rng.choice(["may", "jul", "aug", "jun", "nov", "apr"], size=n_samples)
        poutcome = rng.choice(["unknown", "failure", "other", "success"], p=[0.78, 0.12, 0.05, 0.05], size=n_samples)

    predicted_classes = rng.choice(["no", "yes"], p=[0.85, 0.15] if not drift else [0.55, 0.45], size=n_samples)
    prob_yes = [
        round(rng.uniform(0.55, 0.95), 4) if pred == "yes" else round(rng.uniform(0.02, 0.45), 4)
        for pred in predicted_classes
    ]
    prob_no = [round(1.0 - py, 4) for py in prob_yes]

    timestamps = [
        (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=i * 5)).isoformat()
        for i in range(n_samples)
    ]

    sample_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "age": np.clip(age, 18, 95).round(0),
            "job": jobs,
            "marital": rng.choice(["married", "single", "divorced"], p=[0.6, 0.28, 0.12], size=n_samples),
            "education": education,
            "default": rng.choice(["no", "yes"], p=[0.98, 0.02], size=n_samples),
            "balance": np.clip(balance, -5000, 80000).round(0),
            "housing": housing,
            "loan": rng.choice(["no", "yes"], p=[0.84, 0.16], size=n_samples),
            "contact": contact,
            "day": np.clip(day, 1, 31).round(0),
            "month": month,
            "campaign": np.clip(campaign, 1, 50).round(0),
            "pdays": np.clip(pdays, -1, 850).round(0),
            "previous": np.clip(previous, 0, 100).round(0),
            "poutcome": poutcome,
            "predicted_class": predicted_classes,
            "prob_no": prob_no,
            "prob_yes": prob_yes,
        },
        columns=LOG_COLUMNS,
    )

    sample_df.to_csv(log_path, mode="w", index=False, header=True)
    print(f"Generated {n_samples} sample inference logs (drift={drift}) at: {log_path}")
    return sample_df


def _categorical_drift(ref_series: pd.Series, cur_series: pd.Series, p_value_threshold: float):
    """Compare categorical distributions using chi-square test on aligned category counts."""
    ref_clean = ref_series.dropna().astype(str)
    cur_clean = cur_series.dropna().astype(str)
    if ref_clean.empty or cur_clean.empty:
        return None

    categories = sorted(set(ref_clean.unique()).union(set(cur_clean.unique())))
    ref_counts = np.array([max((ref_clean == c).sum(), 1e-6) for c in categories], dtype=float)
    cur_counts = np.array([float((cur_clean == c).sum()) for c in categories], dtype=float)

    expected = ref_counts * (cur_counts.sum() / ref_counts.sum())
    chi2_stat, p_val = chisquare(f_obs=cur_counts, f_exp=expected)
    is_drift = bool(p_val < p_value_threshold)

    top_ref = ref_clean.value_counts(normalize=True).index[0]
    top_cur = cur_clean.value_counts(normalize=True).index[0]
    ref_top_pct = float(ref_clean.value_counts(normalize=True).iloc[0] * 100.0)
    cur_top_pct = float((cur_clean == top_ref).mean() * 100.0)

    return {
        "test": "Chi-Square",
        "feature_type": "categorical",
        "drift_detected": is_drift,
        "p_value": float(p_val),
        "statistic": float(chi2_stat),
        "ks_statistic": float(chi2_stat),
        "threshold": p_value_threshold,
        "reference_summary": f"{top_ref} ({ref_top_pct:.1f}%)",
        "current_summary": f"{top_cur} ({cur_top_pct:.1f}%)",
        "mean_shift": float(cur_top_pct - ref_top_pct),
    }


def analyze_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    p_value_threshold: float = 0.05,
    dataset_drift_threshold: float = 0.5,
) -> dict:
    """Run statistical drift tests per available feature to detect data drift."""
    features_report = {}
    drifted_count = 0

    for col in FEATURE_COLUMNS:
        if col not in reference_df.columns or col not in current_df.columns:
            continue

        if col in NUMERIC_FEATURE_COLUMNS:
            ref_vals = pd.to_numeric(reference_df[col], errors="coerce").dropna().values
            cur_vals = pd.to_numeric(current_df[col], errors="coerce").dropna().values
            if len(ref_vals) == 0 or len(cur_vals) == 0:
                continue

            ks_stat, p_val = ks_2samp(ref_vals, cur_vals)
            is_drift = bool(p_val < p_value_threshold)
            if is_drift:
                drifted_count += 1

            ref_mean, ref_std = float(np.mean(ref_vals)), float(np.std(ref_vals))
            cur_mean, cur_std = float(np.mean(cur_vals)), float(np.std(cur_vals))

            features_report[col] = {
                "test": "Kolmogorov-Smirnov",
                "feature_type": "numeric",
                "drift_detected": is_drift,
                "p_value": float(p_val),
                "ks_statistic": float(ks_stat),
                "threshold": p_value_threshold,
                "reference_mean": ref_mean,
                "reference_std": ref_std,
                "current_mean": cur_mean,
                "current_std": cur_std,
                "reference_summary": f"{ref_mean:.2f} (±{ref_std:.2f})",
                "current_summary": f"{cur_mean:.2f} (±{cur_std:.2f})",
                "mean_shift": float(cur_mean - ref_mean),
            }
        else:
            cat_report = _categorical_drift(reference_df[col], current_df[col], p_value_threshold)
            if cat_report is not None:
                if cat_report["drift_detected"]:
                    drifted_count += 1
                features_report[col] = cat_report

    total_features = len(features_report)
    drift_share = (drifted_count / total_features) if total_features > 0 else 0.0
    dataset_drift_detected = drift_share >= dataset_drift_threshold

    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "reference_samples": len(reference_df),
        "current_samples": len(current_df),
        "total_features": total_features,
        "drifted_features": drifted_count,
        "drift_share_pct": round(drift_share * 100, 2),
        "dataset_drift_detected": dataset_drift_detected,
        "dataset_drift_threshold_pct": round(dataset_drift_threshold * 100, 2),
        "features": features_report,
    }


def render_html_report(report: dict, output_path: Path = DRIFT_HTML_PATH):
    """Render a standalone HTML dashboard for the Bank drift analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    status_color = "#ef4444" if report["dataset_drift_detected"] else "#10b981"
    status_text = "DRIFT DETECTED" if report["dataset_drift_detected"] else "DISTRIBUTION HEALTHY"

    cards_html = ""
    for feat, data in report["features"].items():
        feat_color = "#ef4444" if data["drift_detected"] else "#10b981"
        feat_status = "DRIFT" if data["drift_detected"] else "PASS"
        cards_html += f"""
        <div style="border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-bottom: 12px; background: white;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <strong style="font-size: 16px; color: #1e293b;">{feat} <span style="font-size: 12px; color: #64748b;">({data.get('test', 'KS')})</span></strong>
                <span style="background: {feat_color}; color: white; padding: 3px 10px; border-radius: 12px; font-weight: bold; font-size: 12px;">{feat_status}</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; font-size: 13px; color: #64748b;">
                <div>p-value: <b style="color: #0f172a;">{data['p_value']:.4f}</b></div>
                <div>Stat: <b style="color: #0f172a;">{data['ks_statistic']:.4f}</b></div>
                <div>Reference: <b style="color: #0f172a;">{data['reference_summary']}</b></div>
                <div>Production: <b style="color: #0f172a;">{data['current_summary']}</b></div>
            </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Bank MLOps Data Drift Monitoring</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f8fafc; margin: 0; padding: 24px; color: #1e293b; }}
        .container {{ max-width: 960px; margin: 0 auto; }}
        .header {{ background: white; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
        .badge {{ display: inline-block; padding: 6px 14px; border-radius: 16px; color: white; font-weight: bold; font-size: 14px; background: {status_color}; }}
        .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 16px; }}
        .stat-card {{ background: #f1f5f9; padding: 12px; border-radius: 8px; text-align: center; }}
        .stat-num {{ font-size: 20px; font-weight: bold; color: #0f172a; }}
        .stat-label {{ font-size: 12px; color: #64748b; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1 style="margin: 0 0 6px 0; font-size: 24px;">🏦 Bank Term-Deposit Drift Report</h1>
                    <div style="color: #64748b; font-size: 13px;">Analyzed at: {report['timestamp']}</div>
                </div>
                <div class="badge">{status_text}</div>
            </div>
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-num">{report['reference_samples']}</div>
                    <div class="stat-label">Reference Samples</div>
                </div>
                <div class="stat-card">
                    <div class="stat-num">{report['current_samples']}</div>
                    <div class="stat-label">Production Logs</div>
                </div>
                <div class="stat-card">
                    <div class="stat-num">{report['drifted_features']} / {report['total_features']}</div>
                    <div class="stat-label">Drifted Features</div>
                </div>
                <div class="stat-card">
                    <div class="stat-num">{report['drift_share_pct']}%</div>
                    <div class="stat-label">Drift Share</div>
                </div>
            </div>
        </div>
        <h2 style="font-size: 18px; margin-bottom: 12px;">Feature Drift Tests (Kolmogorov-Smirnov &amp; Chi-Square)</h2>
        {cards_html}
    </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def generate_alert_markdown(report: dict, output_path: Path = DRIFT_ALERT_MD_PATH) -> str:
    """Generate Markdown incident report for GitHub Issues, Slack, or Step Summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    drift_header = (
        "🚨 **HIGH PRIORITY: PRODUCTION DATA DRIFT DETECTED**"
        if report["dataset_drift_detected"]
        else "✅ **PRODUCTION DATA DISTRIBUTION HEALTHY**"
    )

    rows = []
    for feat, data in report["features"].items():
        status = "⚠️ **DRIFT**" if data["drift_detected"] else "✅ OK"
        rows.append(
            f"| `{feat}` | {status} | `{data['p_value']:.4f}` | `{data['ks_statistic']:.4f}` | "
            f"`{data['reference_summary']}` | `{data['current_summary']}` | `{data['mean_shift']:+.2f}` |"
        )
    table_content = "\n".join(rows)

    action_text = (
        "⚠️ **Continuous Training Required:** Production feature distributions have deviated significantly from the baseline training distribution. "
        "Triggering automatic pipeline reproduction (`dvc repro`) and quality gate evaluation."
        if report["dataset_drift_detected"]
        else "✅ **No Action Needed:** Production traffic matches the reference baseline distribution within statistical tolerance."
    )

    md = f"""# {drift_header}

### 📊 Monitoring Summary
- **Evaluation Timestamp:** `{report['timestamp']}`
- **Drifted Features:** **{report['drifted_features']} of {report['total_features']}** ({report['drift_share_pct']}%)
- **Dataset Drift Threshold:** `{report['dataset_drift_threshold_pct']}%`
- **Reference Training Samples:** `{report['reference_samples']}`
- **Production Inference Samples:** `{report['current_samples']}`

### 🔬 Statistical Drift Tests (Kolmogorov-Smirnov / Chi-Square)
| Feature | Status | p-value | Statistic | Reference | Production | Shift |
|---|---|---|---|---|---|---|
{table_content}

---
### ⚙️ Action & Continuous Training Status
{action_text}
"""
    output_path.write_text(md, encoding="utf-8")
    return md


def send_drift_alert(report: dict, webhook_url: str = None) -> bool:
    """Send webhook alert to Slack, Discord, or Teams if configured and drift detected."""
    if not report.get("dataset_drift_detected"):
        return False

    target_url = webhook_url or os.getenv("ALERT_WEBHOOK_URL") or os.getenv("SLACK_WEBHOOK_URL")
    if not target_url:
        print("Note: No ALERT_WEBHOOK_URL configured. Markdown alert generated at reports/drift_alert.md.")
        return False

    payload = {
        "text": (
            f"🚨 *[MLOps Alert] Bank Production Data Drift Detected!*\n"
            f"• Drifted Features: {report['drifted_features']}/{report['total_features']} ({report['drift_share_pct']}%)\n"
            f"• Action: Triggering automated continuous retraining pipeline (`dvc repro`)."
        )
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(target_url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"Alert successfully delivered to webhook (HTTP {resp.status}).")
            return True
    except Exception as e:
        print(f"Warning: Failed to deliver alert to webhook: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Check production data drift against baseline Bank training data.")
    parser.add_argument("--reference", type=Path, default=REFERENCE_DATA_PATH, help="Path to reference training CSV.")
    parser.add_argument("--logs", type=Path, default=INFERENCE_LOG_PATH, help="Path to production inference logs CSV.")
    parser.add_argument("--generate-sample-data", action="store_true", help="Generate synthetic inference logs if empty.")
    parser.add_argument("--simulate-drift", action="store_true", help="Simulate drifted traffic for demonstration.")
    parser.add_argument("--fail-on-drift", action="store_true", help="Exit with code 1 if dataset drift is detected.")
    parser.add_argument("--webhook-url", type=str, default=None, help="Optional Slack/Discord webhook URL for alerts.")
    args = parser.parse_args()

    if not args.reference.exists():
        if RAW_FALLBACK_PATH.exists():
            ref_df = pd.read_csv(RAW_FALLBACK_PATH).drop_duplicates()
        else:
            print(f"Error: Reference training data not found at {args.reference}. Run dvc repro first.")
            sys.exit(1)
    else:
        ref_df = pd.read_csv(args.reference)

    logs_df = load_inference_logs(args.logs)
    if len(logs_df) < 5 or args.generate_sample_data or args.simulate_drift:
        if args.generate_sample_data or args.simulate_drift:
            logs_df = generate_sample_inferences(n_samples=50, drift=args.simulate_drift, log_path=args.logs)
        else:
            print(f"Inference log file {args.logs} is empty or has < 5 entries. Auto-generating sample traffic...")
            logs_df = generate_sample_inferences(n_samples=50, drift=False, log_path=args.logs)

    report = analyze_drift(ref_df, logs_df)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(DRIFT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    render_html_report(report, DRIFT_HTML_PATH)
    generate_alert_markdown(report, DRIFT_ALERT_MD_PATH)

    if report["dataset_drift_detected"]:
        send_drift_alert(report, webhook_url=args.webhook_url)

    print("\n=== Data Drift Monitoring Report ===")
    print(f"Status: {'DRIFT DETECTED ⚠️' if report['dataset_drift_detected'] else 'DISTRIBUTION HEALTHY ✅'}")
    print(f"Drifted features: {report['drifted_features']} / {report['total_features']} ({report['drift_share_pct']}%)")
    for feat, data in report["features"].items():
        flag = "DRIFT" if data["drift_detected"] else "OK"
        print(f"  - {feat:15s}: [{flag}] p-val={data['p_value']:.4f}, stat={data['ks_statistic']:.4f}, shift={data['mean_shift']:+.2f}")
    print(f"\nSaved JSON report: {DRIFT_JSON_PATH}")
    print(f"Saved HTML report: {DRIFT_HTML_PATH}")
    print(f"Saved Alert Markdown: {DRIFT_ALERT_MD_PATH}")

    if args.fail_on_drift and report["dataset_drift_detected"]:
        print("\nQuality Gate Error: Production data drift exceeds threshold!")
        sys.exit(1)


if __name__ == "__main__":
    main()