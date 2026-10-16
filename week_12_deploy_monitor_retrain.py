import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from aeropure_features import TARGET_COLUMN, build_supervised_frame
from app import load_data, prepare_clean_data


OUTPUT_DIR = "outputs"
PACKAGE_DIR = os.path.join(OUTPUT_DIR, "aeropure_champion_v1")
MODEL_PATH = os.path.join(PACKAGE_DIR, "model.joblib")
MANIFEST_PATH = os.path.join(PACKAGE_DIR, "manifest.json")
BASELINE_DRIFT_PATH = os.path.join(PACKAGE_DIR, "psi_drift_guard.csv")
TEST_FRACTION = 0.2
PUBLIC_FORECAST_ROWS = 500


def save_table(frame, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    frame.to_csv(path, index=False)
    return path


def load_manifest():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Run python week_11_package_champion.py before Week 12.")

    raw = load_data()
    cleaned, _, _ = prepare_clean_data(raw)
    model_frame, feature_columns, train, test, _ = build_supervised_frame(cleaned, TEST_FRACTION)
    model = joblib.load(MODEL_PATH)
    manifest = load_manifest()

    latest = (
        model_frame.sort_values(["datetime", "station"])
        .groupby("station", as_index=False)
        .tail(1)
        .sort_values("station")
    )
    forecast = latest[["station", "datetime", "AQI Value", "tomorrows_aqi_value", TARGET_COLUMN]].copy()
    forecast["hazard_probability"] = model.predict_proba(latest[feature_columns])[:, 1]
    forecast["hazard_prediction"] = (forecast["hazard_probability"] >= 0.5).astype(int)
    forecast["public_status"] = np.where(
        forecast["hazard_prediction"] == 1,
        "High-risk air day expected",
        "No high-risk air day expected",
    )
    forecast = forecast.head(PUBLIC_FORECAST_ROWS)

    baseline_drift = pd.read_csv(BASELINE_DRIFT_PATH)
    monitor_summary = pd.DataFrame(
        [
            {
                "monitor": "psi_drift_guard",
                "status": "fail" if (baseline_drift["status"] == "fail").any() else ("warn" if (baseline_drift["status"] == "warn").any() else "pass"),
                "failing_features": int((baseline_drift["status"] == "fail").sum()),
                "warning_features": int((baseline_drift["status"] == "warn").sum()),
                "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            {
                "monitor": "seasonal_concept_shift",
                "status": "scheduled",
                "failing_features": 0,
                "warning_features": 0,
                "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        ]
    )
    deployment_manifest = pd.DataFrame(
        [
            {
                "service_name": "aeropure-public-forecaster",
                "package_name": manifest["package_name"],
                "model_file": MODEL_PATH,
                "public_output": "outputs/w12_public_forecast.csv",
                "monitoring_output": "outputs/w12_drift_monitoring_status.csv",
                "deployment_mode": "public CSV/API-ready Flask artifact",
                "readiness": "ready",
            }
        ]
    )
    retrain_schedule = pd.DataFrame(
        [
            {
                "job": "seasonal_retrain",
                "cadence": "quarterly",
                "trigger": "calendar season boundary or PSI fail threshold",
                "command": "python week_11_package_champion.py && python week_12_deploy_monitor_retrain.py",
                "approval_gate": "compare sealed-test metrics and drift report before replacing champion",
            }
        ]
    )

    forecast_path = save_table(forecast, "w12_public_forecast.csv")
    monitor_path = save_table(monitor_summary, "w12_drift_monitoring_status.csv")
    deployment_path = save_table(deployment_manifest, "w12_deployment_manifest.csv")
    retrain_path = save_table(retrain_schedule, "w12_seasonal_retrain_schedule.csv")

    print("W12 complete: public forecast, monitoring, and seasonal retrain schedule")
    print(f"Forecast rows: {len(forecast)}")
    print(f"Monitoring status: {monitor_summary.loc[0, 'status']}")
    print("Output files:")
    for path in [forecast_path, monitor_path, deployment_path, retrain_path]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
