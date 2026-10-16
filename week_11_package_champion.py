import json
import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

from aeropure_features import TARGET_COLUMN, build_supervised_frame
from app import load_data, prepare_clean_data


OUTPUT_DIR = "outputs"
PACKAGE_DIR = os.path.join(OUTPUT_DIR, "aeropure_champion_v1")
EXPERIMENT_NAME = "AeroPure_W11_champion_package"
MLFLOW_DB = "mlflow.db"
TEST_FRACTION = 0.2
FIT_SAMPLE_ROWS = 120000
EVAL_SAMPLE_ROWS = 60000
PSI_BINS = 10
PSI_WARN_THRESHOLD = 0.1
PSI_FAIL_THRESHOLD = 0.25


def ensure_package_dir():
    os.makedirs(PACKAGE_DIR, exist_ok=True)


def save_table(frame, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    frame.to_csv(path, index=False)
    return path


def save_package_table(frame, filename):
    ensure_package_dir()
    path = os.path.join(PACKAGE_DIR, filename)
    frame.to_csv(path, index=False)
    return path


def psi(expected, actual, bins=PSI_BINS):
    expected = pd.Series(expected).replace([np.inf, -np.inf], np.nan).dropna()
    actual = pd.Series(actual).replace([np.inf, -np.inf], np.nan).dropna()
    if expected.empty or actual.empty:
        return np.nan

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(expected.quantile(quantiles).to_numpy())
    if len(edges) < 3:
        edges = np.linspace(expected.min(), expected.max() + 1e-6, bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf

    expected_counts = pd.cut(expected, edges, include_lowest=True).value_counts(sort=False)
    actual_counts = pd.cut(actual, edges, include_lowest=True).value_counts(sort=False)
    expected_pct = expected_counts / max(expected_counts.sum(), 1)
    actual_pct = actual_counts / max(actual_counts.sum(), 1)
    expected_pct = expected_pct.replace(0, 1e-6)
    actual_pct = actual_pct.replace(0, 1e-6)
    return float(((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)).sum())


def drift_status(value):
    if pd.isna(value):
        return "unknown"
    if value >= PSI_FAIL_THRESHOLD:
        return "fail"
    if value >= PSI_WARN_THRESHOLD:
        return "warn"
    return "pass"


def metric_row(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "model": "hist_gradient_boosting_champion_v1",
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "brier_score": brier_score_loss(y_true, probabilities),
    }


def main():
    raw = load_data()
    cleaned, _, _ = prepare_clean_data(raw)
    _, feature_columns, train, test, metadata = build_supervised_frame(cleaned, TEST_FRACTION)

    fit_train = train.sample(n=min(FIT_SAMPLE_ROWS, len(train)), random_state=42)
    eval_test = test.sample(n=min(EVAL_SAMPLE_ROWS, len(test)), random_state=42)

    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=220,
        max_leaf_nodes=31,
        min_samples_leaf=120,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=12,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(fit_train[feature_columns], fit_train[TARGET_COLUMN])

    probabilities = model.predict_proba(eval_test[feature_columns])[:, 1]
    metrics = pd.DataFrame([metric_row(eval_test[TARGET_COLUMN], probabilities)])

    schema = pd.DataFrame(
        [
            {
                "feature": column,
                "dtype": str(train[column].dtype),
                "train_missing_rate": round(float(train[column].isna().mean()), 6),
            }
            for column in feature_columns
        ]
    )
    psi_rows = []
    for column in feature_columns:
        value = psi(train[column], test[column])
        psi_rows.append(
            {
                "feature": column,
                "psi": value,
                "status": drift_status(value),
                "warn_threshold": PSI_WARN_THRESHOLD,
                "fail_threshold": PSI_FAIL_THRESHOLD,
            }
        )
    drift_guard = pd.DataFrame(psi_rows).sort_values("psi", ascending=False)
    package_manifest = {
        "package_name": "aeropure_champion_v1",
        "model_file": "model.joblib",
        "feature_module": "aeropure_features.py",
        "feature_schema": "feature_schema.csv",
        "drift_guard": "psi_drift_guard.csv",
        "target": TARGET_COLUMN,
        "feature_count": len(feature_columns),
        "fit_rows": len(fit_train),
        "evaluation_rows": len(eval_test),
        "early_stopping_iterations": int(model.n_iter_),
        "psi_warn_threshold": PSI_WARN_THRESHOLD,
        "psi_fail_threshold": PSI_FAIL_THRESHOLD,
    }

    ensure_package_dir()
    model_path = os.path.join(PACKAGE_DIR, "model.joblib")
    manifest_path = os.path.join(PACKAGE_DIR, "manifest.json")
    joblib.dump(model, model_path)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(package_manifest, handle, indent=2)

    schema_path = save_package_table(schema, "feature_schema.csv")
    drift_package_path = save_package_table(drift_guard, "psi_drift_guard.csv")
    metrics_path = save_table(metrics, "w11_champion_package_metrics.csv")
    schema_output_path = save_table(schema, "w11_feature_schema.csv")
    drift_output_path = save_table(drift_guard, "w11_psi_drift_guard.csv")
    manifest_output_path = save_table(pd.DataFrame([package_manifest]), "w11_package_manifest.csv")

    mlflow_tracking_uri = f"sqlite:///{Path(MLFLOW_DB).resolve().as_posix()}"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="w11_package_champion"):
        mlflow.log_param("package_name", package_manifest["package_name"])
        mlflow.log_param("feature_count", len(feature_columns))
        mlflow.log_param("early_stopping_iterations", model.n_iter_)
        for name, value in metrics.iloc[0].items():
            if name != "model":
                mlflow.log_metric(name, float(value))
        for path in [model_path, manifest_path, schema_path, drift_package_path, metrics_path, schema_output_path, drift_output_path, manifest_output_path]:
            mlflow.log_artifact(path)

    print("W11 complete: packaged champion artifact")
    print(f"Package directory: {PACKAGE_DIR}")
    print(f"ROC_AUC={metrics.loc[0, 'roc_auc']:.3f}, F1={metrics.loc[0, 'f1']:.3f}")
    print("Output files:")
    for path in [metrics_path, schema_output_path, drift_output_path, manifest_output_path]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
