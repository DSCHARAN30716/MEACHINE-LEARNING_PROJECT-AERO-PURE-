import os
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app import load_data, prepare_clean_data
from week_04_regularized_and_logistic import (
    chronological_split,
    engineer_history_features,
)


OUTPUT_DIR = "outputs"
EXPERIMENT_NAME = "AeroPure_W7_random_forest"
MLFLOW_DB = "mlflow.db"
TEST_FRACTION = 0.2
TARGET_COLUMN = "tomorrows_hazardous_air_day_flag"
N_ESTIMATORS = 80
MAX_DEPTH = 12
MIN_SAMPLES_LEAF = 75
PERMUTATION_SAMPLE_ROWS = 20000


def save_table(frame, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    frame.to_csv(path, index=False)
    return path


def classification_metrics(y_true, probabilities, predictions):
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }


def feature_group(feature):
    if feature.startswith("station_"):
        return "station_history"
    if feature in ["hour", "month", "day_of_week"]:
        return "time"
    if "rolling" in feature or "lag" in feature:
        return "sensor_weather_history"
    if feature in ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]:
        return "weather_current"
    if feature in ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "AQI Value"]:
        return "sensor_current"
    return "other"


def main():
    raw = load_data()
    cleaned, _, _ = prepare_clean_data(raw)
    model_frame, feature_columns, _, _, _ = engineer_history_features(cleaned)
    train, test = chronological_split(model_frame, TEST_FRACTION)

    x_train = train[feature_columns]
    x_test = test[feature_columns]
    y_train = train[TARGET_COLUMN]
    y_test = test[TARGET_COLUMN]

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced_subsample",
        oob_score=True,
        bootstrap=True,
        random_state=42,
        n_jobs=1,
    )
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = classification_metrics(y_test, probabilities, predictions)
    metrics["oob_score"] = model.oob_score_
    metrics["oob_error"] = 1 - model.oob_score_

    sample = test.sample(
        n=min(PERMUTATION_SAMPLE_ROWS, len(test)),
        random_state=42,
    )
    permutation = permutation_importance(
        model,
        sample[feature_columns],
        sample[TARGET_COLUMN],
        n_repeats=3,
        random_state=42,
        n_jobs=1,
        scoring="roc_auc",
    )

    metric_table = pd.DataFrame([{"model": "random_forest", **metrics}])
    impurity_importance = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    permutation_table = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "permutation_importance_mean": permutation.importances_mean,
                "permutation_importance_std": permutation.importances_std,
            }
        )
        .sort_values("permutation_importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    permutation_table["feature_group"] = permutation_table["feature"].map(feature_group)
    group_importance = (
        permutation_table.groupby("feature_group", as_index=False)[
            "permutation_importance_mean"
        ]
        .sum()
        .sort_values("permutation_importance_mean", ascending=False)
    )
    unavailable = pd.DataFrame(
        [
            {"feature_group": "traffic", "status": "not_available_in_repo"},
            {"feature_group": "satellite", "status": "not_available_in_repo"},
        ]
    )
    predictions_table = test[
        ["station", "datetime", "AQI Value", TARGET_COLUMN]
    ].copy()
    predictions_table["hazard_probability"] = probabilities
    predictions_table["hazard_prediction"] = predictions

    metric_path = save_table(metric_table, "w7_random_forest_metrics.csv")
    impurity_path = save_table(impurity_importance, "w7_random_forest_impurity_importance.csv")
    permutation_path = save_table(permutation_table, "w7_random_forest_permutation_importance.csv")
    group_path = save_table(group_importance, "w7_random_forest_group_importance.csv")
    unavailable_path = save_table(unavailable, "w7_unavailable_feature_groups.csv")
    prediction_path = save_table(
        predictions_table.head(500),
        "w7_random_forest_predictions.csv",
    )

    mlflow_tracking_uri = f"sqlite:///{Path(MLFLOW_DB).resolve().as_posix()}"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="w7_random_forest_oob_permutation"):
        mlflow.log_param("target", TARGET_COLUMN)
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("max_depth", MAX_DEPTH)
        mlflow.log_param("min_samples_leaf", MIN_SAMPLES_LEAF)
        mlflow.log_param("permutation_sample_rows", len(sample))
        mlflow.log_param("feature_count", len(feature_columns))
        mlflow.log_param("traffic_features_available", False)
        mlflow.log_param("satellite_features_available", False)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        for path in [
            metric_path,
            impurity_path,
            permutation_path,
            group_path,
            unavailable_path,
            prediction_path,
        ]:
            mlflow.log_artifact(path)

    print("W7 complete: random forest with OOB and permutation importance")
    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print(
        "Random forest: "
        f"Accuracy={metrics['accuracy']:.3f}, "
        f"F1={metrics['f1']:.3f}, "
        f"ROC_AUC={metrics['roc_auc']:.3f}, "
        f"OOB error={metrics['oob_error']:.3f}"
    )
    print("Top permutation features:")
    for _, row in permutation_table.head(5).iterrows():
        print(f"- {row['feature']}: {row['permutation_importance_mean']:.4f}")
    print("Output files:")
    for path in [
        metric_path,
        impurity_path,
        permutation_path,
        group_path,
        unavailable_path,
        prediction_path,
    ]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
