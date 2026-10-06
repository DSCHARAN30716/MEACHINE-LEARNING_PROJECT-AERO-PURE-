import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from app import HAZARDOUS_AQI_THRESHOLD, load_data, numeric_feature_columns, prepare_clean_data


OUTPUT_DIR = "outputs"
EXPERIMENT_NAME = "AeroPure_W4_regularized_and_logistic"
MLFLOW_DB = "mlflow.db"
TEST_FRACTION = 0.2
RIDGE_ALPHA = 10.0
LASSO_ALPHA = 0.01
LOGISTIC_C = 1.0


def save_table(frame, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    frame.to_csv(path, index=False)
    return path


def engineer_history_features(cleaned):
    frame = cleaned.sort_values(["station", "datetime"]).copy()
    base_columns = numeric_feature_columns(frame)
    history_columns = ["AQI Value", "PM2.5", "PM10", "NO2", "O3", "TEMP", "WSPM"]
    history_columns = [column for column in history_columns if column in frame.columns]

    for column in history_columns:
        grouped = frame.groupby("station")[column]
        frame[f"{column}_lag_24h"] = grouped.shift(24)
        frame[f"{column}_rolling_24h_mean"] = grouped.transform(
            lambda values: values.shift(1).rolling(24, min_periods=6).mean()
        )

    frame["tomorrows_hazardous_air_day_flag"] = (
        frame["tomorrows_aqi_value"] >= HAZARDOUS_AQI_THRESHOLD
    ).astype(int)
    frame["hour"] = frame["datetime"].dt.hour
    frame["month"] = frame["datetime"].dt.month
    frame["day_of_week"] = frame["datetime"].dt.dayofweek

    station_dummies = pd.get_dummies(frame["station"], prefix="station", dtype=int)
    frame = pd.concat([frame, station_dummies], axis=1)

    engineered_columns = [
        column for column in frame.columns
        if column.endswith("_lag_24h")
        or column.endswith("_rolling_24h_mean")
    ]
    time_columns = ["hour", "month", "day_of_week"]
    station_columns = list(station_dummies.columns)
    feature_columns = base_columns + engineered_columns + time_columns + station_columns

    model_frame = frame.dropna(
        subset=feature_columns + ["tomorrows_aqi_value", "tomorrows_hazardous_air_day_flag"]
    ).reset_index(drop=True)
    return model_frame, feature_columns, base_columns, engineered_columns, station_columns


def chronological_split(frame, test_fraction):
    ordered = frame.sort_values(["datetime", "station"]).reset_index(drop=True)
    split_index = int(len(ordered) * (1 - test_fraction))
    train = ordered.iloc[:split_index].copy()
    test = ordered.iloc[split_index:].copy()
    return train, test


def regression_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "r2": r2_score(y_true, y_pred),
    }


def classification_metrics(y_true, probabilities, predictions):
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }


def main():
    raw = load_data()
    cleaned, _, _ = prepare_clean_data(raw)
    model_frame, feature_columns, base_columns, engineered_columns, station_columns = (
        engineer_history_features(cleaned)
    )
    train, test = chronological_split(model_frame, TEST_FRACTION)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[feature_columns])
    x_test = scaler.transform(test[feature_columns])
    y_train_regression = train["tomorrows_aqi_value"]
    y_test_regression = test["tomorrows_aqi_value"]
    y_train_classification = train["tomorrows_hazardous_air_day_flag"]
    y_test_classification = test["tomorrows_hazardous_air_day_flag"]

    ridge = Ridge(alpha=RIDGE_ALPHA)
    lasso = Lasso(alpha=LASSO_ALPHA, max_iter=10000)
    logistic = LogisticRegression(C=LOGISTIC_C, max_iter=1000, class_weight="balanced")

    ridge.fit(x_train, y_train_regression)
    lasso.fit(x_train, y_train_regression)
    logistic.fit(x_train, y_train_classification)

    ridge_predictions = ridge.predict(x_test)
    lasso_predictions = lasso.predict(x_test)
    logistic_probabilities = logistic.predict_proba(x_test)[:, 1]
    logistic_predictions = (logistic_probabilities >= 0.5).astype(int)

    ridge_metrics = regression_metrics(y_test_regression, ridge_predictions)
    lasso_metrics = regression_metrics(y_test_regression, lasso_predictions)
    logistic_metrics = classification_metrics(
        y_test_classification,
        logistic_probabilities,
        logistic_predictions,
    )

    regression_metrics_table = pd.DataFrame(
        [
            {"model": "ridge_regression", "alpha": RIDGE_ALPHA, **ridge_metrics},
            {"model": "lasso_regression", "alpha": LASSO_ALPHA, **lasso_metrics},
        ]
    )
    classification_metrics_table = pd.DataFrame(
        [
            {
                "model": "logistic_regression",
                "target": "tomorrows_hazardous_air_day_flag",
                **logistic_metrics,
            }
        ]
    )
    coefficients = pd.DataFrame(
        {
            "feature": feature_columns,
            "ridge_coefficient": ridge.coef_,
            "lasso_coefficient": lasso.coef_,
            "logistic_coefficient": logistic.coef_[0],
        }
    ).sort_values("logistic_coefficient", key=lambda values: values.abs(), ascending=False)
    feature_summary = pd.DataFrame(
        [
            {"feature_group": "sensor_and_weather_current", "count": len(base_columns)},
            {"feature_group": "sensor_and_weather_history", "count": len(engineered_columns)},
            {"feature_group": "time_features", "count": 3},
            {"feature_group": "station_features", "count": len(station_columns)},
            {"feature_group": "traffic_features", "count": 0},
            {"feature_group": "satellite_features", "count": 0},
        ]
    )
    predictions = test[
        [
            "station",
            "datetime",
            "AQI Value",
            "tomorrows_aqi_value",
            "tomorrows_hazardous_air_day_flag",
        ]
    ].copy()
    predictions["ridge_prediction"] = ridge_predictions
    predictions["lasso_prediction"] = lasso_predictions
    predictions["hazard_probability"] = logistic_probabilities
    predictions["hazard_prediction"] = logistic_predictions

    regression_path = save_table(
        regression_metrics_table,
        "w4_ridge_lasso_regression_metrics.csv",
    )
    classification_path = save_table(
        classification_metrics_table,
        "w4_logistic_regression_metrics.csv",
    )
    coefficient_path = save_table(
        coefficients,
        "w4_regularized_logistic_coefficients.csv",
    )
    feature_path = save_table(feature_summary, "w4_feature_engineering_summary.csv")
    prediction_path = save_table(
        predictions.head(500),
        "w4_regularized_logistic_predictions.csv",
    )

    mlflow_tracking_uri = f"sqlite:///{Path(MLFLOW_DB).resolve().as_posix()}"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="w4_ridge_lasso_logistic"):
        mlflow.log_param("test_fraction", TEST_FRACTION)
        mlflow.log_param("ridge_alpha", RIDGE_ALPHA)
        mlflow.log_param("lasso_alpha", LASSO_ALPHA)
        mlflow.log_param("logistic_c", LOGISTIC_C)
        mlflow.log_param("feature_count", len(feature_columns))
        mlflow.log_param("train_rows", len(train))
        mlflow.log_param("test_rows", len(test))
        mlflow.log_param("traffic_features_available", False)
        mlflow.log_param("satellite_features_available", False)

        for name, value in ridge_metrics.items():
            mlflow.log_metric(f"ridge_{name}", value)
        for name, value in lasso_metrics.items():
            mlflow.log_metric(f"lasso_{name}", value)
        for name, value in logistic_metrics.items():
            mlflow.log_metric(f"logistic_{name}", value)

        for path in [
            regression_path,
            classification_path,
            coefficient_path,
            feature_path,
            prediction_path,
        ]:
            mlflow.log_artifact(path)

    print("W4 complete: ridge, lasso, and logistic regression")
    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Feature count: {len(feature_columns):,}")
    print(
        "Ridge: "
        f"MAE={ridge_metrics['mae']:.2f}, "
        f"RMSE={ridge_metrics['rmse']:.2f}, "
        f"R2={ridge_metrics['r2']:.3f}"
    )
    print(
        "Lasso: "
        f"MAE={lasso_metrics['mae']:.2f}, "
        f"RMSE={lasso_metrics['rmse']:.2f}, "
        f"R2={lasso_metrics['r2']:.3f}"
    )
    print(
        "Logistic: "
        f"Accuracy={logistic_metrics['accuracy']:.3f}, "
        f"F1={logistic_metrics['f1']:.3f}, "
        f"ROC_AUC={logistic_metrics['roc_auc']:.3f}"
    )
    print("Output files:")
    for path in [
        regression_path,
        classification_path,
        coefficient_path,
        feature_path,
        prediction_path,
    ]:
        print(f"- {path}")
    print("MLflow experiment:")
    print(f"- {EXPERIMENT_NAME}")
    print(f"- tracking URI: {mlflow_tracking_uri}")


if __name__ == "__main__":
    main()
