import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from app import load_data, prepare_clean_data


OUTPUT_DIR = "outputs"
EXPERIMENT_NAME = "AeroPure_W3_linear_baseline"
MLFLOW_DB = "mlflow.db"
TARGET_COLUMN = "tomorrows_aqi_value"
TEST_FRACTION = 0.2
LEARNING_RATE = 0.03
EPOCHS = 800


def save_table(frame, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    frame.to_csv(path, index=False)
    return path


def make_design_matrix(frame):
    feature_columns = [
        column for column in frame.columns
        if column.endswith(" Scaled")
    ]
    x = frame[feature_columns].to_numpy(dtype=float)
    x_with_intercept = np.c_[np.ones(len(x)), x]
    return x_with_intercept, feature_columns


def chronological_split(frame, test_fraction):
    ordered = frame.sort_values(["datetime", "station"]).reset_index(drop=True)
    split_index = int(len(ordered) * (1 - test_fraction))
    train = ordered.iloc[:split_index].copy()
    test = ordered.iloc[split_index:].copy()
    return train, test


def metrics(y_true, y_pred):
    errors = y_true - y_pred
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def fit_ols(x_train, y_train):
    coefficients, _, _, _ = np.linalg.lstsq(x_train, y_train, rcond=None)
    return coefficients


def fit_gradient_descent(x_train, y_train, learning_rate, epochs):
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_scaled = (y_train - y_mean) / y_std

    weights = np.zeros(x_train.shape[1])
    losses = []

    for epoch in range(epochs):
        predictions = x_train @ weights
        errors = predictions - y_scaled
        gradient = (x_train.T @ errors) / len(x_train)
        weights -= learning_rate * gradient

        if epoch == 0 or (epoch + 1) % 50 == 0:
            losses.append(
                {
                    "epoch": epoch + 1,
                    "mse": float(np.mean(errors ** 2)),
                }
            )

    return weights, y_mean, y_std, pd.DataFrame(losses)


def main():
    raw = load_data()
    _, model_ready, _ = prepare_clean_data(raw)
    model_ready = model_ready.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    train, test = chronological_split(model_ready, TEST_FRACTION)
    x_train, feature_columns = make_design_matrix(train)
    x_test, _ = make_design_matrix(test)
    y_train = train[TARGET_COLUMN].to_numpy(dtype=float)
    y_test = test[TARGET_COLUMN].to_numpy(dtype=float)

    ols_coefficients = fit_ols(x_train, y_train)
    ols_predictions = x_test @ ols_coefficients
    ols_metrics = metrics(y_test, ols_predictions)

    gd_weights, y_mean, y_std, gd_loss_history = fit_gradient_descent(
        x_train,
        y_train,
        LEARNING_RATE,
        EPOCHS,
    )
    gd_predictions = (x_test @ gd_weights) * y_std + y_mean
    gd_metrics = metrics(y_test, gd_predictions)

    coefficients = pd.DataFrame(
        {
            "feature": ["intercept"] + feature_columns,
            "ols_coefficient": ols_coefficients,
            "gradient_descent_weight_scaled_target": gd_weights,
        }
    )
    predictions = test[
        ["station", "datetime", "AQI Value", TARGET_COLUMN]
    ].copy()
    predictions["ols_prediction"] = ols_predictions
    predictions["gradient_descent_prediction"] = gd_predictions
    predictions["ols_error"] = predictions[TARGET_COLUMN] - predictions["ols_prediction"]
    predictions["gradient_descent_error"] = (
        predictions[TARGET_COLUMN] - predictions["gradient_descent_prediction"]
    )
    metrics_table = pd.DataFrame(
        [
            {"model": "ols_closed_form", **ols_metrics},
            {"model": "gradient_descent", **gd_metrics},
        ]
    )

    metric_path = save_table(metrics_table, "w3_linear_regression_metrics.csv")
    coefficient_path = save_table(coefficients, "w3_linear_regression_coefficients.csv")
    prediction_path = save_table(
        predictions.head(500),
        "w3_linear_regression_predictions.csv",
    )
    loss_path = save_table(gd_loss_history, "w3_gradient_descent_loss.csv")

    mlflow_tracking_uri = f"sqlite:///{Path(MLFLOW_DB).resolve().as_posix()}"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="w3_ols_and_gradient_descent"):
        mlflow.log_param("target", TARGET_COLUMN)
        mlflow.log_param("test_fraction", TEST_FRACTION)
        mlflow.log_param("features", ",".join(feature_columns))
        mlflow.log_param("gradient_descent_learning_rate", LEARNING_RATE)
        mlflow.log_param("gradient_descent_epochs", EPOCHS)
        mlflow.log_param("train_rows", len(train))
        mlflow.log_param("test_rows", len(test))

        for name, value in ols_metrics.items():
            mlflow.log_metric(f"ols_{name}", value)
        for name, value in gd_metrics.items():
            mlflow.log_metric(f"gradient_descent_{name}", value)

        mlflow.log_artifact(metric_path)
        mlflow.log_artifact(coefficient_path)
        mlflow.log_artifact(prediction_path)
        mlflow.log_artifact(loss_path)

    print("W3 complete: linear-regression baseline")
    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print(
        "OLS metrics: "
        f"MAE={ols_metrics['mae']:.2f}, "
        f"RMSE={ols_metrics['rmse']:.2f}, "
        f"R2={ols_metrics['r2']:.3f}"
    )
    print(
        "Gradient descent metrics: "
        f"MAE={gd_metrics['mae']:.2f}, "
        f"RMSE={gd_metrics['rmse']:.2f}, "
        f"R2={gd_metrics['r2']:.3f}"
    )
    print("Output files:")
    for path in [metric_path, coefficient_path, prediction_path, loss_path]:
        print(f"- {path}")
    print("MLflow experiment:")
    print(f"- {EXPERIMENT_NAME}")
    print(f"- tracking URI: {mlflow_tracking_uri}")


if __name__ == "__main__":
    main()
