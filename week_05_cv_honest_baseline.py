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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from app import load_data, prepare_clean_data
from week_04_regularized_and_logistic import (
    chronological_split,
    engineer_history_features,
)


OUTPUT_DIR = "outputs"
EXPERIMENT_NAME = "AeroPure_W5_cv_honest_baseline"
MLFLOW_DB = "mlflow.db"
TEST_FRACTION = 0.2
CV_SPLITS = 3
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]
LASSO_ALPHAS = [0.001, 0.01, 0.1]
LOGISTIC_CS = [0.1, 1.0, 10.0]


def save_table(frame, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    frame.to_csv(path, index=False)
    return path


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


def cv_regression(train, feature_columns, target_column, model_name, values):
    rows = []
    splitter = TimeSeriesSplit(n_splits=CV_SPLITS)
    x = train[feature_columns].to_numpy()
    y = train[target_column].to_numpy()

    for value in values:
        fold_scores = []
        for fold, (train_index, validation_index) in enumerate(splitter.split(x), start=1):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x[train_index])
            x_validation = scaler.transform(x[validation_index])
            y_train = y[train_index]
            y_validation = y[validation_index]

            if model_name == "ridge_regression":
                model = Ridge(alpha=value)
            else:
                model = Lasso(alpha=value, max_iter=10000)

            model.fit(x_train, y_train)
            predictions = model.predict(x_validation)
            rmse = mean_squared_error(y_validation, predictions) ** 0.5
            fold_scores.append(rmse)
            rows.append(
                {
                    "model": model_name,
                    "parameter": "alpha",
                    "value": value,
                    "fold": fold,
                    "rmse": rmse,
                }
            )

        mean_score = float(np.mean(fold_scores))
        se_score = float(np.std(fold_scores, ddof=1) / np.sqrt(len(fold_scores)))
        rows.append(
            {
                "model": model_name,
                "parameter": "alpha",
                "value": value,
                "fold": "mean",
                "rmse": mean_score,
                "standard_error": se_score,
            }
        )

    return pd.DataFrame(rows)


def cv_logistic(train, feature_columns, target_column, values):
    rows = []
    splitter = TimeSeriesSplit(n_splits=CV_SPLITS)
    x = train[feature_columns].to_numpy()
    y = train[target_column].to_numpy()

    for value in values:
        fold_scores = []
        for fold, (train_index, validation_index) in enumerate(splitter.split(x), start=1):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x[train_index])
            x_validation = scaler.transform(x[validation_index])
            y_train = y[train_index]
            y_validation = y[validation_index]

            model = LogisticRegression(
                C=value,
                max_iter=1000,
                class_weight="balanced",
            )
            model.fit(x_train, y_train)
            probabilities = model.predict_proba(x_validation)[:, 1]
            roc_auc = roc_auc_score(y_validation, probabilities)
            fold_scores.append(roc_auc)
            rows.append(
                {
                    "model": "logistic_regression",
                    "parameter": "C",
                    "value": value,
                    "fold": fold,
                    "roc_auc": roc_auc,
                }
            )

        mean_score = float(np.mean(fold_scores))
        se_score = float(np.std(fold_scores, ddof=1) / np.sqrt(len(fold_scores)))
        rows.append(
            {
                "model": "logistic_regression",
                "parameter": "C",
                "value": value,
                "fold": "mean",
                "roc_auc": mean_score,
                "standard_error": se_score,
            }
        )

    return pd.DataFrame(rows)


def one_standard_error_regression(cv_summary):
    mean_rows = cv_summary[cv_summary["fold"] == "mean"].copy()
    best = mean_rows.loc[mean_rows["rmse"].idxmin()]
    threshold = best["rmse"] + best["standard_error"]
    eligible = mean_rows[mean_rows["rmse"] <= threshold]
    selected = eligible.sort_values("value", ascending=False).iloc[0]
    return selected, best, threshold


def one_standard_error_classification(cv_summary):
    mean_rows = cv_summary[cv_summary["fold"] == "mean"].copy()
    best = mean_rows.loc[mean_rows["roc_auc"].idxmax()]
    threshold = best["roc_auc"] - best["standard_error"]
    eligible = mean_rows[mean_rows["roc_auc"] >= threshold]
    selected = eligible.sort_values("value", ascending=True).iloc[0]
    return selected, best, threshold


def fit_final_models(train, test, feature_columns, selected):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[feature_columns])
    x_test = scaler.transform(test[feature_columns])

    y_train_regression = train["tomorrows_aqi_value"]
    y_test_regression = test["tomorrows_aqi_value"]
    y_train_classification = train["tomorrows_hazardous_air_day_flag"]
    y_test_classification = test["tomorrows_hazardous_air_day_flag"]

    ridge = Ridge(alpha=selected["ridge_alpha"])
    lasso = Lasso(alpha=selected["lasso_alpha"], max_iter=10000)
    logistic = LogisticRegression(
        C=selected["logistic_c"],
        max_iter=1000,
        class_weight="balanced",
    )

    ridge.fit(x_train, y_train_regression)
    lasso.fit(x_train, y_train_regression)
    logistic.fit(x_train, y_train_classification)

    ridge_predictions = ridge.predict(x_test)
    lasso_predictions = lasso.predict(x_test)
    logistic_probabilities = logistic.predict_proba(x_test)[:, 1]
    logistic_predictions = (logistic_probabilities >= 0.5).astype(int)

    regression_rows = [
        {
            "model": "ridge_regression",
            "selected_alpha": selected["ridge_alpha"],
            **regression_metrics(y_test_regression, ridge_predictions),
        },
        {
            "model": "lasso_regression",
            "selected_alpha": selected["lasso_alpha"],
            **regression_metrics(y_test_regression, lasso_predictions),
        },
    ]
    classification_row = {
        "model": "logistic_regression",
        "selected_c": selected["logistic_c"],
        **classification_metrics(
            y_test_classification,
            logistic_probabilities,
            logistic_predictions,
        ),
    }

    predictions = test[
        [
            "station",
            "datetime",
            "tomorrows_aqi_value",
            "tomorrows_hazardous_air_day_flag",
        ]
    ].copy()
    predictions["ridge_prediction"] = ridge_predictions
    predictions["lasso_prediction"] = lasso_predictions
    predictions["hazard_probability"] = logistic_probabilities
    predictions["hazard_prediction"] = logistic_predictions

    return pd.DataFrame(regression_rows), pd.DataFrame([classification_row]), predictions


def main():
    raw = load_data()
    cleaned, _, _ = prepare_clean_data(raw)
    model_frame, feature_columns, _, _, _ = engineer_history_features(cleaned)
    train, test = chronological_split(model_frame, TEST_FRACTION)

    ridge_cv = cv_regression(
        train,
        feature_columns,
        "tomorrows_aqi_value",
        "ridge_regression",
        RIDGE_ALPHAS,
    )
    lasso_cv = cv_regression(
        train,
        feature_columns,
        "tomorrows_aqi_value",
        "lasso_regression",
        LASSO_ALPHAS,
    )
    logistic_cv = cv_logistic(
        train,
        feature_columns,
        "tomorrows_hazardous_air_day_flag",
        LOGISTIC_CS,
    )

    ridge_selected, ridge_best, ridge_threshold = one_standard_error_regression(ridge_cv)
    lasso_selected, lasso_best, lasso_threshold = one_standard_error_regression(lasso_cv)
    logistic_selected, logistic_best, logistic_threshold = one_standard_error_classification(
        logistic_cv
    )
    selected = {
        "ridge_alpha": float(ridge_selected["value"]),
        "lasso_alpha": float(lasso_selected["value"]),
        "logistic_c": float(logistic_selected["value"]),
    }

    final_regression, final_classification, predictions = fit_final_models(
        train,
        test,
        feature_columns,
        selected,
    )
    chosen_regression = final_regression.sort_values("rmse").iloc[0]
    milestone = pd.DataFrame(
        [
            {
                "milestone": "honest_linear_baseline",
                "selected_model": chosen_regression["model"],
                "sealed_test_rmse": chosen_regression["rmse"],
                "sealed_test_mae": chosen_regression["mae"],
                "sealed_test_r2": chosen_regression["r2"],
            },
            {
                "milestone": "honest_logistic_baseline",
                "selected_model": "logistic_regression",
                "sealed_test_roc_auc": final_classification.loc[0, "roc_auc"],
                "sealed_test_f1": final_classification.loc[0, "f1"],
                "sealed_test_accuracy": final_classification.loc[0, "accuracy"],
            },
        ]
    )
    selection_summary = pd.DataFrame(
        [
            {
                "model": "ridge_regression",
                "selected_value": selected["ridge_alpha"],
                "best_value": ridge_best["value"],
                "one_standard_error_threshold": ridge_threshold,
                "metric": "rmse",
            },
            {
                "model": "lasso_regression",
                "selected_value": selected["lasso_alpha"],
                "best_value": lasso_best["value"],
                "one_standard_error_threshold": lasso_threshold,
                "metric": "rmse",
            },
            {
                "model": "logistic_regression",
                "selected_value": selected["logistic_c"],
                "best_value": logistic_best["value"],
                "one_standard_error_threshold": logistic_threshold,
                "metric": "roc_auc",
            },
        ]
    )

    cv_results = pd.concat([ridge_cv, lasso_cv, logistic_cv], ignore_index=True)
    cv_path = save_table(cv_results, "w5_cross_validation_results.csv")
    selection_path = save_table(selection_summary, "w5_one_standard_error_selection.csv")
    regression_path = save_table(final_regression, "w5_honest_regression_metrics.csv")
    classification_path = save_table(
        final_classification,
        "w5_honest_logistic_metrics.csv",
    )
    milestone_path = save_table(milestone, "w5_honest_baseline_milestone.csv")
    prediction_path = save_table(predictions.head(500), "w5_honest_predictions.csv")

    mlflow_tracking_uri = f"sqlite:///{Path(MLFLOW_DB).resolve().as_posix()}"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="w5_cv_one_standard_error"):
        mlflow.log_param("cv_splits", CV_SPLITS)
        mlflow.log_param("test_fraction", TEST_FRACTION)
        mlflow.log_param("feature_count", len(feature_columns))
        mlflow.log_param("ridge_alpha", selected["ridge_alpha"])
        mlflow.log_param("lasso_alpha", selected["lasso_alpha"])
        mlflow.log_param("logistic_c", selected["logistic_c"])
        mlflow.log_param("honest_linear_model", chosen_regression["model"])

        for _, row in final_regression.iterrows():
            for metric in ["mae", "rmse", "r2"]:
                mlflow.log_metric(f"{row['model']}_{metric}", row[metric])
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            mlflow.log_metric(f"logistic_{metric}", final_classification.loc[0, metric])

        for path in [
            cv_path,
            selection_path,
            regression_path,
            classification_path,
            milestone_path,
            prediction_path,
        ]:
            mlflow.log_artifact(path)

    print("W5 complete: cross-validation and honest baseline milestone")
    print(f"Training rows: {len(train):,}")
    print(f"Sealed test rows: {len(test):,}")
    print(f"Selected ridge alpha: {selected['ridge_alpha']}")
    print(f"Selected lasso alpha: {selected['lasso_alpha']}")
    print(f"Selected logistic C: {selected['logistic_c']}")
    print(
        "Honest linear baseline: "
        f"{chosen_regression['model']} "
        f"RMSE={chosen_regression['rmse']:.2f}, "
        f"R2={chosen_regression['r2']:.3f}"
    )
    print(
        "Honest logistic baseline: "
        f"ROC_AUC={final_classification.loc[0, 'roc_auc']:.3f}, "
        f"F1={final_classification.loc[0, 'f1']:.3f}"
    )
    print("Output files:")
    for path in [
        cv_path,
        selection_path,
        regression_path,
        classification_path,
        milestone_path,
        prediction_path,
    ]:
        print(f"- {path}")
    print("MLflow experiment:")
    print(f"- {EXPERIMENT_NAME}")
    print(f"- tracking URI: {mlflow_tracking_uri}")


if __name__ == "__main__":
    main()
