import os
from pathlib import Path

import pandas as pd
import mlflow
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from week_04_regularized_and_logistic import chronological_split, engineer_history_features
from app import load_data, prepare_clean_data


OUTPUT_DIR = "outputs"
EXPERIMENT_NAME = "AeroPure_W8_boosting_champion"
MLFLOW_DB = "mlflow.db"
TARGET_COLUMN = "tomorrows_hazardous_air_day_flag"
TEST_FRACTION = 0.2
FIT_SAMPLE_ROWS = 90000
TEST_SAMPLE_ROWS = 60000
PERMUTATION_SAMPLE_ROWS = 8000


def save_table(frame, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    frame.to_csv(path, index=False)
    return path


def metrics(y_true, probabilities, predictions):
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
    model_frame, feature_columns, _, _, _ = engineer_history_features(cleaned)
    train, test = chronological_split(model_frame, TEST_FRACTION)

    x_train = train[feature_columns]
    y_train = train[TARGET_COLUMN]
    eval_test = test.sample(n=min(TEST_SAMPLE_ROWS, len(test)), random_state=42)
    x_test = eval_test[feature_columns]
    y_test = eval_test[TARGET_COLUMN]

    fit_train = train.sample(n=min(FIT_SAMPLE_ROWS, len(train)), random_state=42)
    x_fit = fit_train[feature_columns]
    y_fit = fit_train[TARGET_COLUMN]

    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=150,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=12,
        tol=1e-4,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(x_fit, y_fit)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metric_values = metrics(y_test, probabilities, predictions)

    sample = eval_test.sample(n=min(PERMUTATION_SAMPLE_ROWS, len(eval_test)), random_state=42)
    permutation = permutation_importance(
        model,
        sample[feature_columns],
        sample[TARGET_COLUMN],
        scoring="roc_auc",
        n_repeats=3,
        random_state=42,
        n_jobs=1,
    )
    importance = (
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
    prediction_slices = (
        eval_test.assign(hazard_probability=probabilities)
        .groupby("station")
        .agg(
            rows=("hazard_probability", "size"),
            mean_probability=("hazard_probability", "mean"),
            observed_hazard_rate=(TARGET_COLUMN, "mean"),
        )
        .round(4)
        .reset_index()
        .sort_values("mean_probability", ascending=False)
    )
    predictions_table = eval_test[["station", "datetime", "AQI Value", TARGET_COLUMN]].copy()
    predictions_table["hazard_probability"] = probabilities
    predictions_table["hazard_prediction"] = predictions
    notes = pd.DataFrame(
        [
            {
                "item": "champion_model",
                "detail": "HistGradientBoostingClassifier used as the local champion because xgboost is not installed locally.",
            },
            {
                "item": "explanation_method",
                "detail": "Permutation importance used because shap is not installed locally.",
            },
            {
                "item": "early_stopping_iterations",
                "detail": model.n_iter_,
            },
        ]
    )

    metric_path = save_table(pd.DataFrame([{"model": "gradient_boosting", **metric_values}]), "w8_boosting_metrics.csv")
    importance_path = save_table(importance, "w8_boosting_permutation_importance.csv")
    slice_path = save_table(prediction_slices, "w8_boosting_prediction_slices.csv")
    prediction_path = save_table(predictions_table.head(500), "w8_boosting_predictions.csv")
    notes_path = save_table(notes, "w8_boosting_notes.csv")

    mlflow_tracking_uri = f"sqlite:///{Path(MLFLOW_DB).resolve().as_posix()}"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="w8_boosting_champion"):
        mlflow.log_param("model", "HistGradientBoostingClassifier")
        mlflow.log_param("target", TARGET_COLUMN)
        mlflow.log_param("early_stopping_iterations", model.n_iter_)
        mlflow.log_param("feature_count", len(feature_columns))
        mlflow.log_param("fit_rows", len(fit_train))
        mlflow.log_param("evaluation_rows", len(eval_test))
        for name, value in metric_values.items():
            mlflow.log_metric(name, value)
        for path in [metric_path, importance_path, slice_path, prediction_path, notes_path]:
            mlflow.log_artifact(path)

    print("W8 complete: boosting champion with early stopping")
    print(f"Early-stopped iterations: {model.n_iter_}")
    print(f"ROC_AUC={metric_values['roc_auc']:.3f}, F1={metric_values['f1']:.3f}")
    print("Output files:")
    for path in [metric_path, importance_path, slice_path, prediction_path, notes_path]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
