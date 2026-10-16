import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from week_04_regularized_and_logistic import chronological_split, engineer_history_features
from app import load_data, prepare_clean_data


OUTPUT_DIR = "outputs"
EXPERIMENT_NAME = "AeroPure_W10_rigorous_evaluation"
MLFLOW_DB = "mlflow.db"
TARGET_COLUMN = "tomorrows_hazardous_air_day_flag"
TEST_FRACTION = 0.2
TRAIN_SAMPLE_ROWS = 100000
TEST_SAMPLE_ROWS = 60000


def save_table(frame, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    frame.to_csv(path, index=False)
    return path


def metric_row(name, y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
        "brier_score": brier_score_loss(y_true, probabilities),
    }


def main():
    raw = load_data()
    cleaned, _, _ = prepare_clean_data(raw)
    model_frame, feature_columns, _, _, _ = engineer_history_features(cleaned)
    train, test = chronological_split(model_frame, TEST_FRACTION)
    fit_train = train.sample(n=min(TRAIN_SAMPLE_ROWS, len(train)), random_state=42)
    eval_test = test.sample(n=min(TEST_SAMPLE_ROWS, len(test)), random_state=42)
    x_train = fit_train[feature_columns]
    y_train = fit_train[TARGET_COLUMN]
    x_test = eval_test[feature_columns]
    y_test = eval_test[TARGET_COLUMN]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    nested_rows = []
    splitter = TimeSeriesSplit(n_splits=3)
    candidate_depths = [6, 10]
    sample_train = fit_train.sort_values(["datetime", "station"]).reset_index(drop=True)
    sample_x = sample_train[feature_columns]
    sample_y = sample_train[TARGET_COLUMN]
    for depth in candidate_depths:
        scores = []
        for fold, (fit_idx, val_idx) in enumerate(splitter.split(sample_x), start=1):
            model = RandomForestClassifier(
                n_estimators=40,
                max_depth=depth,
                min_samples_leaf=100,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=1,
            )
            model.fit(sample_x.iloc[fit_idx], sample_y.iloc[fit_idx])
            probs = model.predict_proba(sample_x.iloc[val_idx])[:, 1]
            score = roc_auc_score(sample_y.iloc[val_idx], probs)
            scores.append(score)
            nested_rows.append({"model": "random_forest", "max_depth": depth, "fold": fold, "roc_auc": score})
        nested_rows.append({"model": "random_forest", "max_depth": depth, "fold": "mean", "roc_auc": float(np.mean(scores))})
    nested_cv = pd.DataFrame(nested_rows)
    best_depth = int(nested_cv[nested_cv["fold"] == "mean"].sort_values("roc_auc", ascending=False).iloc[0]["max_depth"])

    logistic = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced")
    logistic.fit(x_train_scaled, y_train)
    logistic_probs = logistic.predict_proba(x_test_scaled)[:, 1]

    forest = RandomForestClassifier(
        n_estimators=80,
        max_depth=best_depth,
        min_samples_leaf=75,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=1,
    )
    forest.fit(x_train, y_train)
    forest_probs = forest.predict_proba(x_test)[:, 1]

    boosting = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=200,
        max_leaf_nodes=31,
        early_stopping=True,
        class_weight="balanced",
        random_state=42,
    )
    boosting.fit(x_train, y_train)
    boosting_probs = boosting.predict_proba(x_test)[:, 1]

    calibrated = CalibratedClassifierCV(
        estimator=LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced"),
        method="sigmoid",
        cv=3,
    )
    calibrated.fit(x_train_scaled, y_train)
    calibrated_probs = calibrated.predict_proba(x_test_scaled)[:, 1]

    metrics = pd.DataFrame(
        [
            metric_row("logistic_baseline", y_test, logistic_probs),
            metric_row("calibrated_logistic", y_test, calibrated_probs),
            metric_row("random_forest", y_test, forest_probs),
            metric_row("hist_gradient_boosting", y_test, boosting_probs),
        ]
    )
    calibration = metrics[["model", "brier_score"]].copy()

    forest_pred = (forest_probs >= 0.5).astype(int)
    logistic_pred = (logistic_probs >= 0.5).astype(int)
    forest_correct = forest_pred == y_test.to_numpy()
    logistic_correct = logistic_pred == y_test.to_numpy()
    forest_only = int((forest_correct & ~logistic_correct).sum())
    logistic_only = int((logistic_correct & ~forest_correct).sum())
    significance = pd.DataFrame(
        [
            {
                "test": "mcnemar_exact_binomial",
                "model_a": "random_forest",
                "model_b": "logistic_baseline",
                "model_a_only_correct": forest_only,
                "model_b_only_correct": logistic_only,
                "p_value": binomtest(forest_only, forest_only + logistic_only, 0.5).pvalue,
            }
        ]
    )
    champion = metrics.sort_values(["roc_auc", "f1"], ascending=False).iloc[0]
    champion_table = pd.DataFrame([champion])

    nested_path = save_table(nested_cv, "w10_nested_cv_results.csv")
    metrics_path = save_table(metrics, "w10_sealed_test_metrics.csv")
    calibration_path = save_table(calibration, "w10_calibration_metrics.csv")
    significance_path = save_table(significance, "w10_significance_test.csv")
    champion_path = save_table(champion_table, "w10_champion_summary.csv")

    mlflow_tracking_uri = f"sqlite:///{Path(MLFLOW_DB).resolve().as_posix()}"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="w10_rigorous_evaluation"):
        mlflow.log_param("best_rf_depth_from_nested_cv", best_depth)
        mlflow.log_param("test_fraction", TEST_FRACTION)
        mlflow.log_param("champion_model", champion["model"])
        mlflow.log_param("fit_rows", len(fit_train))
        mlflow.log_param("evaluation_rows", len(eval_test))
        for metric in ["roc_auc", "f1", "balanced_accuracy", "average_precision", "brier_score"]:
            mlflow.log_metric(f"champion_{metric}", champion[metric])
        for path in [nested_path, metrics_path, calibration_path, significance_path, champion_path]:
            mlflow.log_artifact(path)

    print("W10 complete: rigorous evaluation")
    print(f"Champion: {champion['model']} ROC_AUC={champion['roc_auc']:.3f}, F1={champion['f1']:.3f}")
    print(f"Significance p-value: {significance.loc[0, 'p_value']:.6f}")
    print("Output files:")
    for path in [nested_path, metrics_path, calibration_path, significance_path, champion_path]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
