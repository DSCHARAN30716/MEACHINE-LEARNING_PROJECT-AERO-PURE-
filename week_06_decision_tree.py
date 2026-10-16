import os
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier, export_text

from app import load_data, prepare_clean_data
from week_04_regularized_and_logistic import (
    chronological_split,
    engineer_history_features,
)


OUTPUT_DIR = "outputs"
EXPERIMENT_NAME = "AeroPure_W6_decision_tree"
MLFLOW_DB = "mlflow.db"
TEST_FRACTION = 0.2
MAX_DEPTH = 4
MIN_SAMPLES_LEAF = 250
TARGET_COLUMN = "tomorrows_hazardous_air_day_flag"


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


def extract_split_rules(model, feature_columns):
    tree = model.tree_
    rows = []
    for node_id in range(tree.node_count):
        feature_index = tree.feature[node_id]
        if feature_index < 0:
            continue
        rows.append(
            {
                "node": node_id,
                "feature": feature_columns[feature_index],
                "threshold": tree.threshold[node_id],
                "samples": tree.n_node_samples[node_id],
                "impurity": tree.impurity[node_id],
            }
        )
    return pd.DataFrame(rows).sort_values(["node"]).reset_index(drop=True)


def main():
    raw = load_data()
    cleaned, _, _ = prepare_clean_data(raw)
    model_frame, feature_columns, _, _, _ = engineer_history_features(cleaned)
    train, test = chronological_split(model_frame, TEST_FRACTION)

    x_train = train[feature_columns]
    x_test = test[feature_columns]
    y_train = train[TARGET_COLUMN]
    y_test = test[TARGET_COLUMN]

    model = DecisionTreeClassifier(
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = classification_metrics(y_test, probabilities, predictions)

    metric_table = pd.DataFrame([{"model": "decision_tree", **metrics}])
    importance = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    split_rules = extract_split_rules(model, feature_columns)
    tree_text = pd.DataFrame(
        {
            "tree_rules": export_text(
                model,
                feature_names=feature_columns,
                max_depth=MAX_DEPTH,
            ).splitlines()
        }
    )
    predictions_table = test[
        ["station", "datetime", "AQI Value", TARGET_COLUMN]
    ].copy()
    predictions_table["hazard_probability"] = probabilities
    predictions_table["hazard_prediction"] = predictions

    metric_path = save_table(metric_table, "w6_decision_tree_metrics.csv")
    importance_path = save_table(importance, "w6_decision_tree_importance.csv")
    split_path = save_table(split_rules, "w6_decision_tree_splits.csv")
    rule_path = save_table(tree_text, "w6_decision_tree_rules.csv")
    prediction_path = save_table(
        predictions_table.head(500),
        "w6_decision_tree_predictions.csv",
    )

    mlflow_tracking_uri = f"sqlite:///{Path(MLFLOW_DB).resolve().as_posix()}"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="w6_interpretable_decision_tree"):
        mlflow.log_param("target", TARGET_COLUMN)
        mlflow.log_param("max_depth", MAX_DEPTH)
        mlflow.log_param("min_samples_leaf", MIN_SAMPLES_LEAF)
        mlflow.log_param("feature_count", len(feature_columns))
        mlflow.log_param("train_rows", len(train))
        mlflow.log_param("test_rows", len(test))
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        for path in [
            metric_path,
            importance_path,
            split_path,
            rule_path,
            prediction_path,
        ]:
            mlflow.log_artifact(path)

    print("W6 complete: interpretable decision tree")
    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print(
        "Decision tree: "
        f"Accuracy={metrics['accuracy']:.3f}, "
        f"F1={metrics['f1']:.3f}, "
        f"ROC_AUC={metrics['roc_auc']:.3f}"
    )
    print("Top split features:")
    for _, row in split_rules.head(5).iterrows():
        print(f"- {row['feature']} <= {row['threshold']:.3f}")
    print("Output files:")
    for path in [metric_path, importance_path, split_path, rule_path, prediction_path]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
