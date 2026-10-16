import os
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from week_04_regularized_and_logistic import engineer_history_features
from app import load_data, prepare_clean_data


OUTPUT_DIR = "outputs"
EXPERIMENT_NAME = "AeroPure_W9_pollution_regime_segments"
MLFLOW_DB = "mlflow.db"
KMEANS_CLUSTERS = 5
DBSCAN_SAMPLE_ROWS = 30000
PCA_SAMPLE_ROWS = 120000


def save_table(frame, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    frame.to_csv(path, index=False)
    return path


def main():
    raw = load_data()
    cleaned, _, _ = prepare_clean_data(raw)
    model_frame, feature_columns, _, _, _ = engineer_history_features(cleaned)
    clustering_features = [
        column for column in [
            "AQI Value", "PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"
        ]
        if column in model_frame.columns
    ]
    clustering_frame = model_frame[clustering_features].copy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x = scaler.fit_transform(imputer.fit_transform(clustering_frame))
    if len(model_frame) > PCA_SAMPLE_ROWS:
        sample_index = model_frame.sample(n=PCA_SAMPLE_ROWS, random_state=42).index
        work_frame = model_frame.loc[sample_index].reset_index(drop=True)
        x_work = x[sample_index]
    else:
        work_frame = model_frame.reset_index(drop=True)
        x_work = x

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(x_work)
    kmeans = KMeans(n_clusters=KMEANS_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(x_work)

    segmented = work_frame[["station", "datetime", "AQI Value", "hazardous_air_day_flag", "tomorrows_aqi_value"]].copy()
    segmented["pca_1"] = components[:, 0]
    segmented["pca_2"] = components[:, 1]
    segmented["pollution_regime_cluster"] = labels

    sample = segmented.sample(n=min(DBSCAN_SAMPLE_ROWS, len(segmented)), random_state=42).copy()
    sample_x = sample[["pca_1", "pca_2"]]
    dbscan = DBSCAN(eps=0.45, min_samples=80)
    sample["dbscan_cluster"] = dbscan.fit_predict(sample_x)

    cluster_profile = (
        segmented.groupby("pollution_regime_cluster")
        .agg(
            rows=("AQI Value", "size"),
            mean_aqi=("AQI Value", "mean"),
            hazardous_rate=("hazardous_air_day_flag", "mean"),
            tomorrow_mean_aqi=("tomorrows_aqi_value", "mean"),
            mean_pca_1=("pca_1", "mean"),
            mean_pca_2=("pca_2", "mean"),
        )
        .round(4)
        .reset_index()
    )
    pca_summary = pd.DataFrame(
        {
            "component": ["pca_1", "pca_2"],
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )
    dbscan_summary = (
        sample["dbscan_cluster"]
        .value_counts()
        .rename_axis("dbscan_cluster")
        .reset_index(name="rows")
        .sort_values("dbscan_cluster")
    )
    quality = pd.DataFrame(
        [
            {
                "method": "kmeans",
                "clusters": KMEANS_CLUSTERS,
                "silhouette_sample": silhouette_score(x_work[: min(50000, len(x_work))], labels[: min(50000, len(labels))]),
            },
            {
                "method": "dbscan",
                "clusters": int(sample.loc[sample["dbscan_cluster"] >= 0, "dbscan_cluster"].nunique()),
                "noise_rows": int((sample["dbscan_cluster"] == -1).sum()),
            },
        ]
    )
    feature_note = pd.DataFrame(
        [
            {
                "new_feature": "pollution_regime_cluster",
                "usage": "Feed this KMeans segment label back into later supervised models as a regime feature.",
            }
        ]
    )

    profile_path = save_table(cluster_profile, "w9_kmeans_cluster_profile.csv")
    pca_path = save_table(pca_summary, "w9_pca_summary.csv")
    dbscan_path = save_table(dbscan_summary, "w9_dbscan_summary.csv")
    quality_path = save_table(quality, "w9_cluster_quality.csv")
    preview_path = save_table(segmented.head(500), "w9_segment_feature_preview.csv")
    note_path = save_table(feature_note, "w9_segment_feature_note.csv")

    mlflow_tracking_uri = f"sqlite:///{Path(MLFLOW_DB).resolve().as_posix()}"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="w9_kmeans_dbscan_pca_segments"):
        mlflow.log_param("kmeans_clusters", KMEANS_CLUSTERS)
        mlflow.log_param("dbscan_sample_rows", len(sample))
        mlflow.log_metric("pca_2d_explained_variance", float(pca.explained_variance_ratio_.sum()))
        for path in [profile_path, pca_path, dbscan_path, quality_path, preview_path, note_path]:
            mlflow.log_artifact(path)

    print("W9 complete: pollution-regime clustering")
    print(f"KMeans clusters: {KMEANS_CLUSTERS}")
    print(f"PCA 2D explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    print("Output files:")
    for path in [profile_path, pca_path, dbscan_path, quality_path, preview_path, note_path]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
