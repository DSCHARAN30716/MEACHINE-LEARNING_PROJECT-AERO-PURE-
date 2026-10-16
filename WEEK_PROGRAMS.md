# AeroPure Week Programs

Run each week independently from the project root.

## Project EDA Report

```powershell
python project_eda_report.py
```

Creates project-level EDA reports for missingness, station risk, hourly/monthly patterns, target correlations, and outliers in `outputs/`.

## Week 1: Target Framing and EDA

```powershell
python week_01_target_eda.py
```

Creates Week 1 source-audit, missing-value, category-distribution, statistics, target-summary, and preview reports in `outputs/`.

## Week 2: Ingestion and Preprocessing Pipeline

```powershell
python week_02_ingest_clean_pipeline.py
```

Creates Week 2 feed-status, validation, pipeline-step, feature-audit, and model-ready preview reports in `outputs/`.

## Week 3: Linear Regression Baseline

```powershell
python week_03_linear_regression_baseline.py
```

Creates Week 3 OLS and gradient-descent reports in `outputs/`, then logs the experiment to local MLflow using `mlflow.db`.

## Week 4: Regularized and Logistic Models

```powershell
python week_04_regularized_and_logistic.py
```

Creates Week 4 ridge, lasso, logistic-regression, feature-engineering, coefficient, and prediction reports in `outputs/`, then logs the experiment to local MLflow using `mlflow.db`.

## Week 5: Cross-Validation and Honest Baseline

```powershell
python week_05_cv_honest_baseline.py
```

Creates Week 5 cross-validation, one-standard-error selection, honest regression/logistic metrics, milestone, and prediction reports in `outputs/`, then logs the experiment to local MLflow using `mlflow.db`.

## Week 6: Interpretable Decision Tree

```powershell
python week_06_decision_tree.py
```

Creates Week 6 tree metrics, feature importance, top split rules, full tree rules, and prediction reports in `outputs/`, then logs the experiment to local MLflow using `mlflow.db`.

## Week 7: Random Forest

```powershell
python week_07_random_forest.py
```

Creates Week 7 random-forest metrics, OOB error, impurity importance, permutation importance, grouped importance, unavailable feature-group notes, and prediction reports in `outputs/`, then logs the experiment to local MLflow using `mlflow.db`.

## Week 8: Boosting Champion

```powershell
python week_08_boosting_champion.py
```

Creates Week 8 early-stopped boosting metrics, permutation-importance explanation, station prediction slices, implementation notes, and prediction reports in `outputs/`, then logs the experiment to local MLflow using `mlflow.db`.

## Week 9: Pollution-Regime Clustering

```powershell
python week_09_clustering_segments.py
```

Creates Week 9 KMeans cluster profiles, DBSCAN summaries, PCA summaries, cluster-quality metrics, segment-feature notes, and preview rows in `outputs/`, then logs the experiment to local MLflow using `mlflow.db`.

## Week 10: Rigorous Evaluation

```powershell
python week_10_rigorous_evaluation.py
```

Creates Week 10 nested-CV results, sealed-test metrics, calibration metrics, significance-test output, and champion summary in `outputs/`, then logs the experiment to local MLflow using `mlflow.db`.

## Week 11: Champion Package

```powershell
python week_11_package_champion.py
```

Creates a versioned champion package in `outputs/aeropure_champion_v1/` plus package metrics, feature schema, package manifest, and PSI drift guard reports in `outputs/`, then logs the package to local MLflow using `mlflow.db`.

## Week 12: Public Forecaster, Monitoring, and Retrain

```powershell
python week_12_deploy_monitor_retrain.py
```

Creates Week 12 public forecast output, drift-monitoring status, deployment manifest, and seasonal retrain schedule in `outputs/`.

## Interactive Website Forecasting

```powershell
python app.py
```

Open `http://127.0.0.1:5000/predict` to enter a station-level pollution, weather, traffic, and satellite-smoke scenario. The website loads the Week 11 packaged champion model and uses the shared feature module from the 12-week pipeline to forecast future hazardous-air-day probability and projected AQI outcomes.
