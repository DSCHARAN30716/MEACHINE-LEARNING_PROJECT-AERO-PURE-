# AeroPure Week Programs

Run each week independently from the project root.

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
