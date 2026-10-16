from flask import Flask, render_template, request
from functools import lru_cache
from glob import glob
import joblib
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

DATA_PATH = "air_quality.csv"
PRSA_PATH = os.path.join(
    "Dataset",
    "PRSA_Data_20130301-20170228",
    "*.csv",
)
HAZARDOUS_AQI_THRESHOLD = 151
TARGET_COLUMN = "tomorrows_hazardous_air_day_flag"
EDA_CHART_DIR = os.path.join("static", "charts")
CHAMPION_MODEL_PATH = os.path.join("outputs", "aeropure_champion_v1", "model.joblib")
CHAMPION_SCHEMA_PATH = os.path.join("outputs", "aeropure_champion_v1", "feature_schema.csv")
WEEK_OUTPUTS = {
    "W1": ["w1_target_summary.csv", "w1_numeric_statistics.csv"],
    "W2": ["w2_validation_report.csv", "w2_feature_audit.csv"],
    "W3": ["w3_linear_regression_metrics.csv"],
    "W4": ["w4_ridge_lasso_regression_metrics.csv", "w4_logistic_regression_metrics.csv"],
    "W5": ["w5_honest_baseline_milestone.csv"],
    "W6": ["w6_decision_tree_metrics.csv"],
    "W7": ["w7_random_forest_metrics.csv"],
    "W8": ["w8_boosting_metrics.csv"],
    "W9": ["w9_cluster_quality.csv"],
    "W10": ["w10_champion_summary.csv"],
    "W11": ["w11_package_manifest.csv"],
    "W12": ["w12_public_forecast.csv", "w12_drift_monitoring_status.csv"],
}


def aqi_level(aqi):
    if aqi <= 50:
        return "good"
    if aqi <= 100:
        return "moderate"
    if aqi <= 150:
        return "sensitive"
    if aqi <= 200:
        return "unhealthy"
    if aqi <= 300:
        return "very-unhealthy"
    return "hazardous"

def pm25_to_aqi(pm25):
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    values = pd.Series(pm25, dtype="float64")
    aqi = pd.Series(np.nan, index=values.index)

    for conc_low, conc_high, aqi_low, aqi_high in breakpoints:
        mask = values.between(conc_low, conc_high, inclusive="both")
        aqi.loc[mask] = (
            (aqi_high - aqi_low)
            / (conc_high - conc_low)
            * (values.loc[mask] - conc_low)
            + aqi_low
        )

    return aqi.clip(0, 500).round()


def aqi_category(aqi):
    bins = [-1, 50, 100, 150, 200, 300, 500]
    labels = [
        "Good",
        "Moderate",
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous",
    ]
    return pd.cut(aqi, bins=bins, labels=labels).astype(str)


def load_legacy_sensor_data():
    raw = pd.read_csv(DATA_PATH)
    raw["datetime"] = pd.NaT
    raw["station"] = raw["City"]
    raw["PM2.5"] = raw["PM2.5 AQI Value"]
    raw["PM10"] = np.nan
    raw["SO2"] = np.nan
    raw["NO2"] = raw["NO2 AQI Value"]
    raw["CO"] = raw["CO AQI Value"]
    raw["O3"] = raw["Ozone AQI Value"]
    raw["TEMP"] = np.nan
    raw["PRES"] = np.nan
    raw["DEWP"] = np.nan
    raw["RAIN"] = np.nan
    raw["WSPM"] = np.nan
    raw["AQI Value"] = raw["AQI Value"].astype(float)
    raw["AQI Category"] = raw["AQI Category"]
    return raw


def load_prsa_sensor_weather_data():
    frames = []
    for path in sorted(glob(PRSA_PATH)):
        frame = pd.read_csv(path)
        frame["source_file"] = os.path.basename(path)
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            "No air-quality feed found. Expected air_quality.csv or PRSA station CSVs."
        )

    raw = pd.concat(frames, ignore_index=True)
    raw["datetime"] = pd.to_datetime(
        raw[["year", "month", "day", "hour"]]
    )
    raw["AQI Value"] = pm25_to_aqi(raw["PM2.5"])
    raw["AQI Category"] = aqi_category(raw["AQI Value"])
    return raw


@lru_cache(maxsize=1)
def load_data():
    if os.path.exists(DATA_PATH):
        return load_legacy_sensor_data()

    return load_prsa_sensor_weather_data()


def numeric_feature_columns(df):
    candidates = [
        "PM2.5",
        "PM10",
        "SO2",
        "NO2",
        "CO",
        "O3",
        "TEMP",
        "PRES",
        "DEWP",
        "RAIN",
        "WSPM",
    ]
    return [column for column in candidates if column in df.columns]


def required_columns(df):
    base_columns = [
        "station",
        "AQI Value",
        "AQI Category",
        "PM2.5",
        "NO2",
        "CO",
        "O3",
    ]
    if df["datetime"].notna().any():
        base_columns.append("datetime")
    return base_columns


def feed_status(df):
    source = (
        "Dataset/PRSA_Data_20130301-20170228/*.csv"
        if not os.path.exists(DATA_PATH)
        else DATA_PATH
    )
    has_weather = {"TEMP", "PRES", "DEWP", "RAIN", "WSPM"}.issubset(df.columns)

    return pd.DataFrame(
        [
            {
                "Feed": "Public air-quality sensors",
                "Status": "Ingested",
                "Current Source": source,
                "W2 Note": "Station-level pollutant history is validated and target-ready.",
            },
            {
                "Feed": "Weather",
                "Status": "Ingested" if has_weather else "Pending source",
                "Current Source": source if has_weather else "Not present in repo",
                "W2 Note": "Weather fields are cleaned with time-safe station medians.",
            },
            {
                "Feed": "Traffic",
                "Status": "Pending source",
                "Current Source": "Not present in repo",
                "W2 Note": "Reserved for future join by station/location/date.",
            },
            {
                "Feed": "Satellite",
                "Status": "Pending source",
                "Current Source": "Not present in repo",
                "W2 Note": "Reserved for future join by location/date.",
            },
        ]
    )


def build_validation_report(df):
    numeric_columns = df.select_dtypes(include=np.number).columns
    missing_required = [
        column for column in required_columns(df)
        if column not in df.columns
    ]

    checks = [
        {
            "Check": "Required schema",
            "Result": "Pass" if not missing_required else "Fail",
            "Detail": (
                "All expected columns are present."
                if not missing_required
                else ", ".join(missing_required)
            ),
        },
        {
            "Check": "Duplicate rows",
            "Result": "Pass" if df.duplicated().sum() == 0 else "Review",
            "Detail": f"{df.duplicated().sum()} duplicates found.",
        },
        {
            "Check": "AQI range",
            "Result": (
                "Pass"
                if df["AQI Value"].dropna().between(0, 500).all()
                else "Review"
            ),
            "Detail": (
                f"Observed {df['AQI Value'].min():.0f} to "
                f"{df['AQI Value'].max():.0f}."
            ),
        },
        {
            "Check": "Time coverage",
            "Result": (
                "Pass"
                if "datetime" in df.columns and df["datetime"].notna().any()
                else "Review"
            ),
            "Detail": (
                f"{df['datetime'].min()} to {df['datetime'].max()}"
                if "datetime" in df.columns and df["datetime"].notna().any()
                else "No timestamp available for true next-day labels."
            ),
        },
        {
            "Check": "Numeric missingness",
            "Result": (
                "Pass"
                if df[numeric_columns].isna().sum().sum() == 0
                else "Review"
            ),
            "Detail": f"{df[numeric_columns].isna().sum().sum()} missing numeric cells.",
        },
    ]

    return pd.DataFrame(checks)


def prepare_clean_data(df):
    cleaned = df.copy()
    duplicates_removed = cleaned.duplicated().sum()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    cleaned["station"] = cleaned["station"].fillna("Unknown")
    feature_columns = numeric_feature_columns(cleaned)

    for column in feature_columns:
        cleaned[column] = cleaned.groupby("station")[column].transform(
            lambda values: values.ffill().bfill()
        )
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    cleaned["AQI Value"] = cleaned["AQI Value"].fillna(
        pm25_to_aqi(cleaned["PM2.5"])
    )
    cleaned["AQI Category"] = aqi_category(cleaned["AQI Value"])
    cleaned["hazardous_air_day_flag"] = (
        cleaned["AQI Value"] >= HAZARDOUS_AQI_THRESHOLD
    ).astype(int)

    if "datetime" in cleaned.columns and cleaned["datetime"].notna().any():
        cleaned = cleaned.sort_values(["station", "datetime"])
        daily = (
            cleaned.groupby(["station", cleaned["datetime"].dt.date])["AQI Value"]
            .mean()
            .reset_index(name="daily_aqi_value")
            .rename(columns={"datetime": "date"})
        )
        daily["tomorrows_aqi_value"] = daily.groupby("station")[
            "daily_aqi_value"
        ].shift(-1)
        cleaned["date"] = cleaned["datetime"].dt.date
        cleaned = cleaned.merge(
            daily[["station", "date", "tomorrows_aqi_value"]],
            on=["station", "date"],
            how="left",
        )
    else:
        cleaned["tomorrows_aqi_value"] = np.nan

    cleaned = cleaned.dropna(subset=["tomorrows_aqi_value"]).reset_index(drop=True)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(cleaned[feature_columns])

    scaled_columns = [
        f"{column} Scaled"
        for column in feature_columns
    ]
    scaled_df = pd.DataFrame(scaled, columns=scaled_columns)

    model_ready = pd.concat(
        [
            cleaned[
                [
                    "station",
                    "datetime",
                    "AQI Value",
                    "AQI Category",
                    "hazardous_air_day_flag",
                    "tomorrows_aqi_value",
                ]
            ],
            scaled_df,
        ],
        axis=1,
    )

    return cleaned, model_ready, duplicates_removed


def build_eda_tables(df, cleaned):
    pollutant_columns = [
        column for column in ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
        if column in cleaned.columns
    ]
    weather_columns = [
        column for column in ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
        if column in cleaned.columns
    ]
    numeric_columns = pollutant_columns + weather_columns + ["AQI Value"]

    station_summary = (
        cleaned.groupby("station")
        .agg(
            rows=("AQI Value", "size"),
            mean_aqi=("AQI Value", "mean"),
            max_aqi=("AQI Value", "max"),
            hazardous_rate=("hazardous_air_day_flag", "mean"),
            next_day_mean_aqi=("tomorrows_aqi_value", "mean"),
        )
        .round(3)
        .sort_values("hazardous_rate", ascending=False)
        .reset_index()
    )

    hourly_pattern = (
        cleaned.assign(hour=cleaned["datetime"].dt.hour)
        .groupby("hour")
        .agg(
            mean_aqi=("AQI Value", "mean"),
            hazardous_rate=("hazardous_air_day_flag", "mean"),
            rows=("AQI Value", "size"),
        )
        .round(3)
        .reset_index()
    )

    monthly_pattern = (
        cleaned.assign(month=cleaned["datetime"].dt.month)
        .groupby("month")
        .agg(
            mean_aqi=("AQI Value", "mean"),
            mean_pm25=("PM2.5", "mean"),
            hazardous_rate=("hazardous_air_day_flag", "mean"),
            rows=("AQI Value", "size"),
        )
        .round(3)
        .reset_index()
    )

    correlation = (
        cleaned[numeric_columns + ["tomorrows_aqi_value", "hazardous_air_day_flag"]]
        .corr(numeric_only=True)
        .round(3)
    )
    target_correlation = (
        correlation[["tomorrows_aqi_value", "hazardous_air_day_flag"]]
        .drop(index=["tomorrows_aqi_value", "hazardous_air_day_flag"], errors="ignore")
        .reset_index()
        .rename(columns={"index": "feature"})
        .sort_values("tomorrows_aqi_value", key=lambda values: values.abs(), ascending=False)
    )

    outlier_rows = []
    for column in pollutant_columns + ["AQI Value"]:
        q1 = cleaned[column].quantile(0.25)
        q3 = cleaned[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((cleaned[column] < lower) | (cleaned[column] > upper)).sum())
        outlier_rows.append(
            {
                "feature": column,
                "lower_fence": round(lower, 3),
                "upper_fence": round(upper, 3),
                "outlier_rows": count,
                "outlier_rate": round(count / len(cleaned), 4),
            }
        )
    outliers = pd.DataFrame(outlier_rows)

    missing_profile = (
        df.isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "feature", 0: "missing_values"})
    )
    missing_profile["missing_rate"] = (
        missing_profile["missing_values"] / len(df)
    ).round(4)
    missing_profile = missing_profile.sort_values(
        "missing_values",
        ascending=False,
    )

    return {
        "station_summary": station_summary,
        "hourly_pattern": hourly_pattern,
        "monthly_pattern": monthly_pattern,
        "target_correlation": target_correlation,
        "outliers": outliers,
        "missing_profile": missing_profile,
    }


def generate_eda_charts(cleaned, eda_tables, force=False):
    os.makedirs(EDA_CHART_DIR, exist_ok=True)

    chart_specs = [
        {
            "title": "AQI Category Distribution",
            "filename": "eda_aqi_category_distribution.png",
        },
        {
            "title": "AQI Value Distribution",
            "filename": "eda_aqi_distribution.png",
        },
        {
            "title": "Station Hazardous-Day Rate",
            "filename": "eda_station_hazardous_rate.png",
        },
        {
            "title": "Hourly Hazardous-Day Pattern",
            "filename": "eda_hourly_hazardous_pattern.png",
        },
        {
            "title": "Monthly AQI and PM2.5 Pattern",
            "filename": "eda_monthly_aqi_pm25_pattern.png",
        },
        {
            "title": "Pollutant and Weather Correlation",
            "filename": "eda_correlation_heatmap.png",
        },
        {
            "title": "Target Correlation Ranking",
            "filename": "eda_target_correlation.png",
        },
        {
            "title": "Pollutant Outlier Boxplots",
            "filename": "eda_pollutant_boxplots.png",
        },
        {
            "title": "PM2.5 vs Tomorrow's AQI",
            "filename": "eda_pm25_vs_tomorrow_aqi.png",
        },
        {
            "title": "Missing Values by Feature",
            "filename": "eda_missing_values.png",
        },
    ]

    expected_paths = [
        os.path.join(EDA_CHART_DIR, spec["filename"])
        for spec in chart_specs
    ]
    if not force and all(os.path.exists(path) for path in expected_paths):
        return chart_specs

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    palette = ["#0f766e", "#2563eb", "#b45309", "#7c3aed", "#dc2626", "#475569"]
    pollutant_columns = [
        column for column in ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
        if column in cleaned.columns
    ]
    correlation_columns = [
        column for column in pollutant_columns + ["TEMP", "PRES", "DEWP", "RAIN", "WSPM", "AQI Value", "tomorrows_aqi_value"]
        if column in cleaned.columns
    ]

    def save_current(filename):
        path = os.path.join(EDA_CHART_DIR, filename)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    plt.figure(figsize=(9, 5))
    order = cleaned["AQI Category"].value_counts().index
    sns.countplot(data=cleaned, y="AQI Category", order=order, color=palette[0])
    plt.title("AQI Category Distribution")
    plt.xlabel("Rows")
    plt.ylabel("AQI Category")
    save_current("eda_aqi_category_distribution.png")

    plt.figure(figsize=(9, 5))
    sns.histplot(cleaned["AQI Value"], bins=50, kde=True, color=palette[1])
    plt.title("AQI Value Distribution")
    plt.xlabel("AQI Value")
    plt.ylabel("Frequency")
    save_current("eda_aqi_distribution.png")

    station_summary = eda_tables["station_summary"].sort_values("hazardous_rate")
    plt.figure(figsize=(9, 6))
    sns.barplot(data=station_summary, x="hazardous_rate", y="station", color=palette[0])
    plt.title("Hazardous-Day Rate by Station")
    plt.xlabel("Hazardous-Day Rate")
    plt.ylabel("Station")
    save_current("eda_station_hazardous_rate.png")

    hourly = eda_tables["hourly_pattern"]
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=hourly, x="hour", y="hazardous_rate", marker="o", color=palette[2])
    plt.title("Hourly Hazardous-Day Pattern")
    plt.xlabel("Hour")
    plt.ylabel("Hazardous-Day Rate")
    plt.xticks(range(0, 24, 2))
    save_current("eda_hourly_hazardous_pattern.png")

    monthly = eda_tables["monthly_pattern"]
    fig, axis = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=monthly, x="month", y="mean_aqi", marker="o", color=palette[0], ax=axis)
    axis.set_ylabel("Mean AQI")
    axis.set_xlabel("Month")
    second_axis = axis.twinx()
    sns.lineplot(data=monthly, x="month", y="mean_pm25", marker="s", color=palette[1], ax=second_axis)
    second_axis.set_ylabel("Mean PM2.5")
    axis.set_title("Monthly AQI and PM2.5 Pattern")
    axis.set_xticks(range(1, 13))
    save_current("eda_monthly_aqi_pm25_pattern.png")

    plt.figure(figsize=(10, 8))
    corr = cleaned[correlation_columns].corr(numeric_only=True)
    sns.heatmap(corr, cmap="vlag", center=0, linewidths=0.4)
    plt.title("Pollutant, Weather, and Target Correlation")
    save_current("eda_correlation_heatmap.png")

    target_corr = eda_tables["target_correlation"].copy()
    target_corr["abs_corr"] = target_corr["tomorrows_aqi_value"].abs()
    target_corr = target_corr.sort_values("abs_corr", ascending=True).tail(12)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=target_corr, x="tomorrows_aqi_value", y="feature", color=palette[1])
    plt.axvline(0, color="#475569", linewidth=1)
    plt.title("Top Correlations with Tomorrow's AQI")
    plt.xlabel("Correlation")
    plt.ylabel("Feature")
    save_current("eda_target_correlation.png")

    plt.figure(figsize=(10, 5))
    sample = cleaned[pollutant_columns].sample(
        n=min(50000, len(cleaned)),
        random_state=42,
    )
    sns.boxplot(data=sample, orient="h", color="#dbeafe")
    plt.title("Pollutant Outlier Boxplots")
    plt.xlabel("Observed Value")
    save_current("eda_pollutant_boxplots.png")

    scatter_sample = cleaned.sample(n=min(40000, len(cleaned)), random_state=42)
    plt.figure(figsize=(8, 6))
    plt.hexbin(
        scatter_sample["PM2.5"],
        scatter_sample["tomorrows_aqi_value"],
        gridsize=45,
        cmap="viridis",
        mincnt=1,
    )
    plt.colorbar(label="Rows")
    plt.title("PM2.5 vs Tomorrow's AQI")
    plt.xlabel("Current PM2.5")
    plt.ylabel("Tomorrow's AQI")
    save_current("eda_pm25_vs_tomorrow_aqi.png")

    missing = eda_tables["missing_profile"]
    missing = missing[missing["missing_values"] > 0].sort_values("missing_values")
    plt.figure(figsize=(9, 5))
    if missing.empty:
        plt.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=14)
        plt.axis("off")
    else:
        sns.barplot(data=missing, x="missing_values", y="feature", color=palette[2])
        plt.xlabel("Missing Values")
        plt.ylabel("Feature")
    plt.title("Missing Values by Feature")
    save_current("eda_missing_values.png")

    return chart_specs


def week_completion_summary():
    rows = []
    for week, filenames in WEEK_OUTPUTS.items():
        present = [
            filename for filename in filenames
            if os.path.exists(os.path.join("outputs", filename))
        ]
        rows.append(
            {
                "week": week,
                "complete": len(present) == len(filenames),
                "artifact_count": len(present),
                "expected_count": len(filenames),
            }
        )
    return rows


@lru_cache(maxsize=1)
def prediction_context():
    from aeropure_features import build_supervised_frame

    raw = load_data()
    cleaned, _, _ = prepare_clean_data(raw)
    model_frame, feature_columns, train, test, metadata = build_supervised_frame(cleaned)
    model = joblib.load(CHAMPION_MODEL_PATH) if os.path.exists(CHAMPION_MODEL_PATH) else None
    station_profiles = (
        model_frame.groupby("station")
        .agg(
            mean_aqi=("AQI Value", "mean"),
            mean_pm25=("PM2.5", "mean"),
            mean_pm10=("PM10", "mean"),
            mean_temp=("TEMP", "mean"),
            mean_wspm=("WSPM", "mean"),
            hazardous_rate=(TARGET_COLUMN, "mean"),
            rows=("AQI Value", "size"),
        )
        .round(3)
        .sort_values("hazardous_rate", ascending=False)
        .reset_index()
    )
    latest_rows = (
        model_frame.sort_values(["datetime", "station"])
        .groupby("station", as_index=False)
        .tail(1)
        .set_index("station")
    )
    defaults = latest_rows.mean(numeric_only=True).to_dict()
    if not latest_rows.empty:
        first_station = latest_rows.index.sort_values()[0]
        defaults.update(latest_rows.loc[first_station].to_dict())
    return {
        "raw": raw,
        "cleaned": cleaned,
        "model_frame": model_frame,
        "feature_columns": feature_columns,
        "train": train,
        "test": test,
        "metadata": metadata,
        "model": model,
        "station_profiles": station_profiles,
        "latest_rows": latest_rows,
        "defaults": defaults,
    }


def form_float(form, name, default):
    try:
        return float(form.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def form_int(form, name, default, low, high):
    try:
        value = int(form.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def scenario_inputs(form, context):
    latest_rows = context["latest_rows"]
    stations = sorted(latest_rows.index.tolist())
    station = form.get("station") or (stations[0] if stations else "Unknown")
    if station not in latest_rows.index and stations:
        station = stations[0]
    station_defaults = latest_rows.loc[station].to_dict() if station in latest_rows.index else context["defaults"]
    return {
        "station": station,
        "forecast_days": form_int(form, "forecast_days", 7, 1, 30),
        "aqi_value": form_float(form, "aqi_value", station_defaults.get("AQI Value", 100)),
        "pm25": form_float(form, "pm25", station_defaults.get("PM2.5", 35)),
        "pm10": form_float(form, "pm10", station_defaults.get("PM10", 80)),
        "so2": form_float(form, "so2", station_defaults.get("SO2", 10)),
        "no2": form_float(form, "no2", station_defaults.get("NO2", 45)),
        "co": form_float(form, "co", station_defaults.get("CO", 700)),
        "o3": form_float(form, "o3", station_defaults.get("O3", 50)),
        "temp": form_float(form, "temp", station_defaults.get("TEMP", 12)),
        "pres": form_float(form, "pres", station_defaults.get("PRES", 1010)),
        "dewp": form_float(form, "dewp", station_defaults.get("DEWP", 2)),
        "rain": form_float(form, "rain", station_defaults.get("RAIN", 0)),
        "wspm": form_float(form, "wspm", station_defaults.get("WSPM", 2)),
        "traffic_factor": form_float(form, "traffic_factor", 1.0),
        "satellite_smoke": form_float(form, "satellite_smoke", 0.0),
    }


def build_scenario_row(context, inputs, forecast_date, day_index, previous_aqi):
    latest_rows = context["latest_rows"]
    feature_columns = context["feature_columns"]
    station = inputs["station"]
    if station in latest_rows.index:
        row = latest_rows.loc[station].copy()
    else:
        row = pd.Series(context["defaults"]).copy()

    traffic_multiplier = 1 + ((inputs["traffic_factor"] - 1) * 0.28)
    smoke_lift = inputs["satellite_smoke"] * 18
    persistence = 0.82 ** max(day_index - 1, 0)
    projected_aqi = (
        inputs["aqi_value"] * persistence
        + row.get("AQI Value", inputs["aqi_value"]) * (1 - persistence)
        + smoke_lift
        + max(inputs["traffic_factor"] - 1, 0) * 12
    )
    projected_aqi = float(np.clip((projected_aqi + previous_aqi) / 2, 0, 500))

    row["AQI Value"] = projected_aqi
    row["PM2.5"] = max(inputs["pm25"] * traffic_multiplier + inputs["satellite_smoke"] * 10, 0)
    row["PM10"] = max(inputs["pm10"] * traffic_multiplier + inputs["satellite_smoke"] * 14, 0)
    row["SO2"] = max(inputs["so2"] * traffic_multiplier, 0)
    row["NO2"] = max(inputs["no2"] * traffic_multiplier, 0)
    row["CO"] = max(inputs["co"] * traffic_multiplier, 0)
    row["O3"] = max(inputs["o3"] + max(inputs["temp"] - 20, 0) * 0.8, 0)
    row["TEMP"] = inputs["temp"]
    row["PRES"] = inputs["pres"]
    row["DEWP"] = inputs["dewp"]
    row["RAIN"] = inputs["rain"]
    row["WSPM"] = inputs["wspm"]
    row["hour"] = 12
    row["month"] = forecast_date.month
    row["day_of_week"] = forecast_date.dayofweek
    row["hazardous_air_day_flag"] = int(projected_aqi >= HAZARDOUS_AQI_THRESHOLD)

    for column in feature_columns:
        if column.startswith("station_"):
            row[column] = int(column == f"station_{station}")
        elif column.endswith("_lag_24h"):
            base = column.removesuffix("_lag_24h")
            row[column] = row.get(base, context["defaults"].get(base, 0))
        elif column.endswith("_rolling_24h_mean"):
            base = column.removesuffix("_rolling_24h_mean")
            row[column] = (row.get(base, 0) + context["defaults"].get(base, row.get(base, 0))) / 2
        elif pd.isna(row.get(column, np.nan)):
            row[column] = context["defaults"].get(column, 0)

    frame = pd.DataFrame([row])
    return frame[feature_columns], projected_aqi


def forecast_from_inputs(inputs):
    context = prediction_context()
    model = context["model"]
    if model is None:
        return context, pd.DataFrame(), "Run python week_11_package_champion.py to create the packaged champion model."

    latest_date = context["model_frame"]["datetime"].max()
    previous_aqi = inputs["aqi_value"]
    rows = []
    for day in range(1, inputs["forecast_days"] + 1):
        forecast_date = latest_date + pd.Timedelta(days=day)
        feature_row, projected_aqi = build_scenario_row(
            context,
            inputs,
            forecast_date,
            day,
            previous_aqi,
        )
        probability = float(model.predict_proba(feature_row)[:, 1][0])
        risk_label = (
            "Severe" if probability >= 0.75 else
            "High" if probability >= 0.5 else
            "Watch" if probability >= 0.3 else
            "Low"
        )
        rows.append(
            {
                "forecast_date": forecast_date.date().isoformat(),
                "station": inputs["station"],
                "projected_aqi": round(projected_aqi, 1),
                "hazard_probability": round(probability, 3),
                "hazard_risk": risk_label,
                "expected_category": aqi_category(pd.Series([projected_aqi])).iloc[0],
            }
        )
        previous_aqi = projected_aqi
    return context, pd.DataFrame(rows), None


@app.route("/")
def home():
    df = load_data()
    cleaned, _, _ = prepare_clean_data(df)
    completion = week_completion_summary()
    forecast_preview = pd.DataFrame()
    if os.path.exists(os.path.join("outputs", "w12_public_forecast.csv")):
        forecast_preview = pd.read_csv(os.path.join("outputs", "w12_public_forecast.csv")).head(5)
    station_summary = (
        cleaned.groupby("station")
        .agg(mean_aqi=("AQI Value", "mean"), hazardous_rate=("hazardous_air_day_flag", "mean"))
        .round(3)
        .sort_values("hazardous_rate", ascending=False)
        .head(5)
        .reset_index()
    )
    current_station_rows = (
        cleaned.sort_values(["datetime", "station"])
        .groupby("station", as_index=False)
        .tail(1)
        .sort_values("AQI Value", ascending=False)
        .head(12)
        .copy()
    )
    current_station_rows["aqi_level"] = current_station_rows["AQI Value"].apply(aqi_level)
    current_station_rows["marker_left"] = np.linspace(12, 84, len(current_station_rows)).round(1)
    current_station_rows["marker_top"] = (
        42 + 24 * np.sin(np.linspace(0, 2 * np.pi, len(current_station_rows), endpoint=False))
    ).round(1)
    latest_aqi = float(current_station_rows["AQI Value"].iloc[0]) if not current_station_rows.empty else 0
    latest_station = current_station_rows["station"].iloc[0] if not current_station_rows.empty else "N/A"
    forecast_average = (
        float(forecast_preview["hazard_probability"].mean())
        if not forecast_preview.empty and "hazard_probability" in forecast_preview.columns
        else 0
    )
    monitor_status = "not generated"
    monitor_path = os.path.join("outputs", "w12_drift_monitoring_status.csv")
    if os.path.exists(monitor_path):
        monitor_frame = pd.read_csv(monitor_path)
        if not monitor_frame.empty:
            monitor_status = monitor_frame.loc[0, "status"]

    return render_template(
        "index.html",
        rows=len(df),
        columns=len(df.columns),
        hazardous_days=int(cleaned["hazardous_air_day_flag"].sum()),
        missing_values=int(df.isna().sum().sum()),
        completion=completion,
        completed_weeks=sum(1 for row in completion if row["complete"]),
        forecast_preview=forecast_preview.to_dict("records"),
        station_summary=station_summary.to_dict("records"),
        station_map=current_station_rows[
            ["station", "AQI Value", "PM2.5", "TEMP", "aqi_level", "marker_left", "marker_top"]
        ].round(1).to_dict("records"),
        latest_aqi=round(latest_aqi, 1),
        latest_station=latest_station,
        latest_level=aqi_level(latest_aqi),
        forecast_average=round(forecast_average * 100, 1),
        monitor_status=monitor_status,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    context = prediction_context()
    stations = sorted(context["latest_rows"].index.tolist())
    inputs = scenario_inputs(request.form if request.method == "POST" else {}, context)
    forecast = pd.DataFrame()
    error = None
    if request.method == "POST":
        context, forecast, error = forecast_from_inputs(inputs)

    profile = context["station_profiles"]
    selected_profile = profile[profile["station"] == inputs["station"]].to_dict("records")
    return render_template(
        "predict.html",
        stations=stations,
        inputs=inputs,
        forecast=forecast.to_dict("records"),
        forecast_table=forecast.to_html(index=False, classes="data-table") if not forecast.empty else "",
        error=error,
        selected_profile=selected_profile[0] if selected_profile else {},
        model_ready=context["model"] is not None,
        feature_count=len(context["feature_columns"]),
        week_completion=week_completion_summary(),
    )


@app.route("/presentation")
def presentation():
    pptx_path = os.path.join("static", "presentations", "aeropure_w1_w12_project.pptx")
    return render_template(
        "presentation.html",
        deck_ready=os.path.exists(pptx_path),
        deck_url="/static/presentations/aeropure_w1_w12_project.pptx",
    )


@app.route("/dataset")
def dataset():
    df = load_data()

    columns = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values
    })

    preview = df.head(10)
    current_feed_status = feed_status(df)

    return render_template(
        "dataset.html",
        rows=len(df),
        columns=len(df.columns),
        feed_status=current_feed_status.to_html(
            index=False,
            classes="data-table"
        ),
        column_data=columns.to_html(
            index=False,
            classes="data-table"
        ),
        preview=preview.to_html(
            index=False,
            classes="data-table"
        )
    )


@app.route("/eda")
def eda():
    df = load_data()
    cleaned, _, _ = prepare_clean_data(df)
    eda_tables = build_eda_tables(df, cleaned)
    eda_charts = generate_eda_charts(cleaned, eda_tables)

    missing = df.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Values"]

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    statistics = df[numeric_columns].describe().round(2)
    category_distribution = (
        df["AQI Category"]
        .value_counts()
        .rename_axis("AQI Category")
        .reset_index(name="Rows")
    )
    hazardous_summary = pd.DataFrame(
        [
            {
                "Target": "hazardous_air_day_flag",
                "Definition": f"1 when AQI Value >= {HAZARDOUS_AQI_THRESHOLD}",
                "Positive Rows": int(cleaned["hazardous_air_day_flag"].sum()),
                "Positive Rate": (
                    f"{cleaned['hazardous_air_day_flag'].mean() * 100:.2f}%"
                ),
            },
            {
                "Target": "tomorrows_aqi_value",
                "Definition": (
                    "Next calendar day's mean AQI Value for the same station, "
                    "computed after sorting by station and timestamp."
                ),
                "Positive Rows": cleaned["tomorrows_aqi_value"].notna().sum(),
                "Positive Rate": "N/A",
            },
        ]
    )

    return render_template(
        "eda.html",
        missing=missing.to_html(
            index=False,
            classes="data-table"
        ),
        statistics=statistics.to_html(
            classes="data-table"
        ),
        category_distribution=category_distribution.to_html(
            index=False,
            classes="data-table"
        ),
        hazardous_summary=hazardous_summary.to_html(
            index=False,
            classes="data-table"
        ),
        station_summary=eda_tables["station_summary"].to_html(
            index=False,
            classes="data-table"
        ),
        hourly_pattern=eda_tables["hourly_pattern"].to_html(
            index=False,
            classes="data-table"
        ),
        monthly_pattern=eda_tables["monthly_pattern"].to_html(
            index=False,
            classes="data-table"
        ),
        target_correlation=eda_tables["target_correlation"].to_html(
            index=False,
            classes="data-table"
        ),
        outliers=eda_tables["outliers"].to_html(
            index=False,
            classes="data-table"
        ),
        missing_profile=eda_tables["missing_profile"].to_html(
            index=False,
            classes="data-table"
        ),
        eda_charts=eda_charts
    )


@app.route("/visualization")
def visualization():
    df = load_data()
    pollutant_columns = [
        column for column in ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
        if column in df.columns
    ]
    pollutant_means = (
        df[pollutant_columns]
        .mean()
        .round(2)
        .rename_axis("Pollutant")
        .reset_index(name="Mean AQI")
    )
    top_stations = (
        df.groupby("station", dropna=False)["AQI Value"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .round(2)
        .rename_axis("Station")
        .reset_index(name="Mean AQI")
    )

    return render_template(
        "visualization.html",
        pollutant_means=pollutant_means.to_html(
            index=False,
            classes="data-table"
        ),
        top_stations=top_stations.to_html(
            index=False,
            classes="data-table"
        ),
    )


@app.route("/preprocessing")
def preprocessing():

    df = load_data()
    cleaned, model_ready, duplicates = prepare_clean_data(df)
    validation = build_validation_report(df)

    return render_template(
        "preprocessing.html",
        duplicates=duplicates,
        original_rows=len(df),
        final_rows=len(cleaned),
        missing_values=int(df.isna().sum().sum()),
        validation=validation.to_html(
            index=False,
            classes="data-table"
        ),
        data=model_ready.head(10).to_html(
            index=False,
            classes="data-table"
        )
    )


@app.route("/w3")
def w3():
    metric_path = os.path.join("outputs", "w3_linear_regression_metrics.csv")
    coefficient_path = os.path.join("outputs", "w3_linear_regression_coefficients.csv")
    prediction_path = os.path.join("outputs", "w3_linear_regression_predictions.csv")
    loss_path = os.path.join("outputs", "w3_gradient_descent_loss.csv")

    if not os.path.exists(metric_path):
        return render_template(
            "w3.html",
            ready=False,
            metrics="",
            coefficients="",
            predictions="",
            loss="",
        )

    metrics = pd.read_csv(metric_path)
    coefficients = pd.read_csv(coefficient_path)
    predictions = pd.read_csv(prediction_path)
    loss = pd.read_csv(loss_path)

    return render_template(
        "w3.html",
        ready=True,
        metrics=metrics.to_html(index=False, classes="data-table"),
        coefficients=coefficients.to_html(index=False, classes="data-table"),
        predictions=predictions.head(25).to_html(index=False, classes="data-table"),
        loss=loss.to_html(index=False, classes="data-table"),
    )


@app.route("/w4")
def w4():
    regression_path = os.path.join("outputs", "w4_ridge_lasso_regression_metrics.csv")
    classification_path = os.path.join("outputs", "w4_logistic_regression_metrics.csv")
    coefficient_path = os.path.join("outputs", "w4_regularized_logistic_coefficients.csv")
    feature_path = os.path.join("outputs", "w4_feature_engineering_summary.csv")
    prediction_path = os.path.join("outputs", "w4_regularized_logistic_predictions.csv")

    if not os.path.exists(regression_path):
        return render_template(
            "w4.html",
            ready=False,
            regression_metrics="",
            classification_metrics="",
            feature_summary="",
            coefficients="",
            predictions="",
        )

    regression_metrics = pd.read_csv(regression_path)
    classification_metrics = pd.read_csv(classification_path)
    feature_summary = pd.read_csv(feature_path)
    coefficients = pd.read_csv(coefficient_path)
    predictions = pd.read_csv(prediction_path)

    return render_template(
        "w4.html",
        ready=True,
        regression_metrics=regression_metrics.to_html(index=False, classes="data-table"),
        classification_metrics=classification_metrics.to_html(index=False, classes="data-table"),
        feature_summary=feature_summary.to_html(index=False, classes="data-table"),
        coefficients=coefficients.head(25).to_html(index=False, classes="data-table"),
        predictions=predictions.head(25).to_html(index=False, classes="data-table"),
    )


@app.route("/w5")
def w5():
    cv_path = os.path.join("outputs", "w5_cross_validation_results.csv")
    selection_path = os.path.join("outputs", "w5_one_standard_error_selection.csv")
    regression_path = os.path.join("outputs", "w5_honest_regression_metrics.csv")
    classification_path = os.path.join("outputs", "w5_honest_logistic_metrics.csv")
    milestone_path = os.path.join("outputs", "w5_honest_baseline_milestone.csv")
    prediction_path = os.path.join("outputs", "w5_honest_predictions.csv")

    if not os.path.exists(milestone_path):
        return render_template(
            "w5.html",
            ready=False,
            milestone="",
            selection="",
            regression_metrics="",
            classification_metrics="",
            cv_results="",
            predictions="",
        )

    cv_results = pd.read_csv(cv_path)
    selection = pd.read_csv(selection_path)
    regression_metrics = pd.read_csv(regression_path)
    classification_metrics = pd.read_csv(classification_path)
    milestone = pd.read_csv(milestone_path)
    predictions = pd.read_csv(prediction_path)

    return render_template(
        "w5.html",
        ready=True,
        milestone=milestone.to_html(index=False, classes="data-table"),
        selection=selection.to_html(index=False, classes="data-table"),
        regression_metrics=regression_metrics.to_html(index=False, classes="data-table"),
        classification_metrics=classification_metrics.to_html(index=False, classes="data-table"),
        cv_results=cv_results.head(40).to_html(index=False, classes="data-table"),
        predictions=predictions.head(25).to_html(index=False, classes="data-table"),
    )


@app.route("/w6")
def w6():
    metric_path = os.path.join("outputs", "w6_decision_tree_metrics.csv")
    importance_path = os.path.join("outputs", "w6_decision_tree_importance.csv")
    split_path = os.path.join("outputs", "w6_decision_tree_splits.csv")
    rule_path = os.path.join("outputs", "w6_decision_tree_rules.csv")
    prediction_path = os.path.join("outputs", "w6_decision_tree_predictions.csv")

    if not os.path.exists(metric_path):
        return render_template(
            "w6.html",
            ready=False,
            metrics="",
            importance="",
            splits="",
            rules="",
            predictions="",
        )

    metrics = pd.read_csv(metric_path)
    importance = pd.read_csv(importance_path)
    splits = pd.read_csv(split_path)
    rules = pd.read_csv(rule_path)
    predictions = pd.read_csv(prediction_path)

    return render_template(
        "w6.html",
        ready=True,
        metrics=metrics.to_html(index=False, classes="data-table"),
        importance=importance.head(25).to_html(index=False, classes="data-table"),
        splits=splits.head(25).to_html(index=False, classes="data-table"),
        rules=rules.to_html(index=False, classes="data-table"),
        predictions=predictions.head(25).to_html(index=False, classes="data-table"),
    )


@app.route("/w7")
def w7():
    metric_path = os.path.join("outputs", "w7_random_forest_metrics.csv")
    impurity_path = os.path.join("outputs", "w7_random_forest_impurity_importance.csv")
    permutation_path = os.path.join("outputs", "w7_random_forest_permutation_importance.csv")
    group_path = os.path.join("outputs", "w7_random_forest_group_importance.csv")
    unavailable_path = os.path.join("outputs", "w7_unavailable_feature_groups.csv")
    prediction_path = os.path.join("outputs", "w7_random_forest_predictions.csv")

    if not os.path.exists(metric_path):
        return render_template(
            "w7.html",
            ready=False,
            metrics="",
            impurity="",
            permutation="",
            groups="",
            unavailable="",
            predictions="",
        )

    metrics = pd.read_csv(metric_path)
    impurity = pd.read_csv(impurity_path)
    permutation = pd.read_csv(permutation_path)
    groups = pd.read_csv(group_path)
    unavailable = pd.read_csv(unavailable_path)
    predictions = pd.read_csv(prediction_path)

    return render_template(
        "w7.html",
        ready=True,
        metrics=metrics.to_html(index=False, classes="data-table"),
        impurity=impurity.head(25).to_html(index=False, classes="data-table"),
        permutation=permutation.head(25).to_html(index=False, classes="data-table"),
        groups=groups.to_html(index=False, classes="data-table"),
        unavailable=unavailable.to_html(index=False, classes="data-table"),
        predictions=predictions.head(25).to_html(index=False, classes="data-table"),
    )


@app.route("/w8")
def w8():
    metric_path = os.path.join("outputs", "w8_boosting_metrics.csv")
    importance_path = os.path.join("outputs", "w8_boosting_permutation_importance.csv")
    slice_path = os.path.join("outputs", "w8_boosting_prediction_slices.csv")
    note_path = os.path.join("outputs", "w8_boosting_notes.csv")
    prediction_path = os.path.join("outputs", "w8_boosting_predictions.csv")

    if not os.path.exists(metric_path):
        return render_template(
            "w8.html",
            ready=False,
            metrics="",
            importance="",
            slices="",
            notes="",
            predictions="",
        )

    return render_template(
        "w8.html",
        ready=True,
        metrics=pd.read_csv(metric_path).to_html(index=False, classes="data-table"),
        importance=pd.read_csv(importance_path).head(25).to_html(index=False, classes="data-table"),
        slices=pd.read_csv(slice_path).to_html(index=False, classes="data-table"),
        notes=pd.read_csv(note_path).to_html(index=False, classes="data-table"),
        predictions=pd.read_csv(prediction_path).head(25).to_html(index=False, classes="data-table"),
    )


@app.route("/w9")
def w9():
    profile_path = os.path.join("outputs", "w9_kmeans_cluster_profile.csv")
    pca_path = os.path.join("outputs", "w9_pca_summary.csv")
    dbscan_path = os.path.join("outputs", "w9_dbscan_summary.csv")
    quality_path = os.path.join("outputs", "w9_cluster_quality.csv")
    preview_path = os.path.join("outputs", "w9_segment_feature_preview.csv")
    note_path = os.path.join("outputs", "w9_segment_feature_note.csv")

    if not os.path.exists(profile_path):
        return render_template(
            "w9.html",
            ready=False,
            profile="",
            pca="",
            dbscan="",
            quality="",
            preview="",
            note="",
        )

    return render_template(
        "w9.html",
        ready=True,
        profile=pd.read_csv(profile_path).to_html(index=False, classes="data-table"),
        pca=pd.read_csv(pca_path).to_html(index=False, classes="data-table"),
        dbscan=pd.read_csv(dbscan_path).to_html(index=False, classes="data-table"),
        quality=pd.read_csv(quality_path).to_html(index=False, classes="data-table"),
        preview=pd.read_csv(preview_path).head(25).to_html(index=False, classes="data-table"),
        note=pd.read_csv(note_path).to_html(index=False, classes="data-table"),
    )


@app.route("/w10")
def w10():
    nested_path = os.path.join("outputs", "w10_nested_cv_results.csv")
    metrics_path = os.path.join("outputs", "w10_sealed_test_metrics.csv")
    calibration_path = os.path.join("outputs", "w10_calibration_metrics.csv")
    significance_path = os.path.join("outputs", "w10_significance_test.csv")
    champion_path = os.path.join("outputs", "w10_champion_summary.csv")

    if not os.path.exists(champion_path):
        return render_template(
            "w10.html",
            ready=False,
            nested="",
            metrics="",
            calibration="",
            significance="",
            champion="",
        )

    return render_template(
        "w10.html",
        ready=True,
        champion=pd.read_csv(champion_path).to_html(index=False, classes="data-table"),
        metrics=pd.read_csv(metrics_path).to_html(index=False, classes="data-table"),
        calibration=pd.read_csv(calibration_path).to_html(index=False, classes="data-table"),
        significance=pd.read_csv(significance_path).to_html(index=False, classes="data-table"),
        nested=pd.read_csv(nested_path).to_html(index=False, classes="data-table"),
    )


@app.route("/w11")
def w11():
    metrics_path = os.path.join("outputs", "w11_champion_package_metrics.csv")
    schema_path = os.path.join("outputs", "w11_feature_schema.csv")
    drift_path = os.path.join("outputs", "w11_psi_drift_guard.csv")
    manifest_path = os.path.join("outputs", "w11_package_manifest.csv")

    if not os.path.exists(manifest_path):
        return render_template(
            "w11.html",
            ready=False,
            metrics="",
            schema="",
            drift="",
            manifest="",
        )

    return render_template(
        "w11.html",
        ready=True,
        metrics=pd.read_csv(metrics_path).to_html(index=False, classes="data-table"),
        manifest=pd.read_csv(manifest_path).to_html(index=False, classes="data-table"),
        drift=pd.read_csv(drift_path).head(40).to_html(index=False, classes="data-table"),
        schema=pd.read_csv(schema_path).head(40).to_html(index=False, classes="data-table"),
    )


@app.route("/w12")
@app.route("/forecast")
def w12():
    forecast_path = os.path.join("outputs", "w12_public_forecast.csv")
    monitor_path = os.path.join("outputs", "w12_drift_monitoring_status.csv")
    deployment_path = os.path.join("outputs", "w12_deployment_manifest.csv")
    retrain_path = os.path.join("outputs", "w12_seasonal_retrain_schedule.csv")

    if not os.path.exists(forecast_path):
        return render_template(
            "w12.html",
            ready=False,
            forecast="",
            monitor="",
            deployment="",
            retrain="",
        )

    return render_template(
        "w12.html",
        ready=True,
        forecast=pd.read_csv(forecast_path).to_html(index=False, classes="data-table"),
        monitor=pd.read_csv(monitor_path).to_html(index=False, classes="data-table"),
        deployment=pd.read_csv(deployment_path).to_html(index=False, classes="data-table"),
        retrain=pd.read_csv(retrain_path).to_html(index=False, classes="data-table"),
    )


if __name__ == "__main__":
    app.run(
        debug=True
    )

