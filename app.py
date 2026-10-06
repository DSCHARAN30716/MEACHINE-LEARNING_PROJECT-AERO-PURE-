from flask import Flask, render_template
from functools import lru_cache
from glob import glob
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


@app.route("/")
def home():
    df = load_data()
    cleaned, _, _ = prepare_clean_data(df)

    return render_template(
        "index.html",
        rows=len(df),
        columns=len(df.columns),
        hazardous_days=int(cleaned["hazardous_air_day_flag"].sum()),
        missing_values=int(df.isna().sum().sum()),
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
        )
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


if __name__ == "__main__":
    app.run(
        debug=True
    )

