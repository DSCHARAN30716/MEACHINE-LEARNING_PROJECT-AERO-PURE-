from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

DATA_PATH = "air_quality.csv"


def load_data():
    return pd.read_csv(DATA_PATH)


@app.route("/")
def home():
    df = load_data()

    return render_template(
        "index.html",
        rows=len(df),
        columns=len(df.columns)
    )


@app.route("/dataset")
def dataset():
    df = load_data()

    columns = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values
    })

    preview = df.head(10)

    return render_template(
        "dataset.html",
        rows=len(df),
        columns=len(df.columns),
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

    missing = df.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Values"]

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    statistics = df[numeric_columns].describe().round(2)

    return render_template(
        "eda.html",
        missing=missing.to_html(
            index=False,
            classes="data-table"
        ),
        statistics=statistics.to_html(
            classes="data-table"
        )
    )


@app.route("/visualization")
def visualization():
    return render_template("visualization.html")


@app.route("/preprocessing")
def preprocessing():

    df = load_data()

    df["Country"] = df["Country"].fillna(
        df["Country"].mode()[0]
    )

    duplicates = df.duplicated().sum()

    df = df.drop_duplicates()

    numeric_features = [
        "AQI Value",
        "CO AQI Value",
        "Ozone AQI Value",
        "NO2 AQI Value",
        "PM2.5 AQI Value",
        "lat",
        "lng"
    ]

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        df[numeric_features]
    )

    scaled_df = pd.DataFrame(
        scaled_data,
        columns=[
            column + " Scaled"
            for column in numeric_features
        ]
    )

    result = scaled_df.head(10)

    return render_template(
        "preprocessing.html",
        duplicates=duplicates,
        original_rows=len(load_data()),
        final_rows=len(df),
        data=result.to_html(
            index=False,
            classes="data-table"
        )
    )


if __name__ == "__main__":
    app.run(
        debug=True
    )

