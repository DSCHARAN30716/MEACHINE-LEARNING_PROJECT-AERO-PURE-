import os

import pandas as pd

from app import (
    HAZARDOUS_AQI_THRESHOLD,
    feed_status,
    load_data,
    prepare_clean_data,
)


OUTPUT_DIR = "outputs"


def save_table(frame, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    frame.to_csv(path, index=False)
    return path


def main():
    raw = load_data()
    cleaned, model_ready, _ = prepare_clean_data(raw)

    missing = (
        raw.isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_values"})
    )
    category_distribution = (
        cleaned["AQI Category"]
        .value_counts()
        .rename_axis("aqi_category")
        .reset_index(name="rows")
    )
    numeric_statistics = raw.select_dtypes(include="number").describe().round(2)
    target_summary = pd.DataFrame(
        [
            {
                "target": "hazardous_air_day_flag",
                "definition": f"1 when AQI Value >= {HAZARDOUS_AQI_THRESHOLD}",
                "available_rows": len(cleaned),
                "positive_rows": int(cleaned["hazardous_air_day_flag"].sum()),
                "positive_rate": round(
                    cleaned["hazardous_air_day_flag"].mean(), 4
                ),
            },
            {
                "target": "tomorrows_aqi_value",
                "definition": (
                    "Next calendar day's mean AQI Value for the same station"
                ),
                "available_rows": int(
                    cleaned["tomorrows_aqi_value"].notna().sum()
                ),
                "positive_rows": "N/A",
                "positive_rate": "N/A",
            },
        ]
    )

    files = [
        save_table(feed_status(raw), "w1_feed_status.csv"),
        save_table(missing, "w1_missing_values.csv"),
        save_table(category_distribution, "w1_aqi_category_distribution.csv"),
        save_table(
            numeric_statistics.reset_index().rename(columns={"index": "metric"}),
            "w1_numeric_statistics.csv",
        ),
        save_table(target_summary, "w1_target_summary.csv"),
        save_table(model_ready.head(100), "w1_model_ready_preview.csv"),
    ]

    print("W1 complete: target framing and first EDA")
    print(f"Rows: {len(raw):,}")
    print(f"Columns: {len(raw.columns):,}")
    print(f"Model-ready rows with next-day target: {len(cleaned):,}")
    print(f"Hazardous air-day rows: {int(cleaned['hazardous_air_day_flag'].sum()):,}")
    print("Output files:")
    for path in files:
        print(f"- {path}")


if __name__ == "__main__":
    main()
