import os

import pandas as pd

from app import (
    build_validation_report,
    feed_status,
    load_data,
    numeric_feature_columns,
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
    cleaned, model_ready, duplicates_removed = prepare_clean_data(raw)
    validation = build_validation_report(raw)
    feature_columns = numeric_feature_columns(cleaned)

    pipeline_steps = pd.DataFrame(
        [
            {
                "step": "ingest",
                "detail": "Load PRSA station-level air-quality and weather CSV files.",
            },
            {
                "step": "validate",
                "detail": (
                    "Check schema, duplicate rows, AQI bounds, time coverage, "
                    "and numeric missingness."
                ),
            },
            {
                "step": "clean",
                "detail": (
                    "Drop duplicates, fill station names, and fill numeric gaps "
                    "with station-aware forward/backward fill plus median fallback."
                ),
            },
            {
                "step": "target",
                "detail": (
                    "Create hazardous_air_day_flag and tomorrows_aqi_value "
                    "without using future features."
                ),
            },
            {
                "step": "scale",
                "detail": (
                    "Scale source-safe pollutant and weather features with "
                    "StandardScaler."
                ),
            },
        ]
    )
    feature_audit = pd.DataFrame(
        {
            "feature": feature_columns,
            "role": "source_safe_model_input",
        }
    )

    files = [
        save_table(feed_status(raw), "w2_feed_status.csv"),
        save_table(validation, "w2_validation_report.csv"),
        save_table(pipeline_steps, "w2_pipeline_steps.csv"),
        save_table(feature_audit, "w2_feature_audit.csv"),
        save_table(model_ready.head(100), "w2_model_ready_preview.csv"),
    ]

    print("W2 complete: ingestion, cleaning, validation, and leakage-safe pipeline")
    print(f"Raw rows: {len(raw):,}")
    print(f"Clean model-ready rows: {len(cleaned):,}")
    print(f"Duplicates removed: {duplicates_removed:,}")
    print(f"Feature columns: {', '.join(feature_columns)}")
    print("Output files:")
    for path in files:
        print(f"- {path}")


if __name__ == "__main__":
    main()
