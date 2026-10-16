import os

from app import build_eda_tables, generate_eda_charts, load_data, prepare_clean_data


OUTPUT_DIR = "outputs"


def save_table(frame, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    frame.to_csv(path, index=False)
    return path


def main():
    raw = load_data()
    cleaned, _, _ = prepare_clean_data(raw)
    tables = build_eda_tables(raw, cleaned)
    charts = generate_eda_charts(cleaned, tables, force=True)

    files = [
        save_table(tables["missing_profile"], "eda_missingness_profile.csv"),
        save_table(tables["station_summary"], "eda_station_risk_summary.csv"),
        save_table(tables["hourly_pattern"], "eda_hourly_pattern.csv"),
        save_table(tables["monthly_pattern"], "eda_monthly_pattern.csv"),
        save_table(tables["target_correlation"], "eda_target_correlation.csv"),
        save_table(tables["outliers"], "eda_outlier_scan.csv"),
    ]

    print("EDA report complete")
    print(f"Raw rows: {len(raw):,}")
    print(f"Clean rows: {len(cleaned):,}")
    print("Output files:")
    for path in files:
        print(f"- {path}")
    print("Chart files:")
    for chart in charts:
        print(f"- static\\charts\\{chart['filename']}")


if __name__ == "__main__":
    main()
