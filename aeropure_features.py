"""Shared feature assembly for AeroPure supervised models."""

from week_04_regularized_and_logistic import chronological_split, engineer_history_features


TARGET_COLUMN = "tomorrows_hazardous_air_day_flag"


def build_supervised_frame(cleaned, test_fraction=0.2):
    """Return the feature frame, ordered columns, and chronological train/test split."""
    model_frame, feature_columns, base_columns, engineered_columns, station_columns = (
        engineer_history_features(cleaned)
    )
    train, test = chronological_split(model_frame, test_fraction)
    metadata = {
        "base_columns": base_columns,
        "engineered_columns": engineered_columns,
        "station_columns": station_columns,
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "test_fraction": test_fraction,
    }
    return model_frame, feature_columns, train, test, metadata
