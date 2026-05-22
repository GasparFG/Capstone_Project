"""
train_model.py

Two-stage workload forecasting model.

Stage 1:
    Activity classifier predicts whether a future 15-minute window will have workload activity.

Stage 2:
    Demand regressors estimate CPU demand, memory demand, duration, and job count
    for active workload windows.

Expected input:
    data/processed/sarima_ready_dataset.parquet

Expected outputs:
    models/forecasting/{job_type}_activity_classifier.pkl
    models/forecasting/{job_type}_{metric}_demand_regressor.pkl
    models/forecasting/{job_type}_forecast_metadata.pkl
"""

from pathlib import Path
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


INPUT_PATH = Path("data/processed/sarima_ready_dataset.parquet")
MODEL_DIR = Path("models/forecasting")

TIME_COLUMN = "forecast_timestamp"
GROUP_COLUMN = "job_type"

TARGET_COLUMNS = [
    "total_cpu_demand",
    "total_memory_demand",
    "median_duration_minutes",
    "job_count",
]

LAG_COLUMNS = TARGET_COLUMNS

LAGS = [1, 2, 4, 8]
ROLLING_WINDOWS = [4, 8]

ACTIVITY_THRESHOLD = 0.35


def load_dataset(input_path: Path) -> pd.DataFrame:
    """Load forecasting-ready dataset."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            "Run prepare_forecast_data.py first."
        )

    return pd.read_parquet(input_path)


def create_features(group_df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal, lag, rolling mean, and activity features."""

    df = group_df.copy()

    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])
    df = df.sort_values(TIME_COLUMN)

    df["hour"] = df[TIME_COLUMN].dt.hour
    df["minute"] = df[TIME_COLUMN].dt.minute
    df["day"] = df[TIME_COLUMN].dt.day
    df["day_of_week"] = df[TIME_COLUMN].dt.dayofweek

    df["is_active_window"] = (df["job_count"] > 0).astype(int)

    for column in LAG_COLUMNS:
        for lag in LAGS:
            df[f"{column}_lag_{lag}"] = df[column].shift(lag)

        for window in ROLLING_WINDOWS:
            df[f"{column}_rolling_mean_{window}"] = (
                df[column]
                .shift(1)
                .rolling(window=window)
                .mean()
            )

    df = df.dropna().reset_index(drop=True)

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature columns used for model training."""

    excluded_columns = [
        TIME_COLUMN,
        GROUP_COLUMN,
        "is_active_window",
        *TARGET_COLUMNS,
    ]

    return [column for column in df.columns if column not in excluded_columns]


def train_activity_classifier(
    X: pd.DataFrame,
    y: pd.Series,
) -> RandomForestClassifier:
    """Train active/inactive workload window classifier."""

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=8,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X, y)

    return model


def train_demand_regressor(
    X: pd.DataFrame,
    y: pd.Series,
) -> RandomForestRegressor:
    """Train conservative demand regressor to reduce extreme spikes."""

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
    )

    model.fit(X, y)

    return model


def save_pickle(object_to_save, output_path: Path) -> None:
    """Save model or metadata as pickle file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as file:
        pickle.dump(object_to_save, file)


def main() -> None:
    """Train two-stage forecasting models by workload type."""

    df = load_dataset(INPUT_PATH)

    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])
    df = df.sort_values([GROUP_COLUMN, TIME_COLUMN])

    for job_type, group_df in df.groupby(GROUP_COLUMN):

        print(f"\nTraining two-stage forecasting models for: {job_type}")

        feature_df = create_features(group_df)

        if len(feature_df) < 20:
            print(f"Skipping {job_type}: not enough observations.")
            continue

        feature_columns = get_feature_columns(feature_df)

        X = feature_df[feature_columns]
        y_activity = feature_df["is_active_window"]

        activity_model = train_activity_classifier(
            X=X,
            y=y_activity,
        )

        activity_model_path = MODEL_DIR / f"{job_type}_activity_classifier.pkl"
        save_pickle(activity_model, activity_model_path)

        print(f"Saved activity classifier: {activity_model_path}")

        active_df = feature_df[feature_df["is_active_window"] == 1].copy()

        if len(active_df) < 10:
            print(
                f"Skipping demand regressors for {job_type}: "
                "not enough active windows."
            )
            continue

        X_active = active_df[feature_columns]

        for target_column in TARGET_COLUMNS:

            y_target = active_df[target_column].clip(lower=0)

            demand_model = train_demand_regressor(
                X=X_active,
                y=y_target,
            )

            demand_model_path = (
                MODEL_DIR
                / f"{job_type}_{target_column}_demand_regressor.pkl"
            )

            save_pickle(demand_model, demand_model_path)

            print(f"Saved demand regressor: {demand_model_path}")

        metadata = {
            "feature_columns": feature_columns,
            "lags": LAGS,
            "rolling_windows": ROLLING_WINDOWS,
            "target_columns": TARGET_COLUMNS,
            "activity_threshold": ACTIVITY_THRESHOLD,
        }

        metadata_path = MODEL_DIR / f"{job_type}_forecast_metadata.pkl"
        save_pickle(metadata, metadata_path)

        print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()