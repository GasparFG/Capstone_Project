"""
generate_predictions.py

Generates future workload forecasts using the two-stage forecasting approach.

Stage 1:
    Predict whether a future 15-minute window is active.

Stage 2:
    If active, estimate CPU demand, memory demand, duration, and job count.

Important:
    A recursive forecast can collapse to zero if earlier predicted windows are inactive.
    To prevent this, the script applies an activity continuity rule based on recent
    historical workload behavior.

Expected input:
    models/forecasting/*_activity_classifier.pkl
    models/forecasting/*_demand_regressor.pkl
    models/forecasting/*_forecast_metadata.pkl
    data/processed/sarima_ready_dataset.parquet

Expected output:
    data/processed/forecast_predictions.parquet
"""

from pathlib import Path
import pickle

import pandas as pd


INPUT_DATA_PATH = Path("data/processed/sarima_ready_dataset.parquet")
MODEL_DIR = Path("models/forecasting")
OUTPUT_PATH = Path("data/processed/forecast_predictions.parquet")

TIME_COLUMN = "forecast_timestamp"
GROUP_COLUMN = "job_type"

TARGET_COLUMNS = [
    "total_cpu_demand",
    "total_memory_demand",
    "median_duration_minutes",
    "job_count",
]

LAG_COLUMNS = TARGET_COLUMNS

FORECAST_STEPS = 96
FORECAST_FREQ = "15min"

MIN_ACTIVE_JOB_COUNT = 1

RECENT_ACTIVITY_WINDOW = 8
MIN_ACTIVITY_PROBABILITY_IF_RECENT_ACTIVITY = 0.45

MAX_DURATION_MINUTES = 720
MAX_DURATION_SLOTS = 48


def load_dataset(input_path: Path) -> pd.DataFrame:
    """Load forecasting-ready dataset."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            "Run prepare_forecast_data.py first."
        )

    return pd.read_parquet(input_path)


def load_pickle(input_path: Path):
    """Load model or metadata from pickle."""

    if not input_path.exists():
        raise FileNotFoundError(f"Missing file: {input_path}")

    with open(input_path, "rb") as file:
        return pickle.load(file)


def build_single_feature_row(
    history_df: pd.DataFrame,
    next_timestamp: pd.Timestamp,
    feature_columns: list[str],
    lags: list[int],
    rolling_windows: list[int],
) -> pd.DataFrame:
    """Build one feature row for recursive future prediction."""

    feature_row = {
        "hour": next_timestamp.hour,
        "minute": next_timestamp.minute,
        "day": next_timestamp.day,
        "day_of_week": next_timestamp.dayofweek,
    }

    for column in LAG_COLUMNS:
        for lag in lags:
            feature_row[f"{column}_lag_{lag}"] = (
                history_df[column].iloc[-lag]
                if len(history_df) >= lag
                else 0
            )

        for window in rolling_windows:
            feature_row[f"{column}_rolling_mean_{window}"] = (
                history_df[column].tail(window).mean()
                if len(history_df) >= window
                else history_df[column].mean()
            )

    feature_df = pd.DataFrame([feature_row])

    for column in feature_columns:
        if column not in feature_df.columns:
            feature_df[column] = 0

    feature_df = feature_df[feature_columns]

    return feature_df


def apply_activity_continuity_rule(
    activity_probability: float,
    history_df: pd.DataFrame,
) -> float:
    """
    Prevent recursive collapse to zero.

    If recent workload activity exists, the activity probability is lifted
    to a conservative minimum value. This avoids a chain where one zero
    prediction causes all following windows to become zero.
    """

    recent_activity = (
        history_df["job_count"]
        .tail(RECENT_ACTIVITY_WINDOW)
        .mean()
    )

    if recent_activity > 0:
        activity_probability = max(
            activity_probability,
            MIN_ACTIVITY_PROBABILITY_IF_RECENT_ACTIVITY,
        )

    return activity_probability


def postprocess_prediction(
    target_column: str,
    prediction: float,
) -> float:
    """Clean and constrain prediction values."""

    prediction = max(prediction, 0)

    if target_column == "job_count":
        prediction = max(
            round(prediction),
            MIN_ACTIVE_JOB_COUNT,
        )

    if target_column == "median_duration_minutes":
        prediction = min(prediction, MAX_DURATION_MINUTES)

    return prediction


def main() -> None:
    """Generate future workload forecasts."""

    df = load_dataset(INPUT_DATA_PATH)

    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])
    df = df.sort_values([GROUP_COLUMN, TIME_COLUMN])

    forecast_results = []

    for job_type, group_df in df.groupby(GROUP_COLUMN):

        print(f"Generating two-stage forecast for: {job_type}")

        group_df = group_df.sort_values(TIME_COLUMN).copy()

        metadata_path = MODEL_DIR / f"{job_type}_forecast_metadata.pkl"
        activity_model_path = MODEL_DIR / f"{job_type}_activity_classifier.pkl"

        metadata = load_pickle(metadata_path)
        activity_model = load_pickle(activity_model_path)

        feature_columns = metadata["feature_columns"]
        lags = metadata["lags"]
        rolling_windows = metadata["rolling_windows"]
        activity_threshold = metadata.get("activity_threshold", 0.35)

        demand_models = {}

        for target_column in TARGET_COLUMNS:
            model_path = (
                MODEL_DIR
                / f"{job_type}_{target_column}_demand_regressor.pkl"
            )

            demand_models[target_column] = load_pickle(model_path)

        history_df = group_df[
            [
                TIME_COLUMN,
                *TARGET_COLUMNS,
            ]
        ].copy()

        last_timestamp = history_df[TIME_COLUMN].max()

        future_rows = []

        for step in range(1, FORECAST_STEPS + 1):

            next_timestamp = last_timestamp + pd.Timedelta(
                minutes=15 * step
            )

            feature_df = build_single_feature_row(
                history_df=history_df,
                next_timestamp=next_timestamp,
                feature_columns=feature_columns,
                lags=lags,
                rolling_windows=rolling_windows,
            )

            activity_probability = activity_model.predict_proba(feature_df)[0][1]

            activity_probability = apply_activity_continuity_rule(
                activity_probability=activity_probability,
                history_df=history_df,
            )

            is_active = int(activity_probability >= activity_threshold)

            row = {
                GROUP_COLUMN: job_type,
                TIME_COLUMN: next_timestamp,
            }

            if is_active == 1:
                for target_column in TARGET_COLUMNS:

                    prediction = demand_models[target_column].predict(feature_df)[0]

                    prediction = postprocess_prediction(
                        target_column=target_column,
                        prediction=prediction,
                    )

                    row[target_column] = prediction
            else:
                for target_column in TARGET_COLUMNS:
                    row[target_column] = 0

            future_rows.append(row)

            history_update = {
                TIME_COLUMN: next_timestamp,
                **{
                    target_column: row[target_column]
                    for target_column in TARGET_COLUMNS
                },
            }

            history_df = pd.concat(
                [
                    history_df,
                    pd.DataFrame([history_update]),
                ],
                ignore_index=True,
            )

        forecast_results.append(pd.DataFrame(future_rows))

    final_forecast_df = pd.concat(
        forecast_results,
        ignore_index=True,
    )

    final_forecast_df["job_count"] = (
        final_forecast_df["job_count"]
        .round()
        .clip(lower=0)
        .astype(int)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    final_forecast_df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(final_forecast_df.head(10))
    print(f"\nForecast predictions saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()