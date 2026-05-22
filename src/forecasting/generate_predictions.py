"""
generate_predictions.py

This script generates future workload forecasts using trained SARIMA models.

Expected input:
    models/forecasting/*.pkl
    data/processed/sarima_ready_dataset.parquet

Expected output:
    data/processed/forecast_predictions.parquet

Forecasting strategy:
    Hybrid forecasting:
        final_forecast =
            0.7 * SARIMA forecast
            + 0.3 * rolling historical mean

    A minimum job-count floor is applied to avoid unrealistic zero-demand
    forecasts in intermittent workload traces.
"""

from pathlib import Path
import pickle

import numpy as np
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

FORECAST_STEPS = 96

SARIMA_WEIGHT = 0.7
ROLLING_MEAN_WEIGHT = 0.3

ROLLING_WINDOW = 8
MIN_ACTIVE_JOB_COUNT = 1


def load_dataset(input_path: Path) -> pd.DataFrame:
    """Load SARIMA-ready dataset."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            "Run prepare_forecast_data.py first."
        )

    return pd.read_parquet(input_path)


def load_model(model_path: Path):
    """Load trained SARIMA model."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Run train_model.py first."
        )

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    return model


def generate_future_timestamps(last_timestamp, periods):
    """Generate future 15-minute forecasting timestamps."""
    return pd.date_range(
        start=last_timestamp + pd.Timedelta(minutes=15),
        periods=periods,
        freq="15min",
    )


def generate_sarima_forecast(model, steps: int) -> np.ndarray:
    """Generate forecast values using inverse log1p transformation."""

    transformed_predictions = model.get_forecast(
        steps=steps
    ).predicted_mean

    predictions = np.expm1(np.array(transformed_predictions))

    predictions = np.maximum(predictions, 0)

    predictions = np.nan_to_num(
        predictions,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return predictions


def generate_rolling_mean_forecast(
    historical_series: pd.Series,
    steps: int,
) -> np.ndarray:
    """Generate rolling mean baseline forecast."""

    rolling_mean = historical_series.tail(ROLLING_WINDOW).mean()
    rolling_mean = max(rolling_mean, 0)

    return np.full(
        shape=steps,
        fill_value=rolling_mean,
    )


def combine_forecasts(
    sarima_forecast: np.ndarray,
    rolling_forecast: np.ndarray,
) -> np.ndarray:
    """Combine SARIMA and rolling mean forecasts."""

    combined = (
        SARIMA_WEIGHT * sarima_forecast
        + ROLLING_MEAN_WEIGHT * rolling_forecast
    )

    combined = np.maximum(combined, 0)

    return combined


def apply_job_count_floor(
    forecast_values: np.ndarray,
    historical_series: pd.Series,
) -> np.ndarray:
    """
    Apply a minimum job-count floor when recent historical demand exists.

    This avoids unrealistic zero-job forecasts in intermittent workload traces.
    """

    recent_activity = historical_series.tail(ROLLING_WINDOW).sum()

    if recent_activity > 0:
        forecast_values = np.where(
            forecast_values > 0,
            np.maximum(forecast_values, MIN_ACTIVE_JOB_COUNT),
            forecast_values,
        )

    return forecast_values


def main() -> None:
    """Generate future workload forecasts."""

    df = load_dataset(INPUT_DATA_PATH)

    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])

    forecast_results = []

    for job_type, group_df in df.groupby(GROUP_COLUMN):

        group_df = group_df.sort_values(TIME_COLUMN)

        last_timestamp = group_df[TIME_COLUMN].max()

        future_timestamps = generate_future_timestamps(
            last_timestamp=last_timestamp,
            periods=FORECAST_STEPS,
        )

        forecast_df = pd.DataFrame(
            {
                GROUP_COLUMN: job_type,
                TIME_COLUMN: future_timestamps,
            }
        )

        for target_column in TARGET_COLUMNS:

            model_path = (
                MODEL_DIR
                / f"{job_type}_{target_column}_sarima_model.pkl"
            )

            model = load_model(model_path)

            historical_series = group_df[target_column]

            sarima_predictions = generate_sarima_forecast(
                model=model,
                steps=FORECAST_STEPS,
            )

            rolling_predictions = generate_rolling_mean_forecast(
                historical_series=historical_series,
                steps=FORECAST_STEPS,
            )

            hybrid_predictions = combine_forecasts(
                sarima_forecast=sarima_predictions,
                rolling_forecast=rolling_predictions,
            )

            if target_column == "job_count":
                hybrid_predictions = apply_job_count_floor(
                    forecast_values=hybrid_predictions,
                    historical_series=historical_series,
                )

            forecast_df[target_column] = hybrid_predictions

        forecast_results.append(forecast_df)

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