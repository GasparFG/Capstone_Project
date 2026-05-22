"""
generate_predictions.py

This script generates future workload forecasts using trained SARIMA models.

Expected input:
    models/forecasting/*.pkl
    data/processed/sarima_ready_dataset.parquet

Expected output:
    data/processed/forecast_predictions.parquet

Forecasting approach:
    - Generate future forecasts by job_type.
    - Predict:
        * CPU demand
        * Memory demand
        * Duration
        * Job count
    - Use 15-minute forecasting windows.
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
    "avg_cpu_request",
    "avg_memory_request",
    "avg_duration_minutes",
    "job_count",
]

FORECAST_STEPS = 24


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


def generate_forecast(model, steps: int) -> np.ndarray:
    """
    Generate forecast values safely.

    This function uses get_forecast().predicted_mean instead of forecast()
    to improve compatibility with SARIMA model results.

    It also replaces NaN and infinite values with 0 to avoid breaking
    downstream optimization inputs.
    """

    predictions = model.get_forecast(
        steps=steps
    ).predicted_mean

    predictions = np.array(predictions)

    predictions = np.maximum(predictions, 0)

    predictions = np.nan_to_num(
        predictions,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return predictions


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

            predictions = generate_forecast(
                model=model,
                steps=FORECAST_STEPS,
            )

            forecast_df[target_column] = predictions

        forecast_results.append(forecast_df)

    final_forecast_df = pd.concat(
        forecast_results,
        ignore_index=True,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    final_forecast_df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(final_forecast_df.head(10))

    print(
        f"\nForecast predictions saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()