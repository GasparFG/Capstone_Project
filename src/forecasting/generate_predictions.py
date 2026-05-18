"""
generate_predictions.py

This script generates future workload forecasts using trained SARIMA models.

Expected inputs:
    data/processed/sarima_ready_dataset.parquet
    models/forecasting/cpu_sarima_model.pkl
    models/forecasting/ram_sarima_model.pkl

Expected output:
    data/processed/forecast_predictions.parquet
"""

from pathlib import Path
import pickle

import pandas as pd


DATA_PATH = Path("data/processed/sarima_ready_dataset.parquet")
CPU_MODEL_PATH = Path("models/forecasting/cpu_sarima_model.pkl")
RAM_MODEL_PATH = Path("models/forecasting/ram_sarima_model.pkl")
OUTPUT_PATH = Path("data/processed/forecast_predictions.parquet")

FORECAST_STEPS = 24
FREQUENCY = "1H"


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load SARIMA-ready dataset."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {data_path}. "
            "Run prepare_forecast_data.py first."
        )

    return pd.read_parquet(data_path)


def load_model(model_path: Path):
    """Load trained SARIMA model."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Run train_model.py first."
        )

    with open(model_path, "rb") as file:
        return pickle.load(file)


def generate_future_timestamps(
    last_timestamp: pd.Timestamp,
    forecast_steps: int,
    frequency: str,
) -> pd.DatetimeIndex:
    """Generate future timestamps for forecast horizon."""
    return pd.date_range(
        start=last_timestamp + pd.Timedelta(hours=1),
        periods=forecast_steps,
        freq=frequency,
    )


def main() -> None:
    """Generate CPU and RAM utilization forecasts."""

    df = load_dataset(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    cpu_model = load_model(CPU_MODEL_PATH)
    ram_model = load_model(RAM_MODEL_PATH)

    last_timestamp = df["timestamp"].max()

    future_timestamps = generate_future_timestamps(
        last_timestamp=last_timestamp,
        forecast_steps=FORECAST_STEPS,
        frequency=FREQUENCY,
    )

    cpu_forecast = cpu_model.forecast(steps=FORECAST_STEPS)
    ram_forecast = ram_model.forecast(steps=FORECAST_STEPS)

    predictions_df = pd.DataFrame(
        {
            "timestamp": future_timestamps,
            "predicted_cpu_utilization": cpu_forecast.values,
            "predicted_ram_utilization": ram_forecast.values,
        }
    )

    predictions_df["predicted_cpu_utilization"] = predictions_df[
        "predicted_cpu_utilization"
    ].clip(lower=0, upper=100)

    predictions_df["predicted_ram_utilization"] = predictions_df[
        "predicted_ram_utilization"
    ].clip(lower=0, upper=100)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_parquet(OUTPUT_PATH, index=False)

    print(predictions_df.head())
    print(f"\nForecast predictions saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()