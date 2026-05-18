"""
evaluate_model.py

This script evaluates SARIMA forecasting accuracy for CPU and RAM utilization.

Expected input:
    data/processed/sarima_ready_dataset.parquet

Expected outputs:
    results/forecasting/forecast_metrics.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from statsmodels.tsa.statespace.sarimax import SARIMAX


INPUT_PATH = Path("data/processed/sarima_ready_dataset.parquet")
OUTPUT_PATH = Path("results/forecasting/forecast_metrics.csv")

SARIMA_ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 1, 1, 24)

TRAIN_SPLIT = 0.8


def load_dataset(input_path: Path) -> pd.DataFrame:
    """Load SARIMA-ready dataset."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            "Run prepare_forecast_data.py first."
        )

    return pd.read_parquet(input_path)


def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    non_zero_mask = y_true != 0

    return (
        np.mean(
            np.abs(
                (y_true[non_zero_mask] - y_pred[non_zero_mask])
                / y_true[non_zero_mask]
            )
        )
        * 100
    )


def evaluate_series(series: pd.Series, metric_name: str) -> dict:
    """Train/test evaluate one time series."""

    split_index = int(len(series) * TRAIN_SPLIT)

    train = series.iloc[:split_index]
    test = series.iloc[split_index:]

    model = SARIMAX(
        train,
        order=SARIMA_ORDER,
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fitted_model = model.fit(disp=False)

    predictions = fitted_model.forecast(steps=len(test))

    mae = mean_absolute_error(test, predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    mape = calculate_mape(test, predictions)

    return {
        "metric": metric_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
    }


def main() -> None:
    """Evaluate CPU and RAM SARIMA forecasting accuracy."""

    df = load_dataset(INPUT_PATH)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    cpu_results = evaluate_series(
        df["cpu_utilization"],
        "cpu_utilization",
    )

    ram_results = evaluate_series(
        df["ram_utilization"],
        "ram_utilization",
    )

    metrics_df = pd.DataFrame([cpu_results, ram_results])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUTPUT_PATH, index=False)

    print(metrics_df)
    print(f"\nForecast metrics saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()