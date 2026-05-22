"""
evaluate_model.py

This script evaluates SARIMA forecasting accuracy for workload resource demand.

Expected input:
    data/processed/sarima_ready_dataset.parquet

Expected output:
    results/forecasting/forecast_metrics.csv

Evaluation approach:
    - Evaluate each job_type separately.
    - Evaluate CPU request, memory request, duration, and job count.
    - Use a time-based train/test split.
    - Calculate MAE, RMSE, and MAPE.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


INPUT_PATH = Path("data/processed/sarima_ready_dataset.parquet")
OUTPUT_PATH = Path("results/forecasting/forecast_metrics.csv")

TIME_COLUMN = "forecast_timestamp"
GROUP_COLUMN = "job_type"

TARGET_COLUMNS = [
    "avg_cpu_request",
    "avg_memory_request",
    "avg_duration_minutes",
    "job_count",
]

SARIMA_ORDER = (1, 1, 1)
SEASONAL_ORDER = (0, 0, 0, 0)

TRAIN_SPLIT = 0.8


def load_dataset(input_path: Path) -> pd.DataFrame:
    """Load SARIMA-ready dataset."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            "Run prepare_forecast_data.py first."
        )

    return pd.read_parquet(input_path)


def calculate_mape(y_true, y_pred) -> float:
    """Calculate Mean Absolute Percentage Error while avoiding division by zero."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    non_zero_mask = y_true != 0

    if non_zero_mask.sum() == 0:
        return np.nan

    return (
        np.mean(
            np.abs(
                (y_true[non_zero_mask] - y_pred[non_zero_mask])
                / y_true[non_zero_mask]
            )
        )
        * 100
    )


def evaluate_series(series: pd.Series, job_type: str, target_column: str) -> dict:
    """Train/test evaluate one time series."""

    split_index = int(len(series) * TRAIN_SPLIT)

    train = series.iloc[:split_index]
    test = series.iloc[split_index:]

    if len(train) < 10 or len(test) < 1:
        return {
            "job_type": job_type,
            "metric": target_column,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MAPE": np.nan,
            "status": "skipped_not_enough_data",
        }

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
        "job_type": job_type,
        "metric": target_column,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
        "status": "evaluated",
    }


def main() -> None:
    """Evaluate SARIMA models by job_type and metric."""

    df = load_dataset(INPUT_PATH)

    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])
    df = df.sort_values([GROUP_COLUMN, TIME_COLUMN])

    results = []

    for job_type, group_df in df.groupby(GROUP_COLUMN):
        group_df = group_df.set_index(TIME_COLUMN).sort_index()

        for target_column in TARGET_COLUMNS:
            result = evaluate_series(
                series=group_df[target_column],
                job_type=job_type,
                target_column=target_column,
            )

            results.append(result)

    metrics_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUTPUT_PATH, index=False)

    print(metrics_df)
    print(f"\nForecast metrics saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()