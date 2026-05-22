"""
evaluate_model.py

This script evaluates SARIMA forecasting accuracy for workload demand.

Expected input:
    data/processed/sarima_ready_dataset.parquet

Expected output:
    results/forecasting/forecast_metrics.csv
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
    "total_cpu_demand",
    "total_memory_demand",
    "median_duration_minutes",
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


def calculate_wape(y_true, y_pred) -> float:
    """Calculate Weighted Absolute Percentage Error."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    denominator = np.sum(np.abs(y_true))

    if denominator == 0:
        return np.nan

    return np.sum(np.abs(y_true - y_pred)) / denominator * 100


def calculate_smape(y_true, y_pred) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    valid_mask = denominator != 0

    if valid_mask.sum() == 0:
        return np.nan

    return np.mean(
        np.abs(y_true[valid_mask] - y_pred[valid_mask])
        / denominator[valid_mask]
    ) * 100


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
            "WAPE": np.nan,
            "SMAPE": np.nan,
            "status": "skipped_not_enough_data",
        }

    transformed_train = np.log1p(train.clip(lower=0))

    model = SARIMAX(
        transformed_train,
        order=SARIMA_ORDER,
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fitted_model = model.fit(disp=False)

    transformed_predictions = fitted_model.get_forecast(
        steps=len(test)
    ).predicted_mean

    predictions = np.expm1(np.array(transformed_predictions))
    predictions = np.maximum(predictions, 0)

    predictions = np.nan_to_num(
        predictions,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    mae = mean_absolute_error(test, predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    wape = calculate_wape(test, predictions)
    smape = calculate_smape(test, predictions)

    return {
        "job_type": job_type,
        "metric": target_column,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "WAPE": round(wape, 4),
        "SMAPE": round(smape, 4),
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