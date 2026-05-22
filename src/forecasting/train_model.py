"""
train_model.py

This script trains SARIMA models for workload demand forecasting.

Expected input:
    data/processed/sarima_ready_dataset.parquet

Expected outputs:
    models/forecasting/{job_type}_{metric}_sarima_model.pkl
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


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

SARIMA_ORDER = (1, 1, 1)
SEASONAL_ORDER = (0, 0, 0, 0)


def load_dataset(input_path: Path) -> pd.DataFrame:
    """Load SARIMA-ready dataset."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            "Run prepare_forecast_data.py first."
        )

    return pd.read_parquet(input_path)


def train_sarima_model(series: pd.Series):
    """Train a SARIMA model using log1p transformation."""
    transformed_series = np.log1p(series.clip(lower=0))

    model = SARIMAX(
        transformed_series,
        order=SARIMA_ORDER,
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fitted_model = model.fit(disp=False)
    return fitted_model


def save_model(model, output_path: Path) -> None:
    """Save trained model as a pickle file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as file:
        pickle.dump(model, file)


def main() -> None:
    """Train SARIMA models by job_type and forecasting metric."""
    df = load_dataset(INPUT_PATH)

    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])
    df = df.sort_values([GROUP_COLUMN, TIME_COLUMN])

    for job_type, group_df in df.groupby(GROUP_COLUMN):
        group_df = group_df.set_index(TIME_COLUMN).sort_index()

        for target_column in TARGET_COLUMNS:
            series = group_df[target_column]

            if series.dropna().shape[0] < 10:
                print(
                    f"Skipping {job_type} - {target_column}: "
                    "not enough observations."
                )
                continue

            fitted_model = train_sarima_model(series)

            model_path = MODEL_DIR / f"{job_type}_{target_column}_sarima_model.pkl"
            save_model(fitted_model, model_path)

            print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()