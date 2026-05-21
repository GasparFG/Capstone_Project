"""
prepare_forecast_data.py

This script prepares the processed forecasting dataset for SARIMA modeling.

Expected input:
    data/interim/cleaned_data.parquet

Expected output:
    data/processed/sarima_ready_dataset.parquet
"""

from pathlib import Path
import pandas as pd


INPUT_PATH = Path("data/interim/cleaned_data.parquet")
OUTPUT_PATH = Path("data/processed/sarima_ready_dataset.parquet")


REQUIRED_COLUMNS = ["uid", "job_type", "start_timestamp", "end_timestamp", "duration_minutes", "cpu_usage", "mem_usage"]




def load_dataset(input_path: Path) -> pd.DataFrame:
    """Load the processed forecasting dataset."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            "This file should be created by the data cleaning and feature engineering branch."
        )

    return pd.read_parquet(input_path)


def validate_columns(df: pd.DataFrame) -> None:
    """Validate that the required columns exist."""
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Expected columns are: {REQUIRED_COLUMNS}"
        )


def prepare_time_series_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare time-series data for SARIMA.

    Steps:
    - Convert timestamp to datetime.
    - Sort records by timestamp.
    - Aggregate CPU and RAM utilization by 15-minute forecasting window.
    - Fill missing time windows using time interpolation.
    """

    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    hourly_df = (
        df.set_index("timestamp")
        .resample("15Min")
        .agg(
            cpu_utilization=("cpu_utilization", "mean"),
            ram_utilization=("ram_utilization", "mean"),
        )
    )

    hourly_df["cpu_utilization"] = hourly_df["cpu_utilization"].interpolate(method="time")
    hourly_df["ram_utilization"] = hourly_df["ram_utilization"].interpolate(method="time")

    hourly_df = hourly_df.dropna().reset_index()

    return hourly_df


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Save the SARIMA-ready dataset."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def main() -> None:
    """Run the forecasting data preparation pipeline."""
    df = load_dataset(INPUT_PATH)
    validate_columns(df)

    sarima_ready_df = prepare_time_series_data(df)
    save_dataset(sarima_ready_df, OUTPUT_PATH)

    print(f"SARIMA-ready dataset saved to: {OUTPUT_PATH}")
    print(f"Rows saved: {len(sarima_ready_df)}")


if __name__ == "__main__":
    main()