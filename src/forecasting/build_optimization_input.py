"""
build_optimization_input.py

This script converts forecast predictions into an optimization-ready dataset.

Expected input:
    data/processed/forecast_predictions.parquet

Expected output:
    data/processed/optimization_input_dataset.parquet

Optimization-ready approach:
    - Use forecasted workload demand by job_type.
    - Convert forecast timestamps into ordered forecast windows.
    - Rename forecast columns into optimization-friendly names.
    - Add priority and resource demand fields required by the optimization model.
"""

from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/processed/forecast_predictions.parquet")
OUTPUT_PATH = Path("data/processed/optimization_input_dataset.parquet")


def load_predictions(input_path: Path) -> pd.DataFrame:
    """Load forecast predictions dataset."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            "Run generate_predictions.py first."
        )

    return pd.read_parquet(input_path)


def build_optimization_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build optimization-ready dataset.

    This dataset represents expected workload resource demand
    by job type and 15-minute forecast window.
    """

    optimization_df = df.copy()

    optimization_df = optimization_df.sort_values(
        ["job_type", "forecast_timestamp"]
    )

    optimization_df["forecast_window"] = (
        optimization_df.groupby("job_type").cumcount() + 1
    )

    optimization_df = optimization_df.rename(
        columns={
            "avg_cpu_request": "predicted_cpu_request",
            "avg_memory_request": "predicted_memory_request",
            "avg_duration_minutes": "predicted_duration_minutes",
            "job_count": "predicted_job_count",
        }
    )

    optimization_df["predicted_job_count"] = (
        optimization_df["predicted_job_count"]
        .round()
        .clip(lower=0)
        .astype(int)
    )

    optimization_df["predicted_cpu_request"] = (
        optimization_df["predicted_cpu_request"]
        .clip(lower=0)
    )

    optimization_df["predicted_memory_request"] = (
        optimization_df["predicted_memory_request"]
        .clip(lower=0)
    )

    optimization_df["predicted_duration_minutes"] = (
        optimization_df["predicted_duration_minutes"]
        .clip(lower=0)
    )

    optimization_df["workload_priority"] = optimization_df["job_type"].map(
        {
            "interactive": "high",
            "batch": "medium",
        }
    )

    optimization_df["forecast_window_minutes"] = 15

    optimization_df = optimization_df[
        [
            "job_type",
            "forecast_window",
            "forecast_timestamp",
            "forecast_window_minutes",
            "predicted_cpu_request",
            "predicted_memory_request",
            "predicted_duration_minutes",
            "predicted_job_count",
            "workload_priority",
        ]
    ]

    return optimization_df


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Save optimization-ready dataset."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, index=False)


def main() -> None:
    """Build optimization-ready forecasting dataset."""

    predictions_df = load_predictions(INPUT_PATH)

    optimization_df = build_optimization_dataset(predictions_df)

    save_dataset(optimization_df, OUTPUT_PATH)

    print(optimization_df.head(10))
    print(f"\nOptimization dataset saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()