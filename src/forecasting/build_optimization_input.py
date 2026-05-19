"""
build_optimization_input.py

This script converts forecast predictions into an optimization-ready dataset.

Expected input:
    data/processed/forecast_predictions.parquet

Expected output:
    data/processed/optimization_input_dataset.parquet
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

    This dataset will later be used by the optimization model
    for workload scheduling and resource allocation decisions.
    """

    optimization_df = df.copy()

    optimization_df["forecast_window"] = range(
        1,
        len(optimization_df) + 1,
    )

    optimization_df["predicted_total_utilization"] = (
        optimization_df["predicted_cpu_utilization"]
        + optimization_df["predicted_ram_utilization"]
    ) / 2

    optimization_df["workload_priority"] = "medium"

    optimization_df["server_group"] = "default_cluster"

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

    print(optimization_df.head())
    print(f"\nOptimization dataset saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()