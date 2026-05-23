"""
build_optimization_input.py

Converts forecast predictions into an optimization-ready dataset.

Expected input:
    data/processed/forecast_predictions.parquet

Expected outputs:
    data/processed/optimization_input_dataset.parquet
    results/forecasting/optimization_input_dataset.xlsx
"""

from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/processed/forecast_predictions.parquet")
OUTPUT_PATH = Path("data/processed/optimization_input_dataset.parquet")
EXCEL_OUTPUT_PATH = Path("results/forecasting/optimization_input_dataset.xlsx")

MAX_DURATION_SLOTS = 48


def load_predictions(input_path: Path) -> pd.DataFrame:
    """Load forecast predictions dataset."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            "Run generate_predictions.py first."
        )

    return pd.read_parquet(input_path)


def build_optimization_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Build the optimization-ready forecasting dataset."""

    optimization_df = df.copy()

    optimization_df = optimization_df.sort_values(
        ["job_type", "forecast_timestamp"]
    )

    optimization_df["forecast_window"] = (
        optimization_df.groupby("job_type").cumcount() + 1
    )

    optimization_df = optimization_df.rename(
        columns={
            "total_cpu_demand": "predicted_cpu_demand",
            "total_memory_demand": "predicted_memory_demand",
            "median_duration_minutes": "predicted_duration_minutes",
            "job_count": "predicted_job_count",
        }
    )

    optimization_df["predicted_job_count"] = (
        optimization_df["predicted_job_count"]
        .round()
        .clip(lower=0)
        .astype(int)
    )

    optimization_df["predicted_cpu_demand"] = (
        optimization_df["predicted_cpu_demand"]
        .clip(lower=0)
    )

    optimization_df["predicted_memory_demand"] = (
        optimization_df["predicted_memory_demand"]
        .clip(lower=0)
    )

    optimization_df["predicted_duration_minutes"] = (
        optimization_df["predicted_duration_minutes"]
        .clip(lower=0)
    )

    optimization_df["predicted_duration_slots"] = (
        optimization_df["predicted_duration_minutes"] / 15
    ).round().clip(lower=1, upper=MAX_DURATION_SLOTS).astype(int)

    optimization_df["workload_priority"] = optimization_df["job_type"].map(
        {
            "interactive": "high",
            "batch": "medium",
        }
    )

    optimization_df["deadline_type"] = optimization_df["job_type"].map(
        {
            "interactive": "hard",
            "batch": "soft",
        }
    )

    optimization_df["forecast_window_minutes"] = 15

    optimization_df["aggregate_resource_demand"] = (
        optimization_df["predicted_cpu_demand"]
        + optimization_df["predicted_memory_demand"]
    )

    optimization_df = optimization_df[
        [
            "job_type",
            "forecast_window",
            "forecast_timestamp",
            "forecast_window_minutes",
            "predicted_cpu_demand",
            "predicted_memory_demand",
            "aggregate_resource_demand",
            "predicted_duration_minutes",
            "predicted_duration_slots",
            "predicted_job_count",
            "workload_priority",
            "deadline_type",
        ]
    ]

    return optimization_df


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Save optimization-ready dataset as parquet."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def save_excel_report(df: pd.DataFrame, output_path: Path) -> None:
    """Save Excel report with data dictionary for easier interpretation."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    column_dictionary = pd.DataFrame(
        [
            [
                "job_type",
                "Workload category. Batch workloads are flexible; interactive workloads are latency-sensitive.",
            ],
            [
                "forecast_window",
                "Sequential 15-minute forecast window number within the 24-hour forecast horizon.",
            ],
            [
                "forecast_timestamp",
                "Timestamp associated with each forecast window.",
            ],
            [
                "forecast_window_minutes",
                "Length of each forecast window in minutes. Fixed at 15 minutes.",
            ],
            [
                "predicted_cpu_demand",
                "Forecasted aggregate CPU demand for the workload type during the window.",
            ],
            [
                "predicted_memory_demand",
                "Forecasted aggregate memory demand for the workload type during the window.",
            ],
            [
                "aggregate_resource_demand",
                "Simplified combined resource pressure indicator: predicted CPU demand plus predicted memory demand.",
            ],
            [
                "predicted_duration_minutes",
                "Forecasted workload processing duration in minutes.",
            ],
            [
                "predicted_duration_slots",
                "Forecasted duration converted into 15-minute scheduling slots. Capped to avoid unrealistic optimization durations.",
            ],
            [
                "predicted_job_count",
                "Forecasted number of jobs expected in the window.",
            ],
            [
                "workload_priority",
                "Priority level assigned from workload type. Interactive = high; batch = medium.",
            ],
            [
                "deadline_type",
                "Deadline behavior used by the optimization model. Interactive = hard; batch = soft.",
            ],
        ],
        columns=["Column", "Description"],
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Optimization Input", index=False)
        column_dictionary.to_excel(
            writer,
            sheet_name="Column Dictionary",
            index=False,
        )


def main() -> None:
    """Build optimization-ready forecasting dataset."""

    predictions_df = load_predictions(INPUT_PATH)

    optimization_df = build_optimization_dataset(predictions_df)

    save_dataset(optimization_df, OUTPUT_PATH)
    save_excel_report(optimization_df, EXCEL_OUTPUT_PATH)

    print(optimization_df.head(10))
    print(f"\nOptimization dataset saved to: {OUTPUT_PATH}")
    print(f"Excel report saved to: {EXCEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()