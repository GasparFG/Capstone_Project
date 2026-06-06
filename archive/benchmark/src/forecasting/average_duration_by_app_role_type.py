"""
average_duration_by_app_role_type.py

Baseline model:
Predict duration_minutes using the historical average duration by:

    app_name + role + job_type

This is not a machine learning regression model.
It is a grouped historical average baseline.

Input:
    data/processed/synthetic_clean_90_days.parquet

Outputs:
    data/forecast/average_duration_by_app_role_type_metrics.csv
    data/forecast/average_duration_by_app_role_type_predictions.csv
    data/forecast/average_duration_by_app_role_type_lookup.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


INPUT_PATH = Path("data/processed/synthetic_clean_90_days.parquet")
OUTPUT_DIR = Path("data/forecast")

METRICS_OUTPUT = OUTPUT_DIR / "average_duration_by_app_role_type_metrics.csv"
PREDICTIONS_OUTPUT = OUTPUT_DIR / "average_duration_by_app_role_type_predictions.csv"
LOOKUP_OUTPUT = OUTPUT_DIR / "average_duration_by_app_role_type_lookup.csv"

GROUP_COLUMNS = [
    "app_name",
    "role",
    "job_type",
]

TARGET_COLUMN = "duration_minutes"

MIN_GROUP_COUNT = 5
TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_data(input_path: Path) -> pd.DataFrame:
    """Load the processed workload dataset."""

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    return pd.read_parquet(input_path)


def prepare_data(workload_data: pd.DataFrame) -> pd.DataFrame:
    """Prepare columns needed for the grouped average baseline."""

    required_columns = GROUP_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column for column in required_columns
        if column not in workload_data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    model_data = workload_data[required_columns].copy()

    for column in GROUP_COLUMNS:
        model_data[column] = model_data[column].astype(str).fillna("unknown")

    model_data[TARGET_COLUMN] = pd.to_numeric(
        model_data[TARGET_COLUMN],
        errors="coerce",
    )

    model_data = model_data.replace([np.inf, -np.inf], np.nan)
    model_data = model_data.dropna(subset=[TARGET_COLUMN])
    model_data = model_data[model_data[TARGET_COLUMN] >= 0]

    return model_data


def build_average_lookup(train_data: pd.DataFrame) -> pd.DataFrame:
    """Build average duration lookup table by app_name + role + job_type."""

    lookup_table = (
        train_data
        .groupby(GROUP_COLUMNS, dropna=False)
        .agg(
            average_duration_minutes=(TARGET_COLUMN, "mean"),
            median_duration_minutes=(TARGET_COLUMN, "median"),
            group_count=(TARGET_COLUMN, "count"),
            min_duration_minutes=(TARGET_COLUMN, "min"),
            max_duration_minutes=(TARGET_COLUMN, "max"),
        )
        .reset_index()
    )

    lookup_table = lookup_table[lookup_table["group_count"] >= MIN_GROUP_COUNT]

    return lookup_table


def predict_with_group_average(
    test_data: pd.DataFrame,
    lookup_table: pd.DataFrame,
    global_average: float,
) -> pd.DataFrame:
    """Predict duration using grouped average. If group is missing, use global average."""

    predictions_data = test_data.merge(
        lookup_table,
        on=GROUP_COLUMNS,
        how="left",
    )

    predictions_data["prediction_source"] = np.where(
        predictions_data["average_duration_minutes"].notna(),
        "app_name_role_job_type_average",
        "global_average_fallback",
    )

    predictions_data["predicted_duration_minutes"] = (
        predictions_data["average_duration_minutes"]
        .fillna(global_average)
        .clip(lower=0)
    )

    predictions_data["actual_duration_minutes"] = predictions_data[TARGET_COLUMN]

    predictions_data["error"] = (
        predictions_data["actual_duration_minutes"]
        - predictions_data["predicted_duration_minutes"]
    )

    predictions_data["absolute_error"] = predictions_data["error"].abs()

    return predictions_data


def evaluate_predictions(predictions_data: pd.DataFrame) -> dict:
    """Calculate regression metrics."""

    y_true = predictions_data["actual_duration_minutes"]
    y_pred = predictions_data["predicted_duration_minutes"]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "model": "average_duration_by_app_name_role_job_type",
        "target": TARGET_COLUMN,
        "grouping": " + ".join(GROUP_COLUMNS),
        "min_group_count": MIN_GROUP_COUNT,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "test_rows": len(predictions_data),
        "predictions_from_group_average": int(
            (predictions_data["prediction_source"] == "app_name_role_job_type_average").sum()
        ),
        "predictions_from_global_average": int(
            (predictions_data["prediction_source"] == "global_average_fallback").sum()
        ),
    }


def save_outputs(
    metrics: dict,
    predictions_data: pd.DataFrame,
    lookup_table: pd.DataFrame,
) -> None:
    """Save metrics, predictions, and lookup table."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame([metrics])

    try:
        metrics_df.to_csv(METRICS_OUTPUT, index=False)
        predictions_data.to_csv(PREDICTIONS_OUTPUT, index=False)
        lookup_table.to_csv(LOOKUP_OUTPUT, index=False)

    except PermissionError as error:
        raise PermissionError(
            "One of the output CSV files is open or locked. "
            "Close Excel/OneDrive preview and run the script again."
        ) from error

    print(f"Metrics saved to: {METRICS_OUTPUT}")
    print(f"Predictions saved to: {PREDICTIONS_OUTPUT}")
    print(f"Lookup table saved to: {LOOKUP_OUTPUT}")


def main() -> None:
    print("Step 1: Loading data...")
    workload_data = load_data(INPUT_PATH)

    print("Step 2: Preparing data...")
    model_data = prepare_data(workload_data)

    print(f"Rows available for modeling: {len(model_data):,}")
    print(f"Unique app_name values: {model_data['app_name'].nunique():,}")
    print(f"Unique role values: {model_data['role'].nunique():,}")
    print(f"Unique job_type values: {model_data['job_type'].nunique():,}")

    print("Step 3: Splitting train and test data...")
    train_data, test_data = train_test_split(
        model_data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"Train rows: {len(train_data):,}")
    print(f"Test rows: {len(test_data):,}")

    print("Step 4: Building average duration lookup table...")
    lookup_table = build_average_lookup(train_data)

    global_average = train_data[TARGET_COLUMN].mean()

    print(f"Groups available after minimum count filter: {len(lookup_table):,}")
    print(f"Global average duration: {global_average:.4f} minutes")

    print("Step 5: Predicting duration using grouped averages...")
    predictions_data = predict_with_group_average(
        test_data=test_data,
        lookup_table=lookup_table,
        global_average=global_average,
    )

    print("Step 6: Evaluating predictions...")
    metrics = evaluate_predictions(predictions_data)

    metrics["rows_used"] = len(model_data)
    metrics["train_rows"] = len(train_data)
    metrics["global_average_duration_minutes"] = global_average
    metrics["groups_available"] = len(lookup_table)

    print("Step 7: Saving outputs...")
    save_outputs(
        metrics=metrics,
        predictions_data=predictions_data,
        lookup_table=lookup_table,
    )

    print("\nAverage Duration Baseline Results")
    print("---------------------------------")
    print(f"Grouping: {metrics['grouping']}")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R2:   {metrics['r2']:.4f}")
    print(f"Predictions from grouped average: {metrics['predictions_from_group_average']:,}")
    print(f"Predictions from global fallback: {metrics['predictions_from_global_average']:,}")


if __name__ == "__main__":
    main()