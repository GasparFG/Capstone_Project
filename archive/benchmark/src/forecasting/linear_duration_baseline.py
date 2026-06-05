"""
linear_duration_baseline.py

Simple baseline model:
Predict duration_minutes using only cpu_request and memory_request.

Input:
    data/processed/synthetic_clean_90_days.parquet

Outputs:
    data/forecast/linear_duration_baseline_metrics.csv
    data/forecast/linear_duration_baseline_predictions.csv
"""

from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


INPUT_PATH = Path("data/processed/synthetic_clean_90_days.parquet")
OUTPUT_DIR = Path("data/forecast")

METRICS_OUTPUT = OUTPUT_DIR / "linear_duration_baseline_metrics.csv"
PREDICTIONS_OUTPUT = OUTPUT_DIR / "linear_duration_baseline_predictions.csv"

FEATURE_COLUMNS = [
    "cpu_request",
    "memory_request",
]

TARGET_COLUMN = "duration_minutes"


def load_data(input_path: Path) -> pd.DataFrame:
    """Load the processed dataset."""

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    return pd.read_parquet(input_path)


def prepare_data(workload_data: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns needed for the linear baseline."""

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column for column in required_columns
        if column not in workload_data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    model_data = workload_data[required_columns].copy()

    for column in required_columns:
        model_data[column] = pd.to_numeric(model_data[column], errors="coerce")

    model_data = model_data.replace([np.inf, -np.inf], np.nan)
    model_data = model_data.dropna()

    model_data = model_data[model_data[TARGET_COLUMN] >= 0]

    return model_data


def train_linear_model(model_data: pd.DataFrame):
    """Train and evaluate a linear regression model."""

    features = model_data[FEATURE_COLUMNS]
    target = model_data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Duration cannot be negative, so we clip predictions at 0.
    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    metrics = {
        "model": "linear_regression_baseline",
        "target": TARGET_COLUMN,
        "features": ", ".join(FEATURE_COLUMNS),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "intercept": model.intercept_,
        "cpu_coefficient": model.coef_[0],
        "memory_coefficient": model.coef_[1],
        "rows_used": len(model_data),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }

    predictions_df = X_test.copy()
    predictions_df["actual_duration_minutes"] = y_test.values
    predictions_df["predicted_duration_minutes"] = predictions
    predictions_df["error"] = (
        predictions_df["actual_duration_minutes"]
        - predictions_df["predicted_duration_minutes"]
    )
    predictions_df["absolute_error"] = predictions_df["error"].abs()

    return metrics, predictions_df


def save_outputs(metrics: dict, predictions_df: pd.DataFrame) -> None:
    """Save metrics and predictions."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame([metrics])

    metrics_df.to_csv(METRICS_OUTPUT, index=False)
    predictions_df.to_csv(PREDICTIONS_OUTPUT, index=False)

    print(f"Metrics saved to: {METRICS_OUTPUT}")
    print(f"Predictions saved to: {PREDICTIONS_OUTPUT}")


def main() -> None:
    print("Step 1: Loading data...")
    workload_data = load_data(INPUT_PATH)

    print("Step 2: Preparing data...")
    model_data = prepare_data(workload_data)

    print(f"Rows available for modeling: {len(model_data):,}")

    print("Step 3: Training linear regression baseline...")
    metrics, predictions_df = train_linear_model(model_data)

    print("Step 4: Saving outputs...")
    save_outputs(metrics, predictions_df)

    print("\nLinear Regression Baseline Results")
    print("----------------------------------")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R2:   {metrics['r2']:.4f}")
    print(f"Intercept: {metrics['intercept']:.4f}")
    print(f"CPU coefficient: {metrics['cpu_coefficient']:.4f}")
    print(f"Memory coefficient: {metrics['memory_coefficient']:.4f}")


if __name__ == "__main__":
    main()