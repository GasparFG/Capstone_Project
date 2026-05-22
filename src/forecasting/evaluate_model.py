"""
evaluate_model.py

Evaluates the two-stage workload forecasting model.

Stage 1:
    Activity classification:
        active vs inactive 15-minute window.

Stage 2:
    Conditional demand regression:
        CPU demand, memory demand, duration, and job count.

Expected input:
    data/processed/sarima_ready_dataset.parquet

Expected output:
    results/forecasting/forecast_metrics.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)


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

LAG_COLUMNS = TARGET_COLUMNS

LAGS = [1, 2, 4, 8]
ROLLING_WINDOWS = [4, 8]

TRAIN_SPLIT = 0.8
ACTIVITY_THRESHOLD = 0.35


def load_dataset(input_path: Path) -> pd.DataFrame:
    """Load forecasting-ready dataset."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            "Run prepare_forecast_data.py first."
        )

    return pd.read_parquet(input_path)


def create_features(group_df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal, lag, rolling mean, and activity features."""

    df = group_df.copy()

    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])
    df = df.sort_values(TIME_COLUMN)

    df["hour"] = df[TIME_COLUMN].dt.hour
    df["minute"] = df[TIME_COLUMN].dt.minute
    df["day"] = df[TIME_COLUMN].dt.day
    df["day_of_week"] = df[TIME_COLUMN].dt.dayofweek

    df["is_active_window"] = (df["job_count"] > 0).astype(int)

    for column in LAG_COLUMNS:
        for lag in LAGS:
            df[f"{column}_lag_{lag}"] = df[column].shift(lag)

        for window in ROLLING_WINDOWS:
            df[f"{column}_rolling_mean_{window}"] = (
                df[column]
                .shift(1)
                .rolling(window=window)
                .mean()
            )

    df = df.dropna().reset_index(drop=True)

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return model feature columns."""

    excluded_columns = [
        TIME_COLUMN,
        GROUP_COLUMN,
        "is_active_window",
        *TARGET_COLUMNS,
    ]

    return [column for column in df.columns if column not in excluded_columns]


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

    return (
        np.mean(
            np.abs(y_true[valid_mask] - y_pred[valid_mask])
            / denominator[valid_mask]
        )
        * 100
    )


def train_activity_classifier(X_train, y_train) -> RandomForestClassifier:
    """Train active/inactive window classifier."""

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=8,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    return model


def train_demand_regressor(X_train, y_train) -> RandomForestRegressor:
    """Train conservative demand regression model."""

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_job_type(
    job_type: str,
    group_df: pd.DataFrame,
) -> list[dict]:
    """Evaluate one workload type using temporal train/test split."""

    feature_df = create_features(group_df)

    split_index = int(len(feature_df) * TRAIN_SPLIT)

    train_df = feature_df.iloc[:split_index].copy()
    test_df = feature_df.iloc[split_index:].copy()

    feature_columns = get_feature_columns(feature_df)

    X_train = train_df[feature_columns]
    X_test = test_df[feature_columns]

    y_activity_train = train_df["is_active_window"]
    y_activity_test = test_df["is_active_window"]

    results = []

    activity_model = train_activity_classifier(
        X_train=X_train,
        y_train=y_activity_train,
    )

    activity_probabilities = activity_model.predict_proba(X_test)[:, 1]

    activity_predictions = (
        activity_probabilities >= ACTIVITY_THRESHOLD
    ).astype(int)

    activity_accuracy = accuracy_score(
        y_activity_test,
        activity_predictions,
    )

    activity_precision = precision_score(
        y_activity_test,
        activity_predictions,
        zero_division=0,
    )

    activity_recall = recall_score(
        y_activity_test,
        activity_predictions,
        zero_division=0,
    )

    activity_f1 = f1_score(
        y_activity_test,
        activity_predictions,
        zero_division=0,
    )

    results.append(
        {
            "job_type": job_type,
            "metric": "activity_classification",
            "MAE": np.nan,
            "RMSE": np.nan,
            "WAPE": np.nan,
            "SMAPE": np.nan,
            "accuracy": round(activity_accuracy, 4),
            "precision": round(activity_precision, 4),
            "recall": round(activity_recall, 4),
            "f1_score": round(activity_f1, 4),
            "status": "evaluated",
        }
    )

    active_train_df = train_df[train_df["is_active_window"] == 1].copy()

    if len(active_train_df) < 10:
        for target_column in TARGET_COLUMNS:
            results.append(
                {
                    "job_type": job_type,
                    "metric": target_column,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "WAPE": np.nan,
                    "SMAPE": np.nan,
                    "accuracy": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1_score": np.nan,
                    "status": "skipped_not_enough_active_windows",
                }
            )

        return results

    X_active_train = active_train_df[feature_columns]

    for target_column in TARGET_COLUMNS:

        y_active_train = active_train_df[target_column].clip(lower=0)

        demand_model = train_demand_regressor(
            X_train=X_active_train,
            y_train=y_active_train,
        )

        raw_predictions = demand_model.predict(X_test)
        raw_predictions = np.maximum(raw_predictions, 0)

        final_predictions = np.where(
            activity_predictions == 1,
            raw_predictions,
            0,
        )

        actuals = test_df[target_column].clip(lower=0).to_numpy()

        if target_column == "job_count":
            final_predictions = np.round(final_predictions).clip(min=0)

        mae = mean_absolute_error(actuals, final_predictions)
        rmse = np.sqrt(mean_squared_error(actuals, final_predictions))
        wape = calculate_wape(actuals, final_predictions)
        smape = calculate_smape(actuals, final_predictions)

        results.append(
            {
                "job_type": job_type,
                "metric": target_column,
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "WAPE": round(wape, 4),
                "SMAPE": round(smape, 4),
                "accuracy": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1_score": np.nan,
                "status": "evaluated",
            }
        )

    return results


def main() -> None:
    """Evaluate two-stage workload forecasting models."""

    df = load_dataset(INPUT_PATH)

    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])
    df = df.sort_values([GROUP_COLUMN, TIME_COLUMN])

    all_results = []

    for job_type, group_df in df.groupby(GROUP_COLUMN):

        print(f"Evaluating two-stage forecasting for: {job_type}")

        job_type_results = evaluate_job_type(
            job_type=job_type,
            group_df=group_df,
        )

        all_results.extend(job_type_results)

    metrics_df = pd.DataFrame(all_results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUTPUT_PATH, index=False)

    print("\nForecast evaluation metrics:")
    print(metrics_df)

    print(f"\nForecast metrics saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()