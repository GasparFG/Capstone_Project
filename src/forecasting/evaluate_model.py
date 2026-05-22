"""
evaluate_model.py

Evaluates the activity classification component of the two-stage workload
forecasting pipeline.

The forecasting problem was reframed as workload activity detection because
the workload traces are sparse and intermittent. Instead of prioritizing
point-value regression metrics, this script focuses on whether the model can
detect active workload windows.

Evaluation metrics:
    - Accuracy
    - Precision
    - Recall
    - F1 Score

Expected input:
    data/processed/sarima_ready_dataset.parquet

Expected output:
    results/forecasting/forecast_metrics.csv
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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
    """
    Create temporal, lag, rolling mean, and activity features.

    The activity target is defined as:
        1 = active workload window
        0 = inactive workload window
    """

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
    """Return feature columns used for activity classification."""

    excluded_columns = [
        TIME_COLUMN,
        GROUP_COLUMN,
        "is_active_window",
        *TARGET_COLUMNS,
    ]

    return [column for column in df.columns if column not in excluded_columns]


def train_activity_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """Train active/inactive workload window classifier."""

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=8,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    return model


def evaluate_activity_model(
    job_type: str,
    group_df: pd.DataFrame,
) -> dict:
    """Evaluate activity classification for one workload type."""

    feature_df = create_features(group_df)

    split_index = int(len(feature_df) * TRAIN_SPLIT)

    train_df = feature_df.iloc[:split_index].copy()
    test_df = feature_df.iloc[split_index:].copy()

    feature_columns = get_feature_columns(feature_df)

    X_train = train_df[feature_columns]
    X_test = test_df[feature_columns]

    y_train = train_df["is_active_window"]
    y_test = test_df["is_active_window"]

    model = train_activity_classifier(
        X_train=X_train,
        y_train=y_train,
    )

    activity_probabilities = model.predict_proba(X_test)[:, 1]

    y_pred = (
        activity_probabilities >= ACTIVITY_THRESHOLD
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    return {
        "job_type": job_type,
        "evaluation_focus": "activity_classification",
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "activity_threshold": ACTIVITY_THRESHOLD,
        "status": "evaluated",
    }


def main() -> None:
    """Evaluate activity classification models by workload type."""

    df = load_dataset(INPUT_PATH)

    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])

    df = df.sort_values(
        [GROUP_COLUMN, TIME_COLUMN]
    )

    results = []

    for job_type, group_df in df.groupby(GROUP_COLUMN):

        print(f"Evaluating activity classification for: {job_type}")

        result = evaluate_activity_model(
            job_type=job_type,
            group_df=group_df,
        )

        results.append(result)

    metrics_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nActivity classification evaluation metrics:")
    print(metrics_df)

    print(f"\nForecast metrics saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()