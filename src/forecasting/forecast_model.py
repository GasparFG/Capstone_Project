"""
forecast_model.py

Final job-level forecasting pipeline in SECONDS.

Key decisions:
    1. One row = one forecasted job.
    2. All time variables are handled in seconds.
    3. CPU and memory are modeled with classification because they are repeated discrete request values.
    4. Duration is modeled with log-regression because it is right-skewed.
    5. gpu_request is preserved as gpu_request.
    6. Forecast horizon is 24 hours (86,400 seconds).

Input:
    data/interim/cleaned_data.parquet

Outputs:
    data/forecast/good_job_level_model_metrics.csv
    data/forecast/good_job_level_forecast.csv
    data/forecast/good_job_level_forecast.parquet
    data/forecast/good_job_level_forecast_validation.csv
    data/processed/optimization_forecast_jobs.parquet
"""

from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor


INPUT_PARQUET = Path("data/interim/cleaned_data.parquet")

FORECAST_DIR = Path("data/forecast")
PROCESSED_DIR = Path("data/processed")

METRICS_OUTPUT = FORECAST_DIR / "good_job_level_model_metrics.csv"
VALIDATION_OUTPUT = FORECAST_DIR / "good_job_level_forecast_validation.csv"

FORECAST_CSV_OUTPUT = FORECAST_DIR / "good_job_level_forecast.csv"
FORECAST_PARQUET_OUTPUT = FORECAST_DIR / "good_job_level_forecast.parquet"

OPTIMIZATION_PARQUET_OUTPUT = PROCESSED_DIR / "optimization_forecast_jobs.parquet"

FORECAST_HORIZON_SECONDS = 86_400
MAX_FORECAST_JOBS = 10_000
RANDOM_STATE = 42


def discrete_accuracy(y_true, y_pred) -> float:
    true_values = pd.Series(y_true).astype(float).round(6).astype(str)
    pred_values = pd.Series(y_pred).astype(float).round(6).astype(str)
    return accuracy_score(true_values, pred_values)


def load_and_prepare_data() -> pd.DataFrame:
    data = pd.read_parquet(INPUT_PARQUET).copy()

    required = [
        "instance_sn",
        "role",
        "app_name",
        "job_type",
        "gpu_request",
        "scheduled_time",
        "cpu_request",
        "memory_request",
        "duration_minutes",
    ]

    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if not pd.api.types.is_timedelta64_dtype(data["scheduled_time"]):
        data["scheduled_time"] = pd.to_timedelta(data["scheduled_time"], unit="s")

    data["scheduled_seconds"] = data["scheduled_time"].dt.total_seconds()

    if "duration_seconds" not in data.columns:
        data["duration_seconds"] = data["duration_minutes"] * 60
    else:
        data["duration_seconds"] = pd.to_numeric(
            data["duration_seconds"],
            errors="coerce",
        )

    for col in [
        "scheduled_seconds",
        "gpu_request",
        "cpu_request",
        "memory_request",
        "duration_seconds",
    ]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(
        subset=[
            "scheduled_seconds",
            "gpu_request",
            "cpu_request",
            "memory_request",
            "duration_seconds",
            "role",
            "app_name",
            "job_type",
        ]
    )

    data = data[
        (data["cpu_request"] > 0)
        & (data["memory_request"] > 0)
        & (data["duration_seconds"] > 0)
    ].copy()

    data["gpu_request"] = data["gpu_request"].astype(int)

    data["role"] = data["role"].astype(str).str.lower().str.strip()
    data["app_name"] = data["app_name"].astype(str).str.lower().str.strip()
    data["job_type"] = data["job_type"].astype(str).str.lower().str.strip()

    data = data.sort_values("scheduled_seconds").reset_index(drop=True)

    data["interarrival_seconds"] = (
        data["scheduled_seconds"].diff().fillna(0).clip(lower=0)
    )

    data["time_of_day_seconds"] = data["scheduled_seconds"] % 86_400
    data["hour"] = (data["time_of_day_seconds"] // 3600).astype(int)
    data["minute"] = ((data["time_of_day_seconds"] % 3600) // 60).astype(int)
    data["second"] = (data["time_of_day_seconds"] % 60).astype(int)

    data["hour_sin"] = np.sin(2 * np.pi * data["time_of_day_seconds"] / 86_400)
    data["hour_cos"] = np.cos(2 * np.pi * data["time_of_day_seconds"] / 86_400)

    for col in [
        "cpu_request",
        "memory_request",
        "duration_seconds",
        "interarrival_seconds",
    ]:
        data[f"{col}_lag_1"] = data[col].shift(1)
        data[f"{col}_lag_2"] = data[col].shift(2)
        data[f"{col}_lag_5"] = data[col].shift(5)
        data[f"{col}_rolling_mean_5"] = data[col].shift(1).rolling(5).mean()

    data = data.dropna().reset_index(drop=True)

    return data


def temporal_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = int(len(data) * 0.80)
    train = data.iloc[:split].copy()
    test = data.iloc[split:].copy()
    return train, test


def make_preprocessor(
    categorical_features: list[str],
    numeric_features: list[str],
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
            ("numeric", "passthrough", numeric_features),
        ]
    )


def train_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    categorical_features: list[str],
    numeric_features: list[str],
) -> tuple[dict, list[dict]]:

    X_train = train[categorical_features + numeric_features]
    X_test = test[categorical_features + numeric_features]

    preprocessor = make_preprocessor(categorical_features, numeric_features)

    models = {}
    metrics = []

    cpu_encoder = LabelEncoder()
    y_train_cpu = cpu_encoder.fit_transform(train["cpu_request"].astype(str))

    cpu_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBClassifier(
                    n_estimators=150,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.90,
                    colsample_bytree=0.90,
                    eval_metric="mlogloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    cpu_model.fit(X_train, y_train_cpu)

    cpu_pred_encoded = cpu_model.predict(X_test)
    cpu_pred = cpu_encoder.inverse_transform(cpu_pred_encoded).astype(float)

    metrics.append(
        {
            "model": "XGBoost classifier",
            "target": "cpu_request",
            "metric_space": "classification_discrete_values",
            "accuracy": discrete_accuracy(test["cpu_request"], cpu_pred),
            "mae": mean_absolute_error(test["cpu_request"], cpu_pred),
            "rmse": np.sqrt(mean_squared_error(test["cpu_request"], cpu_pred)),
            "r2": r2_score(test["cpu_request"], cpu_pred),
        }
    )

    models["cpu_model"] = cpu_model
    models["cpu_encoder"] = cpu_encoder

    memory_encoder = LabelEncoder()
    y_train_memory = memory_encoder.fit_transform(
        train["memory_request"].astype(str)
    )

    memory_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBClassifier(
                    n_estimators=150,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.90,
                    colsample_bytree=0.90,
                    eval_metric="mlogloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    memory_model.fit(X_train, y_train_memory)

    memory_pred_encoded = memory_model.predict(X_test)
    memory_pred = memory_encoder.inverse_transform(memory_pred_encoded).astype(float)

    metrics.append(
        {
            "model": "XGBoost classifier",
            "target": "memory_request",
            "metric_space": "classification_discrete_values",
            "accuracy": discrete_accuracy(test["memory_request"], memory_pred),
            "mae": mean_absolute_error(test["memory_request"], memory_pred),
            "rmse": np.sqrt(mean_squared_error(test["memory_request"], memory_pred)),
            "r2": r2_score(test["memory_request"], memory_pred),
        }
    )

    models["memory_model"] = memory_model
    models["memory_encoder"] = memory_encoder

    duration_numeric_features = numeric_features + [
        "cpu_request",
        "memory_request",
    ]

    duration_features = categorical_features + duration_numeric_features

    duration_preprocessor = make_preprocessor(
        categorical_features=categorical_features,
        numeric_features=duration_numeric_features,
    )

    duration_model = Pipeline(
        steps=[
            ("preprocessor", duration_preprocessor),
            (
                "model",
                XGBRegressor(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.04,
                    subsample=0.90,
                    colsample_bytree=0.90,
                    objective="reg:squarederror",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    y_train_duration_log = np.log1p(train["duration_seconds"])
    y_test_duration_log = np.log1p(test["duration_seconds"])

    duration_model.fit(train[duration_features], y_train_duration_log)

    duration_pred_log = duration_model.predict(test[duration_features])
    duration_pred = np.expm1(duration_pred_log).clip(min=1)

    metrics.append(
        {
            "model": "XGBoost log-regressor",
            "target": "duration_seconds",
            "metric_space": "log_duration",
            "accuracy": np.nan,
            "mae": mean_absolute_error(test["duration_seconds"], duration_pred),
            "rmse": np.sqrt(
                mean_squared_error(test["duration_seconds"], duration_pred)
            ),
            "r2": r2_score(y_test_duration_log, duration_pred_log),
        }
    )

    metrics.append(
        {
            "model": "XGBoost log-regressor",
            "target": "duration_seconds",
            "metric_space": "raw_duration_diagnostic",
            "accuracy": np.nan,
            "mae": mean_absolute_error(test["duration_seconds"], duration_pred),
            "rmse": np.sqrt(
                mean_squared_error(test["duration_seconds"], duration_pred)
            ),
            "r2": r2_score(test["duration_seconds"], duration_pred),
        }
    )

    models["duration_model"] = duration_model
    models["duration_features"] = duration_features

    return models, metrics


def sample_categorical_from_history(
    history: pd.DataFrame,
    column: str,
    rng: np.random.Generator,
    condition_column: str | None = None,
    condition_value: str | None = None,
) -> str:
    if condition_column is not None and condition_value is not None:
        subset = history[history[condition_column] == condition_value]
        if len(subset) > 0:
            values = subset[column].dropna().values
        else:
            values = history[column].dropna().values
    else:
        values = history[column].dropna().values

    if len(values) == 0:
        raise ValueError(f"No values available for {column}")

    return str(rng.choice(values))


def build_future_base_features(
    history: pd.DataFrame,
    absolute_scheduled_seconds: float,
    role: str,
    app_name: str,
    job_type: str,
    gpu_request: int,
) -> dict:
    previous = history.iloc[-1].copy()

    interarrival_seconds = absolute_scheduled_seconds - previous["scheduled_seconds"]
    interarrival_seconds = max(1, interarrival_seconds)

    time_of_day_seconds = absolute_scheduled_seconds % 86_400
    hour = int(time_of_day_seconds // 3600)
    minute = int((time_of_day_seconds % 3600) // 60)
    second = int(time_of_day_seconds % 60)

    row = {
        "role": role,
        "app_name": app_name,
        "job_type": job_type,
        "scheduled_seconds": absolute_scheduled_seconds,
        "time_of_day_seconds": time_of_day_seconds,
        "hour": hour,
        "minute": minute,
        "second": second,
        "hour_sin": np.sin(2 * np.pi * time_of_day_seconds / 86_400),
        "hour_cos": np.cos(2 * np.pi * time_of_day_seconds / 86_400),
        "interarrival_seconds": interarrival_seconds,
        "gpu_request": gpu_request,
        "cpu_request_lag_1": previous["cpu_request"],
        "cpu_request_lag_2": history["cpu_request"].iloc[-2],
        "cpu_request_lag_5": history["cpu_request"].iloc[-5],
        "cpu_request_rolling_mean_5": history["cpu_request"].tail(5).mean(),
        "memory_request_lag_1": previous["memory_request"],
        "memory_request_lag_2": history["memory_request"].iloc[-2],
        "memory_request_lag_5": history["memory_request"].iloc[-5],
        "memory_request_rolling_mean_5": history["memory_request"].tail(5).mean(),
        "duration_seconds_lag_1": previous["duration_seconds"],
        "duration_seconds_lag_2": history["duration_seconds"].iloc[-2],
        "duration_seconds_lag_5": history["duration_seconds"].iloc[-5],
        "duration_seconds_rolling_mean_5": history["duration_seconds"].tail(5).mean(),
        "interarrival_seconds_lag_1": previous["interarrival_seconds"],
        "interarrival_seconds_lag_2": history["interarrival_seconds"].iloc[-2],
        "interarrival_seconds_lag_5": history["interarrival_seconds"].iloc[-5],
        "interarrival_seconds_rolling_mean_5": history[
            "interarrival_seconds"
        ].tail(5).mean(),
    }

    return row


def generate_future_jobs(
    data: pd.DataFrame,
    models: dict,
    categorical_features: list[str],
    numeric_features: list[str],
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)

    history = data.copy().reset_index(drop=True)

    interarrival_pool = history["interarrival_seconds"].copy()
    interarrival_pool = interarrival_pool[interarrival_pool > 0]

    if len(interarrival_pool) == 0:
        interarrival_pool = pd.Series([60.0])

    interarrival_pool = interarrival_pool.clip(
        lower=interarrival_pool.quantile(0.01),
        upper=interarrival_pool.quantile(0.99),
    )

    historical_last_time = float(history["scheduled_seconds"].iloc[-1])

    relative_time = 0.0
    absolute_time = historical_last_time

    future_rows = []

    while relative_time < FORECAST_HORIZON_SECONDS and len(future_rows) < MAX_FORECAST_JOBS:
        sampled_interarrival = float(rng.choice(interarrival_pool.values))
        sampled_interarrival = max(1.0, sampled_interarrival)

        relative_time += sampled_interarrival
        absolute_time += sampled_interarrival

        if relative_time > FORECAST_HORIZON_SECONDS:
            break

        job_type = sample_categorical_from_history(
            history=history,
            column="job_type",
            rng=rng,
        )

        role = sample_categorical_from_history(
            history=history,
            column="role",
            rng=rng,
            condition_column="job_type",
            condition_value=job_type,
        )

        app_name = sample_categorical_from_history(
            history=history,
            column="app_name",
            rng=rng,
            condition_column="job_type",
            condition_value=job_type,
        )

        gpu_values = history.loc[
            history["job_type"] == job_type,
            "gpu_request",
        ].dropna()

        if len(gpu_values) == 0:
            gpu_values = history["gpu_request"].dropna()

        gpu_request = int(rng.choice(gpu_values.values))

        base_row = build_future_base_features(
            history=history,
            absolute_scheduled_seconds=absolute_time,
            role=role,
            app_name=app_name,
            job_type=job_type,
            gpu_request=gpu_request,
        )

        X_future = pd.DataFrame([base_row])
        X_resource = X_future[categorical_features + numeric_features]

        cpu_pred_encoded = models["cpu_model"].predict(X_resource)
        predicted_cpu = float(
            models["cpu_encoder"].inverse_transform(cpu_pred_encoded)[0]
        )

        memory_pred_encoded = models["memory_model"].predict(X_resource)
        predicted_memory = float(
            models["memory_encoder"].inverse_transform(memory_pred_encoded)[0]
        )

        duration_row = base_row.copy()
        duration_row["cpu_request"] = predicted_cpu
        duration_row["memory_request"] = predicted_memory

        X_duration = pd.DataFrame([duration_row])[models["duration_features"]]

        duration_pred_log = models["duration_model"].predict(X_duration)[0]
        predicted_duration = float(np.expm1(duration_pred_log))

        predicted_duration = float(
            np.clip(
                predicted_duration,
                history["duration_seconds"].quantile(0.01),
                history["duration_seconds"].quantile(0.99),
            )
        )

        deletion_seconds = relative_time + predicted_duration

        forecast_row = {
            "forecast_job_id": f"forecast_job_{len(future_rows) + 1:06d}",
            "instance_sn": f"forecast_instance_{len(future_rows) + 1:06d}",
            "role": role,
            "app_name": app_name,
            "job_type": job_type,
            "scheduled_seconds": round(relative_time, 2),
            "cpu_request": predicted_cpu,
            "memory_request": predicted_memory,
            "gpu_request": gpu_request,
            "duration_seconds": round(predicted_duration, 2),
            "deletion_seconds": round(deletion_seconds, 2),
            "interarrival_seconds": round(sampled_interarrival, 2),
        }

        future_rows.append(forecast_row)

        history_row = base_row.copy()
        history_row["forecast_job_id"] = forecast_row["forecast_job_id"]
        history_row["instance_sn"] = forecast_row["instance_sn"]
        history_row["cpu_request"] = predicted_cpu
        history_row["memory_request"] = predicted_memory
        history_row["duration_seconds"] = predicted_duration
        history_row["deletion_seconds"] = absolute_time + predicted_duration
        history_row["scheduled_seconds"] = absolute_time
        history_row["interarrival_seconds"] = sampled_interarrival

        history = pd.concat(
            [history, pd.DataFrame([history_row])],
            ignore_index=True,
        )

    forecast = pd.DataFrame(future_rows)

    return forecast


def prepare_optimization_output(forecast: pd.DataFrame) -> pd.DataFrame:
    optimization = forecast.copy()

    optimization["release_seconds"] = optimization["scheduled_seconds"]
    optimization["processing_duration_seconds"] = optimization["duration_seconds"]

    optimization["deadline_seconds"] = (
        optimization["release_seconds"]
        + optimization["processing_duration_seconds"] * 1.25
    )

    cpu_threshold = optimization["cpu_request"].quantile(0.90)
    memory_threshold = optimization["memory_request"].quantile(0.90)

    optimization["is_critical"] = (
        (optimization["gpu_request"] == 1)
        | (optimization["cpu_request"] >= cpu_threshold)
        | (optimization["memory_request"] >= memory_threshold)
    ).astype(int)

    optimization["replica_count"] = np.where(
        optimization["is_critical"] == 1,
        2,
        1,
    )

    final_columns = [
        "forecast_job_id",
        "instance_sn",
        "role",
        "app_name",
        "job_type",
        "release_seconds",
        "cpu_request",
        "memory_request",
        "gpu_request",
        "processing_duration_seconds",
        "deadline_seconds",
        "is_critical",
        "replica_count",
    ]

    return optimization[final_columns]


def validate_forecast(
    original: pd.DataFrame,
    forecast: pd.DataFrame,
    optimization: pd.DataFrame,
) -> pd.DataFrame:
    """
    Forecast QA summary.

    This validation is intentionally written as a general data-quality report,
    not as a list of fixes from previous versions.
    """

    rows = []

    def add(section, metric, value):
        rows.append(
            {
                "section": section,
                "metric": metric,
                "value": value,
            }
        )

    add("row_counts", "historical_rows_used", len(original))
    add("row_counts", "forecast_rows_generated", len(forecast))

    add("time", "forecast_horizon_seconds", FORECAST_HORIZON_SECONDS)
    add("time", "min_scheduled_seconds", forecast["scheduled_seconds"].min())
    add("time", "max_scheduled_seconds", forecast["scheduled_seconds"].max())
    add("time", "scheduled_seconds_monotonic", forecast["scheduled_seconds"].is_monotonic_increasing)

    add("job_type_distribution", "counts", forecast["job_type"].value_counts().to_dict())
    add("gpu_request_distribution", "counts", forecast["gpu_request"].value_counts().to_dict())
    add("critical_distribution", "counts", optimization["is_critical"].value_counts().to_dict())
    add("replica_distribution", "counts", optimization["replica_count"].value_counts().to_dict())

    add("cpu_request", "historical_min", original["cpu_request"].min())
    add("cpu_request", "historical_max", original["cpu_request"].max())
    add("cpu_request", "forecast_min", forecast["cpu_request"].min())
    add("cpu_request", "forecast_max", forecast["cpu_request"].max())
    add(
        "cpu_request",
        "within_historical_range",
        forecast["cpu_request"]
        .between(original["cpu_request"].min(), original["cpu_request"].max())
        .all(),
    )

    add("memory_request", "historical_min", original["memory_request"].min())
    add("memory_request", "historical_max", original["memory_request"].max())
    add("memory_request", "forecast_min", forecast["memory_request"].min())
    add("memory_request", "forecast_max", forecast["memory_request"].max())
    add(
        "memory_request",
        "within_historical_range",
        forecast["memory_request"]
        .between(original["memory_request"].min(), original["memory_request"].max())
        .all(),
    )

    add("duration_seconds", "historical_min", original["duration_seconds"].min())
    add("duration_seconds", "historical_max", original["duration_seconds"].max())
    add("duration_seconds", "forecast_min", forecast["duration_seconds"].min())
    add("duration_seconds", "forecast_max", forecast["duration_seconds"].max())
    add("duration_seconds", "all_positive", (forecast["duration_seconds"] > 0).all())

    add("app_name", "unique_forecast_apps", forecast["app_name"].nunique())
    add("role", "unique_forecast_roles", forecast["role"].nunique())

    required_columns = [
        "forecast_job_id",
        "instance_sn",
        "role",
        "app_name",
        "job_type",
        "release_seconds",
        "cpu_request",
        "memory_request",
        "gpu_request",
        "processing_duration_seconds",
        "deadline_seconds",
        "is_critical",
        "replica_count",
    ]

    missing_columns = [c for c in required_columns if c not in optimization.columns]
    add("schema", "missing_required_columns", missing_columns)
    add("schema", "schema_complete", len(missing_columns) == 0)

    return pd.DataFrame(rows)


def main() -> None:
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("\nStep 1: Loading and preparing data...")
    data = load_and_prepare_data()

    print("\nStep 2: Temporal train/test split...")
    train, test = temporal_split(data)

    categorical_features = [
        "role",
        "app_name",
        "job_type",
    ]

    numeric_features = [
        "time_of_day_seconds",
        "hour",
        "minute",
        "second",
        "hour_sin",
        "hour_cos",
        "interarrival_seconds",
        "gpu_request",
        "cpu_request_lag_1",
        "cpu_request_lag_2",
        "cpu_request_lag_5",
        "cpu_request_rolling_mean_5",
        "memory_request_lag_1",
        "memory_request_lag_2",
        "memory_request_lag_5",
        "memory_request_rolling_mean_5",
        "duration_seconds_lag_1",
        "duration_seconds_lag_2",
        "duration_seconds_lag_5",
        "duration_seconds_rolling_mean_5",
        "interarrival_seconds_lag_1",
        "interarrival_seconds_lag_2",
        "interarrival_seconds_lag_5",
        "interarrival_seconds_rolling_mean_5",
    ]

    print("\nStep 3: Training models...")
    models, metrics = train_models(
        train=train,
        test=test,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
    )

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(METRICS_OUTPUT, index=False)

    print("\nStep 4: Generating 24-hour job-level forecast...")
    forecast = generate_future_jobs(
        data=data,
        models=models,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
    )

    print("\nStep 5: Saving forecast...")
    forecast.to_csv(FORECAST_CSV_OUTPUT, index=False)
    forecast.to_parquet(FORECAST_PARQUET_OUTPUT, index=False)

    print("\nStep 6: Preparing optimization-style output...")
    optimization_forecast = prepare_optimization_output(forecast)
    optimization_forecast.to_parquet(OPTIMIZATION_PARQUET_OUTPUT, index=False)

    print("\nStep 7: Generating forecast QA summary...")
    validation = validate_forecast(
        original=data,
        forecast=forecast,
        optimization=optimization_forecast,
    )
    validation.to_csv(VALIDATION_OUTPUT, index=False)

    print("\nDone.")
    print(f"Train rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Forecasted jobs: {len(forecast):,}")

    print("\nMetrics:")
    print(metrics_df)

    print("\nForecast job_type distribution:")
    print(forecast["job_type"].value_counts())

    print("\nForecast gpu_request distribution:")
    print(forecast["gpu_request"].value_counts())

    print("\nOptimization is_critical distribution:")
    print(optimization_forecast["is_critical"].value_counts())

    print("\nOptimization replica_count distribution:")
    print(optimization_forecast["replica_count"].value_counts())

    print("\nForecast QA summary:")
    print(validation)

    print("\nForecast output saved to:")
    print(f"- {FORECAST_CSV_OUTPUT}")
    print(f"- {FORECAST_PARQUET_OUTPUT}")

    print("\nOptimization-style output saved to:")
    print(f"- {OPTIMIZATION_PARQUET_OUTPUT}")


if __name__ == "__main__":
    main()