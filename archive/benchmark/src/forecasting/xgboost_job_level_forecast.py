"""
xgboost_job_level_forecast.py

Final job-level XGBoost forecast with categorical workload information.

This version:
    - Keeps the output at job level.
    - Uses app_name and role from the parquet file.
    - Encodes categorical columns numerically.
    - Predicts:
        job_type
        cpu_request
        memory_request
        duration_minutes
    - Generates future jobs one by one.
    - Saves forecast and metrics.

Input:
    data/processed/synthetic_clean_90_days.parquet
    or
    data/processed/synthetic_clean_90_days.csv

Outputs:
    data/forecast/xgboost_job_level_forecast.csv
    data/forecast/xgboost_job_level_metrics.csv
"""

from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
)
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBRegressor, XGBClassifier
except ImportError as exc:
    raise ImportError(
        "XGBoost is not installed. Install it with:\n\n"
        "pip install xgboost\n"
    ) from exc


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

INPUT_PARQUET = Path("data/processed/synthetic_clean_90_days.parquet")
INPUT_CSV = Path("data/processed/synthetic_clean_90_days.csv")

OUTPUT_DIR = Path("data/forecast")
FORECAST_OUTPUT = OUTPUT_DIR / "xgboost_job_level_forecast.csv"
METRICS_OUTPUT = OUTPUT_DIR / "xgboost_job_level_metrics.csv"

FORECAST_JOBS = 1000
RANDOM_STATE = 42


# ------------------------------------------------------------
# Load and clean
# ------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load synthetic clean 90-day dataset."""

    if INPUT_PARQUET.exists():
        print(f"Loading parquet file: {INPUT_PARQUET}")
        return pd.read_parquet(INPUT_PARQUET)

    if INPUT_CSV.exists():
        print(f"Loading CSV file: {INPUT_CSV}")
        return pd.read_csv(INPUT_CSV)

    raise FileNotFoundError(
        "Could not find input dataset. Expected one of:\n"
        f"- {INPUT_PARQUET}\n"
        f"- {INPUT_CSV}"
    )


def convert_time_to_minutes(series: pd.Series) -> pd.Series:
    """
    Convert time column to minutes.

    Handles:
        - numeric seconds
        - numeric minutes
        - timedelta strings
        - pandas timedelta values
    """

    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds() / 60

    numeric_version = pd.to_numeric(series, errors="coerce")

    if numeric_version.notna().mean() > 0.90:
        max_value = numeric_version.max()

        if max_value > 100_000:
            return numeric_version / 60

        return numeric_version

    timedelta_version = pd.to_timedelta(series, errors="coerce")
    return timedelta_version.dt.total_seconds() / 60


def standardize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize columns and clean values.

    Required columns:
        scheduled_time
        job_type
        cpu_request
        memory_request
        duration_minutes
        app_name
        role
    """

    df = df.copy()

    rename_map = {
        "cpu_milli": "cpu_request",
        "memory_mib": "memory_request",
    }

    df = df.rename(columns=rename_map)

    required_columns = [
        "scheduled_time",
        "job_type",
        "cpu_request",
        "memory_request",
        "duration_minutes",
        "app_name",
        "role",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df["scheduled_time_minutes"] = convert_time_to_minutes(df["scheduled_time"])

    df["cpu_request"] = pd.to_numeric(df["cpu_request"], errors="coerce")
    df["memory_request"] = pd.to_numeric(df["memory_request"], errors="coerce")
    df["duration_minutes"] = pd.to_numeric(df["duration_minutes"], errors="coerce")

    df["app_name"] = df["app_name"].astype(str).fillna("unknown_app")
    df["role"] = df["role"].astype(str).fillna("unknown_role")
    df["job_type"] = df["job_type"].astype(str).fillna("unknown_job_type")

    df = df[
        df["scheduled_time_minutes"].notna()
        & df["job_type"].notna()
        & df["app_name"].notna()
        & df["role"].notna()
        & df["cpu_request"].notna()
        & df["memory_request"].notna()
        & df["duration_minutes"].notna()
    ].copy()

    df = df[
        (df["cpu_request"] >= 0)
        & (df["memory_request"] >= 0)
        & (df["duration_minutes"] > 0)
    ].copy()

    df = df.sort_values("scheduled_time_minutes").reset_index(drop=True)

    return df


# ------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------

def add_features(df: pd.DataFrame):
    """
    Add job-level features.

    Each row remains one job.
    app_name and role are encoded and used as predictive features.
    """

    df = df.copy()

    encoders = {
        "job_type": LabelEncoder(),
        "app_name": LabelEncoder(),
        "role": LabelEncoder(),
    }

    df["job_type_encoded"] = encoders["job_type"].fit_transform(df["job_type"])
    df["app_name_encoded"] = encoders["app_name"].fit_transform(df["app_name"])
    df["role_encoded"] = encoders["role"].fit_transform(df["role"])

    df["interarrival_minutes"] = df["scheduled_time_minutes"].diff()

    median_interarrival = df["interarrival_minutes"].median()

    if pd.isna(median_interarrival):
        median_interarrival = 0

    df["interarrival_minutes"] = df["interarrival_minutes"].fillna(
        median_interarrival
    )

    df["interarrival_minutes"] = df["interarrival_minutes"].clip(lower=0)

    df["minute_of_day"] = df["scheduled_time_minutes"] % (24 * 60)
    df["hour"] = (df["minute_of_day"] // 60).astype(int)
    df["day_number"] = (df["scheduled_time_minutes"] // (24 * 60)).astype(int)
    df["day_of_week"] = df["day_number"] % 7
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Historical context features
    df["cpu_lag_1"] = df["cpu_request"].shift(1)
    df["memory_lag_1"] = df["memory_request"].shift(1)
    df["duration_lag_1"] = df["duration_minutes"].shift(1)
    df["job_type_lag_1"] = df["job_type_encoded"].shift(1)
    df["app_name_lag_1"] = df["app_name_encoded"].shift(1)
    df["role_lag_1"] = df["role_encoded"].shift(1)

    df["cpu_rolling_mean_10"] = df["cpu_request"].shift(1).rolling(10).mean()
    df["memory_rolling_mean_10"] = df["memory_request"].shift(1).rolling(10).mean()
    df["duration_rolling_mean_10"] = df["duration_minutes"].shift(1).rolling(10).mean()
    df["interarrival_rolling_mean_10"] = (
        df["interarrival_minutes"].shift(1).rolling(10).mean()
    )

    df = df.dropna().reset_index(drop=True)

    return df, encoders


def get_feature_columns_for_job_type() -> list[str]:
    """Features used to predict job_type."""

    return [
        "scheduled_time_minutes",
        "minute_of_day",
        "hour",
        "day_number",
        "day_of_week",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "interarrival_minutes",
        "interarrival_rolling_mean_10",
        "app_name_encoded",
        "role_encoded",
        "app_name_lag_1",
        "role_lag_1",
    ]


def get_feature_columns_for_app_role() -> list[str]:
    """
    Features used to predict app_name and role for future jobs.

    This is needed because future jobs do not already have app_name/role.
    """

    return [
        "scheduled_time_minutes",
        "minute_of_day",
        "hour",
        "day_number",
        "day_of_week",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "interarrival_minutes",
        "interarrival_rolling_mean_10",
        "app_name_lag_1",
        "role_lag_1",
        "job_type_lag_1",
    ]


def get_feature_columns_for_resources() -> list[str]:
    """Features used to predict CPU, RAM, and duration."""

    return [
        "scheduled_time_minutes",
        "minute_of_day",
        "hour",
        "day_number",
        "day_of_week",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "interarrival_minutes",
        "interarrival_rolling_mean_10",
        "job_type_encoded",
        "app_name_encoded",
        "role_encoded",
        "job_type_lag_1",
        "app_name_lag_1",
        "role_lag_1",
        "cpu_lag_1",
        "memory_lag_1",
        "duration_lag_1",
        "cpu_rolling_mean_10",
        "memory_rolling_mean_10",
        "duration_rolling_mean_10",
    ]


def clean_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Ensure XGBoost receives only numeric finite values."""

    X = X.copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    return X


def temporal_train_test_split(df: pd.DataFrame, train_ratio: float = 0.80):
    """Use first 80% of jobs for training and last 20% for testing."""

    split_index = int(len(df) * train_ratio)

    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()

    return train, test


# ------------------------------------------------------------
# Train models
# ------------------------------------------------------------

def train_models(df: pd.DataFrame):
    """Train job-level XGBoost models."""

    train, test = temporal_train_test_split(df)

    job_type_features = get_feature_columns_for_job_type()
    app_role_features = get_feature_columns_for_app_role()
    resource_features = get_feature_columns_for_resources()

    X_train_type = clean_matrix(train[job_type_features])
    X_test_type = clean_matrix(test[job_type_features])

    X_train_app_role = clean_matrix(train[app_role_features])
    X_test_app_role = clean_matrix(test[app_role_features])

    X_train_resource = clean_matrix(train[resource_features])
    X_test_resource = clean_matrix(test[resource_features])

    models = {}

    # --------------------------------------------------------
    # app_name model
    # --------------------------------------------------------
    app_name_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.90,
        colsample_bytree=0.90,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
    )

    app_name_model.fit(X_train_app_role, train["app_name_encoded"])
    app_name_pred = app_name_model.predict(X_test_app_role)

    models["app_name_model"] = app_name_model

    # --------------------------------------------------------
    # role model
    # --------------------------------------------------------
    role_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=4,
        subsample=0.90,
        colsample_bytree=0.90,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
    )

    role_model.fit(X_train_app_role, train["role_encoded"])
    role_pred = role_model.predict(X_test_app_role)

    models["role_model"] = role_model

    # --------------------------------------------------------
    # job_type model
    # --------------------------------------------------------
    job_type_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=4,
        subsample=0.90,
        colsample_bytree=0.90,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
    )

    job_type_model.fit(X_train_type, train["job_type_encoded"])
    job_type_pred = job_type_model.predict(X_test_type)

    models["job_type_model"] = job_type_model

    # --------------------------------------------------------
    # CPU model
    # --------------------------------------------------------
    cpu_model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.90,
        colsample_bytree=0.90,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
    )

    cpu_model.fit(X_train_resource, np.log1p(train["cpu_request"]))
    cpu_pred = np.expm1(cpu_model.predict(X_test_resource))
    cpu_pred = np.clip(cpu_pred, 0, None)

    models["cpu_model"] = cpu_model

    # --------------------------------------------------------
    # Memory model
    # --------------------------------------------------------
    memory_model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.90,
        colsample_bytree=0.90,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
    )

    memory_model.fit(X_train_resource, np.log1p(train["memory_request"]))
    memory_pred = np.expm1(memory_model.predict(X_test_resource))
    memory_pred = np.clip(memory_pred, 0, None)

    models["memory_model"] = memory_model

    # --------------------------------------------------------
    # Duration model
    # --------------------------------------------------------
    duration_model = XGBRegressor(
        n_estimators=700,
        learning_rate=0.025,
        max_depth=6,
        subsample=0.90,
        colsample_bytree=0.90,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
    )

    duration_model.fit(X_train_resource, np.log1p(train["duration_minutes"]))
    duration_pred = np.expm1(duration_model.predict(X_test_resource))
    duration_pred = np.clip(duration_pred, 0, None)

    models["duration_model"] = duration_model

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    metrics = [
        {
            "model": "app_name_model",
            "target": "app_name",
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "accuracy": accuracy_score(test["app_name_encoded"], app_name_pred),
        },
        {
            "model": "role_model",
            "target": "role",
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "accuracy": accuracy_score(test["role_encoded"], role_pred),
        },
        {
            "model": "job_type_model",
            "target": "job_type",
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "accuracy": accuracy_score(test["job_type_encoded"], job_type_pred),
        },
        {
            "model": "cpu_model",
            "target": "cpu_request",
            "mae": mean_absolute_error(test["cpu_request"], cpu_pred),
            "rmse": np.sqrt(mean_squared_error(test["cpu_request"], cpu_pred)),
            "r2": r2_score(test["cpu_request"], cpu_pred),
            "accuracy": np.nan,
        },
        {
            "model": "memory_model",
            "target": "memory_request",
            "mae": mean_absolute_error(test["memory_request"], memory_pred),
            "rmse": np.sqrt(mean_squared_error(test["memory_request"], memory_pred)),
            "r2": r2_score(test["memory_request"], memory_pred),
            "accuracy": np.nan,
        },
        {
            "model": "duration_model",
            "target": "duration_minutes",
            "mae": mean_absolute_error(test["duration_minutes"], duration_pred),
            "rmse": np.sqrt(mean_squared_error(test["duration_minutes"], duration_pred)),
            "r2": r2_score(test["duration_minutes"], duration_pred),
            "accuracy": np.nan,
        },
    ]

    models["job_type_features"] = job_type_features
    models["app_role_features"] = app_role_features
    models["resource_features"] = resource_features

    return models, metrics, train, test


# ------------------------------------------------------------
# Forecast future jobs
# ------------------------------------------------------------

def build_future_base_row(
    history: pd.DataFrame,
    scheduled_time_minutes: float,
) -> dict:
    """Build one future job feature row."""

    previous = history.iloc[-1]

    interarrival_minutes = scheduled_time_minutes - previous["scheduled_time_minutes"]
    interarrival_minutes = max(0, interarrival_minutes)

    minute_of_day = scheduled_time_minutes % (24 * 60)
    hour = int(minute_of_day // 60)
    day_number = int(scheduled_time_minutes // (24 * 60))
    day_of_week = day_number % 7

    row = {
        "scheduled_time_minutes": scheduled_time_minutes,
        "minute_of_day": minute_of_day,
        "hour": hour,
        "day_number": day_number,
        "day_of_week": day_of_week,
        "is_weekend": int(day_of_week in [5, 6]),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * day_of_week / 7),
        "dow_cos": np.cos(2 * np.pi * day_of_week / 7),
        "interarrival_minutes": interarrival_minutes,
        "interarrival_rolling_mean_10": history["interarrival_minutes"].tail(10).mean(),
        "job_type_lag_1": previous["job_type_encoded"],
        "app_name_lag_1": previous["app_name_encoded"],
        "role_lag_1": previous["role_encoded"],
        "cpu_lag_1": previous["cpu_request"],
        "memory_lag_1": previous["memory_request"],
        "duration_lag_1": previous["duration_minutes"],
        "cpu_rolling_mean_10": history["cpu_request"].tail(10).mean(),
        "memory_rolling_mean_10": history["memory_request"].tail(10).mean(),
        "duration_rolling_mean_10": history["duration_minutes"].tail(10).mean(),
    }

    return row


def forecast_future_jobs(
    df: pd.DataFrame,
    models: dict,
    encoders: dict,
    forecast_jobs: int,
) -> pd.DataFrame:
    """Generate future jobs one by one."""

    history = df.copy().reset_index(drop=True)

    forecast_rows = []

    interarrival_pool = history["interarrival_minutes"]
    interarrival_pool = interarrival_pool[interarrival_pool >= 0]

    interarrival_min = interarrival_pool.quantile(0.01)
    interarrival_max = interarrival_pool.quantile(0.99)

    interarrival_pool = interarrival_pool.clip(
        lower=interarrival_min,
        upper=interarrival_max,
    )

    cpu_min = history["cpu_request"].quantile(0.01)
    cpu_max = history["cpu_request"].quantile(0.99)

    memory_min = history["memory_request"].quantile(0.01)
    memory_max = history["memory_request"].quantile(0.99)

    duration_min = history["duration_minutes"].quantile(0.01)
    duration_max = history["duration_minutes"].quantile(0.99)

    current_time = history["scheduled_time_minutes"].iloc[-1]

    for i in range(1, forecast_jobs + 1):
        sampled_interarrival = float(
            interarrival_pool.sample(1, random_state=RANDOM_STATE + i).iloc[0]
        )

        current_time = current_time + sampled_interarrival

        row = build_future_base_row(
            history=history,
            scheduled_time_minutes=current_time,
        )

        # Predict app_name
        X_app_role = pd.DataFrame([row])
        X_app_role = clean_matrix(X_app_role[models["app_role_features"]])

        app_name_encoded = int(models["app_name_model"].predict(X_app_role)[0])
        role_encoded = int(models["role_model"].predict(X_app_role)[0])

        row["app_name_encoded"] = app_name_encoded
        row["role_encoded"] = role_encoded

        # Predict job_type
        X_type = pd.DataFrame([row])
        X_type = clean_matrix(X_type[models["job_type_features"]])

        job_type_encoded = int(models["job_type_model"].predict(X_type)[0])

        row["job_type_encoded"] = job_type_encoded

        # Predict resources
        X_resource = pd.DataFrame([row])
        X_resource = clean_matrix(X_resource[models["resource_features"]])

        predicted_cpu = np.expm1(models["cpu_model"].predict(X_resource)[0])
        predicted_memory = np.expm1(models["memory_model"].predict(X_resource)[0])
        predicted_duration = np.expm1(models["duration_model"].predict(X_resource)[0])

        predicted_cpu = float(np.clip(predicted_cpu, cpu_min, cpu_max))
        predicted_memory = float(np.clip(predicted_memory, memory_min, memory_max))
        predicted_duration = float(np.clip(predicted_duration, duration_min, duration_max))

        app_name = encoders["app_name"].inverse_transform([app_name_encoded])[0]
        role = encoders["role"].inverse_transform([role_encoded])[0]
        job_type = encoders["job_type"].inverse_transform([job_type_encoded])[0]

        deletion_time_minutes = current_time + predicted_duration

        forecast_row = {
            "forecast_job_id": f"forecast_job_{i:06d}",
            "scheduled_time_minutes": current_time,
            "app_name": app_name,
            "role": role,
            "job_type": job_type,
            "cpu_request": predicted_cpu,
            "memory_request": predicted_memory,
            "duration_minutes": predicted_duration,
            "deletion_time_minutes": deletion_time_minutes,
            "interarrival_minutes": sampled_interarrival,
        }

        forecast_rows.append(forecast_row)

        new_history_row = row.copy()
        new_history_row["app_name"] = app_name
        new_history_row["role"] = role
        new_history_row["job_type"] = job_type
        new_history_row["app_name_encoded"] = app_name_encoded
        new_history_row["role_encoded"] = role_encoded
        new_history_row["job_type_encoded"] = job_type_encoded
        new_history_row["cpu_request"] = predicted_cpu
        new_history_row["memory_request"] = predicted_memory
        new_history_row["duration_minutes"] = predicted_duration
        new_history_row["interarrival_minutes"] = sampled_interarrival

        history = pd.concat(
            [history, pd.DataFrame([new_history_row])],
            ignore_index=True,
        )

    return pd.DataFrame(forecast_rows)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nStep 1: Loading data...")
    raw_data = load_data()

    print("\nStep 2: Standardizing data...")
    clean_data = standardize_data(raw_data)

    print("\nStep 3: Creating job-level features with app_name and role...")
    model_data, encoders = add_features(clean_data)

    print(f"Rows available for modeling: {len(model_data):,}")
    print(f"Unique app_name values: {model_data['app_name'].nunique():,}")
    print(f"Unique role values: {model_data['role'].nunique():,}")
    print(f"Unique job_type values: {model_data['job_type'].nunique():,}")

    print("\nStep 4: Training XGBoost job-level models...")
    models, metrics, train, test = train_models(model_data)

    print("\nStep 5: Saving metrics...")
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(METRICS_OUTPUT, index=False)

    print("\nStep 6: Forecasting future jobs...")
    forecast_df = forecast_future_jobs(
        df=model_data,
        models=models,
        encoders=encoders,
        forecast_jobs=FORECAST_JOBS,
    )

    print("\nStep 7: Saving forecast...")
    forecast_df.to_csv(FORECAST_OUTPUT, index=False)

    print("\nDone.")
    print(f"Metrics saved to: {METRICS_OUTPUT}")
    print(f"Forecast saved to: {FORECAST_OUTPUT}")

    print("\nTrain/Test split")
    print("----------------")
    print(f"Train jobs: {len(train):,}")
    print(f"Test jobs: {len(test):,}")

    print("\nMetrics")
    print("-------")
    print(metrics_df)

    print("\nForecast summary")
    print("----------------")
    print(f"Forecasted jobs: {len(forecast_df):,}")
    print(f"Average CPU request: {forecast_df['cpu_request'].mean():.2f}")
    print(f"Average memory request: {forecast_df['memory_request'].mean():.2f}")
    print(f"Average duration minutes: {forecast_df['duration_minutes'].mean():.2f}")

    print("\nFirst forecasted jobs:")
    print(forecast_df.head(10))


if __name__ == "__main__":
    main()