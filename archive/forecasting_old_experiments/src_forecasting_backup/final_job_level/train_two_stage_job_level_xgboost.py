import numpy as np
import pandas as pd

from xgboost import XGBClassifier, XGBRegressor

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from config import (
    INPUT_DATA,
    PROCESSED_DATA,
    TARGET_COLUMNS,
    DATE_COLUMN,
    TEST_HORIZON,
    SLOTS_PER_DAY,
    FORECAST_FREQ,
    PREDICTIONS_DIR,
    METRICS_DIR
)


MODEL_NAME = "TwoStage_JobLevel_XGBoost"
CLASSIFIER_NAME = "TwoStage_JobLevel_ActiveClassifier"

ACTIVE_THRESHOLD = 0.35

JOB_LEVEL_NUMERIC_TARGETS = [
    "cpu_request",
    "memory_request",
    "duration_minutes"
]

CLASSIFICATION_TARGET = "job_type"


# ============================================================
# METRICS
# ============================================================

def calculate_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    non_zero_mask = y_true != 0

    if non_zero_mask.sum() == 0:
        return np.nan

    return np.mean(
        np.abs(
            (y_true[non_zero_mask] - y_pred[non_zero_mask])
            / y_true[non_zero_mask]
        )
    ) * 100


def calculate_smape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    denominator = np.abs(y_true) + np.abs(y_pred)
    non_zero_mask = denominator != 0

    if non_zero_mask.sum() == 0:
        return np.nan

    return np.mean(
        2 * np.abs(y_pred[non_zero_mask] - y_true[non_zero_mask])
        / denominator[non_zero_mask]
    ) * 100


def calculate_regression_metrics(y_true, y_pred, model_name, target_name):
    mse = mean_squared_error(y_true, y_pred)

    return {
        "model": model_name,
        "target": target_name,
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": calculate_mape(y_true, y_pred),
        "SMAPE": calculate_smape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }


def clean_feature_matrix(feature_matrix):
    feature_matrix = feature_matrix.copy()
    feature_matrix = feature_matrix.apply(pd.to_numeric, errors="coerce")
    feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan)
    feature_matrix = feature_matrix.fillna(0)

    return feature_matrix


# ============================================================
# WINDOW-LEVEL FEATURES
# ============================================================

def create_window_features(forecast_data):
    model_data = forecast_data.copy()

    model_data["is_active_window"] = (
        model_data["job_count"] > 0
    ).astype(int)

    feature_blocks = []

    lags = [
        1,
        2,
        4,
        max(1, SLOTS_PER_DAY // 4),
        max(1, SLOTS_PER_DAY // 2),
        SLOTS_PER_DAY,
        SLOTS_PER_DAY * 2,
        SLOTS_PER_DAY * 3
    ]

    lags = sorted(list(set(lags)))

    for target_column in TARGET_COLUMNS:
        log_column = f"{target_column}_log"
        log_series = np.log1p(model_data[target_column])

        features = pd.DataFrame(index=model_data.index)

        features[log_column] = log_series

        for lag in lags:
            features[f"{target_column}_lag_{lag}"] = (
                log_series.shift(lag)
            )

        features[f"{target_column}_rolling_mean_4"] = (
            log_series.shift(1).rolling(4).mean()
        )

        features[f"{target_column}_rolling_std_4"] = (
            log_series.shift(1).rolling(4).std()
        )

        features[f"{target_column}_rolling_max_4"] = (
            log_series.shift(1).rolling(4).max()
        )

        features[f"{target_column}_rolling_min_4"] = (
            log_series.shift(1).rolling(4).min()
        )

        half_day_window = max(2, SLOTS_PER_DAY // 2)

        features[f"{target_column}_rolling_mean_half_day"] = (
            log_series.shift(1).rolling(half_day_window).mean()
        )

        features[f"{target_column}_rolling_std_half_day"] = (
            log_series.shift(1).rolling(half_day_window).std()
        )

        features[f"{target_column}_rolling_mean_day"] = (
            log_series.shift(1).rolling(SLOTS_PER_DAY).mean()
        )

        features[f"{target_column}_rolling_std_day"] = (
            log_series.shift(1).rolling(SLOTS_PER_DAY).std()
        )

        features[f"{target_column}_rolling_max_day"] = (
            log_series.shift(1).rolling(SLOTS_PER_DAY).max()
        )

        features[f"{target_column}_rolling_min_day"] = (
            log_series.shift(1).rolling(SLOTS_PER_DAY).min()
        )

        features[f"{target_column}_diff_1"] = (
            log_series.shift(1).diff(1)
        )

        features[f"{target_column}_diff_day"] = (
            log_series.shift(1).diff(SLOTS_PER_DAY)
        )

        feature_blocks.append(features)

    active_series = model_data["is_active_window"]

    active_features = pd.DataFrame(index=model_data.index)

    active_features["active_lag_1"] = active_series.shift(1)
    active_features["active_lag_2"] = active_series.shift(2)
    active_features["active_lag_4"] = active_series.shift(4)
    active_features["active_lag_day"] = active_series.shift(SLOTS_PER_DAY)

    active_features["active_rate_4"] = (
        active_series.shift(1).rolling(4).mean()
    )

    active_features["active_rate_half_day"] = (
        active_series.shift(1)
        .rolling(max(2, SLOTS_PER_DAY // 2))
        .mean()
    )

    active_features["active_rate_day"] = (
        active_series.shift(1).rolling(SLOTS_PER_DAY).mean()
    )

    feature_blocks.append(active_features)

    time_features = pd.DataFrame(index=model_data.index)

    time_features["time_sin"] = np.sin(
        2 * np.pi * model_data["slot_of_day"] / 1440
    )

    time_features["time_cos"] = np.cos(
        2 * np.pi * model_data["slot_of_day"] / 1440
    )

    time_features["day_sin"] = np.sin(
        2 * np.pi * model_data["day_of_week"] / 7
    )

    time_features["day_cos"] = np.cos(
        2 * np.pi * model_data["day_of_week"] / 7
    )

    time_features["hour"] = model_data["hour"]
    time_features["minute"] = model_data["minute"]
    time_features["day_of_week"] = model_data["day_of_week"]
    time_features["is_weekend"] = model_data["is_weekend"]
    time_features["slot_of_day"] = model_data["slot_of_day"]

    feature_blocks.append(time_features)

    model_data = pd.concat(
        [model_data] + feature_blocks,
        axis=1
    )

    model_data = model_data.loc[:, ~model_data.columns.duplicated()]
    model_data = model_data.dropna().reset_index(drop=True)

    return model_data


def build_window_feature_columns(model_data):
    target_log_columns = [
        f"{target_column}_log"
        for target_column in TARGET_COLUMNS
    ]

    exclude_columns = (
        TARGET_COLUMNS
        + target_log_columns
        + [
            DATE_COLUMN,
            "is_active_window"
        ]
    )

    feature_columns = [
        column for column in model_data.columns
        if column not in exclude_columns
    ]

    feature_columns = list(dict.fromkeys(feature_columns))

    return feature_columns


# ============================================================
# JOB-LEVEL FEATURES
# ============================================================

def load_job_level_data():
    job_data = pd.read_csv(INPUT_DATA)

    required_columns = [
        "instance_sn",
        "scheduled_time",
        "cpu_request",
        "memory_request",
        "duration_minutes",
        "job_type"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in job_data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    job_data["scheduled_time"] = pd.to_timedelta(
        job_data["scheduled_time"]
    )

    base_date = pd.Timestamp("2025-01-01")

    job_data["scheduled_datetime"] = (
        base_date + job_data["scheduled_time"]
    )

    job_data["job_type"] = (
        job_data["job_type"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    job_data = job_data.sort_values(
        "scheduled_datetime"
    ).reset_index(drop=True)

    return job_data


def create_job_level_features(job_data):
    model_data = job_data.copy()

    model_data["hour"] = model_data["scheduled_datetime"].dt.hour
    model_data["minute"] = model_data["scheduled_datetime"].dt.minute
    model_data["day_of_week"] = model_data["scheduled_datetime"].dt.dayofweek

    model_data["is_weekend"] = (
        model_data["day_of_week"] >= 5
    ).astype(int)

    model_data["minutes_from_start"] = (
        model_data["scheduled_datetime"]
        - model_data["scheduled_datetime"].min()
    ).dt.total_seconds() / 60

    model_data["interarrival_minutes"] = (
        model_data["scheduled_datetime"]
        .diff()
        .dt.total_seconds()
        .div(60)
        .fillna(0)
    )

    model_data["time_of_day_minutes"] = (
        model_data["hour"] * 60 + model_data["minute"]
    )

    model_data["time_sin"] = np.sin(
        2 * np.pi * model_data["time_of_day_minutes"] / 1440
    )

    model_data["time_cos"] = np.cos(
        2 * np.pi * model_data["time_of_day_minutes"] / 1440
    )

    model_data["day_sin"] = np.sin(
        2 * np.pi * model_data["day_of_week"] / 7
    )

    model_data["day_cos"] = np.cos(
        2 * np.pi * model_data["day_of_week"] / 7
    )

    model_data["job_type_encoded"] = (
        model_data["job_type"]
        .astype("category")
        .cat.codes
    )

    lags = [1, 2, 3, 5, 10, 20]

    lag_source_columns = [
        "cpu_request",
        "memory_request",
        "duration_minutes",
        "job_type_encoded",
        "interarrival_minutes"
    ]

    for column in lag_source_columns:
        for lag in lags:
            model_data[f"{column}_lag_{lag}"] = (
                model_data[column].shift(lag)
            )

    rolling_windows = [5, 10, 20]

    for column in [
        "cpu_request",
        "memory_request",
        "duration_minutes"
    ]:
        for window in rolling_windows:
            model_data[f"{column}_rolling_mean_{window}"] = (
                model_data[column]
                .shift(1)
                .rolling(window)
                .mean()
            )

            model_data[f"{column}_rolling_std_{window}"] = (
                model_data[column]
                .shift(1)
                .rolling(window)
                .std()
            )

            model_data[f"{column}_rolling_min_{window}"] = (
                model_data[column]
                .shift(1)
                .rolling(window)
                .min()
            )

            model_data[f"{column}_rolling_max_{window}"] = (
                model_data[column]
                .shift(1)
                .rolling(window)
                .max()
            )

    model_data = model_data.dropna().reset_index(drop=True)

    return model_data


def build_job_feature_columns(model_data):
    exclude_columns = [
        "instance_sn",
        "scheduled_time",
        "scheduled_datetime",
        "job_type",
        "cpu_request",
        "memory_request",
        "duration_minutes"
    ]

    feature_columns = [
        column for column in model_data.columns
        if column not in exclude_columns
    ]

    feature_columns = list(dict.fromkeys(feature_columns))

    return feature_columns


# ============================================================
# MODEL TRAINING HELPERS
# ============================================================

def train_active_window_classifier(X_train, y_train):
    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)

    if positive_count == 0:
        scale_pos_weight = 1
    else:
        scale_pos_weight = negative_count / positive_count

    classifier = XGBClassifier(
        objective="binary:logistic",
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        reg_alpha=0.3,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="logloss"
    )

    classifier.fit(X_train, y_train)

    return classifier


def train_job_count_model(X_train_active, y_train_job_count_log):
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        reg_alpha=0.3,
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

    model.fit(X_train_active, y_train_job_count_log)

    return model


def train_job_level_regressors(job_model_data, job_feature_columns, split_index):
    X_train = job_model_data[job_feature_columns].iloc[:split_index].copy()
    X_train = clean_feature_matrix(X_train)

    regressors = {}

    for target_column in JOB_LEVEL_NUMERIC_TARGETS:
        print(f"Training job-level regressor for {target_column}")

        y_train = np.log1p(
            job_model_data[target_column].iloc[:split_index]
        )

        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=3.0,
            reg_alpha=0.3,
            random_state=42,
            n_jobs=-1,
            tree_method="hist"
        )

        model.fit(X_train, y_train)

        regressors[target_column] = model

    return regressors


def train_job_type_classifier(job_model_data, job_feature_columns, split_index):
    X_train = job_model_data[job_feature_columns].iloc[:split_index].copy()
    X_train = clean_feature_matrix(X_train)

    y_train = job_model_data["job_type_encoded"].iloc[:split_index]

    unique_classes = sorted(y_train.unique())
    number_of_classes = len(unique_classes)

    if number_of_classes < 2:
        return None, unique_classes

    if number_of_classes == 2:
        classifier = XGBClassifier(
            objective="binary:logistic",
            n_estimators=400,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=3.0,
            reg_alpha=0.3,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="logloss"
        )

        classifier.fit(X_train, y_train)

        return classifier, unique_classes

    classifier = XGBClassifier(
        objective="multi:softprob",
        num_class=number_of_classes,
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        reg_alpha=0.3,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="mlogloss"
    )

    classifier.fit(X_train, y_train)

    return classifier, unique_classes


# ============================================================
# JOB GENERATION
# ============================================================

def create_job_type_mapping(job_model_data):
    mapping = (
        job_model_data[["job_type_encoded", "job_type"]]
        .drop_duplicates()
        .sort_values("job_type_encoded")
        .reset_index(drop=True)
    )

    return mapping


def build_job_template_features(
    job_model_data,
    job_feature_columns,
    window_row,
    job_sequence_number
):
    latest_job_features = (
        job_model_data[job_feature_columns]
        .iloc[-1]
        .copy()
    )

    feature_row = latest_job_features.copy()

    if "hour" in feature_row.index:
        feature_row["hour"] = window_row["hour"]

    if "minute" in feature_row.index:
        feature_row["minute"] = window_row["minute"]

    if "day_of_week" in feature_row.index:
        feature_row["day_of_week"] = window_row["day_of_week"]

    if "is_weekend" in feature_row.index:
        feature_row["is_weekend"] = window_row["is_weekend"]

    time_of_day_minutes = (
        window_row["hour"] * 60 + window_row["minute"]
    )

    if "time_of_day_minutes" in feature_row.index:
        feature_row["time_of_day_minutes"] = time_of_day_minutes

    if "time_sin" in feature_row.index:
        feature_row["time_sin"] = np.sin(
            2 * np.pi * time_of_day_minutes / 1440
        )

    if "time_cos" in feature_row.index:
        feature_row["time_cos"] = np.cos(
            2 * np.pi * time_of_day_minutes / 1440
        )

    if "day_sin" in feature_row.index:
        feature_row["day_sin"] = np.sin(
            2 * np.pi * window_row["day_of_week"] / 7
        )

    if "day_cos" in feature_row.index:
        feature_row["day_cos"] = np.cos(
            2 * np.pi * window_row["day_of_week"] / 7
        )

    feature_row["generated_job_sequence_number"] = job_sequence_number

    return feature_row


def generate_jobs_for_test_horizon(
    window_model_data,
    predicted_active,
    active_probabilities,
    predicted_job_counts,
    job_model_data,
    job_feature_columns,
    job_regressors,
    job_type_classifier,
    unique_job_type_classes,
    job_type_mapping
):
    generated_jobs = []

    if "generated_job_sequence_number" not in job_feature_columns:
        job_feature_columns = (
            job_feature_columns + ["generated_job_sequence_number"]
        )

    test_windows = window_model_data.iloc[-TEST_HORIZON:].copy()

    for test_index, (_, window_row) in enumerate(test_windows.iterrows()):
        is_predicted_active = int(predicted_active[test_index])
        active_probability = active_probabilities[test_index]

        predicted_job_count = int(
            np.round(predicted_job_counts[test_index])
        )

        predicted_job_count = max(0, predicted_job_count)

        if is_predicted_active == 0:
            predicted_job_count = 0

        for job_number in range(predicted_job_count):
            feature_row = build_job_template_features(
                job_model_data=job_model_data,
                job_feature_columns=job_feature_columns,
                window_row=window_row,
                job_sequence_number=job_number + 1
            )

            feature_frame = pd.DataFrame([feature_row])

            for column in job_feature_columns:
                if column not in feature_frame.columns:
                    feature_frame[column] = 0

            feature_frame = feature_frame[job_feature_columns]
            feature_frame = clean_feature_matrix(feature_frame)

            predicted_values = {}

            for target_column, regressor in job_regressors.items():
                prediction_log = regressor.predict(feature_frame)[0]
                prediction = np.expm1(prediction_log)
                prediction = max(0, prediction)

                predicted_values[target_column] = prediction

            if job_type_classifier is None:
                predicted_job_type_encoded = unique_job_type_classes[0]
            else:
                predicted_job_type_encoded = int(
                    job_type_classifier.predict(feature_frame)[0]
                )

            job_type_match = job_type_mapping[
                job_type_mapping["job_type_encoded"]
                == predicted_job_type_encoded
            ]

            if len(job_type_match) > 0:
                predicted_job_type = job_type_match["job_type"].iloc[0]
            else:
                predicted_job_type = "unknown"

            generated_jobs.append(
                {
                    DATE_COLUMN: window_row[DATE_COLUMN],
                    "predicted_active": is_predicted_active,
                    "active_probability": active_probability,
                    "predicted_job_count_for_window": predicted_job_count,
                    "generated_job_number_in_window": job_number + 1,
                    "predicted_cpu_request": predicted_values["cpu_request"],
                    "predicted_memory_request": predicted_values["memory_request"],
                    "predicted_duration_minutes": predicted_values["duration_minutes"],
                    "predicted_job_type_encoded": predicted_job_type_encoded,
                    "predicted_job_type": predicted_job_type
                }
            )

    generated_jobs_data = pd.DataFrame(generated_jobs)

    return generated_jobs_data


def aggregate_generated_jobs(generated_jobs_data, test_windows):
    all_windows = test_windows[[DATE_COLUMN]].copy()

    if generated_jobs_data.empty:
        aggregated = all_windows.copy()

        aggregated["cpu_request_sum_predicted"] = 0
        aggregated["memory_request_sum_predicted"] = 0
        aggregated["duration_minutes_mean_predicted"] = 0
        aggregated["job_count_predicted"] = 0
        aggregated["batch_count_predicted"] = 0
        aggregated["interactive_count_predicted"] = 0

        return aggregated

    generated_jobs_data = generated_jobs_data.copy()

    generated_jobs_data["is_batch"] = (
        generated_jobs_data["predicted_job_type"] == "batch"
    ).astype(int)

    generated_jobs_data["is_interactive"] = (
        generated_jobs_data["predicted_job_type"] == "interactive"
    ).astype(int)

    aggregated_jobs = generated_jobs_data.groupby(DATE_COLUMN).agg(
        cpu_request_sum_predicted=("predicted_cpu_request", "sum"),
        memory_request_sum_predicted=("predicted_memory_request", "sum"),
        duration_minutes_mean_predicted=("predicted_duration_minutes", "mean"),
        job_count_predicted=("predicted_cpu_request", "count"),
        batch_count_predicted=("is_batch", "sum"),
        interactive_count_predicted=("is_interactive", "sum")
    ).reset_index()

    aggregated = all_windows.merge(
        aggregated_jobs,
        on=DATE_COLUMN,
        how="left"
    )

    aggregated = aggregated.fillna(0)

    return aggregated


# ============================================================
# MAIN
# ============================================================

def train_two_stage_job_level_model():
    print("Loading forecast-ready window data...")

    forecast_data = pd.read_csv(
        PROCESSED_DATA,
        parse_dates=[DATE_COLUMN]
    )

    forecast_data = forecast_data.sort_values(
        DATE_COLUMN
    ).reset_index(drop=True)

    print(f"Forecast data used: {PROCESSED_DATA}")
    print(f"Forecast frequency: {FORECAST_FREQ}")
    print(f"Forecast data shape: {forecast_data.shape}")

    print("Creating window-level features...")

    window_model_data = create_window_features(forecast_data)

    print(f"Window model-ready shape: {window_model_data.shape}")

    if len(window_model_data) <= TEST_HORIZON:
        raise ValueError(
            f"Not enough window data after feature engineering. "
            f"Rows: {len(window_model_data)}, test horizon: {TEST_HORIZON}"
        )

    window_split_index = len(window_model_data) - TEST_HORIZON

    window_feature_columns = build_window_feature_columns(
        window_model_data
    )

    X_train_windows = window_model_data[
        window_feature_columns
    ].iloc[:window_split_index].copy()

    X_test_windows = window_model_data[
        window_feature_columns
    ].iloc[window_split_index:].copy()

    X_train_windows = clean_feature_matrix(X_train_windows)
    X_test_windows = clean_feature_matrix(X_test_windows)

    y_train_active = window_model_data[
        "is_active_window"
    ].iloc[:window_split_index]

    y_test_active = window_model_data[
        "is_active_window"
    ].iloc[window_split_index:]

    print("Training Layer 1: Active Window Classifier")

    active_classifier = train_active_window_classifier(
        X_train=X_train_windows,
        y_train=y_train_active
    )

    active_probabilities = active_classifier.predict_proba(
        X_test_windows
    )[:, 1]

    predicted_active = (
        active_probabilities >= ACTIVE_THRESHOLD
    ).astype(int)

    classifier_metrics = {
        "model": CLASSIFIER_NAME,
        "forecast_freq": FORECAST_FREQ,
        "threshold": ACTIVE_THRESHOLD,
        "accuracy": accuracy_score(y_test_active, predicted_active),
        "precision": precision_score(
            y_test_active,
            predicted_active,
            zero_division=0
        ),
        "recall": recall_score(
            y_test_active,
            predicted_active,
            zero_division=0
        ),
        "f1": f1_score(
            y_test_active,
            predicted_active,
            zero_division=0
        ),
        "roc_auc": (
            roc_auc_score(y_test_active, active_probabilities)
            if len(np.unique(y_test_active)) > 1
            else np.nan
        )
    }

    print("Training Layer 2A: Job count model for active windows")

    active_train_mask = (
        window_model_data["is_active_window"]
        .iloc[:window_split_index]
        == 1
    ).values

    X_train_active_windows = X_train_windows.loc[active_train_mask]

    y_train_job_count_log = np.log1p(
        window_model_data["job_count"]
        .iloc[:window_split_index]
        .loc[active_train_mask]
    )

    job_count_model = train_job_count_model(
        X_train_active=X_train_active_windows,
        y_train_job_count_log=y_train_job_count_log
    )

    predicted_job_count_log = job_count_model.predict(X_test_windows)
    predicted_job_counts = np.expm1(predicted_job_count_log)
    predicted_job_counts = np.maximum(predicted_job_counts, 0)

    print("Loading job-level data...")

    job_data = load_job_level_data()

    print(f"Job-level data used: {INPUT_DATA}")
    print(f"Raw job-level shape: {job_data.shape}")

    job_model_data = create_job_level_features(job_data)

    print(f"Job model-ready shape: {job_model_data.shape}")

    job_split_index = int(len(job_model_data) * 0.8)

    job_feature_columns = build_job_feature_columns(job_model_data)

    if "generated_job_sequence_number" not in job_feature_columns:
        job_model_data["generated_job_sequence_number"] = 0
        job_feature_columns.append("generated_job_sequence_number")

    print("Training Layer 2B: Job-level resource models")

    job_regressors = train_job_level_regressors(
        job_model_data=job_model_data,
        job_feature_columns=job_feature_columns,
        split_index=job_split_index
    )

    print("Training Layer 2C: Job-level job_type classifier")

    job_type_classifier, unique_job_type_classes = train_job_type_classifier(
        job_model_data=job_model_data,
        job_feature_columns=job_feature_columns,
        split_index=job_split_index
    )

    job_type_mapping = create_job_type_mapping(job_model_data)

    print("Generating predicted jobs inside predicted active windows...")

    test_windows = window_model_data.iloc[window_split_index:].copy()

    generated_jobs_data = generate_jobs_for_test_horizon(
        window_model_data=window_model_data,
        predicted_active=predicted_active,
        active_probabilities=active_probabilities,
        predicted_job_counts=predicted_job_counts,
        job_model_data=job_model_data,
        job_feature_columns=job_feature_columns,
        job_regressors=job_regressors,
        job_type_classifier=job_type_classifier,
        unique_job_type_classes=unique_job_type_classes,
        job_type_mapping=job_type_mapping
    )

    print("Aggregating generated jobs back into windows...")

    aggregated_predictions = aggregate_generated_jobs(
        generated_jobs_data=generated_jobs_data,
        test_windows=test_windows
    )

    actual_windows = test_windows[
        [
            DATE_COLUMN,
            "cpu_request_sum",
            "memory_request_sum",
            "duration_minutes_mean",
            "job_count",
            "batch_count",
            "interactive_count"
        ]
    ].copy()

    evaluation_data = actual_windows.merge(
        aggregated_predictions,
        on=DATE_COLUMN,
        how="left"
    ).fillna(0)

    metrics_list = []

    target_prediction_mapping = {
        "cpu_request_sum": "cpu_request_sum_predicted",
        "memory_request_sum": "memory_request_sum_predicted",
        "duration_minutes_mean": "duration_minutes_mean_predicted",
        "job_count": "job_count_predicted",
        "batch_count": "batch_count_predicted",
        "interactive_count": "interactive_count_predicted"
    }

    for target_column, predicted_column in target_prediction_mapping.items():
        metrics = calculate_regression_metrics(
            y_true=evaluation_data[target_column],
            y_pred=evaluation_data[predicted_column],
            model_name=MODEL_NAME,
            target_name=target_column
        )

        metrics_list.append(metrics)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    generated_jobs_path = (
        PREDICTIONS_DIR
        / f"two_stage_job_level_generated_jobs_{FORECAST_FREQ}.csv"
    )

    aggregated_predictions_path = (
        PREDICTIONS_DIR
        / f"two_stage_job_level_aggregated_predictions_{FORECAST_FREQ}.csv"
    )

    evaluation_path = (
        PREDICTIONS_DIR
        / f"two_stage_job_level_evaluation_data_{FORECAST_FREQ}.csv"
    )

    metrics_path = (
        METRICS_DIR
        / f"two_stage_job_level_metrics_{FORECAST_FREQ}.csv"
    )

    classifier_metrics_path = (
        METRICS_DIR
        / f"two_stage_job_level_classifier_metrics_{FORECAST_FREQ}.csv"
    )

    job_type_mapping_path = (
        METRICS_DIR
        / f"two_stage_job_level_job_type_mapping_{FORECAST_FREQ}.csv"
    )

    generated_jobs_data.to_csv(generated_jobs_path, index=False)
    aggregated_predictions.to_csv(aggregated_predictions_path, index=False)
    evaluation_data.to_csv(evaluation_path, index=False)

    pd.DataFrame(metrics_list).to_csv(metrics_path, index=False)

    pd.DataFrame([classifier_metrics]).to_csv(
        classifier_metrics_path,
        index=False
    )

    job_type_mapping.to_csv(job_type_mapping_path, index=False)

    print(f"Generated jobs saved to: {generated_jobs_path}")
    print(f"Aggregated predictions saved to: {aggregated_predictions_path}")
    print(f"Evaluation data saved to: {evaluation_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Classifier metrics saved to: {classifier_metrics_path}")
    print(f"Job type mapping saved to: {job_type_mapping_path}")

    print("\nTwo-stage job-level window aggregation metrics:")
    print(pd.DataFrame(metrics_list))

    print("\nActive window classifier metrics:")
    print(pd.DataFrame([classifier_metrics]))

    print("\nJob type mapping:")
    print(job_type_mapping)


if __name__ == "__main__":
    train_two_stage_job_level_model()