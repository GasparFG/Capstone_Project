import numpy as np
import pandas as pd

from xgboost import XGBRegressor, XGBClassifier

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from config import (
    INPUT_DATA,
    METRICS_DIR,
    PREDICTIONS_DIR
)


MODEL_NAME = "JobLevel_XGBoost"

TEST_SIZE = 0.2

NUMERIC_TARGETS = [
    "cpu_request",
    "memory_request",
    "duration_minutes"
]

CLASSIFICATION_TARGET = "job_type"


def calculate_regression_metrics(y_true, y_pred, model_name, target_name):
    mse = mean_squared_error(y_true, y_pred)

    return {
        "model": model_name,
        "target": target_name,
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }


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


def get_feature_columns(model_data):
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


def train_numeric_models(
    model_data,
    feature_columns,
    split_index,
    predictions_data
):
    X_train = model_data[feature_columns].iloc[:split_index].copy()
    X_test = model_data[feature_columns].iloc[split_index:].copy()

    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)

    regression_metrics = []

    for target_column in NUMERIC_TARGETS:
        print(f"Training job-level XGBoost for {target_column}")

        y_train = np.log1p(
            model_data[target_column].iloc[:split_index]
        )

        y_test = model_data[target_column].iloc[split_index:]

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

        predictions_log = model.predict(X_test)
        predictions = np.expm1(predictions_log)
        predictions = np.maximum(predictions, 0)

        predictions_data[f"{target_column}_actual"] = y_test.values
        predictions_data[f"{target_column}_predicted"] = predictions

        metrics = calculate_regression_metrics(
            y_true=y_test.values,
            y_pred=predictions,
            model_name=MODEL_NAME,
            target_name=target_column
        )

        regression_metrics.append(metrics)

    return regression_metrics, predictions_data, X_train, X_test


def train_job_type_classifier(
    model_data,
    feature_columns,
    split_index,
    predictions_data
):
    print("Training job-level XGBoost classifier for job_type")

    X_train = model_data[feature_columns].iloc[:split_index].copy()
    X_test = model_data[feature_columns].iloc[split_index:].copy()

    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)

    y_train_class = model_data["job_type_encoded"].iloc[:split_index]
    y_test_class = model_data["job_type_encoded"].iloc[split_index:]

    unique_train_classes = sorted(y_train_class.unique())
    number_of_classes = len(unique_train_classes)

    if number_of_classes < 2:
        print(
            "Only one job_type class found in training data. "
            "Skipping job_type classification."
        )

        class_predictions = np.full(
            shape=len(y_test_class),
            fill_value=unique_train_classes[0]
        )

        predictions_data["job_type_actual_encoded"] = y_test_class.values
        predictions_data["job_type_predicted_encoded"] = class_predictions

        classification_metrics = {
            "model": MODEL_NAME,
            "target": CLASSIFICATION_TARGET,
            "accuracy": accuracy_score(y_test_class, class_predictions),
            "precision_macro": np.nan,
            "recall_macro": np.nan,
            "f1_macro": np.nan,
            "note": (
                "Only one class found in training data. "
                "Classification skipped."
            )
        }

        return classification_metrics, predictions_data

    if number_of_classes == 2:
        print("Detected binary job_type classification.")

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

        classifier.fit(X_train, y_train_class)

        class_predictions = classifier.predict(X_test)

        predictions_data["job_type_actual_encoded"] = y_test_class.values
        predictions_data["job_type_predicted_encoded"] = class_predictions

        classification_metrics = {
            "model": MODEL_NAME,
            "target": CLASSIFICATION_TARGET,
            "accuracy": accuracy_score(y_test_class, class_predictions),
            "precision_macro": precision_score(
                y_test_class,
                class_predictions,
                average="macro",
                zero_division=0
            ),
            "recall_macro": recall_score(
                y_test_class,
                class_predictions,
                average="macro",
                zero_division=0
            ),
            "f1_macro": f1_score(
                y_test_class,
                class_predictions,
                average="macro",
                zero_division=0
            ),
            "note": "Binary classification"
        }

        return classification_metrics, predictions_data

    print(
        f"Detected multiclass job_type classification: "
        f"{number_of_classes} classes."
    )

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

    classifier.fit(X_train, y_train_class)

    class_predictions = classifier.predict(X_test)

    predictions_data["job_type_actual_encoded"] = y_test_class.values
    predictions_data["job_type_predicted_encoded"] = class_predictions

    classification_metrics = {
        "model": MODEL_NAME,
        "target": CLASSIFICATION_TARGET,
        "accuracy": accuracy_score(y_test_class, class_predictions),
        "precision_macro": precision_score(
            y_test_class,
            class_predictions,
            average="macro",
            zero_division=0
        ),
        "recall_macro": recall_score(
            y_test_class,
            class_predictions,
            average="macro",
            zero_division=0
        ),
        "f1_macro": f1_score(
            y_test_class,
            class_predictions,
            average="macro",
            zero_division=0
        ),
        "note": "Multiclass classification"
    }

    return classification_metrics, predictions_data


def train_job_level_xgboost():
    job_data = load_job_level_data()

    print(f"Input data used: {INPUT_DATA}")
    print(f"Raw job-level shape: {job_data.shape}")

    model_data = create_job_level_features(job_data)

    print(f"Model-ready job-level shape: {model_data.shape}")

    split_index = int(len(model_data) * (1 - TEST_SIZE))

    feature_columns = get_feature_columns(model_data)

    predictions_data = model_data[
        [
            "scheduled_datetime",
            "instance_sn"
        ]
    ].iloc[split_index:].copy()

    regression_metrics, predictions_data, _, _ = train_numeric_models(
        model_data=model_data,
        feature_columns=feature_columns,
        split_index=split_index,
        predictions_data=predictions_data
    )

    classification_metrics, predictions_data = train_job_type_classifier(
        model_data=model_data,
        feature_columns=feature_columns,
        split_index=split_index,
        predictions_data=predictions_data
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    regression_metrics_path = (
        METRICS_DIR / "job_level_regression_metrics.csv"
    )

    classification_metrics_path = (
        METRICS_DIR / "job_level_classification_metrics.csv"
    )

    predictions_path = (
        PREDICTIONS_DIR / "job_level_predictions.csv"
    )

    pd.DataFrame(regression_metrics).to_csv(
        regression_metrics_path,
        index=False
    )

    pd.DataFrame([classification_metrics]).to_csv(
        classification_metrics_path,
        index=False
    )

    predictions_data.to_csv(
        predictions_path,
        index=False
    )

    print(f"Regression metrics saved to: {regression_metrics_path}")
    print(f"Classification metrics saved to: {classification_metrics_path}")
    print(f"Predictions saved to: {predictions_path}")

    print("\nRegression metrics:")
    print(pd.DataFrame(regression_metrics))

    print("\nClassification metrics:")
    print(pd.DataFrame([classification_metrics]))


if __name__ == "__main__":
    train_job_level_xgboost()