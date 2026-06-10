import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from config import (
    PROCESSED_DATA,
    TARGET_COLUMNS,
    DATE_COLUMN,
    TEST_HORIZON,
    SLOTS_PER_DAY,
    FORECAST_FREQ,
    PREDICTIONS_DIR,
    METRICS_DIR
)

from metrics import calculate_metrics, save_metrics


MODEL_NAME = "XGBoost"


def create_xgboost_features(forecast_data, target_columns):
    model_data = forecast_data.copy()

    feature_blocks = []

    lags = [
        1,
        2,
        4,
        max(1, SLOTS_PER_DAY // 4),
        max(1, SLOTS_PER_DAY // 2),
        SLOTS_PER_DAY,
        SLOTS_PER_DAY * 2,
        SLOTS_PER_DAY * 3,
    ]

    lags = sorted(list(set(lags)))

    for target_column in target_columns:
        log_column = f"{target_column}_log"

        log_series = np.log1p(model_data[target_column])

        features = pd.DataFrame(index=model_data.index)

        # This is the transformed target column.
        # It will be used as y, not as an input feature.
        features[log_column] = log_series

        for lag in lags:
            features[f"{target_column}_lag_{lag}"] = log_series.shift(lag)

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

        # Important:
        # These use only past values to avoid data leakage.
        features[f"{target_column}_diff_1"] = (
            log_series.shift(1).diff(1)
        )

        features[f"{target_column}_diff_day"] = (
            log_series.shift(1).diff(SLOTS_PER_DAY)
        )

        feature_blocks.append(features)

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

    feature_blocks.append(time_features)

    model_data = pd.concat(
        [model_data] + feature_blocks,
        axis=1
    )

    # Safety check: remove any duplicated columns.
    model_data = model_data.loc[:, ~model_data.columns.duplicated()]

    model_data = model_data.dropna().reset_index(drop=True)

    return model_data


def train_xgboost():
    forecast_data = pd.read_csv(PROCESSED_DATA, parse_dates=[DATE_COLUMN])
    forecast_data = forecast_data.sort_values(DATE_COLUMN).reset_index(drop=True)

    model_data = create_xgboost_features(
        forecast_data=forecast_data,
        target_columns=TARGET_COLUMNS
    )

    if len(model_data) <= TEST_HORIZON:
        raise ValueError(
            f"Not enough data after feature engineering. "
            f"Rows: {len(model_data)}, test horizon: {TEST_HORIZON}"
        )

    split_index = len(model_data) - TEST_HORIZON

    target_log_columns = [
        f"{target_column}_log"
        for target_column in TARGET_COLUMNS
    ]

    exclude_columns = (
        TARGET_COLUMNS
        + target_log_columns
        + [DATE_COLUMN]
    )

    feature_columns = [
        column for column in model_data.columns
        if column not in exclude_columns
    ]

    # Safety check: make sure feature names are unique.
    feature_columns = list(dict.fromkeys(feature_columns))

    X_train = model_data[feature_columns].iloc[:split_index].copy()
    X_test = model_data[feature_columns].iloc[split_index:].copy()

    # XGBoost needs clean numeric input.
    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_test = X_test.apply(pd.to_numeric, errors="coerce")

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    predictions_data = model_data[[DATE_COLUMN]].iloc[split_index:].copy()
    metrics_list = []

    for target_column in TARGET_COLUMNS:
        print(f"Training XGBoost for {target_column}")

        target_log_column = f"{target_column}_log"

        y_train = model_data[target_log_column].iloc[:split_index]
        y_test_actual = model_data[target_column].iloc[split_index:]

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

        actual_values = y_test_actual.values
        predicted_values = predictions

        predictions_data[f"{target_column}_actual"] = actual_values
        predictions_data[f"{target_column}_predicted"] = predicted_values

        metrics = calculate_metrics(
            y_true=actual_values,
            y_pred=predicted_values,
            model_name=MODEL_NAME,
            target_name=target_column
        )

        metrics_list.append(metrics)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = (
        PREDICTIONS_DIR / f"xgboost_predictions_{FORECAST_FREQ}.csv"
    )

    metrics_path = (
        METRICS_DIR / f"xgboost_metrics_{FORECAST_FREQ}.csv"
    )

    predictions_data.to_csv(predictions_path, index=False)
    save_metrics(metrics_list, metrics_path)

    print("XGBoost training completed.")


if __name__ == "__main__":
    train_xgboost()