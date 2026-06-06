import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from config import (
    PROCESSED_DATA,
    TARGET_COLUMNS,
    DATE_COLUMN,
    TEST_HORIZON,
    SEQUENCE_LENGTH,
    FORECAST_FREQ,
    PREDICTIONS_DIR,
    METRICS_DIR
)

from metrics import calculate_metrics, save_metrics


MODEL_NAME = "LSTM"


def create_sequences(feature_values, target_values, sequence_length):
    X = []
    y = []

    for index in range(sequence_length, len(feature_values)):
        X.append(feature_values[index - sequence_length:index])
        y.append(target_values[index])

    return np.array(X), np.array(y)


def train_lstm():
    forecast_data = pd.read_csv(PROCESSED_DATA, parse_dates=[DATE_COLUMN])
    forecast_data = forecast_data.sort_values(DATE_COLUMN).reset_index(drop=True)

    for target_column in TARGET_COLUMNS:
        forecast_data[f"{target_column}_log"] = np.log1p(
            forecast_data[target_column]
        )

    forecast_data["time_sin"] = np.sin(
        2 * np.pi * forecast_data["slot_of_day"] / 1440
    )

    forecast_data["time_cos"] = np.cos(
        2 * np.pi * forecast_data["slot_of_day"] / 1440
    )

    target_log_columns = [
        f"{target_column}_log"
        for target_column in TARGET_COLUMNS
    ]

    feature_columns = (
        target_log_columns
        + [
            "job_count",
            "batch_count",
            "interactive_count",
            "hour",
            "minute",
            "day_of_week",
            "is_weekend",
            "time_sin",
            "time_cos"
        ]
    )

    feature_columns = list(dict.fromkeys(feature_columns))

    missing_columns = [
        column for column in feature_columns + target_log_columns
        if column not in forecast_data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns for LSTM: {missing_columns}")

    feature_values = forecast_data[feature_columns].values
    target_values = forecast_data[target_log_columns].values

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    scaled_features = feature_scaler.fit_transform(feature_values)
    scaled_targets = target_scaler.fit_transform(target_values)

    X, y = create_sequences(
        feature_values=scaled_features,
        target_values=scaled_targets,
        sequence_length=SEQUENCE_LENGTH
    )

    if len(X) <= TEST_HORIZON:
        raise ValueError(
            f"Not enough sequences for LSTM. Available sequences: {len(X)}, "
            f"test horizon: {TEST_HORIZON}"
        )

    split_index = len(X) - TEST_HORIZON

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    model = Sequential(
        [
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(128, return_sequences=True),
            Dropout(0.25),
            LSTM(64, return_sequences=False),
            Dropout(0.25),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(len(TARGET_COLUMNS))
        ]
    )

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-5
    )

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=150,
        batch_size=64,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    scaled_predictions = model.predict(X_test)

    predictions_log = target_scaler.inverse_transform(scaled_predictions)
    actual_log = target_scaler.inverse_transform(y_test)

    predictions = np.expm1(predictions_log)
    actual_values = np.expm1(actual_log)

    predictions = np.maximum(predictions, 0)

    prediction_dates = (
        forecast_data[[DATE_COLUMN]]
        .iloc[SEQUENCE_LENGTH:]
        .iloc[split_index:]
        .copy()
    )

    predictions_data = prediction_dates.copy()
    metrics_list = []

    for index, target_column in enumerate(TARGET_COLUMNS):
        actual_column = actual_values[:, index]
        predicted_column = predictions[:, index]

        predictions_data[f"{target_column}_actual"] = actual_column
        predictions_data[f"{target_column}_predicted"] = predicted_column

        metrics = calculate_metrics(
            y_true=actual_column,
            y_pred=predicted_column,
            model_name=MODEL_NAME,
            target_name=target_column
        )

        metrics_list.append(metrics)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = (
        PREDICTIONS_DIR / f"lstm_predictions_{FORECAST_FREQ}.csv"
    )

    metrics_path = (
        METRICS_DIR / f"lstm_metrics_{FORECAST_FREQ}.csv"
    )

    predictions_data.to_csv(predictions_path, index=False)
    save_metrics(metrics_list, metrics_path)

    print("LSTM training completed.")


if __name__ == "__main__":
    train_lstm()