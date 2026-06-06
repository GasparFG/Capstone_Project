import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX

from config import (
    PROCESSED_DATA,
    TARGET_COLUMNS,
    DATE_COLUMN,
    TEST_HORIZON,
    FORECAST_FREQ,
    PREDICTIONS_DIR,
    METRICS_DIR
)

from metrics import calculate_metrics, save_metrics


MODEL_NAME = "SARIMA_LIGHT"


def train_sarima():
    forecast_data = pd.read_csv(PROCESSED_DATA, parse_dates=[DATE_COLUMN])
    forecast_data = forecast_data.sort_values(DATE_COLUMN).reset_index(drop=True)

    if len(forecast_data) <= TEST_HORIZON:
        raise ValueError(
            f"Not enough data. Rows: {len(forecast_data)}, "
            f"test horizon: {TEST_HORIZON}"
        )

    split_index = len(forecast_data) - TEST_HORIZON

    predictions_data = forecast_data[[DATE_COLUMN]].iloc[split_index:].copy()
    metrics_list = []

    for target_column in TARGET_COLUMNS:
        print(f"Training lightweight SARIMA for {target_column}")

        train_data = forecast_data[target_column].iloc[:split_index]
        test_data = forecast_data[target_column].iloc[split_index:]

        train_log = np.log1p(train_data)

        try:
            model = SARIMAX(
                train_log,
                order=(1, 1, 1),
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            fitted_model = model.fit(
                disp=False,
                maxiter=50
            )

            forecast_log = fitted_model.forecast(
                steps=len(test_data)
            )

            predictions = np.expm1(forecast_log)
            predictions = np.maximum(predictions, 0)

        except Exception as error:
            print(f"SARIMA failed for {target_column}: {error}")
            print("Using naive fallback prediction.")

            last_value = train_data.iloc[-1]
            predictions = pd.Series(
                [last_value] * len(test_data),
                index=test_data.index
            )

        predictions_data[f"{target_column}_actual"] = test_data.values
        predictions_data[f"{target_column}_predicted"] = predictions.values

        metrics = calculate_metrics(
            y_true=test_data.values,
            y_pred=predictions.values,
            model_name=MODEL_NAME,
            target_name=target_column
        )

        metrics_list.append(metrics)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = (
        PREDICTIONS_DIR / f"sarima_predictions_{FORECAST_FREQ}.csv"
    )

    metrics_path = (
        METRICS_DIR / f"sarima_metrics_{FORECAST_FREQ}.csv"
    )

    predictions_data.to_csv(predictions_path, index=False)
    save_metrics(metrics_list, metrics_path)

    print("Lightweight SARIMA training completed.")


if __name__ == "__main__":
    train_sarima()