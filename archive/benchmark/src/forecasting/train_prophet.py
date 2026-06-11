import numpy as np
import pandas as pd

from prophet import Prophet

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


MODEL_NAME = "Prophet"


def train_prophet():
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
        print(f"Training Prophet for {target_column}")

        prophet_data = forecast_data[
            [
                DATE_COLUMN,
                target_column,
                "hour",
                "minute",
                "day_of_week",
                "is_weekend",
                "slot_of_day"
            ]
        ].copy()

        prophet_data[target_column] = np.log1p(prophet_data[target_column])

        prophet_data = prophet_data.rename(
            columns={
                DATE_COLUMN: "ds",
                target_column: "y"
            }
        )

        train_data = prophet_data.iloc[:split_index]
        test_data = prophet_data.iloc[split_index:]

        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            seasonality_mode="additive"
        )

        model.add_regressor("hour")
        model.add_regressor("minute")
        model.add_regressor("day_of_week")
        model.add_regressor("is_weekend")
        model.add_regressor("slot_of_day")

        model.fit(train_data)

        future_data = test_data[
            [
                "ds",
                "hour",
                "minute",
                "day_of_week",
                "is_weekend",
                "slot_of_day"
            ]
        ]

        forecast = model.predict(future_data)

        predictions = np.expm1(forecast["yhat"])
        predictions = np.maximum(predictions, 0)

        actual_values = forecast_data[target_column].iloc[split_index:]

        predictions_data[f"{target_column}_actual"] = actual_values.values
        predictions_data[f"{target_column}_predicted"] = predictions.values

        metrics = calculate_metrics(
            y_true=actual_values.values,
            y_pred=predictions.values,
            model_name=MODEL_NAME,
            target_name=target_column
        )

        metrics_list.append(metrics)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = (
        PREDICTIONS_DIR / f"prophet_predictions_{FORECAST_FREQ}.csv"
    )

    metrics_path = (
        METRICS_DIR / f"prophet_metrics_{FORECAST_FREQ}.csv"
    )

    predictions_data.to_csv(predictions_path, index=False)
    save_metrics(metrics_list, metrics_path)

    print("Prophet training completed.")


if __name__ == "__main__":
    train_prophet()