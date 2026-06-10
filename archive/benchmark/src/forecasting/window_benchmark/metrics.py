import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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


def calculate_metrics(y_true, y_pred, model_name, target_name):
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


def save_metrics(metrics_list, output_path):
    metrics_data = pd.DataFrame(metrics_list)
    metrics_data.to_csv(output_path, index=False)

    print(f"Metrics saved to: {output_path}")