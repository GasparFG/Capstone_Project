import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, precision_score, recall_score
from config import JOB_LEVEL_DATA, OUTPUTS_DIR, MODELS_DIR, TEST_SIZE, DESCRIPTOR_COLUMNS, NUMERIC_TARGETS

PROFILE_LEVELS = [
    ["role", "app_name", "job_type"],
    ["app_name", "job_type"],
    ["role", "job_type"],
    ["app_name"],
    ["job_type"],
]


def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) * 100


def regression_row(target, y_true, y_pred, model_name="JobLevel_HistoricalProfile"):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "model": model_name,
        "target": target,
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "SMAPE": smape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "LOG_R2": r2_score(np.log1p(y_true), np.log1p(np.maximum(y_pred, 0))),
    }


def build_profile_tables(train_data: pd.DataFrame, target: str):
    tables = []
    for keys in PROFILE_LEVELS:
        table = (
            train_data.groupby(keys, dropna=False)[target]
            .agg(predicted_value="median", profile_count="count")
            .reset_index()
        )
        tables.append({"keys": keys, "table": table})
    fallback = float(train_data[target].median())
    return tables, fallback


def profile_predict(test_data: pd.DataFrame, profile_tables, fallback):
    predictions = pd.Series(index=test_data.index, dtype=float)
    matched_level = pd.Series(index=test_data.index, dtype="object")

    for level in profile_tables:
        keys = level["keys"]
        table = level["table"]
        remaining = predictions.isna()
        if not remaining.any():
            break
        merged = (
            test_data.loc[remaining, keys]
            .reset_index()
            .merge(table, on=keys, how="left")
        )
        values = merged.set_index("index")["predicted_value"]
        valid = values.notna()
        predictions.loc[values[valid].index] = values[valid]
        matched_level.loc[values[valid].index] = "+".join(keys)

    predictions = predictions.fillna(fallback)
    matched_level = matched_level.fillna("global_median")
    return predictions.values, matched_level.values


def predict_job_type_from_descriptors(train_data, test_data):
    # Predict job type from role + app_name using historical majority class.
    mode_table = (
        train_data.groupby(["role", "app_name"])["job_type"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
        .rename(columns={"job_type": "predicted_job_type"})
    )
    merged = test_data[["role", "app_name", "job_type"]].reset_index().merge(
        mode_table, on=["role", "app_name"], how="left"
    )
    fallback = train_data["job_type"].mode().iloc[0]
    predicted = merged["predicted_job_type"].fillna(fallback)
    return predicted.values


def save_profile_tables(train_data: pd.DataFrame):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {"profile_levels": PROFILE_LEVELS, "numeric_targets": NUMERIC_TARGETS}

    for target in NUMERIC_TARGETS:
        target_dir = MODELS_DIR / f"profile_{target}"
        target_dir.mkdir(parents=True, exist_ok=True)
        tables, fallback = build_profile_tables(train_data, target)
        metadata[target] = {"fallback_median": fallback, "tables": []}
        for index, level in enumerate(tables):
            filename = f"level_{index + 1}_{'_'.join(level['keys'])}.parquet"
            path = target_dir / filename
            level["table"].to_parquet(path, index=False)
            metadata[target]["tables"].append({"keys": level["keys"], "file": str(path.relative_to(MODELS_DIR))})

    descriptor_distribution = (
        train_data.groupby(DESCRIPTOR_COLUMNS, dropna=False)
        .size()
        .reset_index(name="count")
    )
    descriptor_distribution["probability"] = descriptor_distribution["count"] / descriptor_distribution["count"].sum()
    descriptor_distribution.to_csv(MODELS_DIR / "descriptor_distribution.csv", index=False)

    with open(MODELS_DIR / "job_level_profile_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def train_job_level_capacity_model():
    data = pd.read_parquet(JOB_LEVEL_DATA)
    data = data.sort_values("scheduled_time").reset_index(drop=True)
    split_index = int(len(data) * (1 - TEST_SIZE))
    train_data = data.iloc[:split_index].copy()
    test_data = data.iloc[split_index:].copy()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = []
    predictions = test_data[["arrival_order", "scheduled_time", "scheduled_minutes_from_start", "role", "app_name", "job_type"]].copy()

    for target in NUMERIC_TARGETS:
        tables, fallback = build_profile_tables(train_data, target)
        pred, level = profile_predict(test_data, tables, fallback)
        predictions[f"{target}_actual"] = test_data[target].values
        predictions[f"{target}_predicted"] = pred
        predictions[f"{target}_matched_profile"] = level
        metrics.append(regression_row(target, test_data[target].values, pred))

    job_type_pred = predict_job_type_from_descriptors(train_data, test_data)
    classification_metrics = {
        "model": "JobType_RoleApp_ProfileMode",
        "target": "job_type",
        "accuracy": accuracy_score(test_data["job_type"], job_type_pred),
        "precision_macro": precision_score(test_data["job_type"], job_type_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(test_data["job_type"], job_type_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(test_data["job_type"], job_type_pred, average="macro", zero_division=0),
    }
    predictions["job_type_predicted"] = job_type_pred

    regression_metrics = pd.DataFrame(metrics)
    regression_metrics.to_csv(OUTPUTS_DIR / "job_level_regression_metrics.csv", index=False)
    pd.DataFrame([classification_metrics]).to_csv(OUTPUTS_DIR / "job_level_classification_metrics.csv", index=False)
    predictions.to_csv(OUTPUTS_DIR / "job_level_test_predictions.csv", index=False)
    save_profile_tables(train_data)

    print(f"Training data used: {JOB_LEVEL_DATA}")
    print(f"Shape: {data.shape}")
    print("Regression metrics:")
    print(regression_metrics)
    print("Classification metrics:")
    print(pd.DataFrame([classification_metrics]))
    print(f"Artifacts saved to: {MODELS_DIR}")


if __name__ == "__main__":
    train_job_level_capacity_model()
