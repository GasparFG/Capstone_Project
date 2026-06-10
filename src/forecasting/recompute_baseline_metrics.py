"""
recompute_baseline_metrics.py

Re-trains the XGBoost baseline (same config as forecast_model.py) and saves
metrics in the same format as ensemble_model_metrics.csv so both can be
compared directly.

Output:
    data/forecast/baseline_model_metrics.csv
    Columns: split, model, target, accuracy, macro_f1, mae, rmse, rmsle, r2_log, r2_raw
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor

INPUT_PARQUET = Path("data/interim/cleaned_data.parquet")
OUTPUT_CSV    = Path("data/forecast/metrics/baseline_model_metrics.csv")
RANDOM_STATE  = 42


# ── helpers ──────────────────────────────────────────────────────────────────

def rmsle(y_true, y_pred):
    return float(np.sqrt(np.mean(
        (np.log1p(y_true) - np.log1p(np.clip(y_pred, 0, None))) ** 2
    )))


def clf_row(split, target, y_true_raw, y_pred_raw):
    y_t = pd.Series(y_true_raw).astype(float).round(6).astype(str)
    y_p = pd.Series(y_pred_raw).astype(float).round(6).astype(str)
    labels = sorted(y_t.unique())
    return {
        "split":    split,
        "model":    "XGBoost_baseline",
        "target":   target,
        "accuracy": accuracy_score(y_t, y_p),
        "macro_f1": f1_score(y_t, y_p, labels=labels, average="macro", zero_division=0),
        "mae":      mean_absolute_error(y_true_raw, y_pred_raw),
        "rmse":     float(np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))),
        "rmsle":    None,
        "r2_log":   None,
        "r2_raw":   r2_score(y_true_raw, y_pred_raw),
    }


def reg_row(split, y_true_raw, y_pred_raw, y_true_log, y_pred_log):
    return {
        "split":    split,
        "model":    "XGBoost_baseline",
        "target":   "duration_seconds",
        "accuracy": None,
        "macro_f1": None,
        "mae":      mean_absolute_error(y_true_raw, y_pred_raw),
        "rmse":     float(np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))),
        "rmsle":    rmsle(y_true_raw, y_pred_raw),
        "r2_log":   r2_score(y_true_log, y_pred_log),
        "r2_raw":   r2_score(y_true_raw, y_pred_raw),
    }


# ── data ─────────────────────────────────────────────────────────────────────

def load_data():
    data = pd.read_parquet(INPUT_PARQUET).copy()

    if not pd.api.types.is_timedelta64_dtype(data["scheduled_time"]):
        data["scheduled_time"] = pd.to_timedelta(data["scheduled_time"], unit="s")
    data["scheduled_seconds"] = data["scheduled_time"].dt.total_seconds()

    if "duration_seconds" not in data.columns:
        data["duration_seconds"] = data["duration_minutes"] * 60

    for col in ["scheduled_seconds", "gpu_request", "cpu_request",
                "memory_request", "duration_seconds"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["scheduled_seconds", "gpu_request", "cpu_request",
                                "memory_request", "duration_seconds",
                                "role", "app_name", "job_type"])
    data = data[(data["cpu_request"] > 0) & (data["memory_request"] > 0)
                & (data["duration_seconds"] > 0)].copy()

    for col in ["role", "app_name", "job_type"]:
        data[col] = data[col].astype(str).str.lower().str.strip()

    data = data.sort_values("scheduled_seconds").reset_index(drop=True)
    data["interarrival_seconds"] = data["scheduled_seconds"].diff().fillna(0).clip(lower=0)
    data["time_of_day_seconds"]  = data["scheduled_seconds"] % 86_400
    data["hour"]   = (data["time_of_day_seconds"] // 3600).astype(int)
    data["minute"] = ((data["time_of_day_seconds"] % 3600) // 60).astype(int)
    data["second"] = (data["time_of_day_seconds"] % 60).astype(int)
    data["hour_sin"] = np.sin(2 * np.pi * data["time_of_day_seconds"] / 86_400)
    data["hour_cos"] = np.cos(2 * np.pi * data["time_of_day_seconds"] / 86_400)

    for col in ["cpu_request", "memory_request", "duration_seconds", "interarrival_seconds"]:
        data[f"{col}_lag_1"] = data[col].shift(1)
        data[f"{col}_lag_2"] = data[col].shift(2)
        data[f"{col}_lag_5"] = data[col].shift(5)
        data[f"{col}_rolling_mean_5"] = data[col].shift(1).rolling(5).mean()

    return data.dropna().reset_index(drop=True)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    data  = load_data()
    split = int(len(data) * 0.80)
    train, test = data.iloc[:split].copy(), data.iloc[split:].copy()
    print(f"  Train: {len(train):,}  Test: {len(test):,}")

    cat_feats = ["role", "app_name", "job_type"]
    num_feats = [
        "time_of_day_seconds", "hour", "minute", "second",
        "hour_sin", "hour_cos", "interarrival_seconds", "gpu_request",
        "cpu_request_lag_1", "cpu_request_lag_2", "cpu_request_lag_5",
        "cpu_request_rolling_mean_5",
        "memory_request_lag_1", "memory_request_lag_2", "memory_request_lag_5",
        "memory_request_rolling_mean_5",
        "duration_seconds_lag_1", "duration_seconds_lag_2", "duration_seconds_lag_5",
        "duration_seconds_rolling_mean_5",
        "interarrival_seconds_lag_1", "interarrival_seconds_lag_2",
        "interarrival_seconds_lag_5", "interarrival_seconds_rolling_mean_5",
    ]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_feats),
        ("num", "passthrough", num_feats),
    ])

    X_tr = train[cat_feats + num_feats]
    X_te = test[cat_feats + num_feats]
    rows = []

    # CPU
    print("Training CPU classifier...")
    enc_cpu = LabelEncoder()
    y_tr_cpu = enc_cpu.fit_transform(train["cpu_request"].astype(str))
    cpu_pipe = Pipeline([("pre", preprocessor),
                         ("clf", XGBClassifier(n_estimators=150, max_depth=4,
                             learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                             eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1))])
    cpu_pipe.fit(X_tr, y_tr_cpu)
    rows.append(clf_row("train", "cpu_request",
                        train["cpu_request"].values,
                        enc_cpu.inverse_transform(cpu_pipe.predict(X_tr)).astype(float)))
    rows.append(clf_row("test",  "cpu_request",
                        test["cpu_request"].values,
                        enc_cpu.inverse_transform(cpu_pipe.predict(X_te)).astype(float)))

    # RAM
    print("Training RAM classifier...")
    enc_ram = LabelEncoder()
    y_tr_ram = enc_ram.fit_transform(train["memory_request"].astype(str))
    ram_pipe = Pipeline([("pre", preprocessor),
                         ("clf", XGBClassifier(n_estimators=150, max_depth=4,
                             learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                             eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1))])
    ram_pipe.fit(X_tr, y_tr_ram)
    rows.append(clf_row("train", "memory_request",
                        train["memory_request"].values,
                        enc_ram.inverse_transform(ram_pipe.predict(X_tr)).astype(float)))
    rows.append(clf_row("test",  "memory_request",
                        test["memory_request"].values,
                        enc_ram.inverse_transform(ram_pipe.predict(X_te)).astype(float)))

    # Duration
    print("Training Duration regressor...")
    dur_num  = num_feats + ["cpu_request", "memory_request"]
    dur_feat = cat_feats + dur_num
    dur_pre  = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_feats),
        ("num", "passthrough", dur_num),
    ])
    dur_pipe = Pipeline([("pre", dur_pre),
                         ("reg", XGBRegressor(n_estimators=300, max_depth=4,
                             learning_rate=0.04, subsample=0.9, colsample_bytree=0.9,
                             objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1))])
    y_tr_log = np.log1p(train["duration_seconds"])
    y_te_log = np.log1p(test["duration_seconds"])
    dur_pipe.fit(train[dur_feat], y_tr_log)

    dur_tr_log  = dur_pipe.predict(train[dur_feat])
    dur_tr_pred = np.expm1(dur_tr_log).clip(min=1)
    rows.append(reg_row("train", train["duration_seconds"].values, dur_tr_pred,
                        y_tr_log.values, dur_tr_log))

    dur_te_log  = dur_pipe.predict(test[dur_feat])
    dur_te_pred = np.expm1(dur_te_log).clip(min=1)
    rows.append(reg_row("test",  test["duration_seconds"].values, dur_te_pred,
                        y_te_log.values, dur_te_log))

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
