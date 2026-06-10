"""
train_model.py
==============
Trains the XGBoost forecasting models using the dataset in SECONDS.

Models trained:
  REGRESSION (continuous values):
  - duration_seconds      -> expected job duration

  CLASSIFICATION (discrete values, few unique values):
  - cpu_request           -> 9 possible values  (2, 8, 12, 16, 48, 64, 96, 192...)
  - memory_request        -> 24 possible values
  - interarrival_bucket   -> 5 arrival regime buckets
  - role_encoded          -> 2 classes (CN / HN)
  - app_name_encoded      -> 151 apps
  - job_type_encoded      -> 2 classes (batch / interactive)

Outputs:
  models/forecast_seconds/{target}_model.joblib       <- regressors
  models/forecast_seconds/{target}_classifier.joblib  <- classifiers (cpu, memory, descriptors)
  models/forecast_seconds/model_metadata.json
  outputs/forecast_seconds/regression_metrics.csv
  outputs/forecast_seconds/classification_metrics.csv
  outputs/forecast_seconds/test_predictions.csv

Run from src/forecasting/:
    python train_model.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score,
)
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBRegressor, XGBClassifier
except ImportError as e:
    raise ImportError("XGBoost is not installed. Run: pip install xgboost") from e

from config_seconds import (
    PREPARED_DATA, MODELS_DIR, OUTPUTS_DIR,
    TEST_SIZE, RANDOM_STATE,
    NUMERIC_TARGETS, DISCRETE_TARGETS, DESCRIPTOR_TARGETS,
)

# ── Columns excluded from features ───────────────────────────────────────────
DROP_ALWAYS = [
    "instance_sn", "creation_time", "deletion_time", "gpu_request",
    "scheduled_time", "scheduled_seconds", "arrival_order",
    "role", "app_name", "job_type",
    "duration_minutes",
    "interarrival_seconds", "interarrival_bucket",
    "cpu_request", "memory_request", "duration_seconds",
    "interarrival_seconds_log", "cpu_request_log", "memory_request_log", "duration_seconds_log",
]
CURRENT_DESCRIPTORS = ["role_encoded", "app_name_encoded", "job_type_encoded"]


# ── Metrics ───────────────────────────────────────────────────────────────────

def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    return np.nan if mask.sum() == 0 else np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    return np.nan if mask.sum() == 0 else np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100


def reg_metrics(y_true, y_pred, target, model_name):
    mse = mean_squared_error(y_true, y_pred)
    log_r2 = r2_score(np.log1p(y_true), np.log1p(np.maximum(y_pred, 0)))
    return {
        "model": model_name, "target": target,
        "MSE": mse, "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mape(y_true, y_pred), "SMAPE": smape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred), "LOG_R2": log_r2,
    }


def clean(X: pd.DataFrame) -> pd.DataFrame:
    return X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)


# ── Feature column selection ──────────────────────────────────────────────────

def past_feature_columns(df: pd.DataFrame):
    """Features that do NOT use current job information (past only)."""
    return [c for c in df.columns if c not in DROP_ALWAYS and c not in CURRENT_DESCRIPTORS]


def capacity_feature_columns(df: pd.DataFrame):
    """Features for predicting CPU/memory/duration — includes current job descriptors."""
    past = past_feature_columns(df)
    caps = [c for c in CURRENT_DESCRIPTORS if c in df.columns]
    return list(dict.fromkeys(past + caps))


# ── XGBoost hyperparameter configurations ────────────────────────────────────

XGB_PARAMS = {
    "cpu_request": [
        ("xgb_cpu_balanced", dict(
            n_estimators=800, learning_rate=0.03, max_depth=5,
            min_child_weight=2, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=2.5, reg_alpha=0.1,
        )),
        ("xgb_cpu_regularized", dict(
            n_estimators=700, learning_rate=0.03, max_depth=4,
            min_child_weight=4, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=5, reg_alpha=0.5,
        )),
    ],
    "memory_request": [
        ("xgb_memory_balanced", dict(
            n_estimators=800, learning_rate=0.03, max_depth=5,
            min_child_weight=2, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=2.5, reg_alpha=0.1,
        )),
        ("xgb_memory_regularized", dict(
            n_estimators=700, learning_rate=0.03, max_depth=4,
            min_child_weight=4, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=5, reg_alpha=0.5,
        )),
    ],
    "duration_seconds": [
        ("xgb_duration_regularized", dict(
            n_estimators=600, learning_rate=0.03, max_depth=3,
            min_child_weight=6, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=8, reg_alpha=1,
        )),
        ("xgb_duration_balanced", dict(
            n_estimators=800, learning_rate=0.025, max_depth=4,
            min_child_weight=4, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=5, reg_alpha=0.5,
        )),
    ],
}


# ── Regressor training ────────────────────────────────────────────────────────

def select_best_regressor(target, X_train, X_test, train_df, test_df):
    """Trains all candidates for `target` and returns the best by RMSE."""
    y_train = np.log1p(train_df[target])
    y_test  = test_df[target].values
    best = None

    for name, params in XGB_PARAMS[target]:
        print(f"  Candidate: {name}")
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            **params,
        )
        model.fit(X_train, y_train)
        pred = np.expm1(model.predict(X_test)).clip(0)
        metrics = reg_metrics(y_test, pred, target, name)
        result = {"name": name, "model": model, "pred": pred, "metrics": metrics}
        if best is None or metrics["RMSE"] < best["metrics"]["RMSE"]:
            best = result

    print(f"  -> Best: {best['name']}  RMSE={best['metrics']['RMSE']:.2f}  R2={best['metrics']['R2']:.4f}  LOG_R2={best['metrics']['LOG_R2']:.4f}")
    return best


# ── Classifier training ───────────────────────────────────────────────────────

def train_classifier(target, X_train, X_test, train_df, test_df):
    # Cast to string so sklearn does not confuse discrete floats (320.0, 64.0)
    # with continuous variables. Applies to both integer encodings and float values.
    y_train = train_df[target].astype(str)
    y_test  = test_df[target].astype(str)
    n_classes = int(y_train.nunique())

    if n_classes < 2:
        pred = np.full(len(y_test), y_train.iloc[0])
        return None, None, _cls_metrics(y_test, pred, target, "constant"), pred

    # LabelEncoder remaps classes to contiguous 0..n-1.
    # Needed when the temporal split leaves gaps in training codes
    # (e.g. app_name_encoded=[0,1,3,4,...] but XGBoost requires [0,1,2,3,...]).
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    # Test: classes seen in train -> encode; unseen -> -1 (excluded from metrics)
    known_mask  = y_test.isin(le.classes_)
    y_test_enc  = np.full(len(y_test), -1, dtype=int)
    y_test_enc[known_mask.values] = le.transform(y_test[known_mask])

    n_enc       = len(le.classes_)
    objective   = "multi:softprob" if n_enc > 2 else "binary:logistic"
    extra       = {"num_class": n_enc} if n_enc > 2 else {}
    eval_metric = "mlogloss" if n_enc > 2 else "logloss"

    model = XGBClassifier(
        objective=objective,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        eval_metric=eval_metric,
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=3,
        reg_alpha=0.2,
        **extra,
    )
    model.fit(X_train, y_train_enc)

    # Predict -> inverse_transform to recover the original code (app/role/job_type)
    pred_enc = model.predict(X_test)
    pred     = le.inverse_transform(pred_enc.astype(int))

    # Metrics only over classes known in test set
    valid = known_mask.values
    metrics = _cls_metrics(
        y_test[valid], pred[valid], target, f"xgb_{target}_classifier"
    )
    return model, le, metrics, pred


def _cls_metrics(y_true, y_pred, target, model_name):
    return {
        "model": model_name, "target": target,
        "accuracy":         accuracy_score(y_true, y_pred),
        "precision_macro":  precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro":     recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro":         f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────

def train_model() -> None:
    print(f"Loading prepared dataset: {PREPARED_DATA}")
    data = pd.read_parquet(PREPARED_DATA)
    print(f"Shape: {data.shape}")

    # Temporal split (80% train / 20% test)
    split = int(len(data) * (1 - TEST_SIZE))
    train_df = data.iloc[:split].copy()
    test_df  = data.iloc[split:].copy()
    print(f"Train: {len(train_df)} jobs  |  Test: {len(test_df)} jobs")

    past_cols = past_feature_columns(data)
    cap_cols  = capacity_feature_columns(data)
    print(f"Features (past-only): {len(past_cols)}  |  Features (capacity): {len(cap_cols)}")

    X_train_past = clean(train_df[past_cols])
    X_test_past  = clean(test_df[past_cols])
    X_train_cap  = clean(train_df[cap_cols])
    X_test_cap   = clean(test_df[cap_cols])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    def save_classifier(target, X_tr, X_te, label=""):
        print(f"\n[{target}]{label}")
        model, le, metrics, pred = train_classifier(target, X_tr, X_te, train_df, test_df)
        if model is not None:
            joblib.dump({"model": model, "label_encoder": le},
                        MODELS_DIR / f"{target}_classifier.joblib")
        return metrics, pred

    # ── Descriptor classifiers (role, app_name, job_type) ─────────────────────
    print("\n── Descriptor classifiers ──")
    cls_rows = {}
    for target in [t for t in DESCRIPTOR_TARGETS if t in data.columns]:
        metrics, pred = save_classifier(target, X_train_past, X_test_past)
        cls_rows[target] = {"metrics": metrics, "pred": pred}

    # ── Discrete resource classifiers (cpu, memory, interarrival_bucket) ──────
    print("\n── Discrete resource classifiers ──")
    for target in [t for t in DISCRETE_TARGETS if t in data.columns]:
        # interarrival_bucket: predicts WHEN the next job arrives
        #   -> past-only features (the future job's descriptors are unknown at this point)
        # cpu_request / memory_request: predict WHAT resources the job needs
        #   -> capacity features (include current job descriptors role/app/job_type)
        use_past = (target == "interarrival_bucket")
        X_tr = X_train_past if use_past else X_train_cap
        X_te = X_test_past  if use_past else X_test_cap
        metrics, pred = save_classifier(
            target, X_tr, X_te,
            label=f"  ({data[target].nunique()} unique values)"
        )
        cls_rows[target] = {"metrics": metrics, "pred": pred}

    # ── Continuous regressors (duration) ──────────────────────────────────────
    print("\n── Continuous regressors ──")
    reg_rows = []
    pred_df  = test_df[["arrival_order", "scheduled_seconds", "job_type", "role", "app_name"]].copy()

    for target in NUMERIC_TARGETS:
        print(f"\n[{target}]")
        best = select_best_regressor(target, X_train_cap, X_test_cap, train_df, test_df)
        reg_rows.append(best["metrics"])
        pred_df[f"{target}_actual"]         = test_df[target].values
        pred_df[f"{target}_predicted"]      = best["pred"]
        pred_df[f"{target}_selected_model"] = best["name"]
        joblib.dump(best["model"], MODELS_DIR / f"{target}_model.joblib")

    # ── Save metrics ──────────────────────────────────────────────────────────
    reg_df = pd.DataFrame(reg_rows)
    cls_df = pd.DataFrame([v["metrics"] for v in cls_rows.values()])
    reg_df.to_csv(OUTPUTS_DIR / "regression_metrics.csv", index=False)
    cls_df.to_csv(OUTPUTS_DIR / "classification_metrics.csv", index=False)
    pred_df.to_csv(OUTPUTS_DIR / "test_predictions.csv", index=False)
    pd.DataFrame({"feature": past_cols}).to_csv(OUTPUTS_DIR / "past_feature_columns.csv", index=False)
    pd.DataFrame({"feature": cap_cols}).to_csv(OUTPUTS_DIR / "capacity_feature_columns.csv", index=False)

    # ── Save descriptor distribution ──────────────────────────────────────────
    desc_cols = [c for c in ["role", "app_name", "job_type"] if c in data.columns]
    dist = (
        data[desc_cols].groupby(desc_cols, dropna=False)
        .size().reset_index(name="count")
    )
    dist["probability"] = dist["count"] / dist["count"].sum()
    dist.to_csv(MODELS_DIR / "descriptor_distribution.csv", index=False)

    # ── Save model metadata ───────────────────────────────────────────────────
    metadata = {
        "past_feature_columns":     past_cols,
        "capacity_feature_columns": cap_cols,
        "numeric_targets":          NUMERIC_TARGETS,
        "discrete_targets":         [t for t in DISCRETE_TARGETS if t in data.columns],
        "descriptor_targets":       [t for t in DESCRIPTOR_TARGETS if t in data.columns],
        "split_index":              split,
        "time_unit":                "seconds",
        "note": (
            "interarrival_seconds and duration_seconds -> regression (continuous). "
            "cpu_request and memory_request -> classification (discrete values). "
            "role_encoded, app_name_encoded, job_type_encoded -> classification."
        ),
    }
    with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n══════════════════════════════════════════════")
    print("REGRESSION METRICS — continuous targets")
    print("══════════════════════════════════════════════")
    print(reg_df[["target", "RMSE", "MAE", "MAPE", "R2", "LOG_R2"]].to_string(index=False))
    print("\n══════════════════════════════════════════════")
    print("CLASSIFICATION METRICS — discrete targets + descriptors")
    print("══════════════════════════════════════════════")
    print(cls_df[["target", "accuracy", "f1_macro"]].to_string(index=False))
    print(f"\nModels saved to: {MODELS_DIR}")
    print(f"Metrics saved to: {OUTPUTS_DIR}")


if __name__ == "__main__":
    train_model()
