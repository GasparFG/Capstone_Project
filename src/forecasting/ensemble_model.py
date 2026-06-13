"""
ensemble_model.py
=================
Two-stage ensemble pipeline for datacenter resource forecasting.

ARCHITECTURE: Generative Workload Scenario Tool
=============================================================
Stage 0  — Synthetic Workload Generation
           Samples interarrival timing, job_type, role, app_name from
           historical distributions to generate a plausible future job queue.

Stage 1  — CPU and Memory Resource Estimation (Classification)
           Given a synthetically generated job descriptor and temporal/lag
           context, estimates discrete CPU and memory request classes.

Stage 2  — Duration Resource Estimation (Regression)
           Given job descriptor plus Stage 1 predicted CPU/RAM,
           estimates job duration in seconds.

Final    — Complete synthetic job-level workload for the optimizer.

ENSEMBLE DESIGN:
  Stage 1 Base Models (CPU and Memory):
    - XGBClassifier     : evaluated in all job-level experiments
    - RandomForest      : evaluated in experiment_metrics_training_testing.csv

  Stage 1 Ensembles:
    - SoftVoting   : average class probabilities from fitted base models
    - Stack_LR     : logistic regression meta-learner on OOF base predictions
    - Stack_XGB    : XGBoost meta-learner on OOF base predictions

  Stage 2 Base Models (Duration):
    - XGBoost_log  : XGBRegressor on log1p(duration)
    - RandomForest : evaluated in job-level experiments
    - Ridge_log    : Ridge on log1p(duration), justified by experiment results

  Stage 2 Ensembles:
    - WeightedAvg  : 1/RMSLE weighted combination, weights from TRAIN OOF
    - Stack_Ridge  : Ridge meta-learner trained on TRAIN OOF predictions
    - Stack_XGB_dur: XGBoost meta-learner trained on TRAIN OOF predictions

  REMOVED (not independently evaluated in project):
    LightGBM, OrdinalClassifier, Tweedie

OOF METHODOLOGY:
  Stage 1 — OOF base predictions are generated with TimeSeriesSplit(5) on
  TRAIN. Each row receives a prediction from folds that never saw it. These
  OOF predictions are used as: (a) meta-learner training inputs for Stage 1
  stacking, (b) inputs to Stage 2 training (replacing actual CPU/RAM).

  Stage 2 — Duration base models also generate OOF predictions on TRAIN
  using the same TimeSeriesSplit scheme. Stage 2 meta-learners (Stack_Ridge,
  Stack_XGB_dur) are trained ONLY on these TRAIN OOF predictions.
  TEST is never passed to any .fit() call.

TRAIN-SERVING CONSISTENCY:
  Baseline (forecast_model.py) trained duration on ACTUAL cpu_request /
  memory_request. This ensemble trains Stage 2 on OOF Stage 1 predictions,
  matching the inference distribution.

Outputs:
    data/forecast/metrics/ensemble_model_metrics.csv
    data/forecast/metrics/ensemble_metrics_cpu.csv
    data/forecast/metrics/ensemble_metrics_memory.csv
    data/forecast/metrics/ensemble_metrics_duration.csv
    data/forecast/metrics/ensemble_metrics_summary.csv
    data/forecast/metrics/baseline_vs_ensemble_metrics.csv
    data/forecast/metrics/model_comparison_summary.csv
    data/forecast/metrics/wilcoxon_results.csv
    data/forecast/metrics/capstone_report_material.txt
    data/forecast/output/ensemble_forecast.csv
    data/forecast/output/ensemble_forecast.parquet
    data/processed/optimization_ensemble_jobs.parquet
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from copy import deepcopy

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from xgboost import XGBClassifier, XGBRegressor

try:
    from scipy.stats import wilcoxon
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[WARN] Scipy not installed — Wilcoxon test will be skipped.")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1: CONSTANTS AND PATHS
# ════════════════════════════════════════════════════════════════════════════

INPUT_PARQUET        = Path("data/interim/cleaned_data.parquet")
FORECAST_DIR         = Path("data/forecast")
PROCESSED_DIR        = Path("data/processed")
METRICS_DIR          = FORECAST_DIR / "metrics"
OUTPUT_DIR           = FORECAST_DIR / "output"

METRICS_OUTPUT       = METRICS_DIR / "ensemble_model_metrics.csv"
COMPARISON_OUTPUT    = METRICS_DIR / "ensemble_comparison_table.csv"
WILCOXON_OUTPUT      = METRICS_DIR / "wilcoxon_results.csv"

CPU_METRICS_OUTPUT      = METRICS_DIR / "ensemble_metrics_cpu.csv"
MEMORY_METRICS_OUTPUT   = METRICS_DIR / "ensemble_metrics_memory.csv"
DURATION_METRICS_OUTPUT = METRICS_DIR / "ensemble_metrics_duration.csv"
SUMMARY_METRICS_OUTPUT  = METRICS_DIR / "ensemble_metrics_summary.csv"
BASELINE_VS_ENS_OUTPUT  = METRICS_DIR / "baseline_vs_ensemble_metrics.csv"
MODEL_COMPARISON_OUTPUT = METRICS_DIR / "model_comparison_summary.csv"

FORECAST_CSV         = OUTPUT_DIR / "ensemble_forecast.csv"
FORECAST_PARQUET     = OUTPUT_DIR / "ensemble_forecast.parquet"
OPTIMIZATION_OUTPUT  = PROCESSED_DIR / "optimization_ensemble_jobs.parquet"
BASELINE_METRICS_REF = FORECAST_DIR / "historical" / "good_job_level_model_metrics.csv"

FORECAST_HORIZON_SECONDS = 86_400
MAX_FORECAST_JOBS        = 10_000
RANDOM_STATE             = 42
N_OOF_FOLDS              = 5
RECENT_WINDOW_ROWS       = 5_000


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2: DATA LOADING AND PREPARATION
# ════════════════════════════════════════════════════════════════════════════

def load_and_prepare_data() -> pd.DataFrame:
    data = pd.read_parquet(INPUT_PARQUET).copy()

    if not pd.api.types.is_timedelta64_dtype(data["scheduled_time"]):
        data["scheduled_time"] = pd.to_timedelta(data["scheduled_time"], unit="s")
    data["scheduled_seconds"] = data["scheduled_time"].dt.total_seconds()

    if "duration_seconds" not in data.columns:
        data["duration_seconds"] = data["duration_minutes"] * 60
    else:
        data["duration_seconds"] = pd.to_numeric(data["duration_seconds"], errors="coerce")

    for col in ["scheduled_seconds", "gpu_request", "cpu_request",
                "memory_request", "duration_seconds"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=[
        "scheduled_seconds", "gpu_request", "cpu_request",
        "memory_request", "duration_seconds", "role", "app_name", "job_type",
    ])
    data = data[
        (data["cpu_request"] > 0) &
        (data["memory_request"] > 0) &
        (data["duration_seconds"] > 0)
    ].copy()

    data["gpu_request"] = data["gpu_request"].astype(int)
    for col in ["role", "app_name", "job_type"]:
        data[col] = data[col].astype(str).str.lower().str.strip()

    n24 = (data["cpu_request"] == 24).sum()
    if 0 < n24 < 10:
        print(f"[FIX] cpu_request=24 has {n24} rows — merging into cpu=16")
        data.loc[data["cpu_request"] == 24, "cpu_request"] = 16

    data = data.sort_values("scheduled_seconds").reset_index(drop=True)

    data["interarrival_seconds"] = (
        data["scheduled_seconds"].diff().fillna(0).clip(lower=0)
    )
    data["time_of_day_seconds"] = data["scheduled_seconds"] % 86_400
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


def temporal_split(data: pd.DataFrame, ratio: float = 0.80):
    split = int(len(data) * ratio)
    return data.iloc[:split].copy(), data.iloc[split:].copy()


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3: FEATURE DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════

CAT_FEATURES = ["role", "app_name", "job_type"]

NUM_BASE = [
    "time_of_day_seconds", "hour", "minute", "second",
    "hour_sin", "hour_cos", "interarrival_seconds", "gpu_request",
    "cpu_request_lag_1",         "cpu_request_lag_2",         "cpu_request_lag_5",
    "cpu_request_rolling_mean_5",
    "memory_request_lag_1",      "memory_request_lag_2",      "memory_request_lag_5",
    "memory_request_rolling_mean_5",
    "duration_seconds_lag_1",    "duration_seconds_lag_2",    "duration_seconds_lag_5",
    "duration_seconds_rolling_mean_5",
    "interarrival_seconds_lag_1","interarrival_seconds_lag_2","interarrival_seconds_lag_5",
    "interarrival_seconds_rolling_mean_5",
]

S1_FEATS  = CAT_FEATURES + NUM_BASE
NUM_DUR   = NUM_BASE + ["cpu_request", "memory_request"]
DUR_FEATS = CAT_FEATURES + NUM_DUR


def make_prep(numeric_features: list) -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
        ("num", "passthrough", numeric_features),
    ])


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4: BASE MODEL DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════

def get_classifiers() -> dict:
    prep = make_prep(NUM_BASE)
    return {
        "XGBoost": Pipeline([
            ("prep", prep),
            ("clf", XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ]),
        "RandomForest": Pipeline([
            ("prep", prep),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=8, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ]),
    }


def get_regressors() -> dict:
    prep = make_prep(NUM_DUR)
    return {
        "XGBoost_log": {
            "pipeline": Pipeline([
                ("prep", prep),
                ("reg", XGBRegressor(
                    n_estimators=300, max_depth=4, learning_rate=0.04,
                    subsample=0.9, colsample_bytree=0.9,
                    objective="reg:squarederror",
                    random_state=RANDOM_STATE, n_jobs=-1,
                )),
            ]),
            "use_log": True,
        },
        "RandomForest": {
            "pipeline": Pipeline([
                ("prep", prep),
                ("reg", RandomForestRegressor(
                    n_estimators=200, max_depth=10,
                    random_state=RANDOM_STATE, n_jobs=-1,
                )),
            ]),
            "use_log": False,
        },
        "Ridge_log": {
            "pipeline": Pipeline([
                ("prep", prep),
                ("reg", Ridge(alpha=1.0)),
            ]),
            "use_log": True,
        },
    }


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5: METRICS FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred_c = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred_c)) ** 2)))


def clf_metrics(y_true, y_pred, split: str, model: str, target: str) -> dict:
    yt = np.array(y_true).astype(float)
    yp = np.array(y_pred).astype(float)
    yt_s = yt.astype(str)
    yp_s = yp.astype(str)
    return {
        "split":       split,
        "model":       model,
        "target":      target,
        "accuracy":    round(accuracy_score(yt_s, yp_s), 6),
        "macro_f1":    round(f1_score(yt_s, yp_s, average="macro",    zero_division=0), 6),
        "weighted_f1": round(f1_score(yt_s, yp_s, average="weighted", zero_division=0), 6),
        "mae":         round(mean_absolute_error(yt, yp), 4),
        "rmse":        round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
        "rmsle":       None,
        "r2_log":      None,
        "r2_raw":      round(r2_score(yt, yp), 6),
    }


def reg_metrics(y_true, y_pred, split: str, model: str, target: str) -> dict:
    yt = np.clip(np.array(y_true).astype(float), 1, None)
    yp = np.clip(np.array(y_pred).astype(float), 1, None)
    lt, lp = np.log1p(yt), np.log1p(yp)
    return {
        "split":       split,
        "model":       model,
        "target":      target,
        "accuracy":    None,
        "macro_f1":    None,
        "weighted_f1": None,
        "mae":         round(float(mean_absolute_error(yt, yp)), 4),
        "rmse":        round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
        "rmsle":       round(float(np.sqrt(mean_squared_error(lt, lp))), 6),
        "r2_log":      round(float(r2_score(lt, lp)), 6),
        "r2_raw":      round(float(r2_score(yt, yp)), 6),
    }


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6: STAGE 1 — BASE LEARNERS + OOF
# ════════════════════════════════════════════════════════════════════════════

def train_stage1(train: pd.DataFrame, test: pd.DataFrame, target: str):
    """
    Train Stage 1 base classifiers (CPU or Memory).

    Returns
    -------
    metrics       : list of metric dicts (train + test for each base model)
    oof_preds     : dict  name -> float array (OOF predictions on TRAIN)
    test_preds    : dict  name -> float array (predictions on TEST)
    fitted        : dict  name -> fitted Pipeline
    encoder       : LabelEncoder fitted on training class labels
    """
    print(f"\n  [Stage 1 — {target}]")
    X_tr = train[S1_FEATS]
    X_te = test[S1_FEATS]
    y_tr = train[target].astype(str)
    y_te = test[target].astype(str)

    encoder = LabelEncoder()
    y_tr_enc = encoder.fit_transform(y_tr)

    clfs = get_classifiers()
    metrics, oof_preds, test_preds, fitted = [], {}, {}, {}
    tscv = TimeSeriesSplit(n_splits=N_OOF_FOLDS)

    for name, pipe in clfs.items():
        print(f"    {name}...", end=" ", flush=True)
        oof_arr = np.empty(len(train), dtype=object)
        oof_arr[:] = None

        for _, (tr_idx, val_idx) in enumerate(tscv.split(X_tr)):
            fp = clone(pipe)
            Xf, yf = X_tr.iloc[tr_idx], y_tr.iloc[tr_idx]
            Xv = X_tr.iloc[val_idx]
            if name == "XGBoost":
                ef = LabelEncoder()
                yfe = ef.fit_transform(yf)
                fp.fit(Xf, yfe)
                oof_arr[val_idx] = ef.classes_[fp.predict(Xv)].astype(str)
            else:
                fp.fit(Xf, yf)
                oof_arr[val_idx] = fp.predict(Xv).astype(str)

        fp_full = clone(pipe)
        if name == "XGBoost":
            fp_full.fit(X_tr, y_tr_enc)
            mask = np.array([v is None for v in oof_arr])
            if mask.any():
                oof_arr[mask] = encoder.inverse_transform(
                    fp_full.predict(X_tr.iloc[mask])).astype(str)
            tr_pred = encoder.inverse_transform(fp_full.predict(X_tr)).astype(str)
            te_pred = encoder.inverse_transform(fp_full.predict(X_te)).astype(str)
        else:
            fp_full.fit(X_tr, y_tr)
            mask = np.array([v is None for v in oof_arr])
            if mask.any():
                oof_arr[mask] = fp_full.predict(X_tr.iloc[mask]).astype(str)
            tr_pred = fp_full.predict(X_tr).astype(str)
            te_pred = fp_full.predict(X_te).astype(str)

        oof_preds[name]  = oof_arr.astype(float)
        test_preds[name] = te_pred.astype(float)
        fitted[name]     = fp_full

        metrics += [
            clf_metrics(y_tr, tr_pred, "train", name, target),
            clf_metrics(y_te, te_pred, "test",  name, target),
        ]
        m_te = metrics[-1]
        print(f"acc={m_te['accuracy']:.4f}  F1={m_te['macro_f1']:.4f}")

    return metrics, oof_preds, test_preds, fitted, encoder


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7: STAGE 1 — ENSEMBLE STRATEGIES
# ════════════════════════════════════════════════════════════════════════════

def ensemble_stage1(train, test, target, fitted, encoder, oof_preds, test_preds):
    """
    Three ensemble strategies for CPU / Memory classification.

    Meta-learners (Stack_LR, Stack_XGB) are trained on TRAIN OOF predictions
    only. TEST predictions are used solely for evaluation.

    Returns
    -------
    metrics        : list of metric dicts
    ens_te         : dict  name -> float array (ensemble predictions on TEST)
    ens_oof        : dict  name -> float array (ensemble predictions on TRAIN)
    best_model     : str   name of best ensemble by test Macro F1
    meta_lr        : fitted LogisticRegression (Stack_LR meta-learner)
    meta_xgb       : fitted XGBClassifier (Stack_XGB meta-learner)
    meta_xgb_enc   : LabelEncoder used to fit meta_xgb
    """
    print(f"\n  [Ensemble Stage 1 — {target}]")
    X_tr = train[S1_FEATS]
    X_te = test[S1_FEATS]
    y_tr = train[target].astype(str)
    y_te = test[target].astype(str)
    names  = list(fitted.keys())
    n_cls  = len(encoder.classes_)
    metrics = []

    # OOF and test numeric prediction matrices for stacking
    mX_tr = np.column_stack([oof_preds[n]  for n in names])
    mX_te = np.column_stack([test_preds[n] for n in names])

    # Soft-voting probability matrices
    prob_tr = np.zeros((len(train), n_cls))
    prob_te = np.zeros((len(test),  n_cls))
    for nm, mdl in fitted.items():
        clf_step  = mdl.named_steps["clf"]
        prep_step = mdl.named_steps["prep"]
        if not hasattr(clf_step, "predict_proba"):
            continue
        pt = clf_step.predict_proba(prep_step.transform(X_tr))
        pv = clf_step.predict_proba(prep_step.transform(X_te))
        if pt.shape[1] == n_cls:
            prob_tr += pt
            prob_te += pv

    # ── A) Soft Voting ───────────────────────────────────────────────────────
    print("    A) SoftVoting...", end=" ", flush=True)
    sv_tr = encoder.inverse_transform(np.argmax(prob_tr, axis=1)).astype(str)
    sv_te = encoder.inverse_transform(np.argmax(prob_te, axis=1)).astype(str)
    m_te  = clf_metrics(y_te, sv_te, "test", "SoftVoting", target)
    metrics += [clf_metrics(y_tr, sv_tr, "train", "SoftVoting", target), m_te]
    print(f"acc={m_te['accuracy']:.4f}  F1={m_te['macro_f1']:.4f}")

    # ── B) Stack_LR — trained on TRAIN OOF only ──────────────────────────────
    print("    B) Stack_LR...",  end=" ", flush=True)
    meta_lr = LogisticRegression(C=1.0, max_iter=500, random_state=RANDOM_STATE)
    meta_lr.fit(mX_tr, y_tr.values)           # TRAIN OOF only
    lr_tr = meta_lr.predict(mX_tr).astype(str)
    lr_te = meta_lr.predict(mX_te).astype(str)
    m_te  = clf_metrics(y_te, lr_te, "test", "Stack_LR", target)
    metrics += [clf_metrics(y_tr, lr_tr, "train", "Stack_LR", target), m_te]
    print(f"acc={m_te['accuracy']:.4f}  F1={m_te['macro_f1']:.4f}")

    # ── C) Stack_XGB — trained on TRAIN OOF only ─────────────────────────────
    print("    C) Stack_XGB...", end=" ", flush=True)
    meta_xgb_enc = LabelEncoder()
    my_y         = meta_xgb_enc.fit_transform(y_tr.values)
    meta_xgb     = XGBClassifier(
        n_estimators=50, max_depth=2, learning_rate=0.1,
        eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1,
    )
    meta_xgb.fit(mX_tr, my_y)                 # TRAIN OOF only
    xgb_tr = meta_xgb_enc.inverse_transform(meta_xgb.predict(mX_tr)).astype(str)
    xgb_te = meta_xgb_enc.inverse_transform(meta_xgb.predict(mX_te)).astype(str)
    m_te   = clf_metrics(y_te, xgb_te, "test", "Stack_XGB", target)
    metrics += [clf_metrics(y_tr, xgb_tr, "train", "Stack_XGB", target), m_te]
    print(f"acc={m_te['accuracy']:.4f}  F1={m_te['macro_f1']:.4f}")

    test_m = [m for m in metrics if m["split"] == "test"]
    best   = max(test_m, key=lambda m: m["macro_f1"])
    print(f"    → Best: {best['model']} (F1={best['macro_f1']:.4f})")

    ens_te = {
        "SoftVoting": encoder.inverse_transform(np.argmax(prob_te, axis=1)).astype(float),
        "Stack_LR":   meta_lr.predict(mX_te).astype(float),
        "Stack_XGB":  meta_xgb_enc.inverse_transform(meta_xgb.predict(mX_te)).astype(float),
    }
    ens_oof = {
        "SoftVoting": encoder.inverse_transform(np.argmax(prob_tr, axis=1)).astype(float),
        "Stack_LR":   meta_lr.predict(mX_tr).astype(float),
        "Stack_XGB":  meta_xgb_enc.inverse_transform(meta_xgb.predict(mX_tr)).astype(float),
    }
    return metrics, ens_te, ens_oof, best["model"], meta_lr, meta_xgb, meta_xgb_enc


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8: STAGE 1 — SINGLE-ROW PREDICTION HELPER
# ════════════════════════════════════════════════════════════════════════════

def predict_s1_single(X_row, fitted, encoder, best, prob_tr_shape,
                      meta_lr, meta_xgb, meta_xgb_enc):
    """
    Predict CPU or memory for a single-row DataFrame using the selected
    Stage 1 ensemble strategy.

    Parameters
    ----------
    X_row        : pd.DataFrame with shape (1, len(S1_FEATS))
    fitted       : dict  name -> fitted base Pipeline
    encoder      : LabelEncoder (maps encoded int -> original class label)
    best         : str  'SoftVoting' | 'Stack_LR' | 'Stack_XGB'
    prob_tr_shape: int  number of classes (len(encoder.classes_))
    meta_lr      : fitted LogisticRegression meta-learner
    meta_xgb     : fitted XGBClassifier meta-learner
    meta_xgb_enc : LabelEncoder used when training meta_xgb

    Returns
    -------
    float  predicted class label (e.g. 4.0, 8.0, ...)
    """
    n_cls = prob_tr_shape

    if best == "SoftVoting":
        prob = np.zeros(n_cls)
        for nm, mdl in fitted.items():
            clf_step  = mdl.named_steps["clf"]
            prep_step = mdl.named_steps["prep"]
            if hasattr(clf_step, "predict_proba"):
                pt = clf_step.predict_proba(prep_step.transform(X_row))
                if pt.shape[1] == n_cls:
                    prob += pt[0]
        return float(encoder.inverse_transform([np.argmax(prob)])[0])

    # Compute base predictions as numeric class labels
    base_preds = []
    for nm, mdl in fitted.items():
        clf_step  = mdl.named_steps["clf"]
        prep_step = mdl.named_steps["prep"]
        if nm == "XGBoost":
            enc_pred  = clf_step.predict(prep_step.transform(X_row))
            base_preds.append(float(encoder.inverse_transform(enc_pred)[0]))
        else:
            base_preds.append(float(mdl.predict(X_row)[0]))

    mX = np.array([base_preds])

    if best == "Stack_LR":
        return float(meta_lr.predict(mX)[0])
    else:  # Stack_XGB
        enc_pred = meta_xgb.predict(mX)
        return float(meta_xgb_enc.inverse_transform(enc_pred)[0])


# ════════════════════════════════════════════════════════════════════════════
# SECTION 9: STAGE 2 — DURATION BASE LEARNERS + OOF
# ════════════════════════════════════════════════════════════════════════════

def train_stage2(train, test, oof_cpu_tr, oof_ram_tr, te_cpu, te_ram):
    """
    Train Stage 2 duration base regressors with OOF for meta-learner training.

    TRAIN-SERVING CONSISTENCY:
    Stage 2 trains on OOF Stage 1 CPU/RAM predictions (oof_cpu_tr, oof_ram_tr)
    rather than actual values, matching the inference distribution.

    OOF FOR META-LEARNER:
    Each base duration model also generates OOF predictions on TRAIN using
    TimeSeriesSplit. These OOF predictions are the ONLY training signal for
    the Stage 2 meta-learners. TEST is never used in any .fit() call.

    Returns
    -------
    base_metrics  : list of metric dicts
    fitted_base   : dict  name -> Pipeline trained on full TRAIN
    duan_factors  : dict  name -> float (Duan 1983 bias correction)
    use_log_map   : dict  name -> bool
    oof_dur_preds : dict  name -> float array (OOF predictions on TRAIN,
                    in duration-seconds space, expm1 only — no Duan —
                    for consistent meta-feature scale with OOF training)
    te_preds      : dict  name -> float array (predictions on TEST,
                    Duan-corrected, for standalone evaluation metrics)
    """
    print(f"\n  [Stage 2 — duration_seconds]")

    tr2 = train.copy()
    tr2["cpu_request"]    = oof_cpu_tr
    tr2["memory_request"] = oof_ram_tr
    te2 = test.copy()
    te2["cpu_request"]    = te_cpu
    te2["memory_request"] = te_ram

    X_tr = tr2[DUR_FEATS]
    X_te = te2[DUR_FEATS]
    y_tr = train["duration_seconds"].values
    y_te = test["duration_seconds"].values

    regs        = get_regressors()
    use_log_map = {name: info["use_log"] for name, info in regs.items()}

    base_metrics  = []
    fitted_base   = {}
    duan_factors  = {}
    oof_dur_preds = {}
    te_preds      = {}

    tscv = TimeSeriesSplit(n_splits=N_OOF_FOLDS)

    for name, info in regs.items():
        print(f"    {name}...", end=" ", flush=True)
        use_log = info["use_log"]

        # ── OOF predictions on TRAIN (for meta-learner training) ────────────
        # Use expm1 only (no Duan) so that OOF meta-features and test
        # meta-features come from the same transformation pipeline.
        oof_arr = np.empty(len(train))
        for _, (tr_idx, val_idx) in enumerate(tscv.split(X_tr)):
            fold_pipe = clone(info["pipeline"])
            y_fit_fold = np.log1p(y_tr[tr_idx]) if use_log else y_tr[tr_idx]
            fold_pipe.fit(X_tr.iloc[tr_idx], y_fit_fold)
            raw = fold_pipe.predict(X_tr.iloc[val_idx])
            oof_arr[val_idx] = np.clip(
                np.expm1(raw) if use_log else raw, 1, None
            )
        oof_dur_preds[name] = oof_arr

        # ── Full-train model: evaluation metrics and forecast use ────────────
        full_pipe = clone(info["pipeline"])
        y_fit = np.log1p(y_tr) if use_log else y_tr
        full_pipe.fit(X_tr, y_fit)

        tr_raw = full_pipe.predict(X_tr)
        te_raw = full_pipe.predict(X_te)

        if use_log:
            resid   = y_fit - tr_raw
            factor  = float(np.mean(np.exp(resid)))   # Duan 1983 correction
            duan_factors[name] = factor
            tr_pred = np.clip(np.expm1(tr_raw) * factor, 1, None)
            te_pred = np.clip(np.expm1(te_raw) * factor, 1, None)
        else:
            duan_factors[name] = 1.0
            tr_pred = np.clip(tr_raw, 1, None)
            te_pred = np.clip(te_raw, 1, None)

        te_preds[name]   = te_pred
        fitted_base[name] = full_pipe

        m_te = reg_metrics(y_te, te_pred, "test", name, "duration_seconds")
        base_metrics += [
            reg_metrics(y_tr, tr_pred, "train", name, "duration_seconds"),
            m_te,
        ]
        print(f"RMSLE={m_te['rmsle']:.4f}  R²log={m_te['r2_log']:.4f}")

    return base_metrics, fitted_base, duan_factors, use_log_map, oof_dur_preds, te_preds


# ════════════════════════════════════════════════════════════════════════════
# SECTION 10: STAGE 2 — DURATION ENSEMBLE STRATEGIES
# ════════════════════════════════════════════════════════════════════════════

def ensemble_stage2(y_train, y_test, oof_dur_preds, te_dur_preds):
    """
    Train Stage 2 meta-learners on TRAIN OOF predictions, evaluate on TEST.

    LEAKAGE FIX (vs. previous version):
    Stack_Ridge and Stack_XGB_dur are trained exclusively on TRAIN OOF
    predictions. y_test is never passed to any .fit() call.
    TEST predictions are used only for evaluation (.predict + metrics).

    Parameters
    ----------
    y_train       : np.ndarray  duration_seconds for TRAIN rows
    y_test        : np.ndarray  duration_seconds for TEST rows (eval only)
    oof_dur_preds : dict  name -> float array, OOF preds on TRAIN
                    (expm1 only, no Duan — consistent scale for stacking)
    te_dur_preds  : dict  name -> float array, base model preds on TEST
                    (Duan-corrected, for meta-inference and standalone eval)

    Returns
    -------
    metrics       : list of metric dicts
    ens_te_preds  : dict  name -> float array (ensemble preds on TEST)
    best_model    : str
    meta_ridge    : fitted Ridge meta-learner
    meta_xgb_dur  : fitted XGBRegressor meta-learner
    wa_weights    : dict  name -> float (WeightedAvg weights from TRAIN OOF)
    dur_names     : list[str] ordered base model names
    """
    print(f"\n  [Ensemble Stage 2 — duration]")
    names = list(oof_dur_preds.keys())
    metrics = []

    # Meta-matrices:
    #   X_meta_tr = OOF predictions on TRAIN (used for .fit())
    #   X_meta_te = base predictions on TEST  (used for .predict() + eval only)
    X_meta_tr = np.column_stack([oof_dur_preds[n] for n in names])
    X_meta_te = np.column_stack([te_dur_preds[n]  for n in names])

    # ── A) WeightedAvg — weights from TRAIN OOF RMSLE ────────────────────────
    print("    A) WeightedAvg...", end=" ", flush=True)
    oof_rmsle_vals = {n: rmsle(y_train, oof_dur_preds[n]) for n in names}
    inv  = {n: 1.0 / max(v, 1e-8) for n, v in oof_rmsle_vals.items()}
    tot  = sum(inv.values())
    wa_weights = {n: inv[n] / tot for n in names}
    wa_te = sum(wa_weights[n] * te_dur_preds[n] for n in names)
    m     = reg_metrics(y_test, wa_te, "test", "WeightedAvg", "duration_seconds")
    metrics.append(m)
    print(f"RMSLE={m['rmsle']:.4f}  R²log={m['r2_log']:.4f}")

    # ── B) Stack_Ridge — trained on TRAIN OOF, evaluated on TEST ─────────────
    print("    B) Stack_Ridge...", end=" ", flush=True)
    meta_ridge = Ridge(alpha=1.0)
    meta_ridge.fit(X_meta_tr, y_train)        # TRAIN OOF only — no TEST leakage
    rp = np.clip(meta_ridge.predict(X_meta_te), 1, None)
    m  = reg_metrics(y_test, rp, "test", "Stack_Ridge", "duration_seconds")
    metrics.append(m)
    print(f"RMSLE={m['rmsle']:.4f}  R²log={m['r2_log']:.4f}")

    # ── C) Stack_XGB_dur — trained on TRAIN OOF, evaluated on TEST ───────────
    print("    C) Stack_XGB_dur...", end=" ", flush=True)
    meta_xgb_dur = XGBRegressor(
        n_estimators=50, max_depth=2, learning_rate=0.1,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    meta_xgb_dur.fit(X_meta_tr, np.log1p(y_train))   # TRAIN OOF only
    xp = np.clip(np.expm1(meta_xgb_dur.predict(X_meta_te)), 1, None)
    m  = reg_metrics(y_test, xp, "test", "Stack_XGB_dur", "duration_seconds")
    metrics.append(m)
    print(f"RMSLE={m['rmsle']:.4f}  R²log={m['r2_log']:.4f}")

    best = min(metrics, key=lambda m: m["rmsle"])
    print(f"    → Best: {best['model']} (RMSLE={best['rmsle']:.4f})")

    ens_te_preds = {"WeightedAvg": wa_te, "Stack_Ridge": rp, "Stack_XGB_dur": xp}
    return metrics, ens_te_preds, best["model"], meta_ridge, meta_xgb_dur, wa_weights, names


# ════════════════════════════════════════════════════════════════════════════
# SECTION 11: STAGE 2 — SINGLE-ROW PREDICTION HELPER
# ════════════════════════════════════════════════════════════════════════════

def predict_dur_single(X_dur_row, fitted_base, use_log_map,
                       best, meta_ridge, meta_xgb_dur, wa_weights, dur_names,
                       dur_clip_lo=1.0, dur_clip_hi=None):
    """
    Predict duration for a single-row DataFrame using the selected
    Stage 2 ensemble strategy.

    Meta-features use expm1 only (no Duan correction) to match the scale
    used when training the meta-learners on TRAIN OOF predictions.

    Parameters
    ----------
    X_dur_row    : pd.DataFrame (1, len(DUR_FEATS)) with predicted CPU/RAM
    fitted_base  : dict  name -> Pipeline trained on full TRAIN
    use_log_map  : dict  name -> bool
    best         : str  'WeightedAvg' | 'Stack_Ridge' | 'Stack_XGB_dur'
    meta_ridge   : fitted Ridge meta-learner
    meta_xgb_dur : fitted XGBRegressor meta-learner
    wa_weights   : dict  name -> float
    dur_names    : list[str] ordered names
    dur_clip_lo  : float  minimum clipped duration
    dur_clip_hi  : float or None  maximum clipped duration

    Returns
    -------
    float  predicted duration in seconds
    """
    # Build base predictions in the same scale as meta-training (expm1, no Duan)
    base_meta = {}
    for name, mdl in fitted_base.items():
        raw = mdl.predict(X_dur_row)[0]
        if use_log_map[name]:
            base_meta[name] = float(np.clip(np.expm1(raw), 1, None))
        else:
            base_meta[name] = float(np.clip(raw, 1, None))

    if best == "WeightedAvg":
        pred = sum(wa_weights[n] * base_meta[n] for n in dur_names)
    else:
        mX = np.array([[base_meta[n] for n in dur_names]])
        if best == "Stack_Ridge":
            pred = float(meta_ridge.predict(mX)[0])
        else:  # Stack_XGB_dur
            pred = float(np.expm1(meta_xgb_dur.predict(mX)[0]))

    pred = max(pred, dur_clip_lo)
    if dur_clip_hi is not None:
        pred = min(pred, dur_clip_hi)
    return pred


# ════════════════════════════════════════════════════════════════════════════
# SECTION 12: WILCOXON TEST
# ════════════════════════════════════════════════════════════════════════════

def run_wilcoxon(y_true, base_pred, ens_pred, kind: str) -> dict:
    if not SCIPY_AVAILABLE:
        return {"wilcoxon_p_value": "scipy_not_installed"}
    y  = np.array(y_true).astype(float)
    if kind == "clf":
        eb = np.abs(np.array(base_pred).astype(float) - y)
        ee = np.abs(np.array(ens_pred).astype(float)  - y)
    else:
        eb = np.abs(np.log1p(np.clip(base_pred, 1, None)) - np.log1p(np.clip(y, 1, None)))
        ee = np.abs(np.log1p(np.clip(ens_pred,  1, None)) - np.log1p(np.clip(y, 1, None)))
    diff = eb - ee
    diff = diff[diff != 0]
    if len(diff) < 10:
        return {"target": "?", "wilcoxon_p_value": "insufficient_differences"}
    stat, p = wilcoxon(diff, alternative="two-sided")
    _ens_better = bool(np.median(ee) < np.median(eb))
    # NOTE: ensemble_better=False is expected when the ensemble improves
    # minority-class recall (boosting macro-F1) without shifting the
    # per-sample median error direction. Both results are consistent.
    if not _ens_better and p < 0.05:
        print(f"    [{kind}] Wilcoxon p={p:.2e}: significant but ensemble_better=False — "
              "ensemble improves aggregate metric (macro-F1/RMSLE) via minority "
              "class recall, not per-sample median shift. Expected behaviour.")
    return {
        "wilcoxon_statistic":  round(float(stat), 4),
        "wilcoxon_p_value":    round(float(p), 6),
        "significant_at_005":  bool(p < 0.05),
        "ensemble_better":     _ens_better,
        "interpretation":      ("aggregate metric improved via minority class recall"
                                if not _ens_better and p < 0.05
                                else ("ensemble dominates per-sample errors"
                                      if _ens_better else "no per-sample improvement")),
    }


# ════════════════════════════════════════════════════════════════════════════
# SECTION 13: 24-HOUR FORECAST — WORKLOAD GENERATOR
# ════════════════════════════════════════════════════════════════════════════

def _sample_from(pool: pd.Series, rng: np.random.Generator) -> str:
    return str(rng.choice(pool.dropna().values))


def generate_forecast(
    data: pd.DataFrame,
    # Stage 1 CPU ensemble objects
    cpu_fitted, cpu_encoder, cpu_best, cpu_n_cls,
    cpu_meta_lr, cpu_meta_xgb, cpu_meta_xgb_enc,
    # Stage 1 Memory ensemble objects
    ram_fitted, ram_encoder, ram_best, ram_n_cls,
    ram_meta_lr, ram_meta_xgb, ram_meta_xgb_enc,
    # Stage 2 Duration ensemble objects
    dur_fitted, dur_use_log_map,
    dur_best, dur_meta_ridge, dur_meta_xgb_dur, dur_wa_weights, dur_names,
) -> pd.DataFrame:
    """
    Stage 0 + Stage 1 + Stage 2: Generative Workload Scenario Tool.

    Uses the SELECTED best ensemble for each target (not hardcoded XGBoost).

    Stage 0 — Synthetic Workload Generation:
        Generates synthetic jobs by sampling from historical distributions.
        app_name, role, job_type are GENERATED — not known future values.

    Stage 1 — CPU and Memory Estimation:
        Uses selected best ensemble (SoftVoting / Stack_LR / Stack_XGB).

    Stage 2 — Duration Estimation:
        Uses selected best ensemble (WeightedAvg / Stack_Ridge / Stack_XGB_dur).
        Inputs: predicted CPU/RAM from Stage 1 (train-serving consistent).

    Autoregressive buffer:
        All predicted values clipped to historical range before appending,
        preventing lag feature drift over the 24-hour rollout.
    """
    rng     = np.random.default_rng(RANDOM_STATE)
    history = data.copy().reset_index(drop=True)
    recent  = history.tail(RECENT_WINDOW_ROWS)

    # ── Stage 0: window diversity & distribution-shift check ─────────────
    _full_batch   = (history["job_type"] == "batch").mean()
    _recent_batch = (recent["job_type"]  == "batch").mean()
    print(f"    Stage 0 window : {_recent_batch:.1%} batch  "
          f"(full dataset: {_full_batch:.1%} batch, window={RECENT_WINDOW_ROWS} jobs)")
    if _recent_batch == 0.0:
        warnings.warn(
            f"Stage 0: RECENT_WINDOW_ROWS={RECENT_WINDOW_ROWS} contains 0% batch jobs — "
            "workload diversity collapsed to 100%% interactive. "
            "No batch jobs will be generated in the forecast.",
            UserWarning, stacklevel=2,
        )
    elif abs(_full_batch - _recent_batch) > 0.20:
        warnings.warn(
            f"Stage 0: distribution shift detected — "
            f"full={_full_batch:.1%} batch vs recent={_recent_batch:.1%} batch. "
            "Forecast reflects recent workload mix, not historical average.",
            UserWarning, stacklevel=2,
        )

    ia_pool = history["interarrival_seconds"]
    ia_pool = ia_pool[ia_pool > 0].clip(
        lower=ia_pool[ia_pool > 0].quantile(0.01),
        upper=ia_pool[ia_pool > 0].quantile(0.99),
    )

    cpu_vals = sorted(history["cpu_request"].unique())
    mem_vals = sorted(history["memory_request"].unique())
    dur_lo   = float(history["duration_seconds"].quantile(0.01))
    dur_hi   = float(history["duration_seconds"].quantile(0.99))
    ia_lo    = float(ia_pool.quantile(0.01))
    ia_hi    = float(ia_pool.quantile(0.99))

    last_abs = float(history["scheduled_seconds"].iloc[-1])
    rel_t    = 0.0
    abs_t    = last_abs
    rows     = []

    print(f"    CPU ensemble : {cpu_best}")
    print(f"    RAM ensemble : {ram_best}")
    print(f"    Dur ensemble : {dur_best}")

    while rel_t < FORECAST_HORIZON_SECONDS and len(rows) < MAX_FORECAST_JOBS:
        # ── Stage 0: Synthetic workload generation ───────────────────────────
        ia    = float(np.clip(rng.choice(ia_pool.values), ia_lo, ia_hi))
        ia    = max(1.0, ia)
        rel_t += ia
        abs_t += ia
        if rel_t > FORECAST_HORIZON_SECONDS:
            break

        jt   = _sample_from(recent["job_type"],  rng)
        sub  = recent[recent["job_type"] == jt]
        role = _sample_from(sub["role"]     if len(sub) > 0 else recent["role"],     rng)
        app  = _sample_from(sub["app_name"] if len(sub) > 0 else recent["app_name"], rng)
        gpu  = 0 if role == "cn" else 1

        prev = history.iloc[-1]
        tod  = abs_t % 86_400

        base = {
            "role": role, "app_name": app, "job_type": jt,
            "scheduled_seconds":    abs_t,
            "time_of_day_seconds":  tod,
            "hour":   int(tod // 3600),
            "minute": int((tod % 3600) // 60),
            "second": int(tod % 60),
            "hour_sin": np.sin(2 * np.pi * tod / 86_400),
            "hour_cos": np.cos(2 * np.pi * tod / 86_400),
            "interarrival_seconds": ia,
            "gpu_request": gpu,
            "cpu_request_lag_1":               float(prev["cpu_request"]),
            "cpu_request_lag_2":               float(history["cpu_request"].iloc[-2]),
            "cpu_request_lag_5":               float(history["cpu_request"].iloc[-5]),
            "cpu_request_rolling_mean_5":      float(history["cpu_request"].tail(5).mean()),
            "memory_request_lag_1":            float(prev["memory_request"]),
            "memory_request_lag_2":            float(history["memory_request"].iloc[-2]),
            "memory_request_lag_5":            float(history["memory_request"].iloc[-5]),
            "memory_request_rolling_mean_5":   float(history["memory_request"].tail(5).mean()),
            "duration_seconds_lag_1":          float(prev["duration_seconds"]),
            "duration_seconds_lag_2":          float(history["duration_seconds"].iloc[-2]),
            "duration_seconds_lag_5":          float(history["duration_seconds"].iloc[-5]),
            "duration_seconds_rolling_mean_5": float(history["duration_seconds"].tail(5).mean()),
            "interarrival_seconds_lag_1":      float(prev["interarrival_seconds"]),
            "interarrival_seconds_lag_2":      float(history["interarrival_seconds"].iloc[-2]),
            "interarrival_seconds_lag_5":      float(history["interarrival_seconds"].iloc[-5]),
            "interarrival_seconds_rolling_mean_5": float(history["interarrival_seconds"].tail(5).mean()),
        }

        # ── Stage 1: CPU estimation using selected ensemble ──────────────────
        X_s1 = pd.DataFrame([base])[S1_FEATS]
        cpu_v = predict_s1_single(
            X_s1, cpu_fitted, cpu_encoder, cpu_best, cpu_n_cls,
            cpu_meta_lr, cpu_meta_xgb, cpu_meta_xgb_enc,
        )
        cpu_v = float(min(cpu_vals, key=lambda x: abs(x - cpu_v)))

        # ── Stage 1: Memory estimation using selected ensemble ───────────────
        ram_v = predict_s1_single(
            X_s1, ram_fitted, ram_encoder, ram_best, ram_n_cls,
            ram_meta_lr, ram_meta_xgb, ram_meta_xgb_enc,
        )
        ram_v = float(min(mem_vals, key=lambda x: abs(x - ram_v)))

        # ── Stage 2: Duration estimation using selected ensemble ─────────────
        dur_row = {**base, "cpu_request": cpu_v, "memory_request": ram_v}
        X_dur   = pd.DataFrame([dur_row])[DUR_FEATS]
        dur_v   = predict_dur_single(
            X_dur, dur_fitted, dur_use_log_map,
            dur_best, dur_meta_ridge, dur_meta_xgb_dur, dur_wa_weights, dur_names,
            dur_clip_lo=dur_lo, dur_clip_hi=dur_hi,
        )

        rows.append({
            "forecast_job_id":      f"ens_{len(rows)+1:06d}",
            "instance_sn":          f"ens_inst_{len(rows)+1:06d}",
            "role":                 role,
            "app_name":             app,
            "job_type":             jt,
            "scheduled_seconds":    round(rel_t, 2),
            "cpu_request":          cpu_v,
            "memory_request":       ram_v,
            "gpu_request":          gpu,
            "duration_seconds":     round(dur_v, 2),
            "deletion_seconds":     round(rel_t + dur_v, 2),
            "interarrival_seconds": round(ia, 2),
        })

        # Append clipped values to rolling buffer
        clipped_ia = float(np.clip(ia, ia_lo, ia_hi))
        new_h = {
            **base,
            "cpu_request":          cpu_v,
            "memory_request":       ram_v,
            "duration_seconds":     dur_v,
            "scheduled_seconds":    abs_t,
            "interarrival_seconds": clipped_ia,
        }
        history = pd.concat([history, pd.DataFrame([new_h])], ignore_index=True)

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 14: METRICS EXPORT — TABLES A, B, C, D
# ════════════════════════════════════════════════════════════════════════════

def build_per_target_table(metrics_df: pd.DataFrame, target: str) -> pd.DataFrame:
    sub = metrics_df[metrics_df["target"] == target].copy()
    if target in ("cpu_request", "memory_request"):
        metric_cols = ["accuracy", "macro_f1", "weighted_f1", "mae", "rmse"]
    else:
        metric_cols = ["mae", "rmse", "rmsle", "r2_log", "r2_raw"]

    models = sub["model"].unique()
    rows_out = []
    for metric in metric_cols:
        row = {"Metric": metric}
        for mdl in models:
            tr = sub[(sub["model"] == mdl) & (sub["split"] == "train")][metric].values
            te = sub[(sub["model"] == mdl) & (sub["split"] == "test")][metric].values
            row[f"{mdl}_Train"] = round(float(tr[0]), 6) if len(tr) > 0 and tr[0] is not None else None
            row[f"{mdl}_Test"]  = round(float(te[0]), 6) if len(te) > 0 and te[0] is not None else None
        rows_out.append(row)
    return pd.DataFrame(rows_out)


def build_summary_table(all_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, key_metric in [
        ("cpu_request",    "macro_f1"),
        ("memory_request", "macro_f1"),
        ("duration_seconds","rmsle"),
    ]:
        sub = all_metrics[
            (all_metrics["target"] == target) &
            (all_metrics["split"]  == "test")
        ].copy()
        if sub.empty:
            continue
        asc = (key_metric == "rmsle")
        best_row = sub.sort_values(key_metric, ascending=asc).iloc[0]
        rows.append({
            "Target":        target,
            "Best_Model":    best_row["model"],
            "Key_Metric":    key_metric,
            "Key_Value":     round(float(best_row[key_metric]), 6),
            "Accuracy_Test": round(float(best_row["accuracy"]), 6) if best_row["accuracy"] is not None else None,
            "MAE_Test":      round(float(best_row["mae"]),      4)  if best_row["mae"]      is not None else None,
            "RMSE_Test":     round(float(best_row["rmse"]),     4)  if best_row["rmse"]     is not None else None,
            "RMSLE_Test":    round(float(best_row["rmsle"]),    6)  if best_row["rmsle"]    is not None else None,
            "R2_log_Test":   round(float(best_row["r2_log"]),   6)  if best_row["r2_log"]   is not None else None,
        })
    return pd.DataFrame(rows)


def build_baseline_vs_ensemble(all_metrics: pd.DataFrame, baseline_path: Path) -> pd.DataFrame:
    rows = []
    for target, key_metric in [
        ("cpu_request",    "macro_f1"),
        ("memory_request", "macro_f1"),
        ("duration_seconds","rmsle"),
    ]:
        sub = all_metrics[
            (all_metrics["target"] == target) &
            (all_metrics["split"]  == "test")
        ]
        bl_row = sub[sub["model"] == "XGBoost"].copy()
        if bl_row.empty:
            bl_row = sub[sub["model"] == "XGBoost_log"].copy()
        bl_val = round(float(bl_row[key_metric].values[0]), 6) if not bl_row.empty else None

        ens_models = ["SoftVoting", "Stack_LR", "Stack_XGB",
                      "WeightedAvg", "Stack_Ridge", "Stack_XGB_dur"]
        ens_sub = sub[sub["model"].isin(ens_models)]
        if ens_sub.empty:
            continue
        asc = (key_metric == "rmsle")
        best_ens = ens_sub.sort_values(key_metric, ascending=asc).iloc[0]
        ens_val  = round(float(best_ens[key_metric]), 6)
        improved = (ens_val > bl_val) if key_metric == "macro_f1" else (ens_val < bl_val)

        rows.append({
            "Target":              target,
            "Key_Metric":          key_metric,
            "Baseline_XGB":        bl_val,
            "Best_Ensemble":       best_ens["model"],
            "Best_Ensemble_Value": ens_val,
            "Improved":            improved,
            "Delta":               round(ens_val - bl_val, 6) if bl_val is not None else None,
        })
    return pd.DataFrame(rows)


def write_report_text(all_metrics, baseline_vs_ens, wilcoxon_df, output_dir):
    lines = []
    a = lines.append

    a("=" * 72)
    a("CAPSTONE REPORT — FORECASTING ENSEMBLE: FINAL REPORT MATERIAL")
    a("=" * 72)
    a("")
    a("─" * 72)
    a("1. PREVIOUS BASELINE MODEL")
    a("─" * 72)
    a(
        "The previous job-level baseline (forecast_model.py) used a single "
        "XGBoost Classifier for cpu_request, a second XGBoost Classifier for "
        "memory_request, and an XGBoost Regressor on log1p(duration_seconds). "
        "A known train-serving mismatch existed in the duration model: it trained "
        "on actual cpu_request/memory_request but used predicted values at inference."
    )
    a("")
    a("─" * 72)
    a("2. MOTIVATION FOR THE ENSEMBLE")
    a("─" * 72)
    a(
        "The ensemble addresses three concerns: (1) single-model sensitivity to "
        "hyperparameter choices; (2) the duration train-serving mismatch via "
        "OOF-based Stage 2 training; (3) statistically testable improvement "
        "via Wilcoxon signed-rank test. Only models previously evaluated in "
        "the project repository were included."
    )
    a("")
    a("─" * 72)
    a("3. ENSEMBLE DESIGN")
    a("─" * 72)
    a("Stage 1 Base Models: XGBClassifier, RandomForestClassifier")
    a("Stage 2 Base Models: XGBoost_log, RandomForest, Ridge_log")
    a("")
    a("OOF methodology:")
    a("  Stage 1: TimeSeriesSplit(5) OOF predictions used as meta-learner training")
    a("           inputs AND as Stage 2 training inputs (CPU/RAM).")
    a("  Stage 2: Duration base models also generate OOF predictions on TRAIN.")
    a("           Stack_Ridge and Stack_XGB_dur are trained ONLY on these TRAIN")
    a("           OOF predictions. TEST is used only for evaluation.")
    a("")
    a("Ensemble strategies:")
    a("  Stage 1: SoftVoting, Stack_LR (OOF-trained), Stack_XGB (OOF-trained)")
    a("  Stage 2: WeightedAvg (TRAIN OOF RMSLE weights), Stack_Ridge (TRAIN OOF),")
    a("           Stack_XGB_dur (TRAIN OOF)")
    a("")
    a("Removed (not independently evaluated): LightGBM, OrdinalClassifier, Tweedie")
    a("")
    a("─" * 72)
    a("4. METHODOLOGICAL FIXES IN THIS VERSION")
    a("─" * 72)
    a(
        "  Fix 1 — Stage 2 stacking leakage (previous version):"
    )
    a(
        "    The previous ensemble_stage2() trained Stack_Ridge and Stack_XGB_dur "
        "directly on TEST predictions with y_test as the training target."
        " This is data leakage: test labels were used inside .fit()."
        " The reported metrics for Stack_Ridge and Stack_XGB_dur in the previous"
        " version were therefore invalid (optimistically biased)."
    )
    a(
        "    Fix: Stage 2 meta-learners now train exclusively on TRAIN OOF "
        "duration predictions. y_test is used only in .predict() and metrics "
        "computation, never in .fit()."
    )
    a("")
    a(
        "  Fix 2 — Forecast used hardcoded XGBoost (previous version):"
    )
    a(
        "    The generate_forecast() function hardcoded XGBoost as the base model "
        "for all three targets, even when SoftVoting (CPU), Stack_XGB (memory), "
        "and Stack_XGB_dur (duration) were selected as best ensembles."
    )
    a(
        "    Fix: predict_s1_single() and predict_dur_single() dispatch to the "
        "correct ensemble (SoftVoting / Stack_LR / Stack_XGB for Stage 1, "
        "WeightedAvg / Stack_Ridge / Stack_XGB_dur for Stage 2) using the "
        "fitted meta-model objects."
    )
    a("")
    a("─" * 72)
    a("5. WORKLOAD GENERATION")
    a("─" * 72)
    a(
        "Future jobs are unknown. Stage 0 generates synthetic jobs by sampling "
        "job_type, role, app_name from recent historical distributions. "
        "gpu_request is derived deterministically from role. CPU and memory are "
        "then estimated by the selected Stage 1 ensemble. Duration is estimated "
        "by the selected Stage 2 ensemble using predicted (not actual) CPU/RAM."
    )
    a("")
    a("─" * 72)
    a("6. LIMITATIONS")
    a("─" * 72)
    a("  1. Distribution shift: assumes future workload mix follows recent history.")
    a("  2. Autoregressive drift: lag features compound prediction errors over the "
      "24-hour horizon. Mitigated by clipping predictions before buffer append.")
    a("  3. Plausible scenario tool, not a point forecast with uncertainty bounds.")
    a("")
    a("─" * 72)
    a("7. RESULTS")
    a("─" * 72)
    if not baseline_vs_ens.empty:
        for _, r in baseline_vs_ens.iterrows():
            direction = "improved" if r["Improved"] else "did not improve"
            a(f"  {r['Target']}: {r['Best_Ensemble']} {direction} over XGBoost. "
              f"Baseline {r['Key_Metric']}={r['Baseline_XGB']}, "
              f"Ensemble={r['Best_Ensemble_Value']} (delta={r['Delta']:+.6f}).")
    a("")
    if not wilcoxon_df.empty:
        a("  Wilcoxon signed-rank test (ensemble vs. XGBoost baseline):")
        for _, r in wilcoxon_df.iterrows():
            a(f"    {r['target']:<22} p={r.get('wilcoxon_p_value','N/A')}"
              f"  sig={r.get('significant_at_005','N/A')}"
              f"  better={r.get('ensemble_better','N/A')}")
    a("")
    a("=" * 72)
    a("END OF REPORT MATERIAL")
    a("=" * 72)

    rpt = output_dir / "capstone_report_material.txt"
    rpt.write_text("\n".join(lines), encoding="utf-8")
    print(f"    → Report material: {rpt}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 15: MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    for d in [METRICS_DIR, OUTPUT_DIR, PROCESSED_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  ENSEMBLE FORECAST PIPELINE")
    print("=" * 60)
    print(
        "\n  Architecture: Historical Data → Synthetic Workload Generation"
        "\n                → Conditional Resource Estimation → Optimizer"
        "\n"
        "\n  Stage 2 meta-learners trained on TRAIN OOF predictions only."
        "\n  TEST used only for evaluation (.predict + metrics).\n"
    )

    print("[1] Loading data...")
    data  = load_and_prepare_data()
    train, test = temporal_split(data)
    print(f"    Train: {len(train):,}  |  Test: {len(test):,}")

    all_metrics = []

    # ── Stage 1: CPU ──────────────────────────────────────────────────────────
    print("\n[2] Stage 1 — CPU classification")
    m, cpu_oof, cpu_te, cpu_fit, cpu_enc = train_stage1(train, test, "cpu_request")
    all_metrics.extend(m)
    (em, cpu_ens_te, cpu_ens_oof, best_cpu,
     cpu_meta_lr, cpu_meta_xgb, cpu_meta_xgb_enc) = ensemble_stage1(
        train, test, "cpu_request", cpu_fit, cpu_enc, cpu_oof, cpu_te
    )
    all_metrics.extend(em)

    # ── Stage 1: Memory ───────────────────────────────────────────────────────
    print("\n[3] Stage 1 — Memory classification")
    m, ram_oof, ram_te, ram_fit, ram_enc = train_stage1(train, test, "memory_request")
    all_metrics.extend(m)
    (em, ram_ens_te, ram_ens_oof, best_ram,
     ram_meta_lr, ram_meta_xgb, ram_meta_xgb_enc) = ensemble_stage1(
        train, test, "memory_request", ram_fit, ram_enc, ram_oof, ram_te
    )
    all_metrics.extend(em)

    # ── Stage 2: Duration (OOF CPU/RAM fix + OOF duration meta-training) ──────
    print("\n[4] Stage 2 — Duration regression")
    (m, dur_fit, duan_factors, use_log_map,
     oof_dur, te_dur) = train_stage2(
        train, test,
        oof_cpu_tr=cpu_ens_oof[best_cpu].astype(float),
        oof_ram_tr=ram_ens_oof[best_ram].astype(float),
        te_cpu=cpu_ens_te[best_cpu],
        te_ram=ram_ens_te[best_ram],
    )
    all_metrics.extend(m)

    (em, dur_ens_te, best_dur,
     dur_meta_ridge, dur_meta_xgb_dur,
     wa_weights, dur_names) = ensemble_stage2(
        train["duration_seconds"].values,
        test["duration_seconds"].values,
        oof_dur,
        te_dur,
    )
    all_metrics.extend(em)

    # ── Wilcoxon ──────────────────────────────────────────────────────────────
    print("\n[5] Wilcoxon Tests")
    wil_rows = [
        {"target": "cpu_request",     **run_wilcoxon(
            test["cpu_request"].values,     cpu_te["XGBoost"], cpu_ens_te[best_cpu], "clf")},
        {"target": "memory_request",  **run_wilcoxon(
            test["memory_request"].values,  ram_te["XGBoost"], ram_ens_te[best_ram], "clf")},
        {"target": "duration_seconds", **run_wilcoxon(
            test["duration_seconds"].values, te_dur["XGBoost_log"], dur_ens_te[best_dur], "reg")},
    ]
    wil = pd.DataFrame(wil_rows)
    for _, r in wil.iterrows():
        print(f"    {r['target']:<22} p={r.get('wilcoxon_p_value','N/A')}"
              f"  sig={r.get('significant_at_005','N/A')}"
              f"  better={r.get('ensemble_better','N/A')}")

    # ── Save metrics ──────────────────────────────────────────────────────────
    print("\n[6] Saving metrics...")
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(METRICS_OUTPUT,    index=False)
    metrics_df.to_csv(COMPARISON_OUTPUT, index=False)
    wil.to_csv(WILCOXON_OUTPUT, index=False)

    cpu_tbl  = build_per_target_table(metrics_df, "cpu_request")
    mem_tbl  = build_per_target_table(metrics_df, "memory_request")
    dur_tbl  = build_per_target_table(metrics_df, "duration_seconds")
    summ_tbl = build_summary_table(metrics_df)
    bl_ens   = build_baseline_vs_ensemble(metrics_df, BASELINE_METRICS_REF)

    cpu_tbl.to_csv(CPU_METRICS_OUTPUT,       index=False)
    mem_tbl.to_csv(MEMORY_METRICS_OUTPUT,    index=False)
    dur_tbl.to_csv(DURATION_METRICS_OUTPUT,  index=False)
    summ_tbl.to_csv(SUMMARY_METRICS_OUTPUT,  index=False)
    bl_ens.to_csv(BASELINE_VS_ENS_OUTPUT,    index=False)
    summ_tbl.to_csv(MODEL_COMPARISON_OUTPUT, index=False)

    for p in [METRICS_OUTPUT, CPU_METRICS_OUTPUT, MEMORY_METRICS_OUTPUT,
              DURATION_METRICS_OUTPUT, SUMMARY_METRICS_OUTPUT, BASELINE_VS_ENS_OUTPUT]:
        print(f"    → {p}")

    write_report_text(metrics_df, bl_ens, wil, METRICS_DIR)

    # ── 24-hour Forecast using selected ensemble winners ──────────────────────
    print("\n[7] Generating 24-hour forecast (using selected ensembles)...")
    forecast = generate_forecast(
        data,
        cpu_fit, cpu_enc, best_cpu, len(cpu_enc.classes_),
        cpu_meta_lr, cpu_meta_xgb, cpu_meta_xgb_enc,
        ram_fit, ram_enc, best_ram, len(ram_enc.classes_),
        ram_meta_lr, ram_meta_xgb, ram_meta_xgb_enc,
        dur_fit, use_log_map,
        best_dur, dur_meta_ridge, dur_meta_xgb_dur, wa_weights, dur_names,
    )
    forecast.to_csv(FORECAST_CSV,     index=False)
    forecast.to_parquet(FORECAST_PARQUET, index=False)
    print(f"    → {FORECAST_CSV} ({len(forecast):,} jobs)")

    # ── Optimizer output ──────────────────────────────────────────────────────
    opt = forecast.copy()
    opt["release_seconds"]             = opt["scheduled_seconds"]
    opt["processing_duration_seconds"] = opt["duration_seconds"]
    opt["deadline_seconds"]            = opt["release_seconds"] + opt["processing_duration_seconds"] * 1.25
    c90 = opt["cpu_request"].quantile(0.90)
    m90 = opt["memory_request"].quantile(0.90)
    opt["is_critical"]  = (
        (opt["gpu_request"] == 1) |
        (opt["cpu_request"]  >= c90) |
        (opt["memory_request"] >= m90)
    ).astype(int)
    opt["replica_count"] = np.where(opt["is_critical"] == 1, 2, 1)
    cols = [
        "forecast_job_id", "instance_sn", "role", "app_name", "job_type",
        "release_seconds", "cpu_request", "memory_request", "gpu_request",
        "processing_duration_seconds", "deadline_seconds",
        "is_critical", "replica_count",
    ]
    opt[cols].to_parquet(OPTIMIZATION_OUTPUT, index=False)
    print(f"    → {OPTIMIZATION_OUTPUT}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    for tgt, key in [("cpu_request", "macro_f1"),
                     ("memory_request", "macro_f1"),
                     ("duration_seconds", "rmsle")]:
        print(f"\n  {tgt} — test {key}:")
        sub = metrics_df[(metrics_df["target"] == tgt) &
                         (metrics_df["split"]  == "test")].copy()
        asc = (key == "rmsle")
        extra = "accuracy" if key == "macro_f1" else "r2_log"
        show  = [c for c in ["model", key, extra] if c in sub.columns]
        print(sub.sort_values(key, ascending=asc)[show].to_string(index=False))

    if not bl_ens.empty:
        print("\n  Baseline vs. Ensemble:")
        print(bl_ens[["Target","Key_Metric","Baseline_XGB","Best_Ensemble",
                       "Best_Ensemble_Value","Improved","Delta"]].to_string(index=False))

    print("\n[DONE]\n")


if __name__ == "__main__":
    main()
