"""
generate_forecast.py
====================
Generates a job-level forecast for the next 86,400 seconds (1 day).

Each predicted job includes:
  arrival_offset_seconds    -> arrival time (seconds from last historical job)
  required_cpu              -> required CPUs (real system value, classifier)
  required_memory           -> required memory MB (real system value, classifier)
  expected_duration_seconds -> expected duration (regression)
  role, app_name, job_type  -> job descriptors (classifiers)

Output:
  outputs/forecast_seconds/job_forecast.csv
  data/processed/optimization_input_seconds.parquet

Run from src/forecasting/:
    python generate_forecast.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import joblib
import numpy as np
import pandas as pd

from config_seconds import (
    PREPARED_DATA, FORECAST_OUTPUT, MODELS_DIR, OUTPUTS_DIR,
    NUMERIC_TARGETS, DISCRETE_TARGETS, FORECAST_HORIZON_SECONDS, MAX_GENERATED_JOBS,
    INTERARRIVAL_MEDIANS,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_mapping(name: str) -> dict:
    path = OUTPUTS_DIR / f"{name}_mapping.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {int(r[f"{name}_encoded"]): r[name] for _, r in df.iterrows()}


def pred_reg(model, X: pd.DataFrame) -> float:
    return max(0.0, float(np.expm1(model.predict(X)[0])))


def pred_clf_numeric(bundle, X: pd.DataFrame) -> float:
    """Classifier for discrete numeric targets (cpu, memory).
    inverse_transform returns a string (e.g. '64.0') — converted to float."""
    pred_enc = int(bundle["model"].predict(X)[0])
    val = bundle["label_encoder"].inverse_transform([pred_enc])[0]
    return float(val)


# ── Feature row construction ──────────────────────────────────────────────────

_LAG_COLS = [
    "interarrival_seconds", "cpu_request", "memory_request", "duration_seconds",
    "job_type_encoded",
    "interarrival_seconds_log", "cpu_request_log", "memory_request_log", "duration_seconds_log",
    "role_encoded", "app_name_encoded",
]
_LAG_STEPS    = [1, 2, 3, 5, 10, 20, 50]
_ROLLING_COLS = ["interarrival_seconds", "cpu_request", "memory_request", "duration_seconds"]
_ROLLING_WINS = [5, 10, 20, 50]


def build_feature_row(history: pd.DataFrame, feature_cols: list, abs_seconds: float,
                      extra: dict = None) -> pd.DataFrame:
    """Builds a feature row. Optimized: pre-initializes the dict to 0
    to avoid the per-column existence check on every prediction."""
    # Pre-initialize all keys to 0 — eliminates the 'if c not in X.columns' loop
    row = {c: 0 for c in feature_cols}

    # Daily cycle time features
    tod = abs_seconds % 86_400
    row["time_of_day_s"] = tod
    row["hour"]          = int(tod // 3600)
    row["minute"]        = int((tod % 3600) // 60)
    row["second"]        = int(tod % 60)
    row["time_sin"]      = np.sin(2 * np.pi * tod / 86_400)
    row["time_cos"]      = np.cos(2 * np.pi * tod / 86_400)

    # Lag features — use .values to avoid pandas overhead
    h_vals = {col: history[col].values for col in _LAG_COLS if col in history.columns}
    n = len(history)
    for lag in _LAG_STEPS:
        if n >= lag:
            for col, arr in h_vals.items():
                key = f"{col}_lag_{lag}"
                if key in row:
                    row[key] = arr[-lag]

    # Rolling features — numpy is ~5x faster than pandas .tail().mean()
    for col in _ROLLING_COLS:
        if col not in history.columns:
            continue
        arr = history[col].values
        for w in _ROLLING_WINS:
            recent = arr[-w:] if len(arr) >= w else arr
            if len(recent) == 0:
                continue
            row[f"{col}_rolling_mean_{w}"]   = recent.mean()
            row[f"{col}_rolling_std_{w}"]    = recent.std()    if len(recent) > 1 else 0
            row[f"{col}_rolling_median_{w}"] = np.median(recent)
            row[f"{col}_rolling_min_{w}"]    = recent.min()
            row[f"{col}_rolling_max_{w}"]    = recent.max()

    if extra:
        for k, v in extra.items():
            if k in row:
                row[k] = v

    return pd.DataFrame([row], columns=feature_cols)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def generate_forecast() -> None:
    # ── Load metadata and models ──────────────────────────────────────────────
    meta_path = MODELS_DIR / "model_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {meta_path}.\n"
            "Run train_model.py first."
        )
    meta          = load_json(meta_path)
    past_cols     = meta["past_feature_columns"]
    cap_cols      = meta["capacity_feature_columns"]
    desc_tgts     = meta.get("descriptor_targets", [])
    discrete_tgts = meta.get("discrete_targets", [])

    # Continuous regressor: duration_seconds
    num_models = {}
    for target in NUMERIC_TARGETS:
        p = MODELS_DIR / f"{target}_model.joblib"
        if not p.exists():
            raise FileNotFoundError(f"Missing model: {p}. Run train_model.py.")
        num_models[target] = joblib.load(p)

    def load_clf(target):
        """Loads {"model": xgb, "label_encoder": le} bundle."""
        p = MODELS_DIR / f"{target}_classifier.joblib"
        if not p.exists():
            raise FileNotFoundError(f"Missing classifier: {p}. Run train_model.py.")
        return joblib.load(p)

    def predict_clf(bundle, X):
        """Predicts a descriptor (role/app/job_type) and returns original code as int.
        LabelEncoder was trained on strings (e.g. '3', '42') -> inverse_transform
        returns string -> cast to int."""
        pred_enc = int(bundle["model"].predict(X)[0])
        val = bundle["label_encoder"].inverse_transform([pred_enc])[0]
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return 0

    # Descriptor classifiers (role, app_name, job_type)
    desc_models = {t: load_clf(t) for t in desc_tgts}

    # Discrete resource classifiers (cpu_request, memory_request, interarrival_bucket)
    # inverse_transform returns the real system value directly (e.g. 64, 96, 320...)
    discrete_models = {t: load_clf(t) for t in discrete_tgts}

    # Mappings to decode role/app/job_type from code -> string
    mappings = {name: load_mapping(name) for name in ["role", "app_name", "job_type"]}

    # ── Initial history (last jobs from the full dataset) ─────────────────────
    print(f"Loading history from: {PREPARED_DATA}")
    data = pd.read_parquet(PREPARED_DATA)

    hist_cols = [
        "scheduled_seconds", "interarrival_seconds",
        "cpu_request", "memory_request", "duration_seconds",
        "job_type_encoded", "job_type",
        "interarrival_seconds_log", "cpu_request_log", "memory_request_log", "duration_seconds_log",
        "role_encoded", "app_name_encoded",
    ]
    # Keep only the last 60 rows (max lag=50, max rolling window=50)
    # Trimming here avoids the O(n²) growth from repeated pd.concat
    history = data[[c for c in hist_cols if c in data.columns]].iloc[-60:].copy().reset_index(drop=True)

    # Starting point: immediately after the last historical job
    last_seconds = float(data["scheduled_seconds"].iloc[-1])
    offset       = 0.0
    jobs         = []

    print(f"\nGenerating forecast for {FORECAST_HORIZON_SECONDS:,} seconds (1 day)...")
    print(f"Start: second {last_seconds:,.0f} of the historical dataset\n")

    while offset < FORECAST_HORIZON_SECONDS and len(jobs) < MAX_GENERATED_JOBS:
        abs_s = last_seconds + offset

        # 1. Predict interarrival bucket (classifier) -> convert to seconds
        X_past = build_feature_row(history, past_cols, abs_s)
        bucket = str(predict_clf(discrete_models["interarrival_bucket"], X_past))
        interarrival = INTERARRIVAL_MEDIANS.get(bucket, 24.0)   # fallback: global median
        offset += interarrival
        if offset > FORECAST_HORIZON_SECONDS:
            break

        abs_s  = last_seconds + offset
        X_past = build_feature_row(history, past_cols, abs_s)

        # 2. Predict job descriptors (role, app, job_type)
        generated_desc = {}
        for tgt, bundle in desc_models.items():
            generated_desc[tgt] = int(predict_clf(bundle, X_past))
        # Fallback: historical mode for any missing descriptor
        for tgt in ["role_encoded", "app_name_encoded", "job_type_encoded"]:
            if tgt in cap_cols and tgt not in generated_desc:
                generated_desc[tgt] = int(history[tgt].mode().iloc[0]) if tgt in history.columns else 0

        # 3. Predict resources using capacity features (includes job descriptors)
        X_cap = build_feature_row(history, cap_cols, abs_s, extra=generated_desc)

        # cpu and memory -> discrete classifiers: return real system values
        cpu    = pred_clf_numeric(discrete_models["cpu_request"],    X_cap)
        memory = pred_clf_numeric(discrete_models["memory_request"], X_cap)

        # duration -> regressor: continuous value in seconds
        duration = pred_reg(num_models["duration_seconds"], X_cap)

        # Decode descriptors to strings
        job_type_enc = int(generated_desc.get("job_type_encoded", 0))
        role_enc     = int(generated_desc.get("role_encoded", 0))
        app_enc      = int(generated_desc.get("app_name_encoded", 0))
        job_type     = mappings["job_type"].get(job_type_enc, "unknown")
        role         = mappings["role"].get(role_enc, "unknown")
        app_name     = mappings["app_name"].get(app_enc, "unknown")

        jobs.append({
            "job_id":                    f"pred_{len(jobs) + 1:06d}",
            "arrival_offset_seconds":    round(offset, 2),
            "job_type":                  job_type,   # batch / interactive
            "required_cpu":              cpu,        # real system value (e.g. 64)
            "required_memory":           memory,     # real system value in MB (e.g. 320.0)
            "expected_duration_seconds": round(duration, 2),
            "role":                      role,
            "app_name":                  app_name,
        })

        # Update history with the generated job
        new_row = {
            "scheduled_seconds":        abs_s,
            "interarrival_seconds":     interarrival,
            "cpu_request":              cpu,
            "memory_request":           memory,
            "duration_seconds":         duration,
            "job_type_encoded":         job_type_enc,
            "job_type":                 job_type,
            "interarrival_seconds_log": np.log1p(interarrival),
            "cpu_request_log":          np.log1p(cpu),
            "memory_request_log":       np.log1p(memory),
            "duration_seconds_log":     np.log1p(duration),
            "role_encoded":             role_enc,
            "app_name_encoded":         app_enc,
        }
        history = pd.concat([history.iloc[-60:], pd.DataFrame([new_row])], ignore_index=True)

    # ── Job-level forecast ────────────────────────────────────────────────────
    job_forecast = pd.DataFrame(jobs)

    # ── Save ──────────────────────────────────────────────────────────────────
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FORECAST_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    job_forecast.to_parquet(FORECAST_OUTPUT, index=False)
    job_forecast.to_csv(OUTPUTS_DIR / "job_forecast.csv", index=False)

    # ── Console summary ───────────────────────────────────────────────────────
    n_batch       = (job_forecast["job_type"] == "batch").sum()
    n_interactive = (job_forecast["job_type"] == "interactive").sum()

    print("══════════════════════════════════════════════════════════")
    print("FORECAST COMPLETE — JOB LEVEL")
    print("══════════════════════════════════════════════════════════")
    print(f"  Jobs generated:    {len(job_forecast):,}  "
          f"(batch={n_batch}, interactive={n_interactive})")
    print(f"  Horizon covered:   {offset:,.0f} s  ({offset/3600:.1f} h)")
    print(f"  CPU unique values: {sorted(job_forecast['required_cpu'].unique())}")
    print(f"  Mem unique values: {sorted(job_forecast['required_memory'].unique())[:8]}...")
    print()
    print(job_forecast.head(5).to_string(index=False))
    print()
    print(f"  Saved to: {OUTPUTS_DIR / 'job_forecast.csv'}")
    print("══════════════════════════════════════════════════════════")


if __name__ == "__main__":
    generate_forecast()
