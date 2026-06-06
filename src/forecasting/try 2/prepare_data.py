"""
prepare_data.py
===============
Reads cleaned_data.parquet and builds the training dataset in SECONDS
with lag features and rolling features for the forecasting pipeline.

Output: data/processed/forecast_dataset_seconds.parquet

Run from src/forecasting/:
    python prepare_data.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import numpy as np
import pandas as pd
from config_seconds import (
    INPUT_DATA, PREPARED_DATA, OUTPUTS_DIR, REQUIRED_COLUMNS,
    DESCRIPTOR_TARGETS, INTERARRIVAL_BINS, INTERARRIVAL_LABELS,
)

# ── Columns excluded from features (current targets and metadata) ─────────────
DROP_FROM_FEATURES = [
    "instance_sn", "creation_time", "deletion_time", "gpu_request",
    "scheduled_time", "scheduled_seconds",
    "role", "app_name", "job_type",
    "interarrival_seconds", "interarrival_bucket",
    "cpu_request", "memory_request", "duration_seconds",
    "interarrival_seconds_log", "cpu_request_log", "memory_request_log", "duration_seconds_log",
    "day_index", "day_of_week", "is_weekend", "day_sin", "day_cos",
]
CURRENT_DESCRIPTORS = ["role_encoded", "app_name_encoded", "job_type_encoded"]

LAG_COLS = [
    "interarrival_seconds", "cpu_request", "memory_request", "duration_seconds",
    "job_type_encoded",
    "interarrival_seconds_log", "cpu_request_log", "memory_request_log", "duration_seconds_log",
    "role_encoded", "app_name_encoded",
]
LAG_STEPS    = [1, 2, 3, 5, 10, 20, 50]
ROLLING_COLS = ["interarrival_seconds", "cpu_request", "memory_request", "duration_seconds"]
ROLLING_WINS = [5, 10, 20, 50]


def validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")


def cyclic(series: pd.Series, period: float):
    return np.sin(2 * np.pi * series / period), np.cos(2 * np.pi * series / period)


def prepare_data() -> None:
    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Reading: {INPUT_DATA}")
    df = pd.read_parquet(INPUT_DATA)

    # Drop extra columns (Unnamed, embedded statistics)
    real_cols = [c for c in df.columns if not c.startswith("Unnamed") and c not in ["12374", "100"]]
    df = df[real_cols].copy()

    validate(df)
    print(f"Original shape: {df.shape}")

    # ── Times already stored as timedelta64[ns] in the parquet ───────────────
    # pd.to_timedelta() not needed; just ensure type in case of edge cases
    for col in ["creation_time", "scheduled_time", "deletion_time"]:
        if col in df.columns and not pd.api.types.is_timedelta64_dtype(df[col]):
            df[col] = pd.to_timedelta(df[col])

    # Sort by scheduled_time
    df = df.sort_values("scheduled_time").reset_index(drop=True)

    # Base time in SECONDS from dataset start
    df["scheduled_seconds"] = df["scheduled_time"].dt.total_seconds()

    # duration_seconds already exists in the parquet — use it directly
    if "duration_seconds" not in df.columns:
        df["duration_seconds"] = pd.to_numeric(df["duration_minutes"], errors="coerce").fillna(0).clip(lower=0) * 60
    else:
        df["duration_seconds"] = pd.to_numeric(df["duration_seconds"], errors="coerce").fillna(0).clip(lower=0)

    for col in ["cpu_request", "memory_request"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

    # ── Interarrival in seconds + bucket ─────────────────────────────────────
    df["interarrival_seconds"] = df["scheduled_time"].diff().dt.total_seconds().fillna(0).clip(lower=0)

    # Operational bucket: classifies the arrival REGIME instead of predicting
    # the exact second. 34.9% of jobs are simultaneous (interarrival=0),
    # making precise regression impossible.
    df["interarrival_bucket"] = pd.cut(
        df["interarrival_seconds"],
        bins=INTERARRIVAL_BINS,
        labels=INTERARRIVAL_LABELS,
    ).astype(str)

    # ── Descriptor encodings ─────────────────────────────────────────────────
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    mappings = {}
    for col in ["role", "app_name", "job_type"]:
        df[col] = df[col].astype(str).str.lower().str.strip()
        df[f"{col}_encoded"] = df[col].astype("category").cat.codes
        mapping = (
            df[[f"{col}_encoded", col]].drop_duplicates()
            .sort_values(f"{col}_encoded").reset_index(drop=True)
        )
        mapping.to_csv(OUTPUTS_DIR / f"{col}_mapping.csv", index=False)
        mappings[col] = {int(r[f"{col}_encoded"]): r[col] for _, r in mapping.iterrows()}

    with open(OUTPUTS_DIR / "descriptor_mappings.json", "w", encoding="utf-8") as f:
        json.dump(mappings, f, indent=4)

    # ── Time features ─────────────────────────────────────────────────────────
    # Only intra-day position features (time_of_day_s, hour, minute, second)
    # and cyclic encodings. The dataset is a timedelta from t=0 with no real
    # calendar date, so day_of_week and is_weekend are excluded — we cannot
    # verify which actual weekday corresponds to day 0 of the dataset.
    df["arrival_order"] = np.arange(1, len(df) + 1)
    df["time_of_day_s"] = df["scheduled_seconds"] % 86_400
    df["hour"]          = (df["time_of_day_s"] // 3600).astype(int)
    df["minute"]        = ((df["time_of_day_s"] % 3600) // 60).astype(int)
    df["second"]        = (df["time_of_day_s"] % 60).astype(int)

    # Sine/cosine of the daily cycle: give the model continuity between 23:59 and 00:00
    df["time_sin"], df["time_cos"] = cyclic(df["time_of_day_s"], 86_400)

    # ── Log transforms ────────────────────────────────────────────────────────
    for col in ["interarrival_seconds", "cpu_request", "memory_request", "duration_seconds"]:
        df[f"{col}_log"] = np.log1p(df[col])

    # ── Lag features ──────────────────────────────────────────────────────────
    lag_blocks = []
    for lag in LAG_STEPS:
        lag_blocks.append(df[LAG_COLS].shift(lag).add_suffix(f"_lag_{lag}"))

    # ── Rolling features ──────────────────────────────────────────────────────
    rolling_blocks = []
    for col in ROLLING_COLS:
        shifted = df[col].shift(1)
        for w in ROLLING_WINS:
            rolling_blocks.append(pd.DataFrame({
                f"{col}_rolling_mean_{w}":   shifted.rolling(w).mean(),
                f"{col}_rolling_std_{w}":    shifted.rolling(w).std(),
                f"{col}_rolling_median_{w}": shifted.rolling(w).median(),
                f"{col}_rolling_min_{w}":    shifted.rolling(w).min(),
                f"{col}_rolling_max_{w}":    shifted.rolling(w).max(),
            }))

    # ── Combine all blocks ────────────────────────────────────────────────────
    model_df = (
        pd.concat([df] + lag_blocks + rolling_blocks, axis=1)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
        .copy()
    )

    print(f"Final shape (after dropna): {model_df.shape}")
    print(f"Columns: {len(model_df.columns)}")

    # ── Save ──────────────────────────────────────────────────────────────────
    PREPARED_DATA.parent.mkdir(parents=True, exist_ok=True)
    model_df.to_parquet(PREPARED_DATA, index=False)
    print(f"\nDataset saved to: {PREPARED_DATA}")

    print("\n── Target statistics ──")
    for col in ["interarrival_seconds", "duration_seconds", "cpu_request", "memory_request"]:
        s = model_df[col]
        print(f"  {col}: mean={s.mean():.1f}  median={s.median():.1f}  std={s.std():.1f}  max={s.max():.1f}")


if __name__ == "__main__":
    prepare_data()
