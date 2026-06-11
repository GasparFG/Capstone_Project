"""
config_seconds.py
=================
Configuration for the forecasting pipeline in SECONDS.
Uses cleaned_data.parquet as input (12,374 real jobs).
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]   # Capstone_Project/

# ── Input ─────────────────────────────────────────────────────────────────────
INPUT_DATA = BASE_DIR / "data" / "interim" / "cleaned_data.parquet"

# ── Processed ─────────────────────────────────────────────────────────────────
PREPARED_DATA   = BASE_DIR / "data" / "processed" / "forecast_dataset_seconds.parquet"
FORECAST_OUTPUT = BASE_DIR / "data" / "processed" / "optimization_input_seconds.parquet"

# ── Outputs ───────────────────────────────────────────────────────────────────
OUTPUTS_DIR = BASE_DIR / "outputs" / "forecast_seconds"
MODELS_DIR  = BASE_DIR / "models" / "forecast_seconds"

# ── Required columns in cleaned_data ─────────────────────────────────────────
REQUIRED_COLUMNS = [
    "instance_sn",
    "role",
    "app_name",
    "cpu_request",
    "memory_request",
    "scheduled_time",
    "duration_minutes",
    "job_type",
]

# ── Model targets ─────────────────────────────────────────────────────────────
# Only truly continuous target → XGBoost regression + log1p
NUMERIC_TARGETS  = ["duration_seconds"]

# Discrete values → XGBoost classification
# cpu_request:          9 unique values  (2, 8, 12, 16, 48, 64, 96, 192...)
# memory_request:      24 unique values  (discrete MB values in the system)
# interarrival_bucket: 5 operational buckets (0s / 1-30s / 30-300s / 300-3600s / >3600s)
#   34.9% of jobs arrive simultaneously (interarrival=0). Regression cannot predict
#   exact 0s, destroying R² and MAPE. Classifying the arrival REGIME is more
#   useful for the optimizer than predicting the exact second.
DISCRETE_TARGETS = ["cpu_request", "memory_request", "interarrival_bucket"]

# Job descriptors → classification
DESCRIPTOR_TARGETS = ["role_encoded", "app_name_encoded", "job_type_encoded"]

# Interarrival buckets (seconds) and their representative value for the forecast
INTERARRIVAL_BINS    = [-1, 0, 30, 300, 3600, float("inf")]
INTERARRIVAL_LABELS  = ["0s", "1-30s", "30-300s", "300-3600s", ">3600s"]
INTERARRIVAL_MEDIANS = {       # empirical median of each bucket (seconds)
    "0s":      0.0,
    "1-30s":   8.0,
    "30-300s": 90.0,
    "300-3600s": 600.0,
    ">3600s":  3600.0,
}

# ── Training parameters ───────────────────────────────────────────────────────
TEST_SIZE          = 0.20
RANDOM_STATE       = 42

# ── Forecast horizon ──────────────────────────────────────────────────────────
FORECAST_HORIZON_SECONDS = 86_400   # 1 full day
MAX_GENERATED_JOBS       = 10_000   # safety limit
