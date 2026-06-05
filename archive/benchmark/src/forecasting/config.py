from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_DATA = BASE_DIR / "data" / "interim" / "cleaned_data_extended_90_days.csv"
JOB_LEVEL_DATA = BASE_DIR / "data" / "processed" / "job_level_forecast_dataset.parquet"
OPTIMIZATION_INPUT_DATA = BASE_DIR / "data" / "processed" / "optimization_input_dataset.parquet"

OUTPUTS_DIR = BASE_DIR / "outputs" / "final_job_level"
MODELS_DIR = BASE_DIR / "models" / "forecasting"

TEST_SIZE = 0.20
RANDOM_STATE = 42
FORECAST_HORIZON_MINUTES = 1440
MAX_GENERATED_JOBS = 5000

REQUIRED_RAW_COLUMNS = [
    "instance_sn",
    "role",
    "app_name",
    "cpu_request",
    "memory_request",
    "scheduled_time",
    "duration_minutes",
    "job_type",
]

DESCRIPTOR_COLUMNS = ["role", "app_name", "job_type"]
NUMERIC_TARGETS = ["interarrival_minutes", "cpu_request", "memory_request", "duration_minutes"]
