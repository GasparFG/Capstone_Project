from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parents[2]

# ============================================================
# INPUT DATA
# ============================================================

INPUT_FILE_NAME = "cleaned_data_extended_90_days.csv"

INPUT_DATA = BASE_DIR / "data" / "interim" / INPUT_FILE_NAME
PROCESSED_DATA = BASE_DIR / "data" / "processed" / "forecast_ready.csv"

PREDICTIONS_DIR = BASE_DIR / "outputs" / "predictions"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"

OPTIMIZATION_INPUT_DATA = (
    BASE_DIR / "data" / "processed" / "optimization_input_dataset.parquet"
)


# ============================================================
# FORECAST WINDOW
# ============================================================


FORECAST_FREQ = os.getenv("FORECAST_FREQ", "5min")


def get_slots_per_day(forecast_freq: str) -> int:
    frequency_map = {
        "5min": 288,
        "15min": 96,
        "30min": 48,
        "1h": 24,
        "1H": 24,
        "60min": 24
    }

    if forecast_freq not in frequency_map:
        raise ValueError(
            f"Unsupported FORECAST_FREQ: {forecast_freq}. "
            "Use one of: '5min', '15min', '30min', '1h'."
        )

    return frequency_map[forecast_freq]


SLOTS_PER_DAY = get_slots_per_day(FORECAST_FREQ)

# Last full day as test horizon
TEST_HORIZON = SLOTS_PER_DAY

SEQUENCE_LENGTH = SLOTS_PER_DAY
SEASONAL_PERIOD = SLOTS_PER_DAY

DATE_COLUMN = "time_slot"


# ============================================================
# WINDOW-LEVEL TARGETS
# ============================================================

TARGET_COLUMNS = [
    "cpu_request_sum",
    "memory_request_sum",
    "duration_minutes_mean",
    "job_count",
    "batch_count",
    "interactive_count"
]