import json
import numpy as np
import pandas as pd
from config import INPUT_DATA, JOB_LEVEL_DATA, OUTPUTS_DIR, REQUIRED_RAW_COLUMNS, DESCRIPTOR_COLUMNS


def validate_columns(data: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def prepare_job_level_data() -> None:
    data = pd.read_csv(INPUT_DATA)
    validate_columns(data)

    data = data.copy()
    data["scheduled_time"] = pd.to_timedelta(data["scheduled_time"])
    data = data.sort_values("scheduled_time").reset_index(drop=True)

    for col in DESCRIPTOR_COLUMNS:
        data[col] = data[col].astype(str).str.lower().str.strip()
        data[f"{col}_encoded"] = data[col].astype("category").cat.codes

    for col in ["cpu_request", "memory_request", "duration_minutes"]:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0).clip(lower=0)

    data["arrival_order"] = np.arange(1, len(data) + 1)
    data["scheduled_minutes_from_start"] = data["scheduled_time"].dt.total_seconds() / 60
    data["arrival_offset_in_day"] = data["scheduled_minutes_from_start"] % 1440
    data["day_index"] = (data["scheduled_minutes_from_start"] // 1440).astype(int)
    data["hour"] = (data["arrival_offset_in_day"] // 60).astype(int)
    data["minute"] = (data["arrival_offset_in_day"] % 60).astype(int)
    data["day_of_week"] = data["day_index"] % 7
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)

    data["interarrival_minutes"] = (
        data["scheduled_time"].diff().dt.total_seconds().div(60)
    )
    median_interarrival = data["interarrival_minutes"].median()
    if pd.isna(median_interarrival) or median_interarrival <= 0:
        median_interarrival = 1.0
    data["interarrival_minutes"] = data["interarrival_minutes"].fillna(median_interarrival).clip(lower=0.01)

    JOB_LEVEL_DATA.parent.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    data.to_parquet(JOB_LEVEL_DATA, index=False)

    mappings = {}
    for col in DESCRIPTOR_COLUMNS:
        mapping = (
            data[[f"{col}_encoded", col]]
            .drop_duplicates()
            .sort_values(f"{col}_encoded")
            .reset_index(drop=True)
        )
        mapping.to_csv(OUTPUTS_DIR / f"{col}_mapping.csv", index=False)
        mappings[col] = {int(r[f"{col}_encoded"]): r[col] for _, r in mapping.iterrows()}

    with open(OUTPUTS_DIR / "descriptor_mappings.json", "w", encoding="utf-8") as f:
        json.dump(mappings, f, indent=4)

    print(f"Input data used: {INPUT_DATA}")
    print(f"Raw shape: {data.shape}")
    print(f"Job-level forecast dataset saved to: {JOB_LEVEL_DATA}")
    print(f"Model-ready shape: {data.shape}")
    print(data.head())


if __name__ == "__main__":
    prepare_job_level_data()
