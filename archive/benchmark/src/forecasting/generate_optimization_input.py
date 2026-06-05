import json
import numpy as np
import pandas as pd
from config import MODELS_DIR, OUTPUTS_DIR, OPTIMIZATION_INPUT_DATA, FORECAST_HORIZON_MINUTES, MAX_GENERATED_JOBS, NUMERIC_TARGETS, RANDOM_STATE, DESCRIPTOR_COLUMNS

PROFILE_LEVELS = [
    ["role", "app_name", "job_type"],
    ["app_name", "job_type"],
    ["role", "job_type"],
    ["app_name"],
    ["job_type"],
]


def load_profile_tables(target):
    target_dir = MODELS_DIR / f"profile_{target}"
    tables = []
    for index, keys in enumerate(PROFILE_LEVELS):
        path = target_dir / f"level_{index + 1}_{'_'.join(keys)}.parquet"
        if path.exists():
            tables.append({"keys": keys, "table": pd.read_parquet(path)})
    with open(MODELS_DIR / "job_level_profile_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    fallback = metadata[target]["fallback_median"]
    return tables, fallback


def profile_predict_one(row, tables, fallback):
    for level in tables:
        keys = level["keys"]
        table = level["table"]
        mask = np.ones(len(table), dtype=bool)
        for key in keys:
            mask &= table[key].astype(str).values == str(row[key])
        matched = table.loc[mask]
        if len(matched) > 0:
            return float(matched["predicted_value"].iloc[0])
    return float(fallback)


def generate_optimization_input():
    rng = np.random.default_rng(RANDOM_STATE)
    descriptor_distribution_path = MODELS_DIR / "descriptor_distribution.csv"
    if not descriptor_distribution_path.exists():
        raise FileNotFoundError("Missing descriptor distribution. Run train_job_level_capacity_model.py first.")
    descriptor_distribution = pd.read_csv(descriptor_distribution_path)

    profile_models = {target: load_profile_tables(target) for target in NUMERIC_TARGETS}

    offset = 0.0
    rows = []
    print("Generating next-day job-level forecast...")
    while offset < FORECAST_HORIZON_MINUTES and len(rows) < MAX_GENERATED_JOBS:
        sampled = descriptor_distribution.sample(n=1, weights="probability", random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        descriptor_row = {col: sampled[col] for col in DESCRIPTOR_COLUMNS}
        interarrival = profile_predict_one(descriptor_row, *profile_models["interarrival_minutes"])
        interarrival = max(interarrival, 0.01)
        offset += interarrival
        if offset > FORECAST_HORIZON_MINUTES:
            break
        cpu = profile_predict_one(descriptor_row, *profile_models["cpu_request"])
        memory = profile_predict_one(descriptor_row, *profile_models["memory_request"])
        duration = profile_predict_one(descriptor_row, *profile_models["duration_minutes"])
        rows.append({
            "predicted_job_id": f"pred_{len(rows) + 1:06d}",
            "arrival_order": len(rows) + 1,
            "arrival_offset_minutes": offset,
            "required_cpu": max(cpu, 0),
            "required_memory": max(memory, 0),
            "expected_duration_minutes": max(duration, 0),
            "role": descriptor_row["role"],
            "app_name": descriptor_row["app_name"],
            "job_type": descriptor_row["job_type"],
            "forecasting_approach": "job_level_historical_profile_forecasting",
        })

    output = pd.DataFrame(rows)
    OPTIMIZATION_INPUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OPTIMIZATION_INPUT_DATA, index=False)
    output.to_csv(OUTPUTS_DIR / "optimization_input_dataset_preview.csv", index=False)
    print(f"Optimization input saved to: {OPTIMIZATION_INPUT_DATA}")
    print(f"CSV preview saved to: {OUTPUTS_DIR / 'optimization_input_dataset_preview.csv'}")
    print(f"Generated jobs: {len(output)}")
    print(output.head())


if __name__ == "__main__":
    generate_optimization_input()
