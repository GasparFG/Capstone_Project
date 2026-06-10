import pandas as pd

from config import INPUT_DATA, PROCESSED_DATA, FORECAST_FREQ


def prepare_forecast_data():
    workload_data = pd.read_csv(INPUT_DATA)

    required_columns = [
        "instance_sn",
        "cpu_request",
        "memory_request",
        "duration_minutes",
        "scheduled_time",
        "job_type"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in workload_data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    workload_data["scheduled_time"] = pd.to_timedelta(
        workload_data["scheduled_time"]
    )

    base_date = pd.Timestamp("2025-01-01")

    workload_data["scheduled_datetime"] = (
        base_date + workload_data["scheduled_time"]
    )

    workload_data["job_type"] = (
        workload_data["job_type"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    workload_data["is_batch"] = (
        workload_data["job_type"] == "batch"
    ).astype(int)

    workload_data["is_interactive"] = (
        workload_data["job_type"] == "interactive"
    ).astype(int)

    workload_data = workload_data.set_index("scheduled_datetime")

    forecast_data = workload_data.resample(FORECAST_FREQ).agg(
        cpu_request_sum=("cpu_request", "sum"),
        memory_request_sum=("memory_request", "sum"),
        duration_minutes_mean=("duration_minutes", "mean"),
        job_count=("instance_sn", "count"),
        batch_count=("is_batch", "sum"),
        interactive_count=("is_interactive", "sum")
    ).reset_index()

    forecast_data = forecast_data.rename(
        columns={"scheduled_datetime": "time_slot"}
    )

    forecast_data = forecast_data.fillna(0)

    forecast_data["hour"] = forecast_data["time_slot"].dt.hour
    forecast_data["minute"] = forecast_data["time_slot"].dt.minute
    forecast_data["day_of_week"] = forecast_data["time_slot"].dt.dayofweek
    forecast_data["is_weekend"] = (
        forecast_data["day_of_week"] >= 5
    ).astype(int)

    forecast_data["slot_of_day"] = (
        forecast_data["hour"] * 60 + forecast_data["minute"]
    )

    PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
    forecast_data.to_csv(PROCESSED_DATA, index=False)

    print(f"Input data used: {INPUT_DATA}")
    print(f"Forecast frequency: {FORECAST_FREQ}")
    print(f"Forecast-ready dataset saved to: {PROCESSED_DATA}")
    print(f"Shape: {forecast_data.shape}")
    print(forecast_data.head())


if __name__ == "__main__":
    prepare_forecast_data()