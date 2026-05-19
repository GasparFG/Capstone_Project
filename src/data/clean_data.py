import pandas as pd

def clean_data(input_path, output_path):
    """
    Cleans workload dataset.
    """

    # Load dataset
    df = pd.read_parquet(input_path)

    # Convert timestamps
    timestamp_columns = ["start_timestamp","end_timestamp"]
    for col in timestamp_columns:
        df[col] = pd.to_timedelta(df[col], unit="s")

    # Coalesce resource columns into unified usage signals
    df["cpu_usage"] = df["plan_cpu"].combine_first(df["real_cpu_avg"]).fillna(0)
    df["mem_usage"] = df["plan_mem"].combine_first(df["real_mem_avg"]).fillna(0)

    # Drop unnecesary columns
    cols_to_drop = ["job_id",           # encoded in uid
                    "task_id",          # encoded in uid
                    "machine_id",       # not necessary
                    "plan_cpu",         # merged into cpu_usage
                    "plan_mem",         # merged into mem_usage
                    "real_cpu_avg",     # merged into cpu_usage
                    "real_mem_avg",     # merged into mem_usage
                    "real_cpu_max",     # keeping avg only
                    "real_mem_max",     # keeping avg only
                    "capacity_cpu",     # not necessary
                    "capacity_memory",  # not necessary
                    ]

    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove negative resource values
    for col in ["cpu_usage", "mem_usage"]:
        df = df[df[col] >= 0]

    # Remove invalid or missing start timestamps
    df = df[df["start_timestamp"].notna()]

    # Remove negative durations (end before start)
    df = df[(df["end_timestamp"] >= df["start_timestamp"]) | (df["end_timestamp"].isna())]

    # Create duration feature
    df["duration_minutes"] = ((df["end_timestamp"] - df["start_timestamp"]).dt.total_seconds() / 60)

    # Final column order
    df = df[["uid", "job_type", "start_timestamp", "end_timestamp",
             "duration_minutes", "cpu_usage", "mem_usage"]]

    #Save clean dataset
    df.to_parquet(output_path, index=False)

    # Preview results
    print(f"Cleaned dataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"\n=== dtypes ===\n{df.dtypes}")
    print(f"\n=== null counts ===\n{df.isnull().sum()}")
    print(f"\n=== job_type counts ===\n{df['job_type'].value_counts()}")
    print(f"\n=== sample (5 batch, 5 interactive) ===")
    sample = pd.concat([
        df[df["job_type"] == "batch"].head(5),
        df[df["job_type"] == "interactive"].head(5)
    ])
    print(sample.to_string())

if __name__ == "__main__":
    clean_data(input_path="data/interim/df_jobs.parquet", output_path="data/interim/cleaned_data.parquet")