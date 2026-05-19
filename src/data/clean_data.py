import pandas as pd
import numpy as np

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

    # Compute resource usage per job type

    #CPU
    df["cpu_usage"] = np.where(df["job_type"] == "batch",
                                # batch (cpu_usage = plan_cpu)
                                df["plan_cpu"],
                                # interactive (cpu_usage = (real_cpu_avg * plan_cpu) / 100)
                                (df["real_cpu_avg"] * df["plan_cpu"]) / 100  
                                )
    
    #Memory
    df["mem_usage"] = np.where(df["job_type"] == "batch",
                                # batch (mem_usage = plan_mem * capacity_memory)
                                df["plan_mem"] * df["capacity_memory"],
                                # interactive (mem_usage = (real_mem_avg * plan_mem) / 100 * capacity_memory)
                                ((df["real_mem_avg"] * df["plan_mem"]) / 100) * df["capacity_memory"]  
                                )

    # Drop rows where usage could not be computed
    df = df.dropna(subset=["cpu_usage", "mem_usage"])

    # Drop unnecesary columns
    cols_to_drop = ["job_id",           # encoded in uid
                    "task_id",          # encoded in uid
                    "machine_id",       # not necessary
                    "plan_cpu",         # used in cpu_usage formula
                    "plan_mem",         # used in mem_usage formula
                    "real_cpu_avg",     # used in cpu_usage formula
                    "real_mem_avg",     # used in mem_usage formula
                    "real_cpu_max",     # not necessary
                    "real_mem_max",     # not necessary
                    "capacity_cpu",     # not necessary
                    "capacity_memory",  # used in mem_usage formula
                    ]

    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove negative or zero resource values
    for col in ["cpu_usage", "mem_usage"]:
        df = df[df[col] > 0]

    # Remove invalid or missing start timestamps
    df = df[df["start_timestamp"].notna()]

    # Remove negative durations (end before start)
    df = df[(df["end_timestamp"] >= df["start_timestamp"]) | (df["end_timestamp"].isna())]

    # Create duration feature
    df["duration_minutes"] = ((df["end_timestamp"] - df["start_timestamp"]).dt.total_seconds() / 60)

    # Drop rows where duration is null (end_timestamp was null) or zero
    df = df[df["duration_minutes"].notna()]
    df = df[df["duration_minutes"] > 0]

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
    print(f"\n=== cpu_usage / mem_usage stats ===\n{df[['cpu_usage', 'mem_usage']].describe()}")
    print(f"\n=== sample (5 batch, 5 interactive) ===")
    sample = pd.concat([
        df[df["job_type"] == "batch"].head(5),
        df[df["job_type"] == "interactive"].head(5)
    ])
    print(sample.to_string())

if __name__ == "__main__":
    clean_data(input_path="data/interim/df_jobs.parquet", output_path="data/interim/cleaned_data.parquet")