import pandas as pd
import numpy as np

def clean_data(input_path, output_path):
    """
    Cleans workload dataset.
    """

    # Load dataset
    df_jobs = pd.read_parquet(input_path)

    # Convert timestamps
    timestamp_columns = ["scheduled_time","deletion_time", "creation_time"]
    for col in timestamp_columns:
        df_jobs[col] = pd.to_timedelta(df_jobs[col], unit="s")

     # Remove invalid or missing start timestamps
    df_jobs = df_jobs[df_jobs["scheduled_time"].notna()]

    # Remove negative durations (end before start)
    df_jobs = df_jobs[(df_jobs["deletion_time"] >= df_jobs["scheduled_time"]) | (df_jobs["deletion_time"].isna())]

    # Create duration feature
    df_jobs["duration_seconds"] = (df_jobs["deletion_time"] - df_jobs["scheduled_time"]).dt.total_seconds()
    df_jobs["duration_minutes"] = ((df_jobs["deletion_time"] - df_jobs["scheduled_time"]).dt.total_seconds() / 60)

    # Drop rows where duration is null (deletion_time was null) or zero
    df_jobs = df_jobs[df_jobs["duration_minutes"].notna()]
    df_jobs = df_jobs[df_jobs["duration_minutes"] > 0]
    
    # Categorizing job type (Longer than 60 minutes = Batch. Shorter or equal than 60 minutes = Interactive)
    df_jobs["job_type"] = np.where(df_jobs["duration_minutes"] <= 60,"interactive","batch")

    # Remove negative or zero resource values
    for col in ["cpu_request", "memory_request"]:
        df_jobs = df_jobs[df_jobs[col] > 0]

    # Remove duplicates
    df_jobs = df_jobs.drop_duplicates()

    # Remove outliers in duration_minutes using IQR
    rows_before = len(df_jobs)

    Q1 = df_jobs["duration_minutes"].quantile(0.25) 
    Q3 = df_jobs["duration_minutes"].quantile(0.75)
    IQR = Q3 - Q1 

    lower_bound = Q1 - 1.5 * IQR 
    upper_bound = Q3 + 1.5 * IQR 

    df_jobs = df_jobs[(df_jobs["duration_minutes"] >= lower_bound) & (df_jobs["duration_minutes"] <= upper_bound)].copy()

    rows_after = len(df_jobs)
    
    #Save clean dataset
    df_jobs.to_parquet(output_path, index=False)

    # Preview results
    print(f"Cleaned dataset saved to: {output_path}")
    print(f"Shape: {df_jobs.shape}")
    print(f"\n=== dtypes ===\n{df_jobs.dtypes}")
    print(f"\n=== null counts ===\n{df_jobs.isnull().sum()}")
    print(f"Removed {rows_before - rows_after} outliers "f"({100 * (rows_before - rows_after) / rows_before:.2f}%)")
    print(f"\n=== job_type counts ===\n{df_jobs['job_type'].value_counts()}")
    print(f"\n=== cpu_usage / mem_usage stats ===\n{df_jobs[['cpu_request', 'memory_request']].describe()}")
    print(f"\n=== sample (5 batch, 5 interactive) ===")
    sample = pd.concat([
        df_jobs[df_jobs["job_type"] == "batch"].head(5),
        df_jobs[df_jobs["job_type"] == "interactive"].head(5)
    ])
    print(sample.to_string())

if __name__ == "__main__":
    clean_data(input_path="data/interim/df_jobs.parquet", output_path="data/interim/cleaned_data.parquet")