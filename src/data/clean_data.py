import pandas as pd
import numpy as np

def clean_data(input_path, output_path):
    """
    Cleans workload dataset.
    """

    # Load dataset
    df = pd.read_parquet(input_path)

    # Convert timestamps
    timestamp_columns = ["scheduled_time","deletion_time", "creation_time"]
    for col in timestamp_columns:
        df[col] = pd.to_timedelta(df[col], unit="s")

     # Remove invalid or missing start timestamps
    df = df[df["scheduled_time"].notna()]

    # Remove negative durations (end before start)
    df = df[(df["deletion_time"] >= df["scheduled_time"]) | (df["deletion_time"].isna())]

    # Create duration feature
    df["duration_minutes"] = ((df["deletion_time"] - df["scheduled_time"]).dt.total_seconds() / 60)

    # Drop rows where duration is null (deletion_time was null) or zero
    df = df[df["duration_minutes"].notna()]
    df = df[df["duration_minutes"] > 0]
    
    # Categorizing job type (Longer than 60 minutes = Batch. Shorter or equal than 60 minutes = Interactive)
    df["job_type"] = np.where(df["duration_minutes"] <= 60,"interactive","batch")

    # Remove negative or zero resource values
    for col in ["cpu_request", "memory_request"]:
        df = df[df[col] > 0]

    # Remove duplicates
    df = df.drop_duplicates()

    #Save clean dataset
    df.to_parquet(output_path, index=False)

    # Preview results
    print(f"Cleaned dataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"\n=== dtypes ===\n{df.dtypes}")
    print(f"\n=== null counts ===\n{df.isnull().sum()}")
    print(f"\n=== job_type counts ===\n{df['job_type'].value_counts()}")
    print(f"\n=== cpu_usage / mem_usage stats ===\n{df[['cpu_request', 'memory_request']].describe()}")
    print(f"\n=== sample (5 batch, 5 interactive) ===")
    sample = pd.concat([
        df[df["job_type"] == "batch"].head(5),
        df[df["job_type"] == "interactive"].head(5)
    ])
    print(sample.to_string())

if __name__ == "__main__":
    clean_data(input_path="data/interim/df_jobs.parquet", output_path="data/interim/cleaned_data.parquet")