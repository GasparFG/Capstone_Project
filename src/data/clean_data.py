import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def clean_data(input_path, output_path):
    """
    Cleans workload dataset.
    """

    # Load dataset
    df_jobs = pd.read_parquet(input_path)

    initial_rows = len(df_jobs)

    # Convert timestamps
    timestamp_columns = ["scheduled_time","deletion_time", "creation_time"]
    for col in timestamp_columns:
        df_jobs[col] = pd.to_timedelta(df_jobs[col], unit="s")

     # Remove invalid or missing start timestamps
    df_jobs = df_jobs[df_jobs["scheduled_time"].notna()]
    after_scheduled_time = len(df_jobs)

    # Remove negative durations (end before start)
    df_jobs = df_jobs[(df_jobs["deletion_time"] >= df_jobs["scheduled_time"]) | (df_jobs["deletion_time"].isna())]
    after_negative_duration = len(df_jobs)

    # Create duration feature
    df_jobs["duration_seconds"] = (df_jobs["deletion_time"] - df_jobs["scheduled_time"]).dt.total_seconds()
    df_jobs["duration_minutes"] = ((df_jobs["deletion_time"] - df_jobs["scheduled_time"]).dt.total_seconds() / 60)

    # Drop rows where duration is null (deletion_time was null) or zero
    df_jobs = df_jobs[df_jobs["duration_minutes"].notna()]
    after_null_duration = len(df_jobs)
    df_jobs = df_jobs[df_jobs["duration_minutes"] > 0]
    after_zero_duration = len(df_jobs)


    # Categorizing job type (Longer than 60 minutes = Batch. Shorter or equal than 60 minutes = Interactive)
    df_jobs["job_type"] = np.where(df_jobs["duration_minutes"] <= 60,"interactive","batch")

    # Remove negative or zero resource values
    for col in ["cpu_request", "memory_request"]:
        df_jobs = df_jobs[df_jobs[col] > 0]
    after_cpu_memory = len(df_jobs)

    # Remove duplicates
    df_jobs = df_jobs.drop_duplicates()
    after_duplicates = len(df_jobs)

    # Remove outliers in duration_minutes using IQR

    Q1 = df_jobs["duration_minutes"].quantile(0.25) 
    Q3 = df_jobs["duration_minutes"].quantile(0.75)
    IQR = Q3 - Q1 

    lower_bound = Q1 - 1.5 * IQR 
    upper_bound = Q3 + 1.5 * IQR 

    df_jobs = df_jobs[(df_jobs["duration_minutes"] >= lower_bound) & (df_jobs["duration_minutes"] <= upper_bound)].copy()

    after_outliers = len(df_jobs)
    
    #Remove jobs over 24 hours
    df_jobs = df_jobs[df_jobs["duration_minutes"] < 1440]
    after_24h_removal = len(df_jobs)

    #Save clean dataset
    df_jobs.to_parquet(output_path, index=False)
    
    #Histogram for Memory and CPU disribution (for interactive and batch jobs)
    interactive_jobs = df_jobs[df_jobs["job_type"] == "interactive"]
    batch_jobs = df_jobs[df_jobs["job_type"] == "batch"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(interactive_jobs["cpu_request"], bins=50, edgecolor="black")
    axes[0, 0].set_title("Interactive Jobs - CPU Request")
    axes[0, 0].set_xlabel("CPU Request")
    axes[0, 0].set_ylabel("Frequency")

    axes[0, 1].hist(interactive_jobs["memory_request"], bins=50, edgecolor="black")
    axes[0, 1].set_title("Interactive Jobs - Memory Request")
    axes[0, 1].set_xlabel("Memory Request")
    axes[0, 1].set_ylabel("Frequency")

    axes[1, 0].hist(batch_jobs["cpu_request"], bins=50, edgecolor="black")
    axes[1, 0].set_title("Batch Jobs - CPU Request")
    axes[1, 0].set_xlabel("CPU Request")
    axes[1, 0].set_ylabel("Frequency")

    axes[1, 1].hist(batch_jobs["memory_request"], bins=50, edgecolor="black")
    axes[1, 1].set_title("Batch Jobs - Memory Request")
    axes[1, 1].set_xlabel("Memory Request")
    axes[1, 1].set_ylabel("Frequency")

    plt.tight_layout()

    # Save Histogtam
    plt.savefig("resource_requests_by_job_type.png", dpi=300, bbox_inches="tight")

    print("Histogram saved to: resource_requests_by_job_type.png")

    # Preview results
    print(f"Cleaned dataset saved to: {output_path}")
    print("\nRows remaining after each cleaning step:")
    print(f"Initial dataset: {initial_rows:,}")
    print(f"After removing missing scheduled_time: {after_scheduled_time:,}")
    print(f"After removing negative durations: {after_negative_duration:,}")
    print(f"After removing null durations: {after_null_duration:,}")
    print(f"After removing zero durations: {after_zero_duration:,}")
    print(f"After filtering cpu_and_memory_request > 0: {after_cpu_memory:,}")
    print(f"After removing duplicates: {after_duplicates:,}")
    print(f"After removing IQR outliers: {after_outliers:,}")
    print(f"After removing jobs over 24h: {after_24h_removal:,}")
    print(f"Shape: {df_jobs.shape}")
    print(f"\n=== dtypes ===\n{df_jobs.dtypes}")
    print(f"\n=== null counts ===\n{df_jobs.isnull().sum()}")
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