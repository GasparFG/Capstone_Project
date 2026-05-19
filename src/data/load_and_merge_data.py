import pandas as pd
from pathlib import Path
# ════════════════════════════════════════════════════════════════════════════════
# LOAD
# ════════════════════════════════════════════════════════════════════════════════
def load_raw(data_dir: str = "data/raw") -> dict[str, pd.DataFrame]: 

    p = Path(data_dir)
    
    return {
        "batch_instance": pd.read_csv(p / "batch_instance.csv", names=[
            "start_timestamp", "end_timestamp", "job_id", "task_id", "machineID",
            "status", "seq_no", "total_seq_no",
            "real_cpu_max", "real_cpu_avg", "real_mem_max", "real_mem_avg",
        ]),
 
        "batch_task": pd.read_csv(p / "batch_task.csv", names=[
            "create_timestamp", "modify_timestamp", "job_id", "task_id",
            "instance_num", "status", "plan_cpu", "plan_mem",
        ]),
 
        "container_event": pd.read_csv(p / "container_event.csv", names=[
            "timestamp", "event_type", "instance_id", "machine_id",
            "plan_cpu", "plan_mem", "plan_disk", "cpuset",
        ],index_col=False),
 
        "container_usage": pd.read_csv(p / "container_usage.csv", names=[
            "timestamp", "instance_id", "cpu_util", "mem_util", "disk_util",
            "load1", "load5", "load15",
            "avg_cpi", "avg_mpki", "max_cpi", "max_mpki",
        ]),
 
        "server_event": pd.read_csv(p / "server_event.csv", names=[
            "timestamp", "machineID", "event_type", "event_detail",
            "capacity_cpu", "capacity_memory", "capacity_disk",
        ]),
    }

 

# ════════════════════════════════════════════════════════════════════════════════
# BUILD JOBS
# ════════════════════════════════════════════════════════════════════════════════

def build_jobs(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:

    # Last-known capacity per machine — resolves the server_event 1:N to one
    # canonical spec row per machineID. This is a heuristic, but the best we can do given the data. 
    server_capacity = (
        raw["server_event"]
        .sort_values("timestamp")
        .groupby("machineID", as_index=False)
        .last()
        [["machineID", "capacity_cpu", "capacity_memory"]]
        # capacity_disk dropped: disk is not modelled in the MILP 
    )
    
    # First create event per container
    container_creation = (
        raw["container_event"]
        .sort_values("timestamp")
        .groupby("instance_id", as_index=False)
        .first()
        [["instance_id", "machine_id", "plan_cpu", "plan_mem"]]
        # event_type dropped: zero information content, all values identical
        # plan_disk, cpuset dropped: disk not modelled; cpuset is internal OS detail
    )
    
    # Container lifetime from usage log — no delete events exist in this release,
    # so first/last usage timestamp is the only source of lifetime.
    container_lifetime = (
        raw["container_usage"]
        .groupby("instance_id")["timestamp"]
        .agg(start_timestamp="min", end_timestamp="max")
        .reset_index()
    )
    
    
    # ════════════════════════════════════════════════════════════════════════════════
    # MERGE
    # ════════════════════════════════════════════════════════════════════════════════
    
    df_batch = (
        raw["batch_instance"]
        .merge(raw["batch_task"],   on=["job_id", "task_id"], how="left",
            suffixes=("_instance", "_task"))
        .merge(server_capacity, on="machineID", how="left")
    )
    
    # Aggregate container usage from sample-level to job-level before joining.
    container_usage_agg = (
        raw["container_usage"]
        .groupby("instance_id")
        .agg(
            real_cpu_avg = ("cpu_util", "mean"),
            real_cpu_max = ("cpu_util", "max"),
            real_mem_avg = ("mem_util", "mean"),
            real_mem_max = ("mem_util", "max"),
        )
        # disk_util, load1/5/15 dropped: disk not modelled; 
        # avg_cpi, avg_mpki, max_cpi, max_mpki dropped: microarchitectural metrics not modelled
        .reset_index()
    )
    
    df_container = (
        container_usage_agg
        .merge(container_creation, on="instance_id", how="left")
        .merge(container_lifetime, on="instance_id", how="left")
        .merge(server_capacity.rename(columns={"machineID": "machine_id"}),
            on="machine_id", how="left")
    )
    
    
    # ════════════════════════════════════════════════════════════════════════════════
    # SELECT — drop columns from df_batch
    # ════════════════════════════════════════════════════════════════════════════════
    #
    #   seq_no, total_seq_no  — internal instance sequencing, not a scheduling input
    #   status_instance       — runtime outcome, not a scheduling input
    #   status_task           — task-level outcome, redundant with status_instance
    #   create_timestamp      — task submission time, superseded by start_timestamp
    #   modify_timestamp      — internal bookkeeping
    #   instance_num          — count of parallel instances; recoverable via groupby
    #                           if needed later
    
    df_batch = df_batch[[
        "job_id", "task_id",
        "machineID",
        "plan_cpu", "plan_mem",
        "real_cpu_avg", "real_cpu_max",
        "real_mem_avg", "real_mem_max",
        "start_timestamp", "end_timestamp",
        "capacity_cpu", "capacity_memory",
    ]].rename(columns={"machineID": "machine_id"})
    
    
    # ════════════════════════════════════════════════════════════════════════════════
    # CONCAT — unified jobs table
    # ════════════════════════════════════════════════════════════════════════════════
    
    batch_prepped = df_batch.assign(
        uid      = "B_" + df_batch["job_id"].astype(str)
                        + "_" + df_batch["task_id"].astype(str),
        job_type = "batch",
    )
    
    interactive_prepped = (
        df_container
        .rename(columns={"instance_id": "job_id"})
        .assign(
            uid      = lambda x: "I_" + x["job_id"].astype(str),
            job_type = "interactive",
            task_id  = pd.NA,
        )
    )
    
    df_jobs = (
        pd.concat([batch_prepped, interactive_prepped], ignore_index=True)
        [[
            "uid", "job_id", "task_id", "job_type",
            "machine_id",
            "plan_cpu", "plan_mem",
            "real_cpu_avg", "real_cpu_max",
            "real_mem_avg", "real_mem_max",
            "start_timestamp", "end_timestamp",
            "capacity_cpu", "capacity_memory",
        ]]
    )
    return df_jobs

# ════════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ════════════════════════════════════════════════════════════════════════════════
def save_jobs(df_jobs: pd.DataFrame, out_dir: str = "data/interim") -> Path:
    """
    Write df_jobs to parquet.
 
    Returns
    -------
    Path to the written file.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
 
    file_path = out_path / "merged_jobs.parquet"
    df_jobs.to_parquet(file_path, index=False)
 
    print(f"Saved {len(df_jobs):,} rows → {file_path}")
    return file_path

 
 
# ════════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════
 
if __name__ == "__main__":
    raw     = load_raw()
    df_jobs = build_jobs(raw)
   
    print("=== df_jobs shape ===")
    print(df_jobs.shape)
    
    print("\n=== job_type counts ===")
    print(df_jobs["job_type"].value_counts())
    
    print("\n=== dtypes ===")
    print(df_jobs.dtypes)
    
    print("\n=== null counts ===")
    print(df_jobs.isnull().sum())
    
    print("\n=== sample (5 batch, 5 interactive) ===")
    print(pd.concat([
        df_jobs[df_jobs["job_type"] == "batch"].head(5),
        df_jobs[df_jobs["job_type"] == "interactive"].head(5)
    ]))
    
    save_jobs(df_jobs)

