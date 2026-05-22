import pandas as pd
from pathlib import Path
# ════════════════════════════════════════════════════════════════════════════════
# LOAD
# ════════════════════════════════════════════════════════════════════════════════
def load_raw(data_dir: str = "data/raw") -> dict[str, pd.DataFrame]: 

    p = Path(data_dir)
    
    df_raw =  pd.read_csv(p / "disaggregated_DLRM_trace.csv")

    return df_raw

 
# ════════════════════════════════════════════════════════════════════════════════
# SELECT — drop columns not needed for scheduling input
# ════════════════════════════════════════════════════════════════════════════════

def build_jobs(df_raw: pd.DataFrame) -> pd.DataFrame:
    
    # Drop unnecesary columns
    cols_to_drop = ["cpu_limit",
                    "gpu_limit",
                    "rdma_request",       
                    "rdma_limit",         
                    "memory_limit", 
                    "disk_request",	
                    "disk_limit", 
                    "max_instance_per_node",  
                    ]
 
    df_jobs = df_raw.drop(columns=[c for c in cols_to_drop if c in df_raw.columns])

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
 
    file_path = out_path / "df_jobs.parquet"
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
    
    print("\n=== role counts ===")
    print(df_jobs["role"].value_counts())

    print("\n=== null counts ===")
    print(df_jobs.isnull().sum())
    
    print("\n=== sample ===")
    print(df_jobs.head(10))
    
    save_jobs(df_jobs)

