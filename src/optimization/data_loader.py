"""Load forecast output and convert it into optimization input data."""

from pathlib import Path
from typing import Any, Dict, List
import math
import json


def _normalize_cooling(cooling, K):
    """Ensure cooling["eta"] is always a dict keyed by slot index."""
    eta = cooling.get("eta")
    if isinstance(eta, (int, float)):
        cooling = dict(cooling)
        cooling["eta"] = {k: float(eta) for k in K}
    elif isinstance(eta, list):
        cooling = dict(cooling)
        cooling["eta"] = {k: float(eta[k]) for k in K}
    return cooling


def load_data_from_jobs_json(
    jobs_json_path: Path,
    server_json_path: Path,
) -> Dict[str, Any]:
    """
    Load forecast-translated job parameters and server infrastructure parameters
    from JSON files, then merge them into the complete optimization input
    dictionary expected by solver.py.

    Inputs:
        jobs_json_path:
            data/processed/optimization_jobs_params.json

        server_json_path:
            server_params_42servers_v5.json
    """

    if not jobs_json_path.exists():
        raise FileNotFoundError(
            f"Jobs JSON file not found: {jobs_json_path}. "
            "Run build_jobs_json_from_forecast.py first."
        )

    if not server_json_path.exists():
        raise FileNotFoundError(
            f"Server parameters JSON file not found: {server_json_path}."
        )

    with open(jobs_json_path, "r", encoding="utf-8") as file:
        jobs_data = json.load(file)

    with open(server_json_path, "r", encoding="utf-8") as file:
        server_data = json.load(file)

    required_job_sections = [
        "sets",
        "eligibility",
        "job_params",
    ]

    missing_job_sections = [
        section for section in required_job_sections
        if section not in jobs_data
    ]

    if missing_job_sections:
        raise ValueError(
            f"Missing required sections in {jobs_json_path}: "
            f"{missing_job_sections}. "
            f"Available sections: {list(jobs_data.keys())}"
        )

    required_server_sections = [
        "sets",
        "server_params",
        "thermal",
        "cooling",
        "power",
        "maintenance",
        "costs",
        "demand",
        "redundancy",
        "slot_duration",
    ]

    missing_server_sections = [
        section for section in required_server_sections
        if section not in server_data
    ]

    if missing_server_sections:
        raise ValueError(
            f"Missing required sections in {server_json_path}: "
            f"{missing_server_sections}. "
            f"Available sections: {list(server_data.keys())}"
        )

    job_sets = jobs_data["sets"]
    server_sets = server_data["sets"]

    required_job_sets = [
        "I",
        "I_B",
        "I_V",
        "I_C",
        "E",
        "A",
        "G",
    ]

    missing_job_sets = [
        set_name for set_name in required_job_sets
        if set_name not in job_sets
    ]

    if missing_job_sets:
        raise ValueError(
            f"Missing required job sets in {jobs_json_path}: "
            f"{missing_job_sets}."
        )

    required_server_sets = [
        "J",
        "K",
        "F",
    ]

    missing_server_sets = [
        set_name for set_name in required_server_sets
        if set_name not in server_sets
    ]

    if missing_server_sets:
        raise ValueError(
            f"Missing required server sets in {server_json_path}: "
            f"{missing_server_sets}."
        )

    I = job_sets["I"]
    I_B = job_sets["I_B"]
    I_V = job_sets["I_V"]
    I_C = job_sets["I_C"]

    J = server_sets["J"]
    K = server_sets["K"]
    F = server_sets["F"]

    E = job_sets.get("E", [])
    A = job_sets.get("A", [])
    G = job_sets.get("G", [])

    eligibility = jobs_data["eligibility"]
    job_params = jobs_data["job_params"]

    valid_servers = set(J)

    for job_id, servers in eligibility.items():
        invalid_servers = [
            server for server in servers
            if server not in valid_servers
        ]

        if invalid_servers:
            raise ValueError(
                f"Job {job_id} has invalid eligible servers: "
                f"{invalid_servers}. "
                f"Valid servers are: {min(J)} to {max(J)}."
            )

    required_job_params = [
        "d",
        "r",
        "a",
        "b",
        "q",
        "rho",
    ]

    missing_job_params = [
        param for param in required_job_params
        if param not in job_params
    ]

    if missing_job_params:
        raise ValueError(
            f"Missing required job_params in {jobs_json_path}: "
            f"{missing_job_params}."
        )

    data = {
        "pipeline_note": (
            f"Optimization input loaded from two JSON files. "
            f"Jobs source: {jobs_json_path}. "
            f"Server source: {server_json_path}."
        ),

        "sets": {
            "I": I,
            "I_B": I_B,
            "I_V": I_V,
            "I_C": I_C,
            "J": J,
            "K": K,
            "F": F,
            "E": E,
            "A": A,
            "G": G,
        },

        "eligibility": eligibility,

        "job_params": job_params,

        "server_params": server_data["server_params"],

        "thermal": server_data["thermal"],

        "cooling": _normalize_cooling(server_data["cooling"], K),

        "power": server_data["power"],

        "maintenance": server_data["maintenance"],

        "costs": server_data["costs"],

        "demand": server_data["demand"],

        "redundancy": server_data["redundancy"],

        "slot_duration": server_data["slot_duration"],

        "metadata": {
            "jobs_metadata": jobs_data.get("metadata", {}),
            "server_metadata": server_data.get("_comments", {}),
        },
    }

    return data


def load_data_from_synthetic_parquet(parquet_path: Path) -> Dict[str, Any]:
    """
    Load optimization data directly from optimization_input_dataset.parquet.

    Modelling choice:
    - Each active forecast row becomes one aggregate workload job.
    - The job release slot comes from forecast_window.
    - The job resource requirement comes from aggregate_resource_demand.
    - Duration is set to one forecast window because the forecast row already
      represents demand within that window. predicted_duration_slots is kept as
      forecast metadata but is not used as the scheduling duration to avoid
      double-counting window demand across multiple slots.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Optimization input file not found: {parquet_path}. "
            "Run generate_predictions.py and build_optimization_input.py first."
        )

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required to read optimization_input_dataset.parquet."
        ) from exc

    optimization_data = pd.read_parquet(parquet_path)

    required_columns = [
        "job_type",
        "forecast_window",
        "forecast_timestamp",
        "forecast_window_minutes",
        "predicted_cpu_demand",
        "predicted_memory_demand",
        "aggregate_resource_demand",
        "predicted_duration_minutes",
        "predicted_duration_slots",
        "predicted_job_count",
        "workload_priority",
        "deadline_type",
    ]

    missing_columns = [
        column for column in required_columns if column not in optimization_data.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {parquet_path}: {missing_columns}. "
            f"Available columns: {list(optimization_data.columns)}"
        )

    optimization_data = optimization_data.copy()
    optimization_data["forecast_timestamp"] = pd.to_datetime(
        optimization_data["forecast_timestamp"])
    optimization_data["forecast_window"] = optimization_data["forecast_window"].astype(
        int)
    optimization_data["predicted_job_count"] = optimization_data["predicted_job_count"].fillna(
        0).round().clip(lower=0).astype(int)
    optimization_data["aggregate_resource_demand"] = optimization_data["aggregate_resource_demand"].fillna(
        0).clip(lower=0)
    optimization_data = optimization_data.sort_values(
        ["forecast_window", "job_type"]).reset_index(drop=True)

    forecast_windows = sorted(
        optimization_data["forecast_window"].unique().tolist())
    if not forecast_windows:
        raise ValueError(
            "The optimization input dataset has no forecast windows.")

    K = list(range(len(forecast_windows)))
    window_to_slot = {forecast_window: idx for idx,
                      forecast_window in enumerate(forecast_windows)}

    forecast_window_minutes = int(
        optimization_data["forecast_window_minutes"].dropna().iloc[0])
    slot_duration = forecast_window_minutes / 60.0
    nK = len(K)

    active_df = optimization_data[(optimization_data["predicted_job_count"] > 0) & (
        optimization_data["aggregate_resource_demand"] > 0)].copy()
    if active_df.empty:
        raise ValueError(
            "No active workload rows found in optimization_input_dataset.parquet. "
            "Check predicted_job_count and aggregate_resource_demand."
        )

    active_df = active_df.reset_index(drop=True)
    active_df["job_id"] = active_df.index.astype(int)
    active_df["release_slot"] = active_df["forecast_window"].map(
        window_to_slot).astype(int)

    I = active_df["job_id"].astype(int).tolist()
    I_B = active_df.loc[active_df["job_type"].str.lower(
    ) == "batch", "job_id"].astype(int).tolist()
    I_V = active_df.loc[active_df["job_type"].str.lower(
    ) == "interactive", "job_id"].astype(int).tolist()

    # There is no reliable criticality, precedence, affinity, or anti-affinity
    # information in the forecast output, so these sets are intentionally empty.
    I_C: List[int] = []
    E: List[List[int]] = []
    A: List[List[int]] = []
    G: List[List[int]] = []

    J = [0, 1, 2, 3]
    F = [[0, 1], [2, 3]]
    S = {str(i): J.copy() for i in I}

    # Normalize forecast demand to the same scale as server capacity.
    # Each server has capacity 1.0, so total capacity is 4.0.
    total_server_capacity = 4.0
    target_peak_utilization = 0.75
    max_raw_demand = float(active_df["aggregate_resource_demand"].max())
    if max_raw_demand <= 0:
        raise ValueError(
            "aggregate_resource_demand must contain positive values.")

    active_df["normalized_resource_demand"] = (
        active_df["aggregate_resource_demand"] / max_raw_demand
    ) * total_server_capacity * target_peak_utilization

   # Keep each aggregate job small enough to fit the model capacity rules.
    # Batch jobs must fit within the batch-reserved capacity:
    #     (1 - theta[j]) * C[j]
    # Interactive jobs can use the full server capacity.
    MIN_RESOURCE = 0.01
    SERVER_CAPACITY = 1.0
    INTERACTIVE_RESERVATION = 0.30

    MAX_BATCH_RESOURCE = 0.95 * (1 - INTERACTIVE_RESERVATION) * SERVER_CAPACITY
    MAX_INTERACTIVE_RESOURCE = 0.95 * SERVER_CAPACITY

    active_df["resource_cap"] = active_df["job_type"].str.lower().map(
        {
            "batch": MAX_BATCH_RESOURCE,
            "interactive": MAX_INTERACTIVE_RESOURCE,
        }
    ).fillna(MAX_BATCH_RESOURCE)

    active_df["normalized_resource_demand"] = active_df[
        ["normalized_resource_demand", "resource_cap"]
    ].min(axis=1)

    active_df["normalized_resource_demand"] = active_df["normalized_resource_demand"].clip(
        lower=MIN_RESOURCE
    )

    d = {str(int(row.job_id)): 1 for row in active_df.itertuples(index=False)}
    r = {
        str(int(row.job_id)): float(row.normalized_resource_demand)
        for row in active_df.itertuples(index=False)
    }
    a = {
        str(int(row.job_id)): int(row.release_slot)
        for row in active_df.itertuples(index=False)
    }

    # Interactive workloads have a hard one-window deadline.
    # Batch workloads receive a soft deadline with limited flexibility.
    batch_slack_slots = max(4, int(round(2.0 / slot_duration)))
    b = {}
    for row in active_df.itertuples(index=False):
        job_id = str(int(row.job_id))
        release_slot = int(row.release_slot)
        if str(row.job_type).lower() == "interactive":
            b[job_id] = min(nK, release_slot + 1)
        else:
            b[job_id] = min(nK, release_slot + 1 + batch_slack_slots)

    q = {str(i): 1 for i in I}
    rho = {str(i): 3.0 for i in I_B}

    # Demand by slot is kept for reporting only. It is not enforced as c30,
    # because the forecast is already represented as aggregate jobs.
    demand_by_slot = [0.0 for _ in K]
    for row in active_df.itertuples(index=False):
        demand_by_slot[int(row.release_slot)
                       ] += float(row.normalized_resource_demand)

    hourly_eta = [
        5.0, 5.0, 5.0, 5.0,
        4.5, 4.5, 4.5, 4.5,
        3.5, 3.5, 3.5, 3.5,
        3.0, 3.0, 3.0, 3.0,
        3.5, 3.5, 3.5, 3.5,
        4.5, 4.5, 4.5, 4.5,
    ]
    slots_per_hour = max(1, int(round(60 / forecast_window_minutes)))
    eta_values = [value for value in hourly_eta for _ in range(slots_per_hour)]
    if len(eta_values) < nK:
        repeats = math.ceil(nK / len(eta_values))
        eta_values = (eta_values * repeats)[:nK]
    else:
        eta_values = eta_values[:nK]

    data = {
        "pipeline_note": (
            f"Synthetic optimization input loaded directly from {parquet_path}. "
            "Active forecast rows were converted into aggregate workload jobs."
        ),
        "sets": {
            "I": I,
            "I_B": I_B,
            "I_V": I_V,
            "I_C": I_C,
            "J": J,
            "K": K,
            "F": F,
            "E": E,
            "A": A,
            "G": G,
        },
        "eligibility": S,
        "job_params": {
            "d": d,
            "r": r,
            "a": a,
            "b": b,
            "q": q,
            "rho": rho,
        },
        "server_params": {
            "C": {"0": 1.0, "1": 1.0, "2": 1.0, "3": 1.0},
            "theta": {"0": 0.30, "1": 0.30, "2": 0.30, "3": 0.30},
            "P0": {"0": 100.0, "1": 100.0, "2": 110.0, "3": 120.0},
            "dP": {"0": 150.0, "1": 150.0, "2": 160.0, "3": 180.0},
            "alpha": {"0": 0.85, "1": 0.85, "2": 0.87, "3": 0.90},
            "lambda0": {"0": 0.020, "1": 0.020, "2": 0.022, "3": 0.025},
            "lambda_pm": {"0": 0.005, "1": 0.005, "2": 0.005, "3": 0.006},
            "Lambda": {"0": 4.0, "1": 4.0, "2": 4.0, "3": 4.0},
        },
        "thermal": {
            "T_sup": 18.0,
            "T_busy": 27.0,
            "T_idle": 35.0,
            "M_big": 100.0,
            "D": [
                [0.000, 0.010, 0.005, 0.005],
                [0.010, 0.000, 0.005, 0.005],
                [0.005, 0.005, 0.000, 0.010],
                [0.005, 0.005, 0.010, 0.000],
            ],
        },
        "cooling": {"eta": eta_values},
        "power": {"P_ov": 60.0, "Pi_max": 1.56},
        "maintenance": {"d_pm": max(1, int(round(2.0 / slot_duration))), "c_pm": 50.0, "c_cm": 200.0},
        "costs": {"c_e": 0.15, "c_sw": 1.0, "S_max": max(12, nK * len(J))},
        "demand": {
            "D": demand_by_slot,
            "raw_forecast_rows": len(optimization_data),
            "active_forecast_jobs": len(I),
            "normalization_note": (
                "aggregate_resource_demand was normalized to server capacity scale."
            ),
        },
        "redundancy": {"N_min": 2, "kappa": 1, "Q_max": 999999.0},
        "slot_duration": slot_duration,
    }

    return data
