"""
Data-centre scheduling optimisation model using synthetic forecast output only.

External input:
    data/processed/optimization_input_dataset.parquet

Forecast rows from the synthetic optimization dataset are converted into
aggregate workload jobs. Static server,
thermal, energy, redundancy, and maintenance parameters remain inside this file
as modelling assumptions.

Solver:
    Gurobi

Outputs:
    outputs/optimization/optimization_solution_*.csv
    outputs/optimization/performance_metrics_*.csv
    results/tables/hourly_energy_thermal_*.csv
    results/tables/pm_schedule_*.csv
    outputs/results/reports/optimization_report_*.txt
"""

from __future__ import annotations
import argparse
import copy
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB


# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------

DEFAULT_PARQUET_INPUT = "data/processed/optimization_input_dataset.parquet"
DEFAULT_OUTPUT_ROOT = "."

STATUS_LABELS = {
    GRB.OPTIMAL: "OPTIMAL",
    GRB.TIME_LIMIT: "TIME_LIMIT",
    GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.INF_OR_UNBD: "INF_OR_UNBD",
    GRB.UNBOUNDED: "UNBOUNDED",
}


# ---------------------------------------------------------------------------
# 1. Generic helpers
# ---------------------------------------------------------------------------

def ensure_output_dirs(output_root: Path) -> Dict[str, Path]:
    """Create the folder structure."""
    paths = {
        "optimization": output_root / "outputs" / "optimization",
        "reports": output_root / "outputs" / "results" / "reports",
        "tables": output_root / "results" / "tables",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    """Write rows to CSV even when the row list is empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def slot_to_time(k: int, delta_t: float) -> str:
    """Convert a slot index into HH:MM based on the slot duration."""
    mins = int(round(k * delta_t * 60))
    return f"{mins // 60:02d}:{mins % 60:02d}"


def safe_value(var: gp.Var | gp.LinExpr | gp.QuadExpr | Any, default: float = 0.0) -> float:
    """Safely extract a numerical value from a Gurobi object."""
    try:
        if hasattr(var, "X"):
            return float(var.X)
        if hasattr(var, "getValue"):
            return float(var.getValue())
        return float(var)
    except Exception:
        return default


def get_status_label(status_code: int) -> str:
    """Return a readable Gurobi status label."""
    return STATUS_LABELS.get(status_code, f"STATUS_{status_code}")


# ---------------------------------------------------------------------------
# 2. Synthetic forecast input loader
# ---------------------------------------------------------------------------

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

    df = pd.read_parquet(parquet_path)

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
        column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {parquet_path}: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df["forecast_timestamp"] = pd.to_datetime(df["forecast_timestamp"])
    df["forecast_window"] = df["forecast_window"].astype(int)
    df["predicted_job_count"] = df["predicted_job_count"].fillna(
        0).round().clip(lower=0).astype(int)
    df["aggregate_resource_demand"] = df["aggregate_resource_demand"].fillna(
        0).clip(lower=0)
    df = df.sort_values(["forecast_window", "job_type"]).reset_index(drop=True)

    forecast_windows = sorted(df["forecast_window"].unique().tolist())
    if not forecast_windows:
        raise ValueError(
            "The optimization input dataset has no forecast windows.")

    K = list(range(len(forecast_windows)))
    window_to_slot = {forecast_window: idx for idx,
                      forecast_window in enumerate(forecast_windows)}

    forecast_window_minutes = int(
        df["forecast_window_minutes"].dropna().iloc[0])
    slot_duration = forecast_window_minutes / 60.0
    nK = len(K)

    active_df = df[(df["predicted_job_count"] > 0) & (
        df["aggregate_resource_demand"] > 0)].copy()
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
            "raw_forecast_rows": len(df),
            "active_forecast_jobs": len(I),
            "normalization_note": (
                "aggregate_resource_demand was normalized to server capacity scale."
            ),
        },
        "redundancy": {"N_min": 2, "kappa": 1, "Q_max": 999999.0},
        "slot_duration": slot_duration,
    }

    return data

# ---------------------------------------------------------------------------
# 3. Scenario generation
# ---------------------------------------------------------------------------


def build_scenarios(base_data: Dict[str, Any], run_scenarios: bool) -> Dict[str, Dict[str, Any]]:
    """
    Create simple stress-test scenarios for the orange integration review.

    These scenarios are not replacements for the final forecast scenarios. They
    allow the optimization pipeline to be tested before the other project parts
    are completed.
    """
    scenarios = {"base": copy.deepcopy(base_data)}

    if not run_scenarios:
        return scenarios

    high_demand = copy.deepcopy(base_data)
    high_demand["demand"]["D"] = [
        float(x) * 1.25 for x in high_demand["demand"]["D"]]
    scenarios["high_demand_25pct"] = high_demand

    high_energy = copy.deepcopy(base_data)
    high_energy["costs"]["c_e"] = float(high_energy["costs"]["c_e"]) * 1.50
    scenarios["high_energy_cost_50pct"] = high_energy

    reduced_capacity = copy.deepcopy(base_data)
    reduced_capacity["server_params"]["C"] = {
        str(k): float(v) * 0.90 for k, v in reduced_capacity["server_params"]["C"].items()
    }
    scenarios["reduced_capacity_10pct"] = reduced_capacity

    return scenarios


# ---------------------------------------------------------------------------
# 4. MILP builder and solver
# ---------------------------------------------------------------------------

def solve_datacenter_model(
    data: Dict[str, Any],
    scenario_name: str,
    time_limit: int,
    mip_gap: float,
    verbose: bool,
) -> Dict[str, Any]:
    """Build and solve the MILP for one scenario."""

    # -----------------------------
    # 4.1 Load sets and parameters
    # -----------------------------
    I = data["sets"]["I"]
    I_B = data["sets"]["I_B"]
    I_V = data["sets"]["I_V"]
    I_C = data["sets"]["I_C"]
    J = data["sets"]["J"]
    K = data["sets"]["K"]
    F = data["sets"]["F"]
    E = data["sets"]["E"]
    A = data["sets"]["A"]
    G = data["sets"]["G"]
    S = {int(k): v for k, v in data["eligibility"].items()}

    jp = data["job_params"]
    d = {int(k): v for k, v in jp["d"].items()}
    r = {int(k): v for k, v in jp["r"].items()}
    a = {int(k): v for k, v in jp["a"].items()}
    b = {int(k): v for k, v in jp["b"].items()}
    q = {int(k): v for k, v in jp["q"].items()}
    rho = {int(k): v for k, v in jp["rho"].items()}

    # Non-critical jobs need one replica by definition.
    for i in I:
        if i not in q:
            q[i] = 1

    sp = data["server_params"]
    C = {int(k): v for k, v in sp["C"].items()}
    theta = {int(k): v for k, v in sp["theta"].items()}
    P0 = {int(k): v for k, v in sp["P0"].items()}
    dP = {int(k): v for k, v in sp["dP"].items()}
    alpha = {int(k): v for k, v in sp["alpha"].items()}
    lambda0 = {int(k): v for k, v in sp["lambda0"].items()}
    lambda_pm = {int(k): v for k, v in sp["lambda_pm"].items()}
    Lambda = {int(k): v for k, v in sp["Lambda"].items()}

    th = data["thermal"]
    T_sup = th["T_sup"]
    T_busy = th["T_busy"]
    T_idle = th["T_idle"]
    M_big = th["M_big"]
    D = th["D"]

    eta = {k: data["cooling"]["eta"][k] for k in K}
    P_ov = data["power"]["P_ov"]
    Pi_max = data["power"]["Pi_max"]
    d_pm = data["maintenance"]["d_pm"]
    c_pm = data["maintenance"]["c_pm"]
    c_cm = data["maintenance"]["c_cm"]
    c_e = data["costs"]["c_e"]
    c_sw = data["costs"]["c_sw"]
    S_max = data["costs"]["S_max"]
    Dk = {k: data["demand"]["D"][k] for k in K}
    N_min = data["redundancy"]["N_min"]
    kappa = data["redundancy"]["kappa"]
    Q_max = data["redundancy"]["Q_max"]
    delta_t = data["slot_duration"]
    nK = len(K)

    def local_slot_to_time(k: int) -> str:
        return slot_to_time(k, delta_t)

    def valid_starts(job: int) -> List[int]:
        """Valid start slots: release time, horizon end, and hard deadline for interactive jobs."""
        upper = nK - d[job]
        if job in I_V:
            upper = min(upper, b[job] - d[job])
        return [k for k in K if a[job] <= k <= upper]

    # X is referenced by running_at, so the helper is defined after X exists.
    X: Dict[Tuple[int, int, int], gp.Var] = {}

    def running_at(job: int, server: int, slot: int) -> List[int]:
        """Return start slots where job is running on server during slot."""
        return [
            kp for kp in valid_starts(job)
            if (job, server, kp) in X and kp <= slot < kp + d[job]
        ]

    # -----------------------------
    # 4.2 Build model
    # -----------------------------
    params = {
        "WLSACCESSID": "fc17fa3a-ef7f-41d2-b95c-20c3b221a483",
        "WLSSECRET": "6bee54d1-5c9f-4f12-9d64-0c7b16e0dd52",
        "LICENSEID": 2804943
    }

    env = gp.Env(empty=True)
    for key, value in params.items():
        env.setParam(key, value)
    env.start()
    mdl = gp.Model(f"datacenter_1day_{scenario_name}", env=env)
    mdl.setParam("TimeLimit", time_limit)
    mdl.setParam("MIPGap", mip_gap)
    if not verbose:
        mdl.setParam("OutputFlag", 0)

    # -----------------------------
    # 4.3 Decision variables
    # -----------------------------
    # R_{ijk} remains eliminated. It is fully determined by X.
    X.update({
        (i, j, k): mdl.addVar(vtype=GRB.BINARY, name=f"X_{i}_{j}_{k}")
        for i in I for j in S[i] for k in valid_starts(i)
    })

    y = {(j, k): mdl.addVar(vtype=GRB.BINARY, name=f"y_{j}_{k}")
         for j in J for k in K}
    d_on = {(j, k): mdl.addVar(vtype=GRB.BINARY, name=f"don_{j}_{k}")
            for j in J for k in K[:-1]}
    d_off = {(j, k): mdl.addVar(vtype=GRB.BINARY, name=f"doff_{j}_{k}")
             for j in J for k in K[:-1]}
    m_j = {j: mdl.addVar(vtype=GRB.BINARY, name=f"m_{j}") for j in J}

    # All possible starts for preventive maintenance
    pm_starts = [k for k in K if k <= nK - d_pm]
    v = {(j, k): mdl.addVar(vtype=GRB.BINARY, name=f"v_{j}_{k}")
         for j in J for k in pm_starts}
    z = {(j, k): mdl.addVar(vtype=GRB.BINARY, name=f"z_{j}_{k}")
         for j in J for k in K}

    # Phi or McCormick linearization of mj * yjk
    mk = {(j, k): mdl.addVar(lb=0.0, ub=1.0, name=f"mk_{j}_{k}")
          for j in J for k in K}

    l_var = {i: mdl.addVar(lb=0.0, name=f"l_{i}") for i in I_B}
    L = {(j, k): mdl.addVar(lb=0.0, ub=1.0, name=f"L_{j}_{k}")
         for j in J for k in K}
    H = {(j, k): mdl.addVar(lb=0.0, name=f"H_{j}_{k}") for j in J for k in K}
    PIT = {k: mdl.addVar(lb=0.0, name=f"PIT_{k}") for k in K}
    Pcool = {k: mdl.addVar(lb=0.0, name=f"Pcool_{k}") for k in K}
    Ptot = {k: mdl.addVar(lb=0.0, name=f"Ptot_{k}") for k in K}
    s = {i: mdl.addVar(lb=0.0, ub=nK - 1, name=f"s_{i}") for i in I}
    psi = {(j, k): mdl.addVar(lb=0.0, name=f"psi_{j}_{k}")
           for j in J for k in K}
    # Critical job official start slot (to force all replicas to start at the same time)
    u = {(i, k): mdl.addVar(vtype=GRB.BINARY, name=f"u_{i}_{k}")
         for i in I_C for k in valid_starts(i)}

    mdl.update()

    # -----------------------------
    # 4.4 Objective (#4)
    # -----------------------------
    energy_cost = c_e * delta_t / 1000.0 * gp.quicksum(Ptot[k] for k in K)
    pm_cost = gp.quicksum(c_pm * m_j[j] for j in J)
    cm_cost = c_cm * gp.quicksum(
        lambda0[j] * y[j, k] - (lambda0[j] - lambda_pm[j]) * mk[j, k]
        for j in J for k in K
    )
    sw_cost = c_sw * gp.quicksum(d_on[j, k] + d_off[j, k]
                                 for j in J for k in K[:-1])
    late_cost = gp.quicksum(rho[i] * l_var[i] for i in I_B)

    mdl.setObjective(energy_cost + pm_cost + cm_cost +
                     sw_cost + late_cost, GRB.MINIMIZE)

    # -----------------------------
    # 4.5 Constraints
    # -----------------------------

    # --- #5 Job assignment (exact replica count) ---
    for i in I:
        mdl.addConstr(
            gp.quicksum(X[i, j, k] for j in S[i]
                        for k in valid_starts(i)) == q[i],
            name=f"c5_{i}",
        )

    # --- #6/#7 Release time and interactive hard deadlines are enforced in valid_starts(). ---

    # --- Critical replicas start synchronously ---
    for i in I_C:
        for k in valid_starts(i):
            mdl.addConstr(
                gp.quicksum(X[i, j, k]
                            for j in S[i] if (i, j, k) in X) == q[i] * u[i, k],
                name=f"crit_sync_{i}_{k}"
            )

    # --- #8 Precedence ---
    for i_pred, i_succ in E:
        mdl.addConstr(s[i_succ] >= s[i_pred] + d[i_pred],
                      name=f"c8_{i_pred}_{i_succ}")

    # --- Start-time definition ---
    for i in I:
        if i in I_C:
            mdl.addConstr(
                s[i] == gp.quicksum(k * u[i, k] for k in valid_starts(i)),
                name=f"cs_crit_{i}",
            )
        else:
            mdl.addConstr(
                s[i] == gp.quicksum(k * X[i, j, k]
                                    for j in S[i] for k in valid_starts(i)),
                name=f"cs_{i}",
            )

    # --- #9 Batch-only capacity (interactive reservation) ---
    for j in J:
        for k in K:
            batch_load = gp.quicksum(
                r[i] * X[i, j, kp]
                for i in I_B if j in S[i]
                for kp in running_at(i, j, k)
            )
            mdl.addConstr(batch_load <= (
                1 - theta[j]) * C[j] * y[j, k], name=f"c9_{j}_{k}")

    # --- #10 Batch lateness ---
    for i in I_B:
        if i not in I_C:
            mdl.addConstr(l_var[i] >= s[i] + d[i] - b[i], name=f"c10nc_{i}")

    M_lat = nK
    for i in I_B:
        if i in I_C:
            for j in S[i]:
                for k in valid_starts(i):
                    mdl.addConstr(
                        l_var[i] >= k + d[i] - b[i] - M_lat * (1 - X[i, j, k]),
                        name=f"c10cr_{i}_{j}_{k}",
                    )

    # --- #12/#13 Load definition and server capacity ---
    for j in J:
        for k in K:
            load_expr = gp.quicksum(
                r[i] * X[i, j, kp]
                for i in I if j in S[i]
                for kp in running_at(i, j, k)
            )
            mdl.addConstr(L[j, k] == load_expr, name=f"c13_{j}_{k}")
            mdl.addConstr(L[j, k] <= C[j] * y[j, k], name=f"c12_{j}_{k}")

    # --- #14 Server cannot be active and under preventive maintenance simultaneously ---
    for j in J:
        for k in K:
            mdl.addConstr(y[j, k] + z[j, k] <= 1, name=f"c14_{j}_{k}")

    # --- #15 Total IT power ---
    for k in K:
        mdl.addConstr(
            PIT[k] == gp.quicksum(P0[j] * y[j, k] + dP[j]
                                  * L[j, k] for j in J),
            name=f"c15_{k}",
        )

    # --- #16 Heat per server ---
    for j in J:
        for k in K:
            mdl.addConstr(
                H[j, k] == alpha[j] * (P0[j] * y[j, k] + dP[j] * L[j, k]),
                name=f"c16_{j}_{k}",
            )

    # --- #17 Cooling power ---
    for k in K:
        mdl.addConstr(Pcool[k] == (1.0 / eta[k]) *
                      gp.quicksum(H[j, k] for j in J), name=f"c17_{k}")

    # --- #18 Total facility power ---
    for k in K:
        mdl.addConstr(Ptot[k] == PIT[k] + Pcool[k] + P_ov, name=f"c18_{k}")

    # --- #20 PUE cap ---
    for k in K:
        mdl.addConstr(Ptot[k] <= Pi_max * PIT[k], name=f"c20_{k}")

    # --- #21 Thermal: server inlet temperature ---
    for j in J:
        for k in K:
            recirc = gp.quicksum(
                D[j][jp2] * alpha[jp2] *
                (P0[jp2] * y[jp2, k] + dP[jp2] * L[jp2, k])
                for jp2 in J
            )
            mdl.addConstr(
                T_sup + recirc <= T_idle -
                (T_idle - T_busy) * y[j, k] + M_big * z[j, k],
                name=f"c21_{j}_{k}",
            )

    # --- #22 PM count per server ---
    for j in J:
        mdl.addConstr(gp.quicksum(v[j, k]
                      for k in pm_starts) == m_j[j], name=f"c22_{j}")

    # --- #23 PM active window ---
    for j in J:
        for k in K:
            win = [kp for kp in pm_starts if max(0, k - d_pm + 1) <= kp <= k]
            mdl.addConstr(z[j, k] == gp.quicksum(v[j, kp]
                          for kp in win), name=f"c23_{j}_{k}")

    # --- #24 Max servers under PM per slot ---
    for k in K:
        mdl.addConstr(gp.quicksum(z[j, k] for j in J)
                      <= len(J) - N_min, name=f"c24_{k}")

    # --- #25 Cumulative load ---
    for j in J:
        for k in K:
            if k == 0:
                mdl.addConstr(psi[j, k] == L[j, k], name=f"c25_{j}_{k}")
            else:
                mdl.addConstr(psi[j, k] == psi[j, k - 1] +
                              L[j, k], name=f"c25_{j}_{k}")

    # --- #26 PM may only start once cumulative load reaches threshold ---
    for j in J:
        for k in pm_starts:
            mdl.addConstr(psi[j, k] >= Lambda[j] *
                          v[j, k], name=f"c26_{j}_{k}")

    # --- #27/#28 Server state-change tracking ---
    for j in J:
        for k in K[:-1]:
            mdl.addConstr(y[j, k + 1] - y[j, k] <=
                          d_on[j, k], name=f"c27_{j}_{k}")
            mdl.addConstr(y[j, k] - y[j, k + 1] <=
                          d_off[j, k], name=f"c28_{j}_{k}")

    # --- #29 Total switching budget ---
    mdl.addConstr(gp.quicksum(d_on[j, k] + d_off[j, k]
                  for j in J for k in K[:-1]) <= S_max, name="c29")

    # --- #30 Forecast demand representation ---
    # The synthetic forecast is represented directly as aggregate workload jobs.
    # Therefore, the old hard aggregate demand constraint is not added here.
    # Adding it would double-count the same forecast demand and can make the
    # model infeasible.

    # --- #31 Anti-affinity isolation for critical job pairs ---
    for i1, i2 in G:
        if i1 in I_C and i2 in I_C:
            for j in J:
                for k in K:
                    r1 = gp.quicksum(X[i1, j, kp]
                                     for kp in running_at(i1, j, k))
                    r2 = gp.quicksum(X[i2, j, kp]
                                     for kp in running_at(i2, j, k))
                    mdl.addConstr(r1 + r2 <= 1, name=f"c31_{i1}_{i2}_{j}_{k}")

    # --- #32 Affinity: same server ---
    for i1, i2 in A:
        for j in J:
            s1 = gp.quicksum(X[i1, j, k]
                             for k in valid_starts(i1) if (i1, j, k) in X)
            s2 = gp.quicksum(X[i2, j, k]
                             for k in valid_starts(i2) if (i2, j, k) in X)
            mdl.addConstr(s1 == s2, name=f"c32_{i1}_{i2}_{j}")

    # --- #33 Affinity critical pairs: same replica count data check ---
    for i1, i2 in A:
        if i1 in I_C:
            assert q[i1] == q[i2], f"Affinity pair ({i1},{i2}): replica counts must match"

    # --- #34 Hot standby buffer ---
    for k in K:
        mdl.addConstr(gp.quicksum(y[j, k]
                      for j in J) >= N_min + kappa, name=f"c34_{k}")

    # --- #35 Rack diversity for critical jobs ---
    for i in I_C:
        for f_idx, Ff in enumerate(F):
            mdl.addConstr(
                gp.quicksum(
                    X[i, j, k]
                    for j in Ff if j in S[i]
                    for k in valid_starts(i) if (i, j, k) in X
                ) <= 1,
                name=f"c35_{i}_{f_idx}",
            )

    # --- #36 Replication overhead budget ---
    mdl.addConstr(gp.quicksum((q[i] - 1) * r[i]
                  for i in I_C) <= Q_max, name="c36")

    # --- McCormick linearisation: mk[j,k] = m_j[j] * y[j,k] ---
    for j in J:
        for k in K:
            mdl.addConstr(mk[j, k] <= m_j[j], name=f"mc1_{j}_{k}")
            mdl.addConstr(mk[j, k] <= y[j, k], name=f"mc2_{j}_{k}")
            mdl.addConstr(mk[j, k] >= m_j[j] + y[j, k] -
                          1, name=f"mc3_{j}_{k}")

    # --- Non-critical anti-affinity: jobs cannot share a server ---
    for i1, i2 in G:
        if not (i1 in I_C and i2 in I_C):
            for j in J:
                lhs = gp.quicksum(X[i1, j, k]
                                  for k in valid_starts(i1) if (i1, j, k) in X)
                rhs = gp.quicksum(X[i2, j, k]
                                  for k in valid_starts(i2) if (i2, j, k) in X)
                mdl.addConstr(lhs + rhs <= 1, name=f"c_nca_{i1}_{i2}_{j}")

    # -----------------------------
    # 4.6 Solve
    # -----------------------------

    mdl.optimize()
    if mdl.status == GRB.INFEASIBLE:
        print("\nModel is infeasible. Computing IIS...")
        mdl.computeIIS()

        iis_path = f"infeasible_{scenario_name}.ilp"
        mdl.write(iis_path)

        print(f"\nIIS written to: {iis_path}")

        print("\nConstraints included in IIS:")
        for constr in mdl.getConstrs():
            if constr.IISConstr:
                print(constr.ConstrName)

        print("\nVariable bounds included in IIS:")
        for var in mdl.getVars():
            if var.IISLB or var.IISUB:
                print(
                    var.VarName,
                    "LB" if var.IISLB else "",
                    "UB" if var.IISUB else "",
                )
    feasible_solution = mdl.status in (
        GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) and mdl.SolCount > 0

    result: Dict[str, Any] = {
        "scenario_name": scenario_name,
        "model": mdl,
        "status_code": mdl.status,
        "status_label": get_status_label(mdl.status),
        "feasible_solution": feasible_solution,
        "data": data,
        "sets": {"I": I, "I_B": I_B, "I_V": I_V, "I_C": I_C, "J": J, "K": K},
        "params": {"d": d, "r": r, "q": q, "delta_t": delta_t, "Dk": Dk, "eta": eta},
        "vars": {
            "X": X,
            "y": y,
            "d_on": d_on,
            "d_off": d_off,
            "m_j": m_j,
            "v": v,
            "z": z,
            "mk": mk,
            "l_var": l_var,
            "L": L,
            "H": H,
            "PIT": PIT,
            "Pcool": Pcool,
            "Ptot": Ptot,
            "s": s,
            "psi": psi,
        },
        "objective_terms": {
            "energy_cost": energy_cost,
            "pm_cost": pm_cost,
            "cm_cost": cm_cost,
            "switching_cost": sw_cost,
            "lateness_cost": late_cost,
        },
        "helpers": {"slot_to_time": local_slot_to_time},
    }

    return result


# ---------------------------------------------------------------------------
# 5. Result extraction
# ---------------------------------------------------------------------------

def extract_solution_rows(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create job-level optimization_solution rows."""
    if not result["feasible_solution"]:
        return []

    scenario = result["scenario_name"]
    I_B = result["sets"]["I_B"]
    I_C = result["sets"]["I_C"]
    d = result["params"]["d"]
    delta_t = result["params"]["delta_t"]
    X = result["vars"]["X"]
    local_slot_to_time = result["helpers"]["slot_to_time"]

    rows = []
    for (i, j, k), var in sorted(X.items()):
        if var.X > 0.5:
            rows.append({
                "scenario": scenario,
                "job_id": i,
                "job_type": "batch" if i in I_B else "interactive",
                "is_critical": int(i in I_C),
                "server_id": j,
                "start_slot": k,
                "end_slot": k + d[i],
                "start_time": local_slot_to_time(k),
                "end_time": local_slot_to_time(k + d[i]),
                "duration_hours": d[i] * delta_t,
            })
    return rows


def extract_hourly_rows(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create hourly power/load rows for tables and analysis."""
    if not result["feasible_solution"]:
        return []

    scenario = result["scenario_name"]
    J = result["sets"]["J"]
    K = result["sets"]["K"]
    Dk = result["params"]["Dk"]
    eta = result["params"]["eta"]
    local_slot_to_time = result["helpers"]["slot_to_time"]

    y = result["vars"]["y"]
    L = result["vars"]["L"]
    PIT = result["vars"]["PIT"]
    Pcool = result["vars"]["Pcool"]
    Ptot = result["vars"]["Ptot"]

    rows = []
    for k in K:
        pit = PIT[k].X
        pcool = Pcool[k].X
        ptot = Ptot[k].X
        total_load = sum(L[j, k].X for j in J)
        rows.append({
            "scenario": scenario,
            "slot": k,
            "time": local_slot_to_time(k),
            "demand": Dk[k],
            "served_load": total_load,
            "active_servers": sum(1 for j in J if y[j, k].X > 0.5),
            "PIT_W": pit,
            "Pcool_W": pcool,
            "Ptot_W": ptot,
            "PUE": ptot / pit if pit > 1e-6 else math.nan,
            "COP": eta[k],
        })
    return rows


def extract_pm_rows(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create preventive-maintenance schedule rows."""
    if not result["feasible_solution"]:
        return []

    scenario = result["scenario_name"]
    J = result["sets"]["J"]
    d_pm = result["data"]["maintenance"]["d_pm"]
    K = result["sets"]["K"]
    nK = len(K)
    pm_starts = [k for k in K if k <= nK - d_pm]
    v = result["vars"]["v"]
    m_j = result["vars"]["m_j"]
    local_slot_to_time = result["helpers"]["slot_to_time"]

    rows = []
    for j in J:
        scheduled = int(m_j[j].X > 0.5)
        starts = [k for k in pm_starts if v[j, k].X > 0.5]
        if not starts:
            rows.append({
                "scenario": scenario,
                "server_id": j,
                "pm_scheduled": scheduled,
                "pm_start_slot": "",
                "pm_end_slot": "",
                "pm_start_time": "",
                "pm_end_time": "",
            })
        for k in starts:
            rows.append({
                "scenario": scenario,
                "server_id": j,
                "pm_scheduled": scheduled,
                "pm_start_slot": k,
                "pm_end_slot": k + d_pm,
                "pm_start_time": local_slot_to_time(k),
                "pm_end_time": local_slot_to_time(k + d_pm),
            })
    return rows


def extract_performance_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """Create one scenario-level performance metrics row."""
    scenario = result["scenario_name"]
    mdl = result["model"]

    metrics = {
        "scenario": scenario,
        "status": result["status_label"],
        "has_solution": int(result["feasible_solution"]),
        "objective_value": "",
        "mip_gap_pct": "",
        "runtime_seconds": safe_value(getattr(mdl, "Runtime", 0.0)),
        "num_variables": mdl.NumVars,
        "num_binary_variables": mdl.NumBinVars,
        "num_constraints": mdl.NumConstrs,
        "energy_cost": "",
        "pm_cost": "",
        "expected_cm_cost": "",
        "switching_cost": "",
        "lateness_cost": "",
        "total_facility_energy_kwh": "",
        "average_pue": "",
        "max_pue": "",
        "total_switching_events": "",
        "total_lateness_hours": "",
    }

    if not result["feasible_solution"]:
        return metrics

    K = result["sets"]["K"]
    I_B = result["sets"]["I_B"]
    J = result["sets"]["J"]
    delta_t = result["params"]["delta_t"]
    d_on = result["vars"]["d_on"]
    d_off = result["vars"]["d_off"]
    l_var = result["vars"]["l_var"]
    PIT = result["vars"]["PIT"]
    Ptot = result["vars"]["Ptot"]

    pue_values = [Ptot[k].X / PIT[k].X for k in K if PIT[k].X > 1e-6]
    total_energy_kwh = sum(Ptot[k].X * delta_t / 1000.0 for k in K)
    total_switching = sum(
        d_on[j, k].X + d_off[j, k].X for j in J for k in K[:-1])
    total_lateness_hours = sum(max(0.0, l_var[i].X) * delta_t for i in I_B)

    objective_terms = result["objective_terms"]
    metrics.update({
        "objective_value": mdl.ObjVal,
        "mip_gap_pct": 100.0 * mdl.MIPGap,
        "energy_cost": safe_value(objective_terms["energy_cost"]),
        "pm_cost": safe_value(objective_terms["pm_cost"]),
        "expected_cm_cost": safe_value(objective_terms["cm_cost"]),
        "switching_cost": safe_value(objective_terms["switching_cost"]),
        "lateness_cost": safe_value(objective_terms["lateness_cost"]),
        "total_facility_energy_kwh": total_energy_kwh,
        "average_pue": sum(pue_values) / len(pue_values) if pue_values else math.nan,
        "max_pue": max(pue_values) if pue_values else math.nan,
        "total_switching_events": total_switching,
        "total_lateness_hours": total_lateness_hours,
    })
    return metrics


# ---------------------------------------------------------------------------
# 6. Output writers
# ---------------------------------------------------------------------------

def save_result_files(result: Dict[str, Any], paths: Dict[str, Path]) -> Dict[str, Path]:
    """Save all output artifacts for one scenario."""
    scenario = result["scenario_name"]

    solution_rows = extract_solution_rows(result)
    hourly_rows = extract_hourly_rows(result)
    pm_rows = extract_pm_rows(result)
    metrics_row = extract_performance_metrics(result)

    solution_path = paths["optimization"] / \
        f"optimization_solution_{scenario}.csv"
    hourly_path = paths["tables"] / f"hourly_energy_thermal_{scenario}.csv"
    pm_path = paths["tables"] / f"pm_schedule_{scenario}.csv"
    metrics_path = paths["optimization"] / \
        f"performance_metrics_{scenario}.csv"
    report_path = paths["reports"] / f"optimization_report_{scenario}.txt"

    write_csv(
        solution_path,
        solution_rows,
        ["scenario", "job_id", "job_type", "is_critical", "server_id",
            "start_slot", "end_slot", "start_time", "end_time", "duration_hours"],
    )
    write_csv(
        hourly_path,
        hourly_rows,
        ["scenario", "slot", "time", "demand", "served_load",
            "active_servers", "PIT_W", "Pcool_W", "Ptot_W", "PUE", "COP"],
    )
    write_csv(
        pm_path,
        pm_rows,
        ["scenario", "server_id", "pm_scheduled", "pm_start_slot",
            "pm_end_slot", "pm_start_time", "pm_end_time"],
    )
    write_csv(metrics_path, [metrics_row], list(metrics_row.keys()))

    save_text_report(result, metrics_row, solution_rows,
                     hourly_rows, pm_rows, report_path)

    return {
        "solution": solution_path,
        "hourly": hourly_path,
        "pm": pm_path,
        "metrics": metrics_path,
        "report": report_path,
    }


def save_text_report(
    result: Dict[str, Any],
    metrics: Dict[str, Any],
    solution_rows: List[Dict[str, Any]],
    hourly_rows: List[Dict[str, Any]],
    pm_rows: List[Dict[str, Any]],
    report_path: Path,
) -> None:
    """Save a readable text report for quick review."""
    scenario = result["scenario_name"]
    note = result["data"].get("pipeline_note", "")

    lines = []
    lines.append("Optimization Report")
    lines.append("=" * 70)
    lines.append(f"Scenario: {scenario}")
    lines.append(f"Status: {metrics['status']}")
    lines.append(f"Has solution: {metrics['has_solution']}")
    if note:
        lines.append(f"Input note: {note}")
    lines.append("")

    lines.append("Performance Metrics")
    lines.append("-" * 70)
    for key, value in metrics.items():
        lines.append(f"{key}: {value}")
    lines.append("")

    lines.append("Job Schedule")
    lines.append("-" * 70)
    if solution_rows:
        for row in solution_rows:
            lines.append(
                f"Job {row['job_id']} | {row['job_type']} | server {row['server_id']} | "
                f"{row['start_time']} - {row['end_time']} | critical={row['is_critical']}"
            )
    else:
        lines.append("No job schedule available.")
    lines.append("")

    lines.append("Preventive Maintenance")
    lines.append("-" * 70)
    if pm_rows:
        for row in pm_rows:
            if row["pm_scheduled"]:
                lines.append(
                    f"Server {row['server_id']} | PM {row['pm_start_time']} - {row['pm_end_time']}"
                )
            else:
                lines.append(f"Server {row['server_id']} | no PM scheduled")
    else:
        lines.append("No PM schedule available.")
    lines.append("")

    lines.append("Hourly Energy Snapshot")
    lines.append("-" * 70)
    for row in hourly_rows:
        lines.append(
            f"Slot {row['slot']:>2} ({row['time']}) | served_load={row['served_load']:.3f} | "
            f"Ptot_W={row['Ptot_W']:.2f} | PUE={row['PUE']:.3f} | active_servers={row['active_servers']}"
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def save_combined_files(all_metrics: List[Dict[str, Any]], all_solution_rows: List[Dict[str, Any]], paths: Dict[str, Path]) -> None:
    """Save combined files across all scenarios."""
    if all_metrics:
        write_csv(paths["optimization"] / "performance_metrics.csv",
                  all_metrics, list(all_metrics[0].keys()))
        write_csv(paths["tables"] / "performance_metrics.csv",
                  all_metrics, list(all_metrics[0].keys()))

    if all_solution_rows:
        write_csv(
            paths["optimization"] / "optimization_solution.csv",
            all_solution_rows,
            ["scenario", "job_id", "job_type", "is_critical", "server_id",
                "start_slot", "end_slot", "start_time", "end_time", "duration_hours"],
        )


# ---------------------------------------------------------------------------
# 7. Console summary
# ---------------------------------------------------------------------------

def print_console_summary(result: Dict[str, Any], saved_files: Dict[str, Path]) -> None:
    """Print a compact console summary while keeping CSV/report files as the main outputs."""
    metrics = extract_performance_metrics(result)
    print("\n" + "=" * 70)
    print(f"Scenario: {result['scenario_name']}")
    print(f"Status: {metrics['status']}")
    print(f"Objective: {metrics['objective_value']}")
    print(f"Total facility energy kWh: {metrics['total_facility_energy_kwh']}")
    print(f"Average PUE: {metrics['average_pue']}")
    print("Saved files:")
    for label, path in saved_files.items():
        print(f"  {label}: {path}")


# ---------------------------------------------------------------------------
# 8. Main entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the data-centre optimization pipeline.")
    parser.add_argument("--parquet-input", default=DEFAULT_PARQUET_INPUT,
                        help="Path to synthetic optimization_input_dataset.parquet input file.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT,
                        help="Root folder for generated outputs.")
    parser.add_argument("--run-scenarios", action="store_true",
                        help="Run stress-test scenarios in addition to base.")
    parser.add_argument("--time-limit", type=int, default=120,
                        help="Gurobi time limit in seconds.")
    parser.add_argument("--mip-gap", type=float,
                        default=0.02, help="Target MIP gap.")
    parser.add_argument("--verbose", action="store_true",
                        help="Show full Gurobi solver log.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    parquet_path = Path(args.parquet_input)
    output_root = Path(args.output_root)

    paths = ensure_output_dirs(output_root)

    base_data = load_data_from_synthetic_parquet(parquet_path)
    scenarios = build_scenarios(base_data, args.run_scenarios)

    all_metrics = []
    all_solution_rows = []

    for scenario_name, scenario_data in scenarios.items():
        result = solve_datacenter_model(
            data=scenario_data,
            scenario_name=scenario_name,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            verbose=args.verbose,
        )
        saved_files = save_result_files(result, paths)
        all_metrics.append(extract_performance_metrics(result))
        all_solution_rows.extend(extract_solution_rows(result))
        print_console_summary(result, saved_files)

    save_combined_files(all_metrics, all_solution_rows, paths)

    print("\nPipeline finished.")
    print(
        f"Combined optimization solution: {paths['optimization'] / 'optimization_solution.csv'}")
    print(
        f"Combined performance metrics: {paths['optimization'] / 'performance_metrics.csv'}")
    print(f"Report folder: {paths['reports']}")
    print(f"Tables folder: {paths['tables']}")


if __name__ == "__main__":
    main()
