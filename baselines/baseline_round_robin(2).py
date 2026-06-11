"""
baseline_round_robin.py

Round-Robin scheduling baseline for comparison against the MILP optimizer.

PURPOSE:
    This script implements the non-optimized scheduling policy used as the
    baseline in the capstone research. Round-Robin is a standard heuristic
    where incoming jobs are assigned sequentially to available servers without
    considering current utilization, thermal conditions, energy prices, or
    future demand. It represents a commonly deployed, non-optimized approach
    against which the MILP model's improvements are measured.

SCHEDULING POLICY:
    - Jobs are processed in order of their release (arrival) time.
    - Each job is assigned to the next available eligible server in a rotating
      sequence (its eligibility group, as defined in the jobs JSON).
    - The job starts at the earliest time slot >= its release slot where the
      selected server has enough remaining capacity.
    - Critical jobs requiring multiple replicas are placed on different
      servers (one replica per server) for fault isolation.
    - No lookahead, no demand forecasting, no energy awareness.

ALIGNMENT WITH MILP MODEL (for apples-to-apples comparison):
    - Inputs: the SAME --jobs-json-input / --server-json-input pair consumed
      by data_loader.py / solver.py, so both models operate on an identical
      job set and server fleet for any given robustness-suite case.
    - Job preprocessing: identical d/r/a/b/q/rho fields as job_params in the
      jobs JSON (produced by build_jobs_json_from_forecast.py or
      generate_robustness_tests.py).
    - Power model: P0, dP, alpha, eta, P_ov, Pi_max read from server_params /
      thermal / cooling / power sections of the server JSON, same as
      solver.py.
    - Cost model: c_pm, c_cm, c_e, c_sw, S_max read from the server JSON's
      maintenance / costs sections; lambda0, lambda_pm, Lambda read from
      server_params, same wear-based CM cost formula
      (c_cm * lambda0[j] * psi[j,k]/Lambda[j]) as the MILP objective.
    - psi_0: loaded from server_params["psi_0"] in the same server JSON used
      by the MILP, so both models start each daily run with identical
      inherited wear state.
    - No PM is scheduled in the baseline, so psi[j,k] grows monotonically
      from psi_0[j] -- producing higher CM cost than the MILP, which can reset
      wear via PM. This is the key mechanism behind hypothesis H3.

OUTPUTS (saved to <output-root>/outputs/baseline/):
    rr_slot_metrics.csv       - Per-slot energy, PUE, utilization, and cost
    rr_job_schedule.csv       - Job-level schedule with server, start time, lateness
    rr_server_utilization.csv - Per-server load summary across all slots
    rr_summary.csv            - Single-row aggregate metrics for t-test comparison

USAGE:
    python baseline_round_robin.py \
        --jobs-json-input   data/processed/optimization_jobs_params.json \
        --server-json-input data/raw/server_params_42servers_v6.json \
        --output-root       outputs/robustness/<case_name>
"""

import argparse
import json
import math
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# 1.  CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Round-Robin baseline scheduler on the same "
                     "jobs/server JSON inputs as the MILP pipeline.")
    parser.add_argument(
        "--jobs-json-input",
        default="data/processed/optimization_jobs_params.json",
        help="Path to optimization_jobs_params.json input file.",
    )
    parser.add_argument(
        "--server-json-input",
        default="data/raw/server_params_42servers_v6.json",
        help="Path to server_params_42servers_v6.json input file.",
    )
    parser.add_argument(
        "--output-root",
        default=".",
        help="Root folder for generated outputs "
             "(files are written to <output-root>/outputs/baseline/).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 2.  CONFIGURATION LOADING
#     Derive every parameter the baseline needs from the same two JSON files
#     consumed by data_loader.py / solver.py, so the comparison is
#     apples-to-apples for every robustness-suite case.
# ---------------------------------------------------------------------------
def load_config(jobs_json_path: Path, server_json_path: Path) -> Dict[str, Any]:
    """
    Load jobs_params and server_params JSON files and derive all scheduling,
    power, cooling, and cost parameters needed by the Round-Robin baseline.
    """
    if not jobs_json_path.exists():
        raise FileNotFoundError(f"Jobs JSON file not found: {jobs_json_path}.")
    if not server_json_path.exists():
        raise FileNotFoundError(f"Server JSON file not found: {server_json_path}.")

    with open(jobs_json_path, "r", encoding="utf-8") as f:
        jobs_data = json.load(f)
    with open(server_json_path, "r", encoding="utf-8") as f:
        server_data = json.load(f)

    sets = server_data["sets"]
    J: List[int] = list(sets["J"])
    K: List[int] = list(sets["K"])
    n_j = len(J)
    n_k = len(K)

    sp = server_data["server_params"]
    P0 = {int(j): float(v) for j, v in sp["P0"].items()}
    DP = {int(j): float(v) for j, v in sp["dP"].items()}
    ALPHA = {int(j): float(v) for j, v in sp["alpha"].items()}
    C_J = {int(j): float(v) for j, v in sp["C"].items()}
    THETA = {int(j): float(v) for j, v in sp["theta"].items()}
    LAM0 = {int(j): float(v) for j, v in sp["lambda0"].items()}
    LAM_PM = {int(j): float(v) for j, v in sp["lambda_pm"].items()}
    LAMBDA = {int(j): float(v) for j, v in sp["Lambda"].items()}

    psi_0_raw = sp.get("psi_0", {})
    PSI_0 = {j: float(psi_0_raw.get(str(j), psi_0_raw.get(j, 0.0))) for j in J}

    # --- Cooling COP per slot (eta) ---
    eta_raw = server_data["cooling"]["eta"]
    if isinstance(eta_raw, (int, float)):
        ETA = {k: float(eta_raw) for k in K}
    elif isinstance(eta_raw, list):
        ETA = {k: float(eta_raw[k]) for k in K}
    else:
        ETA = {int(k): float(v) for k, v in eta_raw.items()}

    # --- Power overhead / PUE cap ---
    P_OV = float(server_data["power"]["P_ov"])
    PI_MAX = float(server_data["power"].get("Pi_max", float("inf")))

    # --- Maintenance / cost coefficients ---
    C_PM = float(server_data["maintenance"]["c_pm"])
    C_CM = float(server_data["maintenance"]["c_cm"])
    C_E = {int(k): float(v) for k, v in
           dict(enumerate(server_data["costs"]["c_e"])).items()}
    C_SW = float(server_data["costs"]["c_sw"])

    # --- Redundancy / hot standby ---
    N_MIN = int(server_data["redundancy"]["N_min"])
    KAPPA = int(server_data["redundancy"].get("kappa", 0))
    MIN_ACTIVE = N_MIN + KAPPA

    # --- Slot duration (hours) ---
    DELTA_T = float(server_data["slot_duration"])

    # --- Eligibility-derived server groups ---
    # Identify the smallest eligibility set across all jobs; treat that as
    # the "restricted" server group (e.g. GPU servers), and the remaining
    # servers (or the full set, if every job is eligible everywhere) as the
    # "general" group. This mirrors the GPU/CPU split used by
    # build_jobs_json_from_forecast.py and generate_robustness_tests.py
    # without hardcoding server indices.
    eligibility_raw = jobs_data["eligibility"]
    eligibility = {int(k): list(v) for k, v in eligibility_raw.items()}
    all_servers = set(J)

    restricted_sets = [set(v) for v in eligibility.values() if set(v) != all_servers]
    if restricted_sets:
        # Use the smallest restricted eligibility set (e.g. GPU-only group)
        j_restricted = sorted(min(restricted_sets, key=len))
    else:
        j_restricted = list(J)
    j_general = list(J)  # jobs not in the restricted group are eligible everywhere

    cfg: Dict[str, Any] = {
        "jobs_data": jobs_data,
        "server_data": server_data,
        "J": J,
        "K": K,
        "N_J": n_j,
        "N_SLOTS": n_k,
        "J_RESTRICTED": j_restricted,
        "J_GENERAL": j_general,
        "P0": P0,
        "DP": DP,
        "ALPHA": ALPHA,
        "C_J": C_J,
        "THETA": THETA,
        "LAM0": LAM0,
        "LAM_PM": LAM_PM,
        "LAMBDA": LAMBDA,
        "PSI_0": PSI_0,
        "ETA": ETA,
        "P_OV": P_OV,
        "PI_MAX": PI_MAX,
        "C_PM": C_PM,
        "C_CM": C_CM,
        "C_E": C_E,
        "C_SW": C_SW,
        "N_MIN": N_MIN,
        "KAPPA": KAPPA,
        "MIN_ACTIVE": MIN_ACTIVE,
        "DELTA_T": DELTA_T,
        "RHO_INTER": 5.0,  # fallback lateness penalty for interactive jobs
                            # (interactive jobs have rho=0 in job_params
                            # because the MILP enforces hard deadlines via
                            # valid_starts(); the baseline has no such hard
                            # constraint, so a penalty is applied here for
                            # any interactive job that finishes late).
    }
    return cfg


# ---------------------------------------------------------------------------
# 3.  LOAD JOBS FROM JSON
# ---------------------------------------------------------------------------
def load_jobs(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Build the job DataFrame consumed by round_robin_schedule() directly from
    the jobs_params JSON (sets, eligibility, job_params) -- the same source
    data solver.py uses, so the baseline and MILP operate on an identical
    job set for any robustness-suite case.

    Returns:
        DataFrame sorted by release_slot (arrival order), one row per job.
    """
    jobs_data = cfg["jobs_data"]
    eligibility_raw = jobs_data["eligibility"]
    jp = jobs_data["job_params"]

    I = jobs_data["sets"]["I"]
    I_B = set(jobs_data["sets"]["I_B"])
    I_C = set(jobs_data["sets"]["I_C"])

    d = {int(k): v for k, v in jp["d"].items()}
    r = {int(k): v for k, v in jp["r"].items()}
    a = {int(k): v for k, v in jp["a"].items()}
    b = {int(k): v for k, v in jp["b"].items()}
    q = {int(k): v for k, v in jp["q"].items()}
    rho = {int(k): v for k, v in jp["rho"].items()}
    eligibility = {int(k): list(v) for k, v in eligibility_raw.items()}

    rows = []
    for i in I:
        i = int(i)
        rows.append({
            "job_id": i,
            "job_type": "batch" if i in I_B else "interactive",
            "is_critical": int(i in I_C),
            "release_slot": int(a[i]),
            "duration_slots": int(d[i]),
            "deadline_slot": int(b[i]),
            "replica_count": int(q.get(i, 1)),
            "r": float(r[i]),
            "rho": float(rho.get(i, 0.0)),
            "eligible_servers": eligibility.get(i, list(cfg["J"])),
        })

    fc = pd.DataFrame(rows)

    # Drop jobs that cannot complete within the horizon (defensive; the
    # source jobs JSON should already satisfy this).
    n_slots = cfg["N_SLOTS"]
    fits = (fc["release_slot"] + fc["duration_slots"]) <= n_slots
    n_dropped = (~fits).sum()
    if n_dropped:
        print(f"  Dropping {n_dropped} jobs that cannot complete within the "
              f"{n_slots}-slot horizon.")
    fc = fc[fits].reset_index(drop=True)

    return fc.sort_values("release_slot").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4.  ROUND-ROBIN SCHEDULER
# ---------------------------------------------------------------------------
def round_robin_schedule(jobs: pd.DataFrame, cfg: Dict[str, Any]):
    """
    Assign all jobs to servers using the Round-Robin policy.

    Algorithm:
      1. Jobs are processed in arrival order (sorted by release_slot).
      2. For each job, its eligibility group (from eligibility[i] in the
         jobs JSON) is iterated starting from that group's Round-Robin
         pointer.
      3. For each candidate server, find the earliest feasible start slot
         >= its release slot where the server has enough free capacity.
      4. Assign to the first server with a feasible slot.
      5. Jobs requiring multiple replicas repeat this process, placing each
         replica on a DIFFERENT server for fault isolation.
      6. Fallback: if no feasible slot is found (capacity fully booked),
         the job is placed at its release slot on the next server in
         rotation. This ensures all jobs are scheduled even if capacity is
         exceeded.

    Key difference from the MILP:
      Round-Robin makes no forward-looking decisions. It does not consider
      energy prices, thermal load, PUE, or future demand. Each assignment is
      greedy and irreversible.

    Args:
        jobs: DataFrame of jobs sorted by release_slot.
        cfg:  Config dict from load_config().

    Returns:
        schedule_df: DataFrame with one row per placement (job x replica).
        load:        2D numpy array [N_J x N_SLOTS] of load per server-slot.
    """
    N_J = cfg["N_J"]
    N_SLOTS = cfg["N_SLOTS"]
    J = cfg["J"]
    C_J = cfg["C_J"]
    THETA = cfg["THETA"]
    RHO_INTER = cfg["RHO_INTER"]

    # Map server id -> matrix row index (J may not be a contiguous 0..N-1
    # range in arbitrary server JSONs, though in practice it is).
    j_index = {j: idx for idx, j in enumerate(J)}

    load = np.zeros((N_J, N_SLOTS))
    batch_load = np.zeros((N_J, N_SLOTS))

    schedule = []

    rr_ptr: Dict[tuple, int] = {}

    def get_rr_key(servers):
        return tuple(sorted(servers))

    def find_start_slot(j, release_slot, dur, r_i, is_batch):
        cap_limit = (1 - THETA[j]) * C_J[j] if is_batch else C_J[j]
        jx = j_index[j]
        for k in range(release_slot, N_SLOTS - dur + 1):
            feasible = True
            for kk in range(k, k + dur):
                if load[jx][kk] + r_i > C_J[j] + 1e-6:
                    feasible = False
                    break
                if is_batch and batch_load[jx][kk] + r_i > cap_limit + 1e-6:
                    feasible = False
                    break
            if feasible:
                return k
        return None

    def assign_job(server, start_slot, dur, r_i, is_batch):
        jx = j_index[server]
        for kk in range(start_slot, start_slot + dur):
            load[jx][kk] += r_i
            if is_batch:
                batch_load[jx][kk] += r_i

    for _, job in jobs.iterrows():
        i = int(job["job_id"])
        r_i = float(job["r"])
        release = int(job["release_slot"])
        dur = int(job["duration_slots"])
        deadline = int(job["deadline_slot"])
        replicas = int(job["replica_count"])
        is_batch = job["job_type"] == "batch"
        eligible = list(job["eligible_servers"])
        # Lateness penalty: use rho from job_params for batch jobs (matches
        # the MILP's rho[i] * l_var[i] term); interactive jobs use a fallback
        # penalty since the MILP enforces their deadlines as hard constraints
        # (rho=0 in job_params for interactive jobs).
        rho = float(job["rho"]) if is_batch else RHO_INTER
        rr_key = get_rr_key(eligible)

        dur = min(dur, N_SLOTS - release)
        if dur < 1:
            continue

        replicas_placed = 0
        used_servers: List[int] = []

        for _ in range(replicas):
            placed = False

            for _ in range(len(eligible)):
                ptr = rr_ptr.get(rr_key, 0)
                j = eligible[ptr % len(eligible)]
                rr_ptr[rr_key] = ptr + 1

                if j in used_servers:
                    continue

                start = find_start_slot(j, release, dur, r_i, is_batch)

                if start is not None:
                    assign_job(j, start, dur, r_i, is_batch)
                    lateness = max(0, start + dur - deadline)
                    schedule.append({
                        "job_id": i,
                        "job_type": job["job_type"],
                        "is_critical": int(job["is_critical"]),
                        "replica": replicas_placed + 1,
                        "server": j,
                        "start_slot": start,
                        "end_slot": start + dur,
                        "duration_slots": dur,
                        "r": r_i,
                        "release_slot": release,
                        "deadline_slot": deadline,
                        "lateness_slots": lateness,
                        "lateness_cost": rho * lateness,
                    })
                    used_servers.append(j)
                    replicas_placed += 1
                    placed = True
                    break

            if not placed:
                j = eligible[rr_ptr.get(rr_key, 0) % len(eligible)]
                rr_ptr[rr_key] = rr_ptr.get(rr_key, 0) + 1
                start = release
                assign_job(j, start, dur, r_i, is_batch)
                lateness = max(0, start + dur - deadline)
                schedule.append({
                    "job_id": i,
                    "job_type": job["job_type"],
                    "is_critical": int(job["is_critical"]),
                    "replica": replicas_placed + 1,
                    "server": j,
                    "start_slot": start,
                    "end_slot": start + dur,
                    "duration_slots": dur,
                    "r": r_i,
                    "release_slot": release,
                    "deadline_slot": deadline,
                    "lateness_slots": lateness,
                    "lateness_cost": rho * lateness,
                })
                used_servers.append(j)
                replicas_placed += 1

    return pd.DataFrame(schedule), load


# ---------------------------------------------------------------------------
# 5.  SLOT-LEVEL ENERGY AND THERMAL METRICS
# ---------------------------------------------------------------------------
def compute_slot_metrics(load: np.ndarray, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute energy, PUE, and utilization for each time slot.

    For each slot k, active servers are those with load[j][k] > 0.
    A minimum of MIN_ACTIVE = N_min + kappa servers is always kept active
    (hot standby, matching constraint #34 in the MILP), supplementing with
    idle servers from the restricted (e.g. GPU) group if needed.

    Energy model (same formulas as the MILP objective):
      IT power (PIT):  sum_j( P0[j] + dP[j] * L[j,k] )   for active servers
      Heat per server: H[j,k] = alpha[j] * (P0[j] + dP[j] * L[j,k])
      Cooling power:   Pcool[k] = sum_j(H[j,k]) / eta[k]
      Total power:     Ptot[k] = PIT[k] + Pcool[k] + P_ov
      PUE:             Pi[k] = Ptot[k] / PIT[k]

    Returns:
        DataFrame with one row per slot containing all energy metrics.
    """
    J = cfg["J"]
    K = cfg["K"]
    N_SLOTS = cfg["N_SLOTS"]
    P0, DP, ALPHA = cfg["P0"], cfg["DP"], cfg["ALPHA"]
    ETA, P_OV, DELTA_T = cfg["ETA"], cfg["P_OV"], cfg["DELTA_T"]
    C_E = cfg["C_E"]
    MIN_ACTIVE = cfg["MIN_ACTIVE"]
    J_RESTRICTED = cfg["J_RESTRICTED"]

    j_index = {j: idx for idx, j in enumerate(J)}

    rows = []
    for k_pos, k in enumerate(K):
        active_j = [j for j in J if load[j_index[j]][k_pos] > 1e-6]

        standby_pool = [j for j in J_RESTRICTED if j not in active_j]
        i_s = 0
        while len(active_j) < MIN_ACTIVE and i_s < len(standby_pool):
            active_j.append(standby_pool[i_s])
            i_s += 1

        PIT = sum(P0[j] + DP[j] * float(load[j_index[j]][k_pos]) for j in active_j)
        H_tot = sum(
            ALPHA[j] * (P0[j] + DP[j] * float(load[j_index[j]][k_pos]))
            for j in active_j)

        eta_k = ETA[k]
        Pcool = H_tot / eta_k if eta_k > 0 else float("nan")
        Ptot = PIT + Pcool + P_OV
        PUE = Ptot / PIT if PIT > 0 else float("nan")

        n_active = len(active_j)
        avg_util = (
            sum(float(load[j_index[j]][k_pos]) for j in active_j) / n_active
            if n_active > 0 else 0.0)

        energy_kwh = Ptot * DELTA_T / 1000.0
        c_e_k = C_E.get(k, C_E.get(k_pos, 0.0))
        energy_cost = c_e_k * energy_kwh

        rows.append({
            "slot": k,
            "time": _slot_to_time(k, DELTA_T),
            "n_active": n_active,
            "avg_util": round(avg_util, 4),
            "PIT_W": round(PIT, 2),
            "Pcool_W": round(Pcool, 2),
            "Ptot_W": round(Ptot, 2),
            "PUE": round(PUE, 4) if not math.isnan(PUE) else None,
            "COP": eta_k,
            "energy_kWh": round(energy_kwh, 4),
            "energy_cost": round(energy_cost, 4),
        })

    return pd.DataFrame(rows)


def _slot_to_time(k: int, delta_t: float) -> str:
    """Convert a slot index into HH:MM based on the slot duration."""
    mins = int(round(k * delta_t * 60))
    return f"{mins // 60:02d}:{mins % 60:02d}"


# ---------------------------------------------------------------------------
# 6.  SERVER SWITCHING COST
# ---------------------------------------------------------------------------
def compute_switching_cost(load: np.ndarray, cfg: Dict[str, Any]):
    """
    Count server on/off state transitions and compute the switching cost.

    A transition occurs when a server changes between active (load > 0)
    and idle (load = 0) between consecutive slots. In Round-Robin, servers
    activate and deactivate reactively as jobs arrive and complete, with no
    attempt to consolidate workload or stabilize server states. This results
    in more frequent switching than the MILP optimizer, which explicitly
    minimizes switching through the c_sw penalty term.

    Returns:
        (total_cost, n_switches): monetary switching cost and raw count.
    """
    N_J, N_SLOTS = cfg["N_J"], cfg["N_SLOTS"]
    C_SW = cfg["C_SW"]
    switches = 0
    for jx in range(N_J):
        was_active = False
        for k in range(N_SLOTS):
            is_active = load[jx][k] > 1e-6
            if is_active != was_active:
                switches += 1
            was_active = is_active
    return C_SW * switches, switches


# ---------------------------------------------------------------------------
# 7.  AGGREGATE SUMMARY METRICS
# ---------------------------------------------------------------------------
def build_summary(schedule_df: pd.DataFrame, slot_df: pd.DataFrame,
                  load: np.ndarray, cfg: Dict[str, Any]) -> dict:
    """
    Aggregate all metrics into a single-row summary for comparison against
    the MILP's performance_metrics CSV.

    Total cost follows the same five-component structure as the MILP
    objective function (equation #4 in the model):
      Total cost = Energy cost
                 + Corrective maintenance cost  (no PM scheduled in baseline)
                 + Server switching cost
                 + Batch lateness penalty
                 (Preventive maintenance cost = 0 in baseline)

    Note: Because Round-Robin does not schedule any preventive maintenance,
    servers run at their full base failure rate lambda0 throughout the
    horizon. This results in a higher expected corrective maintenance cost
    compared to the MILP, which can reduce failure rates by scheduling PM
    windows.

    Returns:
        Dictionary with all aggregate performance and cost metrics.
    """
    J = cfg["J"]
    N_SLOTS = cfg["N_SLOTS"]
    LAM0, LAMBDA, PSI_0, LAM_PM = cfg["LAM0"], cfg["LAMBDA"], cfg["PSI_0"], cfg["LAM_PM"]
    C_CM = cfg["C_CM"]
    j_index = {j: idx for idx, j in enumerate(J)}

    sw_cost, n_switches = compute_switching_cost(load, cfg)

    total_energy_kwh = slot_df["energy_kWh"].sum()
    total_energy_cost = slot_df["energy_cost"].sum()
    avg_util = slot_df["avg_util"].mean()
    avg_pue = slot_df["PUE"].dropna().mean()
    total_late_cost = schedule_df["lateness_cost"].sum()
    jobs_placed = schedule_df["job_id"].nunique()

    jobs_on_time = (
        schedule_df.groupby("job_id")["lateness_slots"].max() == 0).sum()

    # Wear-based CM cost -- mirrors the solver's objective term exactly:
    #   cm_cost = c_cm * sum_{j,k} lambda0[j] * (psi[j,k] / Lambda[j]) * active[j,k]
    #
    # In the baseline no PM is ever scheduled, so psi[j,k] grows from psi_0[j]
    # linearly by load[j][k] each slot -- the worst-case wear trajectory.
    cm_cost = 0.0
    psi_end = {}
    for j in J:
        jx = j_index[j]
        psi_jk = PSI_0[j]
        for k in range(N_SLOTS):
            psi_jk += float(load[jx][k])
            active = 1 if load[jx][k] > 1e-6 else 0
            cm_cost += C_CM * (LAM_PM[j] + LAM0[j] * (psi_jk / LAMBDA[j])) * active
        psi_end[j] = round(psi_jk, 6)

    total_cost = total_energy_cost + cm_cost + sw_cost + total_late_cost

    return {
        "policy": "Round-Robin",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "jobs_in_forecast": len(schedule_df["job_id"].unique()),
        "jobs_placed": jobs_placed,
        "jobs_on_time": int(jobs_on_time),
        "total_energy_kWh": round(total_energy_kwh, 3),
        "total_energy_cost_$": round(total_energy_cost, 4),
        "avg_server_util": round(avg_util, 4),
        "avg_PUE": round(avg_pue, 4) if not math.isnan(avg_pue) else None,
        "switching_cost_$": round(sw_cost, 4),
        "n_switches": n_switches,
        "corrective_maint_$": round(cm_cost, 4),
        "preventive_maint_$": 0.0,
        "lateness_cost_$": round(total_late_cost, 4),
        "total_cost_$": round(total_cost, 4),
        "avg_psi_end": round(sum(psi_end.values()) / len(psi_end), 4),
        "max_psi_end": round(max(psi_end.values()), 4),
    }


# ---------------------------------------------------------------------------
# 8.  MAIN ENTRY POINT
# ---------------------------------------------------------------------------
def main():
    """
    Run the full Round-Robin baseline pipeline:
      1. Load jobs/server JSON inputs (same pair the MILP pipeline uses)
      2. Assign all jobs using Round-Robin scheduling
      3. Compute per-slot energy, PUE, and utilization metrics
      4. Build aggregate summary for comparison with the MILP results
      5. Save all four output CSV files to <output-root>/outputs/baseline/
    """
    args = parse_args()

    jobs_json_path = Path(args.jobs_json_input)
    server_json_path = Path(args.server_json_input)
    out_dir = Path(args.output_root) / "outputs" / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  Round-Robin Baseline Scheduler")
    print("=" * 55)
    print(f"Jobs JSON:   {jobs_json_path}")
    print(f"Server JSON: {server_json_path}")

    # Step 1: Load config and jobs
    print("\nLoading configuration and jobs...")
    cfg = load_config(jobs_json_path, server_json_path)
    jobs = load_jobs(cfg)
    print(f"  {len(jobs)} jobs loaded  "
          f"({(jobs.job_type == 'batch').sum()} batch, "
          f"{(jobs.job_type == 'interactive').sum()} interactive, "
          f"{jobs.is_critical.sum()} critical)")
    print(f"  {cfg['N_J']} servers, {cfg['N_SLOTS']} slots")

    # Step 2: Run Round-Robin assignment and get load matrix
    print("\nRunning Round-Robin assignment...")
    schedule_df, load = round_robin_schedule(jobs, cfg)
    print(f"  {len(schedule_df)} placements made "
          f"across {cfg['N_J']} servers, {cfg['N_SLOTS']} slots")

    # Step 3: Compute per-slot energy, PUE, and utilization
    print("\nComputing slot-level metrics...")
    slot_df = compute_slot_metrics(load, cfg)

    # Step 4: Build aggregate summary for comparison
    summary = build_summary(schedule_df, slot_df, load, cfg)

    # Step 5: Save all outputs
    schedule_df.to_csv(out_dir / "rr_job_schedule.csv", index=False)
    slot_df.to_csv(out_dir / "rr_slot_metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "rr_summary.csv", index=False)

    # Per-server utilization breakdown
    J = cfg["J"]
    j_index = {j: idx for idx, j in enumerate(J)}
    server_util = []
    for j in J:
        jx = j_index[j]
        slots_active = sum(1 for k in range(cfg["N_SLOTS"]) if load[jx][k] > 1e-6)
        server_util.append({
            "server": j,
            "slots_active": slots_active,
            "avg_load": round(float(np.mean(load[jx])), 4),
            "max_load": round(float(np.max(load[jx])), 4),
        })
    pd.DataFrame(server_util).to_csv(
        out_dir / "rr_server_utilization.csv", index=False)

    # Print summary to console
    print("\n" + "=" * 55)
    print("  RESULTS SUMMARY")
    print("=" * 55)
    for k, v in summary.items():
        if k not in ("policy", "timestamp"):
            print(f"  {k:<30} {v}")

    print(f"\nOutputs saved to: {out_dir}/")
    print("  rr_job_schedule.csv       - Full job-level schedule")
    print("  rr_slot_metrics.csv       - Per-slot energy and PUE")
    print("  rr_server_utilization.csv - Per-server load summary")
    print("  rr_summary.csv            - Aggregate metrics for comparison")


if __name__ == "__main__":
    main()
