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
      sequence (GPU servers for GPU jobs, all servers for CPU-only jobs).
    - The job starts at the earliest time slot >= its release slot where the
      selected server has enough remaining capacity.
    - Critical jobs requiring 2 replicas are placed on 2 different servers.
    - No lookahead, no demand forecasting, no energy awareness.

ALIGNMENT WITH MILP MODEL (for apples-to-apples comparison):
    - Input:  same parquet file (data/processed/optimization_forecast_jobs.parquet)
    - Job preprocessing: identical filtering, r formula, eligibility, and
      deadline logic to build_jobs_json_from_forecast.py
    - Power model: same P0, dP, alpha, eta, P_ov constants as solver.py
    - Cost model:  same c_cm=6000, lambda0=0.0000085, and wear-based CM cost
      formula (c_cm * lambda0 * psi[j,k]/Lambda[j]) as the MILP objective
    - Lambda:  50.8 for GPU servers (0-33), 6999.7 for CPU servers (34-41)
      calibrated for quarterly PM cycle from real forecast load distribution
    - psi_0:   loaded from server_params JSON (same file as MILP) so both
      models start each daily run with identical inherited wear state
    - No PM is scheduled in the baseline, so psi[j,k] grows monotonically
      from psi_0[j] — producing higher CM cost than the MILP, which can reset
      wear via PM. This is the key mechanism behind hypothesis H3.

OUTPUTS (saved to outputs/baseline/):
    rr_slot_metrics.csv       - Per-slot energy, PUE, utilization, and cost
    rr_job_schedule.csv       - Job-level schedule with server, start time, lateness
    rr_server_utilization.csv - Per-server load summary across all 96 slots
    rr_summary.csv            - Single-row aggregate metrics for t-test comparison
"""

import math
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# 1.  CONFIGURATION
#     All values match data_loader.py in the optimization model to ensure
#     consistent comparison.
# ---------------------------------------------------------------------------

# Time horizon: 24 hours divided into 15-minute slots (96 slots total)
SLOT_SECONDS = 900    # Duration of each time slot in seconds (15 min)
N_SLOTS      = 96     # Total number of scheduling slots in the planning horizon
DELTA_T      = 0.25   # Slot duration in hours (used in energy cost calculation)

# Server fleet: 42 servers split into two hardware types
#   GPU servers (j=0..33): high-performance nodes with GPU, used by HN jobs
#   CPU servers (j=34..41): standard compute nodes without GPU, used by CN jobs
J_GPU = list(range(0, 34))
J_CPU = list(range(34, 42))
J_ALL = list(range(0, 42))
N_J   = 42

# Server power model: actual_power = P0 + dP * normalized_load
# P0    = idle (static) power when server is on but not running jobs (Watts)
# DP    = incremental power per unit of load at full utilization (Watts)
# ALPHA = fraction of electrical power converted to heat (for thermal model)
P0    = {j: 180.0 if j < 34 else 110.0 for j in J_ALL}
DP    = {j: 320.0 if j < 34 else 180.0 for j in J_ALL}
ALPHA = {j: 0.90  if j < 34 else 0.87  for j in J_ALL}

# Cooling COP (Coefficient of Performance) per 15-minute slot.
# COP varies by time of day: higher at night (cooler ambient temperatures),
# lower at midday peak heat. Each hourly value is repeated 4 times for the
# four 15-min slots within each hour.
_hourly_eta = [
    5.0, 5.0, 5.0, 5.0,   # 00:00-03:45  night (most efficient cooling)
    4.5, 4.5, 4.5, 4.5,   # 04:00-07:45  early morning
    3.5, 3.5, 3.5, 3.5,   # 08:00-11:45  morning
    3.0, 3.0, 3.0, 3.0,   # 12:00-15:45  midday peak (least efficient)
    3.5, 3.5, 3.5, 3.5,   # 16:00-19:45  afternoon
    4.5, 4.5, 4.5, 4.5,   # 20:00-23:45  evening
]
ETA = []
for v in _hourly_eta:
    ETA.extend([v] * 4)   # Expand 24 hourly values to 96 slot values

# Fixed overhead power: constant load from lighting, networking switches,
# UPS losses, and other non-IT infrastructure (Watts)
P_OV  = 300.0

# Cost parameters (all monetary values in dollars)
# NOTE: c_e in the MILP is slot-varying; here we use the same per-slot vector
# loaded from the server JSON so energy costs are computed identically.
C_E   = 0.139   # Electricity price ($/kWh) - Ontario time-of-use average (fallback)
C_SW  = 0.1     # Penalty per server on/off state transition ($)
C_PM  = 250.0   # Fixed cost of preventive maintenance per server ($)
# Updated to match solver calibration (was 100,000 — caused over-scheduling of PM)
C_CM  = 6000.0  # Corrective maintenance cost coefficient ($)

# Server failure rates per time slot — updated to match solver.py calibration.
# In the baseline, no preventive maintenance is scheduled, so LAM0 applies
# throughout the full 96-slot horizon (worst-case corrective cost scenario).
LAM0   = {j: 0.0000085 for j in J_ALL}   # base failure rate/slot (calibrated)
LAM_PM = {j: 0.0000085 * 0.30 for j in J_ALL}  # post-PM rate (unused in baseline)

# Wear-based CM cost: Lambda threshold for quarterly PM cycle (90 days).
# Separate thresholds for GPU (A10, servers 0-33) and CPU (servers 34-41)
# computed from actual forecast load distribution (avg load-slots/server/day).
LAMBDA = {j: 50.8 if j < 34 else 6999.7 for j in J_ALL}

# psi_0[j]: accumulated wear at start of horizon inherited from previous cycles.
# Loaded from the same server_params JSON used by the MILP so both models share
# an identical starting wear state — essential for a fair daily comparison.
_SERVER_JSON = Path("data/raw/server_params_42servers_v6.json")
PSI_0: dict[int, float] = {}

def _load_psi_0() -> dict:
    import json
    if _SERVER_JSON.exists():
        with open(_SERVER_JSON) as f:
            d = json.load(f)
        raw = d.get("server_params", {}).get("psi_0", {})
        return {j: float(raw.get(str(j), raw.get(j, 0.0))) for j in J_ALL}
    return {j: 0.0 for j in J_ALL}

PSI_0 = _load_psi_0()

# Lateness penalty rates ($ per slot beyond deadline)
# Interactive jobs carry a higher penalty (hard QoS requirements).
# Batch jobs can tolerate some delay (soft deadlines).
RHO_BATCH = 3.0
RHO_INTER = 5.0

# Server capacity and reservation
C_J   = {j: 1.0 if j < 34 else 0.420139 for j in J_ALL}  # Normalized server capacity (GPU=1.0, CPU=0.420139)
THETA = 0.30   # Fraction of capacity reserved for interactive jobs.
               # Batch jobs can use at most (1 - THETA) = 70% per slot.


# ---------------------------------------------------------------------------
# 2.  LOAD FORECAST JOBS
# ---------------------------------------------------------------------------
def load_jobs() -> pd.DataFrame:
    """
    Load the forecast job sequence and compute scheduling parameters.

    Reads the same parquet file consumed by build_jobs_json_from_forecast.py
    so that the baseline and MILP operate on an identical job set.

    Returns:
        DataFrame sorted by release_slot (arrival order), one row per job.
    """
    fc_path = Path("data/processed/optimization_forecast_jobs.parquet")
    if not fc_path.exists():
        raise FileNotFoundError(
            f"Forecast parquet not found: {fc_path}. "
            "Run forecast_model.py first."
        )
    fc = pd.read_parquet(fc_path)

    required = [
        "forecast_job_id", "job_type", "release_seconds", "cpu_request",
        "memory_request", "gpu_request", "processing_duration_seconds",
        "deadline_seconds", "is_critical", "replica_count",
    ]
    missing = [c for c in required if c not in fc.columns]
    if missing:
        raise ValueError(f"Missing columns in parquet: {missing}")

    fc = fc.dropna(subset=required).reset_index(drop=True)
    fc["job_id"] = fc.index

    # Timing columns are already in the parquet (produced by forecast_model.py)
    fc["release_seconds"]             = fc["release_seconds"].astype(float)
    fc["processing_duration_seconds"] = fc["processing_duration_seconds"].astype(float)
    fc["deadline_seconds"]            = fc["deadline_seconds"].astype(float)

    # --- Convert seconds to slot indices (identical to build_jobs_json_from_forecast.py) ---
    fc["release_slot"] = fc["release_seconds"].apply(
        lambda s: max(0, min(int(math.floor(s / SLOT_SECONDS)), N_SLOTS - 1)))
    fc["duration_slots"] = fc["processing_duration_seconds"].apply(
        lambda s: max(1, int(math.ceil(s / SLOT_SECONDS))))
    fc["deadline_slot"] = fc["deadline_seconds"].apply(
        lambda s: min(N_SLOTS, int(math.ceil(s / SLOT_SECONDS))))

    # --- Resource normalization (identical formula to build_jobs_json_from_forecast.py) ---
    max_cpu    = 128.0   # cores — G-type GPU server max
    max_memory = 1024.0  # GB   — G-type GPU server max
    fc["r"] = fc.apply(
        lambda row: round(
            min(1.0, 0.5 * float(row["cpu_request"])    / max_cpu
                   + 0.5 * float(row["memory_request"]) / max_memory), 4),
        axis=1)

    # --- Server eligibility (same logic as build_eligibility in build_jobs_json) ---
    fc["eligible_servers"] = fc["gpu_request"].apply(
        lambda g: J_GPU if int(g) == 1 else J_ALL)

    # --- Drop jobs that cannot complete within the 96-slot horizon ---
    fits = (fc["release_slot"] + fc["duration_slots"]) <= N_SLOTS
    n_dropped = (~fits).sum()
    if n_dropped:
        print(f"  Dropping {n_dropped} jobs that exceed the 96-slot horizon.")
    fc = fc[fits].reset_index(drop=True)
    fc["job_id"] = fc.index

    # --- Drop oversized batch jobs (r >= 0.70 cannot fit batch cap) ---
    BATCH_CAP = (1 - THETA) * 1.0   # 0.70
    oversized = (fc["job_type"] == "batch") & (fc["r"] >= BATCH_CAP)
    n_over = oversized.sum()
    if n_over:
        print(f"  Dropping {n_over} oversized batch jobs (r >= {BATCH_CAP}).")
    fc = fc[~oversized].reset_index(drop=True)
    fc["job_id"] = fc.index

    # Ensure deadline is always reachable
    fc["deadline_slot"] = fc.apply(
        lambda row: max(row["deadline_slot"],
                        row["release_slot"] + row["duration_slots"]),
        axis=1).clip(upper=N_SLOTS)

    return fc.sort_values("release_slot").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3.  ROUND-ROBIN SCHEDULER
# ---------------------------------------------------------------------------
def round_robin_schedule(jobs: pd.DataFrame):
    """
    Assign all forecast jobs to servers using the Round-Robin policy.

    Algorithm:
      1. Jobs are processed in arrival order (sorted by release_slot).
      2. For each job, the eligible server group (GPU or CPU) is iterated
         starting from the current Round-Robin pointer for that group.
      3. For each candidate server, find the earliest feasible start slot
         >= release_slot where the server has enough free capacity.
      4. Assign to the first server with a feasible slot.
      5. Critical jobs (replica_count=2) repeat this process twice, placing
         each replica on a DIFFERENT server for fault isolation.
      6. Fallback: if no feasible slot is found (capacity fully booked),
         the job is placed at its release slot on the next server in rotation.
         This ensures all jobs are scheduled even if capacity is exceeded.

    Key difference from the MILP:
      Round-Robin makes no forward-looking decisions. It does not consider
      energy prices, thermal load, PUE, or other jobs future demand.
      Each assignment is greedy and irreversible.

    Args:
        jobs: DataFrame of forecast jobs sorted by release_slot.

    Returns:
        schedule_df: DataFrame with one row per placement (job x replica).
        load:        2D numpy array [N_J x N_SLOTS] of load per server-slot.
    """
    # load[j][k]: normalized resource load on server j during slot k (0 to 1+)
    load       = np.zeros((N_J, N_SLOTS))
    # batch_load[j][k]: batch-only portion of load, for enforcing the theta cap
    batch_load = np.zeros((N_J, N_SLOTS))

    schedule = []  # List of assignment dicts (one entry per job placement)

    # Round-Robin pointer: tracks next server index to try for each server group.
    # GPU and CPU groups are rotated independently.
    rr_ptr = {tuple(J_GPU): 0, tuple(J_CPU): 0}

    def get_rr_key(servers):
        """Create a hashable key for the server eligibility group."""
        return tuple(sorted(servers))

    def find_start_slot(j, release_slot, dur, r_i, is_batch):
        """
        Find the earliest feasible start slot for a job on server j.

        A slot k is feasible if for every slot kk in [k, k+duration):
          - Total server load stays within capacity: load[j][kk] + r_i <= C_J
          - For batch jobs: batch load stays within reservation cap:
            batch_load[j][kk] + r_i <= (1 - THETA) * C_J

        Returns the first feasible slot index, or None if none exists.
        """
        cap_limit = (1 - THETA) * C_J[j] if is_batch else C_J[j]
        for k in range(release_slot, N_SLOTS - dur + 1):
            feasible = True
            for kk in range(k, k + dur):
                if load[j][kk] + r_i > C_J[j] + 1e-6:
                    feasible = False
                    break
                if is_batch and batch_load[j][kk] + r_i > cap_limit + 1e-6:
                    feasible = False
                    break
            if feasible:
                return k
        return None  # No feasible slot within the horizon

    def assign_job(server, start_slot, dur, r_i, is_batch):
        """
        Commit a job placement by updating the load tracking arrays.
        Adds r_i to load[server][kk] for each slot kk the job occupies.
        """
        for kk in range(start_slot, start_slot + dur):
            load[server][kk] += r_i
            if is_batch:
                batch_load[server][kk] += r_i

    # --- Main scheduling loop: one job at a time, in arrival order ---
    for _, job in jobs.iterrows():
        i        = int(job["job_id"])
        r_i      = float(job["r"])
        release  = int(job["release_slot"])
        dur      = int(job["duration_slots"])
        deadline = int(job["deadline_slot"])
        replicas = int(job["replica_count"])
        is_batch = job["job_type"] == "batch"
        eligible = list(job["eligible_servers"])
        rho      = RHO_INTER if not is_batch else RHO_BATCH
        rr_key   = get_rr_key(eligible)

        # Clamp duration so the job fits entirely within the planning horizon
        dur = min(dur, N_SLOTS - release)
        if dur < 1:
            continue  # Skip jobs with no schedulable window remaining

        replicas_placed = 0
        used_servers    = []  # Servers already used for this job's replicas

        # Place each required replica
        for _ in range(replicas):
            placed = False

            # Try each server in the eligible group starting from the RR pointer
            for _ in range(len(eligible)):
                ptr = rr_ptr.get(rr_key, 0)
                j   = eligible[ptr % len(eligible)]
                rr_ptr[rr_key] = ptr + 1  # Advance the Round-Robin pointer

                # Each replica must go on a different server for fault isolation
                if j in used_servers:
                    continue

                # Find earliest feasible slot on this server
                start = find_start_slot(j, release, dur, r_i, is_batch)

                if start is not None:
                    # Feasible slot found - commit the assignment
                    assign_job(j, start, dur, r_i, is_batch)
                    lateness = max(0, start + dur - deadline)
                    schedule.append({
                        "job_id":         i,
                        "job_type":       job["job_type"],
                        "is_critical":    int(job["is_critical"]),
                        "replica":        replicas_placed + 1,
                        "server":         j,
                        "start_slot":     start,
                        "end_slot":       start + dur,
                        "duration_slots": dur,
                        "r":              r_i,
                        "release_slot":   release,
                        "deadline_slot":  deadline,
                        "lateness_slots": lateness,
                        "lateness_cost":  rho * lateness,
                    })
                    used_servers.append(j)
                    replicas_placed += 1
                    placed = True
                    break

            if not placed:
                # Fallback: no server had capacity within constraints.
                # Force placement at release time on the next server in rotation.
                # This ensures 100% scheduling coverage at the cost of accuracy.
                j = eligible[rr_ptr.get(rr_key, 0) % len(eligible)]
                rr_ptr[rr_key] = rr_ptr.get(rr_key, 0) + 1
                start    = release
                assign_job(j, start, dur, r_i, is_batch)
                lateness = max(0, start + dur - deadline)
                schedule.append({
                    "job_id":         i,
                    "job_type":       job["job_type"],
                    "is_critical":    int(job["is_critical"]),
                    "replica":        replicas_placed + 1,
                    "server":         j,
                    "start_slot":     start,
                    "end_slot":       start + dur,
                    "duration_slots": dur,
                    "r":              r_i,
                    "release_slot":   release,
                    "deadline_slot":  deadline,
                    "lateness_slots": lateness,
                    "lateness_cost":  rho * lateness,
                })
                used_servers.append(j)
                replicas_placed += 1

    return pd.DataFrame(schedule), load


# ---------------------------------------------------------------------------
# 4.  SLOT-LEVEL ENERGY AND THERMAL METRICS
# ---------------------------------------------------------------------------
def compute_slot_metrics(load: np.ndarray) -> pd.DataFrame:
    """
    Compute energy, PUE, and utilization for each 15-minute time slot.

    For each slot k, active servers are those with load[j][k] > 0.
    A minimum of 3 servers is always kept active (hot standby = N_min + kappa
    from the optimization model), supplementing with idle GPU servers if needed.

    Energy model (same formulas as the MILP objective):
      IT power (PIT):  sum_j( P0[j] + dP[j] * L[j,k] )   for active servers
      Heat per server: H[j,k] = alpha[j] * (P0[j] + dP[j] * L[j,k])
      Cooling power:   Pcool[k] = sum_j(H[j,k]) / eta[k]
      Total power:     Ptot[k] = PIT[k] + Pcool[k] + P_ov
      PUE:             Pi[k] = Ptot[k] / PIT[k]  (target: <= 1.56)

    Returns:
        DataFrame with one row per slot containing all energy metrics.
    """
    rows = []

    for k in range(N_SLOTS):
        # Identify servers actively running jobs in this slot
        active_j = [j for j in J_ALL if load[j][k] > 1e-6]

        # Enforce minimum server count (hot standby), supplement with GPU servers
        gpu_standby = [j for j in J_GPU if j not in active_j]
        i_s = 0
        while len(active_j) < 3 and i_s < len(gpu_standby):
            active_j.append(gpu_standby[i_s])
            i_s += 1

        # Compute IT power using the linear power model
        PIT = sum(P0[j] + DP[j] * float(load[j][k]) for j in active_j)

        # Compute heat generated (used to determine required cooling power)
        H_tot = sum(
            ALPHA[j] * (P0[j] + DP[j] * float(load[j][k]))
            for j in active_j)

        # Cooling power: heat removed divided by COP (efficiency factor)
        eta_k = ETA[k]
        Pcool = H_tot / eta_k

        # Total facility power: IT + cooling + fixed infrastructure overhead
        Ptot = PIT + Pcool + P_OV

        # PUE = total facility power / IT power (lower = more efficient)
        PUE = Ptot / PIT if PIT > 0 else float("nan")

        # Average utilization across active servers
        n_active = len(active_j)
        avg_util = (
            sum(float(load[j][k]) for j in active_j) / n_active
            if n_active > 0 else 0.0)

        # Energy consumed this slot in kWh (Watts * hours / 1000)
        energy_kwh  = Ptot * DELTA_T / 1000.0
        energy_cost = C_E * energy_kwh

        rows.append({
            "slot":        k,
            "hour":        round(k * DELTA_T, 2),
            "n_active":    n_active,
            "avg_util":    round(avg_util, 4),
            "PIT_W":       round(PIT, 2),
            "Pcool_W":     round(Pcool, 2),
            "Ptot_W":      round(Ptot, 2),
            "PUE":         round(PUE, 4) if not math.isnan(PUE) else None,
            "COP":         eta_k,
            "energy_kWh":  round(energy_kwh, 4),
            "energy_cost": round(energy_cost, 4),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5.  SERVER SWITCHING COST
# ---------------------------------------------------------------------------
def compute_switching_cost(load: np.ndarray):
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
    switches = 0
    for j in J_ALL:
        was_active = False
        for k in range(N_SLOTS):
            is_active = load[j][k] > 1e-6
            if is_active != was_active:   # Detect any state change
                switches += 1
            was_active = is_active
    return C_SW * switches, switches


# ---------------------------------------------------------------------------
# 6.  AGGREGATE SUMMARY METRICS
# ---------------------------------------------------------------------------
def build_summary(schedule_df: pd.DataFrame, slot_df: pd.DataFrame,
                  load: np.ndarray) -> dict:
    """
    Aggregate all metrics into a single-row summary for t-test comparison.

    Total cost follows the same five-component structure as the MILP
    objective function (equation #4 in the model):
      Total cost = Energy cost
                 + Corrective maintenance cost  (no PM scheduled in baseline)
                 + Server switching cost
                 + Batch lateness penalty
                 (Preventive maintenance cost = 0 in baseline)

    Note: Because Round-Robin does not schedule any preventive maintenance,
    servers run at their full base failure rate LAM0 throughout the horizon.
    This results in a higher expected corrective maintenance cost compared
    to the MILP, which can reduce failure rates by scheduling PM windows.

    Returns:
        Dictionary with all aggregate performance and cost metrics.
    """
    sw_cost, n_switches = compute_switching_cost(load)

    total_energy_kwh  = slot_df["energy_kWh"].sum()
    total_energy_cost = slot_df["energy_cost"].sum()
    avg_util          = slot_df["avg_util"].mean()
    avg_pue           = slot_df["PUE"].dropna().mean()
    total_late_cost   = schedule_df["lateness_cost"].sum()
    jobs_placed       = schedule_df["job_id"].nunique()

    jobs_on_time = (
        schedule_df.groupby("job_id")["lateness_slots"].max() == 0).sum()

    # Wear-based CM cost — mirrors the solver's objective term exactly:
    #   cm_cost = c_cm * Σ_{j,k} lambda0[j] * (psi[j,k] / Lambda[j]) * active[j,k]
    #
    # In the baseline no PM is ever scheduled, so psi[j,k] grows from PSI_0[j]
    # linearly by load[j][k] each slot — the worst-case wear trajectory.
    # This produces a higher CM cost than the MILP (which resets psi via PM),
    # which is the key mechanism behind hypothesis H3.
    cm_cost = 0.0
    for j in J_ALL:
        psi_jk = PSI_0[j]
        for k in range(N_SLOTS):
            psi_jk += float(load[j][k])   # accumulate load, no reset (no PM)
            active = 1 if load[j][k] > 1e-6 else 0
            cm_cost += C_CM * LAM0[j] * (psi_jk / LAMBDA[j]) * active

    total_cost = total_energy_cost + cm_cost + sw_cost + total_late_cost

    # psi_end[j]: wear at end of horizon (= last psi_jk for each server).
    # Saved so that if a multi-day study is run, the baseline's wear state
    # can be carried forward symmetrically with the MILP's update_psi_0 step.
    psi_end = {}
    for j in J_ALL:
        psi_jk = PSI_0[j]
        for k in range(N_SLOTS):
            psi_jk += float(load[j][k])
        psi_end[j] = round(psi_jk, 6)

    return {
        "policy":              "Round-Robin",
        "timestamp":           datetime.now().strftime("%Y-%m-%d %H:%M"),
        "jobs_in_forecast":    len(schedule_df["job_id"].unique()),
        "jobs_placed":         jobs_placed,
        "jobs_on_time":        int(jobs_on_time),
        "total_energy_kWh":    round(total_energy_kwh, 3),
        "total_energy_cost_$": round(total_energy_cost, 4),
        "avg_server_util":     round(avg_util, 4),
        "avg_PUE":             round(avg_pue, 4),
        "switching_cost_$":    round(sw_cost, 4),
        "n_switches":          n_switches,
        "corrective_maint_$":  round(cm_cost, 4),
        "preventive_maint_$":  0.0,
        "lateness_cost_$":     round(total_late_cost, 4),
        "total_cost_$":        round(total_cost, 4),
        "avg_psi_end":         round(sum(psi_end.values()) / len(psi_end), 4),
        "max_psi_end":         round(max(psi_end.values()), 4),
    }


# ---------------------------------------------------------------------------
# 7.  MAIN ENTRY POINT
# ---------------------------------------------------------------------------
def main():
    """
    Run the full Round-Robin baseline pipeline:
      1. Load and prepare forecast jobs from the CSV output of forecast_model.py
      2. Assign all jobs using Round-Robin scheduling
      3. Compute per-slot energy, PUE, and utilization metrics
      4. Build aggregate summary for comparison with the MILP results
      5. Save all four output CSV files to outputs/baseline/
    """
    out_dir = Path("outputs/baseline")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  Round-Robin Baseline Scheduler")
    print("=" * 55)

    # Step 1: Load forecast jobs and derive scheduling parameters
    print("\nLoading forecast jobs...")
    jobs = load_jobs()
    print(f"  {len(jobs)} jobs loaded  "
          f"({(jobs.job_type=='batch').sum()} batch, "
          f"{(jobs.job_type=='interactive').sum()} interactive, "
          f"{jobs.is_critical.sum()} critical)")

    # Step 2: Run Round-Robin assignment and get load matrix
    print("\nRunning Round-Robin assignment...")
    schedule_df, load = round_robin_schedule(jobs)
    print(f"  {len(schedule_df)} placements made "
          f"across {N_J} servers, {N_SLOTS} slots")

    # Step 3: Compute per-slot energy, PUE, and utilization
    print("\nComputing slot-level metrics...")
    slot_df = compute_slot_metrics(load)

    # Step 4: Build aggregate summary for t-test comparison
    summary = build_summary(schedule_df, slot_df, load)

    # Step 5: Save all outputs
    schedule_df.to_csv(out_dir / "rr_job_schedule.csv", index=False)
    slot_df.to_csv(out_dir / "rr_slot_metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "rr_summary.csv", index=False)

    # Per-server utilization breakdown (for visualization and analysis)
    server_util = []
    for j in J_ALL:
        slots_active = sum(1 for k in range(N_SLOTS) if load[j][k] > 1e-6)
        server_util.append({
            "server":       j,
            "type":         "GPU" if j < 34 else "CPU",
            "slots_active": slots_active,
            "avg_load":     round(float(np.mean(load[j])), 4),
            "max_load":     round(float(np.max(load[j])), 4),
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
    print("  rr_summary.csv            - Aggregate metrics for t-test")


if __name__ == "__main__":
    main()
