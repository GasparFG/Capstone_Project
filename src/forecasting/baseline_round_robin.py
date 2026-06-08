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
      sequence (GPU servers for GPU jobs, CPU servers for CPU-only jobs).
    - The job starts at the earliest time slot >= its release slot where the
      selected server has enough remaining capacity.
    - Critical jobs requiring 2 replicas are placed on 2 different servers.
    - No lookahead, no demand forecasting, no energy awareness.

PARAMETERS:
    All server parameters, slot structure, and cost formulas are identical to
    those used in the MILP optimization model (data_loader.py) to ensure a
    fair, apples-to-apples comparison for the research hypotheses.

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
C_E   = 0.139   # Electricity price ($/kWh) - Ontario time-of-use average
C_SW  = 0.1   # Penalty per server on/off state transition ($)
C_PM  = 250.0   # Fixed cost of preventive maintenance per server ($)
C_CM  = 100000  # Corrective maintenance cost coefficient ($ per failure event)

# Server failure rates per time slot.
# In the baseline, no preventive maintenance (PM) is scheduled, so the full
# base failure rate LAM0 applies throughout the entire 96-slot horizon.
LAM0   = {j: 0.00234 if j < 34 else 0.0020 for j in J_ALL}   # base failure rate/slot
LAM_PM = {j: 0.000702 if j < 34 else 0.0006 for j in J_ALL}   # post-PM rate (unused here)

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

    Reads the job-level forecast produced by forecast_model.py and derives:
      - release_slot:    earliest slot the job can start (from scheduled_seconds)
      - duration_slots:  how many consecutive slots the job occupies
      - deadline_slot:   latest slot by which the job must complete
      - r:               normalized resource requirement (0 to 1, combining CPU+memory)
      - is_critical:     whether the job needs 2 replicas for redundancy
      - eligible_servers: which servers can run this job (GPU or CPU group)

    Returns:
        DataFrame sorted by release_slot (arrival order), one row per job.
    """
    fc_path = Path("data/forecast/good_job_level_forecast.csv")
    fc = pd.read_csv(fc_path).dropna(
        subset=["scheduled_seconds", "duration_seconds", "cpu_request"])
    fc = fc.reset_index(drop=True)

    # Use 0-based job IDs to match the optimization model indexing convention
    fc["job_id"] = fc.index

    # Derive timing from forecast output
    fc["release_seconds"]             = fc["scheduled_seconds"]
    fc["processing_duration_seconds"] = fc["duration_seconds"]
    # Deadline = release + 1.25 * duration (25% time buffer over processing time)
    fc["deadline_seconds"] = (
        fc["release_seconds"] + fc["processing_duration_seconds"] * 1.25)

    # --- Criticality classification ---
    # A job is critical if it uses GPU, or requests top-10% CPU or memory.
    # Critical jobs require 2 replicas placed on different servers for fault tolerance.
    cpu_thresh = fc["cpu_request"].quantile(0.90)
    mem_thresh  = fc["memory_request"].quantile(0.90)
    fc["is_critical"] = (
        (fc["gpu_request"] == 1) |
        (fc["cpu_request"]  >= cpu_thresh) |
        (fc["memory_request"] >= mem_thresh)
    ).astype(int)
    fc["replica_count"] = np.where(fc["is_critical"] == 1, 2, 1)

    # --- Convert seconds to slot indices ---
    # release_slot: floor(seconds / SLOT_SECONDS), clamped to valid range
    fc["release_slot"] = fc["release_seconds"].apply(
        lambda s: max(0, min(int(math.floor(s / SLOT_SECONDS)), N_SLOTS - 1)))
    # duration_slots: ceiling division so the job always gets full processing time
    fc["duration_slots"] = fc["processing_duration_seconds"].apply(
        lambda s: max(1, int(math.ceil(s / SLOT_SECONDS))))
    # deadline_slot: ceiling division, capped at the horizon boundary
    fc["deadline_slot"] = fc["deadline_seconds"].apply(
        lambda s: min(N_SLOTS, int(math.ceil(s / SLOT_SECONDS))))

    # --- Resource normalization ---
    # r[i] combines CPU and memory demand into a single normalized load fraction.
    # Formula: r = 0.5*(cpu/max_cpu) + 0.5*(memory/max_memory)
    # This matches the formula used in build_jobs_json_from_forecast.py.
    max_cpu = 128 #cores (from server parameters based on G type server with highest CPU capacity)
    max_memory = 1024  # GB (from server parameters based on G type server with highest RAM capacity)
    max_mem = max(float(fc["memory_request"].max()), 1.0)
    fc["r"] = fc.apply(
        lambda row: round(
            min(1.0, 0.5 * row["cpu_request"] / max_cpu
                   + 0.5 * row["memory_request"] / max_mem), 4),
        axis=1)

    # --- Server eligibility ---
    # GPU jobs (role=HN, gpu_request=1) run exclusively on GPU servers.
    # CPU-only jobs (role=CN) run on CPU servers.
    fc["eligible_servers"] = fc["gpu_request"].apply(
        lambda g: J_GPU if int(g) == 1 else J_CPU)

    # --- Reclassify oversized batch jobs as interactive ---
    # Batch jobs with r >= 1.0 would violate the batch capacity constraint
    # (they need 100% of server capacity, but batch is capped at 70%).
    # These large, high-priority jobs are reclassified as interactive.
    # This matches the same fix applied in the MILP model.
    large_batch_mask = (fc["job_type"] == "batch") & (fc["r"] >= 1.0)
    fc.loc[large_batch_mask, "job_type"] = "interactive"

    # Ensure deadline is always reachable (>= release + duration)
    fc["deadline_slot"] = fc.apply(
        lambda row: max(
            row["deadline_slot"],
            row["release_slot"] + row["duration_slots"]),
        axis=1).clip(upper=N_SLOTS)

    # Return sorted by arrival time (Round-Robin processes in arrival order)
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

    # A job is "on time" if ALL of its replicas complete before the deadline
    jobs_on_time = (
        schedule_df.groupby("job_id")["lateness_slots"].max() == 0).sum()

    # Expected corrective maintenance cost (no PM in baseline):
    # Each server runs at base failure rate LAM0 for all N_SLOTS slots.
    # Expected failures per server = LAM0[j] * N_SLOTS
    cm_cost = C_CM * sum(LAM0[j] * N_SLOTS for j in J_ALL)

    total_cost = total_energy_cost + cm_cost + sw_cost + total_late_cost

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
        "lateness_cost_$":     round(total_late_cost, 4),
        "total_cost_$":        round(total_cost, 4),
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
