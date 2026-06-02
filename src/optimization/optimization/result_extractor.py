"""Extract CSV-ready result tables from solved optimization models."""

from typing import Any, Dict, List
import math

from .utils import safe_value


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
