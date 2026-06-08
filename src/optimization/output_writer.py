"""Write optimization outputs and human-readable reports."""

from pathlib import Path
from time import time
from typing import Any, Dict, List
import time
from .result_extractor import (
    extract_solution_rows,
    extract_hourly_rows,
    extract_pm_rows,
    extract_performance_metrics,
    extract_server_load_timeseries,
    extract_server_summary,
)
from .utils import write_csv


def save_result_files(result: Dict[str, Any], paths: Dict[str, Path]) -> Dict[str, Path]:
    """Save all output artifacts for one scenario."""
    scenario = result["scenario_name"]
    timestr = time.strftime("%Y%m%d-%H%M%S")

    solution_rows = extract_solution_rows(result)
    hourly_rows = extract_hourly_rows(result)
    pm_rows = extract_pm_rows(result)
    metrics_row = extract_performance_metrics(result)
    server_ts_rows = extract_server_load_timeseries(result)
    server_summary_rows = extract_server_summary(result)

    solution_path = paths["optimization"] / \
        f"optimization_solution_{scenario}_{timestr}.csv"
    hourly_path = paths["tables"] / f"hourly_energy_thermal_{scenario}_{timestr}.csv"
    pm_path = paths["tables"] / f"pm_schedule_{scenario}_{timestr}.csv"
    metrics_path = paths["optimization"] / \
        f"performance_metrics_{scenario}_{timestr}.csv"
    server_ts_path = paths["tables"] / f"server_load_timeseries_{scenario}_{timestr}.csv"
    server_summary_path = paths["optimization"] / f"server_summary_{scenario}_{timestr}.csv"
    report_path = paths["reports"] / f"optimization_report_{scenario}_{timestr}.txt"

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
    write_csv(
        server_ts_path,
        server_ts_rows,
        ["scenario", "server_id", "server_type", "slot", "time",
         "load", "batch_load", "interactive_load", "active", "in_maintenance", "status"],
    )
    write_csv(
        server_summary_path,
        server_summary_rows,
        ["scenario", "server_id", "server_type", "slots_active", "avg_load",
         "max_load", "batch_avg_load", "interactive_avg_load", "utilization_rate_pct",
         "pm_scheduled", "pm_start_time", "pm_end_time", "final_status"],
    )

    save_text_report(result, metrics_row, solution_rows,
                     hourly_rows, pm_rows, report_path)

    return {
        "solution": solution_path,
        "hourly": hourly_path,
        "pm": pm_path,
        "metrics": metrics_path,
        "server_timeseries": server_ts_path,
        "server_summary": server_summary_path,
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
    timestr = time.strftime("%Y%m%d-%H%M%S")

    if all_metrics:
        write_csv(paths["optimization"] / f"performance_metrics_{timestr}.csv",
                  all_metrics, list(all_metrics[0].keys()))
        write_csv(paths["tables"] / f"performance_metrics_{timestr}.csv",
                  all_metrics, list(all_metrics[0].keys()))

    if all_solution_rows:
        write_csv(
            paths["optimization"] / f"optimization_solution_{timestr}.csv",
            all_solution_rows,
            ["scenario", "job_id", "job_type", "is_critical", "server_id",
                "start_slot", "end_slot", "start_time", "end_time", "duration_hours"],
        )


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
