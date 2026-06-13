"""
collect_robustness_results.py
==============================
Collects the seven required reporting metrics from every completed
robustness test case and writes a single summary CSV plus a formatted
console / text table.

Required columns per test case
--------------------------------
1.  n_servers         — number of servers in the fleet
2.  n_jobs            — number of jobs in the workload
3.  n_slots           — number of time slots (K)
4.  num_variables     — total Gurobi variables (from performance_metrics CSV)
5.  num_constraints   — total Gurobi constraints (from performance_metrics CSV)
6.  runtime_seconds   — solver wall-clock time
7.  objective_value   — optimal / best-found objective

Additional context columns are included to make the table self-contained:
  psi_stage, mip_gap_pct, status, has_solution, slot_minutes,
  num_binary_variables.

Data sources (in priority order for each field)
------------------------------------------------
1. performance_metrics_*.csv written by output_writer.save_result_files()
   — contains num_variables, num_constraints, runtime_seconds,
     objective_value, mip_gap_pct, status, has_solution.
2. run_log.jsonl written by run_robustness_suite.py
   — contains n_servers, n_jobs, k_slots, psi_stage, elapsed_s, status.
3. test_manifest.json
   — ground-truth for all axis dimensions; used when the CSV is missing
     (e.g. infeasible cases that produced no solution file).

Usage
-----
    python collect_robustness_results.py \
        --output-root  outputs/robustness_20260609 \
        --manifest     data/robustness_tests/test_manifest.json \
        [--run-log     outputs/robustness_20260609/run_log.jsonl] \
        [--out-csv     outputs/robustness_20260609/robustness_summary.csv] \
        [--out-txt     outputs/robustness_20260609/robustness_summary.txt]

References
----------
Bowly, S. et al. (2020). Generation Techniques for Hard Random MILP
    Instances. INFORMS Journal on Computing, 32(4).
    https://doi.org/10.1287/ijoc.2019.0933
    (Recommends reporting variables, constraints, solve time, and
    objective gap as the standard MILP benchmark metrics.)

Lodi, A. & Tramontani, A. (2023). Performance Variability in
    Mixed-Integer Programming. In: Toth, P. & Vigo, D. (eds),
    Vehicle Routing: Problems, Methods, and Applications (3rd ed.).
    SIAM. (Gap and solve-time distribution analysis methodology.)
"""

import argparse
import csv
import glob
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Field names
# ---------------------------------------------------------------------------

SUMMARY_FIELDS = [
    # ---- the 7 required reporting columns ----
    "case_name",
    "n_servers",
    "n_jobs",
    "n_slots",
    "num_variables",
    "num_constraints",
    "runtime_seconds",
    "objective_value",
    # ---- context / diagnostics ----
    "psi_stage",
    "psi_fraction",
    "slot_minutes",
    "status",
    "has_solution",
    "mip_gap_pct",
    "num_binary_variables",
    "n_gpu",
    "n_cpu",
    "n_batch",
    "n_interactive",
    "n_critical",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_metrics_csv(case_dir: Path) -> Optional[Path]:
    """
    Return the most recent performance_metrics_*.csv in case_dir,
    searching both the case root and the outputs/optimization sub-folder
    that save_result_files() creates.
    """
    patterns = [
        case_dir / "performance_metrics_*.csv",
        case_dir / "outputs" / "optimization" / "performance_metrics_*.csv",
        case_dir / "outputs" / "optimization" / "performance_metrics.csv",
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(str(pat)))
    if not candidates:
        return None
    # most recently modified
    return Path(sorted(candidates, key=os.path.getmtime)[-1])


def _read_metrics_csv(path: Path) -> Optional[Dict[str, Any]]:
    """
    Read the first data row of a performance_metrics CSV.
    Returns a dict with string values (convert to float/int at call site).
    """
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                return dict(row)   # first row only (base scenario)
    except Exception:
        return None
    return None


def _safe_float(val: Any, default: str = "") -> Any:
    if val is None or str(val).strip() == "":
        return default
    try:
        f = float(val)
        return "" if math.isnan(f) else round(f, 6)
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: str = "") -> Any:
    if val is None or str(val).strip() == "":
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Core collector
# ---------------------------------------------------------------------------

def collect(
    output_root: Path,
    manifest_path: Path,
    run_log_path: Optional[Path],
) -> List[Dict[str, Any]]:
    """
    For every case in the manifest, locate its metrics CSV inside
    output_root/<case_name>/ and merge with manifest + run-log data.
    Returns a list of dicts, one per case.
    """

    manifest = json.loads(manifest_path.read_text())

    # Build run-log index keyed by case_name (optional but helpful)
    run_log_index: Dict[str, Dict] = {}
    if run_log_path and run_log_path.exists():
        with open(run_log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        run_log_index[entry["case_name"]] = entry
                    except json.JSONDecodeError:
                        pass

    rows: List[Dict[str, Any]] = []

    for case in manifest:
        name     = case["case_name"]
        case_dir = output_root / name

        # ---- ground-truth dimensions from manifest ----
        n_servers   = case["n_servers"]
        n_jobs      = case["n_jobs"]
        k_slots     = case["k_slots"]
        slot_min    = case.get("slot_minutes", case["slot_seconds"] // 60)
        psi_stage   = case.get("psi_stage", "")
        psi_frac    = case.get("psi_fraction", "")
        n_gpu       = case.get("n_gpu", "")
        n_cpu       = case.get("n_cpu", "")
        n_batch     = case.get("n_batch", "")
        n_inter     = case.get("n_interactive", "")
        n_crit      = case.get("n_critical", "")

        # ---- defaults (filled in if CSV is present) ----
        num_vars    = ""
        num_bvars   = ""
        num_constrs = ""
        runtime_s   = ""
        obj_val     = ""
        mip_gap     = ""
        status      = "NOT_RUN"
        has_sol     = 0

        # ---- try run-log first for status / elapsed ----
        if name in run_log_index:
            rl = run_log_index[name]
            status    = rl.get("status", status)
            runtime_s = _safe_float(rl.get("elapsed_s", ""))

        # ---- load performance_metrics CSV ----
        csv_path = _find_metrics_csv(case_dir)
        if csv_path:
            metrics = _read_metrics_csv(csv_path)
            if metrics:
                num_vars    = _safe_int  (metrics.get("num_variables",        ""))
                num_bvars   = _safe_int  (metrics.get("num_binary_variables", ""))
                num_constrs = _safe_int  (metrics.get("num_constraints",      ""))
                obj_val     = _safe_float(metrics.get("objective_value",      ""))
                mip_gap     = _safe_float(metrics.get("mip_gap_pct",          ""))
                has_sol     = _safe_int  (metrics.get("has_solution",         0))
                # runtime from CSV is more accurate than subprocess elapsed
                csv_rt = _safe_float(metrics.get("runtime_seconds", ""))
                if csv_rt != "":
                    runtime_s = csv_rt
                # status from CSV overrides run-log when solution exists
                csv_status = metrics.get("status", "").strip()
                if csv_status:
                    status = csv_status
        else:
            # No CSV — case either not run yet or was infeasible with no output
            if name not in run_log_index:
                status = "NOT_RUN"

        rows.append({
            "case_name":           name,
            "n_servers":           n_servers,
            "n_jobs":              n_jobs,
            "n_slots":             k_slots,
            "num_variables":       num_vars,
            "num_constraints":     num_constrs,
            "runtime_seconds":     runtime_s,
            "objective_value":     obj_val,
            "psi_stage":           psi_stage,
            "psi_fraction":        psi_frac,
            "slot_minutes":        slot_min,
            "status":              status,
            "has_solution":        has_sol,
            "mip_gap_pct":         mip_gap,
            "num_binary_variables": num_bvars,
            "n_gpu":               n_gpu,
            "n_cpu":               n_cpu,
            "n_batch":             n_batch,
            "n_interactive":       n_inter,
            "n_critical":          n_crit,
        })

    return rows


# ---------------------------------------------------------------------------
# Formatted text table
# ---------------------------------------------------------------------------

def _fmt(val: Any, width: int, align: str = ">") -> str:
    s = "" if val == "" else str(val)
    fmt = f"{{:{align}{width}}}"
    return fmt.format(s[:width])


def write_text_table(rows: List[Dict[str, Any]], path: Path) -> None:
    """
    Write a human-readable fixed-width table focusing on the 7 required
    columns plus status and psi_stage for context.
    Groups rows by psi_stage to make cross-stage comparison easy.
    """
    lines: List[str] = []

    header = (
        f"{'Case':<48} "
        f"{'Svrs':>4} "
        f"{'Jobs':>5} "
        f"{'Kslots':>6} "
        f"{'Vars':>8} "
        f"{'Constrs':>9} "
        f"{'Time(s)':>8} "
        f"{'ObjVal':>14} "
        f"{'Gap%':>6} "
        f"{'Status':<12} "
        f"PSI_stage"
    )
    sep = "-" * len(header)

    lines.append("Robustness Test Suite — Summary of Results")
    lines.append("=" * len(header))
    lines.append("")
    lines.append("Seven required reporting metrics per case")
    lines.append("  (1) n_servers  (2) n_jobs  (3) n_slots  (4) num_variables")
    lines.append("  (5) num_constraints  (6) runtime_seconds  (7) objective_value")
    lines.append("")

    # Group by psi_stage
    stages_order = ["fresh", "mid_life", "near_thresh", "at_thresh", "over_thresh"]
    by_stage: Dict[str, List[Dict]] = {s: [] for s in stages_order}
    other: List[Dict] = []
    for r in rows:
        ps = r.get("psi_stage", "")
        if ps in by_stage:
            by_stage[ps].append(r)
        else:
            other.append(r)

    for stage in stages_order:
        group = by_stage[stage]
        if not group:
            continue
        lines.append(f"PSI stage: {stage}  "
                     f"(psi_0 = {group[0]['psi_fraction']:.2f} × Lambda)")
        lines.append(sep)
        lines.append(header)
        lines.append(sep)
        for r in group:
            rt  = f"{r['runtime_seconds']:.1f}" if r['runtime_seconds'] != "" else "—"
            obj = f"{r['objective_value']:.4f}"  if r['objective_value'] != "" else "—"
            gap = f"{r['mip_gap_pct']:.2f}"      if r['mip_gap_pct'] != "" else "—"
            nv  = str(r['num_variables'])         if r['num_variables'] != "" else "—"
            nc  = str(r['num_constraints'])       if r['num_constraints'] != "" else "—"
            lines.append(
                f"{r['case_name']:<48} "
                f"{r['n_servers']:>4} "
                f"{r['n_jobs']:>5} "
                f"{r['n_slots']:>6} "
                f"{nv:>8} "
                f"{nc:>9} "
                f"{rt:>8} "
                f"{obj:>14} "
                f"{gap:>6} "
                f"{str(r['status']):<12} "
                f"{stage}"
            )
        lines.append("")

    # Completion summary
    total   = len(rows)
    solved  = sum(1 for r in rows if r["has_solution"] == 1)
    timeout = sum(1 for r in rows if str(r["status"]) == "TIME_LIMIT")
    infeas  = sum(1 for r in rows if str(r["status"]) == "INFEASIBLE")
    not_run = sum(1 for r in rows if str(r["status"]) == "NOT_RUN")
    lines.append("=" * len(header))
    lines.append(f"Total cases : {total}")
    lines.append(f"  Solved    : {solved}  (OPTIMAL or suboptimal with solution)")
    lines.append(f"  Time limit: {timeout}  (hit wall clock, may have solution)")
    lines.append(f"  Infeasible: {infeas}")
    lines.append(f"  Not run   : {not_run}")

    # Per-axis scaling summaries (solve time & variable count)
    lines.append("")
    lines.append("Scaling summary — mean solve time (s) by axis")
    lines.append("-" * 50)

    def _mean_rt(subset):
        vals = [r["runtime_seconds"] for r in subset
                if isinstance(r["runtime_seconds"], (int, float))]
        return f"{sum(vals)/len(vals):.1f}" if vals else "—"

    def _mean_vars(subset):
        vals = [r["num_variables"] for r in subset
                if isinstance(r["num_variables"], int)]
        return f"{int(sum(vals)/len(vals)):,}" if vals else "—"

    for axis_key, axis_label in [
        ("n_servers", "Servers"),
        ("n_jobs",    "Jobs   "),
        ("n_slots",   "K slots"),
        ("psi_stage", "PSI stage"),
    ]:
        seen_vals = sorted(set(r[axis_key] for r in rows),
                           key=lambda x: str(x))
        for v in seen_vals:
            sub = [r for r in rows if r[axis_key] == v]
            lines.append(
                f"  {axis_label} = {str(v):<15}  "
                f"mean_time={_mean_rt(sub):>7}s   "
                f"mean_vars={_mean_vars(sub):>10}"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Text table  → {path}")


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_summary_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary CSV → {path}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect robustness suite results into a single summary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-root", required=True,
        help="Root folder passed to run_robustness_suite.py "
             "(contains one sub-folder per case).",
    )
    parser.add_argument(
        "--manifest", required=True,
        help="Path to test_manifest.json produced by generate_robustness_tests.py.",
    )
    parser.add_argument(
        "--run-log", default=None,
        help="Path to run_log.jsonl (defaults to <output-root>/run_log.jsonl).",
    )
    parser.add_argument(
        "--out-csv", default=None,
        help="Output CSV path (defaults to <output-root>/robustness_summary.csv).",
    )
    parser.add_argument(
        "--out-txt", default=None,
        help="Output text table path "
             "(defaults to <output-root>/robustness_summary.txt).",
    )
    args = parser.parse_args()

    output_root  = Path(args.output_root)
    manifest_path = Path(args.manifest)
    run_log_path  = Path(args.run_log) if args.run_log \
                    else output_root / "run_log.jsonl"
    out_csv  = Path(args.out_csv)  if args.out_csv  \
               else output_root / "robustness_summary.csv"
    out_txt  = Path(args.out_txt)  if args.out_txt  \
               else output_root / "robustness_summary.txt"

    if not manifest_path.exists():
        sys.exit(f"Manifest not found: {manifest_path}")

    print(f"Collecting results from  : {output_root}")
    print(f"Manifest                 : {manifest_path}")
    print(f"Run log                  : {run_log_path}")

    rows = collect(output_root, manifest_path, run_log_path)

    write_summary_csv(rows, out_csv)
    write_text_table(rows, out_txt)

    # Quick console preview of the 7 required columns
    solved = [r for r in rows if r["has_solution"] == 1]
    print(f"\n{len(solved)}/{len(rows)} cases have a feasible solution.\n")
    print(f"{'Case':<48} {'Svrs':>4} {'Jobs':>5} {'K':>4} "
          f"{'Vars':>8} {'Cstrs':>8} {'Time':>7} {'ObjVal':>14}")
    print("-" * 105)
    for r in rows:
        rt  = f"{r['runtime_seconds']:.1f}" \
              if isinstance(r['runtime_seconds'], float) else "—"
        obj = f"{r['objective_value']:.4f}" \
              if isinstance(r['objective_value'], float) else "—"
        nv  = f"{r['num_variables']:,}"     if isinstance(r['num_variables'], int) else "—"
        nc  = f"{r['num_constraints']:,}"   if isinstance(r['num_constraints'], int) else "—"
        print(f"{r['case_name']:<48} {r['n_servers']:>4} {r['n_jobs']:>5} "
              f"{r['n_slots']:>4} {nv:>8} {nc:>8} {rt:>7} {obj:>14}")


if __name__ == "__main__":
    main()
