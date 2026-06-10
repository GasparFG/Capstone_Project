"""Small reusable helpers for the optimization pipeline."""

from pathlib import Path
from typing import Any, Dict, List
import csv

import gurobipy as gp

from .config import STATUS_LABELS


def ensure_output_dirs(output_root: Path) -> Dict[str, Path]:
    """Create the folder structure."""
    paths = {
        "optimization": output_root / "outputs" / "optimization",
        "reports": output_root / "outputs" / "results" / "reports",
        "tables": output_root / "outputs" / "results" / "tables",
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
