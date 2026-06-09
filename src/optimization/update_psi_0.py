"""
update_psi_0.py

Updates psi_0[j] in server_params JSON at the end of each daily optimization
cycle. This carries accumulated wear forward into the next day's run.

Usage (called from cli.py after each solve):
    update_psi_0(result, server_json_path)
"""

import json
from pathlib import Path
from typing import Any, Dict


def update_psi_0(result: Dict[str, Any], server_json_path: Path) -> None:
    """
    Extract psi[j, nK-1] from the solved model and write it back to
    server_params["psi_0"] in the server JSON file.

    If the model has no feasible solution, psi_0 is not updated so that
    the previous day's wear state is preserved for the next run.

    Args:
        result:            The result dict returned by solve_datacenter_model().
        server_json_path:  Path to server_params_42servers_v6.json.
    """
    if not result["feasible_solution"]:
        print("  [psi_0 update] No feasible solution — psi_0 unchanged.")
        return

    J = result["sets"]["J"]
    K = result["sets"]["K"]
    psi = result["vars"]["psi"]
    last_k = max(K)

    # psi[j, last_k] is the cumulative load at end of horizon.
    # If server j was under PM at last_k (z[j,last_k]=1), psi was reset to 0
    # by constraint c25_reset, so we naturally get 0 — correct behaviour.
    psi_end = {j: round(psi[j, last_k].X, 6) for j in J}

    with open(server_json_path, "r", encoding="utf-8") as f:
        server_data = json.load(f)

    server_data["server_params"]["psi_0"] = {str(j): v for j, v in psi_end.items()}

    with open(server_json_path, "w", encoding="utf-8") as f:
        json.dump(server_data, f, indent=2)

    avg_wear = sum(psi_end.values()) / len(psi_end) if psi_end else 0.0
    max_wear = max(psi_end.values()) if psi_end else 0.0
    print(f"  [psi_0 update] Written to {server_json_path}")
    print(f"  [psi_0 update] avg wear={avg_wear:.4f}  max wear={max_wear:.4f}")
