"""Configuration constants for the optimization pipeline."""

from gurobipy import GRB

DEFAULT_PARQUET_INPUT = "data/processed/optimization_input_dataset.parquet"
DEFAULT_JOBS_JSON_INPUT = "data/processed/optimization_jobs_params.json"
DEFAULT_SERVER_JSON_INPUT = "data/processed/server_params_42servers_v3.json"
DEFAULT_OUTPUT_ROOT = "."

STATUS_LABELS = {
    GRB.OPTIMAL: "OPTIMAL",
    GRB.TIME_LIMIT: "TIME_LIMIT",
    GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.INF_OR_UNBD: "INF_OR_UNBD",
    GRB.UNBOUNDED: "UNBOUNDED",
}
