"""Command-line entry point for the optimization pipeline."""

import argparse
from pathlib import Path

from .config import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_JOBS_JSON_INPUT,
    DEFAULT_SERVER_JSON_INPUT,
)
from .data_loader import load_data_from_jobs_json
from .build_jobs_json_from_forecast import main as build_jobs_json_from_forecast
from .scenario_builder import build_scenarios
from .solver import solve_datacenter_model
from .utils import ensure_output_dirs
from .output_writer import save_result_files, save_combined_files, print_console_summary
from .result_extractor import extract_performance_metrics, extract_solution_rows
from .update_psi_0 import update_psi_0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the data-centre optimization pipeline.")
    parser.add_argument(
        "--jobs-json-input",
        default=DEFAULT_JOBS_JSON_INPUT,
        help="Path to optimization_jobs_params.json input file.",
    )
    parser.add_argument(
        "--server-json-input",
        default=DEFAULT_SERVER_JSON_INPUT,
        help="Path to server_params_42servers_v6.json input file.",
    )
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

    jobs_json_path = Path(args.jobs_json_input)
    server_json_path = Path(args.server_json_input)
    output_root = Path(args.output_root)

    paths = ensure_output_dirs(output_root)
    print("\nBuilding optimization jobs JSON from forecast output...")
    build_jobs_json_from_forecast()

    base_data = load_data_from_jobs_json(
        jobs_json_path=jobs_json_path,
        server_json_path=server_json_path,
    )
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

        # Update psi_0 only from the base scenario so that the inherited wear
        # reflects the real daily workload, not a stress-test perturbation.
        if scenario_name == "base":
            update_psi_0(result, server_json_path)

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
