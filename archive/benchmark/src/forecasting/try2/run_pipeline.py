"""
run_pipeline.py
===============
Master script for the forecasting pipeline in SECONDS.

Runs in order:
    1. prepare_data.py      → builds dataset with lag features (in seconds)
    2. train_model.py       → trains XGBoost models, saves .joblib files
    3. generate_forecast.py → generates 1-day job-level forecast

Usage (from any directory):
    python src/forecasting/run_pipeline.py

Main outputs:
    outputs/forecast_seconds/job_forecast.csv       <- predicted job list (job-level)
    outputs/forecast_seconds/regression_metrics.csv <- model metrics
    data/processed/optimization_input_seconds.parquet
"""

import sys
import importlib
from pathlib import Path

THIS_DIR     = Path(__file__).resolve().parent   # src/forecasting/
PROJECT_ROOT = THIS_DIR.parents[1]               # Capstone_Project/


def run_step(label: str, module_name: str, func_name: str) -> None:
    print(f"\n{'='*62}")
    print(f"  {label}")
    print(f"{'='*62}")

    dir_str = str(THIS_DIR)
    if dir_str in sys.path:
        sys.path.remove(dir_str)
    sys.path.insert(0, dir_str)

    # Clear import cache to force reload with correct sys.path
    for mod in [module_name, "config_seconds"]:
        if mod in sys.modules:
            del sys.modules[mod]

    module = importlib.import_module(module_name)
    getattr(module, func_name)()


def main():
    steps = [
        ("STEP 1/3  Prepare data in seconds",   "prepare_data",      "prepare_data"),
        ("STEP 2/3  Train XGBoost models",       "train_model",       "train_model"),
        ("STEP 3/3  Generate forecast (1 day)",  "generate_forecast", "generate_forecast"),
    ]

    for label, module, func in steps:
        run_step(label, module, func)

    print(f"\n{'='*62}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*62}")
    print(f"  Job forecast CSV:  {PROJECT_ROOT / 'outputs' / 'forecast_seconds' / 'job_forecast.csv'}")
    print(f"  Metrics:           {PROJECT_ROOT / 'outputs' / 'forecast_seconds' / 'regression_metrics.csv'}")
    print(f"  Parquet (optim):   {PROJECT_ROOT / 'data' / 'processed' / 'optimization_input_seconds.parquet'}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
