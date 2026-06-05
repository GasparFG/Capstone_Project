"""
run_pipeline.py
===============
Script maestro del pipeline de forecasting en SEGUNDOS.

Ejecuta en orden:
    1. prepare_data.py     → genera dataset con lag features (en segundos)
    2. train_model.py      → entrena modelos XGBoost, guarda .joblib
    3. generate_forecast.py → genera forecast de 1 día (job-level + daily summary)

Uso (desde cualquier directorio):
    python src/forecasting/run_pipeline.py

Salidas principales:
    outputs/forecast_seconds/job_forecast.csv       ← lista de jobs predichos (job-level)
    outputs/forecast_seconds/regression_metrics.csv ← métricas del modelo
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

    # Limpiar caché de imports para forzar recarga con el sys.path correcto
    for mod in [module_name, "config_seconds"]:
        if mod in sys.modules:
            del sys.modules[mod]

    module = importlib.import_module(module_name)
    getattr(module, func_name)()


def main():
    steps = [
        ("PASO 1/3  Preparar datos en segundos",    "prepare_data",      "prepare_data"),
        ("PASO 2/3  Entrenar modelos XGBoost",       "train_model",       "train_model"),
        ("PASO 3/3  Generar forecast (1 día)",       "generate_forecast", "generate_forecast"),
    ]

    for label, module, func in steps:
        run_step(label, module, func)

    print(f"\n{'='*62}")
    print("  PIPELINE COMPLETADO")
    print(f"{'='*62}")
    print(f"  Job forecast CSV:     {PROJECT_ROOT / 'outputs' / 'forecast_seconds' / 'job_forecast.csv'}")
    print(f"  Métricas:             {PROJECT_ROOT / 'outputs' / 'forecast_seconds' / 'regression_metrics.csv'}")
    print(f"  Parquet (optim):      {PROJECT_ROOT / 'data' / 'processed' / 'optimization_input_seconds.parquet'}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
