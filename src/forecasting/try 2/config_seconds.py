"""
config_seconds.py
=================
Configuración del pipeline de forecasting en SEGUNDOS.
Usa cleaned_data.parquet como entrada (12,374 jobs reales).
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]   # Capstone_Project/

# ── Entrada ──────────────────────────────────────────────────────────────────
INPUT_DATA = BASE_DIR / "data" / "interim" / "cleaned_data.parquet"

# ── Procesados ───────────────────────────────────────────────────────────────
PREPARED_DATA   = BASE_DIR / "data" / "processed" / "forecast_dataset_seconds.parquet"
FORECAST_OUTPUT = BASE_DIR / "data" / "processed" / "optimization_input_seconds.parquet"

# ── Salidas ──────────────────────────────────────────────────────────────────
OUTPUTS_DIR = BASE_DIR / "outputs" / "forecast_seconds"
MODELS_DIR  = BASE_DIR / "models" / "forecast_seconds"

# ── Columnas requeridas en cleaned_data ──────────────────────────────────────
REQUIRED_COLUMNS = [
    "instance_sn",
    "role",
    "app_name",
    "cpu_request",
    "memory_request",
    "scheduled_time",
    "duration_minutes",
    "job_type",
]

# ── Targets del modelo ───────────────────────────────────────────────────────
# Único target verdaderamente continuo → regresión XGBoost + log1p
NUMERIC_TARGETS  = ["duration_seconds"]

# Valores discretos → clasificación XGBoost
# cpu_request:          9 valores únicos  (2, 8, 12, 16, 48, 64, 96, 192...)
# memory_request:      24 valores únicos  (MB discretos del sistema)
# interarrival_bucket: 5 buckets operacionales (0s / 1-30s / 30-300s / 300-3600s / >3600s)
#   El 34.9% de jobs son simultáneos (interarrival=0). Regresión no puede predecir 0s
#   exactos, destruyendo R² y MAPE. Clasificar el RÉGIMEN de llegada es más
#   útil para el optimizador que predecir el segundo exacto.
DISCRETE_TARGETS = ["cpu_request", "memory_request", "interarrival_bucket"]

# Descriptores del job → clasificación
DESCRIPTOR_TARGETS = ["role_encoded", "app_name_encoded", "job_type_encoded"]

# Buckets de interarrival (segundos) y su valor representativo para el forecast
INTERARRIVAL_BINS    = [-1, 0, 30, 300, 3600, float("inf")]
INTERARRIVAL_LABELS  = ["0s", "1-30s", "30-300s", "300-3600s", ">3600s"]
INTERARRIVAL_MEDIANS = {       # mediana empírica de cada bucket (segundos)
    "0s":      0.0,
    "1-30s":   8.0,
    "30-300s": 90.0,
    "300-3600s": 600.0,
    ">3600s":  3600.0,
}

# ── Parámetros de entrenamiento ──────────────────────────────────────────────
TEST_SIZE          = 0.20
RANDOM_STATE       = 42

# ── Horizonte del forecast ───────────────────────────────────────────────────
FORECAST_HORIZON_SECONDS = 86_400   # 1 día completo
MAX_GENERATED_JOBS       = 10_000   # límite de seguridad
