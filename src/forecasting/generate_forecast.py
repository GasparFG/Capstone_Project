"""
generate_forecast.py
====================
Genera un forecast job-level para las siguientes 86,400 segundos (1 día).

Cada job predicho tiene:
  arrival_offset_seconds    → cuándo llega (segundos desde el último job histórico)
  required_cpu              → CPUs requeridas  (valor real del sistema, clasificador)
  required_memory           → memoria requerida MB (valor real del sistema, clasificador)
  expected_duration_seconds → duración esperada (regresión)
  role, app_name, job_type  → descriptores del job (clasificadores)

Salida:
  outputs/forecast_seconds/job_forecast.csv
  data/processed/optimization_input_seconds.parquet

Ejecutar desde src/forecasting/:
    python generate_forecast.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import joblib
import numpy as np
import pandas as pd

from config_seconds import (
    PREPARED_DATA, FORECAST_OUTPUT, MODELS_DIR, OUTPUTS_DIR,
    NUMERIC_TARGETS, DISCRETE_TARGETS, FORECAST_HORIZON_SECONDS, MAX_GENERATED_JOBS,
    INTERARRIVAL_MEDIANS,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_mapping(name: str) -> dict:
    path = OUTPUTS_DIR / f"{name}_mapping.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {int(r[f"{name}_encoded"]): r[name] for _, r in df.iterrows()}


def clean(X: pd.DataFrame) -> pd.DataFrame:
    return X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)


def pred_reg(model, X: pd.DataFrame) -> float:
    return max(0.0, float(np.expm1(model.predict(X)[0])))


def pred_clf_numeric(bundle, X: pd.DataFrame) -> float:
    """Clasificador para targets numéricos discretos (cpu, memory).
    inverse_transform devuelve string (ej. '64.0') — se convierte a float."""
    pred_enc = int(bundle["model"].predict(X)[0])
    val = bundle["label_encoder"].inverse_transform([pred_enc])[0]
    return float(val)


# ── Construcción de fila de features ─────────────────────────────────────────

_LAG_COLS = [
    "interarrival_seconds", "cpu_request", "memory_request", "duration_seconds",
    "job_type_encoded",
    "interarrival_seconds_log", "cpu_request_log", "memory_request_log", "duration_seconds_log",
    "role_encoded", "app_name_encoded",
]
_LAG_STEPS    = [1, 2, 3, 5, 10, 20, 50]
_ROLLING_COLS = ["interarrival_seconds", "cpu_request", "memory_request", "duration_seconds"]
_ROLLING_WINS = [5, 10, 20, 50]


def build_feature_row(history: pd.DataFrame, feature_cols: list, abs_seconds: float,
                      extra: dict = None) -> pd.DataFrame:
    """Construye una fila de features. Optimizado: pre-inicializa el dict a 0
    para evitar el loop de comprobación de columnas por cada predicción."""
    # Pre-inicializar todo a 0 — elimina el loop 'if c not in X.columns'
    row = {c: 0 for c in feature_cols}

    # Features de tiempo (ciclo diario)
    tod = abs_seconds % 86_400
    row["time_of_day_s"] = tod
    row["hour"]          = int(tod // 3600)
    row["minute"]        = int((tod % 3600) // 60)
    row["second"]        = int(tod % 60)
    row["time_sin"]      = np.sin(2 * np.pi * tod / 86_400)
    row["time_cos"]      = np.cos(2 * np.pi * tod / 86_400)

    # Lag features — usar .values para evitar overhead de pandas
    h_vals = {col: history[col].values for col in _LAG_COLS if col in history.columns}
    n = len(history)
    for lag in _LAG_STEPS:
        if n >= lag:
            for col, arr in h_vals.items():
                key = f"{col}_lag_{lag}"
                if key in row:
                    row[key] = arr[-lag]

    # Rolling features — numpy es ~5x más rápido que pandas .tail().mean()
    for col in _ROLLING_COLS:
        if col not in history.columns:
            continue
        arr = history[col].values
        for w in _ROLLING_WINS:
            recent = arr[-w:] if len(arr) >= w else arr
            if len(recent) == 0:
                continue
            row[f"{col}_rolling_mean_{w}"]   = recent.mean()
            row[f"{col}_rolling_std_{w}"]    = recent.std()    if len(recent) > 1 else 0
            row[f"{col}_rolling_median_{w}"] = np.median(recent)
            row[f"{col}_rolling_min_{w}"]    = recent.min()
            row[f"{col}_rolling_max_{w}"]    = recent.max()

    if extra:
        for k, v in extra.items():
            if k in row:
                row[k] = v

    return pd.DataFrame([row], columns=feature_cols)


# ── Pipeline principal ────────────────────────────────────────────────────────

def generate_forecast() -> None:
    # ── Cargar metadata y modelos ─────────────────────────────────────────────
    meta_path = MODELS_DIR / "model_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No se encontró {meta_path}.\n"
            "Ejecuta train_model.py primero."
        )
    meta         = load_json(meta_path)
    past_cols    = meta["past_feature_columns"]
    cap_cols     = meta["capacity_feature_columns"]
    desc_tgts    = meta.get("descriptor_targets", [])
    discrete_tgts= meta.get("discrete_targets", [])

    # Regresor continuo: duration_seconds
    num_models = {}
    for target in NUMERIC_TARGETS:
        p = MODELS_DIR / f"{target}_model.joblib"
        if not p.exists():
            raise FileNotFoundError(f"Modelo faltante: {p}. Ejecuta train_model.py.")
        num_models[target] = joblib.load(p)

    def load_clf(target):
        """Carga {"model": xgb, "label_encoder": le} y devuelve la tupla."""
        p = MODELS_DIR / f"{target}_classifier.joblib"
        if not p.exists():
            raise FileNotFoundError(f"Clasificador faltante: {p}. Ejecuta train_model.py.")
        return joblib.load(p)

    def predict_clf(bundle, X):
        """Predice descriptor (role/app/job_type) y devuelve el código original como int.
        Le fue entrenado con strings (ej. '3', '42') → inverse_transform devuelve string → cast int."""
        pred_enc = int(bundle["model"].predict(X)[0])
        val = bundle["label_encoder"].inverse_transform([pred_enc])[0]
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return 0

    # Clasificadores de descriptores (role, app_name, job_type)
    desc_models = {t: load_clf(t) for t in desc_tgts}

    # Clasificadores de recursos discretos (cpu_request, memory_request)
    # inverse_transform devuelve directamente el valor real (ej. 64, 96, 320...)
    discrete_models = {t: load_clf(t) for t in discrete_tgts}

    # Mappings para decodificar role/app/job_type de código → string
    mappings = {name: load_mapping(name) for name in ["role", "app_name", "job_type"]}

    # ── Historial inicial (últimos jobs del training set) ─────────────────────
    print(f"Cargando historial desde: {PREPARED_DATA}")
    data = pd.read_parquet(PREPARED_DATA)

    hist_cols = [
        "scheduled_seconds", "interarrival_seconds",
        "cpu_request", "memory_request", "duration_seconds",
        "job_type_encoded", "job_type",
        "interarrival_seconds_log", "cpu_request_log", "memory_request_log", "duration_seconds_log",
        "role_encoded", "app_name_encoded",
    ]
    # Solo necesitamos las últimas 60 filas (max lag=50, max rolling=50)
    # Cargar todo y luego recortar evita el O(n²) del concat creciente
    history = data[[c for c in hist_cols if c in data.columns]].iloc[-60:].copy().reset_index(drop=True)

    # Punto de partida: justo después del último job del dataset
    last_seconds = float(data["scheduled_seconds"].iloc[-1])
    offset       = 0.0
    jobs         = []


    print(f"\nGenerando forecast para {FORECAST_HORIZON_SECONDS:,} segundos (1 día)...")
    print(f"Inicio: segundo {last_seconds:,.0f} del dataset histórico\n")

    while offset < FORECAST_HORIZON_SECONDS and len(jobs) < MAX_GENERATED_JOBS:
        abs_s = last_seconds + offset

        # 1. Predecir bucket de interarrival (clasificador) → convertir a segundos
        X_past  = build_feature_row(history, past_cols, abs_s)
        bucket  = str(predict_clf(discrete_models["interarrival_bucket"], X_past))
        interarrival = INTERARRIVAL_MEDIANS.get(bucket, 24.0)   # fallback: mediana global
        offset += interarrival
        if offset > FORECAST_HORIZON_SECONDS:
            break

        abs_s = last_seconds + offset
        X_past = build_feature_row(history, past_cols, abs_s)

        # 2. Predecir descriptores del job (rol, app, tipo)
        generated_desc = {}
        for tgt, bundle in desc_models.items():
            generated_desc[tgt] = int(predict_clf(bundle, X_past))
        # Fallback: moda histórica si falta algún descriptor
        for tgt in ["role_encoded", "app_name_encoded", "job_type_encoded"]:
            if tgt in cap_cols and tgt not in generated_desc:
                generated_desc[tgt] = int(history[tgt].mode().iloc[0]) if tgt in history.columns else 0

        # 3. Predecir recursos con features de capacidad (incluye descriptores del job)
        X_cap = build_feature_row(history, cap_cols, abs_s, extra=generated_desc)

        # cpu y memory → clasificadores discretos: devuelven el valor real del sistema
        cpu    = pred_clf_numeric(discrete_models["cpu_request"],    X_cap)
        memory = pred_clf_numeric(discrete_models["memory_request"], X_cap)

        # duration → regresor: valor continuo en segundos
        duration = pred_reg(num_models["duration_seconds"], X_cap)

        # Decodificar descriptores a strings
        job_type_enc = int(generated_desc.get("job_type_encoded", 0))
        role_enc     = int(generated_desc.get("role_encoded", 0))
        app_enc      = int(generated_desc.get("app_name_encoded", 0))
        job_type     = mappings["job_type"].get(job_type_enc, "unknown")
        role         = mappings["role"].get(role_enc, "unknown")
        app_name     = mappings["app_name"].get(app_enc, "unknown")

        jobs.append({
            "job_id":                    f"pred_{len(jobs) + 1:06d}",
            "arrival_offset_seconds":    round(offset, 2),
            "job_type":                  job_type,   # batch / interactive
            "required_cpu":              cpu,        # valor real del sistema (ej. 64)
            "required_memory":           memory,     # valor real del sistema MB (ej. 320.0)
            "expected_duration_seconds": round(duration, 2),
            "role":                      role,
            "app_name":                  app_name,
        })

        # Actualizar historial con el job generado
        new_row = {
            "scheduled_seconds":      abs_s,
            "interarrival_seconds":   interarrival,
            "cpu_request":            cpu,
            "memory_request":         memory,
            "duration_seconds":       duration,
            "job_type_encoded":       job_type_enc,
            "job_type":               job_type,
            "interarrival_seconds_log": np.log1p(interarrival),
            "cpu_request_log":          np.log1p(cpu),
            "memory_request_log":       np.log1p(memory),
            "duration_seconds_log":     np.log1p(duration),
            "role_encoded":           role_enc,
            "app_name_encoded":       app_enc,
        }
        history = pd.concat([history.iloc[-60:], pd.DataFrame([new_row])], ignore_index=True)

    # ── Job-level forecast ────────────────────────────────────────────────────
    job_forecast = pd.DataFrame(jobs)

    # ── Guardar ───────────────────────────────────────────────────────────────
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FORECAST_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    job_forecast.to_parquet(FORECAST_OUTPUT, index=False)
    job_forecast.to_csv(OUTPUTS_DIR / "job_forecast.csv", index=False)

    # ── Resumen de consola ────────────────────────────────────────────────────
    n_batch       = (job_forecast["job_type"] == "batch").sum()
    n_interactive = (job_forecast["job_type"] == "interactive").sum()

    print("══════════════════════════════════════════════════════════")
    print("FORECAST COMPLETADO — JOB LEVEL")
    print("══════════════════════════════════════════════════════════")
    print(f"  Jobs generados:    {len(job_forecast):,}  "
          f"(batch={n_batch}, interactive={n_interactive})")
    print(f"  Horizonte:         {offset:,.0f} s  ({offset/3600:.1f} h)")
    print(f"  CPU única vals:    {sorted(job_forecast['required_cpu'].unique())}")
    print(f"  Memory única vals: {sorted(job_forecast['required_memory'].unique())[:8]}...")
    print()
    print(job_forecast.head(5).to_string(index=False))
    print()
    print(f"  Guardado en: {OUTPUTS_DIR / 'job_forecast.csv'}")
    print("══════════════════════════════════════════════════════════")


if __name__ == "__main__":
    generate_forecast()
