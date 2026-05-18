"""
train_model.py

This script trains SARIMA models for CPU and RAM utilization.

Expected input:
    data/processed/sarima_ready_dataset.parquet

Expected outputs:
    models/forecasting/cpu_sarima_model.pkl
    models/forecasting/ram_sarima_model.pkl
"""

from pathlib import Path
import pickle

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


INPUT_PATH = Path("data/processed/sarima_ready_dataset.parquet")
MODEL_DIR = Path("models/forecasting")

CPU_MODEL_PATH = MODEL_DIR / "cpu_sarima_model.pkl"
RAM_MODEL_PATH = MODEL_DIR / "ram_sarima_model.pkl"

SARIMA_ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 1, 1, 24)


def load_dataset(input_path: Path) -> pd.DataFrame:
    """Load SARIMA-ready dataset."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            "Run prepare_forecast_data.py first."
        )

    return pd.read_parquet(input_path)


def train_sarima_model(series: pd.Series):
    """Train a SARIMA model for one time-series variable."""
    model = SARIMAX(
        series,
        order=SARIMA_ORDER,
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fitted_model = model.fit(disp=False)
    return fitted_model


def save_model(model, output_path: Path) -> None:
    """Save trained model as a pickle file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as file:
        pickle.dump(model, file)


def main() -> None:
    """Train SARIMA models for CPU and RAM utilization."""
    df = load_dataset(INPUT_PATH)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    cpu_model = train_sarima_model(df["cpu_utilization"])
    ram_model = train_sarima_model(df["ram_utilization"])

    save_model(cpu_model, CPU_MODEL_PATH)
    save_model(ram_model, RAM_MODEL_PATH)

    print(f"CPU SARIMA model saved to: {CPU_MODEL_PATH}")
    print(f"RAM SARIMA model saved to: {RAM_MODEL_PATH}")


if __name__ == "__main__":
    main()