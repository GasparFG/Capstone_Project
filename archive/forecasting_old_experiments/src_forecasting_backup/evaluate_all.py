import pandas as pd

from config import METRICS_DIR, FORECAST_FREQ


def evaluate_all_models():
    metric_files = list(METRICS_DIR.glob(f"*_metrics_{FORECAST_FREQ}.csv"))

    metric_files = [
        file for file in metric_files
        if file.name != f"all_model_metrics_summary_{FORECAST_FREQ}.csv"
    ]

    if not metric_files:
        raise FileNotFoundError(
            f"No metrics files found for FORECAST_FREQ={FORECAST_FREQ}. "
            "Run the training scripts first."
        )

    all_metrics = []

    for metric_file in metric_files:
        metric_data = pd.read_csv(metric_file)
        all_metrics.append(metric_data)

    summary_data = pd.concat(all_metrics, ignore_index=True)

    summary_data = summary_data.sort_values(
        by=["target", "RMSE"],
        ascending=[True, True]
    )

    output_path = (
        METRICS_DIR / f"all_model_metrics_summary_{FORECAST_FREQ}.csv"
    )

    summary_data.to_csv(output_path, index=False)

    print(f"Final metrics summary saved to: {output_path}")
    print(summary_data)


if __name__ == "__main__":
    evaluate_all_models()