import pandas as pd

from config import METRICS_DIR, FORECAST_FREQ


def compare_forecasting_approaches():
    direct_metrics_path = (
        METRICS_DIR / f"all_model_metrics_summary_{FORECAST_FREQ}.csv"
    )

    two_stage_metrics_path = (
        METRICS_DIR / f"two_stage_metrics_{FORECAST_FREQ}.csv"
    )

    classifier_metrics_path = (
        METRICS_DIR / f"two_stage_classifier_metrics_{FORECAST_FREQ}.csv"
    )

    if not direct_metrics_path.exists():
        raise FileNotFoundError(
            f"Direct forecasting metrics not found: {direct_metrics_path}"
        )

    if not two_stage_metrics_path.exists():
        raise FileNotFoundError(
            f"Two-stage metrics not found: {two_stage_metrics_path}"
        )

    direct_metrics = pd.read_csv(direct_metrics_path)
    two_stage_metrics = pd.read_csv(two_stage_metrics_path)

    if classifier_metrics_path.exists():
        classifier_metrics = pd.read_csv(classifier_metrics_path)
    else:
        classifier_metrics = pd.DataFrame()

    comparison_rows = []

    targets = sorted(
        set(direct_metrics["target"]).intersection(
            set(two_stage_metrics["target"])
        )
    )

    for target in targets:
        direct_target_metrics = direct_metrics[
            direct_metrics["target"] == target
        ].copy()

        direct_best = direct_target_metrics.sort_values(
            by="RMSE",
            ascending=True
        ).iloc[0]

        two_stage_row = two_stage_metrics[
            two_stage_metrics["target"] == target
        ].iloc[0]

        if two_stage_row["RMSE"] < direct_best["RMSE"]:
            winner = "Two-stage model"
        else:
            winner = "Direct aggregate forecast"

        comparison_rows.append(
            {
                "forecast_freq": FORECAST_FREQ,
                "target": target,

                "direct_best_model": direct_best["model"],
                "direct_RMSE": direct_best["RMSE"],
                "direct_MAE": direct_best["MAE"],
                "direct_MAPE": direct_best["MAPE"],
                "direct_SMAPE": direct_best["SMAPE"],
                "direct_R2": direct_best["R2"],

                "two_stage_model": two_stage_row["model"],
                "two_stage_RMSE": two_stage_row["RMSE"],
                "two_stage_MAE": two_stage_row["MAE"],
                "two_stage_MAPE": two_stage_row["MAPE"],
                "two_stage_SMAPE": two_stage_row["SMAPE"],
                "two_stage_R2": two_stage_row["R2"],

                "winner_by_RMSE": winner
            }
        )

    comparison_data = pd.DataFrame(comparison_rows)

    output_path = (
        METRICS_DIR / f"forecasting_approach_comparison_{FORECAST_FREQ}.csv"
    )

    comparison_data.to_csv(output_path, index=False)

    print(f"Forecasting approach comparison saved to: {output_path}")
    print(comparison_data)

    if not classifier_metrics.empty:
        print("\nTwo-stage active window classifier metrics:")
        print(classifier_metrics)


if __name__ == "__main__":
    compare_forecasting_approaches()