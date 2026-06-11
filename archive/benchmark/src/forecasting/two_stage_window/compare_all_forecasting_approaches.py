import pandas as pd

from config import METRICS_DIR, FORECAST_FREQ


def load_metrics_if_exists(path):
    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()


def compare_all_forecasting_approaches():
    direct_metrics_path = (
        METRICS_DIR / f"all_model_metrics_summary_{FORECAST_FREQ}.csv"
    )

    two_stage_aggregate_path = (
        METRICS_DIR / f"two_stage_metrics_{FORECAST_FREQ}.csv"
    )

    two_stage_job_level_path = (
        METRICS_DIR / f"two_stage_job_level_metrics_{FORECAST_FREQ}.csv"
    )

    direct_metrics = load_metrics_if_exists(direct_metrics_path)
    two_stage_aggregate_metrics = load_metrics_if_exists(two_stage_aggregate_path)
    two_stage_job_level_metrics = load_metrics_if_exists(two_stage_job_level_path)

    if direct_metrics.empty and two_stage_aggregate_metrics.empty and two_stage_job_level_metrics.empty:
        raise FileNotFoundError(
            "No forecasting metrics were found. Run the models first."
        )

    targets = set()

    for data in [
        direct_metrics,
        two_stage_aggregate_metrics,
        two_stage_job_level_metrics
    ]:
        if not data.empty:
            targets.update(data["target"].unique())

    comparison_rows = []

    for target in sorted(targets):
        row = {
            "forecast_freq": FORECAST_FREQ,
            "target": target
        }

        candidates = []

        if not direct_metrics.empty:
            direct_target = direct_metrics[
                direct_metrics["target"] == target
            ].copy()

            if not direct_target.empty:
                direct_best = direct_target.sort_values(
                    by="RMSE",
                    ascending=True
                ).iloc[0]

                row["direct_best_model"] = direct_best["model"]
                row["direct_RMSE"] = direct_best["RMSE"]
                row["direct_MAE"] = direct_best["MAE"]
                row["direct_MAPE"] = direct_best["MAPE"]
                row["direct_SMAPE"] = direct_best["SMAPE"]
                row["direct_R2"] = direct_best["R2"]

                candidates.append(
                    (
                        "Direct aggregate forecast",
                        direct_best["RMSE"]
                    )
                )

        if not two_stage_aggregate_metrics.empty:
            two_stage_target = two_stage_aggregate_metrics[
                two_stage_aggregate_metrics["target"] == target
            ].copy()

            if not two_stage_target.empty:
                two_stage_row = two_stage_target.iloc[0]

                row["two_stage_aggregate_model"] = two_stage_row["model"]
                row["two_stage_aggregate_RMSE"] = two_stage_row["RMSE"]
                row["two_stage_aggregate_MAE"] = two_stage_row["MAE"]
                row["two_stage_aggregate_MAPE"] = two_stage_row["MAPE"]
                row["two_stage_aggregate_SMAPE"] = two_stage_row["SMAPE"]
                row["two_stage_aggregate_R2"] = two_stage_row["R2"]

                candidates.append(
                    (
                        "Two-stage aggregate forecast",
                        two_stage_row["RMSE"]
                    )
                )

        if not two_stage_job_level_metrics.empty:
            job_level_target = two_stage_job_level_metrics[
                two_stage_job_level_metrics["target"] == target
            ].copy()

            if not job_level_target.empty:
                job_level_row = job_level_target.iloc[0]

                row["two_stage_job_level_model"] = job_level_row["model"]
                row["two_stage_job_level_RMSE"] = job_level_row["RMSE"]
                row["two_stage_job_level_MAE"] = job_level_row["MAE"]
                row["two_stage_job_level_MAPE"] = job_level_row["MAPE"]
                row["two_stage_job_level_SMAPE"] = job_level_row["SMAPE"]
                row["two_stage_job_level_R2"] = job_level_row["R2"]

                candidates.append(
                    (
                        "Two-stage job-level forecast",
                        job_level_row["RMSE"]
                    )
                )

        if candidates:
            winner = min(candidates, key=lambda item: item[1])[0]
            row["winner_by_RMSE"] = winner

        comparison_rows.append(row)

    comparison_data = pd.DataFrame(comparison_rows)

    output_path = (
        METRICS_DIR / f"all_forecasting_approaches_comparison_{FORECAST_FREQ}.csv"
    )

    comparison_data.to_csv(output_path, index=False)

    print(f"All forecasting approaches comparison saved to: {output_path}")
    print(comparison_data)


if __name__ == "__main__":
    compare_all_forecasting_approaches()