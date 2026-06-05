import pandas as pd
from config import OUTPUTS_DIR


def evaluate_job_level_model():
    regression_path = OUTPUTS_DIR / "job_level_regression_metrics.csv"
    classification_path = OUTPUTS_DIR / "job_level_classification_metrics.csv"
    if not regression_path.exists():
        raise FileNotFoundError(f"Missing {regression_path}. Run train_job_level_capacity_model.py first.")
    if not classification_path.exists():
        raise FileNotFoundError(f"Missing {classification_path}. Run train_job_level_capacity_model.py first.")

    regression = pd.read_csv(regression_path)
    classification = pd.read_csv(classification_path)
    rows = []
    for _, r in regression.iterrows():
        role = "Primary optimization capacity variable" if r["target"] in ["cpu_request", "memory_request"] else "Timing/duration support variable"
        rows.append({
            "target": r["target"],
            "metric_type": "regression",
            "model": r["model"],
            "RMSE": r["RMSE"],
            "MAE": r["MAE"],
            "MAPE": r["MAPE"],
            "SMAPE": r["SMAPE"],
            "R2": r["R2"],
            "LOG_R2": r["LOG_R2"],
            "role": role,
        })
    for _, r in classification.iterrows():
        rows.append({
            "target": r["target"],
            "metric_type": "classification",
            "model": r["model"],
            "accuracy": r["accuracy"],
            "precision_macro": r["precision_macro"],
            "recall_macro": r["recall_macro"],
            "f1_macro": r["f1_macro"],
            "role": "Job type descriptor prediction",
        })
    summary = pd.DataFrame(rows)
    summary_path = OUTPUTS_DIR / "job_level_model_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Job-level model summary saved to: {summary_path}")
    print(summary)


if __name__ == "__main__":
    evaluate_job_level_model()
