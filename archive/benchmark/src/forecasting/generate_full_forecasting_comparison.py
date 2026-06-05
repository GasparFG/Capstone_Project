from pathlib import Path
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[2]

OUTPUTS_DIR = BASE_DIR / "outputs"
DOCS_DIR = BASE_DIR / "docs" / "forecasting_summary"
DATA_DIR = BASE_DIR / "data" / "processed"

DOCS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# General helpers
# ============================================================

def read_csv_if_exists(path: Path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as error:
            print(f"Could not read {path}: {error}")
            return None
    return None


def save_table(df: pd.DataFrame, filename: str):
    output_path = DOCS_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    return output_path


def round_numeric_columns(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    clean = df.copy()

    for column in clean.columns:
        if pd.api.types.is_numeric_dtype(clean[column]):
            clean[column] = clean[column].round(decimals)

    return clean


def standardize_metric_table(
    df: pd.DataFrame,
    approach: str,
    source_file: str,
    model_override: str | None = None,
    notes: str = ""
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    data = df.copy()

    rename_map = {
        "target_column": "target",
        "Target": "target",
        "Model": "model",
        "metric_type": "metric_type",
        "MSE": "MSE",
        "RMSE": "RMSE",
        "MAE": "MAE",
        "MAPE": "MAPE",
        "SMAPE": "SMAPE",
        "R2": "R2",
        "LOG_R2": "LOG_R2",
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }

    data = data.rename(columns=rename_map)

    if "model" not in data.columns:
        data["model"] = model_override if model_override else approach

    if model_override:
        data["model"] = model_override

    if "target" not in data.columns:
        data["target"] = ""

    wanted_columns = [
        "approach",
        "model",
        "target",
        "metric_type",
        "MSE",
        "RMSE",
        "MAE",
        "MAPE",
        "SMAPE",
        "R2",
        "LOG_R2",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "source_file",
        "notes",
    ]

    data["approach"] = approach
    data["source_file"] = source_file
    data["notes"] = notes

    for column in wanted_columns:
        if column not in data.columns:
            data[column] = np.nan

    return data[wanted_columns]


# ============================================================
# Metric collectors
# ============================================================

def collect_direct_window_metrics():
    """
    Direct aggregate forecast benchmark:
    SARIMA, Prophet, LSTM, XGBoost.
    """
    possible_paths = [
        OUTPUTS_DIR / "window_benchmark" / "all_model_metrics_summary_5min.csv",
        OUTPUTS_DIR / "metrics" / "all_model_metrics_summary_5min.csv",
        OUTPUTS_DIR / "metrics" / "all_model_metrics_summary.csv",
        BASE_DIR / "archive" / "forecasting_old_experiments" / "metrics_backup" / "all_model_metrics_summary_5min.csv",
        BASE_DIR / "archive" / "forecasting_old_experiments" / "metrics_backup" / "all_model_metrics_summary.csv",
    ]

    collected = []

    for path in possible_paths:
        df = read_csv_if_exists(path)

        if df is not None:
            collected.append(
                standardize_metric_table(
                    df=df,
                    approach="Direct Aggregate Window Forecasting",
                    source_file=str(path.relative_to(BASE_DIR)),
                    notes=(
                        "Classical benchmark using jobs aggregated into time windows. "
                        "Used to compare SARIMA, Prophet, LSTM, and XGBoost on aggregate demand."
                    ),
                )
            )

    return pd.concat(collected, ignore_index=True) if collected else pd.DataFrame()


def collect_individual_window_model_metrics():
    """
    Backup collector if the all_model_metrics_summary file is missing.
    """
    candidates = [
        ("SARIMA", OUTPUTS_DIR / "metrics" / "sarima_metrics_5min.csv"),
        ("Prophet", OUTPUTS_DIR / "metrics" / "prophet_metrics_5min.csv"),
        ("LSTM", OUTPUTS_DIR / "metrics" / "lstm_metrics_5min.csv"),
        ("XGBoost", OUTPUTS_DIR / "metrics" / "xgboost_metrics_5min.csv"),

        ("SARIMA", OUTPUTS_DIR / "window_benchmark" / "sarima_metrics_5min.csv"),
        ("Prophet", OUTPUTS_DIR / "window_benchmark" / "prophet_metrics_5min.csv"),
        ("LSTM", OUTPUTS_DIR / "window_benchmark" / "lstm_metrics_5min.csv"),
        ("XGBoost", OUTPUTS_DIR / "window_benchmark" / "xgboost_metrics_5min.csv"),

        ("SARIMA", BASE_DIR / "archive" / "forecasting_old_experiments" / "metrics_backup" / "sarima_metrics_5min.csv"),
        ("Prophet", BASE_DIR / "archive" / "forecasting_old_experiments" / "metrics_backup" / "prophet_metrics_5min.csv"),
        ("LSTM", BASE_DIR / "archive" / "forecasting_old_experiments" / "metrics_backup" / "lstm_metrics_5min.csv"),
        ("XGBoost", BASE_DIR / "archive" / "forecasting_old_experiments" / "metrics_backup" / "xgboost_metrics_5min.csv"),
    ]

    collected = []

    for model_name, path in candidates:
        df = read_csv_if_exists(path)

        if df is not None:
            collected.append(
                standardize_metric_table(
                    df=df,
                    approach="Direct Aggregate Window Forecasting",
                    model_override=model_name,
                    source_file=str(path.relative_to(BASE_DIR)),
                    notes="Individual metric file for a direct aggregate window model.",
                )
            )

    return pd.concat(collected, ignore_index=True) if collected else pd.DataFrame()


def collect_two_stage_window_metrics():
    possible_paths = [
        OUTPUTS_DIR / "two_stage_window" / "two_stage_metrics_5min.csv",
        OUTPUTS_DIR / "metrics" / "two_stage_metrics_5min.csv",
        OUTPUTS_DIR / "metrics" / "two_stage_metrics.csv",
        BASE_DIR / "archive" / "forecasting_old_experiments" / "metrics_backup" / "two_stage_metrics_5min.csv",
        BASE_DIR / "archive" / "forecasting_old_experiments" / "metrics_backup" / "two_stage_metrics.csv",
    ]

    collected = []

    for path in possible_paths:
        df = read_csv_if_exists(path)

        if df is not None:
            collected.append(
                standardize_metric_table(
                    df=df,
                    approach="Two-Stage Window Forecasting",
                    source_file=str(path.relative_to(BASE_DIR)),
                    notes=(
                        "Window-level model that first predicts active windows, "
                        "then predicts aggregate demand for active periods."
                    ),
                )
            )

    return pd.concat(collected, ignore_index=True) if collected else pd.DataFrame()


def collect_two_stage_job_level_metrics():
    possible_paths = [
        OUTPUTS_DIR / "final_job_level" / "two_stage_job_level_metrics_5min.csv",
        OUTPUTS_DIR / "final_job_level" / "two_stage_job_level_internal_metrics_5min.csv",
        OUTPUTS_DIR / "metrics" / "two_stage_job_level_metrics_5min.csv",
        OUTPUTS_DIR / "metrics" / "two_stage_job_level_internal_metrics_5min.csv",
        BASE_DIR / "archive" / "forecasting_old_experiments" / "metrics_backup" / "two_stage_job_level_metrics_5min.csv",
        BASE_DIR / "archive" / "forecasting_old_experiments" / "metrics_backup" / "two_stage_job_level_internal_metrics_5min.csv",
    ]

    collected = []

    for path in possible_paths:
        df = read_csv_if_exists(path)

        if df is not None:
            collected.append(
                standardize_metric_table(
                    df=df,
                    approach="Two-Stage Job-Level XGBoost",
                    source_file=str(path.relative_to(BASE_DIR)),
                    notes=(
                        "Hybrid approach using active windows and job-level XGBoost generation. "
                        "Used before shifting to the final historical profile approach."
                    ),
                )
            )

    return pd.concat(collected, ignore_index=True) if collected else pd.DataFrame()


def collect_baseline_xgboost_job_level_metrics():
    possible_paths = [
        OUTPUTS_DIR / "baseline_xgboost_job_level" / "job_level_model_summary.csv",
        OUTPUTS_DIR / "baseline_xgboost_job_level" / "job_level_regression_metrics.csv",
        OUTPUTS_DIR / "baseline_xgboost_job_level" / "job_level_classification_metrics.csv",
    ]

    collected = []

    for path in possible_paths:
        df = read_csv_if_exists(path)

        if df is not None:
            collected.append(
                standardize_metric_table(
                    df=df,
                    approach="Job-Level XGBoost Capacity Forecasting",
                    source_file=str(path.relative_to(BASE_DIR)),
                    notes=(
                        "Job-level XGBoost model using generated or predicted descriptors. "
                        "Kept as baseline before selecting the profile-based model."
                    ),
                )
            )

    return pd.concat(collected, ignore_index=True) if collected else pd.DataFrame()


def collect_final_profile_metrics():
    possible_paths = [
        OUTPUTS_DIR / "final_job_level" / "job_level_model_summary.csv",
        OUTPUTS_DIR / "final_job_level" / "job_level_regression_metrics.csv",
        OUTPUTS_DIR / "final_job_level" / "job_level_classification_metrics.csv",
    ]

    collected = []

    for path in possible_paths:
        df = read_csv_if_exists(path)

        if df is not None:
            collected.append(
                standardize_metric_table(
                    df=df,
                    approach="Job-Level Workload Forecasting using Historical Profiles",
                    source_file=str(path.relative_to(BASE_DIR)),
                    notes=(
                        "Final selected approach. Uses historical profiles based on role, "
                        "app_name, and job_type to generate job-level capacity requirements."
                    ),
                )
            )

    return pd.concat(collected, ignore_index=True) if collected else pd.DataFrame()


# ============================================================
# Methodology and decision tables
# ============================================================

def create_methodology_table():
    rows = [
        {
            "Order": 1,
            "Approach": "Direct Aggregate Window Forecasting",
            "Models Tested": "SARIMA, Prophet, LSTM, XGBoost",
            "Data Level": "Window-level",
            "What It Predicted": (
                "Aggregate CPU, memory, duration, job_count, batch_count, "
                "interactive_count per time window"
            ),
            "Why We Tried It": (
                "Traditional forecasting setup to compare classical and ML forecasting models."
            ),
            "Main Limitation": "The output is aggregate demand per window, not one row per job.",
            "Decision": "Kept as benchmark.",
        },
        {
            "Order": 2,
            "Approach": "Two-Stage Window Forecasting",
            "Models Tested": "Active-window classifier + aggregate demand model",
            "Data Level": "Window-level",
            "What It Predicted": "Whether a window is active, then aggregate demand for active windows",
            "Why We Tried It": (
                "The dataset had many inactive or low-activity periods, so a classification layer "
                "could reduce noise."
            ),
            "Main Limitation": "Still window-level and not directly aligned with the optimization input.",
            "Decision": "Kept as advanced benchmark.",
        },
        {
            "Order": 3,
            "Approach": "Two-Stage Job-Level XGBoost",
            "Models Tested": "Active-window classifier + job count + job-level XGBoost resource models",
            "Data Level": "Hybrid window/job-level",
            "What It Predicted": (
                "Active windows, number of jobs, then generated job-level CPU, memory, duration, and job_type"
            ),
            "Why We Tried It": (
                "Bridge between aggregate window forecasting and job-level optimization input."
            ),
            "Main Limitation": "Still depended on windows and produced extra complexity.",
            "Decision": "Replaced by cleaner job-level approach.",
        },
        {
            "Order": 4,
            "Approach": "Job-Level XGBoost Capacity Forecasting",
            "Models Tested": "XGBoost regressors/classifiers",
            "Data Level": "Job-level",
            "What It Predicted": (
                "Interarrival time, CPU, memory, duration, job type, and job descriptors"
            ),
            "Why We Tried It": (
                "Directly forecasted individual jobs and their capacity requirements."
            ),
            "Main Limitation": (
                "Some descriptor classifiers, especially app_name, were noisy and slow."
            ),
            "Decision": "Kept as job-level baseline if metrics exist.",
        },
        {
            "Order": 5,
            "Approach": "Job-Level Workload Forecasting using Historical Profiles",
            "Models Tested": "Historical profile forecasting by role, app_name, and job_type",
            "Data Level": "Job-level",
            "What It Predicted": (
                "Individual future jobs with arrival offset, required CPU, required memory, "
                "expected duration, and job type"
            ),
            "Why We Tried It": (
                "The optimization model requires capacity requirements per job, not aggregate windows."
            ),
            "Main Limitation": (
                "Duration and interarrival time remain more variable than CPU and memory."
            ),
            "Decision": "Final selected approach.",
        },
    ]

    methodology = pd.DataFrame(rows)
    save_table(methodology, "01_methodology_path_summary.csv")
    return methodology


def create_final_decision_table():
    rows = [
        {
            "Decision Area": "Final selected forecasting approach",
            "Selected Option": "Job-Level Workload Forecasting using Historical Profiles",
            "Reason": (
                "It directly produces one row per predicted job with required CPU, "
                "required memory, expected duration, job type, and arrival offset."
            ),
        },
        {
            "Decision Area": "Why not final window forecasting?",
            "Selected Option": "Window models kept as benchmarks only",
            "Reason": (
                "Window models predict aggregate demand per time interval, but the optimization phase "
                "requires job-level capacity requirements."
            ),
        },
        {
            "Decision Area": "Primary variables for optimization",
            "Selected Option": "required_cpu and required_memory",
            "Reason": (
                "These represent the core resource capacity requirements used by the optimization model."
            ),
        },
        {
            "Decision Area": "Duration interpretation",
            "Selected Option": "Expected duration estimate",
            "Reason": (
                "Duration is more variable and affected by large outliers, so it should not be presented "
                "as an exact forecast."
            ),
        },
        {
            "Decision Area": "Interarrival interpretation",
            "Selected Option": "Timing reference within the planning horizon",
            "Reason": (
                "Interarrival time is used to place predicted jobs in the next-day horizon rather than "
                "as a highly precise arrival-time forecast."
            ),
        },
    ]

    decision = pd.DataFrame(rows)
    save_table(decision, "04_final_decision_summary.csv")
    return decision


# ============================================================
# Ranking and output summary
# ============================================================

def create_target_best_model_table(all_metrics: pd.DataFrame):
    if all_metrics.empty:
        return pd.DataFrame()

    regression = all_metrics[
        all_metrics["RMSE"].notna()
        & all_metrics["target"].notna()
        & (all_metrics["target"] != "")
    ].copy()

    if regression.empty:
        return pd.DataFrame()

    best_rows = []

    for target, group in regression.groupby("target"):
        group = group.copy()

        if group["R2"].notna().any():
            best = group.sort_values("R2", ascending=False).iloc[0]
            criterion = "Highest R2"
        else:
            best = group.sort_values("RMSE", ascending=True).iloc[0]
            criterion = "Lowest RMSE"

        best_rows.append(
            {
                "Target": target,
                "Best Approach": best["approach"],
                "Best Model": best["model"],
                "Selection Criterion": criterion,
                "RMSE": best.get("RMSE", np.nan),
                "MAE": best.get("MAE", np.nan),
                "R2": best.get("R2", np.nan),
                "LOG_R2": best.get("LOG_R2", np.nan),
                "Source File": best.get("source_file", ""),
            }
        )

    best_models = pd.DataFrame(best_rows)
    best_models = round_numeric_columns(best_models)

    save_table(best_models, "03_best_model_by_target.csv")
    return best_models


def create_output_dataset_summary():
    optimization_path = DATA_DIR / "optimization_input_dataset.parquet"

    if not optimization_path.exists():
        print("Optimization input dataset not found. Skipping output summary.")
        return pd.DataFrame()

    df = pd.read_parquet(optimization_path)

    summary = pd.DataFrame(
        [
            {
                "Output File": "data/processed/optimization_input_dataset.parquet",
                "Rows / Predicted Jobs": len(df),
                "Columns": len(df.columns),
                "Level": "Job-level",
                "Purpose": "Direct input for the optimization model",
            }
        ]
    )

    save_table(summary, "05_optimization_input_dataset_summary.csv")

    preview = df.head(20)
    save_table(preview, "06_optimization_input_dataset_preview.csv")

    structure_rows = []

    descriptions = {
        "predicted_job_id": "Unique ID for each predicted job",
        "arrival_order": "Predicted order of arrival",
        "arrival_offset_minutes": "Arrival position within the next-day planning horizon",
        "required_cpu": "Predicted CPU requirement",
        "required_memory": "Predicted memory requirement",
        "expected_duration_minutes": "Expected duration estimate",
        "role": "Categorical job descriptor",
        "app_name": "Anonymized application identifier",
        "job_type": "Predicted or assigned job type",
        "forecasting_approach": "Approach used to generate the record",
    }

    for column in df.columns:
        structure_rows.append(
            {
                "Column": column,
                "Description": descriptions.get(column, "Generated job-level field"),
                "Example": df[column].iloc[0] if len(df) > 0 else "",
            }
        )

    structure = pd.DataFrame(structure_rows)
    save_table(structure, "07_optimization_input_dataset_structure.csv")

    return summary


# ============================================================
# Professor message draft
# ============================================================

def create_professor_message_from_tables():
    message = """
Subject: Forecasting Approach Summary and Model Comparison

Hi Professor,

I wanted to share a structured summary of the forecasting work completed for our capstone project.

We evaluated multiple forecasting paths. First, we tested direct aggregate window forecasting using models such as SARIMA, Prophet, LSTM, and XGBoost. In that stage, jobs were grouped into time windows and the models forecasted aggregate CPU demand, memory demand, duration, and job counts per interval.

Then, we tested a two-stage window-based approach. This added a first layer to classify whether a time window would be active, followed by a second layer to forecast aggregate demand for active windows. This helped us evaluate whether separating active and inactive periods improved aggregate workload forecasting.

After that, we explored job-level forecasting approaches because the optimization phase requires capacity requirements per job rather than aggregate demand per time window. The final selected approach was Job-Level Workload Forecasting using Historical Profiles. This model uses historical workload profiles based on role, app_name, and job_type to generate a next-day workload forecast at the individual job level.

The final output is:

data/processed/optimization_input_dataset.parquet

This file contains one row per predicted job and includes the arrival offset, required CPU, required memory, expected duration, job type, role, and app_name. This format is directly aligned with the optimization model input.

The comparison tables were generated directly from the model outputs and saved under:

docs/forecasting_summary/

The main files are:
- 01_methodology_path_summary.csv
- 02_all_available_forecasting_metrics.csv
- 03_best_model_by_target.csv
- 04_final_decision_summary.csv
- 05_optimization_input_dataset_summary.csv
- 06_optimization_input_dataset_preview.csv
- 07_optimization_input_dataset_structure.csv

Best regards,
Monica
""".strip()

    output_path = DOCS_DIR / "08_professor_message_draft.md"

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(message)

    print(f"Saved: {output_path}")


# ============================================================
# Terminal summary
# ============================================================

def print_terminal_summary(all_metrics: pd.DataFrame):
    print("\n" + "=" * 120)
    print("FULL FORECASTING COMPARISON SUMMARY")
    print("=" * 120)

    if all_metrics.empty:
        print("No metric files were found.")
        return

    display_columns = [
        "approach",
        "model",
        "target",
        "RMSE",
        "MAE",
        "MAPE",
        "R2",
        "LOG_R2",
        "accuracy",
        "f1_macro",
        "source_file",
    ]

    existing_columns = [
        column for column in display_columns
        if column in all_metrics.columns
    ]

    print("\nALL COLLECTED METRICS")
    print("-" * 120)

    sorted_metrics = all_metrics.copy()

    sort_columns = [
        column for column in ["approach", "target", "model"]
        if column in sorted_metrics.columns
    ]

    if sort_columns:
        sorted_metrics = sorted_metrics.sort_values(sort_columns)

    print(sorted_metrics[existing_columns].to_string(index=False))

    regression = all_metrics[
        all_metrics["RMSE"].notna()
        & all_metrics["target"].notna()
        & (all_metrics["target"] != "")
    ].copy()

    if not regression.empty:
        best_rows = []

        for target, group in regression.groupby("target"):
            if group["R2"].notna().any():
                best = group.sort_values("R2", ascending=False).iloc[0]
                criterion = "Highest R2"
            else:
                best = group.sort_values("RMSE", ascending=True).iloc[0]
                criterion = "Lowest RMSE"

            best_rows.append(
                {
                    "target": target,
                    "best_approach": best["approach"],
                    "best_model": best["model"],
                    "criterion": criterion,
                    "RMSE": best.get("RMSE", np.nan),
                    "MAE": best.get("MAE", np.nan),
                    "R2": best.get("R2", np.nan),
                    "LOG_R2": best.get("LOG_R2", np.nan),
                    "source_file": best.get("source_file", ""),
                }
            )

        best_table = pd.DataFrame(best_rows)
        best_table = round_numeric_columns(best_table)

        print("\n" + "=" * 120)
        print("BEST MODEL BY TARGET")
        print("=" * 120)
        print(best_table.to_string(index=False))

    classification = all_metrics[
        all_metrics["accuracy"].notna()
        | all_metrics["f1_macro"].notna()
    ].copy()

    if not classification.empty:
        print("\n" + "=" * 120)
        print("CLASSIFICATION METRICS")
        print("=" * 120)

        class_columns = [
            column for column in [
                "approach",
                "model",
                "target",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "source_file",
            ]
            if column in classification.columns
        ]

        print(classification[class_columns].to_string(index=False))

    optimization_path = DATA_DIR / "optimization_input_dataset.parquet"

    if optimization_path.exists():
        optimization_data = pd.read_parquet(optimization_path)

        print("\n" + "=" * 120)
        print("FINAL OPTIMIZATION INPUT DATASET")
        print("=" * 120)
        print(f"File: {optimization_path}")
        print(f"Shape: {optimization_data.shape}")
        print("\nPreview:")
        print(optimization_data.head(10).to_string(index=False))

    print("\n" + "=" * 120)
    print("FINAL DECISION")
    print("=" * 120)
    print(
        "Final selected approach: Job-Level Workload Forecasting using Historical Profiles\n"
        "Reason: It generates one row per predicted job with required CPU, required memory, "
        "expected duration, job type, and arrival offset. This is the structure needed by "
        "the optimization model.\n"
        "Note: Window-based SARIMA, Prophet, LSTM, XGBoost, and two-stage approaches were used "
        "as benchmarks/comparisons."
    )


# ============================================================
# Main
# ============================================================

def generate_full_forecasting_comparison():
    print("Generating full forecasting comparison...")

    create_methodology_table()

    collected_tables = []

    collectors = [
        collect_direct_window_metrics,
        collect_individual_window_model_metrics,
        collect_two_stage_window_metrics,
        collect_two_stage_job_level_metrics,
        collect_baseline_xgboost_job_level_metrics,
        collect_final_profile_metrics,
    ]

    for collector in collectors:
        result = collector()
        if result is not None and not result.empty:
            collected_tables.append(result)

    if collected_tables:
        all_metrics = pd.concat(collected_tables, ignore_index=True)
        all_metrics = round_numeric_columns(all_metrics)
        save_table(all_metrics, "02_all_available_forecasting_metrics.csv")
        create_target_best_model_table(all_metrics)
    else:
        print("No metric files found. Creating empty metrics table.")
        all_metrics = pd.DataFrame()
        save_table(all_metrics, "02_all_available_forecasting_metrics.csv")

    create_final_decision_table()
    create_output_dataset_summary()
    create_professor_message_from_tables()

    print_terminal_summary(all_metrics)

    print("\nDone. Tables saved in:")
    print(DOCS_DIR)


if __name__ == "__main__":
    generate_full_forecasting_comparison()