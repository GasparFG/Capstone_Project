from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_DATA = (
    BASE_DIR
    / "data"
    / "interim"
    / "cleaned_data_extended_90_days.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "forecasting_experiments"
DOCS_DIR = BASE_DIR / "docs" / "forecasting_summary"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

TARGET_COLUMNS = [
    "cpu_request",
    "memory_request",
    "duration_minutes",
]

BASE_TIME_FEATURES = [
    "hour",
    "day_of_week",
    "is_weekend",
    "time_of_day_minutes",
    "time_sin",
    "time_cos",
    "day_sin",
    "day_cos",
]

PROFILE_GROUPS = [
    ["role", "app_name", "job_type"],
    ["role", "job_type"],
    ["app_name", "job_type"],
    ["app_name"],
    ["job_type"],
    ["role"],
]


# ============================================================
# METRICS
# ============================================================

def calculate_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = y_true != 0

    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_metrics(y_true, y_pred):
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0)

    mse = mean_squared_error(y_true, y_pred)

    return {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": calculate_mape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def add_metrics_to_row(row, prefix, metrics_dict):
    for metric_name, value in metrics_dict.items():
        row[f"{metric_name} {prefix}"] = value

    return row


# ============================================================
# DATA PREPARATION
# ============================================================

def load_and_prepare_data():
    if not INPUT_DATA.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_DATA}"
        )

    data = pd.read_csv(INPUT_DATA)

    required_columns = [
        "instance_sn",
        "role",
        "app_name",
        "cpu_request",
        "memory_request",
        "scheduled_time",
        "duration_minutes",
        "job_type",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    for column in ["role", "app_name", "job_type"]:
        data[column] = (
            data[column]
            .astype(str)
            .str.lower()
            .str.strip()
        )

    data["scheduled_time_td"] = pd.to_timedelta(data["scheduled_time"])
    data["scheduled_minutes"] = (
        data["scheduled_time_td"].dt.total_seconds() / 60
    )

    data = data.sort_values("scheduled_minutes").reset_index(drop=True)

    for column in TARGET_COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        ).fillna(0)

        data[column] = data[column].clip(lower=0)

    data["day_index"] = (data["scheduled_minutes"] // 1440).astype(int)
    data["week_index"] = (data["scheduled_minutes"] // (7 * 1440)).astype(int)

    data["hour"] = ((data["scheduled_minutes"] // 60) % 24).astype(int)
    data["minute"] = (data["scheduled_minutes"] % 60).astype(int)
    data["day_of_week"] = (data["day_index"] % 7).astype(int)

    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)

    data["time_of_day_minutes"] = data["hour"] * 60 + data["minute"]

    data["time_sin"] = np.sin(
        2 * np.pi * data["time_of_day_minutes"] / 1440
    )

    data["time_cos"] = np.cos(
        2 * np.pi * data["time_of_day_minutes"] / 1440
    )

    data["day_sin"] = np.sin(
        2 * np.pi * data["day_of_week"] / 7
    )

    data["day_cos"] = np.cos(
        2 * np.pi * data["day_of_week"] / 7
    )

    data["interarrival_minutes"] = (
        data["scheduled_minutes"]
        .diff()
        .fillna(data["scheduled_minutes"].diff().median())
        .clip(lower=0.01)
    )

    print(f"Input data used: {INPUT_DATA}")
    print(f"Prepared data shape: {data.shape}")
    print(f"Available weeks: {sorted(data['week_index'].unique())}")

    return data


# ============================================================
# EXPERIMENT SPLITS
# ============================================================

def build_experiments(data):
    experiments = []

    experiment_definitions = [
        {
            "experiment": 1,
            "train_weeks": [0, 1, 2],
            "test_weeks": [3],
        },
        {
            "experiment": 2,
            "train_weeks": [4, 5, 6],
            "test_weeks": [7],
        },
        {
            "experiment": 3,
            "train_weeks": [8, 9, 10],
            "test_weeks": [11],
        },
    ]

    for definition in experiment_definitions:
        train_data = data[
            data["week_index"].isin(definition["train_weeks"])
        ].copy()

        test_data = data[
            data["week_index"].isin(definition["test_weeks"])
        ].copy()

        if train_data.empty or test_data.empty:
            print(
                f"Skipping experiment {definition['experiment']} "
                "because training or testing data is empty."
            )
            continue

        experiments.append(
            {
                "experiment": definition["experiment"],
                "train_weeks": definition["train_weeks"],
                "test_weeks": definition["test_weeks"],
                "train_data": train_data,
                "test_data": test_data,
            }
        )

    if not experiments:
        raise ValueError(
            "No valid experiments were created. "
            "Check week_index values in the dataset."
        )

    return experiments


# ============================================================
# HISTORICAL PROFILE FEATURES
# ============================================================

def add_profile_features(train_data, data_to_transform, target):
    transformed = data_to_transform.copy()

    global_median = train_data[target].median()

    for index, group_columns in enumerate(PROFILE_GROUPS, start=1):
        profile_column = f"{target}_profile_median_{index}"
        count_column = f"{target}_profile_count_{index}"

        profile_table = (
            train_data
            .groupby(group_columns)
            .agg(
                profile_prediction=(target, "median"),
                profile_count=(target, "count")
            )
            .reset_index()
        )

        profile_table = profile_table.rename(
            columns={
                "profile_prediction": profile_column,
                "profile_count": count_column,
            }
        )

        transformed = transformed.merge(
            profile_table,
            on=group_columns,
            how="left"
        )

        transformed[profile_column] = (
            transformed[profile_column]
            .fillna(global_median)
        )

        transformed[count_column] = (
            transformed[count_column]
            .fillna(0)
        )

    return transformed


def get_feature_matrix(train_data, data_to_transform, target):
    transformed = add_profile_features(
        train_data=train_data,
        data_to_transform=data_to_transform,
        target=target
    )

    profile_features = []

    for index in range(1, len(PROFILE_GROUPS) + 1):
        profile_features.append(f"{target}_profile_median_{index}")
        profile_features.append(f"{target}_profile_count_{index}")

    feature_columns = BASE_TIME_FEATURES + profile_features

    # Professor feedback:
    # duration should be estimated using its relationship with CPU and memory.
    if target == "duration_minutes":
        transformed["cpu_memory_ratio"] = (
            transformed["memory_request"]
            / transformed["cpu_request"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

        feature_columns = (
            feature_columns
            + [
                "cpu_request",
                "memory_request",
                "cpu_memory_ratio",
            ]
        )

    X = transformed[feature_columns].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X, feature_columns


def historical_profile_predict(train_data, data_to_predict, target):
    transformed = add_profile_features(
        train_data=train_data,
        data_to_transform=data_to_predict,
        target=target
    )

    # Most specific profile:
    # role + app_name + job_type
    prediction = transformed[f"{target}_profile_median_1"].values

    prediction = np.maximum(prediction, 0)

    return prediction


# ============================================================
# MODEL TRAINING AND TUNING
# ============================================================

def tune_ridge(X_train, y_train):
    split_index = int(len(X_train) * 0.8)

    X_fit = X_train.iloc[:split_index]
    y_fit = y_train.iloc[:split_index]

    X_val = X_train.iloc[split_index:]
    y_val = y_train.iloc[split_index:]

    candidates = [
        ("Ridge_alpha_0.1", Ridge(alpha=0.1)),
        ("Ridge_alpha_1.0", Ridge(alpha=1.0)),
        ("Ridge_alpha_10.0", Ridge(alpha=10.0)),
    ]

    return select_best_candidate(
        candidates=candidates,
        X_fit=X_fit,
        y_fit=y_fit,
        X_val=X_val,
        y_val=y_val,
        X_full=X_train,
        y_full=y_train,
    )


def tune_random_forest(X_train, y_train):
    split_index = int(len(X_train) * 0.8)

    X_fit = X_train.iloc[:split_index]
    y_fit = y_train.iloc[:split_index]

    X_val = X_train.iloc[split_index:]
    y_val = y_train.iloc[split_index:]

    candidates = [
        (
            "RF_n30_depth8_leaf2",
            RandomForestRegressor(
                n_estimators=30,
                max_depth=8,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "RF_n60_depth12_leaf1",
            RandomForestRegressor(
                n_estimators=60,
                max_depth=12,
                min_samples_leaf=1,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]

    return select_best_candidate(
        candidates=candidates,
        X_fit=X_fit,
        y_fit=y_fit,
        X_val=X_val,
        y_val=y_val,
        X_full=X_train,
        y_full=y_train,
    )


def tune_xgboost(X_train, y_train):
    if not XGBOOST_AVAILABLE:
        return None, []

    split_index = int(len(X_train) * 0.8)

    X_fit = X_train.iloc[:split_index]
    y_fit = y_train.iloc[:split_index]

    X_val = X_train.iloc[split_index:]
    y_val = y_train.iloc[split_index:]

    candidates = [
        (
            "XGB_n60_depth3_lr0.05",
            XGBRegressor(
                objective="reg:squarederror",
                n_estimators=60,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.90,
                colsample_bytree=0.90,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist",
            ),
        ),
        (
            "XGB_n120_depth4_lr0.03",
            XGBRegressor(
                objective="reg:squarederror",
                n_estimators=120,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist",
            ),
        ),
    ]

    return select_best_candidate(
        candidates=candidates,
        X_fit=X_fit,
        y_fit=y_fit,
        X_val=X_val,
        y_val=y_val,
        X_full=X_train,
        y_full=y_train,
    )


def select_best_candidate(
    candidates,
    X_fit,
    y_fit,
    X_val,
    y_val,
    X_full,
    y_full,
):
    best_result = None
    selection_details = []

    for candidate_name, model in candidates:
        model.fit(X_fit, y_fit)

        validation_prediction = model.predict(X_val)
        validation_rmse = np.sqrt(
            mean_squared_error(y_val, validation_prediction)
        )

        selection_details.append(
            {
                "candidate": candidate_name,
                "validation_RMSE": validation_rmse,
            }
        )

        if best_result is None:
            best_result = {
                "candidate": candidate_name,
                "model": model,
                "validation_RMSE": validation_rmse,
            }
        elif validation_rmse < best_result["validation_RMSE"]:
            best_result = {
                "candidate": candidate_name,
                "model": model,
                "validation_RMSE": validation_rmse,
            }

    best_model = best_result["model"]
    best_model.fit(X_full, y_full)

    best_result["model"] = best_model

    for detail in selection_details:
        detail["selected"] = (
            detail["candidate"] == best_result["candidate"]
        )

    return best_result, selection_details


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_predictions(
    experiment_number,
    target,
    model_name,
    selected_candidate,
    y_train,
    train_prediction,
    y_test,
    test_prediction,
):
    training_metrics = calculate_metrics(y_train, train_prediction)
    testing_metrics = calculate_metrics(y_test, test_prediction)

    row = {
        "Experiment": experiment_number,
        "Target": target,
        "Model": model_name,
        "Selected Candidate": selected_candidate,
    }

    row = add_metrics_to_row(
        row=row,
        prefix="Training",
        metrics_dict=training_metrics
    )

    row = add_metrics_to_row(
        row=row,
        prefix="Testing",
        metrics_dict=testing_metrics
    )

    return row


def run_single_target(
    experiment_number,
    train_data,
    test_data,
    target,
):
    result_rows = []
    tuning_rows = []

    y_train = train_data[target]
    y_test = test_data[target]

    X_train, feature_columns = get_feature_matrix(
        train_data=train_data,
        data_to_transform=train_data,
        target=target
    )

    X_test, _ = get_feature_matrix(
        train_data=train_data,
        data_to_transform=test_data,
        target=target
    )

    ensemble_train_predictions = []
    ensemble_test_predictions = []
    ensemble_members = []

    # --------------------------------------------------------
    # Historical Profile
    # --------------------------------------------------------

    historical_train_prediction = historical_profile_predict(
        train_data=train_data,
        data_to_predict=train_data,
        target=target
    )

    historical_test_prediction = historical_profile_predict(
        train_data=train_data,
        data_to_predict=test_data,
        target=target
    )

    result_rows.append(
        evaluate_predictions(
            experiment_number=experiment_number,
            target=target,
            model_name="HistoricalProfile",
            selected_candidate="Median by role/app_name/job_type",
            y_train=y_train,
            train_prediction=historical_train_prediction,
            y_test=y_test,
            test_prediction=historical_test_prediction,
        )
    )

    ensemble_train_predictions.append(historical_train_prediction)
    ensemble_test_predictions.append(historical_test_prediction)
    ensemble_members.append("HistoricalProfile")

    # --------------------------------------------------------
    # Ridge
    # --------------------------------------------------------

    ridge_result, ridge_tuning = tune_ridge(X_train, y_train)

    ridge_train_prediction = ridge_result["model"].predict(X_train)
    ridge_test_prediction = ridge_result["model"].predict(X_test)

    result_rows.append(
        evaluate_predictions(
            experiment_number=experiment_number,
            target=target,
            model_name="Ridge",
            selected_candidate=ridge_result["candidate"],
            y_train=y_train,
            train_prediction=ridge_train_prediction,
            y_test=y_test,
            test_prediction=ridge_test_prediction,
        )
    )

    for row in ridge_tuning:
        row.update(
            {
                "Experiment": experiment_number,
                "Target": target,
                "Model": "Ridge",
            }
        )
        tuning_rows.append(row)

    ensemble_train_predictions.append(ridge_train_prediction)
    ensemble_test_predictions.append(ridge_test_prediction)
    ensemble_members.append("Ridge")

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    rf_result, rf_tuning = tune_random_forest(X_train, y_train)

    rf_train_prediction = rf_result["model"].predict(X_train)
    rf_test_prediction = rf_result["model"].predict(X_test)

    result_rows.append(
        evaluate_predictions(
            experiment_number=experiment_number,
            target=target,
            model_name="RandomForest",
            selected_candidate=rf_result["candidate"],
            y_train=y_train,
            train_prediction=rf_train_prediction,
            y_test=y_test,
            test_prediction=rf_test_prediction,
        )
    )

    for row in rf_tuning:
        row.update(
            {
                "Experiment": experiment_number,
                "Target": target,
                "Model": "RandomForest",
            }
        )
        tuning_rows.append(row)

    ensemble_train_predictions.append(rf_train_prediction)
    ensemble_test_predictions.append(rf_test_prediction)
    ensemble_members.append("RandomForest")

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    if XGBOOST_AVAILABLE:
        xgb_result, xgb_tuning = tune_xgboost(X_train, y_train)

        xgb_train_prediction = xgb_result["model"].predict(X_train)
        xgb_test_prediction = xgb_result["model"].predict(X_test)

        result_rows.append(
            evaluate_predictions(
                experiment_number=experiment_number,
                target=target,
                model_name="XGBoost",
                selected_candidate=xgb_result["candidate"],
                y_train=y_train,
                train_prediction=xgb_train_prediction,
                y_test=y_test,
                test_prediction=xgb_test_prediction,
            )
        )

        for row in xgb_tuning:
            row.update(
                {
                    "Experiment": experiment_number,
                    "Target": target,
                    "Model": "XGBoost",
                }
            )
            tuning_rows.append(row)

        ensemble_train_predictions.append(xgb_train_prediction)
        ensemble_test_predictions.append(xgb_test_prediction)
        ensemble_members.append("XGBoost")

    # --------------------------------------------------------
    # Ensemble
    # --------------------------------------------------------

    ensemble_train_prediction = np.mean(
        np.vstack(ensemble_train_predictions),
        axis=0
    )

    ensemble_test_prediction = np.mean(
        np.vstack(ensemble_test_predictions),
        axis=0
    )

    result_rows.append(
        evaluate_predictions(
            experiment_number=experiment_number,
            target=target,
            model_name="Ensemble_Average",
            selected_candidate="Average of " + ", ".join(ensemble_members),
            y_train=y_train,
            train_prediction=ensemble_train_prediction,
            y_test=y_test,
            test_prediction=ensemble_test_prediction,
        )
    )

    return result_rows, tuning_rows, feature_columns


# ============================================================
# DURATION CORRELATION
# ============================================================

def duration_correlation_analysis(experiment_number, train_data, test_data):
    return {
        "Experiment": experiment_number,
        "corr_duration_cpu_training": train_data["duration_minutes"].corr(
            train_data["cpu_request"]
        ),
        "corr_duration_memory_training": train_data["duration_minutes"].corr(
            train_data["memory_request"]
        ),
        "corr_duration_cpu_testing": test_data["duration_minutes"].corr(
            test_data["cpu_request"]
        ),
        "corr_duration_memory_testing": test_data["duration_minutes"].corr(
            test_data["memory_request"]
        ),
    }


# ============================================================
# MAIN EXPERIMENT RUNNER
# ============================================================

def run_forecasting_experiments():
    data = load_and_prepare_data()
    experiments = build_experiments(data)

    all_metric_rows = []
    all_tuning_rows = []
    all_correlation_rows = []
    feature_rows = []

    print("\nRunning forecasting experiments...")
    print("=" * 100)

    for experiment in experiments:
        experiment_number = experiment["experiment"]
        train_data = experiment["train_data"]
        test_data = experiment["test_data"]

        print(
            f"\nExperiment {experiment_number}: "
            f"Training weeks {experiment['train_weeks']} "
            f"Testing weeks {experiment['test_weeks']}"
        )

        print(f"Training rows: {len(train_data)}")
        print(f"Testing rows: {len(test_data)}")

        all_correlation_rows.append(
            duration_correlation_analysis(
                experiment_number=experiment_number,
                train_data=train_data,
                test_data=test_data
            )
        )

        for target in TARGET_COLUMNS:
            print(f"  Target: {target}")

            metric_rows, tuning_rows, feature_columns = run_single_target(
                experiment_number=experiment_number,
                train_data=train_data,
                test_data=test_data,
                target=target,
            )

            all_metric_rows.extend(metric_rows)
            all_tuning_rows.extend(tuning_rows)

            for feature in feature_columns:
                feature_rows.append(
                    {
                        "Experiment": experiment_number,
                        "Target": target,
                        "Feature": feature,
                    }
                )

    metrics_table = pd.DataFrame(all_metric_rows)
    tuning_table = pd.DataFrame(all_tuning_rows)
    correlation_table = pd.DataFrame(all_correlation_rows)
    feature_table = pd.DataFrame(feature_rows)

    metrics_table = metrics_table.round(4)
    tuning_table = tuning_table.round(4)
    correlation_table = correlation_table.round(4)

    metrics_output = OUTPUT_DIR / "experiment_metrics_training_testing.csv"
    tuning_output = OUTPUT_DIR / "grid_search_selection_details.csv"
    correlation_output = OUTPUT_DIR / "duration_correlation_analysis.csv"
    feature_output = OUTPUT_DIR / "experiment_feature_columns.csv"

    metrics_table.to_csv(metrics_output, index=False)
    tuning_table.to_csv(tuning_output, index=False)
    correlation_table.to_csv(correlation_output, index=False)
    feature_table.to_csv(feature_output, index=False)

    # Also save copies for professor summary
    metrics_table.to_csv(
        DOCS_DIR / "09_experiment_metrics_training_testing.csv",
        index=False
    )

    tuning_table.to_csv(
        DOCS_DIR / "10_grid_search_selection_details.csv",
        index=False
    )

    correlation_table.to_csv(
        DOCS_DIR / "11_duration_correlation_analysis.csv",
        index=False
    )

    print("\n" + "=" * 100)
    print("EXPERIMENT METRICS: TRAINING VS TESTING")
    print("=" * 100)
    print(metrics_table.to_string(index=False))

    print("\n" + "=" * 100)
    print("GRID SEARCH / PARAMETER TUNING DETAILS")
    print("=" * 100)
    print(tuning_table.to_string(index=False))

    print("\n" + "=" * 100)
    print("DURATION CORRELATION WITH CPU AND MEMORY")
    print("=" * 100)
    print(correlation_table.to_string(index=False))

    print("\nFiles saved to:")
    print(OUTPUT_DIR)
    print(DOCS_DIR)


if __name__ == "__main__":
    run_forecasting_experiments()