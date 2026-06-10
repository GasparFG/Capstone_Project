from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# 1. STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Data Center Optimization Dashboard",
    page_icon="📊",
    layout="wide",
)


st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-size: 17px;
    }

    .stApp {
        font-size: 17px;
    }

    h1 {
        font-size: 2.4rem !important;
    }

    h2 {
        font-size: 2rem !important;
    }

    h3 {
        font-size: 1.45rem !important;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 1rem !important;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
    }

    .overview-section {
        border: 1px solid rgba(250, 250, 250, 0.12);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 18px;
        background-color: rgba(255, 255, 255, 0.03);
    }

    .overview-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 14px;
    }

    .center-title {
        text-align: center;
    }

    .big-total-cost {
        font-size: 2.85rem;
        font-weight: 850;
        line-height: 1.1;
        margin-bottom: 4px;
        text-align: center;
    }

    .big-total-cost-label {
        font-size: 1rem;
        opacity: 0.75;
        margin-bottom: 16px;
        text-align: center;
    }

    .status-subtitle {
        font-size: 0.95rem;
        opacity: 0.8;
        margin-top: -8px;
        margin-bottom: 10px;
    }

    .server-grid-card {
        border-radius: 10px;
        padding: 14px 8px;
        margin: 4px 0px;
        text-align: center;
        font-weight: 700;
        color: #111111;
        min-height: 72px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.25);
    }

    .server-grid-id {
        font-size: 16px;
        margin-bottom: 4px;
    }

    .server-grid-status {
        font-size: 11px;
        text-transform: uppercase;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. PATH CONFIGURATION
# ============================================================

def get_project_root() -> Path:
    """
    Returns project root assuming this file is inside:
        Capstone_Project/src/visualization/dashboard.py
    """
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = get_project_root()


def get_output_priority_dirs() -> list[Path]:
    """
    Folder priority for every output table.

    The dashboard searches in this order:
        1. outputs/optimization
        2. outputs/results/tables
    """

    return [
        PROJECT_ROOT / "outputs" / "optimization",
        PROJECT_ROOT / "outputs" / "results" / "tables",
    ]


OUTPUT_PRIORITY_DIRS = get_output_priority_dirs()


# ============================================================
# 3. FILE DISCOVERY AND LOADING
# ============================================================

def find_matching_files_with_priority(patterns: list[str]) -> list[Path]:
    """
    Finds files using folder priority.

    For each table type, the first folder with matching readable files wins.
    """

    for folder in OUTPUT_PRIORITY_DIRS:
        if not folder.exists():
            continue

        matching_files = []

        for pattern in patterns:
            matching_files.extend(folder.glob(pattern))

        matching_files = sorted(set(matching_files))

        valid_files = []

        for file_path in matching_files:
            try:
                if file_path.stat().st_size > 0:
                    valid_files.append(file_path)
            except OSError:
                continue

        if valid_files:
            return valid_files

    return []


def read_dashboard_table(file_path: Path) -> pd.DataFrame:
    """
    Reads CSV or Excel files with safer CSV handling.
    """

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        try:
            return pd.read_csv(file_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(
                file_path,
                encoding="latin1",
                sep=None,
                engine="python",
            )

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)

    raise ValueError(f"Unsupported file type: {file_path.suffix}")


def infer_dashboard_scenario_from_filename(
    file_name: str,
    table_prefix: str,
) -> str:
    """
    Infers scenario name from filenames like:
        performance_metrics_base_20260609-024827.csv
        optimization_solution_high_demand_20260609-024827.csv
        server_summary_reduced_capacity_20260609-024827.xlsx
    """

    stem = Path(file_name).stem

    if stem.startswith(table_prefix):
        scenario_and_time = stem.replace(table_prefix, "", 1).strip("_")
    else:
        scenario_and_time = stem

    parts = scenario_and_time.split("_")

    if not parts:
        return "unknown"

    if "-" in parts[-1]:
        parts = parts[:-1]

    if parts and parts[-1].isdigit() and len(parts[-1]) >= 6:
        parts = parts[:-1]

    scenario = "_".join(parts).strip("_")

    if not scenario:
        return "unknown"

    return scenario


def read_and_combine_dashboard_files(
    patterns: list[str],
    table_prefix: str,
) -> pd.DataFrame:
    """
    Reads all files matching patterns from the first priority folder
    where that table type exists.
    """

    dashboard_files = find_matching_files_with_priority(patterns)

    if not dashboard_files:
        return pd.DataFrame()

    dashboard_dataframes = []

    for file_path in dashboard_files:
        try:
            df_dashboard_table = read_dashboard_table(file_path)

            df_dashboard_table["source_file"] = file_path.name
            df_dashboard_table["source_path"] = str(file_path)
            df_dashboard_table["file_modified_time"] = pd.to_datetime(
                file_path.stat().st_mtime,
                unit="s",
            )

            if "scenario" not in df_dashboard_table.columns:
                df_dashboard_table["scenario"] = infer_dashboard_scenario_from_filename(
                    file_name=file_path.name,
                    table_prefix=table_prefix,
                )

            dashboard_dataframes.append(df_dashboard_table)

        except pd.errors.EmptyDataError:
            continue

        except Exception as error:
            st.warning(
                f"Could not read file: {file_path.name}. Error: {error}. "
                f"Path: {file_path}"
            )

    if not dashboard_dataframes:
        return pd.DataFrame()

    return pd.concat(dashboard_dataframes, ignore_index=True)


def keep_latest_dashboard_run_per_scenario(
    df_dashboard_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keeps only the latest file per scenario.
    """

    if df_dashboard_table.empty:
        return df_dashboard_table

    required_columns = {
        "scenario",
        "source_file",
        "file_modified_time",
    }

    if not required_columns.issubset(df_dashboard_table.columns):
        return df_dashboard_table

    df_dashboard_latest_files = (
        df_dashboard_table[
            [
                "scenario",
                "source_file",
                "file_modified_time",
            ]
        ]
        .drop_duplicates()
        .sort_values("file_modified_time")
        .groupby("scenario", as_index=False)
        .tail(1)
    )

    df_dashboard_latest = df_dashboard_table.merge(
        df_dashboard_latest_files[
            [
                "scenario",
                "source_file",
            ]
        ],
        on=[
            "scenario",
            "source_file",
        ],
        how="inner",
    )

    return df_dashboard_latest


@st.cache_data(ttl=60)
def load_all_available_dashboard_outputs() -> dict[str, pd.DataFrame]:
    """
    Loads all currently available optimization outputs.

    File priority:
        outputs/optimization first
        outputs/results/tables second
    """

    df_dashboard_performance = keep_latest_dashboard_run_per_scenario(
        read_and_combine_dashboard_files(
            patterns=[
                "performance_metrics_*.csv",
                "performance_metrics_*.xlsx",
                "*performance_metrics*.csv",
                "*performance_metrics*.xlsx",
            ],
            table_prefix="performance_metrics",
        )
    )

    df_dashboard_solution = keep_latest_dashboard_run_per_scenario(
        read_and_combine_dashboard_files(
            patterns=[
                "optimization_solution_*.csv",
                "optimization_solution_*.xlsx",
                "*optimization_solution*.csv",
                "*optimization_solution*.xlsx",
            ],
            table_prefix="optimization_solution",
        )
    )

    df_dashboard_server_summary = keep_latest_dashboard_run_per_scenario(
        read_and_combine_dashboard_files(
            patterns=[
                "server_summary_*.csv",
                "server_summary_*.xlsx",
                "*server_summary*.csv",
                "*server_summary*.xlsx",
            ],
            table_prefix="server_summary",
        )
    )

    df_dashboard_hourly_energy = keep_latest_dashboard_run_per_scenario(
        read_and_combine_dashboard_files(
            patterns=[
                "hourly_energy_thermal_*.csv",
                "hourly_energy_thermal_*.xlsx",
                "*hourly_energy_thermal*.csv",
                "*hourly_energy_thermal*.xlsx",
            ],
            table_prefix="hourly_energy_thermal",
        )
    )

    df_dashboard_server_load = keep_latest_dashboard_run_per_scenario(
        read_and_combine_dashboard_files(
            patterns=[
                "server_load_timeseries_*.csv",
                "server_load_timeseries_*.xlsx",
                "*server_load_timeseries*.csv",
                "*server_load_timeseries*.xlsx",
            ],
            table_prefix="server_load_timeseries",
        )
    )

    return {
        "performance": df_dashboard_performance,
        "solution": df_dashboard_solution,
        "server_summary": df_dashboard_server_summary,
        "hourly_energy": df_dashboard_hourly_energy,
        "server_load": df_dashboard_server_load,
    }


# ============================================================
# 4. GENERAL HELPERS
# ============================================================

def require_dashboard_columns(
    df_dashboard_table: pd.DataFrame,
    required_columns: list[str],
    table_name: str,
) -> bool:
    """
    Checks whether a DataFrame contains required columns.
    """

    if df_dashboard_table.empty:
        st.warning(f"No data found for: {table_name}")
        return False

    missing_columns = [
        column
        for column in required_columns
        if column not in df_dashboard_table.columns
    ]

    if missing_columns:
        st.warning(
            f"Table '{table_name}' is missing columns: {missing_columns}"
        )

        with st.expander(f"Available columns in {table_name}"):
            st.write(list(df_dashboard_table.columns))

        return False

    return True


def safe_dashboard_metric_value(
    dashboard_row: pd.Series,
    column: str,
    default=0,
):
    """
    Safely extracts a value from a row.
    """

    if column not in dashboard_row:
        return default

    value = dashboard_row[column]

    if pd.isna(value):
        return default

    return value


def filter_dashboard_by_scenario(
    df_dashboard_table: pd.DataFrame,
    scenario: str,
) -> pd.DataFrame:
    """
    Filters a DataFrame by one scenario.
    """

    if df_dashboard_table.empty:
        return pd.DataFrame()

    if "scenario" not in df_dashboard_table.columns:
        return pd.DataFrame()

    return df_dashboard_table[
        df_dashboard_table["scenario"] == scenario
    ].copy()


def filter_dashboard_by_scenarios(
    df_dashboard_table: pd.DataFrame,
    scenarios: list[str],
) -> pd.DataFrame:
    """
    Filters a DataFrame by multiple scenarios.
    """

    if df_dashboard_table.empty:
        return pd.DataFrame()

    if "scenario" not in df_dashboard_table.columns:
        return pd.DataFrame()

    return df_dashboard_table[
        df_dashboard_table["scenario"].isin(scenarios)
    ].copy()


def format_dashboard_currency(value) -> str:
    """
    Formats a value as currency.
    """

    try:
        return f"${float(value):,.2f}"
    except Exception:
        return "$0.00"


def format_dashboard_number(value, decimals: int = 2) -> str:
    """
    Formats a number safely.
    """

    try:
        return f"{float(value):,.{decimals}f}"
    except Exception:
        return f"{0:.{decimals}f}"


def get_dashboard_unique_jobs(
    df_dashboard_solution: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns unique jobs if job_id exists.
    """

    if df_dashboard_solution.empty:
        return df_dashboard_solution

    if "job_id" not in df_dashboard_solution.columns:
        return df_dashboard_solution

    return df_dashboard_solution.drop_duplicates(
        subset=["job_id"]
    ).copy()


def classify_dashboard_daily_server_status(
    df_dashboard_server_group: pd.DataFrame,
) -> str:
    """
    Classifies each server once for the whole day.

    Priority:
        maintenance > active > idle
    """

    statuses = set(df_dashboard_server_group["status"].dropna())

    if "maintenance" in statuses:
        return "maintenance"

    if "active" in statuses:
        return "active"

    return "idle"


def build_dashboard_status_occurrence_counts(
    df_dashboard_server_load: pd.DataFrame,
) -> pd.DataFrame:
    """
    Counts server-time-slot records by status.
    """

    if df_dashboard_server_load.empty:
        return pd.DataFrame()

    if "status" not in df_dashboard_server_load.columns:
        return pd.DataFrame()

    df_dashboard_status_occurrences = (
        df_dashboard_server_load
        .groupby("status")
        .size()
        .reset_index(name="server_slot_count")
    )

    return df_dashboard_status_occurrences


def build_dashboard_unique_server_status_counts(
    df_dashboard_server_load: pd.DataFrame,
) -> pd.DataFrame:
    """
    Counts unique servers that appeared at least once in each status.

    A server can be counted in multiple statuses.
    """

    if df_dashboard_server_load.empty:
        return pd.DataFrame()

    required_columns = {
        "server_id",
        "status",
    }

    if not required_columns.issubset(df_dashboard_server_load.columns):
        return pd.DataFrame()

    df_dashboard_unique_status_counts = (
        df_dashboard_server_load
        .groupby("status")["server_id"]
        .nunique()
        .reset_index(name="unique_server_count")
    )

    return df_dashboard_unique_status_counts


def build_dashboard_daily_server_classification(
    df_dashboard_server_load: pd.DataFrame,
) -> pd.DataFrame:
    """
    Builds one final daily status per server.

    Priority:
        maintenance > active > idle
    """

    if df_dashboard_server_load.empty:
        return pd.DataFrame()

    required_columns = {
        "server_id",
        "status",
    }

    if not required_columns.issubset(df_dashboard_server_load.columns):
        return pd.DataFrame()

    df_dashboard_daily_status = (
        df_dashboard_server_load
        .groupby("server_id")
        .apply(classify_dashboard_daily_server_status, include_groups=False)
        .reset_index(name="daily_status")
    )

    return df_dashboard_daily_status


def build_dashboard_daily_server_status_counts(
    df_dashboard_server_load: pd.DataFrame,
) -> pd.DataFrame:
    """
    Counts servers by final daily classification.

    Each server is counted once.
    """

    df_dashboard_daily_status = build_dashboard_daily_server_classification(
        df_dashboard_server_load
    )

    if df_dashboard_daily_status.empty:
        return pd.DataFrame()

    df_dashboard_daily_status_counts = (
        df_dashboard_daily_status
        .groupby("daily_status")["server_id"]
        .nunique()
        .reset_index(name="server_count")
    )

    return df_dashboard_daily_status_counts


def get_dashboard_status_color(status: str) -> str:
    """
    Returns a color for each server status.
    """

    status_colors = {
        "active": "#2ecc71",
        "idle": "#95a5a6",
        "maintenance": "#f1c40f",
    }

    return status_colors.get(str(status).lower(), "#7f8c8d")


def render_dashboard_server_status_grid(
    df_dashboard_server_load_slot: pd.DataFrame,
    servers_per_row: int = 7,
) -> None:
    """
    Renders a compact server grid.
    Each square represents one server and its status at the selected slot.
    """

    if df_dashboard_server_load_slot.empty:
        st.warning("No server data available for the selected slot.")
        return

    df_dashboard_grid = (
        df_dashboard_server_load_slot[
            [
                "server_id",
                "status",
            ]
        ]
        .drop_duplicates(subset=["server_id"])
        .sort_values("server_id")
        .copy()
    )

    dashboard_servers = df_dashboard_grid.to_dict("records")

    for row_start in range(0, len(dashboard_servers), servers_per_row):
        dashboard_row_servers = dashboard_servers[
            row_start: row_start + servers_per_row
        ]

        dashboard_columns = st.columns(servers_per_row)

        for index, dashboard_server in enumerate(dashboard_row_servers):
            dashboard_server_id = dashboard_server["server_id"]
            dashboard_status = dashboard_server["status"]
            dashboard_color = get_dashboard_status_color(dashboard_status)

            with dashboard_columns[index]:
                st.markdown(
                    f"""
                    <div class="server-grid-card" style="background-color:{dashboard_color};">
                        <div class="server-grid-id">Server {dashboard_server_id}</div>
                        <div class="server-grid-status">{dashboard_status}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        for empty_index in range(len(dashboard_row_servers), servers_per_row):
            with dashboard_columns[empty_index]:
                st.empty()


# ============================================================
# 5. LOAD DATA
# ============================================================

dashboard_data = load_all_available_dashboard_outputs()

df_dashboard_performance = dashboard_data["performance"]
df_dashboard_solution = dashboard_data["solution"]
df_dashboard_server_summary = dashboard_data["server_summary"]
df_dashboard_hourly_energy = dashboard_data["hourly_energy"]
df_dashboard_server_load = dashboard_data["server_load"]


# ============================================================
# 6. SIDEBAR
# ============================================================

st.sidebar.title("Dashboard Controls")

if st.sidebar.button("Refresh outputs"):
    st.cache_data.clear()
    st.rerun()


with st.sidebar.expander("Output folders priority"):
    for index, folder in enumerate(OUTPUT_PRIORITY_DIRS, start=1):
        status = "Found" if folder.exists() else "Not found"
        st.write(f"{index}. {status}: `{folder}`")


with st.sidebar.expander("Loaded tables debug"):
    for table_name, df_dashboard_table in dashboard_data.items():
        st.write(f"**{table_name}**: {len(df_dashboard_table)} rows")

        if not df_dashboard_table.empty and "source_file" in df_dashboard_table.columns:
            source_files = (
                df_dashboard_table["source_file"]
                .drop_duplicates()
                .sort_values()
                .tolist()
            )
            st.write(source_files)

        if not df_dashboard_table.empty and "scenario" in df_dashboard_table.columns:
            scenarios = sorted(
                df_dashboard_table["scenario"].dropna().unique()
            )
            st.write(f"Scenarios: {scenarios}")


# ============================================================
# 7. STOP IF NO PERFORMANCE DATA
# ============================================================

if df_dashboard_performance.empty:
    st.title("Data Center Optimization Dashboard")

    st.warning("No optimization performance files found yet.")

    st.write("The dashboard is currently scanning these folders in priority order:")

    for folder in OUTPUT_PRIORITY_DIRS:
        st.code(str(folder))

    st.write("Expected file examples:")

    st.code(
        """
performance_metrics_base_20260609-024827.csv
optimization_solution_base_20260609-024827.csv
server_summary_base_20260609-024827.csv
hourly_energy_thermal_base_20260609-024827.csv
server_load_timeseries_base_20260609-024827.csv
        """.strip()
    )

    st.stop()


if "scenario" not in df_dashboard_performance.columns:
    st.error("The performance metrics data does not contain a scenario column.")
    st.stop()


available_dashboard_scenarios = sorted(
    df_dashboard_performance["scenario"].dropna().unique()
)

if not available_dashboard_scenarios:
    st.warning("No scenarios found in the available output files.")
    st.stop()


st.sidebar.metric(
    "Scenarios available",
    len(available_dashboard_scenarios),
)


main_dashboard_scenario = st.sidebar.selectbox(
    "Main scenario for detailed analysis",
    options=available_dashboard_scenarios,
)


comparison_dashboard_scenarios = st.sidebar.multiselect(
    "Scenarios for comparison",
    options=available_dashboard_scenarios,
    default=available_dashboard_scenarios[:3],
    max_selections=3,
)


if not comparison_dashboard_scenarios:
    st.warning("Select at least one scenario for comparison.")
    st.stop()


with st.sidebar.expander("Available scenarios"):
    dashboard_scenario_columns = [
        column for column in [
            "scenario",
            "status",
            "has_solution",
            "total_cost",
            "average_pue",
            "jobs_late_count",
            "file_modified_time",
            "source_file",
        ]
        if column in df_dashboard_performance.columns
    ]

    st.dataframe(
        df_dashboard_performance[dashboard_scenario_columns].sort_values(
            "file_modified_time",
            ascending=False,
        ),
        width="stretch",
    )


# ============================================================
# 8. FILTER DATA
# ============================================================

df_dashboard_performance_main = filter_dashboard_by_scenario(
    df_dashboard_performance,
    main_dashboard_scenario,
)

df_dashboard_solution_main = filter_dashboard_by_scenario(
    df_dashboard_solution,
    main_dashboard_scenario,
)

df_dashboard_server_summary_main = filter_dashboard_by_scenario(
    df_dashboard_server_summary,
    main_dashboard_scenario,
)

df_dashboard_hourly_energy_main = filter_dashboard_by_scenario(
    df_dashboard_hourly_energy,
    main_dashboard_scenario,
)

df_dashboard_server_load_main = filter_dashboard_by_scenario(
    df_dashboard_server_load,
    main_dashboard_scenario,
)

df_dashboard_performance_comparison = filter_dashboard_by_scenarios(
    df_dashboard_performance,
    comparison_dashboard_scenarios,
)

df_dashboard_solution_comparison = filter_dashboard_by_scenarios(
    df_dashboard_solution,
    comparison_dashboard_scenarios,
)

df_dashboard_server_summary_comparison = filter_dashboard_by_scenarios(
    df_dashboard_server_summary,
    comparison_dashboard_scenarios,
)

df_dashboard_hourly_energy_comparison = filter_dashboard_by_scenarios(
    df_dashboard_hourly_energy,
    comparison_dashboard_scenarios,
)

df_dashboard_server_load_comparison = filter_dashboard_by_scenarios(
    df_dashboard_server_load,
    comparison_dashboard_scenarios,
)


# ============================================================
# 9. TITLE
# ============================================================

st.title("Data Center Job Scheduling Optimization Dashboard")

st.caption(
    f"Main scenario: {main_dashboard_scenario} | "
    f"Comparison scenarios: {', '.join(comparison_dashboard_scenarios)}"
)


# ============================================================
# 10. TABS
# ============================================================

(
    tab_overview,
    tab_jobs,
    tab_servers,
    tab_energy,
    tab_maintenance,
    tab_compare,
    tab_library,
) = st.tabs(
    [
        "Overview",
        "Jobs & Demand",
        "Server Utilization",
        "Energy & PUE",
        "Maintenance",
        "Scenario Comparison",
        "Scenario Library",
    ]
)


# ============================================================
# 11. OVERVIEW TAB
# ============================================================

with tab_overview:
    st.header(f"Overview: {main_dashboard_scenario}")

    if df_dashboard_performance_main.empty:
        st.warning("No performance metrics available for this scenario.")
    else:
        dashboard_metrics = df_dashboard_performance_main.iloc[0]

        dashboard_total_cost = safe_dashboard_metric_value(
            dashboard_metrics,
            "total_cost",
        )

        dashboard_energy_cost = safe_dashboard_metric_value(
            dashboard_metrics,
            "energy_cost",
        )

        dashboard_pm_cost = safe_dashboard_metric_value(
            dashboard_metrics,
            "pm_cost",
        )

        dashboard_expected_cm_cost = safe_dashboard_metric_value(
            dashboard_metrics,
            "expected_cm_cost",
        )

        dashboard_switching_cost = safe_dashboard_metric_value(
            dashboard_metrics,
            "switching_cost",
        )

        dashboard_lateness_cost = safe_dashboard_metric_value(
            dashboard_metrics,
            "lateness_cost",
        )

        dashboard_jobs_in_forecast = safe_dashboard_metric_value(
            dashboard_metrics,
            "jobs_in_forecast",
        )

        dashboard_jobs_placed = safe_dashboard_metric_value(
            dashboard_metrics,
            "jobs_placed",
        )

        dashboard_jobs_on_time = safe_dashboard_metric_value(
            dashboard_metrics,
            "jobs_on_time",
        )

        dashboard_jobs_late = safe_dashboard_metric_value(
            dashboard_metrics,
            "jobs_late_count",
        )

        dashboard_total_energy = safe_dashboard_metric_value(
            dashboard_metrics,
            "total_facility_energy_kwh",
        )

        dashboard_average_pue = safe_dashboard_metric_value(
            dashboard_metrics,
            "average_pue",
        )

        dashboard_max_pue = safe_dashboard_metric_value(
            dashboard_metrics,
            "max_pue",
        )

        dashboard_total_pue = dashboard_average_pue

        # -----------------------------
        # COST BLOCK
        # -----------------------------

        st.markdown('<div class="overview-section">', unsafe_allow_html=True)
        st.markdown(
            '<div class="overview-title center-title">Cost Summary</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="big-total-cost">{format_dashboard_currency(dashboard_total_cost)}</div>
            <div class="big-total-cost-label">Total cost</div>
            """,
            unsafe_allow_html=True,
        )

        df_dashboard_cost_breakdown = pd.DataFrame(
            {
                "scenario": [main_dashboard_scenario],
                "energy_cost": [dashboard_energy_cost],
                "pm_cost": [dashboard_pm_cost],
                "expected_cm_cost": [dashboard_expected_cm_cost],
                "switching_cost": [dashboard_switching_cost],
                "lateness_cost": [dashboard_lateness_cost],
            }
        )

        df_dashboard_cost_breakdown_pct = df_dashboard_cost_breakdown.melt(
            id_vars="scenario",
            var_name="cost_type",
            value_name="cost",
        )

        df_dashboard_cost_breakdown_pct["cost_type"] = (
            df_dashboard_cost_breakdown_pct["cost_type"]
            .replace(
                {
                    "energy_cost": "Energy",
                    "pm_cost": "PM",
                    "expected_cm_cost": "CM",
                    "switching_cost": "Switch",
                    "lateness_cost": "Lateness",
                }
            )
        )

        dashboard_cost_sum = df_dashboard_cost_breakdown_pct["cost"].sum()

        if dashboard_cost_sum > 0:
            df_dashboard_cost_breakdown_pct["percentage"] = (
                df_dashboard_cost_breakdown_pct["cost"]
                / dashboard_cost_sum
                * 100
            )
        else:
            df_dashboard_cost_breakdown_pct["percentage"] = 0

        chart_left_space, chart_middle_space, chart_right_space = st.columns(
            [1, 3, 1]
        )

        with chart_middle_space:
            fig_dashboard_cost_breakdown_pct = px.bar(
                df_dashboard_cost_breakdown_pct,
                x="percentage",
                y="scenario",
                color="cost_type",
                orientation="h",
                text=df_dashboard_cost_breakdown_pct["percentage"].map(
                    lambda value: f"{value:.1f}%"
                ),
                title="Cost Breakdown",
            )

            fig_dashboard_cost_breakdown_pct.update_layout(
                barmode="stack",
                xaxis_title="Cost Share (%)",
                yaxis_title="",
                xaxis=dict(range=[0, 100]),
                legend_title_text="Cost Type",
                height=260,
                margin=dict(l=20, r=20, t=60, b=40),
                title_x=0.5,
            )

            st.plotly_chart(
                fig_dashboard_cost_breakdown_pct,
                width="stretch",
            )

        cost_col1, cost_col2, cost_col3, cost_col4, cost_col5 = st.columns(5)

        cost_col1.metric(
            "Energy Cost",
            format_dashboard_currency(dashboard_energy_cost),
        )

        cost_col2.metric(
            "PM Cost",
            format_dashboard_currency(dashboard_pm_cost),
        )

        cost_col3.metric(
            "CM Cost",
            format_dashboard_currency(dashboard_expected_cm_cost),
        )

        cost_col4.metric(
            "Switch Cost",
            format_dashboard_currency(dashboard_switching_cost),
        )

        cost_col5.metric(
            "Lateness Cost",
            format_dashboard_currency(dashboard_lateness_cost),
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # -----------------------------
        # JOBS BLOCK
        # -----------------------------

        st.markdown('<div class="overview-section">', unsafe_allow_html=True)
        st.markdown(
            '<div class="overview-title">Jobs Summary</div>',
            unsafe_allow_html=True,
        )

        jobs_col1, jobs_col2, jobs_col3, jobs_col4 = st.columns(4)

        jobs_col1.metric(
            "Jobs in Forecast",
            int(dashboard_jobs_in_forecast),
        )

        jobs_col2.metric(
            "Jobs Placed",
            int(dashboard_jobs_placed),
        )

        jobs_col3.metric(
            "Jobs On Time",
            int(dashboard_jobs_on_time),
        )

        jobs_col4.metric(
            "Late Jobs",
            int(dashboard_jobs_late),
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # -----------------------------
        # ENERGY BLOCK
        # -----------------------------

        st.markdown('<div class="overview-section">', unsafe_allow_html=True)
        st.markdown(
            '<div class="overview-title">Energy & PUE Summary</div>',
            unsafe_allow_html=True,
        )

        energy_col1, energy_col2, energy_col3, energy_col4 = st.columns(4)

        energy_col1.metric(
            "Total Energy",
            f"{format_dashboard_number(dashboard_total_energy, 3)} kWh",
        )

        energy_col2.metric(
            "Total PUE",
            format_dashboard_number(dashboard_total_pue, 3),
        )

        energy_col3.metric(
            "Average PUE",
            format_dashboard_number(dashboard_average_pue, 3),
        )

        energy_col4.metric(
            "Max PUE",
            format_dashboard_number(dashboard_max_pue, 3),
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # -----------------------------
        # SERVER BLOCK
        # -----------------------------

        st.markdown('<div class="overview-section">', unsafe_allow_html=True)
        st.markdown(
            '<div class="overview-title">Top Servers by Utilization</div>',
            unsafe_allow_html=True,
        )

        if not df_dashboard_server_summary_main.empty and {
            "server_id",
            "utilization_rate_pct",
        }.issubset(df_dashboard_server_summary_main.columns):

            df_dashboard_top_5_servers = (
                df_dashboard_server_summary_main[
                    [
                        "server_id",
                        "utilization_rate_pct",
                    ]
                ]
                .sort_values(
                    "utilization_rate_pct",
                    ascending=False,
                )
                .head(5)
                .copy()
            )

            df_dashboard_top_5_servers = df_dashboard_top_5_servers.rename(
                columns={
                    "server_id": "Server",
                    "utilization_rate_pct": "Utilization",
                }
            )

            df_dashboard_top_5_servers["Utilization"] = (
                df_dashboard_top_5_servers["Utilization"]
                .map(lambda value: f"{float(value):.2f}%")
            )

            table_left_space, table_middle_space, table_right_space = st.columns(
                [1.3, 1.4, 1.3]
            )

            with table_middle_space:
                st.dataframe(
                    df_dashboard_top_5_servers,
                    width="stretch",
                    hide_index=True,
                )

        else:
            st.warning("No server summary available for top server table.")

        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# 12. JOBS & DEMAND TAB
# ============================================================

with tab_jobs:
    st.header(f"Jobs & Demand: {main_dashboard_scenario}")

    if df_dashboard_solution_main.empty:
        st.warning("No optimization solution data available for this scenario.")

        with st.expander("Debug: solution table"):
            st.write(
                f"Rows in all solution data: {len(df_dashboard_solution)}")

            if not df_dashboard_solution.empty:
                st.write("Columns:")
                st.write(list(df_dashboard_solution.columns))

                if "scenario" in df_dashboard_solution.columns:
                    st.write("Scenarios in solution table:")
                    st.write(
                        sorted(
                            df_dashboard_solution["scenario"]
                            .dropna()
                            .unique()
                        )
                    )

                if "source_file" in df_dashboard_solution.columns:
                    st.write("Source files:")
                    st.write(
                        df_dashboard_solution["source_file"]
                        .drop_duplicates()
                        .tolist()
                    )
    else:
        dashboard_job_metric = st.radio(
            "Job metric",
            [
                "Job Type Proportion",
                "Jobs by Hour and Type",
            ],
            horizontal=True,
            key="job_metric",
        )

        if dashboard_job_metric == "Job Type Proportion":
            if require_dashboard_columns(
                df_dashboard_solution_main,
                [
                    "job_id",
                    "job_type",
                ],
                "optimization_solution",
            ):
                df_dashboard_unique_jobs = get_dashboard_unique_jobs(
                    df_dashboard_solution_main
                )

                df_dashboard_job_type_counts = (
                    df_dashboard_unique_jobs
                    .groupby("job_type")
                    .size()
                    .reset_index(name="job_count")
                )

                fig_dashboard_job_type = px.pie(
                    df_dashboard_job_type_counts,
                    names="job_type",
                    values="job_count",
                    title="Proportion of Job Types",
                )

                st.plotly_chart(fig_dashboard_job_type, width="stretch")

                st.dataframe(
                    df_dashboard_job_type_counts,
                    width="stretch",
                )

        elif dashboard_job_metric == "Jobs by Hour and Type":
            if require_dashboard_columns(
                df_dashboard_solution_main,
                [
                    "job_id",
                    "job_type",
                    "start_slot",
                ],
                "optimization_solution",
            ):
                df_dashboard_unique_jobs = get_dashboard_unique_jobs(
                    df_dashboard_solution_main
                )

                df_dashboard_unique_jobs["start_hour"] = (
                    df_dashboard_unique_jobs["start_slot"]
                )

                df_dashboard_jobs_by_hour_type = (
                    df_dashboard_unique_jobs
                    .groupby(
                        [
                            "start_hour",
                            "job_type",
                        ]
                    )
                    .size()
                    .reset_index(name="job_count")
                )

                fig_dashboard_jobs_by_hour = px.bar(
                    df_dashboard_jobs_by_hour_type,
                    x="start_hour",
                    y="job_count",
                    color="job_type",
                    barmode="stack",
                    title="Jobs by Hour and Type",
                )

                fig_dashboard_jobs_by_hour.update_layout(
                    xaxis_title="Hour / Slot",
                    yaxis_title="Number of Jobs",
                )

                st.plotly_chart(
                    fig_dashboard_jobs_by_hour,
                    width="stretch",
                )

                st.dataframe(
                    df_dashboard_jobs_by_hour_type,
                    width="stretch",
                )


# ============================================================
# 13. SERVER UTILIZATION TAB
# ============================================================

with tab_servers:
    st.header(f"Server Utilization: {main_dashboard_scenario}")

    dashboard_server_metric = st.radio(
        "Server metric",
        [
            "Utilization Summary",
            "Server Load Over Time",
            "Batch vs Interactive Load",
            "Peak Hours",
            "Status Heatmap",
        ],
        horizontal=True,
        key="server_metric",
    )

    if dashboard_server_metric == "Utilization Summary":
        if df_dashboard_server_summary_main.empty:
            st.warning("No server summary data available.")
        else:
            if require_dashboard_columns(
                df_dashboard_server_summary_main,
                [
                    "server_id",
                    "utilization_rate_pct",
                ],
                "server_summary",
            ):
                fig_dashboard_utilization = px.bar(
                    df_dashboard_server_summary_main,
                    x="server_id",
                    y="utilization_rate_pct",
                    color=(
                        "server_type"
                        if "server_type"
                        in df_dashboard_server_summary_main.columns
                        else None
                    ),
                    text_auto=".2f",
                    title="Server Utilization Rate",
                )

                fig_dashboard_utilization.update_layout(
                    xaxis_title="Server ID",
                    yaxis_title="Utilization Rate (%)",
                )

                st.plotly_chart(
                    fig_dashboard_utilization,
                    width="stretch",
                )

                st.dataframe(
                    df_dashboard_server_summary_main.sort_values(
                        "utilization_rate_pct",
                        ascending=False,
                    ),
                    width="stretch",
                )

    elif dashboard_server_metric == "Server Load Over Time":
        if df_dashboard_server_load_main.empty:
            st.warning("No server load timeseries available.")
        else:
            if require_dashboard_columns(
                df_dashboard_server_load_main,
                [
                    "server_id",
                    "time",
                    "load",
                ],
                "server_load_timeseries",
            ):
                dashboard_available_servers = sorted(
                    df_dashboard_server_load_main["server_id"].unique()
                )

                dashboard_selected_servers = st.multiselect(
                    "Select servers",
                    options=dashboard_available_servers,
                    default=dashboard_available_servers[:5],
                )

                df_dashboard_filtered_load = df_dashboard_server_load_main[
                    df_dashboard_server_load_main["server_id"].isin(
                        dashboard_selected_servers
                    )
                ]

                fig_dashboard_load_line = px.line(
                    df_dashboard_filtered_load,
                    x="time",
                    y="load",
                    color="server_id",
                    title="Server Load Over Time",
                )

                fig_dashboard_load_line.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Load",
                )

                st.plotly_chart(
                    fig_dashboard_load_line,
                    width="stretch",
                )

    elif dashboard_server_metric == "Batch vs Interactive Load":
        if df_dashboard_server_summary_main.empty:
            st.warning("No server summary data available.")
        else:
            if require_dashboard_columns(
                df_dashboard_server_summary_main,
                [
                    "server_id",
                    "batch_avg_load",
                    "interactive_avg_load",
                ],
                "server_summary",
            ):
                df_dashboard_avg_load_type = (
                    df_dashboard_server_summary_main[
                        [
                            "server_id",
                            "batch_avg_load",
                            "interactive_avg_load",
                        ]
                    ].melt(
                        id_vars="server_id",
                        value_vars=[
                            "batch_avg_load",
                            "interactive_avg_load",
                        ],
                        var_name="load_type",
                        value_name="average_load",
                    )
                )

                fig_dashboard_load_type = px.bar(
                    df_dashboard_avg_load_type,
                    x="server_id",
                    y="average_load",
                    color="load_type",
                    barmode="group",
                    title="Average Batch vs Interactive Load by Server",
                )

                fig_dashboard_load_type.update_layout(
                    xaxis_title="Server ID",
                    yaxis_title="Average Load",
                )

                st.plotly_chart(
                    fig_dashboard_load_type,
                    width="stretch",
                )

    elif dashboard_server_metric == "Peak Hours":
        if df_dashboard_server_load_main.empty:
            st.warning("No server load timeseries available.")
        else:
            if require_dashboard_columns(
                df_dashboard_server_load_main,
                [
                    "time",
                    "active",
                    "load",
                    "batch_load",
                    "interactive_load",
                ],
                "server_load_timeseries",
            ):
                df_dashboard_peak_hours = (
                    df_dashboard_server_load_main
                    .groupby("time")
                    .agg(
                        active_servers=("active", "sum"),
                        total_load=("load", "sum"),
                        batch_load=("batch_load", "sum"),
                        interactive_load=("interactive_load", "sum"),
                    )
                    .reset_index()
                    .sort_values(
                        [
                            "active_servers",
                            "total_load",
                        ],
                        ascending=False,
                    )
                )

                fig_dashboard_peak_hours = px.line(
                    df_dashboard_peak_hours.sort_values("time"),
                    x="time",
                    y="active_servers",
                    title="Active Servers Over Time",
                )

                fig_dashboard_peak_hours.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Active Servers",
                )

                st.plotly_chart(
                    fig_dashboard_peak_hours,
                    width="stretch",
                )

                st.subheader("Top Peak Hours")

                st.dataframe(
                    df_dashboard_peak_hours.head(10),
                    width="stretch",
                )

    elif dashboard_server_metric == "Status Heatmap":
        if df_dashboard_server_load_main.empty:
            st.warning("No server load timeseries available.")
        else:
            if require_dashboard_columns(
                df_dashboard_server_load_main,
                [
                    "server_id",
                    "slot",
                    "time",
                    "status",
                ],
                "server_load_timeseries",
            ):
                st.subheader("Server Status Grid")

                df_dashboard_server_load_main["slot"] = pd.to_numeric(
                    df_dashboard_server_load_main["slot"],
                    errors="coerce",
                )

                df_dashboard_slots = (
                    df_dashboard_server_load_main[
                        [
                            "slot",
                            "time",
                        ]
                    ]
                    .dropna(subset=["slot"])
                    .drop_duplicates()
                    .sort_values("slot")
                )

                dashboard_slot_options = df_dashboard_slots["slot"].tolist()

                dashboard_selected_slot = st.slider(
                    "Select time slot",
                    min_value=int(min(dashboard_slot_options)),
                    max_value=int(max(dashboard_slot_options)),
                    value=int(min(dashboard_slot_options)),
                    step=1,
                )

                dashboard_selected_time = (
                    df_dashboard_slots[
                        df_dashboard_slots["slot"] == dashboard_selected_slot
                    ]["time"]
                    .iloc[0]
                )

                st.caption(
                    f"Selected slot: {dashboard_selected_slot} | Time: {dashboard_selected_time}"
                )

                legend_col1, legend_col2, legend_col3 = st.columns(3)

                with legend_col1:
                    st.markdown(
                        '<span style="background-color:#2ecc71; color:#111; padding:6px 12px; border-radius:6px;">Active</span>',
                        unsafe_allow_html=True,
                    )

                with legend_col2:
                    st.markdown(
                        '<span style="background-color:#95a5a6; color:#111; padding:6px 12px; border-radius:6px;">Idle</span>',
                        unsafe_allow_html=True,
                    )

                with legend_col3:
                    st.markdown(
                        '<span style="background-color:#f1c40f; color:#111; padding:6px 12px; border-radius:6px;">Maintenance</span>',
                        unsafe_allow_html=True,
                    )

                st.write("")

                df_dashboard_server_load_slot = df_dashboard_server_load_main[
                    df_dashboard_server_load_main["slot"] == dashboard_selected_slot
                ].copy()

                render_dashboard_server_status_grid(
                    df_dashboard_server_load_slot=df_dashboard_server_load_slot,
                    servers_per_row=7,
                )


# ============================================================
# 14. ENERGY & PUE TAB
# ============================================================

with tab_energy:
    st.header(f"Energy & PUE: {main_dashboard_scenario}")

    if df_dashboard_hourly_energy_main.empty:
        st.warning("No hourly energy data available for this scenario.")
    else:
        dashboard_energy_metric = st.radio(
            "Energy metric",
            [
                "PUE Variation",
                "Power Consumption",
            ],
            horizontal=True,
            key="energy_metric",
        )

        if dashboard_energy_metric == "PUE Variation":
            if require_dashboard_columns(
                df_dashboard_hourly_energy_main,
                [
                    "time",
                    "PUE",
                ],
                "hourly_energy_thermal",
            ):
                fig_dashboard_pue = px.line(
                    df_dashboard_hourly_energy_main,
                    x="time",
                    y="PUE",
                    title="PUE Over Time",
                )

                fig_dashboard_pue.update_layout(
                    xaxis_title="Time",
                    yaxis_title="PUE",
                )

                st.plotly_chart(
                    fig_dashboard_pue,
                    width="stretch",
                )

        elif dashboard_energy_metric == "Power Consumption":
            if require_dashboard_columns(
                df_dashboard_hourly_energy_main,
                [
                    "time",
                    "PIT_W",
                    "Pcool_W",
                    "Ptot_W",
                ],
                "hourly_energy_thermal",
            ):
                df_dashboard_power = (
                    df_dashboard_hourly_energy_main[
                        [
                            "time",
                            "PIT_W",
                            "Pcool_W",
                            "Ptot_W",
                        ]
                    ].melt(
                        id_vars="time",
                        value_vars=[
                            "PIT_W",
                            "Pcool_W",
                            "Ptot_W",
                        ],
                        var_name="power_type",
                        value_name="power_watts",
                    )
                )

                fig_dashboard_power = px.line(
                    df_dashboard_power,
                    x="time",
                    y="power_watts",
                    color="power_type",
                    title="IT Power, Cooling Power, and Total Facility Power",
                )

                fig_dashboard_power.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Power",
                )

                st.plotly_chart(
                    fig_dashboard_power,
                    width="stretch",
                )


# ============================================================
# 15. MAINTENANCE TAB
# ============================================================

with tab_maintenance:
    st.header(f"Maintenance: {main_dashboard_scenario}")

    dashboard_maintenance_metric = st.radio(
        "Maintenance metric",
        [
            "PM Schedule",
            "Status Views",
        ],
        horizontal=True,
        key="maintenance_metric",
    )

    if dashboard_maintenance_metric == "PM Schedule":
        if df_dashboard_server_summary_main.empty:
            st.warning("No maintenance data available.")
        else:
            if (
                "pm_scheduled" in df_dashboard_server_summary_main.columns
                and "server_id" in df_dashboard_server_summary_main.columns
            ):
                dashboard_servers_with_pm = int(
                    df_dashboard_server_summary_main["pm_scheduled"].sum()
                )

                dashboard_total_servers = (
                    df_dashboard_server_summary_main["server_id"].nunique()
                )

                col1, col2 = st.columns(2)

                col1.metric(
                    "Servers with PM",
                    dashboard_servers_with_pm,
                )

                col2.metric(
                    "Total Servers",
                    dashboard_total_servers,
                )

            df_dashboard_pm_display = (
                df_dashboard_server_summary_main.copy()
            )

            if "pm_scheduled" in df_dashboard_pm_display.columns:
                df_dashboard_pm_display = df_dashboard_pm_display[
                    df_dashboard_pm_display["pm_scheduled"] == 1
                ]

            dashboard_pm_columns = [
                column
                for column in [
                    "scenario",
                    "server_id",
                    "server_type",
                    "pm_scheduled",
                    "pm_start_time",
                    "pm_end_time",
                    "utilization_rate_pct",
                    "final_status",
                ]
                if column in df_dashboard_pm_display.columns
            ]

            if df_dashboard_pm_display.empty:
                st.info("No servers have scheduled maintenance.")
            else:
                st.dataframe(
                    df_dashboard_pm_display[dashboard_pm_columns],
                    width="stretch",
                )

    elif dashboard_maintenance_metric == "Status Views":
        if df_dashboard_server_load_main.empty:
            st.warning("No server status data available.")
        else:
            if require_dashboard_columns(
                df_dashboard_server_load_main,
                [
                    "server_id",
                    "status",
                ],
                "server_load_timeseries",
            ):
                st.subheader("Server Status Views")

                df_dashboard_status_occurrences = (
                    build_dashboard_status_occurrence_counts(
                        df_dashboard_server_load_main
                    )
                )

                df_dashboard_unique_status_counts = (
                    build_dashboard_unique_server_status_counts(
                        df_dashboard_server_load_main
                    )
                )

                df_dashboard_daily_status_counts = (
                    build_dashboard_daily_server_status_counts(
                        df_dashboard_server_load_main
                    )
                )

                col_status_1, col_status_2, col_status_3 = st.columns(3)

                with col_status_1:
                    st.markdown("### Status Occurrences")
                    st.markdown(
                        '<div class="status-subtitle">Counts every server-slot record. A server active for 10 slots counts 10 times.</div>',
                        unsafe_allow_html=True,
                    )

                    if df_dashboard_status_occurrences.empty:
                        st.warning("Could not calculate status occurrences.")
                    else:
                        fig_dashboard_status_occurrences = px.pie(
                            df_dashboard_status_occurrences,
                            names="status",
                            values="server_slot_count",
                            title="",
                        )

                        st.plotly_chart(
                            fig_dashboard_status_occurrences,
                            width="stretch",
                        )

                        st.dataframe(
                            df_dashboard_status_occurrences,
                            width="stretch",
                        )

                with col_status_2:
                    st.markdown("### Unique Servers by Status")
                    st.markdown(
                        '<div class="status-subtitle">Counts servers that appeared at least once in each status. One server can appear in multiple statuses.</div>',
                        unsafe_allow_html=True,
                    )

                    if df_dashboard_unique_status_counts.empty:
                        st.warning(
                            "Could not calculate unique server status counts.")
                    else:
                        fig_dashboard_unique_status = px.pie(
                            df_dashboard_unique_status_counts,
                            names="status",
                            values="unique_server_count",
                            title="",
                        )

                        st.plotly_chart(
                            fig_dashboard_unique_status,
                            width="stretch",
                        )

                        st.dataframe(
                            df_dashboard_unique_status_counts,
                            width="stretch",
                        )

                with col_status_3:
                    st.markdown("### Daily Server Classification")
                    st.markdown(
                        '<div class="status-subtitle">Counts each server once using this priority: maintenance, active, then idle.</div>',
                        unsafe_allow_html=True,
                    )

                    if df_dashboard_daily_status_counts.empty:
                        st.warning(
                            "Could not calculate daily server classification.")
                    else:
                        fig_dashboard_daily_status = px.pie(
                            df_dashboard_daily_status_counts,
                            names="daily_status",
                            values="server_count",
                            title="",
                        )

                        st.plotly_chart(
                            fig_dashboard_daily_status,
                            width="stretch",
                        )

                        st.dataframe(
                            df_dashboard_daily_status_counts,
                            width="stretch",
                        )


# ============================================================
# 16. SCENARIO COMPARISON TAB
# ============================================================

with tab_compare:
    st.header("Scenario Comparison")

    if df_dashboard_performance_comparison.empty:
        st.warning("No comparison data available.")
    else:
        dashboard_comparison_metric = st.radio(
            "Metric to compare",
            [
                "Total Cost",
                "Cost Breakdown",
                "Jobs Late",
                "Average PUE",
                "Energy Consumption",
                "Server Utilization",
                "Server Status",
            ],
            horizontal=True,
            key="comparison_metric",
        )

        if dashboard_comparison_metric == "Total Cost":
            if require_dashboard_columns(
                df_dashboard_performance_comparison,
                [
                    "scenario",
                    "total_cost",
                ],
                "performance_metrics",
            ):
                fig_dashboard_total_cost = px.bar(
                    df_dashboard_performance_comparison,
                    x="scenario",
                    y="total_cost",
                    text_auto=".2f",
                    title="Total Cost by Scenario",
                )

                fig_dashboard_total_cost.update_layout(
                    xaxis_title="Scenario",
                    yaxis_title="Total Cost",
                )

                st.plotly_chart(
                    fig_dashboard_total_cost,
                    width="stretch",
                )

                dashboard_total_cost_columns = [
                    column
                    for column in [
                        "scenario",
                        "total_cost",
                        "energy_cost",
                        "pm_cost",
                        "expected_cm_cost",
                        "switching_cost",
                        "lateness_cost",
                    ]
                    if column in df_dashboard_performance_comparison.columns
                ]

                st.dataframe(
                    df_dashboard_performance_comparison[
                        dashboard_total_cost_columns
                    ],
                    width="stretch",
                )

        elif dashboard_comparison_metric == "Cost Breakdown":
            dashboard_cost_columns = [
                "energy_cost",
                "pm_cost",
                "expected_cm_cost",
                "switching_cost",
                "lateness_cost",
            ]

            dashboard_available_cost_columns = [
                column
                for column in dashboard_cost_columns
                if column in df_dashboard_performance_comparison.columns
            ]

            if dashboard_available_cost_columns:
                df_dashboard_cost_breakdown_comparison = (
                    df_dashboard_performance_comparison[
                        [
                            "scenario",
                            *dashboard_available_cost_columns,
                        ]
                    ].melt(
                        id_vars="scenario",
                        var_name="cost_type",
                        value_name="cost",
                    )
                )

                df_dashboard_cost_breakdown_comparison["cost_type"] = (
                    df_dashboard_cost_breakdown_comparison["cost_type"]
                    .replace(
                        {
                            "energy_cost": "Energy",
                            "pm_cost": "PM",
                            "expected_cm_cost": "CM",
                            "switching_cost": "Switch",
                            "lateness_cost": "Lateness",
                        }
                    )
                )

                fig_dashboard_cost_breakdown = px.bar(
                    df_dashboard_cost_breakdown_comparison,
                    x="scenario",
                    y="cost",
                    color="cost_type",
                    barmode="stack",
                    text=df_dashboard_cost_breakdown_comparison["cost"].map(
                        lambda value: f"${float(value):,.2f}"
                    ),
                    title="Cost Breakdown by Scenario",
                )

                fig_dashboard_cost_breakdown.update_layout(
                    xaxis_title="Scenario",
                    yaxis_title="Cost",
                    legend_title_text="Cost Type",
                )

                fig_dashboard_cost_breakdown.update_traces(
                    textposition="inside",
                )

                st.plotly_chart(
                    fig_dashboard_cost_breakdown,
                    width="stretch",
                )
            else:
                st.warning("No cost breakdown columns found.")

        elif dashboard_comparison_metric == "Jobs Late":
            if require_dashboard_columns(
                df_dashboard_performance_comparison,
                [
                    "scenario",
                    "jobs_late_count",
                ],
                "performance_metrics",
            ):
                fig_dashboard_late_jobs = px.bar(
                    df_dashboard_performance_comparison,
                    x="scenario",
                    y="jobs_late_count",
                    text_auto=True,
                    title="Late Jobs by Scenario",
                )

                fig_dashboard_late_jobs.update_layout(
                    xaxis_title="Scenario",
                    yaxis_title="Late Jobs",
                )

                st.plotly_chart(
                    fig_dashboard_late_jobs,
                    width="stretch",
                )

        elif dashboard_comparison_metric == "Average PUE":
            if require_dashboard_columns(
                df_dashboard_performance_comparison,
                [
                    "scenario",
                    "average_pue",
                ],
                "performance_metrics",
            ):
                fig_dashboard_average_pue = px.bar(
                    df_dashboard_performance_comparison,
                    x="scenario",
                    y="average_pue",
                    text_auto=".3f",
                    title="Average PUE by Scenario",
                )

                fig_dashboard_average_pue.update_layout(
                    xaxis_title="Scenario",
                    yaxis_title="Average PUE",
                )

                st.plotly_chart(
                    fig_dashboard_average_pue,
                    width="stretch",
                )

            if (
                not df_dashboard_hourly_energy_comparison.empty
                and {
                    "scenario",
                    "time",
                    "PUE",
                }.issubset(df_dashboard_hourly_energy_comparison.columns)
            ):
                fig_dashboard_pue_line = px.line(
                    df_dashboard_hourly_energy_comparison,
                    x="time",
                    y="PUE",
                    color="scenario",
                    title="PUE Variation by Scenario",
                )

                fig_dashboard_pue_line.update_layout(
                    xaxis_title="Time",
                    yaxis_title="PUE",
                )

                st.plotly_chart(
                    fig_dashboard_pue_line,
                    width="stretch",
                )

        elif dashboard_comparison_metric == "Energy Consumption":
            if require_dashboard_columns(
                df_dashboard_performance_comparison,
                [
                    "scenario",
                    "total_facility_energy_kwh",
                ],
                "performance_metrics",
            ):
                fig_dashboard_energy_consumption = px.bar(
                    df_dashboard_performance_comparison,
                    x="scenario",
                    y="total_facility_energy_kwh",
                    text_auto=".3f",
                    title="Total Facility Energy by Scenario",
                )

                fig_dashboard_energy_consumption.update_layout(
                    xaxis_title="Scenario",
                    yaxis_title="Total Facility Energy kWh",
                )

                st.plotly_chart(
                    fig_dashboard_energy_consumption,
                    width="stretch",
                )

            required_dashboard_power_columns = {
                "scenario",
                "time",
                "PIT_W",
                "Pcool_W",
                "Ptot_W",
            }

            if (
                not df_dashboard_hourly_energy_comparison.empty
                and required_dashboard_power_columns.issubset(
                    df_dashboard_hourly_energy_comparison.columns
                )
            ):
                df_dashboard_power_comparison = (
                    df_dashboard_hourly_energy_comparison[
                        [
                            "scenario",
                            "time",
                            "PIT_W",
                            "Pcool_W",
                            "Ptot_W",
                        ]
                    ].melt(
                        id_vars=[
                            "scenario",
                            "time",
                        ],
                        var_name="power_type",
                        value_name="power_watts",
                    )
                )

                fig_dashboard_power_comparison = px.line(
                    df_dashboard_power_comparison,
                    x="time",
                    y="power_watts",
                    color="scenario",
                    line_dash="power_type",
                    title="Power Consumption by Scenario",
                )

                fig_dashboard_power_comparison.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Power",
                )

                st.plotly_chart(
                    fig_dashboard_power_comparison,
                    width="stretch",
                )

        elif dashboard_comparison_metric == "Server Utilization":
            if df_dashboard_server_summary_comparison.empty:
                st.warning("No server summary data available for comparison.")
            else:
                if require_dashboard_columns(
                    df_dashboard_server_summary_comparison,
                    [
                        "scenario",
                        "utilization_rate_pct",
                    ],
                    "server_summary",
                ):
                    df_dashboard_avg_utilization = (
                        df_dashboard_server_summary_comparison
                        .groupby("scenario", as_index=False)
                        .agg(
                            avg_utilization_pct=(
                                "utilization_rate_pct",
                                "mean",
                            ),
                            max_utilization_pct=(
                                "utilization_rate_pct",
                                "max",
                            ),
                        )
                    )

                    fig_dashboard_avg_utilization = px.bar(
                        df_dashboard_avg_utilization,
                        x="scenario",
                        y="avg_utilization_pct",
                        text_auto=".2f",
                        title="Average Server Utilization by Scenario",
                    )

                    fig_dashboard_avg_utilization.update_layout(
                        xaxis_title="Scenario",
                        yaxis_title="Average Utilization (%)",
                    )

                    st.plotly_chart(
                        fig_dashboard_avg_utilization,
                        width="stretch",
                    )

                    fig_dashboard_utilization_distribution = px.box(
                        df_dashboard_server_summary_comparison,
                        x="scenario",
                        y="utilization_rate_pct",
                        title="Server Utilization Distribution by Scenario",
                    )

                    fig_dashboard_utilization_distribution.update_layout(
                        xaxis_title="Scenario",
                        yaxis_title="Utilization Rate (%)",
                    )

                    st.plotly_chart(
                        fig_dashboard_utilization_distribution,
                        width="stretch",
                    )

        elif dashboard_comparison_metric == "Server Status":
            if df_dashboard_server_load_comparison.empty:
                st.warning("No server load data available for comparison.")
            else:
                if require_dashboard_columns(
                    df_dashboard_server_load_comparison,
                    [
                        "scenario",
                        "server_id",
                        "status",
                    ],
                    "server_load_timeseries",
                ):
                    dashboard_status_comparison_view = st.radio(
                        "Status comparison view",
                        [
                            "Status Occurrences",
                            "Unique Servers by Status",
                            "Daily Server Classification",
                        ],
                        horizontal=True,
                        key="status_comparison_view",
                    )

                    if dashboard_status_comparison_view == "Status Occurrences":
                        df_dashboard_status_occurrence_comparison = (
                            df_dashboard_server_load_comparison
                            .groupby(
                                [
                                    "scenario",
                                    "status",
                                ]
                            )
                            .size()
                            .reset_index(name="server_slot_count")
                        )

                        fig_dashboard_status_occurrence_comparison = px.bar(
                            df_dashboard_status_occurrence_comparison,
                            x="scenario",
                            y="server_slot_count",
                            color="status",
                            barmode="group",
                            text_auto=True,
                            title="Status Occurrences by Scenario",
                        )

                        fig_dashboard_status_occurrence_comparison.update_layout(
                            xaxis_title="Scenario",
                            yaxis_title="Server-Time Slot Count",
                        )

                        st.plotly_chart(
                            fig_dashboard_status_occurrence_comparison,
                            width="stretch",
                        )

                        st.dataframe(
                            df_dashboard_status_occurrence_comparison,
                            width="stretch",
                        )

                    elif dashboard_status_comparison_view == "Unique Servers by Status":
                        df_dashboard_unique_status_comparison = (
                            df_dashboard_server_load_comparison
                            .groupby(
                                [
                                    "scenario",
                                    "status",
                                ]
                            )["server_id"]
                            .nunique()
                            .reset_index(name="unique_server_count")
                        )

                        fig_dashboard_unique_status_comparison = px.bar(
                            df_dashboard_unique_status_comparison,
                            x="scenario",
                            y="unique_server_count",
                            color="status",
                            barmode="group",
                            text_auto=True,
                            title="Unique Servers by Status and Scenario",
                        )

                        fig_dashboard_unique_status_comparison.update_layout(
                            xaxis_title="Scenario",
                            yaxis_title="Unique Server Count",
                        )

                        st.plotly_chart(
                            fig_dashboard_unique_status_comparison,
                            width="stretch",
                        )

                        st.dataframe(
                            df_dashboard_unique_status_comparison,
                            width="stretch",
                        )

                    elif dashboard_status_comparison_view == "Daily Server Classification":
                        dashboard_daily_classification_rows = []

                        for scenario, df_dashboard_scenario_load in (
                            df_dashboard_server_load_comparison.groupby(
                                "scenario")
                        ):
                            df_dashboard_daily_status = (
                                build_dashboard_daily_server_classification(
                                    df_dashboard_scenario_load
                                )
                            )

                            if df_dashboard_daily_status.empty:
                                continue

                            df_dashboard_daily_status["scenario"] = scenario

                            dashboard_daily_classification_rows.append(
                                df_dashboard_daily_status
                            )

                        if not dashboard_daily_classification_rows:
                            st.warning(
                                "Could not build daily status comparison.")
                        else:
                            df_dashboard_daily_classification_comparison = (
                                pd.concat(
                                    dashboard_daily_classification_rows,
                                    ignore_index=True,
                                )
                            )

                            df_dashboard_daily_status_counts_comparison = (
                                df_dashboard_daily_classification_comparison
                                .groupby(
                                    [
                                        "scenario",
                                        "daily_status",
                                    ]
                                )["server_id"]
                                .nunique()
                                .reset_index(name="server_count")
                            )

                            fig_dashboard_daily_status_comparison = px.bar(
                                df_dashboard_daily_status_counts_comparison,
                                x="scenario",
                                y="server_count",
                                color="daily_status",
                                barmode="group",
                                text_auto=True,
                                title="Daily Server Classification by Scenario",
                            )

                            fig_dashboard_daily_status_comparison.update_layout(
                                xaxis_title="Scenario",
                                yaxis_title="Number of Servers",
                            )

                            st.plotly_chart(
                                fig_dashboard_daily_status_comparison,
                                width="stretch",
                            )

                            st.dataframe(
                                df_dashboard_daily_status_counts_comparison,
                                width="stretch",
                            )


# ============================================================
# 17. SCENARIO LIBRARY TAB
# ============================================================

with tab_library:
    st.header("Scenario Library")

    st.write(
        "This table shows all scenarios currently detected from the output folders."
    )

    dashboard_library_columns = [
        column
        for column in [
            "scenario",
            "status",
            "has_solution",
            "total_cost",
            "jobs_placed",
            "jobs_late_count",
            "average_pue",
            "max_pue",
            "total_facility_energy_kwh",
            "runtime_seconds",
            "file_modified_time",
            "source_file",
            "source_path",
        ]
        if column in df_dashboard_performance.columns
    ]

    df_dashboard_scenario_library = (
        df_dashboard_performance[dashboard_library_columns]
        .sort_values(
            "file_modified_time",
            ascending=False,
        )
    )

    st.dataframe(
        df_dashboard_scenario_library,
        width="stretch",
    )

    st.subheader("Raw loaded table sizes")

    df_dashboard_table_sizes = pd.DataFrame(
        [
            {
                "table": table_name,
                "rows": len(df_dashboard_table),
                "columns": (
                    len(df_dashboard_table.columns)
                    if not df_dashboard_table.empty
                    else 0
                ),
            }
            for table_name, df_dashboard_table in dashboard_data.items()
        ]
    )

    st.dataframe(
        df_dashboard_table_sizes,
        width="stretch",
    )
