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


# ============================================================
# 2. PATH CONFIGURATION
# ============================================================

def get_project_root() -> Path:
    """
    Returns project root assuming this file is inside:
        Capstone_Project/src/visualization/dashboard.py

    parents[0] = visualization
    parents[1] = src
    parents[2] = Capstone_Project
    """
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = get_project_root()


def get_output_candidate_dirs() -> list[Path]:
    """
    Output folders scanned by the dashboard.

    If later your output folder changes, edit only this function.
    """

    return [
        PROJECT_ROOT / "outputs" / "optimization",
        PROJECT_ROOT / "outputs" / "results" / "optimization",
        PROJECT_ROOT / "outputs" / "results" / "tables",
        PROJECT_ROOT / "outputs" / "tables",
        PROJECT_ROOT / "results" / "optimization",
        PROJECT_ROOT / "results" / "tables",
    ]


OUTPUT_CANDIDATE_DIRS = get_output_candidate_dirs()


# ============================================================
# 3. FILE DISCOVERY AND LOADING
# ============================================================

def find_matching_files(patterns: list[str]) -> list[Path]:
    """
    Finds files matching multiple patterns across all candidate folders.
    """

    matching_files = []

    for folder in OUTPUT_CANDIDATE_DIRS:
        if folder.exists():
            for pattern in patterns:
                matching_files.extend(folder.glob(pattern))

    return sorted(set(matching_files))


def read_table(file_path: Path) -> pd.DataFrame:
    """
    Reads CSV or Excel files.
    """

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(file_path)

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)

    raise ValueError(f"Unsupported file type: {file_path.suffix}")


def infer_scenario_from_filename(file_name: str, table_prefix: str) -> str:
    """
    Infers scenario name from filenames like:

        optimization_solution_base_20260609-024827.csv
        performance_metrics_high_demand_20260609-024827.csv
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


def read_and_combine_files(
    patterns: list[str],
    table_prefix: str,
) -> pd.DataFrame:
    """
    Reads all files matching patterns and combines them into one DataFrame.

    Adds:
        source_file
        source_path
        file_modified_time
        scenario, if missing
    """

    files = find_matching_files(patterns)

    if not files:
        return pd.DataFrame()

    dataframes = []

    for file_path in files:
        try:
            df = read_table(file_path)

            df["source_file"] = file_path.name
            df["source_path"] = str(file_path)
            df["file_modified_time"] = pd.to_datetime(
                file_path.stat().st_mtime,
                unit="s",
            )

            if "scenario" not in df.columns:
                df["scenario"] = infer_scenario_from_filename(
                    file_name=file_path.name,
                    table_prefix=table_prefix,
                )

            dataframes.append(df)

        except Exception as error:
            st.warning(
                f"Could not read file: {file_path.name}. Error: {error}"
            )

    if not dataframes:
        return pd.DataFrame()

    return pd.concat(dataframes, ignore_index=True)


def keep_latest_run_per_scenario(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps only the latest file per scenario.
    """

    if df.empty:
        return df

    if "scenario" not in df.columns:
        return df

    if "source_file" not in df.columns:
        return df

    if "file_modified_time" not in df.columns:
        return df

    latest_files = (
        df[["scenario", "source_file", "file_modified_time"]]
        .drop_duplicates()
        .sort_values("file_modified_time")
        .groupby("scenario", as_index=False)
        .tail(1)
    )

    latest_df = df.merge(
        latest_files[["scenario", "source_file"]],
        on=["scenario", "source_file"],
        how="inner",
    )

    return latest_df


@st.cache_data(ttl=60)
def load_all_available_outputs() -> dict[str, pd.DataFrame]:
    """
    Loads all currently available optimization outputs.
    """

    performance_df = keep_latest_run_per_scenario(
        read_and_combine_files(
            patterns=[
                "performance_metrics_*.csv",
                "performance_metrics_*.xlsx",
                "*performance_metrics*.csv",
                "*performance_metrics*.xlsx",
            ],
            table_prefix="performance_metrics",
        )
    )

    solution_df = keep_latest_run_per_scenario(
        read_and_combine_files(
            patterns=[
                "optimization_solution_*.csv",
                "optimization_solution_*.xlsx",
                "*optimization_solution*.csv",
                "*optimization_solution*.xlsx",
            ],
            table_prefix="optimization_solution",
        )
    )

    server_summary_df = keep_latest_run_per_scenario(
        read_and_combine_files(
            patterns=[
                "server_summary_*.csv",
                "server_summary_*.xlsx",
                "*server_summary*.csv",
                "*server_summary*.xlsx",
            ],
            table_prefix="server_summary",
        )
    )

    hourly_df = keep_latest_run_per_scenario(
        read_and_combine_files(
            patterns=[
                "hourly_energy_thermal_*.csv",
                "hourly_energy_thermal_*.xlsx",
                "*hourly_energy_thermal*.csv",
                "*hourly_energy_thermal*.xlsx",
            ],
            table_prefix="hourly_energy_thermal",
        )
    )

    pm_df = keep_latest_run_per_scenario(
        read_and_combine_files(
            patterns=[
                "pm_schedule_*.csv",
                "pm_schedule_*.xlsx",
                "*pm_schedule*.csv",
                "*pm_schedule*.xlsx",
            ],
            table_prefix="pm_schedule",
        )
    )

    server_load_df = keep_latest_run_per_scenario(
        read_and_combine_files(
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
        "performance": performance_df,
        "solution": solution_df,
        "server_summary": server_summary_df,
        "hourly": hourly_df,
        "pm": pm_df,
        "server_load": server_load_df,
    }


# ============================================================
# 4. GENERAL HELPERS
# ============================================================

def require_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    table_name: str,
) -> bool:
    """
    Checks whether a DataFrame contains required columns.
    """

    if df.empty:
        st.warning(f"No data found for: {table_name}")
        return False

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        st.warning(
            f"Table '{table_name}' is missing columns: {missing_columns}"
        )

        with st.expander(f"Available columns in {table_name}"):
            st.write(list(df.columns))

        return False

    return True


def safe_metric_value(row: pd.Series, column: str, default=0):
    """
    Safely extracts a value from a row.
    """

    if column not in row:
        return default

    value = row[column]

    if pd.isna(value):
        return default

    return value


def filter_by_scenario(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """
    Filters a DataFrame by one scenario.
    """

    if df.empty:
        return pd.DataFrame()

    if "scenario" not in df.columns:
        return pd.DataFrame()

    return df[df["scenario"] == scenario].copy()


def filter_by_scenarios(
    df: pd.DataFrame,
    scenarios: list[str],
) -> pd.DataFrame:
    """
    Filters a DataFrame by multiple scenarios.
    """

    if df.empty:
        return pd.DataFrame()

    if "scenario" not in df.columns:
        return pd.DataFrame()

    return df[df["scenario"].isin(scenarios)].copy()


def format_currency(value) -> str:
    """
    Formats a value as currency.
    """

    try:
        return f"${float(value):,.2f}"
    except Exception:
        return "$0.00"


def format_number(value, decimals: int = 2) -> str:
    """
    Formats a number safely.
    """

    try:
        return f"{float(value):,.{decimals}f}"
    except Exception:
        return f"{0:.{decimals}f}"


def get_unique_jobs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns unique jobs if job_id exists.
    """

    if df.empty:
        return df

    if "job_id" not in df.columns:
        return df

    return df.drop_duplicates(subset=["job_id"]).copy()


def classify_daily_server_status(server_df: pd.DataFrame) -> str:
    """
    Classifies each server once for the whole day.

    Priority:
        maintenance > active > idle
    """

    statuses = set(server_df["status"].dropna())

    if "maintenance" in statuses:
        return "maintenance"

    if "active" in statuses:
        return "active"

    return "idle"


def build_status_occurrence_counts(server_load_df: pd.DataFrame) -> pd.DataFrame:
    """
    Counts server-time-slot records by status.

    Example:
        If server 1 is active for 10 slots, it contributes 10 active records.
    """

    if server_load_df.empty:
        return pd.DataFrame()

    if "status" not in server_load_df.columns:
        return pd.DataFrame()

    return (
        server_load_df
        .groupby("status")
        .size()
        .reset_index(name="server_slot_count")
    )


def build_unique_server_status_counts(server_load_df: pd.DataFrame) -> pd.DataFrame:
    """
    Counts unique servers that appeared at least once in each status.

    A server can be counted in multiple statuses.
    Example:
        Server 1 active in morning and idle at night counts in both.
    """

    if server_load_df.empty:
        return pd.DataFrame()

    required_columns = {"server_id", "status"}

    if not required_columns.issubset(server_load_df.columns):
        return pd.DataFrame()

    return (
        server_load_df
        .groupby("status")["server_id"]
        .nunique()
        .reset_index(name="unique_server_count")
    )


def build_daily_server_classification(server_load_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds one final daily status per server.

    Priority:
        maintenance > active > idle

    This makes sure each server is counted only once.
    """

    if server_load_df.empty:
        return pd.DataFrame()

    required_columns = {"server_id", "status"}

    if not required_columns.issubset(server_load_df.columns):
        return pd.DataFrame()

    daily_status_df = (
        server_load_df
        .groupby("server_id")
        .apply(classify_daily_server_status, include_groups=False)
        .reset_index(name="daily_status")
    )

    return daily_status_df


def build_daily_server_status_counts(server_load_df: pd.DataFrame) -> pd.DataFrame:
    """
    Counts servers by final daily classification.

    Each server is counted once.
    """

    daily_status_df = build_daily_server_classification(server_load_df)

    if daily_status_df.empty:
        return pd.DataFrame()

    return (
        daily_status_df
        .groupby("daily_status")["server_id"]
        .nunique()
        .reset_index(name="server_count")
    )


# ============================================================
# 5. LOAD DATA
# ============================================================

data = load_all_available_outputs()

performance_df = data["performance"]
solution_df = data["solution"]
server_summary_df = data["server_summary"]
hourly_df = data["hourly"]
pm_df = data["pm"]
server_load_df = data["server_load"]


# ============================================================
# 6. SIDEBAR
# ============================================================

st.sidebar.title("Dashboard Controls")

if st.sidebar.button("Refresh outputs"):
    st.cache_data.clear()
    st.rerun()


with st.sidebar.expander("Output folders being scanned"):
    for folder in OUTPUT_CANDIDATE_DIRS:
        status = "Found" if folder.exists() else "Not found"
        st.write(f"{status}: `{folder}`")


with st.sidebar.expander("Loaded tables debug"):
    for table_name, df in data.items():
        st.write(f"**{table_name}**: {len(df)} rows")

        if not df.empty and "source_file" in df.columns:
            source_files = (
                df["source_file"]
                .drop_duplicates()
                .sort_values()
                .tolist()
            )
            st.write(source_files)

        if not df.empty and "scenario" in df.columns:
            scenarios = sorted(df["scenario"].dropna().unique())
            st.write(f"Scenarios: {scenarios}")


# ============================================================
# 7. STOP IF NO PERFORMANCE DATA
# ============================================================

if performance_df.empty:
    st.title("Data Center Optimization Dashboard")

    st.warning("No optimization performance files found yet.")

    st.write("The dashboard is currently scanning these folders:")

    for folder in OUTPUT_CANDIDATE_DIRS:
        st.code(str(folder))

    st.write("Expected file examples:")

    st.code(
        """
performance_metrics_base_20260609-024827.csv
optimization_solution_base_20260609-024827.csv
server_summary_base_20260609-024827.csv
hourly_energy_thermal_base_20260609-024827.csv
pm_schedule_base_20260609-024827.csv
server_load_timeseries_base_20260609-024827.csv
        """.strip()
    )

    st.stop()


if "scenario" not in performance_df.columns:
    st.error("The performance metrics data does not contain a scenario column.")
    st.stop()


available_scenarios = sorted(
    performance_df["scenario"].dropna().unique()
)

if not available_scenarios:
    st.warning("No scenarios found in the available output files.")
    st.stop()


st.sidebar.metric(
    "Scenarios available",
    len(available_scenarios),
)


main_scenario = st.sidebar.selectbox(
    "Main scenario for detailed analysis",
    options=available_scenarios,
)


comparison_scenarios = st.sidebar.multiselect(
    "Scenarios for comparison",
    options=available_scenarios,
    default=available_scenarios[:3],
    max_selections=3,
)


if not comparison_scenarios:
    st.warning("Select at least one scenario for comparison.")
    st.stop()


with st.sidebar.expander("Available scenarios"):
    scenario_columns = [
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
        if column in performance_df.columns
    ]

    st.dataframe(
        performance_df[scenario_columns].sort_values(
            "file_modified_time",
            ascending=False,
        ),
        use_container_width=True,
    )


# ============================================================
# 8. FILTER DATA
# ============================================================

performance_main = filter_by_scenario(performance_df, main_scenario)
solution_main = filter_by_scenario(solution_df, main_scenario)
server_summary_main = filter_by_scenario(server_summary_df, main_scenario)
hourly_main = filter_by_scenario(hourly_df, main_scenario)
pm_main = filter_by_scenario(pm_df, main_scenario)
server_load_main = filter_by_scenario(server_load_df, main_scenario)

performance_comparison = filter_by_scenarios(
    performance_df,
    comparison_scenarios,
)

solution_comparison = filter_by_scenarios(
    solution_df,
    comparison_scenarios,
)

server_summary_comparison = filter_by_scenarios(
    server_summary_df,
    comparison_scenarios,
)

hourly_comparison = filter_by_scenarios(
    hourly_df,
    comparison_scenarios,
)

pm_comparison = filter_by_scenarios(
    pm_df,
    comparison_scenarios,
)

server_load_comparison = filter_by_scenarios(
    server_load_df,
    comparison_scenarios,
)


# ============================================================
# 9. TITLE
# ============================================================

st.title("Data Center Job Scheduling Optimization Dashboard")

st.caption(
    f"Main scenario: {main_scenario} | "
    f"Comparison scenarios: {', '.join(comparison_scenarios)}"
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
    st.header(f"Overview: {main_scenario}")

    if performance_main.empty:
        st.warning("No performance metrics available for this scenario.")
    else:
        metrics = performance_main.iloc[0]

        metric_group = st.radio(
            "Metric group",
            ["Cost", "Jobs", "Energy", "Servers"],
            horizontal=True,
            key="overview_metric_group",
        )

        if metric_group == "Cost":
            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Total Cost",
                format_currency(safe_metric_value(metrics, "total_cost")),
            )

            col2.metric(
                "Energy Cost",
                format_currency(safe_metric_value(metrics, "energy_cost")),
            )

            col3.metric(
                "PM Cost",
                format_currency(safe_metric_value(metrics, "pm_cost")),
            )

            col4.metric(
                "Lateness Cost",
                format_currency(safe_metric_value(metrics, "lateness_cost")),
            )

            cost_columns = [
                "energy_cost",
                "pm_cost",
                "expected_cm_cost",
                "switching_cost",
                "lateness_cost",
            ]

            available_cost_columns = [
                column
                for column in cost_columns
                if column in performance_main.columns
            ]

            if available_cost_columns:
                cost_breakdown_df = performance_main[
                    ["scenario", *available_cost_columns]
                ].melt(
                    id_vars="scenario",
                    var_name="cost_type",
                    value_name="cost",
                )

                fig = px.bar(
                    cost_breakdown_df,
                    x="cost_type",
                    y="cost",
                    text_auto=".2f",
                    title="Cost Breakdown",
                )

                fig.update_layout(
                    xaxis_title="Cost Type",
                    yaxis_title="Cost",
                )

                st.plotly_chart(fig, use_container_width=True)

        elif metric_group == "Jobs":
            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Jobs in Forecast",
                int(safe_metric_value(metrics, "jobs_in_forecast")),
            )

            col2.metric(
                "Jobs Placed",
                int(safe_metric_value(metrics, "jobs_placed")),
            )

            col3.metric(
                "Jobs On Time",
                int(safe_metric_value(metrics, "jobs_on_time")),
            )

            col4.metric(
                "Late Jobs",
                int(safe_metric_value(metrics, "jobs_late_count")),
            )

        elif metric_group == "Energy":
            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Facility Energy",
                f"{format_number(safe_metric_value(metrics, 'total_facility_energy_kwh'), 3)} kWh",
            )

            col2.metric(
                "Average PUE",
                format_number(safe_metric_value(metrics, "average_pue"), 3),
            )

            col3.metric(
                "Max PUE",
                format_number(safe_metric_value(metrics, "max_pue"), 3),
            )

        elif metric_group == "Servers":
            if server_summary_main.empty:
                st.warning("No server summary available for this scenario.")
            else:
                if require_columns(
                    server_summary_main,
                    ["server_id", "utilization_rate_pct"],
                    "server_summary",
                ):
                    most_used = server_summary_main.sort_values(
                        "utilization_rate_pct",
                        ascending=False,
                    ).iloc[0]

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Most Used Server",
                        most_used["server_id"],
                    )

                    col2.metric(
                        "Highest Utilization",
                        f"{float(most_used['utilization_rate_pct']):.2f}%",
                    )

                    if "pm_scheduled" in server_summary_main.columns:
                        servers_with_pm = int(
                            server_summary_main["pm_scheduled"].sum()
                        )
                    else:
                        servers_with_pm = 0

                    col3.metric(
                        "Servers with PM",
                        servers_with_pm,
                    )


# ============================================================
# 12. JOBS & DEMAND TAB
# ============================================================

with tab_jobs:
    st.header(f"Jobs & Demand: {main_scenario}")

    if solution_main.empty:
        st.warning("No optimization solution data available for this scenario.")

        with st.expander("Debug: solution table"):
            st.write(f"Rows in all solution data: {len(solution_df)}")

            if not solution_df.empty:
                st.write("Columns:")
                st.write(list(solution_df.columns))

                if "scenario" in solution_df.columns:
                    st.write("Scenarios in solution table:")
                    st.write(sorted(solution_df["scenario"].dropna().unique()))

                if "source_file" in solution_df.columns:
                    st.write("Source files:")
                    st.write(
                        solution_df["source_file"].drop_duplicates().tolist())
    else:
        job_metric = st.radio(
            "Job metric",
            [
                "Job Type Proportion",
                "Jobs by Hour and Type",
                "Scheduled Jobs Table",
            ],
            horizontal=True,
            key="job_metric",
        )

        if job_metric == "Job Type Proportion":
            if require_columns(
                solution_main,
                ["job_id", "job_type"],
                "optimization_solution",
            ):
                unique_jobs = get_unique_jobs(solution_main)

                job_type_counts = (
                    unique_jobs
                    .groupby("job_type")
                    .size()
                    .reset_index(name="job_count")
                )

                fig = px.pie(
                    job_type_counts,
                    names="job_type",
                    values="job_count",
                    title="Proportion of Job Types",
                )

                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    job_type_counts,
                    use_container_width=True,
                )

        elif job_metric == "Jobs by Hour and Type":
            if require_columns(
                solution_main,
                ["job_id", "job_type", "start_slot"],
                "optimization_solution",
            ):
                unique_jobs = get_unique_jobs(solution_main)

                unique_jobs["start_hour"] = unique_jobs["start_slot"]

                jobs_by_hour_type = (
                    unique_jobs
                    .groupby(["start_hour", "job_type"])
                    .size()
                    .reset_index(name="job_count")
                )

                fig = px.bar(
                    jobs_by_hour_type,
                    x="start_hour",
                    y="job_count",
                    color="job_type",
                    barmode="stack",
                    title="Jobs by Hour and Type",
                )

                fig.update_layout(
                    xaxis_title="Hour / Slot",
                    yaxis_title="Number of Jobs",
                )

                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    jobs_by_hour_type,
                    use_container_width=True,
                )

        elif job_metric == "Scheduled Jobs Table":
            display_columns = [
                column
                for column in [
                    "scenario",
                    "job_id",
                    "job_type",
                    "is_critical",
                    "server_id",
                    "start_slot",
                    "end_slot",
                    "start_time",
                    "end_time",
                    "duration_hours",
                    "source_file",
                ]
                if column in solution_main.columns
            ]

            st.dataframe(
                solution_main[display_columns],
                use_container_width=True,
            )


# ============================================================
# 13. SERVER UTILIZATION TAB
# ============================================================

with tab_servers:
    st.header(f"Server Utilization: {main_scenario}")

    server_metric = st.radio(
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

    if server_metric == "Utilization Summary":
        if server_summary_main.empty:
            st.warning("No server summary data available.")
        else:
            if require_columns(
                server_summary_main,
                ["server_id", "utilization_rate_pct"],
                "server_summary",
            ):
                fig = px.bar(
                    server_summary_main,
                    x="server_id",
                    y="utilization_rate_pct",
                    color=(
                        "server_type"
                        if "server_type" in server_summary_main.columns
                        else None
                    ),
                    text_auto=".2f",
                    title="Server Utilization Rate",
                )

                fig.update_layout(
                    xaxis_title="Server ID",
                    yaxis_title="Utilization Rate (%)",
                )

                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    server_summary_main.sort_values(
                        "utilization_rate_pct",
                        ascending=False,
                    ),
                    use_container_width=True,
                )

    elif server_metric == "Server Load Over Time":
        if server_load_main.empty:
            st.warning("No server load timeseries available.")
        else:
            if require_columns(
                server_load_main,
                ["server_id", "time", "load"],
                "server_load_timeseries",
            ):
                available_servers = sorted(
                    server_load_main["server_id"].unique())

                selected_servers = st.multiselect(
                    "Select servers",
                    options=available_servers,
                    default=available_servers[:5],
                )

                filtered_load = server_load_main[
                    server_load_main["server_id"].isin(selected_servers)
                ]

                fig = px.line(
                    filtered_load,
                    x="time",
                    y="load",
                    color="server_id",
                    title="Server Load Over Time",
                )

                fig.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Load",
                )

                st.plotly_chart(fig, use_container_width=True)

    elif server_metric == "Batch vs Interactive Load":
        if server_summary_main.empty:
            st.warning("No server summary data available.")
        else:
            if require_columns(
                server_summary_main,
                [
                    "server_id",
                    "batch_avg_load",
                    "interactive_avg_load",
                ],
                "server_summary",
            ):
                avg_load_type_df = server_summary_main[
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

                fig = px.bar(
                    avg_load_type_df,
                    x="server_id",
                    y="average_load",
                    color="load_type",
                    barmode="group",
                    title="Average Batch vs Interactive Load by Server",
                )

                fig.update_layout(
                    xaxis_title="Server ID",
                    yaxis_title="Average Load",
                )

                st.plotly_chart(fig, use_container_width=True)

    elif server_metric == "Peak Hours":
        if server_load_main.empty:
            st.warning("No server load timeseries available.")
        else:
            if require_columns(
                server_load_main,
                [
                    "time",
                    "active",
                    "load",
                    "batch_load",
                    "interactive_load",
                ],
                "server_load_timeseries",
            ):
                peak_hours_df = (
                    server_load_main
                    .groupby("time")
                    .agg(
                        active_servers=("active", "sum"),
                        total_load=("load", "sum"),
                        batch_load=("batch_load", "sum"),
                        interactive_load=("interactive_load", "sum"),
                    )
                    .reset_index()
                    .sort_values(
                        ["active_servers", "total_load"],
                        ascending=False,
                    )
                )

                st.subheader("Top Peak Hours")

                st.dataframe(
                    peak_hours_df.head(10),
                    use_container_width=True,
                )

                fig = px.line(
                    peak_hours_df.sort_values("time"),
                    x="time",
                    y="active_servers",
                    title="Active Servers Over Time",
                )

                fig.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Active Servers",
                )

                st.plotly_chart(fig, use_container_width=True)

    elif server_metric == "Status Heatmap":
        if server_load_main.empty:
            st.warning("No server load timeseries available.")
        else:
            if require_columns(
                server_load_main,
                ["server_id", "time", "status"],
                "server_load_timeseries",
            ):
                status_map = {
                    "idle": 0,
                    "active": 1,
                    "maintenance": 2,
                }

                heatmap_df = server_load_main.copy()

                heatmap_df["status_code"] = (
                    heatmap_df["status"]
                    .map(status_map)
                    .fillna(0)
                )

                heatmap_pivot = heatmap_df.pivot_table(
                    index="server_id",
                    columns="time",
                    values="status_code",
                    aggfunc="max",
                )

                fig = px.imshow(
                    heatmap_pivot,
                    aspect="auto",
                    title="Server Status Heatmap: Idle, Active, Maintenance",
                    labels={
                        "x": "Time",
                        "y": "Server ID",
                        "color": "Status",
                    },
                )

                fig.update_layout(
                    coloraxis_colorbar={
                        "tickvals": [0, 1, 2],
                        "ticktext": [
                            "Idle",
                            "Active",
                            "Maintenance",
                        ],
                    }
                )

                st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 14. ENERGY & PUE TAB
# ============================================================

with tab_energy:
    st.header(f"Energy & PUE: {main_scenario}")

    if hourly_main.empty:
        st.warning("No hourly energy data available for this scenario.")
    else:
        energy_metric = st.radio(
            "Energy metric",
            [
                "PUE Variation",
                "Power Consumption",
                "Energy Table",
            ],
            horizontal=True,
            key="energy_metric",
        )

        if energy_metric == "PUE Variation":
            if require_columns(
                hourly_main,
                ["time", "PUE"],
                "hourly_energy_thermal",
            ):
                fig = px.line(
                    hourly_main,
                    x="time",
                    y="PUE",
                    title="PUE Over Time",
                )

                fig.update_layout(
                    xaxis_title="Time",
                    yaxis_title="PUE",
                )

                st.plotly_chart(fig, use_container_width=True)

        elif energy_metric == "Power Consumption":
            if require_columns(
                hourly_main,
                ["time", "PIT_W", "Pcool_W", "Ptot_W"],
                "hourly_energy_thermal",
            ):
                power_df = hourly_main[
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

                fig = px.line(
                    power_df,
                    x="time",
                    y="power_watts",
                    color="power_type",
                    title="IT Power, Cooling Power, and Total Facility Power",
                )

                fig.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Power",
                )

                st.plotly_chart(fig, use_container_width=True)

        elif energy_metric == "Energy Table":
            st.dataframe(
                hourly_main,
                use_container_width=True,
            )


# ============================================================
# 15. MAINTENANCE TAB
# ============================================================

with tab_maintenance:
    st.header(f"Maintenance: {main_scenario}")

    maintenance_metric = st.radio(
        "Maintenance metric",
        [
            "PM Schedule",
            "Status Views",
        ],
        horizontal=True,
        key="maintenance_metric",
    )

    if maintenance_metric == "PM Schedule":
        if pm_main.empty and server_summary_main.empty:
            st.warning("No maintenance data available.")
        else:
            if (
                not server_summary_main.empty
                and "pm_scheduled" in server_summary_main.columns
                and "server_id" in server_summary_main.columns
            ):
                servers_with_pm = int(
                    server_summary_main["pm_scheduled"].sum())
                total_servers = server_summary_main["server_id"].nunique()

                col1, col2 = st.columns(2)

                col1.metric(
                    "Servers with PM",
                    servers_with_pm,
                )

                col2.metric(
                    "Total Servers",
                    total_servers,
                )

            if not pm_main.empty:
                pm_display = pm_main.copy()

                if "pm_scheduled" in pm_display.columns:
                    pm_display = pm_display[
                        pm_display["pm_scheduled"] == 1
                    ]

                if pm_display.empty:
                    st.info("No servers have scheduled maintenance.")
                else:
                    st.dataframe(
                        pm_display,
                        use_container_width=True,
                    )

    elif maintenance_metric == "Status Views":
        if server_load_main.empty:
            st.warning("No server status data available.")
        else:
            if require_columns(
                server_load_main,
                ["server_id", "status"],
                "server_load_timeseries",
            ):
                status_view = st.radio(
                    "Status view",
                    [
                        "Status Occurrences",
                        "Unique Servers by Status",
                        "Daily Server Classification",
                    ],
                    horizontal=True,
                    key="status_view",
                )

                if status_view == "Status Occurrences":
                    st.caption(
                        "Counts server-time-slot records. "
                        "Example: if one server is active for 10 slots, it contributes 10 active records."
                    )

                    status_occurrences = build_status_occurrence_counts(
                        server_load_main
                    )

                    if status_occurrences.empty:
                        st.warning("Could not calculate status occurrences.")
                    else:
                        fig = px.bar(
                            status_occurrences,
                            x="status",
                            y="server_slot_count",
                            text_auto=True,
                            title="Status Occurrences Across All Server-Time Slots",
                        )

                        fig.update_layout(
                            xaxis_title="Status",
                            yaxis_title="Server-Time Slot Count",
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        st.dataframe(
                            status_occurrences,
                            use_container_width=True,
                        )

                elif status_view == "Unique Servers by Status":
                    st.caption(
                        "Counts unique servers that appeared at least once in each status. "
                        "A server can count in more than one status."
                    )

                    unique_status_counts = build_unique_server_status_counts(
                        server_load_main
                    )

                    if unique_status_counts.empty:
                        st.warning(
                            "Could not calculate unique server status counts.")
                    else:
                        fig = px.bar(
                            unique_status_counts,
                            x="status",
                            y="unique_server_count",
                            text_auto=True,
                            title="Unique Servers That Appeared in Each Status",
                        )

                        fig.update_layout(
                            xaxis_title="Status",
                            yaxis_title="Unique Server Count",
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        st.dataframe(
                            unique_status_counts,
                            use_container_width=True,
                        )

                elif status_view == "Daily Server Classification":
                    st.caption(
                        "Each server is counted once using this priority rule: "
                        "maintenance > active > idle."
                    )

                    daily_status_counts = build_daily_server_status_counts(
                        server_load_main
                    )

                    daily_status_df = build_daily_server_classification(
                        server_load_main
                    )

                    if daily_status_counts.empty:
                        st.warning(
                            "Could not calculate daily server classification.")
                    else:
                        fig = px.bar(
                            daily_status_counts,
                            x="daily_status",
                            y="server_count",
                            text_auto=True,
                            title="Daily Server Classification",
                        )

                        fig.update_layout(
                            xaxis_title="Daily Status",
                            yaxis_title="Number of Servers",
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("Daily Status Counts")
                            st.dataframe(
                                daily_status_counts,
                                use_container_width=True,
                            )

                        with col2:
                            st.subheader("Server-Level Classification")
                            st.dataframe(
                                daily_status_df,
                                use_container_width=True,
                            )


# ============================================================
# 16. SCENARIO COMPARISON TAB
# ============================================================

with tab_compare:
    st.header("Scenario Comparison")

    if performance_comparison.empty:
        st.warning("No comparison data available.")
    else:
        comparison_metric = st.radio(
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

        if comparison_metric == "Total Cost":
            if require_columns(
                performance_comparison,
                ["scenario", "total_cost"],
                "performance_metrics",
            ):
                fig = px.bar(
                    performance_comparison,
                    x="scenario",
                    y="total_cost",
                    text_auto=".2f",
                    title="Total Cost by Scenario",
                )

                fig.update_layout(
                    xaxis_title="Scenario",
                    yaxis_title="Total Cost",
                )

                st.plotly_chart(fig, use_container_width=True)

                display_columns = [
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
                    if column in performance_comparison.columns
                ]

                st.dataframe(
                    performance_comparison[display_columns],
                    use_container_width=True,
                )

        elif comparison_metric == "Cost Breakdown":
            cost_columns = [
                "energy_cost",
                "pm_cost",
                "expected_cm_cost",
                "switching_cost",
                "lateness_cost",
            ]

            available_cost_columns = [
                column
                for column in cost_columns
                if column in performance_comparison.columns
            ]

            if available_cost_columns:
                cost_breakdown = performance_comparison[
                    ["scenario", *available_cost_columns]
                ].melt(
                    id_vars="scenario",
                    var_name="cost_type",
                    value_name="cost",
                )

                fig = px.bar(
                    cost_breakdown,
                    x="scenario",
                    y="cost",
                    color="cost_type",
                    barmode="stack",
                    title="Cost Breakdown by Scenario",
                )

                fig.update_layout(
                    xaxis_title="Scenario",
                    yaxis_title="Cost",
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No cost breakdown columns found.")

        elif comparison_metric == "Jobs Late":
            if require_columns(
                performance_comparison,
                ["scenario", "jobs_late_count"],
                "performance_metrics",
            ):
                fig = px.bar(
                    performance_comparison,
                    x="scenario",
                    y="jobs_late_count",
                    text_auto=True,
                    title="Late Jobs by Scenario",
                )

                fig.update_layout(
                    xaxis_title="Scenario",
                    yaxis_title="Late Jobs",
                )

                st.plotly_chart(fig, use_container_width=True)

        elif comparison_metric == "Average PUE":
            if require_columns(
                performance_comparison,
                ["scenario", "average_pue"],
                "performance_metrics",
            ):
                fig = px.bar(
                    performance_comparison,
                    x="scenario",
                    y="average_pue",
                    text_auto=".3f",
                    title="Average PUE by Scenario",
                )

                fig.update_layout(
                    xaxis_title="Scenario",
                    yaxis_title="Average PUE",
                )

                st.plotly_chart(fig, use_container_width=True)

            if (
                not hourly_comparison.empty
                and {"scenario", "time", "PUE"}.issubset(hourly_comparison.columns)
            ):
                fig_line = px.line(
                    hourly_comparison,
                    x="time",
                    y="PUE",
                    color="scenario",
                    title="PUE Variation by Scenario",
                )

                fig_line.update_layout(
                    xaxis_title="Time",
                    yaxis_title="PUE",
                )

                st.plotly_chart(fig_line, use_container_width=True)

        elif comparison_metric == "Energy Consumption":
            if require_columns(
                performance_comparison,
                ["scenario", "total_facility_energy_kwh"],
                "performance_metrics",
            ):
                fig = px.bar(
                    performance_comparison,
                    x="scenario",
                    y="total_facility_energy_kwh",
                    text_auto=".3f",
                    title="Total Facility Energy by Scenario",
                )

                fig.update_layout(
                    xaxis_title="Scenario",
                    yaxis_title="Total Facility Energy kWh",
                )

                st.plotly_chart(fig, use_container_width=True)

            required_power_columns = {
                "scenario",
                "time",
                "PIT_W",
                "Pcool_W",
                "Ptot_W",
            }

            if (
                not hourly_comparison.empty
                and required_power_columns.issubset(hourly_comparison.columns)
            ):
                power_comparison = hourly_comparison[
                    [
                        "scenario",
                        "time",
                        "PIT_W",
                        "Pcool_W",
                        "Ptot_W",
                    ]
                ].melt(
                    id_vars=["scenario", "time"],
                    var_name="power_type",
                    value_name="power_watts",
                )

                fig_power = px.line(
                    power_comparison,
                    x="time",
                    y="power_watts",
                    color="scenario",
                    line_dash="power_type",
                    title="Power Consumption by Scenario",
                )

                fig_power.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Power",
                )

                st.plotly_chart(fig_power, use_container_width=True)

        elif comparison_metric == "Server Utilization":
            if server_summary_comparison.empty:
                st.warning("No server summary data available for comparison.")
            else:
                if require_columns(
                    server_summary_comparison,
                    ["scenario", "utilization_rate_pct"],
                    "server_summary",
                ):
                    avg_utilization = (
                        server_summary_comparison
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

                    fig = px.bar(
                        avg_utilization,
                        x="scenario",
                        y="avg_utilization_pct",
                        text_auto=".2f",
                        title="Average Server Utilization by Scenario",
                    )

                    fig.update_layout(
                        xaxis_title="Scenario",
                        yaxis_title="Average Utilization (%)",
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    fig_box = px.box(
                        server_summary_comparison,
                        x="scenario",
                        y="utilization_rate_pct",
                        title="Server Utilization Distribution by Scenario",
                    )

                    fig_box.update_layout(
                        xaxis_title="Scenario",
                        yaxis_title="Utilization Rate (%)",
                    )

                    st.plotly_chart(fig_box, use_container_width=True)

        elif comparison_metric == "Server Status":
            if server_load_comparison.empty:
                st.warning("No server load data available for comparison.")
            else:
                if require_columns(
                    server_load_comparison,
                    ["scenario", "server_id", "status"],
                    "server_load_timeseries",
                ):
                    status_comparison_view = st.radio(
                        "Status comparison view",
                        [
                            "Status Occurrences",
                            "Unique Servers by Status",
                            "Daily Server Classification",
                        ],
                        horizontal=True,
                        key="status_comparison_view",
                    )

                    if status_comparison_view == "Status Occurrences":
                        status_occurrence_comparison = (
                            server_load_comparison
                            .groupby(["scenario", "status"])
                            .size()
                            .reset_index(name="server_slot_count")
                        )

                        fig = px.bar(
                            status_occurrence_comparison,
                            x="scenario",
                            y="server_slot_count",
                            color="status",
                            barmode="group",
                            text_auto=True,
                            title="Status Occurrences by Scenario",
                        )

                        fig.update_layout(
                            xaxis_title="Scenario",
                            yaxis_title="Server-Time Slot Count",
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        st.dataframe(
                            status_occurrence_comparison,
                            use_container_width=True,
                        )

                    elif status_comparison_view == "Unique Servers by Status":
                        unique_status_comparison = (
                            server_load_comparison
                            .groupby(["scenario", "status"])["server_id"]
                            .nunique()
                            .reset_index(name="unique_server_count")
                        )

                        fig = px.bar(
                            unique_status_comparison,
                            x="scenario",
                            y="unique_server_count",
                            color="status",
                            barmode="group",
                            text_auto=True,
                            title="Unique Servers by Status and Scenario",
                        )

                        fig.update_layout(
                            xaxis_title="Scenario",
                            yaxis_title="Unique Server Count",
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        st.dataframe(
                            unique_status_comparison,
                            use_container_width=True,
                        )

                    elif status_comparison_view == "Daily Server Classification":
                        daily_classification_rows = []

                        for scenario, scenario_df in server_load_comparison.groupby("scenario"):
                            daily_status_df = build_daily_server_classification(
                                scenario_df
                            )

                            if daily_status_df.empty:
                                continue

                            daily_status_df["scenario"] = scenario

                            daily_classification_rows.append(daily_status_df)

                        if not daily_classification_rows:
                            st.warning(
                                "Could not build daily status comparison.")
                        else:
                            daily_classification_comparison = pd.concat(
                                daily_classification_rows,
                                ignore_index=True,
                            )

                            daily_status_counts_comparison = (
                                daily_classification_comparison
                                .groupby(["scenario", "daily_status"])["server_id"]
                                .nunique()
                                .reset_index(name="server_count")
                            )

                            fig = px.bar(
                                daily_status_counts_comparison,
                                x="scenario",
                                y="server_count",
                                color="daily_status",
                                barmode="group",
                                text_auto=True,
                                title="Daily Server Classification by Scenario",
                            )

                            fig.update_layout(
                                xaxis_title="Scenario",
                                yaxis_title="Number of Servers",
                            )

                            st.plotly_chart(fig, use_container_width=True)

                            st.dataframe(
                                daily_status_counts_comparison,
                                use_container_width=True,
                            )


# ============================================================
# 17. SCENARIO LIBRARY TAB
# ============================================================

with tab_library:
    st.header("Scenario Library")

    st.write(
        "This table shows all scenarios currently detected from the output folders."
    )

    display_columns = [
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
        if column in performance_df.columns
    ]

    scenario_library = performance_df[display_columns].sort_values(
        "file_modified_time",
        ascending=False,
    )

    st.dataframe(
        scenario_library,
        use_container_width=True,
    )

    st.subheader("Raw loaded table sizes")

    table_sizes = pd.DataFrame(
        [
            {
                "table": table_name,
                "rows": len(df),
                "columns": len(df.columns) if not df.empty else 0,
            }
            for table_name, df in data.items()
        ]
    )

    st.dataframe(
        table_sizes,
        use_container_width=True,
    )
