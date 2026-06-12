from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# 1. STREAMLIT CONFIG & MOBILE DETECTION
# ============================================================

st.set_page_config(
    page_title="Data Center Optimization Dashboard",
    page_icon="📊",
    layout="wide",
)

# Detect if the mobile flag is passed via the URL parameter (?mobile=true)
IS_MOBILE = st.query_params.get("mobile", "false") == "true"

if IS_MOBILE:
    st.info("📱 Mobile View Activated")
else:
    st.info("🖥️ Desktop View Activated")

# ============================================================
# 1.1 DYNAMIC DIMENSION CONFIGURATION
# ============================================================

if IS_MOBILE:
    DASHBOARD_CHART_HEIGHT = 320
    DASHBOARD_SERVERS_PER_ROW = 3  # Fewer columns on mobile to prevent layout overflow

    DASHBOARD_TITLE_SIZE = 16
    DASHBOARD_AXIS_TITLE_SIZE = 11
    DASHBOARD_AXIS_TICK_SIZE = 9
    DASHBOARD_LEGEND_SIZE = 9
    FONT_BASE_PX = "15px"
else:
    DASHBOARD_CHART_HEIGHT = 550
    DASHBOARD_SERVERS_PER_ROW = 10  # Expanded grid for widescreen layouts

    DASHBOARD_TITLE_SIZE = 24
    DASHBOARD_AXIS_TITLE_SIZE = 17
    DASHBOARD_AXIS_TICK_SIZE = 15
    DASHBOARD_LEGEND_SIZE = 15
    FONT_BASE_PX = "20px"

# ============================================================
# 1.2 ADAPTIVE CSS INJECTION
# ============================================================
st.markdown(
    f"""
    <style>
    html, body, [class*="css"] {{
        font-size: {FONT_BASE_PX} !important;
    }}

    .stApp {{
        font-size: {FONT_BASE_PX} !important;
    }}

    /* Responsive Headings */
    h1 {{ font-size: {"1.8rem" if IS_MOBILE else "2.8rem"} !important; }}
    h2 {{ font-size: {"1.5rem" if IS_MOBILE else "2.35rem"} !important; }}
    h3 {{ font-size: {"1.2rem" if IS_MOBILE else "1.75rem"} !important; }}

    p, span, label, div {{
        font-size: {"0.9rem" if IS_MOBILE else "1.03rem"};
    }}

    /* Streamlit Native Metric Overrides */
    div[data-testid="stMetricLabel"] {{ font-size: {"0.9rem" if IS_MOBILE else "1.15rem"} !important; }}
    div[data-testid="stMetricValue"] {{ font-size: {"1.5rem" if IS_MOBILE else "2.15rem"} !important; }}

    /* Main KPI Card Container */
    .overview-section {{
        border: 1px solid rgba(250, 250, 250, 0.12);
        border-radius: 14px;
        padding: {"12px" if IS_MOBILE else "22px"};
        margin-bottom: 20px;
        background-color: rgba(255, 255, 255, 0.03);
    }}

    .big-total-cost {{
        font-size: {"2.0rem" if IS_MOBILE else "3.45rem"} !important;
        font-weight: 850;
        line-height: 1.1;
        text-align: center;
        color: #56CCF2;
    }}

    .big-total-cost-label {{
        font-size: {"0.9rem" if IS_MOBILE else "1.2rem"};
        opacity: 0.8;
        text-align: center;
        margin-bottom: 10px;
    }}

    /* Server Grid Infrastructure UI Components */
    .server-grid-card {{
        border-radius: 6px;
        padding: 6px 2px;
        margin: 3px 0px;
        text-align: center;
        font-weight: 700;
        color: #111111;
        min-height: {"45px" if IS_MOBILE else "52px"};
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.25);
    }}

    .server-grid-id {{ font-size: {"11px" if IS_MOBILE else "14px"}; }}
    .server-grid-status {{ font-size: {"8px" if IS_MOBILE else "10px"}; }}

    .heatmap-clock-container {{ text-align: center; margin: 10px 0; }}
    .heatmap-clock-time {{
        font-size: {"1.2rem" if IS_MOBILE else "1.7rem"};
        font-weight: 850;
        padding: 4px 12px;
        border-radius: 8px;
        background-color: rgba(255,255,255,0.04);
        display: inline-block;
        border: 1px solid rgba(255,255,255,0.18);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 2. PATH CONFIGURATION & COLOR MAPS
# ============================================================

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]

PROJECT_ROOT = get_project_root()

def get_output_priority_dirs() -> list[Path]:
    """ Defines folder lookup hierarchy for optimization tables. """
    return [
        PROJECT_ROOT / "dashboard_dt",
        PROJECT_ROOT / "outputs" / "optimization",
        PROJECT_ROOT / "outputs" / "results" / "tables",
    ]

OUTPUT_PRIORITY_DIRS = get_output_priority_dirs()

DASHBOARD_NAVY = "#0B1F5B"
DASHBOARD_STATUS_COLOR_MAP = {
    "Active": "#2ECC71",
    "Maintenance": "#F1C40F",
    "Idle": "#E74C3C",
    "Unknown": "#7F8C8D",
}
DASHBOARD_COST_COLOR_MAP = {
    "Energy": "#3498DB",
    "PM": "#9B59B6",
    "CM": "#E67E22",
    "Switch": "#95A5A6",
    "Lateness": "#E74C3C",
}
DASHBOARD_POWER_COLOR_MAP = {
    "IT Power": "#7EC8FF",
    "Cooling Power": "#1565C0",
    "Total Facility Power": "#FFB3B3",
}

# ============================================================
# 3. FILE DISCOVERY & LOADING FUNCTIONS
# ============================================================

def find_matching_files_with_priority(patterns: list[str]) -> list[Path]:
    """ Scans directories sequentially according to priority rules. """
    for folder in OUTPUT_PRIORITY_DIRS:
        if not folder.exists():
            continue
        matching_files = []
        for pattern in patterns:
            matching_files.extend(folder.glob(pattern))
        matching_files = sorted(set(matching_files))
        valid_files = [f for f in matching_files if f.exists() and f.stat().st_size > 0]
        if valid_files:
            return valid_files
    return []

def read_dashboard_table(file_path: Path) -> pd.DataFrame:
    """ Fallback encoding reader safely routing CSV and Excel inputs. """
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(file_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(file_path, encoding="latin1", sep=None, engine="python")
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    raise ValueError(f"Unsupported file type: {file_path.suffix}")

def infer_dashboard_scenario_from_filename(file_name: str, table_prefix: str) -> str:
    """ Strips timestamp strings to isolate core scenario identifier strings. """
    stem = Path(file_name).stem
    scenario = stem.replace(table_prefix, "", 1).strip("_") if stem.startswith(table_prefix) else stem
    parts = scenario.split("_")
    if parts and ("-" in parts[-1] or (parts[-1].isdigit() and len(parts[-1]) >= 6)):
        parts = parts[:-1]
    return "_".join(parts).strip("_") or "unknown"

def read_and_combine_dashboard_files(patterns: list[str], table_prefix: str) -> pd.DataFrame:
    """ Discovers and unifies data slices across targeted execution metrics. """
    dashboard_files = find_matching_files_with_priority(patterns)
    if not dashboard_files:
        return pd.DataFrame()
    
    dashboard_dataframes = []
    for file_path in dashboard_files:
        try:
            df = read_dashboard_table(file_path)
            df["source_file"] = file_path.name
            df["file_modified_time"] = pd.to_datetime(file_path.stat().st_mtime, unit="s")
            if "scenario" not in df.columns:
                df["scenario"] = infer_dashboard_scenario_from_filename(file_path.name, table_prefix)
            dashboard_dataframes.append(df)
        except Exception:
            continue
    return pd.concat(dashboard_dataframes, ignore_index=True) if dashboard_dataframes else pd.DataFrame()

def keep_latest_dashboard_run_per_scenario(df: pd.DataFrame) -> pd.DataFrame:
    """ Deduplicates dataframes by pulling only the newest run per scenario tag. """
    if df.empty or not {"scenario", "source_file", "file_modified_time"}.issubset(df.columns):
        return df
    latest_files = df[["scenario", "source_file", "file_modified_time"]].drop_duplicates().sort_values("file_modified_time").groupby("scenario", as_index=False).tail(1)
    return df.merge(latest_files[["scenario", "source_file"]], on=["scenario", "source_file"], how="inner")

@st.cache_data(ttl=60)
def load_all_available_dashboard_outputs() -> dict[str, pd.DataFrame]:
    """ Primary batch caching engine loading operational optimization metrics. """
    return {
        "performance": keep_latest_dashboard_run_per_scenario(read_and_combine_dashboard_files(["*performance_metrics*.csv", "*performance_metrics*.xlsx"], "performance_metrics")),
        "solution": keep_latest_dashboard_run_per_scenario(read_and_combine_dashboard_files(["*optimization_solution*.csv", "*optimization_solution*.xlsx"], "optimization_solution")),
        "server_summary": keep_latest_dashboard_run_per_scenario(read_and_combine_dashboard_files(["*server_summary*.csv", "*server_summary*.xlsx"], "server_summary")),
        "hourly_energy": keep_latest_dashboard_run_per_scenario(read_and_combine_dashboard_files(["*hourly_energy_thermal*.csv", "*hourly_energy_thermal*.xlsx"], "hourly_energy_thermal")),
        "server_load": keep_latest_dashboard_run_per_scenario(read_and_combine_dashboard_files(["*server_load_timeseries*.csv", "*server_load_timeseries*.xlsx"], "server_load_timeseries")),
    }

# ============================================================
# 4. RENDERING & FORMATTING HELPERS
# ============================================================

def apply_dashboard_chart_font(fig):
    """ Maps text dimensions dynamically to standard Plotly chart frames. """
    fig.update_layout(
        height=DASHBOARD_CHART_HEIGHT,
        title_font=dict(size=DASHBOARD_TITLE_SIZE),
        font=dict(size=DASHBOARD_AXIS_TICK_SIZE),
        legend=dict(font=dict(size=DASHBOARD_LEGEND_SIZE), title=dict(font=dict(size=DASHBOARD_LEGEND_SIZE))),
        xaxis=dict(title_font=dict(size=DASHBOARD_AXIS_TITLE_SIZE), tickfont=dict(size=DASHBOARD_AXIS_TICK_SIZE)),
        yaxis=dict(title_font=dict(size=DASHBOARD_AXIS_TITLE_SIZE), tickfont=dict(size=DASHBOARD_AXIS_TICK_SIZE)),
        margin=dict(l=10, r=10, t=40, b=10) if IS_MOBILE else dict(l=40, r=40, t=50, b=40)
    )
    return fig

def apply_dashboard_pie_font(fig):
    """ Restructures pie charts dynamically to drop messy text overlays on mobile. """
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label" if not IS_MOBILE else "percent",
        textfont=dict(size=12 if IS_MOBILE else 18, color="white"),
    )
    fig.update_layout(
        showlegend=True,
        height=DASHBOARD_CHART_HEIGHT,
        margin=dict(l=5, r=5, t=5, b=5),
        legend=dict(font=dict(size=DASHBOARD_LEGEND_SIZE))
    )
    return fig

def format_dashboard_currency(val) -> str:
    return f"${float(val):,.2f}" if pd.notna(val) else "$0.00"

def format_dashboard_number(val, decimals=2) -> str:
    return f"{float(val):,.{decimals}f}" if pd.notna(val) else "0.00"

def format_dashboard_slot_as_time(slot) -> str:
    """ Computes the string representation of 15-minute engineering step windows. """
    total_mins = int(float(slot)) * 15
    return f"{(total_mins // 60) % 24:02d}:{total_mins % 60:02d}"

def normalize_dashboard_status_label(v) -> str:
    w = str(v).strip().lower()
    return "Active" if w == "active" else "Maintenance" if w == "maintenance" else "Idle" if w == "idle" else "Unknown"

def render_dashboard_server_status_grid(df, servers_per_row) -> None:
    """ Generates responsive grid mapping matrix displaying server status maps. """
    if df.empty:
        st.warning("No data found for this timestep slot")
        return
    df_grid = df[["server_id", "status"]].drop_duplicates(subset=["server_id"]).sort_values("server_id")
    servers = df_grid.to_dict("records")
    
    for i in range(0, len(servers), servers_per_row):
        chunk = servers[i: i + servers_per_row]
        cols = st.columns(servers_per_row)
        for idx, s in enumerate(chunk):
            status = normalize_dashboard_status_label(s["status"])
            color = DASHBOARD_STATUS_COLOR_MAP.get(status, "#7F8C8D")
            with cols[idx]:
                st.markdown(
                    f"""
                    <div class="server-grid-card" style="background-color:{color};">
                        <div class="server-grid-id">S{s['server_id']}</div>
                        <div class="server-grid-status">{status[:3]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ============================================================
# 5. DATA LOADING & SIDEBAR MANAGEMENT
# ============================================================

dashboard_data = load_all_available_dashboard_outputs()
df_perf = dashboard_data["performance"]
df_load = dashboard_data["server_load"]
df_energy = dashboard_data["hourly_energy"]

st.sidebar.title("🛠️ Controls")
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

if df_perf.empty:
    st.title("Data Center Dashboard")
    st.warning("No tracking configuration metrics found.")
    st.stop()

scenarios = sorted(df_perf["scenario"].dropna().unique())
main_scenario = st.sidebar.selectbox("Main Scenario:", scenarios)
comp_scenarios = st.sidebar.multiselect("Compare with:", scenarios, default=scenarios[:2])

df_perf_main = df_perf[df_perf["scenario"] == main_scenario]
df_load_main = df_load[df_load["scenario"] == main_scenario]
df_energy_main = df_energy[df_energy["scenario"] == main_scenario]
df_perf_comp = df_perf[df_perf["scenario"].isin(comp_scenarios)]

# ============================================================
# 6. MAIN MULTI-TAB DISPLAY INTERFACE
# ============================================================

st.title("Data Center Job Scheduling Optimization")

tabs = st.tabs(["📊 Overview", "🖥️ Server Grid", "⚡ Energy Profiles", "🔄 Comparison"])

# ------------------------------------------------------------
# TAB 1: OVERVIEW METRICS
# ------------------------------------------------------------
with tabs[0]:
    st.write(f"### Selected Configuration Model: {main_scenario}")
    if not df_perf_main.empty:
        row = df_perf_main.iloc[0]
        
        # Large Total Objective Cost Hero Display Container
        cost_val = row.get("total_cost", 0)
        st.markdown(
            f"""
            <div class="overview-section">
                <div class="big-total-cost">{format_dashboard_currency(cost_val)}</div>
                <div class="big-total-cost-label">Total Optimization Objective Cost</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Responsive KPI columns
        kpi_col1, kpi_col2 = st.columns(2)
        with kpi_col1:
            st.metric("Avg PUE Score", format_dashboard_number(row.get("average_pue", 1.0), 3))
        with kpi_col2:
            st.metric("Tardy Late Jobs", int(row.get("jobs_late_count", 0)))
            
        st.markdown("---")
        st.write("#### 🪙 Cost Component Shares Breakdown")
        
        # Isolate cost component attributes from dataframe row metrics
        cost_segments = []
        for label, col in [("Energy", "energy_cost"), ("PM", "pm_cost"), ("CM", "cm_cost"), ("Switch", "switch_cost"), ("Lateness", "lateness_cost")]:
            if col in row.index:
                cost_segments.append({"Component": label, "Cost": float(row[col])})
                
        df_pie = pd.DataFrame(cost_segments)
        if not df_pie.empty and df_pie["Cost"].sum() > 0:
            fig_pie = px.pie(df_pie, values="Cost", names="Component", color="Component", color_discrete_map=DASHBOARD_COST_COLOR_MAP, hole=0.3)
            st.plotly_chart(apply_dashboard_pie_font(fig_pie), use_container_width=True)

# ------------------------------------------------------------
# TAB 2: INFRASTRUCTURE STATE VIEW
# ------------------------------------------------------------
with tabs[1]:
    st.write("### 📋 Infrastructure Spatiotemporal State Grid Map")
    if df_load_main.empty:
        st.info("No load timeline profile logs available.")
    else:
        slots = sorted(df_load_main["slot"].dropna().unique())
        selected_slot = st.select_slider("Horizon Planning Index (15-Min Granular Intervals):", options=slots, value=slots[0], format_func=format_dashboard_slot_as_time)
        
        st.markdown(
            f"""
            <div class="heatmap-clock-container">
                <div class="heatmap-clock-time">⏱️ Simulation Clock Target Window: {format_dashboard_slot_as_time(selected_slot)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        df_snapshot = df_load_main[df_load_main["slot"] == selected_slot]
        render_dashboard_server_status_grid(df_snapshot, DASHBOARD_SERVERS_PER_ROW)

# ------------------------------------------------------------
# TAB 3: ENERGY PROFILE PATTERNS
# ------------------------------------------------------------
with tabs[2]:
    st.write("### ⚡ Thermal Telemetry & PUE Load Curves")
    if df_energy_main.empty:
        st.info("No thermal tracking data logs captured.")
    else:
        x_axis = "hour" if "hour" in df_energy_main.columns else "slot"
        
        # Power Load Profile Distributions
        power_vars = [c for c in ["PIT_W", "Pcool_W", "Ptot_W"] if c in df_energy_main.columns]
        if power_vars:
            df_melt = df_energy_main.melt(id_vars=[x_axis], value_vars=power_vars, var_name="Metric", value_name="Watts")
            df_melt["Metric"] = df_melt["Metric"].map({"PIT_W": "IT Power", "Pcool_W": "Cooling Power", "Ptot_W": "Total Facility Power"})
            fig_pow = px.line(df_melt, x=x_axis, y="Watts", color="Metric", title="Power Profiles Over Time", color_discrete_map=DASHBOARD_POWER_COLOR_MAP)
            st.plotly_chart(apply_dashboard_chart_font(fig_pow), use_container_width=True)
            
        # PUE Trend Curve
        pue_col = "pue" if "pue" in df_energy_main.columns else ("PUE" if "PUE" in df_energy_main.columns else None)
        if pue_col:
            fig_pue = px.line(df_energy_main, x=x_axis, y=pue_col, title="Instantaneous PUE Performance Curves")
            fig_pue.update_traces(line_color="#E67E22", line_width=2)
            st.plotly_chart(apply_dashboard_chart_font(fig_pue), use_container_width=True)

# ------------------------------------------------------------
# TAB 4: BENCHMARKING & SCENARIO COMPARISONS
# ------------------------------------------------------------
with tabs[3]:
    st.write("### 🔄 Cross-Scenario Multiobjective Benchmarking")
    if df_perf_comp.empty:
        st.info("Select alternative evaluation tracks from the configuration sidebar panel.")
    else:
        # Cross-Scenario Total Financial Costs Comparison Bar Graph
        fig_comp = px.bar(df_perf_comp, x="scenario", y="total_cost", color="scenario", title="Objective Performance Cost Benchmarking (Lower values are Optimal)")
        st.plotly_chart(apply_dashboard_chart_font(fig_comp), use_container_width=True)
        
        # Clean Datatable Matrix Summary
        st.markdown("#### Direct Scenario Metrics Grid Reference View")
        disp_cols = [c for c in ["scenario", "total_cost", "average_pue", "jobs_late_count"] if c in df_perf_comp.columns]
        st.dataframe(df_perf_comp[disp_cols], use_container_width=True, hide_index=True)