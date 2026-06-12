import streamlit as st

st.title("Workload Prediction and Scheduling Optimization for Energy-Efficient, Reliable, and Service-Aware Data Centers")
st.write("Welcome")

col1, col2 = st.columns(2)

with col1:
    st.link_button(
        "🖥️ Desktop Dashboard",
        "https://dashboard-workload-optimization-and-scheduling.streamlit.app"
    )

with col2:
    st.link_button(
        "📱 Mobile Dashboard",
        "https://dashboard-workload-optimization-and-scheduling.streamlit.app/?mobile=true"
    )

st.divider()

# ---------------- EXECUTIVE SUMMARY ----------------
st.header("Executive Summary")

st.markdown("""
Modern data centers must continuously balance three competing objectives:
**high performance**, **energy efficiency**, and **service reliability**.
Traditional scheduling approaches often react to workload demand as it arrives,
which can lead to inefficient resource utilization, higher operating costs,
and increased risk of service degradation.

This capstone project presents a **forecast-driven workload scheduling framework**
designed to support intelligent, proactive decision-making in data center operations.
Using **12,374 real production jobs** from Alibaba's GPU cluster trace, the system
combines machine learning forecasting with mathematical optimization to anticipate
future workload demand and allocate computing resources more efficiently.

The forecasting component predicts future job arrivals, resource requirements,
and execution durations. These forecasts are then used as inputs to a
**Mixed-Integer Linear Programming (MILP)** model that schedules workloads across
**42 servers** over a **24-hour planning horizon**.

The optimization framework simultaneously considers energy consumption,
cooling costs, maintenance requirements, reliability constraints, and
service-level performance. By evaluating these objectives together,
the model identifies scheduling decisions that are difficult to achieve
using conventional reactive approaches.

Results demonstrate that predictive analytics and mathematical optimization
can be combined to improve workload management, increase operational efficiency,
and support more sustainable and reliable data center operations.
""")

# ---------------- ARCHITECTURE ----------------
st.header("Architecture Diagram")
st.image("docs/architecture_diagram.png")

# ---------------- FORECASTING ----------------
st.header("Forecasting Module")
st.markdown("""
HERE GOES THE FORECASTING SUMMARY
BLABLABLABLABLABLABLA
""")

# ---------------- OPTIMIZATION ----------------
st.header("Optimization Module")
st.markdown("""
The MILP model is the decision-making base of the project. It focuses on scheduling jobs across physical servers over 
96 time slots of 15 minutes each (one full day), and it handles five operational dimensions simultaneously:

The model simultaneously optimizes five operational dimensions:

- **Energy Efficiency:** Minimizes operational costs and Power Usage
  Effectiveness (PUE) by reducing energy waste from cooling and
  infrastructure overhead.

- **Reliability and Maintenance:** Schedules preventive and corrective
  maintenance activities while enforcing redundancy requirements for
  critical workloads.

- **Thermal Management:** Maintains server inlet temperatures within
  ASHRAE A1 operating limits to ensure safe and efficient operation.

- **Workload Prioritization:** Differentiates between batch and
  interactive jobs, reserving capacity for latency-sensitive workloads.

- **Server State Management:** Minimizes unnecessary server power
  transitions and supports affinity and anti-affinity scheduling
  requirements.

The model produces an optimized daily schedule that specifies workload
placement, server activation states, and maintenance windows while
satisfying capacity, reliability, and operational constraints.
""")

# ---------------- LINKS ----------------
st.link_button(
    "💻 View GitHub Repository",
    "https://github.com/GasparFG/Workload_Prediction_and_Scheduling_Optimization_for_Energy_Efficient_Reliable_and_Service-Aware_DC"
)

# ---------------- TEAM ----------------
st.header("Team Members")

st.markdown("""
### Ana Carolina Quiroz Vazquez
ana.quiroz2010@myunfc.ca

### Gaspar Franco Garcia
gaspar.franco6692@myunf.ca

### Maria Jose Agualimpia Martinez
maria.agualimpia7916@myunfc.ca

### Monica Giselle Mendoza Mendoza
monica.mendoza5170@myunfc.ca
""")
