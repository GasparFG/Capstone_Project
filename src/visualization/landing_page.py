import streamlit as st

st.title("Workload Prediction and Scheduling Optimization for Energy-Efficient, Reliable, and Service-Aware Data Centers")
st.write("Welcome")

if st.button("Go to Dashboard"):
    st.switch_page("visualization/dashboard.py")

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
The forecasting module generates a synthetic 24-hour job workload from historical patterns using a
three-stage generative ensemble pipeline trained on real production data from Alibaba's GPU cluster trace.

**Stage 0 — Workload Sampling**

Future jobs are unknown, so the pipeline first generates synthetic job descriptors by sampling
job type, role, and application name from recent historical distributions using non-parametric
empirical sampling. Interarrival times are sampled from the same trace, producing a realistic
arrival sequence of approximately 5,000 jobs for the planning horizon.

**Stage 1 — CPU & Memory Forecasting (Classification)**

Two ensemble classifiers predict discrete CPU and memory request classes for each synthetic job.
Base models — XGBoost and Random Forest — are trained using 5-fold time-series cross-validation,
and their out-of-fold (OOF) predictions are used to train three meta-learner strategies:

- **SoftVoting** — averages class probabilities across base models
- **Stack-LR** — logistic regression meta-learner trained on OOF predictions
- **Stack-XGB** — XGBoost meta-learner trained on OOF predictions

The best-performing strategy per target is selected via Wilcoxon signed-rank test against the
single-model XGBoost baseline (α = 0.05). SoftVoting was selected for CPU (macro-F1: **0.842**,
p = 4×10⁻⁶) and Stack-XGB for memory (macro-F1: **0.630**, p = 3×10⁻⁷).

**Stage 2 — Duration Forecasting (Regression)**

Job duration is predicted using the Stage 1 *predicted* CPU and memory values — not the actual
ones — to maintain train-serving consistency and avoid data leakage. Three regression ensembles
are evaluated (WeightedAvg, Stack-Ridge, Stack-XGB), all trained exclusively on OOF predictions
from the training set. Stack-XGB was selected as best (RMSLE: **1.155**, p = 0.0069), reducing
error on high-duration outliers compared to the XGBoost baseline.

All ensemble comparisons are validated with the Wilcoxon signed-rank test.
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
