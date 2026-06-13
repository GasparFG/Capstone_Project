# Workload Prediction and Scheduling Optimization for Energy-Efficient, Reliable, and Service-Aware Data Centers

**Authors:** Ana Carolina Quiroz Vazquez · Gaspar Franco Garcia · Maria Jose Agualimpia Martinez · Monica Giselle Mendoza Mendoza  
**Instructor:** Hany Osman, PhD. PEng | University of Niagara Falls Canada  

An end-to-end pipeline that forecasts datacenter workloads and schedules jobs using a Mixed-Integer Linear Program (MILP) to minimize energy cost, cooling overhead, and maintenance risk — while meeting job deadlines and reliability constraints.

---

## Project Overview

The pipeline has three sequential stages:

1. **Data Cleaning** — processes a raw DLRM job trace into a clean, feature-engineered dataset
2. **Ensemble Forecasting** — generates a synthetic 24-hour job workload from historical distributions using a three-stage generative model
3. **MILP Optimization** — schedules the forecasted jobs across 42 physical servers, minimizing a multi-objective cost function subject to capacity, thermal, maintenance, and redundancy constraints

A Streamlit dashboard and a round-robin baseline are also included for visualization and comparison.

---

## Repository Structure

```
Capstone_Project/
│
├── src/
│   ├── data/
│   │   ├── load_and_merge_data.py     # Merges raw data sources
│   │   └── clean_data.py              # Cleaning, feature engineering, outlier removal
│   │
│   ├── forecasting/
│   │   ├── ensemble_model.py          # 3-stage ensemble workload generator (main model)
│   │   ├── forecast_model.py          # Single XGBoost baseline forecaster
│   │   ├── recompute_baseline_metrics.py
│   │   └── run_pipeline.py            # End-to-end pipeline runner
│   │
│   ├── optimization/
│   │   ├── solver.py                  # MILP model (Gurobi)
│   │   ├── cli.py                     # CLI entry point
│   │   ├── build_jobs_json_from_forecast.py  # Forecast → optimizer JSON
│   │   ├── scenario_builder.py        # Stress-test scenario generator
│   │   ├── data_loader.py             # Loads JSON inputs for solver
│   │   ├── output_writer.py           # Saves solver outputs
│   │   ├── result_extractor.py        # Extracts metrics from solved model
│   │   ├── update_psi_0.py            # Propagates wear state across daily runs
│   │   ├── config.py                  # Path and Gurobi status constants
│   │   └── utils.py                   # Shared helpers
│   │
│   └── visualization/
│       ├── dashboard.py               # Streamlit dashboard
│       ├── landing_page.py            # Landing page
│       └── plots.py                   # Plot helpers
│
├── baselines/
│   └── baseline_round_robin.py        # Round-robin heuristic for comparison
│
├── data/
│   ├── raw/                           # Raw DLRM trace + server parameters
│   ├── interim/                       # Cleaned data (parquet)
│   ├── processed/                     # Forecast datasets + optimizer inputs
│   ├── forecast/                      # Forecast outputs and metrics
│   ├── thermal_parameters.json        # Thermal model parameters
│   └── robustness_mid/                # Robustness test configurations
│
├── outputs/                           # Optimization results, reports, tables
├── docs/                              # Architecture diagram, final report
├── archive/                           # Benchmark experiments (historical)
├── requirements.txt
└── run_pipeline.py                    # Top-level pipeline runner
```

---

## Pipeline

### Step 1 — Data Cleaning (`src/data/clean_data.py`)

Reads raw job trace parquet, applies the following cleaning steps in order:

- Converts timestamps to timedelta
- Removes missing or negative-duration records
- Derives `duration_minutes` and classifies jobs: **interactive** (≤ 60 min) or **batch** (> 60 min)
- Filters out zero/negative CPU and memory requests
- Removes duplicates
- Removes outliers via IQR on `duration_minutes`
- Drops jobs longer than 24 hours

Outputs cleaned parquet to `data/interim/cleaned_data.parquet`.

### Step 2 — Ensemble Forecasting (`src/forecasting/ensemble_model.py`)

A three-stage generative pipeline that produces a synthetic 24-hour job queue:

**Stage 0 — Synthetic Workload Generation**  
Samples interarrival times, `job_type`, `role`, and `app_name` from recent historical distributions to generate plausible future jobs.

**Stage 1 — CPU & Memory Estimation (Classification)**  
Two classifiers predict discrete CPU and memory request classes per job, using temporal lag features and job descriptor features. Three ensemble strategies are evaluated:
- *SoftVoting* — averages class probabilities across base models
- *Stack_LR* — logistic regression meta-learner trained on out-of-fold (OOF) predictions
- *Stack_XGB* — XGBoost meta-learner trained on OOF predictions

Base models: `XGBClassifier`, `RandomForestClassifier`. Training uses `TimeSeriesSplit(5)` to generate OOF predictions without leakage.

**Stage 2 — Duration Estimation (Regression)**  
Predicts job duration in seconds, using the Stage 1 predicted CPU/RAM (not actual values) for train-serving consistency. Three ensemble strategies:
- *WeightedAvg* — inverse-RMSLE weighted average from training OOF
- *Stack_Ridge* — Ridge meta-learner trained on OOF predictions
- *Stack_XGB_dur* — XGBoost meta-learner trained on OOF predictions

Base models: `XGBoost_log`, `RandomForest`, `Ridge_log`. Duan (1983) bias correction is applied to log-transformed models at evaluation; meta-features use `expm1`-only for scale consistency.

Statistical validation uses the Wilcoxon signed-rank test to compare ensemble vs. XGBoost baseline.

Outputs:
- `data/forecast/output/ensemble_forecast.csv` — raw 24-hour job forecast
- `data/processed/optimization_ensemble_jobs.parquet` — formatted for the optimizer
- `data/forecast/metrics/` — full per-target metrics tables and comparison summaries

### Step 3 — MILP Optimization (`src/optimization/solver.py`)

Schedules the forecasted jobs using Gurobi. The model minimizes:

```
min  energy_cost + pm_cost + cm_cost + switching_cost + lateness_cost
```

**Decision variables:**
- `X[i,j,k]` — binary: job `i` starts on server `j` at time slot `k`
- `y[j,k]` — binary: server `j` is active at slot `k`
- `v[j,k]` — binary: preventive maintenance (PM) starts on server `j` at slot `k`
- `z[j,k]` — binary: server `j` is under PM at slot `k`
- `psi[j,k]` — continuous: cumulative wear on server `j` at slot `k`
- `L[j,k]`, `H[j,k]`, `PIT[k]`, `Pcool[k]`, `Ptot[k]` — load, heat, and power tracking

**Key constraints:**
- Job assignment with exact replica counts (critical jobs require 2 replicas on different rack failure domains)
- Server capacity limits; interactive jobs have hard deadlines
- PM can only trigger when wear `psi ≥ Lambda[j]`; resets wear to 0 during PM window
- Thermal constraint: server inlet temperature bounded by supply + recirculation heat
- PUE cap, hot standby buffer, switching budget
- Anti-affinity (critical and non-critical job isolation) and affinity constraints

Wear state (`psi_0`) is propagated across daily runs via `update_psi_0.py`, creating a multi-day stateful optimization cycle.

**Scenarios** (via `--run-scenarios` flag):
- `base` — standard forecast workload
- `high_demand_25pct` — demand scaled up 25%
- `high_energy_cost_50pct` — energy prices scaled up 50%
- `reduced_capacity_10pct` — server capacity reduced 10%

---

## Running the Pipeline

### Full end-to-end run

```bash
python run_pipeline.py
```

Optional flags:
```bash
python run_pipeline.py --skip-forecast    # reuse existing forecast
python run_pipeline.py --skip-solver      # skip optimization step
python run_pipeline.py --forecast-only    # run forecasting only
```

### Optimization only (with CLI options)

```bash
python -m src.optimization.cli \
    --jobs-json-input data/processed/optimization_jobs_params.json \
    --server-json-input data/raw/server_params_42servers_v6.json \
    --time-limit 120 \
    --mip-gap 0.02 \
    --run-scenarios
```

### Dashboard

```bash
streamlit run src/visualization/dashboard.py
```

---

## Baseline Comparison

`baselines/baseline_round_robin.py` implements a round-robin scheduler using the same power model, cost parameters, and input data as the MILP — but without any lookahead, energy awareness, or PM scheduling. It serves as the apples-to-apples comparison for evaluating the optimizer's improvements in energy cost, CM cost, and lateness.

---

## Results

The MILP model was validated against the round-robin baseline across 12 robustness scenarios (varying servers, jobs, slot sizes, and server wear levels). All comparisons are statistically significant at α = 0.05 (Wilcoxon signed-rank test).

| Metric | Round Robin | MILP | Improvement |
|---|---|---|---|
| Server utilization | 57.59% | 32.56% | −25 pp |
| Total energy | 538.50 kWh | 354.89 kWh | −34% |
| Total cost (USD) | $884.48 | $615.51 | −30% |

Forecasting results (ensemble vs. XGBoost baseline, all p < 0.05):

| Target | Best ensemble | Metric | Score |
|---|---|---|---|
| CPU request | SoftVoting | macro-F1 | 0.842 |
| Memory request | Stack-XGB | macro-F1 | 0.630 |
| Duration | Stack-XGB | RMSLE | 1.155 |

---

## Data Sources

| Source | Description |
|---|---|
| Alibaba Cluster Trace GPU v2025 | Primary job trace |
| External benchmarks | Infrastructure parameters |
| 60 days AI-synthetic workloads | Supplementary training data |

| File | Description |
|---|---|
| `data/raw/disaggregated_DLRM_trace.csv` | Raw job trace |
| `data/raw/server_params_42servers_v6.json` | Server hardware parameters |
| `data/thermal_parameters.json` | Thermal model (T_sup, T_busy, T_idle, recirculation matrix D) |
| `data/interim/cleaned_data.parquet` | Cleaned job dataset |
| `data/processed/optimization_ensemble_jobs.parquet` | Forecasted jobs for optimizer |
| `data/forecast/output/ensemble_forecast.csv` | 24-hour synthetic workload |
| `data/forecast/metrics/` | Model comparison tables, Wilcoxon results |

---

## Requirements

```
streamlit
pandas
plotly
openpyxl
xgboost
scikit-learn
scipy
gurobipy       # requires a valid Gurobi license (WLS or academic)
numpy
```

Install with:
```bash
pip install -r requirements.txt
```

> **Note:** The MILP solver requires a [Gurobi license](https://www.gurobi.com/solutions/licensing/). A free academic license is available. The WLS credentials are currently embedded in `solver.py` — replace with your own before running.

---

## Limitations

- Missing precise rack topology data
- 2023/2025 data mismatches between trace and benchmarks
- Reliance on 60 days of AI-synthetic workloads
- Static day-ahead planning horizon (no real-time rescheduling)
- Two server types only; 24-hour maximum job duration

---

## Outputs

After a full pipeline run, results are written to:

```
outputs/
├── optimization/
│   ├── optimization_solution.csv      # Job-to-server assignment for each scenario
│   ├── performance_metrics.csv        # Cost breakdown per scenario
│   └── server_summary_base_*.csv      # Per-server utilization summary
└── results/
    ├── reports/                       # Text report per scenario
    └── tables/                        # Hourly energy, server load timeseries, PM schedule
```
