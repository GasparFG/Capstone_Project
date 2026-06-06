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