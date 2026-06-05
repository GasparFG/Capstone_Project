from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_DATA = BASE_DIR / 'data' / 'interim' / 'cleaned_data_extended_90_days.csv'
JOB_LEVEL_DATA = BASE_DIR / 'data' / 'processed' / 'job_level_forecast_dataset.parquet'
OPTIMIZATION_INPUT_DATA = BASE_DIR / 'data' / 'processed' / 'optimization_input_dataset.parquet'
MODELS_DIR = BASE_DIR / 'models' / 'forecasting'
OUTPUTS_DIR = BASE_DIR / 'outputs' / 'final_job_level'
RANDOM_STATE = 42
TEST_SIZE = 0.20
FORECAST_HORIZON_MINUTES = 1440
MAX_GENERATED_JOBS = 5000
NUMERIC_TARGETS = ['interarrival_minutes','cpu_request','memory_request','duration_minutes']
DESCRIPTOR_TARGETS = ['role_encoded','app_name_encoded','job_type_encoded']
REQUIRED_RAW_COLUMNS = ['instance_sn','scheduled_time','cpu_request','memory_request','duration_minutes','job_type']
