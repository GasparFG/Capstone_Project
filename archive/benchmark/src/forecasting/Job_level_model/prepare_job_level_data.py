import json
import numpy as np
import pandas as pd
from config import INPUT_DATA, JOB_LEVEL_DATA, OUTPUTS_DIR, REQUIRED_RAW_COLUMNS

def validate_columns(df):
    missing=[c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing: raise ValueError(f'Missing required columns: {missing}')

def cyc(series, period):
    return np.sin(2*np.pi*series/period), np.cos(2*np.pi*series/period)

def prepare_job_level_data():
    df=pd.read_csv(INPUT_DATA)
    validate_columns(df)
    print(f'Input data used: {INPUT_DATA}')
    print(f'Raw shape: {df.shape}')
    df=df.copy()
    df['scheduled_time']=pd.to_timedelta(df['scheduled_time'])
    df['scheduled_datetime']=pd.Timestamp('2025-01-01') + df['scheduled_time']
    df['scheduled_minutes_from_start']=df['scheduled_time'].dt.total_seconds()/60
    for c in ['role','app_name','job_type']:
        if c in df.columns:
            df[c]=df[c].astype(str).str.lower().str.strip()
            df[c+'_encoded']=df[c].astype('category').cat.codes
    for c in ['cpu_request','memory_request','duration_minutes']:
        df[c]=pd.to_numeric(df[c], errors='coerce').fillna(0).clip(lower=0)
    df=df.sort_values('scheduled_datetime').reset_index(drop=True)
    df['interarrival_minutes']=df['scheduled_datetime'].diff().dt.total_seconds().div(60)
    med=df['interarrival_minutes'].median()
    df['interarrival_minutes']=df['interarrival_minutes'].fillna(1.0 if pd.isna(med) or med<=0 else med).clip(lower=0.01)
    df['arrival_order']=np.arange(1,len(df)+1)
    df['hour']=df['scheduled_datetime'].dt.hour
    df['minute']=df['scheduled_datetime'].dt.minute
    df['day_of_week']=df['scheduled_datetime'].dt.dayofweek
    df['is_weekend']=(df['day_of_week']>=5).astype(int)
    df['time_of_day_minutes']=df['hour']*60+df['minute']
    df['time_sin'], df['time_cos']=cyc(df['time_of_day_minutes'],1440)
    df['day_sin'], df['day_cos']=cyc(df['day_of_week'],7)
    for c in ['interarrival_minutes','cpu_request','memory_request','duration_minutes']:
        df[c+'_log']=np.log1p(df[c])
    feature_blocks=[]
    lag_cols=['interarrival_minutes','cpu_request','memory_request','duration_minutes','job_type_encoded','interarrival_minutes_log','cpu_request_log','memory_request_log','duration_minutes_log']
    for optional in ['role_encoded','app_name_encoded']:
        if optional in df.columns: lag_cols.append(optional)
    for lag in [1,2,3,5,10,20,50]:
        feature_blocks.append(df[lag_cols].shift(lag).add_suffix(f'_lag_{lag}'))
    for c in ['interarrival_minutes','cpu_request','memory_request','duration_minutes']:
        shifted=df[c].shift(1)
        for w in [5,10,20,50]:
            feature_blocks.append(pd.DataFrame({
                f'{c}_rolling_mean_{w}': shifted.rolling(w).mean(),
                f'{c}_rolling_std_{w}': shifted.rolling(w).std(),
                f'{c}_rolling_median_{w}': shifted.rolling(w).median(),
                f'{c}_rolling_min_{w}': shifted.rolling(w).min(),
                f'{c}_rolling_max_{w}': shifted.rolling(w).max(),
            }))
    model=pd.concat([df]+feature_blocks, axis=1).replace([np.inf,-np.inf],np.nan).dropna().reset_index(drop=True).copy()
    JOB_LEVEL_DATA.parent.mkdir(parents=True, exist_ok=True)
    model.to_parquet(JOB_LEVEL_DATA, index=False)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    for name in ['role','app_name','job_type']:
        if name+'_encoded' in model.columns:
            mp=model[[name+'_encoded',name]].drop_duplicates().sort_values(name+'_encoded')
            mp.to_csv(OUTPUTS_DIR/f'{name}_mapping.csv', index=False)
            with open(OUTPUTS_DIR/f'{name}_mapping.json','w',encoding='utf-8') as f:
                json.dump({int(r[name+'_encoded']):r[name] for _,r in mp.iterrows()}, f, indent=4)
    print(f'Job-level forecast dataset saved to: {JOB_LEVEL_DATA}')
    print(f'Model-ready shape: {model.shape}')
    print(model.head())
if __name__=='__main__': prepare_job_level_data()
