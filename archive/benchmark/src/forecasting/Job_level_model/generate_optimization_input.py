import json, joblib
import numpy as np
import pandas as pd
from config import JOB_LEVEL_DATA, MODELS_DIR, OUTPUTS_DIR, OPTIMIZATION_INPUT_DATA, FORECAST_HORIZON_MINUTES, MAX_GENERATED_JOBS, NUMERIC_TARGETS

def clean(X):
    return X.apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(0)

def load_json(path):
    with open(path,'r',encoding='utf-8') as f: return json.load(f)

def load_mapping(name):
    path=OUTPUTS_DIR/f'{name}_mapping.csv'
    if not path.exists(): return {}
    df=pd.read_csv(path)
    return {int(r[f'{name}_encoded']):r[name] for _,r in df.iterrows()}

def time_features(abs_min):
    tod=abs_min%1440; hour=int(tod//60); minute=int(tod%60); day=int(abs_min//1440); dow=day%7
    return {'hour':hour,'minute':minute,'day_of_week':dow,'is_weekend':int(dow>=5),'time_of_day_minutes':tod,'time_sin':np.sin(2*np.pi*tod/1440),'time_cos':np.cos(2*np.pi*tod/1440),'day_sin':np.sin(2*np.pi*dow/7),'day_cos':np.cos(2*np.pi*dow/7)}

def build_row(history, columns, abs_min, extra=None):
    row=time_features(abs_min)
    lag_cols=['interarrival_minutes','cpu_request','memory_request','duration_minutes','job_type_encoded','interarrival_minutes_log','cpu_request_log','memory_request_log','duration_minutes_log']
    for opt in ['role_encoded','app_name_encoded']:
        if opt in history.columns: lag_cols.append(opt)
    for lag in [1,2,3,5,10,20,50]:
        for col in lag_cols:
            row[f'{col}_lag_{lag}']=history[col].iloc[-lag] if col in history.columns and len(history)>=lag else 0
    for col in ['interarrival_minutes','cpu_request','memory_request','duration_minutes']:
        vals=history[col] if col in history.columns else pd.Series(dtype=float)
        for w in [5,10,20,50]:
            recent=vals.tail(w)
            row[f'{col}_rolling_mean_{w}']=recent.mean() if len(recent) else 0
            row[f'{col}_rolling_std_{w}']=recent.std() if len(recent) else 0
            row[f'{col}_rolling_median_{w}']=recent.median() if len(recent) else 0
            row[f'{col}_rolling_min_{w}']=recent.min() if len(recent) else 0
            row[f'{col}_rolling_max_{w}']=recent.max() if len(recent) else 0
    if extra: row.update(extra)
    X=pd.DataFrame([row])
    for c in columns:
        if c not in X.columns: X[c]=0
    return clean(X[columns])

def pred_reg(model, X):
    return max(0, float(np.expm1(model.predict(X)[0])))

def generate_optimization_input():
    meta=load_json(MODELS_DIR/'job_level_model_metadata.json')
    past_cols=meta['past_feature_columns']; cap_cols=meta['capacity_feature_columns']
    data=pd.read_parquet(JOB_LEVEL_DATA)
    history_cols=['scheduled_minutes_from_start','interarrival_minutes','cpu_request','memory_request','duration_minutes','job_type_encoded','job_type','interarrival_minutes_log','cpu_request_log','memory_request_log','duration_minutes_log']
    for opt in ['role_encoded','app_name_encoded']:
        if opt in data.columns: history_cols.append(opt)
    hist=data[history_cols].copy()
    mappings={name:load_mapping(name) for name in ['role','app_name','job_type']}
    desc_models={}
    for target in meta.get('descriptor_targets',[]):
        path=MODELS_DIR/f'{target}_classifier.joblib'
        if path.exists(): desc_models[target]=joblib.load(path)
    num_models={target:joblib.load(MODELS_DIR/f'{target}_model.joblib') for target in NUMERIC_TARGETS}
    last_min=float(hist['scheduled_minutes_from_start'].iloc[-1]); offset=0.0; jobs=[]
    print('Generating next-day job-level forecast...')
    while offset<FORECAST_HORIZON_MINUTES and len(jobs)<MAX_GENERATED_JOBS:
        abs_min=last_min+offset
        Xpast=build_row(hist,past_cols,abs_min)
        inter=pred_reg(num_models['interarrival_minutes'],Xpast)
        inter=max(inter,0.01); offset+=inter
        if offset>FORECAST_HORIZON_MINUTES: break
        abs_min=last_min+offset
        Xpast=build_row(hist,past_cols,abs_min)
        generated_desc={}
        for target,model in desc_models.items():
            generated_desc[target]=int(model.predict(Xpast)[0])
        # fallback to historical mode for any missing descriptor
        for target in ['role_encoded','app_name_encoded','job_type_encoded']:
            if target in cap_cols and target not in generated_desc:
                generated_desc[target]=int(hist[target].mode().iloc[0]) if target in hist.columns else 0
        Xcap=build_row(hist,cap_cols,abs_min,extra=generated_desc)
        cpu=pred_reg(num_models['cpu_request'],Xcap)
        mem=pred_reg(num_models['memory_request'],Xcap)
        dur=pred_reg(num_models['duration_minutes'],Xcap)
        job_type_encoded=int(generated_desc.get('job_type_encoded',0))
        job_type=mappings.get('job_type',{}).get(job_type_encoded,'unknown')
        row={'predicted_job_id':f'pred_{len(jobs)+1:06d}','arrival_order':len(jobs)+1,'arrival_offset_minutes':offset,'required_cpu':cpu,'required_memory':mem,'expected_duration_minutes':dur,'job_type':job_type,'job_type_encoded':job_type_encoded,'forecasting_approach':'job_level_capacity_forecasting'}
        if 'role_encoded' in generated_desc: row['role']=mappings.get('role',{}).get(int(generated_desc['role_encoded']),'unknown')
        if 'app_name_encoded' in generated_desc: row['app_name']=mappings.get('app_name',{}).get(int(generated_desc['app_name_encoded']),'unknown')
        jobs.append(row)
        new={'scheduled_minutes_from_start':abs_min,'interarrival_minutes':inter,'cpu_request':cpu,'memory_request':mem,'duration_minutes':dur,'job_type_encoded':job_type_encoded,'job_type':job_type,'interarrival_minutes_log':np.log1p(inter),'cpu_request_log':np.log1p(cpu),'memory_request_log':np.log1p(mem),'duration_minutes_log':np.log1p(dur)}
        for d,v in generated_desc.items(): new[d]=v
        hist=pd.concat([hist,pd.DataFrame([new])],ignore_index=True)
    out=pd.DataFrame(jobs)
    OPTIMIZATION_INPUT_DATA.parent.mkdir(parents=True,exist_ok=True); OUTPUTS_DIR.mkdir(parents=True,exist_ok=True)
    out.to_parquet(OPTIMIZATION_INPUT_DATA,index=False)
    out.to_csv(OUTPUTS_DIR/'optimization_input_dataset_preview.csv',index=False)
    print(f'Optimization input saved to: {OPTIMIZATION_INPUT_DATA}')
    print(f'CSV preview saved to: {OUTPUTS_DIR / "optimization_input_dataset_preview.csv"}')
    print(f'Generated jobs: {len(out)}')
    print(out.head())
if __name__=='__main__': generate_optimization_input()
