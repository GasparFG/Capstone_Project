import json, joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from config import JOB_LEVEL_DATA, MODELS_DIR, OUTPUTS_DIR, TEST_SIZE, RANDOM_STATE, NUMERIC_TARGETS, DESCRIPTOR_TARGETS

DROP_ALWAYS=[
 'instance_sn','scheduled_time','scheduled_datetime','scheduled_minutes_from_start','arrival_order',
 'creation_time','deletion_time','gpu_request','role','app_name','job_type',
 'interarrival_minutes','cpu_request','memory_request','duration_minutes',
 'interarrival_minutes_log','cpu_request_log','memory_request_log','duration_minutes_log'
]
CURRENT_DESCRIPTORS=['role_encoded','app_name_encoded','job_type_encoded']

def mape(y,p):
    y=np.asarray(y); p=np.asarray(p); mask=y!=0
    return np.nan if mask.sum()==0 else np.mean(np.abs((y[mask]-p[mask])/y[mask]))*100

def smape(y,p):
    y=np.asarray(y); p=np.asarray(p); den=np.abs(y)+np.abs(p); mask=den!=0
    return np.nan if mask.sum()==0 else np.mean(2*np.abs(p[mask]-y[mask])/den[mask])*100

def reg_metrics(y,p,target,model):
    mse=mean_squared_error(y,p)
    return {'model':model,'target':target,'MSE':mse,'RMSE':np.sqrt(mse),'MAE':mean_absolute_error(y,p),'MAPE':mape(y,p),'SMAPE':smape(y,p),'R2':r2_score(y,p)}

def clean(X):
    return X.apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(0)

def past_feature_columns(df):
    return [c for c in df.columns if c not in DROP_ALWAYS and c not in CURRENT_DESCRIPTORS]

def capacity_feature_columns(df):
    cols=past_feature_columns(df)+[c for c in CURRENT_DESCRIPTORS if c in df.columns]
    return list(dict.fromkeys(cols))

def xgb_reg(target):
    if target=='interarrival_minutes':
        return [
            ('xgb_interarrival_conservative',dict(n_estimators=500,learning_rate=.03,max_depth=3,min_child_weight=5,subsample=.85,colsample_bytree=.85,reg_lambda=5,reg_alpha=.5)),
            ('xgb_interarrival_flexible',dict(n_estimators=700,learning_rate=.025,max_depth=4,min_child_weight=3,subsample=.9,colsample_bytree=.9,reg_lambda=3,reg_alpha=.2)),
        ]
    if target in ['cpu_request','memory_request']:
        return [
            (f'xgb_{target}_balanced',dict(n_estimators=800,learning_rate=.03,max_depth=5,min_child_weight=2,subsample=.9,colsample_bytree=.9,reg_lambda=2.5,reg_alpha=.1)),
            (f'xgb_{target}_regularized',dict(n_estimators=700,learning_rate=.03,max_depth=4,min_child_weight=4,subsample=.85,colsample_bytree=.85,reg_lambda=5,reg_alpha=.5)),
        ]
    return [
        ('xgb_duration_regularized',dict(n_estimators=600,learning_rate=.03,max_depth=3,min_child_weight=6,subsample=.85,colsample_bytree=.85,reg_lambda=8,reg_alpha=1)),
        ('xgb_duration_balanced',dict(n_estimators=800,learning_rate=.025,max_depth=4,min_child_weight=4,subsample=.9,colsample_bytree=.9,reg_lambda=5,reg_alpha=.5)),
    ]

def fit_reg(params):
    return XGBRegressor(objective='reg:squarederror',random_state=RANDOM_STATE,n_jobs=-1,tree_method='hist',**params)

def select_regressor(target,Xtr,Xte,tr,te):
    ytr=np.log1p(tr[target]); yte=te[target].values
    best=None
    for name,params in xgb_reg(target):
        print(f'Training candidate {name} for {target}')
        model=fit_reg(params); model.fit(Xtr,ytr)
        pred=np.expm1(model.predict(Xte)).clip(0)
        met=reg_metrics(yte,pred,target,name)
        result={'name':name,'model':model,'pred':pred,'metrics':met}
        if best is None or met['RMSE']<best['metrics']['RMSE']: best=result
    print(f"Selected {best['name']} for {target} with RMSE={best['metrics']['RMSE']:.4f}, R2={best['metrics']['R2']:.4f}")
    return best

def train_classifier(target,Xtr,Xte,tr,te):
    ytr=tr[target]; yte=te[target]; n=int(ytr.nunique())
    if n<2:
        pred=np.full(len(yte), ytr.iloc[0]); return None, {'model':'constant','target':target,'accuracy':accuracy_score(yte,pred),'precision_macro':precision_score(yte,pred,average='macro',zero_division=0),'recall_macro':recall_score(yte,pred,average='macro',zero_division=0),'f1_macro':f1_score(yte,pred,average='macro',zero_division=0)}, pred
    params=dict(n_estimators=500,learning_rate=.03,max_depth=4,min_child_weight=3,subsample=.9,colsample_bytree=.9,reg_lambda=3,reg_alpha=.2)
    obj='multi:softprob' if n>2 else 'binary:logistic'
    kwargs={'num_class':n} if n>2 else {}
    name=f'xgb_{target}_classifier'
    print(f'Training {name}')
    model=XGBClassifier(objective=obj,random_state=RANDOM_STATE,n_jobs=-1,tree_method='hist',eval_metric='mlogloss' if n>2 else 'logloss',**params,**kwargs)
    model.fit(Xtr,ytr); pred=model.predict(Xte)
    met={'model':name,'target':target,'accuracy':accuracy_score(yte,pred),'precision_macro':precision_score(yte,pred,average='macro',zero_division=0),'recall_macro':recall_score(yte,pred,average='macro',zero_division=0),'f1_macro':f1_score(yte,pred,average='macro',zero_division=0)}
    return model, met, pred

def train_job_level_capacity_model():
    data=pd.read_parquet(JOB_LEVEL_DATA)
    print(f'Training data used: {JOB_LEVEL_DATA}')
    print(f'Shape: {data.shape}')
    split=int(len(data)*(1-TEST_SIZE)); tr=data.iloc[:split].copy(); te=data.iloc[split:].copy()
    past_cols=past_feature_columns(data); cap_cols=capacity_feature_columns(data)
    print(f'Past-only feature columns: {len(past_cols)}')
    print(f'Capacity feature columns: {len(cap_cols)}')
    print('First 20 capacity features:', cap_cols[:20])
    Xtr_past=clean(tr[past_cols]); Xte_past=clean(te[past_cols])
    Xtr_cap=clean(tr[cap_cols]); Xte_cap=clean(te[cap_cols])
    MODELS_DIR.mkdir(parents=True,exist_ok=True); OUTPUTS_DIR.mkdir(parents=True,exist_ok=True)
    # descriptors
    class_rows=[]; desc_pred={}
    for target in [t for t in DESCRIPTOR_TARGETS if t in data.columns]:
        model,met,pred=train_classifier(target,Xtr_past,Xte_past,tr,te)
        class_rows.append(met); desc_pred[target]=pred
        if model is not None: joblib.dump(model, MODELS_DIR/f'{target}_classifier.joblib')
    # capacity regressors: interarrival uses past only; capacity uses current descriptors, which are generated first in the forecast pipeline
    reg_rows=[]; pred_df=te[['arrival_order','scheduled_datetime','scheduled_minutes_from_start','job_type']].copy()
    for target in NUMERIC_TARGETS:
        Xtr=Xtr_past if target=='interarrival_minutes' else Xtr_cap
        Xte=Xte_past if target=='interarrival_minutes' else Xte_cap
        best=select_regressor(target,Xtr,Xte,tr,te)
        reg_rows.append(best['metrics'])
        pred_df[f'{target}_actual']=te[target].values; pred_df[f'{target}_predicted']=best['pred']; pred_df[f'{target}_selected_model']=best['name']
        joblib.dump(best['model'], MODELS_DIR/f'{target}_model.joblib')
    reg_df=pd.DataFrame(reg_rows); class_df=pd.DataFrame(class_rows)
    reg_df.to_csv(OUTPUTS_DIR/'job_level_regression_metrics.csv',index=False)
    class_df.to_csv(OUTPUTS_DIR/'job_level_classification_metrics.csv',index=False)
    pred_df.to_csv(OUTPUTS_DIR/'job_level_test_predictions.csv',index=False)
    pd.DataFrame({'feature_column':past_cols}).to_csv(OUTPUTS_DIR/'past_feature_columns_used.csv',index=False)
    pd.DataFrame({'feature_column':cap_cols}).to_csv(OUTPUTS_DIR/'capacity_feature_columns_used.csv',index=False)
    # Save descriptor mappings (needed by generate_optimization_input.py to decode encoded labels back to strings)
    descriptor_label_cols = [c.replace('_encoded', '') for c in CURRENT_DESCRIPTORS if c in data.columns]
    for col in descriptor_label_cols:
        enc_col = f'{col}_encoded'
        if enc_col in data.columns and col in data.columns:
            mapping = (
                data[[enc_col, col]].drop_duplicates()
                .sort_values(enc_col).reset_index(drop=True)
            )
            mapping.to_csv(OUTPUTS_DIR / f'{col}_mapping.csv', index=False)
    descriptor_distribution = (
        data[[c for c in ['role', 'app_name', 'job_type'] if c in data.columns]]
        .groupby([c for c in ['role', 'app_name', 'job_type'] if c in data.columns], dropna=False)
        .size().reset_index(name='count')
    )
    descriptor_distribution['probability'] = descriptor_distribution['count'] / descriptor_distribution['count'].sum()
    descriptor_distribution.to_csv(MODELS_DIR / 'descriptor_distribution.csv', index=False)
    metadata={'past_feature_columns':past_cols,'capacity_feature_columns':cap_cols,'numeric_targets':NUMERIC_TARGETS,'descriptor_targets':[t for t in DESCRIPTOR_TARGETS if t in data.columns], 'split_index':split, 'note':'Capacity models use generated job descriptors role/app/job_type. They do not use gpu_request, creation_time, deletion_time, current targets, or current log-transformed targets.'}
    with open(MODELS_DIR/'job_level_model_metadata.json','w',encoding='utf-8') as f: json.dump(metadata,f,indent=4)
    print('Regression metrics:'); print(reg_df)
    print('Classification metrics:'); print(class_df)
    print(f'Mapping CSVs saved to: {OUTPUTS_DIR}')
if __name__=='__main__': train_job_level_capacity_model()
