import pandas as pd
from config import OUTPUTS_DIR

def evaluate_job_level_model():
    reg=pd.read_csv(OUTPUTS_DIR/'job_level_regression_metrics.csv')
    cls=pd.read_csv(OUTPUTS_DIR/'job_level_classification_metrics.csv')
    rows=[]
    for _,r in reg.iterrows():
        rows.append({'target':r.target,'metric_type':'regression','model':r.model,'RMSE':r.RMSE,'MAE':r.MAE,'MAPE':r.MAPE,'SMAPE':r.SMAPE,'R2':r.R2,'role':'Primary optimization variable' if r.target in ['cpu_request','memory_request'] else 'Timing/duration variable'})
    for _,r in cls.iterrows():
        rows.append({'target':r.target,'metric_type':'classification','model':r.model,'accuracy':r.accuracy,'precision_macro':r.precision_macro,'recall_macro':r.recall_macro,'f1_macro':r.f1_macro,'role':'Generated job descriptor'})
    out=pd.DataFrame(rows)
    out.to_csv(OUTPUTS_DIR/'job_level_model_summary.csv',index=False)
    print(f'Job-level model summary saved to: {OUTPUTS_DIR / "job_level_model_summary.csv"}')
    print(out)
if __name__=='__main__': evaluate_job_level_model()
