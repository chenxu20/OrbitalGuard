import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, average_precision_score


def ranking_at_k(df, score, k):
    ranked=df.dropna(subset=[score]).sort_values([score,'event_id'],ascending=[False,True],kind='stable')
    k=min(int(k),len(ranked)); positives=int((ranked.target>=-6).sum())
    tp=int((ranked.head(k).target>=-6).sum())
    return {'k':k,'true_positives':tp,'positive_count':positives,'population':len(ranked),'precision':tp/k if k else None,'recall':tp/positives if positives else None}


def compute_heldout_metrics(df, score='prediction'):
    y=df.target.to_numpy(); p=df[score].to_numpy(); truth=y>=-6; predicted=p>=-6
    tp=int((truth & predicted).sum()); fp=int((~truth & predicted).sum()); fn=int((truth & ~predicted).sum()); tn=int((~truth & ~predicted).sum())
    positives=int(truth.sum()); pp=int(predicted.sum())
    return {'n':len(df),'positive_count':positives,'critical_count':int((y>=-4).sum()),'mae':float(mean_absolute_error(y,p)),'rmse':float(np.sqrt(mean_squared_error(y,p))),'positive_mae':float(np.abs(y[truth]-p[truth]).mean()) if positives else None,'precision':tp/pp if pp else None,'recall':tp/positives if positives else None,'f2':5*tp/(5*tp+4*fn+fp) if positives else None,'average_precision':float(average_precision_score(truth,p)) if positives and positives<len(df) else None,'confusion':{'tp':tp,'fp':fp,'fn':fn,'tn':tn},'undefined_reasons':{'precision':'No predicted positives' if not pp else None,'recall_f2':'No positive outcomes' if not positives else None,'average_precision':'Requires both outcome classes' if not positives or positives==len(df) else None},'ranking':{str(k):ranking_at_k(df,score,k) for k in [10,25,50,100]}}
