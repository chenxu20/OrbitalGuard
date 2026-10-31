"""Event-disjoint model selection and reproducible trusted local artifacts."""
import hashlib
import json
import platform
from pathlib import Path
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from .data import fingerprint, SCHEMA_VERSION
from .features import build_event_snapshot,event_cohort,FEATURE_VERSION
from .evaluation import compute_heldout_metrics,ranking_at_k

MODEL_VERSION='orbitalguard-1.0'


def make_model(columns):
    prep=ColumnTransformer([('type',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1),['c_object_type'])],remainder='passthrough',verbose_feature_names_out=False)
    return make_pipeline(prep,HistGradientBoostingRegressor(max_iter=160,max_leaf_nodes=15,learning_rate=.08,l2_regularization=2,min_samples_leaf=25,early_stopping=False,random_state=42))


def split_events(cohort,seed=42):
    stratified=True
    try:
        train,rest=train_test_split(cohort,test_size=.3,random_state=seed,stratify=cohort.target>=-6)
        val,test=train_test_split(rest,test_size=.5,random_state=seed,stratify=rest.target>=-6)
    except ValueError:
        stratified=False
        train,rest=train_test_split(cohort,test_size=.3,random_state=seed)
        val,test=train_test_split(rest,test_size=.5,random_state=seed)
    ids=[set(x.event_id) for x in [train,val,test]]
    assert not(ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])
    return train,val,test,stratified


def build_tables(df,cohort,augment=False):
    groups={eid:g for eid,g in df.groupby('event_id',sort=False)}
    features=[]; targets=[]; weights=[]; meta=[]
    for r in cohort.itertuples(index=False):
        g=groups[r.event_id]
        times=g.loc[g.time_to_tca>=2,'time_to_tca'].sort_values(ascending=False).to_numpy()
        chosen=np.unique([0,len(times)//2,len(times)-1]) if augment else [len(times)-1]
        for idx in chosen:
            t=float(times[idx]); features.append(build_event_snapshot(g,t));targets.append(r.target);weights.append(1/len(chosen))
            row=g.loc[g.time_to_tca==t].iloc[-1]
            meta.append({'event_id':r.event_id,'target':r.target,'decision_time':t,'baseline_a':row.max_risk_estimate,'miss_distance':row.miss_distance})
    return pd.DataFrame(features),np.array(targets),np.array(weights),pd.DataFrame(meta)


def train_models(df,output='artifacts',seed=42,audit=None,raw_sha256=None):
    from .features import LATEST
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    cohort,exclusions=event_cohort(df)
    if len(cohort)<20: raise ValueError('At least 20 eligible events required; sample is insufficient for credible evaluation.')
    tr,va,te,stratified=split_events(cohort,seed)
    print(f'Eligible events: {len(cohort)}; exclusions: {exclusions}',flush=True)
    tables={}
    for name,c in [('train',tr),('validation',va),('test',te)]:
        print(f'{name}: {len(c)} events; {(c.target>=-6).sum()} elevated; {(c.target>=-4).sum()} critical',flush=True)
        tables[name]=build_tables(df,c,augment=name=='train')
    models={};val_metrics={};columns={}
    for name,cols in [('baseline_b',LATEST),('temporal_c',list(tables['train'][0].columns))]:
        columns[name]=cols;model=make_model(cols)
        X,y,w,_=tables['train'];model.fit(X[cols],y,histgradientboostingregressor__sample_weight=w)
        models[name]=model
        xv,yv,_,mv=tables['validation'];pred=model.predict(xv[cols]);m=mv.copy();m['prediction']=np.minimum(pred,0)
        val_metrics[name]=compute_heldout_metrics(m)
        print(f'{name} validation MAE: {val_metrics[name]["mae"]:.5f}',flush=True)
    selected='temporal_c' if val_metrics['temporal_c']['mae']<val_metrics['baseline_b']['mae'] else 'baseline_b'
    data_id=fingerprint(df)
    bundle={'models':models,'columns':columns,'selected':selected,'model_version':MODEL_VERSION,'feature_version':FEATURE_VERSION,'schema_version':SCHEMA_VERSION,'fingerprint':data_id,'seed':seed,'prediction_mode':'early prefix >=2 days','package_versions':{p:importlib.metadata.version(p) for p in ['pandas','numpy','scikit-learn','joblib']}}
    metrics={'model_version':MODEL_VERSION,'selected_model':selected,'selection':'Lowest validation MAE; ties prefer baseline B. Models fitted on training events only.','fingerprint':data_id,'raw_sha256':raw_sha256,'seed':seed,'stratified':stratified,'audit':audit,'exclusions':exclusions,'eligible_events':len(cohort),'validation_candidates':val_metrics,'risk_floor_count':int((df.get('risk',pd.Series(dtype=float))==-30).sum()),'notes':['Risk -30 floor retained; no invented lower probability floor.','Baseline A max_risk_estimate is a ranking signal only.','No calibrated uncertainty; positive predictions clipped at 0.']}
    manifest={'fingerprint':data_id,'splits':{},'exclusions':exclusions}
    for split,c in [('train',tr),('validation',va),('test',te)]:
        manifest['splits'][split]=c.event_id.tolist()
        if split=='train':continue
        X,y,w,m=tables[split]
        for name,model in models.items():m[name]=np.minimum(model.predict(X[columns[name]]),0)
        raw=models[selected].predict(X[columns[selected]])
        m['raw_prediction']=raw;m['prediction']=np.minimum(raw,0)
        assert m.event_id.is_unique
        metrics[split]=compute_heldout_metrics(m)
        common=m.dropna(subset=['baseline_a'])
        metrics[split]['ranking_comparison']={'n':len(common),'excluded_missing_baseline':len(m)-len(common),'ai':{str(k):ranking_at_k(common,'prediction',k) for k in [10,25,50,100]},'baseline_a':{str(k):ranking_at_k(common,'baseline_a',k) for k in [10,25,50,100]}}
        m.to_csv(output/f'{split}_predictions.csv',index=False)
        if split=='validation':
            fn=m[(m.target>=-6)&(m.prediction<-6)]
            chosen=fn if len(fn) else m.assign(error=(m.target-m.prediction).abs()).sort_values('error',ascending=False)
            metrics['failure_case']={'event_id':str(chosen.iloc[0].event_id),'kind':'false negative' if len(fn) else 'largest absolute error'}
    joblib.dump(bundle,output/'model.joblib')
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
    cohort.to_csv(output/'cohort.csv',index=False)
    print(f'Selected {selected}; test MAE {metrics["test"]["mae"]:.5f}; artifacts: {output}',flush=True)
    return bundle


def predict_event(snapshot,bundle,model_name=None):
    name=model_name or bundle['selected']
    X=pd.DataFrame([snapshot])[bundle['columns'][name]]
    raw=float(bundle['models'][name].predict(X)[0]);pred=min(raw,0.)
    return {'model_version':bundle['model_version'],'predicted_final_log_pc':pred,'raw_prediction':raw,'estimated_final_pc':10.**pred,'status':'Statistical forecast; uncertainty uncalibrated'}
