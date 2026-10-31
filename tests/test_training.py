"""Small, explicitly synthetic end-to-end pipeline tests; never shown as evidence."""
import json
import joblib
import numpy as np
import pandas as pd
from src.data import INPUTS,validate_cdms,fingerprint
from src.models import train_models,predict_event
from src.features import build_event_snapshot


def test_synthetic_training_persistence_invariance_and_unknowns(tmp_path):
    rows=[]
    for i in range(40):
        for time in [6.,3.,.2]:
            row={c:float(i+1) for c in INPUTS}
            row.update(event_id=f'synthetic-{i:03}',c_object_type='DEBRIS',time_to_tca=time,risk=-5. if i%4==0 else -12.,miss_distance=6000+i)
            rows.append(row)
    df,audit=validate_cdms(pd.DataFrame(rows))
    bundle=train_models(df,tmp_path,audit=audit)
    loaded=joblib.load(tmp_path/'model.joblib')
    g=df[df.event_id=='synthetic-000'].copy()
    a=predict_event(build_event_snapshot(g,3),loaded)
    g['risk']=0;g.loc[g.time_to_tca<3,'miss_distance']=1
    b=predict_event(build_event_snapshot(g,3),loaded)
    assert a==b
    g.loc[:,'c_object_type']='UNKNOWN_NEW_TYPE';g.loc[:,'F10']=np.nan
    assert np.isfinite(predict_event(build_event_snapshot(g,3),loaded)['predicted_final_log_pc'])
    m=json.loads((tmp_path/'metrics.json').read_text())
    assert m['fingerprint']==fingerprint(df)
    df.loc[0,'miss_distance']+=1
    assert m['fingerprint']!=fingerprint(df)
    for name in ['validation','test']:
        p=pd.read_csv(tmp_path/f'{name}_predictions.csv')
        assert p.event_id.is_unique
