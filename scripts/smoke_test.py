"""Verify real artifacts, source identity, prediction export and hidden replay."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from src.data import load_cdms,validate_cdms,fingerprint
from src.features import build_event_snapshot
from src.models import predict_event
from src.replay import get_replay_frame

def main():
    base=Path(__file__).resolve().parents[1]
    bundle=joblib.load(base/'artifacts/model.joblib')
    metrics=json.loads((base/'artifacts/metrics.json').read_text())
    manifest=json.loads((base/'artifacts/manifest.json').read_text())
    df,audit=validate_cdms(load_cdms(base/'data/train_data.csv'))
    assert fingerprint(df)==bundle['fingerprint']==metrics['fingerprint']
    splits=[set(manifest['splits'][s]) for s in ['train','validation','test']]
    assert not(splits[0]&splits[1] or splits[0]&splits[2] or splits[1]&splits[2])
    predictions=pd.read_csv(base/'artifacts/test_predictions.csv',dtype={'event_id':str})
    assert predictions.event_id.is_unique and len(predictions)==metrics['test']['n']
    eid=predictions.iloc[0].event_id;g=df[df.event_id==eid];t=g.loc[g.time_to_tca>=2,'time_to_tca'].min()
    p=predict_event(build_event_snapshot(g,t),bundle)
    assert np.isclose(p['predicted_final_log_pc'],predictions.iloc[0].prediction)
    frame=get_replay_frame(g,0)
    assert 'outcome' not in frame and 'risk' not in frame['observations']
    assert not frame['observations'].time_to_tca.lt(2).any()
    assert 'outcome' in get_replay_frame(g,0,True)
    print(f'Real-data smoke passed: {len(df):,} rows; {len(predictions):,} test events; {bundle["selected"]}; prediction/export/replay/identity verified.')


if __name__=='__main__':
    main()
