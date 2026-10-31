import io
import zipfile
import numpy as np
import pandas as pd
import pytest
from src.data import validate_cdms,load_cdms,fingerprint,INPUTS
from src.features import build_event_snapshot,event_cohort
from src.risk_rules import classify_challenge_tier
from src.replay import get_replay_frame,replay_key
from src.evaluation import compute_heldout_metrics,ranking_at_k
from src.models import split_events,train_models
from src.plots import geometry_status


def fixture():
    rows=[]
    for eid in ['001','b']:
        for t in [1.,5.,3.,.2]:
            row={c:1. for c in INPUTS};row.update(event_id=eid,c_object_type='DEBRIS',time_to_tca=t,miss_distance=6000.,risk=-5. if t==.2 else -10.)
            rows.append(row)
    return validate_cdms(pd.DataFrame(rows))[0]

@pytest.mark.parametrize('p,d,tier',[(-7,6000,'SAFE'),(-6,6000,'ELEVATED'),(-5,6000,'ELEVATED'),(-4,6000,'CRITICAL'),(-8,999,'CRITICAL'),(-8,1000,'ELEVATED'),(-8,5000,'ELEVATED'),(-8,5001,'SAFE'),(None,9000,'REVIEW_REQUIRED'),(None,500,'CRITICAL'),(-4.2,None,'REVIEW_REQUIRED'),(None,None,'UNVERIFIED')])
def test_boundaries(p,d,tier):assert classify_challenge_tier(p,d).tier==tier

def test_missing_evidence():
    assert classify_challenge_tier(-8,9000,True).tier=='REVIEW_REQUIRED'
    d=classify_challenge_tier(-5,9000,True)
    assert d.tier=='ELEVATED' and d.review_required
    assert classify_challenge_tier(None,500).review_required

def test_causal_label_and_future_invariance():
    df=fixture();g=df[df.event_id=='001'];a=build_event_snapshot(g,2)
    g=g.copy();g['risk']=0;g.loc[g.time_to_tca<2,'miss_distance']=99999999
    b=build_event_snapshot(g,2)
    assert a.keys()==b.keys()
    for k in a:assert a[k]==b[k] or pd.isna(a[k]) and pd.isna(b[k])
    assert 'event_id' not in a and 'risk' not in a
    assert a['time_to_tca']==3 and a['cdm_count']==2

def test_actual_final_and_conflict():
    df=fixture();df.loc[(df.event_id=='001')&(df.time_to_tca==.2),'risk']=np.nan
    cohort,_=event_cohort(df)
    assert cohort.event_id.tolist()==['b']
    raw=pd.concat([fixture(),fixture().iloc[[0]].assign(risk=-8)])
    cleaned,audit=validate_cdms(raw)
    assert audit['conflicting_time_events']==['001']
    assert '001' not in event_cohort(cleaned)[0].event_id.tolist()

def test_invalid_time_excludes_event():
    raw=pd.concat([fixture(),fixture().iloc[[0]].assign(time_to_tca=-.1)])
    cleaned,audit=validate_cdms(raw)
    assert audit['invalid_time_rows']==1
    assert '001' not in event_cohort(cleaned)[0].event_id.tolist()

def test_replay_hides_outcome_and_future():
    frame=get_replay_frame(fixture().query('event_id=="001"'),0)
    assert 'outcome' not in frame and 'risk' not in frame['observations']
    assert frame['observations'].time_to_tca.tolist()==[5.]
    g=fixture().query('event_id=="001"')
    assert get_replay_frame(g,0)['observations'].time_to_tca.tolist()==[5.]
    assert get_replay_frame(g,1,True)['outcome']['risk']==-5
    assert replay_key('a','b',0)!=replay_key('a','b',1)

def test_metrics_null_and_rank_ties():
    d=pd.DataFrame({'event_id':['b','a'],'target':[-9,-8],'prediction':[-7,-7]})
    m=compute_heldout_metrics(d)
    assert m['recall'] is None and m['precision'] is None and m['f2'] is None and m['average_precision'] is None
    d['target']=[-9,-5]
    assert ranking_at_k(d,'prediction',1)['true_positives']==1
    assert compute_heldout_metrics(d)['recall']==0
    assert ranking_at_k(d,'prediction',999)['k']==2

def test_ingestion_and_fingerprint(tmp_path):
    df=fixture()[INPUTS+['risk']]
    for ext in ['csv','xlsx','zip']:
        p=tmp_path/f'input.{ext}'
        if ext=='csv':df.to_csv(p,index=False)
        elif ext=='xlsx':df.to_excel(p,index=False)
        else:
            with zipfile.ZipFile(p,'w') as z:z.writestr('../nested.csv',df.to_csv(index=False))
        loaded,_=validate_cdms(load_cdms(p))
        assert '001' in loaded.event_id.tolist()
        assert fingerprint(loaded)==fingerprint(df)
    with zipfile.ZipFile(tmp_path/'bad.zip','w') as z:
        z.writestr('a.csv','a');z.writestr('b.csv','b')
    with pytest.raises(ValueError):load_cdms(tmp_path/'bad.zip')
    with pytest.raises(ValueError):validate_cdms(pd.DataFrame({'foo':[1]}))

def test_geometry():
    row={'relative_position_r':3,'relative_position_t':4,'relative_position_n':0,'miss_distance':6000}
    point,message=geometry_status(row)
    assert point==[3,4,0] and 'DISCREPANCY' in message
    assert geometry_status({})[0] is None

def test_disjoint_split():
    c=pd.DataFrame({'event_id':[str(i) for i in range(100)],'target':[-5 if i%5==0 else -10 for i in range(100)]})
    tr,va,te,_=split_events(c)
    assert len(tr)==70 and len(va)==15 and len(te)==15
    assert not(set(tr.event_id)&set(va.event_id)|set(tr.event_id)&set(te.event_id)|set(va.event_id)&set(te.event_id))


def test_fingerprint_covers_chronology_quality():
    df=fixture();changed=df.copy();changed.loc[0,'_ambiguous']=True
    assert fingerprint(df)!=fingerprint(changed)
