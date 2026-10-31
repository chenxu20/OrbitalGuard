"""Causal snapshots: target extraction is deliberately separate."""
import numpy as np
import pandas as pd
from .data import INPUTS

LATEST = [c for c in INPUTS if c != 'event_id']
TREND = ['miss_distance','max_risk_estimate','t_sigma_r','c_sigma_r']
FEATURE_VERSION = 'prefix-v1'


def build_event_snapshot(event_df, snapshot_time_to_tca, temporal=True):
    prefix = event_df.loc[event_df.time_to_tca >= snapshot_time_to_tca].sort_values('time_to_tca', ascending=False,kind='stable')
    if prefix.empty: raise ValueError('No observations available at cutoff')
    row = prefix.iloc[-1]
    result = {c:row.get(c,np.nan) for c in LATEST}
    result['c_object_type'] = str(row.c_object_type) if pd.notna(row.c_object_type) else 'MISSING'
    if temporal:
        span = float(prefix.time_to_tca.iloc[0]-prefix.time_to_tca.iloc[-1])
        result.update(cdm_count=len(prefix),history_span=span)
        for c in TREND:
            v = prefix[c]
            delta = v.iloc[-1]-v.iloc[0]
            result.update({f'{c}_first':v.iloc[0],f'{c}_delta':delta,f'{c}_rate':delta/span if span>0 else np.nan,f'{c}_min':v.min(),f'{c}_max':v.max(),f'{c}_std':v.std(ddof=0)})
    return result


def event_cohort(df):
    records, exclusions = [], {'ambiguous_chronology':0,'no_early_observation':0,'final_not_within_one_day':0,'invalid_final_target':0}
    for eid,g in df.groupby('event_id',sort=False):
        final = g.sort_values('time_to_tca',ascending=False).iloc[-1]
        if g._ambiguous.any(): exclusions['ambiguous_chronology']+=1; continue
        if final.time_to_tca >= 1: exclusions['final_not_within_one_day']+=1; continue
        if pd.isna(final.get('risk')): exclusions['invalid_final_target']+=1; continue
        early = g[g.time_to_tca >= 2]
        if early.empty: exclusions['no_early_observation']+=1; continue
        records.append({'event_id':eid,'target':float(final.risk),'decision_time':float(early.time_to_tca.min()),'final_time':float(final.time_to_tca),'final_distance':float(final.miss_distance),'early_count':len(early)})
    return pd.DataFrame(records), exclusions
