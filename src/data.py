"""Restricted CDM ingestion, conservative cleaning and content identity."""
from pathlib import Path
import hashlib
import io
import zipfile
import numpy as np
import pandas as pd

INPUTS = ['event_id','time_to_tca','miss_distance','relative_speed','relative_position_r','relative_position_t','relative_position_n','t_sigma_r','c_sigma_r','c_object_type','max_risk_estimate','F10','AP']
SCHEMA_VERSION = 'cdm-v1'


def load_cdms(source):
    name = str(getattr(source, 'name', source)).lower()
    if name.endswith('.zip'):
        with zipfile.ZipFile(source) as z:
            members = [m for m in z.infolist() if m.filename.lower().endswith('.csv')]
            if len(members) != 1:
                raise ValueError('ZIP must contain exactly one CSV; select and upload one member to avoid event-ID collisions.')
            if members[0].file_size > 600_000_000 or len(z.infolist()) > 100:
                raise ValueError('ZIP expanded size/count exceeds safety limits.')
            with z.open(members[0]) as f:
                return pd.read_csv(f, dtype={'event_id':str}, usecols=lambda c: c in INPUTS + ['risk'])
    if name.endswith('.xlsx'):
        return pd.read_excel(source, dtype={'event_id':str}, usecols=lambda c: c in INPUTS + ['risk'])
    if not name.endswith('.csv'):
        raise ValueError('Use CSV, XLSX or a ZIP containing one CSV.')
    return pd.read_csv(source, dtype={'event_id':str}, usecols=lambda c: c in INPUTS + ['risk'])


def validate_cdms(raw):
    required = {'event_id','time_to_tca','miss_distance'}
    if not required.issubset(raw.columns):
        raise ValueError('Required columns: event_id, time_to_tca, miss_distance')
    df = raw.loc[:, [c for c in INPUTS + ['risk'] if c in raw]].copy()
    audit = {'input_rows':len(df), 'missing_optional_columns':[c for c in INPUTS if c not in df]}
    for c in INPUTS:
        if c not in df: df[c] = np.nan
    missing_id = df.event_id.isna() | df.event_id.astype(str).str.strip().eq('')
    audit['missing_event_id_rows'] = int(missing_id.sum())
    df = df.loc[~missing_id].copy()
    df['event_id'] = df.event_id.astype(str).str.strip()
    for c in [c for c in INPUTS + ['risk'] if c in df and c not in ['event_id','c_object_type']]:
        df[c] = pd.to_numeric(df[c], errors='coerce').replace([np.inf,-np.inf],np.nan)
        if c in ['time_to_tca','miss_distance','relative_speed','t_sigma_r','c_sigma_r','F10','AP']:
            df.loc[df[c] < 0,c] = np.nan
        if c == 'risk': df.loc[df[c] > 0,c] = np.nan
    invalid_time = df.time_to_tca.isna()
    bad_time_ids = df.loc[invalid_time,'event_id'].unique().tolist()
    audit['invalid_time_rows'] = int(invalid_time.sum())
    df = df.loc[~invalid_time].copy()
    before = len(df)
    df = df.drop_duplicates()
    audit['exact_duplicates'] = before-len(df)
    conflict_ids = df.loc[df.duplicated(['event_id','time_to_tca'],keep=False),'event_id'].unique().tolist()
    df['_ambiguous'] = df.event_id.isin(set(bad_time_ids + conflict_ids))
    df['_quality_missing'] = df[[c for c in INPUTS if c not in ['event_id','c_object_type']]].isna().any(axis=1) | df.c_object_type.isna()
    audit['invalid_time_events'] = bad_time_ids
    audit['conflicting_time_events'] = conflict_ids
    audit['missing_evidence_rows'] = int(df._quality_missing.sum())
    audit['clean_rows'] = len(df)
    audit['events'] = int(df.event_id.nunique())
    df = df.sort_values(['event_id','time_to_tca'], ascending=[True,False],kind='stable').reset_index(drop=True)
    return df, audit


def fingerprint(df):
    cols = [c for c in INPUTS + ['risk'] if c in df]
    canonical = df[cols].copy()
    for c in cols:
        if c not in ['event_id','c_object_type']:
            canonical[c] = canonical[c].astype(float)
    canonical = canonical.sort_values(cols, na_position='last').reset_index(drop=True)
    digest = hashlib.sha256(SCHEMA_VERSION.encode())
    digest.update(pd.util.hash_pandas_object(canonical,index=False).values.tobytes())
    # Invalid-time records can leave identical clean values but alter eligibility.
    ambiguous=sorted(df.loc[df['_ambiguous'],'event_id'].unique()) if '_ambiguous' in df else []
    digest.update(repr(ambiguous).encode())
    return digest.hexdigest()
