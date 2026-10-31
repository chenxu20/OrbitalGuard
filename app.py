"""OrbitalGuard: evidence-first conjunction operations prototype."""
from pathlib import Path
import json
import hashlib
import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.data import load_cdms,validate_cdms,fingerprint,INPUTS,SCHEMA_VERSION
from src.features import build_event_snapshot,FEATURE_VERSION
from src.models import predict_event,MODEL_VERSION
from src.risk_rules import classify_challenge_tier
from src.replay import get_replay_frame,replay_key
from src.evaluation import ranking_at_k
from src.plots import geometry_plot

st.set_page_config(page_title='OrbitalGuard · Mission Control',page_icon='◈',layout='wide')
st.markdown('''<style>.block-container{padding-top:2rem;max-width:1600px}h1{letter-spacing:-1.8px}div[data-testid="stMetric"]{background:#101d31;border:1px solid #22364c;padding:18px;border-radius:12px}div[data-testid="stMetricLabel"]{color:#9bb4c7} .eyebrow{color:#36d9cf;letter-spacing:4px;font-size:12px;font-weight:700}.hero{font-size:19px;color:#9bb4c7;margin-bottom:26px}.pill{background:#102a35;color:#5eead4;padding:7px 12px;border-radius:20px;font-size:12px}button{border-radius:8px!important}</style>''',unsafe_allow_html=True)
st.markdown('<div class="eyebrow">CONJUNCTION INTELLIGENCE / HISTORICAL OPERATIONS</div>',unsafe_allow_html=True)
st.title('OrbitalGuard')
st.markdown('<div class="hero">See tomorrow’s collision risks, today.</div>',unsafe_allow_html=True)
BASE=Path(__file__).parent

@st.cache_data(show_spinner='Reading and validating CDMs…')
def read_path(path,source_digest):
    return validate_cdms(load_cdms(path))

@st.cache_data(show_spinner='Validating upload…')
def read_upload(content,name):
    import io
    f=io.BytesIO(content);f.name=name
    return validate_cdms(load_cdms(f))

@st.cache_resource
def read_bundle(path,mtime):
    b=joblib.load(path)
    if b.get('model_version')!=MODEL_VERSION or b.get('feature_version')!=FEATURE_VERSION or b.get('schema_version')!=SCHEMA_VERSION:raise ValueError('Incompatible model/feature version; retrain local artifacts.')
    return b

def identity(df):return fingerprint(df)

@st.cache_data(show_spinner='Constructing early decision snapshots…')
def triage(df,dataset_identity,model_identity,_bundle):
    features=[];rows=[]
    for eid,g in df.groupby('event_id',sort=False):
        early=g[g.time_to_tca>=2]
        row=(early.iloc[-1] if len(early) else g.iloc[-1])
        eligible=not early.empty and not g._ambiguous.any()
        features.append(build_event_snapshot(g,row.time_to_tca) if eligible else None)
        rows.append({'event_id':eid,'lead_days':row.time_to_tca,'distance_m':row.miss_distance,'max_risk_estimate':row.max_risk_estimate,'incomplete':bool(row._quality_missing or g._ambiguous.any()),'supported_early':eligible,'latest_time':g.time_to_tca.min()})
    result=pd.DataFrame(rows);result['prediction']=np.nan
    if _bundle:
        indices=[i for i,f in enumerate(features) if f is not None]
        X=pd.DataFrame([features[i] for i in indices]);name=_bundle['selected']
        if len(X):result.loc[indices,'prediction']=np.minimum(_bundle['models'][name].predict(X[_bundle['columns'][name]]),0)
    decisions=[classify_challenge_tier(r.prediction,r.distance_m,r.incomplete) for r in result.itertuples()]
    result['tier']=[d.tier for d in decisions];result['review_required']=[d.review_required for d in decisions]
    return result

with st.sidebar:
    st.markdown('### Flight desk')
    st.caption('Prototype · human review only')
    source_mode=st.radio('CDM source',['Local file','Upload'])
    path=st.text_input('Local CSV / XLSX / ZIP',value=os.environ.get('ORBITALGUARD_DATA',str(BASE/'data/train_data.csv'))) if source_mode=='Local file' else None
    upload=st.file_uploader('Upload CDMs',type=['csv','xlsx','zip']) if source_mode=='Upload' else None
    st.divider()
    st.caption('Forecasts use observations ≥2 days before TCA. Times are relative to each encounter; this is historical data, not live tracking.')
    st.caption('Operational thresholds: CRITICAL at predicted logPc ≥−4 or distance <1 km; ELEVATED at logPc ≥−6 or distance ≤5 km. Incomplete evidence requires review.')
try:
    if source_mode=='Upload':
        if upload is None:
            st.info('Upload real conjunction messages to explore encounters. Add a trained local artifact to enable forecasts.');st.stop()
        df,audit=read_upload(upload.getvalue(),upload.name);source_name=upload.name
    else:
        if not path or not Path(path).is_file():
            st.info('Choose a local CDM file or upload CSV, XLSX or a ZIP containing one CSV. No synthetic encounters are generated.');st.stop()
        p=Path(path)
        digest=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
        df,audit=read_path(str(p),digest.hexdigest());source_name=p.name
    if df.empty:st.warning('No valid observations remain after validation.');st.stop()
except Exception as exc:
    st.error(f'Cannot load CDMs: {exc}');st.stop()

bundle=None;metrics=None;manifest=None
try:
    artifact=BASE/'artifacts/model.joblib'
    if artifact.is_file():
        bundle=read_bundle(str(artifact),artifact.stat().st_mtime_ns)
        metrics=json.loads((BASE/'artifacts/metrics.json').read_text());manifest=json.loads((BASE/'artifacts/manifest.json').read_text())
        if not (bundle['fingerprint']==metrics['fingerprint']==manifest['fingerprint']):raise ValueError('Artifact companion identities differ; retrain.')
except Exception as exc:
    bundle=metrics=manifest=None
    st.warning(f'Forecast unavailable: {exc}')
dataset_id=identity(df);matches=bool(bundle and dataset_id==bundle['fingerprint'])
st.markdown(f'<span class="pill">{source_name} · {len(df):,} CDMs · {df.event_id.nunique():,} encounters</span>',unsafe_allow_html=True)
st.caption(f'Model: {bundle["selected"]+" / "+bundle["model_version"] if bundle else "unavailable — exploration and distance screening active"} · '+('Matching training-run source; held-out membership verified.' if matches else 'External / not evaluated on this source.'))
summary=triage(df,dataset_id,str(artifact.stat().st_mtime_ns) if bundle else 'none',bundle)
if metrics and metrics['test']['recall']==0:
    st.warning(f"Prototype limitation: the recorded test model detected 0 of {metrics['test']['positive_count']} elevated outcomes at logPc ≥−6. Do not rely on this forecast to clear encounters; inspect telemetry and proximity rules.")
ops,replay,lab=st.tabs(['◈  Conjunction Ops','↶  Risk Time Machine','▥  Evaluation & Ops Load'])

def render_geometry(row):
    fig,message=geometry_plot(row)
    if fig is not None:st.plotly_chart(fig)
    st.caption(message)
    st.caption('Target origin + supplied relative encounter point. Equal metric axis scale; not a physical orbit or spacecraft trajectory.')

def evidence(row,prediction,incomplete):
    decision=classify_challenge_tier(prediction,row.get('miss_distance'),incomplete)
    st.subheader(decision.tier.replace('_',' '))
    for reason in decision.triggered_rules:st.write('• '+reason)
    a,b=st.columns(2)
    a.metric('Decision lead time',f'{row.time_to_tca:.3f} days')
    b.metric('Reported miss distance',f'{row.miss_distance:,.0f} m' if pd.notna(row.miss_distance) else 'Missing')
    if prediction is not None and np.isfinite(prediction):
        a.metric('Predicted final log₁₀(Pc)',f'{prediction:.3f}')
        b.metric('Statistical predicted Pc',f'{10.**prediction:.2e}')
        st.caption('Estimated final probability from regression; not an independent covariance-based physical Pc calculation. Uncertainty uncalibrated.')
    else:st.info('AI forecast unavailable for this observation.')
    st.caption(f'Existing max-risk estimate (ranking signal only): {row.get("max_risk_estimate",np.nan):.4g}')

with ops:
    st.markdown('### The analyst’s early-warning queue')
    a,b,c,d=st.columns(4)
    a.metric('Encounters inspected',f'{len(summary):,}')
    b.metric('Early proximity <1 km',int(((summary.distance_m<1000)&summary.supported_early).sum()))
    c.metric('Predicted logPc ≥−6',int((summary.prediction>=-6).sum()))
    d.metric('Final observation <24 h',int((summary.latest_time<1).sum()))
    st.caption('The <24 h count uses each event’s final available observation; these are not simultaneous alerts. Queue rules and forecasts share the early decision CDM; unsupported events use explicitly historical distance screening.')
    search=st.text_input('Find event ID',placeholder='Search encounters…')
    tiers=st.multiselect('Screening tiers',sorted(summary.tier.unique()),default=[])
    shown=summary[summary.event_id.str.contains(search,case=False,regex=False)]
    if tiers:shown=shown[shown.tier.isin(tiers)]
    shown=shown.sort_values(['prediction','event_id'],ascending=[False,True],na_position='last')
    left,right=st.columns([1.4,1])
    with left:
        st.dataframe(shown[['event_id','tier','prediction','distance_m','lead_days','review_required','supported_early']],hide_index=True,width='stretch',height=350)
        st.download_button('Export current early decision queue',shown.to_csv(index=False).encode(),'orbitalguard_predictions.csv','text/csv')
        with st.expander('Input quality audit'):st.json(audit)
    with right:
        if len(shown):
            eid=st.selectbox('Inspect encounter',shown.event_id.tolist())
            g=df[df.event_id==eid];s=shown[shown.event_id==eid].iloc[0];row=g[g.time_to_tca==s.lead_days].iloc[-1]
            if not s.supported_early:st.warning('No supported early forecast: no observation at least 2 days before TCA, or ambiguous chronology. Historical distance screening only.')
            evidence(row,s.prediction,bool(s.incomplete));render_geometry(row)
            with st.expander('Exact decision CDM — allowed inputs'):st.dataframe(row[INPUTS].to_frame('value').astype(str))

with replay:
    st.markdown('### Risk Time Machine')
    st.write('Freeze the evidence. Make the forecast. Then reveal the recorded outcome.')
    early_ids=summary.loc[summary.supported_early,'event_id'].tolist()
    if matches:
        allowed=set(manifest['splits']['validation']);early_ids=[e for e in early_ids if e in allowed]
        st.caption('Held-out validation replay · this cohort selected the model. Final assessment is on separate test events.')
    else:st.caption('External / not evaluated replay. Held-out status cannot be established for this source.')
    if not early_ids:st.info('No supported observations at least 2 days before TCA.')
    else:
        failure=metrics['failure_case']['event_id'] if matches else None
        default=early_ids.index(failure) if failure in early_ids else 0
        eid=st.selectbox('Replay encounter',early_ids,index=default,key='replay_event')
        g=df[df.event_id==eid];n=int((g.time_to_tca>=2).sum())
        index=st.slider('Observation — earliest → latest eligible',0,max(1,n-1),0,disabled=n==1)
        index=min(index,n-1)
        token=replay_key(dataset_id,eid,index)
        if st.session_state.get('replay_token')!=token:
            st.session_state.replay_token=token;st.session_state.reveal=False
        st.button('Reveal outcome',on_click=lambda:st.session_state.update(reveal=True),disabled='risk' not in df)
        frame=get_replay_frame(g,index,st.session_state.get('reveal',False));visible=frame['observations'];row=visible.iloc[-1]
        snap=build_event_snapshot(visible,row.time_to_tca)
        pred=predict_event(snap,bundle)['predicted_final_log_pc'] if bundle else None
        l,r=st.columns([1,1.2])
        with l:
            evidence(row,pred,bool(row[INPUTS].isna().any()))
            if 'outcome' in frame:
                o=frame['outcome'];st.success('Recorded outcome revealed')
                st.write(f'Final log₁₀(Pc): {o["risk"]:.4f} · final time: {o["time_to_tca"]:.4f} days · final distance: {o["miss_distance"]:,.0f} m')
                if pred is not None and np.isfinite(o['risk']):
                    bp=predict_event(snap,bundle,'baseline_b')['predicted_final_log_pc']
                    st.write(f'Absolute error: selected model {abs(pred-o["risk"]):.3f} · learned baseline B {abs(bp-o["risk"]):.3f} log units')
                if o['time_to_tca']>=1:st.warning('Final CDM is at least one day before TCA; this is outside the training target cohort.')
                if matches and eid==failure:st.warning(f'Recorded validation {metrics["failure_case"]["kind"]}. This case demonstrates a limitation, not a causal explanation.')
            else:st.info('Final label and future CDMs are concealed.')
        with r:render_geometry(row)
        fig=go.Figure(go.Scatter(x=visible.time_to_tca,y=visible.miss_distance,mode='lines+markers',line=dict(color='#36d9cf'),name='Available miss distance'))
        fig.update_layout(template='plotly_dark',height=250,xaxis=dict(title='Days before TCA',autorange='reversed'),yaxis_title='Miss distance (m)',margin=dict(t=20,b=30))
        st.plotly_chart(fig)
        st.caption('Only visible prefix observations set chart bounds. Replay forecasts at earlier prefixes are historical inference; benchmark metrics apply only to the latest eligible observation.')
        if len(visible)>1:
            with st.expander('What changed? — available input evidence'):
                fields=['miss_distance','max_risk_estimate','t_sigma_r','c_sigma_r']
                previous=visible.iloc[-2][fields].astype(float)
                current=visible.iloc[-1][fields].astype(float)
                changes=pd.DataFrame({'previous':previous,'current':current,'change':current-previous})
                st.dataframe(changes,width='stretch')
                st.caption('Changes in available telemetry, not causal model attributions. Distance and sigmas in metres; max-risk estimate remains a ranking signal.')
        with st.expander('Available source CDMs — no target labels'):st.dataframe(visible,hide_index=True,width='stretch')

with lab:
    st.markdown('### Evidence before confidence')
    if not metrics:st.info('Train local artifacts to enable recorded evaluation: python scripts/train.py --data data/train_data.csv --output artifacts/')
    else:
        st.caption('Recorded training-run cohort only. These metrics do not evaluate newly uploaded files. Event-disjoint 70/15/15 split, seed 42; augmented prefixes only in training. Validation selects the model; test remains assessment only.')
        split=st.radio('Recorded assessment cohort',['test','validation'],horizontal=True)
        m=metrics[split];cols=st.columns(5)
        for col,label,value in zip(cols,['Events','Final logPc ≥−6','Final logPc ≥−4','MAE (log units)','Recall at −6'],[m['n'],m['positive_count'],m['critical_count'],f'{m["mae"]:.3f}',f'{m["recall"]:.1%}' if m['recall'] is not None else 'N/A']):col.metric(label,value)
        st.write(f'Selected model: **{metrics["selected_model"]}**. Selection used validation MAE; uncertainty and rare-event recall remain limitations.')
        st.dataframe(pd.DataFrame(metrics['validation_candidates']).T[['n','positive_count','mae','rmse','precision','recall','f2','average_precision']],width='stretch')
        st.caption('Average precision is the precision-recall ranking summary, requiring both classes. Threshold decisions here use predicted logPc ≥−6, separately from operational distance rules. Undefined values are null, with reasons in the full report.')
        predfile=BASE/f'artifacts/{split}_predictions.csv'
        predictions=pd.read_csv(predfile,dtype={'event_id':str})
        common=predictions.dropna(subset=['baseline_a']);budget=st.slider('How many encounters can our analysts review?',1,max(1,len(common)),min(50,max(1,len(common))))
        results=[{'ranking':label,**ranking_at_k(common,score,budget)} for label,score in [('Selected AI','prediction'),('Existing max-risk signal A','baseline_a')]]
        st.dataframe(pd.DataFrame(results),hide_index=True,width='stretch')
        st.caption(f'Identical comparison subset: {len(common):,} encounters; {len(predictions)-len(common)} excluded for missing baseline signal. Precision = caught/K; recall = caught/positive outcomes. Baseline A is not a physical Pc or regression comparator.')
        ks=sorted(set([1,10,25,50,100,budget,len(common)]));fig=go.Figure()
        for label,score in [('Selected AI','prediction'),('Existing signal A','baseline_a')]:
            points=[ranking_at_k(common,score,k) for k in ks]
            fig.add_trace(go.Scatter(x=[p['k'] for p in points],y=[p['recall'] for p in points],name=label,mode='lines+markers'))
        fig.update_layout(template='plotly_dark',height=300,xaxis_title='Encounter review capacity',yaxis_title='Recall of final logPc ≥−6',yaxis_tickformat='.0%')
        st.plotly_chart(fig)
        st.write(f'Validation failure case: **{metrics["failure_case"]["event_id"]}** ({metrics["failure_case"]["kind"]}); open Risk Time Machine on the matching source to inspect it.')
        with st.expander('Full recorded metrics, exclusions and lineage'):st.json(metrics)
        st.download_button('Download recorded metrics',json.dumps(metrics,indent=2),'metrics.json','application/json')
    st.divider()
    st.write('**Limits & accountability.** Simplified telemetry omits full covariance and hard-body radius. The observed −30 target floor is retained. Predictions can miss elevated outcomes. Relative geometry is an encounter estimate, not propagated orbital motion. Qualified operators must review evidence; OrbitalGuard issues no flight commands.')
