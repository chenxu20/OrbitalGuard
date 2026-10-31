"""Programmatic Streamlit integration checks; no visual browser verification."""
from pathlib import Path
import os
import sys
from tempfile import TemporaryDirectory
import pandas as pd
from streamlit.testing.v1 import AppTest

def main():
    base=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(base))
    at=AppTest.from_file(str(base/'app.py'),default_timeout=180).run()
    assert not len(at.exception),[e.message for e in at.exception]
    # Choose an event with >1 supported early observation so a time change is real.
    pdct=pd.read_csv(base/'artifacts/validation_predictions.csv',dtype={'event_id':str})
    cohort=pd.read_csv(base/'artifacts/cohort.csv',dtype={'event_id':str})
    eid=cohort[cohort.event_id.isin(pdct.event_id)&(cohort.early_count>1)].iloc[0].event_id
    at.selectbox[1].select(eid).run()
    at.button[0].click().run()
    assert not len(at.exception) and at.session_state['reveal']
    at.slider[0].set_value(1).run()
    assert not len(at.exception) and not at.session_state['reveal']
    next(r for r in at.radio if r.label=='Recorded assessment cohort').set_value('validation').run()
    assert not len(at.exception)
    at.sidebar.text_input[0].set_value('/tmp/orbitalguard-does-not-exist.csv').run()
    assert not len(at.exception) and len(at.info)
    with TemporaryDirectory() as directory:
        bad=Path(directory)/'bad.csv';bad.write_text('bad,headers\n1,2\n')
        at.sidebar.text_input[0].set_value(str(bad)).run()
        assert not len(at.exception) and any('Cannot load' in e.value for e in at.error)
    print('Streamlit AppTest passed: real data, replay/reveal/reset, cohort switch, empty path, malformed source.')


if __name__=='__main__':
    main()
