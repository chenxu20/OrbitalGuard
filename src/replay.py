import numpy as np
from .data import INPUTS


def get_replay_frame(event_df, snapshot_index, reveal_outcome=False):
    early=event_df[event_df.time_to_tca>=2].sort_values('time_to_tca',ascending=False,kind='stable')
    if early.empty: return {'observations':early[INPUTS].copy(),'snapshot':None}
    index=max(0,min(int(snapshot_index),len(early)-1))
    visible=early.iloc[:index+1][INPUTS].copy()
    result={'observations':visible,'snapshot':visible.iloc[-1].to_dict()}
    if reveal_outcome:
        final=event_df.sort_values('time_to_tca',ascending=False,kind='stable').iloc[-1]
        result['outcome']={c:final.get(c,np.nan) for c in ['risk','time_to_tca','miss_distance']}
    return result


def replay_key(dataset_fingerprint,event_id,index):
    return f'{dataset_fingerprint}:{event_id}:{index}'
