"""Reproduce the full event-level training and assessment run."""
import argparse
import hashlib
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data import load_cdms,validate_cdms
from src.models import train_models

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Train on full CDMs; small sample spreadsheets cannot establish robust model performance.')
    parser.add_argument('--data',required=True);parser.add_argument('--output',default='artifacts');parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    digest=hashlib.sha256()
    with open(args.data,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    df,audit=validate_cdms(load_cdms(args.data))
    print(f'Loaded {len(df)} clean rows / {df.event_id.nunique()} events',flush=True)
    train_models(df,args.output,args.seed,audit,digest.hexdigest())
