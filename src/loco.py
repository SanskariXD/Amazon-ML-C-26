"""Strict leave-one-country-out stress test with label-free shared retrieval.

Fit and threshold tuning use source-country labels only. Target-country development
labels are opened only for the final evaluation. The final holdout is never read.
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
import psutil
from .audit import truth_for
from .metric import evaluate
from .runtime import atomic_json
from .train_stage1 import train
from . import features,decide


def run_loco(db,work,cfg):
    from .pipeline import feature_shards,frames
    for part in ('fit','tune','dev'):feature_shards(db,work,cfg,part)
    from .pipeline import artifact_paths
    total_pairs=sum(json.loads(p.read_text()).get('pairs',0) for part in ('fit','tune','dev') for p in (artifact_paths(work)['features']/part).glob('[0-9]*.json'))
    if total_pairs*(len(features.NAMES)*4+180)*6 > psutil.virtual_memory().available*.60:
        raise MemoryError('LOCO sample exceeds RAM budget; reduce entity sample sizes')
    tables={};queries={};country={}
    for part in ('fit','tune','dev'):
        pieces=[];queries[part]=[]
        for _,df,ids in frames(work,part):pieces.append(df);queries[part]+=ids
        tables[part]=pd.concat(pieces,ignore_index=True)
        for q in queries[part]:country[q]=db.execute("SELECT country FROM records WHERE split='train' AND id=?",(q,)).fetchone()[0]
    countries=sorted({country[q] for q in queries['fit']})
    results=[]
    for source in countries:
        for target in countries:
            if source==target:continue
            # Safe filename independent of country spelling.
            import hashlib
            tag=hashlib.sha256(f'{source}\0{target}'.encode()).hexdigest()[:12]
            dest=work/'experiments'/f'loco_{tag}.json'
            if dest.exists():results.append(json.loads(dest.read_text()));continue
            fit_ids={q for q in queries['fit'] if country[q]==source}
            tune_ids=[q for q in queries['tune'] if country[q]==source]
            dev_ids=[q for q in queries['dev'] if country[q]==target]
            if not fit_ids or not tune_ids or not dev_ids:continue
            fit=tables['fit'][tables['fit'].q.isin(fit_ids)]
            model=train(fit[features.NAMES].to_numpy(dtype=np.float32),fit.y.to_numpy(),cfg)
            tuning=tables['tune'][tables['tune'].q.isin(tune_ids)].copy()
            tuning['p']=model.predict(tuning[features.NAMES].to_numpy(dtype=np.float32),num_threads=4) if len(tuning) else []
            rule=decide.tune(tuning,truth_for(db,tune_ids),False)
            dev=tables['dev'][tables['dev'].q.isin(dev_ids)].copy()
            dev['p']=model.predict(dev[features.NAMES].to_numpy(dtype=np.float32),num_threads=4) if len(dev) else []
            prediction=decide.predict(dev,rule['method'],rule['threshold'])
            report=evaluate(truth_for(db,dev_ids),prediction,dev_ids)
            report.update(train_country=source,threshold_tuning_country=source,evaluation_country=target,
              final_holdout_used=False,normalization_uses_labels=False,model='single LightGBM stress test',rule=rule)
            atomic_json(dest,report);results.append(report)
    atomic_json(work/'experiments/loco_summary.json',results)
    return results
