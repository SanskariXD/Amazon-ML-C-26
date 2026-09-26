"""Full-target-pool blocking benchmark on fitting entities only; no holdout access."""
import json
from pathlib import Path
import resource
import time
import numpy as np
from .audit import truth_for
from .retrieval import build_index,Retriever
from .runtime import atomic_json


def benchmark(db,work,cfg,index_root):
    destination=work/'experiments/blocking_benchmark.json'
    if destination.exists():return json.loads(destination.read_text())
    started=time.monotonic();root=build_index(db,'train',index_root,cfg)
    index_seconds=time.monotonic()-started
    rows=list(db.execute("SELECT * FROM records WHERE split='train' AND source=1 AND part='fit' ORDER BY h LIMIT ?",(cfg.get('benchmark_entities',256),)))
    if not rows:raise ValueError('No fitting entities for retrieval benchmark')
    truth=truth_for(db,[r['id'] for r in rows]);positive=[q for q,t in truth.items() if t]
    ks=sorted(set([10,20,30,50,cfg['top_k_per_view']]))
    bcfg=dict(cfg,top_k_per_view=max(ks));retriever=Retriever(db,'train',root,bcfg)
    t0=time.monotonic();candidates=retriever.query(rows,work/'checkpoints/benchmark_retrieval.json');elapsed=time.monotonic()-t0
    n_targets=db.execute("SELECT COUNT(*) FROM records WHERE split='train' AND source IN (2,3)").fetchone()[0]
    n_test=db.execute("SELECT COUNT(*) FROM records WHERE split='test' AND source=1").fetchone()[0]
    experiments=[]
    for k in ks:
        sets={r['id']:{tid for tid,feat in c.items() if any(rank >= 1/k-1e-12 for rank in feat[3:])} for r,c in zip(rows,candidates)}
        sizes=np.array([len(v) for v in sets.values()]);tp=sum(len(truth[q]&sets[q]) for q in sets)
        country={}
        for c in sorted({r['country'] for r in rows}):
            ids=[r['id'] for r in rows if r['country']==c];den=sum(len(truth[q]) for q in ids)
            country[c]={'queries':len(ids),'true_pairs':den,'candidate_recall':sum(len(truth[q]&sets[q]) for q in ids)/den if den else None}
        den=sum(map(len,truth.values()))
        experiments.append({'top_k_per_view':k,'candidate_recall':tp/den if den else None,
          'entity_candidate_recall':sum(bool(truth[q]&sets[q]) for q in positive)/len(positive) if positive else None,
          'complete_entity_candidate_recall':sum(truth[q]<=sets[q] for q in positive)/len(positive) if positive else None,
          'average_candidates':float(sizes.mean()),'p90':float(np.quantile(sizes,.9)),'p99':float(np.quantile(sizes,.99)),
          'maximum':int(sizes.max()),'by_country':country,
          'projected_test_numeric_feature_gb':float(sizes.mean()*n_test*27*4/1e9)})
    result={'retrieval_engine':cfg.get('retrieval_engine','tfidf'),'index_metadata':json.loads((root/'complete.json').read_text()),'queries':len(rows),'query_partition':'fit','target_count':n_targets,'full_target_universe':True,
      'country_block':cfg['country_block'],'index_build_or_resume_seconds':index_seconds,'retrieval_seconds_max_k':elapsed,
      'queries_per_second':len(rows)/max(elapsed,1e-9),'linear_test_retrieval_hours_estimate':elapsed/len(rows)*n_test/3600,
      'projection_caveat':'Approximation using train target pool and benchmark batch; test size/density, cache and batch effects differ. Not a deadline guarantee.',
      'peak_ram_gb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'experiments':experiments,
      'note':'Only blocking quality. No classifier score; no model or architecture automatically promoted.'}
    atomic_json(destination,result);return result
