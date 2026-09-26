"""Resumable CLI. Run `python -m src.pipeline --help`."""
import argparse
import csv
import json
import os
import hashlib
import shutil
from pathlib import Path
import resource
import subprocess
import time
import zipfile
import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
from . import audit, blocking, features, decide, submission
from .metric import evaluate
from .runtime import atomic_json, resources, signature, stage_signature
from .train_stage1 import train
from .train_stage2 import train_stack, score


def parquet_atomic(frame,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.parquet');frame.to_parquet(tmp,index=False);tmp.replace(path)


def query_rows(db,part,limit):
    if part=='test':return list(db.execute("SELECT * FROM records WHERE split='test' AND source=1 ORDER BY id LIMIT ?",(limit,)))
    return list(db.execute("SELECT * FROM records WHERE split='train' AND source=1 AND part=? ORDER BY h LIMIT ?",(part,limit)))


def artifact_paths(work):
    marker=work/'artifact_paths.json'
    return {k:Path(v) for k,v in json.loads(marker.read_text()).items()} if marker.exists() else {
        'prepared':work,'index':work/'candidates'/'index','features':work/'features'}


def feature_shards(db,work,cfg,part):
    split='test' if part=='test' else 'train'
    paths=artifact_paths(work)
    root=paths['features']/part;root.mkdir(parents=True,exist_ok=True)
    if (root/'complete.json').exists():return json.loads((root/'complete.json').read_text())
    ix=blocking.build_index(db,split,paths['index'],cfg)
    retriever=blocking.Retriever(ix,cfg)
    if part=='test':cursor=db.execute("SELECT * FROM records WHERE split='test' AND source=1 ORDER BY id")
    else:cursor=db.execute("SELECT * FROM records WHERE split='train' AND source=1 AND part=? ORDER BY h LIMIT ?",(part,cfg[f'{part}_entities']))
    shards=[];i=0;macro=0
    search_batch=max(cfg['query_batch'],cfg.get('search_batch',cfg['query_batch']))
    while True:
        rows=cursor.fetchmany(search_batch)
        if not rows:break
        chunks=[rows[j:j+cfg['query_batch']] for j in range(0,len(rows),cfg['query_batch'])]
        names=[f'{i+j:06d}' for j in range(len(chunks))]
        complete=lambda stem:(root/f'{stem}.parquet').exists() and (root/f'{stem}.json').exists()
        if not all(complete(stem) for stem in names):
            checkpoint=root/f'retrieval_{macro:06d}.json'
            candidates=retriever.query(rows,checkpoint)
            offset=0
            for stem,chunk in zip(names,chunks):
                candidate_chunk=candidates[offset:offset+len(chunk)];offset+=len(chunk)
                if complete(stem):continue
                targets=features.target_records(db,split,{t for c in candidate_chunk for t in c})
                truth=audit.truth_for(db,[r['id'] for r in chunk]) if split=='train' else {}
                output=[]
                for q,c in zip(chunk,candidate_chunk):
                    for t,retrieval in sorted(c.items()):
                        output.append([q['id'],t,int(t in truth.get(q['id'],set()))]+features.pair(q,targets[t],retrieval))
                frame=pd.DataFrame(output,columns=['q','t','y']+features.NAMES)
                for c in features.NAMES:frame[c]=frame[c].astype('float32')
                parquet_atomic(frame,root/f'{stem}.parquet')
                atomic_json(root/f'{stem}.json',{'queries':[r['id'] for r in chunk],'pairs':len(frame)})
            checkpoint.unlink(missing_ok=True)
        shards+=names;i+=len(chunks);macro+=1
        print(f'{part}: {i} complete query shards',flush=True)
    if not shards:raise ValueError(f'No entities for partition {part}')
    obj={'shards':shards,'full_target_pool':True};atomic_json(root/'complete.json',obj)
    return obj


def frames(work,part):
    root=artifact_paths(work)['features']/part
    obj=json.loads((root/'complete.json').read_text())
    for stem in obj['shards']:
        yield stem,pd.read_parquet(root/f'{stem}.parquet'),json.loads((root/f'{stem}.json').read_text())['queries']


def fit(db,work,cfg):
    model_dir=work/'models';model_dir.mkdir(exist_ok=True)
    if (model_dir/'complete.json').exists():return load_models(model_dir)
    feature_shards(db,work,cfg,'fit')
    paths=list((artifact_paths(work)['features']/'fit').glob('[0-9]*.json'))
    count=sum(json.loads(p.read_text())['pairs'] for p in paths)
    # Includes pandas strings, copies during fitting and LightGBM histogram overhead.
    estimate=count*(len(features.NAMES)*4+180)*5
    if estimate>psutil.virtual_memory().available*.60:
        raise MemoryError(f'Estimated fit peak {estimate/2**30:.1f} GB exceeds budget. Reduce fit_entities in config; samples are selected by entity, never by pair.')
    rows=[df for _,df,_ in frames(work,'fit')]
    df=pd.concat(rows,ignore_index=True);del rows
    X=df[features.NAMES].to_numpy(dtype=np.float32);y=df.y.to_numpy();q=df.q.to_numpy()
    if cfg['two_stage']:models,stage2=train_stack(X,y,q,cfg,model_dir)
    else:
        m=train(X,y,cfg)
        model_path=model_dir/'stage1.txt';m.save_model(str(model_path)+'.tmp');Path(str(model_path)+'.tmp').replace(model_path)
        models,stage2=[m],None
    atomic_json(model_dir/'complete.json',{'two_stage':cfg['two_stage'],'folds':cfg['folds'],'feature_names':features.NAMES,'fit_pairs':len(df)})
    return models,stage2


def load_models(root):
    meta=json.loads((root/'complete.json').read_text())
    if meta['feature_names']!=features.NAMES:raise ValueError('Feature manifest mismatch')
    paths=[root/f'stage1_fold{i}.txt' for i in range(meta['folds'])] if meta['two_stage'] else [root/'stage1.txt']
    return [lgb.Booster(model_file=str(p)) for p in paths],lgb.Booster(model_file=str(root/'stage2.txt')) if meta['two_stage'] else None


def scored(work,part,models,stage2):
    root=work/'checkpoints'/part;root.mkdir(parents=True,exist_ok=True)
    for stem,df,ids in frames(work,part):
        path=root/f'{stem}.parquet'
        if path.exists():df=pd.read_parquet(path)
        else:
            df=df.copy()
            df['p']=score(models,stage2,df[features.NAMES].to_numpy(dtype=np.float32),df.q.to_numpy()) if len(df) else np.array([],dtype=float)
            df=df[['q','t','y','p']];parquet_atomic(df,path)
        yield df,ids


def tune_and_evaluate(db,work,cfg,models,stage2,part='dev'):
    rule_path=work/'models'/'decision.json'
    if not rule_path.exists():
        feature_shards(db,work,cfg,'tune')
        parts=[];ids=[]
        for df,qs in scored(work,'tune',models,stage2):parts.append(df);ids+=qs
        rule=decide.tune(pd.concat(parts,ignore_index=True),audit.truth_for(db,ids),cfg.get('expected_f',False))
        atomic_json(rule_path,rule)
    rule=json.loads(rule_path.read_text())
    report_path=work/'experiments'/f'{part}.json'
    if report_path.exists():return json.loads(report_path.read_text())
    feature_shards(db,work,cfg,part)
    ids=[];pred={};candidate={};error_samples=[]
    for df,qs in scored(work,part,models,stage2):
        ids+=qs;pred.update(decide.predict(df,rule['method'],rule['threshold']))
        candidate.update({q:set(g.t) for q,g in df.groupby('q')})
    truth=audit.truth_for(db,ids);report=evaluate(truth,pred,ids)
    hits=sum(len(truth[q]&candidate.get(q,set())) for q in ids)
    nonempty=[q for q in ids if truth[q]];sizes=np.array([len(candidate.get(q,set())) for q in ids])
    report.update(candidate_recall=hits/max(1,sum(map(len,truth.values()))),
      entity_candidate_recall=sum(bool(truth[q]&candidate.get(q,set())) for q in nonempty)/max(1,len(nonempty)),
      complete_entity_candidate_recall=sum(truth[q]<=candidate.get(q,set()) for q in nonempty)/max(1,len(nonempty)),
      avg_candidates=float(sizes.mean()),p90_candidates=float(np.quantile(sizes,.9)),p99_candidates=float(np.quantile(sizes,.99)),max_candidates=int(sizes.max()),
      rule=rule,partition=part,full_target_universe=True,experiment_id=cfg['experiment_id'])
    countries={}
    for q in ids:
        country=db.execute("SELECT country FROM records WHERE split='train' AND id=?",(q,)).fetchone()[0]
        countries.setdefault(country,[]).append(q)
        fp=pred.get(q,set())-truth[q];fn=truth[q]-pred.get(q,set())
        if (fp or fn) and len(error_samples)<100:error_samples.append({'q':q,'country':country,'false_positives':sorted(fp),'false_negatives':sorted(fn),'singleton':not truth[q],'missed_by_blocking':sorted(truth[q]-candidate.get(q,set()))})
    report['by_country']={c:evaluate(truth,pred,qs) for c,qs in countries.items()}
    atomic_json(report_path,report);atomic_json(work/'experiments'/f'{part}_errors.json',error_samples)
    return report


def inference(db,work,cfg,models,stage2):
    feature_shards(db,work,cfg,'test')
    rule=json.loads((work/'models'/'decision.json').read_text())
    out=work/'outputs';out.mkdir(exist_ok=True)
    mp=out/'matching_results.tsv';cp=out/'candidate_pairs.tsv'
    with open(str(mp)+'.tmp','w',encoding='utf-8',newline='') as mf,open(str(cp)+'.tmp','w',encoding='utf-8',newline='') as cf:
        mw=csv.writer(mf,delimiter='\t',lineterminator='\n');cw=csv.writer(cf,delimiter='\t',lineterminator='\n')
        mw.writerow(['source1_entity_id','matched_entity_ids']);cw.writerow(['source1_entity_id','candidate_entity_ids'])
        for df,ids in scored(work,'test',models,stage2):
            pred=decide.predict(df,rule['method'],rule['threshold']);cands={q:set(g.t) for q,g in df.groupby('q')}
            for q in ids:
                if not pred.get(q,set())<=cands.get(q,set()):raise AssertionError('Candidate subset violated')
                mw.writerow([q,','.join(sorted(pred.get(q,set())))]);cw.writerow([q,','.join(sorted(cands.get(q,set())))])
    Path(str(mp)+'.tmp').replace(mp);Path(str(cp)+'.tmp').replace(cp)
    report=submission.validate(mp,cp,db);atomic_json(out/'internal_validation.json',report)
    return report


def official_validate(dataset,work):
    # Organizer utility remains in the user's original resource bundle, not copied from a competitor.
    candidates=[Path(dataset).parent/'utils/validate_submission.py',Path(dataset)/'utils/validate_submission.py']
    script=next((p for p in candidates if p.exists()),None)
    if script is None:raise FileNotFoundError('Organizer validator missing from resource ZIP')
    cmd=['python',str(script),'--matching',str(work/'outputs/matching_results.tsv'),'--candidate',str(work/'outputs/candidate_pairs.tsv'),'--test-dir',str(Path(dataset)/'test')]
    if '--check-ids' in script.read_text():cmd+=['--check-ids']
    result=subprocess.run(cmd,text=True,capture_output=True)
    atomic_json(work/'outputs/official_validation.json',{'command':cmd,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
    print(result.stdout)
    if result.returncode:raise RuntimeError('Official validation failed')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',required=True,help='Folder containing train/ and test/')
    ap.add_argument('--work',required=True,help='Persistent output root (Drive in Colab)')
    ap.add_argument('--config',default='configs/baseline.json')
    ap.add_argument('--stage',choices=['audit','index','benchmark','baseline','holdout','infer','validate','package','report','loco','profile'],default='baseline')
    args=ap.parse_args();cfg=json.loads(Path(args.config).read_text())
    Path(args.work).mkdir(parents=True,exist_ok=True)
    rt=resources(args.work);print(json.dumps(rt),flush=True)
    manifest=audit.manifest(Path(args.dataset));run_id=signature(cfg,manifest)
    work=Path(args.work)/run_id;work.mkdir(exist_ok=True)
    # Persist resource sizing once: reconnecting with different RAM must not invalidate completed shards.
    if (work/'config.json').exists():cfg=json.loads((work/'config.json').read_text())
    else:
        available=psutil.virtual_memory().available
        cfg['target_shard']=max(100,min(cfg['target_shard'],int(available*.08/5000)))
        cfg['query_batch']=max(1,min(cfg['query_batch'],int(available*.025/(cfg['target_shard']*12))))
        cfg['search_batch']=max(cfg['query_batch'],min(cfg.get('search_batch',4096),int(available*.04/30000)))
        atomic_json(work/'config.json',cfg)
    atomic_json(Path(args.work)/'latest_run.json',{'run_id':run_id,'work':str(work.resolve())})
    atomic_json(work/'dataset_manifest.json',manifest);atomic_json(work/'runtime.json',rt)
    print(f'Run directory: {work}',flush=True)
    if args.stage=='profile':
        from .profile_dataset import profile
        print(json.dumps(profile(args.dataset,work),indent=2));return
    if args.stage=='report':
        from .report import export_report
        print(export_report(work));return
    prepared_key=stage_signature({'seed':cfg['seed']},manifest,['audit.py','normalize.py','split.py'])
    shared=Path(args.work)/'_shared'
    prepared=shared/'prepared'/prepared_key
    index_key=stage_signature({k:cfg[k] for k in ('target_shard','vocabulary_sample','max_features')},prepared_key,['blocking.py'])
    feature_params={k:cfg[k] for k in ('seed','fit_entities','tune_entities','dev_entities','holdout_entities','query_batch','top_k_per_view','country_block')}
    feature_params['search_batch']=cfg.get('search_batch',cfg['query_batch'])
    feature_key=stage_signature(feature_params,index_key,['features.py','pipeline.py'])
    paths={'prepared':prepared,'index':shared/'indexes'/index_key,'features':shared/'features'/feature_key}
    atomic_json(work/'artifact_paths.json',{k:str(v.resolve()) for k,v in paths.items()})
    start=time.time();database=audit.build(args.dataset,prepared,cfg['seed'])
    if os.environ.get('ER_LOCAL_CACHE'):
        local_key=hashlib.sha256(str(prepared.resolve()).encode()).hexdigest()[:16]
        local=Path(os.environ['ER_LOCAL_CACHE'])/'prepared'/local_key/'records.sqlite'
        local.parent.mkdir(parents=True,exist_ok=True)
        if not local.exists() or local.stat().st_size!=database.stat().st_size:
            temporary=local.with_suffix('.tmp');shutil.copyfile(database,temporary);temporary.replace(local)
        database=local
    db=audit.connect(database)
    profile=json.loads((prepared/'dataset_profile.json').read_text())
    atomic_json(work/'dataset_profile.json',profile)
    if cfg['country_block'] and profile['checks']['cross_country_pairs']:raise ValueError('Country blocking would discard known matches')
    if args.stage=='audit':return
    if args.stage=='index':blocking.build_index(db,'train',paths['index'],cfg);return
    if args.stage=='benchmark':
        from .benchmark import benchmark
        report=benchmark(db,work,cfg,paths['index']);print(json.dumps(report,indent=2));return
    if args.stage=='loco':
        from .loco import run_loco
        print(json.dumps(run_loco(db,work,cfg),indent=2));return
    if args.stage in ('baseline','holdout','infer'):
        models,stage2=fit(db,work,cfg)
        if args.stage=='baseline':report=tune_and_evaluate(db,work,cfg,models,stage2)
        elif args.stage=='holdout':report=tune_and_evaluate(db,work,cfg,models,stage2,'holdout')
        else:
            if not (work/'models/decision.json').exists():raise RuntimeError('Run baseline before inference')
            report=inference(db,work,cfg,models,stage2);official_validate(args.dataset,work)
        report.update(runtime_seconds=time.time()-start,peak_ram_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,gpu_used=False)
        timing_path=work/'experiments'/f'{args.stage}_run.json'
        if not timing_path.exists():atomic_json(timing_path,report)
        atomic_json(work/'experiments'/f'{args.stage}_last_invocation.json',report)
        print(json.dumps(report,indent=2))
        if args.stage=='baseline' and not (work/'experiments/results.csv').exists():write_experiment(work,cfg,report)
    if args.stage in ('validate','package'):official_validate(args.dataset,work)
    if args.stage=='package':
        from .package import package
        print(package(work))


def write_experiment(work,cfg,report):
    fields=['experiment_id','date','git_commit','description','hypothesis','blocking_method','candidate_recall','entity_candidate_recall','avg_candidates','p99_candidates','model','features','negative_strategy','validation_entities','precision','recall','macro_f0.5','singleton_accuracy','india_f0.5','us_f0.5','threshold_method','runtime','peak_ram','gpu_used','notes','promoted']
    try:commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except subprocess.SubprocessError:commit='uncommitted'
    row={k:report.get(k,'') for k in fields}
    row.update(date=time.strftime('%Y-%m-%d'),git_commit=commit,description='Independent baseline',hypothesis='Full-pool retrieval + lexical GBDT',blocking_method='3-view sharded TF-IDF',model='LightGBM stage2' if cfg['two_stage'] else 'LightGBM',features=len(features.NAMES),negative_strategy='all retrieved negatives for sampled fit entities',validation_entities=report['entities'],threshold_method=report['rule']['method'],runtime=report['runtime_seconds'],peak_ram=report['peak_ram_gb'],promoted=False)
    for c,key in [('India','india_f0.5'),('US','us_f0.5')]:row[key]=report['by_country'].get(c,{}).get('macro_f0.5','')
    path=work/'experiments/results.csv'
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow(row)


if __name__=='__main__':main()
