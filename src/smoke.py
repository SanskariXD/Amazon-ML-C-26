"""Synthetic end-to-end integrity check, never a competition benchmark."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from .runtime import atomic_json
from .audit import connect
from .submission import validate


def make_data(root):
    for split in ('train','test'):
        d=root/split;d.mkdir(parents=True,exist_ok=True)
        sources={1:[],2:[],3:[]};truth=[]
        for i in range(240 if split=='train' else 36):
            country=['India','US','France'][i%3] if split=='test' else ['India','US'][i%2]
            q=f'S1-{i:06d}';name=f'Lotus {i%13} Systems';address=f'{100+i} Cedar Road Sector {i%7}'
            sources[1].append([q,name,address,country]);matches=[]
            if i%5:
                t=f'S2-{i:06d}';sources[2].append([t,name,address.replace('Road','Rd'),country]);matches.append(t)
                if i%3==0:
                    t=f'S3-{i:06d}';sources[3].append([t,name.upper(),address,country]);matches.append(t)
            sources[2].append([f'S2-D{i:06d}',name,f'{9000+i} Elm Street',country])
            sources[3].append([f'S3-D{i:06d}','Unrelated Bakery '+str(i),'77 Distant Avenue',country])
            truth.append([q,','.join(matches)])
        for k,rows in sources.items():
            with (d/f'{split}_source{k}.tsv').open('w',newline='') as f:
                w=csv.writer(f,delimiter='\t');w.writerow(['entity_id','business_name','business_address','country']);w.writerows(rows)
        if split=='train':
            with (d/'train_ground_truth.tsv').open('w',newline='') as f:
                w=csv.writer(f,delimiter='\t');w.writerow(['source1_entity_id','matched_entity_ids']);w.writerows(truth)


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--root');args=ap.parse_args()
    root=Path(args.root or tempfile.mkdtemp(prefix='er-smoke-')).resolve();root.mkdir(parents=True,exist_ok=True)
    make_data(root/'dataset')
    cfg=json.loads(Path('configs/baseline.json').read_text())
    cfg.update(fit_entities=150,tune_entities=30,dev_entities=30,holdout_entities=30,query_batch=40,target_shard=200,vocabulary_sample=500,max_features=1500,top_k_per_view=5,rounds=25,min_child_samples=4,two_stage=True,experiment_id='SMOKE_ONLY')
    atomic_json(root/'smoke.json',cfg)
    cmd=[sys.executable,'-m','src.pipeline','--dataset',str(root/'dataset'),'--work',str(root/'work'),'--config',str(root/'smoke.json')]
    subprocess.run(cmd+['--stage','baseline'],check=True)
    latest=json.loads((root/'work/latest_run.json').read_text());work=Path(latest['work'])
    from .pipeline import load_models,inference
    db=connect(work/'records.sqlite');models,stage2=load_models(work/'models')
    inference(db,work,cfg,models,stage2)
    result=validate(work/'outputs/matching_results.tsv',work/'outputs/candidate_pairs.tsv',db)
    assert result['entities']==36
    # The external holdout remains untouched by the smoke baseline run.
    assert not (work/'features/holdout').exists()
    # Re-running resumes completed features and models, without modifying their timestamps.
    before={str(p):p.stat().st_mtime_ns for p in (work/'models').glob('*.txt')}
    subprocess.run(cmd+['--stage','baseline'],check=True,stdout=subprocess.DEVNULL)
    assert json.loads((root/'work/latest_run.json').read_text())['work']==str(work), 'Resume changed run identity'
    assert before=={str(p):p.stat().st_mtime_ns for p in (work/'models').glob('*.txt')}
    atomic_json(root/'smoke_result.json',{'status':'PASS','synthetic_only':True,'output_entities':36,'work':str(work),'resume_verified':True,'holdout_untouched':True})
    print(f'SMOKE PASS (synthetic only): {root}/smoke_result.json')

if __name__=='__main__':main()
