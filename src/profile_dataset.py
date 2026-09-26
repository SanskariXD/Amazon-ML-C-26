"""Low-memory first-pass counts before normalization/index construction."""
from collections import Counter
from pathlib import Path
import time
from .audit import read_tsv,FIELDS
from .runtime import atomic_json


def profile(dataset,work):
    destination=work/'dataset_profile_preliminary.json'
    if destination.exists():
        import json
        return json.loads(destination.read_text())
    started=time.monotonic();counts=[]
    for split in ('train','test'):
        for source in (1,2,3):
            countries={};invalid=0;longest_name=longest_address=0
            path=Path(dataset)/split/f'{split}_source{source}.tsv'
            for row in read_tsv(path,FIELDS):
                c=countries.setdefault(row['country'],{'n':0,'empty_name':0,'empty_address':0})
                c['n']+=1;c['empty_name']+=not row['business_name'].strip();c['empty_address']+=not row['business_address'].strip()
                invalid+=not row['entity_id'].startswith(f'S{source}-')
                longest_name=max(longest_name,len(row['business_name']));longest_address=max(longest_address,len(row['business_address']))
            counts.append({'split':split,'source':source,'countries':countries,'bytes':path.stat().st_size,
              'invalid_source_prefixes':invalid,'longest_name_chars':longest_name,'longest_address_chars':longest_address})
            print(f'Profiled {split} source {source}: {sum(c["n"] for c in countries.values()):,} rows',flush=True)
    distribution=Counter();duplicate_list_rows=0;rows=0
    for row in read_tsv(Path(dataset)/'train/train_ground_truth.tsv',['source1_entity_id','matched_entity_ids']):
        targets=[t.strip() for t in row['matched_entity_ids'].split(',') if t.strip()]
        distribution[len(targets)]+=1;rows+=1;duplicate_list_rows+=len(targets)!=len(set(targets))
    result={'profile_kind':'preliminary_streaming','counts':counts,'ground_truth_rows':rows,
      'matches_per_entity_distribution':dict(sorted(distribution.items())),'singletons':distribution[0],
      'duplicate_ids_within_gt_list_rows':duplicate_list_rows,'elapsed_seconds':time.monotonic()-started,
      'not_yet_verified':['global ID uniqueness','ground-truth ID membership','cross-country matches','shared target ownership'],
      'next_step':'Full audit performs relational checks before training. No model score or blocking measurement exists yet.'}
    atomic_json(destination,result);return result
